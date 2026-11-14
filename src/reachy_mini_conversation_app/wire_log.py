"""The wire log (the companion's W1, owner 11-14): one record of what goes in and out of the realtime connection.

Every call on the connection's ``session``, ``conversation``, ``response`` and
``input_audio_buffer`` is an "out" row; every event the receive loop takes is an "in" row.
No media is kept (the owner): an image or audio payload becomes its hash and size. Audio
chunks and text deltas are counted, not stored: audio as bytes per response or per
committed user turn, deltas not at all (their ``.done`` events carry the whole text).

The tap does one thing on the event loop: a reference on a bounded queue. A writer
thread serializes, compresses (zlib) and commits in batches. A full queue drops the event
and counts it; a database error turns the log off for the process with one warning.
Nothing here can block or break the conversation.

On only when ``REACHY_COMPANION_WIRE_LOG`` names a database path. Rows older than
RETENTION_DAYS are deleted at start and once a day.
"""

from __future__ import annotations
import os
import json
import time
import zlib
import queue
import hashlib
import logging
import sqlite3
import threading
import subprocess
from typing import Any
from pathlib import Path


logger = logging.getLogger(__name__)

ENV = "REACHY_COMPANION_WIRE_LOG"
RETENTION_DAYS = 90  # the owner, 11-14
QUEUE_MAX = 5000  # about 40 s of the busiest stream (mic chunks) with the writer stalled
BATCH_S = 1.0  # the writer's wake
BLOB_MIN = 8192  # payloads this large (raw), the instructions above all, are stored once by hash
ZLEVEL = 3
NAMESPACES = ("session", "conversation", "response", "input_audio_buffer", "output_audio_buffer")
MEDIA_KEYS = ("image_url", "audio", "image", "data")  # payload keys of media parts
MEDIA_TYPES = ("input_image", "input_audio", "image", "audio")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY, opened_at REAL NOT NULL, closed_at REAL,
  backend_host TEXT, app_commit TEXT, companion_commit TEXT, profile TEXT
);
CREATE TABLE IF NOT EXISTS blobs (hash TEXT PRIMARY KEY, size INTEGER NOT NULL, z BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, t REAL NOT NULL, dir TEXT NOT NULL,
  type TEXT NOT NULL, item_id TEXT, response_id TEXT, call_id TEXT, size INTEGER NOT NULL,
  z BLOB, blob TEXT
);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, id);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
CREATE INDEX IF NOT EXISTS events_item ON events(item_id);
CREATE TABLE IF NOT EXISTS usage (
  session_id INTEGER NOT NULL, response_id TEXT, t REAL NOT NULL,
  input INTEGER, cached INTEGER, output INTEGER
);
"""


def media_stub(value: str) -> dict[str, Any]:
    """Return the placeholder kept for a media payload: its kind, hash and size. Pure."""
    kind = "media"
    raw = value
    if value.startswith("data:") and ";base64," in value:
        head, raw = value.split(";base64,", 1)
        kind = head[5:] or kind
    return {
        "media": kind,
        "sha256": hashlib.sha256(raw.encode()).hexdigest()[:16],
        "bytes": len(raw) * 3 // 4 if raw is not value else len(raw),
    }


def strip_media(obj: Any) -> Any:
    """Return a copy of a payload with every media part reduced to its stub. Pure.

    A media part is a data URL anywhere, or the payload keys of a part whose type is an
    image or audio (a picture from a channel may be a plain URL; it goes too).
    """
    if isinstance(obj, str):
        return media_stub(obj) if obj.startswith("data:") and ";base64," in obj else obj
    if isinstance(obj, list):
        return [strip_media(x) for x in obj]
    if isinstance(obj, dict):
        media = obj.get("type") in MEDIA_TYPES
        out = {}
        for k, v in obj.items():
            if media and k in MEDIA_KEYS and isinstance(v, str):
                out[k] = media_stub(v)
            else:
                out[k] = strip_media(v)
        return out
    return obj


def to_jsonable(obj: Any) -> Any:
    """Return an event or argument as plain JSON data (pydantic models dumped)."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # an SDK model that refuses: fall through
            pass
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return {"media": "bytes", "bytes": len(obj)}
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return {k: to_jsonable(v) for k, v in d.items() if not k.startswith("_")}
    return repr(obj)


def b64_bytes(value: Any) -> int:
    """Return the decoded size of a base64 audio chunk (0 for anything else). Pure."""
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value) * 3 // 4 - value[-2:].count("=")
    return 0


def _ids(kind: str, data: Any) -> tuple[str | None, str | None, str | None]:
    """Return (item id, response id, call id) found in a payload. Pure."""
    if not isinstance(data, dict):
        return None, None, None
    item = data.get("item") if isinstance(data.get("item"), dict) else {}
    resp = data.get("response") if isinstance(data.get("response"), dict) else {}
    item_id = data.get("item_id") or item.get("id")
    response_id = data.get("response_id") or resp.get("id")
    call_id = data.get("call_id") or item.get("call_id")
    return item_id, response_id, call_id


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class WireLog:
    """The recorder: ``wrap()`` a connection at its open, ``close()`` it at its end."""

    def __init__(self, path: str | os.PathLike[str], *, retention_days: int = RETENTION_DAYS,
                 queue_max: int = QUEUE_MAX, batch_s: float = BATCH_S) -> None:
        """Open (or create) the database lazily on the writer thread; start the writer."""
        self.path = Path(path)
        self.retention_days = retention_days
        self.batch_s = batch_s
        self._q: queue.Queue[tuple[Any, ...]] = queue.Queue(maxsize=queue_max)
        self.dropped = 0
        self.enabled = True
        self._next_session = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="wire-log", daemon=True)
        self._thread.start()

    # --- the tap (event loop side): a reference on the queue, nothing else -------------

    def _put(self, entry: tuple[Any, ...]) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(entry)
        except queue.Full:
            self.dropped += 1

    def open_session(self, *, backend_host: str | None = None, profile: str | None = None) -> int:
        """Start a session row and return its local number."""
        with self._lock:
            self._next_session += 1
            sid = self._next_session
        self._put(("open", sid, time.time(), backend_host, profile))
        return sid

    def close_session(self, sid: int) -> None:
        """Mark a session closed."""
        self._put(("close", sid, time.time()))

    def out(self, sid: int, kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Note a call sent on the connection."""
        self._put(("out", sid, time.time(), kind, args, kwargs))

    def received(self, sid: int, event: Any) -> None:
        """Note an event received."""
        self._put(("in", sid, time.time(), event))

    def wrap(self, conn: Any, **meta: Any) -> Any:
        """Return ``conn`` wrapped so every send and receive is recorded."""
        return Tapped(conn, self, self.open_session(**meta))

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until the queue is written (tests, shutdown)."""
        self._put(("flush",))
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)

    # --- the writer thread --------------------------------------------------------------

    def _run(self) -> None:
        db = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            db = sqlite3.connect(self.path, isolation_level=None)
            if new:
                db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.executescript(SCHEMA)
        except Exception as exc:
            self._disable(f"cannot open {self.path}: {exc}")
            return
        self._sessions: dict[int, int] = {}  # local number -> row id
        self._audio_out: dict[tuple[int, str | None], int] = {}
        self._audio_in: dict[int, int] = {}
        pruned_at = 0.0
        reported = 0
        while self.enabled:
            batch = []
            try:
                batch.append(self._q.get(timeout=self.batch_s))
                while len(batch) < 500:
                    batch.append(self._q.get_nowait())
            except queue.Empty:
                pass
            try:
                if batch:
                    db.execute("BEGIN")
                    for entry in batch:
                        self._write(db, entry)
                    if self.dropped > reported:
                        for row in set(self._sessions.values()):
                            db.execute(
                                "INSERT INTO events (session_id, t, dir, type, size) VALUES (?,?,?,?,?)",
                                (row, time.time(), "meta", "wire.dropped", self.dropped - reported),
                            )
                            break
                        reported = self.dropped
                    db.execute("COMMIT")
                if time.time() - pruned_at > 86400:
                    pruned_at = time.time()
                    self._prune(db)
            except Exception as exc:
                self._disable(f"write failed: {exc}")
            finally:
                for _ in batch:
                    self._q.task_done()
        if db is not None:
            db.close()

    def _disable(self, why: str) -> None:
        if self.enabled:
            self.enabled = False
            logger.warning("wire log off for this run: %s", why)
        while True:  # release anyone waiting on flush
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break

    def _prune(self, db: sqlite3.Connection) -> None:
        cutoff = time.time() - self.retention_days * 86400
        db.execute("DELETE FROM events WHERE t < ?", (cutoff,))
        db.execute("DELETE FROM usage WHERE t < ?", (cutoff,))
        db.execute("DELETE FROM sessions WHERE opened_at < ? AND id NOT IN (SELECT DISTINCT session_id FROM events)", (cutoff,))
        db.execute("DELETE FROM blobs WHERE hash NOT IN (SELECT blob FROM events WHERE blob IS NOT NULL)")
        db.execute("PRAGMA incremental_vacuum")

    def _write(self, db: sqlite3.Connection, entry: tuple[Any, ...]) -> None:
        kind = entry[0]
        if kind == "flush":
            return
        if kind == "open":
            _, sid, t, host, profile = entry
            cur = db.execute(
                "INSERT INTO sessions (opened_at, backend_host, app_commit, companion_commit, profile) VALUES (?,?,?,?,?)",
                (t, host, _git_head(Path(__file__).parent), _git_head(Path.home() / "reachy-companion"), profile),
            )
            self._sessions[sid] = cur.lastrowid
            return
        row = self._sessions.get(entry[1])
        if row is None:
            return
        if kind == "close":
            db.execute("UPDATE sessions SET closed_at=? WHERE id=?", (entry[2], row))
            self._sessions.pop(entry[1], None)
            return
        if kind == "out":
            _, sid, t, name, args, kwargs = entry
            if name == "input_audio_buffer.append":  # the mic stream: counted per turn
                self._audio_in[sid] = self._audio_in.get(sid, 0) + b64_bytes(kwargs.get("audio", args[0] if args else None))
                return
            data: Any = to_jsonable(kwargs) if kwargs else {}
            if args:
                data = {"args": to_jsonable(list(args)), **(data if isinstance(data, dict) else {})}
            self._row(db, row, t, "out", name, data)
            return
        _, sid, t, event = entry
        etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None) or "?"
        if etype.endswith("audio.delta"):
            key = (sid, getattr(event, "response_id", None))
            self._audio_out[key] = self._audio_out.get(key, 0) + b64_bytes(getattr(event, "delta", None))
            return
        if etype.endswith(".delta"):
            return
        data = to_jsonable(event)
        self._row(db, row, t, "in", etype, data)
        if etype == "input_audio_buffer.committed":
            n = self._audio_in.pop(sid, 0)
            self._row(db, row, t, "meta", "audio.in", {"pcm_bytes": n}, size=n,
                      ids=(_ids(etype, data)[0], None, None))
        elif etype == "response.done":
            resp = data.get("response", {}) if isinstance(data, dict) else {}
            rid = resp.get("id")
            n = self._audio_out.pop((sid, rid), 0)
            if n:
                self._row(db, row, t, "meta", "audio.out", {"pcm_bytes": n}, size=n, ids=(None, rid, None))
            u = resp.get("usage") or {}
            if u:
                cached = (u.get("input_token_details") or {}).get("cached_tokens")
                db.execute(
                    "INSERT INTO usage (session_id, response_id, t, input, cached, output) VALUES (?,?,?,?,?,?)",
                    (row, rid, t, u.get("input_tokens"), cached, u.get("output_tokens")),
                )

    def _row(self, db: sqlite3.Connection, row: int, t: float, direction: str, kind: str, data: Any,
             size: int | None = None, ids: tuple[str | None, str | None, str | None] | None = None) -> None:
        data = strip_media(data)
        raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
        z = zlib.compress(raw, ZLEVEL)
        item_id, response_id, call_id = ids or _ids(kind, data)
        blob = None
        if len(raw) >= BLOB_MIN:
            blob = hashlib.sha256(raw).hexdigest()
            db.execute("INSERT OR IGNORE INTO blobs (hash, size, z) VALUES (?,?,?)", (blob, len(raw), z))
            z = None
        db.execute(
            "INSERT INTO events (session_id, t, dir, type, item_id, response_id, call_id, size, z, blob) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row, t, direction, kind, item_id, response_id, call_id, len(raw) if size is None else size, z, blob),
        )


class _Space:
    """A namespace of the connection (``conn.conversation.item``): its calls are recorded."""

    def __init__(self, target: Any, path: str, log: WireLog, sid: int) -> None:
        self._target, self._path, self._log, self._sid = target, path, log, sid

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = f"{self._path}.{name}"
        if callable(attr):
            log, sid = self._log, self._sid

            def call(*args: Any, **kwargs: Any) -> Any:
                log.out(sid, path, args, kwargs)
                return attr(*args, **kwargs)

            return call
        return _Space(attr, path, self._log, self._sid)


class Tapped:
    """A realtime connection whose sends and receives go to the wire log; otherwise the same."""

    def __init__(self, conn: Any, log: WireLog, sid: int) -> None:
        """Wrap ``conn`` as log session ``sid``."""
        self._conn, self._log, self._sid = conn, log, sid

    def __getattr__(self, name: str) -> Any:
        """Return the connection's attribute; a namespace of calls comes back recorded."""
        attr = getattr(self._conn, name)
        if name in NAMESPACES:
            return _Space(attr, name, self._log, self._sid)
        return attr

    def __aiter__(self) -> Any:
        """Iterate the connection's events, each one recorded."""
        return self._events()

    async def _events(self) -> Any:
        try:
            async for event in self._conn:
                self._log.received(self._sid, event)
                yield event
        finally:  # the receive loop ends with the connection: so does its session
            self._log.close_session(self._sid)


_LOG: WireLog | None = None


def from_env() -> WireLog | None:
    """Return the process's wire log, started on first use, or None when it is off."""
    global _LOG
    path = os.environ.get(ENV, "").strip()
    if not path or path.lower() == "off":
        return None
    if _LOG is None or str(_LOG.path) != os.path.expanduser(path):
        _LOG = WireLog(os.path.expanduser(path))
    return _LOG if _LOG.enabled else None
