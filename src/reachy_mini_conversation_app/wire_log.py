"""The wire log (the companion's W1, owner 11-14): one record of what goes in and out of the realtime connection.

Every call on the connection's ``session``, ``conversation`` and ``response`` is an "out"
row; every event the receive loop takes is an "in" row. No media is kept (the owner): an
image or audio payload becomes its hash and size. The mic stream (``input_audio_buffer.
append``) is not recorded (the server's committed and transcription events carry each
turn's span and words); his audio is counted per response; text deltas are skipped (their
``.done`` events carry the whole text). The instructions and the tool list are stored once
per distinct text, by hash.

The tap does one thing on the event loop: a reference on a bounded queue. A writer thread
serializes, compresses (zlib) and commits about once a second. A full queue drops the event
and counts it. A database error rolls back that batch, is logged once per kind and counted,
and the writer goes on (the owner's rule: no layer switches itself off); a database that
cannot open is retried each minute. Nothing here can block or break the conversation.

On only when ``REACHY_COMPANION_WIRE_LOG`` names a database path. Rows older than the
companion's turn retention (``REACHY_COMPANION_TURN_RETENTION_DAYS``, 90) are deleted at
start and daily.
"""

from __future__ import annotations
import os
import re
import json
import time
import zlib
import queue
import hashlib
import logging
import sqlite3
import threading
import contextlib
import subprocess
from typing import Any, AsyncIterator
from pathlib import Path


logger = logging.getLogger(__name__)

ENV = "REACHY_COMPANION_WIRE_LOG"
RETENTION_ENV = "REACHY_COMPANION_TURN_RETENTION_DAYS"  # one setting for kept talk (90, owner 11-08)
SCHEMA_VERSION = 1
QUEUE_MAX = 5000
BATCH_S = 1.0  # the writer commits about this often
OPEN_RETRY_S = 60.0
BLOB_MIN = 8192  # other payloads this large (raw) are stored once by hash
ZLEVEL = 3
SOURCE = "realtime"  # this connection; the thinker, summarizer and evals can share the log later
NAMESPACES = ("session", "conversation", "response", "input_audio_buffer", "output_audio_buffer")
NOT_RECORDED = ("input_audio_buffer.append",)  # the mic stream: about 50 a second while connected
MEDIA_KEYS_ALWAYS = ("audio", "image_url", "image")  # a string under these is media, any part type
MEDIA_TYPES = ("input_image", "input_audio", "output_audio", "image", "audio")
DATA_URL = re.compile(r"data:([\w.+/-]*);base64,([A-Za-z0-9+/=]{16,})")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY, source TEXT NOT NULL DEFAULT 'realtime',
  opened_at REAL NOT NULL, closed_at REAL,
  backend_host TEXT, app_commit TEXT, companion_commit TEXT, profile TEXT
);
CREATE TABLE IF NOT EXISTS blobs (hash TEXT PRIMARY KEY, size INTEGER NOT NULL, z BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, t REAL NOT NULL, dir TEXT NOT NULL,
  type TEXT NOT NULL, item_id TEXT, response_id TEXT, call_id TEXT, size INTEGER NOT NULL,
  z BLOB, blob TEXT, instr_blob TEXT, tools_blob TEXT
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
    m = DATA_URL.fullmatch(value)
    kind, raw, size = "media", value, len(value)
    if m:
        kind, raw = m.group(1) or kind, m.group(2)
        size = len(raw) * 3 // 4 - raw[-2:].count("=")
    return {"media": kind, "sha256": hashlib.sha256(raw.encode()).hexdigest()[:16], "bytes": size}


def _stub_inline(text: str) -> str:
    """Replace data URLs inside a longer string (JSON text, a tool output) with a short tag. Pure."""
    def tag(m: re.Match[str]) -> str:
        s = media_stub(m.group(0))
        return f"[media {s['media']} {s['sha256']} {s['bytes']} bytes]"

    return DATA_URL.sub(tag, text)


def strip_media(obj: Any) -> Any:
    """Return a copy of a payload with every media payload reduced to its stub. Pure.

    Media: a data URL anywhere (whole string or inside one); any string under ``audio``,
    ``image_url`` or ``image``; and ``data`` in a part whose type is an image or audio.
    """
    if isinstance(obj, str):
        if DATA_URL.fullmatch(obj):
            return media_stub(obj)
        return _stub_inline(obj) if "base64," in obj else obj
    if isinstance(obj, list):
        return [strip_media(x) for x in obj]
    if isinstance(obj, dict):
        media = obj.get("type") in MEDIA_TYPES
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and (k in MEDIA_KEYS_ALWAYS or (media and k == "data")):
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
    if isinstance(obj, (bytes, bytearray)):
        return {"media": "bytes", "bytes": len(obj)}
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return {k: to_jsonable(v) for k, v in d.items() if not k.startswith("_")}
    return repr(obj)


def b64_bytes(value: Any) -> int:
    """Return the decoded size of a base64 audio chunk (0 for anything else). Pure."""
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str) and value:
        return len(value) * 3 // 4 - value[-2:].count("=")
    return 0


def _ids(data: Any) -> tuple[str | None, str | None, str | None]:
    """Return (item id, response id, call id) found in a payload. Pure."""
    if not isinstance(data, dict):
        return None, None, None
    item = data.get("item") if isinstance(data.get("item"), dict) else {}
    resp = data.get("response") if isinstance(data.get("response"), dict) else {}
    item_id = data.get("item_id") or item.get("id")
    response_id = data.get("response_id") or resp.get("id")
    call_id = data.get("call_id") or item.get("call_id")
    return item_id, response_id, call_id


def retention_days() -> int:
    """Return the days kept: the companion's turn retention, 90 when unset or not a number."""
    try:
        return max(1, int(os.environ.get(RETENTION_ENV, "90")))
    except ValueError:
        return 90


def _app_commit() -> str | None:
    """Return the fork's commit: it runs from a git checkout on the robot."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _companion_commit() -> str | None:
    """Return the companion's commit, the first word of DEPLOYED_FROM (no .git on the robot)."""
    try:
        words = (Path.home() / "reachy-companion" / "DEPLOYED_FROM").read_text().split()
        return words[0] if words else None
    except OSError:
        return None


class WireLog:
    """The recorder: ``tap()`` a connection at its open; the session ends with it."""

    def __init__(self, path: str | os.PathLike[str], *, retention: int | None = None,
                 queue_max: int = QUEUE_MAX, batch_s: float = BATCH_S, open_retry_s: float = OPEN_RETRY_S) -> None:
        """Start the writer, which opens (or creates) the database on its own thread."""
        self.path = Path(path)
        self.retention_days = retention or retention_days()
        self.batch_s = batch_s
        self.open_retry_s = open_retry_s
        self._q: queue.Queue[tuple[Any, ...]] = queue.Queue(maxsize=queue_max)
        self.dropped = 0  # events the full queue turned away
        self.failed = 0  # batches a database error rolled back
        self._warned: set[str] = set()
        self._next_session = 0
        self._lock = threading.Lock()
        self._commits = (_app_commit(), _companion_commit())  # once per process
        self._thread = threading.Thread(target=self._run, name="wire-log", daemon=True)
        self._thread.start()

    # --- the tap (event loop side): a reference on the queue, nothing else -------------

    def _put(self, entry: tuple[Any, ...]) -> None:
        try:
            self._q.put_nowait(entry)
        except queue.Full:
            self.dropped += 1

    def open_session(self, *, backend_host: str | None = None, profile: str | None = None,
                     source: str = SOURCE) -> int:
        """Start a session row and return its local number."""
        with self._lock:
            self._next_session += 1
            sid = self._next_session
        self._put(("open", sid, time.time(), backend_host, profile, source))
        return sid

    def close_session(self, sid: int) -> None:
        """Mark a session closed (twice is harmless)."""
        self._put(("close", sid, time.time()))

    def out(self, sid: int, kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Note a call sent on the connection."""
        if kind not in NOT_RECORDED:
            self._put(("out", sid, time.time(), kind, args, kwargs))

    def received(self, sid: int, event: Any) -> None:
        """Note an event received."""
        self._put(("in", sid, time.time(), event))

    def mark(self, sid: int, kind: str, data: dict[str, Any]) -> None:
        """Note something the app decided about the session (a meta row), such as a drop."""
        self._put(("mark", sid, time.time(), kind, data))

    def wrap(self, conn: Any, **meta: Any) -> Tapped:
        """Return ``conn`` wrapped so every send and receive is recorded."""
        return Tapped(conn, self, self.open_session(**meta))

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until what is queued is written or given up (tests, shutdown)."""
        self._put(("flush",))
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)

    # --- the writer thread --------------------------------------------------------------

    def _warn(self, kind: str, why: str) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            logger.warning("wire log: %s (further ones are counted, not logged)", why)

    def _connect(self) -> sqlite3.Connection | None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            if new:
                db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA busy_timeout=5000")
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            return db
        except Exception as exc:
            self._warn("open", f"cannot open {self.path}: {exc}; retrying every {self.open_retry_s:.0f} s")
            return None

    def _take(self) -> list[tuple[Any, ...]]:
        """Wait for the first entry, let a batch gather for batch_s, then take it all."""
        try:
            batch = [self._q.get(timeout=self.batch_s)]
        except queue.Empty:
            return []
        if batch[0][0] != "flush":
            time.sleep(self.batch_s)
        while True:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                return batch

    def _run(self) -> None:
        db = None
        self._sessions: dict[int, int] = {}  # local number -> row id
        self._audio_out: dict[tuple[int, str | None], int] = {}
        pruned_at = 0.0
        reported = 0
        next_open = 0.0
        while True:
            batch = self._take()
            try:
                if db is None and time.monotonic() >= next_open:
                    db = self._connect()
                    next_open = time.monotonic() + self.open_retry_s
                if db is None:
                    self.failed += bool(batch)  # the batch is lost, and counted
                    continue
                opened: list[int] = []
                wrote = False
                try:
                    db.execute("BEGIN")
                    for entry in batch:
                        wrote |= self._write(db, entry, opened)
                    if self.dropped > reported and self._sessions:
                        row = next(iter(self._sessions.values()))
                        db.execute(
                            "INSERT INTO events (session_id, t, dir, type, size) VALUES (?,?,?,?,?)",
                            (row, time.time(), "meta", "wire.dropped", self.dropped - reported),
                        )
                        reported, wrote = self.dropped, True
                    db.execute("COMMIT" if wrote else "ROLLBACK")
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        db.execute("ROLLBACK")
                    for sid in opened:  # their rows were rolled back with the batch
                        self._sessions.pop(sid, None)
                    self.failed += 1
                    self._warn(type(exc).__name__, f"a batch of {len(batch)} not written: {exc}")
                if time.time() - pruned_at > 86400:
                    pruned_at = time.time()
                    try:
                        self._prune(db)
                    except Exception as exc:
                        self._warn("prune", f"prune failed: {exc}")
            finally:
                for _ in batch:
                    self._q.task_done()

    def _prune(self, db: sqlite3.Connection) -> None:
        cutoff = time.time() - self.retention_days * 86400
        db.execute("DELETE FROM events WHERE t < ?", (cutoff,))
        db.execute("DELETE FROM usage WHERE t < ?", (cutoff,))
        db.execute(
            "DELETE FROM sessions WHERE opened_at < ? AND id NOT IN (SELECT DISTINCT session_id FROM events)",
            (cutoff,),
        )
        db.execute(
            "DELETE FROM blobs WHERE hash NOT IN (SELECT blob FROM events WHERE blob IS NOT NULL"
            " UNION SELECT instr_blob FROM events WHERE instr_blob IS NOT NULL"
            " UNION SELECT tools_blob FROM events WHERE tools_blob IS NOT NULL)"
        )
        db.execute("PRAGMA incremental_vacuum")

    def _write(self, db: sqlite3.Connection, entry: tuple[Any, ...], opened: list[int]) -> bool:
        """Write one entry; return whether a row changed."""
        kind = entry[0]
        if kind == "flush":
            return False
        if kind == "open":
            _, sid, t, host, profile, source = entry
            cur = db.execute(
                "INSERT INTO sessions (source, opened_at, backend_host, app_commit, companion_commit, profile)"
                " VALUES (?,?,?,?,?,?)",
                (source, t, host, *self._commits, profile),
            )
            self._sessions[sid] = cur.lastrowid
            opened.append(sid)
            return True
        row = self._sessions.get(entry[1])
        if row is None:
            return False
        if kind == "close":
            db.execute("UPDATE sessions SET closed_at=? WHERE id=?", (entry[2], row))
            self._sessions.pop(entry[1], None)
            for key in [k for k in self._audio_out if k[0] == entry[1]]:
                del self._audio_out[key]
            return True
        if kind == "mark":
            _, _, t, name, data = entry
            self._row(db, row, t, "meta", name, data)
            return True
        if kind == "out":
            _, sid, t, name, args, kwargs = entry
            data: Any = to_jsonable(kwargs) if kwargs else {}
            if args:
                data = {"args": to_jsonable(list(args)), **(data if isinstance(data, dict) else {})}
            self._row(db, row, t, "out", name, data)
            return True
        _, sid, t, event = entry
        etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None) or "?"
        if etype.endswith("audio.delta"):
            key = (sid, getattr(event, "response_id", None))
            self._audio_out[key] = self._audio_out.get(key, 0) + b64_bytes(getattr(event, "delta", None))
            return False
        if etype.endswith(".delta"):
            return False
        data = to_jsonable(event)
        self._row(db, row, t, "in", etype, data)
        if etype == "response.done":
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
        return True

    @staticmethod
    def _blob(db: sqlite3.Connection, raw: bytes) -> str:
        h = hashlib.sha256(raw).hexdigest()
        if db.execute("SELECT 1 FROM blobs WHERE hash=?", (h,)).fetchone() is None:
            db.execute("INSERT INTO blobs (hash, size, z) VALUES (?,?,?)", (h, len(raw), zlib.compress(raw, ZLEVEL)))
        return h

    @staticmethod
    def _dumps(data: Any) -> bytes:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()

    def _row(self, db: sqlite3.Connection, row: int, t: float, direction: str, kind: str, data: Any,
             size: int | None = None, ids: tuple[str | None, str | None, str | None] | None = None) -> None:
        data = strip_media(data)
        item_id, response_id, call_id = ids or _ids(data)
        instr = tools = None
        session = data.get("session") if isinstance(data, dict) else None
        if isinstance(session, dict):  # the instructions and tools: once per distinct text
            session = dict(session)
            if isinstance(session.get("instructions"), str):
                instr = self._blob(db, session.pop("instructions").encode())
            if isinstance(session.get("tools"), list):
                tools = self._blob(db, self._dumps(session.pop("tools")))
            data = {**data, "session": session}
        raw = self._dumps(data)
        z: bytes | None = zlib.compress(raw, ZLEVEL)
        blob = None
        if len(raw) >= BLOB_MIN:
            blob, z = self._blob(db, raw), None
        db.execute(
            "INSERT INTO events (session_id, t, dir, type, item_id, response_id, call_id, size, z, blob,"
            " instr_blob, tools_blob) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row, t, direction, kind, item_id, response_id, call_id, len(raw) if size is None else size,
             z, blob, instr, tools),
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

    def wire_mark(self, kind: str, **data: Any) -> None:
        """Note an app decision about this session in the log (``context.dropped``)."""
        self._log.mark(self._sid, kind, data)

    def __aiter__(self) -> Any:
        """Iterate the connection's events, each one recorded."""
        return self._events()

    async def _events(self) -> Any:
        async for event in self._conn:
            self._log.received(self._sid, event)
            yield event


_LOG: WireLog | None = None


def from_env() -> WireLog | None:
    """Return the process's wire log, started on first use, or None when it is off."""
    global _LOG
    path = os.environ.get(ENV, "").strip()
    if not path or path.lower() == "off":
        return None
    path = os.path.expanduser(path)
    if _LOG is None or str(_LOG.path) != path:
        _LOG = WireLog(path)
    return _LOG


@contextlib.asynccontextmanager
async def tap(conn: Any, **meta: Any) -> AsyncIterator[Any]:
    """Yield ``conn`` recorded while the block runs (or as is when the log is off); close its session after."""
    log = from_env()
    if log is None:
        yield conn
        return
    tapped = log.wrap(conn, **meta)
    try:
        yield tapped
    finally:
        log.close_session(tapped._sid)
