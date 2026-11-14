"""The wire log (the companion's W1, 11-14): what goes in and out of the connection, no media."""

from __future__ import annotations
import json
import time
import zlib
import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pytest

from reachy_mini_conversation_app import wire_log
from reachy_mini_conversation_app.wire_log import WireLog, strip_media


B64 = "QUJD" * 300  # 900 bytes decoded


def _rows(path: Path) -> list[tuple[Any, ...]]:
    db = sqlite3.connect(path)
    try:
        return db.execute("SELECT dir, type, item_id, response_id, size, z, blob FROM events ORDER BY id").fetchall()
    finally:
        db.close()


def _payload(path: Path, row: tuple[Any, ...]) -> Any:
    z = row[5]
    if z is None:
        db = sqlite3.connect(path)
        z = db.execute("SELECT z FROM blobs WHERE hash=?", (row[6],)).fetchone()[0]
        db.close()
    return json.loads(zlib.decompress(z))


class _Conn:
    """A fake realtime connection: namespaces of async calls and a scripted event stream."""

    def __init__(self, events: list[Any]) -> None:
        self.calls: list[tuple[str, Any]] = []

        def ns(name: str, *methods: str) -> SimpleNamespace:
            async def make(m: str, *a: Any, **kw: Any) -> None:
                self.calls.append((f"{name}.{m}", kw))
            return SimpleNamespace(**{m: (lambda *a, _m=m, **kw: make(_m, *a, **kw)) for m in methods})

        self.session = ns("session", "update")
        self.conversation = SimpleNamespace(item=ns("conversation.item", "create", "delete", "truncate"))
        self.response = ns("response", "create", "cancel")
        self.input_audio_buffer = ns("input_audio_buffer", "append")
        self._events = events

    def __aiter__(self) -> Any:
        """Iterate the scripted events."""

        async def gen() -> Any:
            for e in self._events:
                yield e
        return gen()


def _ev(type_: str, **kw: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **kw)


def test_media_becomes_its_hash_and_size() -> None:
    """The owner, 11-14: no media at all, camera or channel; a placeholder with hash and size."""
    item = {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{B64}"},
        {"type": "input_image", "image_url": "https://cdn.example/pic.png"},
    ]}
    out = strip_media(item)
    text, cam, chan = out["content"]
    assert text == {"type": "input_text", "text": "look"}
    assert cam["image_url"]["media"] == "image/jpeg" and cam["image_url"]["bytes"] == 900
    assert len(cam["image_url"]["sha256"]) == 16
    assert set(chan["image_url"]) == {"media", "sha256", "bytes"}
    assert "base64" not in json.dumps(out)
    assert item["content"][1]["image_url"].startswith("data:")  # the original is untouched


def test_every_send_and_receive_is_recorded_with_audio_counted(tmp_path: Path) -> None:
    """Out calls and in events become rows; audio and deltas are counted; usage has its row."""
    path = tmp_path / "wire.db"
    log = WireLog(path)
    big = "You are Reachy. " * 4000
    events = [
        _ev("session.updated"),
        _ev("input_audio_buffer.committed", item_id="u1"),
        _ev("response.created", response={"id": "r1"}),
        _ev("response.output_audio.delta", response_id="r1", delta=B64),
        _ev("response.output_audio.delta", response_id="r1", delta=B64),
        _ev("response.output_audio_transcript.delta", response_id="r1", delta="Hel"),
        _ev("response.done", response={"id": "r1", "usage": {"input_tokens": 9000, "output_tokens": 12,
                                                              "input_token_details": {"cached_tokens": 8000}}}),
    ]
    conn = log.wrap(_Conn(events), backend_host="wss://x", profile="companion")

    async def run() -> None:
        await conn.session.update(session={"instructions": big})
        await conn.session.update(session={"instructions": big})
        await conn.input_audio_buffer.append(audio=B64)
        await conn.input_audio_buffer.append(audio=B64)
        await conn.conversation.item.create(item={"id": "n1", "type": "message", "role": "system",
                                                  "content": [{"type": "input_text", "text": "a note"}]})
        await conn.response.create(response={"metadata": {"companion": "x"}})
        async for _ in conn:
            pass

    asyncio.run(run())
    log.flush()
    rows = _rows(path)
    kinds = [(r[0], r[1]) for r in rows]
    assert kinds == [
        ("out", "session.update"), ("out", "session.update"),
        ("out", "conversation.item.create"), ("out", "response.create"),
        ("in", "session.updated"), ("in", "input_audio_buffer.committed"), ("meta", "audio.in"),
        ("in", "response.created"), ("in", "response.done"), ("meta", "audio.out"),
    ]
    audio_in = rows[6]
    assert audio_in[2] == "u1" and audio_in[4] == 1800
    audio_out = rows[9]
    assert audio_out[3] == "r1" and audio_out[4] == 1800
    assert rows[2][2] == "n1" and _payload(path, rows[2])["item"]["content"][0]["text"] == "a note"
    db = sqlite3.connect(path)
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1  # the instructions, once
    assert rows[0][6] == rows[1][6] and rows[0][5] is None
    assert db.execute("SELECT input, cached, output FROM usage").fetchall() == [(9000, 8000, 12)]
    s = db.execute("SELECT backend_host, profile, closed_at IS NOT NULL FROM sessions").fetchall()
    assert s == [("wss://x", "companion", 1)]


def test_a_full_queue_drops_and_counts_never_blocks() -> None:
    """The tap never waits on the writer: past the bound an event is dropped and counted."""

    class Stalled(WireLog):
        def _run(self) -> None:  # a writer that never drains
            threading.Event().wait(0.5)

    log = Stalled("/nonexistent/wire.db", queue_max=2)
    t0 = time.monotonic()
    for _ in range(10):
        log.received(1, _ev("response.created"))
    assert time.monotonic() - t0 < 0.1
    assert log.dropped == 8


def test_a_database_that_cannot_open_turns_the_log_off(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A database error is one warning; the tap goes on accepting calls and records nothing."""
    blocker = tmp_path / "file"
    blocker.write_text("")
    log = WireLog(blocker / "wire.db")
    caplog.set_level("WARNING", logger=wire_log.__name__)
    log._thread.join(timeout=10)  # the writer ends once it has turned the log off
    assert not log.enabled
    log.received(1, _ev("x"))  # no raise
    assert "wire log off" in caplog.text


def test_rows_past_retention_are_pruned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At start and daily, rows older than 90 days go, with blobs nothing uses."""
    path = tmp_path / "wire.db"
    log = WireLog(path)
    old = time.time() - 91 * 86400
    monkeypatch.setattr(wire_log.time, "time", lambda: old)
    sid = log.open_session()
    log.out(sid, "session.update", (), {"session": {"instructions": "x" * 40000}})
    monkeypatch.undo()
    log.out(sid, "response.create", (), {})
    log.flush()
    db = sqlite3.connect(path)
    log._prune(db)
    assert [r[0] for r in db.execute("SELECT type FROM events")] == ["response.create"]
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0


def test_off_unless_its_path_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests and a Mac run record nothing: the log starts only when its env path is set."""
    monkeypatch.delenv(wire_log.ENV, raising=False)
    assert wire_log.from_env() is None
    monkeypatch.setenv(wire_log.ENV, "off")
    assert wire_log.from_env() is None


@pytest.mark.asyncio
async def test_the_handlers_session_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real session path, its connect wrapped: its session.update and events reach the log."""
    from test_huggingface_realtime import _FakeEvent, _ordering_client, _session_handler

    path = tmp_path / "wire.db"
    monkeypatch.setenv(wire_log.ENV, str(path))
    monkeypatch.setattr(wire_log, "_LOG", None)
    sent: list[tuple[str, Any]] = []
    handler = _session_handler(monkeypatch, ())
    handler.client = _ordering_client(
        (_FakeEvent("conversation.item.input_audio_transcription.completed", item_id="u1", transcript="hello"),),
        sent, handler,
    )
    await handler._run_realtime_session()
    log = wire_log.from_env()
    assert log is not None
    log.flush()
    kinds = [(r[0], r[1]) for r in _rows(path)]
    assert ("out", "session.update") in kinds
    assert ("in", "conversation.item.input_audio_transcription.completed") in kinds
    monkeypatch.setattr(wire_log, "_LOG", None)
