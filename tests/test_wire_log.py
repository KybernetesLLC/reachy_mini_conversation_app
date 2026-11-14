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
FAST = {"batch_s": 0.05}


def _rows(path: Path) -> list[tuple[Any, ...]]:
    db = sqlite3.connect(path)
    try:
        return db.execute(
            "SELECT dir, type, item_id, response_id, size, z, blob, instr_blob FROM events ORDER BY id"
        ).fetchall()
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
    """The owner, 11-14: no media at all, camera, channel or his own audio; hash and size."""
    item = {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{B64}"},
        {"type": "input_image", "image_url": "https://cdn.example/pic.png"},
        {"type": "output_audio", "audio": B64, "transcript": "hello"},
        {"type": "input_text", "text": json.dumps({"pic": f"data:image/png;base64,{B64}"})},
    ]}
    out = strip_media(item)
    text, cam, chan, his, inline = out["content"]
    assert text == {"type": "input_text", "text": "look"}
    assert cam["image_url"]["media"] == "image/jpeg" and cam["image_url"]["bytes"] == 900
    assert len(cam["image_url"]["sha256"]) == 16
    assert set(chan["image_url"]) == {"media", "sha256", "bytes"}
    assert set(his["audio"]) == {"media", "sha256", "bytes"} and his["transcript"] == "hello"
    assert "[media image/png" in inline["text"]
    assert B64[:40] not in json.dumps(out)
    assert item["content"][1]["image_url"].startswith("data:")  # the original is untouched


def test_every_send_and_receive_is_recorded_with_audio_counted(tmp_path: Path) -> None:
    """Out calls and in events become rows; the mic stream and deltas are not kept; usage has its row."""
    path = tmp_path / "wire.db"
    log = WireLog(path, **FAST)
    big = "You are Reachy. " * 4000
    events = [
        _ev("session.updated", event_id="e1", session={"instructions": big}),
        _ev("input_audio_buffer.committed", item_id="u1"),
        _ev("response.created", response={"id": "r1"}),
        _ev("response.output_audio.delta", response_id="r1", delta=B64),
        _ev("response.output_audio.delta", response_id="r1", delta=B64),
        _ev("response.output_audio_transcript.delta", response_id="r1", delta="Hel"),
        _ev("conversation.item.done", item={"id": "a1", "type": "message", "role": "assistant",
                                            "content": [{"type": "output_audio", "audio": B64, "transcript": "Hello"}]}),
        _ev("response.done", response={"id": "r1", "usage": {"input_tokens": 9000, "output_tokens": 12,
                                                              "input_token_details": {"cached_tokens": 8000}}}),
    ]

    async def run() -> None:
        async with wire_log.tap(_Conn(events)) as conn:
            pass  # off: the env path is unset
        assert isinstance(conn, _Conn)
        tapped = log.wrap(_Conn(events), backend_host="wss://x", profile="companion")
        await tapped.session.update(session={"instructions": big, "tools": [{"name": "express"}]})
        await tapped.session.update(session={"instructions": big, "tools": [{"name": "express"}]})
        await tapped.input_audio_buffer.append(audio=B64)
        await tapped.conversation.item.create(item={"id": "n1", "type": "message", "role": "system",
                                                    "content": [{"type": "input_text", "text": "a note"}]})
        await tapped.response.create(response={"metadata": {"companion": "x"}})
        async for _ in tapped:
            pass
        log.close_session(tapped._sid)

    asyncio.run(run())
    log.flush()
    rows = _rows(path)
    assert [(r[0], r[1]) for r in rows] == [
        ("out", "session.update"), ("out", "session.update"),
        ("out", "conversation.item.create"), ("out", "response.create"),
        ("in", "session.updated"), ("in", "input_audio_buffer.committed"),
        ("in", "response.created"), ("in", "conversation.item.done"),
        ("in", "response.done"), ("meta", "audio.out"),
    ]
    assert rows[9][3] == "r1" and rows[9][4] == 1800
    assert rows[2][2] == "n1" and _payload(path, rows[2])["item"]["content"][0]["text"] == "a note"
    assert rows[0][7] == rows[1][7] == rows[4][7]  # the instructions: one copy, sent twice and echoed
    assert B64[:40] not in json.dumps(_payload(path, rows[7]))
    db = sqlite3.connect(path)
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 2  # the instructions, the tools
    assert db.execute("SELECT input, cached, output FROM usage").fetchall() == [(9000, 8000, 12)]
    s = db.execute("SELECT source, backend_host, profile, closed_at IS NOT NULL FROM sessions").fetchall()
    assert s == [("realtime", "wss://x", "companion", 1)]
    assert db.execute("PRAGMA user_version").fetchone()[0] == wire_log.SCHEMA_VERSION


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


def test_a_database_that_cannot_open_is_retried_never_given_up(tmp_path: Path) -> None:
    """The owner's rule (no self-disabling layers): a database that cannot open is tried again."""
    blocker = tmp_path / "dir"
    blocker.write_text("")  # a file where the directory must go
    log = WireLog(blocker / "wire.db", batch_s=0.02, open_retry_s=0.05)
    log.open_session()
    log.flush(1.0)
    assert log.failed >= 1
    blocker.unlink()
    time.sleep(0.1)
    sid = log.open_session()
    log.out(sid, "response.create", (), {})
    log.flush(2.0)
    assert [r[1] for r in _rows(blocker / "wire.db")] == ["response.create"]


def test_a_failed_batch_rolls_back_and_the_writer_goes_on(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A write error loses its batch, logs once, and the next batch is written."""
    path = tmp_path / "wire.db"
    log = WireLog(path, **FAST)
    real = log._row
    calls = {"n": 0}

    def flaky(*a: Any, **kw: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("disk I/O error")
        real(*a, **kw)

    log._row = flaky  # type: ignore[method-assign]
    sid = log.open_session()
    log.out(sid, "response.create", (), {})
    log.flush()
    sid = log.open_session()
    log.out(sid, "response.cancel", (), {})
    log.out(sid, "conversation.item.create", (), {"item": {"text": "lone \ud800 surrogate"}})
    log.flush()
    assert log.failed == 1 and "disk I/O error" in caplog.text
    assert [r[1] for r in _rows(path)] == ["response.cancel", "conversation.item.create"]
    db = sqlite3.connect(path)
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1  # the rolled-back open is gone


def test_rows_past_retention_are_pruned_and_live_blobs_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Daily, rows older than the retention go, with blobs nothing uses; a blob in use stays."""
    path = tmp_path / "wire.db"
    log = WireLog(path, **FAST)
    old = time.time() - 91 * 86400
    monkeypatch.setattr(wire_log.time, "time", lambda: old)
    sid = log.open_session()
    log.out(sid, "session.update", (), {"session": {"instructions": "old " * 5000}})
    log.out(sid, "session.update", (), {"session": {"instructions": "kept " * 5000}})
    monkeypatch.undo()
    log.out(sid, "session.update", (), {"session": {"instructions": "kept " * 5000}})
    log.flush()
    db = sqlite3.connect(path)
    log._prune(db)
    assert [r[1] for r in db.execute("SELECT id, type FROM events")] == ["session.update"]
    assert db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1


def test_retention_follows_the_companions_turn_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """One setting for how long talk is kept: the companion's turn retention."""
    monkeypatch.setenv(wire_log.RETENTION_ENV, "30")
    assert wire_log.retention_days() == 30
    monkeypatch.setenv(wire_log.RETENTION_ENV, "x")
    assert wire_log.retention_days() == 90


def test_off_unless_its_path_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests and a Mac run record nothing: the log starts only when its env path is set."""
    monkeypatch.delenv(wire_log.ENV, raising=False)
    assert wire_log.from_env() is None
    monkeypatch.setenv(wire_log.ENV, "off")
    assert wire_log.from_env() is None


@pytest.mark.asyncio
async def test_the_session_closes_when_the_block_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A session.update that raises before the receive loop still closes the session row."""
    path = tmp_path / "wire.db"
    monkeypatch.setenv(wire_log.ENV, str(path))
    monkeypatch.setattr(wire_log, "_LOG", None)
    with pytest.raises(RuntimeError):
        async with wire_log.tap(_Conn([])) as conn:
            await conn.session.update(session={"instructions": "x"})
            raise RuntimeError("session.update refused")
    log = wire_log.from_env()
    assert log is not None
    log.flush()
    db = sqlite3.connect(path)
    assert db.execute("SELECT closed_at IS NOT NULL FROM sessions").fetchall() == [(1,)]
    monkeypatch.setattr(wire_log, "_LOG", None)


@pytest.mark.asyncio
async def test_the_handlers_session_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real session path, its connect tapped: its session.update and events reach the log."""
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


def test_an_app_decision_is_a_meta_row(tmp_path: Path) -> None:
    """The handler marks where it decided the backend dropped the conversation (W2 review)."""
    path = tmp_path / "wire.db"
    log = WireLog(path, **FAST)
    tapped = log.wrap(_Conn([]))
    tapped.wire_mark("context.dropped", input_tokens=4100)
    log.flush()
    (row,) = _rows(path)
    assert row[:2] == ("meta", "context.dropped")
    assert _payload(path, row) == {"input_tokens": 4100}
