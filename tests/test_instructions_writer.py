"""The writer of his instructions (route all, Unit 4), against the companion's ConvoAll round 6.

Each test is named for the model counterexample (docs/formal/conversation-d/model.md on the
companion's arch-d-model branch) or the round-6 review item it keeps out.
"""

from __future__ import annotations
import ast
import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

from reachy_mini_conversation_app import instructions_writer as iw


SESSION = 5.0
S1 = {"version": 10.0, "row": 2, "up_to": 1000.0, "text": "Alli asked about ferries.", "cut": "u2"}
S2 = {"version": 20.0, "row": 4, "up_to": 2000.0, "text": "Then about the bus.", "cut": "u3"}
HI = "(a thought from Alli, answered) hi / (your reply) hello"


class Conn:
    """A session: every session.update and item delete, in the order sent."""

    def __init__(self) -> None:
        self.wire: list[tuple[str, object]] = []

        async def update(session: dict) -> None:
            self.wire.append(("update", session.get("instructions")))

        async def delete(item_id: str) -> None:
            self.wire.append(("delete", item_id))

        self.session = SimpleNamespace(update=update)
        self.conversation = SimpleNamespace(item=SimpleNamespace(delete=delete))


class Prune:
    def __init__(self, conn: Conn) -> None:
        self.calls: list[tuple] = []
        self.conn = conn

    async def delete_before(self, conn, item_id, *, through=False, up_to=None) -> int:
        self.calls.append((item_id, through, up_to))
        conn.wire.append(("delete_before", item_id))
        return 1


def _part(name: str, version: int, value: object, session: float | None = SESSION) -> SimpleNamespace:
    return SimpleNamespace(name=name, version=version, value=value, session=session, expires_at=None)


def _block(version: int, text: str, *, s=None, top=1, through=None, catch_up=None) -> SimpleNamespace:
    value = {"text": text, "top": top, "through": through, "s": s, "catch_up": catch_up}
    return _part("block", version, value)


def _writer(batch_s: float = 100.0) -> tuple[iw.InstructionsWriter, Conn, Prune, list[str]]:
    conn = Conn()
    prune = Prune(conn)
    measured: list[str] = []
    writer = iw.InstructionsWriter(
        prune, connection=lambda: conn, batch_s=batch_s, on_sent=lambda: measured.append("reset")
    )
    return writer, conn, prune, measured


async def _open(writer: iw.InstructionsWriter, conn: Conn, header: dict | None = None, held=None) -> None:
    text = writer.connect(SESSION, "BASE", "## Your memory\nold turns", header or {"top": 1}, held or {})
    await writer.open(conn, {"type": "realtime", "instructions": text})
    conn.wire.clear()


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def test_the_composition_is_the_stable_base_then_the_parts() -> None:
    block = iw.Block("## Your memory\nturns", 3, None, None, 1, 0.0)
    view = iw.View("BASE", SESSION, block, S1, 0.0, ("(a thought from Alli, answered) hi / (your reply) hello",))
    text = iw.compose(view)
    assert text.startswith("BASE\n\n## Your memory\nturns\n\n")
    assert f"{iw.S_HEADING}\nAlli asked about ferries." in text
    assert text.endswith(f"{iw.MESSAGES_HEADING}\n(a thought from Alli, answered) hi / (your reply) hello")


@pytest.mark.asyncio
async def test_fjoinbatch_s_joins_at_once_in_one_send_before_its_deletes() -> None:
    """Fjoinbatch and a_delfirst_race: S goes at once (never batched), and its deletes follow
    the write that carries it, in one coroutine; deletes first would leave a copy with neither."""
    writer, conn, prune, measured = _writer()
    await _open(writer, conn)
    writer.parts_changed({"summary": _part("summary", 2, S1, session=None)})
    await _settle()
    assert conn.wire[0][0] == "update" and "Alli asked about ferries." in conn.wire[0][1]
    assert conn.wire[1] == ("delete_before", "u2")
    assert prune.calls == [("u2", False, 1000.0)]  # the time cut's fallback travels with it
    # a join starts the drop detector's measure again (item 3), and again after its deletes
    assert measured == ["reset", "reset"]


@pytest.mark.asyncio
async def test_fresendtask_one_owner_sends_are_one_at_a_time_and_end_current() -> None:
    """Fresendtask/OneOwner: two sends started together never interleave; the session ends with
    the composition as it is now, written once."""
    writer, conn, _, _ = _writer()
    await _open(writer, conn)
    writer.view = writer.view.__class__(**{**writer.view.__dict__, "base": "NEW"})
    await asyncio.gather(writer.send("one"), writer.send("two"))
    assert [k for k, _ in conn.wire] == ["update"] and conn.wire[0][1] == writer.current()


@pytest.mark.asyncio
async def test_fstaleparts_an_older_or_another_sessions_block_is_never_taken() -> None:
    """Fstaleparts: a block part from another session, or not newer than the one held, is not
    taken; a cleared S (a newer version with no text) leaves no S in the composition at once."""
    writer, conn, _, _ = _writer()
    await _open(writer, conn)
    writer.parts_changed({"block": _block(5, "## Your memory\nfive")})
    writer.parts_changed({"block": _block(4, "## Your memory\nfour")})
    writer.parts_changed({"block": _part("block", 9, {"text": "other session"}, session=4.0)})
    assert "five" in writer.current() and "four" not in writer.current()
    writer.parts_changed({"summary": _part("summary", 6, S1, session=None)})
    await _settle()
    cleared = {"version": 15.0, "row": None, "text": None, "cut": None}
    writer.parts_changed({"summary": _part("summary", 7, cleared, session=None)})
    await _settle()
    assert "Alli asked" not in conn.wire[-1][1]  # ClearedGone: the clear went at once


@pytest.mark.asyncio
async def test_fblockskew_the_s_the_block_names_joins_with_it() -> None:
    """a_blockskew_GD: the block is cut at S2 while the link still has S1; the composition
    carries S2, so neither S2's turns nor S2 are missing."""
    writer, conn, _, _ = _writer()
    await _open(writer, conn, held={"summary": _part("summary", 1, S1, session=None)})
    writer.parts_changed({"summary": _part("summary", 1, S1, session=None), "block": _block(3, "cut at S2", s=S2)})
    await _settle()
    assert "Then about the bus." in writer.current() and "Alli asked" not in writer.current()


@pytest.mark.asyncio
async def test_a_nonewer_an_older_s_on_the_link_never_replaces_the_blocks() -> None:
    """a_nonewer_GL: S arrives by both paths; a link S older than the one the block named (a
    snapshot still in flight) is not taken (rule 8: S's own version decides)."""
    writer, conn, _, _ = _writer()
    await _open(writer, conn, header={"top": 1, "s": S2})
    writer.parts_changed({"summary": _part("summary", 9, S1, session=None)})
    assert "Then about the bus." in writer.current() and "Alli asked" not in writer.current()


@pytest.mark.asyncio
async def test_fstaleblock_each_block_part_replaces_the_block_cut_at_its_own_s() -> None:
    writer, conn, _, _ = _writer()
    await _open(writer, conn, header={"top": 1, "s": S1})
    writer.parts_changed({"block": _block(2, "## Your memory\ncut at S2", s=S2)})
    await _settle()
    assert "cut at S2" in writer.current() and "old turns" not in writer.current()
    assert "Then about the bus." in writer.current()


@pytest.mark.asyncio
async def test_a_nocatchup_and_a_catchupatend_the_catch_up_goes_at_once() -> None:
    """a_nocatchup_GDne and a_catchupatend_GDne: the companion marks the block that holds more
    than the read at the open; it goes at once on the handler's loop, not at the next reply's
    end. Its decision, not the app's (round-6 review, item 5): a higher row with no mark batches."""
    writer, conn, _, measured = _writer(batch_s=100.0)
    await _open(writer, conn)
    writer.parts_changed({"block": _block(2, "## Your memory\nmore", top=5)})
    await _settle()
    assert conn.wire == []  # batched: no catch-up mark
    writer.parts_changed({"block": _block(3, "## Your memory\nmore still", top=6, catch_up=6)})
    await _settle()
    assert conn.wire == [("update", writer.current())] and measured == ["reset"]


@pytest.mark.asyncio
async def test_a_noctxdel_a_drops_items_go_after_the_block_that_holds_them() -> None:
    """a_noctxdel_GD (rule 9): after a reported drop, the block bounded at the item goes at once,
    then the items through that one; nothing is deleted before the block names it."""
    writer, conn, prune, _ = _writer()
    await _open(writer, conn)
    writer.report_drop("u7")
    writer.parts_changed({"block": _block(2, "## Your memory\nold", top=1)})
    await writer.send("a part changed")
    assert ("delete_before", "u7") not in conn.wire
    writer.parts_changed({"block": _block(3, "## Your memory\nthrough u7", top=7, through="u7")})
    await _settle()
    kinds = [k for k, _ in conn.wire]
    assert kinds[-2:] == ["update", "delete_before"] and prune.calls[-1] == ("u7", True, None)
    assert writer._drop is None


@pytest.mark.asyncio
async def test_a_done_write_and_a_batched_write_never_reset_the_drop_measure() -> None:
    """Round-6 review, item 3: only a write that changes the block or S at once (startup, the
    catch-up, the S join) starts the measure again, or a drop would never be seen."""
    writer, conn, _, measured = _writer(batch_s=0.01)
    await _open(writer, conn)
    writer.parts_changed({"messages": _part("messages", 2, [HI], session=None)})
    await asyncio.sleep(0.05)  # an answered message: at once, with no measure reset
    writer.parts_changed({"block": _block(2, "## Your memory\nmood changed")})
    await writer.at_done()
    assert [k for k, _ in conn.wire] == ["update", "update"] and measured == []


@pytest.mark.asyncio
async def test_a_change_goes_at_a_replys_done_and_a_quiet_done_sends_nothing() -> None:
    writer, conn, _, _ = _writer()
    await _open(writer, conn)
    await writer.at_done()
    assert conn.wire == []
    writer.parts_changed({"block": _block(2, "## Your memory\nnew")})
    assert writer.pending()
    await writer.at_done()
    assert conn.wire == [("update", writer.current())] and not writer.pending()


@pytest.mark.asyncio
async def test_a_new_session_spends_s_deletes_and_drops_a_pending_drop() -> None:
    """A new session has no items: the S it opens with deletes nothing, and the last session's
    drop is not carried."""
    writer, conn, prune, _ = _writer()
    writer.report_drop("u1")
    await _open(writer, conn, header={"top": 1, "s": S1})
    await writer.send("x")
    assert prune.calls == [] and writer._drop is None


@pytest.mark.asyncio
async def test_a_failed_write_is_logged_never_raised_and_deletes_nothing() -> None:
    writer, conn, prune, _ = _writer()
    await _open(writer, conn)

    async def fail(session: dict) -> None:
        raise RuntimeError("closed")

    conn.session.update = fail
    writer.parts_changed({"summary": _part("summary", 2, S1, session=None)})
    await _settle()
    assert prune.calls == [] and writer.pending()


def test_nothing_else_sends_his_session_instructions() -> None:
    """Rule 4 (OneOwner): only the writer sends session instructions. Any other session.update
    in the app is the voice change's, and carries none."""
    src = Path(iw.__file__).parent
    others = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "instructions_writer.py":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                func = getattr(node, "func", None)
                if (
                    isinstance(node, ast.Call)
                    and isinstance(func, ast.Attribute)
                    and func.attr == "update"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "session"
                ):
                    segment = ast.get_source_segment(text, node) or ""
                    others.append((path.name, fn.name, "instructions" in segment))
    assert others == [("huggingface_realtime.py", "change_voice", False)]


@pytest.mark.asyncio
async def test_batched_writes_keep_a_floor_and_at_once_writes_do_not(caplog) -> None:
    """The Units 3+4 review, finding 2: a batched change goes at most once every gap_s; an S join
    is not held by it. The send log counts the writes in the last minute."""
    conn = Conn()
    writer = iw.InstructionsWriter(Prune(conn), connection=lambda: conn, batch_s=0.01, gap_s=0.3)
    await _open(writer, conn)
    writer.parts_changed({"block": _block(2, "## Your memory\ntwo")})
    await asyncio.sleep(0.05)
    assert len(conn.wire) == 1  # the first batched write: none before it
    writer.parts_changed({"block": _block(3, "## Your memory\nthree")})
    await asyncio.sleep(0.1)
    assert len(conn.wire) == 1  # held by the floor
    with caplog.at_level("INFO", logger=iw.__name__):
        writer.parts_changed({"summary": _part("summary", 4, S1, session=None)})
        await _settle()
    assert conn.wire[1][0] == "update" and "Alli asked" in conn.wire[1][1]  # S joins at once
    assert "2 write(s) in the last minute" in caplog.text
    await asyncio.sleep(0.4)
    assert [k for k, _ in conn.wire].count("update") == 2  # the batch found nothing left to send


def test_a_reported_drop_with_no_block_is_let_go() -> None:
    """The Units 3+4 review, finding 3: a companion restarted mid-session publishes no block; the
    wait ends after DROP_WAIT_S, so trims and drops are not stalled for the session."""
    clock = {"t": 0.0}
    conn = Conn()
    writer = iw.InstructionsWriter(Prune(conn), connection=lambda: conn, clock=lambda: clock["t"])
    writer.report_drop("u4")
    clock["t"] = iw.DROP_WAIT_S - 1
    assert writer.drop_pending()
    clock["t"] = iw.DROP_WAIT_S + 1
    assert not writer.drop_pending() and writer._drop is None


# --- Unit 6: the recent text messages part --------------------------------------------------


def _messages(version: int, lines: list[str] | None) -> SimpleNamespace:
    return _part("messages", version, lines, session=None)


@pytest.mark.asyncio
async def test_the_messages_part_is_in_the_composition_at_once_and_never_in_the_slot() -> None:
    """An answered message is the companion's part, sent at once, in the instructions, never an item."""
    writer, conn, _, measured = _writer()  # a batch that never fires: only an at-once send goes
    await _open(writer, conn)
    writer.parts_changed({"messages": _messages(2, [HI])})
    await _settle()
    assert [k for k, _ in conn.wire] == ["update"]
    assert measured == []  # the drop measure is not started again
    assert conn.wire[0][1].endswith(f"{iw.MESSAGES_HEADING}\n{HI}")


def test_only_lines_of_text_are_taken_from_the_part() -> None:
    """A blank or a non-text entry in the part is never a line he is told."""
    assert iw.part_messages(_messages(2, [HI, "", "   ", 3, None, ["x"]])) == (HI,)
    assert iw.part_messages(_messages(2, {"lines": [HI]})) == ()
    assert iw.part_messages(None) == ()


@pytest.mark.asyncio
async def test_the_messages_part_survives_a_reconnects_snapshot_and_a_new_session() -> None:
    """A snapshot keeps the part, and a new session's composition holds it (the talk's part)."""
    from reachy_mini_conversation_app import parts as parts_mod

    record = parts_mod.Parts(held={})
    wire = {"name": "messages", "version": 3, "value": [HI], "session": None, "expires_at": None}
    record, _, _ = parts_mod.apply(record, {"epoch": "e1", "snapshot": False, "parts": [wire]})
    record, _, _ = parts_mod.apply(record, {"epoch": "e1", "snapshot": True, "version": 3, "parts": [wire]})
    assert record.held["messages"].value == [HI]
    writer, conn, _, _ = _writer()
    await _open(writer, conn)
    text = writer.connect(SESSION + 1, "BASE", "## Your memory\nold turns", {"top": 1}, dict(record.held))
    assert text.endswith(f"{iw.MESSAGES_HEADING}\n{HI}")


@pytest.mark.asyncio
async def test_a_forget_clears_the_messages_at_once() -> None:
    """A clear (a forget, the talk's end) goes at once, as S's does (ClearedGone)."""
    writer, conn, _, _ = _writer()
    await _open(writer, conn, held={"messages": _messages(2, [HI])})
    assert HI in writer.sent
    writer.parts_changed({"messages": _messages(3, None)})
    await _settle()
    assert [k for k, _ in conn.wire] == ["update"] and HI not in conn.wire[0][1]
    assert iw.MESSAGES_HEADING not in conn.wire[0][1]


def test_an_older_messages_version_is_ignored() -> None:
    """A late resend of an older version, after the clear, never brings the lines back."""
    from reachy_mini_conversation_app import parts as parts_mod

    def wire(version: int, value: object) -> dict:
        return {"name": "messages", "version": version, "value": value, "session": None, "expires_at": None}

    record, _, _ = parts_mod.apply(
        parts_mod.Parts(held={}), {"epoch": "e1", "snapshot": True, "version": 4, "parts": [wire(4, None)]}
    )
    record, _, changes = parts_mod.apply(record, {"epoch": "e1", "snapshot": False, "parts": [wire(3, [HI])]})
    assert changes == ["messages v3 ignored (older)"]
    block = iw.Block("", 0, None, None, 0, 0.0)
    view, urgent = iw.take(iw.View("BASE", SESSION, block, None, 0.0), dict(record.held), 0.0)
    assert view.messages == () and HI not in iw.compose(view)


def test_the_writer_builds_no_record_of_its_own() -> None:
    """The thought record retired (Unit 6): the writer has no way to add a line itself."""
    writer, _, _, _ = _writer()
    assert not hasattr(writer, "add_thought")
    assert not hasattr(iw, "THOUGHTS_MAX")


# --- the "just now" part (route all, Unit 5; ConvoAll rule 10) --------------------------------


def _now(version: int, *, speaker=None, guest=False, here=(), lines=(), time="It's noon.", expires=None):
    value = {
        "time": time,
        "here": list(here),
        "guest": guest,
        "speaker": speaker,
        "lines": list(lines),
        "at_wall": 1_800_000_000.0,
        "at_mono": 4.0,
    }
    return SimpleNamespace(name="now", version=version, value=value, session=None, expires_at=expires)


def _just_now(text: str) -> str:
    return text.split(f"{iw.NOW_HEADING}\n", 1)[1] if iw.NOW_HEADING in text else ""


def test_just_now_is_composed_last_in_its_order() -> None:
    """The section goes last, after the messages: the time, who, the speaker, the lines."""
    block = iw.Block("## Your memory\nturns", 3, None, None, 1, 0.0)
    j = iw.Now(
        _now(1, speaker={"text": "Ada is speaking to you.", "at": SESSION}, here=["Ada is here."],
             lines=["Your hearing is off right now.", "You're bright."]).value,
        1,
        None,
    )  # fmt: skip
    hi = "(a thought from Ada, answered) hi / (your reply) hello"
    text = iw.compose(iw.View("BASE", SESSION, block, None, 0.0, (hi,), j))
    assert text.endswith(
        f"{iw.MESSAGES_HEADING}\n{hi}\n\n{iw.NOW_HEADING}\nIt's noon.\nAda is here.\n"
        "Ada is speaking to you.\nYour hearing is off right now.\nYou're bright."
    )
    empty = iw.Now(_now(2, time=None).value, 2, None)
    assert iw.NOW_HEADING not in iw.compose(iw.View("BASE", SESSION, block, None, 0.0, (), empty))


@pytest.mark.asyncio
async def test_a_speaker_stamped_after_memory_read_at_is_shown() -> None:
    """ConvoAll JInTime: a speaker decided at or after this view's memory_read_at is shown.

    It goes at once, before the reply it is for, and is not a drop-measure reset.
    """
    writer, conn, _, measured = _writer()
    await _open(writer, conn, held={"now": _now(1)})
    writer.parts_changed({"now": _now(2, speaker={"text": "Ada is speaking to you.", "at": SESSION + 1})})
    await _settle()
    assert [k for k, _ in conn.wire] == ["update"]
    assert _just_now(conn.wire[0][1]) == "It's noon.\nAda is speaking to you."
    assert measured == []
    writer.parts_changed({"now": _now(3, speaker={"text": "Ada is speaking to you.", "at": SESSION})})
    await _settle()
    assert len(conn.wire) == 1  # a stamp equal to the read is shown too: no change, no write
    writer.parts_changed({"now": _now(4, speaker=None)})  # a withdrawal (answer (a)) goes at once too
    await _settle()
    assert len(conn.wire) == 2 and "Ada is speaking" not in conn.wire[1][1] and measured == []


@pytest.mark.asyncio
async def test_an_old_sessions_speaker_is_not_in_the_startup_write() -> None:
    """ConvoAll NoStaleSpeaker: the startup write leaves out an older session's speaker.

    The record at a reconnect's connect still holds the last session's speaker, stamped before
    this view's memory_read_at. The companion's re-stamp at the next clear verdict shows it at once.
    """
    writer, conn, _, _ = _writer()
    held = {"now": _now(1, speaker={"text": "Ada is speaking to you.", "at": SESSION - 0.5})}
    text = writer.connect(SESSION, "BASE", "## Your memory\nold turns", {"top": 1}, held)
    assert _just_now(text) == "It's noon." and "Ada" not in text
    await writer.open(conn, {"type": "realtime", "instructions": text})
    conn.wire.clear()
    writer.parts_changed({"now": _now(2, speaker={"text": "Ada is speaking to you.", "at": SESSION + 2})})
    await _settle()
    assert [k for k, _ in conn.wire] == ["update"] and "Ada is speaking to you." in conn.wire[0][1]


@pytest.mark.asyncio
async def test_a_just_now_write_during_a_reply_leaves_the_drop_measure() -> None:
    """ConvoAll NoGapHis (Fjmeasure): a just-now write never starts the drop measure again.

    The speaker and a guest's arrival go at once with no measure reset, or the drop detector
    would start again at every turn; taken with S's join, the one send measures by S's reason.
    Any other change (the time, who is here, the lines, a guest leaving) is batched.
    """
    writer, conn, _, measured = _writer()
    await _open(writer, conn, held={"now": _now(1)})
    writer.parts_changed({"now": _now(2, guest=True, here=["A guest is here."])})
    await _settle()
    assert [k for k, _ in conn.wire] == ["update"] and "A guest is here." in conn.wire[0][1]
    assert measured == []
    for version, change in enumerate(
        [{"time": "It's five past noon.", "guest": True, "here": ["A guest is here."]},
         {"guest": True, "here": ["A guest is here.", "Ben is here."]},
         {"guest": True, "here": ["A guest is here."], "lines": ["Your hearing is off right now."]},
         {"guest": False, "here": []}],
        start=3,
    ):  # fmt: skip
        writer.parts_changed({"now": _now(version, **change)})
        await _settle()
        assert len(conn.wire) == 1 and writer.pending()  # batched: nothing at once
    writer.parts_changed(
        {
            "now": _now(8, speaker={"text": "Ben is speaking to you.", "at": SESSION + 3}),
            "summary": _part("summary", 9, {**S1, "text": "Ada asked about the weather."}, session=None),
        }
    )
    await _settle()
    assert conn.wire[1][0] == "update" and "Ben is speaking to you." in conn.wire[1][1]
    assert measured[0] == "reset"  # S's join measures; the speaker rode with it


@pytest.mark.asyncio
async def test_a_silent_companions_part_is_dropped_at_its_expiry(caplog) -> None:
    """ConvoAll NoExpiredJ (Fnojwake): a silent companion's part is dropped at its expiry.

    With no newer version (a silent companion, a dead link), the writer's one timer at the held
    part's expiry drops it and writes at once, so he has no "just now" rather than a stale one.
    A part already past its expiry is never taken, and a snapshot that lacks the part drops it
    at once too. None of it resets the drop measure.
    """
    import time

    writer, conn, _, measured = _writer()
    await _open(writer, conn, held={"now": _now(1, expires=time.monotonic() + 0.05)})
    assert iw.NOW_HEADING in writer.sent
    with caplog.at_level("INFO", logger=iw.__name__):
        await asyncio.sleep(0.15)
        await _settle()
    assert [k for k, _ in conn.wire] == ["update"] and iw.NOW_HEADING not in conn.wire[0][1]
    assert "(just now dropped)" in caplog.text and "no just now" in caplog.text
    writer.parts_changed({"now": _now(2, expires=time.monotonic() - 1.0)})  # already past it
    await _settle()
    assert writer.view.now is None and len(conn.wire) == 1
    writer.parts_changed({"now": _now(3, expires=time.monotonic() + 60.0)})
    writer.parts_changed({})  # a snapshot without it
    await _settle()
    assert iw.NOW_HEADING not in conn.wire[-1][1] and writer.view.now is None
    assert measured == []
    assert writer._expiry is None  # nothing held, no timer


@pytest.mark.asyncio
async def test_a_newer_version_moves_the_expiry_timer_later() -> None:
    """The heartbeat: each version's expiry is later, so its timer replaces the last one's."""
    import time

    writer, conn, _, _ = _writer()
    await _open(writer, conn, held={"now": _now(1, expires=time.monotonic() + 0.05)})
    writer.parts_changed({"now": _now(2, expires=time.monotonic() + 60.0)})
    await asyncio.sleep(0.15)
    assert conn.wire == [] and writer.view.now is not None and writer._expiry is not None


@pytest.mark.asyncio
async def test_the_send_log_names_the_parts_version_and_age_never_its_text(caplog) -> None:
    """The send log carries the part's version and age, never household text."""
    writer, conn, _, _ = _writer()
    await _open(writer, conn, held={"now": _now(1)})
    with caplog.at_level("INFO", logger=iw.__name__):
        writer.parts_changed({"now": _now(2, speaker={"text": "Ada is speaking to you.", "at": SESSION + 1})})
        await _settle()
    assert "just now v2, " in caplog.text and " s old, its speaker shown" in caplog.text
    assert "Ada" not in caplog.text


@pytest.mark.asyncio
async def test_a_part_that_arrives_after_a_connect_without_one_arms_the_expiry() -> None:
    """The expiry timer is armed at each take, not only at the connect.

    A session opened with no "now" part held, then a part arrives: its expiry still drops it.
    """
    import time

    writer, conn, _, measured = _writer()
    await _open(writer, conn, held={})
    assert writer._expiry is None
    writer.parts_changed({"now": _now(1, expires=time.monotonic() + 0.05)})
    assert writer._expiry is not None
    await asyncio.sleep(0.15)
    await _settle()
    assert writer.view.now is None and iw.NOW_HEADING not in (writer.sent or "")
    assert measured == []
