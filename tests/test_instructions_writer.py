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
    assert text.endswith(f"{iw.THOUGHTS_HEADING}\n(a thought from Alli, answered) hi / (your reply) hello")


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
    assert measured == ["reset"]  # a join starts the drop detector's measure again (item 3)


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
    writer.add_thought("(a thought from Alli, answered) hi / (your reply) hello")
    await asyncio.sleep(0.05)  # the batch
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
