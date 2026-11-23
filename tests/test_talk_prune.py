"""The talk prune under route all: delete-only, by turn (the companion's ConvoAll round 6)."""

import asyncio
from types import SimpleNamespace

from reachy_mini_conversation_app import talk_prune as tp


class Conn:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[dict] = []
        self.conversation = SimpleNamespace(item=SimpleNamespace(delete=self._delete, create=self._create))

    async def _delete(self, item_id: str) -> None:
        self.deleted.append(item_id)

    async def _create(self, item: dict) -> None:
        self.created.append(item)


def _item(item_id: str, type_: str = "message", call: str = "", role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=item_id, type=type_, role=role, call_id=call, content=[SimpleNamespace(text="hi")])


def _ledger() -> tp.TalkPrune:
    prune = tp.TalkPrune()
    prune.seen(_item("u1"), at=100.0)
    prune.seen(_item("a1", role="assistant"), at=101.0)
    prune.seen(_item("call", type_="function_call", call="c1"), at=102.0)
    prune.seen(_item("out", type_="function_call_output", call="c1"), at=103.0)
    prune.seen(_item("owed", type_="function_call", call="c2"), at=104.0)  # its output not sent yet
    prune.seen(_item("u2"), at=200.0)  # the first item S does not cover
    prune.seen(_item("a2", role="assistant"), at=201.0)
    return prune


def test_a_lagcut_s_cuts_by_turn_at_the_first_item_it_does_not_cover() -> None:
    """a_lagcut_GS: every item that arrived before S's cut item goes, his replies and complete
    tool pairs included; no summary item is created (TalkPrune is delete-only)."""
    prune, conn = _ledger(), Conn()
    assert asyncio.run(prune.delete_before(conn, "u2")) == 3
    assert conn.deleted == ["u1", "a1", "call", "out"] and conn.created == []
    assert list(prune.items) == ["u2", "a2"]


def test_a_call_whose_output_is_owed_is_never_deleted() -> None:
    """Round-6 review, item 1: calls go only as complete pairs. A call whose output is still owed
    stays, or the backend refuses the output (chat.py:299) and the follow-up runs blind."""
    prune, conn = _ledger(), Conn()
    asyncio.run(prune.delete_before(conn, "a2", through=True))  # a drop's catch-up, through a2
    assert "owed" not in conn.deleted and "c2" in prune.calls
    assert conn.deleted == ["u1", "a1", "u2", "a2", "call", "out"]


def test_a_cut_item_not_in_the_ledger_falls_back_to_the_time_cut() -> None:
    """Round-6 review, item 4: with S's cut item absent (another session's, evicted, or none), the
    cut is by S's time, LAG_S before it; with no time either, nothing goes."""
    prune, conn = _ledger(), Conn()
    assert asyncio.run(prune.delete_before(conn, "gone", up_to=150.0 + tp.LAG_S)) == 3
    assert conn.deleted == ["u1", "a1", "call", "out"]
    prune, conn = _ledger(), Conn()
    assert asyncio.run(prune.delete_before(conn, None)) == 0 and conn.deleted == []


def test_a_scrubbed_item_is_never_deleted_again_and_a_reset_forgets_all() -> None:
    prune, conn = _ledger(), Conn()
    prune.forget("u1")
    asyncio.run(prune.delete_before(conn, "u2"))
    assert "u1" not in conn.deleted
    prune.reset()
    assert not prune.items and not prune.calls
    prune.seen(_item("sys", role="system"))
    assert not prune.items  # instructions stay


def _chat(prune: tp.TalkPrune, turns: int, *, notes: tuple[int, ...] = (), owed_after: int | None = None) -> list[str]:
    """`turns` heard user turns, each with his reply; a note before the turns in `notes`."""
    users = []
    for n in range(turns):
        if n in notes:
            prune.seen(_item(f"note{n}"))
            users.append(f"note{n}")
        prune.seen(_item(f"u{n}"))
        prune.mark_heard(f"u{n}")
        users.append(f"u{n}")
        prune.seen(_item(f"a{n}", role="assistant"))
        if owed_after == n:
            prune.seen(_item("owed", type_="function_call", call="c9"))
    return users


def test_a_trim_starts_at_trim_at_user_items_and_leaves_at_most_trim_to() -> None:
    """The owner's trim (11-22): notes count; deleting through the trim point leaves at most
    TRIM_TO user items in the chat, and never the latest user turn."""
    prune = tp.TalkPrune()
    _chat(prune, tp.TRIM_AT - 3, notes=(1,))
    assert prune.trim_point() is None  # under TRIM_AT
    prune = tp.TalkPrune()
    users = _chat(prune, tp.TRIM_AT - 2, notes=(2, 5))  # 8 turns and 2 notes: 10 user items
    point = prune.trim_point()
    assert point is not None and point != users[-1] and point in prune.heard
    conn = Conn()
    asyncio.run(prune.delete_before(conn, point, through=True))
    assert len([i for i in prune.items if i in prune.users]) <= tp.TRIM_TO
    assert users[-1] in prune.items
    kept = list(prune.items)
    assert all(i not in kept for i in conn.deleted)
    # a turn the trim deletes is one at or before its point, which the companion's block holds
    assert set(conn.deleted) <= set(users[: users.index(point) + 1]) | {f"a{n}" for n in range(10)}
    assert all(int(i[1:]) < int(point[1:]) for i in conn.deleted if i.startswith("a"))


def test_a_trim_never_takes_an_open_tool_pair_or_an_unheard_point() -> None:
    prune = tp.TalkPrune()
    _chat(prune, tp.TRIM_AT, owed_after=1)
    point = prune.trim_point()
    conn = Conn()
    asyncio.run(prune.delete_before(conn, point, through=True))
    assert "owed" not in conn.deleted and "c9" in prune.calls
    prune = tp.TalkPrune()
    for n in range(tp.TRIM_AT):
        prune.seen(_item(f"note{n}"))  # notes only: the companion has no turn to bound at
    assert prune.trim_point() is None


def test_a_trim_point_is_never_the_latest_user_item_when_the_newer_are_notes() -> None:
    """The Units 3+4 review, finding 6: with the newer user items all unheard notes, no heard turn
    lies where the trim would cut, so there is no point; never the latest user item."""
    prune = tp.TalkPrune()
    notes = tp.TRIM_TO + 3  # more notes than the trim leaves: the cut falls among them
    _chat(prune, tp.TRIM_AT - notes)
    for n in range(notes):
        prune.seen(_item(f"note{n}"))
    assert len(prune.users) == tp.TRIM_AT
    assert prune.trim_point() is None
    # the same with the latest a heard turn: the only heard item the cut reaches is the latest
    prune.forget(f"note{notes - 1}")
    prune.seen(_item("last"))
    prune.mark_heard("last")
    assert prune.trim_point() is None
    # all heard: the point is exactly the turn that leaves TRIM_TO
    prune = tp.TalkPrune()
    users = _chat(prune, tp.TRIM_AT)
    assert prune.trim_point() == users[tp.TRIM_AT - tp.TRIM_TO - 1]
