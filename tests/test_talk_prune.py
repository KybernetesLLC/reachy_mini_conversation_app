"""The companion's context budget (2026-11-04): a long talk kept small."""

import asyncio
import json
import sqlite3
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


def _item(item_id: str, text: str = "hi") -> SimpleNamespace:
    return SimpleNamespace(id=item_id, content=[SimpleNamespace(text=text)])


def test_items_older_than_the_summary_go_and_the_summary_replaces_the_last(tmp_path) -> None:
    p = tmp_path / "companion.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO settings VALUES ('talk_summary', ?)",
                (json.dumps({"session": 1.0, "up_to": 150.0, "text": "Alli asked about ferries."}),))
    con.commit()
    prune, conn = tp.TalkPrune(), Conn()
    prune.seen(_item("a"), at=100.0)
    prune.seen(_item("b"), at=200.0)
    assert asyncio.run(prune.apply(conn, tp.read_summary(p))) == 1
    assert conn.deleted == ["a"] and conn.created[0]["role"] == "system"
    assert conn.created[0]["content"][0]["text"] == tp.PREFIX + "Alli asked about ferries."
    prune.seen(_item("s1", tp.PREFIX + "Alli asked about ferries."), at=210.0)
    assert prune.summary_item == "s1" and "s1" not in prune.items
    assert asyncio.run(prune.apply(conn, tp.read_summary(p))) == 0  # no newer summary
    assert asyncio.run(prune.apply(conn, (250.0, "More."))) == 1
    assert conn.deleted == ["a", "b", "s1"]  # the old summary goes with them
    assert tp.read_summary(tmp_path / "none.db") is None
