"""A long talk kept small (the companion's context budget, 2026-11-04), delete-only under route all.

The companion folds a talk's turns older than 20 minutes into one running summary, S. Under
route all (the companion's ConvoAll, docs/formal/conversation-d/model.md, round 6, rules 4, 5
and 9) S reaches him only as a part of his instructions (instructions_writer), and this module
only deletes: no summary item is created, and nothing is read from the companion's store.

- When S joins his instructions, the items it covers go: S names the first item it does not
  cover (the user item of the next turn), and every item that arrived before that one goes,
  his replies and tool pairs included (the cut by turn; round 6, a_lagcut).
- After a context drop the companion re-sends the block holding the turns through the app's
  latest heard user item; the surviving items through that one go (rule 9, a_noctxdel).

When S's cut item is not in this session's ledger (another session's, one the backend evicted,
or none named), the cut falls back to S's time: the items that arrived LAG_S before it (the
round-6 review, item 4). A tool call goes only with its output: a call whose output is still
owed is never deleted, or the backend refuses the output (item 1). Ids and counts only are
logged.
"""

from __future__ import annotations
import time
import logging
import itertools
from typing import Any
from collections import OrderedDict


logger = logging.getLogger(__name__)

LEDGER_MAX = 2000
LAG_S = 60.0  # the time cut: an item arrives before its transcript is stamped


class TalkPrune:
    """Each conversation item's id in the order it arrived, and tool call pairs by call_id."""

    def __init__(self) -> None:
        """Start with an empty ledger."""
        self._order = itertools.count()
        self.items: OrderedDict[str, tuple[int, float]] = OrderedDict()  # (order, wall time)
        # a tool call and its output by call_id: [(item id, (order, wall)) of call, of output]
        self.calls: OrderedDict[str, list] = OrderedDict()

    def seen(self, item: Any, at: float | None = None) -> None:
        """Record an item the backend created, in the order the events came, at `at` (wall)."""
        stamp = (next(self._order), time.time() if at is None else at)
        item_id = getattr(item, "id", None)
        kind = getattr(item, "type", "message")
        if kind in ("function_call", "function_call_output"):  # pruned only as a pair (11-04)
            call = getattr(item, "call_id", None)
            if item_id and call:
                pair = self.calls.setdefault(call, [None, None])
                pair[0 if kind == "function_call" else 1] = (item_id, stamp)
                while len(self.calls) > LEDGER_MAX:
                    self.calls.popitem(last=False)
            return
        if kind != "message" or getattr(item, "role", None) == "system":
            return  # instructions stay
        if item_id and item_id not in self.items:
            self.items[item_id] = stamp
            while len(self.items) > LEDGER_MAX:
                self.items.popitem(last=False)

    def forget(self, item_id: str) -> None:
        """Drop an item deleted elsewhere (a leaked reply's scrub): never deleted again here."""
        self.items.pop(item_id, None)

    def reset(self) -> None:
        """Start again for a new session: nothing of the last one is in its conversation."""
        self.__init__()

    async def delete_before(
        self, conn: Any, item_id: str | None, *, through: bool = False, up_to: float | None = None
    ) -> int:
        """Delete every item that arrived before `item_id`, and each tool pair whose halves did.

        `through` deletes `item_id` too. When the item is not in this session's ledger, the cut
        is by time: what arrived LAG_S before `up_to` (wall), or nothing without it. Returns the
        count deleted.
        """
        if item_id and item_id in self.items:
            order = self.items[item_id][0]

            def before(stamp: tuple[int, float], own: bool = False) -> bool:
                return stamp[0] < order or (own and through and stamp[0] == order)

        elif up_to is not None:
            logger.info("Talk prune: the cut item is not in this session; cut by time")
            cut = up_to - LAG_S

            def before(stamp: tuple[int, float], own: bool = False) -> bool:
                return stamp[1] < cut

        else:
            logger.info("Talk prune: the cut item is not in this session; nothing deleted")
            return 0
        old = [i for i, stamp in self.items.items() if before(stamp, own=True)]
        for i in old:
            self.items.pop(i, None)
            await conn.conversation.item.delete(item_id=i)
        # a call whose output is owed has no output here: it stays (the backend would refuse it)
        pairs = [c for c, (call, out) in self.calls.items()
                 if call and out and before(call[1]) and before(out[1])]  # fmt: skip
        for c in pairs:  # both halves before the cut: what was found is in what covers them
            call, out = self.calls.pop(c)
            await conn.conversation.item.delete(item_id=call[0])
            await conn.conversation.item.delete(item_id=out[0])
        logger.info("Talk prune: %d messages and %d tool call pairs deleted", len(old), len(pairs))
        return len(old) + len(pairs)
