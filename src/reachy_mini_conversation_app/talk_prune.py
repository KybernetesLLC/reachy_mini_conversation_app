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

The trim (the owner's decision, 11-22): the backend compacts a chat on its own once it holds
chat_size user items, and S then covers turns it has already summarised. So the app trims first.
When the ledger holds TRIM_AT user items (notes count), the oldest go down to TRIM_TO, by rule
9's sequence started on purpose: the companion is told the trim point (a heard user turn's item),
bounds the block there and re-sends it, and the deletes go only after the write that carries it
(instructions_writer). The latest user turn and a pending tool call are never trimmed.
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
# The trim (the owner, 11-23): the hosted backend's chat_size is not known. At speech-to-speech
# ff4aec3 it defaults to 30 (arguments_classes/language_model_base_arguments.py:22), which the
# realtime pipeline passes on (s2s_pipeline.py:633); the 10 at api/openai_realtime/service.py:370
# is only RealtimeService's own default. The lowest evidence for the hosted value: compactions at
# about 14 user items on 11-14, while notes still counted. So the trim starts at 13, under that,
# and leaves 6: at 10 down to 3 he kept about 3 exchanges and a fast talk trimmed every ~90 s. An
# overnight measurement of chat_size will set both.
TRIM_AT = 13
TRIM_TO = 6


class TalkPrune:
    """Each conversation item's id in the order it arrived, and tool call pairs by call_id."""

    def __init__(self) -> None:
        """Start with an empty ledger."""
        self._order = itertools.count()
        self.items: OrderedDict[str, tuple[int, float]] = OrderedDict()  # (order, wall time)
        self.users: set[str] = set()  # the user-role items among them (notes count)
        self.heard: set[str] = set()  # user items whose transcript was heard (the companion has it)
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
            if getattr(item, "role", None) == "user":
                self.users.add(item_id)
            while len(self.items) > LEDGER_MAX:
                self._drop(self.items.popitem(last=False)[0])

    def _drop(self, item_id: str) -> None:
        self.items.pop(item_id, None)
        self.users.discard(item_id)
        self.heard.discard(item_id)

    def mark_heard(self, item_id: str | None) -> None:
        """Note a user item whose transcript was heard: the companion holds its turn."""
        if item_id and item_id in self.items:
            self.heard.add(item_id)

    def trim_point(self) -> str | None:
        """Return the heard user item to trim through, once the chat holds TRIM_AT user items.

        The first heard user turn at or after the one that leaves TRIM_TO, never the latest user
        item; None when the count is under TRIM_AT or no such turn is heard yet.
        """
        users = [i for i in self.items if i in self.users]
        if len(users) < TRIM_AT:
            return None
        need = len(users) - TRIM_TO  # this many of the oldest go
        for i in users[need - 1 : -1]:
            if i in self.heard:
                return i
        return None

    def forget(self, item_id: str) -> None:
        """Drop an item deleted elsewhere (a leaked reply's scrub): never deleted again here."""
        self._drop(item_id)

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
            self._drop(i)
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
