"""A long talk kept small (the companion's context budget, 2026-11-04).

The companion folds a talk's turns older than 20 minutes into one running summary
(talk_summary in companion.db). The memory block carries it at a session start, but a live
session grows inside the backend's own conversation. So at a reply's end, at most once every
CHECK_S, the items older than the summary's `up_to` are deleted, and the summary goes in as one
system item, replacing the one before. The backend honours conversation.item.delete and a
system item created mid-session (checked in a dry session, 11-04). A drop's resend re-reads the
block, which carries the same summary. Ids and times only are logged.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHECK_S = 60.0
LAG_S = 60.0  # an item arrives before its transcript is stamped: this much older than up_to
LEDGER_MAX = 2000
KEY = "talk_summary"
PREFIX = "Earlier in this talk, in short: "


def companion_db() -> Path:
    named = os.environ.get("REACHY_COMPANION_DB")
    return Path(named) if named else Path.home() / ".local/share/reachy-companion/companion.db"


def read_summary(path: Path | None = None) -> tuple[float, str] | None:
    """(up_to, text) from the companion's store, read only; None without one."""
    try:
        con = sqlite3.connect(f"file:{path or companion_db()}?mode=ro", uri=True, timeout=1.0)
        try:
            row = con.execute("SELECT value FROM settings WHERE key = ?", (KEY,)).fetchone()
        finally:
            con.close()
        d = json.loads(row[0]) if row and row[0] else None
        if isinstance(d, dict) and str(d.get("text") or "").strip():
            return float(d.get("up_to") or 0), str(d["text"]).strip()
    except (sqlite3.Error, ValueError, TypeError):
        return None
    return None


class TalkPrune:
    """Each conversation item's id and when it arrived, and the summary item now in place."""

    def __init__(self) -> None:
        self.items: OrderedDict[str, float] = OrderedDict()
        self.summary_item: str | None = None
        self.up_to = 0.0
        self.checked_at = float("-inf")

    def seen(self, item: Any, at: float | None = None) -> None:
        """An item the backend created: the summary's own (by its text) is kept apart."""
        item_id = getattr(item, "id", None)
        content = getattr(item, "content", None) or []
        text = getattr(content[0], "text", "") if content else ""
        if item_id and isinstance(text, str) and text.startswith(PREFIX):
            self.summary_item = item_id
            return
        if getattr(item, "type", "message") != "message" or getattr(item, "role", None) == "system":
            return  # a tool call and its output stay paired, and instructions stay (the review)
        if item_id and item_id not in self.items and item_id != self.summary_item:
            self.items[item_id] = time.time() if at is None else at
            while len(self.items) > LEDGER_MAX:
                self.items.popitem(last=False)

    def reset(self) -> None:
        """A new session: nothing of the last one is in its conversation."""
        self.__init__()

    def due(self, now: float) -> bool:
        return now - self.checked_at >= CHECK_S

    async def apply(self, conn: Any, summary: tuple[float, str] | None) -> int:
        """Delete the items older than the summary's up_to and put the summary in; the count
        deleted. Nothing without a newer summary."""
        self.checked_at = time.monotonic()
        if summary is None:  # cleared (a forget, the talk's end): its item goes too
            if self.summary_item is not None:
                await conn.conversation.item.delete(item_id=self.summary_item)
                self.summary_item, self.up_to = None, 0.0
            return 0
        if summary[0] <= self.up_to:
            return 0
        up_to, text = summary
        old = [i for i, at in self.items.items() if at < up_to - LAG_S]
        for item_id in old:
            await conn.conversation.item.delete(item_id=item_id)
            self.items.pop(item_id, None)
        if self.summary_item is not None:
            await conn.conversation.item.delete(item_id=self.summary_item)
            self.summary_item = None  # the new one's id comes back in its created event
        await conn.conversation.item.create(
            item={
                "type": "message", "role": "system",
                "content": [{"type": "input_text", "text": PREFIX + text}],
            }
        )  # fmt: skip
        self.up_to = up_to
        logger.info("Talk prune: %d items older than the summary deleted", len(old))
        return len(old)
