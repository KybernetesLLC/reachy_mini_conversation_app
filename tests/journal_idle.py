"""The replay check: every stretch of a journal excerpt with the speech stream open and idle.

The 05:22 phantom (2026-10-26) came 9 h 19 m after the speech stream's last write. Run
against the robot's journal, it lists every idle stretch over a threshold in hours::

    journalctl --user -u reachy-mini-daemon --since "<from>" -o short-iso |
      uv run python tests/journal_idle.py 3
"""

from __future__ import annotations
import re
import sys
import datetime
from typing import Iterable, NamedTuple


# Journal lines that mean the speech stream was written to or made new, and the line that
# closes it (capture off sets the pipeline to NULL).
STREAM_WRITE_LINES = (
    "Reply heard after",
    "Reply not heard",
    "Speaker stream rebuilt",
    "Audio capture started",
)
STREAM_CLOSE_LINES = ("Audio capture stopped",)
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")


class IdleStretch(NamedTuple):
    """A stretch with the speech stream open and nothing written to it."""

    start: datetime.datetime
    end: datetime.datetime
    ongoing: bool  # still idle at the excerpt's last line

    @property
    def seconds(self) -> float:
        """The stretch's length."""
        return (self.end - self.start).total_seconds()


def idle_stretches(lines: Iterable[str], idle_s: float) -> list[IdleStretch]:
    """Return every stretch of a journal excerpt with the speech stream idle over ``idle_s``. Pure.

    The excerpt's first timed line counts as a write: what came before it is not known.
    Reads ``journalctl -o short-iso`` or the classic format, by the app's own timestamp.
    """
    found: list[IdleStretch] = []
    last_write: datetime.datetime | None = None
    seen = False
    at: datetime.datetime | None = None
    for line in lines:
        stamp = _STAMP.search(line)
        if stamp is None:
            continue
        at = datetime.datetime.fromisoformat(f"{stamp[1]} {stamp[2]}")
        writes = any(key in line for key in STREAM_WRITE_LINES)
        closes = any(key in line for key in STREAM_CLOSE_LINES)
        if not seen:
            seen, last_write = True, at
        if writes or closes:
            if last_write is not None and (at - last_write).total_seconds() > idle_s:
                found.append(IdleStretch(last_write, at, ongoing=False))
            last_write = at if writes else None
    if at is not None and last_write is not None and (at - last_write).total_seconds() > idle_s:
        found.append(IdleStretch(last_write, at, ongoing=True))
    return found


if __name__ == "__main__":
    for found in idle_stretches(sys.stdin, float(sys.argv[1]) * 3600.0):
        print(found.start, found.end, f"{found.seconds / 3600:.2f} h", "ongoing" * found.ongoing)
