"""The writer of his instructions (route all, Unit 4): the only sender of session instructions.

The companion's model is ConvoAll, round 6 (docs/formal/conversation-d/model.md on its branch
arch-d-model), rules 2-5, 7 and 9. His instructions are composed here from the stable base (the
profile, first, for the prompt cache) and then the parts:

- the session block: at a session's open, the companion's file read unbounded (it names the S it
  is cut at, with its text); after that, the companion's "block" part for this session, bounded
  at the turns this session's chat does not hold;
- S, the talk's earlier part in short: the newer, by S's own version, of the S the block names
  and the companion's "summary" part (rule 8: S arrives by both; a clear is a newer version);
- the recent text messages he answered in this talk: the companion's "messages" part (Unit 6,
  ConvoAll rule 4 as S), its lines composed as they come. The app keeps no record of its own.

`take` and `compose` are pure: what the parts record changes, what goes at once and why, and the
text. The writer sends (rules 3-5, 9):

- a join of S, a clear (of S or of the messages), the reconnect's catch-up (the companion marks it: the read at the open
  held less than the bound) and a drop's block go at once, on the handler's loop;
- any other change is batched, BATCH_S, and goes in a tool's window too (no hold), and never
  sooner than BATCHED_GAP_S after the last write (the Units 3+4 review: trims and awareness
  refreshes came to 5-8 writes a minute, 70-117 KB each with the echo);
- at a reply's done, any change goes before the done event is set (H calls `at_done`);
- after the write that carries a newly joined S, the items S covers are deleted; after a drop's
  block, the items through the reported item (talk_prune), in that order, in one coroutine.

Every send is logged with its parts' versions and ages, the writes in the last minute, and
counts only. A reported drop or trim whose block has not come in DROP_WAIT_S (a companion
restarted mid-session) is let go, so trims and drops are not stalled for the session.
"""

from __future__ import annotations
import time
import asyncio
import logging
from typing import Any, Callable, Optional
from collections import deque
from dataclasses import replace, dataclass


logger = logging.getLogger(__name__)

BATCH_S = 1.0
BATCHED_GAP_S = 5.0  # a batched write goes at most this often; the at-once ones are not held
DROP_WAIT_S = 30.0  # a drop's or trim's block not come by then is let go
S_HEADING = "### Earlier in this talk, in short"
MESSAGES_HEADING = "### Thoughts you answered in this conversation"
_NONE = float("-inf")


@dataclass(frozen=True)
class Block:
    """The session block he is told.

    Its text, the highest turn row it holds, the drop item it was bounded for, the S it is cut
    at, the link's version (0: the read at the open), when it was taken (monotonic), and the
    bound the companion marked as the catch-up (the read at the open held less; its decision).
    """

    text: str
    top: int
    through: Optional[str]
    s: Optional[dict[str, Any]]
    version: int
    at: float
    catch_up: Optional[int] = None


@dataclass(frozen=True)
class View:
    """What his instructions are composed from, for one realtime session (its memory_read_at)."""

    base: str
    session: Optional[float]
    block: Block
    s: Optional[dict[str, Any]]
    s_at: float
    messages: tuple[str, ...] = ()


def _s(raw: Any) -> Optional[dict[str, Any]]:
    """Return an S record (the companion's talk_summary.named plus its cut), or None."""
    if not isinstance(raw, dict):
        return None
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, (int, float)):
        return None
    return raw


def s_version(s: Optional[dict[str, Any]]) -> float:
    """Return S's own version; no S is older than any."""
    return float(s["version"]) if s is not None else _NONE


def _top(raw: Any) -> int:
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0


def file_block(text: str, header: dict[str, Any], at: float) -> Block:
    """Return the block as read from the companion's file at a session's open."""
    return Block(text, _top(header.get("top")), None, _s(header.get("s")), 0, at)


def part_block(part: Any, at: float) -> Optional[Block]:
    """Return the block from the companion's "block" part, or None for a value that is not one."""
    value = getattr(part, "value", None)
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        return None
    through = value.get("through")
    return Block(
        value["text"],
        _top(value.get("top")),
        through if isinstance(through, str) and through else None,
        _s(value.get("s")),
        int(getattr(part, "version", 0)),
        at,
        value["catch_up"] if _top(value.get("catch_up")) else None,
    )


def part_messages(part: Any) -> tuple[str, ...]:
    """Return the lines of the companion's "messages" part; none for a clear or no part."""
    value = getattr(part, "value", None)
    if not isinstance(value, list):
        return ()
    return tuple(line for line in value if isinstance(line, str) and line.strip())


def take(view: View, held: dict[str, Any], now: float) -> tuple[View, list[str]]:
    """Return the view after the parts record `held` changed, and why it goes at once.

    An empty list of reasons means it is batched. The block part is taken only for this session and only when newer (Fstaleparts). S is the
    newest by its own version of the S held, the block's and the link's (a_nonewer), so an
    older S on the link never replaces the one a block named, and a clear is never undone.
    """
    urgent: list[str] = []
    block = view.block
    part = held.get("block")
    if part is not None and part.session is not None and part.session == view.session:
        new = part_block(part, now) if part.version > block.version else None
        if new is not None:
            if new.through is not None and new.through != block.through:
                urgent.append("a drop's block")
            elif new.catch_up is not None and new.catch_up != block.catch_up:
                urgent.append("the catch-up")  # the companion's decision (round-6 review, 5)
            block = new
    s, s_at = view.s, view.s_at
    link = held.get("summary")
    for candidate in (block.s, _s(getattr(link, "value", None))):
        if candidate is not None and s_version(candidate) > s_version(s):
            s, s_at = candidate, now
    if s is not view.s:
        urgent.append("S joined" if s is not None and s.get("text") else "S cleared")
    # The messages are read from the record as held: it keeps the newest version (parts.apply)
    # and drops a part a snapshot lacks. An answered message is batched; a clear (the talk's
    # end, a forget) goes at once, as S's does (ClearedGone).
    messages = part_messages(held.get("messages"))
    if view.messages and not messages:
        urgent.append("messages cleared")
    return replace(view, block=block, s=s, s_at=s_at, messages=messages), urgent


def compose(view: View) -> str:
    """Return his instructions: the stable base first, then the block, S and the messages answered."""
    parts = [view.base.strip()]
    if view.block.text.strip():
        parts.append(view.block.text.strip())
    if view.s is not None and str(view.s.get("text") or "").strip():
        parts.append(f"{S_HEADING}\n{' '.join(str(view.s['text']).split())}")
    if view.messages:
        parts.append(f"{MESSAGES_HEADING}\n" + "\n".join(view.messages))
    return "\n\n".join(p for p in parts if p)


class InstructionsWriter:
    """The one sender of his session instructions, on the handler's loop."""

    def __init__(
        self,
        prune: Any,
        *,
        connection: Callable[[], Any],
        clock: Callable[[], float] = time.monotonic,
        batch_s: float = BATCH_S,
        gap_s: float = BATCHED_GAP_S,
        on_sent: Callable[[], None] | None = None,
    ) -> None:
        """Hold no view until a session's open (`connect`)."""
        self.prune = prune
        self._connection = connection
        self.clock = clock
        self.batch_s = batch_s
        self.gap_s = gap_s
        self.on_sent = on_sent
        self.view: View | None = None
        self.sent: str | None = None  # the instructions the session holds, as last written
        self._joined = _NONE  # the S version whose deletes went
        self._drop: str | None = None  # the drop item reported, until its block is sent
        self._drop_at = 0.0  # when it was reported (the clock)
        self._writes: deque[float] = deque()  # when each write went, the last minute's
        self._lock: asyncio.Lock | None = None
        self._batch: asyncio.Task[None] | None = None
        self._now: set[asyncio.Task[None]] = set()

    # --- a session's open ---------------------------------------------------------------------

    def connect(
        self,
        session: float | None,
        base: str,
        text: str,
        header: dict[str, Any],
        held: dict[str, Any],
    ) -> str:
        """Start a new session's view and return the composition its config carries.

        The view is the file read now (unbounded, naming its S) and the parts held. A new
        session has no items, so S's deletes are spent and no drop is pending.
        """
        now = self.clock()
        block = file_block(text, header, now)
        view, _ = take(View(base, session, block, block.s, now), held, now)
        self.view, self.sent, self._drop = view, None, None
        self._joined = s_version(view.s)
        if self._batch is not None:
            self._batch.cancel()
            self._batch = None
        return compose(view)

    async def open(self, conn: Any, session_config: dict[str, Any]) -> None:
        """Send the session's first write: its config whole, with the instructions composed.

        Rule 4: a session's first write carries the writer's record. Raises as session.update.
        """
        await conn.session.update(session=session_config)
        self.sent = session_config.get("instructions")
        self._log("the session's open", self.sent or "")
        if self.view is not None and compose(self.view) != self.sent:
            self._schedule()  # a part came between the read and the write

    # --- what changes the view ------------------------------------------------------------------

    def current(self) -> str | None:
        """Return the composition as it is now (rule 7: the nudge's create carries it at each send)."""
        return compose(self.view) if self.view is not None else None

    def parts_changed(self, held: dict[str, Any]) -> None:
        """Take the companion's changed parts record (on the handler's loop)."""
        if self.view is None:
            return
        view, urgent = take(self.view, held, self.clock())
        self.view = view
        if urgent:
            self._send_now(", ".join(urgent))
        elif compose(view) != self.sent:
            self._schedule()

    def report_drop(self, item: str | None) -> None:
        """Note a reported drop's item (rule 9): its items go after the block naming it."""
        self._drop, self._drop_at = item, self.clock()

    def drop_pending(self) -> bool:
        """Return whether a reported drop or trim waits for its block, letting go of a stale one.

        A block not come in DROP_WAIT_S (a companion restarted mid-session publishes none) is
        let go and logged; its items stay in the chat until the next trim or S.
        """
        if self._drop is not None and self.clock() - self._drop_at > DROP_WAIT_S:
            logger.info("Talk prune: no block for the reported item in %.0f s; let go", DROP_WAIT_S)
            self._drop = None
        return self._drop is not None

    def want(self, why: str) -> None:
        """Batch a send asked for from outside (a session opened on a live connection)."""
        if self.view is not None:
            logger.debug("Instructions wanted (%s)", why)
            self._schedule()

    def pending(self) -> bool:
        """Return whether the composition differs from what the session holds."""
        return self.view is not None and compose(self.view) != self.sent

    async def at_done(self) -> None:
        """Send any change at a reply's done, before the done event is set (rule 2)."""
        if self.pending():
            await self.send("a reply's done")

    # --- sending --------------------------------------------------------------------------------

    def _schedule(self) -> None:
        if self._batch is not None and not self._batch.done():
            return
        try:
            self._batch = asyncio.get_running_loop().create_task(self._after_batch())
        except RuntimeError:  # no loop: the next done or join sends it
            self._batch = None

    async def _after_batch(self) -> None:
        await asyncio.sleep(self.batch_s)
        last = self._writes[-1] if self._writes else None
        if last is not None:  # the floor on batched writes, from the last write of any kind
            wait = last + self.gap_s - self.clock()
            if wait > 0:
                await asyncio.sleep(wait)
        await self.send("a part changed")

    def _send_now(self, why: str) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self.send(why, measure=True))
        except RuntimeError:
            return
        self._now.add(task)
        task.add_done_callback(self._now.discard)

    async def send(self, why: str, *, measure: bool = False) -> None:
        """Write the composition when it changed, then the deletes it makes due.

        S's covered items go after the write that carries S, a drop's items after the block
        naming it. One send at a time; never raises. `measure`: a write that changes the block or
        S at once (a join, a clear, the catch-up, a drop's block) starts the drop detector's
        measure again (`on_sent`); a done write or a batched one does not, or a drop would never
        show (the round-6 review, item 3). Deletes shrink the chat on purpose, so once they
        are done the measure starts again too: a trim or S's join is not a drop.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            conn, view = self._connection(), self.view
            if conn is None or view is None:
                return
            target = compose(view)
            if target != self.sent:
                try:
                    await conn.session.update(session={"type": "realtime", "instructions": target})
                except Exception as exc:
                    logger.warning("Instructions not sent (%s): %s", why, exc)
                    return
                self.sent = target
                now = self.clock()
                self._writes.append(now)
                while self._writes and self._writes[0] < now - 60.0:
                    self._writes.popleft()
                self._log(why, target)
                if measure and self.on_sent is not None:
                    self.on_sent()
            try:
                deleted = await self._deletes(conn, view)
            except Exception as exc:
                logger.warning("Talk prune: not applied (%s): %s", why, exc)
                deleted = 1  # some may have gone
            if deleted and self.on_sent is not None:
                self.on_sent()

    async def _deletes(self, conn: Any, view: View) -> int:
        deleted = 0
        s = view.s
        if s is not None and s_version(s) > self._joined:
            self._joined = s_version(s)
            if s.get("text"):
                up_to = s.get("up_to")
                up_to = float(up_to) if isinstance(up_to, (int, float)) else None
                # by its cut item; by its time when the item is not in this session's ledger
                deleted += await self.prune.delete_before(conn, s.get("cut") or None, up_to=up_to)
        if self._drop is not None and view.block.through == self._drop:
            item, self._drop = self._drop, None
            deleted += await self.prune.delete_before(conn, item, through=True)
        return deleted

    def _log(self, why: str, text: str) -> None:
        view, now = self.view, self.clock()
        if view is None:
            return
        block = view.block
        s = view.s
        s_part = (
            "no S"
            if s is None
            else f"S v{s_version(s):.3f} {'cleared' if not s.get('text') else 'held'}, "
            f"{now - view.s_at:.1f} s old"
        )
        # with tokens at 3.5 chars each: drops have come at 14.4k-22k input tokens (11-03)
        logger.info(
            "Instructions sent (%s): %d chars, about %d tokens; block v%d to row %d, %.1f s old; "
            "%s; %d message(s); %d write(s) in the last minute",
            why,
            len(text),
            round(len(text) / 3.5),
            block.version,
            block.top,
            now - block.at,
            s_part,
            len(view.messages),
            len(self._writes),
        )
