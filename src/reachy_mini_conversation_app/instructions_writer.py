"""The writer of his instructions (route all, Unit 4): the only sender of session instructions.

The companion's model is ConvoAll, round 6 (docs/formal/conversation-d/model.md on its branch
arch-d-model), rules 2-5, 7 and 9. His instructions are composed here from the parts, then the
stable base (the profile), near the turn, as the fork before route all had it (the owner, 11-23:
after route all put the profile first he stopped calling recall, and the prompt cache it was
first for never paid off, `cached 0` on all 995 turns from 11-21 to 11-23), then the moment
thinker's line last of all:

- the session block: at a session's open, the companion's file read unbounded (it names the S it
  is cut at, with its text); after that, the companion's "block" part for this session, bounded
  at the turns this session's chat does not hold;
- S, the talk's earlier part in short: the newer, by S's own version, of the S the block names
  and the companion's "summary" part (rule 8: S arrives by both; a clear is a newer version);
- the recent text messages he answered in this talk: the companion's "messages" part (Unit 6,
  ConvoAll rule 4 as S), its lines composed as they come. The app keeps no record of its own;
- what is true just now: the companion's "now" part (Unit 5, ConvoAll rule 10), the last part,
  before the base. Its speaker line is shown only when its decision's stamp is not before
  this view's memory_read_at (both on the host's monotonic clock): an older session's speaker is
  never shown. A part past its expiry is not taken. His agenda (Unit 10) is the part's `agenda`
  lines, after the speaker: what is set, in the companion's words;
- the stable base;
- this moment (Units 8-9, model.md "The final rule set (F)", rules 6-9): the companion's "line"
  part, the moment thinker's act for his next reply, after the base (the owner's design: ticket
  11's gain was measured with it at the end). It is shown only while its item is the open item:
  the person's turn whose speech_started the writer has seen and whose transcript has not ended
  (completed, empty included, or failed); a session's open has none. The take has no check of its
  own, so a resend or a snapshot of a line whose item ended writes nothing. "quiet" is shown as
  QUIET_LINE. About 20-40 tokens at 3.5 characters a token: the quiet line 19, an act at the
  companion's 120-character cap 39. The line is not held to its expiry: the open item bounds it.

`take` and `compose` are pure: what the parts record changes, what goes at once and why, and the
text. The writer sends (rules 3-5, 9):

- a join of S, a clear (of S or of the messages), the reconnect's catch-up (the companion marks it: the read at the open
  held less than the bound) and a drop's block go at once, on the handler's loop;
- an answered message goes at once too, but does not start the drop measure again (Unit 6
  review: S's join goes at once, and the record is S-shaped);
- a speaker change, a guest's arrival, the agenda guarded (Unit 10: a guest changed a line) and
  "just now" dropped (its expiry, or a snapshot that lacks it) go at once and do not start the drop measure again either, unless a block or S reason goes
  in the same send; one timer at the held part's expiry drops it and sends at once;
- any other change is batched, BATCH_S, and goes in a tool's window too (no hold), and never
  sooner than BATCHED_GAP_S after the last write (the Units 3+4 review: trims and awareness
  refreshes came to 5-8 writes a minute, 70-117 KB each with the echo);
- at a reply's done, any change goes before the done event is set (H calls `at_done`);
- the line goes at once when what is shown of it changes, and never starts the drop measure again
  (LINE_CHANGED is a J reason, Fu8measure). At speech_started any pending change goes in one write
  (the withdrawal of an older item's line and whatever else is pending; the companion's ticket 77,
  item 2), and at an empty or failed end of the open item any pending change goes, before H's
  re-create; neither starts the measure again. A transcript with text sends nothing: the line
  stays on the server until his reply's done, so the reply reads it;
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
NOW_HEADING = "### Just now"
NOW_PART = "now"
# The "just now" reasons: at once, and never a drop-measure reset of their own (ConvoAll rule 10).
SPEAKER_CHANGED = "the speaker changed"
GUEST_ARRIVED = "a guest arrived"
AGENDA_GUARDED = "the agenda guarded"  # Unit 10: a guest changed a line of his agenda
NOW_DROPPED = "just now dropped"
# The moment thinker's line (Units 8-9): its own at-once reason, never a measure reset (Fu8measure).
LINE_PART = "line"
LINE_HEADING = "### This moment"
QUIET = "quiet"  # the companion's moment_thinker.QUIET: the words were not said to him
QUIET_LINE = "These words were not said to you: call stay_quiet."  # the profile's form
LINE_CHANGED = "the moment's line"
J_REASONS = frozenset({SPEAKER_CHANGED, GUEST_ARRIVED, AGENDA_GUARDED, NOW_DROPPED, LINE_CHANGED})
ENDED_KEPT = 16  # the items whose end this session has seen, kept for the line's log
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
    now: Optional["Now"] = None
    line: Optional["Line"] = None
    open_item: Optional[str] = None  # the person's turn whose transcript has not ended (rule 6)


@dataclass(frozen=True)
class Now:
    """The companion's "now" part as taken: its value, its version and its expiry (monotonic)."""

    value: dict[str, Any]
    version: int
    expires_at: Optional[float]


@dataclass(frozen=True)
class Line:
    """The companion's "line" part as taken: its item, call, text, version, expiry and arrival."""

    item: str
    call: int
    text: str
    version: int
    expires_at: Optional[float]
    at: float


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


def part_now(part: Any, now: float) -> Optional[Now]:
    """Return the companion's "now" part, or None.

    None is no part, a clear, a value that is not one, or a part past its expiry.
    """
    value = getattr(part, "value", None)
    if not isinstance(value, dict):
        return None
    expires = getattr(part, "expires_at", None)
    if expires is not None and expires <= now:
        return None
    return Now(value, int(getattr(part, "version", 0)), expires)


def part_line(part: Any, at: float) -> Optional[Line]:
    """Return the companion's "line" part, or None for no part, a clear or a value that is not one.

    No other check (rule 8): its session, call and expiry are not read; compose shows it only
    while its item is open.
    """
    value = getattr(part, "value", None)
    if not isinstance(value, dict):
        return None
    item, text, call = value.get("item"), value.get("text"), value.get("call")
    if not isinstance(item, str) or not item or not isinstance(text, str) or not text.strip():
        return None
    return Line(
        item,
        call if isinstance(call, int) and not isinstance(call, bool) else 0,
        text.strip(),
        int(getattr(part, "version", 0)),
        getattr(part, "expires_at", None),
        at,
    )


def shown_line(view: View) -> Optional[Line]:
    """Return the line shown in this view: only while its item is the open item (rule 7)."""
    line = view.line
    if line is None or view.open_item is None or line.item != view.open_item:
        return None
    return line


def line_text(view: View) -> str:
    """Return the "### This moment" section, or "" when no line is shown."""
    line = shown_line(view)
    if line is None:
        return ""
    words = QUIET_LINE if line.text.rstrip(".!").casefold() == QUIET else line.text
    return f"{LINE_HEADING}\n{words}"


def now_speaker(view: View) -> Optional[str]:
    """Return the speaker line shown in this view: the stamp filter (ConvoAll rule 10).

    Its decision's stamp must not be before the view's memory_read_at, so a speaker decided for
    an earlier realtime session is not shown (NoStaleSpeaker).
    """
    j = view.now
    speaker = j.value.get("speaker") if j is not None else None
    if not isinstance(speaker, dict) or view.session is None:
        return None
    text, at = speaker.get("text"), speaker.get("at")
    if not isinstance(text, str) or not text.strip():
        return None
    if isinstance(at, bool) or not isinstance(at, (int, float)) or at < view.session:
        return None
    return text.strip()


def _guest(j: Optional[Now]) -> bool:
    return j is not None and j.value.get("guest") is True


def _guarded(j: Optional[Now]) -> bool:
    return j is not None and j.value.get("guarded") is True


def _lines(raw: Any) -> list[str]:
    return [x.strip() for x in raw if isinstance(x, str) and x.strip()] if isinstance(raw, list) else []


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
    # and drops a part a snapshot lacks. A clear (the talk's end, a forget) is urgent, as S's
    # is (ClearedGone); an answered message goes at once too, from parts_changed, with no
    # measure reset, so it is not listed here.
    messages = part_messages(held.get("messages"))
    if view.messages and not messages:
        urgent.append("messages cleared")
    # "Just now" (Unit 5): a speaker change (a naming, a withdrawal, or a re-stamp a new session
    # can show), a guest's arrival, the agenda guarded (Unit 10) and the part dropped go at once;
    # the rest is batched, the agenda's other changes included.
    line = part_line(held.get(LINE_PART), now)
    if line is not None and view.line is not None and line.version == view.line.version:
        line = view.line  # the same version: its arrival stays
    new = replace(
        view, block=block, s=s, s_at=s_at, messages=messages, now=part_now(held.get(NOW_PART), now), line=line
    )
    if view.now is not None and new.now is None:
        urgent.append(NOW_DROPPED)
    elif now_speaker(new) != now_speaker(view):
        urgent.append(SPEAKER_CHANGED)
    if _guest(new.now) and not _guest(view.now):
        urgent.append(GUEST_ARRIVED)
    if _guarded(new.now) and not _guarded(view.now):
        urgent.append(AGENDA_GUARDED)
    # The line (rule 8): at once whenever what is shown changes; a line for an item not open
    # changes nothing shown, so it writes nothing.
    if line_text(new) != line_text(view):
        urgent.append(LINE_CHANGED)
    return new, urgent


def now_text(view: View) -> str:
    """Return the "### Just now" section, or "" when it has nothing to say.

    The time, who is here, the speaker shown, his agenda, then the lines: the companion's order
    and text.
    """
    j = view.now
    if j is None:
        return ""
    time_line = j.value.get("time")
    lines = [time_line.strip()] if isinstance(time_line, str) and time_line.strip() else []
    lines += _lines(j.value.get("here"))
    speaker = now_speaker(view)
    if speaker:
        lines.append(speaker)
    lines += _lines(j.value.get("agenda"))
    lines += _lines(j.value.get("lines"))
    return f"{NOW_HEADING}\n" + "\n".join(lines) if lines else ""


def compose(view: View) -> str:
    """Return his instructions, the moment's line last (rule 7, nearest the turn).

    The block, S, the messages answered, "just now", the base, then this moment's line.
    """
    parts = [view.block.text.strip()]
    if view.s is not None and str(view.s.get("text") or "").strip():
        parts.append(f"{S_HEADING}\n{' '.join(str(view.s['text']).split())}")
    if view.messages:
        parts.append(f"{MESSAGES_HEADING}\n" + "\n".join(view.messages))
    parts.append(now_text(view))
    parts.append(view.base.strip())
    parts.append(line_text(view))
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
        self._held: dict[str, Any] = {}  # the parts record last taken, which the expiry retakes
        self._expiry: asyncio.TimerHandle | None = None  # the one timer at "now"'s expiry
        self._ended: deque[str] = deque(maxlen=ENDED_KEPT)  # items whose end this session saw
        self._line_sent: Line | None = None  # the line the last write showed, and when it went
        self._line_sent_at = 0.0

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
        self._ended.clear()
        self._line_sent = None
        self._line_arrived(None)
        self._joined = s_version(view.s)
        if self._batch is not None:
            self._batch.cancel()
            self._batch = None
        self._held = held
        self._arm_expiry()
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
        self._held = held
        before, line = self.view.messages, self.view.line
        view, urgent = take(self.view, held, self.clock())
        self.view = view
        self._line_arrived(line)
        self._arm_expiry()
        if urgent:
            # J's reasons alone never start the drop measure again; with a block or S reason,
            # that reason's write measures as it does today (ConvoAll rule 10, Fjmeasure).
            self._send_now(", ".join(urgent), measure=any(r not in J_REASONS for r in urgent))
        elif view.messages and view.messages != before:
            self._send_now("a message answered", measure=False)
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

    # --- the person's turn: the open item (Units 8-9, rules 6 and 9) ----------------------------

    async def turn_started(self, item: str | None) -> None:
        """Take speech_started's item as the open item, and send any pending change in one write.

        The write withdraws an older item's line and carries whatever else waits (rule 9; the
        companion's ticket 77, item 2). It never starts the drop measure again (Fu8measureclr).
        """
        if self.view is None:
            return
        self.view = replace(self.view, open_item=item or None)
        if self.pending():
            await self.send("speech started")

    def turn_ended(self, item: str | None) -> bool:
        """End the open item at its transcript's end (completed, empty included, or failed).

        Return whether it was the open item. Nothing is sent: at a transcript with text the line
        stays on the server for his reply, and the reply's done withdraws it (rule 9).
        """
        if item:
            self._ended.append(item)
        if self.view is None or not item or self.view.open_item != item:
            return False
        self.view = replace(self.view, open_item=None)
        return True

    async def at_empty_end(self, item: str | None, why: str) -> bool:
        """At an empty or failed end of the open item, send any pending change now (rule 9).

        H awaits it before the empty turn's re-create, so the re-create never reads the line.
        It never starts the drop measure again (Fu8measureclr). Return whether it was the open
        item.
        """
        if not self.turn_ended(item):
            return False
        if self.pending():
            await self.send(why)
        return True

    def _arm_expiry(self) -> None:
        """Set the one timer at the held "now" part's expiry (ConvoAll rule 10, JExpire).

        A silent companion or a dead link leaves him no "just now" rather than a stale one.
        """
        if self._expiry is not None:
            self._expiry.cancel()
            self._expiry = None
        j = self.view.now if self.view is not None else None
        if j is None or j.expires_at is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop: the next take drops it
            return
        self._expiry = loop.call_later(max(0.0, j.expires_at - self.clock()), self._expired)

    def _expired(self) -> None:
        self._expiry = None
        if self.view is None:
            return
        view, urgent = take(self.view, self._held, self.clock())
        self.view = view
        if urgent:
            self._send_now(", ".join(urgent), measure=any(r not in J_REASONS for r in urgent))
        else:
            self._arm_expiry()  # the clock read early: wait again

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

    def _send_now(self, why: str, *, measure: bool = True) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self.send(why, measure=measure))
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
                self._log_line(why, shown_line(view), now)
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

    def _line_arrived(self, before: Line | None) -> None:
        """Log a newly taken line version: counts and timings only, never its text or item."""
        view = self.view
        line = view.line if view is not None else None
        if line is None or (before is not None and before.version == line.version):
            return
        if view is not None and line.item == view.open_item:
            where = "its item open"
        elif line.item in self._ended:
            where = "after its item ended"
        else:
            where = "not the open item"
        logger.info(
            "Moment line taken: call %d, v%d, %s, %s; %s",
            line.call,
            line.version,
            "quiet" if line.text.rstrip(".!").casefold() == QUIET else "an act",
            where,
            self._expiry_left(line, line.at),
        )

    def _log_line(self, why: str, shown: Line | None, now: float) -> None:
        """Log a write that showed a new line or withdrew one: counts and timings only."""
        sent, self._line_sent = self._line_sent, shown
        if shown is not None and (sent is None or sent.version != shown.version):
            self._line_sent_at = now
            logger.info(
                "Moment line written (%s): call %d, v%d, %.2f s after it was taken; %s",
                why,
                shown.call,
                shown.version,
                now - shown.at,
                self._expiry_left(shown, now),
            )
        elif shown is None and sent is not None:
            logger.info(
                "Moment line withdrawn (%s): call %d, v%d, shown %.1f s",
                why,
                sent.call,
                sent.version,
                now - self._line_sent_at,
            )

    @staticmethod
    def _expiry_left(line: Line, now: float) -> str:
        expires = line.expires_at
        return "no expiry" if expires is None else f"{expires - now:.1f} s to its expiry"

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
        j = view.now
        at = j.value.get("at_mono") if j is not None else None
        j_part = (
            "no just now"
            if j is None
            else f"just now v{j.version}"
            + (f", {now - at:.1f} s old" if isinstance(at, (int, float)) and not isinstance(at, bool) else "")
            + (", its speaker shown" if now_speaker(view) else "")
        )
        # with tokens at 3.5 chars each: drops have come at 14.4k-22k input tokens (11-03)
        logger.info(
            "Instructions sent (%s): %d chars, about %d tokens; block v%d to row %d, %.1f s old; "
            "%s; %d message(s); %s; %s; %d write(s) in the last minute",
            why,
            len(text),
            round(len(text) / 3.5),
            block.version,
            block.top,
            now - block.at,
            s_part,
            len(view.messages),
            j_part,
            "a line shown" if shown_line(view) is not None else "no line",
            len(self._writes),
        )
