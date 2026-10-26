"""Playback health: keep the speaker stream's latency bounded, and say so in the log.

The 65 s lag (the robot, 2026-10-14/15): speech played 64-65 s late, steadily, and a
barge-in could not clear it; recreating the stream did. pulsesink writes to
pipewire-pulse at absolute stream offsets taken from its ring buffer position, and
asks for the default maxlength, 4 MiB: 65.5 s of the app's F32LE mono 16 kHz.
The SDK keeps capture and playback in one pipeline, which takes its clock from the
microphone's pulsesrc, so pulsesink is slaved to the microphone. After a capture
stall (the robot had them: 10 s of nothing, 80 s lost in 200 s) pulsesink re-maps
its offsets, and its write pointer lands far from the server's read pointer. Ahead,
the gap is queued silence the server pins at maxlength (the robot's 65 s). Behind,
every reply is written into the past and never heard (the trixie container, after a
75 s PipeWire stop). Neither pulsesink nor pipewire-pulse offers a way to lower
maxlength: pulsesink always asks for -1, and the server only honours a smaller
client value.

Playback latency is measured end to end, per reply: from its first push to the first
sound of it on the speaker's monitor (ReplyTracker). pa_stream_get_latency, read
through pulsesink's ring buffer, is no measure: it read 0 on the robot while speech
played (2026-10-15) and in every trixie-container reading. A reply heard more than
LATENCY_LIMIT_S late is logged as a WARNING and nothing else: the measure can be
fooled (a pause in a backlog, or a motif, reads as a reply's onset), so it may not
rebuild the speaker and cut a reply off (owner, 2026-10-16: log only). Once a minute
the watchdog logs the last measured lag and its age.
"""

from __future__ import annotations
import os
import re
import math
import time
import logging
import datetime
import functools
import threading
from typing import Any, TypeVar, Callable, Iterable, Sequence, NamedTuple


logger = logging.getLogger(__name__)

LATENCY_LIMIT_S = 5.0
CHECK_INTERVAL_S = 60.0
# One cooldown for every rebuild trigger: 30 s after a rebuild, doubling with each
# further rebuild in the last 10 minutes, capped at 10 minutes. A fault a rebuild
# cannot clear costs a handful of rebuilds, then about one every 10 minutes.
REBUILD_COOLDOWN_S = 30.0
REBUILD_COOLDOWN_MAX_S = 600.0
REBUILD_WINDOW_S = 600.0

# Held by capture on/off (LocalStream's conversation.capture) and by every reading or
# rebuild of the speaker, so neither runs while the pipeline is being stopped or started.
MEDIA_LOCK = threading.RLock()
_F = TypeVar("_F", bound=Callable[..., Any])


def under_media_lock(func: _F) -> _F:
    """Run ``func`` holding MEDIA_LOCK."""

    @functools.wraps(func)
    def locked(*args: Any, **kwargs: Any) -> Any:
        with MEDIA_LOCK:
            return func(*args, **kwargs)

    return locked  # type: ignore[return-value]


class Verdict(NamedTuple):
    """What the watchdog does about one reading, and why."""

    action: str  # "ok" | "rebuild" | "unmeasured"
    cause: str


def decide(latency_s: float | None, limit_s: float = LATENCY_LIMIT_S) -> Verdict:
    """Decide from one latency reading. Pure."""
    if latency_s is None or not math.isfinite(latency_s) or latency_s < 0:
        return Verdict("unmeasured", "no playback stream latency to read")
    if latency_s > limit_s:
        return Verdict("rebuild", f"playback latency {latency_s:.1f} s is over the {limit_s:.1f} s bound")
    return Verdict("ok", f"playback latency {latency_s:.3f} s is within the {limit_s:.1f} s bound")


def latency_bounded(latency_s: float | None, limit_s: float = LATENCY_LIMIT_S) -> bool:
    """Check the invariant, playback latency stays bounded; no reading is not a violation."""
    return decide(latency_s, limit_s).action != "rebuild"


def _iterate(iterator: Any) -> list[Any]:
    """Walk a Gst.Iterator with next(): the robot has no gst-python overrides."""
    from gi.repository import Gst

    found: list[Any] = []
    while True:
        result, item = iterator.next()
        if result == Gst.IteratorResult.RESYNC:
            iterator.resync()
            found.clear()
            continue
        if result != Gst.IteratorResult.OK:
            return found
        found.append(item)


def speaker_sink(pipeline: Any) -> Any:
    """Find the playback branch's audio sink: the sink downstream of the speech appsrc, not an appsink."""
    if pipeline is None:
        return None
    sources = [
        e
        for e in _iterate(pipeline.iterate_recurse())
        if e.get_factory() is not None and e.get_factory().get_name() == "appsrc"
    ]
    seen: list[Any] = []
    frontier = list(sources)
    while frontier:
        element = frontier.pop(0)
        if any(element is s for s in seen):
            continue
        seen.append(element)
        factory = element.get_factory()
        name = factory.get_name() if factory is not None else ""
        src_pads = _iterate(element.iterate_src_pads())
        if not src_pads and name != "appsink":
            return element  # a leaf of the branch: the sink
        for pad in src_pads:
            peer = pad.get_peer()
            parent = peer.get_parent_element() if peer is not None else None
            if parent is not None:
                frontier.append(parent)
    return None


def rebuild_sink(sink: Any, timeout_s: float = 2.0) -> bool:
    """Recreate the speaker's stream: the sink element alone, the pipeline stays PLAYING.

    The sink's upstream pad is blocked first, so its streaming thread never sees the
    sink flushing (a FLUSHING return would stop the branch for good), and the pads are
    relinked so the stream's caps and segment are sent again. Returns whether the
    sink came back.
    """
    from gi.repository import Gst

    sinkpad = sink.get_static_pad("sink")
    upstream = sinkpad.get_peer() if sinkpad is not None else None
    if upstream is None:
        return False
    blocked = threading.Event()

    def on_block(_pad: Any, _info: Any) -> Any:
        blocked.set()
        return Gst.PadProbeReturn.OK

    probe = upstream.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, on_block)
    try:
        if not blocked.wait(timeout_s):
            logger.warning("Speaker stream rebuild: no audio reached the sink in %.1f s to block", timeout_s)
        upstream.unlink(sinkpad)
        sink.set_state(Gst.State.NULL)
        upstream.link(sinkpad)
        # Rejoin without prerolling: an async sink in a PLAYING pipeline would post
        # ASYNC_START and the pipeline would report its state lost until the preroll.
        was_async = sink.get_property("async")
        sink.set_property("async", False)
        ok = sink.sync_state_with_parent()
        sink.set_property("async", was_async)
    finally:
        upstream.remove_probe(probe)
    return bool(ok)


def rebuild_decision(
    now: float,
    history: Sequence[float],
    base_s: float = REBUILD_COOLDOWN_S,
    max_s: float = REBUILD_COOLDOWN_MAX_S,
    window_s: float = REBUILD_WINDOW_S,
) -> Verdict:
    """Decide whether a rebuild may run now, from the times of earlier ones. Pure; shared by every trigger."""
    recent = [t for t in history if now - t < window_s]
    if not recent:
        return Verdict("rebuild", f"no rebuild in the last {window_s:.0f} s")
    cooldown = min(max_s, base_s * 2 ** (len(recent) - 1))
    left = recent[-1] + cooldown - now
    if left > 0:
        return Verdict(
            "cooldown",
            f"{len(recent)} rebuild(s) in the last {window_s:.0f} s, cooldown {cooldown:.0f} s, "
            f"next allowed in {left:.0f} s",
        )
    return Verdict("rebuild", f"the {cooldown:.0f} s cooldown after {len(recent)} rebuild(s) has passed")


def _playing(pipeline: Any) -> bool:
    """Return whether the pipeline is PLAYING now; capture off sets it to NULL."""
    from gi.repository import Gst

    try:
        return bool(pipeline is not None and pipeline.get_state(0)[1] == Gst.State.PLAYING)
    except Exception:
        return False


def restart_speaker_clock(audio: Any) -> bool:
    """Restart the speaker's clock after its sink went through NULL; nothing to do on the microphone's clock."""
    return False


class RebuildGovernor:
    """Run speaker-stream rebuilds for every trigger, under one cooldown, each logged with its trigger."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Start with no rebuild history; ``clock`` is injectable for tests."""
        self._clock = clock
        self._history: list[float] = []
        self._skipped: str | None = None  # the last skip's line, logged once until it changes
        self.last_rebuild_at: float | None = None

    def _skip(self, trigger: str, why: str) -> bool:
        """Log a skipped rebuild at INFO once per cause, then at DEBUG. Holds MEDIA_LOCK."""
        line = f"Speaker stream rebuild skipped (trigger: {trigger}): {why}"
        logger.log(logging.DEBUG if line == self._skipped else logging.INFO, "%s", line)
        self._skipped = line
        return False

    def rebuild(self, audio: Any, trigger: str, cause: str, level: int = logging.WARNING) -> bool:
        """Rebuild the speaker stream for ``trigger`` unless the cooldown holds it back.

        Returns whether a rebuild ran and the sink came back. A pipeline that is not
        PLAYING (capture off) is left alone and uses no cooldown. ``level`` is the
        "Rebuilding" line's: a fault's rebuild is a WARNING, a routine one need not be.
        """
        with MEDIA_LOCK:
            pipeline = getattr(audio, "_pipeline", None)
            if not _playing(pipeline):
                return self._skip(trigger, "the pipeline is not playing")
            now = self._clock()
            verdict = rebuild_decision(now, self._history)
            if verdict.action != "rebuild":
                logger.info(
                    "Speaker stream rebuild held back (trigger: %s, cause: %s): %s", trigger, cause, verdict.cause
                )
                return False
            sink = speaker_sink(pipeline)
            if sink is None:
                return self._skip(trigger, "no speaker sink")
            self._skipped = None
            self._history = [t for t in self._history if now - t < REBUILD_WINDOW_S] + [now]
            self.last_rebuild_at = now
            logger.log(
                level,
                "Rebuilding the speaker stream (trigger: %s, cause: %s; rebuild %d in the last %.0f s)",
                trigger,
                cause,
                len(self._history),
                REBUILD_WINDOW_S,
            )
            ok = rebuild_sink(sink)
            try:
                restarted = restart_speaker_clock(audio)
            except Exception:
                logger.warning("Could not restart the speaker's clock after the rebuild", exc_info=True)
                restarted = False
            logger.info(
                "Speaker stream rebuilt (trigger: %s): sink back=%s%s", trigger, ok, ", primed" if restarted else ""
            )
            return bool(ok)


GOVERNOR = RebuildGovernor()

# The idle refresh (2026-10-18, 18:40): after 75 minutes with nothing played, the
# first four replies of a wake were pushed but not heard, until the unheard-reply
# trigger rebuilt the stream 33 s in. So a session that opens after this long with
# nothing pushed rebuilds the speaker stream first, through GOVERNOR. 0 turns it off.
REFRESH_IDLE_ENV = "REACHY_COMPANION_SPEAKER_REFRESH_IDLE_S"
REFRESH_IDLE_S = 600.0


def seconds_from_env(name: str, default: float, environ: Any = os.environ) -> float:
    """Return a threshold in seconds from the environment variable ``name``; 0 means off."""
    raw = str(environ.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        logger.warning("%s=%r is not a number of seconds >= 0; using %.0f", name, raw, default)
        return default
    return value


def refresh_decision(now: float, fresh_at: Sequence[float | None], idle_s: float = REFRESH_IDLE_S) -> Verdict:
    """Decide whether a session opening at ``now`` rebuilds the speaker stream first. Pure.

    ``fresh_at``: when the stream last carried a reply, was rebuilt, or was started;
    None for what has not happened.
    """
    known = [t for t in fresh_at if t is not None]
    if idle_s <= 0 or not known:
        return Verdict("ok", "the idle refresh is off" if idle_s <= 0 else "nothing known about the stream")
    idle = now - max(known)
    if idle > idle_s:
        return Verdict("refresh", f"{idle / 60:.0f} min idle")
    return Verdict("ok", f"{idle:.0f} s idle, under {idle_s:.0f} s")


def refresh_after_idle(
    audio: Any, cause: str, governor: RebuildGovernor | None = None, level: int = logging.WARNING
) -> bool:
    """Rebuild the speaker stream for the idle refresh, and log whether it was refreshed."""
    ok = (governor or GOVERNOR).rebuild(audio, "idle refresh", cause, level)
    if ok:
        logger.info("Speaker stream refreshed after %s", cause)
    return ok


# The periodic idle rebuild (2026-10-26, 05:22): a speech stream with no write for hours
# replayed the last 65 s it carried. With prebuf 0, pipewire-pulse advances the stream's
# read index through an underrun; 2^31 bytes on (9 h 19 m at 64,000 B/s) the 32-bit
# difference to the write index wraps, and the server plays the last 4 MiB written. So
# the watchdog rebuilds the stream after this long with no write, through GOVERNOR, when
# no reply, session or sound is playing. 0 turns it off.
IDLE_REBUILD_ENV = "REACHY_COMPANION_SPEAKER_IDLE_REBUILD_S"
IDLE_REBUILD_S = 3 * 3600.0


def idle_rebuild_decision(
    now: float,
    fresh_at: Sequence[float | None],
    idle_s: float = IDLE_REBUILD_S,
    *,
    reply: bool = False,
    session: bool = False,
    sound: bool = False,
) -> Verdict:
    """Decide whether the watchdog rebuilds a speech stream left idle. Pure.

    ``fresh_at`` as for refresh_decision; ``reply``, ``session`` and ``sound`` say what
    of the app's is playing or open, and any of them holds the rebuild back.
    """
    busy = (
        (reply, "a reply is playing"),
        (session, "a session is open"),
        (sound, "a sound is playing"),
    )
    held = [why for on, why in busy if on]
    if held:
        return Verdict("ok", "held: " + ", ".join(held))
    verdict = refresh_decision(now, fresh_at, idle_s)
    if verdict.action != "refresh":
        return verdict
    cause = f"{verdict.cause} with no write to the speech stream (periodic rebuild)"
    return Verdict("rebuild", cause)


# The replay check: journal lines that mean the speech stream was written to or made new,
# and the line that closes it (capture off sets the pipeline to NULL).
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


def lag_report(last: tuple[float, float] | None, now: float) -> str:
    """Say what the minute line reports: the last measured lag and its age, or that none was. Pure."""
    if last is None:
        return "no reply measured"
    lag_s, at = last
    return f"last reply heard after {lag_s:.2f} s, {max(0.0, now - at):.0f} s ago"


def check_playback(
    pipeline: Any, limit_s: float = LATENCY_LIMIT_S, replies: Any = None, now: float | None = None
) -> Verdict:
    """Log the minute line, the last measured reply lag; the rebuild belongs to the per-reply trigger."""
    with MEDIA_LOCK:
        if not _playing(pipeline):
            return Verdict("unmeasured", "the pipeline is not playing")
        sink = speaker_sink(pipeline)
    name = sink.get_name() if sink is not None else "none"
    last = replies.last_lag() if replies is not None else None
    line = lag_report(last, time.monotonic() if now is None else now)
    logger.info("Playback latency: %s (speaker %s, bound %.1f s)", line, name, limit_s)
    if last is None:
        return Verdict("unmeasured", line)
    return Verdict("ok" if latency_bounded(last[0], limit_s) else "over", line)


# --- the capture-gap trigger ---------------------------------------------------------------
#
# A capture stall is what desynced pulsesink from the server (the module docstring):
# after one, replies may be written behind the server's read pointer and never heard,
# and pa_stream_get_latency reads 0 then, so the latency watchdog cannot see it.
# Frames normally arrive every 10 ms; the robot's harmful gaps were 10 s and more
# (2026-10-10), the container's 75 s. 5 s is far above any scheduling hiccup (the
# record loop's own pull stalls were 20 ms) and well under the robot's gaps, and a
# needless rebuild costs a ~0.1 s blip in playback. While frames are missing, after
# 3 s one WARNING says so and the speaker's clock is restarted (it may be what stopped).

CAPTURE_GAP_S = 5.0
CAPTURE_STALL_S = 3.0


def capture_gap_decision(gap_s: float, limit_s: float = CAPTURE_GAP_S) -> Verdict:
    """Decide from a gap between two microphone frames, seen when the second arrives. Pure."""
    if gap_s > limit_s:
        return Verdict(
            "rebuild",
            f"a {gap_s:.1f} s gap in microphone frames with capture on (bound {limit_s:.1f} s); "
            "a capture stall can leave pulsesink writing where the server never plays",
        )
    return Verdict("ok", f"microphone frames {gap_s:.3f} s apart")


def capture_stall_decision(age_s: float, warned: bool, stall_s: float = CAPTURE_STALL_S) -> Verdict:
    """Decide while no frame comes: once per stall, past ``stall_s``, it is reported. Pure."""
    if age_s > stall_s and not warned:
        return Verdict("stalled", f"no microphone frames for {age_s:.1f} s with capture on")
    return Verdict("ok", f"last microphone frame {age_s:.1f} s ago")


class CaptureWatch:
    """Track microphone frame arrivals for the record loop; times are passed in, so it is deterministic."""

    def __init__(self, gap_s: float = CAPTURE_GAP_S, stall_s: float = CAPTURE_STALL_S) -> None:
        """Start with no frame seen."""
        self._gap_s = gap_s
        self._stall_s = stall_s
        self.reset()

    def reset(self) -> None:
        """Forget the last frame: capture was off, and that time is no gap."""
        self._last: float | None = None
        self._warned = False

    def frame(self, now: float) -> Verdict:
        """Note a frame at ``now``; a gap over the bound before it asks for a rebuild."""
        last, self._last, self._warned = self._last, now, False
        if last is None:
            return Verdict("ok", "first microphone frame")
        return capture_gap_decision(now - last, self._gap_s)

    def idle(self, now: float) -> Verdict:
        """Note a poll with no frame at ``now``; the first such poll after a reset starts the wait."""
        if self._last is None:
            self._last = now
            return Verdict("ok", "waiting for the first microphone frame")
        verdict = capture_stall_decision(now - self._last, self._warned, self._stall_s)
        if verdict.action == "stalled":
            self._warned = True
        return verdict


def act_on_capture(verdict: Verdict, audio: Any, governor: RebuildGovernor | None = None) -> None:
    """Carry out a CaptureWatch verdict, off the event loop: report a stall, or rebuild after a gap."""
    if verdict.action == "stalled":
        logger.warning("Capture stalled: %s; restarting the speaker's clock in case it stopped", verdict.cause)
        try:
            restart_speaker_clock(audio)
        except Exception:
            logger.warning("Could not restart the speaker's clock", exc_info=True)
    elif verdict.action == "rebuild":
        (governor or GOVERNOR).rebuild(audio, "capture gap", verdict.cause)


# --- the unheard-reply trigger -------------------------------------------------------------
#
# "Heard" is measured on the speaker's monitor source: what the server actually mixed
# into the device, in a separate capture pipeline. Nothing inside the app's pipeline
# can tell: measured in the trixie container under the microphone's clock, before and
# after a 75 s capture stall (every reply heard in 0.14 s, then none), per 1 s reply
# the bytes reaching pulsesink's pad (208,000 vs 208,640), basesink's rendered count
# (325 vs 326), the ring buffer's samples_done, the sink clock's advance (3.25 s vs
# 3.26 s) and its position query were the same whether the reply was heard or not:
# pulsesink consumes and clocks every buffer as usual, and the server discards the
# writes that land behind its read pointer. pa_stream_get_latency reads 0 then. The
# monitor carries the mix before the sink's volume (PipeWire's monitor.channel-volumes
# is false by default), so a quiet or muted speaker still reads as heard.

UNHEARD_WINDOW_S = 5.0  # a reply is judged this long after its first push
UNHEARD_LIMIT = 3  # unheard replies in a row before a rebuild
REPLY_GAP_S = 1.5  # pushes further apart than this start a new reply
LOUD_RMS = 0.003  # about -50 dBFS: speech, not silence or dither
HEARD_FRACTION = 0.25  # of the speech pushed (up to the window less 1 s), heard by the deadline
MIN_SPEECH_S = 0.3  # less speech than this is not judged

# The measured lag. Each reply's lag is the time from its first push to the first
# sound of it on the monitor. Replies play in the order pushed, so sounds are matched
# to replies first in, first out: the oldest reply not yet heard claims the next sound
# that starts (a loud monitor buffer after a quiet one) after its push. Sound already
# playing at the push (a motif) has no start, so it is never a reply's. A reply pushed
# while the one before it still plays is queued, not lagged: its lag counts from when
# the one before ends (heard at + seconds pushed), and the first loud buffer from then
# is its sound, start or not. LAG_WINDOW_S is longer than the robot's 65 s lag, so
# that lag is measured; a reply heard by then has no lag, and a reply never heard is
# the unheard trigger's case. A rebuild drops the replies pushed before it, so a reply
# it cut off never claims a later reply's sound, and a barge-in drops every reply still
# queued; neither leaves a reply to be queued behind. So does a monitor that delivered too
# little to judge a reply by. The limit: a reply lost without a rebuild (one or two
# unheard, then one heard) claims the next reply's sound, and its lag is logged as
# heard late (a WARNING, never a rebuild).
LAG_WINDOW_S = 90.0
QUEUE_SLACK_S = 0.3  # a queued reply's sound may arrive this much before the estimate
REPLY_TAIL_S = 3.0  # a heard reply is the app's own sound until it has played, and this long


def rms(samples: Any) -> float | None:
    """Return the RMS of a 1-D array of float samples, or None for none. Pure.

    The speaker monitor (loud_seconds) and the phantom detector's pad probe both use it,
    with the one LOUD_RMS floor, so the two cannot disagree about what is loud.
    """
    if samples.size == 0:
        return None
    return math.sqrt(float(samples.dot(samples)) / samples.size)


def loud_seconds(samples: Any, rate: int) -> tuple[float, float]:
    """Return (seconds, seconds above LOUD_RMS) of a buffer of float samples, in 10 ms blocks."""
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0 or rate <= 0:
        return 0.0, 0.0
    block = max(1, rate // 100)
    usable = (data.size // block) * block
    loud = 0
    if usable:
        blocks = np.sqrt(np.mean(np.square(data[:usable].reshape(-1, block)), axis=1))
        loud = int(np.count_nonzero(blocks > LOUD_RMS)) * block
    rest = data[usable:]
    if rest.size and (rms(rest) or 0.0) > LOUD_RMS:
        loud += rest.size
    return data.size / rate, loud / rate


def reply_verdict(
    speech_pushed_s: float, heard_s: float, monitored_s: float, window_s: float = UNHEARD_WINDOW_S
) -> Verdict:
    """Judge one reply at its deadline, from speech pushed and speech the monitor heard in its window. Pure."""
    expected = min(speech_pushed_s, window_s - 1.0)
    if expected < MIN_SPEECH_S:
        return Verdict("skip", f"{speech_pushed_s:.2f} s of speech pushed, too little to judge")
    if monitored_s < 0.5 * window_s:
        return Verdict("unmeasured", f"the speaker's monitor delivered {monitored_s:.1f} s of {window_s:.0f} s")
    if heard_s >= HEARD_FRACTION * expected:
        return Verdict("heard", f"{heard_s:.2f} s of speech heard of {expected:.2f} s expected")
    return Verdict("unheard", f"{heard_s:.2f} s of speech heard of {expected:.2f} s expected within {window_s:.0f} s")


def unheard_streak_decision(count: int, limit: int = UNHEARD_LIMIT) -> Verdict:
    """Decide from the count of unheard replies in a row. Pure."""
    if count >= limit:
        return Verdict("rebuild", f"{count} replies in a row pushed but not heard within {UNHEARD_WINDOW_S:.0f} s")
    return Verdict("ok", f"{count} unheard in a row")


class ReplyTracker:
    """Group pushed audio into replies, judge each against the speaker's monitor, count unheard ones.

    Times are passed in, so it is deterministic. Thread-safe: pushes and flushes come
    from the event loop, monitor buffers from GStreamer's thread.
    """

    def __init__(
        self,
        window_s: float = UNHEARD_WINDOW_S,
        limit: int = UNHEARD_LIMIT,
        rebuilt_at: Callable[[], float | None] | None = None,
    ) -> None:
        """Start with nothing pushed, and disarmed until the monitor has heard one reply.

        ``rebuilt_at`` returns the time of the last speaker rebuild (the governor's, by default).
        """
        self._window_s = window_s
        self._limit = limit
        self._rebuilt_at = rebuilt_at or (lambda: GOVERNOR.last_rebuild_at)
        self._lock = threading.Lock()
        self._replies: list[dict[str, Any]] = []
        self._monitor: list[tuple[float, float, float]] = []  # (arrival, seconds, speech seconds)
        self._was_loud = False  # the last monitor buffer had speech in it
        self._before: dict[str, Any] | None = None  # the last reply heard, which a queued reply waits for
        self._lags: list[float] = []  # measured, not yet reported by tick
        self._last_lag: tuple[float, float] | None = None  # (lag, measured at)
        self._heard_until: float | None = None  # when the last heard reply has played out
        self._heard_at: float | None = None  # when the last heard reply was first heard
        self.unheard_in_a_row = 0
        self.armed = False
        self.last_pushed_at: float | None = None

    def pushed(self, now: float, seconds: float, speech_s: float) -> None:
        """Note audio pushed to the speaker at ``now``."""
        with self._lock:
            self.last_pushed_at = now
            current = self._replies[-1] if self._replies else None
            if current is None or current["flushed"] or now - current["last"] > REPLY_GAP_S:
                current = {
                    "start": now,
                    "last": now,
                    "seconds": 0.0,
                    "speech": 0.0,
                    "flushed": False,
                    "judged": False,
                    "lag": "pending",  # "pending" | "measured" | "none"
                }
                self._replies.append(current)
            current["last"] = now
            current["seconds"] += seconds
            current["speech"] += speech_s
            if current["lag"] == "measured":  # heard while still pushed: it plays this much longer
                ends = current["heard_at"] + current["seconds"]
                self._heard_until = max(self._heard_until or ends, ends)

    def flushed(self, now: float) -> None:
        """Note a barge-in: the reply being pushed is dropped audio, never unheard; earlier ones are judged."""
        with self._lock:
            if self._replies and not self._replies[-1]["judged"]:
                self._replies[-1]["flushed"] = True
            for reply in self._replies:  # everything queued was dropped: no lag, and nothing to wait for
                if reply["lag"] == "pending":
                    reply["lag"] = "none"
            self._before = None
            if self._heard_until is not None:  # the heard reply's queued rest was dropped too
                self._heard_until = min(self._heard_until, now)

    def monitored(self, now: float, seconds: float, speech_s: float) -> None:
        """Note a buffer from the speaker's monitor, arriving at ``now``."""
        with self._lock:
            self._monitor.append((now, seconds, speech_s))
            loud = speech_s > 0.0
            starts = loud and not self._was_loud
            self._was_loud = loud
            if loud:
                self._match(now, starts)

    def _expire(self, now: float) -> None:
        """Give up on replies past the lag window, and on replies a rebuild cut off. Holds the lock."""
        rebuilt = self._rebuilt_at()
        if rebuilt is not None and self._before is not None and self._before["heard_at"] < rebuilt:
            self._before = None  # the rebuild dropped whatever of it was still queued
        heard_at, until = self._heard_at, self._heard_until
        if rebuilt is not None and heard_at is not None and until is not None:
            if heard_at < rebuilt < until:
                self._heard_until = rebuilt  # the rebuild dropped the heard reply's rest too
        for reply in self._replies:
            if reply["lag"] != "pending":
                continue
            if now > reply["start"] + LAG_WINDOW_S or (rebuilt is not None and reply["start"] < rebuilt):
                reply["lag"] = "none"
                self._before = None

    def _match(self, now: float, starts: bool) -> None:
        """Give a loud monitor buffer to the oldest reply not yet heard, if it is that reply's sound."""
        self._expire(now)
        reply = None
        for candidate in self._replies:
            if candidate["lag"] != "pending":
                continue
            if candidate["speech"] < MIN_SPEECH_S and now - candidate["last"] > REPLY_GAP_S:
                candidate["lag"] = "none"  # finished with too little speech to be heard
                continue
            reply = candidate
            break
        if reply is None or now <= reply["start"] or reply["speech"] < MIN_SPEECH_S:
            return
        before = self._before
        ends = before["heard_at"] + before["seconds"] if before is not None else None
        if ends is not None and ends > reply["start"]:
            if now < ends - QUEUE_SLACK_S:
                return  # still the reply before it
            expected = ends
        elif starts:
            expected = reply["start"]
        else:
            return  # sound that started before the push
        reply["lag"] = "measured"
        reply["heard_at"] = now
        self._heard_at = now
        self._heard_until = max(self._heard_until or now, now + reply["seconds"])
        self._before = reply
        lag = max(0.0, now - expected)
        self._lags.append(lag)
        self._last_lag = (lag, now)

    def pending(self, now: float) -> bool:
        """Return whether a pushed reply may still be playing: unheard yet, or heard and not over.

        A reply may play up to LAG_WINDOW_S late (the robot's 65 s lag), and once heard it
        plays for as long as was pushed. The phantom-speech detector counts both as the
        app's own sound: a late reply is never a phantom, its start or its tail.
        """
        with self._lock:
            self._expire(now)  # first, so a rebuild has cut the heard reply short
            if self._heard_until is not None and now < self._heard_until + REPLY_TAIL_S:
                return True
            for reply in self._replies:
                if reply["lag"] == "pending" and not reply["flushed"]:
                    if reply["speech"] >= MIN_SPEECH_S:
                        return True
            return False

    def last_lag(self) -> tuple[float, float] | None:
        """Return (seconds, measured at) of the last measured reply lag, or None."""
        with self._lock:
            return self._last_lag

    def tick(self, now: float) -> list[Verdict]:
        """Judge every reply whose window has closed; a rebuild verdict follows the third unheard in a row."""
        verdicts: list[Verdict] = []
        with self._lock:
            self._expire(now)
            for lag in self._lags:
                verdicts.append(Verdict("heard_after", f"{lag:.2f} s"))
                if not latency_bounded(lag):
                    verdicts.append(
                        Verdict(
                            "heard_late",
                            f"a reply was heard {lag:.1f} s after its first push, "
                            f"over the {LATENCY_LIMIT_S:.1f} s bound",
                        )
                    )
            self._lags.clear()
            for reply in self._replies:
                if reply["judged"] or now < reply["start"] + self._window_s:
                    continue
                reply["judged"] = True
                if reply["flushed"]:
                    verdicts.append(Verdict("flushed", "barge-in: dropped audio, not judged"))
                    continue
                start, end = reply["start"], reply["start"] + self._window_s
                window = [m for m in self._monitor if start <= m[0] <= end]
                verdict = reply_verdict(
                    reply["speech"], sum(m[2] for m in window), sum(m[1] for m in window), self._window_s
                )
                verdicts.append(verdict)
                if verdict.action == "unmeasured" and reply["lag"] == "pending":
                    reply["lag"] = "none"  # the monitor was down: its sound may come later, or never
                    self._before = None
                if verdict.action == "heard":
                    self.unheard_in_a_row = 0
                    self.armed = True
                elif verdict.action == "unheard" and self.armed:
                    self.unheard_in_a_row += 1
                    streak = unheard_streak_decision(self.unheard_in_a_row, self._limit)
                    if streak.action == "rebuild":
                        verdicts.append(Verdict("rebuild", f"{streak.cause}; last: {verdict.cause}"))
                        self.unheard_in_a_row = 0
            horizon = now - 2 * self._window_s
            self._replies = [
                r for r in self._replies if not r["judged"] or r["last"] > horizon or r["lag"] == "pending"
            ]
            self._monitor = [m for m in self._monitor if m[0] > horizon]
        return verdicts


class SpeakerMonitor:
    """Capture the speaker's monitor source in its own pipeline and report each buffer's speech."""

    def __init__(self, on_buffer: Callable[[float, float, float], None]) -> None:
        """``on_buffer(now, seconds, speech_seconds)`` is called from GStreamer's thread."""
        self._on_buffer = on_buffer
        self._pipeline: Any = None
        self.source: str | None = None
        self.last_buffer_at: float | None = None

    @staticmethod
    def monitor_source(sink: Any) -> str | None:
        """Name the monitor of the device pulsesink is playing to, or None for any other sink."""
        factory = sink.get_factory() if sink is not None else None
        if factory is None or factory.get_name() != "pulsesink":
            return None
        device = sink.get_property("current-device") or sink.get_property("device")
        return f"{device}.monitor" if device else None

    def start(self, source: str) -> bool:
        """Start capturing ``source``; returns whether it is running."""
        from gi.repository import Gst

        self.stop()
        pipeline = Gst.Pipeline.new("speaker_monitor")
        src = Gst.ElementFactory.make("pulsesrc")
        convert = Gst.ElementFactory.make("audioconvert")
        resample = Gst.ElementFactory.make("audioresample")
        appsink = Gst.ElementFactory.make("appsink")
        if not all((src, convert, resample, appsink)):
            return False
        src.set_property("device", source)
        src.set_property("buffer-time", 200_000)
        src.set_property("latency-time", 50_000)
        appsink.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=F32LE,channels=1,rate=16000"))
        appsink.set_property("sync", False)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 50)
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_sample)
        for element in (src, convert, resample, appsink):
            pipeline.add(element)
        src.link(convert)
        convert.link(resample)
        resample.link(appsink)
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            return False
        self._pipeline = pipeline
        self.source = source
        self.last_buffer_at = time.monotonic()
        logger.info("Speaker monitor started on %s (heard replies are measured here)", source)
        return True

    def _on_sample(self, appsink: Any) -> Any:
        from gi.repository import Gst

        sample = appsink.emit("pull-sample")
        if sample is not None:
            import numpy as np

            buf = sample.get_buffer()
            data = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.float32)
            now = time.monotonic()
            self.last_buffer_at = now
            seconds, speech = loud_seconds(data, 16000)
            self._on_buffer(now, seconds, speech)
        return Gst.FlowReturn.OK

    def stop(self) -> None:
        """Stop capturing."""
        from gi.repository import Gst

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self.source = None


def current_speaker_sink(audio: Any, cache: dict[str, Any] | None = None) -> tuple[bool, Any]:
    """Return (the app pipeline is playing, its speaker sink or None), under MEDIA_LOCK.

    With ``cache``, the sink found for a pipeline is reused while the pipeline is the same:
    a rebuild keeps the sink element (rebuild_sink).
    """
    with MEDIA_LOCK:
        pipeline = getattr(audio, "_pipeline", None)
        if not _playing(pipeline):
            return False, None
        if cache is not None and cache.get("pipeline") is pipeline:
            return True, cache["sink"]
        sink = speaker_sink(pipeline)
        if cache is not None and sink is not None:  # a sink not found yet is looked up again
            cache.update(pipeline=pipeline, sink=sink)
        return True, sink


def keep_monitor(
    monitor: SpeakerMonitor,
    audio: Any,
    now: float,
    stale_s: float = 5.0,
    found: tuple[bool, Any] | None = None,
) -> str:
    """Keep the speaker monitor on the device the speaker plays to; restart it if it went quiet.

    ``found`` is current_speaker_sink's answer when the caller already looked it up.
    Returns what it did: "running", "started", "stopped" or "unavailable".
    """
    playing, sink = found if found is not None else current_speaker_sink(audio)
    if not playing:
        if monitor.source is not None:
            monitor.stop()
            return "stopped"
        return "unavailable"
    source = SpeakerMonitor.monitor_source(sink)
    if source is None:
        return "unavailable"
    stale = monitor.last_buffer_at is None or now - monitor.last_buffer_at > stale_s
    if monitor.source == source and not stale:
        return "running"
    if monitor.source is not None and stale:
        logger.info("Speaker monitor on %s delivered nothing for %.0f s: restarting it", monitor.source, stale_s)
    return "started" if monitor.start(source) else "unavailable"


# --- play_sound playbins: a second speaker stream per sound -------------------------------
#
# The SDK's play_sound builds a playbin with its own speaker sink and never tears it
# down: its stream stays open after the sound, and pipewire-pulse replays the last
# 4 MiB of a stream left open with nothing written (every 6.8 h at 22,050 Hz). Every
# play_sound caller (conversation.sound, the app tools, the SDK's play_move) passes
# through the instance wrap in phantom_speech.SoundStarts, which holds SOUND_LOCK while
# the SDK replaces audio._playbin and then hands the new playbin to watch_sound.


SOUND_LOCK = threading.Lock()
SOUND_SLACK_S = 5.0  # a sound with no end this long past its duration is torn down
SOUND_MAX_S = 1800.0  # the same bound for a sound whose duration is not known


class SoundActivity:
    """Which play_sound playbins are open, and when the last one closed.

    The phantom-speech detector and the idle rebuild read it: a sound is the app's own.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Start with no sound open."""
        self._clock = clock
        self._lock = threading.Lock()
        self._open: set[int] = set()
        self._closed_at: float | None = None

    def opened(self, playbin: Any) -> None:
        """Note a sound's playbin as playing."""
        with self._lock:
            self._open.add(id(playbin))

    def closed(self, playbin: Any) -> None:
        """Note a sound's playbin as ended or torn down; closing one twice is harmless."""
        with self._lock:
            self._open.discard(id(playbin))
            self._closed_at = self._clock()

    def is_open(self, playbin: Any) -> bool:
        """Return whether ``playbin`` is open (watched and not yet closed)."""
        with self._lock:
            return id(playbin) in self._open

    def state(self) -> tuple[bool, float | None]:
        """Return (a sound is open, when the last one closed)."""
        with self._lock:
            return bool(self._open), self._closed_at


SOUNDS = SoundActivity()


def watch_sound(audio: Any, playbin: Any, path: Any) -> threading.Thread:
    """Mark a play_sound playbin open, and tear it down from its own thread when the sound ends.

    The play_sound wrap calls this holding SOUND_LOCK, so it must not take the lock
    itself before it returns. The thread waits on the playbin's bus for the end or an
    error, gives up on a sound with no end SOUND_SLACK_S past its duration, and sets it
    to NULL if it is still the current sound. A newer sound has replaced it already
    (the SDK's play_sound set it to NULL), and so has capture off (stop_playing).
    """
    from gi.repository import Gst

    logger.info("Sound stream opened: %s (a second speaker stream beside speech)", path)
    SOUNDS.opened(playbin)
    started = time.monotonic()

    def watch() -> tuple[str, bool]:
        """Wait for the sound's end; returns (the cause, whether it ran past its bound)."""
        bus, duration_of = playbin.get_bus(), playbin.query_duration
        bound = SOUND_MAX_S
        ends = Gst.MessageType.EOS | Gst.MessageType.ERROR
        while True:
            current = getattr(audio, "_playbin", None)
            if current is None:
                return "stopped", False
            if current is not playbin:
                return "replaced by a newer sound", False
            msg = bus.timed_pop_filtered(200 * Gst.MSECOND, ends)
            if msg is not None and msg.type == Gst.MessageType.ERROR:
                err, _debug = msg.parse_error()
                return f"error: {err.message}", False
            if msg is not None:
                return "the sound ended", False
            known, duration = duration_of(Gst.Format.TIME)
            if known and duration >= 0:
                bound = duration / Gst.SECOND + SOUND_SLACK_S
            if time.monotonic() - started > bound:
                return f"no end {bound:.0f} s after it started", True

    def wait() -> None:
        try:
            cause, overdue = watch()
            if not cause.startswith(("the sound ended", "error", "no end")):
                logger.info("Sound stream closed (%s): %s", cause, path)  # NULL already
                return
            with SOUND_LOCK:  # a new sound cannot take _playbin between the check and the clear
                current = getattr(audio, "_playbin", None)
                ours = current is not None and current is playbin
                if ours:
                    audio._playbin = None
                    playbin.set_state(Gst.State.NULL)
            how = "torn down" if ours else "then replaced or stopped, left to the SDK"
            log = logger.warning if overdue else logger.info
            log("Sound stream closed (%s, %s): %s", cause, how, path)
        finally:
            SOUNDS.closed(playbin)

    thread = threading.Thread(target=wait, name="sound-watch", daemon=True)
    thread.start()
    return thread


def stop_sound(audio: Any) -> bool:
    """conversation.sound's stop: the current sound's stream alone, never speech."""
    from gi.repository import Gst

    with SOUND_LOCK:
        playbin = getattr(audio, "_playbin", None)
        if playbin is None:
            return False
        audio._playbin = None
        playbin.set_state(Gst.State.NULL)
    SOUNDS.closed(playbin)  # its watcher logs the close
    logger.debug("Sound stream stop requested")
    return True
