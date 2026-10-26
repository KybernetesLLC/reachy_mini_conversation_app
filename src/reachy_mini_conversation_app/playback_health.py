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
import json
import math
import time
import logging
import datetime
import functools
import threading
import subprocess
from typing import Any, TypeVar, Callable, Sequence, NamedTuple
from collections import deque
from dataclasses import dataclass


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
        self.last_rebuild_at: float | None = None

    def rebuild(self, audio: Any, trigger: str, cause: str) -> bool:
        """Rebuild the speaker stream for ``trigger`` unless the cooldown holds it back.

        Returns whether a rebuild ran and the sink came back. A pipeline that is not
        PLAYING (capture off) is left alone and uses no cooldown.
        """
        with MEDIA_LOCK:
            pipeline = getattr(audio, "_pipeline", None)
            if not _playing(pipeline):
                logger.info("Speaker stream rebuild skipped (trigger: %s): the pipeline is not playing", trigger)
                return False
            now = self._clock()
            verdict = rebuild_decision(now, self._history)
            if verdict.action != "rebuild":
                logger.info(
                    "Speaker stream rebuild held back (trigger: %s, cause: %s): %s", trigger, cause, verdict.cause
                )
                return False
            sink = speaker_sink(pipeline)
            if sink is None:
                logger.info("Speaker stream rebuild skipped (trigger: %s): no speaker sink", trigger)
                return False
            self._history = [t for t in self._history if now - t < REBUILD_WINDOW_S] + [now]
            self.last_rebuild_at = now
            logger.warning(
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


def refresh_idle_s_from_env(environ: Any = os.environ) -> float:
    """Return the idle refresh threshold in seconds from the environment; 0 means off."""
    raw = str(environ.get(REFRESH_IDLE_ENV, "")).strip()
    if not raw:
        return REFRESH_IDLE_S
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        logger.warning("%s=%r is not a number of seconds >= 0; using %.0f", REFRESH_IDLE_ENV, raw, REFRESH_IDLE_S)
        return REFRESH_IDLE_S
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


def refresh_after_idle(audio: Any, cause: str, governor: RebuildGovernor | None = None) -> bool:
    """Rebuild the speaker stream for the idle refresh, and log whether it was refreshed."""
    ok = (governor or GOVERNOR).rebuild(audio, "idle refresh", cause)
    if ok:
        logger.info("Speaker stream refreshed after %s", cause)
    return ok


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
        rms = np.sqrt(np.mean(np.square(data[:usable].reshape(-1, block)), axis=1))
        loud = int(np.count_nonzero(rms > LOUD_RMS)) * block
    rest = data[usable:]
    if rest.size and float(np.sqrt(np.mean(np.square(rest)))) > LOUD_RMS:
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

    def flushed(self, now: float) -> None:
        """Note a barge-in: the reply being pushed is dropped audio, never unheard; earlier ones are judged."""
        with self._lock:
            if self._replies and not self._replies[-1]["judged"]:
                self._replies[-1]["flushed"] = True
            for reply in self._replies:  # everything queued was dropped: no lag, and nothing to wait for
                if reply["lag"] == "pending":
                    reply["lag"] = "none"
            self._before = None

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
        self._before = reply
        lag = max(0.0, now - expected)
        self._lags.append(lag)
        self._last_lag = (lag, now)

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


def keep_monitor(monitor: SpeakerMonitor, audio: Any, now: float, stale_s: float = 5.0) -> str:
    """Keep the speaker monitor on the device the speaker plays to; restart it if it went quiet.

    Returns what it did: "running", "started", "stopped" or "unavailable".
    """
    with MEDIA_LOCK:
        pipeline = getattr(audio, "_pipeline", None)
        if not _playing(pipeline):
            if monitor.source is not None:
                monitor.stop()
                return "stopped"
            return "unavailable"
        source = SpeakerMonitor.monitor_source(speaker_sink(pipeline))
    if source is None:
        return "unavailable"
    stale = monitor.last_buffer_at is None or now - monitor.last_buffer_at > stale_s
    if monitor.source == source and not stale:
        return "running"
    if monitor.source is not None and stale:
        logger.info("Speaker monitor on %s delivered nothing for %.0f s: restarting it", monitor.source, stale_s)
    return "started" if monitor.start(source) else "unavailable"


# --- the motif playbin: conversation.sound's second speaker stream ------------------------


MOTIF_LOCK = threading.Lock()


class MotifActivity:
    """Which motif playbins the fork's watcher holds open, and when the last one closed.

    The phantom-speech detector reads it: a motif is sound the app made without a push.
    Only motifs started through watch_motif are known; the SDK's play_move sounds are not.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Start with no motif open."""
        self._clock = clock
        self._lock = threading.Lock()
        self._open: set[int] = set()
        self._closed_at: float | None = None

    def opened(self, playbin: Any) -> None:
        """Note a motif's playbin as playing."""
        with self._lock:
            self._open.add(id(playbin))

    def closed(self, playbin: Any) -> None:
        """Note a motif's playbin as ended or torn down; closing one twice is harmless."""
        with self._lock:
            self._open.discard(id(playbin))
            self._closed_at = self._clock()

    def state(self) -> tuple[bool, float | None]:
        """Return (a motif is open, when the last one closed)."""
        with self._lock:
            return bool(self._open), self._closed_at


MOTIFS = MotifActivity()


def play_motif(media: Any, audio: Any, path: str) -> Any:
    """Start a sound with the SDK's play_sound under MOTIF_LOCK; return its playbin, to watch."""
    with MOTIF_LOCK:
        media.play_sound(path)
        return getattr(audio, "_playbin", None)


def watch_motif(audio: Any, playbin: Any, path: str) -> threading.Thread:
    """Log a motif's playbin as a second stream, and tear it down when the sound ends.

    The SDK's play_sound builds a playbin with its own speaker sink and never tears it
    down: its stream stays open after the sound. This waits for its end (or error) on
    the playbin's bus, from its own thread, and sets it to NULL if it is still the
    current sound. A newer sound has replaced it already (the SDK set it to NULL).
    """
    from gi.repository import Gst

    logger.info("Motif stream opened: %s (a second speaker stream beside speech)", path)
    MOTIFS.opened(playbin)

    def wait() -> None:
        try:
            watch()
        finally:
            MOTIFS.closed(playbin)

    def watch() -> None:
        bus = playbin.get_bus()
        cause = None
        while cause is None:
            if getattr(audio, "_playbin", None) is not playbin:
                cause = "replaced by a newer sound or stopped"
                break
            msg = bus.timed_pop_filtered(200 * Gst.MSECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if msg is None:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, _debug = msg.parse_error()
                cause = f"error: {err.message}"
            else:
                cause = "the sound ended"
        if "replaced" in cause:
            logger.info("Motif stream closed (%s): %s", cause, path)
            return
        with MOTIF_LOCK:  # a new sound cannot take _playbin between the check and the clear
            current = getattr(audio, "_playbin", None)
            if current is not None and current is playbin:
                audio._playbin = None
                current.set_state(Gst.State.NULL)
        logger.info("Motif stream closed (%s, torn down): %s", cause, path)

    thread = threading.Thread(target=wait, name="motif-watch", daemon=True)
    thread.start()
    return thread


def stop_motif(audio: Any) -> bool:
    """conversation.sound's stop: the motif's stream alone, never speech."""
    from gi.repository import Gst

    with MOTIF_LOCK:
        playbin = getattr(audio, "_playbin", None)
        if playbin is None:
            return False
        audio._playbin = None
        playbin.set_state(Gst.State.NULL)
    MOTIFS.closed(playbin)
    logger.info("Motif stream closed (stop requested, torn down)")
    return True


# --- the phantom-speech detector -----------------------------------------------------------
#
# 2026-10-26, 05:22: the speaker replayed speech from a conversation at 20:03 the night
# before. The app logged no push and the head did not move; the wobbler is fed from a tee
# inside the app's speech pipeline, so the sound most likely never passed through it.
# This detector logs any such episode: the speaker's monitor loud for PHANTOM_LOUD_S while
# the app pushed nothing (and its queued audio has played out) for PHANTOM_QUIET_S and no
# motif played. One WARNING per episode, at most one every PHANTOM_REPORT_EVERY_S, names
# every sink input (pw-dump, or pactl), whether the app's speech sink carried sound in
# the episode (a pad probe on it: RMS of each buffer), and whether the head wobbler fired.
# Nothing is kept but RMS numbers and times. Nothing here runs on the event loop or blocks
# a streaming thread: the monitor and the pad probe only record, under a short lock; the
# listing and the report run in a worker thread.

PHANTOM_LOUD_S = 0.5  # monitor speech, within one candidate, that opens an episode
PHANTOM_QUIET_S = 3.0  # the app pushed nothing, its queued audio and any motif ended, this long ago
PHANTOM_END_S = 2.0  # a quiet monitor for this long closes an episode
PHANTOM_REPORT_EVERY_S = 60.0  # at most one WARNING in this long
SINK_INPUTS_TIMEOUT_S = 5.0
SinkInputs = tuple[list[dict[str, Any]], str]  # (the sink inputs, where they were read from)


@dataclass
class PhantomEpisode:
    """One stretch of speaker sound the app did not push. Times are monotonic seconds."""

    start: float
    last_loud: float
    loud_s: float
    last_push_age_s: float | None
    report: bool
    end: float | None = None
    end_cause: str = ""
    inputs: SinkInputs | None = None


class PhantomEvent(NamedTuple):
    """An episode "started" or "ended"."""

    kind: str
    episode: PhantomEpisode


class PhantomDetector:
    """Find speaker sound the app did not push. Times are passed in; thread-safe.

    The monitor feeds it from GStreamer's thread, pushes and barge-ins from the event
    loop. Events are queued for take(), so nothing is done on the caller's thread.
    """

    def __init__(self, motif: Callable[[], tuple[bool, float | None]] | None = None) -> None:
        """``motif`` returns (a motif is playing, when the last one closed); MOTIFS by default."""
        self._motif = motif or MOTIFS.state
        self._lock = threading.Lock()
        self._last_push: float | None = None
        self._play_end: float | None = None  # when the audio pushed so far has played out
        # (start, loud seconds, last loud) of sound that may open an episode
        self._candidate: tuple[float, float, float] | None = None
        self._episode: PhantomEpisode | None = None
        self._last_report: float | None = None
        self._events: list[PhantomEvent] = []

    def pushed(self, now: float, seconds: float) -> None:
        """Note audio pushed at ``now``: it plays after what is queued; an open episode ends."""
        with self._lock:
            self._last_push = now
            self._play_end = max(self._play_end or now, now) + seconds
            self._candidate = None
            if self._episode is not None:
                self._end(now, "the app pushed audio")

    def flushed(self, now: float) -> None:
        """Note a barge-in: the queued audio was dropped."""
        with self._lock:
            if self._play_end is not None:
                self._play_end = min(self._play_end, now)

    def _app_quiet(self, now: float) -> bool:
        """Whether no sound of the app's own played for PHANTOM_QUIET_S. Holds the lock."""
        for t in (self._last_push, self._play_end):
            if t is not None and now - t < PHANTOM_QUIET_S:
                return False
        playing, closed_at = self._motif()
        if playing:
            return False
        return closed_at is None or now - closed_at >= PHANTOM_QUIET_S

    def _end(self, end: float, cause: str) -> None:
        """Close the open episode. Holds the lock."""
        episode = self._episode
        if episode is None:
            return
        episode.end = end
        episode.end_cause = cause
        self._episode = None
        self._events.append(PhantomEvent("ended", episode))

    def monitored(self, now: float, seconds: float, speech_s: float) -> None:
        """Note a buffer from the speaker's monitor, arriving at ``now``."""
        with self._lock:
            loud = speech_s > 0.0
            episode = self._episode
            if episode is not None:
                if loud:
                    episode.loud_s += speech_s
                    episode.last_loud = now
                elif now - episode.last_loud >= PHANTOM_END_S:
                    self._end(episode.last_loud, "the speaker went quiet")
                return
            if not loud:
                return
            if not self._app_quiet(now):
                self._candidate = None
                return
            candidate = self._candidate
            if candidate is None or now - candidate[2] > PHANTOM_END_S:
                candidate = (now - seconds, 0.0, now)
            start, loud_s, _ = candidate
            loud_s += speech_s
            self._candidate = (start, loud_s, now)
            if loud_s < PHANTOM_LOUD_S - 1e-9:
                return
            self._candidate = None
            report = self._last_report is None or now - self._last_report >= PHANTOM_REPORT_EVERY_S
            if report:
                self._last_report = now
            age = None if self._last_push is None else start - self._last_push
            self._episode = PhantomEpisode(start, now, loud_s, age, report)
            self._events.append(PhantomEvent("started", self._episode))

    def take(self) -> list[PhantomEvent]:
        """Return the events since the last call, oldest first."""
        with self._lock:
            events, self._events = self._events, []
        return events


# The sink inputs: every playback stream the sound server has, from pw-dump (or pactl).


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sink_inputs_from_pw_dump(objects: Sequence[Any], own_pid: int) -> list[dict[str, Any]]:
    """Return the playback streams in a pw-dump listing, the app's own marked. Pure."""
    found = []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "PipeWire:Interface:Node":
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        if props.get("media.class") != "Stream/Output/Audio":
            continue
        formats = (info.get("params") or {}).get("Format") or []
        rate = _int(formats[0].get("rate")) if formats and isinstance(formats[0], dict) else None
        if rate is None:
            rate = _int(str(props.get("node.rate", "")).rpartition("/")[2])
        pid = _int(props.get("application.process.id"))
        found.append(
            {
                "node": obj.get("id"),
                "serial": _int(props.get("object.serial")),
                "rate": rate,
                "pid": pid,
                "ours": pid is not None and pid == own_pid,
                "binary": props.get("application.process.binary") or props.get("application.name"),
                "media": props.get("media.name"),
                "maxlength": _int(props.get("pulse.attr.maxlength")),
                "state": info.get("state"),
                "corked": props.get("pulse.corked") in (True, "true"),
            }
        )
    return found


def sink_inputs_from_pactl(entries: Sequence[Any], own_pid: int) -> list[dict[str, Any]]:
    """Return the sink inputs of ``pactl -f json list sink-inputs``, the app's own marked. Pure."""
    found = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        props = entry.get("properties") or {}
        rate = re.search(r"(\d+)\s*Hz", str(entry.get("sample_specification", "")))
        pid = _int(props.get("application.process.id"))
        found.append(
            {
                "node": entry.get("index"),
                "serial": _int(props.get("object.serial")),
                "rate": int(rate.group(1)) if rate else None,
                "pid": pid,
                "ours": pid is not None and pid == own_pid,
                "binary": props.get("application.process.binary") or props.get("application.name"),
                "media": props.get("media.name"),
                "maxlength": None,
                "state": None,
                "corked": bool(entry.get("corked")),
            }
        )
    return found


def list_sink_inputs(own_pid: int, timeout_s: float = SINK_INPUTS_TIMEOUT_S) -> SinkInputs:
    """List the sink inputs with pw-dump, else pactl; returns (inputs, source). Never raises.

    Runs subprocesses: call it from a worker thread, never the event loop.
    """
    tries: list[tuple[list[str], Callable[[Any, int], list[dict[str, Any]]]]] = [
        (["pw-dump"], sink_inputs_from_pw_dump),
        (["pactl", "-f", "json", "list", "sink-inputs"], sink_inputs_from_pactl),
    ]
    errors = []
    for cmd, parse in tries:
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if done.returncode != 0:
                errors.append(f"{cmd[0]} exited {done.returncode}")
                continue
            return parse(json.loads(done.stdout), own_pid), cmd[0]
        except Exception as exc:
            errors.append(f"{cmd[0]}: {type(exc).__name__}: {exc}")
    return [], "unavailable (" + "; ".join(errors) + ")"


def describe_sink_inputs(inputs: Sequence[dict[str, Any]]) -> str:
    """One line for the sink inputs, the app's own marked. Pure."""
    if not inputs:
        return "none"
    parts = []
    for i in inputs:
        part = f"node {i['node']} serial {i['serial']} {i['rate']} Hz pid {i['pid']}"
        if i["ours"]:
            part += " (this app)"
        part += f" {i['binary']} '{i['media']}'"
        if i["maxlength"] is not None:
            part += f" maxlength {i['maxlength']}"
        if i["state"]:
            part += f" {i['state']}"
        if i["corked"]:
            part += " corked"
        parts.append(part)
    return "; ".join(parts)


# The app's speech sink: a pad probe records the RMS of every buffer that reaches it.


def buffer_rms(data: bytes, fmt: str) -> float | None:
    """Return the RMS of raw F32LE or S16LE samples, or None for another format or none. Pure."""
    import numpy as np

    if fmt == "F32LE":
        samples = np.frombuffer(data, dtype="<f4", count=len(data) // 4)
    elif fmt == "S16LE":
        ints = np.frombuffer(data, dtype="<i2", count=len(data) // 2)
        samples = ints.astype(np.float32) / 32768.0
    else:
        return None
    if samples.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class PadSummary(NamedTuple):
    """What reached a pad in a window: buffers, those above LOUD_RMS, the peak RMS, the format."""

    buffers: int
    loud: int
    max_rms: float | None
    fmt: str | None


class PadLevel:
    """A pad probe on the speaker sink's input, keeping (time, RMS, format) of recent buffers."""

    def __init__(self, keep: int = 12000) -> None:
        """Keep the last ``keep`` buffers: a minute or more at pulsesink's buffer sizes."""
        self._lock = threading.Lock()
        self._buffers: deque[tuple[float, float | None, str]] = deque(maxlen=keep)
        self._sink: Any = None
        self._pad: Any = None
        self._probe: Any = None

    @property
    def attached(self) -> bool:
        """Whether a sink is being measured."""
        return self._sink is not None

    def record(self, now: float, rms: float | None, fmt: str) -> None:
        """Note one buffer."""
        with self._lock:
            self._buffers.append((now, rms, fmt))

    def summary(self, start: float, end: float) -> PadSummary:
        """Summarise the buffers that arrived from ``start`` to ``end``."""
        with self._lock:
            window = [b for b in self._buffers if start <= b[0] <= end]
        levels = [b[1] for b in window if b[1] is not None]
        return PadSummary(
            len(window),
            sum(1 for r in levels if r > LOUD_RMS),
            max(levels) if levels else None,
            window[-1][2] if window else None,
        )

    def attach(self, sink: Any) -> bool:
        """Measure ``sink``'s input pad; a different sink moves the probe. Returns success."""
        from gi.repository import Gst

        if sink is self._sink:
            return True
        self.detach()
        pad = sink.get_static_pad("sink") if sink is not None else None
        if pad is None:
            return False
        self._probe = pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer)
        self._sink, self._pad = sink, pad
        return True

    def detach(self) -> None:
        """Remove the probe, if any."""
        if self._pad is not None and self._probe is not None:
            try:
                self._pad.remove_probe(self._probe)
            except Exception:
                logger.debug("Could not remove the speech sink's pad probe", exc_info=True)
        self._sink = self._pad = self._probe = None

    def _on_buffer(self, pad: Any, info: Any) -> Any:
        """Record one buffer's RMS on the streaming thread: never raises, blocks or logs."""
        from gi.repository import Gst

        try:
            buf = info.get_buffer()
            caps = pad.get_current_caps()
            fmt = str(caps.get_structure(0).get_value("format")) if caps is not None else "unknown"
            self.record(time.monotonic(), buffer_rms(buf.extract_dup(0, buf.get_size()), fmt), fmt)
        except Exception:
            pass
        return Gst.PadProbeReturn.OK


class WobblerWatch:
    """Count the head wobbler's moves: each nonzero offset it applies, with its time."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, keep: int = 20000) -> None:
        """Start with no wobbler watched."""
        self._clock = clock
        self._lock = threading.Lock()
        self._fires: deque[float] = deque(maxlen=keep)

    def keep(self, audio: Any) -> bool:
        """Watch the SDK's current wobbler, once; a new one is watched too. Returns success."""
        wobbler = getattr(audio, "_head_wobbler", None)
        apply = getattr(wobbler, "_apply_offsets", None)
        if apply is None:
            return False
        if getattr(apply, "_phantom_watch", None) is self:
            return True
        watch = self

        def counted(offsets: Any) -> Any:
            try:
                if any(abs(float(v)) > 0.0 for v in offsets):
                    with watch._lock:
                        watch._fires.append(watch._clock())
            except Exception:
                pass
            return apply(offsets)

        counted._phantom_watch = self  # type: ignore[attr-defined]
        wobbler._apply_offsets = counted
        return True

    def fired(self, start: float, end: float) -> int:
        """Return how many nonzero offsets the wobbler applied from ``start`` to ``end``."""
        with self._lock:
            return sum(1 for t in self._fires if start <= t <= end)


def _wall(t: float) -> str:
    """Return a monotonic time as the wall-clock time it was."""
    when = datetime.datetime.now() - datetime.timedelta(seconds=time.monotonic() - t)
    return when.strftime("%H:%M:%S")


def _describe_playbin(audio: Any) -> str:
    """Describe the SDK's motif playbin: none, or its state and position. Never raises."""
    playbin = getattr(audio, "_playbin", None)
    if playbin is None:
        return "no motif playbin"
    try:
        from gi.repository import Gst

        state = playbin.get_state(0)[1].value_nick
        ok_pos, pos = playbin.query_position(Gst.Format.TIME)
        ok_dur, dur = playbin.query_duration(Gst.Format.TIME)
        at = f", at {pos / Gst.SECOND:.1f} of {dur / Gst.SECOND:.1f} s" if ok_pos and ok_dur else ""
        return f"motif playbin {playbin.get_name()} {state}{at}"
    except Exception as exc:
        return f"motif playbin present (unreadable: {exc})"


class PhantomWatch:
    """The detector, the speech-sink probe and the wobbler count, and the report, for the console.

    monitored() runs on the speaker monitor's streaming thread; pushed(), flushed() and
    take() on the event loop; keep() and act() in a worker thread.
    """

    def __init__(
        self,
        motif: Callable[[], tuple[bool, float | None]] | None = None,
        list_inputs: Callable[[], SinkInputs] | None = None,
        pad: PadLevel | None = None,
        wobbler: WobblerWatch | None = None,
    ) -> None:
        """Build with the module's motif state and pw-dump by default."""
        self.detector = PhantomDetector(motif=motif)
        self.pad = pad or PadLevel()
        self.wobbler = wobbler or WobblerWatch()
        self._list_inputs = list_inputs or (lambda: list_sink_inputs(os.getpid()))

    def monitored(self, now: float, seconds: float, speech_s: float) -> None:
        """Note a speaker-monitor buffer, from GStreamer's thread. Never raises."""
        try:
            self.detector.monitored(now, seconds, speech_s)
        except Exception:
            pass

    def pushed(self, now: float, seconds: float) -> None:
        """Note audio the app pushed."""
        self.detector.pushed(now, seconds)

    def flushed(self, now: float) -> None:
        """Note a barge-in: the queued audio was dropped."""
        self.detector.flushed(now)

    def take(self) -> list[PhantomEvent]:
        """Return the events to act on, off the event loop."""
        return self.detector.take()

    def keep(self, audio: Any) -> None:
        """Keep the probe on the current speaker sink and the count on the current wobbler.

        Runs in a worker thread.
        """
        with MEDIA_LOCK:
            pipeline = getattr(audio, "_pipeline", None)
            sink = speaker_sink(pipeline) if _playing(pipeline) else None
        if sink is not None:
            self.pad.attach(sink)
        self.wobbler.keep(audio)

    def act(self, event: PhantomEvent, audio: Any) -> None:
        """List the sink inputs when a reported episode starts; log it at its end. Never raises."""
        episode = event.episode
        try:
            if event.kind == "started" and episode.report:
                try:
                    episode.inputs = self._list_inputs()
                except Exception as exc:
                    episode.inputs = ([], f"unavailable ({type(exc).__name__}: {exc})")
            elif event.kind == "ended" and episode.report:
                logger.warning("%s", self.report(episode, audio))
            elif event.kind == "ended":
                logger.info(
                    "Phantom speech again within %.0f s of the last report: %.1f s loud, %s to %s",
                    PHANTOM_REPORT_EVERY_S,
                    episode.loud_s,
                    _wall(episode.start),
                    _wall(episode.end if episode.end is not None else episode.last_loud),
                )
        except Exception:
            logger.warning("Phantom speech report failed", exc_info=True)

    def report(self, episode: PhantomEpisode, audio: Any) -> str:
        """Return the episode's WARNING line."""
        end = episode.end if episode.end is not None else episode.last_loud
        if episode.last_push_age_s is None:
            push = "nothing pushed since the app started"
        else:
            push = f"last push {episode.last_push_age_s:.0f} s before"
        pad = self.pad.summary(episode.start - 1.0, end)
        if pad.buffers:
            peak = f"{pad.max_rms:.4f}" if pad.max_rms is not None else "n/a"
            counts = f"{pad.buffers} buffers, {pad.loud} loud"
            pad_line = f"speech sink pad: {counts}, max RMS {peak} ({pad.fmt})"
        elif self.pad.attached:
            pad_line = "speech sink pad: no buffers"
        else:
            pad_line = "speech sink pad: not probed"
        fired = self.wobbler.fired(episode.start - 1.0, end)
        inputs, source = episode.inputs if episode.inputs is not None else ([], "not listed")
        if source in ("pw-dump", "pactl"):
            listed = f"sink inputs ({source}): {describe_sink_inputs(inputs)}"
        else:
            listed = f"sink inputs {source}"
        cause = episode.end_cause or "still open"
        return (
            f"Phantom speech: the speaker played {episode.loud_s:.1f} s loud "
            f"from {_wall(episode.start)} to {_wall(end)} ({cause}) with no push from the app "
            f"({push}) and no motif; {pad_line}; head wobbler fired {fired} times; "
            f"{_describe_playbin(audio)}; {listed}"
        )
