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

So the bound comes from the clock. Clocked from the speaker, pulsesink is not slaved
and never re-maps; capture follows the speaker instead. Measured in the container with
the SDK's own elements, a pipe-fed microphone starved for 75 s while PipeWire and the
speaker ran: with the microphone's clock 0 of 32 replies were heard in the next 100 s;
with the speaker's, replies were heard in 0.14-0.17 s from the first second (two runs
each, and the same as a pytest gate). A stop of the PipeWire daemon itself desyncs
either clock and recovers in 22-35 s about four runs in five.

The watchdog is the net under that: once a minute it logs the stream's latency, and
over LATENCY_LIMIT_S it recreates the speaker sink alone (the record branch is never
paused or relinked); measured, that cleared a desync at once, replies heard in 0.16 s.
pa_stream_get_latency reports a queue ahead of the read pointer, the robot's case; a
write pointer behind it reads as 0, so the watchdog cannot see that case.
"""

from __future__ import annotations
import math
import logging
import threading
from typing import Any, NamedTuple


logger = logging.getLogger(__name__)

LATENCY_LIMIT_S = 5.0
CHECK_INTERVAL_S = 60.0


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


def clock_playback_from_speaker(pipeline: Any, sink: Any) -> bool:
    """Clock the shared pipeline from the speaker, never the microphone. Call before PLAYING.

    Returns whether a clock was set; if so, prime_speaker must follow the start. A pipeline already PLAYING takes it at its next
    start; it is never paused for it, since pausing corks the microphone.
    """
    if sink is None:
        return False
    # Before PLAYING provide_clock() is None (the ring buffer is not acquired yet), and
    # use_clock(None) would run the pipeline with no clock at all. The clock object
    # exists from the sink's creation: GstAudioBaseSink's provided_clock field.
    clock = getattr(sink, "provided_clock", None) or sink.provide_clock()
    if clock is None:
        return False
    pipeline.use_clock(clock)
    return True


def prime_speaker(audio: Any) -> None:
    """Push 20 ms of silence, so the speaker's stream starts and its clock with it. Call after starting.

    Clocked from the speaker, the pipeline cannot reach PLAYING on its own: pulsesink's
    clock runs only once its stream has audio, and the live silence source and the mixer
    wait on that clock. Measured in the trixie container: 0 microphone buffers until the
    first reply; with this push, 100 buffers/s from the first second.
    """
    import numpy as np

    channels = getattr(audio, "CHANNELS", 2)
    rate = getattr(audio, "SAMPLE_RATE", 16000)
    audio.push_audio_sample(np.zeros((rate // 50, channels), dtype=np.float32))


def measure_latency_s(sink: Any) -> float | None:
    """Read the speaker stream's own latency (pa_stream_get_latency, via the ring buffer), in seconds."""
    ringbuffer = getattr(sink, "ringbuffer", None)
    try:
        if ringbuffer is None or not ringbuffer.is_acquired():
            return None
        rate = ringbuffer.spec.info.rate
        if not rate:
            return None
        return float(ringbuffer.delay()) / float(rate)
    except Exception:
        logger.debug("could not read the speaker stream's latency", exc_info=True)
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


def check_playback(pipeline: Any, limit_s: float = LATENCY_LIMIT_S) -> Verdict:
    """One watchdog check: log the latency, rebuild the speaker stream if it ran away, check the invariant."""
    sink = speaker_sink(pipeline)
    latency = measure_latency_s(sink) if sink is not None else None
    name = sink.get_name() if sink is not None else "none"
    shown = "unmeasured" if latency is None else f"{latency:.3f} s"
    logger.info("Playback latency: %s (speaker %s, bound %.1f s)", shown, name, limit_s)
    verdict = decide(latency, limit_s)
    if verdict.action != "rebuild":
        return verdict
    logger.warning("Playback latency %.1f s: rebuilding the speaker stream (cause: %s)", latency, verdict.cause)
    rebuilt = rebuild_sink(sink)
    after = measure_latency_s(sink)
    shown_after = "unmeasured" if after is None else f"{after:.3f} s"
    if rebuilt and latency_bounded(after, limit_s):
        logger.info("Speaker stream rebuilt; playback latency now %s, bounded (invariant held)", shown_after)
    else:
        logger.error(
            "Invariant violated: playback latency stays bounded (rebuilt=%s, latency now %s)", rebuilt, shown_after
        )
    return verdict


# --- the motif playbin: conversation.sound's second speaker stream ------------------------


def watch_motif(audio: Any, playbin: Any, path: str) -> threading.Thread:
    """Log a motif's playbin as a second stream, and tear it down when the sound ends.

    The SDK's play_sound builds a playbin with its own speaker sink and never tears it
    down: its stream stays open after the sound. This waits for its end (or error) on
    the playbin's bus, from its own thread, and sets it to NULL if it is still the
    current sound. A newer sound has replaced it already (the SDK set it to NULL).
    """
    from gi.repository import Gst

    logger.info("Motif stream opened: %s (a second speaker stream beside speech)", path)

    def wait() -> None:
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

    playbin = getattr(audio, "_playbin", None)
    if playbin is None:
        return False
    audio._playbin = None
    playbin.set_state(Gst.State.NULL)
    logger.info("Motif stream closed (stop requested, torn down)")
    return True
