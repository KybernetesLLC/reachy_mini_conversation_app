"""Playback health: the 65 s speech lag (2026-10-15) and what keeps it bounded.

The robot's speech played 64-65 s late, steadily: 4 MiB of pipewire-pulse's default
maxlength at 64,000 B/s. pulsesink writes at absolute stream offsets taken from its
ring buffer position; slaved to the microphone's clock (the SDK's one pipeline picks
pulsesrc's clock), a capture stall makes it re-map those offsets, and the write
pointer lands far from the server's read pointer: ahead, and the gap is queued
silence pinned at maxlength (the robot); behind, and every reply is written into the
past (the trixie container). Clocking the pipeline from the speaker removes the
re-mapping; the watchdog is the net under it.

Runs silent and deterministic by default (a live test source for the microphone,
fakesink sync=true for the speaker). With REACHY_PLAYBACK_TEST_SINK=sdk the SDK
builds both itself (pulsesrc/pulsesink on the robot and in the trixie container),
and "heard" is measured on the speaker's monitor source, past the server's queue.
"""

from __future__ import annotations
import os
import re
import math
import time
import logging
import threading
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reachy_mini_conversation_app import playback_health as ph


USE_SDK_DEVICES = os.environ.get("REACHY_PLAYBACK_TEST_SINK") == "sdk"
# A stall of the PipeWire daemon itself: container only, never on the robot.
STALL_KIND = os.environ.get("REACHY_PLAYBACK_TEST_STALL", "capture")
STALL_S = float(os.environ.get("REACHY_PLAYBACK_TEST_STALL_S", "3"))
RECOVERY_S = float(os.environ.get("REACHY_PLAYBACK_TEST_RECOVERY_S", "10"))
# The trixie container's pipe-fed microphone pauses while this file exists.
MIC_FEED_PAUSE = "/tmp/reachy_mic_feed_paused"


# --- the watchdog's decision: pure, table-tested --------------------------------------


@pytest.mark.parametrize(
    ("latency_s", "action"),
    [
        (None, "unmeasured"),
        (math.nan, "unmeasured"),
        (-0.1, "unmeasured"),
        (0.0, "ok"),
        (0.05, "ok"),
        (4.99, "ok"),
        (5.0, "ok"),
        (5.01, "rebuild"),
        (65.5, "rebuild"),
    ],
)
def test_the_watchdog_decides_from_the_measured_latency(latency_s: float | None, action: str) -> None:
    """Over the bound rebuilds; at or under it is fine; no number is no action."""
    verdict = ph.decide(latency_s, limit_s=5.0)
    assert verdict.action == action
    assert verdict.cause  # every action carries its cause, for the log line


def test_the_rebuild_cause_names_the_value_and_the_bound() -> None:
    """The WARNING line says what was measured against what."""
    cause = ph.decide(65.5, limit_s=5.0).cause
    assert "65.5" in cause and "5.0" in cause


def test_the_invariant_is_the_bound() -> None:
    """Playback latency stays bounded: unmeasured is not a violation, over the bound is."""
    assert ph.latency_bounded(None)
    assert ph.latency_bounded(0.2)
    assert ph.latency_bounded(ph.LATENCY_LIMIT_S)
    assert not ph.latency_bounded(ph.LATENCY_LIMIT_S + 0.01)


def test_the_pipeline_is_clocked_from_the_speakers_own_clock_before_it_starts() -> None:
    """The fix for the lag: the speaker's clock, never the microphone's.

    Before PLAYING, an audio sink's provide_clock() is None (its ring buffer is not
    acquired yet), and use_clock(None) runs the pipeline with no clock at all (the
    trixie container, 2026-10-15). The clock object itself exists from the sink's
    creation, as the provided_clock field.
    """
    clock = object()

    class Sink:
        provided_clock = clock

        def provide_clock(self) -> None:
            return None  # not acquired yet

    used: list[object] = []
    pipeline = type("P", (), {"use_clock": lambda self, c: used.append(c)})()
    assert ph.clock_playback_from_speaker(pipeline, Sink()) is True
    assert used == [clock]


def test_a_speaker_without_a_clock_leaves_the_pipeline_clock_alone() -> None:
    """Fakesink and friends provide none: nothing is changed, never use_clock(None)."""
    used: list[object] = []
    pipeline = type("P", (), {"use_clock": lambda self, c: used.append(c)})()
    sink = type("S", (), {"provide_clock": lambda self: None})()
    assert ph.clock_playback_from_speaker(pipeline, sink) is False
    assert ph.clock_playback_from_speaker(pipeline, None) is False
    assert used == []


# --- the SDK's real playback layout -----------------------------------------------------


def _gst() -> Any:
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


class SpeakerClock:
    """A speaker's clock for the local rig, run as pulsesink's runs: from its first audio, while alive.

    fakesink provides no clock, so without this the speaker's clock was never exercised
    outside the container. Like pulsesink's, it starts only once the sink renders audio
    (hence the prime), and it stops while the speaker stream is dead (kill/revive).
    """

    def __init__(self) -> None:
        """Make the clock; it reads 0 until the sink renders."""
        gi = pytest.importorskip("gi")
        gi.require_version("GstAudio", "1.0")
        from gi.repository import GstAudio

        self._lock = threading.Lock()
        self._now = 0
        self._last: int | None = None
        self.started = False
        self.alive = True
        self.clock = GstAudio.AudioClock.new("speakerclock", self._time, None)

    def _time(self, _clock: Any, *_args: Any) -> int:
        with self._lock:
            mono = time.monotonic_ns()
            if self._last is not None and self.started and self.alive:
                self._now += mono - self._last
            self._last = mono
            return self._now

    def kill(self) -> None:
        """Stop the clock, as a dead speaker stream stops it."""
        self._time(None)
        self.alive = False

    def revive(self) -> None:
        """Run the clock again: the speaker's stream is back."""
        self._time(None)
        self.alive = True


class Rig:
    """The SDK's GStreamerAudio, built as the robot builds it, plus a way to hear it.

    With ``clocked`` (local only), the speaker provides a SpeakerClock and the microphone
    is device-driven: an appsrc fed in real time from its own thread, stamped by sample
    count, as pulsesrc is fed by the server whatever the pipeline clock does.
    """

    def __init__(self, clocked: bool = False) -> None:
        """Build the SDK's layout; nothing runs yet."""
        Gst = _gst()
        self.Gst = Gst
        audio_gstreamer = pytest.importorskip("reachy_mini.media.audio_gstreamer")
        self.rendered: list[tuple[float, float]] = []  # (monotonic, peak) heard
        self.mic_buffers: list[float] = []
        self.clocked = clocked and not USE_SDK_DEVICES
        self.speaker_clock = SpeakerClock() if self.clocked else None
        rig = self

        class Playback(audio_gstreamer.GStreamerAudio):  # type: ignore[misc]
            def __init__(self) -> None:
                if USE_SDK_DEVICES:
                    super().__init__()
                    return
                self.logger = logging.getLogger("playback-test")
                self._appsrc_pts = -1
                self._head_wobbler = None
                self._webrtcechoprobe = None
                self._playbin = None
                self._pipeline = Gst.Pipeline.new("reachymini_audio")
                source = (
                    "appsrc name=mic is-live=true format=time do-timestamp=false "
                    "caps=audio/x-raw,rate=16000,channels=2,format=F32LE,layout=interleaved"
                    if rig.clocked
                    else "audiotestsrc is-live=true wave=silence name=mic"
                )
                mic = Gst.parse_bin_from_description(
                    f"{source} ! queue ! audioconvert ! audioresample "
                    "! appsink name=micsink caps=audio/x-raw,rate=16000,channels=2,format=F32LE,layout=interleaved "
                    "drop=true max-buffers=200",
                    False,
                )
                self._pipeline.add(mic)
                self._appsink_audio = mic.get_by_name("micsink")
                self._init_pipeline_playback(self._pipeline)

            def _build_audiosink_element(self) -> Any:
                if USE_SDK_DEVICES:
                    sink = super()._build_audiosink_element()
                else:
                    sink = Gst.ElementFactory.make("fakesink")
                    sink.set_property("sync", True)
                    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, rig._on_rendered)
                return sink

        self.audio = Playback()
        if not hasattr(self.audio._appsrc, "push_buffer"):
            # An appsrc made before GstApp was imported leaves PyGObject's wrapper
            # without GstApp's methods; the signal is the same call.
            appsrc = self.audio._appsrc
            appsrc.push_buffer = lambda buf: appsrc.emit("push-buffer", buf)
        self.pipeline = self.audio._pipeline
        self.speaker = ph.speaker_sink(self.pipeline)
        assert self.speaker is not None
        if self.speaker_clock is not None:
            self.speaker.provided_clock = self.speaker_clock.clock  # read by clock_playback_from_speaker
            clock = self.speaker_clock
            self.speaker.get_static_pad("sink").add_probe(
                Gst.PadProbeType.BUFFER, lambda _p, _i: setattr(clock, "started", True) or Gst.PadProbeReturn.OK
            )  # the stream has audio: its clock runs
        self.mic = (
            self._first("pulsesrc") or self._first("audiotestsrc", name="mic") or self._first("appsrc", name="mic")
        )
        self.monitor: Any = None
        self._stop = threading.Event()

    def _first(self, factory: str, name: str | None = None) -> Any:
        it = self.pipeline.iterate_recurse()
        while True:
            result, element = it.next()
            if result != self.Gst.IteratorResult.OK:
                return None
            f = element.get_factory()
            if f is not None and f.get_name() == factory and (name is None or element.get_name() == name):
                return element

    def _on_rendered(self, pad: Any, info: Any) -> Any:
        Gst = self.Gst
        buf = info.get_buffer()
        ok, mapped = buf.map(Gst.MapFlags.READ)
        if ok:
            fmt = pad.get_current_caps().get_structure(0).get_string("format")
            kind, scale = {"S16LE": (np.int16, 32768.0), "S32LE": (np.int32, 2.0**31)}.get(fmt, (np.float32, 1.0))
            samples = np.frombuffer(mapped.data, dtype=kind)
            self.rendered.append((time.monotonic(), float(np.abs(samples).max()) / scale if samples.size else 0.0))
            buf.unmap(mapped)
        return Gst.PadProbeReturn.OK

    def start(self, prime: bool = True) -> None:
        """Start as the app does: capture off the clock, the speaker's clock, PLAYING, then the prime."""
        Gst = self.Gst
        ph.free_capture_from_clock(self.audio)  # as the app does, before PLAYING
        clocked = ph.clock_playback_from_speaker(self.pipeline, self.speaker)
        if (USE_SDK_DEVICES and self.speaker.get_factory().get_name() == "pulsesink") or self.clocked:
            assert clocked, "the speaker's clock was not applied"  # a silent no-op passed once
        self.pipeline.set_state(Gst.State.PLAYING)
        if clocked and prime:
            ph.prime_speaker(self.audio)
        self.pipeline.get_state(5 * Gst.SECOND)
        if self.clocked:
            threading.Thread(target=self._feed_mic, daemon=True).start()
        threading.Thread(target=self._pull_mic, daemon=True).start()
        if USE_SDK_DEVICES:
            self._start_monitor()
        time.sleep(1.0)

    def _start_monitor(self) -> None:
        """Hear the speaker past the server's queue: its monitor source, in a separate pipeline."""
        Gst = self.Gst
        if self.monitor is not None:
            self.monitor.set_state(Gst.State.NULL)
        device = self.speaker.get_property("device")
        self.monitor = Gst.parse_launch(
            f"pulsesrc device={device}.monitor ! audioconvert ! audio/x-raw,format=F32LE,channels=1 "
            "! appsink name=a drop=true max-buffers=2000 sync=false"
        )
        appsink = self.monitor.get_by_name("a")
        self.monitor.set_state(Gst.State.PLAYING)

        def pull() -> None:
            me = self.monitor
            while not self._stop.is_set() and self.monitor is me:
                sample = appsink.emit("try-pull-sample", 20 * Gst.MSECOND)
                if sample is None:
                    continue
                b = sample.get_buffer()
                a = np.frombuffer(b.extract_dup(0, b.get_size()), dtype=np.float32)
                self.rendered.append((time.monotonic(), float(np.abs(a).max()) if a.size else 0.0))

        threading.Thread(target=pull, daemon=True).start()

    def _feed_mic(self) -> None:
        """Feed the clocked rig's microphone: 10 ms of silence every 10 ms, stamped by sample count.

        As pulsesrc stamps: the pipeline's running time when capture starts, then sample count.
        """
        Gst = self.Gst
        silence = np.zeros((160, 2), dtype=np.float32).tobytes()
        clock = self.pipeline.get_clock()
        first = max(0, clock.get_time() - self.pipeline.get_base_time()) if clock is not None else 0
        n = 0
        next_at = time.monotonic()
        while not self._stop.is_set():
            next_at += 0.01
            buf = Gst.Buffer.new_wrapped(silence)
            buf.pts = first + n * 10 * Gst.MSECOND
            buf.duration = 10 * Gst.MSECOND
            self.mic.emit("push-buffer", buf)
            n += 1
            time.sleep(max(0.0, next_at - time.monotonic()))

    def mic_gaps(self, since: float) -> float:
        """Return the longest gap between microphone buffers since ``since``, up to now."""
        times = [since] + [t for t in list(self.mic_buffers) if t >= since] + [time.monotonic()]
        return max(b - a for a, b in zip(times, times[1:]))

    def kill_speaker(self) -> None:
        """Kill the speaker's stream: its clock stops (locally), or its sink goes to NULL (pulsesink)."""
        if self.speaker_clock is not None:
            self.speaker_clock.kill()
            return
        Gst = self.Gst
        sinkpad = self.speaker.get_static_pad("sink")
        self._upstream = sinkpad.get_peer()
        blocked = threading.Event()
        self._block = self._upstream.add_probe(
            Gst.PadProbeType.BLOCK_DOWNSTREAM, lambda _p, _i: blocked.set() or Gst.PadProbeReturn.OK
        )
        blocked.wait(2.0)
        self._upstream.unlink(sinkpad)
        self.speaker.set_state(Gst.State.NULL)

    def revive_speaker(self) -> None:
        """Bring the speaker's stream back after kill_speaker."""
        if self.speaker_clock is not None:
            self.speaker_clock.revive()
            return
        self._upstream.link(self.speaker.get_static_pad("sink"))
        self.speaker.set_property("async", False)
        self.speaker.sync_state_with_parent()
        self.speaker.set_property("async", True)
        self._upstream.remove_probe(self._block)
        ph.prime_speaker(self.audio)

    def _pull_mic(self) -> None:
        Gst = self.Gst
        while not self._stop.is_set():
            sample = self.audio._appsink_audio.emit("try-pull-sample", 20 * Gst.MSECOND)
            now = time.monotonic()
            if sample is not None:
                self.mic_buffers.append(now)
            on_poll = getattr(self, "on_mic_poll", None)  # (now, got_frame), as the app's record loop polls
            if on_poll is not None:
                on_poll(now, sample is not None)

    def push_reply(self, seconds: float, level: float = 0.3) -> float:
        """Push a reply of constant ``level`` and return when its push began."""
        pushed_at = time.monotonic()
        tone = np.full((int(seconds * 16000), 2), level, dtype=np.float32)
        for start in range(0, len(tone), 1600):  # 100 ms chunks, faster than real time, as the app pushes
            self.audio._push_appsrc_buffer(tone[start : start + 1600])
            time.sleep(0.025)
        return pushed_at

    def heard_after(self, after: float, level: float = 0.3, timeout: float = 3.0) -> float | None:
        """Seconds from ``after`` until the reply at ``level`` is first heard, or None."""
        deadline = after + timeout
        while time.monotonic() < deadline:
            for t, peak in list(self.rendered):
                if t >= after and abs(peak - level) < 0.01:
                    return t - after
            time.sleep(0.01)
        return None

    def reply_latency(self, level: float) -> float | None:
        """Push a 0.5 s reply and return its push-to-heard latency, or None."""
        pushed_at = self.push_reply(0.5, level)
        latency = self.heard_after(pushed_at, level)
        time.sleep(0.6)  # let it finish before the next one
        return latency

    def mic_rate(self, seconds: float = 2.0) -> float:
        """Return the microphone's buffers per second over ``seconds``."""
        before = len(self.mic_buffers)
        time.sleep(seconds)
        return (len(self.mic_buffers) - before) / seconds

    def stall(self, kind: str, seconds: float) -> None:
        """Stall for ``seconds`` while replies keep coming."""
        Gst = self.Gst
        until = time.monotonic() + seconds
        pid = None
        if kind == "pipewire":  # the daemon itself: capture and speaker together (container only)
            pid = int(subprocess.run(["pidof", "pipewire"], capture_output=True, text=True).stdout.split()[0])
            os.kill(pid, 19)
        elif kind == "micfeed":  # a pipe-fed microphone starved, PipeWire healthy (container only)
            open(MIC_FEED_PAUSE, "w").close()
        else:  # the capture thread, held in GStreamer: robot-safe

            def hold(_pad: Any, _info: Any) -> Any:
                while time.monotonic() < until:
                    time.sleep(0.01)
                return Gst.PadProbeReturn.REMOVE

            self.mic.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, hold)
        try:
            while time.monotonic() < until:
                self.push_reply(1.0, 0.05)
                time.sleep(min(4.0, max(0.0, until - time.monotonic())))
        finally:
            if pid is not None:
                os.kill(pid, 18)
            if kind == "micfeed":
                os.remove(MIC_FEED_PAUSE)
        time.sleep(1.0)
        if USE_SDK_DEVICES and kind == "pipewire":
            self._start_monitor()
            time.sleep(1.0)

    def close(self) -> None:
        """Stop the pipelines, off this thread."""
        self._stop.set()
        Gst = self.Gst
        if self.monitor is not None:
            self.monitor.set_state(Gst.State.NULL)
        threading.Thread(target=self.pipeline.set_state, args=(Gst.State.NULL,), daemon=True).start()


@pytest.fixture
def rig() -> Any:
    """Yield a Rig and close it."""
    r = Rig()
    try:
        yield r
    finally:
        r.close()


@pytest.fixture
def clocked_rig() -> Any:
    """Yield a Rig whose speaker provides the clock (the SDK's devices in the container), and close it."""
    r = Rig(clocked=True)
    try:
        yield r
    finally:
        r.close()


def test_the_speaker_sink_is_found_in_the_sdk_layout(rig: Rig) -> None:
    """The playback branch's sink, not the microphone's or the wobbler's appsink."""
    factory = rig.speaker.get_factory().get_name()
    if USE_SDK_DEVICES:
        assert factory not in ("appsink", "fakesink")
    else:
        assert factory == "fakesink"


def test_the_microphone_delivers_from_the_start_before_any_reply(clocked_rig: Rig) -> None:
    """Clocked from the speaker, nothing ran until the speaker's stream had audio.

    The trixie container, 2026-10-15: with pulsesink's clock chosen before PLAYING and
    no priming, the pipeline sat in PAUSED and the microphone delivered 0 buffers until
    the first reply was pushed. prime_speaker's 20 ms of silence starts it.
    """
    clocked_rig.start()
    assert clocked_rig.pipeline.get_state(0)[1] == clocked_rig.Gst.State.PLAYING
    assert clocked_rig.mic_rate(1.0) > 5, "the microphone delivered nothing before the first reply"


def test_playback_latency_stays_bounded_after_a_stall(rig: Rig) -> None:
    """The regression for the 65 s lag: after a stall, replies come back to under a second.

    By default the capture thread is held for REACHY_PLAYBACK_TEST_STALL_S (3 s), which
    is safe on the robot. In the trixie container:

    - REACHY_PLAYBACK_TEST_STALL=micfeed starves a pipe-fed microphone while PipeWire
      and the speaker run, the robot's capture gaps. 75 s, measured: the microphone's
      clock heard 0 of 32 replies in 100 s; the speaker's heard them in 0.16-0.17 s from
      the first second (two runs each).
    - REACHY_PLAYBACK_TEST_STALL=pipewire stops the daemon itself. 75 s, measured:
      recovery in 22-35 s, or none within 60 s, about 1 run in 5 with either clock; a
      stress check, not a gate.

    The assertion is convergence within REACHY_PLAYBACK_TEST_RECOVERY_S: three replies
    in a row heard within a second.
    """
    rig.start()
    mic_before = rig.mic_rate()
    assert mic_before > 5, "the microphone delivered nothing before the first reply"
    before = rig.reply_latency(0.3)
    assert before is not None and before < 1.0, f"baseline reply latency {before}"
    rig.stall(STALL_KIND, STALL_S)
    stalled_at = time.monotonic()
    levels = [round(0.35 + 0.05 * i, 2) for i in range(13)]
    latencies: list[float | None] = []
    converged_after = None
    while time.monotonic() - stalled_at < RECOVERY_S:
        latencies.append(rig.reply_latency(levels[len(latencies) % len(levels)]))
        if len(latencies) >= 3 and all(lat is not None and lat < 1.0 for lat in latencies[-3:]):
            converged_after = time.monotonic() - stalled_at
            break
    mic_after = rig.mic_rate()
    print(
        f"stall {STALL_KIND} {STALL_S:.0f}s: converged after {converged_after}s, latencies {latencies}, "
        f"mic {mic_before:.0f}/s before, {mic_after:.0f}/s after"
    )
    assert converged_after is not None, f"no recovery {RECOVERY_S:.0f}s after a {STALL_S:.0f}s stall: {latencies}"
    assert mic_after > 0.5 * mic_before, (
        f"the microphone slowed after the stall: {mic_before:.0f}/s, {mic_after:.0f}/s"
    )


def test_a_rebuilt_speaker_stream_plays_the_next_reply_and_keeps_the_microphone(rig: Rig) -> None:
    """The watchdog's remedy: the speaker sink alone is recreated, nothing else moves."""
    Gst = rig.Gst
    rig.start()
    mic_before = rig.mic_rate(1.0)
    assert mic_before > 5
    assert ph.rebuild_sink(rig.speaker) is True
    assert rig.pipeline.get_state(0)[1] == Gst.State.PLAYING
    latencies = [rig.reply_latency(level) for level in (0.4, 0.5)]
    # The first reply may wait for the new stream to start; the next plays at once.
    assert latencies[0] is not None and latencies[0] < 3.0, latencies
    assert latencies[1] is not None and latencies[1] < 1.0, latencies
    assert rig.mic_rate(1.0) > 0.5 * mic_before, "the microphone stalled during the rebuild"


def test_the_watchdog_logs_one_info_line_and_does_nothing_when_healthy(
    rig: Rig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drift data from day one: one INFO line per check, no action under the bound."""
    rig.start()
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a healthy stream"))
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        verdict = ph.check_playback(rig.pipeline)
    assert verdict.action in ("ok", "unmeasured")
    lines = [r for r in caplog.records if r.name == ph.logger.name]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "Playback latency" in lines[0].getMessage()


# --- the motif playbin, the second stream -------------------------------------------------


def test_a_motif_stream_is_logged_and_torn_down_when_the_sound_ends(
    rig: Rig, caplog: pytest.LogCaptureFixture, tmp_path: Any
) -> None:
    """conversation.sound's playbin is a second speaker stream: opened, then closed at its end."""
    import wave

    Gst = rig.Gst
    rig.start()
    path = tmp_path / "motif.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes((np.ones(5512) * 1000).astype(np.int16).tobytes())  # 0.25 s

    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        rig.audio.play_sound(str(path))
        playbin = rig.audio._playbin
        watcher = ph.watch_motif(rig.audio, playbin, str(path))
        watcher.join(timeout=5.0)

    messages = [r.getMessage() for r in caplog.records if r.name == ph.logger.name]
    assert any("Motif stream opened" in m and "motif.wav" in m for m in messages), messages
    assert any("Motif stream closed" in m and "ended" in m for m in messages), messages
    assert rig.audio._playbin is None
    assert playbin.get_state(0)[1] == Gst.State.NULL
    # Speech is untouched by the second stream's end.
    lat = rig.reply_latency(0.6)
    assert lat is not None and lat < 1.0


def test_a_replaced_motif_is_not_torn_down_twice(rig: Rig, caplog: pytest.LogCaptureFixture, tmp_path: Any) -> None:
    """A newer sound owns audio._playbin; the old watcher only logs the replacement."""
    import wave

    rig.start()
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes((np.ones(22050 * 3) * 1000).astype(np.int16).tobytes())  # 3 s

    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        rig.audio.play_sound(str(path))
        first = rig.audio._playbin
        old = ph.watch_motif(rig.audio, first, str(path))
        time.sleep(0.3)
        rig.audio.play_sound(str(path))  # the SDK sets the first to NULL and replaces it
        second = rig.audio._playbin
        old.join(timeout=3.0)
        assert not old.is_alive()
        assert rig.audio._playbin is second  # the newer sound is left alone
        ph.stop_motif(rig.audio)

    messages = [r.getMessage() for r in caplog.records if r.name == ph.logger.name]
    assert any("Motif stream closed" in m and "replaced" in m for m in messages), messages
    assert any("Motif stream closed" in m and "stop" in m for m in messages), messages
    assert rig.audio._playbin is None


# --- one cooldown for every rebuild trigger -----------------------------------------------


@pytest.fixture(autouse=True)
def fresh_governor(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Each test starts with no rebuild history."""
    governor = ph.RebuildGovernor()
    monkeypatch.setattr(ph, "GOVERNOR", governor)
    return governor


@pytest.mark.parametrize(
    ("now", "history", "action"),
    [
        (1000.0, (), "rebuild"),  # the first rebuild is at once
        (1000.0, (990.0,), "cooldown"),  # 30 s after one
        (1000.0, (970.5,), "cooldown"),
        (1000.0, (970.0,), "rebuild"),
        (1000.0, (900.0, 960.0), "cooldown"),  # 60 s after the second
        (1000.0, (880.0, 940.0), "rebuild"),
        (1000.0, (700.0, 760.0, 890.0), "cooldown"),  # 120 s after the third
        (1000.0, (700.0, 760.0, 870.0), "rebuild"),
        (5000.0, tuple(4410.0 + 60 * i for i in range(10)), "cooldown"),  # capped at 600 s
        (1000.0, (100.0, 200.0, 300.0), "rebuild"),  # older than the 600 s window: forgotten
    ],
)
def test_rebuilds_share_one_cooldown_that_doubles(now: float, history: tuple[float, ...], action: str) -> None:
    """30 s after a rebuild, doubling with each further rebuild in 600 s, capped at 600 s."""
    verdict = ph.rebuild_decision(now, history)
    assert verdict.action == action
    assert verdict.cause


def test_the_cooldown_cause_says_how_long_is_left() -> None:
    """The held-back log line says when the next rebuild is allowed."""
    cause = ph.rebuild_decision(1000.0, (990.0,)).cause
    assert "20" in cause and "30" in cause


class _Pipeline:
    """A stand-in pipeline in a given state."""

    def __init__(self, playing: bool = True, state: str | None = None) -> None:
        self.playing = playing
        self.state = state  # a Gst.State name, overriding ``playing``

    def get_state(self, _timeout: int) -> tuple[Any, Any, Any]:
        Gst = _gst()
        if self.state is not None:
            return (None, getattr(Gst.State, self.state), None)
        return (None, Gst.State.PLAYING if self.playing else Gst.State.NULL, None)


def test_a_rebuild_is_logged_once_with_its_trigger_and_restarts_the_speaker_clock(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every rebuild logs one WARNING naming its trigger and cause; the speaker's clock is restarted after it."""
    order: list[str] = []
    audio = SimpleNamespace(_pipeline=_Pipeline())
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: "sink")
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: order.append(f"rebuild:{sink}") or True)
    monkeypatch.setattr(ph, "restart_speaker_clock", lambda a: order.append("restart") or True)
    governor = ph.RebuildGovernor(clock=lambda: 100.0)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        assert governor.rebuild(audio, "unheard replies", "3 replies in a row not heard") is True
    assert order == ["rebuild:sink", "restart"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "unheard replies" in warnings[0] and "3 replies in a row not heard" in warnings[0]


def test_a_second_trigger_inside_the_cooldown_is_held_back_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No rebuild loops: triggers share the cooldown, and a held-back one says so."""
    now = [100.0]
    rebuilt: list[str] = []
    audio = SimpleNamespace(_pipeline=_Pipeline())
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: "sink")
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: rebuilt.append(sink) or True)
    governor = ph.RebuildGovernor(clock=lambda: now[0])
    assert governor.rebuild(audio, "capture gap", "no microphone frames for 7.0 s")
    now[0] = 110.0
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        assert governor.rebuild(audio, "playback latency", "latency 65.5 s") is False
    assert rebuilt == ["sink"]
    held = [r.getMessage() for r in caplog.records if "held back" in r.getMessage()]
    assert len(held) == 1 and "playback latency" in held[0]
    now[0] = 131.0
    assert governor.rebuild(audio, "unheard replies", "3 in a row")
    assert rebuilt == ["sink", "sink"]


def test_a_rebuild_never_touches_a_pipeline_that_capture_off_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trigger racing conversation.capture {"on": false} finds the pipeline not PLAYING and leaves it."""
    audio = SimpleNamespace(_pipeline=_Pipeline(playing=False))
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: pytest.fail("looked for the sink"))
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a stopped pipeline"))
    governor = ph.RebuildGovernor(clock=lambda: 100.0)
    assert governor.rebuild(audio, "capture gap", "test") is False
    assert governor.rebuild(audio, "capture gap", "test") is False  # and a skip uses no cooldown


def test_the_latency_check_leaves_a_stopped_pipeline_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """measure_latency_s and the rebuild race a capture off; a pipeline not PLAYING is not read."""
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: pytest.fail("walked a stopped pipeline"))
    assert ph.check_playback(_Pipeline(playing=False)).action == "unmeasured"


def test_a_rebuild_waits_for_a_capture_change_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEDIA_LOCK: a rebuild requested mid capture-off runs after it, and finds the pipeline stopped."""
    pipeline = _Pipeline()
    audio = SimpleNamespace(_pipeline=pipeline)
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt mid capture-off"))
    in_change = threading.Event()
    result: list[bool] = []

    @ph.under_media_lock
    def capture_off() -> None:
        in_change.set()
        time.sleep(0.2)
        pipeline.playing = False

    t = threading.Thread(target=capture_off)
    t.start()
    in_change.wait(1.0)
    result.append(ph.RebuildGovernor().rebuild(audio, "capture gap", "test"))
    t.join()
    assert result == [False]


# --- the capture-gap trigger ---------------------------------------------------------------

IDLE_S = float(os.environ.get("REACHY_PLAYBACK_TEST_IDLE_S", "15"))


@pytest.mark.parametrize(
    ("gap_s", "action"),
    [
        (0.01, "ok"),  # one 10 ms buffer to the next
        (1.0, "ok"),
        (5.0, "ok"),
        (5.01, "rebuild"),
        (10.0, "rebuild"),  # the robot's gaps, 2026-10-10
        (75.0, "rebuild"),  # the container's stall
    ],
)
def test_a_capture_gap_over_the_bound_asks_for_a_rebuild(gap_s: float, action: str) -> None:
    """A gap in microphone frames longer than CAPTURE_GAP_S (5 s) rebuilds the speaker stream."""
    verdict = ph.capture_gap_decision(gap_s, limit_s=5.0)
    assert verdict.action == action
    assert verdict.cause


def test_the_capture_gap_cause_names_the_gap() -> None:
    """The WARNING line carries the gap length and the cause."""
    cause = ph.capture_gap_decision(7.25).cause
    assert "7.2" in cause and "microphone" in cause and "pulsesink" in cause


@pytest.mark.parametrize(
    ("age_s", "warned", "action"),
    [
        (0.5, False, "ok"),
        (3.0, False, "ok"),
        (3.01, False, "stalled"),  # no frames for 3 s with capture on: say so, restart the clock
        (30.0, False, "stalled"),
        (3.01, True, "ok"),  # once per stall
        (30.0, True, "ok"),
    ],
)
def test_a_capture_stall_is_reported_once(age_s: float, warned: bool, action: str) -> None:
    """While no frames come, the stall is reported once."""
    assert ph.capture_stall_decision(age_s, warned, stall_s=3.0).action == action


def test_the_capture_watch_reports_a_stall_and_rebuilds_once_capture_recovers() -> None:
    """The sequence the record loop drives: frames, a stall, one report, recovery, one rebuild."""
    watch = ph.CaptureWatch()
    assert watch.frame(0.0).action == "ok"
    assert watch.frame(0.01).action == "ok"
    assert watch.idle(1.0).action == "ok"
    assert watch.idle(3.5).action == "stalled"
    assert watch.idle(4.0).action == "ok"
    assert watch.idle(6.0).action == "ok"  # no rebuild while capture is still stalled
    verdict = watch.frame(7.21)
    assert verdict.action == "rebuild" and "7.2" in verdict.cause
    assert watch.frame(7.22).action == "ok"
    assert watch.idle(7.5).action == "ok"
    assert watch.idle(10.6).action == "stalled"  # a later stall is reported again


def test_capture_off_is_not_a_gap() -> None:
    """reset() at capture off: the time capture was off never counts, and waiting restarts."""
    watch = ph.CaptureWatch()
    watch.frame(0.0)
    watch.reset()
    assert watch.idle(100.0).action == "ok"  # the first poll after a restart only starts the wait
    assert watch.frame(100.5).action == "ok"
    watch.reset()
    assert watch.idle(200.0).action == "ok"
    assert watch.idle(203.5).action == "stalled"  # a pipeline that never delivers is reported


def test_short_gaps_never_trigger() -> None:
    """Scheduling hiccups, a slow pull, a busy Pi: anything up to 5 s is no gap."""
    watch = ph.CaptureWatch()
    t = 0.0
    for gap in (0.01, 0.02, 0.5, 1.9, 2.9, 0.01, 4.9):
        t += gap
        assert watch.frame(t).action == "ok"


def test_acting_on_a_gap_rebuilds_through_the_governor(monkeypatch: pytest.MonkeyPatch) -> None:
    """act_on_capture: a gap is a "capture gap" rebuild; a stall restarts the clock and rebuilds nothing."""
    calls: list[tuple[str, str]] = []
    governor = SimpleNamespace(rebuild=lambda audio, trigger, cause: calls.append((trigger, cause)) or True)
    monkeypatch.setattr(ph, "restart_speaker_clock", lambda audio: calls.append(("restart", "")) or True)
    ph.act_on_capture(ph.capture_stall_decision(3.5, False), "audio", governor)
    ph.act_on_capture(ph.capture_gap_decision(7.2), "audio", governor)
    ph.act_on_capture(ph.Verdict("ok", ""), "audio", governor)
    assert [c[0] for c in calls] == ["restart", "capture gap"]
    assert "7.2" in calls[1][1]


def _drive_capture_watch(rig: Rig) -> list[ph.Verdict]:
    """Feed a CaptureWatch from the rig's microphone polls and act as the app does, off the poll thread."""
    watch = ph.CaptureWatch()
    acted: list[ph.Verdict] = []

    def on_poll(now: float, got: bool) -> None:
        verdict = watch.frame(now) if got else watch.idle(now)
        if verdict.action != "ok":
            acted.append(verdict)
            threading.Thread(target=ph.act_on_capture, args=(verdict, rig.audio), daemon=True).start()

    rig.on_mic_poll = on_poll
    return acted


def _hold_capture(rig: Rig, seconds: float) -> None:
    """Hold the capture thread in GStreamer for ``seconds``: an app-visible gap, robot-safe."""
    Gst = rig.Gst
    until = time.monotonic() + seconds

    def hold(_pad: Any, _info: Any) -> Any:
        while time.monotonic() < until:
            time.sleep(0.01)
        return Gst.PadProbeReturn.REMOVE

    rig.mic.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, hold)
    time.sleep(seconds)


def test_the_capture_gap_trigger_rebuilds_the_speaker_once_capture_recovers(
    rig: Rig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration: a 6 s capture gap is reported once, then one rebuild when frames return, and replies play."""
    rebuilt: list[Any] = []
    real_rebuild = ph.rebuild_sink
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: rebuilt.append(sink) or real_rebuild(sink))
    rig.start()
    acted = _drive_capture_watch(rig)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        _hold_capture(rig, 6.0)
        time.sleep(1.5)
    assert [v.action for v in acted] == ["stalled", "rebuild"], acted
    assert rebuilt == [rig.speaker]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    rebuilds = [m for m in warnings if "Rebuilding the speaker stream" in m]
    assert len(rebuilds) == 1 and "trigger: capture gap" in rebuilds[0], warnings
    assert re.search(r"a [5-7]\.\d s gap in microphone frames", rebuilds[0]), rebuilds  # the gap length (6 s held)
    lat = rig.reply_latency(0.6)
    assert lat is not None and lat < 1.0, lat


def _barge_in(audio: Any) -> None:
    """Flush as the app's barge-in does; the container lacks the app's other imports, so there the appsrc alone."""
    try:
        from reachy_mini_conversation_app.console import _flush_appsrc
    except ImportError:
        Gst = _gst()
        audio._appsrc.send_event(Gst.Event.new_flush_start())
        audio._appsrc.send_event(Gst.Event.new_flush_stop(False))
        return
    _flush_appsrc(audio._appsrc)


def test_the_capture_gap_trigger_stays_quiet_through_speech_barge_ins_and_idle(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal replies, a barge-in flush and an idle stretch: no stall, no gap, no rebuild."""
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a healthy stream"))
    rig.start()
    acted = _drive_capture_watch(rig)
    for level in (0.4, 0.5):
        assert rig.reply_latency(level) is not None
    threading.Thread(target=rig.push_reply, args=(2.0, 0.45), daemon=True).start()
    time.sleep(0.5)
    _barge_in(rig.audio)
    time.sleep(IDLE_S)
    assert acted == []


# --- the unheard-reply trigger -------------------------------------------------------------


@pytest.mark.parametrize(
    ("speech_pushed_s", "heard_s", "monitored_s", "action"),
    [
        (0.1, 0.0, 5.0, "skip"),  # too little speech to judge
        (2.0, 1.9, 5.0, "heard"),
        (2.0, 0.5, 5.0, "heard"),  # a quarter of what was pushed
        (2.0, 0.49, 5.0, "unheard"),
        (2.0, 0.0, 5.0, "unheard"),  # the behind case: nothing reaches the device
        (10.0, 1.0, 5.0, "heard"),  # a long reply: judged on the first 4 s
        (10.0, 0.99, 5.0, "unheard"),
        (2.0, 0.0, 2.4, "unmeasured"),  # the monitor itself stalled: no verdict
    ],
)
def test_a_reply_is_judged_by_the_speech_the_monitor_heard(
    speech_pushed_s: float, heard_s: float, monitored_s: float, action: str
) -> None:
    """Heard at the deadline: at least a quarter of the speech pushed (up to 4 s), on the speaker's monitor."""
    verdict = ph.reply_verdict(speech_pushed_s, heard_s, monitored_s, window_s=5.0)
    assert verdict.action == action
    assert verdict.cause


@pytest.mark.parametrize(("count", "action"), [(0, "ok"), (1, "ok"), (2, "ok"), (3, "rebuild"), (4, "rebuild")])
def test_three_unheard_replies_in_a_row_ask_for_a_rebuild(count: int, action: str) -> None:
    """The streak bound is UNHEARD_LIMIT = 3, and the cause carries the count."""
    verdict = ph.unheard_streak_decision(count, limit=3)
    assert verdict.action == action
    assert str(count) in verdict.cause


def test_speech_seconds_count_only_audio_above_the_floor() -> None:
    """loud_seconds: silence is 0 s of speech, a tone is all speech, half and half is half."""
    assert ph.loud_seconds(np.zeros(16000, dtype=np.float32), 16000) == (1.0, 0.0)
    assert ph.loud_seconds(np.full(16000, 0.3, dtype=np.float32), 16000) == (1.0, 1.0)
    half = np.concatenate([np.zeros(8000), np.full(8000, 0.3)]).astype(np.float32)
    assert ph.loud_seconds(half, 16000) == (1.0, 0.5)
    assert ph.loud_seconds(np.zeros(0, dtype=np.float32), 16000) == (0.0, 0.0)


def _reply(tracker: ph.ReplyTracker, at: float, heard: bool, speech_s: float = 1.0) -> None:
    """Push one reply of ``speech_s`` at ``at`` in 100 ms chunks; the monitor hears it, or silence."""
    for i in range(int(speech_s * 10)):
        tracker.pushed(at + 0.025 * i, 0.1, 0.1)
    for i in range(50):  # 5 s of monitor buffers, 100 ms each
        t = at + 0.1 * i
        tracker.monitored(t, 0.1, 0.1 if heard and 0.2 <= t - at < 0.2 + speech_s else 0.0)


def _judge(tracker: ph.ReplyTracker, now: float) -> list[str]:
    """Return the unheard trigger's verdicts; the measured lags are tested on their own."""
    return [v.action for v in tracker.tick(now) if v.action not in ("heard_after", "heard_late")]


def test_the_third_unheard_reply_in_a_row_rebuilds_and_starts_the_count_again() -> None:
    """Heard, then three unheard: the third asks for a rebuild, and the count starts again."""
    tracker = ph.ReplyTracker()
    _reply(tracker, 0.0, heard=True)
    assert _judge(tracker, 5.1) == ["heard"]
    _reply(tracker, 10.0, heard=False)
    assert _judge(tracker, 15.1) == ["unheard"]
    _reply(tracker, 20.0, heard=False)
    assert _judge(tracker, 25.1) == ["unheard"]
    assert tracker.unheard_in_a_row == 2
    _reply(tracker, 30.0, heard=False)
    assert _judge(tracker, 35.1) == ["unheard", "rebuild"]
    assert tracker.unheard_in_a_row == 0


def test_a_reply_that_plays_normally_resets_the_count() -> None:
    """Two unheard, one heard, two unheard: no rebuild."""
    tracker = ph.ReplyTracker()
    _reply(tracker, 0.0, heard=True)
    t = 10.0
    for heard in (False, False, True, False, False):
        _reply(tracker, t, heard=heard)
        assert "rebuild" not in _judge(tracker, t + 5.1)
        t += 10.0
    assert tracker.unheard_in_a_row == 2


def test_a_barge_in_flush_is_never_unheard() -> None:
    """Dropped audio is expected: a flushed reply is not judged, and the count stands."""
    tracker = ph.ReplyTracker()
    _reply(tracker, 0.0, heard=True)
    _judge(tracker, 5.1)
    for t in (10.0, 20.0):
        _reply(tracker, t, heard=False)
        _judge(tracker, t + 5.1)
    assert tracker.unheard_in_a_row == 2
    _reply(tracker, 30.0, heard=False)
    tracker.flushed(30.5)  # barge-in before the window closed
    assert _judge(tracker, 35.1) == ["flushed"]
    assert tracker.unheard_in_a_row == 2
    # the next reply after the barge-in is a new reply, judged as usual
    _reply(tracker, 31.0, heard=True)
    assert _judge(tracker, 36.1) == ["heard"]
    assert tracker.unheard_in_a_row == 0


def test_a_barge_in_leaves_earlier_replies_to_be_judged() -> None:
    """Only the reply being pushed is dropped; one that finished earlier is still judged."""
    tracker = ph.ReplyTracker()
    _reply(tracker, 0.0, heard=True)
    _reply(tracker, 2.2, heard=True)
    tracker.flushed(3.0)
    assert _judge(tracker, 7.3) == ["heard", "flushed"]


def test_pushes_close_together_are_one_reply_and_a_pause_starts_another() -> None:
    """Chunks up to REPLY_GAP_S (1.5 s) apart are one reply."""
    tracker = ph.ReplyTracker()
    for t in (0.0, 0.5, 1.9, 3.3):  # one reply, chunks 1.4 s apart
        tracker.pushed(t, 0.5, 0.5)
    tracker.pushed(5.0, 0.5, 0.5)  # 1.7 s later: a second reply
    for i in range(110):
        tracker.monitored(0.1 * i, 0.1, 0.1)
    assert _judge(tracker, 11.0) == ["heard", "heard"]


def test_unheard_replies_count_only_once_the_monitor_has_heard_one() -> None:
    """A monitor on the wrong device would hear nothing at all; until one reply is heard, nothing is counted."""
    tracker = ph.ReplyTracker()
    for t in (0.0, 10.0, 20.0, 30.0):
        _reply(tracker, t, heard=False)
        assert "rebuild" not in _judge(tracker, t + 5.1)
    assert tracker.unheard_in_a_row == 0 and not tracker.armed


def test_a_stalled_monitor_counts_nothing() -> None:
    """No monitor buffers in the window: unmeasured, never unheard."""
    tracker = ph.ReplyTracker()
    _reply(tracker, 0.0, heard=True)
    _judge(tracker, 5.1)
    tracker.pushed(10.0, 1.0, 1.0)
    assert _judge(tracker, 15.1) == ["unmeasured"]
    assert tracker.unheard_in_a_row == 0


# --- the measured lag: first push to sound on the speaker's monitor -------------------------


def _push(tracker: ph.ReplyTracker, at: float, seconds: float, chunk_s: float = 0.1) -> None:
    """Push ``seconds`` of speech from ``at``, four times faster than real time, as the app does."""
    for i in range(int(round(seconds / chunk_s))):
        tracker.pushed(at + 0.025 * i, chunk_s, chunk_s)


def _hear(tracker: ph.ReplyTracker, start: float, end: float, loud: list[tuple[float, float]]) -> None:
    """Monitor buffers of 50 ms from ``start`` to ``end``, loud inside any of the ``loud`` spans."""
    for i in range(int(round((end - start) / 0.05))):
        t = round(start + 0.05 * (i + 1), 3)  # a buffer arrives at the end of its 50 ms
        speech = 0.05 if any(a <= t < b for a, b in loud) else 0.0
        tracker.monitored(t, 0.05, speech)


def _lags(verdicts: list[ph.Verdict]) -> list[str]:
    return [v.cause for v in verdicts if v.action == "heard_after"]


def test_a_reply_heard_reports_its_lag_once() -> None:
    """The first sound after the first push, 0.3 s later: one heard_after verdict, and the last lag kept."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 9.0, 12.0, [(10.3, 11.3)])
    verdicts = tracker.tick(12.0)
    assert _lags(verdicts) == ["0.30 s"]
    assert "heard_late" not in [v.action for v in verdicts]
    assert tracker.last_lag() == pytest.approx((0.3, 10.3))
    assert _lags(tracker.tick(12.5)) == []  # reported once


def test_a_lag_over_the_bound_is_heard_late_naming_it() -> None:
    """A 6 s lag is past the 5 s bound: a heard_late verdict whose cause names the value and the bound."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 9.0, 18.0, [(16.0, 17.0)])
    verdicts = tracker.tick(18.0)
    assert _lags(verdicts) == ["6.00 s"]
    late = [v.cause for v in verdicts if v.action == "heard_late"]
    assert len(late) == 1 and "6.0" in late[0] and "5.0" in late[0]


def test_a_65_s_lag_is_measured() -> None:
    """The robot's 65 s lag fits inside the lag window, so it is measured, not lost."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 0.0, 1.0)
    _hear(tracker, 0.0, 70.0, [(65.0, 66.0)])
    assert _lags(tracker.tick(70.0)) == ["65.00 s"]


def test_a_flushed_reply_never_heard_records_no_lag() -> None:
    """Barge-in drops the reply; a sound after the flush is not its lag."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 10.0, 1.0)
    tracker.flushed(10.2)
    _hear(tracker, 9.0, 14.0, [(12.0, 13.0)])  # a motif, say
    assert _lags(tracker.tick(14.0)) == []
    assert tracker.last_lag() is None


def test_a_reply_heard_then_flushed_keeps_its_lag() -> None:
    """Barge-in after the reply was heard: its lag stands."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 10.0, 3.0)
    _hear(tracker, 9.0, 11.0, [(10.25, 11.0)])
    tracker.flushed(11.0)
    assert _lags(tracker.tick(11.0)) == ["0.25 s"]


def test_sound_already_playing_at_the_push_is_not_the_reply() -> None:
    """A motif loud from before the push to past the window: no sound starts after the push, no lag."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _hear(tracker, 8.0, 10.0, [(9.0, 200.0)])
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 10.0, 12.0, [(9.0, 200.0)])
    assert _lags(tracker.tick(12.0)) == []


def test_a_sound_before_the_push_is_not_the_reply() -> None:
    """A motif that started and ended before the push is ignored; the reply's own sound is its lag."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _hear(tracker, 8.0, 10.0, [(8.5, 9.5)])
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 10.0, 12.0, [(10.4, 11.4)])
    assert _lags(tracker.tick(12.0)) == ["0.40 s"]


def test_a_reply_queued_behind_a_long_one_is_not_lagged() -> None:
    """Reply 2 pushed while reply 1 still plays waits its turn: that queueing is not lag."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 0.0, 8.0)  # 8 s of speech pushed in 2 s
    _push(tracker, 4.0, 1.0)  # 2 s after the last chunk of reply 1: a new reply
    _hear(tracker, 0.0, 10.0, [(0.3, 9.3)])  # 8.3 to 9.3 is reply 2, contiguous
    lags = _lags(tracker.tick(10.0))
    assert lags[0] == "0.30 s" and len(lags) == 2
    assert float(lags[1].split()[0]) < 0.5, lags


def test_a_reply_never_heard_expires_and_does_not_block_the_next() -> None:
    """Never heard within the lag window: no lag for it (the unheard trigger's case); the next is measured."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 0.0, 1.0)
    _hear(tracker, 0.0, ph.LAG_WINDOW_S + 1.0, [])
    tracker.tick(ph.LAG_WINDOW_S + 1.0)
    at = ph.LAG_WINDOW_S + 2.0
    _push(tracker, at, 1.0)
    _hear(tracker, at, at + 2.0, [(at + 0.3, at + 1.3)])
    assert _lags(tracker.tick(at + 2.0)) == ["0.30 s"]


def test_a_reply_the_monitor_could_not_judge_claims_no_later_sound() -> None:
    """The monitor down for a reply: unmeasured, and the next reply's sound is the next reply's."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 0.0, 1.0)
    tracker.tick(5.1)  # no monitor buffers at all
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 9.0, 12.0, [(10.3, 11.3)])
    assert _lags(tracker.tick(12.0)) == ["0.30 s"]


def test_a_reply_too_quiet_to_hear_claims_no_later_sound() -> None:
    """A finished reply with less than MIN_SPEECH_S of speech is never waited for."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    tracker.pushed(0.0, 0.5, 0.0)  # silence pushed
    _hear(tracker, 0.0, 3.0, [])
    _push(tracker, 3.0, 1.0)
    _hear(tracker, 3.0, 5.0, [(3.3, 4.3)])
    assert _lags(tracker.tick(5.0)) == ["0.30 s"]


def test_a_reply_after_a_barge_in_counts_from_its_own_push() -> None:
    """A barge-in cuts the playing reply short: the next reply is not queued behind the rest of it."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 0.0, 8.0)
    _hear(tracker, 0.0, 3.0, [(0.3, 3.0)])
    tracker.flushed(3.0)
    _push(tracker, 5.0, 2.0)
    _hear(tracker, 3.0, 8.0, [(5.3, 7.3)])
    assert _lags(tracker.tick(8.0)) == ["0.30 s", "0.30 s"]


def test_a_reply_after_a_rebuild_counts_from_its_own_push() -> None:
    """A rebuild drops what was queued: the first reply after it is not queued behind the lagged one."""
    rebuilt: list[float | None] = [None]
    tracker = ph.ReplyTracker(rebuilt_at=lambda: rebuilt[0])
    _push(tracker, 0.0, 8.0)
    _hear(tracker, 0.0, 66.0, [(65.0, 66.0)])
    assert _lags(tracker.tick(66.0)) == ["65.00 s"]
    rebuilt[0] = 66.5
    _push(tracker, 70.0, 1.0)
    _hear(tracker, 66.0, 72.0, [(70.3, 71.3)])
    assert _lags(tracker.tick(72.0)) == ["0.30 s"]


def test_a_rebuild_drops_the_replies_it_cut_off() -> None:
    """A reply lost before a rebuild never claims the next reply's sound."""
    rebuilt: list[float | None] = [None]
    tracker = ph.ReplyTracker(rebuilt_at=lambda: rebuilt[0])
    _push(tracker, 0.0, 1.0)  # lost: the speaker was desynced
    _hear(tracker, 0.0, 3.0, [])
    rebuilt[0] = 3.0
    tracker.tick(3.5)
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 3.0, 12.0, [(10.3, 11.3)])
    assert _lags(tracker.tick(12.0)) == ["0.30 s"]


@pytest.mark.parametrize(
    ("last", "now", "line"),
    [
        (None, 100.0, "no reply measured"),
        ((0.31, 58.0), 100.0, "last reply heard after 0.31 s, 42 s ago"),
        ((65.0, 99.6), 100.0, "last reply heard after 65.00 s, 0 s ago"),
    ],
)
def test_the_minute_line_reports_the_last_lag_and_its_age(
    last: tuple[float, float] | None, now: float, line: str
) -> None:
    """Never a fake 0.000: the last measured lag with its age, or none."""
    assert ph.lag_report(last, now) == line


def test_the_minute_check_reports_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One INFO line from the tracker's last lag; the rebuild belongs to the per-reply trigger."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    _push(tracker, 10.0, 1.0)
    _hear(tracker, 9.0, 18.0, [(16.0, 17.0)])
    tracker.tick(18.0)
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: SimpleNamespace(get_name=lambda: "pulsesink0"))
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("the minute check rebuilt"))
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        ph.check_playback(_Pipeline(), replies=tracker, now=40.0)
    lines = [r.getMessage() for r in caplog.records if r.name == ph.logger.name]
    assert lines == ["Playback latency: last reply heard after 6.00 s, 24 s ago (speaker pulsesink0, bound 5.0 s)"]


def test_only_pulsesink_has_a_monitor_to_hear_on(rig: Rig) -> None:
    """fakesink, alsasink and the rest: the unheard trigger is unavailable, not wrong."""
    if USE_SDK_DEVICES:
        assert ph.SpeakerMonitor.monitor_source(rig.speaker).endswith(".monitor")
    else:
        assert ph.SpeakerMonitor.monitor_source(rig.speaker) is None
    assert ph.SpeakerMonitor.monitor_source(None) is None


class _Desync:
    """The behind case, on the local rig: the device plays silence, whatever is pushed, until a rebuild."""

    def __init__(self, rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
        self.on = False
        self.rebuilds = 0
        rig.silenced = lambda: self.on  # read by the rig's heard probe
        real_rebuild = ph.rebuild_sink

        def rebuild(sink: Any) -> bool:
            self.rebuilds += 1
            self.on = False
            return real_rebuild(sink)

        monkeypatch.setattr(ph, "rebuild_sink", rebuild)


def _drive_reply_tracker(rig: Rig, stop: threading.Event) -> tuple[ph.ReplyTracker, list[ph.Verdict]]:
    """Feed a ReplyTracker as the app does: pushes, the speaker heard, a tick every 0.2 s acting on verdicts.

    Heard is the app's SpeakerMonitor in the container; locally, the buffers the
    fakesink (sync=true) renders, counted on its pad.
    """
    Gst = rig.Gst
    tracker = ph.ReplyTracker()
    verdicts: list[ph.Verdict] = []
    push = rig.audio._push_appsrc_buffer

    def pushed(chunk: Any) -> Any:
        seconds, speech = ph.loud_seconds(np.asarray(chunk)[:, 0], 16000)
        tracker.pushed(time.monotonic(), seconds, speech)
        return push(chunk)

    rig.audio._push_appsrc_buffer = pushed
    monitor = None
    if USE_SDK_DEVICES:
        monitor = ph.SpeakerMonitor(tracker.monitored)
        assert monitor.start(ph.SpeakerMonitor.monitor_source(rig.speaker))
    else:

        def heard(pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            data = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.float32)
            channels = pad.get_current_caps().get_structure(0).get_value("channels")
            seconds, speech = ph.loud_seconds(data.reshape(-1, channels)[:, 0], 16000)
            silenced = getattr(rig, "silenced", None)
            tracker.monitored(time.monotonic(), seconds, 0.0 if silenced is not None and silenced() else speech)
            return Gst.PadProbeReturn.OK

        rig.speaker.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, heard)

    def ticker() -> None:
        while not stop.wait(0.2):
            for verdict in tracker.tick(time.monotonic()):
                verdicts.append(verdict)
                trigger = {"rebuild": "unheard replies"}.get(verdict.action)
                if trigger is not None:  # as LocalStream._judge_replies does; heard_late only warns
                    threading.Thread(
                        target=ph.GOVERNOR.rebuild, args=(rig.audio, trigger, verdict.cause), daemon=True
                    ).start()
        if monitor is not None:
            monitor.stop()

    threading.Thread(target=ticker, daemon=True).start()
    return tracker, verdicts


def _wait_judged(verdicts: list[ph.Verdict], count: int, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(_judged(verdicts)) < count:
        time.sleep(0.05)


def _judged(verdicts: list[ph.Verdict]) -> list[ph.Verdict]:
    """Return the unheard trigger's verdicts, one per reply."""
    return [v for v in verdicts if v.action not in ("rebuild", "heard_after", "heard_late")]


def _wait_lags(verdicts: list[ph.Verdict], count: int, timeout: float) -> list[float]:
    """Wait for ``count`` measured lags; return every lag measured."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len([v for v in verdicts if v.action == "heard_after"]) < count:
        time.sleep(0.05)
    return [float(v.cause.split()[0]) for v in verdicts if v.action == "heard_after"]


def test_three_unheard_replies_rebuild_the_speaker_and_the_next_is_heard(
    rig: Rig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration: one heard reply, then the device plays silence; the third unheard reply rebuilds it."""
    if USE_SDK_DEVICES:
        pytest.skip("the behind case itself is reproduced in test_the_triggers_recover_the_behind_case")
    desync = _Desync(rig, monkeypatch)
    rig.start()
    stop = threading.Event()
    try:
        tracker, verdicts = _drive_reply_tracker(rig, stop)
        with caplog.at_level(logging.INFO, logger=ph.logger.name):
            rig.push_reply(0.5, 0.4)
            _wait_judged(verdicts, 1)
            desync.on = True
            for _ in range(3):
                rig.push_reply(0.5, 0.4)
                time.sleep(2.0)  # a pause between replies
            _wait_judged(verdicts, 4)
            time.sleep(1.0)
            assert desync.rebuilds == 1
            rig.push_reply(0.5, 0.4)
            _wait_judged(verdicts, 5)
    finally:
        stop.set()
    actions = [v.action for v in verdicts if v.action not in ("heard_after", "heard_late")]
    assert actions == ["heard", "unheard", "unheard", "unheard", "rebuild", "heard"], verdicts
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    rebuilds = [m for m in warnings if "Rebuilding the speaker stream" in m]
    assert len(rebuilds) == 1 and "trigger: unheard replies" in rebuilds[0] and "3 replies in a row" in rebuilds[0]


def test_the_unheard_trigger_stays_quiet_through_speech_barge_ins_and_idle(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal replies, a barge-in mid-reply and an idle stretch: every judged reply heard or flushed."""
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a healthy stream"))
    rig.start()
    stop = threading.Event()
    try:
        tracker, verdicts = _drive_reply_tracker(rig, stop)
        rig.push_reply(0.8, 0.4)
        time.sleep(2.0)
        rig.push_reply(0.8, 0.5)
        time.sleep(2.0)
        pusher = threading.Thread(target=rig.push_reply, args=(3.0, 0.45), daemon=True)
        pusher.start()
        time.sleep(0.6)
        _barge_in(rig.audio)
        tracker.flushed(time.monotonic())  # as LocalStream.clear_audio_queue does
        pusher.join()
        time.sleep(IDLE_S)
        _wait_judged(verdicts, 3)
    finally:
        stop.set()
    actions = [v.action for v in verdicts]
    assert "heard_late" not in actions, verdicts
    actions = [v.action for v in _judged(verdicts)]
    assert actions[:2] == ["heard", "heard"] and "unheard" not in actions and "rebuild" not in actions, verdicts
    assert tracker.unheard_in_a_row == 0


def test_a_reply_lag_is_measured_during_speech_and_reported_each_minute(
    rig: Rig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration: replies' first push to sound on the monitor, a fraction of a second, then the minute line.

    In the trixie container (REACHY_PLAYBACK_TEST_SINK=sdk) it is the app's SpeakerMonitor
    on pulsesink's monitor source, as on the robot (measured there: about 0.3 s).
    """
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a healthy stream"))
    rig.start()
    stop = threading.Event()
    try:
        tracker, verdicts = _drive_reply_tracker(rig, stop)
        for level in (0.4, 0.5, 0.45):
            rig.push_reply(0.8, level)
            time.sleep(2.0)
        lags = _wait_lags(verdicts, 3, timeout=5.0)
    finally:
        stop.set()
    print(f"\nmeasured lags: {lags}")
    assert len(lags) == 3, verdicts
    low = 0.1 if USE_SDK_DEVICES else 0.0
    assert all(low <= lag <= 0.5 for lag in lags), lags
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        ph.check_playback(rig.pipeline, replies=tracker)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Playback latency")]
    print(line)
    assert len(line) == 1 and f"last reply heard after {lags[-1]:.2f} s" in line[0], line


@pytest.mark.skipif(not USE_SDK_DEVICES, reason="pulsesink and its monitor: the trixie container or the robot")
def test_a_reply_heard_over_5_s_late_is_logged_and_never_rebuilds(rig: Rig, caplog: pytest.LogCaptureFixture) -> None:
    """Integration: a reply heard 6 s late is measured and logged as heard late; the speaker is not rebuilt.

    The lag is pulsesink's ts-offset of 6 s: it writes 6 s ahead of the server's read
    pointer, the robot's 65 s case in small, and nothing stalls. ringbuffer.delay()
    reads 0 throughout, and the latency query returns the configured 124 ms: neither
    sees the lag.
    """
    Gst = rig.Gst
    rig.start()
    stop = threading.Event()
    try:
        tracker, verdicts = _drive_reply_tracker(rig, stop)
        with caplog.at_level(logging.INFO, logger=ph.logger.name):
            rig.push_reply(0.8, 0.4)
            assert len(_wait_lags(verdicts, 1, timeout=5.0)) == 1, verdicts
            time.sleep(2.0)
            rig.speaker.set_property("ts-offset", 6 * Gst.SECOND)
            rig.push_reply(0.8, 0.5)
            lags = _wait_lags(verdicts, 2, timeout=12.0)
            time.sleep(1.0)
        lines = [r.getMessage() for r in caplog.records if r.name == ph.logger.name]
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=ph.logger.name):
            ph.check_playback(rig.pipeline, replies=tracker)
    finally:
        stop.set()
    minute = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Playback latency")]
    print(f"\nmeasured lags: {lags}")
    print(f"verdicts: {verdicts}")
    print(f"minute line: {minute}")
    assert len(lags) >= 2 and 5.5 < lags[1] < 7.0, lags
    late = [v.cause for v in verdicts if v.action == "heard_late"]
    assert len(late) == 1 and f"heard {lags[1]:.1f} s after its first push, over the 5.0 s bound" in late[0], late
    assert not [m for m in lines if "ebuil" in m], lines
    assert len(minute) == 1 and f"last reply heard after {lags[1]:.2f} s" in minute[0], minute


# --- both triggers against the behind case (the trixie container) ---------------------------

# The trigger expected to fire: "unheard replies" or "capture gap".
BEHIND = os.environ.get("REACHY_PLAYBACK_TEST_BEHIND", "")


@pytest.mark.skipif(not (USE_SDK_DEVICES and BEHIND), reason="the trixie container's behind-case reproduction only")
def test_the_triggers_recover_the_behind_case(rig: Rig, caplog: pytest.LogCaptureFixture) -> None:
    """Both triggers live, the speaker desynced behind the server's read pointer, replies heard again.

    Run on the microphone's clock (6522fb7 reverted), where a stall leaves pulsesink
    writing behind the server's read pointer and every reply is silent while the
    latency reads 0. REACHY_PLAYBACK_TEST_STALL=micfeed starves the pipe-fed
    microphone (frames keep coming: the unheard trigger is the one that can see it);
    =pipewire stops the daemon (frames stop: the capture-gap trigger). "Heard" for the
    assertion is the rig's own monitor; the triggers use the app's SpeakerMonitor.
    """
    rig.start()
    acted = _drive_capture_watch(rig)
    stop = threading.Event()
    try:
        tracker, verdicts = _drive_reply_tracker(rig, stop)
        with caplog.at_level(logging.INFO, logger=ph.logger.name):
            before = rig.reply_latency(0.3)
            assert before is not None and before < 1.0, f"baseline reply latency {before}"
            time.sleep(5.0)
            assert tracker.armed, verdicts
            rig.stall(STALL_KIND, STALL_S)
            stalled_at = time.monotonic()
            rebuilt_at = None
            latencies: list[tuple[float, float | None]] = []
            levels = [round(0.35 + 0.01 * i, 2) for i in range(60)]
            while time.monotonic() - stalled_at < RECOVERY_S:
                lat = rig.reply_latency(levels[len(latencies) % len(levels)])
                latencies.append((round(time.monotonic() - stalled_at, 1), None if lat is None else round(lat, 2)))
                time.sleep(1.0)  # a pause: each reply is its own
                rebuilds = [r for r in caplog.records if "Rebuilding the speaker stream" in r.getMessage()]
                if rebuilds and rebuilt_at is None:
                    rebuilt_at = time.monotonic() - stalled_at
                heard = [lat for t, lat in latencies if rebuilt_at is not None and t > rebuilt_at]
                if len(heard) >= 3 and all(lat is not None and lat < 1.0 for lat in heard[-3:]):
                    break
    finally:
        stop.set()
    messages = [r.getMessage() for r in caplog.records if r.name == ph.logger.name]
    rebuild_lines = [m for m in messages if "Rebuilding the speaker stream" in m]
    print(f"\nstall {STALL_KIND} {STALL_S:.0f}s; rebuilds: {rebuild_lines}")
    print(f"capture watch: {[v.action for v in acted]}; reply verdicts: {[v.action for v in verdicts]}")
    print(f"replies after the stall (s after it, push-to-heard s): {latencies}")
    print(f"first rebuild {rebuilt_at} s after the stall")
    assert any(f"trigger: {BEHIND}" in m for m in rebuild_lines), rebuild_lines
    after = [lat for t, lat in latencies if rebuilt_at is not None and t > rebuilt_at]
    assert len(after) >= 3 and all(lat is not None and lat < 1.0 for lat in after[-3:]), latencies


# --- the motif playbin's teardown race ---------------------------------------------------


def test_a_sound_started_during_a_motif_teardown_is_not_orphaned(monkeypatch: pytest.MonkeyPatch) -> None:
    """watch_motif's check-and-clear and play_motif's start are one under MOTIF_LOCK.

    The review (2026-10-15): between the watcher's identity check and its
    ``_playbin = None``, a new sound could take ``_playbin``; the watcher then dropped
    the new playbin's handle, and nothing could stop or tear it down.
    """
    Gst = _gst()
    ended = Gst.Message.new_eos(None)
    bus = SimpleNamespace(timed_pop_filtered=lambda timeout, types: ended)

    class Playbin:
        def __init__(self, name: str) -> None:
            self.name = name
            self.states: list[Any] = []

        def get_bus(self) -> Any:
            return bus

        def set_state(self, state: Any) -> None:
            self.states.append(state)

    old, new = Playbin("old"), Playbin("new")
    started = threading.Event()

    class Audio:
        reads = 0

        def __init__(self) -> None:
            self.value: Any = old

        @property
        def _playbin(self) -> Any:
            Audio.reads += 1
            value = self.value
            if Audio.reads == 2:  # the watcher's identity check at the end: a new sound starts right after it
                threading.Thread(target=ph.play_motif, args=(media, self, "new.wav"), daemon=True).start()
                started.wait(0.2)  # without the lock, play_motif runs to the end here
                time.sleep(0.1)
            return value

        @_playbin.setter
        def _playbin(self, value: Any) -> None:
            self.value = value

    audio = Audio()

    def play_sound(path: str) -> None:
        started.set()
        audio.value = new

    media = SimpleNamespace(play_sound=play_sound)
    ph.watch_motif(audio, old, "old.wav").join(timeout=3.0)
    deadline = time.monotonic() + 2.0
    while audio.value is not new and time.monotonic() < deadline:
        time.sleep(0.01)
    assert audio.value is new  # the new sound kept its handle
    assert old.states == [Gst.State.NULL] and new.states == []


def test_play_motif_returns_the_new_playbin() -> None:
    """The sound route starts the SDK's player under MOTIF_LOCK and watches what it made."""
    audio = SimpleNamespace(_playbin=None)
    media = SimpleNamespace(play_sound=lambda path: setattr(audio, "_playbin", f"playbin for {path}"))
    assert ph.play_motif(media, audio, "ring.wav") == "playbin for ring.wav"


# --- capture off the speaker's clock -------------------------------------------------------


def test_the_prime_is_still_needed_with_the_microphone_off_the_clock(clocked_rig: Rig) -> None:
    """sync=False does not replace the prime: unprimed, the speaker cannot preroll and nothing runs.

    The trixie container, 2026-10-15, microphone sync=False and no prime: the pipeline
    stayed PAUSED and the microphone delivered 0 buffers/s until the first reply.
    """
    clocked_rig.start(prime=False)
    assert clocked_rig.mic_rate(1.0) == 0
    ph.prime_speaker(clocked_rig.audio)
    assert clocked_rig.mic_rate(1.0) > 5


def test_the_microphone_appsink_takes_frames_off_the_clock_and_the_wobbler_stays_on_it(rig: Rig) -> None:
    """The capture sink never waits on the pipeline clock; the playback-side wobbler still does."""
    rig.start()
    assert rig.audio._appsink_audio.get_property("sync") is False
    wobblers = [
        e
        for e in ph._iterate(rig.pipeline.iterate_recurse())
        if e.get_factory() is not None
        and e.get_factory().get_name() == "appsink"
        and e is not rig.audio._appsink_audio
    ]
    assert wobblers and all(w.get_property("sync") is True for w in wobblers)


def test_free_capture_from_clock_without_an_appsink_does_nothing() -> None:
    """The WebRTC backend and stubs have no capture appsink."""
    assert ph.free_capture_from_clock(None) is False
    assert ph.free_capture_from_clock(object()) is False


def test_the_microphone_delivers_while_the_speaker_stream_is_dead(clocked_rig: Rig) -> None:
    """The review's critical (2026-10-15): clocked from the speaker, a synced microphone stops with it.

    A dead speaker stream stops the speaker's clock, and a capture sink that waited on
    it delivered nothing; only a pushed reply restarted it, and a deaf microphone never
    produces one. The trixie container, speaker sink NULL for 5 s: 0 buffers/s with
    sync=True, and still 0 after; 100/s throughout with sync=False.
    """
    rig = clocked_rig
    rig.start()
    assert rig.mic_rate(1.0) > 5
    rig.kill_speaker()
    killed_at = time.monotonic()
    try:
        time.sleep(3.0)
        gap = rig.mic_gaps(killed_at)
    finally:
        rig.revive_speaker()
    assert gap < 0.5, f"the microphone stopped for {gap:.1f} s while the speaker stream was dead"
    lat = rig.reply_latency(0.6)
    assert lat is not None and lat < 3.0, lat


def test_the_microphone_delivers_through_a_rebuild_with_no_reply(clocked_rig: Rig) -> None:
    """A rebuild sets the speaker sink to NULL, which stops its clock; capture goes on regardless."""
    rig = clocked_rig
    rig.start()
    assert rig.mic_rate(1.0) > 5
    since = time.monotonic()
    assert ph.GOVERNOR.rebuild(rig.audio, "test", "a rebuild with no reply after it")
    time.sleep(3.0)  # no reply pushed
    gap = rig.mic_gaps(since)
    assert gap < 0.5, f"the microphone stopped for {gap:.1f} s after a rebuild with no reply"
    lat = rig.reply_latency(0.6)
    assert lat is not None and lat < 1.0, lat


def test_the_microphone_delivers_through_a_long_idle(clocked_rig: Rig) -> None:
    """No reply for REACHY_PLAYBACK_TEST_IDLE_S (15 s here; 300 s in the container run): frames never stop."""
    rig = clocked_rig
    rig.start()
    since = time.monotonic()
    time.sleep(IDLE_S)
    gap = rig.mic_gaps(since)
    assert gap < 0.5, f"the microphone stopped for {gap:.1f} s in a {IDLE_S:.0f} s idle"
    assert rig.mic_rate(1.0) > 5
    lat = rig.reply_latency(0.6)
    assert lat is not None and lat < 1.0, lat


def test_a_rebuild_and_a_capture_stall_prime_the_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The speaker's clock runs again only once its stream has audio: both push 20 ms of silence."""
    primed: list[Any] = []
    audio = SimpleNamespace(_pipeline=_Pipeline(), push_audio_sample=lambda s: None)
    monkeypatch.setattr(ph, "prime_speaker", primed.append)
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: "sink")
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: True)
    assert ph.RebuildGovernor().rebuild(audio, "test", "test")
    ph.act_on_capture(ph.capture_stall_decision(3.5, False), audio)
    assert primed == [audio, audio]
    assert ph.restart_speaker_clock(SimpleNamespace()) is False  # no player to push to


# --- a pipeline stuck in PAUSED (review, 2026-10-16) ---------------------------------------


def test_the_minute_check_warns_about_a_pipeline_stuck_in_paused_and_primes_the_speaker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Capture off is NULL; the app never rests in PAUSED. Clocked from the speaker, PAUSED is the unprimed wedge."""
    primed: list[Any] = []
    audio = SimpleNamespace(push_audio_sample=lambda s: None)
    monkeypatch.setattr(ph, "prime_speaker", primed.append)
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a paused pipeline"))
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        verdict = ph.check_playback(_Pipeline(state="PAUSED"), audio=audio)
    assert verdict.action == "paused"
    assert primed == [audio]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "PAUSED" in warnings[0] and "priming" in warnings[0]


def test_the_minute_check_stays_silent_for_capture_off(caplog: pytest.LogCaptureFixture) -> None:
    """NULL is capture off, the normal case: no line, no prime."""
    audio = SimpleNamespace(push_audio_sample=lambda s: pytest.fail("primed a stopped pipeline"))
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        assert ph.check_playback(_Pipeline(state="NULL"), audio=audio).action == "unmeasured"
    assert not [r for r in caplog.records if r.name == ph.logger.name]


def test_a_rebuild_skipped_on_a_paused_pipeline_names_the_state(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The governor still leaves a pipeline that is not PLAYING alone, but says it is PAUSED."""
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("rebuilt a paused pipeline"))
    audio = SimpleNamespace(_pipeline=_Pipeline(state="PAUSED"))
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        assert ph.RebuildGovernor(clock=lambda: 100.0).rebuild(audio, "capture gap", "test") is False
    assert any("PAUSED" in r.getMessage() for r in caplog.records)


# --- the speaker monitor's restarts are bounded (review, 2026-10-16) -----------------------


class _Monitor(ph.SpeakerMonitor):
    """A SpeakerMonitor whose start records the attempt and never builds a pipeline."""

    def __init__(self, ok: bool = True) -> None:
        super().__init__(lambda now, seconds, speech: None)
        self.ok = ok
        self.starts: list[float] = []
        self.clock = 0.0

    def start(self, source: str) -> bool:
        self.starts.append(self.clock)
        self.buffers = 0
        if not self.ok:
            self.source = None
            return False
        self.source = source
        self.last_buffer_at = self.clock
        return True

    def deliver(self, now: float) -> None:
        self.buffers += 1
        self.last_buffer_at = now


def _keep(monitor: _Monitor, now: float) -> str:
    monitor.clock = now
    return ph.keep_monitor(monitor, _MONITORED_AUDIO, now)


_MONITORED_AUDIO = SimpleNamespace(_pipeline=_Pipeline())


@pytest.fixture
def monitored(monkeypatch: pytest.MonkeyPatch) -> None:
    """One speaker device with a monitor; any rebuild fails the test."""
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: "sink")
    monkeypatch.setattr(ph.SpeakerMonitor, "monitor_source", staticmethod(lambda sink: "speaker.monitor"))
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: pytest.fail("the monitor rebuilt the speaker"))
    monkeypatch.setattr(ph.GOVERNOR, "rebuild", lambda *a: pytest.fail("the monitor asked for a rebuild"))


@pytest.mark.parametrize(
    ("fruitless", "wait_s"),
    [(0, 5.0), (1, 10.0), (2, 20.0), (3, 40.0), (4, 60.0), (10, 60.0)],
)
def test_the_monitor_retry_doubles_and_is_capped(fruitless: int, wait_s: float) -> None:
    """5 s, doubling, at most 60 s: back within a minute of a PipeWire restart (22-35 s)."""
    assert ph.monitor_retry_s(fruitless) == wait_s


@pytest.mark.usefixtures("monitored")
def test_a_monitor_that_never_delivers_backs_off_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """A monitor that delivers nothing is restarted 5, 10, 20, 40, then every 60 s, with one WARNING."""
    monitor = _Monitor()
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        for t in range(0, 400):
            _keep(monitor, float(t))
    assert monitor.starts[:7] == [0.0, 6.0, 16.0, 36.0, 76.0, 136.0, 196.0]
    assert len(monitor.starts) == 10  # then 256, 316, 376
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "backing off" in warnings[0] and "speaker.monitor" in warnings[0]


@pytest.mark.usefixtures("monitored")
def test_a_monitor_that_cannot_start_backs_off_too() -> None:
    """A start that fails is a fruitless try too, never one a second."""
    monitor = _Monitor(ok=False)
    for t in range(0, 100):
        _keep(monitor, float(t))
    assert len(monitor.starts) <= 5


@pytest.mark.usefixtures("monitored")
def test_a_delivery_ends_the_back_off_and_the_next_quiet_spell_warns_again(caplog: pytest.LogCaptureFixture) -> None:
    """A monitor that delivers again is restarted at once when it next goes quiet, and warns once more."""
    monitor = _Monitor()
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        for t in range(0, 40):
            _keep(monitor, float(t))
        monitor.deliver(40.0)
        for t in range(40, 46):
            assert _keep(monitor, float(t)) == "running"
            monitor.deliver(float(t))
        starts = len(monitor.starts)
        for t in range(46, 80):  # quiet again: restarted once after 5 s, then backed off with a new WARNING
            _keep(monitor, float(t))
    assert monitor.starts[starts] == 51.0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


@pytest.mark.usefixtures("monitored")
def test_a_new_speaker_device_restarts_the_monitor_at_once() -> None:
    """The back-off is per device: a new one is started on the next check."""
    monitor = _Monitor()
    for t in range(0, 20):
        _keep(monitor, float(t))
    before = len(monitor.starts)
    ph.SpeakerMonitor.monitor_source = staticmethod(lambda sink: "other.monitor")  # undone by monkeypatch
    _keep(monitor, 20.0)
    assert len(monitor.starts) == before + 1 and monitor.source == "other.monitor"
