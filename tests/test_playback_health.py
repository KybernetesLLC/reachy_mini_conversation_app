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


class Rig:
    """The SDK's GStreamerAudio, built as the robot builds it, plus a way to hear it."""

    def __init__(self) -> None:
        """Build the SDK's layout; nothing runs yet."""
        Gst = _gst()
        self.Gst = Gst
        audio_gstreamer = pytest.importorskip("reachy_mini.media.audio_gstreamer")
        self.rendered: list[tuple[float, float]] = []  # (monotonic, peak) heard
        self.mic_buffers: list[float] = []
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
                mic = Gst.parse_bin_from_description(
                    "audiotestsrc is-live=true wave=silence name=mic ! queue ! audioconvert ! audioresample "
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
        self.mic = self._first("pulsesrc") or self._first("audiotestsrc", name="mic")
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

    def start(self) -> None:
        """Start as the app does: the speaker's clock, PLAYING, then the prime."""
        Gst = self.Gst
        clocked = ph.clock_playback_from_speaker(self.pipeline, self.speaker)  # as the app does, before PLAYING
        if USE_SDK_DEVICES and self.speaker.get_factory().get_name() == "pulsesink":
            assert clocked, "the speaker's clock was not applied"  # a silent no-op passed once
        self.pipeline.set_state(Gst.State.PLAYING)
        if clocked:
            ph.prime_speaker(self.audio)
        self.pipeline.get_state(5 * Gst.SECOND)
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

    def _pull_mic(self) -> None:
        Gst = self.Gst
        while not self._stop.is_set():
            sample = self.audio._appsink_audio.emit("try-pull-sample", 20 * Gst.MSECOND)
            if sample is not None:
                self.mic_buffers.append(time.monotonic())

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


def test_the_speaker_sink_is_found_in_the_sdk_layout(rig: Rig) -> None:
    """The playback branch's sink, not the microphone's or the wobbler's appsink."""
    factory = rig.speaker.get_factory().get_name()
    if USE_SDK_DEVICES:
        assert factory not in ("appsink", "fakesink")
    else:
        assert factory == "fakesink"


def test_the_microphone_delivers_from_the_start_before_any_reply(rig: Rig) -> None:
    """Clocked from the speaker, nothing ran until the speaker's stream had audio.

    The trixie container, 2026-10-15: with pulsesink's clock chosen before PLAYING and
    no priming, the pipeline sat in PAUSED and the microphone delivered 0 buffers until
    the first reply was pushed. prime_speaker's 20 ms of silence starts it.
    """
    rig.start()
    assert rig.pipeline.get_state(0)[1] == rig.Gst.State.PLAYING
    assert rig.mic_rate(1.0) > 5, "the microphone delivered nothing before the first reply"


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
    assert ph.latency_bounded(ph.measure_latency_s(rig.speaker))
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


def test_the_watchdog_rebuilds_the_speaker_when_latency_runs_away(
    rig: Rig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration: a measured 65.5 s logs a WARNING, rebuilds the speaker, and checks the invariant."""
    rig.start()
    readings = iter([65.5, 0.03])
    monkeypatch.setattr(ph, "measure_latency_s", lambda sink: next(readings))
    rebuilt: list[Any] = []
    real_rebuild = ph.rebuild_sink
    monkeypatch.setattr(ph, "rebuild_sink", lambda sink: rebuilt.append(sink) or real_rebuild(sink))

    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        verdict = ph.check_playback(rig.pipeline, audio=rig.audio)

    assert verdict.action == "rebuild"
    assert rebuilt == [rig.speaker]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Playback latency" in m and "65.5" in m for m in infos)
    assert len(warnings) == 1 and "65.5" in warnings[0] and "trigger: playback latency" in warnings[0]
    assert any("bounded" in m and "0.03" in m for m in infos), infos  # the invariant, checked after
    lat = rig.reply_latency(0.6)
    assert lat is not None and lat < 3.0


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

    def __init__(self, playing: bool = True) -> None:
        self.playing = playing

    def get_state(self, _timeout: int) -> tuple[Any, Any, Any]:
        Gst = _gst()
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
