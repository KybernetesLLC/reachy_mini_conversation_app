"""The phantom-speech detector (2026-10-26).

At 05:22 the speaker replayed speech from a conversation at 20:03 the night before:
the app logged no push and the head did not move. The detector logs any such
episode: the speaker's monitor loud while the app pushed nothing and no motif
played, with every sink input, whether the app's own speech sink carried sound,
RMS numbers only; no audio is kept.
"""

from __future__ import annotations
import re
import json
import time
import logging
import threading
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini_conversation_app import phantom_speech as ps
from reachy_mini_conversation_app import playback_health as ph


def _gst() -> Any:
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _monitor(det: ps.PhantomDetector, start: float, end: float, loud: bool) -> None:
    """Feed the detector 0.1 s monitor buffers from ``start`` to ``end``."""
    step = 0.1
    t = start
    while t < end - 1e-9:
        t = round(t + step, 6)
        det.monitored(t, step, step if loud else 0.0)


def _kinds(det: ps.PhantomDetector) -> list[str]:
    return [e.kind for e in det.take()]


# --- the detector: pure, times passed in ---------------------------------------------------


def test_a_loud_monitor_with_nothing_pushed_is_one_episode_with_its_start_and_end() -> None:
    """Half a second of sound with nothing pushed opens an episode; two quiet seconds close it."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    _monitor(det, 100.0, 100.4, loud=True)
    assert det.take() == []  # 0.4 s loud: not yet
    _monitor(det, 100.4, 103.0, loud=True)
    events = det.take()
    assert [e.kind for e in events] == ["started"]
    assert events[0].episode.start == pytest.approx(100.0)
    _monitor(det, 103.0, 106.0, loud=False)
    events = det.take()
    assert [e.kind for e in events] == ["ended"]
    episode = events[0].episode
    assert (episode.start, episode.end) == (pytest.approx(100.0), pytest.approx(103.0))
    assert episode.loud_s == pytest.approx(3.0)
    assert episode.last_push_age_s is None  # nothing was ever pushed


def test_a_blip_under_half_a_second_is_not_an_episode() -> None:
    """Short sounds far apart never add up to an episode."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    _monitor(det, 10.0, 10.3, loud=True)
    _monitor(det, 10.3, 14.0, loud=False)
    _monitor(det, 14.0, 14.3, loud=True)
    _monitor(det, 14.3, 18.0, loud=False)
    assert det.take() == []


def test_the_apps_own_reply_is_never_an_episode() -> None:
    """Sound while the app's reply plays is the reply."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    det.pushed(10.0, 2.0)
    _monitor(det, 10.0, 12.2, loud=True)
    _monitor(det, 12.2, 20.0, loud=False)
    assert det.take() == []


def test_audio_queued_faster_than_real_time_is_still_the_apps_while_it_plays() -> None:
    """Six seconds pushed in half a second plays until 16 s: the push 5 s ago is no phantom."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    for i in range(6):
        det.pushed(10.0 + 0.1 * i, 1.0)
    _monitor(det, 10.2, 16.2, loud=True)
    _monitor(det, 16.2, 22.0, loud=False)
    assert det.take() == []


def test_a_barge_in_drops_the_queued_estimate() -> None:
    """After a barge-in the dropped audio no longer counts as playing."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    det.pushed(10.0, 6.0)
    det.flushed(11.0)
    _monitor(det, 11.0, 14.5, loud=False)
    _monitor(det, 14.5, 15.5, loud=True)
    assert _kinds(det) == ["started"]


def test_a_motif_playing_or_just_ended_is_not_an_episode() -> None:
    """A motif, and its tail for PHANTOM_QUIET_S, is the app's own sound."""
    state = {"motif": (True, None)}
    det = ps.PhantomDetector(sound=lambda: state["motif"])
    _monitor(det, 20.0, 21.0, loud=True)
    state["motif"] = (False, 21.0)
    _monitor(det, 21.0, 22.0, loud=True)  # the motif's tail
    _monitor(det, 22.0, 23.0, loud=False)
    assert det.take() == []
    _monitor(det, 24.0, 25.0, loud=True)
    assert _kinds(det) == ["started"]


def test_the_app_pushing_ends_an_episode() -> None:
    """A push during an episode ends it there."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    _monitor(det, 100.0, 101.0, loud=True)
    det.pushed(101.0, 1.0)
    _monitor(det, 101.0, 101.5, loud=True)
    events = det.take()
    assert [e.kind for e in events] == ["started", "ended"]
    assert events[1].episode.end == pytest.approx(101.0)
    assert "pushed" in events[1].episode.end_cause


def test_the_last_push_age_is_kept_on_the_episode() -> None:
    """The episode says how long before it the app last pushed."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    det.pushed(50.0, 1.0)
    _monitor(det, 100.0, 101.0, loud=True)
    (event,) = det.take()
    assert event.episode.last_push_age_s == pytest.approx(50.0, abs=0.2)


# --- the sink inputs: pw-dump, or pactl ---------------------------------------------------


def _node(node_id: int, cls: str, pid: int, rate: int, serial: int, **extra: Any) -> dict[str, Any]:
    props = {
        "media.class": cls,
        "application.process.id": pid,
        "application.process.binary": "python3.12",
        "application.name": "python",
        "media.name": "Playback Stream",
        "object.serial": serial,
        "node.rate": f"1/{rate}",
        "pulse.attr.maxlength": 4194304,
        "pulse.corked": False,
    }
    props.update(extra)
    return {
        "id": node_id,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "running",
            "props": props,
            "params": {"Format": [{"format": "F32LE", "rate": rate, "channels": 1}]},
        },
    }


ROBOT_DUMP = [
    {"id": 0, "type": "PipeWire:Interface:Core", "info": {}},
    _node(90, "Stream/Input/Audio", 1173, 44100, 109),  # the daemon's microphone
    _node(108, "Stream/Output/Audio", 1330, 16000, 1347),  # the app's speech
    _node(128, "Stream/Output/Audio", 1330, 22050, 1830),  # the app's playbin
    _node(140, "Stream/Output/Audio", 1173, 48000, 1900, **{"pulse.attr.maxlength": 64000}),
]


def test_the_sink_inputs_are_the_output_streams_with_the_apps_marked() -> None:
    """Playback streams only, each with its node, serial, rate and client, the app's marked."""
    inputs = ps.sink_inputs_from_pw_dump(ROBOT_DUMP, own_pid=1330)
    assert [(i["node"], i["serial"], i["rate"], i["pid"], i["ours"]) for i in inputs] == [
        (108, 1347, 16000, 1330, True),
        (128, 1830, 22050, 1330, True),
        (140, 1900, 48000, 1173, False),
    ]
    assert inputs[0]["maxlength"] == 4194304 and inputs[2]["maxlength"] == 64000


def test_the_sink_inputs_line_names_each_node() -> None:
    """The WARNING's listing names every node, the app's marked."""
    line = ps.describe_sink_inputs(ps.sink_inputs_from_pw_dump(ROBOT_DUMP, own_pid=1330))
    assert "node 108 serial 1347 16000 Hz pid 1330 (this app)" in line
    assert "node 140 serial 1900 48000 Hz pid 1173" in line
    assert "maxlength 4194304" in line


PACTL_JSON = [
    {
        "index": 108,
        "corked": False,
        "sample_specification": "float32le 1ch 16000Hz",
        "properties": {
            "object.serial": "1347",
            "application.process.id": "1330",
            "application.process.binary": "python3.12",
            "media.name": "Playback Stream",
        },
    }
]


def test_pactl_sink_inputs_are_read_the_same_way() -> None:
    """Where pactl is installed, its JSON gives the same fields."""
    (entry,) = ps.sink_inputs_from_pactl(PACTL_JSON, own_pid=1330)
    assert (entry["node"], entry["serial"], entry["rate"], entry["pid"], entry["ours"]) == (
        108,
        1347,
        16000,
        1330,
        True,
    )


def test_listing_the_sink_inputs_uses_pw_dump_then_pactl_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pw-dump first, pactl second, each with a timeout; a failure is a reason, not an exception."""
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        assert kwargs.get("timeout")  # never waits forever
        if cmd[0] == "pw-dump":
            raise FileNotFoundError("pw-dump")
        return SimpleNamespace(returncode=0, stdout=json.dumps(PACTL_JSON))

    monkeypatch.setattr(ps.subprocess, "run", run)
    inputs, source = ps.list_sink_inputs(own_pid=1330)
    assert source == "pactl" and inputs[0]["node"] == 108
    assert [c[0] for c in calls] == ["pw-dump", "pactl"]

    def broken(cmd: list[str], **kwargs: Any) -> Any:
        if cmd[0] == "pw-dump":
            raise subprocess.TimeoutExpired(cmd, 5)
        return SimpleNamespace(returncode=0, stdout="not json")

    monkeypatch.setattr(ps.subprocess, "run", broken)
    inputs, source = ps.list_sink_inputs(own_pid=1330)
    assert inputs == [] and "unavailable" in source


# --- the speech sink's pad: RMS of what reaches it ----------------------------------------


@pytest.mark.parametrize(
    ("fmt", "samples", "rms"),
    [
        ("F32LE", [0.5, -0.5, 0.5, -0.5], 0.5),
        ("S16LE", [16384, -16384], 0.5),
        ("F32LE", [], None),
        ("U8", [1, 2, 3], None),
    ],
)
def test_buffer_rms_reads_the_pad_format(fmt: str, samples: list[float], rms: float | None) -> None:
    """RMS of F32LE and S16LE buffers; None for anything else."""
    import numpy as np

    dtype = {"F32LE": "<f4", "S16LE": "<i2", "U8": "u1"}[fmt]
    data = np.asarray(samples, dtype=dtype).tobytes()
    got = ps.buffer_rms(data, fmt)
    assert got == (None if rms is None else pytest.approx(rms, rel=1e-3))


def test_the_pad_level_summarises_a_window() -> None:
    """Buffers in the window, those above LOUD_RMS, the peak and the format."""
    level = ps.PadLevel()
    for i in range(10):
        level._record(100.0 + 0.1 * i, 0.2 if i < 5 else 0.0, "F32LE")
    level._record(200.0, 0.9, "F32LE")
    assert level.summary(100.0, 101.0) == "10 buffers, 5 loud, max RMS 0.2000 (F32LE)"
    assert level.summary(300.0, 301.0) == "no buffers"


def test_a_pad_probe_measures_buffers_reaching_a_real_sink() -> None:
    """A real pipeline: every buffer reaching the sink is measured, by one probe."""
    Gst = _gst()
    pipeline = Gst.parse_launch(
        "audiotestsrc num-buffers=20 wave=sine volume=0.5 ! "
        "audio/x-raw,format=F32LE,rate=16000,channels=1 ! fakesink name=sink sync=false"
    )
    level = ps.PadLevel()
    sink = pipeline.get_by_name("sink")
    assert level.attach(sink)
    assert level.attach(sink)  # the same sink again: one probe, not two
    pipeline.set_state(Gst.State.PLAYING)
    done = Gst.MessageType.EOS | Gst.MessageType.ERROR
    pipeline.get_bus().timed_pop_filtered(5 * Gst.SECOND, done)
    pipeline.set_state(Gst.State.NULL)
    summary = level.summary(0.0, time.monotonic() + 1.0)
    found = re.fullmatch(r"20 buffers, 20 loud, max RMS ([\d.]+) \(F32LE\)", summary)
    assert found and float(found.group(1)) == pytest.approx(0.5 / 2**0.5, rel=0.05)


def test_a_failing_measurement_never_reaches_the_streaming_thread() -> None:
    """The probe returns OK whatever goes wrong inside it."""
    level = ps.PadLevel()
    info = SimpleNamespace(get_buffer=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    pad = SimpleNamespace(get_current_caps=lambda: None)
    Gst = _gst()
    assert level._on_probe(pad, info) == Gst.PadProbeReturn.OK


# --- motifs: open and closed, from the fork's own watcher --------------------------------


def test_watch_sound_marks_the_sound_open_until_it_ends() -> None:
    """watch_sound tells the detector a sound is playing, then that it closed."""
    Gst = _gst()
    ended = Gst.Message.new_eos(None)
    popped = []

    def pop(timeout: Any, types: Any) -> Any:
        popped.append(ph.SOUNDS.state()[0])
        return ended

    bus = SimpleNamespace(timed_pop_filtered=pop)
    playbin = SimpleNamespace(get_bus=lambda: bus, set_state=lambda s: None, query_duration=None)
    audio = SimpleNamespace(_playbin=playbin)
    ph.watch_sound(audio, playbin, "ring.wav").join(timeout=3.0)
    assert popped == [True]  # open while it played
    active, closed_at = ph.SOUNDS.state()
    assert active is False and closed_at is not None


# --- the watch: the detector, the probes and the report, as the console runs them -------


class _Fakes:
    def __init__(self) -> None:
        self.pad = ps.PadLevel()
        self.snapshots = 0

    def list_inputs(self) -> tuple[list[dict[str, Any]], str]:
        self.snapshots += 1
        return ps.sink_inputs_from_pw_dump(ROBOT_DUMP, own_pid=1330), "pw-dump"


def test_an_episode_is_reported_once_with_the_sink_inputs_and_the_pad(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One WARNING at the episode's end, with what was listed at its start."""
    _gst()
    fakes = _Fakes()
    fakes.pad = ps.PadLevel(clock=lambda: 99.5)
    fakes.pad.attach(SimpleNamespace(get_static_pad=lambda name: _FakePad()))
    watch = ps.PhantomWatch(
        sound=lambda: (False, None),
        list_inputs=fakes.list_inputs,
        pad=fakes.pad,
    )
    watch.pushed(10.0, 1.0)
    for i in range(20):
        fakes.pad._record(100.0 + 0.1 * i, 0.0, "F32LE")  # the app's sink: silence only
    _monitor(watch.detector, 100.0, 102.0, loud=True)
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        for event in watch.take():
            watch.act(event, audio)
        _monitor(watch.detector, 102.0, 105.0, loud=False)
        for event in watch.take():
            watch.act(event, audio)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    line = warnings[0]
    assert "Phantom speech" in line
    assert "2.0 s loud" in line
    assert "last push 90 s before" in line
    assert "node 108 serial 1347 16000 Hz pid 1330 (this app)" in line
    assert "speech sink pad, probed from " in line and ": 20 buffers, 0 loud" in line
    assert "wobbler" not in line
    assert "and no sound;" in line and "no sound playbin" in line and "motif" not in line
    assert fakes.snapshots == 1


def test_a_second_episode_within_the_minute_is_one_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second episode in the minute is one INFO line, with no listing."""
    fakes = _Fakes()
    watch = ps.PhantomWatch(
        sound=lambda: (False, None),
        list_inputs=fakes.list_inputs,
        pad=fakes.pad,
    )
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        for start in (100.0, 120.0):
            _monitor(watch.detector, start, start + 1.0, loud=True)
            _monitor(watch.detector, start + 1.0, start + 4.0, loud=False)
            for event in watch.take():
                watch.act(event, audio)
    levels = [r.levelno for r in caplog.records if "hantom" in r.getMessage()]
    assert levels == [logging.WARNING, logging.INFO]
    assert fakes.snapshots == 1


def test_a_report_that_fails_is_logged_and_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    """A listing that fails still leaves the WARNING, saying why."""

    def broken() -> Any:
        raise RuntimeError("no pw-dump today")

    watch = ps.PhantomWatch(sound=lambda: (False, None), list_inputs=broken)
    _monitor(watch.detector, 100.0, 101.0, loud=True)
    _monitor(watch.detector, 101.0, 104.0, loud=False)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        for event in watch.take():
            watch.act(event, SimpleNamespace(_playbin=None))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "sink inputs unavailable" in warnings[0]


def test_the_monitor_callback_never_raises_into_gstreamer() -> None:
    """The monitor's callback swallows a failure."""
    watch = ps.PhantomWatch(sound=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    watch.monitored(1.0, 0.1, 0.1)  # no exception


def test_an_episode_ended_before_its_listing_ran_still_lists_the_sink_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each event runs in its own worker: an end handled first lists the sink inputs itself."""
    fakes = _Fakes()
    watch = ps.PhantomWatch(sound=lambda: (False, None), list_inputs=fakes.list_inputs)
    _monitor(watch.detector, 100.0, 101.0, loud=True)
    watch.pushed(101.0, 1.0)  # ends the episode: started and ended come out together
    started, ended = watch.take()
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        watch.act(ended, audio)
        watch.act(started, audio)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "node 108 serial 1347 16000 Hz pid 1330 (this app)" in warnings[0]
    assert fakes.snapshots == 1


class _Playbin:
    """The SDK's play_sound playbin, as the app tools' music and express sounds leave it."""

    def __init__(self, position_s: float, duration_s: float) -> None:
        self.position_s, self.duration_s = position_s, duration_s

    def get_name(self) -> str:
        return "player"

    def get_state(self, timeout: int) -> Any:
        Gst = _gst()
        return (Gst.StateChangeReturn.SUCCESS, Gst.State.PLAYING, Gst.State.VOID_PENDING)

    def query_position(self, fmt: Any) -> tuple[bool, int]:
        return True, int(self.position_s * _gst().SECOND)

    def query_duration(self, fmt: Any) -> tuple[bool, int]:
        return True, int(self.duration_s * _gst().SECOND)


@pytest.mark.parametrize(("open_", "warnings"), [(True, 0), (False, 1)])
def test_a_sound_open_is_the_apps_own_and_a_playbin_left_behind_is_no_excuse(
    caplog: pytest.LogCaptureFixture, open_: bool, warnings: int
) -> None:
    """An open sound opens no episode and logs nothing; a playbin left behind is no excuse."""
    watch = ps.PhantomWatch(sound=lambda: (open_, None), list_inputs=_Fakes().list_inputs)
    audio = SimpleNamespace(_playbin=_Playbin(12.0, 47.5))
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        _monitor(watch.detector, 100.0, 101.0, loud=True)
        for event in watch.take():
            watch.act(event, audio)
        _monitor(watch.detector, 101.0, 104.0, loud=False)
        for event in watch.take():
            watch.act(event, audio)
    lines = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == ps.logger.name]
    assert len(lines) == warnings
    assert all(level == logging.WARNING and "Phantom speech" in m for level, m in lines)


# --- fix round 1 (the review, 2026-10-26) --------------------------------------------------


def _run(watch: ps.PhantomWatch, audio: Any) -> None:
    for event in watch.take():
        watch.act(event, audio)


def _episode(watch: ps.PhantomWatch, start: float, audio: Any) -> None:
    """One second of sound from ``start`` and three of quiet, acted on as the console does."""
    _monitor(watch.detector, start, start + 1.0, loud=True)
    _run(watch, audio)
    _monitor(watch.detector, start + 1.0, start + 4.0, loud=False)
    _run(watch, audio)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_at_most_one_phantom_warning_a_minute(caplog: pytest.LogCaptureFixture) -> None:
    """Episodes within PHANTOM_REPORT_EVERY_S of a WARNING are INFO lines."""
    fakes = _Fakes()
    watch = ps.PhantomWatch(sound=lambda: (False, None), list_inputs=fakes.list_inputs)
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        for start in (100.0, 130.0, 165.0):
            _episode(watch, start, audio)
    assert len(_warnings(caplog)) == 2
    assert fakes.snapshots == 2


def test_an_app_sound_does_not_use_up_the_minutes_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sound opens no episode, so the minute is free for the phantom 20 s later."""
    state = {"sound": (True, None)}
    watch = ps.PhantomWatch(sound=lambda: state["sound"], list_inputs=_Fakes().list_inputs)
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        _monitor(watch.detector, 100.0, 101.0, loud=True)
        state["sound"] = (False, 101.0)
        _monitor(watch.detector, 101.0, 104.0, loud=False)
        _run(watch, audio)
        _episode(watch, 120.0, audio)
    assert len(_warnings(caplog)) == 1


def test_a_short_sound_closed_before_the_report_runs_is_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 1.2 s express motif: closed at its end, and its tail is the app's for PHANTOM_QUIET_S."""
    state = {"sound": (True, None)}
    watch = ps.PhantomWatch(sound=lambda: state["sound"], list_inputs=_Fakes().list_inputs)
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        _monitor(watch.detector, 100.1, 101.3, loud=True)
        state["sound"] = (False, 101.3)
        _monitor(watch.detector, 101.3, 101.6, loud=True)  # the sink's last buffers after the close
        _monitor(watch.detector, 101.6, 104.0, loud=False)
        _run(watch, audio)
    assert [r for r in caplog.records if r.name == ps.logger.name] == []


class _SdkAudio:
    """The SDK's play_sound as it treats _playbin: the old one to NULL, a new one each call."""

    def __init__(self, make: Any) -> None:
        self._playbin: Any = None
        self.make = make
        self.calls: list[str] = []

    def play_sound(self, sound_file: str) -> None:
        self.calls.append(sound_file)
        if self._playbin is not None:
            self._playbin.set_state(_gst().State.NULL)
        self._playbin = self.make(sound_file)


class _SoundPlaybin:
    """A play_sound playbin whose bus says EOS after ``pops`` polls; it records its states."""

    def __init__(self, name: str, pops: int = 2, duration_s: float = 0.3) -> None:
        self.name, self.pops, self.duration_s = name, pops, duration_s
        self.states: list[Any] = []

    def get_bus(self) -> Any:
        return self

    def timed_pop_filtered(self, timeout: Any, types: Any) -> Any:
        self.pops -= 1
        if self.pops <= 0:
            return _gst().Message.new_eos(None)
        time.sleep(0.01)  # a real bus waits out its timeout; never a busy loop
        return None

    def query_duration(self, fmt: Any) -> tuple[bool, int]:
        return True, int(self.duration_s * _gst().SECOND)

    def set_state(self, state: Any) -> None:
        self.states.append(state)


def _wait_until(check: Any, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not check() and time.monotonic() < deadline:
        time.sleep(0.01)
    return bool(check())


@pytest.mark.parametrize("caller", ["conversation.sound", "app tools", "play_move"])
def test_every_play_sound_caller_is_closed_once_at_its_end(caller: str) -> None:
    """The motif route, a direct app_tools-style call and play_move's MediaManager path.

    Each sound plays once, closes once and leaves no stream behind: its playbin goes to
    NULL exactly once, audio._playbin is cleared, and no sound is left open.
    """
    from reachy_mini.media.media_manager import MediaManager

    Gst = _gst()
    made: list[_SoundPlaybin] = []
    audio = _SdkAudio(lambda path: made.append(_SoundPlaybin(path)) or made[-1])
    sounds = ps.SoundStarts()
    sounds.listeners.append(ph.watch_sound)
    assert sounds.install(audio)
    media = SimpleNamespace(audio=audio, logger=ps.logger)  # MediaManager.play_sound's self
    if caller == "app tools":
        audio.play_sound("express.wav")
    else:  # conversation.sound's media.play_sound and ReachyMini.play_move both reach this
        MediaManager.play_sound(media, "ring.wav")  # type: ignore[arg-type]
    assert len(made) == 1 and len(audio.calls) == 1
    assert _wait_until(lambda: made[0].states == [Gst.State.NULL])
    assert _wait_until(lambda: ph.SOUNDS.state()[0] is False)
    assert audio._playbin is None
    time.sleep(0.3)
    assert made[0].states == [Gst.State.NULL]  # torn down once


def test_a_sound_replaced_mid_play_is_torn_down_by_the_sdk_not_twice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A newer sound's play_sound sets the old playbin to NULL; its watcher leaves it."""
    Gst = _gst()
    made: list[_SoundPlaybin] = []

    def make(path: str) -> _SoundPlaybin:
        made.append(_SoundPlaybin(path, pops=3 if made else 10**9))  # the first never ends
        return made[-1]

    audio = _SdkAudio(make)
    sounds = ps.SoundStarts()
    sounds.listeners.append(ph.watch_sound)
    sounds.install(audio)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        audio.play_sound("song.wav")
        time.sleep(0.3)
        audio.play_sound("ring.wav")
        assert _wait_until(lambda: made[1].states == [Gst.State.NULL])
        assert _wait_until(lambda: ph.SOUNDS.state()[0] is False)
        time.sleep(0.3)
    assert made[0].states == [Gst.State.NULL]  # by the SDK, once
    assert audio._playbin is None
    closes = [r.getMessage() for r in caplog.records if "Sound stream closed" in r.getMessage()]
    assert closes == [
        "Sound stream closed (replaced by a newer sound): song.wav",
        "Sound stream closed (the sound ended, torn down): ring.wav",
    ]


def test_the_wrapper_calls_the_original_once_holding_the_sound_lock() -> None:
    """No recursion and no re-entry: SOUND_LOCK taken once, the original run inside it."""
    held: list[bool] = []
    audio = SimpleNamespace(_playbin=None)

    def original(path: str) -> str:
        held.append(ph.SOUND_LOCK.locked())
        audio._playbin = f"playbin for {path}"
        return "done"

    audio.play_sound = original
    sounds = ps.SoundStarts()
    listened: list[bool] = []
    sounds.listeners.append(lambda a, playbin, path: listened.append(ph.SOUND_LOCK.locked()))
    assert sounds.install(audio) and sounds.install(audio)  # wrapped once, not twice
    caller = threading.Thread(target=lambda: held.append(audio.play_sound("x.wav") == "done"))
    caller.start()
    caller.join(timeout=2.0)
    assert not caller.is_alive()  # a re-entered plain Lock would hang here
    assert held == [True, True] and listened == [True]
    assert ps.SoundStarts().install(SimpleNamespace()) is False


def test_the_sound_wrapper_tells_its_listeners_of_a_new_playbin_only() -> None:
    """A call that made no playbin (the SDK could not) hands the old one to nobody."""
    audio = SimpleNamespace(_playbin=None)
    audio.play_sound = lambda path: setattr(audio, "_playbin", f"playbin for {path}")
    sounds = ps.SoundStarts()
    heard: list[tuple[str, Any]] = []
    sounds.listeners.append(lambda a, playbin, path: heard.append((path, playbin)))
    sounds.install(audio)
    audio.play_sound("x.wav")
    audio.play_sound = lambda path: None  # the SDK failing to make a playbin
    audio._playbin = None  # the first sound was torn down at its end
    sounds.install(audio)
    audio.play_sound("y.wav")
    assert heard == [("x.wav", "playbin for x.wav")]


def test_a_sound_with_no_end_is_torn_down_past_its_duration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A watcher never holds a sound open for good: no EOS by its duration and slack, it goes."""
    Gst = _gst()
    monkeypatch.setattr(ph, "SOUND_SLACK_S", 0.2)
    playbin = _SoundPlaybin("stuck.wav", pops=10**9, duration_s=0.1)
    audio = SimpleNamespace(_playbin=playbin)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        watcher = ph.watch_sound(audio, playbin, "stuck.wav")
        watcher.join(timeout=3.0)
    assert not watcher.is_alive()
    assert playbin.states == [Gst.State.NULL] and audio._playbin is None
    assert ph.SOUNDS.state()[0] is False
    late = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert late == ["Sound stream closed (no end 0 s after it started, torn down): stuck.wav"]


class _FakePad:
    """A pad that counts its probes."""

    def __init__(self) -> None:
        self.probes: dict[int, Any] = {}
        self.next = 0

    def add_probe(self, mask: Any, callback: Any) -> int:
        self.next += 1
        self.probes[self.next] = callback
        return self.next

    def remove_probe(self, probe: int) -> None:
        del self.probes[probe]

    def get_current_caps(self) -> Any:
        return None


def _probe_rig(monkeypatch: pytest.MonkeyPatch, playing: list[bool]) -> tuple[Any, _FakePad]:
    pad = _FakePad()
    sink = SimpleNamespace(get_static_pad=lambda name: pad)
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: sink)
    monkeypatch.setattr(ph, "_playing", lambda pipeline: playing[0])
    return SimpleNamespace(_pipeline="p", _playbin=None), pad


def test_the_probe_is_on_only_while_the_app_is_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A push removes it before the reply's buffers flow; a later suspicion puts it back."""
    _gst()
    audio, pad = _probe_rig(monkeypatch, [True])
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    _suspect(watch, 99.7, 100.0)  # loud with nothing ever pushed: suspected
    watch.keep(audio, 100.0)
    assert len(pad.probes) == 1
    watch.pushed(101.0, 2.0)
    assert pad.probes == {}  # removed at the push, before the audio reaches the sink
    _monitor(watch.detector, 103.5, 104.0, loud=True)  # the reply's sound: not suspected
    watch.keep(audio, 104.0)
    assert pad.probes == {}
    _suspect(watch, 106.0, 106.5)  # loud again once the app is quiet (from 106)
    watch.keep(audio, 106.5)
    assert len(pad.probes) == 1


def test_the_probe_comes_off_when_the_pipeline_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture off: the report says "not probed", not "no buffers"."""
    _gst()
    playing = [True]
    audio, pad = _probe_rig(monkeypatch, playing)
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    _suspect(watch, 99.7, 100.0)
    watch.keep(audio, 100.0)
    assert len(pad.probes) == 1
    playing[0] = False
    watch.keep(audio, 101.0)
    assert pad.probes == {} and not watch.pad.attached


def test_a_detached_probe_sees_none_of_a_real_reply() -> None:
    """On a real pipeline: after pushed(), the buffers reach the sink unmeasured."""
    Gst = _gst()
    pipeline = Gst.parse_launch(
        "audiotestsrc num-buffers=20 wave=sine volume=0.5 ! "
        "audio/x-raw,format=F32LE,rate=16000,channels=1 ! fakesink name=sink sync=false"
    )
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    assert watch.pad.attach(pipeline.get_by_name("sink"))
    watch.pushed(time.monotonic(), 1.0)
    pipeline.set_state(Gst.State.PLAYING)
    done = Gst.MessageType.EOS | Gst.MessageType.ERROR
    pipeline.get_bus().timed_pop_filtered(5 * Gst.SECOND, done)
    pipeline.set_state(Gst.State.NULL)
    assert watch.pad.summary(0.0, time.monotonic() + 1.0) == "no buffers"


def test_the_pad_format_is_read_from_caps_events_not_per_buffer() -> None:
    """Each buffer uses the format of the last CAPS event; the pad's caps are not queried."""
    Gst = _gst()
    level = ps.PadLevel()
    caps = Gst.Caps.from_string("audio/x-raw,format=S16LE,rate=16000,channels=1")
    event = Gst.Event.new_caps(caps)
    event_info = SimpleNamespace(type=Gst.PadProbeType.EVENT_DOWNSTREAM, get_event=lambda: event)
    level._on_probe(None, event_info)
    buf = Gst.Buffer.new_wrapped(b"\x00\x40" * 4)  # 16384: RMS 0.5
    pad = SimpleNamespace(get_current_caps=lambda: pytest.fail("caps queried per buffer"))
    info = SimpleNamespace(type=Gst.PadProbeType.BUFFER, get_buffer=lambda: buf)
    assert level._on_probe(pad, info) == Gst.PadProbeReturn.OK
    summary = level.summary(0.0, time.monotonic() + 1.0)
    assert summary == "1 buffers, 1 loud, max RMS 0.5000 (S16LE)"


def test_a_huge_or_undecodable_listing_is_a_reason_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pw-dump output past the cap fails with its size; bad bytes are replaced, not fatal."""
    seen: dict[str, Any] = {}

    def run(cmd: list[str], **kwargs: Any) -> Any:
        seen.update(kwargs)
        if cmd[0] == "pw-dump":
            return SimpleNamespace(returncode=0, stdout="x" * (ps.SINK_INPUTS_MAX_BYTES + 1))
        raise FileNotFoundError("pactl")

    monkeypatch.setattr(ps.subprocess, "run", run)
    inputs, source = ps.list_sink_inputs(own_pid=1)
    assert inputs == [] and "bytes" in source
    assert seen.get("errors") == "replace"


def test_a_listing_that_hangs_is_killed_at_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real child: back within the timeout, reason given, no child left behind."""
    import os

    monkeypatch.setattr(ps, "SINK_INPUT_COMMANDS", [["sleep", "10"]])
    began = time.monotonic()
    inputs, source = ps.list_sink_inputs(own_pid=1, timeout_s=0.2)
    assert time.monotonic() - began < 3.0
    assert inputs == [] and "TimeoutExpired" in source
    pgrep = ["pgrep", "-P", str(os.getpid()), "sleep"]
    left = subprocess.run(pgrep, capture_output=True, text=True)
    assert left.stdout.strip() == ""


# --- re-review (fix round 1) -----------------------------------------------------------------


def test_a_phantom_soon_after_a_short_sound_is_still_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reviewer's case: a 1 s sound, 1.8 s quiet, then 57 s of sound."""
    state = {"sound": (True, None)}
    watch = ps.PhantomWatch(sound=lambda: state["sound"], list_inputs=_Fakes().list_inputs)
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        _monitor(watch.detector, 99.0, 100.0, loud=True)
        state["sound"] = (False, 100.0)
        _run(watch, audio)
        _monitor(watch.detector, 100.0, 101.8, loud=False)
        _monitor(watch.detector, 101.8, 158.8, loud=True)
        _run(watch, audio)
        _monitor(watch.detector, 158.8, 162.0, loud=False)
        _run(watch, audio)
    assert len(_warnings(caplog)) == 1


def test_a_long_song_still_open_is_no_episode(caplog: pytest.LogCaptureFixture) -> None:
    """A sound open covers all of a long song."""
    watch = ps.PhantomWatch(sound=lambda: (True, None), list_inputs=_Fakes().list_inputs)
    audio = SimpleNamespace(_playbin=_Playbin(1.0, 180.0))
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        _monitor(watch.detector, 99.0, 139.0, loud=True)
        _run(watch, audio)
        _monitor(watch.detector, 139.0, 142.0, loud=False)
        _run(watch, audio)
    assert [r for r in caplog.records if r.name == ps.logger.name] == []


def test_a_keep_that_checked_quiet_before_a_push_cannot_attach_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate: keep's quiet check and attach are one step, so a push's detach comes after."""
    _gst()
    audio, pad = _probe_rig(monkeypatch, [True])
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    inside, go = threading.Event(), threading.Event()
    quiet = watch.detector.app_quiet

    def slow_quiet(now: float) -> bool:
        result = quiet(now)
        inside.set()
        go.wait(2.0)  # the push arrives between the check and the attach
        return result

    monkeypatch.setattr(watch.detector, "app_quiet", slow_quiet)
    _suspect(watch, 99.7, 100.0)
    keeper = threading.Thread(target=watch.keep, args=(audio, 100.0))
    keeper.start()
    assert inside.wait(2.0)
    pusher = threading.Thread(target=watch.pushed, args=(100.1, 2.0))
    pusher.start()
    pusher.join(0.2)
    go.set()
    keeper.join(2.0)
    pusher.join(2.0)
    assert pad.probes == {}  # the push's detach came after the attach


# --- /code-review fixes ------------------------------------------------------------------------


def test_a_reply_pushed_but_not_yet_heard_is_never_a_phantom() -> None:
    """A reply the server plays 50 s late is the app's own: ReplyTracker still has it pending."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    tracker.pushed(10.0, 2.0, 2.0)
    det = ps.PhantomDetector(sound=lambda: (False, None), pending=tracker.pending)
    det.pushed(10.0, 2.0)
    _monitor(det, 60.0, 62.0, loud=True)  # the late reply, at last
    assert det.take() == []


def _suspect(watch: ps.PhantomWatch, start: float, end: float) -> None:
    _monitor(watch.detector, start, end, loud=True)


def test_an_idle_stretch_with_a_quiet_monitor_attaches_no_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Python per buffer on the speech sink while nothing is suspected."""
    _gst()
    audio, pad = _probe_rig(monkeypatch, [True])
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    for t in range(100, 400, 1):
        _monitor(watch.detector, float(t), t + 1.0, loud=False)
        watch.keep(audio, t + 1.0)
    assert pad.probes == {}


def test_the_probe_is_on_from_a_suspected_onset_until_the_episode_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On at a suspected onset, off when the episode ends."""
    _gst()
    audio, pad = _probe_rig(monkeypatch, [True])
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    _suspect(watch, 100.0, 100.3)  # loud while quiet: a candidate, not yet an episode
    watch.keep(audio, 100.3)
    assert len(pad.probes) == 1
    _suspect(watch, 100.3, 104.0)
    _monitor(watch.detector, 104.0, 106.5, loud=False)  # the episode ends at 104
    watch.keep(audio, 106.5)
    assert pad.probes == {}


def test_the_probe_comes_off_after_30_s_of_one_suspicion(monkeypatch: pytest.MonkeyPatch) -> None:
    """One suspicion keeps the probe on PROBE_MAX_S at most."""
    _gst()
    audio, pad = _probe_rig(monkeypatch, [True])
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    _suspect(watch, 100.0, 101.0)
    watch.keep(audio, 101.0)
    assert len(pad.probes) == 1
    _suspect(watch, 101.0, 140.0)
    watch.keep(audio, 132.0)
    assert pad.probes == {}
    watch.keep(audio, 140.0)  # the same episode: it stays off
    assert pad.probes == {}


def test_the_report_says_from_when_the_pad_was_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The WARNING says from when the speech sink was measured."""
    _gst()
    audio, pad = _probe_rig(monkeypatch, [True])
    clock = [100.5]
    fakes = _Fakes()
    watch = ps.PhantomWatch(
        sound=lambda: (False, None),
        list_inputs=fakes.list_inputs,
        pad=ps.PadLevel(clock=lambda: clock[0]),
    )
    _suspect(watch, 100.0, 100.5)
    watch.keep(audio, 100.5)
    _suspect(watch, 100.5, 102.0)
    events = watch.take()
    line = watch.report(events[0].episode, audio)
    assert "speech sink pad, probed from " in line


def test_an_episode_closes_when_the_monitor_stops() -> None:
    """Capture off or sleep closes the episode at its last loud buffer."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    _monitor(det, 100.0, 103.0, loud=True)
    det.monitor_stopped(110.0)
    events = det.take()
    assert [e.kind for e in events] == ["started", "ended"]
    assert events[1].episode.end == pytest.approx(103.0)
    assert "monitor" in events[1].episode.end_cause


def test_an_episode_closes_after_120_s_with_no_monitor_buffers() -> None:
    """A monitor that went silent without stopping closes it after MONITOR_GONE_S."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    _monitor(det, 100.0, 103.0, loud=True)
    det.expire(200.0)
    assert [e.kind for e in det.take()] == ["started"]
    det.expire(223.1)
    (ended,) = det.take()
    assert ended.episode.end == pytest.approx(103.0)


def test_an_episode_a_push_closes_ends_at_its_last_loud_buffer() -> None:
    """The pad window ends at the sound, not at the push."""
    det = ps.PhantomDetector(sound=lambda: (False, None))
    _monitor(det, 100.0, 101.0, loud=True)
    _monitor(det, 101.0, 102.5, loud=False)
    det.pushed(102.5, 1.0)
    ended = det.take()[-1]
    assert ended.episode.end == pytest.approx(101.0)


def test_a_missing_speaker_sink_is_never_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sink not found yet is looked up again next time."""
    sinks = [None, "sink"]
    monkeypatch.setattr(ph, "speaker_sink", lambda pipeline: sinks.pop(0))
    monkeypatch.setattr(ph, "_playing", lambda pipeline: True)
    audio = SimpleNamespace(_pipeline="p")
    cache: dict[str, Any] = {}
    assert ph.current_speaker_sink(audio, cache) == (True, None)
    assert ph.current_speaker_sink(audio, cache) == (True, "sink")


def test_play_sound_replaced_on_the_same_audio_is_wrapped_again() -> None:
    """wrap_once runs each tick, so a fresh method is wrapped again."""
    played: list[str] = []
    audio = SimpleNamespace(_playbin=None, _pipeline=None, play_sound=played.append)
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    heard: list[Any] = []
    watch.sounds.listeners.append(lambda a, playbin, path: heard.append(path))
    watch.keep(audio, 1.0)
    audio.play_sound = lambda path: played.append(path) or setattr(audio, "_playbin", path)
    watch.keep(audio, 2.0)
    audio.play_sound("x.wav")
    assert played == ["x.wav"] and heard == ["x.wav"]


def test_the_head_wobbler_is_left_alone() -> None:
    """AGENTS.md: no monkey-patching of SDK internals."""
    apply = print
    audio = SimpleNamespace(_head_wobbler=SimpleNamespace(_apply_offsets=apply), _pipeline=None)
    ps.PhantomWatch(sound=lambda: (False, None)).keep(audio, 1.0)
    assert audio._head_wobbler._apply_offsets is apply


def test_probe_errors_are_counted_and_logged_once_a_minute(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Errors in the probe are counted and logged at most once a minute."""
    Gst = _gst()
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    bad = SimpleNamespace(type=Gst.PadProbeType.BUFFER, get_buffer=lambda: None)
    for _ in range(3):
        assert watch.pad._on_probe(None, bad) == Gst.PadProbeReturn.OK
    audio = SimpleNamespace(_pipeline=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        watch.keep(audio, 100.0)
        watch.keep(audio, 130.0)
        watch.keep(audio, 161.0)
    warnings = _warnings(caplog)
    assert len(warnings) == 1 and "3" in warnings[0] and "probe" in warnings[0]


def test_monitor_errors_are_counted_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    """An error in the monitor callback is counted and logged, never raised."""

    def broken() -> Any:
        raise RuntimeError("boom")

    watch = ps.PhantomWatch(sound=broken)
    watch.monitored(1.0, 0.1, 0.1)  # no exception into GStreamer
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        watch.keep(SimpleNamespace(_pipeline=None), 100.0)
    assert any("monitor" in w and "RuntimeError" in w for w in _warnings(caplog))


# --- re-review of the /code-review fixes ------------------------------------------------------


def test_a_late_reply_opens_no_episode_while_it_plays() -> None:
    """In console order: each monitor buffer goes to the real tracker, then the detector."""
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    det = ps.PhantomDetector(sound=lambda: (False, None), pending=tracker.pending)
    for i in range(20):  # 2 s of reply pushed at 10 s, in 0.1 s chunks
        tracker.pushed(10.0 + 0.01 * i, 0.1, 0.1)
        det.pushed(10.0 + 0.01 * i, 0.1)

    def feed(start: float, end: float, loud: bool) -> None:
        t = start
        while t < end - 1e-9:
            t = round(t + 0.1, 6)
            speech = 0.1 if loud else 0.0
            tracker.monitored(t, 0.1, speech)
            det.monitored(t, 0.1, speech)
            if round(t * 10) % 5 == 0:
                tracker.tick(t)

    feed(10.2, 59.9, loud=False)
    feed(59.9, 61.9, loud=True)  # the reply, heard 50 s late, plays to its end
    feed(61.9, 66.0, loud=False)
    assert det.take() == []


def _console_rig(rebuilt_at: Any = None) -> tuple[ph.ReplyTracker, ps.PhantomDetector]:
    """Build the real tracker and detector, wired as the console wires them."""
    tracker = ph.ReplyTracker(rebuilt_at=rebuilt_at or (lambda: None))
    det = ps.PhantomDetector(sound=lambda: (False, None), pending=tracker.pending)
    return tracker, det


@pytest.mark.parametrize("rate", [1.0, 1.5])
def test_a_late_reply_heard_while_still_pushed_opens_no_episode(rate: float) -> None:
    """A 15 s reply pushed from 10 s at ``rate`` times real time, heard 5 s late (15 to 30 s)."""
    tracker, det = _console_rig()
    pushed = 0.0
    for step in range(1, 261):  # 10.1 s to 36.0 s, in 0.1 s steps, in console order
        t = round(10.0 + 0.1 * step, 6)
        if pushed < 15.0 - 1e-9:
            chunk = min(0.1 * rate, 15.0 - pushed)
            tracker.pushed(t, chunk, chunk)
            det.pushed(t, chunk)
            pushed += chunk
        speech = 0.1 if 15.0 < t <= 30.0 + 1e-9 else 0.0
        tracker.monitored(t, 0.1, speech)
        det.monitored(t, 0.1, speech)
        if step % 5 == 0:
            tracker.tick(t)
    assert det.take() == []


def _late_reply_then_sound(tracker: ph.ReplyTracker, det: ps.PhantomDetector, cut: Any) -> None:
    """Push a 30 s reply at 10 s, heard from 15 s; ``cut(t)`` at 22 s; sound 24 to 30 s."""
    for i in range(300):
        tracker.pushed(10.0 + 0.001 * i, 0.1, 0.1)
        det.pushed(10.0 + 0.001 * i, 0.1)
    for step in range(1, 251):  # 10.1 s to 35.0 s
        t = round(10.0 + 0.1 * step, 6)
        if abs(t - 22.0) < 1e-9:
            cut(t)
        speech = 0.1 if 15.0 < t <= 22.0 + 1e-9 or 24.0 < t <= 30.0 + 1e-9 else 0.0
        tracker.monitored(t, 0.1, speech)
        det.monitored(t, 0.1, speech)


def test_sound_after_a_barge_in_on_a_late_reply_opens_an_episode() -> None:
    """A barge-in cuts the heard reply short, as it cuts the detector's queued audio."""
    tracker, det = _console_rig()

    def barge_in(t: float) -> None:
        tracker.flushed(t)  # console order: the tracker, then the detector
        det.flushed(t)

    _late_reply_then_sound(tracker, det, barge_in)
    started = [e.episode.start for e in det.take() if e.kind == "started"]
    assert len(started) == 1 and 24.0 <= started[0] < 30.0


def test_a_rebuild_cuts_a_heard_reply_short() -> None:
    """A speaker rebuild drops what was queued, so the heard reply stops counting then."""
    rebuilt: list[float] = []
    tracker, det = _console_rig(rebuilt_at=lambda: rebuilt[-1] if rebuilt else None)
    _late_reply_then_sound(tracker, det, rebuilt.append)
    assert tracker.pending(24.9)  # the tail after the rebuild
    assert not tracker.pending(25.1)


def test_the_probe_follows_a_new_speaker_sink_at_the_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline restart mid-suspicion: the probe moves to the new sink."""
    _gst()
    old_pad, new_pad = _FakePad(), _FakePad()
    old = SimpleNamespace(get_static_pad=lambda name: old_pad)
    new = SimpleNamespace(get_static_pad=lambda name: new_pad)
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    _suspect(watch, 99.7, 100.0)
    watch.keep(SimpleNamespace(), 100.0, found=(True, old))
    assert len(old_pad.probes) == 1
    watch.keep(SimpleNamespace(), 101.0, found=(True, new))
    assert old_pad.probes == {} and len(new_pad.probes) == 1


def test_errors_are_logged_at_most_once_a_minute(caplog: pytest.LogCaptureFixture) -> None:
    """Errors between logs are held and logged together a minute after the last."""
    Gst = _gst()
    watch = ps.PhantomWatch(sound=lambda: (False, None))
    bad = SimpleNamespace(type=Gst.PadProbeType.BUFFER, get_buffer=lambda: None)
    audio = SimpleNamespace(_pipeline=None)
    with caplog.at_level(logging.INFO, logger=ps.logger.name):
        for now in (100.0, 130.0, 161.0):
            watch.pad._on_probe(None, bad)
            watch.keep(audio, now)
    warnings = _warnings(caplog)
    assert len(warnings) == 2
    assert "probe 1 " in warnings[0] and "probe 2 " in warnings[1]


def test_keep_monitor_uses_the_sink_it_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """No second lookup (and no second MEDIA_LOCK) when the caller found the sink."""

    def lookup(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("looked the sink up again")

    monkeypatch.setattr(ph, "current_speaker_sink", lookup)
    stopped: list[bool] = []
    monitor = SimpleNamespace(source="dev.monitor", stop=lambda: stopped.append(True))
    assert ph.keep_monitor(monitor, None, 10.0, found=(False, None)) == "stopped"
    assert stopped == [True]
    factory = SimpleNamespace(get_name=lambda: "pulsesink")
    sink = SimpleNamespace(get_factory=lambda: factory, get_property=lambda name: "dev")
    monitor = SimpleNamespace(source="dev.monitor", last_buffer_at=9.0)
    assert ph.keep_monitor(monitor, None, 10.0, found=(True, sink)) == "running"


# --- review round (2026-10-26) ----------------------------------------------------------


def test_a_stop_closes_the_sound_at_once_and_logs_one_close_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """conversation.sound's stop: closed in SOUNDS before the watcher wakes, one close line."""
    Gst = _gst()
    playbin = _SoundPlaybin("ring.wav", pops=10**9)
    audio = SimpleNamespace(_playbin=playbin)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        watcher = ph.watch_sound(audio, playbin, "ring.wav")
        assert ph.SOUNDS.state()[0] is True
        assert ph.stop_sound(audio) is True
        assert ph.SOUNDS.state()[0] is False  # at once, not when the watcher next looks
        watcher.join(timeout=3.0)
    assert not watcher.is_alive()
    assert playbin.states == [Gst.State.NULL] and audio._playbin is None
    closes = [r.getMessage() for r in caplog.records if "Sound stream closed" in r.getMessage()]
    assert closes == ["Sound stream closed (stopped): ring.wav"]


def test_a_sound_playing_before_the_wrap_is_handed_to_the_listeners_once() -> None:
    """The SDK's wake-up sound can play before the wrap is installed: it is watched from then."""
    started = SimpleNamespace(name="wake_up")
    audio = SimpleNamespace(_playbin=started, play_sound=lambda path: None)
    sounds = ps.SoundStarts()
    heard: list[tuple[Any, Any]] = []

    def listen(a: Any, playbin: Any, path: Any) -> None:
        heard.append((playbin, path))
        ph.SOUNDS.opened(playbin)

    sounds.listeners.append(listen)
    try:
        assert sounds.install(audio)
        assert sounds.install(audio)  # already wrapped: not handed over again
        audio.play_sound = lambda path: None  # a fresh method: wrapped again, but already watched
        assert sounds.install(audio)
        assert heard == [(started, "(a sound started before the wrap)")]
    finally:
        ph.SOUNDS.closed(started)
