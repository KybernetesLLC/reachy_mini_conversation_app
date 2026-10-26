"""The phantom-speech detector (2026-10-26).

At 05:22 the speaker replayed speech from a conversation at 20:03 the night before:
the app logged no push and the head did not move. The detector logs any such
episode: the speaker's monitor loud while the app pushed nothing and no motif
played, with every sink input, whether the app's own speech sink carried sound,
and whether the head wobbler fired. RMS numbers only; no audio is kept.
"""

from __future__ import annotations
import json
import time
import logging
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from reachy_mini_conversation_app import playback_health as ph


def _gst() -> Any:
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def _monitor(det: ph.PhantomDetector, start: float, end: float, loud: bool) -> None:
    """Feed the detector 0.1 s monitor buffers from ``start`` to ``end``."""
    step = 0.1
    t = start
    while t < end - 1e-9:
        t = round(t + step, 6)
        det.monitored(t, step, step if loud else 0.0)


def _kinds(det: ph.PhantomDetector) -> list[str]:
    return [e.kind for e in det.take()]


# --- the detector: pure, times passed in ---------------------------------------------------


def test_a_loud_monitor_with_nothing_pushed_is_one_episode_with_its_start_and_end() -> None:
    """Half a second of sound with nothing pushed opens an episode; two quiet seconds close it."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
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
    det = ph.PhantomDetector(motif=lambda: (False, None))
    _monitor(det, 10.0, 10.3, loud=True)
    _monitor(det, 10.3, 14.0, loud=False)
    _monitor(det, 14.0, 14.3, loud=True)
    _monitor(det, 14.3, 18.0, loud=False)
    assert det.take() == []


def test_the_apps_own_reply_is_never_an_episode() -> None:
    """Sound while the app's reply plays is the reply."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
    det.pushed(10.0, 2.0)
    _monitor(det, 10.0, 12.2, loud=True)
    _monitor(det, 12.2, 20.0, loud=False)
    assert det.take() == []


def test_audio_queued_faster_than_real_time_is_still_the_apps_while_it_plays() -> None:
    """Six seconds pushed in half a second plays until 16 s: the push 5 s ago is no phantom."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
    for i in range(6):
        det.pushed(10.0 + 0.1 * i, 1.0)
    _monitor(det, 10.2, 16.2, loud=True)
    _monitor(det, 16.2, 22.0, loud=False)
    assert det.take() == []


def test_a_barge_in_drops_the_queued_estimate() -> None:
    """After a barge-in the dropped audio no longer counts as playing."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
    det.pushed(10.0, 6.0)
    det.flushed(11.0)
    _monitor(det, 11.0, 14.5, loud=False)
    _monitor(det, 14.5, 15.5, loud=True)
    assert _kinds(det) == ["started"]


def test_a_motif_playing_or_just_ended_is_not_an_episode() -> None:
    """A motif, and its tail for PHANTOM_QUIET_S, is the app's own sound."""
    state = {"motif": (True, None)}
    det = ph.PhantomDetector(motif=lambda: state["motif"])
    _monitor(det, 20.0, 21.0, loud=True)
    state["motif"] = (False, 21.0)
    _monitor(det, 21.0, 22.0, loud=True)  # the motif's tail
    _monitor(det, 22.0, 23.0, loud=False)
    assert det.take() == []
    _monitor(det, 24.0, 25.0, loud=True)
    assert _kinds(det) == ["started"]


def test_the_app_pushing_ends_an_episode() -> None:
    """A push during an episode ends it there."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
    _monitor(det, 100.0, 101.0, loud=True)
    det.pushed(101.0, 1.0)
    _monitor(det, 101.0, 101.5, loud=True)
    events = det.take()
    assert [e.kind for e in events] == ["started", "ended"]
    assert events[1].episode.end == pytest.approx(101.0)
    assert "pushed" in events[1].episode.end_cause


def test_at_most_one_reported_episode_a_minute() -> None:
    """Episodes within PHANTOM_REPORT_EVERY_S of a reported one are not reported."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
    reports = []
    for start in (100.0, 130.0, 165.0):
        _monitor(det, start, start + 1.0, loud=True)
        _monitor(det, start + 1.0, start + 4.0, loud=False)
        reports += [e.episode.report for e in det.take() if e.kind == "started"]
    assert reports == [True, False, True]


def test_the_last_push_age_is_kept_on_the_episode() -> None:
    """The episode says how long before it the app last pushed."""
    det = ph.PhantomDetector(motif=lambda: (False, None))
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
    inputs = ph.sink_inputs_from_pw_dump(ROBOT_DUMP, own_pid=1330)
    assert [(i["node"], i["serial"], i["rate"], i["pid"], i["ours"]) for i in inputs] == [
        (108, 1347, 16000, 1330, True),
        (128, 1830, 22050, 1330, True),
        (140, 1900, 48000, 1173, False),
    ]
    assert inputs[0]["maxlength"] == 4194304 and inputs[2]["maxlength"] == 64000


def test_the_sink_inputs_line_names_each_node() -> None:
    """The WARNING's listing names every node, the app's marked."""
    line = ph.describe_sink_inputs(ph.sink_inputs_from_pw_dump(ROBOT_DUMP, own_pid=1330))
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
    (entry,) = ph.sink_inputs_from_pactl(PACTL_JSON, own_pid=1330)
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

    monkeypatch.setattr(ph.subprocess, "run", run)
    inputs, source = ph.list_sink_inputs(own_pid=1330)
    assert source == "pactl" and inputs[0]["node"] == 108
    assert [c[0] for c in calls] == ["pw-dump", "pactl"]

    def broken(cmd: list[str], **kwargs: Any) -> Any:
        if cmd[0] == "pw-dump":
            raise subprocess.TimeoutExpired(cmd, 5)
        return SimpleNamespace(returncode=0, stdout="not json")

    monkeypatch.setattr(ph.subprocess, "run", broken)
    inputs, source = ph.list_sink_inputs(own_pid=1330)
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
    got = ph.buffer_rms(data, fmt)
    assert got == (None if rms is None else pytest.approx(rms, rel=1e-3))


def test_the_pad_level_summarises_a_window() -> None:
    """Buffers in the window, those above LOUD_RMS, the peak and the format."""
    level = ph.PadLevel()
    for i in range(10):
        level.record(100.0 + 0.1 * i, 0.2 if i < 5 else 0.0, "F32LE")
    level.record(200.0, 0.9, "F32LE")
    summary = level.summary(100.0, 101.0)
    assert (summary.buffers, summary.loud) == (10, 5)
    assert summary.max_rms == pytest.approx(0.2)
    assert summary.fmt == "F32LE"
    assert level.summary(300.0, 301.0).buffers == 0


def test_a_pad_probe_measures_buffers_reaching_a_real_sink() -> None:
    """A real pipeline: every buffer reaching the sink is measured, by one probe."""
    Gst = _gst()
    pipeline = Gst.parse_launch(
        "audiotestsrc num-buffers=20 wave=sine volume=0.5 ! "
        "audio/x-raw,format=F32LE,rate=16000,channels=1 ! fakesink name=sink sync=false"
    )
    level = ph.PadLevel()
    sink = pipeline.get_by_name("sink")
    assert level.attach(sink)
    assert level.attach(sink)  # the same sink again: one probe, not two
    pipeline.set_state(Gst.State.PLAYING)
    done = Gst.MessageType.EOS | Gst.MessageType.ERROR
    pipeline.get_bus().timed_pop_filtered(5 * Gst.SECOND, done)
    pipeline.set_state(Gst.State.NULL)
    summary = level.summary(0.0, time.monotonic() + 1.0)
    assert summary.buffers == 20 and summary.loud == 20
    assert summary.max_rms == pytest.approx(0.5 / 2**0.5, rel=0.05)


def test_a_failing_measurement_never_reaches_the_streaming_thread() -> None:
    """The probe returns OK whatever goes wrong inside it."""
    level = ph.PadLevel()
    info = SimpleNamespace(get_buffer=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    pad = SimpleNamespace(get_current_caps=lambda: None)
    Gst = _gst()
    assert level._on_buffer(pad, info) == Gst.PadProbeReturn.OK


# --- the head wobbler: did it fire ----------------------------------------------------------


def test_the_wobbler_watch_counts_nonzero_offsets_and_passes_them_on() -> None:
    """Each nonzero head offset is counted and still applied; a new wobbler is watched too."""
    applied: list[tuple[float, ...]] = []
    wobbler = SimpleNamespace(_apply_offsets=applied.append)
    audio = SimpleNamespace(_head_wobbler=wobbler)
    watch = ph.WobblerWatch(clock=lambda: 50.0)
    assert watch.keep(audio) is True
    assert watch.keep(audio) is True  # wrapped once, not twice
    wobbler._apply_offsets((0.0,) * 6)
    wobbler._apply_offsets((0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert applied == [(0.0,) * 6, (0.001, 0.0, 0.0, 0.0, 0.0, 0.0)]
    assert watch.fired(49.0, 51.0) == 1
    assert watch.fired(51.0, 52.0) == 0
    newer = SimpleNamespace(_apply_offsets=applied.append)
    audio._head_wobbler = newer
    watch.keep(audio)
    newer._apply_offsets((0.0, 0.002, 0.0, 0.0, 0.0, 0.0))
    assert watch.fired(49.0, 51.0) == 2
    assert ph.WobblerWatch().keep(SimpleNamespace(_head_wobbler=None)) is False


# --- motifs: open and closed, from the fork's own watcher --------------------------------


def test_watch_motif_marks_the_motif_active_until_it_ends() -> None:
    """The fork's motif watcher tells the detector a motif is playing, then that it closed."""
    Gst = _gst()
    ended = Gst.Message.new_eos(None)
    popped = []

    def pop(timeout: Any, types: Any) -> Any:
        popped.append(ph.MOTIFS.state()[0])
        return ended

    bus = SimpleNamespace(timed_pop_filtered=pop)
    playbin = SimpleNamespace(get_bus=lambda: bus, set_state=lambda s: None)
    audio = SimpleNamespace(_playbin=playbin)
    ph.watch_motif(audio, playbin, "ring.wav").join(timeout=3.0)
    assert popped == [True]  # active while it played
    active, closed_at = ph.MOTIFS.state()
    assert active is False and closed_at is not None


# --- the watch: the detector, the probes and the report, as the console runs them -------


class _Fakes:
    def __init__(self) -> None:
        self.pad = ph.PadLevel()
        self.wobbler = ph.WobblerWatch()
        self.snapshots = 0

    def list_inputs(self) -> tuple[list[dict[str, Any]], str]:
        self.snapshots += 1
        return ph.sink_inputs_from_pw_dump(ROBOT_DUMP, own_pid=1330), "pw-dump"


def test_an_episode_is_reported_once_with_the_sink_inputs_the_pad_and_the_wobbler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One WARNING at the episode's end, with what was listed at its start."""
    fakes = _Fakes()
    watch = ph.PhantomWatch(
        motif=lambda: (False, None),
        list_inputs=fakes.list_inputs,
        pad=fakes.pad,
        wobbler=fakes.wobbler,
    )
    watch.pushed(10.0, 1.0)
    for i in range(20):
        fakes.pad.record(100.0 + 0.1 * i, 0.0, "F32LE")  # the app's sink: silence only
    _monitor(watch.detector, 100.0, 102.0, loud=True)
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
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
    assert "speech sink pad: 20 buffers, 0 loud" in line
    assert "head wobbler fired 0 times" in line
    assert "no motif playbin" in line
    assert fakes.snapshots == 1


def test_a_second_episode_within_the_minute_is_one_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second episode in the minute is one INFO line, with no listing."""
    fakes = _Fakes()
    watch = ph.PhantomWatch(
        motif=lambda: (False, None),
        list_inputs=fakes.list_inputs,
        pad=fakes.pad,
        wobbler=fakes.wobbler,
    )
    audio = SimpleNamespace(_playbin=None)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
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

    watch = ph.PhantomWatch(motif=lambda: (False, None), list_inputs=broken)
    _monitor(watch.detector, 100.0, 101.0, loud=True)
    _monitor(watch.detector, 101.0, 104.0, loud=False)
    with caplog.at_level(logging.INFO, logger=ph.logger.name):
        for event in watch.take():
            watch.act(event, SimpleNamespace(_playbin=None))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "sink inputs unavailable" in warnings[0]


def test_the_monitor_callback_never_raises_into_gstreamer() -> None:
    """The monitor's callback swallows a failure."""
    watch = ph.PhantomWatch(motif=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    watch.monitored(1.0, 0.1, 0.1)  # no exception
