"""The phantom-speech detector: speaker sound the app did not push.

2026-10-26, 05:22: the speaker replayed speech from a conversation at 20:03 the night
before. The app logged no push and the head did not move; the wobbler is fed from a tee
inside the app's speech pipeline, so the sound most likely never passed through it.
This detector logs any such episode: the speaker's monitor loud for PHANTOM_LOUD_S while
the app pushed nothing (and its queued audio has played out) for PHANTOM_QUIET_S and no
motif played. Sound from the SDK's play_sound (the app tools' music and express sounds
bypass watch_motif) is one INFO line, not phantom speech: a sound started within
SDK_SOUND_S before an episode no longer than that, or a playbin still mid-sound.
One WARNING per episode, at most one every PHANTOM_REPORT_EVERY_S, names
every sink input (pw-dump, or pactl), whether the app's speech sink carried sound in
the episode (a pad probe on it: RMS of each buffer), and whether the head wobbler fired.
Nothing is kept but RMS numbers and times. Nothing runs on the event loop. The pad probe
runs Python on the speaker's streaming thread, which must wait for the GIL, so it is on
only while the app is quiet: put on by keep() once nothing of the app's has played for
PHANTOM_QUIET_S, taken off by pushed() before a reply's audio is pushed. Any wait then
falls on silence or on the phantom itself, never on a reply. The listing and the report
run in a worker thread.
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
import datetime
import threading
import subprocess
from typing import Any, Callable, Sequence, NamedTuple
from collections import deque
from dataclasses import dataclass

import numpy as np

from reachy_mini_conversation_app import playback_health
from reachy_mini_conversation_app.playback_health import MOTIFS, LOUD_RMS, rms


logger = logging.getLogger(__name__)

PHANTOM_LOUD_S = 0.5  # monitor speech, within one candidate, that opens an episode
PHANTOM_QUIET_S = 3.0  # the app pushed nothing, its queued audio and any motif ended, this long ago
PHANTOM_END_S = 2.0  # a quiet monitor for this long closes an episode
PHANTOM_REPORT_EVERY_S = 60.0  # at most one WARNING in this long
SDK_SOUND_S = 3.0  # sound this soon after a play_sound call is that sound (motifs run 0.4-1.9 s)
SINK_INPUTS_TIMEOUT_S = 5.0
SINK_INPUTS_MAX_BYTES = 8 * 1024 * 1024  # pw-dump printed 342 KB on the robot (2026-10-26)
SINK_INPUT_COMMANDS = [["pw-dump"], ["pactl", "-f", "json", "list", "sink-inputs"]]
Row = dict[str, Any]  # one sink input
SinkInputs = tuple[list[Row], str]  # (the sink inputs, where they were read from)


@dataclass
class PhantomEpisode:
    """One stretch of speaker sound the app did not push. Times are monotonic seconds."""

    start: float
    last_loud: float
    loud_s: float
    last_push_age_s: float | None
    end: float | None = None
    end_cause: str = ""
    inputs: SinkInputs | None = None
    sound_started: str | None = None  # a play_sound call just before the episode
    sound_playing: str | None = None  # the SDK's playbin mid-sound when it was checked
    sdk_checked: bool = False

    def sdk_sound(self) -> str | None:
        """Name the SDK sound that accounts for the episode, or None.

        A start time just before only accounts for a short episode: a phantom beginning
        within PHANTOM_END_S of a short sound's end is folded into its episode, and must
        still be warned of. A playbin still mid-sound accounts for a long song.
        """
        if self.sound_playing is not None:
            return self.sound_playing
        if self.sound_started is not None and self.loud_s <= SDK_SOUND_S:
            return self.sound_started
        return None

    @property
    def end_or_last(self) -> float:
        """The episode's end, or its last loud moment while it is open."""
        return self.end if self.end is not None else self.last_loud


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

    def app_quiet(self, now: float) -> bool:
        """Return whether no sound of the app's own played for PHANTOM_QUIET_S."""
        with self._lock:
            return self._app_quiet(now)

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
        """Note a speaker-monitor buffer arriving at ``now``, from GStreamer's thread.

        Never raises: it is the monitor's streaming thread that calls it.
        """
        if speech_s <= 0.0 and self._episode is None:
            return  # silence with no episode open: nothing to do
        try:
            with self._lock:
                self._monitored(now, seconds, speech_s)
        except Exception:
            pass

    def _monitored(self, now: float, seconds: float, speech_s: float) -> None:
        """Note a monitor buffer. Holds the lock."""
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
        age = None if self._last_push is None else start - self._last_push
        self._episode = PhantomEpisode(start, now, loud_s, age)
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


def _sink_input(node: Any, props: Row, rate: int | None, own_pid: int, **rest: Any) -> Row:
    """Build one sink-input row from a stream's properties; both listings use it."""
    pid = _int(props.get("application.process.id"))
    return {
        "node": node,
        "serial": _int(props.get("object.serial")),
        "rate": rate,
        "pid": pid,
        "ours": pid is not None and pid == own_pid,
        "binary": props.get("application.process.binary") or props.get("application.name"),
        "media": props.get("media.name"),
        "maxlength": rest.get("maxlength"),
        "state": rest.get("state"),
        "corked": bool(rest.get("corked")),
    }


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
        found.append(
            _sink_input(
                obj.get("id"),
                props,
                rate,
                own_pid,
                maxlength=_int(props.get("pulse.attr.maxlength")),
                state=info.get("state"),
                corked=props.get("pulse.corked") in (True, "true"),
            )
        )
    return found


def sink_inputs_from_pactl(entries: Sequence[Any], own_pid: int) -> list[dict[str, Any]]:
    """Return the sink inputs of ``pactl -f json list sink-inputs``, the app's own marked. Pure."""
    found = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rate = re.search(r"(\d+)\s*Hz", str(entry.get("sample_specification", "")))
        found.append(
            _sink_input(
                entry.get("index"),
                entry.get("properties") or {},
                int(rate.group(1)) if rate else None,
                own_pid,
                corked=entry.get("corked"),
            )
        )
    return found


def list_sink_inputs(own_pid: int, timeout_s: float = SINK_INPUTS_TIMEOUT_S) -> SinkInputs:
    """List the sink inputs with pw-dump, else pactl; returns (inputs, source). Never raises.

    Runs subprocesses: call it from a worker thread, never the event loop.
    """
    errors = []
    for cmd in SINK_INPUT_COMMANDS:
        parse = sink_inputs_from_pactl if cmd[0] == "pactl" else sink_inputs_from_pw_dump
        try:
            run = {"capture_output": True, "text": True, "errors": "replace"}
            done = subprocess.run(cmd, timeout=timeout_s, **run)
            if done.returncode != 0:
                errors.append(f"{cmd[0]} exited {done.returncode}")
                continue
            if len(done.stdout) > SINK_INPUTS_MAX_BYTES:
                errors.append(f"{cmd[0]} printed {len(done.stdout)} bytes, over the cap")
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
    if fmt == "F32LE":
        return rms(np.frombuffer(data, dtype="<f4", count=len(data) // 4))
    if fmt == "S16LE":
        ints = np.frombuffer(data, dtype="<i2", count=len(data) // 2)
        if ints.size == 0:
            return None
        wide = ints.astype(np.int64)
        return float(np.sqrt(np.dot(wide, wide) / ints.size)) / 32768.0
    return None


class PadLevel:
    """A pad probe on the speaker sink's input, keeping (time, RMS, format) of recent buffers."""

    def __init__(self, keep: int = 12000) -> None:
        """Keep the last ``keep`` buffers: a minute or more at pulsesink's buffer sizes."""
        self._lock = threading.Lock()
        self._buffers: deque[tuple[float, float | None, str]] = deque(maxlen=keep)
        self._probe_lock = threading.Lock()  # the sink, pad and probe below
        self._sink: Any = None
        self._pad: Any = None
        self._probe: Any = None
        self._fmt = "unknown"  # from the pad's caps when attached, then each CAPS event
        self._gst: Any = None  # gi.repository.Gst, bound when attached

    @property
    def attached(self) -> bool:
        """Whether a sink is being measured."""
        return self._sink is not None

    def measures(self, sink: Any) -> bool:
        """Whether the probe is on ``sink``."""
        return sink is not None and sink is self._sink

    def _record(self, now: float, rms_: float | None, fmt: str) -> None:
        """Note one buffer."""
        with self._lock:
            self._buffers.append((now, rms_, fmt))

    def summary(self, start: float, end: float) -> str:
        """Describe the buffers that arrived from ``start`` to ``end``, newest scanned first."""
        with self._lock:
            buffers = list(self._buffers)
        count = loud = 0
        peak: float | None = None
        fmt = None
        for t, level, f in reversed(buffers):
            if t > end:
                continue
            if t < start:
                break
            fmt = fmt or f
            count += 1
            if level is not None:
                loud += level > LOUD_RMS
                peak = level if peak is None else max(peak, level)
        if not count:
            return "no buffers"
        shown = f"{peak:.4f}" if peak is not None else "n/a"
        return f"{count} buffers, {loud} loud, max RMS {shown} ({fmt})"

    def attach(self, sink: Any) -> bool:
        """Measure ``sink``'s input pad; a different sink moves the probe. Returns success.

        Runs in a worker; detach() may run on the event loop at the same time.
        """
        from gi.repository import Gst

        with self._probe_lock:
            if sink is not None and sink is self._sink:
                return True
            self._detach()
            pad = sink.get_static_pad("sink") if sink is not None else None
            if pad is None:
                return False
            caps = pad.get_current_caps()
            self._fmt = _caps_format(caps) if caps is not None else "unknown"
            self._gst = Gst
            mask = Gst.PadProbeType.BUFFER | Gst.PadProbeType.EVENT_DOWNSTREAM
            self._probe = pad.add_probe(mask, self._on_probe)
            self._sink, self._pad = sink, pad
            return True

    def detach(self) -> None:
        """Remove the probe, if any. Never blocks on the streaming thread."""
        with self._probe_lock:
            self._detach()

    def _detach(self) -> None:
        """Remove the probe. Holds the probe lock."""
        if self._pad is not None and self._probe is not None:
            try:
                self._pad.remove_probe(self._probe)
            except Exception:
                logger.debug("Could not remove the speech sink's pad probe", exc_info=True)
        self._sink = self._pad = self._probe = None

    def _on_probe(self, pad: Any, info: Any) -> Any:
        """Record a buffer's RMS, or a new format, on the streaming thread: never raises or logs."""
        Gst = self._gst
        if Gst is None:
            from gi.repository import Gst
        try:
            if info.type & Gst.PadProbeType.BUFFER:
                buf = info.get_buffer()
                level = buffer_rms(buf.extract_dup(0, buf.get_size()), self._fmt)
                self._record(time.monotonic(), level, self._fmt)
            else:
                event = info.get_event()
                if event.type == Gst.EventType.CAPS:
                    self._fmt = _caps_format(event.parse_caps())
        except Exception:
            pass
        return Gst.PadProbeReturn.OK


def _caps_format(caps: Any) -> str:
    """Return the sample format named in audio caps."""
    return str(caps.get_structure(0).get_value("format"))


def wrap_once(obj: Any, name: str, tag: Any, make_wrapper: Callable[[Any], Any]) -> bool:
    """Replace ``obj.name`` with ``make_wrapper(original)``, once per ``tag``.

    The wrapper is built around the original, captured here before wrapping, so it never
    calls itself. Returns whether ``obj.name`` is wrapped.
    """
    original = getattr(obj, name, None)
    if original is None:
        return False
    if getattr(original, "_wrapped_by", None) is tag:
        return True
    wrapper = make_wrapper(original)
    wrapper._wrapped_by = tag
    setattr(obj, name, wrapper)
    return True


class WobblerWatch:
    """Count the head wobbler's moves: nonzero offsets it applies, counted per second."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, keep: int = 3600) -> None:
        """Start with no wobbler watched; ``keep`` seconds of counts are kept."""
        self._clock = clock
        self._seconds: deque[list[int]] = deque(maxlen=keep)  # [second, nonzero offsets]

    def keep(self, audio: Any) -> bool:
        """Watch the SDK's current wobbler, once; a new one is watched too. Returns success."""
        wobbler = getattr(audio, "_head_wobbler", None)
        if wobbler is None:
            return False

        def make(apply: Callable[[Any], Any]) -> Callable[[Any], Any]:
            def counted(offsets: Any) -> Any:
                try:
                    if any(abs(float(v)) > 0.0 for v in offsets):
                        self._count(int(self._clock()))
                except Exception:
                    pass
                return apply(offsets)

            return counted

        return wrap_once(wobbler, "_apply_offsets", self, make)

    def _count(self, second: int) -> None:
        """Add one move at ``second``. Only GLib's main loop calls this, so there is one writer."""
        seconds = self._seconds
        if seconds and seconds[-1][0] == second:
            seconds[-1][1] += 1
        else:
            seconds.append([second, 1])

    def fired(self, start: float, end: float) -> int:
        """Return how many nonzero offsets the wobbler applied from ``start`` to ``end``."""
        total = 0
        for second, count in reversed(list(self._seconds)):
            if second > end:
                continue
            if second < int(start):
                break
            total += count
        return total


class SoundStarts:
    """Wrap the SDK audio's play_sound once, to note when each sound starts.

    Every caller goes through it: MediaManager.play_sound calls audio.play_sound, and so
    do play_motif and the app tools' music and express sounds. The wrapper calls the
    original method, captured before wrapping, never itself: play_motif calls it under
    MOTIF_LOCK. ``listeners`` are called after the original, with (audio, path, playbin),
    for anything that has to follow each sound's playbin.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, keep: int = 1000) -> None:
        """Start with no sound seen."""
        self._clock = clock
        self._lock = threading.Lock()
        self._starts: deque[float] = deque(maxlen=keep)
        self.listeners: list[Callable[[Any, Any, Any], None]] = []

    def install(self, audio: Any) -> bool:
        """Wrap ``audio.play_sound``, once. Returns whether it is wrapped."""

        def make(original: Callable[..., Any]) -> Callable[..., Any]:
            def play_sound(sound_file: Any, *args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    self._starts.append(self._clock())
                result = original(sound_file, *args, **kwargs)
                for listener in list(self.listeners):
                    try:
                        listener(audio, sound_file, getattr(audio, "_playbin", None))
                    except Exception:
                        logger.warning("A play_sound listener failed", exc_info=True)
                return result

            return play_sound

        return wrap_once(audio, "play_sound", self, make)

    def starts(self) -> list[float]:
        """Return the times of the sounds started, oldest first."""
        with self._lock:
            return list(self._starts)

    def last_before(self, t: float) -> float | None:
        """Return the time of the last sound started at or before ``t``, or None."""
        with self._lock:
            before = [s for s in self._starts if s <= t]
        return before[-1] if before else None


def _wall(t: float) -> str:
    """Return a monotonic time as the wall-clock time it was."""
    when = datetime.datetime.now() - datetime.timedelta(seconds=time.monotonic() - t)
    return when.strftime("%H:%M:%S")


class PlaybinStatus(NamedTuple):
    """The SDK's play_sound playbin: name, state, and position and duration in seconds."""

    name: str
    state: Any
    position_s: float | None
    duration_s: float | None


def _playbin_status(audio: Any) -> PlaybinStatus | None:
    """Read the SDK's play_sound playbin, or None when there is none. May raise."""
    playbin = getattr(audio, "_playbin", None)
    if playbin is None:
        return None
    from gi.repository import Gst

    ok_pos, pos = playbin.query_position(Gst.Format.TIME)
    ok_dur, dur = playbin.query_duration(Gst.Format.TIME)
    return PlaybinStatus(
        playbin.get_name(),
        playbin.get_state(0)[1],
        pos / Gst.SECOND if ok_pos else None,
        dur / Gst.SECOND if ok_dur else None,
    )


def _sdk_sound(audio: Any) -> str | None:
    """Describe the SDK's play_sound playbin if it is playing a sound now, else None. Never raises.

    The app tools' music and express sounds use play_sound directly, not watch_motif. A
    playbin left at its end (the SDK never tears it down) is not playing.
    """
    try:
        from gi.repository import Gst

        status = _playbin_status(audio)
        if status is None or status.state != Gst.State.PLAYING:
            return None
        pos, dur = status.position_s, status.duration_s
        if pos is None or dur is None or pos >= dur - 0.1:
            return None
        return f"{status.name} at {pos:.1f} of {dur:.1f} s"
    except Exception:
        return None


def _describe_playbin(audio: Any) -> str:
    """Describe the SDK's motif playbin: none, or its state and position. Never raises."""
    try:
        status = _playbin_status(audio)
        if status is None:
            return "no motif playbin"
        pos, dur = status.position_s, status.duration_s
        at = f", at {pos:.1f} of {dur:.1f} s" if pos is not None and dur is not None else ""
        return f"motif playbin {status.name} {status.state.value_nick}{at}"
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build with the module's motif state, its own play_sound wrapper and pw-dump."""
        self.detector = PhantomDetector(motif=motif)
        self.monitored = self.detector.monitored  # the monitor's streaming thread; never raises
        self.pad = pad or PadLevel()
        self.wobbler = wobbler or WobblerWatch()
        self.sounds = SoundStarts()
        self._clock = clock
        self._list_inputs = list_inputs or (lambda: list_sink_inputs(os.getpid()))
        # Each event runs in its own worker: one episode's start and end are not interleaved.
        self._act_lock = threading.Lock()
        # keep()'s quiet check and attach, and pushed()'s detach, happen in one order.
        self._probe_gate = threading.Lock()
        self._last_warning: float | None = None  # the start of the last episode WARNED about
        self._sink_cache: dict[str, Any] = {}  # the speaker sink, by pipeline
        self._wrapped_for: tuple[int, int] | None = None  # (audio, wobbler) last wrapped

    def pushed(self, now: float, seconds: float) -> None:
        """Note audio the app is pushing, before it is pushed: the speech-sink probe comes off."""
        with self._probe_gate:
            self.detector.pushed(now, seconds)
            self.pad.detach()

    def flushed(self, now: float) -> None:
        """Note a barge-in: the queued audio was dropped."""
        self.detector.flushed(now)

    def take(self) -> list[PhantomEvent]:
        """Return the events to act on, off the event loop."""
        return self.detector.take()

    def keep(self, audio: Any, now: float | None = None) -> None:
        """Keep the probe on the current speaker sink while the app is quiet, and the counts on.

        Runs in a worker thread. A pipeline that is not playing takes the probe off.
        """
        now = self._clock() if now is None else now
        _, sink = playback_health.current_speaker_sink(audio, self._sink_cache)
        if sink is None:
            self.pad.detach()
        elif not self.pad.measures(sink):
            with self._probe_gate:
                if self.detector.app_quiet(now):
                    self.pad.attach(sink)
        wrapped_for = (id(audio), id(getattr(audio, "_head_wobbler", None)))
        if wrapped_for != self._wrapped_for:
            self.wobbler.keep(audio)
            self.sounds.install(audio)
            self._wrapped_for = wrapped_for

    def act(self, event: PhantomEvent, audio: Any) -> None:
        """List the sink inputs when an episode may be warned of; log it at its end.

        Never raises.
        """
        try:
            with self._act_lock:
                self._act(event, audio)
        except Exception:
            logger.warning("Phantom speech report failed", exc_info=True)

    def _sound_before(self, episode: PhantomEpisode) -> str | None:
        """Name a play_sound call made within SDK_SOUND_S before the episode, or None."""
        at = self.sounds.last_before(episode.start + 0.5)
        if at is None or episode.start - at > SDK_SOUND_S:
            return None
        return f"a play_sound call {max(0.0, episode.start - at):.1f} s before"

    def _list(self, episode: PhantomEpisode) -> None:
        """List the sink inputs onto ``episode``, once."""
        if episode.inputs is not None:
            return
        try:
            episode.inputs = self._list_inputs()
        except Exception as exc:
            episode.inputs = ([], f"unavailable ({type(exc).__name__}: {exc})")

    def _act(self, event: PhantomEvent, audio: Any) -> None:
        """Carry out one event. Holds the act lock."""
        episode = event.episode
        if not episode.sdk_checked:  # at the start, or at the end if the start has not run
            episode.sdk_checked = True
            episode.sound_started = self._sound_before(episode)
            episode.sound_playing = _sdk_sound(audio)
        sdk_sound = episode.sdk_sound()
        last = self._last_warning
        may_warn = last is None or episode.start - last >= PHANTOM_REPORT_EVERY_S
        if event.kind == "started":
            if may_warn and sdk_sound is None:
                self._list(episode)
            return
        span = (episode.loud_s, _wall(episode.start), _wall(episode.end_or_last))
        if sdk_sound is not None:
            logger.info(
                "Speaker sound with no push from the app while the SDK's sound player was playing "
                "(%s): %.1f s loud, %s to %s; not phantom speech",
                sdk_sound,
                *span,
            )
        elif may_warn:
            self._list(episode)  # its start's listing may not have run yet
            self._last_warning = episode.start
            logger.warning("%s", self.report(episode, audio))
        else:
            logger.info(
                "Phantom speech again within %.0f s of the last report: %.1f s loud, %s to %s",
                PHANTOM_REPORT_EVERY_S,
                *span,
            )

    def report(self, episode: PhantomEpisode, audio: Any) -> str:
        """Return the episode's WARNING line."""
        end = episode.end_or_last
        if episode.last_push_age_s is None:
            push = "nothing pushed since the app started"
        else:
            push = f"last push {episode.last_push_age_s:.0f} s before"
        pad = self.pad.summary(episode.start - 1.0, end)
        if pad == "no buffers" and not self.pad.attached:
            pad = "not probed"
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
            f"({push}) and no motif; speech sink pad: {pad}; head wobbler fired {fired} times; "
            f"{_describe_playbin(audio)}; {listed}"
        )
