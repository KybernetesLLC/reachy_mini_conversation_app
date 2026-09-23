"""Tests for the headless console stream."""

import time
import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Callable

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from reachy_mini.utils import create_head_pose
import reachy_mini_conversation_app.console as console_mod
from reachy_mini_conversation_app.moves import HoldPoseMove, BreathingMove, MovementManager
from reachy_mini_conversation_app.config import HF_AVAILABLE_VOICES, config
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.startup_settings import (
    StartupSettings,
    load_startup_settings_into_runtime,
)
from reachy_mini_conversation_app.personality_routes import (
    RouteError,
    build_personality_ops,
)


def _rpc_call(app: FastAPI, method: str, params: Any = None) -> dict[str, Any]:
    """Send one JSON-RPC request over /rpc and return the response envelope."""
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        return ws.receive_json()


async def _wait_until(predicate: Any, timeout: float = 1.0) -> None:
    """Wait until a test predicate becomes true."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


def test_clear_audio_queue_prefers_clear_player() -> None:
    """clear_player() is the canonical flush and is used whenever available."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler.output_queue.put_nowait((24000, np.zeros(4, dtype=np.int16)))
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()
    assert handler.output_queue.empty()


def test_clear_audio_queue_falls_back_to_output_buffer() -> None:
    """Older SDKs without clear_player() still flush via clear_output_buffer()."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    audio = SimpleNamespace(clear_output_buffer=MagicMock())  # no clear_player
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    assert handler.output_queue.empty()


def test_clear_audio_queue_drains_queue_in_place() -> None:
    """The output queue is drained in place, not replaced with a new object."""
    handler = MagicMock()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    queue.put_nowait((24000, np.zeros(4, dtype=np.int16)))
    queue.put_nowait((24000, np.zeros(4, dtype=np.int16)))
    handler.output_queue = queue
    audio = SimpleNamespace(clear_player=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    stream = LocalStream(handler, robot)

    stream.clear_audio_queue()

    assert handler.output_queue is queue  # same object, not replaced
    assert queue.empty()


def test_mic_reports_and_toggles_mute_state_over_rpc() -> None:
    """The mic starts live; conversation.mic exposes and flips the pause state."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()

    assert _rpc_call(app, "conversation.mic")["result"] == {"muted": False}
    assert _rpc_call(app, "conversation.mic", {"muted": True})["result"] == {"muted": True}
    assert stream._mic_muted is True
    assert _rpc_call(app, "conversation.mic", {"muted": False})["result"] == {"muted": False}
    assert stream._mic_muted is False

    # headless streams keep the mic live
    assert LocalStream(MagicMock(), robot)._mic_muted is False


def _must_not_be_called(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("the session gate must not rebuild the handler")


def _gate_stream(last_activity: float | None = None) -> LocalStream:
    """A headless LocalStream whose handler has a real activity clock.

    MagicMock's attributes are mocks, and seconds_since_activity() subtracts
    last_activity_time from time.monotonic(), so a bare MagicMock errors on the
    arithmetic instead of measuring anything.
    """
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = MagicMock()
    handler.last_activity_time = time.monotonic() if last_activity is None else last_activity
    return LocalStream(handler, robot)


@pytest.mark.asyncio
async def test_session_gate_parks_the_startup_loop_when_the_session_is_closed(
    monkeypatch: Any,
) -> None:
    """A cleared gate stops the loop before start_up(), and never rebuilds the handler."""
    stream = _gate_stream()
    started: list[str] = []

    async def _start_up() -> None:
        started.append("start_up")

    stream.handler.start_up = _start_up
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)

    stream._session_wanted.clear()
    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    await asyncio.sleep(0.05)
    assert started == []          # parked: no session was opened

    stream._session_wanted.set()
    await _wait_until(lambda: started == ["start_up"])  # woken by the gate alone

    stream._stop_event.set()
    loop_task.cancel()


@pytest.mark.asyncio
async def test_closing_the_session_does_not_request_a_restart() -> None:
    """The open path must not touch _restart_requested: that rebuilds the handler,
    and a fresh handler greets again — the defect this whole change closes."""
    stream = _gate_stream()
    stream.handler.shutdown = AsyncMock()
    # No startup loop is running here, so nothing will ever mark the gate parked;
    # mark it directly so close_session's retry-until-parked wait resolves at once.
    stream._session_parked.set()

    await stream.close_session()
    assert stream._session_wanted.is_set() is False
    assert stream._restart_requested.is_set() is False

    await stream.open_session()
    assert stream._session_wanted.is_set() is True
    assert stream._restart_requested.is_set() is False


@pytest.mark.asyncio
async def test_a_closed_session_does_not_age_the_app_toward_sleep(monkeypatch: Any) -> None:
    """seconds_since_activity is session-scoped; a deliberate close must not
    look like idleness, or the inactivity timeout stops the app."""
    stream = _gate_stream(last_activity=time.monotonic() - 10_000.0)
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    stream._session_closed_keepalive = 0.01
    assert stream.seconds_since_activity() > 9_000.0

    stream._session_wanted.clear()
    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    await _wait_until(lambda: stream.seconds_since_activity() < 5.0)

    stream._stop_event.set()
    stream._session_wanted.set()
    loop_task.cancel()


@pytest.mark.asyncio
async def test_closing_drops_responses_queued_for_the_next_session() -> None:
    """A close mid-turn must not leave a response.create() for the next session.

    shutdown() drains output_queue and not _pending_responses, and the sender loop
    runs `while self.connection:`, so a queued item survives the close and the next
    session's sender fires it at once — the robot speaking unprompted after the
    microphone was switched off and on again.
    """
    stream = _gate_stream()
    stream.handler.shutdown = AsyncMock()
    stream.handler._pending_responses = asyncio.Queue()
    stream.handler._pending_responses.put_nowait({"response": {}})
    stream.handler._pending_responses.put_nowait({"response": {}})
    # No startup loop is running here; mark the gate parked directly (see the
    # identical comment in test_closing_the_session_does_not_request_a_restart).
    stream._session_parked.set()

    await stream.close_session()

    assert stream.handler._pending_responses.empty()


def test_status_payload_distinguishes_closed_on_purpose_from_broken() -> None:
    """Closed on purpose must be distinguishable from merely disconnected."""
    stream = _gate_stream()
    stream.handler.connection = None

    stream._session_wanted.clear()
    payload = stream._backend_connection_status()
    assert payload["session_wanted"] is False
    assert payload["backend_connected"] is False
    assert payload["backend_connection_state"] == "session_closed"

    stream._session_wanted.set()
    stream._backend_connection_state = "disconnected"
    payload = stream._backend_connection_status()
    assert payload["session_wanted"] is True
    assert payload["backend_connection_state"] == "disconnected"


def test_conversation_session_opens_closes_and_reads_back() -> None:
    """conversation.session mirrors conversation.mic: optional param, state returned."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = MagicMock()
    handler.connection = None
    handler.last_activity_time = time.monotonic()
    handler.shutdown = AsyncMock()
    stream = LocalStream(handler, robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    # No startup loop is running here; mark the gate parked directly (see the
    # identical comment in test_closing_the_session_does_not_request_a_restart).
    stream._session_parked.set()

    assert _rpc_call(app, "conversation.session")["result"] == {
        "wanted": True,
        "connected": False,
    }
    assert _rpc_call(app, "conversation.session", {"open": False})["result"] == {
        "wanted": False,
        "connected": False,
    }
    assert stream._session_wanted.is_set() is False
    assert stream._restart_requested.is_set() is False

    assert _rpc_call(app, "conversation.session", {"open": True})["result"] == {
        "wanted": True,
        "connected": False,
    }
    assert stream._session_wanted.is_set() is True


def test_status_over_rpc_carries_session_wanted() -> None:
    """The supervisor reads the off state from conversation.status, not only from
    the verb's own return value."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = MagicMock()
    handler.connection = None
    handler.last_activity_time = time.monotonic()
    handler.shutdown = AsyncMock()
    stream = LocalStream(handler, robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    # No startup loop is running here; mark the gate parked directly (see the
    # identical comment in test_closing_the_session_does_not_request_a_restart).
    stream._session_parked.set()

    assert _rpc_call(app, "conversation.status")["result"]["session_wanted"] is True
    _rpc_call(app, "conversation.session", {"open": False})
    status = _rpc_call(app, "conversation.status")["result"]
    assert status["session_wanted"] is False
    assert status["backend_connection_state"] == "session_closed"


def test_capture_verb_stops_and_starts_both_pipelines() -> None:
    """One GStreamer pipeline carries record and playback, so the verb moves both.

    Stopping only the recorder would leave the app able to speak while claiming
    the microphone is off -- a readback that lies about the hardware state.
    """
    app = FastAPI()
    robot = _audio_robot(
        stop_recording=MagicMock(),
        start_recording=MagicMock(),
        stop_playing=MagicMock(),
        start_playing=MagicMock(),
    )
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()

    assert _rpc_call(app, "conversation.capture", {"on": False})["result"] == {"on": False}
    assert robot.media.stop_recording.called
    assert robot.media.stop_playing.called

    assert _rpc_call(app, "conversation.capture", {"on": True})["result"] == {"on": True}
    assert robot.media.start_recording.called
    assert robot.media.start_playing.called


def test_capture_verb_without_a_parameter_only_reports() -> None:
    """Reading conversation.capture back must not itself change the state."""
    app = FastAPI()
    robot = _audio_robot(start_recording=MagicMock())
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    stream._capture_on = False

    assert _rpc_call(app, "conversation.capture", {})["result"] == {"on": False}
    assert not robot.media.start_recording.called


def test_capture_state_appears_in_status() -> None:
    """The supervisor reads this back; a verb with no readback can lie."""
    app = FastAPI()
    robot = _audio_robot(
        stop_recording=MagicMock(),
        start_recording=MagicMock(),
        stop_playing=MagicMock(),
        start_playing=MagicMock(),
    )
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()

    _rpc_call(app, "conversation.capture", {"on": False})
    assert _rpc_call(app, "conversation.status")["result"]["capture"] is False


def _pose_stream() -> tuple[LocalStream, MovementManager, FastAPI]:
    """Return a LocalStream wired to a real, unstarted MovementManager for conversation.pose tests."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    manager = MovementManager(MagicMock())
    handler = MagicMock()
    handler.deps = SimpleNamespace(movement_manager=manager)
    stream = LocalStream(handler, robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    return stream, manager, app


def _drive_manager(manager: MovementManager, t: float | None = None) -> None:
    """Run one worker-thread tick's worth of command handling, without a live thread."""
    now = manager._now() if t is None else t
    manager._poll_signals(now)
    manager._update_primary_motion(now)


def test_pose_hold_over_rpc_queues_a_hold_move() -> None:
    """A hold request returns {"holding": True} and leaves a HoldPoseMove current or queued."""
    _stream, manager, app = _pose_stream()

    resp = _rpc_call(
        app,
        "conversation.pose",
        {"head_pose": {"pitch": 0.15}, "antennas": [0.2, -0.2], "duration": 0.5},
    )

    assert resp["result"] == {"holding": True}
    _drive_manager(manager)
    assert isinstance(manager.state.current_move, HoldPoseMove)
    assert manager.is_holding() is True


def test_pose_hold_replaces_a_previous_hold() -> None:
    """A second hold leaves exactly one HoldPoseMove current, targeting the new pose."""
    _stream, manager, app = _pose_stream()

    _rpc_call(app, "conversation.pose", {"antennas": [0.2, -0.2], "duration": 0.5})
    _drive_manager(manager)
    first_hold = manager.state.current_move
    assert isinstance(first_hold, HoldPoseMove)

    _rpc_call(app, "conversation.pose", {"antennas": [-0.15, 0.15], "duration": 0.5})
    _drive_manager(manager)

    assert manager.state.current_move is not first_hold
    second_hold = manager.state.current_move
    assert isinstance(second_hold, HoldPoseMove)
    assert len(manager.move_queue) == 0
    np.testing.assert_array_equal(second_hold.target_antennas, [-0.15, 0.15])


def test_pose_release_over_rpc_clears_the_hold() -> None:
    """Release reports {"holding": False} and leaves no HoldPoseMove current or queued."""
    _stream, manager, app = _pose_stream()
    _rpc_call(app, "conversation.pose", {"antennas": [0.1, -0.1], "duration": 0.5})
    _drive_manager(manager)
    assert manager.is_holding() is True

    resp = _rpc_call(app, "conversation.pose", {"release": True})

    assert resp["result"] == {"holding": False}
    _drive_manager(manager)
    assert manager.is_holding() is False
    assert manager.state.current_move is None
    assert len(manager.move_queue) == 0


def test_pose_release_over_rpc_is_a_no_op_when_nothing_is_held() -> None:
    """Releasing with no hold in play does not disturb another current move."""
    _stream, manager, app = _pose_stream()
    breathing_move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(0.0, 0.0),
    )
    manager.state.current_move = breathing_move
    manager.state.move_start_time = manager._now()

    resp = _rpc_call(app, "conversation.pose", {"release": True})

    assert resp["result"] == {"holding": False}
    _drive_manager(manager)
    assert manager.state.current_move is breathing_move


def test_pose_read_only_reports_without_changing_anything() -> None:
    """An empty-params read reports current state and queues no command."""
    _stream, manager, app = _pose_stream()

    resp = _rpc_call(app, "conversation.pose", {})
    assert resp["result"] == {"holding": False}
    assert manager._command_queue.empty()

    _rpc_call(app, "conversation.pose", {"antennas": [0.1, -0.1], "duration": 0.5})
    _drive_manager(manager)
    assert manager.is_holding() is True

    resp = _rpc_call(app, "conversation.pose", {})
    assert resp["result"] == {"holding": True}
    assert manager._command_queue.empty()
    assert manager.is_holding() is True


@pytest.mark.parametrize(
    "params",
    [
        {"duration": 1.0},  # missing antennas
        {"antennas": [0.1]},  # wrong antenna count
        {"antennas": [0.1, "nope"]},  # non-numeric antenna
        {"antennas": [0.1, -0.1], "head_pose": {"pitch": "nope"}},  # non-numeric head_pose field
        {"antennas": [0.1, -0.1], "duration": "nope"},  # non-numeric duration
        {"antennas": [0.1, -0.1], "duration": 0.0},  # duration must be > 0
        {"antennas": [0.1, -0.1], "duration": -1.0},  # duration must be > 0
        {"antennas": [float("nan"), -0.1]},  # NaN antenna
        {"antennas": [0.1, -0.1], "duration": float("inf")},  # infinite duration
        {"antennas": [0.1, -0.1], "head_pose": {"pitch": float("nan")}},  # NaN head_pose field
    ],
)
def test_pose_invalid_params_are_rejected(params: dict[str, Any]) -> None:
    """Malformed hold params raise the file's invalid_params convention and queue nothing."""
    _stream, manager, app = _pose_stream()

    resp = _rpc_call(app, "conversation.pose", params)

    assert resp["error"]["data"]["reason"] == "invalid_params"
    assert resp["error"]["code"] == -32602
    assert manager._command_queue.empty()


def test_pose_without_a_movement_manager_reports_not_running() -> None:
    """A handler with no wired movement manager fails predictably, not silently."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = MagicMock()
    handler.deps = SimpleNamespace(movement_manager=None)
    stream = LocalStream(handler, robot, settings_app=app)
    stream._init_settings_ui_if_needed()

    resp = _rpc_call(app, "conversation.pose", {"antennas": [0.1, -0.1]})

    assert resp["error"]["data"]["reason"] == "not_running"


@pytest.mark.asyncio
async def test_close_session_keeps_closing_until_the_loop_actually_parks(
    monkeypatch: Any,
) -> None:
    """A close landing while a session is being established must not be satisfied
    by a single shutdown() call that has nothing yet to close.

    _run_realtime_session assigns handler.connection only after the websocket
    handshake and the first session.update; start_up() spends real time with
    connection still None before that, including retry backoff sleeps. A close
    arriving in that window finds shutdown()'s `if self.connection:` guard is a
    no-op, so clearing the gate and calling shutdown() exactly once would let
    that session finish establishing anyway, with the gate already reporting
    closed -- audio would keep reaching the provider while the readback claims
    otherwise. close_session must keep closing until the loop is actually
    parked with nothing connected.
    """
    stream = _gate_stream()
    stream.handler.connection = None  # unset until the fake start_up "connects"
    stream._backend_retry_delay = 0.05  # keep the loop's own retry sleep short
    monkeypatch.setattr(
        type(stream), "_build_handler_for_current_backend", _must_not_be_called
    )
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)

    close_live_calls: list[str] = []
    original_close_live_session = type(stream)._close_live_session

    async def _counting_close_live_session(self: Any) -> None:
        close_live_calls.append("close")
        await original_close_live_session(self)

    monkeypatch.setattr(type(stream), "_close_live_session", _counting_close_live_session)

    entered_connecting = asyncio.Event()
    session_ended = asyncio.Event()

    async def _start_up() -> None:
        entered_connecting.set()
        # Simulate the handshake + session.update delay, during which the real
        # handler's connection attribute is still None.
        await asyncio.sleep(0.1)
        stream.handler.connection = object()  # assigned only once "connected"
        await session_ended.wait()
        stream.handler.connection = None  # mirrors the real start_up's own `finally`

    async def _shutdown() -> None:
        # Mirrors handler.shutdown()'s own `if self.connection:` guard: nothing
        # to close while still connecting.
        if stream.handler.connection is not None:
            stream.handler.connection = None
            session_ended.set()

    stream.handler.start_up = _start_up
    stream.handler.shutdown = _shutdown

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    await entered_connecting.wait()

    # Land the close while the fake handshake is still in flight: connection is
    # still None at this instant, so a single close attempt has nothing to close.
    assert stream.handler.connection is None
    result = await stream.close_session(timeout=2.0)

    assert result is True
    assert stream._session_wanted.is_set() is False
    assert stream._session_parked.is_set() is True
    assert stream._backend_connected() is False
    assert len(close_live_calls) > 1  # the no-op attempt, and the one that closed

    stream._stop_event.set()
    stream._session_wanted.set()
    loop_task.cancel()


@pytest.mark.asyncio
async def test_close_session_returns_false_when_it_cannot_settle() -> None:
    """A close that cannot get the connection down and the loop parked within its
    budget reports failure rather than claiming success it did not achieve."""
    stream = _gate_stream()
    stream.handler.connection = object()  # never actually closes
    stream.handler.shutdown = AsyncMock()  # does not touch handler.connection

    result = await stream.close_session(timeout=0.1)

    assert result is False
    assert stream._session_wanted.is_set() is False  # the intent to close still recorded


@pytest.mark.asyncio
async def test_the_retry_sleep_wakes_when_the_session_gate_is_cleared() -> None:
    """A voice-operated off switch cannot wait out a five-second retry delay."""
    stream = _bare_stream()
    stream._session_wanted.set()

    task = asyncio.ensure_future(stream._sleep_or_restart_requested(5.0))
    await asyncio.sleep(0.05)
    stream._session_wanted.clear()
    stream._session_gate_changed.set()
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_the_retry_sleep_still_wakes_on_a_restart_request() -> None:
    """Waking on the session gate must not cost the existing wake on a restart."""
    stream = _bare_stream()

    task = asyncio.ensure_future(stream._sleep_or_restart_requested(5.0))
    await asyncio.sleep(0.05)
    stream._restart_requested.set()
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_close_session_settles_quickly_during_a_real_retry_sleep(
    monkeypatch: Any,
) -> None:
    """The end-to-end promise: a close must not wait out a real retry delay.

    Runs the real _run_handler_startup_loop(), with a start_up() that fails,
    so the loop is genuinely parked inside its retry sleep at the real
    BACKEND_RETRY_DELAY_SECONDS -- not a hand-set gate event and not a
    shortened delay -- when close_session() is called.
    """
    stream = _gate_stream()
    stream.handler.connection = None
    stream.handler.shutdown = AsyncMock()
    stream._backend_retry_delay = 5.0
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)

    async def _failing_start_up() -> None:
        raise RuntimeError("simulated connect failure")

    stream.handler.start_up = _failing_start_up

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    await asyncio.sleep(0.1)  # let the loop fail start_up() and enter the retry sleep

    result = await asyncio.wait_for(stream.close_session(), timeout=1.0)

    assert result is True

    stream._stop_event.set()
    stream._session_wanted.set()
    loop_task.cancel()


def test_rest_api_is_removed_in_favor_of_rpc() -> None:
    """The /api/v1 REST + SSE surface is gone; control is JSON-RPC over /rpc."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)

    for path in (
        "/api/v1/status",
        "/api/v1/mic",
        "/api/v1/personalities",
        "/api/v1/voices",
        "/api/v1/tool_spaces",
        "/api/v1/profile_tools",
    ):
        assert client.get(path).status_code == 404
    assert client.get("/api/v1/conversation_events").status_code == 404

    # ...but /rpc drives it fine.
    assert _rpc_call(app, "conversation.status")["result"]["backend"]


def test_settings_ui_detaches_framework_catch_all_before_own_routes() -> None:
    """Framework fallback routes should not shadow the UI or the /rpc endpoint."""
    app = FastAPI()

    @app.get("/{path:path}")
    def _framework_fallback(path: str) -> None:
        raise HTTPException(status_code=404)

    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/static/js/api.js").status_code == 200
    assert _rpc_call(app, "conversation.status")["result"]["backend"]


@pytest.mark.asyncio
async def test_activity_from_rebuilt_handler_reaches_rpc_clients() -> None:
    """Activity from a rebuilt handler must still reach /rpc subscribers."""

    class FakeHandler:
        def __init__(self) -> None:
            self.observer: Any = None

        def set_activity_observer(self, observer: Any) -> None:
            self.observer = observer

    rebuilt = FakeHandler()
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(FakeHandler(), robot, settings_app=app, handler_factory=lambda voice: rebuilt)
    stream._init_settings_ui_if_needed()
    stream._build_handler_for_current_backend()  # rebuild re-wires the observer

    with TestClient(app).websocket_connect("/rpc") as ws:
        rebuilt.observer("assistant_audio_delta")
        # First frame is conversation.activity (raw reason).
        msg = ws.receive_json()
    assert msg["method"] == "conversation.activity"
    assert msg["params"] == {"reason": "assistant_audio_delta"}


def test_backend_config_requests_in_process_restart_with_handler_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild-capable LocalStream should reconnect in process after a connection change."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.delenv("HF_REALTIME_WS_URL", raising=False)

    app = FastAPI()
    handler = MagicMock()
    handler.shutdown = AsyncMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(
        handler,
        robot,
        settings_app=app,
        instance_path=str(tmp_path),
        handler_factory=lambda _voice: handler,
    )
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {"hf_mode": "local", "hf_host": "localhost", "hf_port": 8765})["result"]

    assert data["ok"] is True
    assert data["message"] == "Connection saved. Reconnecting backend."
    assert data["backend"] == "huggingface"
    assert data["requires_restart"] is False
    assert data["can_proceed"] is True
    assert data["backend_connection_state"] == "connecting"
    assert stream._restart_requested.is_set()


def test_backend_config_persists_local_hf_selection_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should persist a direct Hugging Face websocket target."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.delenv("HF_REALTIME_SESSION_URL", raising=False)
    monkeypatch.delenv("HF_REALTIME_WS_URL", raising=False)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {"hf_mode": "local", "hf_host": "localhost", "hf_port": 8765})["result"]

    assert data["ok"] is True
    assert data["backend"] == "huggingface"
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "localhost"
    assert data["hf_direct_port"] == 8765

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text


def test_backend_config_persists_deployed_mode_without_clearing_local_hf_ws_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving deployed mode should make env selection explicit and remove stale allocator URLs."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_REALTIME_SESSION_URL=https://lb.example.test/session\n"
        "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")
    monkeypatch.delenv("HF_REALTIME_CONNECTION_MODE", raising=False)
    monkeypatch.setenv("HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://localhost:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {"hf_mode": "deployed"})["result"]

    assert data["ok"] is True
    assert data["has_hf_session_url"] is True
    assert data["has_hf_ws_url"] is True
    assert data["hf_connection_mode"] == "deployed"

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=deployed" in env_text
    assert "HF_REALTIME_SESSION_URL=" not in env_text
    assert "HF_REALTIME_WS_URL=ws://localhost:8765/v1/realtime" in env_text


def test_backend_config_switches_to_saved_local_hf_connection_without_payload_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching back to a saved local Hugging Face backend should reuse the persisted target."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_REALTIME_CONNECTION_MODE=local\nHF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")
    monkeypatch.setenv("HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setenv("HF_REALTIME_WS_URL", "ws://192.168.1.42:8766/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "backend.config", {})["result"]

    assert data["ok"] is True
    assert data["backend"] == "huggingface"
    assert data["hf_connection_mode"] == "local"
    assert data["hf_direct_host"] == "192.168.1.42"
    assert data["hf_direct_port"] == 8766

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_REALTIME_CONNECTION_MODE=local" in env_text
    assert "HF_REALTIME_WS_URL=ws://192.168.1.42:8766/v1/realtime" in env_text


def test_backend_config_rejects_invalid_hf_port_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should reject invalid local Hugging Face ports from direct callers."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    resp = _rpc_call(
        app,
        "backend.config",
        {"backend": "huggingface", "hf_mode": "local", "hf_host": "localhost", "hf_port": 0},
    )

    assert resp["error"]["data"]["reason"] == "invalid_hf_port"


def test_status_reports_direct_hf_ws_url_as_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should treat a direct Hugging Face websocket as a valid configuration."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app, instance_path=str(tmp_path))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "conversation.status")["result"]

    assert data["backend"] == "huggingface"
    assert data["has_hf_session_url"] is False
    assert data["has_hf_ws_url"] is True
    assert data["has_hf_connection"] is True
    assert data["hf_connection_mode"] == "local"
    assert data["can_proceed_with_hf"] is True


def test_status_reports_backend_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings API should expose backend connection failures without hiding controls."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    handler = MagicMock()
    handler.connection = None
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot, settings_app=app, instance_path=str(tmp_path))
    stream._set_backend_connection_state("disconnected", RuntimeError("connect failed"))
    stream._init_settings_ui_if_needed()

    data = _rpc_call(app, "conversation.status")["result"]
    assert data["backend"] == "huggingface"
    assert data["backend_connected"] is False
    assert data["backend_connection_state"] == "disconnected"
    assert data["backend_error"] == "RuntimeError: connect failed"
    assert data["can_proceed"] is True
    assert data["can_proceed_with_hf"] is True


def test_backend_startup_failure_is_recorded_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend startup failures should become status state instead of killing LocalStream."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    app = FastAPI()
    handler = MagicMock()
    handler.connection = None
    handler.shutdown = AsyncMock()
    media = SimpleNamespace(
        audio=None,
        backend=None,
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    stream = LocalStream(handler, robot, settings_app=app, instance_path=str(tmp_path))
    stream._backend_retry_delay = 0
    stream.record_loop = AsyncMock(return_value=None)  # type: ignore[method-assign]
    stream.play_loop = AsyncMock(return_value=None)  # type: ignore[method-assign]
    monkeypatch.setattr("reachy_mini_conversation_app.console.apply_audio_startup_config", MagicMock())

    async def fail_and_stop() -> None:
        stream._stop_event.set()
        raise RuntimeError("local server unavailable")

    handler.start_up = AsyncMock(side_effect=fail_and_stop)

    try:
        stream.launch()
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())

    handler.start_up.assert_awaited_once()
    data = _rpc_call(app, "conversation.status")["result"]
    assert data["backend_connected"] is False
    assert data["backend_connection_state"] == "disconnected"
    assert data["backend_error"] == "RuntimeError: local server unavailable"


def test_media_warmup_overlaps_audio_startup_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audio configuration should run while the media pipelines warm up."""
    monkeypatch.setattr("reachy_mini_conversation_app.console.has_hf_realtime_target", lambda: True)

    handler = MagicMock()
    handler.shutdown = AsyncMock()
    media = SimpleNamespace(
        audio=None,
        backend=None,
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    stream = LocalStream(handler, SimpleNamespace(media=media))
    stream.record_loop = AsyncMock(return_value=None)  # type: ignore[method-assign]
    stream.play_loop = AsyncMock(return_value=None)  # type: ignore[method-assign]

    startup_barrier = threading.Barrier(2)

    async def wait_for_audio_config(_delay: float) -> None:
        await asyncio.to_thread(startup_barrier.wait, 5.0)

    def apply_audio_config(*_args: Any, **_kwargs: Any) -> bool:
        startup_barrier.wait(5.0)
        return True

    async def start_and_stop() -> None:
        stream._stop_event.set()

    handler.start_up = AsyncMock(side_effect=start_and_stop)
    monkeypatch.setattr("reachy_mini_conversation_app.console.asyncio.sleep", wait_for_audio_config)
    monkeypatch.setattr("reachy_mini_conversation_app.console.apply_audio_startup_config", apply_audio_config)

    try:
        stream.launch()
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.mark.asyncio
async def test_startup_loop_rebuilds_handler_on_restart_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """LocalStream should shut down and rebuild the handler when a restart is requested."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")

    class FakeHandler:
        def __init__(self) -> None:
            self.connection = None
            self.output_queue = asyncio.Queue()
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()
            self.shutdown_calls = 0

        async def start_up(self) -> None:
            self.connection = object()
            self.started.set()
            await self.stopped.wait()
            self.connection = None

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.stopped.set()

        async def receive(self, _frame: Any) -> None:
            return None

        async def emit(self) -> None:
            return None

    handlers: list[FakeHandler] = []

    def handler_factory(_voice: str | None) -> FakeHandler:
        handler = FakeHandler()
        handlers.append(handler)
        return handler

    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    initial_handler = handler_factory(None)
    stream = LocalStream(initial_handler, robot, handler_factory=handler_factory)
    stream._backend_retry_delay = 0.01

    startup_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await _wait_until(lambda: initial_handler.started.is_set())

        await stream.request_backend_restart("backend_config_changed")

        await _wait_until(lambda: len(handlers) == 2 and handlers[1].started.is_set())

        assert initial_handler.shutdown_calls >= 1
        assert stream.handler is handlers[1]
        assert stream._backend_connected() is True
    finally:
        stream._stop_event.set()
        await stream._shutdown_active_handler()
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_personality_ops_return_hf_voices() -> None:
    """With no running loop, voices() falls back to the Hugging Face catalog."""
    ops = build_personality_ops(MagicMock(), lambda: None)
    assert await ops.voices() == HF_AVAILABLE_VOICES


def test_personality_ops_delete_builtin_is_not_deletable() -> None:
    """Deleting a built-in personality raises not_deletable (was REST 404)."""
    ops = build_personality_ops(MagicMock(), lambda: None)
    with pytest.raises(RouteError) as ei:
        ops.delete("mad_scientist_assistant")
    assert ei.value.reason == "not_deletable"


def test_personality_ops_load_builtin_default_profile() -> None:
    """The bulk personality API should retain the complete profile payload."""
    ops = build_personality_ops(MagicMock(), lambda: None)
    data = ops.load("default")
    assert "Reachy Mini" in data["instructions"]
    assert data["tools_text"]
    assert data["enabled_tools"]
    assert data["available_tools"]


@pytest.mark.asyncio
async def test_personality_ops_apply_voice() -> None:
    """apply_voice delegates to the handler and reports the status."""
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to cedar.")
    ops = build_personality_ops(handler, lambda: asyncio.get_running_loop())

    result = await ops.apply_voice("cedar")

    assert result == {"ok": True, "status": "Voice changed to cedar."}
    handler.change_voice.assert_awaited_once_with("cedar")


@pytest.mark.asyncio
async def test_personality_ops_persist_startup_with_voice_override() -> None:
    """Applying with persist=True saves the active manual voice override."""
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="Applied personality and restarted realtime session.")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    persist_personality = MagicMock()
    ops = build_personality_ops(handler, lambda: asyncio.get_running_loop(), persist_personality=persist_personality)

    result = await ops.apply("sorry_bro", persist=True)

    assert result["ok"] is True
    handler.apply_personality.assert_awaited_once_with("sorry_bro")
    persist_personality.assert_called_once_with("sorry_bro", "shimmer")


@pytest.mark.asyncio
async def test_personality_ops_apply_same_profile_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-applying the active personality is a no-op for the realtime handler."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "sorry_bro")
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="should not be called")
    handler.get_current_voice = MagicMock(return_value="shimmer")
    ops = build_personality_ops(handler, lambda: None)

    result = await ops.apply("sorry_bro")

    assert result["status"] == "Personality unchanged."
    handler.apply_personality.assert_not_awaited()
    handler.get_current_voice.assert_not_called()


def test_personality_ops_startup_choice_survives_runtime_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime profile switching should not redefine the saved startup personality."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "captain_circuit")
    ops = build_personality_ops(MagicMock(), lambda: None)

    first = ops.get_choices()
    assert first["current"] == "captain_circuit"
    assert first["startup"] == "captain_circuit"

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "chess_coach")

    second = ops.get_choices()
    assert second["current"] == "chess_coach"
    assert second["startup"] == "captain_circuit"


@pytest.mark.asyncio
async def test_personality_ops_use_apply_callback() -> None:
    """Apply delegates to the injected apply_personality callback, not the handler."""
    handler = MagicMock()
    handler.apply_personality = AsyncMock(return_value="handler should not be called")
    apply_personality = AsyncMock(return_value="Applied personality and restarting backend.")
    get_current_voice = MagicMock(return_value="cedar")
    ops = build_personality_ops(
        handler,
        lambda: asyncio.get_running_loop(),
        apply_personality=apply_personality,
        get_current_voice=get_current_voice,
    )

    result = await ops.apply("sorry_bro")

    assert result["status"] == "Applied personality and restarting backend."
    apply_personality.assert_awaited_once_with("sorry_bro")
    handler.apply_personality.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_personality_propagates_restart_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation during backend restart should not be converted into a status string."""
    monkeypatch.setattr(console_mod, "set_custom_profile", lambda _profile: None)
    monkeypatch.setattr(console_mod, "get_session_instructions", lambda _instance_path=None: "instructions")
    monkeypatch.setattr(console_mod, "get_session_voice", lambda default: default)

    stream = LocalStream(MagicMock(), MagicMock())

    async def cancel_restart(_reason: str) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(stream, "request_backend_restart", cancel_restart)

    with pytest.raises(asyncio.CancelledError):
        await stream.apply_personality("sorry_bro")


@pytest.mark.asyncio
async def test_apply_personality_restores_profile_when_tool_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed tool rebuild must not leave the rejected profile selected."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(
        console_mod,
        "set_custom_profile",
        lambda profile: setattr(config, "REACHY_MINI_CUSTOM_PROFILE", profile),
    )
    monkeypatch.setattr(console_mod, "get_session_instructions", lambda: "instructions")
    monkeypatch.setattr(console_mod, "get_session_voice", lambda default: default)
    monkeypatch.setattr(console_mod, "initialize_tools", MagicMock(side_effect=RuntimeError("invalid tools")))
    stream = LocalStream(MagicMock(), MagicMock())

    with pytest.raises(RuntimeError, match="invalid tools"):
        await stream.apply_personality("broken")

    assert config.REACHY_MINI_CUSTOM_PROFILE == "default"


@pytest.mark.asyncio
async def test_local_stream_change_voice_delegates_without_backend_restart() -> None:
    """LocalStream voice changes should update the active handler without rebuilding it."""
    handler = MagicMock()
    handler.change_voice = AsyncMock(return_value="Voice changed to Serena.")
    handler.get_current_voice = MagicMock(return_value="Serena")
    stream = LocalStream(handler, MagicMock())

    status = await stream.change_voice("Serena")

    assert status == "Voice changed to Serena."
    handler.change_voice.assert_awaited_once_with("Serena")
    assert stream._voice_override == "Serena"
    assert not stream._restart_requested.is_set()


def test_local_stream_persist_personality_stores_voice_override(tmp_path) -> None:
    """Persisting startup settings should write both profile and voice override."""
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality("sorry_bro", "shimmer")

    settings_path = tmp_path / "startup_settings.json"
    assert settings_path.exists()
    assert settings_path.read_text(encoding="utf-8") == '{\n  "profile": "sorry_bro",\n  "voice": "shimmer"\n}\n'
    assert stream._read_persisted_personality() == "sorry_bro"


def test_local_stream_persist_personality_clears_legacy_startup_env_overrides(tmp_path, monkeypatch) -> None:
    """Saving startup settings should remove legacy `.env` profile and voice overrides."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "HF_TOKEN=test-token\n"
        "REACHY_MINI_CUSTOM_PROFILE=mad_scientist_assistant\n"
        "REACHY_MINI_VOICE_OVERRIDE=shimmer\n",
        encoding="utf-8",
    )
    stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

    stream._persist_personality(None, "Aiden")

    env_text = env_path.read_text(encoding="utf-8")
    assert "HF_TOKEN=test-token" in env_text
    assert "REACHY_MINI_CUSTOM_PROFILE=" not in env_text
    assert "REACHY_MINI_VOICE_OVERRIDE=" not in env_text

    applied_profiles: list[str | None] = []
    monkeypatch.delenv("REACHY_MINI_CUSTOM_PROFILE", raising=False)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.set_custom_profile",
        lambda profile: applied_profiles.append(profile),
    )

    settings = load_startup_settings_into_runtime(tmp_path)

    assert settings == StartupSettings(voice="Aiden")
    assert applied_profiles == [None]


def test_local_stream_launch_waits_for_missing_hf_target_without_starting_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup should wait for settings input when the Hugging Face target is missing."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", None)
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", None)

    media = SimpleNamespace(
        start_recording=MagicMock(),
        start_playing=MagicMock(),
    )
    robot = SimpleNamespace(media=media)
    stream = LocalStream(MagicMock(), robot, settings_app=FastAPI(), instance_path=str(tmp_path))

    init_settings_ui = MagicMock()
    monkeypatch.setattr(stream, "_init_settings_ui_if_needed", init_settings_ui)
    monkeypatch.setattr("reachy_mini_conversation_app.console.time.sleep", MagicMock(side_effect=KeyboardInterrupt))

    stream.launch()

    init_settings_ui.assert_called_once()
    media.start_recording.assert_not_called()
    media.start_playing.assert_not_called()


def _bare_stream() -> LocalStream:
    """Return a LocalStream with a no-audio robot, enough for helper-method tests."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    return LocalStream(MagicMock(), robot)


def test_read_env_lines_prefers_existing_file(tmp_path: Path) -> None:
    """An existing .env is read verbatim, ignoring the template."""
    env_path = tmp_path / ".env"
    env_path.write_text("A=1\nB=2\n", encoding="utf-8")

    assert _bare_stream()._read_env_lines(env_path) == ["A=1", "B=2"]


def test_read_env_lines_falls_back_to_example_template(tmp_path: Path) -> None:
    """When no .env exists, the sibling .env.example is used as the template."""
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")

    assert _bare_stream()._read_env_lines(tmp_path / ".env") == ["OPENAI_API_KEY="]


def test_seconds_since_activity_reads_handler() -> None:
    """seconds_since_activity is measured from the handler's last activity time."""
    stream = _bare_stream()
    stream.handler.last_activity_time = time.monotonic() - 5.0

    assert stream.seconds_since_activity() >= 5.0


def test_get_current_voice_prefers_override() -> None:
    """A manual voice override wins over the profile voice."""
    stream = _bare_stream()
    stream._voice_override = "Serena"

    assert stream.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_change_voice_reports_handler_failure() -> None:
    """A failing handler voice change is surfaced as an error string, not raised."""
    handler = MagicMock()
    handler.change_voice = AsyncMock(side_effect=RuntimeError("backend down"))
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot)

    result = await stream.change_voice("Serena")

    assert "Failed to change voice" in result


def _audio_robot(**media_attrs: Any) -> SimpleNamespace:
    """Return a robot whose media exposes only the attributes a test drives."""
    return SimpleNamespace(media=SimpleNamespace(audio=None, backend=None, **media_attrs))


def _stop_after(stream: LocalStream, value: Any) -> Callable[[], Any]:
    """Return a side effect that stops the stream after one iteration, yielding `value`."""

    def _side_effect() -> Any:
        stream._stop_event.set()
        return value

    return _side_effect


@pytest.mark.asyncio
async def test_record_loop_forwards_unmuted_frames() -> None:
    """A recorded frame is forwarded to the handler with the input sample rate."""
    frame = np.zeros(4, dtype=np.int16)
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    robot.media.get_audio_sample.side_effect = _stop_after(stream, frame)

    await stream.record_loop()

    handler.receive.assert_awaited_once_with((16000, frame))


@pytest.mark.asyncio
async def test_record_loop_skips_frames_while_muted() -> None:
    """No frames are forwarded while the mic is muted."""
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._mic_muted = True
    robot.media.get_audio_sample.side_effect = _stop_after(stream, np.zeros(4, dtype=np.int16))

    await stream.record_loop()

    handler.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_loop_skips_missing_frames() -> None:
    """A None frame from the recorder is not forwarded."""
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    robot.media.get_audio_sample.side_effect = _stop_after(stream, None)

    await stream.record_loop()

    handler.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_loop_backs_off_instead_of_polling_while_capture_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While capture is off, record_loop must not poll get_audio_sample() or spin.

    A None frame from a NULL pipeline plus asyncio.sleep(0) to yield is a busy
    loop: measured on the robot, it dragged the motor control loop from 49.5 Hz
    to 45.9 Hz.
    """

    class _RecordingAsyncio:
        """Proxies the real asyncio module but records record_loop's sleep calls."""

        def __init__(self, real: Any) -> None:
            self._real = real
            self.sleep_delays: list[float] = []

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

        async def sleep(self, delay: float) -> None:
            self.sleep_delays.append(delay)
            await self._real.sleep(delay)

    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._capture_on = False

    fake_asyncio = _RecordingAsyncio(asyncio)
    monkeypatch.setattr(console_mod, "asyncio", fake_asyncio)

    task = asyncio.ensure_future(stream.record_loop())
    await asyncio.sleep(0.35)
    stream._stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    robot.media.get_audio_sample.assert_not_called()
    handler.receive.assert_not_awaited()
    assert 1 <= len(fake_asyncio.sleep_delays) <= 20
    assert all(delay == 0.1 for delay in fake_asyncio.sleep_delays)


class _FakeSessionHandler:
    """A minimal handler standing in for ConversationHandler in pre-roll tests.

    ``connection`` flips to a real object as soon as start_up() begins --
    mirroring huggingface_realtime.py setting self.connection right after the
    handshake -- and back to None once the session ends: either because the
    test calls shutdown(), or because it sets `stopped` directly to stand in
    for the realtime side closing the connection on its own.
    """

    def __init__(self) -> None:
        self.connection: object | None = None
        self.received: list[tuple[int, Any]] = []
        self.last_activity_time = time.monotonic()
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def start_up(self) -> None:
        self.connection = object()
        self.started.set()
        await self.stopped.wait()
        self.connection = None

    async def shutdown(self) -> None:
        self.stopped.set()
        self.connection = None

    async def receive(self, frame: tuple[int, Any]) -> None:
        # Mirrors HuggingFaceRealtimeHandler.receive(): a no-op while
        # disconnected, real frames like get sent silently dropped. Without
        # this guard the fake would record frames record_loop already
        # forwards unconditionally while disconnected -- which the real
        # handler discards -- as if they had reached a live session.
        if self.connection is None:
            return
        await asyncio.sleep(0)  # a real send yields; let other tasks interleave here too
        self.received.append(frame)


async def _stop_fake_session_loop(stream: LocalStream, handler: _FakeSessionHandler, loop_task: Any) -> None:
    """Tear down a startup loop driven by a _FakeSessionHandler."""
    stream._stop_event.set()
    handler.stopped.set()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_the_preroll_buffer_holds_at_most_two_seconds() -> None:
    """Feed 5 s of audio through record_loop.

    At most PREROLL_SECONDS ends up buffered, and what remains is the most
    recent slice -- not whatever happened to arrive first.
    """
    sample_rate = 16000
    frame_seconds = 0.1
    frame_len = int(sample_rate * frame_seconds)
    total_frames = 50  # 5.0 s at 0.1 s/frame

    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=sample_rate), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.connection = None  # no realtime connection up -- the buffering path
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)

    remaining = [np.full(frame_len, i, dtype=np.int16) for i in range(total_frames)]

    def _next_frame() -> np.ndarray:
        frame = remaining.pop(0)
        if not remaining:
            stream._stop_event.set()
        return frame

    robot.media.get_audio_sample.side_effect = _next_frame

    await stream.record_loop()

    total_duration = sum(len(samples) / rate for rate, samples in stream._preroll)
    assert total_duration <= console_mod.PREROLL_SECONDS + 1e-9
    assert total_duration > console_mod.PREROLL_SECONDS - (2 * frame_seconds)

    kept_indices = [int(samples[0]) for _, samples in stream._preroll]
    assert kept_indices == list(range(total_frames - len(kept_indices), total_frames))


def test_append_preroll_bounds_stereo_channels_last_frames_by_sample_count() -> None:
    """The recorder's real frames are (num_samples, 2) -- channels-last.

    The duration bound must be measured in samples (axis 0, ``len()``), not
    miscounted from the 2-channel axis, or a stereo frame's duration would
    read as a fixed 2 samples regardless of how long it actually is.
    """
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    sample_rate = 16000

    frame = (sample_rate, np.zeros((1600, 2), dtype=np.float32))  # 0.1 s of stereo audio
    for _ in range(25):  # 2.5 s worth, well over the 2.0 s bound
        stream._append_preroll(frame)

    total_duration = sum(len(samples) / rate for rate, samples in stream._preroll)
    assert total_duration <= console_mod.PREROLL_SECONDS + 1e-9
    assert 19 <= len(stream._preroll) <= 20  # ~2.0 s / 0.1 s per frame, float-rounding tolerant


@pytest.mark.asyncio
async def test_open_with_preroll_flushes_the_buffer_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open with preroll and expect the buffer flushed in order, then emptied.

    conversation.session {"open": true, "preroll": true} feeds the buffered
    frames into the handler, in order, right after the connection comes up.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot)
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01

    frame_a = (16000, np.full(4, 1, dtype=np.int16))
    frame_b = (16000, np.full(4, 2, dtype=np.int16))
    stream._append_preroll(frame_a)
    stream._append_preroll(frame_b)

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await stream.open_session(preroll=True)
        await _wait_until(lambda: len(handler.received) >= 2)

        assert [r for r, _ in handler.received] == [16000, 16000]
        assert [s.tolist() for _, s in handler.received] == [frame_a[1].tolist(), frame_b[1].tolist()]
        assert len(stream._preroll) == 0
        assert stream._preroll_flush_pending is False
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


@pytest.mark.asyncio
async def test_record_loop_holds_live_frames_behind_an_armed_flush() -> None:
    """Expect a live frame not to reach the handler directly while a flush is armed.

    A real handler.receive() is a network send that can yield mid-call; if a
    live frame went straight through, it could overtake -- or land between --
    the pre-roll frames the flush hasn't sent yet, scrambling the utterance.
    """
    frame = np.full(4, 9, dtype=np.int16)
    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=16000), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.connection = object()  # a connection is already up
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._preroll_flush_pending = True
    stream._append_preroll((16000, np.full(4, 1, dtype=np.int16)))  # already queued for the flush
    robot.media.get_audio_sample.side_effect = _stop_after(stream, frame)

    await stream.record_loop()

    handler.receive.assert_not_awaited()  # not sent directly
    queued = [samples.tolist() for _, samples in stream._preroll]
    assert queued == [[1, 1, 1, 1], [9, 9, 9, 9]]  # queued behind the existing frame, in order


@pytest.mark.asyncio
async def test_frames_queued_behind_a_flush_stay_within_the_two_second_bound() -> None:
    """Route frames queued behind an in-flight flush through the same bounded append.

    If the drain does not keep up (here it never runs at all -- the worst
    case), the frames record_loop queues behind it must still be trimmed the
    same way everything else in the buffer is, so the 2 s bound holds
    regardless of which path added a frame.
    """
    sample_rate = 16000
    frame_seconds = 0.1
    frame_len = int(sample_rate * frame_seconds)
    total_frames = 50  # 5.0 s worth, well over the 2.0 s bound

    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=sample_rate), get_audio_sample=MagicMock())
    handler = MagicMock()
    handler.connection = object()  # already connected
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._preroll_flush_pending = True  # a flush is (nominally) in flight; nothing drains it here

    remaining = [np.full(frame_len, i, dtype=np.int16) for i in range(total_frames)]

    def _next_frame() -> np.ndarray:
        frame = remaining.pop(0)
        if not remaining:
            stream._stop_event.set()
        return frame

    robot.media.get_audio_sample.side_effect = _next_frame

    await stream.record_loop()

    handler.receive.assert_not_awaited()  # nothing sent directly while the flush is pending
    total_duration = sum(len(samples) / rate for rate, samples in stream._preroll)
    assert total_duration <= console_mod.PREROLL_SECONDS + 1e-9


@pytest.mark.asyncio
async def test_the_flush_drains_frames_queued_during_the_drain_in_order() -> None:
    """Expect a frame queued mid-drain to arrive too, after what was ahead of it.

    The drain does not stop until the buffer it is reading from is genuinely
    empty.
    """
    received: list[tuple[int, Any]] = []
    appended = False

    class _DrainProbeHandler:
        def __init__(self) -> None:
            self.connection = object()  # instance attribute: _backend_connected() reads vars(handler)

        async def receive(self_inner, frame: tuple[int, Any]) -> None:
            nonlocal appended
            received.append(frame)
            if not appended:
                appended = True
                # Stand in for record_loop queuing a live frame mid-drain --
                # via the same helper record_loop itself now uses, so the
                # timestamp bookkeeping stays consistent.
                stream._append_preroll((16000, np.full(4, 3, dtype=np.int16)))

    handler = _DrainProbeHandler()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(handler, robot)
    stream._preroll_flush_pending = True
    stream._append_preroll((16000, np.full(4, 1, dtype=np.int16)))
    stream._append_preroll((16000, np.full(4, 2, dtype=np.int16)))

    await stream._settle_preroll_on_connect()

    tags = [int(samples[0]) for _, samples in received]
    assert tags == [1, 2, 3]
    assert len(stream._preroll) == 0
    assert stream._preroll_flush_pending is False


@pytest.mark.asyncio
async def test_preroll_frames_and_live_frames_arrive_in_one_unbroken_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run record_loop and the startup loop concurrently, end to end.

    With a real yielding handler.receive(), frames must still arrive strictly
    in the order they were spoken -- pre-roll frames first, then whatever
    record_loop produced while the connection was coming up and the flush
    was draining, with nothing skipped, duplicated, or reordered.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    sample_rate = 16000
    total_frames = 8
    produced = {"n": 0}

    robot = _audio_robot(get_input_audio_samplerate=MagicMock(return_value=sample_rate), get_audio_sample=MagicMock())
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot)
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01

    def _next_frame() -> np.ndarray | None:
        # The first two frames arrive before the connection exists (the
        # pre-roll); the rest arrive only once it does, so they exercise the
        # "live frame during an active flush" path record_loop now guards.
        if handler.connection is None and produced["n"] >= 2:
            return None
        tag = produced["n"]
        produced["n"] += 1
        if tag >= total_frames:
            stream._stop_event.set()
            return None
        return np.full(4, tag, dtype=np.int16)

    robot.media.get_audio_sample.side_effect = _next_frame

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    record_task = asyncio.create_task(stream.record_loop())
    try:
        await _wait_until(lambda: produced["n"] >= 2)  # the pre-roll frames buffered
        await stream.open_session(preroll=True)
        await _wait_until(lambda: len(handler.received) >= total_frames)

        tags = [int(samples[0]) for _, samples in handler.received]
        assert tags == list(range(total_frames))
    finally:
        stream._stop_event.set()
        handler.stopped.set()
        for task in (loop_task, record_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_open_without_preroll_discards_the_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session without preroll and expect the buffer discarded, not flushed.

    A session opened for any other reason has no authorisation for the two
    seconds already buffered.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot)
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01
    stream._append_preroll((16000, np.full(4, 9, dtype=np.int16)))

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await stream.open_session()  # preroll defaults to False
        await _wait_until(lambda: handler.started.is_set())
        await _wait_until(lambda: len(stream._preroll) == 0)

        assert handler.received == []
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


@pytest.mark.asyncio
async def test_the_apps_own_reconnect_does_not_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expect no flush when the session ends on its own and the app reconnects.

    The gate stays set through that reconnect -- it is not the wake word
    opening a new session, so whatever has accumulated since must be
    discarded, not flushed.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot)
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01
    stream._append_preroll((16000, np.full(4, 1, dtype=np.int16)))

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await stream.open_session(preroll=True)
        await _wait_until(lambda: len(handler.received) >= 1)
        assert stream._preroll_flush_pending is False  # consumed by the first connect

        # The realtime side ends the session on its own -- not a
        # close_session() call, so the gate (_session_wanted) is never
        # touched, unlike a deliberate close/reopen.
        handler.started.clear()
        handler.stopped.set()
        await _wait_until(lambda: handler.connection is None)
        handler.stopped.clear()
        assert stream._session_wanted.is_set() is True  # gate untouched

        stream._append_preroll((16000, np.full(4, 2, dtype=np.int16)))
        await _wait_until(lambda: handler.started.is_set())
        await _wait_until(lambda: len(stream._preroll) == 0)

        assert len(handler.received) == 1  # the second frame was never flushed
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


@pytest.mark.asyncio
async def test_preroll_on_an_already_open_session_arms_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expect preroll=True on an already-open gate to be a no-op for the flag.

    An already-open session has no "before it connected" left to authorise.
    If it armed the flag anyway, a later reconnect after this session ends on
    its own would flush audio the wake word never asked for, which is
    exactly what the "app's own reconnect" case above forbids.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot)  # gate starts open (the default)
    stream._backend_retry_delay = 0.01

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await _wait_until(lambda: handler.started.is_set())

        await stream.open_session(preroll=True)  # already open: must not arm anything
        assert stream._preroll_flush_pending is False

        stream._append_preroll((16000, np.full(4, 7, dtype=np.int16)))

        # The session ends on its own and the app reconnects.
        handler.started.clear()
        handler.stopped.set()
        await _wait_until(lambda: handler.connection is None)
        handler.stopped.clear()

        await _wait_until(lambda: handler.started.is_set())
        await _wait_until(lambda: len(stream._preroll) == 0)

        assert handler.received == []  # nothing was ever flushed
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


class _FlakySessionHandler(_FakeSessionHandler):
    """Fails the first start_up() attempt before ever connecting, then behaves normally."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def start_up(self) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("transient connect failure")
        await super().start_up()


@pytest.mark.asyncio
async def test_a_failed_connect_does_not_leave_the_flush_flag_armed_for_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The authorisation covers the attempt open_session(preroll=True) caused.

    If that attempt fails before handler.connection is ever set, a later
    retry must not inherit the flag and flush whatever has accumulated by
    the time it connects -- that connection was never actually asked for.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FlakySessionHandler()
    stream = LocalStream(handler, robot)
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01
    stream._append_preroll((16000, np.full(4, 4, dtype=np.int16)))

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await stream.open_session(preroll=True)
        await _wait_until(lambda: handler.attempts >= 1)
        await _wait_until(lambda: stream._preroll_flush_pending is False)  # cleared despite the failure

        await _wait_until(lambda: handler.started.is_set())  # the retry connects
        await _wait_until(lambda: len(stream._preroll) == 0)  # discarded, not flushed

        assert handler.received == []
        assert handler.attempts >= 2
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


@pytest.mark.asyncio
async def test_a_hard_age_limit_drops_stale_buffered_frames_at_flush_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expect a frame older than PREROLL_SECONDS by flush time to be dropped, not sent.

    The duration bound only limits how much audio accumulates; it does not
    expire with wall-clock time on its own. With no new frames arriving to
    trim it, a buffered frame can sit well past PREROLL_SECONDS in real time
    while still reading as "within budget" by duration alone -- the flush
    must catch this with each frame's own arrival timestamp instead.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    fake_now = [0.0]
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot, preroll_clock=lambda: fake_now[0])
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01
    stream._append_preroll((16000, np.full(4, 1, dtype=np.int16)))

    fake_now[0] += console_mod.PREROLL_SECONDS + 1.0  # 3.0 s later, nothing new arrived

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await stream.open_session(preroll=True)
        await _wait_until(lambda: handler.started.is_set())
        await _wait_until(lambda: stream._preroll_flush_pending is False)

        assert handler.received == []  # too old by the time it would have flushed
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


def test_capture_off_clears_the_preroll_buffer_immediately() -> None:
    """No audio from before a capture-off may survive into a later session."""
    app = FastAPI()
    robot = _audio_robot(
        stop_recording=MagicMock(),
        start_recording=MagicMock(),
        stop_playing=MagicMock(),
        start_playing=MagicMock(),
    )
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    stream._preroll.append((16000, np.zeros(4, dtype=np.int16)))
    stream._preroll_duration = 4 / 16000

    assert _rpc_call(app, "conversation.capture", {"on": False})["result"] == {"on": False}

    assert len(stream._preroll) == 0
    assert stream._preroll_duration == 0.0


@pytest.mark.asyncio
async def test_the_preroll_buffer_stays_empty_while_capture_is_off() -> None:
    """Drive record_loop with capture off, a pre-existing buffer, and frames available.

    Confirms record_loop's own off-branch actively empties the buffer (and
    keeps it empty), not just that the capture-off RPC clears whatever was
    there at the moment it was called.
    """
    robot = _audio_robot(
        get_input_audio_samplerate=MagicMock(return_value=16000),
        get_audio_sample=MagicMock(return_value=np.zeros(4, dtype=np.int16)),
    )
    handler = MagicMock()
    handler.connection = None
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._capture_on = False
    stream._preroll.append((16000, np.full(4, 5, dtype=np.int16)))  # left over from before capture went off
    stream._preroll_duration = 4 / 16000

    task = asyncio.ensure_future(stream.record_loop())
    await asyncio.sleep(0.05)
    stream._stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    robot.media.get_audio_sample.assert_not_called()  # off means off -- no polling either
    assert len(stream._preroll) == 0
    assert stream._preroll_duration == 0.0


def test_muting_clears_the_preroll_buffer_immediately() -> None:
    """No audio from before a mute may sit in the buffer and later flush as if it were fresh."""
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    stream = LocalStream(MagicMock(), robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    stream._append_preroll((16000, np.zeros(4, dtype=np.int16)))

    assert _rpc_call(app, "conversation.mic", {"muted": True})["result"] == {"muted": True}

    assert len(stream._preroll) == 0
    assert stream._preroll_duration == 0.0


@pytest.mark.asyncio
async def test_the_preroll_buffer_stays_empty_while_muted() -> None:
    """Drive record_loop muted, with a pre-existing buffer and frames available.

    Confirms record_loop's own mute branch actively empties the buffer (and
    keeps it empty), mirroring the capture-off branch above -- duration is
    tracked in audio time, not wall time, so nothing would otherwise age a
    frozen buffer out on its own for as long as the mute lasts.
    """
    robot = _audio_robot(
        get_input_audio_samplerate=MagicMock(return_value=16000),
        get_audio_sample=MagicMock(return_value=np.zeros(4, dtype=np.int16)),
    )
    handler = MagicMock()
    handler.connection = None
    handler.receive = AsyncMock()
    stream = LocalStream(handler, robot)
    stream._mic_muted = True
    stream._append_preroll((16000, np.full(4, 5, dtype=np.int16)))  # left over from before the mute

    task = asyncio.ensure_future(stream.record_loop())
    await asyncio.sleep(0.05)
    stream._stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    handler.receive.assert_not_awaited()
    assert len(stream._preroll) == 0
    assert stream._preroll_duration == 0.0


@pytest.mark.asyncio
async def test_mute_then_reopen_with_preroll_flushes_only_post_unmute_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: audio buffered before a mute must never reach the handler.

    Not directly, and not through a later preroll flush either -- only audio
    buffered after the unmute may.
    """
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    sample_rate = 16000
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = _FakeSessionHandler()
    stream = LocalStream(handler, robot)
    stream._session_wanted.clear()
    stream._backend_retry_delay = 0.01

    # Pre-mute audio buffers while disconnected (AMBIENT); then the mic is
    # muted, which clears it (what conversation.mic {"muted": true} does).
    stream._append_preroll((sample_rate, np.full(4, 1, dtype=np.int16)))
    stream._mic_muted = True
    stream._clear_preroll()

    loop_task = asyncio.create_task(stream._run_handler_startup_loop())
    try:
        await stream.open_session(preroll=True)  # the wake word, still muted
        await _wait_until(lambda: handler.started.is_set())
        await _wait_until(lambda: stream._preroll_flush_pending is False)
        assert handler.received == []  # nothing pre-mute was ever flushed

        # Back to AMBIENT, unmute, fresh audio buffers, wake word again.
        assert await stream.close_session() is True
        stream._mic_muted = False
        stream._append_preroll((sample_rate, np.full(4, 2, dtype=np.int16)))
        handler.stopped.clear()
        await stream.open_session(preroll=True)
        await _wait_until(lambda: len(handler.received) >= 1)

        tags = [int(s[0]) for _, s in handler.received]
        assert tags == [2]  # only the post-unmute frame, never the pre-mute one
    finally:
        await _stop_fake_session_loop(stream, handler, loop_task)


@pytest.mark.asyncio
async def test_closing_the_session_clears_the_preroll_buffer() -> None:
    """Closing the session clears the buffer -- and cancels any pending flush."""
    stream = _gate_stream()
    stream.handler.shutdown = AsyncMock()
    stream._session_parked.set()
    stream._preroll.append((16000, np.zeros(4, dtype=np.int16)))
    stream._preroll_duration = 4 / 16000
    stream._preroll_flush_pending = True

    await stream.close_session()

    assert len(stream._preroll) == 0
    assert stream._preroll_duration == 0.0
    assert stream._preroll_flush_pending is False


def test_conversation_session_open_without_preroll_is_unchanged() -> None:
    """Expect a caller that omits 'preroll' to get the prior behaviour byte for byte.

    Omitting the parameter must also never arm the pre-roll flush.
    """
    app = FastAPI()
    robot = SimpleNamespace(media=SimpleNamespace(audio=None, backend=None))
    handler = MagicMock()
    handler.connection = None
    handler.last_activity_time = time.monotonic()
    handler.shutdown = AsyncMock()
    stream = LocalStream(handler, robot, settings_app=app)
    stream._init_settings_ui_if_needed()
    stream._session_parked.set()
    stream._session_wanted.clear()

    result = _rpc_call(app, "conversation.session", {"open": True})["result"]

    assert result == {"wanted": True, "connected": False}
    assert stream._preroll_flush_pending is False


@pytest.mark.asyncio
async def test_play_loop_logs_text_outputs() -> None:
    """Text outputs are logged, not pushed to the speaker."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    output = AdditionalOutputs({"role": "assistant", "content": "hi"})
    handler.emit = AsyncMock(side_effect=_stop_after(stream, output))

    await stream.play_loop()

    robot.media.push_audio_sample.assert_not_called()


@pytest.mark.asyncio
async def test_play_loop_pushes_mono_audio_as_float32() -> None:
    """A mono int16 frame is pushed to the speaker as float32."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    handler.emit = AsyncMock(side_effect=_stop_after(stream, (24000, np.zeros(4, dtype=np.int16))))

    await stream.play_loop()

    robot.media.push_audio_sample.assert_called_once()
    pushed = robot.media.push_audio_sample.call_args.args[0]
    assert pushed.ndim == 1
    assert pushed.dtype == np.float32


@pytest.mark.asyncio
async def test_play_loop_downmixes_stereo_before_pushing() -> None:
    """A stereo frame is reduced to a single mono channel before playback."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    stereo = np.zeros((4, 2), dtype=np.int16)
    handler.emit = AsyncMock(side_effect=_stop_after(stream, (24000, stereo)))

    await stream.play_loop()

    pushed = robot.media.push_audio_sample.call_args.args[0]
    assert pushed.ndim == 1


@pytest.mark.asyncio
async def test_play_loop_skips_empty_audio() -> None:
    """An empty audio frame is skipped, not pushed."""
    robot = _audio_robot(push_audio_sample=MagicMock())
    handler = MagicMock()
    stream = LocalStream(handler, robot)
    handler.emit = AsyncMock(side_effect=_stop_after(stream, (24000, np.array([], dtype=np.int16))))

    await stream.play_loop()

    robot.media.push_audio_sample.assert_not_called()


def test_close_without_running_loop_stops_media() -> None:
    """Closing without a running loop stops the media pipelines and sets the stop event."""
    robot = _audio_robot(stop_recording=MagicMock(), stop_playing=MagicMock())
    stream = LocalStream(MagicMock(), robot)
    stream._asyncio_loop = None

    stream.close()

    robot.media.stop_recording.assert_called_once()
    robot.media.stop_playing.assert_called_once()
    assert stream._stop_event.is_set()


def test_drain_output_queue_empties_in_place() -> None:
    """The output queue is drained without being replaced."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    queue.put_nowait("a")
    queue.put_nowait("b")
    handler = MagicMock()
    handler.output_queue = queue
    stream = LocalStream(handler, _audio_robot())

    stream._drain_output_queue()

    assert stream.handler.output_queue is queue
    assert queue.empty()


def test_drain_output_queue_tolerates_missing_queue() -> None:
    """Draining is a no-op when the handler has no output queue."""
    handler = MagicMock()
    handler.output_queue = None
    stream = LocalStream(handler, _audio_robot())

    stream._drain_output_queue()  # must not raise


def _rpc_robot() -> SimpleNamespace:
    """Return a robot mock whose audio pipeline supports clear_audio_queue()."""
    audio = SimpleNamespace(clear_player=MagicMock(), clear_output_buffer=MagicMock())
    return SimpleNamespace(media=SimpleNamespace(audio=audio))


def test_rpc_status_and_mic_over_websocket() -> None:
    """conversation.status/mic are reachable over the /rpc JSON-RPC WebSocket."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    client = TestClient(app)
    with client.websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "conversation.status"})
        resp = ws.receive_json()
        assert resp["id"] == "1"
        assert "result" in resp

        ws.send_json({"jsonrpc": "2.0", "id": "2", "method": "conversation.mic", "params": {"muted": True}})
        resp = ws.receive_json()
        assert resp["result"] == {"muted": True}
    assert stream._mic_muted is True


def test_rpc_interrupt_broadcasts_turn_listening() -> None:
    """conversation.interrupt clears playback and pushes a turn:listening event."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler._is_connected.return_value = True
    app = FastAPI()
    stream = LocalStream(handler, _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "conversation.interrupt"})
        msgs = [ws.receive_json(), ws.receive_json()]
    results = [m for m in msgs if "result" in m]
    notes = [m for m in msgs if m.get("method") == "conversation.turn"]
    assert results and results[0]["result"] == {"ok": True}
    assert notes and notes[0]["params"] == {"state": "listening", "reason": "interrupted"}


def test_rpc_say_requires_active_session() -> None:
    """conversation.say fails with not_running when no session is connected."""
    handler = MagicMock()
    handler._is_connected.return_value = False
    app = FastAPI()
    stream = LocalStream(handler, _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "conversation.say", "params": {"text": "hi"}})
        resp = ws.receive_json()
    assert resp["error"]["data"]["reason"] == "not_running"


def test_rpc_transcript_notification_broadcast() -> None:
    """The handler's transcript observer pushes conversation.transcript events."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        stream._dispatch_transcript("assistant", "hello there", True)
        msg = ws.receive_json()
    assert msg["method"] == "conversation.transcript"
    assert msg["params"] == {"role": "assistant", "text": "hello there", "final": True}


def test_rpc_settings_methods() -> None:
    """Personality, voice, and tool settings are reachable over /rpc."""
    app = FastAPI()
    stream = LocalStream(MagicMock(), _rpc_robot(), settings_app=app)
    stream._init_settings_ui_if_needed()
    with TestClient(app).websocket_connect("/rpc") as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "1", "method": "personalities.list"})
        r1 = ws.receive_json()
        ws.send_json({"jsonrpc": "2.0", "id": "2", "method": "voices.list"})
        r2 = ws.receive_json()
        ws.send_json({"jsonrpc": "2.0", "id": "3", "method": "tool_spaces.list"})
        r3 = ws.receive_json()
        ws.send_json({"jsonrpc": "2.0", "id": "4", "method": "profile_tools.get"})
        r4 = ws.receive_json()
    assert "choices" in r1["result"] and "current" in r1["result"]
    assert isinstance(r2["result"], list)
    assert "spaces" in r3["result"]
    assert "enabled_tools" in r4["result"]
