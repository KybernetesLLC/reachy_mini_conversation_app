import math
import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import (
    HoldPoseMove,
    BreathingMove,
    MovementManager,
    LoopFrequencyStats,
    clone_full_body_pose,
)
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove, EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_clone_full_body_pose_is_a_deep_copy() -> None:
    """Cloning a pose must not alias the head-pose array of the original."""
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    original = (head, (1.0, 2.0), 3.0)
    clone = clone_full_body_pose(original)
    head[0, 0] = 999.0
    assert clone[0][0, 0] != 999.0
    assert clone[1] == (1.0, 2.0)
    assert clone[2] == 3.0


def test_loop_frequency_stats_reset_keeps_last_potential() -> None:
    """Reset clears accumulators but preserves last/potential frequency."""
    stats = LoopFrequencyStats(mean=5.0, m2=2.0, min_freq=1.0, count=10, last_freq=59.0, potential_freq=61.0)
    stats.reset()
    assert stats.mean == 0.0
    assert stats.m2 == 0.0
    assert stats.count == 0
    assert stats.min_freq == float("inf")
    assert stats.last_freq == 59.0
    assert stats.potential_freq == 61.0


def test_breathing_move_interpolates_then_breathes() -> None:
    """Phase 1 starts at the given antennas; phase 2 keeps body yaw neutral."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(0.3, -0.3),
        interpolation_duration=1.0,
    )
    head_start, antennas_start, body_yaw_start = move.evaluate(0.0)
    assert head_start is not None
    np.testing.assert_allclose(antennas_start, [0.3, -0.3])
    assert body_yaw_start == 0.0

    head_breathe, antennas_breathe, body_yaw_breathe = move.evaluate(5.0)
    assert head_breathe is not None
    assert antennas_breathe is not None and antennas_breathe.shape == (2,)
    assert body_yaw_breathe == 0.0


def test_hold_pose_move_interpolates_then_holds_exactly() -> None:
    """Hold reaches the target at `duration` and stays there indefinitely after, sway-free."""
    start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    target_pose = create_head_pose(0, 0, 0.02, 0, 10, 0, degrees=True)
    move = HoldPoseMove(
        target_head_pose=target_pose,
        target_antennas=(0.2, -0.2),
        interpolation_start_pose=start_pose,
        interpolation_start_antennas=(0.0, 0.0),
        interpolation_duration=2.0,
        interpolation_start_body_yaw=0.3,
    )
    assert move.duration == float("inf")

    head_start, antennas_start, body_yaw_start = move.evaluate(0.0)
    np.testing.assert_array_equal(head_start, start_pose)
    np.testing.assert_array_equal(antennas_start, [0.0, 0.0])
    assert body_yaw_start == 0.3

    # Body yaw is not part of the RPC target, but it still interpolates
    # smoothly to neutral rather than snapping -- avoiding a jump if a hold
    # is commanded while the manager's last commanded pose had some body yaw.
    _, _, body_yaw_mid = move.evaluate(1.0)
    assert body_yaw_mid == pytest.approx(0.15)

    head_end, antennas_end, body_yaw_end = move.evaluate(2.0)
    np.testing.assert_array_equal(head_end, target_pose)
    np.testing.assert_array_equal(antennas_end, [0.2, -0.2])
    assert body_yaw_end == 0.0

    head_later, antennas_later, body_yaw_later = move.evaluate(62.0)
    np.testing.assert_array_equal(head_later, target_pose)
    np.testing.assert_array_equal(antennas_later, [0.2, -0.2])
    assert body_yaw_later == 0.0


def test_hold_blocks_breathing_and_release_lets_it_resume() -> None:
    """A held pose blocks idle breathing; releasing it lets breathing resume after the delay.

    Drives the manager's own decision functions (_manage_move_queue /
    _manage_breathing) with a controllable clock rather than the worker
    thread, per the brief's fallback for a manager that only runs on its
    own thread.
    """
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    robot.get_current_head_pose.return_value = np.eye(4)
    manager = MovementManager(robot)

    target_head_pose = create_head_pose(0, 0, 0, 0, 10, 0, degrees=True)
    t0 = manager._now()
    manager._handle_command("hold_pose", (target_head_pose, (0.2, -0.2), 1.0), t0)
    manager._manage_move_queue(t0)  # promote the queued hold to the current move
    assert isinstance(manager.state.current_move, HoldPoseMove)

    # Well past both the hold's own interpolation and the idle-inactivity delay:
    # breathing must not start while the hold is current.
    t_holding = t0 + manager.idle_inactivity_delay + 5.0
    manager._update_primary_motion(t_holding)
    assert isinstance(manager.state.current_move, HoldPoseMove)
    assert len(manager.move_queue) == 0
    _, antennas, _ = manager.state.current_move.evaluate(t_holding - t0)
    np.testing.assert_array_equal(antennas, [0.2, -0.2])

    # Positive control: releasing hands control back to idle behaviour, and
    # breathing starts once its own inactivity delay elapses from the release.
    manager._handle_command("release_hold", None, t_holding)
    assert manager.state.current_move is None
    assert len(manager.move_queue) == 0

    t_after_release = manager._now() + manager.idle_inactivity_delay + 1.0
    manager._update_primary_motion(t_after_release)
    assert len(manager.move_queue) == 1
    assert isinstance(manager.move_queue[0], BreathingMove)


def test_hold_pose_antennas_bypass_the_listening_freeze() -> None:
    """A held pose's own antennas win over a stuck listening freeze.

    _calculate_blended_antennas normally commands the frozen listening
    snapshot while _is_listening is True (see the positive control below).
    That freeze can outlive the session that set it -- nothing clears it on
    shutdown between speech_started and speech_stopped, and a same-window
    set_listening(False) is dropped by its own debounce -- so a hold must
    win over it directly rather than by depending on the flag ever clearing.
    """
    manager = MovementManager(MagicMock())
    manager._is_listening = True
    manager._listening_antennas = (0.05, -0.05)

    # Positive control: without a hold, listening still freezes antennas at
    # the snapshot, regardless of what the current move is commanding.
    frozen = manager._calculate_blended_antennas((0.3, -0.3))
    assert frozen == (0.05, -0.05)

    now = manager._now()
    target_head_pose = create_head_pose(0, 0, 0, 0, 10, 0, degrees=True)
    manager._handle_command("hold_pose", (target_head_pose, (0.2, -0.2), 1.0), now)
    manager._manage_move_queue(now)  # promote the queued hold to the current move
    assert isinstance(manager.state.current_move, HoldPoseMove)

    # Drive a tick well past the hold's own interpolation, with the listening
    # freeze still set: the commanded antennas must be the hold's target, not
    # the frozen snapshot.
    t_holding = now + 5.0
    manager._update_primary_motion(t_holding)
    _, antennas, _ = manager._get_primary_pose(t_holding)
    antennas_cmd = manager._calculate_blended_antennas((float(antennas[0]), float(antennas[1])))
    assert antennas_cmd == (0.2, -0.2)

    # Bypassing the freeze must not mutate listening state itself: it stays
    # truthful for whatever else reads it once the hold releases.
    assert manager._is_listening is True
    assert manager._listening_antennas == (0.05, -0.05)


def test_release_hold_reseeds_the_listening_freeze_to_avoid_a_jump() -> None:
    """Releasing a hold while still listening re-freezes at the hold's antennas, not a stale snapshot.

    Without the reseed, the freeze's own snapshot predates the hold (the hold
    bypasses it entirely -- see test_hold_pose_antennas_bypass_the_listening_freeze
    above) and falling through to it unchanged would snap the antennas back
    to a stale pre-hold position in one tick, violating moves.py's own
    "avoid jumps at all times" invariant.
    """
    manager = MovementManager(MagicMock())
    manager._is_listening = True
    manager._listening_antennas = (0.05, -0.05)  # stale, pre-hold snapshot

    now = manager._now()
    target_head_pose = create_head_pose(0, 0, 0, 0, 10, 0, degrees=True)
    manager._handle_command("hold_pose", (target_head_pose, (0.2, -0.2), 1.0), now)
    manager._manage_move_queue(now)  # promote the queued hold to the current move
    assert isinstance(manager.state.current_move, HoldPoseMove)

    # Tick past the hold's own interpolation: the bypass (see the test above)
    # commands the hold's target, and _issue_control_command records it as
    # the last-commanded pose, exactly as working_loop's real per-tick order
    # would.
    t_holding = now + 5.0
    manager._update_primary_motion(t_holding)
    head, antennas, body_yaw = manager._get_primary_pose(t_holding)
    antennas_cmd = manager._calculate_blended_antennas((float(antennas[0]), float(antennas[1])))
    assert antennas_cmd == (0.2, -0.2)
    manager._issue_control_command(head, antennas_cmd, body_yaw)

    # Release while still listening.
    manager._handle_command("release_hold", None, t_holding)
    assert manager.state.current_move is None
    assert manager._is_listening is True  # release does not clear the flag itself

    # One more tick, still listening: no jump -- the freeze now holds the
    # hold's own last-commanded antennas, not the stale (0.05, -0.05)
    # snapshot from before the hold. Whatever the next move/idle target
    # would be (here a plausible neutral) is irrelevant while listening.
    next_target = (-0.1745, 0.1745)
    antennas_after_release = manager._calculate_blended_antennas(next_target)
    assert antennas_after_release == (0.2, -0.2)

    # Clearing listening lets the existing blend run, gradually, from the
    # true (re-seeded) position toward the next target -- not a further jump
    # in either direction. _last_listening_blend_time/_antenna_unfreeze_blend
    # are set directly (bypassing set_listening's own real-time debounce) so
    # the elapsed blend time is deterministic rather than dependent on how
    # fast this test happens to run.
    manager._is_listening = False
    manager._last_listening_blend_time = manager._now() - (manager._antenna_blend_duration / 2)
    antennas_mid_blend = manager._calculate_blended_antennas(next_target)
    assert antennas_mid_blend != antennas_after_release
    assert antennas_mid_blend != next_target
    assert -0.1745 < antennas_mid_blend[0] < 0.2
    assert -0.2 < antennas_mid_blend[1] < 0.1745


def test_release_hold_is_a_no_op_when_nothing_is_held() -> None:
    """Releasing with no hold current leaves any other current move untouched."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    breathing_move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(0.0, 0.0),
    )
    manager.state.current_move = breathing_move
    manager.state.move_start_time = now
    manager._breathing_active = True

    manager._handle_command("release_hold", None, now)

    assert manager.state.current_move is breathing_move
    assert manager._breathing_active is True


def test_hold_pose_seeds_from_last_commanded_pose_and_replaces_a_previous_hold() -> None:
    """A second hold interpolates from wherever the manager last commanded, not the first target."""
    manager = MovementManager(MagicMock())
    now = manager._now()

    first_target = create_head_pose(0, 0, 0, 0, 10, 0, degrees=True)
    manager._handle_command("hold_pose", (first_target, (0.2, -0.2), 1.0), now)
    manager._manage_move_queue(now)  # promote the queued hold to the current move
    first_hold = manager.state.current_move
    assert isinstance(first_hold, HoldPoseMove)

    # Simulate the control loop having commanded the pose partway through the hold,
    # including a non-zero body yaw (e.g. left over from a dance move).
    commanded_head = create_head_pose(0, 0, 0, 0, 4, 0, degrees=True)
    manager._last_commanded_pose = (commanded_head, (0.08, -0.08), 0.3)

    second_target = create_head_pose(0, 0, 0, 0, -5, 0, degrees=True)
    manager._handle_command("hold_pose", (second_target, (-0.1, 0.1), 1.0), now + 0.5)
    manager._manage_move_queue(now + 0.5)

    assert manager.state.current_move is not first_hold
    second_hold = manager.state.current_move
    assert isinstance(second_hold, HoldPoseMove)
    assert len(manager.move_queue) == 0
    np.testing.assert_array_equal(second_hold.interpolation_start_pose, commanded_head)
    np.testing.assert_array_equal(second_hold.interpolation_start_antennas, [0.08, -0.08])
    assert second_hold.interpolation_start_body_yaw == 0.3
    # The companion (decision 026): a hold given no body yaw keeps the last one.
    assert second_hold.target_body_yaw == 0.3
    np.testing.assert_array_equal(second_hold.target_head_pose, second_target)


def test_is_holding_reports_current_or_queued_hold() -> None:
    """is_holding is true for a current hold, false once released, with no other move disturbed."""
    manager = MovementManager(MagicMock())
    assert manager.is_holding() is False

    now = manager._now()
    target = create_head_pose(0, 0, 0, 0, 10, 0, degrees=True)
    manager._handle_command("hold_pose", (target, (0.2, -0.2), 1.0), now)
    assert manager.is_holding() is True

    manager._handle_command("release_hold", None, now)
    assert manager.is_holding() is False


def test_is_idle_reflects_listening_and_activity() -> None:
    """is_idle is False while listening and True once past the inactivity delay."""
    manager = MovementManager(MagicMock())

    manager._shared_is_listening = True
    assert manager.is_idle() is False

    manager._shared_is_listening = False
    manager._shared_last_activity_time = manager._now()
    assert manager.is_idle() is False

    manager._shared_last_activity_time = manager._now() - 10.0
    assert manager.is_idle() is True


def test_handle_command_queue_and_clear() -> None:
    """queue_move appends real moves, ignores bad payloads, and clear empties the queue."""
    manager = MovementManager(MagicMock())
    now = manager._now()
    move = GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))

    manager._handle_command("queue_move", move, now)
    assert list(manager.move_queue) == [move]

    manager._handle_command("queue_move", "not-a-move", now)
    assert list(manager.move_queue) == [move]

    manager._handle_command("clear_queue", None, now)
    assert len(manager.move_queue) == 0
    assert manager.state.current_move is None


def _move() -> BreathingMove:
    return BreathingMove(
        interpolation_start_pose=np.eye(4),
        interpolation_start_antennas=[-0.1745, 0.1745],
        interpolation_duration=1.0,
    )


def test_the_idle_sway_is_calm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Breathing move uses calm defaults when environment variables are unset."""
    for name in ("REACHY_BREATHING_ANTENNA_DEG", "REACHY_BREATHING_ANTENNA_HZ", "REACHY_BREATHING_Z_MM"):
        monkeypatch.delenv(name, raising=False)
    move = _move()
    assert move.antenna_sway_amplitude == pytest.approx(math.radians(2.9))
    assert move.antenna_frequency == pytest.approx(0.25)
    assert move.breathing_z_amplitude == pytest.approx(0.003)


def test_the_sway_can_be_tuned_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Breathing move reads tuning parameters from environment variables."""
    monkeypatch.setenv("REACHY_BREATHING_ANTENNA_DEG", "2")
    monkeypatch.setenv("REACHY_BREATHING_ANTENNA_HZ", "0.2")
    monkeypatch.setenv("REACHY_BREATHING_Z_MM", "0")
    move = _move()
    assert move.antenna_sway_amplitude == pytest.approx(math.radians(2.0))
    assert move.antenna_frequency == pytest.approx(0.2)
    assert move.breathing_z_amplitude == 0.0


@pytest.mark.parametrize("bad", ["", "lots", "-3", "nan", "inf"])
def test_a_bad_value_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """Invalid environment values (empty, non-numeric, negative, non-finite) fall back to defaults."""
    monkeypatch.setenv("REACHY_BREATHING_ANTENNA_DEG", bad)
    assert _move().antenna_sway_amplitude == pytest.approx(math.radians(2.9))


def test_the_sway_never_takes_an_antenna_past_2_9_degrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap the sway at 2.9 degrees off vertical (the owner, 2026-10-10).

    The right antenna overloaded against its stop past that. A wider setting is capped.
    """
    monkeypatch.setenv("REACHY_BREATHING_ANTENNA_DEG", "8")
    move = _move()
    assert move.antenna_sway_amplitude == pytest.approx(math.radians(2.9))
    for t in (0.0, 1.0, 2.0, 3.0, 5.5):
        _, antennas, _ = move.evaluate(t + move.interpolation_duration)
        assert max(abs(a) for a in antennas) <= math.radians(2.9) + 1e-9


# --- the companion's facing (decision 026) ------------------------------------

from scipy.spatial.transform import Rotation as R  # noqa: E402


def _yaw_of(pose: np.ndarray) -> float:
    return float(R.from_matrix(pose[:3, :3]).as_euler("xyz")[2])


def test_breathing_holds_the_body_where_it_began() -> None:
    """Breathing keeps the body yaw it began with, and the head faces that way.

    The companion (decision 026): breathing used to return body yaw 0, a
    snap back to straight ahead.
    """
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0.4, degrees=False),
        interpolation_start_antennas=(-0.1745, 0.1745),
        body_yaw=0.4,
    )
    head0, _, yaw0 = move.evaluate(0.0)
    np.testing.assert_allclose(head0, create_head_pose(0, 0, 0, 0, 0, 0.4, degrees=False), atol=1e-9)
    assert yaw0 == pytest.approx(0.4)
    head5, _, yaw5 = move.evaluate(5.0)
    assert yaw5 == pytest.approx(0.4)
    assert _yaw_of(head5) == pytest.approx(0.4)


def test_a_turn_while_breathing_is_min_jerk_and_ends_on_target() -> None:
    """A turn eases in and out, and the body stays at the target afterwards."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(-0.1745, 0.1745),
    )
    move.turn_body(0.6, 2.0, 3.0)
    assert move.evaluate(3.0)[2] == pytest.approx(0.0)
    assert move.evaluate(3.2)[2] < 0.6 * 0.1  # slower than a straight line at the start
    assert move.evaluate(4.0)[2] == pytest.approx(0.3)  # min-jerk is half way at half time
    assert move.evaluate(5.0)[2] == pytest.approx(0.6)
    head, _, yaw = move.evaluate(60.0)
    assert yaw == pytest.approx(0.6) and _yaw_of(head) == pytest.approx(0.6)


def test_a_new_turn_starts_from_where_the_body_is() -> None:
    """A turn asked for mid-turn starts from the body's yaw at that moment."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(-0.1745, 0.1745),
    )
    move.turn_body(0.6, 2.0, 0.0)
    move.turn_body(-0.2, 2.0, 1.0)
    assert move.evaluate(1.0)[2] == pytest.approx(0.3)
    assert move.evaluate(3.0)[2] == pytest.approx(-0.2)


def _robot(body_yaw: float = 0.0) -> MagicMock:
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([body_yaw] + [0.0] * 6, [-0.1745, 0.1745])
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, body_yaw, degrees=False)
    return robot


def test_breathing_starts_from_the_robots_present_body_yaw() -> None:
    """Idle breathing reads the body yaw from joint 0 when it begins."""
    manager = MovementManager(_robot(0.5))
    t = manager._now() + manager.idle_inactivity_delay + 1.0
    manager._update_primary_motion(t)
    (move,) = manager.move_queue
    assert isinstance(move, BreathingMove) and move.start_body_yaw == pytest.approx(0.5)


def test_an_idle_turn_turns_the_breathing_body() -> None:
    """turn_idle_body applies to the current breathing move at once."""
    manager = MovementManager(_robot())
    t = manager._now() + manager.idle_inactivity_delay + 1.0
    manager._update_primary_motion(t)
    manager._manage_move_queue(t)
    assert isinstance(manager.state.current_move, BreathingMove)
    manager._handle_command("turn_idle_body", (0.6, 2.0), t + 1.0)
    assert manager.state.current_move.evaluate(3.0)[2] == pytest.approx(0.6)


def test_a_turn_before_breathing_starts_waits_for_it() -> None:
    """A turn asked for before breathing begins starts with the next breathing."""
    manager = MovementManager(_robot())
    t0 = manager._now()
    manager._handle_command("turn_idle_body", (0.6, 2.0), t0)
    assert manager._pending_turn == (0.6, 2.0)
    t = t0 + manager.idle_inactivity_delay + 1.0
    manager._update_primary_motion(t)
    manager._manage_move_queue(t)
    move = manager.state.current_move
    assert isinstance(move, BreathingMove) and manager._pending_turn is None
    assert move.evaluate(2.0)[2] == pytest.approx(0.6)


def test_a_hold_keeps_the_body_yaw_unless_given_one() -> None:
    """A hold without a body yaw keeps the last commanded one."""
    manager = MovementManager(_robot())
    manager._last_commanded_pose = (np.eye(4), (0.0, 0.0), 0.4)
    t = manager._now()
    manager._handle_command("hold_pose", (np.eye(4), (0.2, -0.2), 1.0), t)
    assert manager.move_queue[-1].target_body_yaw == pytest.approx(0.4)
    manager._handle_command("hold_pose", (np.eye(4), (0.2, -0.2), 1.0, 0.1), t)
    assert manager.move_queue[-1].target_body_yaw == pytest.approx(0.1)


def test_a_move_without_a_body_yaw_keeps_the_last_one() -> None:
    """A move that returns no body yaw leaves the body where it is."""

    class _NoYaw:
        duration = 10.0

        def evaluate(self, t: float):
            return (np.eye(4), np.array([0.0, 0.0]), None)

    manager = MovementManager(_robot())
    manager.state.last_primary_pose = (np.eye(4), (0.0, 0.0), 0.4)
    t = manager._now()
    manager.state.current_move = _NoYaw()
    manager.state.move_start_time = t
    assert manager._get_primary_pose(t)[2] == pytest.approx(0.4)


def test_start_commands_the_robots_present_pose_not_neutral() -> None:
    """The first command is the robot's present pose.

    The startup snap: the loop used to command neutral, body yaw 0 and
    antennas (0, 0), for the 0.3 s before breathing began.
    """
    robot = _robot(0.5)
    manager = MovementManager(robot)
    manager.start()
    try:
        assert _wait_for(lambda: robot.set_target.called)
        first = robot.set_target.call_args_list[0].kwargs
        assert first["body_yaw"] == pytest.approx(0.5)
        assert first["antennas"] == pytest.approx((-0.1745, 0.1745))
    finally:
        manager.stop(reset_to_neutral=False)


# --- the companion's wake from the fold ---------------------------------------

# The supervisor's folded pose (reachy-companion body.py, Pose.FOLDED).
_FOLD_HEAD = create_head_pose(x=-0.021, y=0, z=-0.044, roll=0, pitch=0.426, yaw=0, degrees=False, mm=False)
_FOLD_ANTENNAS = (-3.05, 3.05)


def test_the_ease_out_of_the_fold_is_slow_and_ends_inside_the_settle() -> None:
    """The ease out of the fold takes its time from the distance.

    The owner, 2026-10-04: woken from the fold he "POPPED UP like a
    spring-loaded snake-in-a-can". A fixed 1.0 s ease moved the antennas
    about 2.9 rad. The ease is capped below the supervisor's 3 s
    press-settle window.
    """
    move = BreathingMove(interpolation_start_pose=_FOLD_HEAD, interpolation_start_antennas=_FOLD_ANTENNAS)
    assert 2.0 <= move.interpolation_duration <= 2.5


def test_a_short_ease_keeps_the_old_one_second() -> None:
    """From near neutral (after a dance, a hold) the ease is still about 1 s."""
    move = BreathingMove(
        interpolation_start_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        interpolation_start_antennas=(-0.3, 0.3),
    )
    assert move.interpolation_duration == pytest.approx(1.0)


def test_the_ease_starts_and_ends_at_rest() -> None:
    """Min-jerk, not a straight line: no jump in speed when it begins or ends."""
    move = BreathingMove(interpolation_start_pose=_FOLD_HEAD, interpolation_start_antennas=_FOLD_ANTENNAS)
    d = move.interpolation_duration
    step = d / 100
    start = move.evaluate(0.0)[1]
    early = move.evaluate(step)[1]
    # A straight line would have covered 1% of the way; min-jerk covers ~0.001%.
    assert abs(early[1] - start[1]) < 0.001 * abs(start[1] - 0.1745)
    late = move.evaluate(d - step)[1]
    end = move.evaluate(d)[1]
    assert abs(end[1] - late[1]) < 0.001
    np.testing.assert_allclose(move.evaluate(d)[1], move.evaluate(d + 1e-9)[1], atol=1e-6)


def test_idle_breathing_out_of_the_fold_takes_its_time() -> None:
    """The manager no longer forces 1.0 s on the breathing it starts."""
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 7, list(_FOLD_ANTENNAS))
    robot.get_current_head_pose.return_value = _FOLD_HEAD
    manager = MovementManager(robot)
    t = manager._now() + manager.idle_inactivity_delay + 1.0
    manager._update_primary_motion(t)
    (move,) = manager.move_queue
    assert move.interpolation_duration > 2.0


# --- the companion's still start (R-36) ---------------------------------------


def test_a_still_start_holds_the_robots_present_pose_before_anything_moves() -> None:
    """A still start holds the robot where it is.

    Started by the supervisor while folded, the app must not lift the robot:
    breathing waits for a pose command.
    """
    manager = MovementManager(_robot(0.4))
    manager._seed_from_robot()
    manager._queue_hold_still()
    name, (head, antennas, duration, body_yaw) = manager._command_queue.get_nowait()
    assert name == "hold_pose"
    np.testing.assert_allclose(head, create_head_pose(0, 0, 0, 0, 0, 0.4, degrees=False))
    assert antennas == (-0.1745, 0.1745) and body_yaw == pytest.approx(0.4)
    t = manager._now()
    manager._handle_command(name, (head, antennas, duration, body_yaw), t)
    manager._update_primary_motion(t + 5.0)  # well past the 0.3 s breathing delay
    assert manager.is_holding()
    assert not any(isinstance(m, BreathingMove) for m in manager.move_queue)


# --- the companion: a queued move preempts the supervisor's hold (R-39) -----------


def _held_manager() -> tuple[MovementManager, float]:
    manager = MovementManager(_robot())
    manager._seed_from_robot()
    t = manager._now()
    attentive = create_head_pose(0, 0, 0, 0, -6, 0, degrees=True)
    manager._handle_command("hold_pose", (attentive, (-0.4, 0.4), 0.8, 0.2), t)
    manager._update_primary_motion(t)
    assert isinstance(manager.state.current_move, HoldPoseMove)
    return manager, t


def test_a_queued_move_runs_while_a_pose_is_held_and_the_hold_comes_back() -> None:
    """2026-10-06 07:10: move_head right returned 'looking right' three times and
    nothing moved, because the ENGAGED hold never ends and the queue waits."""
    manager, t = _held_manager()
    look = GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, -30, degrees=True), duration=1.0)
    manager._handle_command("queue_move", look, t + 0.1)
    manager._update_primary_motion(t + 0.2)
    assert manager.state.current_move is look
    assert manager.is_holding()  # the supervisor's hold is set aside, not dropped

    manager._update_primary_motion(t + 1.3)  # the look has ended
    back = manager.state.current_move
    assert isinstance(back, HoldPoseMove)
    np.testing.assert_allclose(back.target_antennas, (-0.4, 0.4))
    assert back.target_body_yaw == pytest.approx(0.2)


def test_a_new_hold_or_a_release_replaces_the_one_set_aside() -> None:
    manager, t = _held_manager()
    look = GotoQueueMove(target_head_pose=create_head_pose(0, 0, 0, 0, 0, -30, degrees=True), duration=1.0)
    manager._handle_command("queue_move", look, t + 0.1)
    manager._update_primary_motion(t + 0.2)
    manager._handle_command("release_hold", None, t + 0.3)
    manager._update_primary_motion(t + 1.3)
    assert not isinstance(manager.state.current_move, HoldPoseMove)
    assert not manager.is_holding()


# --- the review of 2026-10-06: a recorded move keeps the body's facing --------------


from reachy_mini.motion.move import Move  # noqa: E402


class _FrontFacingMove(Move):  # type: ignore[misc]
    """A recorded move as the library holds one: about a body facing front."""

    relative_to_body = True

    @property
    def duration(self) -> float:
        return 1.0

    def evaluate(self, t: float):
        return create_head_pose(0, 0, 0, 0, 10, 0, degrees=True), np.array([0.3, -0.3]), 0.0


def test_a_recorded_move_turns_with_the_body_it_starts_on() -> None:
    """Facing a person at +0.7 rad, an emotion or a cue must not swing the body to
    the front and back (it did, at every wake word)."""
    manager = MovementManager(_robot(0.7))
    manager._seed_from_robot()
    t = manager._now()
    attentive = create_head_pose(0, 0, 0, 0, -6, 0, degrees=True)
    manager._handle_command("hold_pose", (attentive, (-0.4, 0.4), 0.8, 0.7), t)
    manager._update_primary_motion(t)
    manager._update_primary_motion(t + 1.0)  # the hold has arrived
    manager._handle_command("queue_move", _FrontFacingMove(), t + 1.1)
    manager._update_primary_motion(t + 1.2)

    head, _antennas, body_yaw = manager._get_primary_pose(t + 1.3)
    assert body_yaw == pytest.approx(0.7)
    yaw = float(np.arctan2(head[1, 0], head[0, 0]))
    assert yaw == pytest.approx(0.7, abs=1e-6)  # the head turns with the body
