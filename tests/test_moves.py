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
    assert second_hold.target_body_yaw == 0.0
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
