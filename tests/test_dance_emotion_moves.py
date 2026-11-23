from unittest.mock import MagicMock

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove, DanceQueueMove


def test_goto_evaluate_at_start_returns_start_state() -> None:
    """At t=0 the goto move sits at its start antennas and body yaw."""
    move = GotoQueueMove(
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 40, degrees=True),
        start_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
        target_antennas=(1.0, 1.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=10.0,
        start_body_yaw=0.0,
        duration=2.0,
    )
    _, antennas, body_yaw = move.evaluate(0.0)
    assert antennas is not None
    np.testing.assert_allclose(antennas, [0.0, 0.0])
    assert body_yaw == 0.0


def test_goto_evaluate_midpoint_interpolates_linearly() -> None:
    """Halfway through the duration antennas and body yaw are halfway to target."""
    move = GotoQueueMove(
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 40, degrees=True),
        target_antennas=(1.0, 1.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=10.0,
        start_body_yaw=0.0,
        duration=2.0,
    )
    _, antennas, body_yaw = move.evaluate(1.0)
    assert antennas is not None
    np.testing.assert_allclose(antennas, [0.5, 0.5])
    assert body_yaw == 5.0


def test_goto_evaluate_clamps_past_duration() -> None:
    """Past the duration t is clamped so antennas reach the target."""
    move = GotoQueueMove(
        target_head_pose=create_head_pose(0, 0, 0, 0, 0, 40, degrees=True),
        target_antennas=(1.0, 1.0),
        start_antennas=(0.0, 0.0),
        target_body_yaw=10.0,
        start_body_yaw=0.0,
        duration=2.0,
    )
    _, antennas, body_yaw = move.evaluate(99.0)
    assert antennas is not None
    np.testing.assert_allclose(antennas, [1.0, 1.0])
    assert body_yaw == 10.0


def test_dance_queue_move_falls_back_to_neutral_on_error() -> None:
    """A failing wrapped dance move yields a neutral pose instead of raising."""
    move = DanceQueueMove.__new__(DanceQueueMove)
    move.move_name = "broken"
    move.dance_move = MagicMock()
    move.dance_move.evaluate.side_effect = RuntimeError("boom")
    head_pose, antennas, body_yaw = move.evaluate(0.5)
    assert head_pose is not None
    np.testing.assert_allclose(antennas, [0.0, 0.0])
    assert body_yaw == 0.0


def test_dance_queue_move_converts_tuple_antennas() -> None:
    """Tuple antennas from the wrapped move are converted to a numpy array."""
    move = DanceQueueMove.__new__(DanceQueueMove)
    move.move_name = "ok"
    move.dance_move = MagicMock()
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    move.dance_move.evaluate.return_value = (head, (0.1, 0.2), 0.5)
    _, antennas, body_yaw = move.evaluate(0.0)
    assert isinstance(antennas, np.ndarray)
    np.testing.assert_allclose(antennas, [0.1, 0.2])
    assert body_yaw == 0.5


def test_a_dance_move_takes_its_tempo_beats_and_amplitude_and_leaves_the_library_alone() -> None:
    """Tempo and beats set a conversation.dance move's length; the library is never changed.

    DanceMove.__init__ updates the library's shared parameter dict in place.
    """
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES

    before = dict(AVAILABLE_MOVES["head_tilt_roll"][1])
    move = DanceQueueMove("head_tilt_roll", bpm=90.0, beats=8, amplitude=0.5, antennas=False)
    again = DanceQueueMove("head_tilt_roll", bpm=90.0, beats=8, amplitude=0.5, antennas=False)

    assert move.duration == 8 * 60 / 90.0
    assert move.dance_move.default_bpm == 90.0
    assert move.dance_move.move_params["amplitude_rad"] == before["amplitude_rad"] * 0.5
    assert again.dance_move.move_params["amplitude_rad"] == before["amplitude_rad"] * 0.5
    assert move.dance_move.move_params["antenna_amplitude_rad"] == 0.0
    assert AVAILABLE_MOVES["head_tilt_roll"][1] == before
    plain = DanceQueueMove("head_tilt_roll")
    assert plain.dance_move.default_bpm == 114 and plain.ends_on_release is False


def test_a_still_antenna_dance_move_leaves_the_antennas_to_the_manager() -> None:
    """antennas=False leaves the antennas to the manager's rest value, and never turns the body.

    0 would be 10 degrees off the held pose and read as a press.
    """
    move = DanceQueueMove("groovy_sway_and_roll", bpm=120.0, beats=4, antennas=False)
    for t in np.linspace(0.0, move.duration, 9):
        _, antennas, body_yaw = move.evaluate(float(t))
        assert antennas is None
        assert body_yaw == 0


class _Step:
    """A step of known length that reports where in it it was asked."""

    def __init__(self, name: str, duration: float) -> None:
        self.name, self.duration = name, duration

    def evaluate(self, t: float):  # type: ignore[no-untyped-def]
        return (self.name, t, None)


def _grid_dance(now: float, at: float = 100.0, period: float = 0.5):  # type: ignore[no-untyped-def]
    from reachy_mini_conversation_app.dance_emotion_moves import GridDanceMove

    steps = [_Step("a", 2.0), _Step("b", 4.0), _Step("c", 2.0)]
    return GridDanceMove(steps, grid_at=at, period_s=period, clock=lambda: now)  # type: ignore[arg-type]


def test_a_grid_dance_starts_on_the_next_beat_and_each_step_on_its_own() -> None:
    """Begun 0.2 s after a beat (100.0 + k * 0.5), the first step waits 0.3 s for the next."""
    move = _grid_dance(now=1000.2)
    assert move.duration == 0.5 + 8.0  # unplaced: at most one beat's wait
    assert move.evaluate(0.0) == ("a", 0.0, None)  # holds the first step's opening pose
    assert move.wait_s is not None and abs(move.wait_s - 0.3) < 1e-9
    assert abs(move.duration - 8.3) < 1e-9
    assert move.evaluate(0.1) == ("a", 0.0, None)
    name, t, _ = move.evaluate(0.3 + 1.0)
    assert name == "a" and abs(t - 1.0) < 1e-9
    name, t, _ = move.evaluate(0.3 + 2.0)
    assert name == "b" and abs(t) < 1e-9  # the second step on its own beat, nothing lost
    name, t, _ = move.evaluate(0.3 + 7.5)
    assert name == "c" and abs(t - 1.5) < 1e-9
    name, t, _ = move.evaluate(9.0)  # past the end: the last step's last pose
    assert name == "c" and t == 2.0


def test_a_grid_dance_on_a_beat_waits_nothing_and_the_grid_may_be_old() -> None:
    """A grid's beat long before (or after) now places the same: the period carries it."""
    on_beat = _grid_dance(now=1000.0)
    on_beat.evaluate(0.0)
    assert on_beat.wait_s is not None and abs(on_beat.wait_s) < 1e-9
    later_grid = _grid_dance(now=1000.2, at=1003.0)  # a beat in the future works too
    later_grid.evaluate(0.0)
    assert later_grid.wait_s is not None and abs(later_grid.wait_s - 0.3) < 1e-9


def test_a_grid_dance_first_evaluated_late_places_from_its_own_start() -> None:
    """The manager's first sample may come a tick after the move began: it is placed from then."""
    move = _grid_dance(now=1000.25)  # began at 1000.2, first sampled 0.05 s in
    assert move.evaluate(0.05) == ("a", 0.0, None)
    assert move.wait_s is not None and abs(move.wait_s - 0.3) < 1e-9
