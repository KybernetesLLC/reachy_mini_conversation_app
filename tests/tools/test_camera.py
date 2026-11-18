"""Tests for the camera tool: since the companion's unit 62b, a grab request to the companion.

The companion's request writer (reachy_companion.grab_request) is replaced by a fake here: these
are the tool's side of the seam. The companion's own tests run the request's lifecycle.
"""

import base64
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import model_image
from reachy_mini_conversation_app.tools import camera as camera_mod
from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _jpeg(width: int, height: int) -> bytes:
    """Return a JPEG header with a baseline frame of width x height, enough for jpeg_size."""
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + bytes(9)
    sof = b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return b"\xff\xd8" + app0 + sof + bytes(12) + b"\xff\xd9"


class FakeCompanion:
    """The companion's grab_request: `ask` answers from a script, one answer per call."""

    def __init__(self, answers: List[Dict[str, Any]]) -> None:
        """Hold the script of answers."""
        self.answers = list(answers)
        self.asked: List[tuple] = []
        self.cancels: List[str] = []

    WAIT_SECONDS = 8.0

    async def ask(
        self, purpose: str, closer: Optional[str], question: Optional[str], *, wait_s: float, cancelable: bool
    ) -> Dict:
        """Answer the next scripted grab."""
        assert 0 < wait_s <= self.WAIT_SECONDS  # one deadline for the call, the retry inside it
        self.asked.append((purpose, closer, question, cancelable))
        return self.answers.pop(0)

    def cancel_all(self, why: str) -> int:
        """Record a cancel of every grab."""
        self.cancels.append(why)
        return 1

    def function(self, module: str, name: str) -> Any:
        """Return the fake's function by name."""
        assert module == "grab_request"
        return getattr(self, name)


def _companion(monkeypatch, answers: List[Dict[str, Any]]) -> FakeCompanion:
    fake = FakeCompanion(answers)
    monkeypatch.setattr(camera_mod.companion, "function", fake.function)
    return fake


def _deps(enabled: bool = True) -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_enabled=enabled)


def _done(path, **extra: Any) -> Dict[str, Any]:
    answer = {
        "status": "done",
        "path": str(path),
        "frame_id": "f1",
        "luma": 91.0,
        "frame": [1920, 1080],
        "box": [0, 0, 1920, 1080],
        "taken_at": "2026-11-17T18:00:00.000-06:00",
    }
    answer.update(extra)
    return answer


@pytest.mark.asyncio
async def test_the_picture_is_the_companions_file_and_its_facts(monkeypatch, tmp_path) -> None:
    """Send the companion's kept file as it is, with its facts, pose and Frame id.

    Under the cap it is not encoded again; a kept file is left where it is.
    """
    kept = tmp_path / "kept.jpg"
    jpeg = _jpeg(1056, 594)
    kept.write_bytes(jpeg)
    pose = {"head_yaw": 0.1, "head_pitch": -0.05, "body_yaw": 0.2}
    fake = _companion(monkeypatch, [_done(kept, id="2026-11-17/x-camera", pose=pose)])

    result = await Camera()(_deps(), question="What color is this?")

    assert base64.b64decode(result["b64_im"]) == jpeg
    assert fake.asked == [("camera", None, "What color is this?", True)]
    assert result["pose"] == pose and result["frame_id"] == "f1"
    assert result["frame"] == [1920, 1080] and result["box"] == [0, 0, 1920, 1080]
    assert (result["image_width"], result["image_height"]) == (1056, 594) and result["zoom"] == 1.0
    assert result["taken_at"].startswith("2026-11-17T18") and "view" not in result
    assert kept.exists()


@pytest.mark.asyncio
async def test_a_scratch_picture_is_deleted_once_read(monkeypatch, tmp_path) -> None:
    """With keeping paused the companion writes a scratch file (no id): read once, then deleted."""
    scratch = tmp_path / "grab-1.jpg"
    scratch.write_bytes(_jpeg(1056, 594))
    _companion(monkeypatch, [_done(scratch)])

    result = await Camera()(_deps(), question="what is it")

    assert "b64_im" in result and not scratch.exists()


@pytest.mark.asyncio
async def test_a_picture_over_the_cap_is_fitted_again(monkeypatch, tmp_path) -> None:
    """Over the model's byte cap the picture is encoded again, smaller."""
    big = tmp_path / "big.jpg"
    big.write_bytes(b"x" * (model_image.MODEL_IMAGE.max_bytes + 1))
    monkeypatch.setattr(model_image, "jpeg_from_jpeg", lambda jpeg: b"fitted")
    _companion(monkeypatch, [_done(big, id="k")])

    result = await Camera()(_deps(), question="what is it")

    assert base64.b64decode(result["b64_im"]) == b"fitted"


@pytest.mark.parametrize(
    "answer,reason",
    [
        ({"status": "rejected", "refused": "asleep"}, "asleep"),
        ({"status": "rejected", "refused": "Eyes failed"}, "Eyes failed"),
        ({"status": "rejected", "refused": "too late"}, "too late"),
        ({"status": "rejected", "refused": "no camera socket"}, "no camera socket"),
        ({"status": "timeout"}, "the camera did not answer in time"),
        ({"status": "canceled"}, "the turn ended"),
        ({"status": "failed"}, "the camera failed"),
    ],
)
@pytest.mark.asyncio
async def test_no_picture_says_why(monkeypatch, caplog, answer, reason) -> None:
    """Tell the model it has no picture, with the reason, and log the miss as a WARNING (2026-10-17).

    For a refusal, a timeout, a cancel or a failure.
    """
    _companion(monkeypatch, [answer])

    with caplog.at_level("WARNING"):
        result = await Camera()(_deps(), question="what is it")

    assert "b64_im" not in result and result["no_picture"] is True
    assert "(no frame)" in result["instruction"] and "Do not describe" in result["instruction"]
    assert reason not in result["instruction"]  # the model is told the old phrase; the log says why
    assert f"camera: no frame for camera/{reason}" in caplog.text


@pytest.mark.asyncio
async def test_with_no_companion_there_is_no_camera(monkeypatch) -> None:
    """No fallback to a grab of the app's own (the companion's LookAfterMove `fork62`)."""
    monkeypatch.setattr(camera_mod.companion, "function", lambda module, name: None)

    result = await Camera()(_deps(), question="what is it")

    assert result["no_picture"] is True and "(no frame)" in result["instruction"]
    assert camera_mod.cancel_grabs("x") == 0


@pytest.mark.asyncio
async def test_a_dark_frame_in_a_lit_room_is_asked_for_once_more(monkeypatch, tmp_path) -> None:
    """The companion refuses a dark frame in a lit room; the tool asks once more, then says so."""
    dark = {"status": "rejected", "refused": camera_mod.DARK_FRAME_REASON, "luma": 4.0}
    fake = _companion(monkeypatch, [dark, dark])

    result = await Camera()(_deps(), question="what is it")

    assert len(fake.asked) == 2 and result["no_picture"] is True
    assert camera_mod.DARK_FRAME_REASON in result["instruction"]

    kept = tmp_path / "k.jpg"
    kept.write_bytes(_jpeg(1056, 594))
    fake = _companion(monkeypatch, [dark, _done(kept, id="k")])
    assert "b64_im" in await Camera()(_deps(), question="what is it")
    assert len(fake.asked) == 2


@pytest.mark.asyncio
async def test_a_missing_file_is_no_picture(monkeypatch, tmp_path) -> None:
    """A forget may erase the kept picture before it is read."""
    _companion(monkeypatch, [_done(tmp_path / "gone.jpg", id="k")])

    result = await Camera()(_deps(), question="what is it")

    assert result["no_picture"] is True and "(no frame)" in result["instruction"]


@pytest.mark.asyncio
async def test_a_failed_request_is_no_picture(monkeypatch, caplog) -> None:
    """A store the request cannot be written to (locked, migrating) is no picture, not its error text."""

    async def ask(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(camera_mod.companion, "function", lambda module, name: ask if name == "ask" else None)

    with caplog.at_level("WARNING"):
        result = await Camera()(_deps(), question="what is it")

    assert result["no_picture"] is True and "(no frame)" in result["instruction"]
    assert "database is locked" in caplog.text and "camera/the request failed" in caplog.text


@pytest.mark.asyncio
async def test_camera_disabled_asks_nothing(monkeypatch) -> None:
    """Ask nothing with the camera disabled."""
    fake = _companion(monkeypatch, [])

    result = await Camera()(_deps(enabled=False), question="What color is this?")

    assert result["no_picture"] is True and "camera disabled" in result["instruction"]
    assert fake.asked == []


@pytest.mark.asyncio
async def test_an_empty_question_is_an_error(monkeypatch) -> None:
    """Ask nothing for an empty question."""
    fake = _companion(monkeypatch, [])
    assert "error" in await Camera()(_deps(), question="  ")
    assert fake.asked == []


# --- Look closer (companion ticket visual-perception/11-look-closer.md, 11-09) ---------


@pytest.mark.asyncio
async def test_closer_asks_for_that_region_and_says_it_is_a_close_up(monkeypatch, tmp_path) -> None:
    """Ask for the close-up's region and say the picture is one."""
    kept = tmp_path / "k.jpg"
    kept.write_bytes(_jpeg(1056, 594))
    fake = _companion(monkeypatch, [_done(kept, id="k", box=[864, 0, 1920, 594])])

    result = await Camera()(_deps(), question="read it", closer="top right")

    assert fake.asked[0][1] == "top right" and result["zoom"] == 1.82
    assert result["view"] == camera_mod.CLOSE_UP_NOTE.format(where="the top right of your view")


@pytest.mark.asyncio
async def test_held_says_where_it_looked(monkeypatch, tmp_path) -> None:
    """Say where a held thing was looked for."""
    kept = tmp_path / "k.jpg"
    kept.write_bytes(_jpeg(1056, 594))
    _companion(monkeypatch, [_done(kept, id="k", box=[432, 486, 1488, 1080])])

    result = await Camera()(_deps(), question="what am I holding", closer="held")

    assert "below where a face would be" in result["view"]


@pytest.mark.asyncio
async def test_a_frame_no_larger_than_the_cut_is_not_called_a_close_up(monkeypatch, tmp_path) -> None:
    """The review, 11-09: a 640x480 frame is sent whole, so the note would be false."""
    kept = tmp_path / "k.jpg"
    kept.write_bytes(_jpeg(640, 480))
    _companion(monkeypatch, [_done(kept, id="k", frame=[640, 480], box=[0, 0, 640, 480])])

    result = await Camera()(_deps(), question="what is it", closer="right")

    assert "view" not in result and result["zoom"] == 1.0


@pytest.mark.asyncio
async def test_closer_takes_any_case_spacing_and_the_british_spelling(monkeypatch, tmp_path) -> None:
    """Take the region in any case and spacing, and the British spelling."""
    kept = tmp_path / "k.jpg"
    kept.write_bytes(_jpeg(1056, 594))
    fake = _companion(monkeypatch, [_done(kept, id="k")])

    await Camera()(_deps(), question="what is it", closer="  Centre ")

    assert fake.asked[0][1] == "center"


@pytest.mark.parametrize("closer", ["sideways", 3])
@pytest.mark.asyncio
async def test_a_closer_that_is_not_a_region_is_an_error_and_asks_nothing(monkeypatch, closer) -> None:
    """Refuse a closer that is not a region before asking."""
    fake = _companion(monkeypatch, [])

    result = await Camera()(_deps(), question="what is it", closer=closer)

    assert "error" in result and fake.asked == []


def test_the_schema_offers_the_regions() -> None:
    """Offer the regions in the schema."""
    closer = Camera.parameters_schema["properties"]["closer"]
    assert closer["enum"] == list(camera_mod.REGIONS) and "held" in closer["enum"]


def test_jpeg_size_reads_the_frame_header() -> None:
    """The size sent comes from the JPEG itself; no header, no size."""
    assert model_image.jpeg_size(_jpeg(1056, 594)) == (1056, 594)
    assert model_image.jpeg_size(b"\xff\xd8fresh\xff\xd9") is None


def test_the_frame_id_and_pose_reach_the_sight_note() -> None:
    """The companion keeps the picture once, at the grab: the Frame id tells its sight handler so."""
    assert "frame_id" in camera_mod.FACT_KEYS and "pose" in camera_mod.FACT_KEYS


def test_the_turn_end_cancels_the_tools_grabs(monkeypatch) -> None:
    """Cancel the tool's grabs at the turn's end."""
    fake = _companion(monkeypatch, [])
    assert camera_mod.cancel_grabs("interrupted") == 1 and fake.cancels == ["interrupted"]


@pytest.mark.asyncio
async def test_the_glance_asks_too_and_is_not_canceled_with_the_turn(monkeypatch, tmp_path) -> None:
    """Ask for the session-open and scene glance (console.py) the same way.

    Only its deadline and a cancel of its own task end it; its scratch picture is deleted once read.
    """
    scratch = tmp_path / "grab-g.jpg"
    scratch.write_bytes(_jpeg(1056, 594))
    fake = _companion(monkeypatch, [_done(scratch)])

    jpeg, luma, why_not, answer = await camera_mod.fresh_jpeg("glance")

    assert jpeg == _jpeg(1056, 594) and luma == 91.0 and why_not is None and answer["frame_id"] == "f1"
    assert fake.asked == [("glance", None, None, False)] and not scratch.exists()
