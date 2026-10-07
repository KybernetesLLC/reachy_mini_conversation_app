"""Tests for the camera tool."""

import base64
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_returns_base64_of_sdk_jpeg() -> None:
    """The tool base64-encodes the JPEG bytes returned by the SDK."""
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = jpeg_bytes

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert result["b64_im"] == base64.b64encode(jpeg_bytes).decode("utf-8")


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_no_frame() -> None:
    """With no frame available the tool returns an error."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = None

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_camera_disabled() -> None:
    """With the camera disabled the tool returns an error and never reads a frame."""
    reachy_mini = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=False,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    reachy_mini.media.get_frame_jpeg.assert_not_called()


# --- 2026-10-06: a camera pipeline that died at a sleep stays dead -----------------
#
# The daemon releases its media at every sleep and re-acquires it at the wake. The
# app's IPC camera pipeline hit end-of-stream at the first sleep (17:54) and never
# came back: at 20:13 the tool returned the last frame left in the buffer, from the
# fold (the model said "dark or blurry"), and at 20:15 nothing. A fresh pipeline at
# 20:17 read mean luma 88-91 in a room light.csv measured at 91-96.

import numpy as np  # noqa: E402

from reachy_mini_conversation_app.tools import camera as camera_mod  # noqa: E402


class FakeCamera:
    """An IPC camera: each new one serves the next script of frames."""

    scripts: list[list[float | None]] = []
    made: list["FakeCamera"] = []

    def __init__(self, camera_specs=None) -> None:
        self.camera_specs = camera_specs
        self.frames = list(FakeCamera.scripts.pop(0)) if FakeCamera.scripts else []
        self.opened = self.closed = False
        FakeCamera.made.append(self)

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def read(self):
        value = self.frames.pop(0) if self.frames else None
        return None if value is None else np.full((4, 4, 3), value, dtype=np.uint8)

    def read_jpeg(self) -> bytes:
        return b"\xff\xd8fresh\xff\xd9"


def _deps(monkeypatch, tmp_path, *, stale: FakeCamera, room_luma: float | None):
    FakeCamera.made = [stale]
    monkeypatch.setattr(camera_mod, "_REOPENABLE", (FakeCamera,))
    monkeypatch.setattr(camera_mod, "FRAME_WAIT_SECONDS", 0.05)
    light = tmp_path / "light.csv"
    if room_luma is not None:
        import datetime

        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        light.write_text(f"ts,posture,quiet,people,luma_mean\n{now},ambient,0,2,{room_luma}\n")
    monkeypatch.setenv("REACHY_COMPANION_LIGHT_LOG", str(light))
    reachy_mini = MagicMock()
    reachy_mini.media.camera = stale
    return reachy_mini, ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock(), camera_enabled=True)


@pytest.mark.asyncio
async def test_each_call_reads_a_fresh_pipeline(monkeypatch, tmp_path) -> None:
    stale = FakeCamera.__new__(FakeCamera)
    stale.camera_specs, stale.frames, stale.closed = "specs", [5.0], False
    FakeCamera.scripts = [[None, 91.0]]  # the first read comes before the first frame
    reachy_mini, deps = _deps(monkeypatch, tmp_path, stale=stale, room_luma=93.0)

    result = await Camera()(deps, question="what do you see")

    assert result["b64_im"] == base64.b64encode(b"\xff\xd8fresh\xff\xd9").decode()
    assert stale.closed
    fresh = reachy_mini.media.camera
    assert fresh is not stale and fresh.opened and fresh.camera_specs == "specs"


@pytest.mark.asyncio
async def test_a_dark_frame_in_a_lit_room_is_tried_again_then_refused(monkeypatch, tmp_path) -> None:
    stale = FakeCamera.__new__(FakeCamera)
    stale.camera_specs, stale.frames, stale.closed = None, [], False
    FakeCamera.scripts = [[4.0], [3.0]]
    _, deps = _deps(monkeypatch, tmp_path, stale=stale, room_luma=93.0)

    result = await Camera()(deps, question="what do you see")

    assert "b64_im" not in result
    assert "dark" in result["error"] and "lit" in result["error"]
    assert len(FakeCamera.made) == 3  # the stale one, and two fresh tries


@pytest.mark.asyncio
async def test_a_dark_frame_in_a_dark_room_is_sent(monkeypatch, tmp_path) -> None:
    stale = FakeCamera.__new__(FakeCamera)
    stale.camera_specs, stale.frames, stale.closed = None, [], False
    FakeCamera.scripts = [[4.0]]
    _, deps = _deps(monkeypatch, tmp_path, stale=stale, room_luma=8.0)

    result = await Camera()(deps, question="what do you see")

    assert "b64_im" in result
