"""Tests for the camera tool."""

import base64
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.asyncio
async def test_camera_tool_returns_base64_of_sdk_jpeg(monkeypatch) -> None:
    """The tool base64-encodes the JPEG bytes returned by the SDK, as fitted for the model."""
    from reachy_mini_conversation_app import model_image

    monkeypatch.setattr(model_image, "jpeg_from_jpeg", lambda jpeg, cut=None: b"fitted:" + jpeg)
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = jpeg_bytes

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert result["b64_im"] == base64.b64encode(b"fitted:" + jpeg_bytes).decode("utf-8")


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_no_frame(caplog) -> None:
    """With no frame the result tells the model it has no picture and must not describe
    a scene, in its own words; the miss is a WARNING (2026-10-17)."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = None

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    with caplog.at_level("WARNING"):
        result = await Camera()(deps, question="What color is this?")

    assert "b64_im" not in result and result["no_picture"] is True
    text = result["instruction"]
    assert "no picture" in text and "Do not describe" in text and "own words" in text
    assert "camera: no frame for camera/no frame" in caplog.text


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

    assert result["no_picture"] is True and "camera disabled" in result["instruction"]
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

    def read_jpeg(self) -> bytes | None:
        # As the SDK's does: a second read() (the review, 2026-10-07).
        frame = self.read()
        return None if frame is None else b"\xff\xd8fresh\xff\xd9"


def _deps(monkeypatch, tmp_path, *, stale: FakeCamera, room_luma: float | None):
    FakeCamera.made = [stale]
    monkeypatch.setattr(camera_mod, "_REOPENABLE", (FakeCamera,))
    # The frame measured is the frame encoded.
    monkeypatch.setattr(camera_mod, "_encode", lambda camera, frame: b"\xff\xd8fresh\xff\xd9")
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
    assert result["no_picture"] is True
    assert "dark frame in a lit room" in result["instruction"]
    assert len(FakeCamera.made) == 3  # the stale one, and two fresh tries


@pytest.mark.asyncio
async def test_a_dark_frame_in_a_dark_room_is_sent(monkeypatch, tmp_path) -> None:
    stale = FakeCamera.__new__(FakeCamera)
    stale.camera_specs, stale.frames, stale.closed = None, [], False
    FakeCamera.scripts = [[4.0]]
    _, deps = _deps(monkeypatch, tmp_path, stale=stale, room_luma=8.0)

    result = await Camera()(deps, question="what do you see")

    assert "b64_im" in result


def test_the_jpeg_is_the_frame_that_was_measured(monkeypatch) -> None:
    """The review, 2026-10-07: read_jpeg read a second frame, often none at all."""
    stale = FakeCamera.__new__(FakeCamera)
    stale.camera_specs, stale.frames, stale.closed = None, [], False
    FakeCamera.made = [stale]
    FakeCamera.scripts = [[None, 91.0]]  # one frame, then the buffer is empty
    monkeypatch.setattr(camera_mod, "_REOPENABLE", (FakeCamera,))
    monkeypatch.setattr(camera_mod, "FRAME_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(camera_mod, "_encode", lambda camera, frame: f"jpeg:{frame.mean():.0f}".encode())
    media = MagicMock()
    media.camera = stale

    jpeg, luma = camera_mod._fresh_frame(media)

    assert jpeg == b"jpeg:91" and luma == 91.0


def test_two_fresh_frames_at_once_never_share_or_leak_a_camera(monkeypatch) -> None:
    """The review, 2026-10-07: the tool and a glance could both reopen the camera."""
    import threading
    import time as _time

    open_now = {"n": 0, "most": 0}

    class Slow(FakeCamera):
        def open(self) -> None:
            open_now["n"] += 1
            open_now["most"] = max(open_now["most"], open_now["n"])
            _time.sleep(0.05)
            self.opened = True

        def close(self) -> None:
            if getattr(self, "opened", False) and not self.closed:
                open_now["n"] -= 1
            self.closed = True

    stale = Slow.__new__(Slow)
    stale.camera_specs, stale.frames, stale.closed, stale.opened = None, [], False, False
    FakeCamera.made = [stale]
    FakeCamera.scripts = [[91.0], [92.0]]
    monkeypatch.setattr(camera_mod, "_REOPENABLE", (Slow,))
    monkeypatch.setattr(camera_mod, "_encode", lambda camera, frame: b"j")
    media = MagicMock()
    media.camera = stale

    threads = [threading.Thread(target=camera_mod._fresh_frame, args=(media,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert open_now["most"] == 1
    assert [c.closed for c in FakeCamera.made[1:-1]] == [True]  # only the newest is left open


# --- 2026-10-16: images sent to the model are fitted to one size and a byte cap ---


def test_a_fresh_frame_is_encoded_for_the_model(monkeypatch) -> None:
    """The IPC path encodes the numpy frame it measured with the model's spec."""
    from reachy_mini_conversation_app import model_image

    seen: list[tuple[int, ...]] = []

    def fake_from_frame(frame):
        seen.append(frame.shape)
        return b"for the model"

    monkeypatch.setattr(model_image, "jpeg_from_frame", fake_from_frame)

    assert camera_mod._encode(object(), np.zeros((1080, 1920, 3), dtype=np.uint8)) == b"for the model"
    assert seen == [(1080, 1920, 3)]


# --- 2026-11-09: look closer, a region of the view at native pixels ---


def test_close_boxes_cut_the_1080p_frame_at_native_size() -> None:
    """Each region is 1056x594 of the 1920x1080 frame; held is the middle below a face."""
    assert camera_mod.close_box(1920, 1080, "top left") == (0, 0, 1056, 594)
    assert camera_mod.close_box(1920, 1080, "center") == (432, 243, 1488, 837)
    assert camera_mod.close_box(1920, 1080, "bottom right") == (864, 486, 1920, 1080)
    assert camera_mod.close_box(1920, 1080, "held") == (432, 486, 1488, 1080)
    assert camera_mod.close_box(640, 480, "right") == (0, 0, 640, 480)  # a smaller frame is sent whole


class BigCamera(FakeCamera):
    """An IPC camera serving one 1920x1080 frame whose pixels say where they are."""

    def read(self):
        """Return the frame once."""
        if not self.frames:
            return None
        self.frames.pop(0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(1920) // 8)[None, :]  # column / 8
        frame[:, :, 1] = (np.arange(1080) // 8)[:, None]  # row / 8
        return frame


def _big_deps(monkeypatch, tmp_path, encoded: list):
    stale = BigCamera.__new__(BigCamera)
    stale.camera_specs, stale.frames, stale.closed = None, [], False
    FakeCamera.made = [stale]
    FakeCamera.scripts = [[1.0]]
    monkeypatch.setattr(camera_mod, "_REOPENABLE", (BigCamera,))
    monkeypatch.setattr(camera_mod, "FRAME_WAIT_SECONDS", 0.05)

    def encode(camera, frame):
        encoded.append(frame)
        return b"\xff\xd8close\xff\xd9"

    monkeypatch.setattr(camera_mod, "_encode", encode)
    monkeypatch.setenv("REACHY_COMPANION_LIGHT_LOG", str(tmp_path / "none.csv"))
    reachy_mini = MagicMock()
    reachy_mini.media.camera = stale
    return ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock(), camera_enabled=True)


@pytest.mark.asyncio
async def test_closer_sends_that_region_and_says_it_is_a_close_up(monkeypatch, tmp_path, caplog) -> None:
    """The cut is the named region at native pixels, the result says so and the log gives the box."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)

    with caplog.at_level("INFO"):
        result = await Camera()(deps, question="what is the title", closer="bottom right")

    (frame,) = encoded
    assert frame.shape == (594, 1056, 3)
    assert frame[0, 0, 0] == 864 // 8 and frame[0, 0, 1] == 486 // 8  # cut at the bottom right
    assert "close-up of the bottom right of your view" in result["view"]
    assert "closer=bottom right box=864,486,1920,1080 of 1920x1080" in caplog.text


@pytest.mark.asyncio
async def test_without_closer_the_whole_view_is_sent_as_before(monkeypatch, tmp_path) -> None:
    """No closer: the whole frame, and no close-up note."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)

    result = await Camera()(deps, question="what do you see")

    assert encoded[0].shape == (1080, 1920, 3)
    assert "view" not in result and "b64_im" in result


@pytest.mark.asyncio
async def test_held_says_where_it_looked(monkeypatch, tmp_path) -> None:
    """Held is the middle below a face, and the note says so."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)

    result = await Camera()(deps, question="what book is this", closer="held")

    assert encoded[0][0, 0, 1] == 486 // 8
    assert "the middle of your view, below where a face would be" in result["view"]


@pytest.mark.asyncio
async def test_an_unknown_closer_is_an_error_and_takes_no_picture(monkeypatch, tmp_path) -> None:
    """A region not in the list takes no picture."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)

    result = await Camera()(deps, question="what is that", closer="far left")

    assert "closer must be one of" in result["error"] and encoded == []


@pytest.mark.asyncio
async def test_closer_cuts_the_sdk_jpeg_path_too(monkeypatch) -> None:
    """A camera that is not the IPC reader is cut as well."""
    from reachy_mini_conversation_app import model_image

    cuts: list = []

    def fitted(jpeg, cut=None):
        cuts.append(cut(np.zeros((1080, 1920, 3), dtype=np.uint8)).shape)
        return b"fitted"

    monkeypatch.setattr(model_image, "jpeg_from_jpeg", fitted)
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg\xff\xd9"
    deps = ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock(), camera_enabled=True)

    result = await Camera()(deps, question="read it", closer="top")

    assert cuts == [(594, 1056, 3)] and "b64_im" in result


def test_the_schema_offers_the_regions() -> None:
    """The model sees the nine regions and held; question stays the only required field."""
    closer = Camera.parameters_schema["properties"]["closer"]
    assert closer["enum"] == list(camera_mod.REGIONS) and "held" in closer["enum"]
    assert Camera.parameters_schema["required"] == ["question"]


@pytest.mark.asyncio
async def test_closer_takes_any_case_spacing_and_the_british_spelling(monkeypatch, tmp_path) -> None:
    """Case, spacing and "centre" give the region meant, not a failed round trip."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)

    result = await Camera()(deps, question="what is that", closer=" Centre ")

    assert encoded[0][0, 0, 0] == 432 // 8 and "the center of your view" in result["view"]


@pytest.mark.asyncio
async def test_a_closer_that_is_not_a_string_is_an_error(monkeypatch, tmp_path) -> None:
    """A list from the model returns the error, not a TypeError."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)

    result = await Camera()(deps, question="what is that", closer=["top"])

    assert "closer must be one of" in result["error"] and encoded == []


# --- 2026-11-09: what the result says about the picture ---


def _jpeg(width: int, height: int) -> bytes:
    """Return a JPEG header with a baseline frame of width x height, enough for jpeg_size."""
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + bytes(9)
    sof = b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return b"\xff\xd8" + app0 + sof + bytes(12) + b"\xff\xd9"


@pytest.mark.parametrize(
    "closer,sent,zoom",
    [(None, (1056, 594), 1.0), ("top left", (1056, 594), 1.82), ("top left", (960, 540), 1.65)],
)
@pytest.mark.asyncio
async def test_the_result_carries_the_picture_facts(monkeypatch, tmp_path, closer, sent, zoom) -> None:
    """Region, box, frame, size sent, bytes, zoom and when it was taken, close-up or not."""
    encoded: list = []
    deps = _big_deps(monkeypatch, tmp_path, encoded)
    jpeg = _jpeg(*sent)
    monkeypatch.setattr(camera_mod, "_encode", lambda camera, frame: encoded.append(frame) or jpeg)

    result = await Camera()(deps, question="what is it", closer=closer)

    assert result["closer"] == closer and result["frame"] == [1920, 1080]
    assert result["box"] == ([0, 0, 1920, 1080] if closer is None else [0, 0, 1056, 594])
    assert (result["image_width"], result["image_height"]) == sent and result["jpeg_bytes"] == len(jpeg)
    assert result["zoom"] == zoom
    assert result["taken_at"][:2] == "20" and "T" in result["taken_at"]


def test_a_frame_no_larger_than_the_cut_is_not_called_a_close_up() -> None:
    """The review, 11-09: a 640x480 frame is sent whole, so the note would be false."""
    meta = {"frame": [640, 480], "box": [0, 0, 640, 480], "taken_at": "t"}
    facts = camera_mod.picture_facts("right", meta, _jpeg(640, 480))
    assert facts["zoom"] == 1.0


@pytest.mark.asyncio
async def test_a_small_frame_with_closer_has_no_close_up_note(monkeypatch, tmp_path) -> None:
    """A closer on a frame the size of the cut sends it whole and says nothing about a close-up."""
    stale = FakeCamera.__new__(FakeCamera)
    stale.camera_specs, stale.frames, stale.closed = None, [], False
    FakeCamera.scripts = [[91.0]]
    _, deps = _deps(monkeypatch, tmp_path, stale=stale, room_luma=None)

    result = await Camera()(deps, question="what is it", closer="right")

    assert "view" not in result and result["box"] == [0, 0, 4, 4]


def test_jpeg_size_reads_the_frame_header() -> None:
    """The size sent comes from the JPEG itself; no header, no size."""
    from reachy_mini_conversation_app import model_image

    assert model_image.jpeg_size(_jpeg(1056, 594)) == (1056, 594)
    assert model_image.jpeg_size(b"\xff\xd8fresh\xff\xd9") is None
