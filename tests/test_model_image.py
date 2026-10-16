"""Tests for the images sent to the model: size, quality and the byte cap.

2026-10-16: the client closed twice with "1009 (message too big)" about 2 s after
an image was added to the session. The hosted server echoes the whole data URL
back in conversation.item.created, and full-resolution frames at the encoder's
default quality pushed that echo past the client's 1 MiB receive limit.
"""

import logging

import numpy as np
import pytest


gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")

from reachy_mini_conversation_app import model_image  # noqa: E402
from reachy_mini_conversation_app.model_image import MODEL_IMAGE, ModelImageSpec  # noqa: E402


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) from a JPEG's start-of-frame marker."""
    i = 2
    while i < len(data):
        assert data[i] == 0xFF
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        i += 2 + length
    raise AssertionError("no start-of-frame marker")


def _gradient(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    return np.stack([x * 255 // width, y * 255 // height, (x + y) * 255 // (width + height)], -1).astype(np.uint8)


def _textured(width: int, height: int, amplitude: int) -> np.ndarray:
    """Return a gradient with noise: over the cap at 1280 px, under it at 960 px (measured)."""
    rng = np.random.default_rng(0)
    noise = rng.integers(-amplitude, amplitude + 1, (height, width, 3))
    return np.clip(_gradient(width, height).astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_the_numbers_live_in_one_spec() -> None:
    """Size, quality and cap are one spec, retuned in one place."""
    assert MODEL_IMAGE == ModelImageSpec(long_side=1280, quality=80, max_bytes=400_000, step_down_long_side=960)


def test_a_full_hd_frame_goes_at_1280_long_side() -> None:
    """A 1920x1080 frame is sent at 1280x720."""
    jpeg = model_image.jpeg_from_frame(_gradient(1920, 1080))

    assert jpeg is not None
    assert _jpeg_size(jpeg) == (1280, 720)


def test_the_aspect_is_kept() -> None:
    """A 4:3 frame stays 4:3."""
    jpeg = model_image.jpeg_from_frame(_gradient(1600, 1200))

    assert jpeg is not None
    assert _jpeg_size(jpeg) == (1280, 960)


def test_a_small_frame_is_not_enlarged() -> None:
    """A frame under 1280 px keeps its size."""
    jpeg = model_image.jpeg_from_frame(_gradient(640, 480))

    assert jpeg is not None
    assert _jpeg_size(jpeg) == (640, 480)


def test_a_frame_over_the_cap_steps_down_to_960_and_fits() -> None:
    """A frame over 400 KB at 1280 px is sent at 960 px, under the cap."""
    jpeg = model_image.jpeg_from_frame(_textured(1920, 1080, amplitude=32))

    assert jpeg is not None
    assert _jpeg_size(jpeg) == (960, 540)
    assert len(jpeg) <= MODEL_IMAGE.max_bytes


def test_pure_noise_steps_down_once_and_is_still_sent(caplog: pytest.LogCaptureFixture) -> None:
    """Noise cannot fit the cap even at 960 px; one step down is all there is."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)

    with caplog.at_level(logging.INFO, logger=model_image.__name__):
        jpeg = model_image.jpeg_from_frame(noise)

    assert jpeg is not None
    assert _jpeg_size(jpeg) == (960, 540)
    assert "stepped down from 1280x720" in caplog.text
    assert any(r.levelno == logging.WARNING and "over the cap" in r.getMessage() for r in caplog.records)


def test_jpeg_bytes_are_resized_too() -> None:
    """The SDK's get_frame_jpeg() fallback gives JPEG bytes only."""
    full = model_image.jpeg_from_frame(
        _gradient(1920, 1080), ModelImageSpec(long_side=1920, quality=95, max_bytes=10**7)
    )
    assert full is not None and _jpeg_size(full) == (1920, 1080)

    jpeg = model_image.jpeg_from_jpeg(full)

    assert jpeg is not None
    assert _jpeg_size(jpeg) == (1280, 720)


def test_bytes_that_are_not_a_jpeg_give_nothing() -> None:
    """Bytes that do not decode give no image."""
    assert model_image.jpeg_from_jpeg(b"\xff\xd8not a jpeg\xff\xd9") is None


def test_each_image_is_logged_with_size_and_quality(caplog: pytest.LogCaptureFixture) -> None:
    """One INFO line per image: resolution, size and quality."""
    with caplog.at_level(logging.INFO, logger=model_image.__name__):
        jpeg = model_image.jpeg_from_frame(_gradient(1920, 1080))

    assert jpeg is not None
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert lines == [f"Image to the model: 1280x720, {round(len(jpeg) / 1000)} KB (q80)"]


def test_a_failed_step_down_keeps_the_first_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over the cap at 1280 and the 960 encode fails: the first JPEG is still sent, not None."""
    big = b"\xff" * 500_000
    calls: list[tuple[int, int]] = []

    def encode(frame: np.ndarray, width: int, height: int, quality: int) -> bytes | None:
        calls.append((width, height))
        return big if len(calls) == 1 else None

    monkeypatch.setattr(model_image, "_encode", encode)
    assert model_image.jpeg_from_frame(np.zeros((1080, 1920, 3), dtype=np.uint8)) == big
    assert calls == [(1280, 720), (960, 540)]


def test_an_encode_that_raises_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline error never reaches the camera tool or the glance: None, logged."""

    def encode(frame: np.ndarray, width: int, height: int, quality: int) -> bytes | None:
        raise RuntimeError("no jpegenc")

    monkeypatch.setattr(model_image, "_encode", encode)
    assert model_image.jpeg_from_frame(np.zeros((720, 1280, 3), dtype=np.uint8)) is None
