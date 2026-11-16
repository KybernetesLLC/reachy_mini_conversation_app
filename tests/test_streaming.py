import asyncio

import numpy as np
import pytest

from reachy_mini_conversation_app.streaming import (
    AdditionalOutputs,
    frame_rms,
    wait_for_item,
    audio_to_int16,
    audio_to_float32,
)


def test_additional_outputs_stores_messages() -> None:
    """Additional outputs should keep the emitted role/content payloads."""
    message = {"role": "assistant", "content": "hello"}

    outputs = AdditionalOutputs(message)

    assert outputs.args == (message,)


@pytest.mark.asyncio
async def test_wait_for_item_returns_none_on_timeout() -> None:
    """Queue waits should time out without raising."""
    queue: asyncio.Queue[int] = asyncio.Queue()

    assert await wait_for_item(queue, timeout=0.01) is None


def test_audio_to_int16_scales_float32() -> None:
    """Float audio should scale into the int16 sample range."""
    audio = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(audio_to_int16(audio), np.array([-32767, 0, 32767], dtype=np.int16))


def test_audio_to_float32_scales_int16() -> None:
    """Int16 audio should scale into the float sample range."""
    audio = np.array([-32768, 0, 32767], dtype=np.int16)

    np.testing.assert_allclose(audio_to_float32(audio), np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))


def test_audio_converters_reject_unsupported_dtype() -> None:
    """Unsupported audio dtypes should fail explicitly."""
    audio = np.array([1.0], dtype=np.float64)

    with pytest.raises(TypeError, match="Unsupported audio data type"):
        audio_to_int16(audio)


def test_frame_rms_is_one_measure_for_floats_and_int16() -> None:
    """Floats as they are, int16 at full scale 32768, and none for an empty frame."""
    floats = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    ints = np.array([16384, -16384, 16384, -16384], dtype=np.int16)
    assert frame_rms(floats) == pytest.approx(0.5)
    assert frame_rms(ints) == pytest.approx(0.5)
    assert frame_rms(ints) == pytest.approx(frame_rms(audio_to_float32(ints)))
    assert frame_rms(np.zeros(0, dtype=np.float32)) is None
    assert frame_rms(np.zeros(0, dtype=np.int16)) is None


def test_frame_rms_counts_every_element_of_a_two_dimensional_frame() -> None:
    """The microphone may hand over (samples, channels): the RMS is over all of it."""
    frame = np.array([[0.5, 0.1], [-0.5, 0.1], [0.5, 0.1]], dtype=np.float32)
    assert frame_rms(frame) == pytest.approx(float(np.sqrt(np.mean(np.square(frame)))))
    assert frame_rms((frame * 32768).astype(np.int16)) == pytest.approx(frame_rms(frame), rel=1e-3)
