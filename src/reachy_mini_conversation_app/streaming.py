import math
import asyncio
from typing import TypeVar, TypeAlias
from collections.abc import Mapping, Callable

import numpy as np
from numpy.typing import NDArray


StreamMessage: TypeAlias = Mapping[str, object]
AudioArray: TypeAlias = NDArray[np.int16] | NDArray[np.float32]

QueueItem = TypeVar("QueueItem")


class AdditionalOutputs:
    """Text or metadata emitted alongside audio frames."""

    def __init__(self, *args: StreamMessage) -> None:
        """Initialize with one or more emitted messages."""
        self.args = args


class AsyncStreamHandler:
    """Minimal async stream handler state used by the local audio loop."""

    def __init__(self) -> None:
        """Initialize shared stream handler state."""
        self._clear_queue: Callable[[], None] | None = None


async def wait_for_item(queue: asyncio.Queue[QueueItem], timeout: float = 0.1) -> QueueItem | None:
    """Return the next queue item, or None when no item arrives before timeout."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def audio_to_int16(audio: AudioArray) -> NDArray[np.int16]:
    """Convert int16 or float32 audio data to int16 samples."""
    if audio.dtype == np.int16:
        return audio.astype(np.int16, copy=False)
    if audio.dtype == np.float32:
        return (audio * 32767.0).astype(np.int16)
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def audio_to_float32(audio: AudioArray) -> NDArray[np.float32]:
    """Convert int16 or float32 audio data to float32 samples."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.float32:
        return audio.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def frame_rms(samples: NDArray[np.generic]) -> float | None:
    """Return the RMS of an array of samples (any shape: every element counts), or None for none. Pure.

    Floats are taken as they are; int16 samples count as full scale 32768. The one RMS of the app:
    the speaker monitor, the phantom detector's pad probe and the console's level and health
    readings all use it.
    """
    if samples.size == 0:
        return None
    samples = samples.reshape(-1)
    if samples.dtype.kind in "iu":
        wide = samples.astype(np.int64)
        return math.sqrt(float(np.dot(wide, wide)) / samples.size) / 32768.0
    return math.sqrt(float(samples.dot(samples)) / samples.size)
