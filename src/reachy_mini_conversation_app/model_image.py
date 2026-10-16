"""Images sent to the model: one size, one JPEG quality, and a byte cap.

2026-10-16: the client closed twice with "1009 (message too big)" about 2 s after
an image was added to the session. The hosted server echoes the whole data URL
back in conversation.item.created, and full-resolution camera frames at the
encoder's default quality made that echo larger than the client's 1 MiB receive
limit. Every image for the model now goes through here.

The resizing and encoding run in GStreamer (videoscale and jpegenc), which the
SDK's own camera encoder already uses: OpenCV and Pillow are not in the app's
environment on the robot.
"""

import logging
from typing import Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelImageSpec:
    """Size, quality and cap of an image sent to the model; retune here."""

    long_side: int = 1280  # pixels; the aspect is kept and frames are never enlarged
    quality: int = 80  # jpegenc quality, 0-100
    max_bytes: int = 400_000  # over this, the image is encoded again at step_down_long_side
    step_down_long_side: int = 960


MODEL_IMAGE = ModelImageSpec()

_TIMEOUT_NS = 2_000_000_000


def _gst() -> Any:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    if not Gst.is_initialized():
        Gst.init(None)
    return Gst


def _target_size(width: int, height: int, long_side: int) -> Tuple[int, int]:
    """Scale (width, height) so the long side is at most `long_side`; even sides, aspect kept."""
    scale = min(1.0, long_side / max(width, height))
    if scale == 1.0:
        return width, height
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def _run(launch: str, data: bytes) -> Tuple[Optional[bytes], Any]:
    """Push `data` through a one-shot pipeline; return (output bytes, output caps)."""
    Gst = _gst()
    pipeline = Gst.parse_launch(launch)
    try:
        pipeline.set_state(Gst.State.PLAYING)
        pipeline.get_by_name("src").emit("push-buffer", Gst.Buffer.new_wrapped(data))
        pipeline.get_by_name("src").emit("end-of-stream")
        sample = pipeline.get_by_name("sink").emit("try-pull-sample", _TIMEOUT_NS)
        if sample is None:
            return None, None
        buffer = sample.get_buffer()
        return buffer.extract_dup(0, buffer.get_size()), sample.get_caps()
    finally:
        pipeline.set_state(Gst.State.NULL)


def _encode(frame: np.ndarray, width: int, height: int, quality: int) -> Optional[bytes]:
    """Encode a BGR frame as a JPEG of `width` x `height` at `quality`."""
    source_height, source_width = frame.shape[:2]
    launch = (
        f"appsrc name=src caps=video/x-raw,format=BGR,width={source_width},height={source_height},framerate=1/1 "
        f"! videoconvert ! videoscale ! video/x-raw,width={width},height={height},pixel-aspect-ratio=1/1 "
        f"! jpegenc quality={quality} ! appsink name=sink sync=false"
    )
    jpeg, _caps = _run(launch, np.ascontiguousarray(frame).tobytes())
    return jpeg


def jpeg_from_frame(frame: np.ndarray, spec: ModelImageSpec = MODEL_IMAGE) -> Optional[bytes]:
    """Return a BGR frame as a JPEG for the model, fitted to `spec`, and log it."""
    height, width = frame.shape[:2]
    size = _target_size(width, height, spec.long_side)
    try:
        jpeg = _encode(frame, *size, spec.quality)
        stepped_from: Optional[Tuple[int, int]] = None
        if jpeg is not None and len(jpeg) > spec.max_bytes:
            smaller = _target_size(width, height, spec.step_down_long_side)
            if smaller != size:
                fitted = _encode(frame, *smaller, spec.quality)
                if fitted is not None:  # a failed step-down keeps the first JPEG: it still fits the 16 MiB limit
                    stepped_from, size, jpeg = size, smaller, fitted
    except Exception:
        logger.warning("Image to the model: encoding a %dx%d frame raised", width, height, exc_info=True)
        return None
    if jpeg is None:
        logger.warning("Image to the model: encoding a %dx%d frame failed", width, height)
        return None
    note = f", stepped down from {stepped_from[0]}x{stepped_from[1]}" if stepped_from else ""
    logger.info(
        "Image to the model: %dx%d, %d KB (q%d)%s", size[0], size[1], round(len(jpeg) / 1000), spec.quality, note
    )
    if len(jpeg) > spec.max_bytes:
        logger.warning("Image to the model is over the cap: %d > %d bytes", len(jpeg), spec.max_bytes)
    return jpeg


def jpeg_from_jpeg(jpeg: bytes, spec: ModelImageSpec = MODEL_IMAGE) -> Optional[bytes]:
    """Return JPEG bytes as a JPEG for the model, fitted to `spec`; None if they do not decode."""
    launch = (
        "appsrc name=src caps=image/jpeg ! jpegdec ! videoconvert ! video/x-raw,format=BGR "
        "! appsink name=sink sync=false"
    )
    try:
        raw, caps = _run(launch, jpeg)
    except Exception:
        logger.warning("Image to the model: decoding a JPEG failed", exc_info=True)
        return None
    if raw is None or caps is None:
        logger.warning("Image to the model: %d bytes did not decode as a JPEG", len(jpeg))
        return None
    structure = caps.get_structure(0)
    width, height = structure.get_value("width"), structure.get_value("height")
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, len(raw) // height)
    return jpeg_from_frame(rows[:, : width * 3].reshape(height, width, 3), spec)
