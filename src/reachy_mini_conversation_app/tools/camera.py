import os
import time
import base64
import asyncio
import logging
import datetime
import threading
from typing import Any, Dict, Tuple, Callable, Optional
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app import model_image
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

try:
    from reachy_mini.media.camera_gstreamer import GStreamerCamera

    # Cameras reopened on every call (the daemon's IPC reader). Tests swap in a fake.
    _REOPENABLE: Tuple[type, ...] = (GStreamerCamera,)
except Exception:  # no GStreamer here: nothing to reopen
    _REOPENABLE = ()

# 2026-10-06: the daemon releases its media at every sleep and re-acquires it at
# the wake, and the app's IPC camera pipeline ends at the release and never comes
# back: the tool then returned the last buffered frame (a dark one, from the fold)
# and later nothing. So every call opens a fresh pipeline; on the robot that took
# 0.4 s, and its frames matched the room's light. FRAME_WAIT_SECONDS bounds the
# wait for the first frame.
FRAME_WAIT_SECONDS = 2.0
# A frame this dark (mean of BGR, 0-255) while the companion's light log, under
# LIGHT_LOG_MAX_AGE_SECONDS old, says the room is at least ROOM_LIT is not the
# room: it is tried once more and then refused rather than shown to the model.
DARK_FRAME = 20.0
ROOM_LIT = 40.0
LIGHT_LOG_MAX_AGE_SECONDS = 120.0


def _light_log() -> Path:
    value = os.getenv("REACHY_COMPANION_LIGHT_LOG")
    return Path(value) if value else Path.home() / ".local/share/reachy-companion/light.csv"


def _room_luma() -> Optional[float]:
    """Return the companion's last light reading if recent: its camera's mean luma."""
    try:
        lines = _light_log().read_text().splitlines()
        header, last = lines[0].split(","), lines[-1].split(",")
        row = dict(zip(header, last))
        at = datetime.datetime.fromisoformat(row["ts"])
        age = (datetime.datetime.now(at.tzinfo) - at).total_seconds()
        return float(row["luma_mean"]) if age <= LIGHT_LOG_MAX_AGE_SECONDS else None
    except (OSError, IndexError, KeyError, ValueError):
        return None


# Look closer (companion ticket visual-perception/11-look-closer.md, the owner, 11-09): a
# CLOSE_W x CLOSE_H cut of the frame at native pixels instead of the whole frame scaled down.
# The hosted model rescales every image to about 1056 wide, so the whole 1920x1080 view
# reaches it at about half its pixels; the cut reaches it at all of them, 1.8x the detail
# across about 45 of the view's 82 degrees, at the same size and cost. The nine regions
# overlap. `held` is the middle below where a face would be: head tracking keeps a face near
# the centre, and a thing held up to show him sits below it. It is fixed for now; the
# detector aims it later only if the log shows it missing. The cut is fixed in pixels for
# the default 1080p stream: a frame no larger than it is sent whole, and at 4K the nine
# regions would leave gaps (the review, 11-09; neither is configured on the robot).
CLOSE_W, CLOSE_H = 1056, 594
REGIONS: Dict[str, Tuple[float, float]] = {  # name: (x, y), 0 the left or top edge, 1 the other
    "top left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom right": (1.0, 1.0),
    "held": (0.5, 1.0),
}
CLOSE_UP_NOTE = "This picture is a close-up of {where} at full detail, not your whole view."


def close_box(width: int, height: int, closer: str) -> Tuple[int, int, int, int]:
    """Return the (x0, y0, x1, y1) cut of a width x height frame for the region `closer`."""
    fx, fy = REGIONS[closer]
    w, h = min(CLOSE_W, width), min(CLOSE_H, height)
    x0, y0 = round((width - w) * fx), round((height - h) * fy)
    return x0, y0, x0 + w, y0 + h


def _cut(closer: Optional[str], meta: Optional[Dict[str, Any]] = None) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return a function cutting a frame to the region `closer`, or None for the whole frame.

    `meta`, when given, is filled with the frame's size, the box sent and when the frame
    was taken; the whole frame then goes through the function too.
    """
    if closer is None and meta is None:
        return None

    def cut(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = (0, 0, width, height) if closer is None else close_box(width, height, closer)
        if closer is not None:
            logger.info("camera: closer=%s box=%d,%d,%d,%d of %dx%d", closer, x0, y0, x1, y1, width, height)
        if meta is not None:
            meta.update(
                frame=[width, height],
                box=[x0, y0, x1, y1],
                taken_at=datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
            )
        return frame[y0:y1, x0:x1]

    return cut


# One reopen at a time: the camera tool and a glance may both want a frame, and
# each closes and replaces the shared camera (the review, 2026-10-07).
_CAMERA_LOCK = threading.Lock()


def _encode(camera: Any, frame: Any) -> Optional[bytes]:
    """Encode this frame, the one measured, as a JPEG for the model.

    2026-10-16: no longer the camera's own encoder, at full resolution and its
    default quality: model_image fits it to one size and a byte cap. The frame
    goes nowhere else. OpenCV is not in the app's environment on the robot.
    """
    return model_image.jpeg_from_frame(frame)


def _fresh_frame(
    media: Any, closer: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[bytes], Optional[float]]:
    """Reopen the IPC camera; return (JPEG, mean luma) of one frame, under the lock."""
    with _CAMERA_LOCK:
        return _fresh_frame_held(media, closer, meta)


def _fresh_frame_held(
    media: Any, closer: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[bytes], Optional[float]]:
    """Reopen the IPC camera and return (JPEG for the model, the whole frame's mean luma).

    Runs in a worker thread: opening waits for the pipeline. A camera that is
    not the IPC reader is read as before, with no luma. `closer` cuts the JPEG to
    one region of the frame; `meta` is filled as `_cut` says.
    """
    cut = _cut(closer, meta)
    old = getattr(media, "camera", None)
    if not _REOPENABLE or not isinstance(old, _REOPENABLE):
        jpeg = media.get_frame_jpeg()
        if jpeg is None:
            return None, None
        return (model_image.jpeg_from_jpeg(jpeg) if cut is None else model_image.jpeg_from_jpeg(jpeg, cut=cut)), None
    specs = getattr(old, "camera_specs", None)
    try:
        old.close()
    except Exception:
        logger.debug("closing the old camera pipeline failed", exc_info=True)
    camera = type(old)(camera_specs=specs)
    camera.open()
    media.camera = camera
    deadline = time.monotonic() + FRAME_WAIT_SECONDS
    frame = camera.read()
    while frame is None and time.monotonic() < deadline:
        time.sleep(0.05)
        frame = camera.read()
    if frame is None:
        return None, None
    return _encode(camera, frame if cut is None else cut(frame)), float(np.mean(frame))


def fresh_jpeg(
    media: Any, closer: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[bytes], Optional[float], Optional[str]]:
    """Return (JPEG, luma, why not) from a fresh frame; for the tool and the glance.

    A dark frame in a lit room is tried once more, then refused. Runs in a worker
    thread: reopening the camera waits for its first frame. `closer` names a region
    of the view (REGIONS) to send at full detail instead of the whole view; `meta`
    is filled with the frame, the box and the time it was taken.
    """
    room = _room_luma()
    jpeg, luma = _fresh_frame(media, closer, meta)
    if luma is not None and room is not None and luma < DARK_FRAME <= ROOM_LIT <= room:
        logger.warning("camera frame dark (luma %.1f) in a lit room (%.1f); trying once more", luma, room)
        jpeg, luma = _fresh_frame(media, closer, meta)
        if luma is not None and luma < DARK_FRAME:
            return None, luma, "dark frame in a lit room"
    if jpeg is None:
        return None, None, "no frame"
    logger.info("camera frame: luma %s, room %s", "-" if luma is None else f"{luma:.1f}", room)
    return jpeg, luma, None


# 2026-10-17: with no frame the tool said only "No frame available", and the model
# described a view it never saw ("city lights and a few tall buildings"). The owner:
# "he shouldn't respond like a machine but it should be clearly logged that no frame
# came thru". So the result tells the model it has no picture and leaves the words to
# it, and every miss is a WARNING the stats count.
NO_PICTURE = (
    "You have no picture right now ({reason}), so you cannot see anything. Do not describe "
    "the scene or guess what is there. Tell the person briefly, in your own words, that you "
    "can't see right now."
)


def no_picture(source: str, reason: str) -> Dict[str, Any]:
    """Log the miss and return the tool result that tells the model it has no picture."""
    logger.warning("camera: no frame for %s/%s", source, reason)
    return {"no_picture": True, "instruction": NO_PICTURE.format(reason=reason)}


def picture_facts(closer: Optional[str], meta: Dict[str, Any], jpeg: bytes) -> Dict[str, Any]:
    """Return what the result says about the picture, for the model and the analyses.

    zoom is the detail against today's whole-view picture: 1.0 for the whole view sent at
    the model's long side, 1.82 for a 1080p close-up, less when the encoder stepped down.
    """
    facts: Dict[str, Any] = {"closer": closer, "jpeg_bytes": len(jpeg)}
    size = model_image.jpeg_size(jpeg)
    if size is not None:
        facts["image_width"], facts["image_height"] = size
    if "box" in meta:
        frame_w = meta["frame"][0]
        cut_w = meta["box"][2] - meta["box"][0]
        facts.update(frame=meta["frame"], box=meta["box"], taken_at=meta["taken_at"])
        if size is not None and cut_w > 0:
            whole_w = min(model_image.MODEL_IMAGE.long_side, frame_w)
            facts["zoom"] = round(size[0] / cut_w * frame_w / whole_w, 2)
    return facts


class Camera(Tool):
    """Take a picture with the camera to see what is in front of the robot."""

    name = "camera"
    description = (
        "Take a picture to see what is in front of you now: what someone is holding, how "
        "they look, the scene, or your visual opinion. If asked to look without saying at "
        "what, don't ask; call it and describe what you see. To read small text or see a small or "
        "far thing, set closer to the part of your view it is in, or to held for something a "
        "person holds up to you."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to look for in the picture.",
            },
            "closer": {
                "type": "string",
                "enum": list(REGIONS),
                "description": "Optional: a sharper picture of only this part of your view.",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and return the base64-encoded JPEG."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        raw = kwargs.get("closer")
        closer = " ".join(raw.lower().replace("centre", "center").split()) if isinstance(raw, str) else raw
        closer = closer or None
        if closer is not None and (not isinstance(closer, str) or closer not in REGIONS):
            logger.warning("camera: unknown closer %r", closer)
            return {"error": f"closer must be one of: {', '.join(REGIONS)}"}

        logger.info("Tool call: camera question=%s closer=%s", question[:120], closer or "-")

        if not deps.camera_enabled:
            return no_picture("camera", "camera disabled")

        meta: Dict[str, Any] = {}
        jpeg_bytes, _luma, why_not = await asyncio.to_thread(fresh_jpeg, deps.reachy_mini.media, closer, meta)
        if jpeg_bytes is None:
            return no_picture("camera", why_not or "no frame")
        result: Dict[str, Any] = {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
        result.update(picture_facts(closer, meta, jpeg_bytes))
        whole = "box" in meta and meta["box"] == [0, 0, *meta["frame"]]  # a frame no larger than the cut
        if closer is not None and not whole:
            where = (
                "the middle of your view, below where a face would be"
                if closer == "held"
                else f"the {closer} of your view"
            )
            result["view"] = CLOSE_UP_NOTE.format(where=where)
        return result
