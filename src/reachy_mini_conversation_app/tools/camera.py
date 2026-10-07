import os
import time
import base64
import asyncio
import logging
import datetime
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

import numpy as np

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
    """The companion's last light reading, if recent: its own camera's mean luma."""
    try:
        lines = _light_log().read_text().splitlines()
        header, last = lines[0].split(","), lines[-1].split(",")
        row = dict(zip(header, last))
        at = datetime.datetime.fromisoformat(row["ts"])
        age = (datetime.datetime.now(at.tzinfo) - at).total_seconds()
        return float(row["luma_mean"]) if age <= LIGHT_LOG_MAX_AGE_SECONDS else None
    except (OSError, IndexError, KeyError, ValueError):
        return None


def _fresh_frame(media: Any) -> Tuple[Optional[bytes], Optional[float]]:
    """Reopen the IPC camera and return (JPEG, the frame's mean luma).

    Runs in a worker thread: opening waits for the pipeline. A camera that is
    not the IPC reader is read as before, with no luma."""
    old = getattr(media, "camera", None)
    if not _REOPENABLE or not isinstance(old, _REOPENABLE):
        return media.get_frame_jpeg(), None
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
    return camera.read_jpeg(), float(np.mean(frame))


class Camera(Tool):
    """Take a picture with the camera to see what is in front of the robot."""

    name = "camera"
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "Use this when the user asks you to look at something, see what they are holding, "
        "check their appearance, describe the scene, or comment on how they look. "
        "Also use it when the user asks what you can see or wants your visual opinion. "
        "The camera is live, each call captures the current moment. "
        "If the user asks you to look without saying at what, do not ask for clarification, call this tool and describe what you see. "
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to observe or ask about in the picture. "
                    "Examples: what is the user holding, describe the user's outfit, "
                    "what do you see around you, how does the user look today."
                ),
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

        logger.info("Tool call: camera question=%s", question[:120])

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        media = deps.reachy_mini.media
        room = _room_luma()
        jpeg_bytes, luma = await asyncio.to_thread(_fresh_frame, media)
        dark_in_lit_room = luma is not None and room is not None and luma < DARK_FRAME <= ROOM_LIT <= room
        if dark_in_lit_room:
            logger.warning("camera frame dark (luma %.1f) in a lit room (%.1f); trying once more", luma, room)
            jpeg_bytes, luma = await asyncio.to_thread(_fresh_frame, media)
            dark_in_lit_room = luma is not None and luma < DARK_FRAME
        if jpeg_bytes is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}
        logger.info("camera frame: luma %s, room %s", "-" if luma is None else f"{luma:.1f}", room)
        if dark_in_lit_room:
            return {
                "error": (
                    "The camera returned a dark frame although the room is lit, so there is "
                    "nothing to describe. Say the camera isn't working right now."
                )
            }

        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
