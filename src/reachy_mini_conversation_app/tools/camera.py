import base64
import logging
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

from reachy_mini_conversation_app import companion, model_image
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Companion unit 62b (its ticket 62, the owner's ticket 48 Q5): the app no longer grabs. The camera
# tool and the glance ask the companion's supervisor for a picture through its request queue
# (reachy_companion.grab_request): it takes the camera under the same lock as its own moves and
# looks, reads his pose just before the pull, and its Eyes cuts the region, encodes the picture as
# the model gets it (1056 px, quality 80: model_image.MODEL_IMAGE) and keeps it (the camera tool's,
# 24 h) or writes a scratch file (a glance's). The app reads the file and fits it for the model.
# With no companion there is no picture: no fallback to a grab of the app's own, which could run
# during one of his moves or while he sleeps (the companion's LookAfterMove.tla `fork62`).
# The dark-frame check (a frame under 20 mean luma in a room the light log says is lit) is the
# companion's; a dark frame is refused, never written, and asked for once more here.
DARK_FRAME_REASON = "dark frame in a lit room"

# Look closer (companion ticket visual-perception/11-look-closer.md, the owner, 11-09): the parts
# of the view the tool can send at full detail. The cut itself is the companion's (people.close_box).
REGIONS: Tuple[str, ...] = (
    "top left",
    "top",
    "top right",
    "left",
    "center",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
    "held",
)
CLOSE_UP_NOTE = "This picture is a close-up of {where} at full detail, not your whole view."


_NOT_DONE = {
    "timeout": "the camera did not answer in time",
    "canceled": "the turn ended",
    "failed": "the camera failed",
}


def _fit(jpeg: bytes) -> Optional[bytes]:
    """Return the companion's picture fitted for the model.

    As it is when under the byte cap (it is already at the model's size and quality), else encoded
    again smaller.
    """
    if len(jpeg) <= model_image.MODEL_IMAGE.max_bytes:
        return jpeg
    return model_image.jpeg_from_jpeg(jpeg)


async def fresh_jpeg(
    purpose: str, closer: Optional[str] = None, question: Optional[str] = None
) -> Tuple[Optional[bytes], Optional[float], Optional[str], Dict[str, Any]]:
    """Return (JPEG, luma, why not, the companion's answer) from a fresh picture; for the tool and the glance.

    `purpose` is "camera" (the tool: canceled when its turn ends, its picture kept by the
    companion) or "glance". A dark frame in a lit room is asked for once more, then refused.
    """
    ask = companion.function("grab_request", "ask")
    if ask is None:
        return None, None, "no camera", {}
    answer: Dict[str, Any] = {}
    for _ in range(2):
        answer = await ask(purpose, closer, question, cancelable=purpose == "camera")
        if answer.get("refused") != DARK_FRAME_REASON:
            break
        logger.warning("camera frame dark (luma %s) in a lit room; trying once more", answer.get("luma"))
    status = answer.get("status")
    if status != "done":
        why = answer.get("refused") or _NOT_DONE.get(str(status), status)
        return None, None, str(why or "no frame"), answer
    path = Path(str(answer.get("path")))
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None, "the picture was not there", answer
    finally:
        if "id" not in answer:  # a scratch file, not kept: read once
            path.unlink(missing_ok=True)
    jpeg = _fit(raw)
    if jpeg is None:
        return None, None, "the picture did not decode", answer
    luma = answer.get("luma")
    logger.info("camera frame %s: luma %s", answer.get("frame_id"), "-" if luma is None else luma)
    return jpeg, luma if isinstance(luma, (int, float)) else None, None, answer


def cancel_grabs(why: str) -> int:
    """Cancel the camera tool's grabs still waiting, as the turn ended (the glance's are not); return how many."""
    cancel_all = companion.function("grab_request", "cancel_all")
    return 0 if cancel_all is None else cancel_all(why)


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


# The facts handed to the companion with the frame, for its sight note (the frame store's `camera`
# kind, 24 h, the owner 11-09, is now kept by the companion at the grab): what the result says, the
# question asked, and the Frame id, by which the companion knows the picture is kept already.
FACT_KEYS = (
    "closer",
    "frame",
    "box",
    "image_width",
    "image_height",
    "jpeg_bytes",
    "zoom",
    "taken_at",
    "pose",
    "frame_id",
)


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
        facts.update(frame=meta["frame"], box=meta["box"], taken_at=meta.get("taken_at"))
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

        jpeg_bytes, _luma, why_not, meta = await fresh_jpeg("camera", closer, question)
        if jpeg_bytes is None:
            return no_picture("camera", why_not or "no frame")
        result: Dict[str, Any] = {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
        result.update(picture_facts(closer, meta, jpeg_bytes))
        if isinstance(meta.get("pose"), dict) and meta["pose"]:  # read under the lock just before the pull
            result["pose"] = meta["pose"]
        if isinstance(meta.get("frame_id"), str):
            result["frame_id"] = meta["frame_id"]
        whole = "box" in meta and meta["box"] == [0, 0, *meta["frame"]]  # a frame no larger than the cut
        if closer is not None and not whole:
            where = (
                "the middle of your view, below where a face would be"
                if closer == "held"
                else f"the {closer} of your view"
            )
            result["view"] = CLOSE_UP_NOTE.format(where=where)
        return result
