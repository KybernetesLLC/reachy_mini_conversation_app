import re
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.config import get_default_voice, get_available_voices
from reachy_mini_conversation_app.personality import save_user_personality
from reachy_mini_conversation_app.personality_tag import to_tag_token
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_WRITE_MOVES = None


def _queue_write_move(deps: ToolDependencies) -> None:
    global _WRITE_MOVES
    if deps.movement_manager is None:
        return
    try:
        from reachy_mini.motion.recorded_move import RecordedMoves
        from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove

        if _WRITE_MOVES is None:
            _WRITE_MOVES = RecordedMoves("glannuzel/local-dataset")
        deps.movement_manager.queue_move(EmotionQueueMove("write-tag-6", _WRITE_MOVES))
    except Exception as exc:
        logger.warning("_queue_write_move: failed to queue movement: %s", exc)


# Authored tool defaults for a personality the robot creates for itself. nfc_writer
# is deliberately absent: core_tools adds it to every profile while a reader is
# attached, so listing it here would only pin a stale copy into the profile document.
_DEFAULT_TOOLS = (
    "camera",
    "dance",
    "head_tracking",
    "move_head",
    "play_emotion",
    "stop_dance",
    "stop_emotion",
)


def _sanitize_name(name: str) -> str:
    """Coerce a model-supplied name into a valid profile folder name."""
    sanitized = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^a-zA-Z0-9_-]", "", sanitized)


class NfcWriter(Tool):
    """Create a new personality profile and write its NFC code to the blank tag on the reader."""

    name = "nfc_writer"
    description = (
        "Create a new personality profile from a name and a system-prompt, "
        "then write its unique code onto the accessory currently placed on the head. "
        "Call this only after the user has confirmed they want to create a new personality "
        "and has described what it should be like."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Short identifier for the personality (e.g. 'pirate', 'chef'). "
                    "Used as the profile folder name — letters, digits, hyphens and underscores only."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Full system-prompt for this personality, crafted from the user's description. "
                    "Should be concise and capture the character, tone and language rules."
                ),
            },
            "voice": {
                "type": "string",
                "description": "TTS voice to use for this personality.",
                # Read from the backend rather than hardcoded: a stale list would
                # be silently rejected and every new personality would sound alike.
                "enum": get_available_voices(),
            },
        },
        "required": ["name", "instructions"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Save the personality, then write its token to the accessory on the reader."""
        name = (kwargs.get("name") or "").strip()
        instructions = (kwargs.get("instructions") or "").strip()
        voice = (kwargs.get("voice") or "").strip() or get_default_voice()

        if not name or not instructions:
            return {"error": "name and instructions are required"}

        name_s = _sanitize_name(name)
        if not name_s:
            return {"error": f"Invalid personality name: {name!r}"}

        logger.info("nfc_writer: creating profile %r voice=%r", name_s, voice)

        try:
            personality = save_user_personality(
                name_s,
                instructions,
                voice=voice,
                overwrite=True,
                default_tools=_DEFAULT_TOOLS,
            )
        except Exception as exc:
            logger.error("nfc_writer: failed to write profile: %s", exc)
            return {"error": f"Failed to write profile: {exc}"}

        # The tag carries the personality itself, so there is nothing to
        # allocate or remember: the same personality always yields the same
        # token, and the tag means the same thing on any robot.
        code = to_tag_token(personality)
        if code is None:
            logger.error("nfc_writer: no tag token for personality %r", personality)
            return {"error": f"Cannot write personality {personality!r} to a tag"}
        logger.info("nfc_writer: tag token %r for %r", code, personality)

        # Check whether a blank tag is currently on the reader
        rfid_serial = deps.rfid_serial
        if deps.blank_tag_present:
            # Accessory is on the reader — send RFID write command.
            # The write-tag-6 movement is triggered later on WRITE_OK so it starts
            # simultaneously with the "I feel the new personality" speech.
            deps.pending_nfc_write = None
            deps.recently_written_codes.add(code)
            if rfid_serial is not None and rfid_serial.is_connected():
                write_status = rfid_serial.write_tag(code)
                logger.info("nfc_writer: write_tag(%r) → %s", code, write_status)
            else:
                write_status = "RFID reader not connected"
                logger.warning("nfc_writer: %s", write_status)
            return {
                "status": "writing",
                "personality": personality,
                "code": code,
                "write_status": write_status,
            }
        else:
            # No blank tag on reader — store pending write so console.py writes on next detection
            deps.pending_nfc_write = {"code": code, "personality": personality}
            logger.info("nfc_writer: no blank tag present — stored pending write for %r", personality)
            return {
                "status": "waiting_for_tag",
                "personality": personality,
                "code": code,
                "message": (
                    "No accessory on the head right now. "
                    "Ask the user to place the accessory back on the head to program it."
                ),
            }
