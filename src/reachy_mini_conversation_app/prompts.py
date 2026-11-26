"""Resolve active profile prompts and voice settings."""

import os
import json
import logging
from typing import Any
from pathlib import Path

from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileDefinition,
    ProfileFormatError,
    read_profile,
    read_packaged_default_profile,
)


logger = logging.getLogger(__name__)

def _active_profile() -> ProfileDefinition:
    return read_profile(config.REACHY_MINI_CUSTOM_PROFILE)


# The companion supervisor keeps the robot's memory and writes it as one block,
# read at each session start.
COMPANION_MEMORY_ENV = "REACHY_COMPANION_SESSION_MEMORY"
# The same block with only the household's words from the room (the companion, thoughts unit c3):
# the instructions of a written reply to a thought.
THOUGHT_MEMORY_ENV = "REACHY_COMPANION_THOUGHT_MEMORY"


def _companion_memory(env: str = COMPANION_MEMORY_ENV) -> str:
    try:
        return Path(os.environ[env]).read_text(encoding="utf-8").strip()
    except (KeyError, OSError) as exc:
        logger.info("No companion memory block (%s): %s", env, exc)
        return ""


def _profile_instructions() -> str:
    """Return the active profile's instructions, the bundled default's if it has none."""
    selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
    profile_name = selected_profile or DEFAULT_PROFILE_NAME
    try:
        profile = _active_profile()
        instructions = profile.instructions.strip()
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load profile %r: %s", profile_name, exc)
        instructions = ""

    if not instructions and selected_profile and selected_profile != DEFAULT_PROFILE_NAME:
        logger.warning("Using bundled default instructions because profile %r is incomplete", selected_profile)
        try:
            instructions = read_packaged_default_profile().instructions.strip()
        except (FileNotFoundError, ProfileFormatError) as exc:
            raise RuntimeError("Default profile has no usable instructions") from exc
    if not instructions:
        raise RuntimeError("Default profile has no usable instructions")
    return instructions


def get_profile_instructions() -> str:
    """Return the active profile's instructions, the stable base of his instructions.

    Sent last, after the companion's parts (instructions_writer.compose).
    """
    return _profile_instructions()


# The session file's first line (the companion's memory.SESSION_HEADER): the S it is cut at and
# the highest turn row the write read, as JSON. Only the writer of his instructions reads it.
SESSION_HEADER = "<!-- block "
SESSION_HEADER_END = " -->"


def read_session_block() -> tuple[str, dict[str, Any]]:
    """Return the companion's session block as a session's open reads it: (block, header).

    The header's fields are {} with none. Unbounded, with S's text in the header, since a new
    session has no parts yet (the companion's ConvoAll round 6, rule 4).
    """
    raw = _companion_memory()
    first, _, rest = raw.partition("\n")
    if not (first.startswith(SESSION_HEADER) and first.endswith(SESSION_HEADER_END)):
        return raw, {}
    try:
        header = json.loads(first[len(SESSION_HEADER) : -len(SESSION_HEADER_END)])
    except ValueError:
        logger.warning("The companion's session block header is not JSON: read without it")
        return rest.strip(), {}
    return rest.strip(), header if isinstance(header, dict) else {}


def get_thought_instructions() -> str:
    """Return the instructions of a written reply to a thought.

    The companion's thought block, then the profile. With no
    thought block (unset, missing, unreadable or empty) the profile goes alone, never the session's
    block, which carries words that stay in the room.
    """
    instructions = _profile_instructions()
    memory_prompt = _companion_memory(THOUGHT_MEMORY_ENV)
    if memory_prompt:
        return f"{memory_prompt}\n\n{instructions}"
    return instructions


def get_session_voice(default: str | None = None) -> str:
    """Return the active profile voice or the backend default."""
    fallback = get_default_voice() if default is None else default
    try:
        return _active_profile().voice or fallback
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile voice: %s", exc)
        return fallback
