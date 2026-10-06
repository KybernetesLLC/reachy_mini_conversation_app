"""Resolve active profile prompts and voice settings."""

import os
import random
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.memory import format_memory_for_prompt
from reachy_mini_conversation_app.profile_store import (
    DEFAULT_PROFILE_NAME,
    ProfileDefinition,
    ProfileFormatError,
    read_profile,
    canonical_profile_name,
    read_packaged_default_profile,
)


logger = logging.getLogger(__name__)

DEFAULT_GREETING_PROMPT = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)
# A profile may list exact opening lines, one per line, beside its profile.md. One is chosen at
# random for each greeting, never the one said last; asked to "vary the wording", the model
# settled on the same few phrases.
STARTUP_GREETINGS_FILENAME = "startup_greetings.txt"
_last_startup_greeting: str | None = None


def _active_profile() -> ProfileDefinition:
    return read_profile(config.REACHY_MINI_CUSTOM_PROFILE)


# The companion supervisor keeps the robot's memory and writes it as one block,
# read at each session start in place of the app's own memory.v1.json facts.
COMPANION_MEMORY_ENV = "REACHY_COMPANION_SESSION_MEMORY"


def _companion_memory() -> str:
    try:
        return Path(os.environ[COMPANION_MEMORY_ENV]).read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.info("No companion memory block: %s", exc)
        return ""


def get_session_instructions(instance_path: str | Path | None = None) -> str:
    """Return instructions for the active profile with memory context."""
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

    memory_prompt = (
        _companion_memory() if COMPANION_MEMORY_ENV in os.environ else format_memory_for_prompt(instance_path)
    )
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


def _startup_greeting_lines() -> list[str]:
    profile_name = canonical_profile_name(config.REACHY_MINI_CUSTOM_PROFILE)
    if profile_name == DEFAULT_PROFILE_NAME:
        return []
    try:
        text = (config.resolve_profile_dir(profile_name) / STARTUP_GREETINGS_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def get_session_greeting_prompt() -> str:
    """Return a line from the profile's startup greetings, else its greeting prompt, else the default."""
    global _last_startup_greeting
    lines = _startup_greeting_lines()
    if lines:
        choices = [line for line in lines if line != _last_startup_greeting] or lines
        _last_startup_greeting = random.choice(choices)
        return f'Start the conversation by saying exactly this, and nothing else: "{_last_startup_greeting}"'
    try:
        return _active_profile().greeting or DEFAULT_GREETING_PROMPT
    except (FileNotFoundError, ProfileFormatError) as exc:
        logger.warning("Failed to load the active profile greeting: %s", exc)
        return DEFAULT_GREETING_PROMPT
