from pathlib import Path

import pytest

import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.config import config


def test_the_companions_memory_file_is_prepended_to_the_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensouled C3: the companion supervisor writes the memory block; the app prepends it."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    block = tmp_path / "session-memory.md"
    block.write_text("## Your memory\nThe plant is by my left antenna.\n")
    monkeypatch.setenv("REACHY_COMPANION_SESSION_MEMORY", str(block))

    instructions = prompts_mod.get_session_instructions(instance_path=tmp_path)

    assert instructions.startswith("## Your memory\nThe plant is by my left antenna.\n\n")
    assert "## IDENTITY" in instructions

    block.unlink()  # no memories yet: no block
    assert prompts_mod.get_session_instructions(instance_path=tmp_path) == prompts_mod._profile_instructions()


def test_a_thought_with_no_thought_block_gets_the_profile_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It fails closed: never the session's block, which holds words that stay in the room."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    session = tmp_path / "session-memory.md"
    session.write_text("## Your memory\n[10:01] Someone: the guest's words\n")
    monkeypatch.setenv("REACHY_COMPANION_SESSION_MEMORY", str(session))
    profile = prompts_mod._profile_instructions()

    monkeypatch.setenv("REACHY_COMPANION_THOUGHT_MEMORY", str(tmp_path / "thought-memory.md"))
    assert prompts_mod.get_thought_instructions() == profile  # missing
    (tmp_path / "thought-memory.md").write_text("  \n")
    assert prompts_mod.get_thought_instructions() == profile  # empty
    monkeypatch.delenv("REACHY_COMPANION_THOUGHT_MEMORY")
    assert prompts_mod.get_thought_instructions() == profile  # unset
