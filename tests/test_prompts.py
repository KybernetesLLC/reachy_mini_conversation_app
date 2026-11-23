from pathlib import Path

import pytest

import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.config import config


def test_the_companions_session_file_is_read_with_its_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route all, Unit 4: the read at a session's open takes the companion's file whole and its
    first line apart, the S the block is cut at and its highest turn row (rule 4)."""
    block = tmp_path / "session-memory.md"
    header = '<!-- block {"top": 7, "s": {"version": 3.0, "row": 2, "text": "Ferries."}} -->'
    block.write_text(header + "\n## Your memory\nThe plant is by my left antenna.\n")
    monkeypatch.setenv("REACHY_COMPANION_SESSION_MEMORY", str(block))
    text, named = prompts_mod.read_session_block()
    assert text == "## Your memory\nThe plant is by my left antenna."
    assert named == {"top": 7, "s": {"version": 3.0, "row": 2, "text": "Ferries."}}

    block.write_text("## Your memory\nNo header (an older companion).\n")
    assert prompts_mod.read_session_block() == ("## Your memory\nNo header (an older companion).", {})
    block.write_text("<!-- block {not json -->\n## Your memory\n")
    assert prompts_mod.read_session_block() == ("## Your memory", {})
    block.unlink()  # no memories yet: no block
    assert prompts_mod.read_session_block() == ("", {})


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
