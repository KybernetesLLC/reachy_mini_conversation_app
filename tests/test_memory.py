import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import (
    MAX_FACTS,
    MAX_FACT_CHARS,
    MemoryFact,
    add_memory_fact,
    list_memory_facts,
    clear_memory_facts,
    forget_memory_fact,
    format_memory_for_prompt,
    memory_path_for_instance,
)
from reachy_mini_conversation_app.tools.forget import Forget
from reachy_mini_conversation_app.tools.remember import Remember
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def test_memory_store_adds_dedupes_caps_and_formats(tmp_path: Path) -> None:
    """Memory facts should be normalized, deduplicated, capped, and prompt-formatted."""
    first = add_memory_fact(tmp_path, "  Likes   jazz  ")
    duplicate = add_memory_fact(tmp_path, "likes jazz")

    assert first is not None
    assert duplicate == first
    assert [fact.text for fact in list_memory_facts(tmp_path)] == ["Likes jazz"]

    long_text = "x" * (MAX_FACT_CHARS + 20)
    stored_long = add_memory_fact(tmp_path, long_text)

    assert stored_long is not None
    assert len(stored_long.text) == MAX_FACT_CHARS
    assert stored_long.text.endswith("...")

    for index in range(MAX_FACTS + 5):
        add_memory_fact(tmp_path, f"Fact {index}")

    facts = list_memory_facts(tmp_path)
    assert len(facts) == MAX_FACTS
    assert facts[0].text == f"Fact {MAX_FACTS + 4}"
    assert "Likes jazz" not in [fact.text for fact in facts]

    prompt = format_memory_for_prompt(tmp_path)
    assert prompt.startswith("Things you remember about the user")
    assert f"- Fact {MAX_FACTS + 4}" in prompt


def test_memory_store_reads_mobile_json_shape_and_forgets_by_query(tmp_path: Path) -> None:
    """The Python store should preserve the mobile app JSON envelope shape."""
    path = memory_path_for_instance(tmp_path)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "facts": [
                    {"id": "m_1", "text": "Likes jazz", "createdAt": 1000},
                    {"id": "m_2", "text": "Likes jazz piano", "createdAt": 900},
                    {"id": "bad"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = forget_memory_fact(tmp_path, query="jazz")

    assert result.removed == MemoryFact(id="m_1", text="Likes jazz", created_at=1000)
    assert [candidate.text for candidate in result.candidates] == ["Likes jazz", "Likes jazz piano"]
    assert [fact.text for fact in list_memory_facts(tmp_path)] == ["Likes jazz piano"]


@pytest.mark.asyncio
async def test_memory_tools_use_instance_storage(tmp_path: Path) -> None:
    """Remember and forget tools should read/write through ToolDependencies.instance_path."""
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        instance_path=tmp_path,
    )

    remember_result = await Remember()(deps, fact="Has a dog named Mochi")
    forget_result = await Forget()(deps, query="Mochi")

    assert remember_result["saved"] == "Has a dog named Mochi"
    assert forget_result["removed"] == "Has a dog named Mochi"
    assert list_memory_facts(tmp_path) == []


def test_prompt_includes_memory_fragment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Session instructions should prepend saved memories for the active app instance."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    clear_memory_facts(tmp_path)
    add_memory_fact(tmp_path, "Prefers concise answers")

    instructions = prompts_mod.get_session_instructions(instance_path=tmp_path)

    assert instructions.startswith("Things you remember about the user")
    assert "- Prefers concise answers" in instructions
    assert "## IDENTITY" in instructions


def test_the_companions_memory_file_replaces_the_apps_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensouled C3: the companion supervisor writes the memory block; the app prepends it."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    clear_memory_facts(tmp_path)
    add_memory_fact(tmp_path, "Prefers concise answers")
    block = tmp_path / "session-memory.md"
    block.write_text("## Your memory\nThe plant is by my left antenna.\n")
    monkeypatch.setenv("REACHY_COMPANION_SESSION_MEMORY", str(block))

    instructions = prompts_mod.get_session_instructions(instance_path=tmp_path)

    assert instructions.startswith("## Your memory\nThe plant is by my left antenna.\n\n")
    assert "Prefers concise answers" not in instructions
    assert "## IDENTITY" in instructions

    block.unlink()  # no memories yet: no block, and still not the app's own
    instructions = prompts_mod.get_session_instructions(instance_path=tmp_path)
    assert not instructions.startswith("Things you remember")


def test_a_thoughts_instructions_are_the_thought_block_and_the_same_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thoughts unit c3: the thought block in place of the session's, the profile part the same."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    session = tmp_path / "session-memory.md"
    session.write_text("## Your memory\n[10:01] Someone: the guest's words\n")
    thought = tmp_path / "thought-memory.md"
    thought.write_text("## Your memory\n[10:01] (someone else talked with you here)\n")
    monkeypatch.setenv("REACHY_COMPANION_SESSION_MEMORY", str(session))
    monkeypatch.setenv("REACHY_COMPANION_THOUGHT_MEMORY", str(thought))

    instructions = prompts_mod.get_thought_instructions()
    profile = prompts_mod.get_session_instructions().split("\n\n", 1)[1]  # the session's own

    assert instructions == f"## Your memory\n[10:01] (someone else talked with you here)\n\n{profile}"
    assert "guest's words" not in instructions


def test_a_thought_with_no_thought_block_gets_the_profile_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It fails closed: never the session's block, nor the app's own memories."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    add_memory_fact(tmp_path, "Prefers concise answers")
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
