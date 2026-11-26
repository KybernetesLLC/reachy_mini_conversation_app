import logging
from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.prompts as prompts_mod
import reachy_mini_conversation_app.profile_store as profile_store_mod
from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
from reachy_mini_conversation_app.profile_store import write_profile, read_profile_from_directory


def test_profile_name_resolves_directly_to_storage_dir() -> None:
    """Built-in profile names should map directly to their on-disk directory."""
    profile_dir = config.resolve_profile_dir("mad_scientist_assistant")

    assert profile_dir.name == "mad_scientist_assistant"
    assert (profile_dir / "profile.md").is_file()


def test_prompts_load_from_compact_builtin_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt loading should read compact built-in profile instructions directly."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "mad_scientist_assistant")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", DEFAULT_PROFILES_DIRECTORY)

    expected = read_profile_from_directory(
        "mad_scientist_assistant",
        DEFAULT_PROFILES_DIRECTORY / "mad_scientist_assistant",
    ).instructions

    assert prompts_mod.get_profile_instructions() == expected


def test_default_session_instructions_load_from_default_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-profile session prompt should come from the built-in default profile."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    expected = read_profile_from_directory("default", DEFAULT_PROFILES_DIRECTORY / "default").instructions

    assert prompts_mod.get_profile_instructions() == expected


def test_bracketed_prompt_line_stays_plain_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bracketed prompt text should not be treated as an include."""
    profile_dir = tmp_path / "literal_prompt"
    write_profile("literal_prompt", profile_dir, "[custom_prompt]\n\nStay extra brief.", [])

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "literal_prompt")

    assert prompts_mod.get_profile_instructions() == "[custom_prompt]\n\nStay extra brief."


def test_session_instructions_fall_back_to_default_for_incomplete_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete selected profiles should not stop session startup."""
    profile_dir = tmp_path / "incomplete_prompt"
    profile_dir.mkdir()

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "incomplete_prompt")

    expected = read_profile_from_directory("default", DEFAULT_PROFILES_DIRECTORY / "default").instructions

    assert prompts_mod.get_profile_instructions() == expected


def test_explicit_default_profile_does_not_fall_back_to_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken explicit default profile should fail without a self-fallback log."""
    default_dir = tmp_path / "default"
    default_dir.mkdir()

    monkeypatch.setattr(config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(profile_store_mod, "DEFAULT_PROFILES_DIRECTORY", tmp_path)

    with caplog.at_level(logging.WARNING, logger="reachy_mini_conversation_app.prompts"):
        with pytest.raises(RuntimeError, match="Default profile has no usable instructions"):
            prompts_mod.get_profile_instructions()

    assert "Using default profile instructions" not in caplog.text


def test_session_voice_defaults_to_hf_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session voice should fall back to the Hugging Face default voice."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    assert prompts_mod.get_session_voice() == "Aiden"


def test_packaged_profiles_win_outside_source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed builds should use packaged profiles, not an unrelated sibling folder."""
    unrelated_profiles = tmp_path / "profiles"
    unrelated_profiles.mkdir()
    packaged_profiles = tmp_path / "package_data" / "profiles"
    packaged_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "_packaged_profiles_directory", lambda: packaged_profiles)

    assert config_mod._resolve_default_profiles_directory() == packaged_profiles
