"""Per-personality voice resolution: authored voice, instance override, fallback."""

from pathlib import Path

import pytest

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice
from reachy_mini_conversation_app.personality import save_user_personality
from reachy_mini_conversation_app.profile_store import read_profile_from_directory
from reachy_mini_conversation_app.profile_voices import (
    read_profile_voice,
    get_profile_voices_path,
    read_profile_voice_override,
    clear_profile_voice_override,
    write_profile_voice_override,
)


@pytest.fixture
def instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a writable throwaway instance directory."""
    monkeypatch.setattr(config, "INSTANCE_PATH", tmp_path)
    return tmp_path


def test_an_empty_voice_authors_none(instance: Path) -> None:
    """An empty voice is how the UI says "no voice of its own", not "say nothing"."""
    save_user_personality("quiet_one", "Be brief.", voice="")

    profile = read_profile_from_directory("quiet_one", instance / "user_personalities" / "quiet_one")
    assert profile.voice is None


def test_an_authored_voice_is_written(instance: Path) -> None:
    """A voice chosen in the modal lands in the profile document."""
    save_user_personality("bard", "Be lyrical.", voice="Serena")

    profile = read_profile_from_directory("bard", instance / "user_personalities" / "bard")
    assert profile.voice == "Serena"


def test_an_override_wins_over_the_authored_voice(instance: Path) -> None:
    """Packaged profiles are read-only, so the override is what makes them editable."""
    save_user_personality("bard", "Be lyrical.", voice="Serena")
    write_profile_voice_override("user_personalities/bard", "Dylan", instance)

    assert read_profile_voice("user_personalities/bard", instance) == "Dylan"


def test_clearing_an_override_restores_the_authored_voice(instance: Path) -> None:
    """Clearing is a restore, not a reset to the backend default."""
    save_user_personality("bard", "Be lyrical.", voice="Serena")
    write_profile_voice_override("user_personalities/bard", "Dylan", instance)

    assert clear_profile_voice_override("user_personalities/bard", instance) is True
    assert read_profile_voice_override("user_personalities/bard", instance) is None
    assert read_profile_voice("user_personalities/bard", instance) == "Serena"


def test_the_store_file_goes_away_with_its_last_override(instance: Path) -> None:
    """No leftover empty file once nothing is overridden."""
    save_user_personality("bard", "Be lyrical.", voice="Serena")
    write_profile_voice_override("user_personalities/bard", "Dylan", instance)
    assert get_profile_voices_path(instance).exists()

    clear_profile_voice_override("user_personalities/bard", instance)
    assert not get_profile_voices_path(instance).exists()


def test_saving_a_personality_drops_its_override(instance: Path) -> None:
    """Editing the voice in the modal must not be shadowed by a stale override."""
    save_user_personality("bard", "Be lyrical.", voice="Serena")
    write_profile_voice_override("user_personalities/bard", "Dylan", instance)

    save_user_personality("bard", "Be lyrical.", voice="Vivian", overwrite=True)

    assert read_profile_voice_override("user_personalities/bard", instance) is None
    assert read_profile_voice("user_personalities/bard", instance) == "Vivian"


def test_the_session_voice_follows_the_active_personality(
    instance: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each personality speaks with its own voice, with no handler in the loop."""
    save_user_personality("bard", "Be lyrical.", voice="Serena")
    save_user_personality("scout", "Be terse.", voice="Vivian")

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/bard")
    assert get_session_voice(default="fallback") == "Serena"

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/scout")
    assert get_session_voice(default="fallback") == "Vivian"


def test_a_personality_without_a_voice_falls_back(
    instance: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unauthored voice resolves to the backend default at session time."""
    save_user_personality("quiet_one", "Be brief.", voice="")
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/quiet_one")

    assert get_session_voice(default="fallback") == "fallback"
