"""When the create_accessory_personality tool is offered to a profile."""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.profile_store import write_profile
from reachy_mini_conversation_app.profile_toolsets import write_profile_tool_override
from reachy_mini_conversation_app.tools.core_tools import (
    ACCESSORY_PERSONALITY_TOOL_NAME,
    _read_profile_tool_names,
    set_accessory_personality_tool_available,
)


@pytest.fixture
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Build a profile whose authored tools do not mention the accessory tool."""
    name = "accessory_probe"
    write_profile(name, tmp_path / name, "hello", ["dance"])
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", name)
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config_mod.config, "INSTANCE_PATH", tmp_path)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    return name


@pytest.fixture(autouse=True)
def _restore_availability() -> object:
    """Availability is process-wide state; put it back for the next test."""
    yield
    set_accessory_personality_tool_available(False)


def test_no_reader_means_the_tool_is_never_offered(profile: str, tmp_path: Path) -> None:
    """The tool cannot run without a reader, so it must not reach the model."""
    set_accessory_personality_tool_available(False)

    assert ACCESSORY_PERSONALITY_TOOL_NAME not in _read_profile_tool_names(tmp_path)


def test_a_reader_offers_the_tool_to_a_profile_that_never_mentioned_it(profile: str, tmp_path: Path) -> None:
    """Shipped profiles do not list it, yet the flow has to work out of the box."""
    set_accessory_personality_tool_available(True)

    assert ACCESSORY_PERSONALITY_TOOL_NAME in _read_profile_tool_names(tmp_path)


def test_unticking_the_tool_actually_disables_it(profile: str, tmp_path: Path) -> None:
    """An explicit toolset is a decision: the default must not override it."""
    set_accessory_personality_tool_available(True)
    write_profile_tool_override(profile, ["dance"], tmp_path)

    assert ACCESSORY_PERSONALITY_TOOL_NAME not in _read_profile_tool_names(tmp_path)


def test_an_explicit_toolset_may_keep_the_tool(profile: str, tmp_path: Path) -> None:
    """Ticking it is a decision too, and survives for the same reason."""
    set_accessory_personality_tool_available(True)
    write_profile_tool_override(profile, ["dance", ACCESSORY_PERSONALITY_TOOL_NAME], tmp_path)

    assert ACCESSORY_PERSONALITY_TOOL_NAME in _read_profile_tool_names(tmp_path)


def test_losing_the_reader_withdraws_an_explicitly_enabled_tool(profile: str, tmp_path: Path) -> None:
    """Even a profile that asked for it cannot have it with no hardware."""
    write_profile_tool_override(profile, ["dance", ACCESSORY_PERSONALITY_TOOL_NAME], tmp_path)
    set_accessory_personality_tool_available(False)

    assert ACCESSORY_PERSONALITY_TOOL_NAME not in _read_profile_tool_names(tmp_path)
