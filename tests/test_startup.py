"""Start-up settings for a supervised app: parked, deaf and still when told (R-36)."""

import pytest

from reachy_mini_conversation_app import startup


SETTINGS = [
    ("REACHY_MINI_SESSION_ON_START", "session_on_start", "closed"),
    ("REACHY_MINI_CAPTURE_ON_START", "capture_on_start", "off"),
    ("REACHY_MINI_MOTION_ON_START", "motion_on_start", "still"),
]


@pytest.mark.parametrize("name, fn, off", SETTINGS)
def test_unset_keeps_upstream(monkeypatch: pytest.MonkeyPatch, name: str, fn: str, off: str) -> None:
    """Upstream's behaviour when nothing is set."""
    monkeypatch.delenv(name, raising=False)
    assert getattr(startup, fn)() is True


@pytest.mark.parametrize("name, fn, off", SETTINGS)
def test_each_setting_turns_its_part_off(monkeypatch: pytest.MonkeyPatch, name: str, fn: str, off: str) -> None:
    """The one word for each turns it off; case and spaces do not matter."""
    monkeypatch.setenv(name, f" {off.upper()} ")
    assert getattr(startup, fn)() is False


@pytest.mark.parametrize("name, fn, off", SETTINGS)
def test_an_unknown_value_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, name: str, fn: str, off: str
) -> None:
    """A typo must not silently change start-up."""
    monkeypatch.setenv(name, "maybe")
    assert getattr(startup, fn)() is True
    assert name in caplog.text
