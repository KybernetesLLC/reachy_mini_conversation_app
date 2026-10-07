"""Tests for Reachy Mini audio startup configuration."""

from __future__ import annotations
from types import SimpleNamespace

from reachy_mini_conversation_app.audio.startup_config import (
    AUDIO_STARTUP_CONFIG,
    WRITE_SETTLE_SECONDS,
    apply_audio_startup_config,
)


class FakeAudio:
    """Fake SDK audio wrapper."""

    def __init__(self, *, result: bool = True, error: Exception | None = None) -> None:
        """Initialize the fake audio wrapper."""
        self.result = result
        self.error = error
        self.calls: list[tuple[object, bool, float]] = []

    def apply_audio_config(
        self,
        config: object,
        *,
        verify: bool = True,
        write_settle_seconds: float = WRITE_SETTLE_SECONDS,
    ) -> bool:
        """Record SDK audio config calls."""
        if self.error is not None:
            raise self.error
        self.calls.append((config, verify, write_settle_seconds))
        return self.result


def test_apply_audio_startup_config_uses_sdk_audio_config_api() -> None:
    """Startup config should delegate writes and verification to the SDK audio API."""
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot)

    assert applied is True
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, True, WRITE_SETTLE_SECONDS)]


def test_apply_audio_startup_config_forwards_sdk_options() -> None:
    """SDK verification options should stay configurable for tests and callers."""
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot, verify=False, write_settle_seconds=0)

    assert applied is True
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, False, 0)]


def test_apply_audio_startup_config_returns_false_without_audio() -> None:
    """Startup should continue when the SDK audio object is unavailable."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=None))

    applied = apply_audio_startup_config(robot)

    assert applied is False


def test_apply_audio_startup_config_returns_false_without_sdk_api() -> None:
    """Startup should continue when the installed SDK does not expose audio config helpers."""
    robot = SimpleNamespace(media=SimpleNamespace(audio=object()))

    applied = apply_audio_startup_config(robot)

    assert applied is False


def test_apply_audio_startup_config_returns_false_when_sdk_returns_false() -> None:
    """SDK application failures should be reported without raising."""
    audio = FakeAudio(result=False)
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot)

    assert applied is False
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, True, WRITE_SETTLE_SECONDS)]


def test_apply_audio_startup_config_returns_false_when_sdk_raises() -> None:
    """Unexpected SDK audio config errors should not prevent app startup."""
    audio = FakeAudio(error=RuntimeError("audio board unavailable"))
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    applied = apply_audio_startup_config(robot)

    assert applied is False
    assert audio.calls == []


# --- the companion, 2026-10-07: the household tunes the mic from a file ------------
#
# Alli's voice reached the backend 12-15 dB quieter than Matt's; the mic's own gain
# control is the lever, and its values are written here at every app start.

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from reachy_mini_conversation_app import config as config_mod  # noqa: E402
from reachy_mini_conversation_app.audio import startup_config as sc  # noqa: E402


def _applied(audio: FakeAudio) -> dict[str, tuple]:
    ((config, _, _),) = audio.calls
    return {name: values for name, values in config}


def test_a_profiles_audio_file_overrides_and_adds_parameters(tmp_path: Path) -> None:
    (tmp_path / "audio.toml").write_text("PP_AGCMAXGAIN = 25.0\nPP_AGCTIME = 0.3\nPP_MGSCALE = [4.0, 1.0, 1.0]\n")
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    assert apply_audio_startup_config(robot, overrides_path=tmp_path / "audio.toml") is True

    applied = _applied(audio)
    assert applied["PP_AGCMAXGAIN"] == (25.0,)
    assert applied["PP_AGCTIME"] == (0.3,)
    assert applied["PP_MIN_NS"] == (0.8,)  # Pollen's tuning stays for the rest
    assert list(applied)[: len(AUDIO_STARTUP_CONFIG)] == [n for n, _ in AUDIO_STARTUP_CONFIG]


def test_a_bad_audio_file_is_reported_and_the_defaults_used(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "audio.toml").write_text("PP_AGCMAXGAIN = 'loud'\n")
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))
    with caplog.at_level(logging.WARNING):
        apply_audio_startup_config(robot, overrides_path=tmp_path / "audio.toml")
    assert _applied(audio) == dict(AUDIO_STARTUP_CONFIG)
    assert any("audio.toml" in r.getMessage() for r in caplog.records)


def test_the_active_profiles_audio_file_is_found_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "companion").mkdir()
    (tmp_path / "companion" / "audio.toml").write_text("PP_AGCMAXGAIN = 30\n")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "companion")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    audio = FakeAudio()
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio))

    apply_audio_startup_config(robot)

    assert _applied(audio)["PP_AGCMAXGAIN"] == (30.0,)


def test_no_audio_file_means_pollens_tuning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "companion")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", tmp_path)
    audio = FakeAudio()
    apply_audio_startup_config(SimpleNamespace(media=SimpleNamespace(audio=audio)))
    assert _applied(audio) == dict(AUDIO_STARTUP_CONFIG)
    assert sc.AUDIO_OVERRIDES_FILE == "audio.toml"
