"""Startup configuration for the Reachy Mini audio processor."""

from __future__ import annotations
import logging
import tomllib
from pathlib import Path
from collections.abc import Sequence

from reachy_mini_conversation_app.config import config


AudioControlValue = float | int
AudioStartupParameter = tuple[str, tuple[AudioControlValue, ...]]
WRITE_SETTLE_SECONDS = 0.1
# A file beside the active profile's profile.md: parameter names to values, TOML.
# Its entries replace or add to AUDIO_STARTUP_CONFIG. The companion's household
# tunes the mic's own processing there (2026-10-07: one voice reached the
# backend 12-15 dB quieter than the other, and the mic's gain control is the setting to change).
AUDIO_OVERRIDES_FILE = "audio.toml"

AUDIO_STARTUP_CONFIG: tuple[AudioStartupParameter, ...] = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


def default_overrides_path() -> Path | None:
    """Return the active custom profile's audio.toml path, if a custom profile is selected."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return None
    return config.resolve_profile_dir(profile) / AUDIO_OVERRIDES_FILE


def load_overrides(path: Path | None, log: logging.Logger) -> tuple[AudioStartupParameter, ...]:
    """Read `path` as parameter overrides; an absent file is none, a bad one is none with a warning."""
    if path is None or not path.is_file():
        return ()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        overrides: list[AudioStartupParameter] = []
        for name, value in raw.items():
            values = value if isinstance(value, list) else [value]
            if not values or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                raise ValueError(f"{name}: values must be numbers")
            overrides.append((str(name), tuple(values)))
        return tuple(overrides)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        log.warning("Ignoring %s: %s", path, exc)
        return ()


def merged_config(overrides: Sequence[AudioStartupParameter]) -> tuple[AudioStartupParameter, ...]:
    """AUDIO_STARTUP_CONFIG with `overrides` replacing matching names, new names appended."""
    by_name = dict(overrides)
    merged = [(name, by_name.pop(name, values)) for name, values in AUDIO_STARTUP_CONFIG]
    merged.extend(by_name.items())
    return tuple(merged)


def apply_audio_startup_config(
    robot: object,
    *,
    logger: logging.Logger | None = None,
    verify: bool = True,
    write_settle_seconds: float = WRITE_SETTLE_SECONDS,
    overrides_path: Path | None = None,
) -> bool:
    """Apply the tuned XVF3800 audio configuration for the conversation app."""
    log = logger or logging.getLogger(__name__)
    audio = getattr(getattr(robot, "media", None), "audio", None)
    startup_config = merged_config(
        load_overrides(overrides_path if overrides_path is not None else default_overrides_path(), log)
    )

    if audio is None:
        log.warning("Skipping Reachy audio startup config: robot media audio is unavailable.")
        return False

    apply_audio_config = getattr(audio, "apply_audio_config", None)
    if not callable(apply_audio_config):
        log.warning("Skipping Reachy audio startup config: SDK audio config API is unavailable.")
        return False

    try:
        applied = bool(
            apply_audio_config(
                startup_config,
                verify=verify,
                write_settle_seconds=write_settle_seconds,
            )
        )
    except Exception as exc:
        log.warning("Skipping Reachy audio startup config: SDK audio config failed: %s", exc)
        return False

    if applied:
        log.info("Applied Reachy audio startup config: %s", _format_config(startup_config))
    else:
        log.warning("Reachy audio startup config was not applied.")

    return applied


def _format_config(config: Sequence[AudioStartupParameter]) -> str:
    return ", ".join(f"{name}={' '.join(str(value) for value in values)}" for name, values in config)
