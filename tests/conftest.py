"""Pytest configuration for path setup."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1].resolve()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Make tests reproducible by ignoring machine-specific profile/tool env config.
# Without this, importing config during test collection can pick up a developer's
# local .env and fail before tests run.
os.environ["REACHY_MINI_SKIP_DOTENV"] = "1"
os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
os.environ.pop("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY", None)
os.environ.pop("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY", None)
# The wire log records only when its path is set: never from a test run, even on the robot.
os.environ.pop("REACHY_COMPANION_WIRE_LOG", None)


# The companion's read functions (companion ticket 33). The fork's .venv cannot import the
# companion: a test that needs it names a companion checkout in REACHY_COMPANION_CHECKOUT, and is
# skipped without one (or with one older than ticket 33). Every other test runs on the app's own reads.
COMPANION_ENV = "REACHY_COMPANION_CHECKOUT"


def _forget_companion() -> None:
    for name in [m for m in sys.modules if m == "reachy_companion" or m.startswith("reachy_companion.")]:
        del sys.modules[name]


@pytest.fixture
def companion_on(monkeypatch: pytest.MonkeyPatch):
    """Point the app at a companion checkout the way the deploy does (its app tools folder)."""
    from reachy_mini_conversation_app import companion

    checkout = os.environ.get(COMPANION_ENV, "").strip()
    if not checkout or not (Path(checkout) / "src" / "reachy_companion" / "luma.py").exists():
        pytest.skip(f"{COMPANION_ENV} names no companion checkout with ticket 33's reads")
    # The companion's own dependencies are this venv's dev dependencies (pyproject.toml): a missing one
    # fails here, loudly, not as the app's defaults under a read that caches None (companion ticket 218).
    import msgspec  # noqa: F401  (the settings module's)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(companion, "_found", {})
    monkeypatch.setenv(companion.TOOLS_ENV, str(Path(checkout) / "app_tools"))
    _forget_companion()
    yield Path(checkout)
    _forget_companion()


@pytest.fixture
def companion_off(monkeypatch: pytest.MonkeyPatch):
    """Run the app with no companion to import: each read is the app's own."""
    from reachy_mini_conversation_app import companion

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(companion, "_found", {})
    monkeypatch.delenv(companion.TOOLS_ENV, raising=False)
    _forget_companion()
    yield
