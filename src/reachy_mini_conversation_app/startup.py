"""Start-up settings for an app another process supervises (the companion, R-36).

Each is read from the environment, which the app inherits from the daemon that
starts it. Unset, each keeps upstream's behaviour exactly.
"""

import os
import logging


logger = logging.getLogger(__name__)


def _flag(name: str, off: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "default"):
        return True
    if raw == off:
        return False
    logger.warning("%s=%r is not %r; keeping the default", name, raw, off)
    return True


def session_on_start() -> bool:
    """Return False (``closed``) to start parked: no session, so no greeting, until one is opened."""
    return _flag("REACHY_MINI_SESSION_ON_START", "closed")


def capture_on_start() -> bool:
    """Return False (``off``) to start with the microphone and speaker pipelines stopped."""
    return _flag("REACHY_MINI_CAPTURE_ON_START", "off")


def motion_on_start() -> bool:
    """Return False (``still``) for no wake-up move and no idle breathing until a pose command."""
    return _flag("REACHY_MINI_MOTION_ON_START", "still")
