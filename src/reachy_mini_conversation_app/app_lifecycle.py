"""Helpers for app startup and shutdown lifecycle behavior."""

import logging
import urllib.error
import urllib.request
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.config import config, set_custom_profile
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME, migrate_legacy_profiles
from reachy_mini_conversation_app.tools.core_tools import initialize_tools


_STOP_CURRENT_APP_PATH = "/api/apps/stop-current-app"
_STOP_CURRENT_APP_TIMEOUT_S = 2.0


def initialize_tools_with_default_fallback(
    instance_path: str | Path | None,
    logger: logging.Logger,
) -> str | None:
    """Load the tool registry, degrading to the default profile if need be.

    Returns the profile that was abandoned, or None when the selection loaded.
    """
    # User profiles predating profile.md were never migrated on disk; convert
    # them here so the strict readers below (and the settings UI) see them.
    try:
        migrate_legacy_profiles(config.user_personalities_root())
    except Exception as exc:
        logger.warning("Legacy profile migration failed: %s", exc)

    try:
        initialize_tools(instance_path=instance_path)
        return None
    except Exception as exc:
        selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
        if not selected_profile or selected_profile == DEFAULT_PROFILE_NAME:
            raise

        # set_custom_profile is a no-op while LOCKED_PROFILE is pinned, so
        # confirm the switch took effect before announcing a fallback.
        set_custom_profile(DEFAULT_PROFILE_NAME)
        if config.REACHY_MINI_CUSTOM_PROFILE != DEFAULT_PROFILE_NAME:
            logger.error(
                "Profile %r could not be loaded (%s) and this build is locked to it, "
                "so there is no profile to fall back to.",
                selected_profile,
                exc,
            )
            raise

        logger.error(
            "Profile %r could not be loaded (%s); starting on the %r profile instead. "
            "Reselect or repair it from the settings UI to restore it.",
            selected_profile,
            exc,
            DEFAULT_PROFILE_NAME,
        )
        initialize_tools(instance_path=instance_path, force=True)
        return selected_profile


def request_stop_current_app(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Request the Reachy Mini daemon to stop the current app."""
    stop_current_app_url = f"http://{robot.client.host}:{robot.client.port}{_STOP_CURRENT_APP_PATH}"
    request = urllib.request.Request(stop_current_app_url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_STOP_CURRENT_APP_TIMEOUT_S) as response:
            response.read()
    except urllib.error.URLError as e:
        logger.error("Failed to request current app stop via %s: %s", stop_current_app_url, e)
        return False

    logger.info("Requested current app stop via %s", stop_current_app_url)
    return True
