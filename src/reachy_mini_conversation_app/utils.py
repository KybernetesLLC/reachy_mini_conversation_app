from __future__ import annotations
import os
import math
import stat
import logging
import argparse
import warnings
from typing import Mapping, Callable, Optional
from pathlib import Path

from reachy_mini_conversation_app import companion


logger = logging.getLogger(__name__)


_clamp: Callable[[float, float, float], float] | None = None


def clamp(x: float, lo: float, hi: float) -> float:
    """Return ``x`` held within ``lo`` to ``hi``: the companion's numeric.clamp, else the same expression.

    The companion's function is looked up once, at the first call.
    """
    global _clamp
    if _clamp is None:
        _clamp = companion.value("numeric", "clamp", lambda v, low, high: max(low, min(high, v)))
    return _clamp(x, lo, hi)


def finite_number(value: object) -> float | None:
    """Return a finite int or float as a float; None for a bool, a string, None, NaN or infinity.

    The one number check for what arrives over JSON-RPC or a command: a bool or a string is no number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def b64_size(text: str) -> int:
    """Return the exact decoded size in bytes of a base64 string (its padding is not data). Pure."""
    return len(text) * 3 // 4 - text[-2:].count("=")


def env_number(name: str, default: float, environ: Mapping[str, str] | None = None) -> float:
    """Return a number >= 0 from the environment, or ``default`` when it is unset, blank or invalid (invalid warns)."""
    raw = str((os.environ if environ is None else environ).get(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        logger.warning("%s=%r is not a number >= 0; using %s", name, raw, default)
        return default
    return value


def env_flag(name: str, off: str) -> bool:
    """Return False when the environment value is ``off``; True when it is unset, blank or ``default``.

    Any other value warns and keeps True.
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "default"):
        return True
    if raw == off:
        return False
    logger.warning("%s=%r is not %r; keeping the default", name, raw, off)
    return True


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8, whole or not at all, keeping the file's mode (the companion's
    jsonfile.write_bytes_atomic); a plain write when the companion has none."""
    write = companion.value("jsonfile", "write_bytes_atomic", None)
    if write is None:
        path.write_text(text, encoding="utf-8")
        return
    try:
        mode: int | None = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        mode = None
    write(path, text.encode("utf-8"), mode=mode, mkdir=False)


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--ui",
        default=False,
        action="store_true",
        help="Serve the web UI at http://127.0.0.1:7860/, in addition to console mode",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    subparsers = parser.add_subparsers(dest="command")
    tool_spaces_parser = subparsers.add_parser("tool-spaces", help="Manage installed Hugging Face Space tool sources")
    tool_spaces_subparsers = tool_spaces_parser.add_subparsers(dest="tool_spaces_command", required=True)

    add_parser = tool_spaces_subparsers.add_parser("add", help="Install one Space tool source by slug")
    add_parser.add_argument("space_slug", help="Hugging Face Space slug in the form owner/space-name")
    add_parser.add_argument(
        "--install-only",
        action="store_true",
        default=False,
        help="Install the Space without enabling its tools in any profile.",
    )
    add_parser.add_argument(
        "--profile",
        dest="profile",
        default=None,
        metavar="PROFILE",
        help="Enable tools in this profile instead of the active profile.",
    )

    remove_parser = tool_spaces_subparsers.add_parser("remove", help="Remove one installed Space tool source")
    remove_parser.add_argument("space_slug", help="Installed Hugging Face Space slug in the form owner/space-name")

    tool_spaces_subparsers.add_parser("list", help="List installed Space tool sources")
    return parser.parse_known_args()


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
