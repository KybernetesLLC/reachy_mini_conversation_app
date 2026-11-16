"""The companion's own read functions, imported from its code (companion ticket 33, the owner's option A, 11-16).

The fork reads four things the companion owns: the commit it was deployed from, the talk summary,
the room's light, and the wire log's path, open and retention. Each goes through the companion
module that owns it. The companion's code sits beside its app tools: the deploy names that folder
in REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY (its deploy/environment.conf), and the companion's app
tools put its ``src`` on sys.path the same way. Nothing goes on PYTHONPATH, which would reach the
daemon too.

When the companion cannot be imported (a fork run on its own, a test) or is an older one without
the function, ``function`` returns None and the caller keeps its own read; that is logged once.
"""

from __future__ import annotations
import os
import sys
import logging
import importlib
from typing import Any, Callable, Optional
from pathlib import Path


logger = logging.getLogger(__name__)

TOOLS_ENV = "REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY"
_found: dict[tuple[str, str], Optional[Callable[..., Any]]] = {}


def src_dir() -> Optional[Path]:
    """Return the companion's ``src``, beside its app tools folder, or None when no folder is named."""
    tools = os.environ.get(TOOLS_ENV, "").strip()
    return Path(tools).expanduser().resolve().parent / "src" if tools else None


def function(module: str, name: str) -> Optional[Callable[..., Any]]:
    """Return ``reachy_companion.<module>.<name>``, or None when it cannot be imported (logged once)."""
    key = (module, name)
    if key not in _found:
        src = src_dir()
        if src is not None and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            _found[key] = getattr(importlib.import_module(f"reachy_companion.{module}"), name)
        except Exception as exc:  # not there, an older companion, or one that fails to import
            _found[key] = None
            logger.warning("companion: no %s.%s (%s); the app's own read is used", module, name, exc)
    return _found[key]
