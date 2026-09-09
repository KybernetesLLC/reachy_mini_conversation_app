"""Encoding of a personality selection into the text written on an NFC tag.

Tags used to carry an opaque code — ``976F33E2`` — that only meant something
through a JSON mapping kept beside the app. A tag was therefore useless on its
own: move it to another robot, or lose the mapping file, and it pointed at
nothing. The tag now carries the personality itself, so it is self-describing.

Two prefixes keep the namespaces apart:

    built-in ``noir_detective``            -> ``hf_noir_detective``
    user ``user_personalities/my_bot``     -> ``usr_my_bot``

Both sides are prefixed rather than only the built-ins. Nothing stops someone
from naming their own personality after a built-in one, or literally naming it
``hf_noir_detective`` — the name sanitizer allows it. Prefixing one side only
would make those cases ambiguous; prefixing both closes the question.

The "(built-in default)" selection has no token on purpose: a tag exists to
*apply* a personality, and reverting to the default is what removing the tag
already does. :func:`to_tag_token` returns None for it so callers refuse the
write rather than burning a tag on a no-op.
"""

from __future__ import annotations

from typing import Optional

BUILTIN_PREFIX = "hf_"
USER_PREFIX = "usr_"
USER_DIR = "user_personalities"

# The sentinel used across the app for "no personality profile selected".
DEFAULT_SELECTION = "(built-in default)"


def to_tag_token(selection: str) -> Optional[str]:
    """Return the text to write on a tag for a personality selection.

    Returns None when the selection cannot be written: the default sentinel,
    or an empty name.
    """
    s = (selection or "").strip()
    if not s or s == DEFAULT_SELECTION:
        return None
    if s.startswith(USER_DIR + "/"):
        name = s[len(USER_DIR) + 1:].strip("/")
        return f"{USER_PREFIX}{name}" if name else None
    return f"{BUILTIN_PREFIX}{s}"


def from_tag_token(token: str) -> Optional[str]:
    """Return the personality selection a tag's text refers to.

    Returns None for anything that is not a personality token — a blank tag, a
    leftover code from the old mapping scheme, or a tag written by something
    else entirely. The caller decides what to do with an unknown tag; this
    function does not guess.
    """
    t = (token or "").strip()
    if t.startswith(USER_PREFIX):
        name = t[len(USER_PREFIX):]
        return f"{USER_DIR}/{name}" if name else None
    if t.startswith(BUILTIN_PREFIX):
        name = t[len(BUILTIN_PREFIX):]
        return name or None
    return None


def is_personality_token(token: str) -> bool:
    """Whether a tag's text looks like a personality token at all."""
    return from_tag_token(token) is not None
