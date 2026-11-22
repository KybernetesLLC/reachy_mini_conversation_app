"""The companion's parts, kept by version (route all, Unit 2; ConvoAll rule 8)."""

from typing import Any

from reachy_mini_conversation_app.parts import Parts, apply


def _p(name: str, version: int, value: Any = "v", **extra: Any) -> dict[str, Any]:
    return {"name": name, "version": version, "value": value, **extra}


def _held(parts: Parts) -> dict[str, tuple[int, Any]]:
    return {n: (p.version, p.value) for n, p in parts.held.items()}


def test_a_newer_version_wins() -> None:
    """Newer wins."""
    parts, acks, _ = apply(Parts(held={}), {"parts": [_p("summary", 1, "s1")]})
    parts, acks, _ = apply(parts, {"parts": [_p("summary", 3, "s3")]})
    assert _held(parts) == {"summary": (3, "s3")}
    assert acks == [{"name": "summary", "version": 3}]


def test_an_older_version_arriving_late_is_ignored_and_acked_with_the_held_one() -> None:
    """An older version arriving late is ignored; the ack names the version held."""
    parts, _, _ = apply(Parts(held={}), {"parts": [_p("summary", 3, "s3")]})
    parts, acks, changes = apply(parts, {"parts": [_p("summary", 2, "s2")]})
    assert _held(parts) == {"summary": (3, "s3")}
    assert acks == [{"name": "summary", "version": 3}]
    assert changes == ["summary v2 ignored (older)"]


def test_a_clear_is_a_value() -> None:
    """A clear is a version like any value."""
    parts, _, _ = apply(Parts(held={}), {"parts": [_p("summary", 1, "s1")]})
    parts, acks, _ = apply(parts, {"parts": [_p("summary", 2, None)]})
    assert _held(parts) == {"summary": (2, None)}
    assert acks == [{"name": "summary", "version": 2}]


def test_a_snapshot_drops_the_parts_not_in_it() -> None:
    """A snapshot is the whole state."""
    parts, _, _ = apply(
        Parts(held={}), {"parts": [_p("summary", 1), _p("just_now", 2, expires_at=110.0)]}
    )
    parts, acks, changes = apply(
        parts, {"snapshot": True, "version": 4, "parts": [_p("summary", 3, "s3")]}
    )
    assert _held(parts) == {"summary": (3, "s3")}
    assert acks == [{"name": "summary", "version": 3}]
    assert "just_now dropped (not in the snapshot)" in changes


def test_a_snapshot_replaces_higher_versions_from_a_companion_that_restarted() -> None:
    """The companion's versions restart with it; its snapshot is taken whole."""
    parts, _, _ = apply(Parts(held={}), {"parts": [_p("summary", 40)]})
    parts, _, _ = apply(parts, {"snapshot": True, "version": 1, "parts": [_p("summary", 1, "s")]})
    assert _held(parts) == {"summary": (1, "s")}


def test_a_part_older_than_the_snapshot_is_ignored() -> None:
    """A part sent before the snapshot, arriving after it, does not come back."""
    parts, _, _ = apply(Parts(held={}), {"snapshot": True, "version": 5, "parts": []})
    parts, acks, _ = apply(parts, {"parts": [_p("just_now", 4)]})
    assert parts.held == {}
    assert acks == []
    parts, acks, _ = apply(parts, {"parts": [_p("just_now", 6)]})
    assert acks == [{"name": "just_now", "version": 6}]


def test_the_session_and_expiry_are_kept() -> None:
    """The session and the expiry are stored as sent."""
    parts, _, _ = apply(
        Parts(held={}), {"parts": [_p("just_now", 1, session=12.5, expires_at=99.0)]}
    )
    part = parts.held["just_now"]
    assert (part.session, part.expires_at) == (12.5, 99.0)


def test_malformed_parts_are_skipped() -> None:
    """A part without a name or an integer version is skipped."""
    parts, acks, _ = apply(
        Parts(held={}),
        {"parts": [{"name": "x"}, {"version": 1}, _p("y", True), "junk", _p("ok", 1)]},
    )
    assert set(parts.held) == {"ok"}
    assert acks == [{"name": "ok", "version": 1}]
