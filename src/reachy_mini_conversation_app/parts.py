"""The parts the companion publishes: named, versioned pieces of what he is told (route all, Unit 2).

ConvoAll rule 8 (the companion's docs/formal/conversation-d/model.md, round 5). The companion's
`parts_link.py` sends `conversation.parts`; this keeps the latest version of each part, as a ROS 2
transient-local, keep-last-1 subscriber would. A snapshot is the whole state: a part not in it
goes. A non-snapshot part is taken only when it is newer than the part held and than the last
snapshot. Each part named that is held is acknowledged with the version held, taken or not.

Nothing reads the parts yet (Unit 4 will). `expires_at` is stored, not acted on (Unit 5).
"""

from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Part:
    """One part's latest version. `value` None is a clear."""

    name: str
    version: int
    value: Any
    session: Optional[float]
    expires_at: Optional[float]


@dataclass(frozen=True)
class Parts:
    """The parts held, and the version of the last snapshot taken."""

    held: dict[str, Part]
    floor: int = 0


def _part(raw: Any) -> Optional[Part]:
    if not isinstance(raw, dict):
        return None
    name, version = raw.get("name"), raw.get("version")
    if not isinstance(name, str) or not name or isinstance(version, bool):
        return None
    if not isinstance(version, int):
        return None
    session, expires = raw.get("session"), raw.get("expires_at")
    return Part(
        name=name,
        version=version,
        value=raw.get("value"),
        session=float(session) if isinstance(session, (int, float)) else None,
        expires_at=float(expires) if isinstance(expires, (int, float)) else None,
    )


def apply(parts: Parts, params: dict[str, Any]) -> tuple[Parts, list[dict[str, Any]], list[str]]:
    """Take one `conversation.parts` message: (the new record, the acks, what changed and why)."""
    raw = params.get("parts")
    incoming = [p for p in (_part(r) for r in (raw if isinstance(raw, list) else [])) if p]
    changes: list[str] = []
    if params.get("snapshot") is True:
        floor = params.get("version")
        held = {p.name: p for p in incoming}
        for name in sorted(set(parts.held) - set(held)):
            changes.append(f"{name} dropped (not in the snapshot)")
        changes.extend(f"{p.name} v{p.version} (snapshot)" for p in incoming)
        new = Parts(held=held, floor=floor if isinstance(floor, int) else 0)
    else:
        held = dict(parts.held)
        for p in incoming:
            current = held.get(p.name)
            if p.version <= parts.floor or (current is not None and p.version <= current.version):
                changes.append(f"{p.name} v{p.version} ignored (older)")
                continue
            held[p.name] = p
            changes.append(f"{p.name} v{p.version}")
        new = Parts(held=held, floor=parts.floor)
    names = {p.name for p in incoming}
    acks = [{"name": n, "version": new.held[n].version} for n in sorted(names) if n in new.held]
    return new, acks, changes
