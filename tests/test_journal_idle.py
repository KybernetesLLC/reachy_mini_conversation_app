"""The replay check: a journal excerpt in, every speech stream idle past the threshold out.

The 05:22 phantom (2026-10-26) came 9 h 19 m after the speech stream's last write; this
finds each such stretch in ``journalctl`` output (journal_idle.py, beside this file).
"""

from __future__ import annotations

import journal_idle as ji


def _journal(stamp: str, source: str, message: str) -> str:
    """One journal line as the robot writes it (journalctl -o short-iso)."""
    day, clock = stamp.split()
    return (
        f"{day}T{clock}-04:00 brain python[1173]: runner - WARNING - {stamp},000 INFO "
        f"reachy_mini_conversation_app.{source} | {message}"
    )


_LAG = "Playback latency: last reply heard after 0.21 s"
_JOURNAL = [
    _journal("2026-10-25 19:44:45", "playback_health:252", "Speaker stream rebuilt (trigger: x)"),
    _journal("2026-10-25 20:03:14", "console:1093", "Reply heard after 0.21 s (first push to ...)"),
    _journal("2026-10-25 21:00:00", "playback_health:325", f"{_LAG}, 3406 s ago"),
    _journal("2026-10-26 05:22:40", "playback_health:325", f"{_LAG}, 33566 s ago"),
    _journal("2026-10-26 06:38:29", "playback_health:252", "Speaker stream rebuilt (trigger: x)"),
    _journal("2026-10-26 06:40:00", "console:1360", "Audio capture stopped via /rpc"),
    _journal("2026-10-26 12:40:00", "console:1360", "Audio capture started via /rpc"),
    _journal("2026-10-26 13:00:00", "console:1093", "Reply not heard (1 in a row): 0.00 s heard"),
    _journal("2026-10-26 17:00:00", "playback_health:325", f"{_LAG}, 14400 s ago"),
]


def test_the_replay_check_finds_every_speech_stream_idle_past_the_threshold() -> None:
    """10-25: 10 h 35 min from the last reply to the next rebuild; capture off is no idle."""
    import datetime

    found = ji.idle_stretches(_JOURNAL, 3 * 3600.0)
    at = datetime.datetime.fromisoformat
    assert found == [
        ji.IdleStretch(at("2026-10-25 20:03:14"), at("2026-10-26 06:38:29"), ongoing=False),
        ji.IdleStretch(at("2026-10-26 13:00:00"), at("2026-10-26 17:00:00"), ongoing=True),
    ]
    assert [round(s.seconds) for s in found] == [38115, 14400]
    assert ji.idle_stretches(_JOURNAL, 12 * 3600.0) == []


def test_the_replay_check_reads_the_classic_journal_format() -> None:
    """Without -o short-iso, the app's own timestamp inside the line is used."""
    lines = [
        "Oct 25 20:03:14 brain python[1]: 2026-10-25 20:03:14,483 INFO x | Reply heard after 0.2 s",
        "Oct 26 05:22:40 brain python[1]: 2026-10-26 05:22:40,001 INFO x | Playback latency",
    ]
    (stretch,) = ji.idle_stretches(lines, 3600.0)
    assert stretch.ongoing and round(stretch.seconds) == 33566


def test_the_replay_check_counts_a_capture_start_and_the_first_line_as_writes() -> None:
    """The stream is new at capture on, and nothing before the excerpt is known."""
    import datetime

    at = datetime.datetime.fromisoformat
    lag = "Playback latency: none"
    first = [
        _journal("2026-10-26 01:00:00", "playback_health:325", lag),
        _journal("2026-10-26 05:00:00", "playback_health:325", lag),
    ]
    assert ji.idle_stretches(first, 3 * 3600.0) == [
        ji.IdleStretch(at("2026-10-26 01:00:00"), at("2026-10-26 05:00:00"), ongoing=True)
    ]
    capture = [
        _journal("2026-10-26 01:00:00", "console:1093", "Reply heard after 0.2 s"),
        _journal("2026-10-26 01:30:00", "console:1360", "Audio capture stopped via /rpc"),
        _journal("2026-10-26 03:00:00", "console:1360", "Audio capture started via /rpc"),
        _journal("2026-10-26 05:00:00", "playback_health:325", lag),
    ]
    assert ji.idle_stretches(capture, 3 * 3600.0) == []  # 2 h since capture on
    assert ji.idle_stretches(capture, 1.5 * 3600.0) == [
        ji.IdleStretch(at("2026-10-26 03:00:00"), at("2026-10-26 05:00:00"), ongoing=True)
    ]
