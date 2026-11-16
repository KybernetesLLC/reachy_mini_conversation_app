"""The companion's read functions, with the companion importable and not (companion ticket 33)."""

from __future__ import annotations
import json
import logging
import sqlite3
import datetime
from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app import wire_log, companion, talk_prune
from reachy_mini_conversation_app.tools import camera as camera_mod


def _summary_db(path: Path, value: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")
    if value is not None:
        con.execute("INSERT INTO settings VALUES ('talk_summary', ?, 0)", (value,))
    con.commit()
    con.close()
    return path


SUMMARIES = [
    None,
    "",
    "{not json",
    json.dumps(["a list"]),
    json.dumps({"session": 1.0, "up_to": 150.0, "text": "   "}),
    json.dumps({"session": 1.0, "up_to": None, "text": " Alli asked about ferries. "}),
    json.dumps({"session": 9.0, "up_to": "1793800000.5", "text": "Another talk's."}),
    json.dumps({"up_to": "soon", "text": "A bad time."}),
]


# --- the helper ----------------------------------------------------------------------------


def test_without_the_companion_each_read_is_the_apps_own_and_logged_once(companion_off, caplog) -> None:
    caplog.set_level(logging.WARNING, logger=companion.__name__)
    assert companion.function("data_paths", "deployed_commit") is None
    assert companion.function("data_paths", "deployed_commit") is None
    assert len([r for r in caplog.records if "data_paths.deployed_commit" in r.getMessage()]) == 1


def test_the_companion_is_found_beside_its_app_tools(companion_on) -> None:
    read = companion.function("data_paths", "deployed_commit")
    assert read is not None and read.__module__ == "reachy_companion.data_paths"
    assert companion.src_dir() == (companion_on / "src").resolve()


def test_an_older_companion_without_the_function_falls_back(companion_on, caplog) -> None:
    caplog.set_level(logging.WARNING, logger=companion.__name__)
    assert companion.function("data_paths", "no_such_read") is None
    assert "no data_paths.no_such_read" in caplog.text


# --- the talk summary: the same rows either way --------------------------------------------


def _reads(tmp_path: Path, folder: str) -> list:
    out = []
    for i, value in enumerate(SUMMARIES):
        out.append(talk_prune.read_summary(_summary_db(tmp_path / folder / f"{i}.db", value)))
    out.append(talk_prune.read_summary(tmp_path / folder / "none.db"))
    return out


def test_the_talk_summary_reads_the_same_through_the_companion(tmp_path, request) -> None:
    request.getfixturevalue("companion_off")
    own = _reads(tmp_path, "own")
    assert own[5] == (0.0, "Alli asked about ferries.") and own[6] == (1793800000.5, "Another talk's.")
    request.getfixturevalue("companion_on")
    assert companion.function("talk_summary", "read") is not None
    assert _reads(tmp_path, "theirs") == own


def test_a_path_the_old_uri_misread_is_read_through_the_companion(tmp_path, companion_on) -> None:
    """The app's own read put the path in a URI unescaped: a '#' in it cut the path short."""
    value = json.dumps({"up_to": 5.0, "text": "Kept."})
    p = _summary_db(tmp_path / "a#b" / "companion.db", value)
    assert talk_prune.read_summary(p) == (5.0, "Kept.")


def test_the_companion_db_is_its_own_when_no_path_is_given(tmp_path, monkeypatch, companion_on) -> None:
    p = _summary_db(tmp_path / "c.db", json.dumps({"up_to": 7.0, "text": "Named."}))
    monkeypatch.setenv("REACHY_COMPANION_DB", str(p))
    assert talk_prune.read_summary() == (7.0, "Named.")


# --- the wire log ---------------------------------------------------------------------------


def test_the_deployed_commit_is_the_companions(companion_on) -> None:
    assert wire_log._companion_commit() == companion.function("data_paths", "deployed_commit")()


@pytest.mark.parametrize("on", [False, True])
def test_a_new_wire_log_is_incrementally_vacuumed(tmp_path, request, on) -> None:
    request.getfixturevalue("companion_on" if on else "companion_off")
    log = wire_log.WireLog(tmp_path / "w" / "wire.db", retention=90)
    log.open_session()
    log.flush()
    db = sqlite3.connect(tmp_path / "w" / "wire.db")
    assert db.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # INCREMENTAL
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_the_wire_log_path_is_the_companions(tmp_path, monkeypatch, companion_on) -> None:
    monkeypatch.setattr(wire_log, "_LOG", None)
    monkeypatch.setenv("WIRE_HOME", str(tmp_path))
    monkeypatch.setenv(wire_log.ENV, "$WIRE_HOME/wire.db")  # expanded, as data_paths does
    log = wire_log.from_env()
    assert log is not None and log.path == tmp_path / "wire.db"
    assert wire_log.from_env() is log  # one log per path
    monkeypatch.setenv(wire_log.ENV, "off")
    assert wire_log.from_env() is None


@pytest.mark.parametrize("raw", ["x", "0", "4000"])
def test_a_retention_the_supervisor_refuses_keeps_the_log_off(tmp_path, monkeypatch, companion_on, raw, caplog) -> None:
    monkeypatch.setattr(wire_log, "_LOG", None)
    monkeypatch.setenv(wire_log.ENV, str(tmp_path / "wire.db"))
    monkeypatch.setenv(wire_log.RETENTION_ENV, raw)
    with caplog.at_level(logging.ERROR, logger=wire_log.__name__):
        assert wire_log.from_env() is None
    assert "wire log: off" in caplog.text and not (tmp_path / "wire.db").exists()
    monkeypatch.setenv(wire_log.RETENTION_ENV, "30")
    assert wire_log.retention_days() == 30


def test_without_the_companion_a_bad_retention_is_still_ninety(monkeypatch, companion_off) -> None:
    monkeypatch.setenv(wire_log.RETENTION_ENV, "x")
    assert wire_log.retention_days() == 90


# --- the camera: one brightness -------------------------------------------------------------


def test_the_frame_is_measured_in_the_companions_luma(companion_on) -> None:
    blue = np.zeros((4, 4, 3), np.uint8)
    blue[..., 0] = 255  # BGR
    assert camera_mod._brightness(blue) == pytest.approx(18.4, abs=0.05)


def test_without_the_companion_the_frame_is_the_plain_mean(companion_off) -> None:
    blue = np.zeros((4, 4, 3), np.uint8)
    blue[..., 0] = 255
    assert camera_mod._brightness(blue) == pytest.approx(85.0)


@pytest.mark.parametrize("on", [False, True])
def test_the_room_reads_the_same_light_log_either_way(tmp_path, monkeypatch, request, on) -> None:
    request.getfixturevalue("companion_on" if on else "companion_off")
    light = tmp_path / "light.csv"
    now = datetime.datetime.now().astimezone()
    old = (now - datetime.timedelta(minutes=5)).isoformat(timespec="seconds")
    light.write_text(f"ts,posture,quiet,people,luma_mean\n{old},awake,0,1,12.0\n"
                     f"{now.isoformat(timespec='seconds')},awake,0,1,61.5\n")
    monkeypatch.setenv("REACHY_COMPANION_LIGHT_LOG", str(light))
    assert camera_mod._room_luma() == 61.5
    light.write_text(f"ts,posture,quiet,people,luma_mean\n{old},awake,0,1,61.5\n")
    assert camera_mod._room_luma() is None


# --- shared constants and helpers (companion ticket 38) --------------------------------------


def test_a_constant_comes_from_the_companion_or_the_default(companion_on) -> None:
    assert companion.value("facing", "NO_SUCH_CONSTANT", 7) == 7
    assert companion.value("facing", "TURN_LEFT_DEFAULT_DEG", 7) == 106.0


def test_the_fallback_literals_equal_the_companions_constants(companion_on) -> None:
    """The fork keeps its own figure for each constant it reads; a pin that the two agree today."""
    from reachy_mini_conversation_app import moves, console

    if companion.function("idle", "BREATHING_SCALE_RANGE") is None:
        pytest.skip("the companion has no idle.BREATHING_SCALE_RANGE (older than ticket 38)")
    assert companion.value("facing", "TURN_LEFT_DEFAULT_DEG", None) == moves.TURN_LEFT_DEFAULT_DEG == 106.0
    assert companion.value("facing", "TURN_RIGHT_DEFAULT_DEG", None) == moves.TURN_RIGHT_DEFAULT_DEG == 115.0
    assert tuple(companion.value("facing", "TURN_RANGE_DEG", None)) == moves.TURN_RANGE_DEG == (10.0, 160.0)
    assert companion.value("facing", "TURN_SPEED", None) == moves.TURN_SPEED
    assert tuple(companion.value("idle", "BREATHING_SCALE_RANGE", None)) == console.BREATHING_SCALE_RANGE
    assert companion.value("idle", "BREATHING_PITCH_MAX", None) == console.BREATHING_PITCH_MAX
    from reachy_companion import mood_expression as me  # re-exports the names it had

    assert me.SCALE_RANGE == console.BREATHING_SCALE_RANGE and me.PITCH_MAX == console.BREATHING_PITCH_MAX


def test_the_chars_per_token_is_the_companions(companion_on) -> None:
    from reachy_mini_conversation_app import huggingface_realtime as hf

    cfg = {"instructions": "x" * 100, "tools": []}
    assert companion.value("context_size", "CHARS_PER_TOKEN", None) == 4
    assert hf.session_size(cfg).endswith("~25 tokens")  # (100 + len("[]")) // 4


def test_the_model_image_is_sized_as_the_companions_glance(companion_on) -> None:
    """The idle glance sends the camera's frame scaled to the app's own conversation frames."""
    from reachy_mini_conversation_app import model_image

    pytest.importorskip("gi")
    glance_size = companion.function("people", "glance_size")
    if glance_size is None:
        pytest.skip("the companion has no people.glance_size")
    long_side = model_image.MODEL_IMAGE.long_side
    assert long_side == companion.value("people", "GLANCE_LONG_SIDE", None)
    for width, height in [(1920, 1080), (1080, 1920), (1056, 1056), (640, 480), (3, 1), (4000, 3001), (1057, 33)]:
        assert model_image._target_size(width, height, long_side) == glance_size(width, height)


def test_an_atomic_write_leaves_no_temp_file_and_falls_back_without_the_companion(tmp_path, request) -> None:
    from reachy_mini_conversation_app.utils import write_text_atomic

    for on in (True, False):
        request.getfixturevalue("companion_on" if on else "companion_off")
        target = tmp_path / ("on" if on else "off") / "settings.json"
        target.parent.mkdir()
        write_text_atomic(target, "oneé\n")
        write_text_atomic(target, "twoé\n")
        assert target.read_text(encoding="utf-8") == "twoé\n"
        assert [p.name for p in target.parent.iterdir()] == ["settings.json"]
