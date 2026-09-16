"""What the reader reports for the UI: no reader is not the same as no accessory."""

import pytest

from reachy_mini_conversation_app import rfid_routes
from reachy_mini_conversation_app.personality_tag import to_tag_token
from reachy_mini_conversation_app.nfc_daemon_client import NfcTagSnapshot, NfcDaemonClient


@pytest.fixture
def controller(monkeypatch):
    """Build a controller whose move datasets and daemon calls never leave the machine."""
    monkeypatch.setattr(rfid_routes, "_load_move_dataset", lambda repo_id: None)
    return rfid_routes.RfidController(
        get_handler=lambda: pytest.fail("the handler must not be needed to report state"),
        get_loop=lambda: None,
        robot=None,
    )


def fake_reader(monkeypatch, *, connected, tag=None):
    """Answer the daemon's status and tag endpoints without a reader."""
    monkeypatch.setattr(
        NfcDaemonClient,
        "get_status",
        lambda self: {
            "connected": connected,
            "chip_detected": connected,
            "driver_available": True,
            "port": "/dev/nfc",
        },
    )
    absent = NfcTagSnapshot(present=False, uid=None, content=None, blank=False)
    monkeypatch.setattr(NfcDaemonClient, "get_tag", lambda self: tag or absent)


def test_an_unplugged_reader_is_unavailable_not_empty(controller, monkeypatch):
    """The regression: an unplugged module read as "None", as if only a tag were missing."""
    fake_reader(monkeypatch, connected=False)
    assert controller.snapshot()["accessory"]["state"] == "unavailable"
    assert controller.poll_once()["accessory"]["state"] == "unavailable"


def test_a_connected_reader_with_nothing_on_it_is_empty(controller, monkeypatch):
    """A reader that answers but holds no tag is the "None" the badge shows."""
    fake_reader(monkeypatch, connected=True)
    assert controller.snapshot()["accessory"]["state"] == "none"


@pytest.mark.parametrize(
    "tag, state",
    [
        (NfcTagSnapshot(present=True, uid="04", content=None, blank=True), "blank"),
        (NfcTagSnapshot(present=True, uid="04", content="written-elsewhere", blank=False), "unknown"),
        (NfcTagSnapshot(present=True, uid="04", content=to_tag_token("default"), blank=False), "known"),
    ],
)
def test_a_tag_on_a_connected_reader_is_described_by_what_it_carries(controller, monkeypatch, tag, state):
    """Each tag state stays distinct from both "none" and "unavailable"."""
    fake_reader(monkeypatch, connected=True, tag=tag)
    assert controller.snapshot()["accessory"]["state"] == state
