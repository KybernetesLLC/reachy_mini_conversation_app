"""What the reader reports for the UI: no reader is not the same as no accessory."""

import webbrowser

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


def fake_reader(monkeypatch, *, connected, tag=None, error=None):
    """Answer the daemon's status and tag endpoints without a reader."""
    monkeypatch.setattr(
        NfcDaemonClient,
        "get_status",
        lambda self: {
            "connected": connected,
            "chip_detected": connected,
            "driver_available": True,
            "port": "/dev/nfc",
            "error": error,
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


def test_the_daemon_reason_for_a_down_link_reaches_the_panel(controller, monkeypatch):
    """ "Not connected" alone leaves a user guessing; the daemon says which it is."""
    fake_reader(monkeypatch, connected=False, error="no NFC reader board found")
    assert controller.connection_status()["error"] == "no NFC reader board found"
    assert controller.last_status()["error"] == "no NFC reader board found"


@pytest.fixture
def add_on_store(controller):
    """Register the rfid.* methods and return the one that opens the add-on store."""
    methods = {}

    class FakeRpc:
        def register(self, name, handler):
            methods[name] = handler

    rfid_routes.register_rfid_methods(FakeRpc(), controller)
    return methods["rfid.open_add_on_store"]


@pytest.mark.asyncio
async def test_the_store_opens_on_the_machine_running_the_app(monkeypatch, add_on_store):
    """The panel cannot raise a browser window from its webview, so the app does it."""
    asked = []
    monkeypatch.setattr(webbrowser, "open", lambda url: bool(asked.append(url)) or True)
    result = await add_on_store({})
    assert asked == [rfid_routes.ADD_ON_STORE_URL]
    assert result == {"opened": True, "url": rfid_routes.ADD_ON_STORE_URL}


@pytest.mark.asyncio
async def test_a_host_with_no_browser_still_hands_back_the_address(monkeypatch, add_on_store):
    """A robot with no screen of its own: the panel shows the URL instead."""
    monkeypatch.setattr(webbrowser, "open", lambda url: False)
    assert await add_on_store({}) == {"opened": False, "url": rfid_routes.ADD_ON_STORE_URL}
