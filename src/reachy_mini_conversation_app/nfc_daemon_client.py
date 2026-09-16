"""HTTP client for the Reachy Mini daemon's NFC reader API.

The daemon owns the serial link to the NFC reader board — a CLRC663 driven
straight from the host, with no microcontroller in the chain — and exposes it
at ``/api/nfc``. This module wraps those endpoints so the conversation app
never touches the serial port directly.

write_tag() is non-blocking: it starts a background thread and enqueues the
``(success, raw_msg)`` result. Call drain_write_results() (e.g. in a poll
loop) to consume those results.

Failures come back as stable codes rather than sentences, so callers can
branch on them; :func:`describe_write_error` turns one into a line for a user.
"""

from __future__ import annotations
import queue
import logging
import threading
from typing import Any, Optional
from dataclasses import dataclass

import requests


logger = logging.getLogger(__name__)

# The daemon refuses anything longer, and the tag's own declared capacity
# applies on top (496 bytes of NDEF on an NTAG215, 144 on an NTAG213). Checked
# here so an oversized code fails as TOO_LONG rather than as a 422 from the
# daemon's request validator.
MAX_WRITE_CHARS = 860

# Stable error codes returned by the daemon's write and erase endpoints.
WRITE_ERROR_MESSAGES = {
    "NO_TAG": "No tag on the reader",
    "TOO_LONG": "Code too long for this tag's capacity",
    "LOCKED": "Tag is locked and can no longer be written",
    "UNKNOWN_TAG": "Unrecognised tag model",
    "WRITE_REFUSED": "Tag refused the write",
    "WRITE_ERROR": "Write failed partway through",
    "COLLISION": "Several tags on the reader — present only one",
    "NOT_CONNECTED": "NFC reader not connected",
    "LINK_LOST": "Lost the link to the NFC reader",
    "TIMEOUT": "Tag was not presented in time",
    "DRIVER_MISSING": "NFC driver not installed on the robot",
}


def describe_write_error(code: str) -> str:
    """Turn a write/erase error code into a line for a user.

    Unknown codes are returned as-is: a code we have never seen is more useful
    on screen than a generic "unknown error" that hides it.
    """
    return WRITE_ERROR_MESSAGES.get(code, code or "Unknown error")


@dataclass
class NfcTagSnapshot:
    """Point-in-time state of the NFC reader."""

    present: bool
    uid: Optional[str]
    content: Optional[str]  # text content; None if blank or no tag
    blank: bool  # tag present but no content written
    writable: bool = False  # identified, formatted and not locked
    model: Optional[str] = None  # "NTAG215", ...
    capacity: Optional[int] = None  # bytes of NDEF the tag declares


class NfcDaemonClient:
    """HTTP client wrapping the daemon's /api/nfc routes.

    Usage::

        client = NfcDaemonClient()          # default http://localhost:8000
        tag = client.get_tag()
        if tag.present and not tag.blank:
            print("Code:", tag.content)
        client.write_tag("HELLO")           # non-blocking; result via drain_write_results()
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0) -> None:
        """Build a client for the daemon's NFC endpoints."""
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._write_queue: queue.SimpleQueue[tuple[bool, str]] = queue.SimpleQueue()
        # The reader is polled a few times a second for as long as the app
        # runs; without a session each poll opens and closes a TCP connection
        # to the daemon. The session is used by the two read endpoints only —
        # writes and erases run on their own threads, and requests.Session is
        # not documented as thread-safe.
        self._session = requests.Session()

    # -- Tag state ----------------------------------------------------------------

    def get_tag(self) -> NfcTagSnapshot:
        """Return the current tag state (never raises; returns absent on error)."""
        try:
            r = self._session.get(f"{self.base}/api/nfc/tag", timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
            return NfcTagSnapshot(
                present=bool(d.get("present")),
                uid=d.get("uid"),
                content=d.get("content") or None,
                blank=bool(d.get("blank")),
                writable=bool(d.get("writable")),
                model=d.get("model"),
                capacity=d.get("capacity"),
            )
        except Exception as exc:
            logger.debug("NFC get_tag error: %s", exc)
            return NfcTagSnapshot(present=False, uid=None, content=None, blank=False)

    # -- Reader status ------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return the daemon NFC reader status (never raises; returns disconnected on error).

        Mirrors the daemon's ``NfcStatus``: ``connected``, ``chip_detected``,
        ``driver_available``, ``port``, ``chip_version``, ``error``,
        ``last_seen_at``.
        """
        try:
            r = self._session.get(f"{self.base}/api/nfc/status", timeout=self.timeout)
            r.raise_for_status()
            status: dict[str, Any] = r.json()
            return status
        except Exception as exc:
            logger.debug("NFC get_status error: %s", exc)
            return {
                "connected": False,
                "chip_detected": False,
                "driver_available": False,
                "port": None,
                "chip_version": None,
                "error": str(exc),
            }

    def is_connected(self) -> bool:
        """Return True if the daemon reports the NFC reader as connected."""
        return bool(self.get_status().get("connected"))

    def driver_available(self) -> bool:
        """Whether the robot has the NFC driver package installed.

        Distinct from ``is_connected``: without the driver the daemon disables
        the reader entirely, and the board being plugged in changes nothing.
        """
        return bool(self.get_status().get("driver_available"))

    # -- Write --------------------------------------------------------------------

    def _post_write(self, text: str, timeout: float) -> tuple[bool, str]:
        """POST one write and normalise the outcome to (success, code)."""
        if not text:
            return False, "EMPTY_CODE"
        if len(text) > MAX_WRITE_CHARS:
            return False, "TOO_LONG"
        try:
            r = requests.post(
                f"{self.base}/api/nfc/write",
                json={"text": text},
                timeout=timeout,
            )
            if r.status_code == 503:
                # The reader itself is unavailable: disabled or link down.
                detail = r.json().get("detail", "unavailable")
                return False, "NOT_CONNECTED" if "connect" in detail.lower() else detail
            if r.status_code == 422:
                # The daemon's own validator rejected the payload.
                return False, "TOO_LONG"
            r.raise_for_status()
            d = r.json()
            if d.get("success"):
                return True, "WRITE_OK"
            return False, d.get("error") or "WRITE_ERROR"
        except Exception as exc:
            logger.warning("NFC write error: %s", exc)
            return False, "LINK_LOST"

    def write_tag(self, code: str) -> str:
        """Start an async write of ``code`` onto the next presented tag.

        Returns a human-readable status string immediately. The write result
        (``WRITE_OK`` or ``WRITE_FAIL:<CODE>``) is enqueued and available via
        :meth:`drain_write_results`, where ``<CODE>`` is one of the stable
        codes in :data:`WRITE_ERROR_MESSAGES`.
        """
        text = code

        def _worker() -> None:
            success, result = self._post_write(text, max(self.timeout, 12.0))
            self._write_queue.put((True, "WRITE_OK") if success else (False, f"WRITE_FAIL:{result}"))

        threading.Thread(target=_worker, daemon=True).start()
        return f"Bring a tag close to write '{text}'…"

    def write_tag_sync(self, code: str, timeout: float = 12.0) -> tuple[bool, str]:
        """Write ``code`` synchronously (tag must already be on the reader).

        Returns ``(True, "WRITE_OK")`` on success, or ``(False, code)`` where
        ``code`` is a stable error code — pass it to :func:`describe_write_error`
        to show it to a user.
        """
        return self._post_write(code, timeout)

    # -- Erase --------------------------------------------------------------------

    def erase_tag_sync(self, full: bool = False, timeout: float = 30.0) -> tuple[bool, str]:
        """Make the tag on the reader blank again.

        ``full`` also zeroes the whole user memory, which is what it takes to
        remove a payload that is not NDEF. It writes one page at a time, so it
        takes a few seconds — hence the wider default timeout.
        """
        try:
            r = requests.post(
                f"{self.base}/api/nfc/erase",
                json={"full": full},
                timeout=timeout,
            )
            if r.status_code == 503:
                detail = r.json().get("detail", "unavailable")
                return False, "NOT_CONNECTED" if "connect" in detail.lower() else detail
            r.raise_for_status()
            d = r.json()
            if d.get("success"):
                return True, "ERASE_OK"
            return False, d.get("error") or "WRITE_ERROR"
        except Exception as exc:
            logger.warning("NFC erase error: %s", exc)
            return False, "LINK_LOST"

    def close(self) -> None:
        """Close the pooled connection used by the read endpoints."""
        self._session.close()

    def drain_write_results(self) -> list[tuple[bool, str]]:
        """Drain and return all pending write results (non-blocking)."""
        results: list[tuple[bool, str]] = []
        try:
            while True:
                results.append(self._write_queue.get_nowait())
        except queue.Empty:
            pass
        return results
