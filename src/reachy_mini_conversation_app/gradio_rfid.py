"""Gradio RFID Manager UI.

Provides two integration modes:

1. Standalone page (always available):
   ``rfid_ui.create_rfid_blocks()`` returns a ``gr.Blocks`` that can be
   mounted at any path (e.g. ``/rfid``) via ``gr.mount_gradio_app``.

2. Inline accordion (when the main dashboard is in Gradio mode):
   ``rfid_ui.add_to_dashboard(stream_manager)`` injects an accordion into
   an existing ``gr.Blocks``.

Both modes use the Reachy Mini daemon's HTTP API (via NfcDaemonClient) rather
than owning the serial link directly. The daemon handles hot-plug and
reconnection automatically.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from reachy_mini_conversation_app.nfc_daemon_client import (  # noqa: E402
    NfcDaemonClient,
    describe_write_error,
)
from reachy_mini_conversation_app.headless_personality import (  # noqa: E402
    list_personalities,
)
from reachy_mini_conversation_app.personality_tag import (  # noqa: E402
    from_tag_token,
    to_tag_token,
)


class RFIDManagerUI:
    """RFID Manager — can be embedded as an accordion or served as a standalone page."""

    def __init__(
        self,
        data_dir: Path | None = None,
        handler: Any = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._nfc = NfcDaemonClient()
        self._handler = handler
        self._loop = loop

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _choices(self) -> list[str]:
        """Personalities that can be written to an accessory, in selection form."""
        return list_personalities()

    # ── Shared UI builder ─────────────────────────────────────────────────────

    def _build_rfid_ui(self) -> None:
        """Create all RFID components inside the current Blocks context."""
        # ── Connection status ─────────────────────────────────────────────────
        conn_status = gr.Markdown("● Vérifie la connexion au daemon…")

        # ── Write a personality onto a tag ────────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("**Écrire une personnalité sur un accessoire**")
                personality_dd = gr.Dropdown(
                    label="Personnalité",
                    choices=self._choices(),
                    value=None,
                    allow_custom_value=False,
                )
                # Shown before writing: the tag carries this text verbatim, so
                # there is no reason to keep it hidden from whoever writes it.
                token_md = gr.Markdown("**Écrit sur l'accessoire :** —")
                write_btn = gr.Button("→ Écrire sur l'accessoire", variant="primary")

        # ── Status bar ────────────────────────────────────────────────────────
        gr.Markdown("---")
        last_tag_md = gr.Markdown("**Accessoire :** aucun accessoire détecté (personnalité par défaut)")
        op_status_md = gr.Markdown("")

        # ── Hidden state + timer ──────────────────────────────────────────────
        code_state: gr.State = gr.State(value=None)
        timer = gr.Timer(value=1.0)

        # ── Previous tag state for change detection ───────────────────────────
        _prev: list[Any] = [None]

        # ── Callbacks ────────────────────────────────────────────────────────

        def _refresh_status() -> str:
            status = self._nfc.get_status()
            connected = status.get("connected", False)
            chip = status.get("chip_detected", False)
            # Checked before the link: without the driver the daemon disables
            # the reader outright, and the board being plugged in changes
            # nothing — reporting "non connecté" there would send you hunting
            # for a cable instead of an install.
            if not status.get("driver_available", False):
                return (
                    "● **Driver NFC absent sur le robot** — "
                    "installer le paquet `winnie_nfc` puis relancer le daemon"
                )
            if not connected:
                return "● **Daemon NFC : non connecté** (carte débranchée ou daemon non démarré)"
            if not chip:
                return "● **Port ouvert** — CLRC663 non détecté"
            return "● **Connecté** — lecteur NFC prêt"

        def _on_personality_select(personality: str | None) -> tuple[str | None, str]:
            token = to_tag_token(personality) if personality else None
            if token is None:
                return personality, "**Écrit sur l'accessoire :** —"
            return personality, f"**Écrit sur l'accessoire :** `{token}`"

        def _write_tag(personality: str | None) -> str:
            if not personality:
                return "Aucune personnalité sélectionnée."
            token = to_tag_token(personality)
            if token is None:
                return "Cette personnalité ne peut pas être écrite sur un accessoire."
            return self._nfc.write_tag(token)

        def _poll() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            status = self._nfc.get_status()
            connected = status.get("connected", False)
            if not connected:
                _prev[0] = None
                return (
                    gr.update(value=_refresh_status()),
                    gr.update(),
                    gr.update(),
                )

            tag = self._nfc.get_tag()
            prev = _prev[0]
            _prev[0] = tag

            conn_text = _refresh_status()
            last_tag_text: str | None = None
            op_text: str | None = None

            # Write results
            for _success, result_msg in self._nfc.drain_write_results():
                if _success:
                    op_text = "✓ Accessoire écrit avec succès"
                else:
                    code = result_msg.split(":", 1)[-1] if ":" in result_msg else result_msg
                    op_text = f"⚠ Échec écriture : {describe_write_error(code)}"

            # Tag transitions
            if prev is not None:
                if not tag.present and prev.present:
                    last_tag_text = (
                        "**Accessoire :** aucun accessoire détecté "
                        "(personnalité par défaut)"
                    )
                    if self._handler is not None and self._loop is not None:
                        try:
                            fut = asyncio.run_coroutine_threadsafe(
                                self._handler.apply_personality(None), self._loop
                            )
                            fut.result(timeout=10)
                            op_text = "Retour à la personnalité par défaut"
                        except Exception as exc:
                            logger.warning("RFID default revert failed: %s", exc)
                elif tag.present and not prev.present:
                    if tag.blank:
                        last_tag_text = "**Accessoire :** nouvel accessoire vierge"
                    elif tag.content:
                        personality = from_tag_token(tag.content)
                        if personality and personality not in list_personalities():
                            # The tag names a personality this robot does not
                            # have — an old opaque code, or a profile deleted
                            # since. Say which, rather than staying silent.
                            personality = None
                        if personality:
                            last_tag_text = f"**Accessoire :** **{personality}**"
                            if self._handler is not None and self._loop is not None:
                                try:
                                    profile = None if personality == "(built-in default)" else personality
                                    fut = asyncio.run_coroutine_threadsafe(
                                        self._handler.apply_personality(profile), self._loop
                                    )
                                    fut.result(timeout=10)
                                    op_text = f"Personnalité appliquée : **{personality}**"
                                except Exception as exc:
                                    op_text = f"Erreur changement personnalité : {exc}"
                        else:
                            last_tag_text = (
                                f"**Accessoire :** inconnu (`{tag.content}`)"
                            )

            return (
                gr.update(value=conn_text),
                gr.update(value=last_tag_text) if last_tag_text is not None else gr.update(),
                gr.update(value=op_text) if op_text is not None else gr.update(),
            )

        # ── Event wiring ──────────────────────────────────────────────────────

        personality_dd.change(
            fn=_on_personality_select,
            inputs=[personality_dd],
            outputs=[code_state, token_md],
        )

        write_btn.click(
            fn=_write_tag,
            inputs=[code_state],
            outputs=[op_status_md],
        )

        timer.tick(
            fn=_poll,
            outputs=[conn_status, last_tag_md, op_status_md],
        )

    # ── Public integration methods ────────────────────────────────────────────

    def create_rfid_blocks(self) -> gr.Blocks:
        """Return a standalone ``gr.Blocks`` with the full RFID Manager UI."""
        with gr.Blocks(title="RFID Manager") as demo:
            gr.Markdown("## RFID Manager")
            self._build_rfid_ui()
        return demo

    def add_to_dashboard(self, blocks: gr.Blocks) -> None:
        """Inject the RFID Manager as an accordion into an existing ``gr.Blocks``."""
        with blocks:
            with gr.Accordion("RFID Manager", open=True):
                self._build_rfid_ui()
