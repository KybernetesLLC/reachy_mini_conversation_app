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
import uuid
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
from external_content.rfid_manager.rfid_store import RFIDStore  # noqa: E402


class RFIDManagerUI:
    """RFID Manager — can be embedded as an accordion or served as a standalone page."""

    def __init__(
        self,
        data_dir: Path | None = None,
        handler: Any = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._nfc = NfcDaemonClient()
        self._store = RFIDStore(data_dir=data_dir)
        self._handler = handler
        self._loop = loop

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _choices(self) -> list[str]:
        return [
            f"{code} — {personality}"
            for code, personality in self._store.all().items()
        ]

    def _code_from_choice(self, choice: str) -> str | None:
        if not choice:
            return None
        return choice.split(" — ")[0].strip()

    # ── Shared UI builder ─────────────────────────────────────────────────────

    def _build_rfid_ui(self) -> None:
        """Create all RFID components inside the current Blocks context."""
        # ── Connection status ─────────────────────────────────────────────────
        conn_status = gr.Markdown("● Vérifie la connexion au daemon…")

        # ── Tag list + editor ─────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=200):
                gr.Markdown("**Tags enregistrés**")
                with gr.Row():
                    new_btn = gr.Button("+ Nouveau", size="sm")
                    delete_btn = gr.Button("✕ Supprimer", size="sm")
                tag_dd = gr.Dropdown(
                    label="Sélectionner un tag",
                    choices=self._choices(),
                    value=None,
                    allow_custom_value=False,
                )

            with gr.Column(scale=2, min_width=320):
                gr.Markdown("**Éditeur**")
                code_id_md = gr.Markdown("**Code ID :** —")
                personality_tb = gr.Textbox(
                    label="Personnalité",
                    placeholder="Nom du profil (ex: user_personalities/my_bot)",
                )
                with gr.Row():
                    save_btn = gr.Button("Sauvegarder", variant="primary")
                    write_btn = gr.Button(
                        "→ Écrire sur tag",
                        interactive=True,
                        variant="secondary",
                    )

        # ── Status bar ────────────────────────────────────────────────────────
        gr.Markdown("---")
        last_tag_md = gr.Markdown("**Tag détecté :** —")
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
            port = status.get("port") or "?"
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
                return f"● **Port ouvert** ({port}) — CLRC663 non détecté"
            return f"● **Connecté** ({port}) — lecteur NFC prêt"

        def _on_tag_select(choice: str) -> tuple[str | None, str, str]:
            code = self._code_from_choice(choice)
            if not code:
                return None, "**Code ID :** —", ""
            personality = self._store.get(code) or ""
            return (
                code,
                f"**Code ID :** `{code}`",
                personality,
            )

        def _new_tag() -> tuple[dict[str, Any], str, str, str]:
            new_code = uuid.uuid4().hex[:8].upper()
            self._store.save(new_code, "")
            choices = self._choices()
            new_choice = next((c for c in choices if c.startswith(new_code)), None)
            return (
                gr.update(choices=choices, value=new_choice),
                new_code,
                f"**Code ID :** `{new_code}`",
                "",
            )

        def _delete_tag(code: str | None) -> tuple[dict[str, Any], None, str, str, str]:
            if not code:
                return gr.update(), None, "**Code ID :** —", "", "Aucun tag sélectionné."
            self._store.delete(code)
            return (
                gr.update(choices=self._choices(), value=None),
                None,
                "**Code ID :** —",
                "",
                "Tag supprimé.",
            )

        def _save_tag(code: str | None, personality: str) -> tuple[dict[str, Any], str]:
            if not code:
                return gr.update(), "Aucun tag sélectionné."
            self._store.save(code, personality.strip())
            choices = self._choices()
            matching = next((c for c in choices if c.startswith(code)), None)
            return gr.update(choices=choices, value=matching), "✓ Sauvegardé."

        def _write_tag(code: str | None) -> str:
            if not code:
                return "Aucun tag sélectionné."
            return self._nfc.write_tag(code)

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
                    op_text = "✓ Tag écrit avec succès"
                else:
                    code = result_msg.split(":", 1)[-1] if ":" in result_msg else result_msg
                    op_text = f"⚠ Échec écriture : {describe_write_error(code)}"

            # Tag transitions
            if prev is not None:
                if not tag.present and prev.present:
                    last_tag_text = "**Tag détecté :** —"
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
                        last_tag_text = "**Tag détecté :** (vide) — tag non initialisé"
                    elif tag.content:
                        personality = self._store.get(tag.content)
                        if personality:
                            last_tag_text = f"**Tag détecté :** `{tag.content}` — **{personality}**"
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
                            last_tag_text = f"**Tag détecté :** `{tag.content}` — code inconnu dans la base"

            return (
                gr.update(value=conn_text),
                gr.update(value=last_tag_text) if last_tag_text is not None else gr.update(),
                gr.update(value=op_text) if op_text is not None else gr.update(),
            )

        # ── Event wiring ──────────────────────────────────────────────────────

        tag_dd.change(
            fn=_on_tag_select,
            inputs=[tag_dd],
            outputs=[code_state, code_id_md, personality_tb],
        )

        new_btn.click(
            fn=_new_tag,
            outputs=[tag_dd, code_state, code_id_md, personality_tb],
        )

        delete_btn.click(
            fn=_delete_tag,
            inputs=[code_state],
            outputs=[tag_dd, code_state, code_id_md, personality_tb, op_status_md],
        )

        save_btn.click(
            fn=_save_tag,
            inputs=[code_state, personality_tb],
            outputs=[tag_dd, op_status_md],
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
