"""Bidirectional local audio stream with optional settings UI.

In headless mode, there is no Gradio UI. If the selected backend is missing
its required API key, we expose a minimal settings page via the Reachy Mini
Apps settings server so users can pick a backend and provide any missing
credentials.

The settings UI is served from this package's ``static/`` folder. It persists
the selected backend and any provided API keys into the app instance's ``.env``
file when available.
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Callable, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.config import (
    HF_BACKEND,
    GEMINI_BACKEND,
    LOCKED_PROFILE,
    OPENAI_BACKEND,
    HF_REALTIME_WS_URL_ENV,
    HF_LOCAL_CONNECTION_MODE,
    HF_DEPLOYED_CONNECTION_MODE,
    HF_REALTIME_CONNECTION_MODE_ENV,
    config,
    get_backend_choice,
    get_hf_session_url,
    get_hf_direct_ws_url,
    build_hf_direct_ws_url,
    has_hf_realtime_target,
    parse_hf_direct_target,
    get_model_name_for_backend,
    get_hf_connection_selection,
    get_default_voice_for_backend,
    refresh_runtime_config_from_env,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.startup_settings import read_startup_settings, write_startup_settings
from reachy_mini_conversation_app.audio.startup_config import apply_audio_startup_config
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore


logger = logging.getLogger(__name__)
HandlerFactory = Callable[[Optional[str]], ConversationHandler]

LEGACY_STARTUP_ENV_NAMES = (
    "REACHY_MINI_CUSTOM_PROFILE",
    "REACHY_MINI_VOICE_OVERRIDE",
)
BACKEND_RETRY_DELAY_SECONDS = 5.0


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: ConversationHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
        handler_factory: HandlerFactory | None = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the stream with a realtime handler and pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        - ``handler_factory``: builds a fresh handler for the currently selected backend.
        """
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._restart_requested = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        self._handler_factory = handler_factory
        self._voice_override = startup_voice
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._settings_initialized = False
        self._asyncio_loop = None
        self._active_backend_name = get_backend_choice()
        self._backend_connection_state = "not_started"
        self._backend_error: str | None = None
        self._backend_retry_delay = BACKEND_RETRY_DELAY_SECONDS
        self._install_handler(handler)

    def _install_handler(self, handler: ConversationHandler) -> None:
        """Set the active handler and wire LocalStream-owned helpers into it."""
        self.handler = handler
        self.handler._clear_queue = self.clear_audio_queue
        # Re-inject RFID deps whenever the handler is replaced
        if getattr(self, "_rfid_serial", None) is not None:
            self.handler.deps.rfid_serial = self._rfid_serial

    # ---- Settings UI ----
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []

    def _active_backend(self) -> str:
        """Return the backend family of the currently running handler."""
        return self._active_backend_name

    def _backend_connected(self) -> bool:
        """Return whether the active handler currently has a realtime connection."""
        try:
            handler_state = vars(self.handler)
        except TypeError:
            handler_state = {}
        return any(handler_state.get(attr) is not None for attr in ("connection", "session"))

    def _can_rebuild_handler(self) -> bool:
        """Return whether LocalStream can construct handlers for backend changes."""
        return self._handler_factory is not None

    def _build_handler_for_current_backend(self) -> ConversationHandler:
        """Create and install a fresh handler for the current runtime backend config."""
        if self._handler_factory is None:
            return self.handler
        handler = self._handler_factory(self._voice_override)
        self._install_handler(handler)
        self._active_backend_name = get_backend_choice()
        return handler

    async def _shutdown_active_handler(self) -> None:
        """Best-effort shutdown for the currently active handler."""
        try:
            await self.handler.shutdown()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("Active handler shutdown ignored during restart: %s", e)

    def _mark_restart_requested(self, reason: str) -> None:
        """Request a backend restart from a synchronous route handler."""
        logger.info("Backend restart requested: %s", reason)
        self._set_backend_connection_state("connecting")
        loop = self._asyncio_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.request_backend_restart(reason), loop)
            return
        self._restart_requested.set()

    async def request_backend_restart(self, reason: str) -> None:
        """Ask the startup loop to rebuild the backend and stop the current handler."""
        self._set_backend_connection_state("connecting")
        self._restart_requested.set()
        await self._shutdown_active_handler()

    async def _sleep_or_restart_requested(self, delay: float) -> None:
        """Sleep for a retry interval, waking early if a restart is requested."""
        if self._restart_requested.is_set():
            return
        try:
            await asyncio.wait_for(self._restart_requested.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    @staticmethod
    def _format_backend_error(error: BaseException | str) -> str:
        """Return a compact user-facing backend error string."""
        if isinstance(error, str):
            return error
        message = str(error).strip()
        if message:
            return f"{type(error).__name__}: {message}"
        return type(error).__name__

    def _set_backend_connection_state(self, state: str, error: BaseException | str | None = None) -> None:
        """Update backend connection status exposed through the settings UI."""
        self._backend_connection_state = state
        if error is not None:
            self._backend_error = self._format_backend_error(error)
        elif state != "disconnected":
            self._backend_error = None

    def _backend_connection_status(self) -> dict[str, object]:
        """Return the backend connection state exposed in /status."""
        connected = self._backend_connected()
        state = "connected" if connected else self._backend_connection_state
        return {
            "backend_connected": connected,
            "backend_connection_state": state,
            "backend_error": None if connected else self._backend_error,
        }

    @staticmethod
    def _has_key(value: Optional[str]) -> bool:
        """Return whether a runtime credential value is present."""
        return bool(value and str(value).strip())

    def _has_required_key(self, backend: str) -> bool:
        """Return whether the requested backend has its required credential."""
        if backend == GEMINI_BACKEND:
            return self._has_key(config.GEMINI_API_KEY)
        if backend == HF_BACKEND:
            return has_hf_realtime_target()
        return self._has_key(config.OPENAI_API_KEY)

    @staticmethod
    def _requirement_name(backend: str) -> str:
        """Return the env var users need for a backend, if any."""
        if backend == GEMINI_BACKEND:
            return "GEMINI_API_KEY"
        if backend == HF_BACKEND:
            return HF_REALTIME_WS_URL_ENV
        return "OPENAI_API_KEY"

    def _persist_env_value(self, env_name: str, value: str) -> None:
        """Persist a non-empty environment value in memory and in the instance `.env`."""
        self._persist_env_values({env_name: value})

    def _persist_env_values(self, updates: dict[str, str]) -> None:
        """Persist non-empty environment values in memory and in the instance `.env`."""
        normalized_updates = {name: (value or "").strip() for name, value in updates.items()}
        normalized_updates = {name: value for name, value in normalized_updates.items() if value}
        if not normalized_updates:
            return

        for env_name, value in normalized_updates.items():
            try:
                os.environ[env_name] = value
            except Exception:
                pass
        refresh_runtime_config_from_env()

        if not self._instance_path:
            return
        try:
            inst = Path(self._instance_path)
            env_path = inst / ".env"
            lines = self._read_env_lines(env_path)
            for env_name, value in normalized_updates.items():
                replaced = False
                for i, ln in enumerate(lines):
                    if ln.strip().startswith(f"{env_name}="):
                        lines[i] = f"{env_name}={value}"
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"{env_name}={value}")
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted %s to %s", ", ".join(sorted(normalized_updates)), env_path)

            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path))
            except Exception:
                pass
            refresh_runtime_config_from_env()
        except Exception as e:
            logger.warning("Failed to persist %s: %s", ", ".join(sorted(normalized_updates)), e)

    def _remove_persisted_env_values(self, env_names: tuple[str, ...]) -> None:
        """Remove keys from the instance `.env` without mutating the current runtime."""
        normalized_names = tuple(sorted({name.strip() for name in env_names if name and name.strip()}))
        if not normalized_names or not self._instance_path:
            return

        env_path = Path(self._instance_path) / ".env"
        if not env_path.exists():
            return

        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
            filtered_lines = [
                line
                for line in lines
                if not any(line.strip().startswith(f"{env_name}=") for env_name in normalized_names)
            ]
            if filtered_lines == lines:
                return

            final_text = "\n".join(filtered_lines)
            if final_text:
                final_text += "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Removed %s from %s", ", ".join(normalized_names), env_path)
        except Exception as e:
            logger.warning("Failed to remove %s: %s", ", ".join(normalized_names), e)

    def _persist_hf_direct_connection(self, host: str, port: int) -> None:
        """Persist a direct Hugging Face websocket target."""
        self._persist_env_values(
            {
                HF_REALTIME_CONNECTION_MODE_ENV: HF_LOCAL_CONNECTION_MODE,
                HF_REALTIME_WS_URL_ENV: build_hf_direct_ws_url(host, port),
            }
        )

    def _persist_hf_allocator_connection(self) -> None:
        """Persist the deployed Hugging Face allocator mode."""
        self._persist_env_value(HF_REALTIME_CONNECTION_MODE_ENV, HF_DEPLOYED_CONNECTION_MODE)
        self._remove_persisted_env_values(("HF_REALTIME_SESSION_URL",))

    def _persist_api_key(self, key: str) -> None:
        """Persist OPENAI_API_KEY to environment and instance `.env`."""
        self._persist_env_value("OPENAI_API_KEY", key)

    def _persist_gemini_api_key(self, key: str) -> None:
        """Persist GEMINI_API_KEY to environment and instance `.env`."""
        self._persist_env_value("GEMINI_API_KEY", key)

    def _persist_backend_choice(self, backend: str) -> None:
        """Persist the selected backend without clobbering explicit model overrides."""
        current_backend = get_backend_choice()
        current_model_name = (os.getenv("MODEL_NAME") or "").strip()
        updates = {"BACKEND_PROVIDER": backend}
        if backend == HF_BACKEND:
            self._persist_env_values(updates)
            try:
                os.environ.pop("MODEL_NAME", None)
            except Exception:
                pass
            self._remove_persisted_env_values(("MODEL_NAME",))
            refresh_runtime_config_from_env()
            return

        if current_model_name and current_model_name != get_model_name_for_backend(current_backend):
            updates["MODEL_NAME"] = current_model_name
        else:
            updates["MODEL_NAME"] = get_model_name_for_backend(backend)
        self._persist_env_values(updates)

    def _persist_personality(self, profile: Optional[str], voice_override: Optional[str] = None) -> None:
        """Persist startup profile and voice in instance-local UI settings."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        normalized_voice_override = (voice_override or "").strip() or None
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            write_startup_settings(
                self._instance_path,
                profile=selection,
                voice=normalized_voice_override,
            )
            self._remove_persisted_env_values(LEGACY_STARTUP_ENV_NAMES)
            logger.info("Persisted startup personality settings to %s", Path(self._instance_path))
        except Exception as e:
            logger.warning("Failed to persist startup personality settings: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read the saved startup personality from instance-local UI settings."""
        return read_startup_settings(self._instance_path).profile

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Apply a personality by updating config and restarting the active backend."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions

            previous_profile = getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
            set_custom_profile(profile)
            try:
                get_session_instructions()
                get_session_voice(default=get_default_voice_for_backend(get_backend_choice()))
            except BaseException:
                set_custom_profile(previous_profile)
                raise
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"
        except BaseException as e:
            logger.error("Failed to resolve personality content: %s", e)
            return f"Failed to apply personality: {e}"
        await self.request_backend_restart("personality_changed")
        return "Applied personality and restarting backend."

    async def get_available_voices(self) -> list[str]:
        """Return voices available for the currently selected backend."""
        return get_available_voices_for_backend(get_backend_choice())

    def get_current_voice(self) -> str:
        """Return the currently selected voice override or backend profile voice."""
        if self._voice_override:
            return self._voice_override
        try:
            from reachy_mini_conversation_app.prompts import get_session_voice

            return get_session_voice(default=get_default_voice_for_backend(get_backend_choice()))
        except Exception:
            return get_default_voice_for_backend(get_backend_choice())

    async def change_voice(self, voice: str) -> str:
        """Change the voice by rebuilding the active backend from LocalStream."""
        available_voices = get_available_voices_for_backend(get_backend_choice())
        default_voice = get_default_voice_for_backend(get_backend_choice())
        resolved_voice = voice if voice in available_voices else default_voice
        if resolved_voice != voice:
            logger.warning(
                "Ignoring unsupported voice %r for backend=%r; using %r",
                voice,
                get_backend_choice(),
                resolved_voice,
            )
        self._voice_override = resolved_voice
        await self.request_backend_restart("voice_changed")
        return f"Voice changed to {resolved_voice}."

    def _init_rfid_routes(self) -> None:
        """Add RFID personality-mapping endpoints to the settings app.

        Each RFID code is mapped to a personality name. When a tag is read,
        the corresponding personality is applied automatically.

        The serial link is owned by the Reachy Mini daemon; this app acts as
        an HTTP client of the daemon's /api/nfc endpoints.
        """
        import uuid as _uuid
        import sys as _sys
        import asyncio as _asyncio
        import threading as _threading

        _proj_root = Path(__file__).parent.parent.parent
        if str(_proj_root) not in _sys.path:
            _sys.path.insert(0, str(_proj_root))

        from reachy_mini_conversation_app.nfc_daemon_client import (
            NfcDaemonClient,
            describe_write_error,
        )
        from reachy_mini_conversation_app.personality_tag import (
            from_tag_token as _from_tag_token,
            to_tag_token as _to_tag_token,
        )

        _TRANSITION_MOVES = None
        _MOVES = None
        try:
            from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
            from reachy_mini.motion.recorded_move import RecordedMoves
            _TRANSITION_MOVES = RecordedMoves("cdeplanne/local-dataset")
            _MOVES = RecordedMoves("glannuzel/local-dataset")
        except Exception:
            pass

        _nfc_client = NfcDaemonClient()
        # Store as instance attrs so _install_handler() can re-inject on backend restart
        self._rfid_serial = _nfc_client
        _app = self._settings_app
        _get_handler = lambda: self.handler  # noqa: E731 — always returns the current handler
        _get_loop = lambda: self._asyncio_loop  # noqa: E731
        _current_rfid_personality: list[str | None] = [None]
        _blank_tag_active: list[bool] = [False]
        _blank_tag_cooldown_until: list[float] = [0.0]  # prevents re-injection after brief NO_TAG
        _delayed_switch_future: list = [None]
        _inject_move_future: list = [None]  # tracks _inject_after_move coroutine for cancellation
        _prev_tag: list = [None]  # last NfcTagSnapshot seen by /rfid/poll

        # Inject NFC deps into ToolDependencies so nfc_writer can use them
        self.handler.deps.rfid_serial = _nfc_client
        _apply_lock = _threading.Lock()

        _DEFAULT_PROFILE = "(built-in default)"

        class _WriteBody(BaseModel):
            code: str

        @_app.get("/rfid/connection")
        def _rfid_connection() -> JSONResponse:
            status = _nfc_client.get_status()
            return JSONResponse({
                "connected": status.get("connected", False),
                "port": status.get("port"),
                "chip_detected": status.get("chip_detected", False),
                "driver_available": status.get("driver_available", False),
                "chip_version": status.get("chip_version"),
            })

        @_app.post("/rfid/write")
        def _rfid_write(body: _WriteBody) -> JSONResponse:
            msg = _nfc_client.write_tag(body.code)
            return JSONResponse({"ok": True, "message": msg})

        class _ClearBody(BaseModel):
            full: bool = False

        @_app.post("/rfid/clear")
        def _rfid_clear(body: _ClearBody | None = None) -> JSONResponse:
            """Make the tag on the reader blank again.

            ``full`` also zeroes the whole user memory — the only way to remove
            a payload that is not NDEF. It writes one page at a time, so it
            takes a few seconds on an NTAG215.
            """
            full = bool(body.full) if body is not None else False
            success, result = _nfc_client.erase_tag_sync(full=full)
            if not success:
                logger.warning("[RFID] Erase failed: %s", result)
                return JSONResponse({
                    "ok": False,
                    "message": describe_write_error(result),
                    "code": result,
                })
            return JSONResponse({"ok": True, "message": "Tag erased"})

        class _LinkTagBody(BaseModel):
            personality: str

        @_app.post("/rfid/link_tag")
        def _rfid_link_tag(body: _LinkTagBody) -> JSONResponse:
            """Link the tag currently on the reader to the given personality.

            If the tag is blank, a fresh unique code is generated and written to it first.
            """
            personality = (body.personality or "").strip()
            if not personality:
                return JSONResponse({"ok": False, "error": "no_personality"}, status_code=400)
            tag = _nfc_client.get_tag()
            if not tag.present:
                return JSONResponse({"ok": False, "error": "no_tag"})
            code = _to_tag_token(personality)
            if code is None:
                return JSONResponse({"ok": False, "error": "not_writable"}, status_code=400)
            # A tag already carrying this exact personality needs no write: the
            # token is derived from the personality, so it cannot drift.
            if tag.content == code:
                logger.info("[RFID] Tag already carries %r", personality)
                return JSONResponse({"ok": True, "code": code, "personality": personality, "written": False})
            logger.info("[RFID] Writing %r to tag for personality %r", code, personality)
            success, detail = _nfc_client.write_tag_sync(code)
            if not success:
                logger.warning("[RFID] Write failed for %r: %s", code, detail)
                return JSONResponse({
                    "ok": False,
                    "error": "write_failed",
                    "detail": describe_write_error(detail),
                    "code": detail,
                })
            logger.info("[RFID] Tag written with %r", code)
            return JSONResponse({"ok": True, "code": code, "personality": personality, "written": True})

        @_app.get("/rfid/poll")
        def _rfid_poll() -> JSONResponse:
            status = _nfc_client.get_status()
            port = status.get("port")
            _driver_ok = status.get("driver_available", False)
            if not status.get("connected"):
                _prev_tag[0] = None
                return JSONResponse({"messages": [], "connected": False, "applied": None, "port": None,
                                     "driver_available": _driver_ok})

            tag = _nfc_client.get_tag()
            prev = _prev_tag[0]
            _prev_tag[0] = tag

            # Build synthetic message list from write results + tag state transitions.
            # Message format mirrors the firmware line protocol so the processing loop below
            # works unchanged: "NO_TAG", "READ:", "READ:<code>", "WRITE_OK", "WRITE_FAIL:…"
            msgs: list[str] = []

            for _success, _result_msg in _nfc_client.drain_write_results():
                msgs.append(_result_msg)

            if prev is not None:
                if not tag.present and prev.present:
                    msgs.append("NO_TAG")
                elif tag.present and not prev.present:
                    msgs.append("READ:" if tag.blank else f"READ:{tag.content or ''}")
                elif tag.present and prev.present:
                    if tag.blank and not prev.blank:
                        msgs.append("READ:")
                    elif not tag.blank and tag.content and tag.content != prev.content:
                        msgs.append(f"READ:{tag.content}")

            applied = None
            if not msgs:
                return JSONResponse({"messages": [], "connected": True, "applied": None, "port": port,
                                     "driver_available": _driver_ok})
            if not _apply_lock.acquire(blocking=False):
                return JSONResponse({"messages": msgs, "connected": True, "applied": None, "port": port,
                                     "driver_available": _driver_ok})
            try:
                handler = _get_handler()
                logger.info("[RFID] events: %r", msgs)

                for msg in msgs:
                    if msg.strip() == "NO_TAG":
                        was_blank = _blank_tag_active[0]
                        _blank_tag_active[0] = False
                        _blank_tag_cooldown_until[0] = time.monotonic() + 3.0
                        handler.deps.blank_tag_present = False
                        if _inject_move_future[0] is not None:
                            _inject_move_future[0].cancel()
                            _inject_move_future[0] = None
                        if _delayed_switch_future[0] is not None:
                            _delayed_switch_future[0].cancel()
                            _delayed_switch_future[0] = None
                        loop = _get_loop()
                        if loop is not None:
                            try:
                                _asyncio.run_coroutine_threadsafe(
                                    handler.abort_nfc_collection(), loop
                                ).result(timeout=5)
                            except Exception as _ae:
                                logger.warning("[RFID] >>> abort_nfc_collection FAILED: %s", _ae)
                                handler._nfc_transition = False
                                handler._nfc_speech_done_event.set()
                        else:
                            handler._nfc_transition = False
                            handler._nfc_speech_done_event.set()
                        if was_blank:
                            logger.info("[RFID] >>> blank tag removed (blank_tag_present cleared)")
                            handler.deps.pending_nfc_write = None
                        if _current_rfid_personality[0] is not None:
                            logger.info("[RFID] >>> NO_TAG received — reverting to default")
                            if _TRANSITION_MOVES is not None:
                                handler.deps.movement_manager.queue_move(
                                    EmotionQueueMove("switch-personnality-5", _TRANSITION_MOVES)
                                )
                            loop = _get_loop()
                            if loop is not None:
                                try:
                                    fut = _asyncio.run_coroutine_threadsafe(
                                        handler.apply_personality(None), loop
                                    )
                                    fut.result(timeout=10)
                                    _current_rfid_personality[0] = None
                                    applied = {"code": None, "personality": _DEFAULT_PROFILE}
                                    logger.info("[RFID] >>> default personality applied OK")
                                except Exception as exc:
                                    logger.warning("[RFID] >>> default revert FAILED: %s", exc)
                    elif msg.startswith("READ:"):
                        code = msg[5:].strip().rstrip("\x00").strip()
                        if not code:
                            handler.deps.blank_tag_present = True
                            pending = handler.deps.pending_nfc_write
                            if pending is not None:
                                _blank_tag_active[0] = True
                                handler.deps.pending_nfc_write = None
                                logger.info("[RFID] >>> blank tag with pending write — writing code %r", pending["code"])
                                handler.deps.recently_written_codes.add(pending["code"])
                                _nfc_client.write_tag(pending["code"])
                                loop = _get_loop()
                                if loop is not None:
                                    try:
                                        fut = _asyncio.run_coroutine_threadsafe(
                                            handler.inject_nfc_writing_started(pending["personality"]), loop
                                        )
                                        fut.result(timeout=10)
                                    except Exception as exc:
                                        logger.warning("[RFID] >>> inject_nfc_writing_started FAILED: %s", exc)
                            elif not _blank_tag_active[0] and time.monotonic() >= _blank_tag_cooldown_until[0]:
                                _blank_tag_active[0] = True
                                logger.info("[RFID] >>> blank tag detected — injecting event to LLM")
                                loop = _get_loop()
                                if loop is not None:
                                    try:
                                        fut = _asyncio.run_coroutine_threadsafe(
                                            handler.inject_blank_nfc_tag(), loop
                                        )
                                        fut.result(timeout=10)
                                    except Exception as exc:
                                        logger.warning("[RFID] >>> blank tag inject FAILED: %s", exc)
                        elif code:
                            handler.deps.blank_tag_present = False
                            personality_name = _from_tag_token(code)
                            if personality_name is not None and personality_name not in _list_personalities():
                                # A token for a personality this robot does not
                                # have: say so rather than silently ignoring a
                                # tag the user just presented.
                                logger.warning("[RFID] >>> unknown personality %r on tag", personality_name)
                                personality_name = None
                            if personality_name is not None:
                                if personality_name == _current_rfid_personality[0]:
                                    logger.debug("[RFID] >>> same personality %r, skipping", personality_name)
                                elif code in handler.deps.recently_written_codes:
                                    handler.deps.recently_written_codes.discard(code)
                                    _current_rfid_personality[0] = personality_name
                                    logger.info("[RFID] >>> newly written tag %r — delaying personality switch", code)
                                    loop = _get_loop()
                                    if loop is not None:
                                        if _delayed_switch_future[0] is not None:
                                            _delayed_switch_future[0].cancel()
                                        profile = None if personality_name == _DEFAULT_PROFILE else personality_name

                                        async def _delayed_switch(p=profile, pn=personality_name):
                                            try:
                                                try:
                                                    await _asyncio.wait_for(
                                                        handler._nfc_speech_done_event.wait(), timeout=12.0
                                                    )
                                                except _asyncio.TimeoutError:
                                                    logger.warning("[RFID] >>> NFC speech done event timed out, switching anyway")
                                                start = handler._nfc_speech_start_time
                                                samples = handler._nfc_speech_samples
                                                sr = handler.output_sample_rate
                                                if start is not None and samples > 0 and sr > 0:
                                                    speech_duration = samples / sr
                                                    expected_end = start + speech_duration + 0.8
                                                    remaining = expected_end - _asyncio.get_event_loop().time()
                                                    logger.info(
                                                        "[RFID] >>> welcome speech: %.2fs, waiting %.2fs more before switch",
                                                        speech_duration, max(0.0, remaining),
                                                    )
                                                    if remaining > 0:
                                                        await _asyncio.sleep(remaining)
                                                else:
                                                    logger.warning("[RFID] >>> no speech audio tracked, falling back to drain+sleep")
                                                    try:
                                                        async def _drain():
                                                            while not handler.output_queue.empty():
                                                                await _asyncio.sleep(0.05)
                                                        await _asyncio.wait_for(_drain(), timeout=10.0)
                                                    except _asyncio.TimeoutError:
                                                        logger.warning("[RFID] >>> audio queue drain timed out")
                                                    await _asyncio.sleep(1.0)
                                                if _TRANSITION_MOVES is not None:
                                                    handler.deps.movement_manager.queue_move(
                                                        EmotionQueueMove("switch-personnality-5", _TRANSITION_MOVES)
                                                    )
                                                await handler.apply_personality(p)
                                                logger.info("[RFID] >>> delayed personality switch to %r done", pn)
                                            except _asyncio.CancelledError:
                                                logger.info("[RFID] >>> delayed personality switch cancelled (tag removed)")
                                                _current_rfid_personality[0] = None
                                            except Exception as exc:
                                                logger.warning("[RFID] >>> delayed personality switch FAILED: %s", exc)
                                            finally:
                                                handler._nfc_transition = False
                                                handler._nfc_speech_done_event.set()
                                                _delayed_switch_future[0] = None

                                        _delayed_switch_future[0] = _asyncio.run_coroutine_threadsafe(_delayed_switch(), loop)
                                else:
                                    logger.info("[RFID] >>> applying personality %r for code %r", personality_name, code)
                                    if _TRANSITION_MOVES is not None:
                                        handler.deps.movement_manager.queue_move(
                                            EmotionQueueMove("switch-personnality-5", _TRANSITION_MOVES)
                                        )
                                    loop = _get_loop()
                                    if loop is not None:
                                        try:
                                            profile = None if personality_name == _DEFAULT_PROFILE else personality_name
                                            fut = _asyncio.run_coroutine_threadsafe(
                                                handler.apply_personality(profile), loop
                                            )
                                            fut.result(timeout=10)
                                            _current_rfid_personality[0] = personality_name
                                            applied = {"code": code, "personality": personality_name}
                                            logger.info("[RFID] >>> personality applied OK")
                                        except Exception as exc:
                                            logger.warning("[RFID] >>> apply FAILED: %s", exc)
                            else:
                                logger.info("[RFID] >>> code %r not in store, keeping current personality", code)
                    elif msg.startswith("WRITE_"):
                        logger.info("[RFID] >>> %s", msg)
                        loop = _get_loop()
                        if loop is not None:
                            success = msg.upper().startswith("WRITE_OK")
                            move_duration = 0.0
                            if success:
                                try:
                                    _asyncio.run_coroutine_threadsafe(
                                        handler.stop_current_speech(), loop
                                    ).result(timeout=6)
                                except Exception as _se:
                                    logger.warning("[RFID] >>> stop_current_speech failed: %s", _se)
                            if success and _MOVES is not None and handler.deps.movement_manager is not None:
                                try:
                                    _write_move = EmotionQueueMove("write-tag-6", _MOVES)
                                    move_duration = float(_write_move.duration)
                                    _SOUND_LEAD_S = 0.15
                                    _sound_path = getattr(
                                        getattr(_write_move, "emotion_move", None),
                                        "sound_path", None,
                                    )
                                    if _sound_path is not None:
                                        self._robot.media.play_sound(str(_sound_path))
                                        logger.info("[RFID] >>> write-tag-6 sound started (%.0fms lead): %s", _SOUND_LEAD_S * 1000, _sound_path)
                                        import time as _time; _time.sleep(_SOUND_LEAD_S)
                                    else:
                                        logger.warning("[RFID] >>> write-tag-6: no sound_path found on emotion_move")
                                    handler.deps.movement_manager.queue_move(_write_move)
                                    logger.info("[RFID] >>> write-tag-6 queued (%.2fs), speech delayed", move_duration)
                                except Exception as exc:
                                    logger.warning("[RFID] >>> write-tag-6 move failed: %s", exc)
                                    move_duration = 0.0
                            if move_duration > 0.0:
                                def _arm_gate():
                                    handler._nfc_speech_done_event.clear()
                                    handler._nfc_speech_start_time = None
                                    handler._nfc_speech_samples = 0
                                loop.call_soon_threadsafe(_arm_gate)

                                async def _inject_after_move(dur=move_duration, s=success, m=msg):
                                    try:
                                        await _asyncio.sleep(dur)
                                        await handler.inject_nfc_write_result(s, m)
                                    finally:
                                        _inject_move_future[0] = None
                                _inject_move_future[0] = _asyncio.run_coroutine_threadsafe(_inject_after_move(), loop)
                            else:
                                try:
                                    fut = _asyncio.run_coroutine_threadsafe(
                                        handler.inject_nfc_write_result(success, msg),
                                        loop,
                                    )
                                    fut.result(timeout=10)
                                except Exception as exc:
                                    logger.warning("[RFID] >>> write result inject FAILED: %s", exc)
                    else:
                        logger.debug("[RFID] >>> unhandled event: %r", msg)
            finally:
                _apply_lock.release()
            return JSONResponse({"messages": msgs, "connected": True, "applied": applied, "port": port,
                                 "driver_available": _driver_ok})

        # ── Personality management (under /rfid/ to avoid runtime route conflicts) ──

        from reachy_mini_conversation_app.headless_personality import (
            DEFAULT_OPTION as _DEFAULT_OPTION,
            list_personalities as _list_personalities,
            read_instructions_for as _read_instructions_for,
            available_tools_for as _available_tools_for,
            resolve_profile_dir as _resolve_profile_dir,
            _write_profile,
            _sanitize_name,
        )

        class _PersonalitySaveBody(BaseModel):
            name: str
            instructions: str
            tools_text: str
            voice: str = "cedar"

        class _PersonalityApplyBody(BaseModel):
            name: str

        @_app.get("/rfid/personalities/list")
        def _rfid_pers_list() -> JSONResponse:
            choices = [_DEFAULT_OPTION, *_list_personalities()]
            from reachy_mini_conversation_app.config import config as _cfg
            current = getattr(_cfg, "REACHY_MINI_CUSTOM_PROFILE", None) or _DEFAULT_OPTION
            personality_to_code = {
                p: t for p in _list_personalities() if (t := _to_tag_token(p)) is not None
            }
            return JSONResponse({"choices": choices, "current": current, "personality_to_code": personality_to_code})

        @_app.get("/rfid/personalities/load")
        def _rfid_pers_load(name: str = "") -> JSONResponse:
            load_name = name or _DEFAULT_OPTION
            instr = _read_instructions_for(load_name)
            tools_txt = ""
            voice = "cedar"
            if load_name != _DEFAULT_OPTION:
                pdir = _resolve_profile_dir(load_name)
                tp = pdir / "tools.txt"
                if tp.exists():
                    tools_txt = tp.read_text(encoding="utf-8")
                vf = pdir / "voice.txt"
                if vf.exists():
                    v = vf.read_text(encoding="utf-8").strip()
                    voice = v or "cedar"
            avail = _available_tools_for(load_name)
            enabled = [
                ln.strip()
                for ln in tools_txt.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            return JSONResponse({
                "instructions": instr,
                "tools_text": tools_txt,
                "voice": voice,
                "available_tools": avail,
                "enabled_tools": enabled,
            })

        @_app.post("/rfid/personalities/save")
        def _rfid_pers_save(body: _PersonalitySaveBody) -> JSONResponse:
            name_s = _sanitize_name(body.name)
            if not name_s:
                return JSONResponse({"ok": False, "error": "invalid_name"}, status_code=400)
            try:
                _write_profile(name_s, body.instructions, body.tools_text, body.voice or "cedar")
                value = f"user_personalities/{name_s}"
                choices = [_DEFAULT_OPTION, *_list_personalities()]
                personality_to_code = {
                    p: t for p in _list_personalities() if (t := _to_tag_token(p)) is not None
                }
                return JSONResponse({"ok": True, "value": value, "code": _to_tag_token(value), "choices": choices, "personality_to_code": personality_to_code})
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        class _PersonalityDeleteBody(BaseModel):
            name: str

        @_app.post("/rfid/personalities/delete")
        def _rfid_pers_delete(body: _PersonalityDeleteBody) -> JSONResponse:
            import shutil as _shutil
            if body.name == _DEFAULT_OPTION:
                return JSONResponse({"ok": False, "error": "cannot_delete_default"}, status_code=400)
            try:
                pdir = _resolve_profile_dir(body.name)
                if pdir.exists():
                    _shutil.rmtree(pdir)
            except Exception:
                pass
            choices = [_DEFAULT_OPTION, *_list_personalities()]
            personality_to_code = {
                p: t for p in _list_personalities() if (t := _to_tag_token(p)) is not None
            }
            return JSONResponse({"ok": True, "choices": choices, "personality_to_code": personality_to_code})

        @_app.post("/rfid/personalities/apply")
        def _rfid_pers_apply(body: _PersonalityApplyBody) -> JSONResponse:
            loop = _get_loop()
            if loop is None:
                return JSONResponse({"ok": False, "error": "loop_unavailable"}, status_code=503)

            async def _do_apply() -> str:
                handler = _get_handler()
                profile = None if body.name == _DEFAULT_OPTION else body.name
                return await handler.apply_personality(profile)

            try:
                fut = _asyncio.run_coroutine_threadsafe(_do_apply(), loop)
                status = fut.result(timeout=10)
                return JSONResponse({"ok": True, "status": status})
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

        @_app.get("/rfid/voices")
        def _rfid_voices() -> JSONResponse:
            loop = _get_loop()
            if loop is None:
                return JSONResponse(["cedar"])

            async def _get_v() -> list:
                try:
                    handler = _get_handler()
                    return await handler.get_available_voices()
                except Exception:
                    return ["cedar"]

            try:
                fut = _asyncio.run_coroutine_threadsafe(_get_v(), loop)
                return JSONResponse(fut.result(timeout=5))
            except Exception:
                return JSONResponse(["cedar"])

        logger.info("RFID routes initialized.")

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                # Serve /static/* assets
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        class ApiKeyPayload(BaseModel):
            openai_api_key: str

        class BackendPayload(BaseModel):
            backend: str
            api_key: Optional[str] = None
            hf_mode: Optional[str] = None
            hf_host: Optional[str] = None
            hf_port: Optional[int] = None

        def _status_payload() -> dict[str, object]:
            backend_provider = get_backend_choice()
            active_backend = self._active_backend()
            has_openai_key = self._has_required_key(OPENAI_BACKEND)
            has_gemini_key = self._has_required_key(GEMINI_BACKEND)
            hf_session_url = get_hf_session_url()
            hf_ws_url = get_hf_direct_ws_url()
            hf_direct_host, hf_direct_port = parse_hf_direct_target(hf_ws_url)
            has_hf_session_url = bool(hf_session_url)
            has_hf_ws_url = bool(hf_ws_url)
            hf_connection_selection = get_hf_connection_selection()
            hf_connection_mode = hf_connection_selection.mode
            has_hf_connection = hf_connection_selection.has_target
            can_proceed_with_openai = has_openai_key
            can_proceed_with_gemini = has_gemini_key
            can_proceed_with_hf = has_hf_connection
            readiness_backend = backend_provider if self._can_rebuild_handler() else active_backend
            can_proceed = self._has_required_key(readiness_backend)
            requires_restart = backend_provider != active_backend and not self._can_rebuild_handler()
            backend_connection = self._backend_connection_status()
            return {
                "active_backend": active_backend,
                "backend_provider": backend_provider,
                "has_key": can_proceed,
                "has_openai_key": has_openai_key,
                "has_gemini_key": has_gemini_key,
                "has_hf_session_url": has_hf_session_url,
                "has_hf_ws_url": has_hf_ws_url,
                "has_hf_connection": has_hf_connection,
                "hf_connection_mode": hf_connection_mode,
                "hf_direct_host": hf_direct_host,
                "hf_direct_port": hf_direct_port,
                "can_proceed": can_proceed,
                "can_proceed_with_openai": can_proceed_with_openai,
                "can_proceed_with_gemini": can_proceed_with_gemini,
                "can_proceed_with_hf": can_proceed_with_hf,
                "requires_restart": requires_restart,
                **backend_connection,
            }

        # GET / -> index.html
        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # GET /status -> whether key is set
        @self._settings_app.get("/status")
        def _status() -> JSONResponse:
            return JSONResponse(_status_payload())

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("reachy_mini_conversation_app.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # POST /openai_api_key -> set/persist key
        @self._settings_app.post("/openai_api_key")
        def _set_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)
            self._persist_api_key(key)
            return JSONResponse({"ok": True, **_status_payload()})

        @self._settings_app.post("/backend_config")
        def _set_backend(payload: BackendPayload) -> JSONResponse:
            backend = payload.backend.strip().lower()
            if backend not in {OPENAI_BACKEND, GEMINI_BACKEND, HF_BACKEND}:
                return JSONResponse({"ok": False, "error": "invalid_backend"}, status_code=400)

            api_key = (payload.api_key or "").strip()
            if backend == GEMINI_BACKEND and not api_key and not self._has_required_key(GEMINI_BACKEND):
                return JSONResponse({"ok": False, "error": "empty_key"}, status_code=400)

            if backend == OPENAI_BACKEND and api_key:
                self._persist_api_key(api_key)
            if backend == GEMINI_BACKEND and api_key:
                self._persist_gemini_api_key(api_key)
            if backend == HF_BACKEND:
                hf_selection = get_hf_connection_selection()
                hf_mode = (payload.hf_mode or hf_selection.mode).strip().lower()
                if hf_mode == HF_LOCAL_CONNECTION_MODE:
                    existing_host, existing_port = parse_hf_direct_target(hf_selection.direct_ws_url)
                    host = (payload.hf_host or "").strip() or existing_host or ""
                    if not host:
                        return JSONResponse({"ok": False, "error": "empty_hf_host"}, status_code=400)
                    if "://" in host or "/" in host or "?" in host or "#" in host:
                        return JSONResponse({"ok": False, "error": "invalid_hf_host"}, status_code=400)

                    port = payload.hf_port if payload.hf_port is not None else existing_port or 8765
                    if port < 1 or port > 65535:
                        return JSONResponse({"ok": False, "error": "invalid_hf_port"}, status_code=400)

                    self._persist_hf_direct_connection(host, port)
                elif hf_mode == HF_DEPLOYED_CONNECTION_MODE:
                    if not bool(get_hf_session_url()):
                        return JSONResponse({"ok": False, "error": "missing_hf_session_url"}, status_code=400)
                    self._persist_hf_allocator_connection()
                else:
                    return JSONResponse({"ok": False, "error": "invalid_hf_mode"}, status_code=400)

            self._persist_backend_choice(backend)
            if self._can_rebuild_handler():
                self._mark_restart_requested("backend_config_changed")
            payload_data = _status_payload()
            message = "Backend saved."
            if payload_data["requires_restart"]:
                message = "Backend saved. Restart Reachy Mini Conversation from the desktop app to apply it."
            elif self._can_rebuild_handler():
                message = "Backend saved. Reconnecting backend."
            return JSONResponse(
                {
                    "ok": True,
                    "message": message,
                    **payload_data,
                }
            )

        # POST /validate_api_key -> validate key without persisting it
        @self._settings_app.post("/validate_api_key")
        async def _validate_key(payload: ApiKeyPayload) -> JSONResponse:
            key = (payload.openai_api_key or "").strip()
            if not key:
                return JSONResponse({"valid": False, "error": "empty_key"}, status_code=400)

            # Try to validate by checking if we can fetch the models
            try:
                import httpx

                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("https://api.openai.com/v1/models", headers=headers)
                    if response.status_code == 200:
                        return JSONResponse({"valid": True})
                    elif response.status_code == 401:
                        return JSONResponse({"valid": False, "error": "invalid_api_key"}, status_code=401)
                    else:
                        return JSONResponse(
                            {"valid": False, "error": "validation_failed"}, status_code=response.status_code
                        )
            except Exception as e:
                logger.warning(f"API key validation failed: {e}")
                return JSONResponse({"valid": False, "error": "validation_error"}, status_code=500)

        try:
            self._init_rfid_routes()
        except Exception as _rfid_exc:
            logger.warning("RFID routes could not be loaded: %s", _rfid_exc)

        self._settings_initialized = True

    async def _run_handler_startup_loop(self) -> None:
        """Start the realtime handler and keep settings UI alive after backend failures."""
        while not self._stop_event.is_set():
            selected_backend = get_backend_choice()
            if selected_backend != self._active_backend() or self._restart_requested.is_set():
                await self._shutdown_active_handler()
                if not self._can_rebuild_handler():
                    self._restart_requested.clear()
                    self._set_backend_connection_state("restart_required")
                    await self._sleep_or_restart_requested(0.5)
                    continue
                self._restart_requested.clear()
                try:
                    self._build_handler_for_current_backend()
                except Exception as e:
                    self._set_backend_connection_state("disconnected", e)
                    logger.warning(
                        "%s backend handler failed to initialize: %s. Retrying in %.1f seconds.",
                        selected_backend,
                        e,
                        self._backend_retry_delay,
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                    await self._sleep_or_restart_requested(self._backend_retry_delay)
                    continue

            active_backend = self._active_backend()
            if not self._has_required_key(active_backend):
                requirement_name = self._requirement_name(active_backend)
                self._set_backend_connection_state("waiting_for_config", f"{requirement_name} is not configured.")
                await self._sleep_or_restart_requested(0.5)
                continue

            self._set_backend_connection_state("connecting")
            try:
                await self.handler.start_up()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._set_backend_connection_state("disconnected", e)
                logger.warning(
                    "%s backend failed to start: %s. Settings UI remains available; retrying in %.1f seconds.",
                    active_backend,
                    e,
                    self._backend_retry_delay,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
            else:
                if self._stop_event.is_set():
                    return
                self._set_backend_connection_state("disconnected")
                if self._restart_requested.is_set():
                    logger.info("%s backend stopped for requested restart.", active_backend)
                    continue
                logger.info(
                    "%s backend session ended. Settings UI remains available; retrying in %.1f seconds.",
                    active_backend,
                    self._backend_retry_delay,
                )

            await self._sleep_or_restart_requested(self._backend_retry_delay)

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the selected backend is missing its required key, expose a tiny
        settings UI via the Reachy Mini settings server to collect it before
        starting streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    refresh_runtime_config_from_env()
            except Exception:
                pass  # Instance .env loading is optional; continue with defaults

        active_backend = self._active_backend()

        # Always expose settings UI if a settings app is available
        # (do this AFTER loading the instance .env so status endpoint sees the right value)
        self._init_settings_ui_if_needed()

        # If key is still missing -> wait until provided via the settings UI
        if not self._has_required_key(active_backend):
            requirement_name = self._requirement_name(active_backend)
            self._set_backend_connection_state("waiting_for_config", f"{requirement_name} is not configured.")
            if active_backend == HF_BACKEND and self._settings_app is None:
                logger.error(
                    "%s not found. Set it in the app .env before starting the Hugging Face backend.", requirement_name
                )
                return
            logger.warning("%s not found. Open the app settings page to configure it.", requirement_name)
            # Poll until the key becomes available (set via the settings UI)
            try:
                while not self._stop_event.is_set() and not self._has_required_key(active_backend):
                    selected_backend = get_backend_choice()
                    if selected_backend != active_backend:
                        if self._can_rebuild_handler():
                            active_backend = selected_backend
                            self._active_backend_name = selected_backend
                            self._restart_requested.set()
                            self._set_backend_connection_state("waiting_for_config")
                        else:
                            self._set_backend_connection_state("restart_required")
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for API key.")
                return
            if self._stop_event.is_set():
                return
            self._set_backend_connection_state("not_started")

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start
        apply_audio_startup_config(self._robot, logger=logger)

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                        apply_personality=self.apply_personality,
                        get_available_voices=self.get_available_voices,
                        get_current_voice=self.get_current_voice,
                        change_voice=self.change_voice,
                    )
            except Exception:
                pass
            self._tasks = [
                asyncio.create_task(self._run_handler_startup_loop(), name="realtime-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Now signal async loops to stop
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush queued playback audio immediately on user barge-in.

        Calls the SDK's ``clear_player()`` — now a first-class flush on both
        the local GStreamer and WebRTC backends (the WebRTC one also tells the
        daemon to drop audio already queued for the speaker). Falls back to the
        deprecated ``clear_output_buffer()`` only for older SDKs.
        """
        logger.info("User intervention: flushing player queue")
        audio = getattr(self._robot.media, "audio", None)
        if audio is not None:
            if hasattr(audio, "clear_player") and callable(audio.clear_player):
                audio.clear_player()
            elif hasattr(audio, "clear_output_buffer") and callable(audio.clear_output_buffer):
                # Older SDK without clear_player(); best-effort.
                audio.clear_output_buffer()
        # Drain the handler's pending output in place — do NOT replace the
        # queue object, since emit() may be awaiting it (wait_for_item).
        self._drain_output_queue()

    def _drain_output_queue(self) -> None:
        """Empty the handler's output queue in place without replacing it."""
        queue = getattr(self.handler, "output_queue", None)
        if queue is None:
            return
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler = self.handler
            try:
                handler_output = await asyncio.wait_for(handler.emit(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                # Skip empty audio frames
                if audio_data.size == 0:
                    continue

                # Reshape if needed
                if audio_data.ndim == 2:
                    # Scipy channels last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    num_samples = int(len(audio_frame) * output_sample_rate / input_sample_rate)
                    if num_samples == 0:
                        continue
                    audio_frame = resample(
                        audio_frame,
                        num_samples,
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
