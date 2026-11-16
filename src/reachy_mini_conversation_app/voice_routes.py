"""Voice selection exposed over JSON-RPC."""

import asyncio
import logging
from typing import Any, TypeVar
from collections.abc import Callable, Awaitable, Coroutine

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import get_default_voice, get_available_voices
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


class RouteError(Exception):
    """Represent a JSON-RPC operation error with a stable reason."""

    def __init__(
        self,
        reason: str,
        *,
        extra: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize an operation error."""
        super().__init__(message or reason)
        self.reason = reason
        self.extra = extra or {}
        self.message = message or reason


class VoiceOps:
    """Provide transport-independent voice operations."""

    def __init__(
        self,
        handler: ConversationHandler,
        get_loop: Callable[[], asyncio.AbstractEventLoop | None],
        *,
        get_voices: Callable[[], Awaitable[list[str]]] | None = None,
        get_current_voice: Callable[[], str] | None = None,
        change_voice: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        """Initialize operations with runtime callbacks."""
        self._handler = handler
        self._get_loop = get_loop
        self._get_voices = get_voices
        self._get_current_voice = get_current_voice
        self._change_voice = change_voice

    async def _run_on_loop(
        self,
        coroutine: Coroutine[Any, Any, ResultT],
        timeout: float = 10.0,
    ) -> ResultT:
        loop = self._get_loop()
        if loop is None:
            coroutine.close()
            raise RouteError("loop_unavailable")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)

    async def voices(self) -> list[str]:
        """List voices available for the active backend."""
        if self._get_loop() is None:
            return get_available_voices()

        async def _get_available() -> list[str]:
            if self._get_voices is not None:
                return await self._get_voices()
            return await self._handler.get_available_voices()

        try:
            return await self._run_on_loop(_get_available())
        except Exception as exc:
            logger.warning("Failed to read available voices: %s", exc)
            return get_available_voices()

    def current_voice(self) -> dict[str, str]:
        """Return the current voice."""
        try:
            callback = self._get_current_voice or self._handler.get_current_voice
            return {"voice": callback()}
        except Exception as exc:
            logger.warning("Failed to read current voice: %s", exc)
            return {"voice": get_default_voice()}

    async def apply_voice(self, voice: str) -> dict[str, Any]:
        """Change the current voice without rebuilding the backend."""
        selected_voice = voice.strip()
        if not selected_voice:
            raise RouteError("missing_voice")

        async def _apply() -> str:
            if self._change_voice is not None:
                return await self._change_voice(selected_voice)
            return await self._handler.change_voice(selected_voice)

        try:
            status = await self._run_on_loop(_apply())
        except RouteError:
            raise
        except Exception as exc:
            raise RouteError("voice_apply_failed", message=str(exc)) from exc
        return {"ok": True, "status": status}


def build_voice_ops(
    handler: ConversationHandler,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    *,
    get_voices: Callable[[], Awaitable[list[str]]] | None = None,
    get_current_voice: Callable[[], str] | None = None,
    change_voice: Callable[[str], Awaitable[str]] | None = None,
) -> VoiceOps:
    """Build voice operations for a control transport."""
    return VoiceOps(
        handler,
        get_loop,
        get_voices=get_voices,
        get_current_voice=get_current_voice,
        change_voice=change_voice,
    )


def register_voice_methods(rpc: JsonRpcServer, ops: VoiceOps) -> None:
    """Register voice operations as JSON-RPC methods."""

    def _wrap(operation: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        async def _method(params: dict[str, Any]) -> Any:
            try:
                result = operation(params)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except RouteError as exc:
                raise JsonRpcError(
                    exc.message,
                    reason=exc.reason,
                    data=exc.extra,
                    code=-32000,
                ) from exc

        return _method

    rpc.register("voices.list", _wrap(lambda params: ops.voices()))
    rpc.register("voices.current", _wrap(lambda params: ops.current_voice()))
    rpc.register("voices.apply", _wrap(lambda params: ops.apply_voice(str(params.get("voice", "")))))
