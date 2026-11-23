from __future__ import annotations
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeAlias
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
HandlerOutput: TypeAlias = AudioFrame | AdditionalOutputs | None
QueueItem: TypeAlias = AudioFrame | AdditionalOutputs


def drop_pending_responses(pending: asyncio.Queue[Any]) -> int:
    """Empty a handler's queued response.create requests in place; return how many were dropped."""
    dropped = 0
    while not pending.empty():
        try:
            pending.get_nowait()
            dropped += 1
        except asyncio.QueueEmpty:
            break
    return dropped


class ConversationHandler(AsyncStreamHandler, ABC):
    """Shared app handler contract for realtime conversation backends."""

    deps: ToolDependencies
    tool_manager: BackgroundToolManager
    output_queue: asyncio.Queue[QueueItem]
    last_activity_time: float
    _activity_observer: Callable[[str], None] | None = None
    _transcript_observer: Callable[[str, str, bool], None] | None = None
    # The tool-call-log brief (2026-10-11): one report per finished (or refused)
    # tool call, pushed to console.py's /rpc dispatch. Returns whether a peer
    # could receive it, so the handler's capped queue knows what to retry.
    _tool_call_observer: Callable[[dict[str, Any]], bool] | None = None
    # Each new realtime session (the restarts included), for the companion's turn pairing.
    _session_observer: Callable[[str], None] | None = None

    def __init__(self) -> None:
        """Initialize the stream handler and its activity tracking."""
        super().__init__()
        self.last_activity_time = time.monotonic()

    def set_activity_observer(self, observer: Callable[..., None] | None) -> None:
        """Attach or detach an activity observer, called (reason) or, for a user turn's realtime
        item, (reason, item_id=...). Pass None to clear."""
        self._activity_observer = observer

    def set_session_observer(self, observer: Callable[[str], None] | None) -> None:
        """Attach/detach the observer told "started" at each new realtime session."""
        self._session_observer = observer

    def _notify_session(self, state: str, **extra: Any) -> None:
        """Tell the session observer, if attached. Not activity: it holds off nothing.

        `extra`: a "context_dropped" report's item_id (route all, rule 9).
        """
        observer = self._session_observer
        if observer is not None:
            try:
                observer(state, **extra) if extra else observer(state)
            except Exception:
                logger.debug("session observer raised (ignored)", exc_info=True)

    def set_transcript_observer(self, observer: Callable[..., None] | None) -> None:
        """Attach/detach a transcript observer, called (role, text, final, artifact): artifact
        is the output gate's cut kind for a reply that was not spoken, else None. A user
        transcript with a realtime item also passes item_id=...; a thought's lines pass thought=<id>;
        his final that answered a person's item passes answers=<that item> (the next-exchange line)."""
        self._transcript_observer = observer

    def set_tool_call_observer(self, observer: Callable[[dict[str, Any]], bool] | None) -> None:
        """Attach/detach the tool-call-report observer. Pass None to clear."""
        self._tool_call_observer = observer

    def _emit_transcript(
        self,
        role: str,
        text: str,
        final: bool = True,
        artifact: str | None = None,
        item_id: str | None = None,
        thought: str | None = None,
        answers: str | None = None,
    ) -> None:
        """Forward one transcript chunk to the observer, if attached; a thought's lines carry its id."""
        observer = self._transcript_observer
        if observer is not None and text:
            try:
                if thought is not None:
                    observer(role, text, final, artifact, thought=thought)
                elif answers is not None:
                    observer(role, text, final, artifact, answers=answers)
                elif item_id is None:
                    observer(role, text, final, artifact)
                else:
                    observer(role, text, final, artifact, item_id=item_id)
            except Exception:
                logger.debug("transcript observer raised (ignored)", exc_info=True)

    def _mark_activity(self, reason: str, item_id: str | None = None) -> None:
        """Record non-idle conversation activity for the idle timer. `item_id` is the realtime
        item of a user turn's speech or transcription, for the companion's turn pairing."""
        self.last_activity_time = time.monotonic()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)
        if self._activity_observer is not None:
            try:
                if item_id is None:
                    self._activity_observer(reason)
                else:
                    self._activity_observer(reason, item_id=item_id)
            except Exception:
                logger.debug("activity observer raised (ignored)", exc_info=True)

    async def emit(self) -> HandlerOutput:
        """Emit the next queued output."""
        return await wait_for_item(self.output_queue)

    @abstractmethod
    def _is_connected(self) -> bool:
        """Return whether the backend session/connection is currently open."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Return voices available for the active backend."""
        ...

    @abstractmethod
    def get_current_voice(self) -> str:
        """Return the current voice."""
        ...

    @abstractmethod
    async def change_voice(self, voice: str) -> str:
        """Change the current voice."""
        ...

    def want_instructions(self, why: str) -> None:
        """Ask for the session's instructions to be sent again when it is safe (the companion,
        10-29). Any thread may call it: it only marks the wish. A backend whose session cannot
        be updated live ignores it."""

    @abstractmethod
    async def say(self, text: str) -> None:
        """Make the robot speak ``text`` now (injected turn; not verbatim TTS).

        The backend is speech-to-speech, so ``text`` is an instruction the
        model voices, not a guaranteed-literal string. Raises if no session is
        open.
        """
        ...

    @abstractmethod
    async def cancel_response(self) -> None:
        """Stop the reply still being generated, so none of its rest plays.

        The companion's ``conversation.interrupt`` calls it after flushing the player.
        """
        ...
