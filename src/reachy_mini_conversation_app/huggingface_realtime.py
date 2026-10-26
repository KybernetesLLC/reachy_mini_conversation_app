import re
import json
import time
import uuid
import base64
import random
import asyncio
import logging
import contextlib
from typing import TYPE_CHECKING, Any, Final, Tuple, Callable, Optional
from collections import deque

import httpx
import numpy as np
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from numpy.typing import NDArray
from huggingface_hub import get_token
from typing_extensions import Literal, TypedDict
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeToolsConfigParam,
    RealtimeFunctionToolParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from websockets.exceptions import ConnectionClosedError
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.config import (
    HF_LOCAL_CONNECTION_MODE,
    config,
    get_default_voice,
    set_custom_profile,
    get_available_voices,
    get_hf_direct_ws_url,
    parse_hf_realtime_url,
    get_hf_connection_selection,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_instructions,
    get_session_greeting_prompt,
)
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.output_gate import (  # noqa: F401  (spoken_artifact: tests)
    CUT,
    PLAY,
    UNDECIDED,
    PLAY_AND_LOG,
    CUT_AND_EXECUTE,
    Verdict,
    ToolNames,
    judge,
    spoken_artifact,
    without_calls,
)
from reachy_mini_conversation_app.tools.core_tools import (
    ToolSpec,
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


if TYPE_CHECKING:
    from openai.resources.realtime.realtime import AsyncRealtimeConnection


logger = logging.getLogger(__name__)

_RESPONSE_DONE_TIMEOUT: Final[float] = 30.0
_RESPONSE_REJECTION_RETRY_DELAY: Final[float] = 0.5
_OWED_OUTPUT_TIMEOUT: float = _RESPONSE_DONE_TIMEOUT  # longest hold of an in-band create
# Carries a say() announcement's text (a timer, a greeting) on its queued create: spoken as
# its own reply, never merged with another create. The sender adds the text as a user item
# just before the create, so a tool's follow-up reply never sees it (2026-10-16, 15:18:45:
# the follow-up voiced only the timer and dropped the search answer). Internal: stripped
# before the create is sent.
_ANNOUNCEMENT: Final[str] = "_announcement"


class InputTranscriptChunksByItem(BaseModel):
    """Current item_id and its accumulated deltas. Only one item at a time."""

    item_id: str | None = None
    deltas: list[str] = Field(default_factory=list)


# Tools taken off the list whose leaked names the gate still cuts, as leak-safe (2026-10-21: idle_do_nothing
# left the list, but the model leaked it 8 times in the week before and may say it from habit).
RETIRED_LEAK_SAFE = ("idle_do_nothing",)


def gate_tool_names(tool_specs: list[ToolSpec], registry: dict[str, Any]) -> ToolNames:
    """Build the output gate's names: the session's tools, leak_safe from each tool's class, then the retired ones."""
    loaded = [(spec["name"], bool(getattr(registry.get(spec["name"]), "leak_safe", False))) for spec in tool_specs]
    names = {name for name, _ in loaded}
    return ToolNames(loaded + [(name, True) for name in RETIRED_LEAK_SAFE if name not in names])


def to_realtime_tools_config(tool_specs: list[ToolSpec]) -> RealtimeToolsConfigParam:
    """Convert app tool specs to the OpenAI-compatible realtime session shape."""
    realtime_tools: RealtimeToolsConfigParam = []
    for spec in tool_specs:
        realtime_tools.append(
            RealtimeFunctionToolParam(
                type="function",
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
            )
        )
    return realtime_tools


class HFNativeRateAudioPCM(TypedDict):
    """Hugging Face extension for native-rate PCM audio."""

    type: Literal["audio/pcm"]
    rate: None


def _native_rate_audio_pcm() -> HFNativeRateAudioPCM:
    """Return the Hugging Face native-rate PCM config."""
    return {"type": "audio/pcm", "rate": None}


def _build_openai_compatible_client_from_realtime_url(
    realtime_url: str,
    bearer_token: str | None,
) -> tuple[AsyncOpenAI, dict[str, str]]:
    """Build an OpenAI-compatible realtime client from a direct websocket/base URL."""
    parsed = parse_hf_realtime_url(realtime_url)
    client = AsyncOpenAI(
        api_key=bearer_token or "DUMMY",
        base_url=parsed.base_url,
        websocket_base_url=parsed.websocket_base_url,
    )
    return client, parsed.connect_query


# The inner monologue: since 2026-10-17 the supervisor writes sight notes with Claude
# Haiku. The app hands it each frame it adds (`on_sight`) and asks the voice service for
# nothing more: before, every glance cost a second upload and a vision pass there,
# serialised against the household's turn (the cold-open brief).

# Reliable search (the companion, 2026-10-09). At 09:50:26 "Let me check." was spoken
# as a response of its own and no call followed, twice. The app now speaks the filler
# itself (an out-of-band audio response, tagged), and a response that promises a lookup
# without a call gets one nudge per user turn.
FILLER_TAG = {"companion": "filler"}
FILLER_LINES = ("Let me check.", "One moment.", "Let me look that up.", "Just a second.")
FILLER_AFTER_S = 1.0  # a lookup tool still running this long gets a filler
SEARCH_TOOLS = ("search_web", "web_search")  # by name ending: always slow enough
# Matched by name ending: "weather" is the companion's tool (2026-10-24) and Pollen's get_weather.
LOOKUP_TOOLS = ("flights", "boats", "transit", "events", "why_lit", "weather")
# Tools that only move the body. A response that already spoke and called only these gets
# no follow-up: on 2026-10-23 08:42 the greeting's follow-up read its context aloud.
GESTURE_TOOLS = frozenset({"express"})
LOOKUP_PROMISE = re.compile(
    r"\b(let me (check|look|find out|search)(?! at you)|let me see (what|if|whether)"
    r"|(looking|look) (it|that|this) up|i'?ll (check|find out|look (it|that|this) up))\b",
    re.IGNORECASE,
)
MISSED_CALL_NOTE = (
    "You said you would look that up but called no tool. Call the tool you meant now "
    "(web_search for facts from the web) with the person's question."
)
# A glance's frame is context, not a question (the companion, 2026-10-08): after a
# song, with nothing left to say, the model described the frame taken as the
# conversation opened.
GLANCE_CONTEXT_NOTE = (
    "(A frame from your camera, for context only. Do not describe it or mention it "
    "unless the person asks what you see, or it bears on what they asked.)"
)

# Largest message the client accepts from the server (websockets' max_size); the
# default, 1 MiB, closed the session with 1009 on an echoed image (2026-10-16).
REALTIME_MAX_MESSAGE_BYTES = 16 * 2**20


# A companion note waits for a quiet moment: nobody speaking for NOTE_QUIET_SECONDS and
# no response active, so it never sits between a person's last word and the reply (the
# backend takes one response at a time).
NOTE_QUIET_SECONDS = 2.0
_NOTE_POLL_SECONDS = 0.25

# A companion note (person recognition, task 11: who has joined, who is no longer
# in view) is dropped after this long without a quiet moment.
COMPANION_NOTE_WAIT_S = 30.0

NOTE_TEXT = "[Companion note, not spoken: {text}]"


def note_item_text(text: str) -> str:
    """Return the note as sent, with brackets in the text turned to parentheses so it cannot close the wrapper."""
    return NOTE_TEXT.format(text=text.replace("[", "(").replace("]", ")"))


# The startup greeting (the companion, 2026-10-24): only when an open asks for it, and only
# into silence. It waits for the console to finish the open (the pre-roll flush), at most
# GREETING_SETTLE_WAIT_S, then GREETING_HOLD_S more; speech or a reply by then skips it.
GREETING_HOLD_S = 0.8
GREETING_SETTLE_WAIT_S = 2.0
# The greeting is its own out-of-band response, tagged, so nothing of it stays in the
# conversation to be answered late. Re-checked at send; cancelled if the person starts.
GREETING_TAG = {"companion": "greeting"}


def _out_of_band_request(
    tag: dict[str, str], modalities: list[str], instructions: str, content: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a tagged response.create body outside the conversation (a filler)."""
    return {
        "conversation": "none",
        "output_modalities": modalities,
        "metadata": dict(tag),
        "instructions": instructions,
        "tool_choice": "none",
        "input": [{"type": "message", "role": "user", "content": content}],
    }


def session_size(session: Any) -> str:
    """What a session.update sends, sized: instructions and tool definitions, with a rough
    token count (4 characters a token). The owner, 2026-10-20: leaks grew with the context."""
    instructions = str(session.get("instructions") or "")
    tools = list(session.get("tools") or [])
    tools_chars = len(json.dumps(tools, default=str))
    return (
        f"instructions {len(instructions)} chars, {len(tools)} tools {tools_chars} chars, "
        f"~{(len(instructions) + tools_chars) // 4} tokens"
    )


def usage_line(usage: Any, *, images: int, image_bytes: int) -> str | None:
    """The server's token counts for one response, or None when it sent none."""
    if usage is None:
        return None
    details = getattr(usage, "input_token_details", None)
    parts = [
        f"{key} {value}"
        for key in ("text", "audio", "image", "cached")
        if (value := getattr(details, f"{key}_tokens", None)) is not None
    ]
    breakdown = f" ({', '.join(parts)})" if parts else ""
    return (
        f"Context: {getattr(usage, 'input_tokens', '?')} input tokens{breakdown}, "
        f"{getattr(usage, 'output_tokens', '?')} output; {images} images ({image_bytes // 1024} KiB) in this session"
    )


def _companion_tag(event: Any) -> str | None:
    """Return the companion tag of a response.created/done event's response, if any."""
    metadata = getattr(getattr(event, "response", None), "metadata", None)
    return metadata.get("companion") if isinstance(metadata, dict) else None


# The audio cut (2026-10-16): the output gate (output_gate.py) judges each reply's
# transcript as it streams; audio that arrives first is held for at most
# SPOKEN_ARTIFACT_HOLD_S, then released.
SPOKEN_ARTIFACT_HOLD_S = 0.3


# The tool-call-log brief (2026-10-11): one /rpc notification per finished (or
# refused) tool call -- console.py's _dispatch_tool_call sends it as
# "conversation.tool" -- so the supervisor can keep a durable record
# (events.jsonl, kind "tool") of which of the robot's tools actually get used.
# Logging only: nothing here changes a tool's own behaviour, result, or timing.
#
# While the supervisor (or no /rpc peer at all) is unreachable, reports pile up
# here instead of blocking a tool call; past this many, the oldest is dropped.
TOOL_REPORT_QUEUE_MAX: Final[int] = 100
_TOOL_ARG_STRING_LIMIT: Final[int] = 200

_TOOL_REPORT_STATUS: Final[dict[ToolState, str]] = {
    ToolState.COMPLETED: "ok",
    ToolState.FAILED: "error",
    ToolState.CANCELLED: "cancelled",
}


def _cut_long_strings(value: Any, limit: int = _TOOL_ARG_STRING_LIMIT) -> Any:
    """Return `value` with every string truncated to `limit` characters, recursively.

    The brief: args are logged as-is (this is the household's own robot, no
    redaction), just capped so one long string can't bloat the report.
    """
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        return {key: _cut_long_strings(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_cut_long_strings(item, limit) for item in value]
    return value


def _tool_duration_ms_and_bytes(completed: ToolNotification) -> tuple[float, int]:
    """Return (ms from call to result, bytes of the JSON result or error)."""
    ms = completed.duration_s * 1000 if completed.duration_s is not None else -1.0
    payload = completed.error if completed.error is not None else completed.result
    size = len(json.dumps(payload, default=str)) if payload is not None else 0
    return ms, size


def _tool_report_status_and_reason(completed: ToolNotification) -> Tuple[str, Optional[str]]:
    """Map a finished call to the conversation.tool report's status + a short reason.

    A tool that refuses (the companion's stay_quiet, express, or a quiet-hours
    gate) returns an ordinary successful result, not an error -- the generic
    dispatcher in core_tools.py cannot tell a refusal from a success by itself.
    A dict result is reported as `status: "refused"` when it has `"ok": False`
    or `"refused": True`. The short text is its "why", else its "reason" (""
    for `ok: False` with neither). An error still wins; a result without `ok`
    stays "ok".
    """
    status = _TOOL_REPORT_STATUS.get(completed.status, "error")
    reason = completed.error
    result = completed.result
    if status == "ok" and isinstance(result, dict):
        if result.get("ok") is False:
            status = "refused"
            reason = result.get("why") or result.get("reason") or ""
        elif result.get("refused") is True:
            status = "refused"
            reason = result.get("why") or result.get("reason")
    return status, reason


class HuggingFaceRealtimeHandler(ConversationHandler):
    """Realtime stream handler for the Hugging Face OpenAI-compatible endpoint."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__()

        self.deps = deps

        self.client: AsyncOpenAI
        self.connection: "AsyncRealtimeConnection | None" = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.instance_path = instance_path
        self._voice_override: str | None = self._normalize_startup_voice(startup_voice)
        self._realtime_connect_query: dict[str, str] = {}

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_debounce_delay = 0.5  # seconds
        self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._response_started_or_rejected_event: asyncio.Event = asyncio.Event()
        self._last_response_rejected: bool = False
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None
        # Set by LocalStream.open_session for an open that asked for the greeting; spent
        # when the greeting is sent or skipped, so a reconnect within it stays quiet.
        self.greet_at_open = False
        self._spoke_since_open = False
        self._greeting_live = False  # its create is sent and its response not yet done
        self._open_settled: asyncio.Event = asyncio.Event()
        self._in_flight_tool_calls: set[str] = set()
        # Function calls whose function_call_output is not yet sent. The server rejects an
        # in-band response.create while any is owed ("Cannot create a response while function
        # call outputs are pending", 2026-10-16), so the sender holds those creates until this
        # empties. Unlike _in_flight_tool_calls, a new user turn does not clear it.
        self._owed_outputs: set[str] = set()
        self._no_outputs_owed: asyncio.Event = asyncio.Event()
        self._no_outputs_owed.set()
        # One refused reply recovered per user turn at most (_recover_refused_reply).
        self._reply_recovered_this_turn = False
        # A tool's follow-up is queued (pending) or is the live response (live): say() does
        # not cancel it.
        self._followup_pending = False
        self._followup_live = False
        # Reliable search: this response's transcript and whether it called a tool; this
        # user turn's guard and filler, each used at most once; the last filler line.
        self._resp_text = ""
        self._resp_called = False
        self._turn_guarded = False
        self._turn_filled = False
        self._last_filler: str | None = None
        self._search_answer_pending = False
        self._filler_tasks: set[asyncio.Task[None]] = set()
        self._tool_batch_needs_response = False
        # Whether every tool in this batch so far is a gesture whose response already spoke.
        self._tool_batch_only_gestured = True
        # Whether this response spoke; the recent responses that did, by id (2026-10-23).
        self._resp_spoke = False
        self._spoken_responses: "deque[str]" = deque(maxlen=64)
        # Each frame added to the conversation, handed to the companion for its sight
        # note (Claude Haiku, 2026-10-17): (base64 JPEG, source). Set by the console.
        self.on_sight: Callable[[str, str], None] | None = None
        # The cold-open brief (2026-10-17), per session: what went to the voice service,
        # and the glance at open against the first reply to a person's turn (monotonic).
        self._session_images = 0
        self._usage_missing_logged = False
        self._session_image_bytes = 0
        self._session_creates = 0
        self._open_glance_at: float | None = None
        self._first_reply_at: float | None = None
        self._user_speaking = False
        self._last_speech_at = 0.0  # monotonic; the session's open counts as speech
        self._note_tasks: set[asyncio.Task[Any]] = set()
        # Why the last note was dropped, for the RPC to report (a now note's answer).
        self.note_drop_reason: str | None = None
        # The tool-call-log brief: a capped, in-memory backlog of conversation.tool
        # reports, drained through console.py's /rpc dispatch (see set_tool_call_observer).
        self._tool_report_queue: "deque[dict[str, Any]]" = deque(maxlen=TOOL_REPORT_QUEUE_MAX)
        self._tool_reports_dropped = 0
        # The audio cut: this response's gate ("pass", "hold", "open" or "cut"), its
        # transcript so far, audio held back, whether any was released, and the timer.
        self._gate = "pass"
        self._gate_text = ""
        self._gate_held: list[Tuple[int, NDArray[np.int16]]] = []
        self._gate_released = False
        self._gate_held_at = 0.0
        self._gate_timer: asyncio.Task[None] | None = None
        # The output gate's tool names (the live registry, at each session open), the
        # kind of this response's cut (the transcript notification's mark), and the
        # leaked calls of leak_safe tools still running.
        self._gate_tools = ToolNames()
        self._gate_cut_kind: str | None = None
        # Whether this response's transcript.done came, and whether it had any audio: the
        # response.done fallback is logged when a reply had none (the owner, 2026-10-17).
        self._gate_transcript_done = False
        self._gate_had_audio = False
        self._leak_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _sanitize_tool_result_for_model(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Remove bulky transport-only fields before echoing tool output back to the model."""
        if tool_name == "camera" and "b64_im" in tool_result:
            sanitized = dict(tool_result)
            sanitized.pop("b64_im", None)
            sanitized["image_attached"] = True
            return sanitized
        return tool_result

    def _normalize_startup_voice(self, voice: str | None) -> str | None:
        """Return a valid persisted startup voice, or None."""
        return self._resolve_backend_voice(voice, source="persisted startup voice")

    async def _wait_for_response_done_before_tool_result(self) -> bool:
        """Return whether the function-call response finished before sending tool output."""
        if self._response_done_event.is_set():
            return True

        try:
            await asyncio.wait_for(
                self._response_done_event.wait(),
                timeout=_RESPONSE_DONE_TIMEOUT,
            )
            return True
        except asyncio.TimeoutError:
            return False

    def _resolve_backend_voice(
        self,
        voice: str | None,
        *,
        source: str,
        fallback: str | None = None,
    ) -> str | None:
        """Return a backend-supported voice, optionally falling back when unsupported."""
        available_voices = get_available_voices()
        voice_value = (voice or "").strip()
        if not voice_value:
            return fallback

        voice_by_lowercase = {candidate.lower(): candidate for candidate in available_voices}
        normalized_voice = voice_by_lowercase.get(voice_value.lower())
        if normalized_voice is not None:
            return normalized_voice

        if voice:
            logger.warning(
                "Ignoring unsupported %s %r; expected one of %s",
                source,
                voice,
                available_voices,
            )
        return fallback

    def _get_session_config(self, tool_specs: list[ToolSpec]) -> RealtimeSessionCreateRequestParam:
        """Return the Hugging Face OpenAI-compatible session config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=get_session_instructions(self.instance_path),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    # The OpenAI SDK type only includes 24 kHz PCM, but the HF
                    # compatible server uses rate=None for native 16 kHz mode.
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language=config.REALTIME_TRANSCRIPTION_LANGUAGE,
                    ),
                    turn_detection=ServerVad(type="server_vad", interrupt_response=True),
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    def _is_connected(self) -> bool:
        """Return whether the realtime connection is open."""
        return self.connection is not None

    def _idle_behavior_ready(self) -> bool:
        """Hold idle behavior while a model response is still active."""
        return self._response_done_event.is_set()

    async def _cancel_partial_transcript_task(self) -> None:
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

    async def change_voice(self, voice: str) -> str:
        """Change only the voice, updating the active session when possible."""
        default_voice = get_default_voice()
        resolved_voice = (
            self._resolve_backend_voice(voice, source="requested voice", fallback=default_voice) or default_voice
        )
        self._voice_override = resolved_voice
        if self.connection is not None:
            try:
                await self.connection.session.update(
                    session=RealtimeSessionCreateRequestParam(
                        type="realtime",
                        audio=RealtimeAudioConfigParam(
                            output=RealtimeAudioConfigOutputParam(
                                voice=resolved_voice,
                            ),
                        ),
                    ),
                )
                return f"Voice changed to {resolved_voice}."
            except Exception as e:
                logger.warning("Failed to update live session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        default_voice = get_default_voice()
        voice = self._voice_override or get_session_voice(default=default_voice)
        return self._resolve_backend_voice(voice, source="session voice", fallback=default_voice) or default_voice

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality to the active or next realtime connection."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            instructions = get_session_instructions(self.instance_path)
            voice = self.get_current_voice()
            core_tools.initialize_tools(force=True)
        except Exception as exc:
            set_custom_profile(previous_profile)
            logger.error("Failed to resolve personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

        if self.connection is not None:
            try:
                await self.connection.session.update(
                    session=RealtimeSessionCreateRequestParam(
                        type="realtime",
                        instructions=instructions,
                        audio=RealtimeAudioConfigParam(
                            output=RealtimeAudioConfigOutputParam(
                                voice=voice,
                            ),
                        ),
                    ),
                )
                logger.info("Applied personality via live update: %s", profile or "default")
            except Exception as exc:
                logger.warning("Live update failed; will restart session: %s", exc)

            try:
                await self._restart_session()
                return "Applied personality and restarted realtime session."
            except Exception as exc:
                logger.warning("Failed to restart session after apply: %s", exc)
                return "Applied personality. Will take effect on next connection."

        logger.info(
            "Applied personality recorded: %s (no live connection; will apply on next session)",
            profile or "default",
        )
        return "Applied personality. Will take effect on next connection."

    async def _emit_debounced_partial(self, transcript: str, item_id: str, sequence_counter: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)

            input_transcript = self.input_transcript_chunks_by_item
            if input_transcript.item_id == item_id and len(input_transcript.deltas) - 1 == sequence_counter:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a Hugging Face partial transcript snapshot."""
        input_transcript.item_id = item_id
        input_transcript.deltas = [delta]

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        self.client = await self._build_realtime_client()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    self.client = await self._build_realtime_client()
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: realtime client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            self.client = await self._build_realtime_client()
            asyncio.create_task(self._run_realtime_session(), name="realtime-session-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _recover_refused_reply(self) -> None:
        """Send a reply the server refused while a tool's output was owed, once the outputs are in.

        2026-10-21 11:26: "Absolutely." in an express wait went unanswered. The sender holds the
        create, and a tool's own follow-up merges with it. Only while an output is owed, and once per
        user turn: a dropped output, or a tool past the 30 s hold, made every retry be refused again.
        """
        if not self._owed_outputs:
            logger.info("a reply the server refused with no tool output owed here: not retried")
            return
        if self._reply_recovered_this_turn:
            logger.info("a reply the server refused again this turn: not retried")
            return
        self._reply_recovered_this_turn = True
        logger.info("a reply the server refused while tool outputs were owed: sending it once they are in")
        await self._safe_response_create()

    def _new_user_turn(self) -> None:
        self._reply_recovered_this_turn = False

    def _owe_output(self, call_id: str) -> None:
        self._owed_outputs.add(call_id)
        self._no_outputs_owed.clear()

    def _output_settled(self, call_id: str) -> None:
        """Stop holding creates for a call whose output was sent, or never will be."""
        self._owed_outputs.discard(call_id)
        if not self._owed_outputs:
            self._no_outputs_owed.set()

    def _reset_owed_outputs(self) -> None:
        self._owed_outputs.clear()
        self._no_outputs_owed.set()

    async def _wait_for_owed_outputs(self, kwargs: dict[str, Any]) -> bool:
        """Hold an in-band response.create until every function call has its output; True if held.

        Out-of-band creates (fillers) are not checked by the server and go at once.
        """
        if self._no_outputs_owed.is_set() or kwargs.get("response", {}).get("conversation") == "none":
            return False
        logger.info("response.create deferred: %d function call output(s) owed", len(self._owed_outputs))
        try:
            await asyncio.wait_for(self._no_outputs_owed.wait(), timeout=_OWED_OUTPUT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "Function call outputs still owed after %.0f s; sending response.create", _OWED_OUTPUT_TIMEOUT
            )
            return True
        logger.info("response.create released: function call outputs sent")
        return True

    def _requeue(self, keep: Callable[[dict[str, Any]], bool], front: dict[str, Any] | None = None) -> int:
        """Drain the queue and put back `front`, then the entries `keep` accepts, in order.

        Returns how many entries were dropped (merged into the create being sent).
        """
        kept: list[dict[str, Any]] = [] if front is None else [front]
        dropped = 0
        while not self._pending_responses.empty():
            queued = self._pending_responses.get_nowait()
            if keep(queued):
                kept.append(queued)
            else:
                dropped += 1
        for queued in kept:
            self._pending_responses.put_nowait(queued)
        if dropped:
            logger.info("response.create: merged %d queued create(s) into the one being sent", dropped)
        return dropped

    async def _safe_response_create(self, **kwargs: Any) -> None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        await self._pending_responses.put(kwargs)

    # --- the audio cut (2026-10-16) -----------------------------------------------------
    #
    # States: "hold" (a response started, nothing decided: audio waits), "open" (released
    # by the timeout, still undecided: audio plays, a late artifact flushes it), "pass"
    # (a reply, or no response being tracked: audio plays) and "cut" (audio dropped).

    def _gate_cancel_timer(self) -> None:
        if self._gate_timer is not None and not self._gate_timer.done():
            self._gate_timer.cancel()
        self._gate_timer = None

    def _gate_start(self) -> None:
        """Hold a new response's audio until its transcript decides."""
        self._gate_cancel_timer()
        self._gate = "hold"
        self._gate_text = ""
        self._gate_cut_kind = None
        self._gate_transcript_done = False
        self._gate_had_audio = False
        self._gate_held = []
        self._gate_released = False

    def _gate_release(self, state: str) -> None:
        """Queue the held audio in order, in one synchronous burst, and move to ``state``."""
        self._gate_cancel_timer()
        if self._gate_held:
            logger.debug(
                "reply audio held %.0f ms before its transcript decided",
                (time.monotonic() - self._gate_held_at) * 1000,
            )
            self._gate_released = True
        for frame in self._gate_held:
            self.output_queue.put_nowait(frame)
        self._gate_held = []
        self._gate = state

    def _gate_drop_held(self) -> None:
        """Drop audio not yet released (a barge-in or a closed session): it is never played."""
        self._gate_cancel_timer()
        self._gate_held = []

    def _gate_cut(self, verdict: Verdict) -> None:
        kind = verdict.kind or "leaked tool call"
        logger.warning("spoken artifact cut (%s): %r", kind, self._gate_text.strip())
        self._gate_cut_kind = kind
        for call in verdict.calls:
            self._gate_leak(call)
        self._gate_cancel_timer()
        self._gate_held = []
        self._gate = "cut"
        # Flush only what this response put on the player: a cut of held audio alone
        # leaves the end of the previous reply playing. The console marks the flushed
        # reply as dropped (ReplyTracker.flushed), never unheard.
        if self._gate_released and self._clear_queue:
            self._clear_queue()

    def _gate_transcript(self, text: str, final: bool = False) -> None:
        """Judge the response's transcript so far: release on a reply, cut an artifact, else
        wait. `final` is the whole transcript: a bare tool name is judged only on it.
        """
        if self._gate == "cut" or (self._gate == "pass" and not final):
            return
        self._gate_text = text
        verdict = judge(text, self._gate_tools, final=final)
        if verdict.action == PLAY_AND_LOG:
            for call in verdict.calls:
                self._gate_leak(call)
        elif self._gate == "pass" and verdict.action in (CUT, CUT_AND_EXECUTE):
            # Already heard, so too late to cut: counted as played, never run (10-20 22:59:
            # a bare express{...} left no gate event at all).
            for heard in verdict.calls:
                self._gate_leak(Verdict(PLAY_AND_LOG, kind=heard.kind, tool=heard.tool, shape=heard.shape))
        if self._gate not in ("hold", "open"):
            return
        if verdict.action in (CUT, CUT_AND_EXECUTE):
            self._gate_cut(verdict)
        elif verdict.action != UNDECIDED:
            self._gate_release("pass")

    def _gate_leak(self, verdict: Verdict) -> None:
        """Log one leaked call, report it (source "leaked"), and run it if leak_safe."""
        name = verdict.tool or ""
        if verdict.action == PLAY_AND_LOG:
            logger.warning("possible leaked tool call: %s (%s) played", name, verdict.shape)
            self._report_tool_call(self._leak_report(name, "played"))
            return
        executed = verdict.action == CUT_AND_EXECUTE
        logger.warning("leaked tool call: %s (%s) cut; executed=%s", name, verdict.shape, executed)
        if not executed:
            self._report_tool_call(self._leak_report(name, "cut"))
            return
        task = asyncio.create_task(self._run_leaked_tool(name))
        self._leak_tasks.add(task)
        task.add_done_callback(self._leak_tasks.discard)

    def _leak_report(self, name: str, status: str, ms: float = 0.0, error: str | None = None) -> dict[str, Any]:
        return {
            "name": name,
            "args": {},
            "status": status,
            "error": error,
            "ms": round(ms, 1),
            "result_bytes": 0,
            "background": False,
            "session": self.connection is not None,
            "source": "leaked",
            "turn_id": None,
        }

    async def _run_leaked_tool(self, name: str) -> None:
        """Run a leak_safe tool the model wrote as text, locally: it has no call_id, so
        nothing goes back to the server.
        """
        tool = core_tools.get_tools().get(name)
        if tool is None and name in RETIRED_LEAK_SAFE:
            logger.debug("leaked tool call: %s is retired; cut, not run", name)
            return
        started = time.monotonic()
        status, error = "ok", None
        try:
            if tool is None:
                raise LookupError(f"unknown tool: {name}")
            await tool(self.deps)
        except Exception as exc:  # a report, never a crash
            status, error = "error", f"{type(exc).__name__}: {exc}"
            logger.warning("leaked tool call: %s failed: %s", name, error)
        self._report_tool_call(self._leak_report(name, status, (time.monotonic() - started) * 1000, error))

    def _gate_audio(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        self._gate_had_audio = True
        if self._gate == "cut":
            return
        if self._gate == "hold":
            if not self._gate_held:
                self._gate_held_at = time.monotonic()
                self._gate_cancel_timer()
                self._gate_timer = asyncio.create_task(self._gate_timeout())
            self._gate_held.append(frame)
            return
        self.output_queue.put_nowait(frame)
        if self._gate == "open":
            self._gate_released = True

    async def _gate_timeout(self) -> None:
        await asyncio.sleep(SPOKEN_ARTIFACT_HOLD_S)
        if self._gate == "hold":
            logger.info("reply audio released after %.0f ms with no transcript", SPOKEN_ARTIFACT_HOLD_S * 1000)
            self._gate_timer = None  # this task: not cancelled by the release
            self._gate_release("open")

    # --- reliable search (the companion, 2026-10-09) ------------------------------------

    def _start_user_turn(self) -> None:
        self._turn_guarded = False
        self._turn_filled = False

    def _response_started(self) -> None:
        self._resp_text = ""
        self._resp_called = False
        self._resp_spoke = False

    def _response_spoke(self) -> None:
        self._resp_spoke = True

    def _record_spoken_response(self, event: Any) -> None:
        """Remember a finished response that was heard, for its tools' follow-up."""
        response_id = getattr(getattr(event, "response", None), "id", None)
        # A cut response was never heard: its tools keep their follow-up.
        if self._resp_spoke and self._gate != "cut" and isinstance(response_id, str):
            self._spoken_responses.append(response_id)

    def _spoke_and_only_gestured(self, completed_tool: ToolNotification) -> bool:
        return (
            completed_tool.tool_name in GESTURE_TOOLS
            and completed_tool.error is None
            and completed_tool.turn_id in self._spoken_responses
        )

    def _response_transcript(self, text: str) -> None:
        self._resp_text += " " + (text or "")

    def _response_called_a_tool(self) -> None:
        self._resp_called = True

    async def _response_finished(self, event: Any) -> bool:
        """Nudge a response that promised a lookup and called nothing, once per user turn.

        The missed-call guard. True when it nudged.
        """
        if _companion_tag(event) == FILLER_TAG["companion"]:
            return False  # the app's own line
        text = self._resp_text.strip()
        if self._resp_called or self._turn_guarded or not LOOKUP_PROMISE.search(text):
            return False
        self._turn_guarded = True
        logger.warning("missed tool call: %r", text)
        if not self.connection:
            return False
        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": MISSED_CALL_NOTE}],
                },
            )
        except Exception as e:
            logger.warning("the missed-call note could not be added (%s)", e)
            return False
        await self._safe_response_create()
        return True

    async def _tool_started(self, tool_name: str, call_id: str) -> None:
        """Play a filler while a lookup runs.

        At once for a search, after FILLER_AFTER_S for another lookup tool. Never for a
        tool that is itself the response.
        """
        name = tool_name or ""
        if name.endswith(SEARCH_TOOLS):
            await self._play_filler()
        elif name.endswith(LOOKUP_TOOLS):
            task = asyncio.create_task(self._filler_if_slow(call_id))
            self._filler_tasks.add(task)  # held until done, so it is not collected early
            task.add_done_callback(self._filler_tasks.discard)

    async def _filler_if_slow(self, call_id: str) -> None:
        await asyncio.sleep(FILLER_AFTER_S)
        if call_id in self._in_flight_tool_calls:
            await self._play_filler()

    async def _play_filler(self) -> None:
        """Speak one short line in his voice, once per turn, never over his own audio."""
        if self._turn_filled or self._turn_first_audio_at is not None or not self.connection:
            return
        self._turn_filled = True
        line = random.choice([x for x in FILLER_LINES if x != self._last_filler])
        self._last_filler = line
        await self._safe_response_create(
            response=_out_of_band_request(
                FILLER_TAG,
                ["audio"],
                f"Say exactly these words, briefly, and nothing else: {line}",
                [{"type": "input_text", "text": "(Say the line now.)"}],
            )
        )

    def _log_tool_latency(self, completed: Any) -> None:
        ms, size = _tool_duration_ms_and_bytes(completed)
        # Parsed by the companion's tools/search_stats.py: keep the wording.
        logger.info("tool latency: %s %.0f ms, %d bytes, %s", completed.tool_name, ms, size, completed.status.value)

    def _build_tool_report(self, completed_tool: ToolNotification) -> dict[str, Any]:
        """Build the conversation.tool /rpc payload for one finished (or refused) call."""
        ms, result_bytes = _tool_duration_ms_and_bytes(completed_tool)
        status, reason = _tool_report_status_and_reason(completed_tool)
        return {
            "name": completed_tool.tool_name,
            "args": _cut_long_strings(completed_tool.args),
            "status": status,
            "error": reason,
            "ms": round(ms, 1),
            "result_bytes": result_bytes,
            # Every tool call in this app runs through BackgroundToolManager
            # (core_tools.dispatch_tool_call has no other caller); always true today.
            "background": True,
            "session": self.connection is not None,
            "source": completed_tool.source,
            "turn_id": completed_tool.turn_id,
        }

    def _report_dropped_tool(self, completed_tool: ToolNotification) -> None:
        """Report a result from a closed session as cancelled; nothing goes to the model."""
        payload = self._build_tool_report(completed_tool)
        payload["status"] = "cancelled"
        payload["session"] = False
        self._report_tool_call(payload)

    def _report_tool_call(self, payload: dict[str, Any]) -> None:
        """Queue one tool-call report and flush it; never blocks or slows a tool call."""
        queue = self._tool_report_queue
        if len(queue) >= (queue.maxlen or TOOL_REPORT_QUEUE_MAX):
            self._tool_reports_dropped += 1
            logger.warning(
                "tool report queue full (%d); dropping the oldest (%d dropped so far)",
                queue.maxlen,
                self._tool_reports_dropped,
            )
        queue.append(payload)
        self._flush_tool_reports()

    def _flush_tool_reports(self) -> None:
        """Send queued reports oldest-first; stop at the first one not delivered.

        The observer (console.py's _dispatch_tool_call) returns whether a /rpc
        peer could receive it. A report is popped only once delivered, so a
        report is never sent twice, and a down supervisor's backlog is retried
        -- in order -- on the next tool call.
        """
        observer = self._tool_call_observer
        if observer is None:
            return
        while self._tool_report_queue:
            item = self._tool_report_queue[0]
            try:
                delivered = observer(item)
            except Exception:
                logger.debug("tool-call observer raised; will retry later", exc_info=True)
                return
            if not delivered:
                return
            self._tool_report_queue.popleft()

    async def say(self, text: str) -> None:
        """Have the model voice ``text`` as its own reply.

        Queues the text with its ``response.create`` through the serial sender, which
        adds it as a user message item just before that create: at once with nothing
        else queued, after a running tool's follow-up reply otherwise. Not verbatim
        TTS (speech-to-speech may rephrase). Raises if the session is closed.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self.connection:
            raise RuntimeError("say: no active session")
        if not self._response_done_event.is_set() and not self._followup_live:
            # The companion's audio audit, A5: a live response would otherwise keep
            # streaming ahead of this say (an announcement behind an old reply).
            # A tool's follow-up is not cancelled: it carries the tool's answer.
            try:
                await self.connection.response.cancel()
            except Exception as e:
                logger.debug("say: could not cancel the live response (%s)", e)
        self._mark_activity("say")
        await self._safe_response_create(**{_ANNOUNCEMENT: text})

    async def end_user_turn(self) -> None:
        """End a half-heard user turn: the microphone was muted or stopped mid-utterance.

        The companion's audio audit (2026-10-09, A4): no speech_stopped would ever
        come, so _user_speaking stayed set and the listening pose stayed held.
        """
        self._user_speaking = False
        try:
            self.deps.movement_manager.set_listening(False)
        except Exception:
            logger.debug("end_user_turn: could not release the listening pose", exc_info=True)
        # No input_audio_buffer.clear: the hosted backend answers it "Unknown or invalid
        # event" (2026-10-09, 11:22), and the muted microphone sends nothing more anyway.

    async def type_text(self, text: str) -> None:
        """Add ``text`` as the person's own words and answer it as speech would be.

        Type to Reachy (the companion, 2026-10-08): tools included, and reported to
        the transcript observer as a user turn so the companion's memory keeps it.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("type_text: empty text")
        if not self.connection:
            raise RuntimeError("type_text: no active session")
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )
        self._mark_activity("typed")
        self._emit_transcript("user", text, True)
        await self._safe_response_create()

    async def add_image(self, b64_jpeg: str) -> None:
        """Add a camera frame to the conversation with no response of its own.

        The companion's glance (2026-10-06): the hosted backend keeps an image item
        until the next response, which sees it, and strips it after that response.
        """
        if not self.connection:
            raise RuntimeError("add_image: no active session")
        self._count_image(b64_jpeg)
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": GLANCE_CONTEXT_NOTE},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_jpeg}"},
                ],
            },
        )

    def note_image(self, b64_jpeg: str, source: str) -> None:
        """Hand a frame just added to the conversation to the companion, for its sight note.

        `source`: "glance" or "camera". The supervisor writes the note with Claude Haiku,
        off the voice service (the cold-open brief, 2026-10-17); nothing here waits on it.
        """
        sink = self.on_sight
        if sink is None:
            return
        try:
            sink(b64_jpeg, source)
        except Exception:
            logger.warning("A frame was not handed over for its sight note", exc_info=True)

    def _log_usage(self, usage: Any) -> None:
        line = usage_line(usage, images=self._session_images, image_bytes=self._session_image_bytes)
        if line is not None:
            logger.info(line)
        elif not self._usage_missing_logged:
            self._usage_missing_logged = True
            logger.info("Context: the server reported no token usage")

    def _count_image(self, b64_jpeg: str) -> None:
        self._session_images += 1
        self._session_image_bytes += (len(b64_jpeg) * 3) // 4

    def open_glance_in(self) -> None:
        """Mark the glance at a session's open as in, and judge it against the first reply."""
        now = time.monotonic()
        self._open_glance_at = now
        if self._first_reply_at is not None:
            logger.warning("First reply created %.1f s before the glance", now - self._first_reply_at)

    def _first_reply_created(self) -> None:
        now = time.monotonic()
        self._first_reply_at = now
        if self._open_glance_at is not None:
            logger.info("Glance in %.1f s before the first reply", now - self._open_glance_at)

    async def _wait_notes(self) -> None:
        """Wait for the pending companion notes to be queued or dropped (the tests)."""
        if self._note_tasks:
            await asyncio.gather(*list(self._note_tasks), return_exceptions=True)

    def add_note(self, text: str, now: bool = False) -> "asyncio.Task[Any]":
        """Add a companion note at the next quiet moment, with no response of its own.

        Who has joined, who is no longer in view. A user-role item: system items
        were unreliable on the hosted backend (2026-10-09). Never reaches the
        transcript observer.

        With now=True the note is sent at once, even while the person is speaking. It is
        still refused while a response is active or queued, and then dropped, never held.
        The task's result is then True once the note was sent and False if it was dropped
        (the reason is in note_drop_reason).
        """
        if not self.connection:
            raise RuntimeError("add_note: no active session")
        # A note belongs to the session it was asked for (fix round 1, the review):
        # recorded here, and checked again after the quiet wait.
        session = self.connection
        if now:
            coro = self._note_text_now(text, session)
        else:
            coro = self._note_text_when_quiet(text, session)
        task = asyncio.create_task(coro, name="companion-note")
        self._note_tasks.add(task)
        task.add_done_callback(self._note_tasks.discard)
        return task

    async def _note_text_now(self, text: str, session: "AsyncRealtimeConnection") -> bool:
        """Send a note now, even mid-speech; drop it if a response is in the way.

        True if the note was sent, False if it was dropped.
        """
        # Not airtight: _response_done_event is cleared on response.created, so between a
        # response.create being sent and response.created arriving this check still passes.
        if not self._response_done_event.is_set() or not self._pending_responses.empty():
            return self._note_dropped("response active")
        return await self._send_note(text, session)

    def _note_dropped(self, reason: str) -> bool:
        self.note_drop_reason = reason
        logger.info("Companion note dropped: %s", reason)
        return False

    async def _note_text_when_quiet(self, text: str, session: "AsyncRealtimeConnection") -> None:
        started = time.monotonic()
        while True:
            now = time.monotonic()
            quiet = not self._user_speaking and now - self._last_speech_at >= NOTE_QUIET_SECONDS
            if quiet and self._response_done_event.is_set():
                break
            if now - started >= COMPANION_NOTE_WAIT_S:
                logger.info("Companion note dropped: no quiet moment in %.0f s", COMPANION_NOTE_WAIT_S)
                return
            await asyncio.sleep(_NOTE_POLL_SECONDS)
        await self._send_note(text, session)

    async def _send_note(self, text: str, session: "AsyncRealtimeConnection") -> bool:
        if not self.connection:
            return self._note_dropped("session closed")
        if self.connection is not session:
            return self._note_dropped("session changed")
        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": note_item_text(text)}],
                },
            )
        except Exception as exc:
            return self._note_dropped(str(exc))
        return True

    def open_settled(self) -> None:
        """Start the greeting's hold: the console has finished this open, its pre-roll sent."""
        self._open_settled.set()

    async def _greet_into_silence(self) -> None:
        """Greet once the open has settled and the person has stayed quiet.

        A wake's pre-roll carries the name and often the request; the server hears it
        and answers, so a greeting then would talk over the person or queue a second
        reply. Skipped silently: his body already shows he is listening.
        """
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._open_settled.wait(), GREETING_SETTLE_WAIT_S)
        await asyncio.sleep(GREETING_HOLD_S)
        if self._spoke_since_open or not self._response_done_event.is_set():
            self.greet_at_open = False
            logger.info("Startup greeting skipped: the person has started speaking")
            return
        await self._send_startup_greeting_prompt()

    async def _send_startup_greeting_prompt(self) -> None:
        """Prompt the model to open the conversation, once, for an open that asked."""
        if not self.greet_at_open or not self.connection:
            return
        self.greet_at_open = False
        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            return

        self._mark_activity("startup_greeting_prompt")
        await self._safe_response_create(
            response=_out_of_band_request(
                GREETING_TAG, ["audio"], greeting_prompt, [{"type": "input_text", "text": "(Say the line now.)"}]
            )
        )
        logger.info("Queued startup greeting prompt")

    async def _cancel_greeting(self) -> None:
        """Stop the greeting's response at once: the person has started speaking.

        As a barge-in does: drop its held and queued audio, and cancel it on the server.
        """
        self._greeting_live = False
        logger.info("Startup greeting cancelled: the person started speaking")
        self._gate_drop_held()
        if self._clear_queue:
            self._clear_queue()
        if self.connection:
            try:
                await self.connection.response.cancel()
            except Exception as e:
                logger.debug("could not cancel the greeting's response (%s)", e)

    async def _response_sender_loop(self) -> None:
        """Dedicated worker that sends ``response.create()`` calls serially.

        This logic was designed to comply with the response.create() docstring specification for event ordering:
        https://github.com/openai/openai-python/blob/3e0c05b84a2056870abf3bd6a5e7849020209cc3/src/openai/resources/realtime/realtime.py#L649C1-L651C30

        For each queued request the worker:
        1. Waits until no response is active (_response_done_event).
        2. Sends response.create().
        3. Waits until the receiver observes response.created or a rejection.
        4. Waits for the response cycle to complete (response.done).
        5. If the server rejected with active_response, retries from step 1.
        """
        while self.connection:
            try:
                kwargs = await self._pending_responses.get()
            except asyncio.CancelledError:
                return

            # Parallel tool calls enqueue duplicate empty requests; coalesce to one.
            # An announcement is never merged: it keeps its place in the queue.
            if not kwargs:
                self._requeue(lambda queued: bool(queued.get(_ANNOUNCEMENT)))
            announcement: str | None = kwargs.pop(_ANNOUNCEMENT, None)
            announcement_added = False

            sent = False
            max_retries = 5
            attempts = 0
            while not sent and self.connection and attempts < max_retries:
                if await self._wait_for_owed_outputs(kwargs) and not kwargs:
                    # Released behind a tool: the tool's plain follow-up, queued just
                    # before the release, merges with a plain create. An announcement
                    # lets that follow-up answer first, then goes as its own reply.
                    if self._requeue(bool) and announcement:
                        logger.info("Announcement waits for the tool's follow-up reply")
                        self._requeue(bool, front={_ANNOUNCEMENT: announcement})
                        announcement = None
                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.info("Timed out waiting for previous response to finish; forcing ahead")
                    self._response_done_event.set()

                if not self.connection:
                    break

                if announcement and not announcement_added:
                    try:
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": announcement}],
                            },
                        )
                    except Exception as e:
                        logger.warning("Announcement dropped: its item could not be added (%s)", e)
                        break
                    announcement_added = True

                greeting = (kwargs.get("response") or {}).get("metadata") == GREETING_TAG
                if greeting:
                    if self._spoke_since_open:
                        logger.info("Startup greeting dropped at send: the person has started speaking")
                        break
                    self._greeting_live = True
                self._last_response_rejected = False
                self._response_started_or_rejected_event.clear()
                try:
                    await self.connection.response.create(**kwargs)
                    self._session_creates += 1
                    # 2026-10-21 16:51: a tool's follow-up after a say was never answered, and
                    # these steps logged only at debug. One line per create sent.
                    if kwargs.get("response", {}).get("conversation") == "none":
                        kind = "filler"  # out of band: never a follow-up (the review of eea54bb)
                    elif announcement_added:
                        kind = "announcement"
                    else:
                        kind = "follow-up" if self._followup_pending else "plain"
                    logger.info("response.create sent (%s, attempt %d)", kind, attempts + 1)
                except Exception as e:
                    logger.info("_response_sender_loop: send failed: %s", e)
                    self._response_done_event.set()
                    break

                try:
                    await asyncio.wait_for(
                        self._response_started_or_rejected_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.info("Timed out waiting for response.created or response rejection")

                # Check if the receiver loop observed an asynchronous rejection.
                if self._last_response_rejected:
                    attempts += 1
                    if attempts >= max_retries:
                        logger.info("response.create rejected %d times; giving up", attempts)
                        break
                    logger.info("response.create was rejected; retrying (%d/%d)", attempts, max_retries)
                    await asyncio.sleep(_RESPONSE_REJECTION_RETRY_DELAY)
                    continue

                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.info("Timed out waiting for response.done; assuming response completed")
                    self._response_done_event.set()
                    break

                sent = True
            if (kwargs.get("response") or {}).get("metadata") == GREETING_TAG:
                self._greeting_live = False  # done, cancelled, rejected or never sent

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        self._log_tool_latency(completed_tool)
        if completed_tool.tool_name.endswith(SEARCH_TOOLS):
            self._search_answer_pending = True  # its answer's first audio is logged
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed with error: %s",
                completed_tool.tool_name,
                completed_tool.id,
                completed_tool.error,
            )
            tool_result = {"error": completed_tool.error}
            tool_result_for_model = tool_result
        elif completed_tool.result is not None:
            tool_result = completed_tool.result
            tool_result_for_model = (
                self._sanitize_tool_result_for_model(completed_tool.tool_name, tool_result)
                if isinstance(tool_result, dict)
                else tool_result
            )
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                completed_tool.tool_name,
                completed_tool.id,
            )
            logger.debug("Tool '%s' model-visible result: %s", completed_tool.tool_name, tool_result_for_model)
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        # The tool-call-log brief (2026-10-11): built once, here, so `session`
        # reflects whether a conversation was open when the call finished, and
        # sent exactly once on every exit below (including a closed connection).
        report_payload = self._build_tool_report(completed_tool)

        # Connection may have closed while tool was running
        if not self.connection:
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                completed_tool.tool_name,
                completed_tool.id,
            )
            self._report_tool_call(report_payload)
            return

        try:
            send_result_to_model = not completed_tool.is_idle_tool_call
            if send_result_to_model:
                self._mark_activity("tool_result_ready")
            model_result_submitted = False
            if send_result_to_model and isinstance(completed_tool.id, str):
                if not await self._wait_for_response_done_before_tool_result():
                    send_result_to_model = False
                if not send_result_to_model:
                    logger.warning(
                        "Dropping realtime model result for tool '%s' (id=%s) because response.done was not observed",
                        completed_tool.tool_name,
                        completed_tool.id,
                    )
                elif not self.connection:
                    logger.warning(
                        "Connection closed before sending tool '%s' (id=%s) result back",
                        completed_tool.tool_name,
                        completed_tool.id,
                    )
                    return
                else:
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": completed_tool.id,
                            "output": json.dumps(tool_result_for_model),
                        },
                    )
                    model_result_submitted = True

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result_for_model),
                    },
                ),
            )

            sight_note: str | None = None
            if model_result_submitted and completed_tool.tool_name == "camera" and "b64_im" in tool_result:
                # use raw base64, don't json.dumps (which adds quotes)
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                sight_note = b64_im
                image_width = tool_result.get("image_width")
                image_height = tool_result.get("image_height")
                jpeg_bytes_value = tool_result.get("jpeg_bytes")
                jpeg_bytes = jpeg_bytes_value if isinstance(jpeg_bytes_value, int) else (len(b64_im) * 3) // 4
                self._count_image(b64_im)
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_im}",
                            },
                        ],
                    },
                )
                if isinstance(image_width, int) and isinstance(image_height, int):
                    logger.info(
                        "Added camera image to conversation frame=%sx%s jpeg_bytes=%s",
                        image_width,
                        image_height,
                        jpeg_bytes,
                    )
                else:
                    logger.info(
                        "Added camera image to conversation jpeg_bytes=%s",
                        jpeg_bytes,
                    )

            if isinstance(completed_tool.id, str):
                self._in_flight_tool_calls.discard(completed_tool.id)

            tool = core_tools.get_tools().get(completed_tool.tool_name)
            # Always surface errors, skip the spoken follow-up for tools that opt out.
            if model_result_submitted and (completed_tool.error is not None or tool is None or tool.needs_response):
                self._tool_batch_needs_response = True
                if not self._spoke_and_only_gestured(completed_tool):
                    self._tool_batch_only_gestured = False

            # Parallel tool calls in one turn: respond once every result is in, not per tool.
            if self._tool_batch_needs_response and not self._in_flight_tool_calls:
                self._tool_batch_needs_response = False
                if self._tool_batch_only_gestured:
                    # 2026-10-23: the greeting spoke and called express; its follow-up, with
                    # nothing left to say, read the context aloud. A skipped follow-up no longer
                    # adds a stored assistant reply, which the express trial counts.
                    logger.info(
                        "tool follow-up skipped: the response already spoke and only gestured (%s)",
                        completed_tool.tool_name,
                    )
                else:
                    self._followup_pending = True
                    logger.info("follow-up reply queued after %s", completed_tool.tool_name)
                    await self._safe_response_create()
                self._tool_batch_only_gestured = True
            if sight_note is not None:
                self.note_image(sight_note, "camera")

        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._response_done_event.set()
        finally:
            if isinstance(completed_tool.id, str):
                # After the follow-up create is queued, so a held create merges with it.
                # A dropped or failed output is settled too: nothing more will be sent.
                self._output_settled(completed_tool.id)
            # Sent after the result reaches the model (or the attempt fails):
            # the report itself is in-memory only and never blocks a tool call.
            self._report_tool_call(report_payload)

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        tool_specs = get_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
        self._gate_tools = gate_tool_names(tool_specs, core_tools.get_tools())
        connect_kwargs: dict[str, Any] = {}
        if self._realtime_connect_query:
            connect_kwargs["extra_query"] = self._realtime_connect_query
        # 2026-10-16: the server echoes image data URLs back in conversation.item.created;
        # the websockets default of 1 MiB closed the session with 1009. Upstream's value.
        connect_kwargs["websocket_connection_options"] = {"max_size": REALTIME_MAX_MESSAGE_BYTES}
        async with self.client.realtime.connect(**connect_kwargs) as conn:
            try:
                session_config = self._get_session_config(tool_specs)
                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                raise

            logger.info("Realtime session updated successfully")
            logger.info("Realtime session size: %s", session_size(session_config))

            # Reset the partial-transcript accumulator for each new session
            self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

            # Manage events received from the realtime server. The greeting's hold is
            # this open's: reset before the console can see the connection.
            self._spoke_since_open = self._greeting_live = False
            self._open_settled.clear()
            self.connection = conn
            self._reset_owed_outputs()
            self._followup_pending = self._followup_live = False
            # Session-state hygiene: the last session's batch is over. Its cancelled tools'
            # results are dropped by the tool manager, never handled here.
            self._in_flight_tool_calls.clear()
            self._tool_batch_needs_response = False
            self._tool_batch_only_gestured = True
            # The open counts as speech: the first note waits for a quiet moment.
            self._user_speaking = False
            self._last_speech_at = time.monotonic()
            self._session_images = self._session_image_bytes = self._session_creates = 0
            self._usage_missing_logged = False  # each session says once whether usage comes
            self._open_glance_at = self._first_reply_at = None
            # A turn left open by the session before is not this session's first turn
            # (the review: its greeting would be judged as the first reply).
            self._turn_user_done_at = self._turn_response_created_at = None
            try:
                self._connected_event.set()
            except Exception:
                pass

            response_sender_task: asyncio.Task[None] | None = None
            greeting_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(
                    tool_callbacks=[self._handle_tool_result], dropped_callbacks=[self._report_dropped_tool]
                )

                # Start the response sender worker
                response_sender_task = asyncio.create_task(self._response_sender_loop(), name="response-sender")
                if self.greet_at_open:
                    greeting_task = asyncio.create_task(self._greet_into_silence(), name="startup-greeting")

                async for event in self.connection:
                    logger.debug("Realtime event: %s", event.type)
                    if event.type == "input_audio_buffer.speech_started":
                        self._spoke_since_open = True
                        self._new_user_turn()
                        self._user_speaking = True
                        self._last_speech_at = time.monotonic()
                        self._mark_activity("user_speech_started")
                        self._turn_user_done_at = None
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        self._gate_drop_held()
                        if self._clear_queue:
                            self._clear_queue()
                        if self._greeting_live:
                            await self._cancel_greeting()
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self._user_speaking = False
                        self._last_speech_at = time.monotonic()
                        self._mark_activity("user_speech_stopped")
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type == "response.output_audio.done":
                        self.deps.movement_manager.set_speaking(False)
                        logger.debug("response completed")

                    if event.type == "response.output_text.delta":
                        self._response_spoke()
                        logger.debug("response text delta")

                    if event.type == "response.output_text.done":
                        logger.debug("response text done: %s", event.text)

                    if event.type == "response.created":
                        self._response_done_event.clear()
                        self._response_started_or_rejected_event.set()
                        if self._followup_pending and _companion_tag(event) is None:
                            self._followup_pending, self._followup_live = False, True
                        self._response_started()
                        self._gate_start()
                        if _companion_tag(event) == GREETING_TAG["companion"] and self._spoke_since_open:
                            await self._cancel_greeting()  # speech came between its create and now
                        self._mark_activity("response_created")
                        self.deps.movement_manager.set_speaking(True)
                        if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                            self._turn_response_created_at = time.perf_counter()
                            delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)
                            if self._first_reply_at is None:
                                self._first_reply_created()
                        logger.info("Response created (active)")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        # Resume tracking for responses that emit no audio (text-only / tool-only).
                        self.deps.movement_manager.set_speaking(False)
                        self._response_done_event.set()
                        self._response_started_or_rejected_event.set()
                        if _companion_tag(event) is None:
                            self._followup_live = False
                        elif _companion_tag(event) == GREETING_TAG["companion"]:
                            self._greeting_live = False
                        # Checked against the function call's response_id on the robot (2026-10-23).
                        logger.debug("Response done: id=%s", getattr(getattr(event, "response", None), "id", None))
                        self._log_usage(getattr(getattr(event, "response", None), "usage", None))
                        if not self._gate_transcript_done and (self._gate_text or self._gate_had_audio):
                            # Parsed by counting: keep the wording. Measuring only; the
                            # fallback below is unchanged.
                            logger.warning(
                                "output gate: reply judged at response.done with no transcript.done "
                                "(%d chars, audio=%s, gate=%s)",
                                len(self._gate_text),
                                self._gate_had_audio,
                                self._gate,
                            )
                        # With no transcript.done this is the whole reply: judged as final (the review).
                        self._gate_transcript(self._gate_text, final=not self._gate_transcript_done)
                        if self._gate == "hold":  # no transcript at all: a reply, not an artifact
                            self._gate_release("pass")
                        # Before the first await: a finished tool waits on this response's done.
                        self._record_spoken_response(event)
                        await self._response_finished(event)

                    if event.type == "conversation.item.input_audio_transcription.delta":
                        self._mark_activity("user_transcription_delta")
                        logger.debug(f"User partial transcript: {event.delta}")

                        item_id = event.item_id
                        delta = event.delta or ""

                        input_transcript = self.input_transcript_chunks_by_item
                        self._record_partial_transcript_delta(input_transcript, item_id, delta)

                        current_partial = "".join(input_transcript.deltas)
                        sequence_counter = len(input_transcript.deltas) - 1

                        await self._cancel_partial_transcript_task()

                        # Start new debounce timer with the last delta
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(current_partial, item_id, sequence_counter)
                        )

                    # A failed transcription has no completion. The companion pairs each user
                    # speech window with its item's completion, so it hears of this one too.
                    if event.type == "conversation.item.input_audio_transcription.failed":
                        self._mark_activity("user_transcription_failed")
                        logger.warning("User transcription failed for item %s", getattr(event, "item_id", None))

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        self._mark_activity("user_transcription_completed")
                        raw_transcript = event.transcript or ""
                        transcript = raw_transcript.strip()
                        logger.debug("User transcript: %s", raw_transcript)
                        self.deps.movement_manager.set_listening(False)

                        await self._cancel_partial_transcript_task()

                        if not transcript:
                            logger.debug("Ignoring empty user transcript")
                            continue

                        self._turn_user_done_at = time.perf_counter()
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        self._in_flight_tool_calls.clear()
                        self._tool_batch_needs_response = False
                        self._tool_batch_only_gestured = True
                        self._start_user_turn()

                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                        self._emit_transcript("user", transcript, True)

                    # Handle assistant transcription
                    if event.type == "response.output_audio_transcript.delta":
                        if getattr(event, "delta", ""):
                            self._response_spoke()
                        self._gate_transcript(self._gate_text + (getattr(event, "delta", "") or ""))

                    if event.type == "response.output_audio_transcript.done":
                        self._mark_activity("assistant_transcript_done")
                        self._gate_transcript_done = True
                        self._gate_transcript(event.transcript or "", final=True)
                        logger.debug(f"Assistant transcript: {event.transcript}")
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event.transcript})
                        )
                        said = event.transcript or ""
                        kept = said if self._gate_cut_kind else without_calls(said, self._gate_tools)  # 2026-10-20
                        if kept or not said:
                            self._emit_transcript("assistant", kept, True, artifact=self._gate_cut_kind)
                        else:  # played, but only markup: logged by the supervisor, never kept (the review)
                            self._emit_transcript("assistant", said, True, artifact="leaked tool call")
                        self._response_transcript(event.transcript or "")

                    # Handle audio delta
                    if event.type == "response.output_audio.delta":
                        decoded_pcm_bytes = base64.b64decode(event.delta)
                        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
                        self._mark_activity("assistant_audio_delta")
                        self._response_spoke()
                        if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
                            self._turn_first_audio_at = time.perf_counter()
                            delta_ms = (self._turn_first_audio_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: first audio delta %.0f ms after user transcript", delta_ms)
                        if self._search_answer_pending and self._turn_user_done_at is not None:
                            self._search_answer_pending = False
                            logger.info(
                                "search answer: first audio %.0f ms after the user's transcript",
                                (time.perf_counter() - self._turn_user_done_at) * 1000,
                            )
                        self._gate_audio((self.SAMPLE_RATE, decoded_pcm))
                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        self._mark_activity("tool_call_received")
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, args=%s",
                            tool_name,
                            call_id,
                            args_json_str,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name,
                                type(tool_name).__name__,
                                args_json_str,
                                type(args_json_str).__name__,
                                call_id,
                            )
                            continue

                        self._in_flight_tool_calls.add(call_id)
                        self._owe_output(call_id)
                        self._response_called_a_tool()
                        # The tool-call-log brief: who asked -- the missed-call guard's
                        # nudge, still in force for the rest of this user turn, or an
                        # ordinary turn. is_idle_tool_call (never True here) covers the
                        # third case, the model unprompted via the idle policy.
                        source = "missed_call_guard" if self._turn_guarded else "turn"
                        background_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=args_json_str,
                                deps=self.deps,
                            ),
                            is_idle_tool_call=False,
                            source=source,
                            turn_id=getattr(event, "response_id", None),
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. The tool is now running. Tool ID: {background_tool.tool_id}",
                                },
                            ),
                        )
                        logger.info(
                            "Started background tool: %s (id=%s, call_id=%s)",
                            tool_name,
                            background_tool.tool_id,
                            call_id,
                        )
                        await self._tool_started(tool_name, call_id)

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", str(err) if err else "unknown error")
                        code = getattr(err, "code", "") or getattr(err, "type", "")

                        if code == "conversation_already_has_active_response":
                            # response.create was rejected.  The sender worker
                            # is waiting on _response_done_event; when the active
                            # response finishes it will wake up and see this flag.
                            self._last_response_rejected = True
                            self._response_started_or_rejected_event.set()
                            logger.info("response.create rejected; worker will retry after active response finishes")
                        else:
                            self._response_started_or_rejected_event.set()
                            logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)
                            if "function call outputs are pending" in str(msg):
                                await self._recover_refused_reply()

                        if code == "input_audio_buffer_commit_empty":
                            self.deps.movement_manager.set_listening(False)

                        # Only show user-facing errors, not internal state errors.
                        if code not in (
                            "input_audio_buffer_commit_empty",
                            "conversation_already_has_active_response",
                        ) and "cancel" not in str(code):  # a say's cancel that came late
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                            )
            finally:
                logger.info(
                    "Session to the voice service: %d images, %d KB, %d response.create",
                    self._session_images,
                    self._session_image_bytes // 1024,
                    self._session_creates,
                )
                # Audio still held belongs to a response this session will not finish.
                self._gate_drop_held()
                self._gate = "pass"

                if greeting_task is not None:
                    greeting_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await greeting_task

                # Stop the response sender worker.
                if response_sender_task is not None:
                    response_sender_task.cancel()
                    try:
                        await response_sender_task
                    except asyncio.CancelledError:
                        pass

                # Stop background tool manager tasks (listener + cleanup) in all paths.
                await self.tool_manager.shutdown()

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the realtime server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for the realtime API.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        _, audio_frame = frame
        if audio_frame.size == 0:
            return

        # Reshape if needed
        if audio_frame.ndim == 2:
            # channels-last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to the realtime input buffer (guard against races during reconnect).
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # A session that closes between speech_started and speech_stopped never hears the
        # stop: release the listening pose here, or breathing (and every idle body turn,
        # which waits for it) stays off with no session open (the companion, 2026-10-19).
        await self.end_user_turn()
        # Unblock the response sender worker so it can exit
        self._response_done_event.set()

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()

        await self._cancel_partial_transcript_task()

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_available_voices(self) -> list[str]:
        """Return the available Hugging Face voices."""
        return get_available_voices()

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the Hugging Face OpenAI-compatible realtime client."""
        configured_bearer_token = (config.HF_TOKEN or "").strip()
        connection_selection = get_hf_connection_selection()
        direct_realtime_url = get_hf_direct_ws_url()
        if connection_selection.mode == HF_LOCAL_CONNECTION_MODE:
            if not direct_realtime_url:
                raise RuntimeError("HF_REALTIME_WS_URL must be set when HF_REALTIME_CONNECTION_MODE=local")
            client, connect_query = _build_openai_compatible_client_from_realtime_url(
                direct_realtime_url,
                configured_bearer_token,
            )
            self._realtime_connect_query = connect_query
            logger.info("Using direct Hugging Face realtime endpoint %s", direct_realtime_url)
            return client

        session_url = connection_selection.session_url
        if not session_url:
            raise RuntimeError("Built-in Hugging Face session proxy URL is unavailable")
        if direct_realtime_url:
            logger.info("HF_REALTIME_CONNECTION_MODE=deployed; ignoring HF_REALTIME_WS_URL.")

        bearer_token = configured_bearer_token or (get_token() or "").strip()
        allocator_headers = {"User-Agent": "reachy-mini-conversation-app"}
        if bearer_token:
            allocator_headers["X-Reachy-Mini-Authorization"] = f"Bearer {bearer_token}"
        allocator_payload: dict[str, str] = {}
        try:
            hardware_id = self.deps.reachy_mini.client.get_status(wait=False).hardware_id
        except (AssertionError, ConnectionError, TimeoutError) as e:
            logger.warning("Daemon status unavailable for realtime session allocation: %s", e)
        else:
            if hardware_id:
                allocator_payload["hardware_id"] = hardware_id

        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(session_url, headers=allocator_headers, json=allocator_payload)
            response.raise_for_status()
            payload = response.json()

        connect_url = payload.get("connect_url")
        if not isinstance(connect_url, str) or not connect_url:
            raise RuntimeError(f"Session allocator response did not contain a valid connect_url: {payload!r}")

        parsed_connect_url = parse_hf_realtime_url(connect_url)
        if not parsed_connect_url.has_realtime_path:
            raise ValueError(f"Expected realtime connect URL ending with /realtime, got: {connect_url}")

        logger.info("Allocated realtime session %s", payload.get("session_id") or "<unknown>")
        client, connect_query = _build_openai_compatible_client_from_realtime_url(
            connect_url,
            bearer_token,
        )
        self._realtime_connect_query = connect_query
        return client
