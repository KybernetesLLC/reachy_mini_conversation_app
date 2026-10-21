import json
import time
import base64
import asyncio
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosedError

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


class _FakeEvent:
    """A minimal realtime event: a `type` plus arbitrary attributes."""

    def __init__(self, event_type: str, **fields: Any) -> None:
        """Store the event type and any extra attributes."""
        self.type = event_type
        self.__dict__.update(fields)


def _make_fake_realtime_client(
    *,
    events: tuple[Any, ...] = (),
    captured_update: dict[str, Any] | None = None,
    captured_connect: dict[str, Any] | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if captured_update is not None:
                captured_update.update(kwargs)

    class FakeNoop:
        async def append(self, **_kw: Any) -> None:
            pass

        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

        def __init__(self) -> None:
            self._events = iter(events)

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> _FakeEvent:
            while True:
                try:
                    item = next(self._events)
                except StopIteration:
                    raise StopAsyncIteration
                if callable(item):  # a probe or pause between events: run it, yield nothing
                    result = item()
                    if asyncio.iscoroutine(result):
                        await result
                    continue
                return item

    class FakeRealtime:
        def connect(self, **kwargs: Any) -> FakeConn:
            if captured_connect is not None:
                captured_connect.update(kwargs)
            return FakeConn()

    class FakeClient:
        realtime = FakeRealtime()

    return FakeClient()


def _fake_openai_client(captured_kwargs: dict[str, Any]) -> type:
    """Return a fake AsyncOpenAI class that records its constructor kwargs."""

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    return FakeClient


def _fake_allocator(
    connect_url: str,
    posts: list[tuple[str, dict[str, str] | None, dict[str, str] | None]],
) -> type:
    """Return a fake httpx.AsyncClient that records allocator requests."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"session_id": "session-123", "connect_url": connect_url}

    class FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, str] | None = None,
        ) -> FakeResponse:
            posts.append((url, headers, json))
            return FakeResponse()

    return FakeAsyncClient


def _plain_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def _session_handler(monkeypatch: Any, events: tuple[Any, ...]) -> HuggingFaceRealtimeHandler:
    """Build a handler wired to a fake realtime client yielding `events`, ready to run a session."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    handler = _plain_handler()
    handler.client = _make_fake_realtime_client(events=events)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    return handler


def _drain(handler: HuggingFaceRealtimeHandler) -> list[Any]:
    items: list[Any] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def _messages(items: list[Any]) -> list[dict[str, Any]]:
    return [item.args[0] for item in items if isinstance(item, AdditionalOutputs)]


def _capture_tool_reports(handler: HuggingFaceRealtimeHandler) -> list[dict[str, Any]]:
    """Attach an always-delivers observer to `handler` and return the list it appends to."""
    reports: list[dict[str, Any]] = []

    def _observer(payload: dict[str, Any]) -> bool:
        reports.append(payload)
        return True

    handler.set_tool_call_observer(_observer)
    return reports


def _report_payload(name: str) -> dict[str, Any]:
    return {"name": name}


@pytest.mark.asyncio
async def test_partial_transcription_uses_latest_snapshot(monkeypatch: Any) -> None:
    """Partial transcription snapshots should replace older snapshots for the same item."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey"),
            _FakeEvent(
                "conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey, how are you?"
            ),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert handler.input_transcript_chunks_by_item.item_id == "item-1"
    assert handler.input_transcript_chunks_by_item.deltas == ["Hey, how are you?"]


@pytest.mark.asyncio
async def test_emit_skips_idle_signal_while_response_active(monkeypatch: Any) -> None:
    """Idle tools should not trigger while a response is still active."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = HuggingFaceRealtimeHandler(deps)
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler._response_done_event.clear()

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_parallel_tool_calls_trigger_single_response(monkeypatch: Any) -> None:
    """Parallel tool calls in one turn should yield one response, not one per completed tool."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    handler._in_flight_tool_calls = {"call_a", "call_b"}

    def _completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="test__parallel_probe",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    await handler._handle_tool_result(_completed("call_a"))
    assert create.await_count == 0

    await handler._handle_tool_result(_completed("call_b"))
    assert create.await_count == 1


def test_handler_uses_hf_startup_voice_at_startup(monkeypatch: Any) -> None:
    """Hugging Face startup should restore persisted HF voices."""
    handler = HuggingFaceRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Aiden",
    )

    assert handler.get_current_voice() == "Aiden"


def test_handler_ignores_unsupported_hf_profile_voice(monkeypatch: Any) -> None:
    """Unsupported profile voices should not be sent to the Hugging Face backend."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "cedar")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == HF_DEFAULT_VOICE
    session = handler._get_session_config([])
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_handler_normalizes_hf_voice_case(monkeypatch: Any) -> None:
    """Lowercase Hugging Face speaker names should resolve to the curated UI value."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "serena")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_run_realtime_session_uses_default_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Use the backend default speaker when no profile voice is selected for the hf LB."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_update=captured_update)

    await handler._run_realtime_session()

    session = captured_update["session"]
    # HF at 16 kHz passes None so the backend uses its optimal default (16 kHz).
    assert session["audio"]["input"]["format"]["rate"] is None
    assert session["audio"]["output"]["format"]["rate"] is None
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_huggingface_session_uses_configured_transcription_language(monkeypatch: Any) -> None:
    """Hugging Face realtime sessions should forward the configured transcription language."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "zh")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    session = handler._get_session_config([])

    assert session["audio"]["input"]["transcription"]["language"] == "zh"


@pytest.mark.asyncio
async def test_run_realtime_session_passes_allocated_session_query(monkeypatch: Any) -> None:
    """Hugging Face sessions must forward the allocated session token to the websocket connect call."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    captured_connect: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_connect=captured_connect)
    handler._realtime_connect_query = {"session_token": "abc123"}

    await handler._run_realtime_session()

    assert "model" not in captured_connect
    assert captured_connect["extra_query"] == {"session_token": "abc123"}


@pytest.mark.asyncio
async def test_run_realtime_session_raises_the_receive_limit(monkeypatch: Any) -> None:
    """The server echoes image data URLs back; 1 MiB (the websockets default) closed with 1009."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    captured_connect: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_connect=captured_connect)

    await handler._run_realtime_session()

    assert captured_connect["websocket_connection_options"] == {"max_size": 16 * 2**20}


@pytest.mark.parametrize(("hf_token", "expected_api_key"), [(None, "DUMMY"), ("hf-secret", "hf-secret")])
@pytest.mark.asyncio
async def test_build_realtime_client_local_uses_explicit_hf_token_only(
    monkeypatch: Any,
    hf_token: str | None,
    expected_api_key: str,
) -> None:
    """Local websocket mode must never forward cached Hugging Face credentials."""
    client_kwargs: dict[str, Any] = {}

    def _no_allocator(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session allocator should not be called in direct websocket mode")

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _no_allocator)
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)
    monkeypatch.setattr(hf_mod, "get_token", lambda: "hf-cached")
    monkeypatch.setattr(
        config,
        "HF_REALTIME_WS_URL",
        "ws://127.0.0.1:8765/v1/realtime?session_token=abc123&model=ignored-by-sdk",
    )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
    assert client_kwargs["websocket_base_url"] == "ws://127.0.0.1:8765/v1"
    assert handler._realtime_connect_query == {"session_token": "abc123"}


@pytest.mark.parametrize(
    (
        "hf_token",
        "cached_token",
        "hardware_id",
        "status_error",
        "expected_header",
        "expected_api_key",
        "expected_payload",
    ),
    [
        (
            "hf-secret",
            "hf-cached",
            "0123456789abcdef",
            None,
            {
                "User-Agent": "reachy-mini-conversation-app",
                "X-Reachy-Mini-Authorization": "Bearer hf-secret",
            },
            "hf-secret",
            {"hardware_id": "0123456789abcdef"},
        ),
        (
            None,
            "hf-cached",
            None,
            None,
            {
                "User-Agent": "reachy-mini-conversation-app",
                "X-Reachy-Mini-Authorization": "Bearer hf-cached",
            },
            "hf-cached",
            {},
        ),
        (None, None, None, None, {"User-Agent": "reachy-mini-conversation-app"}, "DUMMY", {}),
        (
            None,
            None,
            None,
            TimeoutError("status unavailable"),
            {"User-Agent": "reachy-mini-conversation-app"},
            "DUMMY",
            {},
        ),
    ],
)
@pytest.mark.asyncio
async def test_build_realtime_client_deployed_resolves_hf_token(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
    hf_token: str | None,
    cached_token: str | None,
    hardware_id: str | None,
    status_error: Exception | None,
    expected_header: dict[str, str],
    expected_api_key: str,
    expected_payload: dict[str, str],
) -> None:
    """Deployed allocation reports available credentials and robot identity."""
    client_kwargs: dict[str, Any] = {}
    posts: list[tuple[str, dict[str, str] | None, dict[str, str] | None]] = []
    connect_url = "wss://hf.example.test/v1/realtime?session_token=allocated"
    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _fake_allocator(connect_url, posts))
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    # A stale local URL must be ignored in deployed mode.
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)
    monkeypatch.setattr(hf_mod, "get_token", lambda: cached_token)

    reachy_mini = MagicMock()
    reachy_mini.client.get_status.return_value.hardware_id = hardware_id
    if status_error:
        reachy_mini.client.get_status.side_effect = status_error
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert posts == [("https://lb.example.test/session", expected_header, expected_payload)]
    reachy_mini.client.get_status.assert_called_once_with(wait=False)
    if status_error:
        assert "Daemon status unavailable for realtime session allocation" in caplog.text
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "https://hf.example.test/v1"
    assert client_kwargs["websocket_base_url"] == "wss://hf.example.test/v1"
    assert handler._realtime_connect_query == {"session_token": "allocated"}


@pytest.mark.asyncio
async def test_apply_personality_uses_selected_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Live personality updates should honor the selected Qwen CustomVoice speaker."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Serena")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    monkeypatch.setattr(handler, "_restart_session", AsyncMock(return_value=None))

    result = await handler.apply_personality("mars_rover")

    assert "restarted realtime session" in result.lower()
    session = captured_update["session"]
    assert session["instructions"] == "new instructions"
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_apply_personality_restores_profile_when_tools_fail(monkeypatch: Any) -> None:
    """A failed tool reload should leave the previous profile selected."""
    selected_profiles: list[str | None] = []

    def select_profile(profile: str | None) -> None:
        selected_profiles.append(profile)
        config.REACHY_MINI_CUSTOM_PROFILE = profile

    def fail_tool_reload(*, force: bool = False) -> None:
        raise RuntimeError("tool reload failed")

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(hf_mod, "set_custom_profile", select_profile)
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod.core_tools, "initialize_tools", fail_tool_reload)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    result = await handler.apply_personality("broken")

    assert result == "Failed to apply personality: tool reload failed"
    assert config.REACHY_MINI_CUSTOM_PROFILE == "default"
    assert selected_profiles == ["broken", "default"]


@pytest.mark.asyncio
async def test_change_voice_updates_live_hf_session_without_restart(monkeypatch: Any) -> None:
    """Changing Hugging Face voice should update the active session in place."""
    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    restart = AsyncMock(return_value=None)
    monkeypatch.setattr(handler, "_restart_session", restart)

    result = await handler.change_voice("Serena")

    assert result == "Voice changed to Serena."
    assert handler.get_current_voice() == "Serena"
    restart.assert_not_awaited()
    session = captured_update["session"]
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_run_session_decodes_audio_delta(monkeypatch: Any) -> None:
    """An audio delta is base64-decoded and enqueued as an (rate, int16) frame."""
    pcm = np.array([1, 2, 3, 4], dtype=np.int16)
    delta = base64.b64encode(pcm.tobytes()).decode("utf-8")
    handler = _session_handler(monkeypatch, (_FakeEvent("response.output_audio.delta", delta=delta),))

    await handler._run_realtime_session()

    frames = [item for item in _drain(handler) if isinstance(item, tuple)]
    assert len(frames) == 1
    rate, array = frames[0]
    assert rate == handler.SAMPLE_RATE
    np.testing.assert_array_equal(array.reshape(-1), pcm)


@pytest.mark.asyncio
async def test_run_session_emits_completed_transcript(monkeypatch: Any) -> None:
    """A non-empty completed transcript is enqueued as a user message and stops listening."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("conversation.item.input_audio_transcription.completed", transcript="Hello there"),),
    )

    await handler._run_realtime_session()

    assert {"role": "user", "content": "Hello there"} in _messages(_drain(handler))
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_run_session_skips_empty_completed_transcript(monkeypatch: Any) -> None:
    """An empty completed transcript is ignored but still stops listening."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("conversation.item.input_audio_transcription.completed", transcript="   "),),
    )

    await handler._run_realtime_session()

    assert [msg for msg in _messages(_drain(handler)) if msg.get("role") == "user"] == []
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_run_session_generic_error_is_surfaced(monkeypatch: Any) -> None:
    """A generic realtime error is forwarded to the UI as an assistant message."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("error", error=SimpleNamespace(message="bad", code="server_error")),),
    )

    await handler._run_realtime_session()

    assert {"role": "assistant", "content": "[error] bad"} in _messages(_drain(handler))


@pytest.mark.asyncio
async def test_run_session_commit_empty_error_is_internal(monkeypatch: Any) -> None:
    """A commit-empty error stops listening and is not surfaced to the UI."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("error", error=SimpleNamespace(message="empty", code="input_audio_buffer_commit_empty")),),
    )

    await handler._run_realtime_session()

    assert _messages(_drain(handler)) == []
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_run_session_active_response_error_flags_rejection(monkeypatch: Any) -> None:
    """A duplicate-response error is recorded for the sender worker, not surfaced."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("error", error=SimpleNamespace(message="x", code="conversation_already_has_active_response")),),
    )

    await handler._run_realtime_session()

    assert handler._last_response_rejected is True
    assert _messages(_drain(handler)) == []


@pytest.mark.asyncio
async def test_run_session_dispatches_valid_tool_call(monkeypatch: Any) -> None:
    """A valid function call starts a background tool and announces it."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("response.function_call_arguments.done", name="dance", arguments="{}", call_id="c1"),),
    )
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="dance-1"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    start_tool.assert_awaited_once()
    assert any("Used tool dance" in msg["content"] for msg in _messages(_drain(handler)))


@pytest.mark.asyncio
async def test_run_session_ignores_invalid_tool_call(monkeypatch: Any) -> None:
    """A function call with a non-string name is ignored."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("response.function_call_arguments.done", name=None, arguments="{}", call_id="c1"),),
    )
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    start_tool.assert_not_awaited()
    assert not any("Used tool" in msg["content"] for msg in _messages(_drain(handler)))


def test_sanitize_tool_result_strips_camera_image() -> None:
    """Camera results drop the raw image bytes and flag that an image was attached."""
    sanitized = HuggingFaceRealtimeHandler._sanitize_tool_result_for_model("camera", {"b64_im": "x", "seen": "cat"})
    assert sanitized == {"seen": "cat", "image_attached": True}


def test_sanitize_tool_result_passthrough_for_non_camera() -> None:
    """Non-camera results are returned unchanged."""
    result = {"status": "ok"}
    assert HuggingFaceRealtimeHandler._sanitize_tool_result_for_model("dance", result) == result


class _FakeConnection:
    def __init__(self) -> None:
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=AsyncMock()))


@pytest.mark.asyncio
async def test_handle_tool_result_error_is_reported_and_prompts_response() -> None:
    """A failed tool reports its error to the UI and requests a spoken follow-up."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    note = ToolNotification(id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.FAILED, error="boom")

    await handler._handle_tool_result(note)

    assert {"role": "assistant", "content": json.dumps({"error": "boom"})} in _messages(_drain(handler))
    assert not handler._pending_responses.empty()
    handler.connection.conversation.item.create.assert_awaited()


@pytest.mark.asyncio
async def test_handle_tool_result_no_connection_is_dropped() -> None:
    """With no connection the tool result is dropped without enqueuing anything."""
    handler = _plain_handler()
    handler.connection = None
    note = ToolNotification(
        id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": 1}
    )

    await handler._handle_tool_result(note)

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_handle_tool_result_missing_result_reports_error() -> None:
    """A tool that returns neither result nor error is reported as a failure."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    note = ToolNotification(id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED)

    await handler._handle_tool_result(note)

    expected = json.dumps({"error": "No result returned from tool execution"})
    assert {"role": "assistant", "content": expected} in _messages(_drain(handler))


# ---------------------------------------------------------------------------
# The tool-call-log brief (2026-10-11): conversation.tool reports
# ---------------------------------------------------------------------------


def test_cut_long_strings_truncates_nested_values() -> None:
    """Every string, at any nesting depth, is cut to 200 characters; other types pass through."""
    value = {
        "question": "x" * 250,
        "nested": {"note": "y" * 300},
        "list": ["z" * 210, 5, True, None],
        "n": 42,
    }

    cut = hf_mod._cut_long_strings(value)

    assert cut["question"] == "x" * 200
    assert cut["nested"]["note"] == "y" * 200
    assert cut["list"][0] == "z" * 200
    assert cut["list"][1:] == [5, True, None]
    assert cut["n"] == 42


def test_tool_report_status_maps_completed_failed_cancelled() -> None:
    """ok/error/cancelled map directly from the BackgroundTool's ToolState."""
    completed = ToolNotification(
        id="1", tool_name="t", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": 1}
    )
    failed = ToolNotification(id="1", tool_name="t", is_idle_tool_call=False, status=ToolState.FAILED, error="boom")
    cancelled = ToolNotification(
        id="1", tool_name="t", is_idle_tool_call=False, status=ToolState.CANCELLED, error="Tool cancelled"
    )

    assert hf_mod._tool_report_status_and_reason(completed) == ("ok", None)
    assert hf_mod._tool_report_status_and_reason(failed) == ("error", "boom")
    assert hf_mod._tool_report_status_and_reason(cancelled) == ("cancelled", "Tool cancelled")


def test_tool_report_status_refused_via_result_marker() -> None:
    """A successful result marked `"refused": True` reports status='refused' with its reason."""
    refused = ToolNotification(
        id="1",
        tool_name="express",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={"played": False, "refused": True, "why": "quiet hours"},
    )
    assert hf_mod._tool_report_status_and_reason(refused) == ("refused", "quiet hours")


def test_tool_report_status_ok_result_without_marker_stays_ok() -> None:
    """A successful result with no refusal marker stays 'ok' (no tool sets one today)."""
    ok = ToolNotification(
        id="1", tool_name="express", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"played": "x"}
    )
    assert hf_mod._tool_report_status_and_reason(ok) == ("ok", None)


@pytest.mark.asyncio
async def test_handle_tool_result_reports_ok_call_with_full_payload() -> None:
    """A completed call is reported exactly once, with the brief's full field set."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    reports = _capture_tool_reports(handler)
    note = ToolNotification(
        id="c1",
        tool_name="camera",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={"seen": "a cat"},
        duration_s=0.25,
        args={"question": "x" * 250},
        source="turn",
        turn_id="resp-1",
    )

    await handler._handle_tool_result(note)

    assert len(reports) == 1
    report = reports[0]
    assert report["name"] == "camera"
    assert report["status"] == "ok"
    assert report["error"] is None
    assert report["ms"] == 250.0
    assert report["result_bytes"] == len(json.dumps({"seen": "a cat"}))
    assert report["background"] is True
    assert report["session"] is True
    assert report["source"] == "turn"
    assert report["turn_id"] == "resp-1"
    assert report["args"] == {"question": "x" * 200}


@pytest.mark.asyncio
async def test_handle_tool_result_reports_error_call() -> None:
    """A failed call is reported with status='error' and the short error text."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    reports = _capture_tool_reports(handler)
    note = ToolNotification(id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.FAILED, error="boom")

    await handler._handle_tool_result(note)

    assert len(reports) == 1
    assert reports[0]["status"] == "error"
    assert reports[0]["error"] == "boom"


@pytest.mark.asyncio
async def test_handle_tool_result_reports_cancelled_call() -> None:
    """A cancelled call is reported with status='cancelled'."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    reports = _capture_tool_reports(handler)
    note = ToolNotification(
        id="c1", tool_name="flights", is_idle_tool_call=False, status=ToolState.CANCELLED, error="Tool cancelled"
    )

    await handler._handle_tool_result(note)

    assert len(reports) == 1
    assert reports[0]["status"] == "cancelled"
    assert reports[0]["error"] == "Tool cancelled"


@pytest.mark.asyncio
async def test_handle_tool_result_reports_refused_call() -> None:
    """A tool that marks its own result refused is reported with status='refused'."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    reports = _capture_tool_reports(handler)
    note = ToolNotification(
        id="c1",
        tool_name="express",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={"played": False, "refused": True, "reason": "quiet hours"},
    )

    await handler._handle_tool_result(note)

    assert len(reports) == 1
    assert reports[0]["status"] == "refused"
    assert reports[0]["error"] == "quiet hours"


@pytest.mark.asyncio
async def test_handle_tool_result_reports_even_without_connection() -> None:
    """With no connection, the call is still reported once, with session=False."""
    handler = _plain_handler()
    handler.connection = None
    reports = _capture_tool_reports(handler)
    note = ToolNotification(
        id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": 1}
    )

    await handler._handle_tool_result(note)

    assert len(reports) == 1
    assert reports[0]["session"] is False


@pytest.mark.asyncio
async def test_handle_tool_result_reports_once_on_connection_closed_error() -> None:
    """A send that raises ConnectionClosedError is still reported, exactly once."""
    handler = _plain_handler()
    conn = _FakeConnection()
    conn.conversation.item.create.side_effect = ConnectionClosedError(None, None)
    handler.connection = conn
    reports = _capture_tool_reports(handler)
    note = ToolNotification(
        id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": 1}
    )

    await handler._handle_tool_result(note)

    assert len(reports) == 1
    assert handler.connection is None


def test_report_tool_call_caps_queue_and_counts_drops() -> None:
    """With no observer attached, reports pile up capped at 100, dropping the oldest beyond that."""
    handler = _plain_handler()
    for i in range(105):
        handler._report_tool_call(_report_payload(f"tool{i}"))

    assert len(handler._tool_report_queue) == 100
    assert handler._tool_reports_dropped == 5
    assert handler._tool_report_queue[0]["name"] == "tool5"
    assert handler._tool_report_queue[-1]["name"] == "tool104"


def test_report_tool_call_is_plain_sync_never_awaited() -> None:
    """The reporting path is synchronous, in-memory only -- never something a tool call awaits."""
    handler = _plain_handler()
    assert not asyncio.iscoroutinefunction(handler._report_tool_call)
    assert not asyncio.iscoroutinefunction(handler._flush_tool_reports)


def test_report_tool_call_retries_backlog_in_order_without_duplicates() -> None:
    """A flaky observer leaves items queued; once it recovers, the backlog flushes once each, in order."""
    handler = _plain_handler()
    delivered: list[dict[str, Any]] = []
    attempts = {"n": 0}

    def flaky_observer(payload: dict[str, Any]) -> bool:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            return False  # the supervisor (or no /rpc peer at all) is down
        delivered.append(payload)
        return True

    handler.set_tool_call_observer(flaky_observer)

    handler._report_tool_call(_report_payload("a"))
    assert delivered == []
    assert len(handler._tool_report_queue) == 1

    handler._report_tool_call(_report_payload("b"))
    assert delivered == []
    assert len(handler._tool_report_queue) == 2

    handler._report_tool_call(_report_payload("c"))
    assert delivered == [_report_payload("a"), _report_payload("b"), _report_payload("c")]
    assert len(handler._tool_report_queue) == 0


def test_report_tool_call_stops_draining_on_observer_exception() -> None:
    """An observer that raises leaves the item queued for a later attempt; nothing is dropped silently."""
    handler = _plain_handler()
    calls = {"n": 0}

    def raising_observer(payload: dict[str, Any]) -> bool:
        calls["n"] += 1
        raise RuntimeError("rpc broadcast failed")

    handler.set_tool_call_observer(raising_observer)
    handler._report_tool_call(_report_payload("a"))

    assert calls["n"] == 1
    assert len(handler._tool_report_queue) == 1


@pytest.mark.asyncio
async def test_run_session_turn_based_tool_call_reports_turn_source(monkeypatch: Any) -> None:
    """An ordinary user-turn tool call is started with source='turn' and the response id."""
    handler = _session_handler(
        monkeypatch,
        (
            _FakeEvent(
                "response.function_call_arguments.done",
                name="dance",
                arguments="{}",
                call_id="c1",
                response_id="resp-7",
            ),
        ),
    )
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="dance-1"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    kwargs = start_tool.call_args.kwargs
    assert kwargs["source"] == "turn"
    assert kwargs["turn_id"] == "resp-7"


@pytest.mark.asyncio
async def test_run_session_guard_nudged_tool_call_reports_missed_call_guard_source(monkeypatch: Any) -> None:
    """A tool call made while the missed-call guard's nudge is still in force for this turn is reported as such."""
    handler = _session_handler(
        monkeypatch,
        (
            _FakeEvent(
                "response.function_call_arguments.done",
                name="web_search",
                arguments="{}",
                call_id="c1",
                response_id="resp-8",
            ),
        ),
    )
    handler._turn_guarded = True  # the guard already nudged this turn
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="web_search-1"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    kwargs = start_tool.call_args.kwargs
    assert kwargs["source"] == "missed_call_guard"
    assert kwargs["turn_id"] == "resp-8"


@pytest.mark.asyncio
async def test_wait_for_response_done_returns_true_when_set() -> None:
    """The tool-result gate passes immediately when no response is active."""
    handler = _plain_handler()
    handler._response_done_event.set()
    assert await handler._wait_for_response_done_before_tool_result() is True


@pytest.mark.asyncio
async def test_wait_for_response_done_times_out(monkeypatch: Any) -> None:
    """The tool-result gate fails when the active response never finishes."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)
    handler = _plain_handler()
    handler._response_done_event.clear()
    assert await handler._wait_for_response_done_before_tool_result() is False


@pytest.mark.asyncio
async def test_emit_debounced_partial_emits_current_snapshot() -> None:
    """A partial transcript is emitted when it is still the latest snapshot."""
    handler = _plain_handler()
    handler.partial_debounce_delay = 0
    handler.input_transcript_chunks_by_item.item_id = "item-1"
    handler.input_transcript_chunks_by_item.deltas = ["hi"]

    await handler._emit_debounced_partial("hi", "item-1", 0)

    assert {"role": "user_partial", "content": "hi"} in _messages(_drain(handler))


@pytest.mark.asyncio
async def test_emit_debounced_partial_skips_stale_snapshot() -> None:
    """A partial transcript is dropped once a newer delta has arrived."""
    handler = _plain_handler()
    handler.partial_debounce_delay = 0
    handler.input_transcript_chunks_by_item.item_id = "item-1"
    handler.input_transcript_chunks_by_item.deltas = ["hi", "hi there"]

    await handler._emit_debounced_partial("hi", "item-1", 0)

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_emit_runs_idle_behavior_when_due(monkeypatch: Any) -> None:
    """Emit runs the idle behavior once the idle thresholds are exceeded and the robot is idle."""
    handler = _plain_handler()
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler.last_idle_behavior_time = handler.last_activity_time
    handler.deps.movement_manager.is_idle.return_value = True
    handler._response_done_event.set()
    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    before = time.monotonic()
    await handler.emit()

    send_idle_signal.assert_awaited_once()
    assert handler.last_idle_behavior_time >= before


@pytest.mark.asyncio
async def test_emit_keeps_idle_timer_when_idle_behavior_fails(monkeypatch: Any) -> None:
    """A failing idle behavior returns None and does not advance the idle-behavior timer."""
    handler = _plain_handler()
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler.last_idle_behavior_time = handler.last_activity_time
    handler.deps.movement_manager.is_idle.return_value = True
    handler._response_done_event.set()
    monkeypatch.setattr(handler, "send_idle_signal", AsyncMock(side_effect=RuntimeError("closed")))

    previous = handler.last_idle_behavior_time
    result = await handler.emit()

    assert result is None
    assert handler.last_idle_behavior_time == previous


def test_mark_activity_swallows_observer_errors() -> None:
    """The activity observer is notified, and a raising observer is ignored."""
    handler = _plain_handler()
    seen: list[str] = []
    handler.set_activity_observer(seen.append)
    handler._mark_activity("user_speech_started")
    assert seen == ["user_speech_started"]

    def _raise(_reason: str) -> None:
        raise RuntimeError("observer boom")

    handler.set_activity_observer(_raise)
    handler._mark_activity("later")  # must not raise


@pytest.mark.asyncio
async def test_send_idle_signal_noop_without_connection(monkeypatch: Any) -> None:
    """No idle tool runs when there is no active connection."""
    handler = _plain_handler()
    handler.connection = None
    start_idle = AsyncMock()
    monkeypatch.setattr(conv_mod, "start_idle_tool_call", start_idle)

    await handler.send_idle_signal(200.0)

    start_idle.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_idle_signal_dispatches_idle_tool(monkeypatch: Any) -> None:
    """With an active connection the locally selected idle tool is dispatched."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    monkeypatch.setattr(conv_mod, "get_tool_specs", lambda: [{"name": "dance"}])
    start_idle = AsyncMock()
    monkeypatch.setattr(conv_mod, "start_idle_tool_call", start_idle)

    await handler.send_idle_signal(200.0)

    start_idle.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_without_connection_is_dropped() -> None:
    """A frame received before the connection opens is dropped."""
    handler = _plain_handler()
    handler.connection = None
    await handler.receive((16000, np.zeros(4, dtype=np.int16)))  # must not raise


@pytest.mark.asyncio
async def test_receive_ignores_empty_frame() -> None:
    """An empty audio frame is not forwarded to the realtime buffer."""
    handler = _plain_handler()
    append = AsyncMock()
    handler.connection = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=append))
    await handler.receive((16000, np.array([], dtype=np.int16)))
    append.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_downmixes_stereo_and_forwards() -> None:
    """A stereo frame is reduced to mono and appended to the realtime buffer."""
    handler = _plain_handler()
    append = AsyncMock()
    handler.connection = SimpleNamespace(input_audio_buffer=SimpleNamespace(append=append))
    stereo = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.int16)
    await handler.receive((16000, stereo))
    append.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_session_speech_started_clears_queue_and_listens(monkeypatch: Any) -> None:
    """User speech start clears the playback queue and enters listening mode."""
    handler = _session_handler(monkeypatch, (_FakeEvent("input_audio_buffer.speech_started"),))
    clear_queue = MagicMock()
    handler._clear_queue = clear_queue

    await handler._run_realtime_session()

    clear_queue.assert_called_once()
    handler.deps.movement_manager.set_listening.assert_any_call(True)


@pytest.mark.asyncio
async def test_run_session_response_lifecycle_toggles_done_event(monkeypatch: Any) -> None:
    """response.created clears the done gate and response.done sets it again."""
    handler = _session_handler(
        monkeypatch,
        (_FakeEvent("response.created"), _FakeEvent("response.done")),
    )

    await handler._run_realtime_session()

    assert handler._response_done_event.is_set()
    handler.deps.movement_manager.set_speaking.assert_any_call(True)
    handler.deps.movement_manager.set_speaking.assert_any_call(False)


@pytest.mark.asyncio
async def test_handler_reopens_a_session_after_shutdown(monkeypatch: Any) -> None:
    """A handler that has been shut down can run a second session on the same object.

    Session control closes the realtime session and later reopens it without
    rebuilding the handler, because a rebuilt handler greets again. Nothing
    upstream exercises shutdown() followed by start_up() on one instance:
    _shutdown_active_handler's callers always rebuild. This pins it down.
    """
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "hello there")

    handler = _plain_handler()
    greetings: list[str] = []

    # Capture the original BEFORE patching. Reaching for it through type(handler)
    # inside the wrapper would find the wrapper, and recurse forever.
    original_greeting = type(handler)._send_startup_greeting_prompt

    async def _count_greeting(self: Any) -> None:
        before = self._startup_greeting_sent
        await original_greeting(self)
        if not before and self._startup_greeting_sent:
            greetings.append("greeted")

    monkeypatch.setattr(type(handler), "_send_startup_greeting_prompt", _count_greeting)

    def _fresh_client() -> Any:
        return _make_fake_realtime_client(events=(_FakeEvent("response.created"),))

    async def _build(self: Any) -> Any:
        return _fresh_client()

    monkeypatch.setattr(type(handler), "_build_realtime_client", _build)

    # First session: opens, greets, and ends when the fake connection runs dry.
    await handler.start_up()
    assert greetings == ["greeted"]
    assert handler.connection is None

    # Close it the way the close verb will.
    await handler.shutdown()

    # Second session on the same object.
    await handler.start_up()

    # It tore down cleanly a second time.
    assert handler.tool_manager._lifecycle_tasks == []
    # And it did not greet, which is what closes the plan 3 defect.
    assert greetings == ["greeted"]


class _LiveFakeConn:
    """A fake realtime connection that stays open until `close()` is called.

    `_make_fake_realtime_client`'s FakeConn ends the session itself once its
    `events` run out -- by the time `start_up()` returns, `handler.connection`
    is already `None` (its own `finally` guarantees that), so a `shutdown()`
    called afterward never reaches its `connection.close()` branch. This fake
    keeps the connection idle-but-open so `shutdown()` is the thing that ends
    the session, while `_run_realtime_session` is still parked in `async for`.
    """

    def __init__(self) -> None:
        """Build a connection with no queued events and an open `close()` gate."""
        self.session = SimpleNamespace(update=AsyncMock())
        self.input_audio_buffer = SimpleNamespace(append=AsyncMock())
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=AsyncMock(), cancel=AsyncMock()))
        self.response = SimpleNamespace(create=AsyncMock())
        self._closed = asyncio.Event()

    async def __aenter__(self) -> "_LiveFakeConn":
        """Support use as an async context manager."""
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        """Propagate any exception raised inside the `async with` block."""
        return False

    async def close(self) -> None:
        """Signal the pending `__anext__` to end the session, like a real close."""
        self._closed.set()

    def __aiter__(self) -> "_LiveFakeConn":
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> _FakeEvent:
        """Block until closed, the way an idle live websocket would."""
        await self._closed.wait()
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_handler_reopens_a_session_after_closing_a_live_connection(monkeypatch: Any) -> None:
    """shutdown() must close a session that is still open, not just a finished one.

    test_handler_reopens_a_session_after_shutdown proves reuse in the natural-end
    case, where the fake connection has already run dry and start_up()'s own
    `finally` has set `self.connection = None` before shutdown() is ever called --
    shutdown()'s `if self.connection: await self.connection.close()` branch never
    runs there. This test keeps the connection open so shutdown() is the thing
    that ends it, concurrently with _run_realtime_session() still awaiting the
    next event -- the situation Task 3's close verb actually faces.
    """
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "hello there")

    handler = _plain_handler()
    greetings: list[str] = []
    original_greeting = type(handler)._send_startup_greeting_prompt

    async def _count_greeting(self: Any) -> None:
        before = self._startup_greeting_sent
        await original_greeting(self)
        if not before and self._startup_greeting_sent:
            greetings.append("greeted")

    monkeypatch.setattr(type(handler), "_send_startup_greeting_prompt", _count_greeting)

    live_conns: list[_LiveFakeConn] = []
    build_calls = {"n": 0}

    async def _build(self: Any) -> Any:
        build_calls["n"] += 1
        if build_calls["n"] == 1:
            conn = _LiveFakeConn()
            live_conns.append(conn)

            class _LiveClient:
                realtime = SimpleNamespace(connect=lambda **_kw: conn)

            return _LiveClient()
        # Second session: a session that ends on its own, like the other test.
        return _make_fake_realtime_client(events=(_FakeEvent("response.created"),))

    monkeypatch.setattr(type(handler), "_build_realtime_client", _build)

    # First session: starts in the background and stays open (no queued events).
    task = asyncio.create_task(handler.start_up())
    await asyncio.wait_for(handler._connected_event.wait(), timeout=1.0)
    assert handler.connection is not None
    assert greetings == ["greeted"]

    # Close it the way the close verb will, while the session is still live.
    await asyncio.wait_for(handler.shutdown(), timeout=1.0)

    # The concurrent _run_realtime_session() must actually unwind and return --
    # start_up() cannot be called again on the same object while it is still running.
    await asyncio.wait_for(task, timeout=1.0)
    assert handler.connection is None

    # Second session on the same object.
    await handler.start_up()

    # It tore down cleanly a second time.
    assert handler.tool_manager._lifecycle_tasks == []
    # And it did not greet, which is what closes the plan 3 defect.
    assert greetings == ["greeted"]


# --- the inner monologue: sight notes, written by the companion since 2026-10-17 -----


def test_a_frame_is_handed_over_and_nothing_is_asked_of_the_voice_service(monkeypatch: Any) -> None:
    """The cold-open brief: the supervisor writes the note with Claude Haiku."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler.note_image("abc", "glance")  # nobody to hand it to: nothing, and no error
    got: list[tuple[str, str]] = []
    handler.on_sight = lambda b64, source: got.append((b64, source))
    handler.note_image("abc", "glance")
    assert got == [("abc", "glance")]
    create.assert_not_awaited()
    handler.connection.conversation.item.create.assert_not_awaited()


def test_a_failing_hand_over_is_logged_not_raised(caplog: Any) -> None:
    """A broken hand-over never reaches the conversation."""
    handler = _plain_handler()

    def broken(b64: str, source: str) -> None:
        raise RuntimeError("no rpc")

    handler.on_sight = broken
    with caplog.at_level(logging.WARNING):
        handler.note_image("abc", "camera")
    assert "not handed over for its sight note" in caplog.text


@pytest.mark.asyncio
async def test_a_camera_picture_is_handed_over_and_asks_only_for_the_reply(monkeypatch: Any) -> None:
    """The camera tool's picture goes into the conversation and to the companion.

    The only response.create is the spoken follow-up.
    """
    handler = _plain_handler()
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    got: list[tuple[str, str]] = []
    handler.on_sight = lambda b64, source: got.append((b64, source))
    handler._in_flight_tool_calls = {"cam"}

    await handler._handle_tool_result(
        ToolNotification(
            id="cam",
            tool_name="camera",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"b64_im": "abc"},
        )
    )

    items = [c.kwargs["item"] for c in handler.connection.conversation.item.create.await_args_list]
    assert items[-1]["content"] == [{"type": "input_image", "image_url": "data:image/jpeg;base64,abc"}]
    assert [c.kwargs for c in create.await_args_list] == [{}]
    assert got == [("abc", "camera")]
    assert handler._session_images == 1


@pytest.mark.asyncio
async def test_a_text_response_is_never_an_inner_note(monkeypatch: Any) -> None:
    """With notes off the voice service, the app writes no inner transcript at all."""
    from types import SimpleNamespace

    tagged = SimpleNamespace(metadata={"companion": "sight-note"}, conversation_id=None)
    handler = _session_handler(
        monkeypatch,
        events=(
            _FakeEvent("response.created", response=tagged),
            _FakeEvent("response.output_text.done", text="I saw a red mug on the table."),
            _FakeEvent("response.done", response=tagged),
        ),
    )
    seen: list[tuple[str, str, bool]] = []
    handler.set_transcript_observer(lambda role, text, final, artifact=None: seen.append((role, text, final)))
    await handler._run_realtime_session()
    assert [s for s in seen if s[0] == "inner"] == []


@pytest.mark.asyncio
async def test_speech_marks_the_moment_for_the_notes(monkeypatch: Any) -> None:
    """The receiver records when speech starts and stops, for the quiet check."""
    handler = _session_handler(
        monkeypatch,
        events=(
            _FakeEvent("input_audio_buffer.speech_started"),
            _FakeEvent("input_audio_buffer.speech_stopped"),
        ),
    )
    before = time.monotonic()
    await handler._run_realtime_session()
    assert handler._user_speaking is False and handler._last_speech_at >= before


# --- the cold-open brief (2026-10-17): the glance at open against the first reply ----


def test_a_glance_in_before_the_first_reply_logs_its_margin(caplog: Any) -> None:
    """The ordinary path: the frame is in, the reply sees it."""
    handler = _plain_handler()
    with caplog.at_level(logging.INFO):
        handler.open_glance_in()
        handler._first_reply_created()
    assert "Glance in 0.0 s before the first reply" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_first_reply_before_the_glance_is_a_warning(caplog: Any) -> None:
    """The pre-agreed rollback trigger: the reply may not have seen the frame."""
    handler = _plain_handler()
    with caplog.at_level(logging.INFO):
        handler._first_reply_created()
        handler.open_glance_in()
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.getMessage().startswith("First reply created 0.0 s before the glance")
    assert "Glance in" not in caplog.text


@pytest.mark.asyncio
async def test_only_the_first_reply_to_a_person_is_judged_and_each_session_starts_fresh(
    monkeypatch: Any,
) -> None:
    """Not the startup greeting, not a later turn; nothing carried over from a session before."""
    from types import SimpleNamespace

    spoken = SimpleNamespace(metadata=None, conversation_id="conv_1")
    turn = (
        _FakeEvent("conversation.item.input_audio_transcription.completed", transcript="hi", item_id="u1"),
        _FakeEvent("response.created", response=spoken),
        _FakeEvent("response.done", response=spoken),
    )
    greeting = (_FakeEvent("response.created", response=spoken), _FakeEvent("response.done", response=spoken))
    handler = _session_handler(monkeypatch, events=greeting + turn + turn)
    calls: list[float | None] = []
    real = handler._first_reply_created

    def counted() -> None:
        calls.append(handler._open_glance_at)
        real()

    monkeypatch.setattr(handler, "_first_reply_created", counted)
    handler._open_glance_at = handler._first_reply_at = 1.0  # left over from a session before
    await handler._run_realtime_session()
    assert calls == [None]  # once, for the person's turn, not the greeting; the old glance cleared


@pytest.mark.asyncio
async def test_a_turn_left_open_by_the_session_before_does_not_make_the_greeting_first(
    monkeypatch: Any,
) -> None:
    """The review (2026-10-17): a transcript with no reply before a close; the next
    session's greeting would have been judged as its first reply (a false rollback warning)."""
    from types import SimpleNamespace

    spoken = SimpleNamespace(metadata=None, conversation_id="conv_1")
    handler = _session_handler(
        monkeypatch,
        events=(_FakeEvent("response.created", response=spoken), _FakeEvent("response.done", response=spoken)),
    )
    handler._turn_user_done_at = time.perf_counter()
    handler._turn_response_created_at = None
    await handler._run_realtime_session()
    assert handler._first_reply_at is None


@pytest.mark.asyncio
async def test_a_session_logs_what_it_sent_the_voice_service(monkeypatch: Any, caplog: Any) -> None:
    """Images, their bytes and response.create calls per session (the brief's measure)."""
    sent: list[tuple[str, Any]] = []
    handler = _session_handler(monkeypatch, ())
    handler._session_images = 7  # from a session before: reset at the open

    async def _traffic() -> None:
        await handler.add_image("A" * 4096)
        await handler._safe_response_create()
        await asyncio.sleep(0.05)

    handler.client = _ordering_client((_traffic,), sent, handler)
    with caplog.at_level(logging.INFO):
        await handler._run_realtime_session()
    assert "Session to the voice service: 1 images, 3 KB, 1 response.create" in caplog.text


# --- the companion note (task 11, person recognition): who has joined, who left view --


@pytest.mark.asyncio
async def test_a_note_waits_for_quiet_and_asks_for_no_response(monkeypatch: Any) -> None:
    """A companion note is a user-role item at the next quiet moment, never a response."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(hf_mod, "_NOTE_POLL_SECONDS", 0.01)
    handler._response_done_event.clear()

    handler.add_note("Alli (she/her) has joined.")
    await asyncio.sleep(0.05)
    handler.connection.conversation.item.create.assert_not_awaited()
    handler._response_done_event.set()
    await handler._wait_notes()

    item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert item["role"] == "user"
    assert item["content"] == [
        {"type": "input_text", "text": "[Companion note, not spoken: Alli (she/her) has joined.]"}
    ]
    create.assert_not_awaited()


def test_a_note_needs_a_session() -> None:
    """Without a session there is nothing to note into."""
    handler = _plain_handler()
    handler.connection = None
    with pytest.raises(RuntimeError):
        handler.add_note("x")


@pytest.mark.asyncio
async def test_a_note_held_too_long_is_dropped(monkeypatch: Any) -> None:
    """Without a quiet moment inside COMPANION_NOTE_WAIT_S, the note is dropped."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    monkeypatch.setattr(hf_mod, "_NOTE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(hf_mod, "COMPANION_NOTE_WAIT_S", 0.05)
    handler._user_speaking = True

    handler.add_note("x")
    await handler._wait_notes()

    handler.connection.conversation.item.create.assert_not_awaited()
    assert not handler._note_tasks


@pytest.mark.asyncio
async def test_a_notes_text_never_reaches_the_transcript_observer() -> None:
    """The note is for the model alone; the companion's own memory never sees it."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    seen: list[tuple[str, str, bool]] = []
    handler.set_transcript_observer(lambda role, text, final, artifact=None: seen.append((role, text, final)))

    handler.add_note("x")
    await handler._wait_notes()

    assert seen == []


@pytest.mark.asyncio
async def test_a_notes_failed_send_is_dropped_and_logged(caplog: Any) -> None:
    """A send that raises (the review, fix round 1) is caught, not fatal."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    handler.connection.conversation.item.create.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        handler.add_note("x")
        await handler._wait_notes()

    assert any("Companion note dropped" in r.getMessage() and "boom" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_note_whose_session_closed_before_quiet_is_dropped_and_logged(monkeypatch: Any, caplog: Any) -> None:
    """The session ended while the note waited: nothing left to note into."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    monkeypatch.setattr(hf_mod, "_NOTE_POLL_SECONDS", 0.01)
    handler._response_done_event.clear()

    with caplog.at_level(logging.INFO):
        handler.add_note("x")
        await asyncio.sleep(0.03)
        handler.connection = None
        handler._response_done_event.set()
        await handler._wait_notes()

    assert any("session closed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_note_dropped_when_the_session_is_replaced_before_quiet(monkeypatch: Any, caplog: Any) -> None:
    """A note belongs to the session it was asked for (the review, fix round 1).

    A reconnect between the ask and the quiet moment is a different session.
    """
    handler = _plain_handler()
    original = AsyncMock()
    handler.connection = original
    monkeypatch.setattr(hf_mod, "_NOTE_POLL_SECONDS", 0.01)
    handler._response_done_event.clear()

    with caplog.at_level(logging.INFO):
        handler.add_note("x")
        await asyncio.sleep(0.03)
        new_session = AsyncMock()
        handler.connection = new_session
        handler._response_done_event.set()
        await handler._wait_notes()

    original.conversation.item.create.assert_not_awaited()
    new_session.conversation.item.create.assert_not_awaited()
    assert any("session changed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_typed_text_is_a_user_turn_answered_and_reported() -> None:
    """Add typed text as a user message, ask for a response, and report it.

    The observer hears it as the person's words, so the companion's memory keeps it.
    """
    handler = _plain_handler()
    handler.connection = AsyncMock()
    create = AsyncMock()
    handler._safe_response_create = create
    seen: list[tuple[str, str, bool]] = []
    handler.set_transcript_observer(lambda role, text, final, artifact=None: seen.append((role, text, final)))
    await handler.type_text("  play it again ")
    item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert item["role"] == "user" and item["content"] == [{"type": "input_text", "text": "play it again"}]
    create.assert_awaited_once()
    assert seen == [("user", "play it again", True)]


@pytest.mark.asyncio
async def test_a_skipped_greeting_stays_skipped_on_a_reconnect(monkeypatch: Any) -> None:
    """Skip the greeting until the next open says otherwise.

    The review (2026-10-17): a reconnect within a conversation, on the same handler and
    with no new open, spoke "Go ahead." once the skip had been used up.
    """
    import reachy_mini_conversation_app.huggingface_realtime as hr

    monkeypatch.setattr(hr, "get_session_greeting_prompt", lambda: "Greet them.")
    handler = _plain_handler()
    handler.connection = AsyncMock()
    handler._safe_response_create = AsyncMock()
    handler.skip_next_greeting = True
    await handler._send_startup_greeting_prompt()
    await handler._send_startup_greeting_prompt()  # the reconnect
    handler.connection.conversation.item.create.assert_not_awaited()
    assert handler.skip_next_greeting is True and handler._startup_greeting_sent is False


@pytest.mark.asyncio
async def test_a_glance_frame_is_marked_as_context_only() -> None:
    """Mark the frame so the model does not describe it unasked.

    The companion, 2026-10-08: after a song the model, with nothing left to say,
    described the frame taken as the conversation opened.
    """
    handler = _plain_handler()
    handler.connection = AsyncMock()
    await handler.add_image("abc")
    item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    kinds = [part["type"] for part in item["content"]]
    assert kinds == ["input_text", "input_image"]
    note = item["content"][0]["text"].lower()
    assert "context only" in note and "unless" in note
    assert item["content"][1]["image_url"] == "data:image/jpeg;base64,abc"


@pytest.mark.asyncio
async def test_ending_a_user_turn_clears_the_input_and_the_listening_pose() -> None:
    """End a half-heard turn: the companion's audio audit, A4.

    A mute or capture-off mid-utterance left _user_speaking set, the listening pose
    held, and the server's input buffer mid-turn.
    """
    handler = _plain_handler()
    handler.connection = AsyncMock()
    handler._user_speaking = True
    await handler.end_user_turn()
    assert handler._user_speaking is False
    handler.deps.movement_manager.set_listening.assert_called_with(False)
    # Never input_audio_buffer.clear: the hosted backend rejects it (2026-10-09, 11:22).
    handler.connection.input_audio_buffer.clear.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_releases_the_listening_pose() -> None:
    """The companion, 2026-10-19: a session closed between speech_started and
    speech_stopped left listening on, so breathing and every idle body turn stopped."""
    handler = _plain_handler()
    handler._user_speaking = True
    await handler.shutdown()
    assert handler._user_speaking is False
    handler.deps.movement_manager.set_listening.assert_called_with(False)


@pytest.mark.asyncio
async def test_ending_a_turn_with_no_session_is_harmless() -> None:
    """With no session there is no input buffer to clear; the flags still reset."""
    handler = _plain_handler()
    handler.connection = None
    await handler.end_user_turn()
    assert handler._user_speaking is False


@pytest.mark.asyncio
async def test_a_say_during_a_live_response_cancels_it_first(monkeypatch: Any) -> None:
    """The companion's audio audit, A5: a say behind a live response waited for it to
    finish streaming, so the old reply played ahead of the announcement."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", AsyncMock())
    handler._response_done_event.clear()  # a response is active
    await handler.say("Your timer is up")
    handler.connection.response.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_say_with_no_live_response_cancels_nothing(monkeypatch: Any) -> None:
    """Explain the case: no cancel when nothing is streaming."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", AsyncMock())
    await handler.say("Your timer is up")
    handler.connection.response.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_say_during_a_tools_follow_up_does_not_cancel_it(monkeypatch: Any) -> None:
    """The follow-up carries the tool's answer: the announcement waits behind it instead."""
    handler = _plain_handler()
    handler.connection = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", AsyncMock())
    handler._response_done_event.clear()
    handler._followup_live = True
    await handler.say("Your timer is up")
    handler.connection.response.cancel.assert_not_awaited()


# --- reliable search (the companion, 2026-10-09) ---------------------------------------------


def _search_handler(monkeypatch: Any) -> tuple[HuggingFaceRealtimeHandler, AsyncMock]:
    handler = _plain_handler()
    handler.connection = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._start_user_turn()
    return handler, create


def _done(tag: str | None = None) -> Any:
    metadata = {"companion": tag} if tag else None
    return SimpleNamespace(response=SimpleNamespace(metadata=metadata))


@pytest.mark.asyncio
async def test_a_spoken_promise_with_no_call_is_caught_once(monkeypatch: Any, caplog: Any) -> None:
    """09:50:26: 'Let me check.' and no call; then 'I'm looking it up' and no call again."""
    handler, create = _search_handler(monkeypatch)
    handler._response_started()
    handler._response_transcript("Let me check.")
    with caplog.at_level(logging.WARNING):
        assert await handler._response_finished(_done()) is True
    assert any("missed tool call" in r.getMessage() for r in caplog.records)
    item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert item["role"] == "system" and "web_search" in item["content"][0]["text"]
    assert create.await_count == 1
    # The same turn: never again, never a loop.
    handler._response_started()
    handler._response_transcript("Yes, I'm looking it up for you now.")
    assert await handler._response_finished(_done()) is False
    assert create.await_count == 1
    # A new turn may be guarded again.
    handler._start_user_turn()
    handler._response_started()
    handler._response_transcript("Let me find out.")
    assert await handler._response_finished(_done()) is True


@pytest.mark.asyncio
async def test_a_promise_with_its_call_is_left_alone(monkeypatch: Any) -> None:
    handler, create = _search_handler(monkeypatch)
    handler._response_started()
    handler._response_transcript("Let me check.")
    handler._response_called_a_tool()
    assert await handler._response_finished(_done()) is False
    assert create.await_count == 0


@pytest.mark.asyncio
async def test_let_me_see_you_is_not_a_lookup(monkeypatch: Any) -> None:
    """A camera request: 'let me see you' promises a look, not a search."""
    handler, create = _search_handler(monkeypatch)
    handler._response_started()
    handler._response_transcript("Sure, let me see you.")
    assert await handler._response_finished(_done()) is False


@pytest.mark.asyncio
async def test_the_apps_own_filler_is_never_guarded(monkeypatch: Any) -> None:
    handler, create = _search_handler(monkeypatch)
    handler._response_started()
    handler._response_transcript("Let me check.")
    assert await handler._response_finished(_done("filler")) is False


@pytest.mark.asyncio
async def test_a_search_call_gets_one_filler_never_over_speech(monkeypatch: Any) -> None:
    handler, create = _search_handler(monkeypatch)
    await handler._tool_started("pollen_robotics_reachy_mini_search_tool__search_web", "c1")
    assert create.await_count == 1
    body = create.await_args.kwargs["response"]
    assert body["metadata"] == {"companion": "filler"} and body["conversation"] == "none"
    assert body["output_modalities"] == ["audio"]
    first = handler._last_filler
    await handler._tool_started("pollen_robotics_reachy_mini_search_tool__search_web", "c2")
    assert create.await_count == 1  # once per turn
    handler._start_user_turn()
    handler._turn_first_audio_at = 1.0  # he already said something this turn
    await handler._tool_started("web_search", "c3")
    assert create.await_count == 1  # never over the response's own audio
    handler._start_user_turn()
    handler._turn_first_audio_at = None  # the event loop resets it at each user turn
    await handler._tool_started("web_search", "c4")
    assert create.await_count == 2 and handler._last_filler != first  # not the same twice


@pytest.mark.asyncio
async def test_a_slow_lookup_gets_a_filler_a_quick_one_and_a_song_do_not(monkeypatch: Any) -> None:
    monkeypatch.setattr(hf_mod, "FILLER_AFTER_S", 0.05)
    handler, create = _search_handler(monkeypatch)
    handler._in_flight_tool_calls.add("slow")
    await handler._tool_started("flights", "slow")
    await asyncio.sleep(0.1)
    assert create.await_count == 1
    handler._start_user_turn()
    await handler._tool_started("flights", "quick")  # not in flight: already answered
    await asyncio.sleep(0.1)
    handler._start_user_turn()
    handler._in_flight_tool_calls.add("song")
    await handler._tool_started("music", "song")  # the tool is the response
    await asyncio.sleep(0.1)
    assert create.await_count == 1


def test_every_tool_result_logs_its_latency(caplog: Any) -> None:
    handler = _plain_handler()
    done = ToolNotification(
        id="c1",
        tool_name="web_search",
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={"answer": "x" * 40},
        duration_s=1.25,
    )
    with caplog.at_level(logging.INFO):
        handler._log_tool_latency(done)
    line = [r.getMessage() for r in caplog.records if "tool latency" in r.getMessage()][0]
    assert line.startswith("tool latency: web_search 1250 ms") and "completed" in line


# --- the audio cut for spoken artifacts ------------------------------------------------------


def _pcm_delta(*samples: int) -> str:
    return base64.b64encode(np.array(samples, dtype=np.int16).tobytes()).decode("utf-8")


def _frames(handler: HuggingFaceRealtimeHandler) -> list[Any]:
    return [item for item in _drain(handler) if isinstance(item, tuple)]


def _audio(*samples: int) -> _FakeEvent:
    return _FakeEvent("response.output_audio.delta", delta=_pcm_delta(*samples))


def _said(delta: str) -> _FakeEvent:
    return _FakeEvent("response.output_audio_transcript.delta", delta=delta)


def _reply_events(transcript: str, *, audio_first: bool) -> tuple[Any, ...]:
    """One response as speech-to-speech sends it, or with the audio ahead of the transcript."""
    created = _FakeEvent("response.created", response=SimpleNamespace(metadata=None))
    audio = (_audio(1, 2), _audio(3, 4))
    said = (_said(transcript),)
    body = audio + said if audio_first else said + audio
    return (
        created,
        *body,
        _FakeEvent("response.output_audio.done"),
        _FakeEvent("response.output_audio_transcript.done", transcript=transcript),
        _FakeEvent("response.done", response=SimpleNamespace(status="completed", output=[])),
    )


# The artifact rule's table (output_gate.spoken_artifact). The supervisor keeps no copy
# since 2026-10-17: it stores what the transcript notification's "artifact" mark says.
_S, _C, _T, _U = "stage direction", "leaked tool call", "thinking", "undecided"
SPOKEN_ARTIFACT_CASES = [
    # seen on the robot
    ("call:stay_quiet{}", _C),
    ("<call:stay_quiet/>", _C),
    ("(stops talking)", _S),
    ("(A small, contented tilt of the head)", _S),
    # Gemma 4
    (' CALL: express {"mood": "happy"} ', _C),
    ('call:express{mood:<|"|>happy}<|"|>}', _C),
    ('<|tool_call>call:express{mood:<|"|>happy<|"|>}<tool_call|>', _C),
    ("<call:stay_quiet>", _C),
    ("<call:stay_quiet></call>", _C),
    ("<call:stay_quiet>{}</call>", _C),
    ('<call:express {"mood": "happy"}/>', _C),
    ("<|tool_call>", _C),
    ("<tool_call|>", _C),
    ("<|tool_response>", _C),
    ("<tool_response|>", _C),
    ('<|"|>', _C),
    ("<|tool>", _C),
    ("<tool|>", _C),
    ("<|im_end|>", _C),
    # Qwen3, Qwen3.5 / Qwen3-Coder
    ('<tool_call>{"name": "stay_quiet", "arguments": {}}</tool_call>', _C),
    ("<tool_call><function=stay_quiet><parameter=x>1</parameter></function></tool_call>", _C),
    ("<function=stay_quiet><parameter=x>1</parameter></function>", _C),
    # Gemma 3
    ("[stay_quiet()]", _C),
    ('[express(mood="happy"), look(at=[1, 2])]', _C),
    ("```tool_code\nstay_quiet()\n```", _C),
    # generic
    ("<think>the user wants quiet</think>", _T),
    ('<function_call>{"name": "stay_quiet"}</function_call>', _C),
    ('{"name": "stay_quiet", "arguments": {}}', _C),
    ('{"name": "express", "parameters": {"mood": "happy"}}', _C),
    # stage directions
    ("  *sighs*", _S),
    ("_shrugs_", _S),
    (" [says nothing]. ", _S),
    ("(stops talking).", _S),
    # two in a row
    ("<call:stay_quiet/> (stays silent)", _C),
    ("(stays silent) call:stay_quiet{};", _S),
    # replies that play whole
    ("Sure, (I think) yes.", None),
    ("(Laughs) yes, the flight leaves at 3.", None),
    ("(I saw a mug) and a plate", None),
    ("I'll stay quiet (for now).", None),
    ("*waves* Hello there!", None),
    ("I'll call: the vet {maybe}", None),
    ("Calling you now.", None),
    ("Call me later.", None),
    ("can", None),
    ("call:stay_quiet{} and then hello", None),
    ("Use <b>bold</b> text", None),
    ("<b>bold</b> text", None),
    ('{"name": "Matt", "age": 40}', None),
    ("```python\nprint(1)\n```", None),
    ("(" + "x" * 81 + ")", None),
    # still forming
    ("", _U),
    ("   ", _U),
    ("<", _U),
    ("<|tool", _U),
    ("(stops", _U),
    ("call:stay", _U),
    ("call:stay_quiet", _U),
    ("<think>hmm", _U),
    ("ca", _U),
    ("Cal", _U),
    ("<call:express {", _U),
    ("<tool_call>{", _U),
    ("```tool_c", _U),
    ("<call:stay_quiet/> (stays", _U),
    ("<think>" + "x" * 200, None),
]


@pytest.mark.parametrize(("text", "verdict"), SPOKEN_ARTIFACT_CASES)
def test_spoken_artifact_rule(text: str, verdict: str | None) -> None:
    """Only text made wholly of artifact shapes is an artifact; an unfinished one waits."""
    assert hf_mod.spoken_artifact(text) == verdict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "kind"),
    [
        ("(stops talking)", "stage direction"),
        ("call:stay_quiet{}", "leaked tool call"),
        ("<call:stay_quiet/>", "leaked tool call"),
        ("*sighs*", "stage direction"),
    ],
)
@pytest.mark.parametrize("audio_first", [False, True])
async def test_a_spoken_artifact_never_reaches_the_player(
    monkeypatch: Any, caplog: Any, transcript: str, kind: str, audio_first: bool
) -> None:
    """None of an artifact's audio is queued, nothing is flushed, and the cut is logged."""
    handler = _session_handler(monkeypatch, _reply_events(transcript, audio_first=audio_first))
    cleared: list[bool] = []
    handler._clear_queue = lambda: cleared.append(True)

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await handler._run_realtime_session()

    assert _frames(handler) == []
    assert cleared == []  # nothing of it was released, so nothing else is flushed
    cuts = [r.getMessage() for r in caplog.records if "spoken artifact cut" in r.getMessage()]
    assert cuts == [f"spoken artifact cut ({kind}): {transcript!r}"]
    assert all(r.levelno == logging.WARNING for r in caplog.records if "spoken artifact cut" in r.getMessage())


@pytest.mark.asyncio
async def test_a_tag_call_split_across_deltas_is_cut(monkeypatch: Any) -> None:
    """14:02:35, 2026-10-16: "<call:stay_quiet/>" slipped through; its first deltas wait."""
    handler = _session_handler(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _said("<"),
            _audio(1, 2),
            _said("call:stay"),
            _said("_quiet/>"),
            _audio(3, 4),
        ),
    )
    handler._clear_queue = lambda: None

    await handler._run_realtime_session()

    assert _frames(handler) == []


@pytest.mark.asyncio
async def test_a_leaked_tool_call_is_not_executed(monkeypatch: Any) -> None:
    """Spoken "call:stay_quiet{}" starts no tool."""
    handler = _session_handler(monkeypatch, _reply_events("call:stay_quiet{}", audio_first=False))
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    await handler._run_realtime_session()

    start_tool.assert_not_awaited()


class _FakeTool:
    """A registry entry for the tool-name leak tests: counts its runs."""

    def __init__(self, leak_safe: bool) -> None:
        self.leak_safe = leak_safe
        self.runs = 0

    async def __call__(self, deps: Any, **kwargs: Any) -> dict[str, Any]:
        self.runs += 1
        return {"status": "quiet"}


def _leak_session(monkeypatch: Any, events: tuple[Any, ...]) -> tuple[HuggingFaceRealtimeHandler, dict[str, _FakeTool]]:
    """A session whose live registry is stay_quiet (leak_safe), web_search and dance."""
    handler = _session_handler(monkeypatch, events)
    tools = {"stay_quiet": _FakeTool(True), "web_search": _FakeTool(False), "dance": _FakeTool(False)}
    specs = [{"type": "function", "name": name, "description": "", "parameters": {}} for name in tools]
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: specs)
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: dict(tools))
    handler._clear_queue = lambda: None
    return handler, tools


async def _run_with_leaks(handler: HuggingFaceRealtimeHandler) -> None:
    await handler._run_realtime_session()
    if handler._leak_tasks:
        await asyncio.gather(*handler._leak_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_first", [False, True])
async def test_a_bare_leak_safe_name_is_cut_run_reported_and_marked(
    monkeypatch: Any, caplog: Any, audio_first: bool
) -> None:
    """20:44:56, 2026-10-17: "stay_quiet" was heard. Now none of it plays; the tool runs
    locally (never start_tool, which owes the server an output); the report says
    "leaked"; and the supervisor is told not to keep it."""
    handler, tools = _leak_session(monkeypatch, _reply_events("stay_quiet", audio_first=audio_first))
    reports = _capture_tool_reports(handler)
    seen: list[tuple[Any, ...]] = []
    handler.set_transcript_observer(lambda *args: seen.append(args))
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await _run_with_leaks(handler)

    assert _frames(handler) == []
    assert "leaked tool call: stay_quiet (underscore name) cut; executed=True" in caplog.messages
    assert tools["stay_quiet"].runs == 1
    start_tool.assert_not_awaited()
    assert [(r["name"], r["status"], r["source"]) for r in reports] == [("stay_quiet", "ok", "leaked")]
    assert seen == [("assistant", "stay_quiet", True, "leaked tool call")]


@pytest.mark.asyncio
async def test_a_leaked_tool_that_is_not_leak_safe_is_cut_and_never_run(monkeypatch: Any, caplog: Any) -> None:
    handler, tools = _leak_session(monkeypatch, _reply_events("Web search.", audio_first=True))
    reports = _capture_tool_reports(handler)

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await _run_with_leaks(handler)

    assert _frames(handler) == []
    assert "leaked tool call: web_search (spoken name) cut; executed=False" in caplog.messages
    assert tools["web_search"].runs == 0
    assert [(r["name"], r["status"], r["source"]) for r in reports] == [("web_search", "cut", "leaked")]


@pytest.mark.asyncio
async def test_a_single_plain_word_name_plays_whole_and_is_logged(monkeypatch: Any, caplog: Any) -> None:
    handler, tools = _leak_session(monkeypatch, _reply_events("Dance!", audio_first=False))
    reports = _capture_tool_reports(handler)
    seen: list[tuple[Any, ...]] = []
    handler.set_transcript_observer(lambda *args: seen.append(args))

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await _run_with_leaks(handler)

    assert [list(array.reshape(-1)) for _rate, array in _frames(handler)] == [[1, 2], [3, 4]]
    assert "possible leaked tool call: dance (plain word) played" in caplog.messages
    assert tools["dance"].runs == 0
    assert [(r["name"], r["status"], r["source"]) for r in reports] == [("dance", "played", "leaked")]
    assert seen == [("assistant", "Dance!", True, None)]


@pytest.mark.asyncio
async def test_a_spoken_name_split_across_deltas_waits_for_the_whole_reply(monkeypatch: Any) -> None:
    handler, tools = _leak_session(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _said("Stay"),
            _audio(1, 2),
            _said(" quiet."),
            _audio(3, 4),
            _FakeEvent("response.output_audio_transcript.done", transcript="Stay quiet."),
            _FakeEvent("response.done", response=SimpleNamespace(status="completed", output=[])),
        ),
    )

    await _run_with_leaks(handler)

    assert _frames(handler) == []
    assert tools["stay_quiet"].runs == 1


@pytest.mark.asyncio
async def test_a_reply_with_no_transcript_done_is_logged_and_still_falls_back(monkeypatch: Any, caplog: Any) -> None:
    """Measuring (c): the response.done fallback logs once, and plays as it always did."""
    handler, _tools = _leak_session(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _said("Hello there."),
            _audio(1, 2),
            _FakeEvent("response.done", response=SimpleNamespace(status="completed", output=[])),
        ),
    )

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await _run_with_leaks(handler)

    assert [list(array.reshape(-1)) for _rate, array in _frames(handler)] == [[1, 2]]
    fallbacks = [m for m in caplog.messages if m.startswith("output gate: reply judged at response.done")]
    assert fallbacks == ["output gate: reply judged at response.done with no transcript.done (12 chars, audio=True, gate=pass)"]


@pytest.mark.asyncio
async def test_a_reply_with_its_transcript_done_logs_no_fallback(monkeypatch: Any, caplog: Any) -> None:
    handler, _tools = _leak_session(monkeypatch, _reply_events("Hello there.", audio_first=True))

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await _run_with_leaks(handler)

    assert not [m for m in caplog.messages if m.startswith("output gate:")]


@pytest.mark.asyncio
async def test_a_sentence_that_starts_like_a_name_plays_whole(monkeypatch: Any) -> None:
    handler, _tools = _leak_session(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _said("Stay"),
            _audio(1, 2),
            _said(" warm tonight."),
            _audio(3, 4),
            _FakeEvent("response.output_audio_transcript.done", transcript="Stay warm tonight."),
            _FakeEvent("response.done", response=SimpleNamespace(status="completed", output=[])),
        ),
    )

    await _run_with_leaks(handler)

    assert [list(array.reshape(-1)) for _rate, array in _frames(handler)] == [[1, 2], [3, 4]]


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_first", [False, True])
async def test_a_normal_reply_plays_in_full(monkeypatch: Any, caplog: Any, audio_first: bool) -> None:
    """A reply with a bracket later on is not an artifact: every frame reaches the player, in order."""
    handler = _session_handler(monkeypatch, _reply_events("Sure, (I think) yes.", audio_first=audio_first))

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await handler._run_realtime_session()

    frames = _frames(handler)
    assert [list(array.reshape(-1)) for _rate, array in frames] == [[1, 2], [3, 4]]
    assert not [r for r in caplog.records if "spoken artifact cut" in r.getMessage()]


@pytest.mark.asyncio
async def test_audio_before_the_transcript_is_held_then_released(monkeypatch: Any) -> None:
    """Audio ahead of the first transcript delta waits; the delta that decides releases it."""
    sizes: list[int] = []
    handler: HuggingFaceRealtimeHandler
    events = (
        _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
        _audio(1, 2),
        lambda: sizes.append(handler.output_queue.qsize()),  # held
        _said("Sure"),
        lambda: sizes.append(handler.output_queue.qsize()),  # released
        _audio(3, 4),
        lambda: sizes.append(handler.output_queue.qsize()),  # passes straight through
    )
    handler = _session_handler(monkeypatch, events)

    await handler._run_realtime_session()

    assert sizes == [0, 1, 2]


@pytest.mark.asyncio
async def test_the_timeout_releases_audio_when_no_transcript_arrives(monkeypatch: Any) -> None:
    """With no transcript delta, held audio goes to the player after the hold timeout."""
    monkeypatch.setattr(hf_mod, "SPOKEN_ARTIFACT_HOLD_S", 0.05)
    sizes: list[int] = []
    handler: HuggingFaceRealtimeHandler
    events = (
        _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
        _audio(1, 2),
        lambda: sizes.append(handler.output_queue.qsize()),  # held
        lambda: asyncio.sleep(0.15),
        lambda: sizes.append(handler.output_queue.qsize()),  # released by the timeout
        _audio(3, 4),
        lambda: sizes.append(handler.output_queue.qsize()),  # passes straight through
    )
    handler = _session_handler(monkeypatch, events)

    await handler._run_realtime_session()

    assert sizes == [0, 1, 2]


@pytest.mark.asyncio
async def test_an_artifact_found_after_release_flushes_the_player(monkeypatch: Any, caplog: Any) -> None:
    """Audio released by the timeout, then a stage direction: the player is flushed, later audio dropped."""
    monkeypatch.setattr(hf_mod, "SPOKEN_ARTIFACT_HOLD_S", 0.05)
    handler = _session_handler(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _audio(1, 2),
            lambda: asyncio.sleep(0.15),
            _said("(stops talking)"),
            _audio(3, 4),
        ),
    )
    cleared: list[bool] = []
    handler._clear_queue = lambda: cleared.append(True)

    with caplog.at_level(logging.WARNING, logger=hf_mod.logger.name):
        await handler._run_realtime_session()

    assert cleared == [True]
    assert [list(a.reshape(-1)) for _r, a in _frames(handler)] == [[1, 2]]  # the fake flush drained nothing
    assert "spoken artifact cut (stage direction): '(stops talking)'" in caplog.text


@pytest.mark.asyncio
async def test_the_done_transcript_decides_when_no_delta_came(monkeypatch: Any) -> None:
    """A done-only backend: the terminal transcript cuts a stage direction still held."""
    handler = _session_handler(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _audio(1, 2),
            _FakeEvent("response.output_audio.done"),
            _FakeEvent("response.output_audio_transcript.done", transcript="(stays silent)"),
        ),
    )

    await handler._run_realtime_session()

    assert _frames(handler) == []


@pytest.mark.asyncio
async def test_a_barge_in_drops_held_audio(monkeypatch: Any) -> None:
    """Audio still held when the user starts speaking is never released afterwards."""
    handler = _session_handler(
        monkeypatch,
        (
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _audio(1, 2),
            _FakeEvent("input_audio_buffer.speech_started"),
            _said("Sure"),
        ),
    )
    handler._clear_queue = lambda: None

    await handler._run_realtime_session()

    assert _frames(handler) == []


@pytest.mark.asyncio
async def test_a_cut_reply_is_never_counted_unheard(monkeypatch: Any) -> None:
    """A cut reply is either never pushed or flushed like a barge-in: the unheard count stands."""
    from reachy_mini_conversation_app import playback_health as ph

    monkeypatch.setattr(hf_mod, "SPOKEN_ARTIFACT_HOLD_S", 0.05)
    tracker = ph.ReplyTracker(rebuilt_at=lambda: None)
    clock = {"now": 100.0}

    def push_released() -> None:
        # what the console does with each frame it takes from the queue
        for _rate, array in _frames(handler):
            tracker.pushed(clock["now"], array.size / handler.SAMPLE_RATE + 1.0, 1.0)

    handler: HuggingFaceRealtimeHandler
    handler = _session_handler(
        monkeypatch,
        (
            # held, then cut: never pushed
            *_reply_events("(stops talking)", audio_first=True),
            push_released,
            # released by the timeout, then cut: flushed
            _FakeEvent("response.created", response=SimpleNamespace(metadata=None)),
            _audio(1, 2),
            lambda: asyncio.sleep(0.15),
            push_released,
            _said("call:stay_quiet{}"),
        ),
    )
    handler._clear_queue = lambda: tracker.flushed(clock["now"] + 0.1)

    await handler._run_realtime_session()

    verdicts = [v.action for v in tracker.tick(clock["now"] + ph.UNHEARD_WINDOW_S + 1)]
    assert "unheard" not in verdicts
    assert verdicts == ["flushed"]
    assert tracker.unheard_in_a_row == 0


# --- function call outputs before response.create (2026-10-16) ---------------------


def _ordering_client(events: tuple[Any, ...], sent: list[tuple[str, Any]], handler: Any) -> Any:
    """Build a fake realtime client that records items and response.create calls in order."""

    async def _item_create(**kw: Any) -> None:
        sent.append(("item", kw["item"]))

    async def _response_create(**kw: Any) -> None:
        sent.append(("response.create", kw))
        handler._response_started_or_rejected_event.set()

    class _Conn:
        session = SimpleNamespace(update=AsyncMock())
        input_audio_buffer = SimpleNamespace(append=AsyncMock())
        conversation = SimpleNamespace(item=SimpleNamespace(create=_item_create))
        response = SimpleNamespace(create=_response_create, cancel=AsyncMock())

        def __init__(self) -> None:
            self._events = iter(events)

        async def __aenter__(self) -> "_Conn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        def __aiter__(self) -> "_Conn":
            return self

        async def __anext__(self) -> Any:
            while True:
                try:
                    item = next(self._events)
                except StopIteration:
                    raise StopAsyncIteration
                if callable(item):
                    result = item()
                    if asyncio.iscoroutine(result):
                        await result
                    continue
                return item

    return SimpleNamespace(realtime=SimpleNamespace(connect=lambda **_kw: _Conn()))


@pytest.mark.asyncio
async def test_no_response_create_while_a_function_call_output_is_owed(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A say() during a running tool waits for that tool's function_call_output.

    The robot's journal, 2026-10-06 to 10-16: "[error] Cannot create a response while
    function call outputs are pending." The server rejects an in-band response.create
    while any function call lacks its output.
    """
    sent: list[tuple[str, Any]] = []
    handler = _session_handler(monkeypatch, ())
    monkeypatch.setattr(
        type(handler.tool_manager), "start_tool", AsyncMock(return_value=SimpleNamespace(tool_id="dance-1"))
    )
    done = ToolNotification(
        id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": True}
    )

    async def _say() -> None:
        await handler.say("The timer is done.")
        await asyncio.sleep(0.05)  # the sender's chance to send too early

    async def _finish_tool() -> None:
        await handler._handle_tool_result(done)
        await asyncio.sleep(0.05)

    handler.client = _ordering_client(
        (
            _FakeEvent("response.function_call_arguments.done", name="dance", arguments="{}", call_id="c1"),
            _say,
            _finish_tool,
        ),
        sent,
        handler,
    )

    with caplog.at_level(logging.INFO, logger=hf_mod.__name__):
        await handler._run_realtime_session()

    kinds = [
        "output" if kind == "item" and payload.get("type") == "function_call_output" else kind
        for kind, payload in sent
    ]
    assert "response.create" in kinds
    assert kinds.index("output") < kinds.index("response.create")
    assert "response.create deferred" in caplog.text


@pytest.mark.asyncio
async def test_an_out_of_band_create_is_not_held_for_an_owed_output(monkeypatch: Any) -> None:
    """A filler is outside the conversation; the server never checks it."""
    sent: list[tuple[str, Any]] = []
    handler = _session_handler(monkeypatch, ())
    monkeypatch.setattr(
        type(handler.tool_manager), "start_tool", AsyncMock(return_value=SimpleNamespace(tool_id="dance-1"))
    )

    async def _filler() -> None:
        await handler._safe_response_create(
            response=hf_mod._out_of_band_request(hf_mod.FILLER_TAG, ["audio"], "Say: One moment.", [])
        )
        await asyncio.sleep(0.05)

    handler.client = _ordering_client(
        (
            _FakeEvent("response.function_call_arguments.done", name="dance", arguments="{}", call_id="c1"),
            _filler,
        ),
        sent,
        handler,
    )

    await handler._run_realtime_session()

    assert [kind for kind, _ in sent] == ["response.create"]


async def _line_during_a_search(monkeypatch: Any, speak: str) -> list[tuple[str, Any]]:
    """Run a session where `speak` ("say" or "type_text") lands while a web search runs."""
    sent: list[tuple[str, Any]] = []
    handler = _session_handler(monkeypatch, ())
    monkeypatch.setattr(
        type(handler.tool_manager), "start_tool", AsyncMock(return_value=SimpleNamespace(tool_id="search-1"))
    )
    done = ToolNotification(
        id="c1",
        tool_name="test__web_search",  # not a registered tool: its follow-up is always asked for
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
        result={"answer": "sunny"},
    )

    async def _line() -> None:
        await getattr(handler, speak)("The timer is done.")
        await asyncio.sleep(0.05)

    async def _finish_tool() -> None:
        await handler._handle_tool_result(done)
        await asyncio.sleep(0.05)

    handler.client = _ordering_client(
        (
            _FakeEvent("response.function_call_arguments.done", name="test__web_search", arguments="{}", call_id="c1"),
            _line,
            _finish_tool,
        ),
        sent,
        handler,
    )
    await handler._run_realtime_session()
    return sent


def _kinds(sent: list[tuple[str, Any]]) -> list[str]:
    """Name each sent event: output, line (the user item), filler, create, or item."""
    kinds = []
    for kind, payload in sent:
        if kind == "item":
            if payload.get("type") == "function_call_output":
                kinds.append("output")
            elif payload.get("role") == "user":
                kinds.append("line")
            else:
                kinds.append("item")
        elif payload.get("response", {}).get("conversation") == "none":
            kinds.append("filler")
        else:
            assert "_announcement" not in payload  # the marker never reaches the server
            kinds.append("create")
    return kinds


@pytest.mark.asyncio
async def test_an_announcement_during_a_search_is_its_own_reply_after_the_answer(monkeypatch: Any) -> None:
    """2026-10-16, 15:18: a timer's say() during a web_search; the follow-up voiced only the timer.

    The follow-up's create goes before the announcement's user item exists; then the
    item, then the announcement's own create.
    """
    kinds = _kinds(await _line_during_a_search(monkeypatch, "say"))

    assert kinds == ["filler", "output", "create", "line", "create"]


@pytest.mark.asyncio
async def test_a_typed_line_during_a_search_merges_with_the_answer(monkeypatch: Any) -> None:
    """A typed question and the search's result are answered together: one reply."""
    kinds = _kinds(await _line_during_a_search(monkeypatch, "type_text"))

    assert kinds == ["line", "filler", "output", "create"]


@pytest.mark.asyncio
async def test_a_say_with_no_tool_running_is_one_reply(monkeypatch: Any) -> None:
    """The plain announce path: the user item, then one create, nothing held."""
    sent: list[tuple[str, Any]] = []
    handler = _session_handler(monkeypatch, ())

    async def _say() -> None:
        await handler.say("The timer is done.")
        await asyncio.sleep(0.05)

    handler.client = _ordering_client((_say,), sent, handler)
    await handler._run_realtime_session()

    assert _kinds(sent) == ["line", "create"]
    assert sent[1][1] == {}


@pytest.mark.asyncio
async def test_a_held_create_is_released_after_the_timeout(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    """An output that never comes holds a create only up to the timeout."""
    monkeypatch.setattr(hf_mod, "_OWED_OUTPUT_TIMEOUT", 0)
    handler = _plain_handler()
    handler._owe_output("c1")

    with caplog.at_level(logging.WARNING, logger=hf_mod.__name__):
        held = await handler._wait_for_owed_outputs({})

    assert held is True
    assert "still owed" in caplog.text


@pytest.mark.asyncio
async def test_a_dropped_tool_result_settles_its_owed_output(monkeypatch: Any) -> None:
    """No response.done in time: the output is dropped, and creates are no longer held for it."""
    handler = _plain_handler()
    handler.connection = _FakeConnection()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=False))
    handler._owe_output("c1")
    note = ToolNotification(
        id="c1", tool_name="dance", is_idle_tool_call=False, status=ToolState.COMPLETED, result={"ok": 1}
    )

    await handler._handle_tool_result(note)

    handler.connection.conversation.item.create.assert_not_awaited()
    assert handler._owed_outputs == set()
    assert handler._no_outputs_owed.is_set()


@pytest.mark.asyncio
async def test_owed_outputs_reset_on_reconnect(monkeypatch: Any) -> None:
    """A call owed by the last session holds nothing in the next one."""
    seen: list[set[str]] = []
    handler = _session_handler(monkeypatch, (lambda: seen.append(set(handler._owed_outputs)),))
    handler._owe_output("stale")

    await handler._run_realtime_session()

    assert seen == [set()]
    assert handler._no_outputs_owed.is_set()


@pytest.mark.asyncio
async def test_a_played_reply_reaches_the_supervisor_without_its_call_markup(monkeypatch: Any) -> None:
    """2026-10-20: "<call:express{...}/>I'm sorry I'm being glitchy." plays whole, but the
    markup kept in his word-for-word memory taught him to leak more."""
    q = '<|"|>'
    text = f"<call:express{{feeling:{q}shy{q},intensity:0.4}}/>I'm sorry I'm being glitchy."
    handler = _session_handler(monkeypatch, _reply_events(text, audio_first=False))
    handler._clear_queue = lambda: None
    seen: list = []
    handler.set_transcript_observer(lambda role, said, final, artifact=None: seen.append((role, said, artifact)))

    await handler._run_realtime_session()

    assert _frames(handler) != []  # the audio plays as before
    assert seen == [("assistant", "I'm sorry I'm being glitchy.", None)]
