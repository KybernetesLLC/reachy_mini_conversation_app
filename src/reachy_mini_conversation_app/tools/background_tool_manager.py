"""Background tool orchestrator for non-blocking tool execution.

Allows tools to run long operations asynchronously while the robot
continues conversing. Tools can be tracked, cancelled, and their
completion is announced vocally via a silent notification queue.
"""

from __future__ import annotations
import json
import time
import asyncio
import logging
from typing import Any, Dict, Callable, Optional, Coroutine

from pydantic import Field, BaseModel, PrivateAttr

from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    dispatch_tool_call_with_manager,
)
from reachy_mini_conversation_app.tools.tool_constants import ToolState, SystemTool


logger = logging.getLogger(__name__)

_SYSTEM_TOOL_NAMES: set[str] = {t.value for t in SystemTool}


def _safe_load_args(args_json_str: str) -> Dict[str, Any]:
    """Best-effort parse of a call's raw JSON args, for logging only.

    Execution parses ``args_json_str`` itself (core_tools._safe_load_obj); this is
    a separate, independent parse so a logging failure can never affect dispatch.
    """
    try:
        parsed = json.loads(args_json_str or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class ToolProgress(BaseModel):
    """Progress of a background tool."""

    progress: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None


class ToolCallRoutine(BaseModel):
    """Encapsulates an async callable with its arguments for deferred execution."""

    model_config = {"arbitrary_types_allowed": True}

    tool_name: str
    args_json_str: str
    deps: "ToolDependencies"

    async def __call__(self, tool_manager: BackgroundToolManager) -> Any:
        """Execute the stored callable with its arguments."""
        if self.tool_name in _SYSTEM_TOOL_NAMES:
            # For safety purposes, we only allow system tools to be called with the tool manager
            return await dispatch_tool_call_with_manager(
                tool_name=self.tool_name, args_json=self.args_json_str, deps=self.deps, tool_manager=tool_manager
            )
        return await dispatch_tool_call(tool_name=self.tool_name, args_json=self.args_json_str, deps=self.deps)


class ToolNotification(BaseModel):
    """Notification payload for completed tools."""

    id: str
    tool_name: str
    is_idle_tool_call: bool
    status: ToolState
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Seconds from the call to its result (the companion's latency log, 2026-10-09).
    duration_s: Optional[float] = None
    # The tool-call-log brief (2026-10-11): the call's own arguments, for the
    # app's /rpc report; who asked for it (the user's turn, the missed-call
    # guard, or the model unprompted via the idle policy); and the response
    # or turn id, so a report can be tied back to the transcript.
    args: Dict[str, Any] = Field(default_factory=dict)
    source: str = "turn"
    turn_id: Optional[str] = None
    # The manager's session (counted by start_up) the call started in. A result from an
    # earlier session is dropped: that session's connection is gone (2026-10-23).
    session_index: int = 0


class BackgroundTool(ToolNotification):
    """Represents a background tool."""

    progress: Optional[ToolProgress] = None
    started_at: float = Field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    _task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)

    @property
    def tool_id(self) -> str:
        """Return a unique composite identifier for this tool instance."""
        return f"{self.tool_name}-{self.id}-{self.started_at}"

    def get_notification(self) -> ToolNotification:
        """Get the notification for the tool."""
        return ToolNotification(
            id=self.id,
            tool_name=self.tool_name,
            is_idle_tool_call=self.is_idle_tool_call,
            status=self.status,
            result=self.result,
            error=self.error,
            duration_s=(self.completed_at - self.started_at) if self.completed_at else None,
            args=self.args,
            source=self.source,
            turn_id=self.turn_id,
            session_index=self.session_index,
        )


class BackgroundToolManager(BaseModel):
    """Manages background tools for non-blocking tool execution.

    Features:
    - Start async tools without blocking the conversation
    - Track tool status and progress
    - Cancel running tools

    """

    _tools: Dict[str, BackgroundTool] = PrivateAttr(default_factory=dict)
    _notification_queue: asyncio.Queue[ToolNotification] = PrivateAttr(default_factory=asyncio.Queue)
    _loop: Optional[asyncio.AbstractEventLoop] = PrivateAttr(default=None)
    _lifecycle_tasks: list[asyncio.Task[None]] = PrivateAttr(default_factory=list)
    _max_tool_duration_seconds: float = PrivateAttr(default=86400)  # 1 day
    _max_tool_memory_seconds: float = PrivateAttr(default=3600)  # 1 hour
    _session_index: int = PrivateAttr(default=0)

    def set_loop(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Set the event loop.

        Args:
            loop: The event loop (defaults to current running loop)

        """
        if loop is not None:
            self._loop = loop
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        logger.debug("BackgroundToolManager: event loop set")

    async def start_tool(
        self,
        call_id: str,
        tool_call_routine: ToolCallRoutine,
        is_idle_tool_call: bool,
        with_progress: bool = False,
        source: str = "turn",
        turn_id: Optional[str] = None,
    ) -> BackgroundTool:
        """Start a new background tool.

        Args:
            call_id: The ID of the tool
            tool_call_routine: The ToolCallRoutine containing the callable and its arguments
            with_progress: Whether to track progress (0.0-1.0)
            is_idle_tool_call: Whether the tool call was triggered by an idle signal
            source: Who asked for this call -- "turn" (the user's turn, the default),
                "missed_call_guard" (the reliable-search nudge), or "idle" (the local
                idle policy, unprompted). Carried through to the completion notification
                for the tool-call-log brief's /rpc report; never used for dispatch.
            turn_id: The realtime response id this call belongs to, if any. Also
                carried through unchanged, for the same report.

        Returns:
            BackgroundTool object with tool ID

        """
        tool_name = tool_call_routine.tool_name
        id = call_id
        background_tool = BackgroundTool(
            id=id,
            tool_name=tool_name,
            is_idle_tool_call=is_idle_tool_call,
            progress=ToolProgress(progress=0.0) if with_progress else None,
            status=ToolState.RUNNING,
            args=_safe_load_args(tool_call_routine.args_json_str),
            source=source,
            turn_id=turn_id,
            session_index=self._session_index,
        )
        self._tools[background_tool.tool_id] = background_tool

        async_task = asyncio.create_task(
            self._run_tool(background_tool, tool_call_routine),
            name=f"bg-{tool_name}-{id}",
        )
        background_tool._task = async_task

        logger.info(f"Started background tool: {background_tool.tool_name} (id={id})")

        return background_tool

    async def _run_tool(
        self,
        background_tool: BackgroundTool,
        tool_call_routine: ToolCallRoutine,
    ) -> None:
        """Execute the tool and handle completion."""
        result: dict[str, Any] = await tool_call_routine(self)
        background_tool.completed_at = time.monotonic()
        error = result.get("error")

        if error is not None:
            if error == "Tool cancelled":
                background_tool.status = ToolState.CANCELLED
                logger.debug(f"Background tool cancelled: {background_tool.tool_name} (id={background_tool.id})")
            else:
                background_tool.status = ToolState.FAILED
                logger.debug(
                    f"Background tool failed: {background_tool.tool_name} (id={background_tool.id}): {background_tool.error}"
                )
            background_tool.error = result["error"]

        else:
            background_tool.result = result
            background_tool.status = ToolState.COMPLETED
            logger.debug(f"Background tool completed: {background_tool.tool_name} (id={background_tool.id})")

        await self._notification_queue.put(background_tool.get_notification())
        logger.debug(f"Queued notification for tool: {background_tool.tool_name} (id={background_tool.id})")

    async def update_progress(
        self,
        tool_id: str,
        progress: float,
        message: Optional[str] = None,
    ) -> bool:
        """Update progress for a tool (for tools with with_progress=True).

        Args:
            tool_id: The tool ID
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message (e.g., "50% downloaded")

        Returns:
            True if updated successfully, False if tool not found or not tracking progress

        """
        tool = self._tools.get(tool_id)
        if tool is None:
            return False

        if tool.progress is None:
            # Tool not tracking progress
            return False

        tool.progress = ToolProgress(progress=max(0.0, min(1.0, progress)), message=message)
        logger.debug(f"Tool {tool_id} progress: {progress:.1%} - {message or ''}")
        return True

    async def cancel_tool(self, tool_id: str, log: bool = True) -> bool:
        """Cancel a running tool by ID.

        Args:
            tool_id: The tool ID to cancel
            log: Whether to log the cancellation

        Returns:
            True if cancelled, False if tool not found or not running

        """
        tool = self._tools.get(tool_id)
        if tool is None:
            if log:
                logger.warning(f"Cannot cancel tool {tool_id}: not found")
            return False

        if tool.status != ToolState.RUNNING:
            if log:
                logger.warning(f"Cannot cancel tool {tool_id}: status is {tool.status.value}")
            return True

        if tool._task:
            tool._task.cancel()
            if log:
                logger.info(f"Cancelled tool: {tool.tool_name} (id={tool_id})")
            return True

        return False

    def start_up(
        self,
        tool_callbacks: list[Callable[[ToolNotification], Coroutine[Any, Any, None]]],
        dropped_callbacks: Optional[list[Callable[[ToolNotification], None]]] = None,
    ) -> None:
        """Start the background tool manager.

        This method starts two concurrent tasks:
        - _listener: Listens for completed BackgroundTool notifications and calls the callbacks.
        - _cleanup: Cleans up completed/failed/cancelled tools that have been in memory for too long and times out tools that have been running too long.

        Args:
            tool_callbacks: A list of async or sync callables that receive the completed BackgroundTool notifications.
            dropped_callbacks: Callables that receive a notification from an earlier session, which is
                dropped instead of handled (so it can still be reported).

        """
        self.set_loop()
        # shutdown() cancels running tools, but each still queues its "Tool cancelled"
        # result, possibly after this start_up. Tagging beats draining the queue here.
        self._session_index += 1

        async def _listener() -> None:
            while True:
                background_tool = await self._notification_queue.get()
                if background_tool.session_index != self._session_index:
                    logger.info(
                        "dropped a tool result from a closed session: %s (%s)",
                        background_tool.tool_name,
                        background_tool.id,
                    )
                    for dropped in dropped_callbacks or []:
                        dropped(background_tool)
                    continue
                for callback in tool_callbacks:
                    await callback(background_tool)

        async def _cleanup(interval_seconds: float = 5 * 60) -> None:
            while True:
                await asyncio.sleep(interval_seconds)
                await self.cleanup_tools()
                await self.timeout_tools()

        self._lifecycle_tasks = [
            asyncio.create_task(_cleanup(), name="bg-tool-cleanup"),
            asyncio.create_task(_listener(), name="bg-tool-listener-callback"),
        ]

        logger.info(
            "BackgroundToolManager started. "
            "Max tool execution duration: %s seconds (tools running longer will be auto-cancelled). "
            "Max tool memory retention: %s seconds (completed/failed/cancelled tools older than this are purged).",
            self._max_tool_duration_seconds,
            self._max_tool_memory_seconds,
        )

    async def shutdown(self) -> None:
        """Cancel all background tasks (listener, cleanup) and running tools."""
        for task in self._lifecycle_tasks:
            task.cancel()
        for task in self._lifecycle_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._lifecycle_tasks.clear()

        for tool_id in list(self._tools):
            await self.cancel_tool(tool_id, log=False)

        logger.info("BackgroundToolManager shut down")

    async def timeout_tools(self) -> int:
        """Cancel tools that have been running too long.

        Returns:
            Number of tools cancelled

        """
        now = time.monotonic()
        to_cancel = []

        for tool_id, tool in self._tools.items():
            if tool.status == ToolState.RUNNING:
                if tool.started_at and (now - tool.started_at) > self._max_tool_duration_seconds:
                    to_cancel.append(tool_id)

        for tool_id in to_cancel:
            await self.cancel_tool(tool_id)

        if to_cancel:
            logger.debug(f"Timed out {len(to_cancel)} tools")

        return len(to_cancel)

    async def cleanup_tools(self) -> int:
        """Remove completed/failed/cancelled tools that have been in memory for too long.

        Returns:
            Number of tools removed

        """
        now = time.monotonic()
        to_remove = []

        for tool_id, tool in self._tools.items():
            if tool.status in (ToolState.COMPLETED, ToolState.FAILED, ToolState.CANCELLED):
                if tool.completed_at and (now - tool.completed_at) > self._max_tool_memory_seconds:
                    to_remove.append(tool_id)

        for tool_id in to_remove:
            del self._tools[tool_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tools")

        return len(to_remove)

    def get_tool(self, tool_id: str) -> Optional[BackgroundTool]:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def get_running_tools(self) -> list[BackgroundTool]:
        """Get all currently running tools."""
        return [t for t in self._tools.values() if t.status == ToolState.RUNNING]

    def get_all_tools(self, limit: Optional[int] = None) -> list[BackgroundTool]:
        """Get recent tools (most recent first).

        Args:
            limit: Maximum number of tools to return (None means all)

        Returns:
            List of tools sorted by start time (most recent first)

        """
        sorted_tools = sorted(
            self._tools.values(),
            key=lambda t: t.started_at,
            reverse=True,
        )
        if limit is not None:
            return sorted_tools[:limit]
        return sorted_tools
