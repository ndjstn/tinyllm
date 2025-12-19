"""Tool audit logging for TinyLLM.

This module provides comprehensive audit logging for tool executions,
tracking who did what, when, and with what result.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    EXECUTION_START = "execution_start"
    EXECUTION_SUCCESS = "execution_success"
    EXECUTION_FAILURE = "execution_failure"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    GUARD_BLOCKED = "guard_blocked"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    WARNING = "warning"


class AuditLevel(str, Enum):
    """Levels for audit logging."""

    MINIMAL = "minimal"  # Only failures
    STANDARD = "standard"  # Success and failures
    DETAILED = "detailed"  # All events with input/output
    FULL = "full"  # Everything including internal state


@dataclass
class AuditEvent:
    """An audit event."""

    id: str
    event_type: AuditEventType
    tool_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "tool_id": self.tool_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditSink(ABC):
    """Abstract base for audit sinks."""

    @abstractmethod
    def write(self, event: AuditEvent) -> None:
        """Write an audit event.

        Args:
            event: Event to write.
        """
        pass

    def flush(self) -> None:
        """Flush pending events."""
        pass

    def close(self) -> None:
        """Close the sink."""
        pass


class ConsoleSink(AuditSink):
    """Writes audit events to console."""

    def __init__(self, format_fn: Optional[Callable[[AuditEvent], str]] = None):
        """Initialize console sink.

        Args:
            format_fn: Custom format function.
        """
        self.format_fn = format_fn or self._default_format

    def _default_format(self, event: AuditEvent) -> str:
        """Default format for events."""
        status = "✓" if event.success else "✗"
        duration = f"{event.duration_ms:.1f}ms" if event.duration_ms else "N/A"
        return (
            f"[{status}] {event.timestamp.isoformat()} | "
            f"{event.event_type.value} | {event.tool_id} | {duration}"
        )

    def write(self, event: AuditEvent) -> None:
        """Write event to console."""
        print(self.format_fn(event))


class LoggingSink(AuditSink):
    """Writes audit events to Python logging."""

    def __init__(self, logger_name: str = "tinyllm.audit"):
        """Initialize logging sink.

        Args:
            logger_name: Logger name to use.
        """
        self.logger = logging.getLogger(logger_name)

    def write(self, event: AuditEvent) -> None:
        """Write event to logging."""
        level = logging.INFO if event.success else logging.WARNING
        self.logger.log(
            level,
            f"Audit: {event.event_type.value} - {event.tool_id}",
            extra={"audit_event": event.to_dict()},
        )


class FileSink(AuditSink):
    """Writes audit events to a file."""

    def __init__(self, file_path: str, append: bool = True):
        """Initialize file sink.

        Args:
            file_path: Path to audit file.
            append: Whether to append to existing file.
        """
        self.file_path = file_path
        mode = "a" if append else "w"
        self._file = open(file_path, mode)

    def write(self, event: AuditEvent) -> None:
        """Write event to file."""
        self._file.write(event.to_json() + "\n")

    def flush(self) -> None:
        """Flush file buffer."""
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        self._file.close()


class InMemorySink(AuditSink):
    """Stores audit events in memory."""

    def __init__(self, max_events: int = 10000):
        """Initialize in-memory sink.

        Args:
            max_events: Maximum events to store.
        """
        self.max_events = max_events
        self.events: List[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        """Store event in memory."""
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_events(
        self,
        tool_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        success: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Get stored events with filtering.

        Args:
            tool_id: Filter by tool ID.
            event_type: Filter by event type.
            success: Filter by success status.
            limit: Maximum events to return.

        Returns:
            Matching events.
        """
        result = self.events

        if tool_id:
            result = [e for e in result if e.tool_id == tool_id]

        if event_type:
            result = [e for e in result if e.event_type == event_type]

        if success is not None:
            result = [e for e in result if e.success == success]

        if limit:
            result = result[-limit:]

        return result

    def clear(self) -> None:
        """Clear stored events."""
        self.events.clear()


class CallbackSink(AuditSink):
    """Calls a callback for each event."""

    def __init__(self, callback: Callable[[AuditEvent], None]):
        """Initialize callback sink.

        Args:
            callback: Function to call for each event.
        """
        self.callback = callback

    def write(self, event: AuditEvent) -> None:
        """Call callback with event."""
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Audit callback error: {e}")


class MultiplexSink(AuditSink):
    """Writes to multiple sinks."""

    def __init__(self, sinks: Optional[List[AuditSink]] = None):
        """Initialize multiplex sink.

        Args:
            sinks: List of sinks to write to.
        """
        self.sinks: List[AuditSink] = sinks or []

    def add(self, sink: AuditSink) -> "MultiplexSink":
        """Add a sink.

        Args:
            sink: Sink to add.

        Returns:
            Self for chaining.
        """
        self.sinks.append(sink)
        return self

    def write(self, event: AuditEvent) -> None:
        """Write to all sinks."""
        for sink in self.sinks:
            try:
                sink.write(event)
            except Exception as e:
                logger.error(f"Audit sink error: {e}")

    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            sink.close()


class AuditLogger:
    """Main audit logger."""

    def __init__(
        self,
        sink: Optional[AuditSink] = None,
        level: AuditLevel = AuditLevel.STANDARD,
        max_input_length: int = 500,
        max_output_length: int = 500,
    ):
        """Initialize audit logger.

        Args:
            sink: Audit sink to use.
            level: Audit level.
            max_input_length: Max input summary length.
            max_output_length: Max output summary length.
        """
        self.sink = sink or InMemorySink()
        self.level = level
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self._session_id: Optional[str] = None
        self._user_id: Optional[str] = None

    def set_session(self, session_id: str) -> None:
        """Set the current session ID.

        Args:
            session_id: Session identifier.
        """
        self._session_id = session_id

    def set_user(self, user_id: str) -> None:
        """Set the current user ID.

        Args:
            user_id: User identifier.
        """
        self._user_id = user_id

    def _summarize(self, data: Any, max_length: int) -> str:
        """Create summary of data.

        Args:
            data: Data to summarize.
            max_length: Maximum length.

        Returns:
            Summarized string.
        """
        try:
            if hasattr(data, "model_dump"):
                text = json.dumps(data.model_dump(), default=str)
            elif isinstance(data, dict):
                text = json.dumps(data, default=str)
            else:
                text = str(data)

            if len(text) > max_length:
                return text[: max_length - 3] + "..."
            return text

        except Exception:
            return "<unable to summarize>"

    def log(
        self,
        event_type: AuditEventType,
        tool_id: str,
        success: bool = True,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event.
            tool_id: Tool identifier.
            success: Whether operation succeeded.
            input_data: Input data.
            output_data: Output data.
            duration_ms: Duration in milliseconds.
            error: Error message.
            metadata: Additional metadata.

        Returns:
            The created event.
        """
        # Check if we should log this event based on level
        if self.level == AuditLevel.MINIMAL and success:
            if event_type not in {
                AuditEventType.ERROR,
                AuditEventType.EXECUTION_FAILURE,
                AuditEventType.GUARD_BLOCKED,
            }:
                return None

        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            tool_id=tool_id,
            user_id=self._user_id,
            session_id=self._session_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )

        # Add input/output summaries based on level
        if self.level in {AuditLevel.DETAILED, AuditLevel.FULL}:
            if input_data is not None:
                event.input_summary = self._summarize(input_data, self.max_input_length)
            if output_data is not None:
                event.output_summary = self._summarize(
                    output_data, self.max_output_length
                )

        self.sink.write(event)
        return event

    def log_execution_start(
        self,
        tool_id: str,
        input_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log execution start.

        Args:
            tool_id: Tool identifier.
            input_data: Input data.
            metadata: Additional metadata.

        Returns:
            The created event.
        """
        return self.log(
            event_type=AuditEventType.EXECUTION_START,
            tool_id=tool_id,
            input_data=input_data,
            metadata=metadata,
        )

    def log_execution_success(
        self,
        tool_id: str,
        output_data: Any = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log successful execution.

        Args:
            tool_id: Tool identifier.
            output_data: Output data.
            duration_ms: Duration.
            metadata: Additional metadata.

        Returns:
            The created event.
        """
        return self.log(
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id=tool_id,
            success=True,
            output_data=output_data,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def log_execution_failure(
        self,
        tool_id: str,
        error: str,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log failed execution.

        Args:
            tool_id: Tool identifier.
            error: Error message.
            duration_ms: Duration.
            metadata: Additional metadata.

        Returns:
            The created event.
        """
        return self.log(
            event_type=AuditEventType.EXECUTION_FAILURE,
            tool_id=tool_id,
            success=False,
            error=error,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def log_guard_blocked(
        self,
        tool_id: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log guard block.

        Args:
            tool_id: Tool identifier.
            reason: Block reason.
            metadata: Additional metadata.

        Returns:
            The created event.
        """
        return self.log(
            event_type=AuditEventType.GUARD_BLOCKED,
            tool_id=tool_id,
            success=False,
            error=reason,
            metadata=metadata,
        )

    def flush(self) -> None:
        """Flush the sink."""
        self.sink.flush()

    def close(self) -> None:
        """Close the audit logger."""
        self.sink.close()


class AuditedToolWrapper:
    """Wrapper that adds audit logging to tools."""

    def __init__(
        self,
        tool: Any,
        audit_logger: Optional[AuditLogger] = None,
        log_input: bool = True,
        log_output: bool = True,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            audit_logger: Audit logger.
            log_input: Whether to log input.
            log_output: Whether to log output.
        """
        self.tool = tool
        self.audit_logger = audit_logger or AuditLogger()
        self.log_input = log_input
        self.log_output = log_output

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input_data: Any) -> Any:
        """Execute tool with audit logging.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        tool_id = self.tool.metadata.id
        start_time = time.time()

        # Log start
        self.audit_logger.log_execution_start(
            tool_id=tool_id,
            input_data=input_data if self.log_input else None,
        )

        try:
            result = await self.tool.execute(input_data)
            duration_ms = (time.time() - start_time) * 1000

            # Log success
            self.audit_logger.log_execution_success(
                tool_id=tool_id,
                output_data=result if self.log_output else None,
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log failure
            self.audit_logger.log_execution_failure(
                tool_id=tool_id,
                error=str(e),
                duration_ms=duration_ms,
            )

            raise


# Convenience functions


def with_audit(
    tool: Any,
    audit_logger: Optional[AuditLogger] = None,
    log_input: bool = True,
    log_output: bool = True,
) -> AuditedToolWrapper:
    """Add audit logging to a tool.

    Args:
        tool: Tool to wrap.
        audit_logger: Audit logger.
        log_input: Whether to log input.
        log_output: Whether to log output.

    Returns:
        AuditedToolWrapper.
    """
    return AuditedToolWrapper(
        tool,
        audit_logger=audit_logger,
        log_input=log_input,
        log_output=log_output,
    )


def create_audit_logger(
    sink: Optional[AuditSink] = None,
    level: AuditLevel = AuditLevel.STANDARD,
) -> AuditLogger:
    """Create an audit logger.

    Args:
        sink: Audit sink.
        level: Audit level.

    Returns:
        AuditLogger instance.
    """
    return AuditLogger(sink=sink, level=level)


def create_file_audit_logger(
    file_path: str,
    level: AuditLevel = AuditLevel.STANDARD,
) -> AuditLogger:
    """Create an audit logger that writes to file.

    Args:
        file_path: Path to audit file.
        level: Audit level.

    Returns:
        AuditLogger instance.
    """
    return AuditLogger(sink=FileSink(file_path), level=level)
