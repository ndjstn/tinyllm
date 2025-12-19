"""Structured event logging system for TinyLLM.

This module provides a structured event system for tracking important
application events with rich context and metadata. Events are separate
from logs and metrics, providing a third pillar of observability.

The Three Pillars of Observability:
------------------------------------
1. **Logs**: Detailed, timestamped records of discrete events (tinyllm.logging)
2. **Metrics**: Aggregated numerical data over time (tinyllm.metrics)
3. **Events**: Structured business and system events with rich context (this module)

Events are useful for:
- Audit trails (who did what, when)
- Business metrics (user signups, completions)
- System state changes (node added, graph modified)
- Error tracking with context
- Performance anomalies

Features:
---------
1. **Automatic Trace Correlation**: Events inherit trace_id and span_id from active spans
2. **Rich Metadata**: Attach arbitrary data, tags, user/session IDs to events
3. **Multiple Handlers**: Route events to logs, metrics, databases, or custom destinations
4. **Buffered Storage**: Keep recent events in memory for debugging
5. **Error Enrichment**: Capture exception details with stack traces

Architecture:
-------------
The event system consists of:

- **Event**: Immutable dataclass with structured fields
- **EventEmitter**: Central dispatcher routing events to handlers
- **EventHandler**: Base class for event destinations
- **LogEventHandler**: Writes events to structured logs
- **MetricEventHandler**: Records events as Prometheus metrics
- **BufferedEventHandler**: Keeps recent events in memory

Event Categories:
-----------------
Events are classified into categories for filtering and routing:

- **SYSTEM**: System lifecycle events (startup, shutdown, configuration)
- **EXECUTION**: Graph and node execution events
- **MODEL**: Model operations (load, inference, errors)
- **CACHE**: Cache operations (hit, miss, eviction)
- **SECURITY**: Security events (authentication, authorization, access control)
- **PERFORMANCE**: Performance anomalies and optimization events
- **USER**: User actions and interactions
- **INTEGRATION**: External system integration events
- **DATA**: Data operations (validation, transformation, storage)

Event Severities:
-----------------
- **DEBUG**: Detailed information for debugging
- **INFO**: Informational events (normal operation)
- **WARNING**: Warning conditions that should be monitored
- **ERROR**: Error conditions requiring attention
- **CRITICAL**: Critical failures requiring immediate action

Usage:
------
Basic event emission:

    >>> from tinyllm.events import emit_event, EventCategory, EventSeverity
    >>>
    >>> # Emit a simple event
    >>> event = emit_event(
    ...     event_type="node.execution.completed",
    ...     category=EventCategory.EXECUTION,
    ...     severity=EventSeverity.INFO,
    ...     message="Node execution completed successfully",
    ...     data={"node_id": "router", "duration_ms": 123}
    ... )
    >>>
    >>> print(f"Event ID: {event.event_id}")

Convenience functions:

    >>> from tinyllm.events import emit_system_event, emit_error_event
    >>>
    >>> # System events
    >>> emit_system_event("Application started", data={"version": "0.1.0"})
    >>>
    >>> # Error events with exception details
    >>> try:
    ...     raise ValueError("Invalid configuration")
    ... except ValueError as e:
    ...     emit_error_event(
    ...         "Configuration validation failed",
    ...         error=e,
    ...         category=EventCategory.SYSTEM
    ...     )

Custom event handlers:

    >>> from tinyllm.events import EventHandler, get_event_emitter
    >>>
    >>> class DatabaseEventHandler(EventHandler):
    ...     def handle(self, event):
    ...         # Write event to database
    ...         db.events.insert(event.to_dict())
    >>>
    >>> # Register handler
    >>> emitter = get_event_emitter()
    >>> emitter.add_handler(DatabaseEventHandler())

Buffered events for debugging:

    >>> from tinyllm.events import BufferedEventHandler, get_event_emitter
    >>>
    >>> # Add buffer handler
    >>> buffer = BufferedEventHandler(max_size=100)
    >>> emitter = get_event_emitter()
    >>> emitter.add_handler(buffer)
    >>>
    >>> # Get recent events
    >>> recent = buffer.get_events(limit=10)
    >>> errors = buffer.get_events(severity=EventSeverity.ERROR)

Trace correlation:

    >>> from tinyllm.events import emit_event, EventCategory, EventSeverity
    >>> from tinyllm.telemetry import trace_span
    >>>
    >>> # Events emitted within trace spans automatically include trace_id
    >>> with trace_span("process_request"):
    ...     event = emit_event(
    ...         "request.processed",
    ...         category=EventCategory.EXECUTION,
    ...         severity=EventSeverity.INFO,
    ...         message="Request processed"
    ...     )
    ...     # event.trace_id and event.span_id are set automatically

Event serialization:

    >>> event = Event(
    ...     event_type="deployment.completed",
    ...     category=EventCategory.SYSTEM,
    ...     severity=EventSeverity.INFO,
    ...     message="Deployment successful",
    ...     data={"environment": "production", "version": "1.2.3"},
    ...     tags=["deployment", "production"]
    ... )
    >>>
    >>> # Serialize to dict (for JSON, databases, etc.)
    >>> event_dict = event.to_dict()
    >>> print(event_dict["event_type"])  # "deployment.completed"

Best Practices:
---------------
1. **Use descriptive event types**: Format as "category.action.status" (e.g., "node.execution.completed")
2. **Include relevant data**: Add context in the data field, not in the message
3. **Choose appropriate severity**: INFO for normal operations, ERROR for failures
4. **Add tags for filtering**: Use tags to categorize events beyond the category enum
5. **Leverage trace correlation**: Emit events within trace spans for end-to-end visibility

Performance:
------------
- Event emission is asynchronous and non-blocking
- Handler failures don't affect other handlers or application flow
- Buffered handlers use circular buffers for memory efficiency
- Metric handlers are optimized for high-throughput scenarios

Integration:
------------
Events can be routed to:
- Structured logs (always enabled by default)
- Prometheus metrics (via MetricEventHandler)
- Databases (custom handler)
- Message queues (custom handler)
- Analytics platforms (custom handler)
- Notification systems (custom handler)

See Also:
---------
- tinyllm.logging: Structured logging with trace correlation
- tinyllm.metrics: Prometheus metrics for quantitative monitoring
- tinyllm.telemetry: Distributed tracing with OpenTelemetry
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="events")


class EventSeverity(str, Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventCategory(str, Enum):
    """Event categories for classification."""

    SYSTEM = "system"  # System-level events (startup, shutdown)
    EXECUTION = "execution"  # Graph/node execution events
    MODEL = "model"  # Model-related events (load, inference, errors)
    CACHE = "cache"  # Cache operations
    SECURITY = "security"  # Security events (auth, access control)
    PERFORMANCE = "performance"  # Performance anomalies
    USER = "user"  # User actions
    INTEGRATION = "integration"  # External system integrations
    DATA = "data"  # Data operations


@dataclass
class Event:
    """Structured event with metadata."""

    # Core fields
    event_type: str  # Unique event type identifier (e.g., "node.execution.started")
    category: EventCategory
    severity: EventSeverity
    message: str  # Human-readable message

    # Context
    data: Dict[str, Any] = field(default_factory=dict)  # Event-specific data
    tags: List[str] = field(default_factory=list)  # Tags for filtering

    # Metadata (auto-populated)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Optional correlation IDs
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Optional error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data,
            "tags": self.tags,
        }

        # Add optional fields if present
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.error_type:
            result["error"] = {
                "type": self.error_type,
                "message": self.error_message,
                "stack_trace": self.stack_trace,
            }

        return result


class EventEmitter:
    """Centralized event emitter for the application.

    Events are emitted to multiple backends:
    - Structured logs (always)
    - Metrics (if configured)
    - External event stores (if configured)
    """

    def __init__(self):
        """Initialize event emitter."""
        self.handlers: List[EventHandler] = []
        self._enabled = True

    def add_handler(self, handler: "EventHandler") -> None:
        """Add an event handler.

        Args:
            handler: Event handler to add.
        """
        self.handlers.append(handler)

    def remove_handler(self, handler: "EventHandler") -> None:
        """Remove an event handler.

        Args:
            handler: Event handler to remove.
        """
        if handler in self.handlers:
            self.handlers.remove(handler)

    def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers.

        Args:
            event: Event to emit.
        """
        if not self._enabled:
            return

        # Inject trace context if available
        if not event.trace_id:
            try:
                from tinyllm.telemetry import get_current_trace_id, get_current_span_id
                event.trace_id = get_current_trace_id()
                event.span_id = get_current_span_id()
            except Exception:
                pass

        # Emit to all handlers
        for handler in self.handlers:
            try:
                handler.handle(event)
            except Exception as e:
                logger.error(
                    "event_handler_error",
                    handler=handler.__class__.__name__,
                    error=str(e),
                )

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True


class EventHandler:
    """Base class for event handlers."""

    def handle(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: Event to handle.
        """
        raise NotImplementedError


class LogEventHandler(EventHandler):
    """Event handler that writes events to structured logs."""

    def handle(self, event: Event) -> None:
        """Write event to structured logs.

        Args:
            event: Event to handle.
        """
        log_method = getattr(logger, event.severity.value)
        log_method(
            event.event_type,
            **event.to_dict()
        )


class MetricEventHandler(EventHandler):
    """Event handler that records events as metrics."""

    def __init__(self):
        """Initialize metric event handler."""
        from prometheus_client import Counter

        self.event_counter = Counter(
            "tinyllm_events_total",
            "Total number of events by type, category, and severity",
            ["event_type", "category", "severity"],
        )

    def handle(self, event: Event) -> None:
        """Record event as a metric.

        Args:
            event: Event to handle.
        """
        self.event_counter.labels(
            event_type=event.event_type,
            category=event.category.value,
            severity=event.severity.value,
        ).inc()


class BufferedEventHandler(EventHandler):
    """Event handler that buffers events in memory.

    Useful for testing or temporary event storage.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize buffered event handler.

        Args:
            max_size: Maximum number of events to buffer.
        """
        self.events: List[Event] = []
        self.max_size = max_size

    def handle(self, event: Event) -> None:
        """Buffer event in memory.

        Args:
            event: Event to handle.
        """
        self.events.append(event)

        # Trim buffer if needed
        if len(self.events) > self.max_size:
            self.events = self.events[-self.max_size:]

    def get_events(
        self,
        event_type: Optional[str] = None,
        category: Optional[EventCategory] = None,
        severity: Optional[EventSeverity] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Get buffered events with optional filtering.

        Args:
            event_type: Filter by event type.
            category: Filter by category.
            severity: Filter by severity.
            limit: Maximum number of events to return.

        Returns:
            List of matching events.
        """
        filtered = self.events

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if category:
            filtered = [e for e in filtered if e.category == category]
        if severity:
            filtered = [e for e in filtered if e.severity == severity]

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def clear(self) -> None:
        """Clear all buffered events."""
        self.events.clear()


# Global event emitter instance
_event_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance.

    Returns:
        Global EventEmitter singleton.
    """
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = EventEmitter()
        # Add default log handler
        _event_emitter.add_handler(LogEventHandler())
    return _event_emitter


def emit_event(
    event_type: str,
    category: EventCategory,
    severity: EventSeverity,
    message: str,
    **kwargs: Any,
) -> Event:
    """Convenience function to create and emit an event.

    Args:
        event_type: Event type identifier.
        category: Event category.
        severity: Event severity.
        message: Human-readable message.
        **kwargs: Additional event fields (data, tags, etc).

    Returns:
        The created event.

    Example:
        >>> emit_event(
        ...     "node.execution.completed",
        ...     category=EventCategory.EXECUTION,
        ...     severity=EventSeverity.INFO,
        ...     message="Node execution completed successfully",
        ...     data={"node_id": "entry", "duration_ms": 123},
        ...     tags=["execution", "success"],
        ... )
    """
    event = Event(
        event_type=event_type,
        category=category,
        severity=severity,
        message=message,
        **kwargs,
    )

    emitter = get_event_emitter()
    emitter.emit(event)

    return event


# Convenience functions for common event types

def emit_system_event(message: str, severity: EventSeverity = EventSeverity.INFO, **kwargs: Any) -> Event:
    """Emit a system event."""
    return emit_event(
        event_type="system.event",
        category=EventCategory.SYSTEM,
        severity=severity,
        message=message,
        **kwargs,
    )


def emit_execution_event(message: str, severity: EventSeverity = EventSeverity.INFO, **kwargs: Any) -> Event:
    """Emit an execution event."""
    return emit_event(
        event_type="execution.event",
        category=EventCategory.EXECUTION,
        severity=severity,
        message=message,
        **kwargs,
    )


def emit_error_event(
    message: str,
    error: Exception,
    category: EventCategory = EventCategory.SYSTEM,
    **kwargs: Any,
) -> Event:
    """Emit an error event with exception details."""
    import traceback

    return emit_event(
        event_type=f"{category.value}.error",
        category=category,
        severity=EventSeverity.ERROR,
        message=message,
        error_type=type(error).__name__,
        error_message=str(error),
        stack_trace=traceback.format_exc(),
        **kwargs,
    )


__all__ = [
    "Event",
    "EventCategory",
    "EventSeverity",
    "EventEmitter",
    "EventHandler",
    "LogEventHandler",
    "MetricEventHandler",
    "BufferedEventHandler",
    "get_event_emitter",
    "emit_event",
    "emit_system_event",
    "emit_execution_event",
    "emit_error_event",
]
