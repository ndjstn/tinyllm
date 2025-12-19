"""Tests for the structured event system."""

import pytest

from tinyllm.events import (
    Event,
    EventCategory,
    EventEmitter,
    EventSeverity,
    BufferedEventHandler,
    LogEventHandler,
    emit_event,
    emit_system_event,
    emit_error_event,
    get_event_emitter,
)


class TestEvent:
    """Test Event dataclass."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            event_type="test.event",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Test event",
        )

        assert event.event_type == "test.event"
        assert event.category == EventCategory.SYSTEM
        assert event.severity == EventSeverity.INFO
        assert event.message == "Test event"
        assert event.event_id is not None
        assert event.timestamp > 0

    def test_event_with_data(self):
        """Test event with additional data."""
        event = Event(
            event_type="test.event",
            category=EventCategory.EXECUTION,
            severity=EventSeverity.INFO,
            message="Test",
            data={"key": "value"},
            tags=["test", "example"],
        )

        assert event.data == {"key": "value"}
        assert event.tags == ["test", "example"]

    def test_event_to_dict(self):
        """Test event serialization."""
        event = Event(
            event_type="test.event",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.WARNING,
            message="Test",
            data={"foo": "bar"},
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "test.event"
        assert event_dict["category"] == "system"
        assert event_dict["severity"] == "warning"
        assert event_dict["message"] == "Test"
        assert event_dict["data"] == {"foo": "bar"}


class TestEventEmitter:
    """Test EventEmitter."""

    def test_event_emitter_creation(self):
        """Test creating an event emitter."""
        emitter = EventEmitter()
        assert emitter._enabled is True
        assert len(emitter.handlers) == 0

    def test_add_remove_handler(self):
        """Test adding and removing handlers."""
        emitter = EventEmitter()
        handler = LogEventHandler()

        emitter.add_handler(handler)
        assert len(emitter.handlers) == 1

        emitter.remove_handler(handler)
        assert len(emitter.handlers) == 0

    def test_emit_event(self):
        """Test emitting an event."""
        emitter = EventEmitter()
        buffer = BufferedEventHandler(max_size=10)
        emitter.add_handler(buffer)

        event = Event(
            event_type="test.event",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Test",
        )

        emitter.emit(event)

        assert len(buffer.events) == 1
        assert buffer.events[0].event_type == "test.event"

    def test_disable_enable(self):
        """Test disabling and enabling emitter."""
        emitter = EventEmitter()
        buffer = BufferedEventHandler()
        emitter.add_handler(buffer)

        # Emit while enabled
        event = Event(
            event_type="test.event",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Test",
        )
        emitter.emit(event)
        assert len(buffer.events) == 1

        # Disable and emit
        emitter.disable()
        emitter.emit(event)
        assert len(buffer.events) == 1  # Still 1

        # Re-enable and emit
        emitter.enable()
        emitter.emit(event)
        assert len(buffer.events) == 2


class TestBufferedEventHandler:
    """Test BufferedEventHandler."""

    def test_buffer_events(self):
        """Test buffering events."""
        buffer = BufferedEventHandler(max_size=10)

        for i in range(5):
            event = Event(
                event_type=f"test.event.{i}",
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                message=f"Event {i}",
            )
            buffer.handle(event)

        assert len(buffer.events) == 5

    def test_max_size_limit(self):
        """Test that buffer respects max size."""
        buffer = BufferedEventHandler(max_size=3)

        for i in range(5):
            event = Event(
                event_type=f"test.event.{i}",
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                message=f"Event {i}",
            )
            buffer.handle(event)

        assert len(buffer.events) == 3
        # Should keep the most recent 3
        assert buffer.events[0].event_type == "test.event.2"
        assert buffer.events[1].event_type == "test.event.3"
        assert buffer.events[2].event_type == "test.event.4"

    def test_get_events_filter(self):
        """Test filtering buffered events."""
        buffer = BufferedEventHandler()

        # Add events with different categories and severities
        buffer.handle(
            Event(
                event_type="test.1",
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                message="Test 1",
            )
        )
        buffer.handle(
            Event(
                event_type="test.2",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.ERROR,
                message="Test 2",
            )
        )
        buffer.handle(
            Event(
                event_type="test.3",
                category=EventCategory.SYSTEM,
                severity=EventSeverity.ERROR,
                message="Test 3",
            )
        )

        # Filter by category
        system_events = buffer.get_events(category=EventCategory.SYSTEM)
        assert len(system_events) == 2

        # Filter by severity
        error_events = buffer.get_events(severity=EventSeverity.ERROR)
        assert len(error_events) == 2

        # Filter by both
        system_errors = buffer.get_events(
            category=EventCategory.SYSTEM, severity=EventSeverity.ERROR
        )
        assert len(system_errors) == 1
        assert system_errors[0].event_type == "test.3"

    def test_clear(self):
        """Test clearing buffered events."""
        buffer = BufferedEventHandler()

        buffer.handle(
            Event(
                event_type="test.event",
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                message="Test",
            )
        )

        assert len(buffer.events) == 1

        buffer.clear()
        assert len(buffer.events) == 0


class TestConvenienceFunctions:
    """Test convenience functions for emitting events."""

    def test_emit_event(self):
        """Test emit_event function."""
        emitter = get_event_emitter()
        buffer = BufferedEventHandler()
        emitter.add_handler(buffer)

        event = emit_event(
            event_type="test.event",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Test event",
            data={"key": "value"},
        )

        assert event.event_type == "test.event"
        assert len(buffer.events) >= 1

        # Clean up
        emitter.remove_handler(buffer)

    def test_emit_system_event(self):
        """Test emit_system_event convenience function."""
        emitter = get_event_emitter()
        buffer = BufferedEventHandler()
        emitter.add_handler(buffer)

        event = emit_system_event("System started", data={"version": "1.0"})

        assert event.category == EventCategory.SYSTEM
        assert event.severity == EventSeverity.INFO

        # Clean up
        emitter.remove_handler(buffer)

    def test_emit_error_event(self):
        """Test emit_error_event convenience function."""
        emitter = get_event_emitter()
        buffer = BufferedEventHandler()
        emitter.add_handler(buffer)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            event = emit_error_event(
                message="An error occurred", error=e, category=EventCategory.EXECUTION
            )

        assert event.severity == EventSeverity.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "Test error"
        assert event.stack_trace is not None

        # Clean up
        emitter.remove_handler(buffer)


class TestStructuredEventLogging:
    """Test structured event logging features (Task 48)."""

    def test_event_trace_correlation_injection(self, monkeypatch):
        """Test that events automatically inherit trace context."""
        import sys
        from unittest.mock import MagicMock

        # Setup mock telemetry
        mock_telemetry = MagicMock()
        mock_telemetry.get_current_trace_id = lambda: "trace-abc-123"
        mock_telemetry.get_current_span_id = lambda: "span-xyz-456"
        sys.modules['tinyllm.telemetry'] = mock_telemetry

        try:
            emitter = EventEmitter()
            buffer = BufferedEventHandler()
            emitter.add_handler(buffer)

            event = Event(
                event_type="test.correlated",
                category=EventCategory.EXECUTION,
                severity=EventSeverity.INFO,
                message="Test event with correlation",
            )

            emitter.emit(event)

            # Event should have trace context injected
            buffered_event = buffer.events[-1]
            assert buffered_event.trace_id == "trace-abc-123"
            assert buffered_event.span_id == "span-xyz-456"
        finally:
            if 'tinyllm.telemetry' in sys.modules:
                del sys.modules['tinyllm.telemetry']

    def test_event_serialization_with_all_fields(self):
        """Test event serialization includes all optional fields."""
        event = Event(
            event_type="test.full",
            category=EventCategory.SECURITY,
            severity=EventSeverity.CRITICAL,
            message="Full event",
            data={"key": "value"},
            tags=["important", "security"],
            trace_id="trace-123",
            span_id="span-456",
            user_id="user-789",
            session_id="session-abc",
            error_type="SecurityError",
            error_message="Unauthorized access",
            stack_trace="stack trace here",
        )

        event_dict = event.to_dict()

        assert event_dict["trace_id"] == "trace-123"
        assert event_dict["span_id"] == "span-456"
        assert event_dict["user_id"] == "user-789"
        assert event_dict["session_id"] == "session-abc"
        assert "error" in event_dict
        assert event_dict["error"]["type"] == "SecurityError"
        assert event_dict["error"]["message"] == "Unauthorized access"

    def test_event_categories_comprehensive(self):
        """Test all event categories are available."""
        categories = [
            EventCategory.SYSTEM,
            EventCategory.EXECUTION,
            EventCategory.MODEL,
            EventCategory.CACHE,
            EventCategory.SECURITY,
            EventCategory.PERFORMANCE,
            EventCategory.USER,
            EventCategory.INTEGRATION,
            EventCategory.DATA,
        ]

        for category in categories:
            event = Event(
                event_type=f"test.{category.value}",
                category=category,
                severity=EventSeverity.INFO,
                message=f"Test {category.value}",
            )
            assert event.category == category

    def test_event_severities_comprehensive(self):
        """Test all event severities are available."""
        severities = [
            EventSeverity.DEBUG,
            EventSeverity.INFO,
            EventSeverity.WARNING,
            EventSeverity.ERROR,
            EventSeverity.CRITICAL,
        ]

        for severity in severities:
            event = Event(
                event_type=f"test.{severity.value}",
                category=EventCategory.SYSTEM,
                severity=severity,
                message=f"Test {severity.value}",
            )
            assert event.severity == severity

    def test_buffered_handler_limit_parameter(self):
        """Test buffered handler respects limit parameter."""
        buffer = BufferedEventHandler()

        for i in range(20):
            event = Event(
                event_type=f"test.{i}",
                category=EventCategory.SYSTEM,
                severity=EventSeverity.INFO,
                message=f"Event {i}",
            )
            buffer.handle(event)

        # Get limited events
        limited = buffer.get_events(limit=5)
        assert len(limited) == 5
        # Should get the last 5
        assert limited[0].event_type == "test.15"
        assert limited[-1].event_type == "test.19"

    def test_metric_event_handler_integration(self):
        """Test that MetricEventHandler records events as metrics."""
        from tinyllm.events import MetricEventHandler

        handler = MetricEventHandler()
        event = Event(
            event_type="test.metrics",
            category=EventCategory.PERFORMANCE,
            severity=EventSeverity.WARNING,
            message="Performance issue detected",
        )

        # Should not raise
        handler.handle(event)

    def test_event_emitter_handler_error_resilience(self):
        """Test that emitter continues if one handler fails."""
        from tinyllm.events import EventHandler

        class FailingHandler(EventHandler):
            def handle(self, event):
                raise RuntimeError("Handler failed")

        emitter = EventEmitter()
        buffer = BufferedEventHandler()
        failing = FailingHandler()

        emitter.add_handler(failing)
        emitter.add_handler(buffer)

        event = Event(
            event_type="test.resilience",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Test",
        )

        # Should not raise even though one handler fails
        emitter.emit(event)

        # Buffer should still receive the event
        assert len(buffer.events) == 1

    def test_event_immutability_after_creation(self):
        """Test that event fields can be set after creation."""
        event = Event(
            event_type="test.mutable",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Test",
        )

        # Can set trace_id after creation
        event.trace_id = "new-trace-id"
        assert event.trace_id == "new-trace-id"

    def test_structured_event_with_nested_data(self):
        """Test events can handle complex nested data structures."""
        complex_data = {
            "user": {
                "id": "user-123",
                "roles": ["admin", "developer"],
                "metadata": {"team": "engineering", "level": 5},
            },
            "action": "deploy",
            "targets": ["prod-1", "prod-2"],
        }

        event = Event(
            event_type="deployment.started",
            category=EventCategory.SYSTEM,
            severity=EventSeverity.INFO,
            message="Deployment initiated",
            data=complex_data,
        )

        assert event.data["user"]["roles"] == ["admin", "developer"]
        assert event.data["targets"] == ["prod-1", "prod-2"]

        event_dict = event.to_dict()
        assert event_dict["data"]["user"]["metadata"]["team"] == "engineering"
