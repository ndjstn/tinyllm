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
