"""Tests for tool audit logging."""

import pytest
import tempfile
import os
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.audit import (
    AuditedToolWrapper,
    AuditEvent,
    AuditEventType,
    AuditLevel,
    AuditLogger,
    CallbackSink,
    ConsoleSink,
    FileSink,
    InMemorySink,
    LoggingSink,
    MultiplexSink,
    create_audit_logger,
    create_file_audit_logger,
    with_audit,
)


class AuditInput(BaseModel):
    """Input for audit tests."""

    value: int = 0


class AuditOutput(BaseModel):
    """Output for audit tests."""

    result: int = 0
    success: bool = True


class SuccessTool(BaseTool[AuditInput, AuditOutput]):
    """Tool that succeeds."""

    metadata = ToolMetadata(
        id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        category="utility",
    )
    input_type = AuditInput
    output_type = AuditOutput

    async def execute(self, input: AuditInput) -> AuditOutput:
        return AuditOutput(result=input.value * 2)


class FailTool(BaseTool[AuditInput, AuditOutput]):
    """Tool that fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = AuditInput
    output_type = AuditOutput

    async def execute(self, input: AuditInput) -> AuditOutput:
        raise ValueError("Intentional failure")


class TestAuditEvent:
    """Tests for AuditEvent."""

    def test_creation(self):
        """Test event creation."""
        event = AuditEvent(
            id="test-123",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="my_tool",
        )

        assert event.id == "test-123"
        assert event.tool_id == "my_tool"
        assert event.success is True

    def test_to_dict(self):
        """Test converting to dictionary."""
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_FAILURE,
            tool_id="tool",
            user_id="user1",
            error="Something failed",
        )

        d = event.to_dict()

        assert d["id"] == "test"
        assert d["event_type"] == "execution_failure"
        assert d["user_id"] == "user1"
        assert d["error"] == "Something failed"

    def test_to_json(self):
        """Test converting to JSON."""
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
        )

        json_str = event.to_json()

        assert '"id": "test"' in json_str
        assert "execution_success" in json_str


class TestConsoleSink:
    """Tests for ConsoleSink."""

    def test_write(self, capsys):
        """Test writing to console."""
        sink = ConsoleSink()
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="my_tool",
        )

        sink.write(event)

        captured = capsys.readouterr()
        assert "my_tool" in captured.out
        assert "✓" in captured.out

    def test_custom_format(self, capsys):
        """Test custom format function."""

        def custom_format(event):
            return f"CUSTOM: {event.tool_id}"

        sink = ConsoleSink(format_fn=custom_format)
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="my_tool",
        )

        sink.write(event)

        captured = capsys.readouterr()
        assert "CUSTOM: my_tool" in captured.out


class TestLoggingSink:
    """Tests for LoggingSink."""

    def test_write(self, caplog):
        """Test writing to logging."""
        import logging

        with caplog.at_level(logging.INFO, logger="tinyllm.audit"):
            sink = LoggingSink()
            event = AuditEvent(
                id="test",
                event_type=AuditEventType.EXECUTION_SUCCESS,
                tool_id="my_tool",
            )

            sink.write(event)

            assert "my_tool" in caplog.text


class TestFileSink:
    """Tests for FileSink."""

    def test_write(self):
        """Test writing to file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            file_path = f.name

        try:
            sink = FileSink(file_path)
            event = AuditEvent(
                id="test",
                event_type=AuditEventType.EXECUTION_SUCCESS,
                tool_id="my_tool",
            )

            sink.write(event)
            sink.flush()
            sink.close()

            with open(file_path) as f:
                content = f.read()

            assert "my_tool" in content
            assert "test" in content

        finally:
            os.unlink(file_path)


class TestInMemorySink:
    """Tests for InMemorySink."""

    def test_write(self):
        """Test writing to memory."""
        sink = InMemorySink()
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="my_tool",
        )

        sink.write(event)

        assert len(sink.events) == 1
        assert sink.events[0].id == "test"

    def test_get_events_filter_tool_id(self):
        """Test filtering by tool ID."""
        sink = InMemorySink()
        sink.write(
            AuditEvent(id="1", event_type=AuditEventType.EXECUTION_SUCCESS, tool_id="tool1")
        )
        sink.write(
            AuditEvent(id="2", event_type=AuditEventType.EXECUTION_SUCCESS, tool_id="tool2")
        )
        sink.write(
            AuditEvent(id="3", event_type=AuditEventType.EXECUTION_SUCCESS, tool_id="tool1")
        )

        events = sink.get_events(tool_id="tool1")

        assert len(events) == 2

    def test_get_events_filter_success(self):
        """Test filtering by success."""
        sink = InMemorySink()
        sink.write(
            AuditEvent(
                id="1",
                event_type=AuditEventType.EXECUTION_SUCCESS,
                tool_id="tool",
                success=True,
            )
        )
        sink.write(
            AuditEvent(
                id="2",
                event_type=AuditEventType.EXECUTION_FAILURE,
                tool_id="tool",
                success=False,
            )
        )

        failures = sink.get_events(success=False)

        assert len(failures) == 1

    def test_max_events(self):
        """Test max events limit."""
        sink = InMemorySink(max_events=5)

        for i in range(10):
            sink.write(
                AuditEvent(
                    id=str(i),
                    event_type=AuditEventType.EXECUTION_SUCCESS,
                    tool_id="tool",
                )
            )

        assert len(sink.events) == 5

    def test_clear(self):
        """Test clearing events."""
        sink = InMemorySink()
        sink.write(
            AuditEvent(id="1", event_type=AuditEventType.EXECUTION_SUCCESS, tool_id="tool")
        )

        sink.clear()

        assert len(sink.events) == 0


class TestCallbackSink:
    """Tests for CallbackSink."""

    def test_callback_called(self):
        """Test callback is called."""
        events = []

        def callback(event):
            events.append(event)

        sink = CallbackSink(callback)
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
        )

        sink.write(event)

        assert len(events) == 1
        assert events[0].id == "test"

    def test_callback_error_handling(self):
        """Test error handling in callback."""

        def failing_callback(event):
            raise ValueError("Callback error")

        sink = CallbackSink(failing_callback)
        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
        )

        # Should not raise
        sink.write(event)


class TestMultiplexSink:
    """Tests for MultiplexSink."""

    def test_writes_to_all(self):
        """Test writing to all sinks."""
        sink1 = InMemorySink()
        sink2 = InMemorySink()
        multiplex = MultiplexSink(sinks=[sink1, sink2])

        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
        )

        multiplex.write(event)

        assert len(sink1.events) == 1
        assert len(sink2.events) == 1

    def test_add_sink(self):
        """Test adding sinks."""
        sink1 = InMemorySink()
        sink2 = InMemorySink()

        multiplex = MultiplexSink().add(sink1).add(sink2)

        event = AuditEvent(
            id="test",
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
        )

        multiplex.write(event)

        assert len(sink1.events) == 1
        assert len(sink2.events) == 1


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_log_event(self):
        """Test logging an event."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink)

        logger.log(
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="my_tool",
        )

        assert len(sink.events) == 1
        assert sink.events[0].tool_id == "my_tool"

    def test_log_with_session_and_user(self):
        """Test logging with session and user."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink)
        logger.set_session("session-123")
        logger.set_user("user-456")

        logger.log(
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
        )

        assert sink.events[0].session_id == "session-123"
        assert sink.events[0].user_id == "user-456"

    def test_log_execution_start(self):
        """Test log_execution_start."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink, level=AuditLevel.DETAILED)

        logger.log_execution_start("tool", input_data={"value": 5})

        assert sink.events[0].event_type == AuditEventType.EXECUTION_START
        assert "value" in sink.events[0].input_summary

    def test_log_execution_success(self):
        """Test log_execution_success."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink)

        logger.log_execution_success("tool", duration_ms=100.5)

        assert sink.events[0].event_type == AuditEventType.EXECUTION_SUCCESS
        assert sink.events[0].duration_ms == 100.5

    def test_log_execution_failure(self):
        """Test log_execution_failure."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink)

        logger.log_execution_failure("tool", error="Something failed")

        assert sink.events[0].event_type == AuditEventType.EXECUTION_FAILURE
        assert sink.events[0].error == "Something failed"
        assert sink.events[0].success is False

    def test_log_guard_blocked(self):
        """Test log_guard_blocked."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink)

        logger.log_guard_blocked("tool", reason="Dangerous pattern")

        assert sink.events[0].event_type == AuditEventType.GUARD_BLOCKED

    def test_minimal_level_filters_success(self):
        """Test minimal level filters success events."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink, level=AuditLevel.MINIMAL)

        logger.log_execution_success("tool")
        logger.log_execution_failure("tool", error="Error")

        # Only failure should be logged
        assert len(sink.events) == 1
        assert sink.events[0].event_type == AuditEventType.EXECUTION_FAILURE

    def test_detailed_level_includes_data(self):
        """Test detailed level includes input/output."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink, level=AuditLevel.DETAILED)

        logger.log(
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
            input_data={"key": "value"},
            output_data={"result": 42},
        )

        assert sink.events[0].input_summary is not None
        assert sink.events[0].output_summary is not None

    def test_truncates_long_data(self):
        """Test truncation of long data."""
        sink = InMemorySink()
        logger = AuditLogger(sink=sink, level=AuditLevel.DETAILED, max_input_length=50)

        logger.log(
            event_type=AuditEventType.EXECUTION_SUCCESS,
            tool_id="tool",
            input_data={"data": "x" * 1000},
        )

        assert len(sink.events[0].input_summary) <= 50


class TestAuditedToolWrapper:
    """Tests for AuditedToolWrapper."""

    @pytest.mark.asyncio
    async def test_success_logged(self):
        """Test successful execution is logged."""
        sink = InMemorySink()
        audit_logger = AuditLogger(sink=sink)
        wrapper = AuditedToolWrapper(SuccessTool(), audit_logger=audit_logger)

        result = await wrapper.execute(AuditInput(value=5))

        assert result.result == 10
        assert len(sink.events) == 2  # Start and success

        event_types = [e.event_type for e in sink.events]
        assert AuditEventType.EXECUTION_START in event_types
        assert AuditEventType.EXECUTION_SUCCESS in event_types

    @pytest.mark.asyncio
    async def test_failure_logged(self):
        """Test failed execution is logged."""
        sink = InMemorySink()
        audit_logger = AuditLogger(sink=sink)
        wrapper = AuditedToolWrapper(FailTool(), audit_logger=audit_logger)

        with pytest.raises(ValueError):
            await wrapper.execute(AuditInput(value=5))

        event_types = [e.event_type for e in sink.events]
        assert AuditEventType.EXECUTION_START in event_types
        assert AuditEventType.EXECUTION_FAILURE in event_types

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        """Test duration is recorded."""
        sink = InMemorySink()
        audit_logger = AuditLogger(sink=sink)
        wrapper = AuditedToolWrapper(SuccessTool(), audit_logger=audit_logger)

        await wrapper.execute(AuditInput(value=5))

        success_event = [e for e in sink.events if e.event_type == AuditEventType.EXECUTION_SUCCESS][0]
        assert success_event.duration_ms is not None
        assert success_event.duration_ms > 0

    @pytest.mark.asyncio
    async def test_log_input_disabled(self):
        """Test disabling input logging."""
        sink = InMemorySink()
        audit_logger = AuditLogger(sink=sink, level=AuditLevel.DETAILED)
        wrapper = AuditedToolWrapper(
            SuccessTool(),
            audit_logger=audit_logger,
            log_input=False,
        )

        await wrapper.execute(AuditInput(value=5))

        start_event = [e for e in sink.events if e.event_type == AuditEventType.EXECUTION_START][0]
        assert start_event.input_summary is None

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        wrapper = AuditedToolWrapper(SuccessTool())

        assert wrapper.metadata.id == "success_tool"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_audit(self):
        """Test with_audit function."""
        wrapper = with_audit(SuccessTool())

        result = await wrapper.execute(AuditInput(value=5))

        assert result.result == 10

    def test_create_audit_logger(self):
        """Test create_audit_logger function."""
        sink = InMemorySink()
        logger = create_audit_logger(sink=sink, level=AuditLevel.DETAILED)

        assert logger.level == AuditLevel.DETAILED

    def test_create_file_audit_logger(self):
        """Test create_file_audit_logger function."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            file_path = f.name

        try:
            logger = create_file_audit_logger(file_path)
            logger.log(
                event_type=AuditEventType.EXECUTION_SUCCESS,
                tool_id="test",
            )
            logger.close()

            with open(file_path) as f:
                content = f.read()

            assert "test" in content

        finally:
            os.unlink(file_path)
