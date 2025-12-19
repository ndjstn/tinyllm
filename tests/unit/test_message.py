"""Tests for message types."""

import pytest
from tinyllm.core.message import (
    Message,
    MessagePayload,
    MessageMetadata,
    ToolCall,
    ToolResult,
    ErrorInfo,
    TaskPayload,
    TaskResponse,
)


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """Test basic message creation."""
        msg = Message(
            trace_id="trace-123",
            source_node="test_node",
            payload=MessagePayload(content="test content"),
        )

        assert msg.trace_id == "trace-123"
        assert msg.source_node == "test_node"
        assert msg.payload.content == "test content"
        assert msg.message_id is not None
        assert msg.parent_id is None

    def test_create_child_message(self):
        """Test creating child message."""
        parent = Message(
            trace_id="trace-123",
            source_node="parent_node",
            payload=MessagePayload(content="parent content"),
        )

        child = parent.create_child(
            source_node="child_node",
            target_node="target_node",
        )

        assert child.trace_id == parent.trace_id
        assert child.parent_id == parent.message_id
        assert child.source_node == "child_node"
        assert child.target_node == "target_node"

    def test_message_with_custom_payload(self):
        """Test message with custom payload."""
        child = Message(
            trace_id="trace-123",
            source_node="node",
            payload=MessagePayload(
                task="original task",
                route="code",
                confidence=0.95,
            ),
        )

        assert child.payload.route == "code"
        assert child.payload.confidence == 0.95


class TestMessagePayload:
    """Tests for MessagePayload class."""

    def test_empty_payload(self):
        """Test creating empty payload."""
        payload = MessagePayload()
        assert payload.task is None
        assert payload.content is None
        assert payload.route is None

    def test_payload_with_tool_call(self):
        """Test payload with tool call."""
        tool_call = ToolCall(
            tool_id="calculator",
            input={"expression": "2 + 2"},
        )
        payload = MessagePayload(tool_call=tool_call)

        assert payload.tool_call is not None
        assert payload.tool_call.tool_id == "calculator"
        assert payload.tool_call.call_id is not None


class TestToolCall:
    """Tests for ToolCall class."""

    def test_create_tool_call(self):
        """Test creating tool call."""
        call = ToolCall(
            tool_id="calculator",
            input={"expression": "sqrt(16)"},
        )

        assert call.tool_id == "calculator"
        assert call.input == {"expression": "sqrt(16)"}
        assert call.call_id is not None


class TestToolResult:
    """Tests for ToolResult class."""

    def test_successful_result(self):
        """Test successful tool result."""
        result = ToolResult(
            call_id="call-123",
            tool_id="calculator",
            success=True,
            output=4.0,
            latency_ms=10,
        )

        assert result.success is True
        assert result.output == 4.0
        assert result.error is None

    def test_failed_result(self):
        """Test failed tool result."""
        result = ToolResult(
            call_id="call-123",
            tool_id="calculator",
            success=False,
            error="Division by zero",
        )

        assert result.success is False
        assert result.error == "Division by zero"


class TestErrorInfo:
    """Tests for ErrorInfo class."""

    def test_create_error(self):
        """Test creating error info."""
        error = ErrorInfo(
            code=ErrorInfo.Codes.VALIDATION_ERROR,
            message="Invalid input",
            recoverable=True,
        )

        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid input"
        assert error.recoverable is True


class TestTaskPayload:
    """Tests for TaskPayload class."""

    def test_valid_payload(self):
        """Test valid task payload."""
        payload = TaskPayload(content="Test query")
        assert payload.content == "Test query"
        assert payload.context is None

    def test_payload_with_context(self):
        """Test payload with context."""
        payload = TaskPayload(
            content="Test query",
            context="Some context",
            options={"key": "value"},
        )
        assert payload.context == "Some context"
        assert payload.options == {"key": "value"}

    def test_empty_content_fails(self):
        """Test that empty content fails validation."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TaskPayload(content="")


class TestTaskResponse:
    """Tests for TaskResponse class."""

    def test_successful_response(self):
        """Test successful task response."""
        response = TaskResponse(
            trace_id="trace-123",
            success=True,
            content="Response content",
            total_latency_ms=500,
            nodes_executed=3,
        )

        assert response.success is True
        assert response.content == "Response content"
        assert response.error is None

    def test_failed_response(self):
        """Test failed task response."""
        response = TaskResponse(
            trace_id="trace-123",
            success=False,
            error=ErrorInfo(
                code=ErrorInfo.Codes.MODEL_ERROR,
                message="Model failed",
            ),
            total_latency_ms=100,
            nodes_executed=1,
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "MODEL_ERROR"
