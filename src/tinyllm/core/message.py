"""Message types for inter-node communication.

This module defines the core data structures used to pass information
between nodes in the TinyLLM graph.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Request to invoke a tool."""

    tool_id: str = Field(description="ID of tool to invoke")
    input: Dict[str, Any] = Field(description="Tool input parameters")
    call_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique call identifier",
    )


class ToolResult(BaseModel):
    """Result from a tool invocation."""

    call_id: str = Field(description="Matching call_id from ToolCall")
    tool_id: str = Field(description="Tool that was invoked")
    success: bool = Field(description="Whether tool succeeded")
    output: Optional[Any] = Field(default=None, description="Tool output on success")
    error: Optional[str] = Field(default=None, description="Error message on failure")
    latency_ms: int = Field(default=0, ge=0, description="Tool execution time")


class ErrorInfo(BaseModel):
    """Structured error information."""

    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    recoverable: bool = Field(default=True, description="Whether error is recoverable")

    class Codes:
        """Standard error codes."""

        VALIDATION_ERROR = "VALIDATION_ERROR"
        MODEL_ERROR = "MODEL_ERROR"
        TOOL_ERROR = "TOOL_ERROR"
        TIMEOUT = "TIMEOUT"
        ROUTING_ERROR = "ROUTING_ERROR"
        GATE_FAILED = "GATE_FAILED"
        UNKNOWN = "UNKNOWN"


class MessagePayload(BaseModel):
    """Content payload of a message."""

    model_config = {"extra": "allow"}

    # Primary content
    task: Optional[str] = Field(default=None, description="Original task/query")
    content: Optional[str] = Field(
        default=None, description="Text content (input or output)"
    )

    # Structured data
    structured: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured/parsed data"
    )

    # Routing info
    route: Optional[str] = Field(default=None, description="Selected route (from router)")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score"
    )

    # Tool interactions
    tool_call: Optional[ToolCall] = Field(
        default=None, description="Tool call request"
    )
    tool_result: Optional[ToolResult] = Field(
        default=None, description="Tool call result"
    )

    # Error info
    error: Optional[ErrorInfo] = Field(
        default=None, description="Error information if failed"
    )


class MessageMetadata(BaseModel):
    """Execution metadata for a message."""

    # Timing
    latency_ms: Optional[int] = Field(
        default=None, ge=0, description="Processing time in milliseconds"
    )

    # Model info
    model_used: Optional[str] = Field(default=None, description="Ollama model used")

    # Token counts
    tokens_in: Optional[int] = Field(default=None, ge=0, description="Input token count")
    tokens_out: Optional[int] = Field(
        default=None, ge=0, description="Output token count"
    )

    # Quality
    quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Quality score from evaluation"
    )

    # Iteration tracking
    iteration: Optional[int] = Field(
        default=None, ge=0, description="Iteration number in loops"
    )
    retry_count: Optional[int] = Field(default=None, ge=0, description="Number of retries")

    # Flags
    is_cached: bool = Field(default=False, description="Whether result was cached")


class Message(BaseModel):
    """Core message type for inter-node communication."""

    model_config = {"extra": "forbid"}

    # Identifiers
    trace_id: str = Field(description="Trace ID for the entire request")
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message identifier",
    )
    parent_id: Optional[str] = Field(
        default=None, description="Parent message ID (for tracking lineage)"
    )

    # Routing
    source_node: str = Field(description="Node that created this message")
    target_node: Optional[str] = Field(
        default=None, description="Intended recipient node"
    )

    # Content
    payload: MessagePayload = Field(description="Message content")

    # Metadata
    metadata: MessageMetadata = Field(
        default_factory=MessageMetadata, description="Execution metadata"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def create_child(
        self,
        source_node: str,
        target_node: Optional[str] = None,
        payload: Optional[MessagePayload] = None,
    ) -> "Message":
        """Create a child message inheriting trace_id."""
        return Message(
            trace_id=self.trace_id,
            parent_id=self.message_id,
            source_node=source_node,
            target_node=target_node,
            payload=payload or self.payload.model_copy(),
        )


class TaskPayload(BaseModel):
    """Input payload for a task."""

    content: str = Field(
        description="The user's task or query", min_length=1, max_length=50000
    )
    context: Optional[str] = Field(default=None, description="Additional context")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Task options")


class TaskResponse(BaseModel):
    """Output response for a task."""

    trace_id: str = Field(description="Trace ID for debugging")
    success: bool = Field(description="Whether task completed successfully")
    content: Optional[str] = Field(default=None, description="Response content")
    structured: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured response data"
    )
    error: Optional[ErrorInfo] = Field(default=None, description="Error info if failed")

    # Metrics
    total_latency_ms: int = Field(description="Total processing time")
    nodes_executed: int = Field(description="Number of nodes executed")
    tokens_used: int = Field(default=0, description="Total tokens consumed")
