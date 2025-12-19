# Message Specification

## Overview

Messages are the data packets that flow between nodes in TinyLLM. Every piece of information passed through the graph is wrapped in a Message.

## Dependencies

- `pydantic>=2.0.0`
- `uuid` (standard library)
- `datetime` (standard library)

---

## Core Message Types

### Message

The primary data structure for inter-node communication.

```python
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Core message type for inter-node communication."""

    model_config = {"extra": "forbid"}

    # Identifiers
    trace_id: str = Field(description="Trace ID for the entire request")
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message identifier"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent message ID (for tracking lineage)"
    )

    # Routing
    source_node: str = Field(description="Node that created this message")
    target_node: Optional[str] = Field(
        default=None,
        description="Intended recipient node"
    )

    # Content
    payload: "MessagePayload" = Field(description="Message content")

    # Metadata
    metadata: "MessageMetadata" = Field(
        default_factory=lambda: MessageMetadata(),
        description="Execution metadata"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def create_child(
        self,
        source_node: str,
        target_node: Optional[str] = None,
        payload: Optional["MessagePayload"] = None
    ) -> "Message":
        """Create a child message inheriting trace_id."""
        return Message(
            trace_id=self.trace_id,
            parent_id=self.message_id,
            source_node=source_node,
            target_node=target_node,
            payload=payload or self.payload.model_copy(),
        )
```

### MessagePayload

The actual content of a message.

```python
class MessagePayload(BaseModel):
    """Content payload of a message."""

    model_config = {"extra": "allow"}  # Allow additional fields

    # Primary content
    task: Optional[str] = Field(
        default=None,
        description="Original task/query"
    )
    content: Optional[str] = Field(
        default=None,
        description="Text content (input or output)"
    )

    # Structured data
    structured: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured/parsed data"
    )

    # Routing info
    route: Optional[str] = Field(
        default=None,
        description="Selected route (from router)"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )

    # Tool interactions
    tool_call: Optional["ToolCall"] = Field(
        default=None,
        description="Tool call request"
    )
    tool_result: Optional["ToolResult"] = Field(
        default=None,
        description="Tool call result"
    )

    # Error info
    error: Optional["ErrorInfo"] = Field(
        default=None,
        description="Error information if failed"
    )
```

### MessageMetadata

Execution metadata attached to messages.

```python
class MessageMetadata(BaseModel):
    """Execution metadata for a message."""

    # Timing
    latency_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Processing time in milliseconds"
    )

    # Model info
    model_used: Optional[str] = Field(
        default=None,
        description="Ollama model used"
    )

    # Token counts
    tokens_in: Optional[int] = Field(
        default=None,
        ge=0,
        description="Input token count"
    )
    tokens_out: Optional[int] = Field(
        default=None,
        ge=0,
        description="Output token count"
    )

    # Quality
    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score from evaluation"
    )

    # Iteration tracking
    iteration: Optional[int] = Field(
        default=None,
        ge=0,
        description="Iteration number in loops"
    )
    retry_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of retries"
    )

    # Flags
    is_cached: bool = Field(
        default=False,
        description="Whether result was cached"
    )
```

---

## Supporting Types

### ToolCall

Request to invoke a tool.

```python
class ToolCall(BaseModel):
    """Request to invoke a tool."""

    tool_id: str = Field(description="ID of tool to invoke")
    input: Dict[str, Any] = Field(description="Tool input parameters")
    call_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique call identifier"
    )
```

### ToolResult

Result from a tool invocation.

```python
class ToolResult(BaseModel):
    """Result from a tool invocation."""

    call_id: str = Field(description="Matching call_id from ToolCall")
    tool_id: str = Field(description="Tool that was invoked")
    success: bool = Field(description="Whether tool succeeded")
    output: Optional[Any] = Field(
        default=None,
        description="Tool output on success"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message on failure"
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Tool execution time"
    )
```

### ErrorInfo

Structured error information.

```python
class ErrorInfo(BaseModel):
    """Structured error information."""

    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    recoverable: bool = Field(
        default=True,
        description="Whether error is recoverable"
    )

    class ErrorCodes:
        """Standard error codes."""
        VALIDATION_ERROR = "VALIDATION_ERROR"
        MODEL_ERROR = "MODEL_ERROR"
        TOOL_ERROR = "TOOL_ERROR"
        TIMEOUT = "TIMEOUT"
        ROUTING_ERROR = "ROUTING_ERROR"
        GATE_FAILED = "GATE_FAILED"
        UNKNOWN = "UNKNOWN"
```

---

## Input/Output Types

### TaskPayload

Initial input to the system.

```python
class TaskPayload(BaseModel):
    """Input payload for a task."""

    content: str = Field(
        description="The user's task or query",
        min_length=1,
        max_length=50000
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task options"
    )
```

### TaskResponse

Final output from the system.

```python
class TaskResponse(BaseModel):
    """Output response for a task."""

    trace_id: str = Field(description="Trace ID for debugging")
    success: bool = Field(description="Whether task completed successfully")
    content: Optional[str] = Field(
        default=None,
        description="Response content"
    )
    structured: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured response data"
    )
    error: Optional[ErrorInfo] = Field(
        default=None,
        description="Error info if failed"
    )

    # Metrics
    total_latency_ms: int = Field(description="Total processing time")
    nodes_executed: int = Field(description="Number of nodes executed")
    tokens_used: int = Field(
        default=0,
        description="Total tokens consumed"
    )
```

---

## Trace Types

### ExecutionTrace

Complete trace of an execution.

```python
class NodeExecution(BaseModel):
    """Record of a single node execution."""

    node_id: str
    node_type: str
    input_message_id: str
    output_message_ids: List[str]
    started_at: datetime
    completed_at: datetime
    latency_ms: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Complete execution trace."""

    trace_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = Field(pattern=r"^(running|completed|failed)$")

    input: TaskPayload
    output: Optional[TaskResponse] = None

    nodes_executed: List[NodeExecution] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)

    total_latency_ms: Optional[int] = None
    total_tokens: int = 0
```

---

## Usage Examples

### Creating an initial message

```python
# From user input
task = TaskPayload(content="Write a hello world function in Python")

initial_message = Message(
    trace_id=str(uuid4()),
    source_node="api",
    target_node="entry.main",
    payload=MessagePayload(task=task.content),
)
```

### Creating a child message

```python
# After processing
child = parent_message.create_child(
    source_node="router.task_type",
    target_node="specialist.code",
    payload=MessagePayload(
        task=parent_message.payload.task,
        route="code",
        confidence=0.95,
    ),
)
```

### Recording tool call and result

```python
# Tool call
message.payload.tool_call = ToolCall(
    tool_id="code_executor",
    input={"code": "print('hello')"},
)

# After tool execution
message.payload.tool_result = ToolResult(
    call_id=message.payload.tool_call.call_id,
    tool_id="code_executor",
    success=True,
    output={"stdout": "hello\n", "stderr": "", "exit_code": 0},
    latency_ms=150,
)
```

---

## File Location

`src/tinyllm/core/message.py`

---

## Test Cases

| Test | Input | Expected |
|------|-------|----------|
| Create message | Valid fields | Message instance |
| Create child | Parent message | Child with same trace_id, new message_id |
| Payload validation | content="test" | Valid payload |
| Tool call creation | tool_id, input | ToolCall with generated call_id |
| Tool result matching | call_id from ToolCall | Matching ToolResult |
| Error info creation | code, message | Valid ErrorInfo |
| Trace recording | Multiple executions | Complete trace |
