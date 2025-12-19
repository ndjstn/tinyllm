"""Trace recording for TinyLLM execution.

This module provides trace recording for debugging and observability,
tracking all node executions and messages during graph execution.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.core.message import Message, TaskPayload, TaskResponse
from tinyllm.core.node import NodeResult


class NodeExecution(BaseModel):
    """Record of a single node execution."""

    model_config = {"extra": "forbid"}

    node_id: str = Field(description="Node that was executed")
    node_type: str = Field(description="Type of the node")
    input_message_id: str = Field(description="Input message ID")
    output_message_ids: List[str] = Field(
        default_factory=list, description="Output message IDs"
    )
    started_at: datetime = Field(description="When execution started")
    completed_at: Optional[datetime] = Field(
        default=None, description="When execution completed"
    )
    latency_ms: int = Field(default=0, ge=0, description="Execution time in ms")
    success: bool = Field(description="Whether execution succeeded")
    error: Optional[str] = Field(default=None, description="Error if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ExecutionTrace(BaseModel):
    """Complete execution trace for a request."""

    model_config = {"extra": "forbid"}

    trace_id: str = Field(description="Unique trace identifier")
    graph_id: str = Field(description="Graph that was executed")
    started_at: datetime = Field(description="When execution started")
    completed_at: Optional[datetime] = Field(
        default=None, description="When execution completed"
    )
    status: str = Field(
        default="running", description="running, completed, or failed"
    )

    input: TaskPayload = Field(description="Original input")
    output: Optional[TaskResponse] = Field(
        default=None, description="Final output"
    )

    nodes_executed: List[NodeExecution] = Field(
        default_factory=list, description="All node executions"
    )
    messages: List[Message] = Field(
        default_factory=list, description="All messages"
    )

    total_latency_ms: Optional[int] = Field(
        default=None, description="Total execution time"
    )
    total_tokens_in: int = Field(default=0, description="Total input tokens")
    total_tokens_out: int = Field(default=0, description="Total output tokens")

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_tokens_in + self.total_tokens_out

    @property
    def node_count(self) -> int:
        """Get number of nodes executed."""
        return len(self.nodes_executed)


class TraceRecorder:
    """Records execution trace during graph execution.

    Usage:
        recorder = TraceRecorder(trace_id, graph_id, input_payload)
        recorder.start_node("router.main", "router", message)
        # ... execute node ...
        recorder.complete_node("router.main", result)
        trace = recorder.complete(response)
    """

    def __init__(self, trace_id: str, graph_id: str, input_payload: TaskPayload):
        """Initialize trace recorder.

        Args:
            trace_id: Unique trace identifier.
            graph_id: Graph being executed.
            input_payload: Original input payload.
        """
        self._trace = ExecutionTrace(
            trace_id=trace_id,
            graph_id=graph_id,
            started_at=datetime.utcnow(),
            input=input_payload,
        )
        self._pending_nodes: Dict[str, NodeExecution] = {}

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace.trace_id

    def start_node(
        self, node_id: str, node_type: str, input_message: Message
    ) -> None:
        """Record start of node execution.

        Args:
            node_id: Node being executed.
            node_type: Type of the node.
            input_message: Input message to the node.
        """
        execution = NodeExecution(
            node_id=node_id,
            node_type=node_type,
            input_message_id=input_message.message_id,
            started_at=datetime.utcnow(),
            success=False,  # Will be updated on completion
        )
        self._pending_nodes[node_id] = execution

    def complete_node(self, node_id: str, result: NodeResult) -> None:
        """Record completion of node execution.

        Args:
            node_id: Node that completed.
            result: Result of the execution.
        """
        execution = self._pending_nodes.pop(node_id, None)
        if execution is None:
            # Node wasn't started - create a record anyway
            execution = NodeExecution(
                node_id=node_id,
                node_type="unknown",
                input_message_id="unknown",
                started_at=datetime.utcnow(),
                success=result.success,
            )

        execution.completed_at = datetime.utcnow()
        execution.success = result.success
        execution.error = result.error
        execution.latency_ms = result.latency_ms
        execution.output_message_ids = [m.message_id for m in result.output_messages]
        execution.metadata = result.metadata

        self._trace.nodes_executed.append(execution)

    def add_message(self, message: Message) -> None:
        """Add a message to the trace.

        Args:
            message: Message to record.
        """
        self._trace.messages.append(message)

    def add_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Add to token counts.

        Args:
            tokens_in: Input tokens.
            tokens_out: Output tokens.
        """
        self._trace.total_tokens_in += tokens_in
        self._trace.total_tokens_out += tokens_out

    def fail(self, error: str) -> ExecutionTrace:
        """Mark execution as failed.

        Args:
            error: Error description.

        Returns:
            Completed trace.
        """
        self._trace.completed_at = datetime.utcnow()
        self._trace.status = "failed"
        self._trace.total_latency_ms = int(
            (self._trace.completed_at - self._trace.started_at).total_seconds() * 1000
        )
        return self._trace

    def complete(self, response: TaskResponse) -> ExecutionTrace:
        """Complete the trace with final response.

        Args:
            response: Final task response.

        Returns:
            Completed trace.
        """
        self._trace.completed_at = datetime.utcnow()
        self._trace.output = response
        self._trace.status = "completed" if response.success else "failed"
        self._trace.total_latency_ms = int(
            (self._trace.completed_at - self._trace.started_at).total_seconds() * 1000
        )
        return self._trace

    def get_trace(self) -> ExecutionTrace:
        """Get the current trace state.

        Returns:
            Current trace.
        """
        return self._trace
