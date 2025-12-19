"""Graph executor for TinyLLM.

This module provides the Executor class that runs messages through
a graph, managing state and producing traces.
"""

import asyncio
import time
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.graph import Graph
from tinyllm.core.message import (
    ErrorInfo,
    Message,
    MessageMetadata,
    MessagePayload,
    TaskPayload,
    TaskResponse,
)
from tinyllm.core.node import BaseNode, NodeResult
from tinyllm.core.trace import ExecutionTrace, TraceRecorder


class ExecutorConfig(BaseModel):
    """Configuration for the executor."""

    model_config = {"extra": "forbid"}

    max_steps: int = Field(default=100, ge=1, le=1000, description="Maximum execution steps")
    timeout_ms: int = Field(
        default=60000, ge=1000, le=600000, description="Total execution timeout in ms"
    )
    enable_tracing: bool = Field(default=True, description="Whether to record execution trace")
    fail_fast: bool = Field(default=False, description="Stop on first node failure")


class ExecutionError(Exception):
    """Exception raised during graph execution."""

    def __init__(self, message: str, node_id: Optional[str] = None):
        super().__init__(message)
        self.node_id = node_id


class Executor:
    """Executes messages through a graph.

    The Executor manages the flow of messages through graph nodes,
    handling routing, timeouts, retries, and producing execution traces.
    """

    def __init__(
        self,
        graph: Graph,
        config: Optional[ExecutorConfig] = None,
        system_config: Optional[Config] = None,
    ):
        """Initialize executor.

        Args:
            graph: Graph to execute.
            config: Executor configuration.
            system_config: System configuration.
        """
        self.graph = graph
        self.config = config or ExecutorConfig()
        self.system_config = system_config

    async def execute(self, task: TaskPayload) -> TaskResponse:
        """Execute a task through the graph.

        Args:
            task: Input task payload.

        Returns:
            Task response with results.
        """
        trace_id = str(uuid4())
        start_time = time.perf_counter()

        # Initialize trace recorder
        recorder = TraceRecorder(trace_id, self.graph.id, task)

        # Create execution context
        context = ExecutionContext(
            trace_id=trace_id,
            graph_id=self.graph.id,
            config=self.system_config or Config(),
        )

        try:
            # Create initial message
            entry_node = self.graph.get_entry_node()
            if not entry_node:
                raise ExecutionError("No entry node found in graph")

            initial_message = Message(
                trace_id=trace_id,
                source_node="executor",
                target_node=entry_node.id,
                payload=MessagePayload(
                    task=task.content,
                    content=task.content,
                    structured={"context": task.context} if task.context else None,
                ),
            )

            context.add_message(initial_message)
            recorder.add_message(initial_message)

            # Execute graph
            result = await self._execute_loop(
                initial_message, entry_node, context, recorder
            )

            # Build response
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            response = TaskResponse(
                trace_id=trace_id,
                success=result.success,
                content=self._extract_content(result),
                total_latency_ms=elapsed_ms,
                nodes_executed=context.step_count,
                tokens_used=context.total_tokens,
            )

            if not result.success and result.error:
                response.error = ErrorInfo(
                    code=ErrorInfo.Codes.UNKNOWN,
                    message=result.error,
                )

            recorder.complete(response)
            return response

        except asyncio.TimeoutError:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            response = TaskResponse(
                trace_id=trace_id,
                success=False,
                error=ErrorInfo(
                    code=ErrorInfo.Codes.TIMEOUT,
                    message=f"Execution timed out after {self.config.timeout_ms}ms",
                    recoverable=False,
                ),
                total_latency_ms=elapsed_ms,
                nodes_executed=context.step_count,
                tokens_used=context.total_tokens,
            )
            recorder.fail(f"Timeout after {self.config.timeout_ms}ms")
            return response

        except ExecutionError as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            response = TaskResponse(
                trace_id=trace_id,
                success=False,
                error=ErrorInfo(
                    code=ErrorInfo.Codes.UNKNOWN,
                    message=str(e),
                    details={"node_id": e.node_id} if e.node_id else None,
                ),
                total_latency_ms=elapsed_ms,
                nodes_executed=context.step_count,
                tokens_used=context.total_tokens,
            )
            recorder.fail(str(e))
            return response

    async def _execute_loop(
        self,
        message: Message,
        start_node: BaseNode,
        context: ExecutionContext,
        recorder: TraceRecorder,
    ) -> NodeResult:
        """Execute the main graph traversal loop.

        Args:
            message: Initial message.
            start_node: Starting node.
            context: Execution context.
            recorder: Trace recorder.

        Returns:
            Final node result.
        """
        current_message = message
        current_node = start_node
        last_result: Optional[NodeResult] = None

        while context.step_count < self.config.max_steps:
            # Check timeout
            if context.elapsed_ms > self.config.timeout_ms:
                raise asyncio.TimeoutError()

            # Execute current node
            context.visit_node(current_node.id)
            recorder.start_node(
                current_node.id, current_node.type.value, current_message
            )

            result = await self._execute_node(current_node, current_message, context)

            recorder.complete_node(current_node.id, result)
            current_node.update_stats(result.success, result.latency_ms)

            # Handle failure
            if not result.success:
                if self.config.fail_fast:
                    return result
                # For non-fail-fast, we still return on terminal failure
                if not result.next_nodes:
                    return result

            # Check for exit
            if self.graph.is_exit_point(current_node.id):
                return result

            # Get next node
            if not result.next_nodes:
                # Use edges to determine next nodes
                next_node_ids = self.graph.get_next_nodes(
                    current_node.id, current_message
                )
                if not next_node_ids:
                    return result
                result.next_nodes = next_node_ids

            # For now, take first next node (simple sequential)
            next_node_id = result.next_nodes[0]
            next_node = self.graph.get_node(next_node_id)
            if not next_node:
                raise ExecutionError(
                    f"Next node '{next_node_id}' not found", current_node.id
                )

            # Get output message for next node
            if result.output_messages:
                current_message = result.output_messages[0]
                current_message.target_node = next_node_id
                context.add_message(current_message)
                recorder.add_message(current_message)
            else:
                # Create continuation message
                current_message = current_message.create_child(
                    source_node=current_node.id,
                    target_node=next_node_id,
                )
                context.add_message(current_message)
                recorder.add_message(current_message)

            current_node = next_node
            last_result = result

        # Max steps reached
        raise ExecutionError(f"Max steps ({self.config.max_steps}) exceeded")

    async def _execute_node(
        self,
        node: BaseNode,
        message: Message,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute a single node with timeout.

        Args:
            node: Node to execute.
            message: Input message.
            context: Execution context.

        Returns:
            Node execution result.
        """
        start_time = time.perf_counter()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                node.execute(message, context),
                timeout=node.config.timeout_ms / 1000,
            )
            result.latency_ms = int((time.perf_counter() - start_time) * 1000)
            return result

        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return NodeResult.failure_result(
                error=f"Node {node.id} timed out after {node.config.timeout_ms}ms",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return NodeResult.failure_result(
                error=f"Node {node.id} failed: {str(e)}",
                latency_ms=latency_ms,
            )

    def _extract_content(self, result: NodeResult) -> Optional[str]:
        """Extract content from a node result.

        Args:
            result: Node result.

        Returns:
            Content string or None.
        """
        if result.output_messages:
            msg = result.output_messages[-1]
            return msg.payload.content
        return None

    def get_trace(self) -> Optional[ExecutionTrace]:
        """Get the execution trace (if tracing enabled).

        Returns:
            Execution trace or None.
        """
        # TODO: Store and return trace from last execution
        return None
