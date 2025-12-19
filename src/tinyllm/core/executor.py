"""Graph executor for TinyLLM.

This module provides the Executor class that runs messages through
a graph, managing state and producing traces.
"""

import asyncio
import time
from enum import Enum
from typing import List, Optional
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


class DegradationMode(str, Enum):
    """Degradation modes for handling failures."""

    FAIL_FAST = "fail_fast"  # Stop immediately on any failure
    SKIP_FAILED = "skip_failed"  # Skip failed nodes and continue with available paths
    FALLBACK_SIMPLE = "fallback_simple"  # Use simplified responses for failed nodes
    BEST_EFFORT = "best_effort"  # Continue with whatever partial results available


class ExecutorConfig(BaseModel):
    """Configuration for the executor."""

    model_config = {"extra": "forbid"}

    max_steps: int = Field(default=100, ge=1, le=1000, description="Maximum execution steps")
    timeout_ms: int = Field(
        default=60000, ge=1000, le=600000, description="Total execution timeout in ms"
    )
    enable_tracing: bool = Field(default=True, description="Whether to record execution trace")
    fail_fast: bool = Field(default=False, description="Stop on first node failure")
    degradation_mode: str = Field(
        default="fail_fast",
        description="How to handle node failures (fail_fast, skip_failed, fallback_simple, best_effort)"
    )
    max_node_failures: int = Field(
        default=3, ge=1, le=100, description="Maximum node failures before stopping (for degradation modes)"
    )
    enable_partial_results: bool = Field(
        default=True, description="Return partial results even when execution doesn't complete fully"
    )


class ExecutionError(Exception):
    """Exception raised during graph execution."""

    def __init__(self, message: str, node_id: Optional[str] = None):
        super().__init__(message)
        self.node_id = node_id


class Executor:
    """Executes messages through a graph.

    The Executor manages the flow of messages through graph nodes,
    handling routing, timeouts, retries, and producing execution traces.

    Supports graceful degradation modes:
    - fail_fast: Stop immediately on any failure
    - skip_failed: Skip failed nodes and continue with alternative paths
    - fallback_simple: Provide simplified responses for failed nodes
    - best_effort: Continue with partial results even on failures
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
        self._failed_nodes: List[str] = []
        self._partial_results: List[NodeResult] = []

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

            # Handle failure with degradation
            if not result.success:
                self._failed_nodes.append(current_node.id)
                self._partial_results.append(result)

                # Check if we've exceeded failure threshold
                if len(self._failed_nodes) >= self.config.max_node_failures:
                    if self.config.enable_partial_results:
                        return self._build_partial_result(result, context)
                    return result

                # Handle based on degradation mode
                degradation_mode = self.config.degradation_mode

                if degradation_mode == "fail_fast" or self.config.fail_fast:
                    return result
                elif degradation_mode == "skip_failed":
                    # Try to find alternative path
                    alternative_nodes = self._find_alternative_paths(current_node.id, message)
                    if alternative_nodes:
                        result.next_nodes = alternative_nodes
                    elif self.config.enable_partial_results:
                        return self._build_partial_result(result, context)
                    else:
                        return result
                elif degradation_mode == "fallback_simple":
                    # Create a simple fallback response
                    result = self._create_fallback_result(current_node, message, result)
                elif degradation_mode == "best_effort":
                    # Continue with whatever we have
                    if not result.next_nodes:
                        alternative_nodes = self._find_alternative_paths(current_node.id, message)
                        if alternative_nodes:
                            result.next_nodes = alternative_nodes
                        elif self.config.enable_partial_results:
                            return self._build_partial_result(result, context)
                        else:
                            return result

                # For non-fail-fast, check if we can continue
                if not result.next_nodes:
                    if self.config.enable_partial_results:
                        return self._build_partial_result(result, context)
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

    def _find_alternative_paths(self, failed_node_id: str, message: Message) -> list[str]:
        """Find alternative execution paths when a node fails.

        Args:
            failed_node_id: ID of the failed node.
            message: Current message.

        Returns:
            List of alternative node IDs to try.
        """
        # Get all outgoing edges from the failed node
        edges = self.graph.get_outgoing_edges(failed_node_id)

        alternative_nodes = []
        for edge in edges:
            next_node_id = edge.to_node
            # Skip already failed nodes
            if next_node_id not in self._failed_nodes:
                # Check if the edge has a condition
                if edge.condition:
                    if self.graph._evaluate_condition(edge.condition, message):
                        alternative_nodes.append(next_node_id)
                else:
                    alternative_nodes.append(next_node_id)

        return alternative_nodes

    def _create_fallback_result(
        self, node: BaseNode, message: Message, failed_result: NodeResult
    ) -> NodeResult:
        """Create a simplified fallback result when a node fails.

        Args:
            node: The failed node.
            message: Input message.
            failed_result: The original failed result.

        Returns:
            Fallback NodeResult with simplified response.
        """
        # Create a simple fallback message
        fallback_content = (
            f"The system encountered an issue processing your request at {node.id}. "
            f"Providing a simplified response based on available information."
        )

        fallback_message = Message(
            trace_id=message.trace_id,
            source_node=node.id,
            target_node=message.target_node,
            payload=MessagePayload(
                content=fallback_content,
                task=message.payload.task,
                structured={"fallback": True, "original_error": failed_result.error},
            ),
        )

        # Try to find next nodes
        next_nodes = self._find_alternative_paths(node.id, message)

        return NodeResult(
            success=True,  # Mark as success to allow continuation
            output_messages=[fallback_message],
            next_nodes=next_nodes,
            metadata={
                "degraded": True,
                "original_error": failed_result.error,
                "fallback_mode": "simple",
            },
            latency_ms=failed_result.latency_ms,
        )

    def _build_partial_result(self, last_result: NodeResult, context: ExecutionContext) -> NodeResult:
        """Build a partial result from successful nodes when execution can't complete.

        Args:
            last_result: The last result (possibly failed).
            context: Execution context.

        Returns:
            NodeResult with partial results aggregated.
        """
        # Collect all successful partial results
        successful_results = [r for r in self._partial_results if r.success]

        # If we have no successful results, return the last result
        if not successful_results:
            return last_result

        # Aggregate content from successful results
        aggregated_content = []
        for result in successful_results:
            if result.output_messages:
                for msg in result.output_messages:
                    if msg.payload.content:
                        aggregated_content.append(msg.payload.content)

        # Create a partial response message
        partial_message = Message(
            trace_id=context.trace_id,
            source_node="executor",
            target_node="exit",
            payload=MessagePayload(
                content=" ".join(aggregated_content) if aggregated_content else "Partial results available.",
                task="partial_result",
                structured={
                    "partial": True,
                    "successful_nodes": len(successful_results),
                    "failed_nodes": len(self._failed_nodes),
                    "failed_node_ids": self._failed_nodes,
                },
            ),
        )

        return NodeResult(
            success=True,  # Partial success
            output_messages=[partial_message],
            next_nodes=[],
            metadata={
                "partial_result": True,
                "successful_nodes": len(successful_results),
                "failed_nodes": len(self._failed_nodes),
                "degradation_mode": self.config.degradation_mode,
            },
            latency_ms=last_result.latency_ms,
        )

    def get_trace(self) -> Optional[ExecutionTrace]:
        """Get the execution trace (if tracing enabled).

        Returns:
            Execution trace or None.
        """
        # TODO: Store and return trace from last execution
        return None
