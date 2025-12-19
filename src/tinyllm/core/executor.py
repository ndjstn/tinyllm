"""Graph executor for TinyLLM.

This module provides the Executor class that runs messages through
a graph, managing state and producing traces.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional
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
from tinyllm.profiling import get_profiling_context


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
    enable_checkpointing: bool = Field(
        default=False, description="Enable checkpointing for recovery"
    )
    checkpoint_interval: int = Field(
        default=5, ge=1, description="Create checkpoint every N nodes"
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
        self._checkpoints: List[Dict[str, Any]] = []
        self._last_checkpoint_step = 0

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

            # Execute graph with profiling
            profiling = get_profiling_context(enable_performance=True, enable_memory=False)
            async with profiling.profile_async(f"graph_{self.graph.id}"):
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

            # Execute current node with profiling
            context.visit_node(current_node.id)
            recorder.start_node(
                current_node.id, current_node.type.value, current_message
            )

            profiling = get_profiling_context(enable_performance=True, enable_memory=False)
            async with profiling.profile_async(f"node_{current_node.id}"):
                result = await self._execute_node(current_node, current_message, context)

            recorder.complete_node(current_node.id, result)
            current_node.update_stats(result.success, result.latency_ms)

            # Create checkpoint if needed
            if self._should_create_checkpoint(context):
                self._create_checkpoint(current_node, current_message, context, result)

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

    def _create_checkpoint(
        self,
        current_node: BaseNode,
        current_message: Message,
        context: ExecutionContext,
        result: NodeResult,
    ) -> None:
        """Create a checkpoint of current execution state.

        Args:
            current_node: Current node being executed.
            current_message: Current message.
            context: Execution context.
            result: Last node result.
        """
        checkpoint = {
            "step": context.step_count,
            "node_id": current_node.id,
            "message_id": current_message.message_id,
            "visited_nodes": list(context.visited_nodes),
            "success": result.success,
            "timestamp": time.time(),
            "partial_results": len(self._partial_results),
        }

        self._checkpoints.append(checkpoint)
        self._last_checkpoint_step = context.step_count

        from tinyllm.logging import get_logger

        logger = get_logger(__name__)
        logger.info(
            "checkpoint_created",
            trace_id=context.trace_id,
            step=context.step_count,
            node_id=current_node.id,
        )

    def _should_create_checkpoint(self, context: ExecutionContext) -> bool:
        """Check if we should create a checkpoint.

        Args:
            context: Execution context.

        Returns:
            True if checkpoint should be created.
        """
        if not self.config.enable_checkpointing:
            return False

        steps_since_checkpoint = context.step_count - self._last_checkpoint_step
        return steps_since_checkpoint >= self.config.checkpoint_interval

    def get_recovery_point(self) -> Optional[Dict[str, Any]]:
        """Get the last successful checkpoint for recovery.

        Returns:
            Last successful checkpoint or None.
        """
        # Find last successful checkpoint
        for checkpoint in reversed(self._checkpoints):
            if checkpoint.get("success", False):
                return checkpoint

        return None

    async def recover_from_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        task: TaskPayload,
    ) -> TaskResponse:
        """Attempt to recover execution from a checkpoint.

        Args:
            checkpoint: Checkpoint to recover from.
            task: Original task payload.

        Returns:
            TaskResponse from recovered execution.
        """
        from tinyllm.logging import get_logger

        logger = get_logger(__name__)
        logger.info(
            "recovering_from_checkpoint",
            checkpoint_step=checkpoint.get("step"),
            node_id=checkpoint.get("node_id"),
        )

        # Create a new execution context from checkpoint
        trace_id = str(uuid4())
        context = ExecutionContext(
            trace_id=trace_id,
            graph_id=self.graph.id,
            config=self.system_config or Config(),
        )

        # Restore visited nodes
        visited_nodes = checkpoint.get("visited_nodes", [])
        for node_id in visited_nodes:
            context.visit_node(node_id)

        # Get the node to resume from
        resume_node_id = checkpoint.get("node_id")
        resume_node = self.graph.get_node(resume_node_id)

        if not resume_node:
            raise ExecutionError(f"Cannot resume: node {resume_node_id} not found")

        # Create message for resumption
        resume_message = Message(
            trace_id=trace_id,
            source_node="executor",
            target_node=resume_node_id,
            payload=MessagePayload(
                task=task.content,
                content=task.content,
                structured={"recovered": True, "from_checkpoint": checkpoint.get("step")},
            ),
        )

        recorder = TraceRecorder(trace_id, self.graph.id, task)
        context.add_message(resume_message)
        recorder.add_message(resume_message)

        # Continue execution from checkpoint
        result = await self._execute_loop(resume_message, resume_node, context, recorder)

        # Build response
        start_time = time.perf_counter()
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

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._last_checkpoint_step = 0

        from tinyllm.logging import get_logger

        logger = get_logger(__name__)
        logger.info("checkpoints_cleared")

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints.

        Returns:
            Dictionary with checkpoint statistics.
        """
        successful = sum(1 for cp in self._checkpoints if cp.get("success", False))
        failed = len(self._checkpoints) - successful

        return {
            "total_checkpoints": len(self._checkpoints),
            "successful_checkpoints": successful,
            "failed_checkpoints": failed,
            "last_checkpoint_step": self._last_checkpoint_step,
            "checkpointing_enabled": self.config.enable_checkpointing,
            "checkpoint_interval": self.config.checkpoint_interval,
        }

    def get_trace(self) -> Optional[ExecutionTrace]:
        """Get the execution trace (if tracing enabled).

        Returns:
            Execution trace or None.
        """
        # TODO: Store and return trace from last execution
        return None


# ============================================================================
# Transaction Rollback System
# ============================================================================


class TransactionLog(BaseModel):
    """Log entry for a node operation in a transaction."""

    step: int
    node_id: str
    operation: str  # execute, update_context, etc.
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    timestamp: float
    reversible: bool = True


class Transaction:
    """Transaction manager for multi-node graph operations with rollback support."""

    def __init__(self, trace_id: str, graph_id: str):
        """Initialize transaction.

        Args:
            trace_id: Unique trace ID for this transaction.
            graph_id: Graph ID being executed.
        """
        self.trace_id = trace_id
        self.graph_id = graph_id
        self.logs: List[TransactionLog] = []
        self.committed = False
        self.rolled_back = False
        self._lock = asyncio.Lock()
        self.start_time = time.time()

    def log_operation(
        self,
        step: int,
        node_id: str,
        operation: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        reversible: bool = True,
    ) -> None:
        """Log a node operation for potential rollback.

        Args:
            step: Execution step number.
            node_id: Node that performed the operation.
            operation: Type of operation.
            state_before: State before operation.
            state_after: State after operation.
            reversible: Whether this operation can be reversed.
        """
        log_entry = TransactionLog(
            step=step,
            node_id=node_id,
            operation=operation,
            state_before=state_before,
            state_after=state_after,
            timestamp=time.time(),
            reversible=reversible,
        )
        self.logs.append(log_entry)

    async def commit(self) -> None:
        """Commit the transaction, making all operations permanent."""
        async with self._lock:
            if self.committed or self.rolled_back:
                raise RuntimeError(
                    f"Transaction already {'committed' if self.committed else 'rolled back'}"
                )

            self.committed = True

            from tinyllm.logging import get_logger

            logger = get_logger(__name__)
            logger.info(
                "transaction_committed",
                trace_id=self.trace_id,
                graph_id=self.graph_id,
                operations=len(self.logs),
                duration_s=time.time() - self.start_time,
            )

    async def rollback(self, to_step: Optional[int] = None) -> Dict[str, Any]:
        """Rollback the transaction, undoing operations.

        Args:
            to_step: Roll back to this step (None = full rollback).

        Returns:
            Dictionary with rollback results.
        """
        async with self._lock:
            if self.committed:
                raise RuntimeError("Cannot rollback committed transaction")

            if self.rolled_back:
                raise RuntimeError("Transaction already rolled back")

            from tinyllm.logging import get_logger

            logger = get_logger(__name__)

            # Determine which operations to rollback
            if to_step is None:
                operations_to_undo = list(reversed(self.logs))
            else:
                operations_to_undo = [log for log in reversed(self.logs) if log.step > to_step]

            # Track rollback results
            rolled_back_count = 0
            skipped_count = 0
            errors = []

            logger.info(
                "transaction_rollback_starting",
                trace_id=self.trace_id,
                total_operations=len(operations_to_undo),
                to_step=to_step,
            )

            # Undo operations in reverse order
            for log_entry in operations_to_undo:
                if not log_entry.reversible:
                    skipped_count += 1
                    logger.warning(
                        "operation_not_reversible",
                        node_id=log_entry.node_id,
                        operation=log_entry.operation,
                        step=log_entry.step,
                    )
                    continue

                try:
                    # Perform rollback (restore state_before)
                    await self._undo_operation(log_entry)
                    rolled_back_count += 1

                except Exception as e:
                    error_msg = f"Failed to rollback step {log_entry.step}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(
                        "rollback_operation_failed",
                        node_id=log_entry.node_id,
                        operation=log_entry.operation,
                        step=log_entry.step,
                        error=str(e),
                    )

            self.rolled_back = True

            result = {
                "success": len(errors) == 0,
                "rolled_back": rolled_back_count,
                "skipped": skipped_count,
                "errors": errors,
                "to_step": to_step,
                "duration_s": time.time() - self.start_time,
            }

            logger.info(
                "transaction_rolled_back",
                trace_id=self.trace_id,
                result=result,
            )

            return result

    async def _undo_operation(self, log_entry: TransactionLog) -> None:
        """Undo a single operation.

        Args:
            log_entry: Transaction log entry to undo.
        """
        # This is a simplified implementation. In a real system,
        # each operation type would have specific undo logic.
        # For now, we just log the rollback intent.

        from tinyllm.logging import get_logger

        logger = get_logger(__name__)
        logger.debug(
            "undoing_operation",
            node_id=log_entry.node_id,
            operation=log_entry.operation,
            step=log_entry.step,
            state_before=log_entry.state_before,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get transaction status.

        Returns:
            Dictionary with transaction status.
        """
        return {
            "trace_id": self.trace_id,
            "graph_id": self.graph_id,
            "operations_count": len(self.logs),
            "committed": self.committed,
            "rolled_back": self.rolled_back,
            "duration_s": time.time() - self.start_time,
            "reversible_operations": sum(1 for log in self.logs if log.reversible),
            "irreversible_operations": sum(1 for log in self.logs if not log.reversible),
        }


class TransactionalExecutor(Executor):
    """Executor with transaction support for multi-node operations.

    Extends the base Executor with transaction logging and rollback capabilities.
    When enabled, all node operations are logged and can be rolled back on failure.
    """

    def __init__(
        self,
        graph: Graph,
        config: Optional[ExecutorConfig] = None,
        system_config: Optional[Config] = None,
        enable_transactions: bool = True,
    ):
        """Initialize transactional executor.

        Args:
            graph: Graph to execute.
            config: Executor configuration.
            system_config: System configuration.
            enable_transactions: Whether to enable transaction logging.
        """
        super().__init__(graph, config, system_config)
        self.enable_transactions = enable_transactions
        self._current_transaction: Optional[Transaction] = None

    async def execute(self, task: TaskPayload) -> TaskResponse:
        """Execute a task through the graph with transaction support.

        Args:
            task: Input task payload.

        Returns:
            Task response with results.
        """
        if not self.enable_transactions:
            return await super().execute(task)

        # Start transaction
        trace_id = str(uuid4())
        self._current_transaction = Transaction(trace_id, self.graph.id)

        try:
            response = await super().execute(task)

            # Commit transaction on success
            if response.success:
                await self._current_transaction.commit()
            else:
                # Rollback on failure
                rollback_result = await self._current_transaction.rollback()

                from tinyllm.logging import get_logger

                logger = get_logger(__name__)
                logger.info(
                    "execution_rolled_back",
                    trace_id=trace_id,
                    rollback_result=rollback_result,
                )

            return response

        except Exception as e:
            # Rollback on exception
            if self._current_transaction and not self._current_transaction.rolled_back:
                rollback_result = await self._current_transaction.rollback()

                from tinyllm.logging import get_logger

                logger = get_logger(__name__)
                logger.error(
                    "execution_failed_rolled_back",
                    trace_id=trace_id,
                    error=str(e),
                    rollback_result=rollback_result,
                )

            raise

        finally:
            self._current_transaction = None

    async def _execute_node(
        self,
        node: BaseNode,
        message: Message,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute a single node with transaction logging.

        Args:
            node: Node to execute.
            message: Input message.
            context: Execution context.

        Returns:
            Node execution result.
        """
        # Capture state before execution
        state_before = {
            "step_count": context.step_count,
            "total_tokens": context.total_tokens,
            "message_count": len(context.messages),
        }

        # Execute node
        result = await super()._execute_node(node, message, context)

        # Log operation in transaction
        if self._current_transaction:
            state_after = {
                "step_count": context.step_count,
                "total_tokens": context.total_tokens,
                "message_count": len(context.messages),
                "success": result.success,
            }

            self._current_transaction.log_operation(
                step=context.step_count,
                node_id=node.id,
                operation="execute",
                state_before=state_before,
                state_after=state_after,
                reversible=True,  # Most operations are reversible
            )

        return result

    def get_transaction_status(self) -> Optional[Dict[str, Any]]:
        """Get current transaction status.

        Returns:
            Transaction status or None if no active transaction.
        """
        if self._current_transaction:
            return self._current_transaction.get_status()
        return None

    async def rollback_to_checkpoint(self, checkpoint_step: int) -> Dict[str, Any]:
        """Rollback transaction to a specific checkpoint.

        Args:
            checkpoint_step: Step number to rollback to.

        Returns:
            Rollback result dictionary.
        """
        if not self._current_transaction:
            raise RuntimeError("No active transaction to rollback")

        return await self._current_transaction.rollback(to_step=checkpoint_step)
