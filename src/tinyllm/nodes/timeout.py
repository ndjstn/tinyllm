"""Timeout wrapper node implementation.

Wraps any node execution with configurable timeout handling.
"""

import asyncio
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.logging import get_logger

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext

logger = get_logger(__name__, component="timeout")


class TimeoutAction(str, Enum):
    """Actions to take when timeout occurs."""

    ERROR = "error"
    FALLBACK = "fallback"
    SKIP = "skip"


class TimeoutConfig(NodeConfig):
    """Configuration for timeout nodes."""

    model_config = ConfigDict(strict=False, frozen=True, extra="forbid")

    timeout_ms: int = Field(
        default=30000,
        ge=100,
        le=300000,
        description="Timeout in milliseconds",
    )
    on_timeout: TimeoutAction = Field(
        default=TimeoutAction.ERROR,
        description="Action to take when timeout occurs",
    )
    fallback_response: Optional[str] = Field(
        default=None,
        description="Response when on_timeout='fallback'",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of retries before timeout",
    )
    retry_delay_ms: int = Field(
        default=1000,
        ge=0,
        le=10000,
        description="Delay between retries in milliseconds",
    )
    inner_node: str = Field(
        description="Node ID to wrap with timeout",
        pattern=r"^[a-z][a-z0-9_\.]*$",
    )
    propagate_metadata: bool = Field(
        default=True,
        description="Whether to propagate timeout metadata to output",
    )


class TimeoutMetrics(BaseModel):
    """Metrics collected during timeout execution."""

    model_config = ConfigDict(strict=False, frozen=True, extra="forbid")

    total_attempts: int = Field(default=0, ge=0, description="Total execution attempts")
    timeouts_triggered: int = Field(default=0, ge=0, description="Number of timeouts")
    successful_executions: int = Field(default=0, ge=0, description="Successful executions")
    fallback_used: int = Field(default=0, ge=0, description="Times fallback was used")
    skipped: int = Field(default=0, ge=0, description="Times execution was skipped")
    total_elapsed_ms: int = Field(default=0, ge=0, description="Total time spent")
    avg_execution_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average execution time",
    )


@NodeRegistry.register(NodeType.TIMEOUT)
class TimeoutNode(BaseNode):
    """Timeout wrapper node for enforcing execution time limits.

    The TimeoutNode wraps any other node and enforces a timeout on its
    execution. When a timeout occurs, it can:
    - ERROR: Raise a timeout error
    - FALLBACK: Return a predefined fallback response
    - SKIP: Skip execution and return empty result

    The node also supports retry logic, attempting execution multiple
    times before applying the timeout action.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize timeout node."""
        super().__init__(definition)
        # Validate and freeze configuration
        self._timeout_config = TimeoutConfig(**definition.config)
        self._metrics = TimeoutMetrics()

    @property
    def timeout_config(self) -> TimeoutConfig:
        """Get timeout-specific configuration."""
        return self._timeout_config

    @property
    def metrics(self) -> TimeoutMetrics:
        """Get timeout metrics."""
        return self._metrics

    def _update_metrics(
        self,
        attempts: int = 0,
        timeouts: int = 0,
        successes: int = 0,
        fallbacks: int = 0,
        skips: int = 0,
        elapsed_ms: int = 0,
    ) -> None:
        """Update timeout metrics.

        Args:
            attempts: Number of attempts to add.
            timeouts: Number of timeouts to add.
            successes: Number of successes to add.
            fallbacks: Number of fallbacks to add.
            skips: Number of skips to add.
            elapsed_ms: Elapsed time to add.
        """
        total_attempts = self._metrics.total_attempts + attempts
        total_timeouts = self._metrics.timeouts_triggered + timeouts
        total_successes = self._metrics.successful_executions + successes
        total_fallbacks = self._metrics.fallback_used + fallbacks
        total_skips = self._metrics.skipped + skips
        total_time = self._metrics.total_elapsed_ms + elapsed_ms

        avg_time = total_time / total_attempts if total_attempts > 0 else 0.0

        self._metrics = TimeoutMetrics(
            total_attempts=total_attempts,
            timeouts_triggered=total_timeouts,
            successful_executions=total_successes,
            fallback_used=total_fallbacks,
            skipped=total_skips,
            total_elapsed_ms=total_time,
            avg_execution_time_ms=avg_time,
        )

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute inner node with timeout protection.

        Args:
            message: Input message to process.
            context: Execution context.

        Returns:
            NodeResult with timeout handling applied.
        """
        start_time = time.time()

        # Validate configuration
        validation_error = self._validate_config()
        if validation_error:
            return NodeResult.failure_result(
                error=validation_error,
                metadata={"metrics": self._metrics.model_dump()},
            )

        # Execute with retry logic
        for attempt in range(self._timeout_config.retry_count + 1):
            try:
                # Execute inner node with timeout
                result = await self._execute_with_timeout(message, context)

                # Success - update metrics and return
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._update_metrics(attempts=1, successes=1, elapsed_ms=elapsed_ms)

                # Add timeout metadata if configured
                if self._timeout_config.propagate_metadata:
                    result.metadata["timeout"] = {
                        "timeout_ms": self._timeout_config.timeout_ms,
                        "attempt": attempt + 1,
                        "elapsed_ms": elapsed_ms,
                        "timed_out": False,
                    }

                logger.info(
                    f"Timeout node {self.id} completed successfully",
                    node_id=self.id,
                    attempt=attempt + 1,
                    elapsed_ms=elapsed_ms,
                )

                return result

            except asyncio.TimeoutError:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._update_metrics(attempts=1, timeouts=1, elapsed_ms=elapsed_ms)

                logger.warning(
                    f"Timeout occurred in node {self.id}",
                    node_id=self.id,
                    attempt=attempt + 1,
                    timeout_ms=self._timeout_config.timeout_ms,
                    elapsed_ms=elapsed_ms,
                )

                # Check if we should retry
                if attempt < self._timeout_config.retry_count:
                    # Wait before retry
                    if self._timeout_config.retry_delay_ms > 0:
                        await asyncio.sleep(self._timeout_config.retry_delay_ms / 1000.0)
                    continue

                # Final attempt failed - handle timeout
                return await self._handle_timeout(message, elapsed_ms)

            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                self._update_metrics(attempts=1, elapsed_ms=elapsed_ms)

                logger.error(
                    f"Error in timeout node {self.id}",
                    node_id=self.id,
                    error=str(e),
                    attempt=attempt + 1,
                )

                # Don't retry on non-timeout errors
                return NodeResult.failure_result(
                    error=f"Inner node execution failed: {str(e)}",
                    latency_ms=elapsed_ms,
                    metadata={
                        "attempt": attempt + 1,
                        "metrics": self._metrics.model_dump(),
                    },
                )

        # Should never reach here, but just in case
        elapsed_ms = int((time.time() - start_time) * 1000)
        return NodeResult.failure_result(
            error="Maximum retries exceeded",
            latency_ms=elapsed_ms,
            metadata={"metrics": self._metrics.model_dump()},
        )

    async def _execute_with_timeout(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute inner node with timeout.

        Args:
            message: Input message.
            context: Execution context.

        Returns:
            NodeResult from inner node.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        # Create message for inner node
        inner_message = message.create_child(
            source_node=self.id,
            target_node=self._timeout_config.inner_node,
            payload=message.payload.model_copy(),
        )

        # Get inner node from context and execute with timeout
        # Note: In real execution, this would be handled by the executor
        # For now, we simulate the inner node execution
        timeout_seconds = self._timeout_config.timeout_ms / 1000.0

        result = await asyncio.wait_for(
            self._simulate_inner_execution(inner_message, context),
            timeout=timeout_seconds,
        )

        return result

    async def _simulate_inner_execution(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Simulate inner node execution.

        In production, the executor would handle this by routing to the inner node.
        This is a placeholder for testing.

        Args:
            message: Message to process.
            context: Execution context.

        Returns:
            Simulated execution result.
        """
        # This is a placeholder - in real execution, the executor
        # would route to the inner node and return its result
        output_payload = MessagePayload(
            task=message.payload.task,
            content=f"Inner node {self._timeout_config.inner_node} executed",
            metadata=message.payload.metadata,
        )

        output_message = message.create_child(
            source_node=self._timeout_config.inner_node,
            payload=output_payload,
        )

        return NodeResult.success_result(
            output_messages=[output_message],
            next_nodes=[],
            metadata={"simulated": True},
        )

    async def _handle_timeout(self, message: Message, elapsed_ms: int) -> NodeResult:
        """Handle timeout based on configured action.

        Args:
            message: Original input message.
            elapsed_ms: Time elapsed before timeout.

        Returns:
            NodeResult based on timeout action.
        """
        action = self._timeout_config.on_timeout

        timeout_metadata = {
            "timeout_ms": self._timeout_config.timeout_ms,
            "elapsed_ms": elapsed_ms,
            "timed_out": True,
            "action": action.value,
            "metrics": self._metrics.model_dump(),
        }

        if action == TimeoutAction.ERROR:
            logger.error(
                f"Timeout error in node {self.id}",
                node_id=self.id,
                timeout_ms=self._timeout_config.timeout_ms,
                elapsed_ms=elapsed_ms,
            )
            return NodeResult.failure_result(
                error=f"Timeout after {self._timeout_config.timeout_ms}ms",
                latency_ms=elapsed_ms,
                metadata=timeout_metadata,
            )

        elif action == TimeoutAction.FALLBACK:
            self._update_metrics(fallbacks=1)

            fallback_content = (
                self._timeout_config.fallback_response
                or "Timeout occurred - using fallback response"
            )

            logger.info(
                f"Using fallback response in node {self.id}",
                node_id=self.id,
                timeout_ms=self._timeout_config.timeout_ms,
            )

            output_payload = MessagePayload(
                task=message.payload.task,
                content=fallback_content,
                metadata={
                    **message.payload.metadata,
                    "fallback": True,
                },
            )

            output_message = message.create_child(
                source_node=self.id,
                payload=output_payload,
            )

            if self._timeout_config.propagate_metadata:
                timeout_metadata["fallback_used"] = True

            return NodeResult.success_result(
                output_messages=[output_message],
                next_nodes=[],
                latency_ms=elapsed_ms,
                metadata=timeout_metadata,
            )

        elif action == TimeoutAction.SKIP:
            self._update_metrics(skips=1)

            logger.info(
                f"Skipping execution in node {self.id}",
                node_id=self.id,
                timeout_ms=self._timeout_config.timeout_ms,
            )

            # Return empty result with skip signal
            if self._timeout_config.propagate_metadata:
                timeout_metadata["skipped"] = True

            return NodeResult.success_result(
                output_messages=[],
                next_nodes=[],
                latency_ms=elapsed_ms,
                metadata=timeout_metadata,
            )

        else:
            # Should never reach here
            return NodeResult.failure_result(
                error=f"Unknown timeout action: {action}",
                latency_ms=elapsed_ms,
                metadata=timeout_metadata,
            )

    def _validate_config(self) -> Optional[str]:
        """Validate timeout configuration.

        Returns:
            Error message if invalid, None if valid.
        """
        # Validate fallback response if action is FALLBACK
        if self._timeout_config.on_timeout == TimeoutAction.FALLBACK:
            if not self._timeout_config.fallback_response:
                return "FALLBACK action requires 'fallback_response' parameter"

        # Validate inner_node is specified
        if not self._timeout_config.inner_node:
            return "Timeout node requires 'inner_node' parameter"

        return None
