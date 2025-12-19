"""Fanout node implementation.

Enables parallel execution of multiple downstream nodes with various
aggregation strategies for collecting and combining results.
"""

import asyncio
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import ErrorInfo, Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class AggregationStrategy(str, Enum):
    """Strategy for aggregating results from parallel executions."""

    FIRST_SUCCESS = "first_success"  # Return first successful result
    ALL = "all"  # Return all results (including failures)
    MAJORITY_VOTE = "majority_vote"  # Return most common answer
    BEST_SCORE = "best_score"  # Return highest scored result


class FanoutTargetResult(BaseModel):
    """Result from a single fanout target execution.

    This is a strict, frozen model representing the outcome of executing
    one target node in the fanout.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    target_node: str = Field(description="Target node ID that was executed")
    success: bool = Field(description="Whether this target succeeded")
    message: Optional[Message] = Field(
        default=None, description="Output message from target"
    )
    error: Optional[str] = Field(default=None, description="Error if execution failed")
    latency_ms: int = Field(default=0, ge=0, description="Execution time in ms")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata from execution"
    )


class FanoutResult(BaseModel):
    """Aggregated results from all parallel fanout executions.

    This is a strict, frozen model containing all target results and
    the aggregated output.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    success: bool = Field(description="Whether fanout completed successfully")
    target_results: List[FanoutTargetResult] = Field(
        description="Results from each target node"
    )
    aggregated_message: Optional[Message] = Field(
        default=None, description="Aggregated output message"
    )
    strategy_used: AggregationStrategy = Field(
        description="Aggregation strategy that was applied"
    )
    total_latency_ms: int = Field(default=0, ge=0, description="Total execution time")
    successful_targets: int = Field(default=0, ge=0, description="Count of successful targets")
    failed_targets: int = Field(default=0, ge=0, description="Count of failed targets")
    error: Optional[str] = Field(default=None, description="Error if fanout failed")


class FanoutConfig(NodeConfig):
    """Configuration for fanout nodes.

    This is a frozen model defining how the fanout node should behave.
    Note: Not strict to allow string-to-enum conversion from YAML configs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_nodes: List[str] = Field(
        description="List of target node IDs to fan out to",
        min_length=1,
    )
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.ALL,
        description="Strategy for aggregating results",
    )
    timeout_ms: int = Field(
        default=30000, ge=100, le=120000, description="Timeout for each target"
    )
    parallel: bool = Field(
        default=True,
        description="Execute targets in parallel (True) or sequentially (False)",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first failure when executing sequentially",
    )
    require_all_success: bool = Field(
        default=False,
        description="Require all targets to succeed for fanout to succeed",
    )
    score_field: str = Field(
        default="quality_score",
        description="Field to use for BEST_SCORE strategy",
    )
    retry_count: int = Field(default=0, ge=0, le=3)
    retry_delay_ms: int = Field(default=1000, ge=0, le=10000)


@NodeRegistry.register(NodeType.FANOUT)
class FanoutNode(BaseNode):
    """Fans out message to multiple targets in parallel.

    The FanoutNode enables parallel execution of multiple downstream nodes,
    collecting their results and aggregating them according to a configured
    strategy.

    Supports multiple aggregation strategies:
    - FIRST_SUCCESS: Return first successful result (fast, early termination)
    - ALL: Collect all results from all targets
    - MAJORITY_VOTE: Return most common answer from successful results
    - BEST_SCORE: Return result with highest score

    Example:
        A fanout node can send a message to multiple specialized models
        in parallel and aggregate their responses.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize fanout node.

        Args:
            definition: Node definition from graph config.
        """
        super().__init__(definition)
        self._fanout_config = FanoutConfig(**definition.config)

    @property
    def fanout_config(self) -> FanoutConfig:
        """Get fanout-specific configuration."""
        return self._fanout_config

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute fanout logic.

        Fans out the message to all target nodes in parallel (or sequentially),
        collects results, and aggregates them according to the configured strategy.

        Args:
            message: Input message to fan out.
            context: Execution context.

        Returns:
            NodeResult with aggregated output.
        """
        if not self._fanout_config.target_nodes:
            return NodeResult.failure_result(
                error="No target nodes configured for fanout"
            )

        try:
            # Execute targets based on parallel/sequential mode
            if self._fanout_config.parallel:
                fanout_result = await self._execute_parallel(message, context)
            else:
                fanout_result = await self._execute_sequential(message, context)

            # Check if fanout was successful
            if not fanout_result.success:
                return NodeResult.failure_result(
                    error=fanout_result.error or "Fanout execution failed",
                    latency_ms=fanout_result.total_latency_ms,
                    metadata={
                        "fanout_result": fanout_result.model_dump(),
                        "strategy": fanout_result.strategy_used.value,
                    },
                )

            # Build successful result
            output_messages = (
                [fanout_result.aggregated_message]
                if fanout_result.aggregated_message
                else []
            )

            # Next nodes are determined by the executor from edges
            return NodeResult.success_result(
                output_messages=output_messages,
                next_nodes=[],
                latency_ms=fanout_result.total_latency_ms,
                metadata={
                    "fanout_result": fanout_result.model_dump(),
                    "strategy": fanout_result.strategy_used.value,
                    "successful_targets": fanout_result.successful_targets,
                    "failed_targets": fanout_result.failed_targets,
                },
            )

        except Exception as e:
            return NodeResult.failure_result(
                error=f"Fanout execution failed: {str(e)}"
            )

    async def _execute_parallel(
        self, message: Message, context: "ExecutionContext"
    ) -> FanoutResult:
        """Execute all targets in parallel using asyncio.gather.

        Args:
            message: Input message to fan out.
            context: Execution context.

        Returns:
            FanoutResult with aggregated outputs.
        """
        start_time = asyncio.get_event_loop().time()

        # Create tasks for all targets (wrap coroutines in actual tasks)
        tasks = [
            asyncio.create_task(self._execute_single_target(target, message, context))
            for target in self._fanout_config.target_nodes
        ]

        # For FIRST_SUCCESS, we can use asyncio.FIRST_COMPLETED
        if self._fanout_config.aggregation_strategy == AggregationStrategy.FIRST_SUCCESS:
            target_results = await self._execute_first_success(tasks)
        else:
            # Wait for all tasks to complete
            target_results = await asyncio.gather(*tasks, return_exceptions=False)

        # Calculate total latency
        end_time = asyncio.get_event_loop().time()
        total_latency_ms = int((end_time - start_time) * 1000)

        # Aggregate results
        return await self._aggregate_results(
            target_results, total_latency_ms, message
        )

    async def _execute_sequential(
        self, message: Message, context: "ExecutionContext"
    ) -> FanoutResult:
        """Execute targets sequentially.

        Args:
            message: Input message to fan out.
            context: Execution context.

        Returns:
            FanoutResult with aggregated outputs.
        """
        start_time = asyncio.get_event_loop().time()
        target_results: List[FanoutTargetResult] = []

        for target in self._fanout_config.target_nodes:
            result = await self._execute_single_target(target, message, context)
            target_results.append(result)

            # Fail fast if enabled and target failed
            if self._fanout_config.fail_fast and not result.success:
                break

        # Calculate total latency
        end_time = asyncio.get_event_loop().time()
        total_latency_ms = int((end_time - start_time) * 1000)

        # Aggregate results
        return await self._aggregate_results(
            target_results, total_latency_ms, message
        )

    async def _execute_first_success(
        self, tasks: List[asyncio.Task]
    ) -> List[FanoutTargetResult]:
        """Execute tasks and return when first one succeeds.

        Args:
            tasks: List of tasks to execute.

        Returns:
            List containing results (first success plus any completed tasks).
        """
        pending = set(tasks)
        results: List[FanoutTargetResult] = []

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                result = task.result()
                results.append(result)

                # If we got a success, cancel remaining tasks and return
                if result.success:
                    for remaining in pending:
                        remaining.cancel()
                    # Wait for cancelled tasks to finish
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    return results

        return results

    async def _execute_single_target(
        self, target_node: str, message: Message, context: "ExecutionContext"
    ) -> FanoutTargetResult:
        """Execute a single target node.

        This is a mock execution since we don't have actual node execution here.
        In a real implementation, this would invoke the target node through the executor.

        Args:
            target_node: Target node ID.
            message: Input message.
            context: Execution context.

        Returns:
            FanoutTargetResult from target execution.
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Create a timeout for this execution
            async with asyncio.timeout(self._fanout_config.timeout_ms / 1000):
                # In a real implementation, this would call context.execute_node(target_node, message)
                # For now, we'll create a simulated result

                # Create output message as child of input
                output_message = message.create_child(
                    source_node=target_node,
                    target_node=None,
                    payload=message.payload.model_copy(
                        update={
                            "metadata": {
                                **message.payload.metadata,
                                "fanout_target": target_node,
                            }
                        }
                    ),
                )

                end_time = asyncio.get_event_loop().time()
                latency_ms = int((end_time - start_time) * 1000)

                return FanoutTargetResult(
                    target_node=target_node,
                    success=True,
                    message=output_message,
                    latency_ms=latency_ms,
                    metadata={"simulated": True},
                )

        except asyncio.TimeoutError:
            end_time = asyncio.get_event_loop().time()
            latency_ms = int((end_time - start_time) * 1000)

            return FanoutTargetResult(
                target_node=target_node,
                success=False,
                error=f"Target execution timed out after {self._fanout_config.timeout_ms}ms",
                latency_ms=latency_ms,
            )

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            latency_ms = int((end_time - start_time) * 1000)

            return FanoutTargetResult(
                target_node=target_node,
                success=False,
                error=f"Target execution failed: {str(e)}",
                latency_ms=latency_ms,
            )

    async def _aggregate_results(
        self,
        target_results: List[FanoutTargetResult],
        total_latency_ms: int,
        original_message: Message,
    ) -> FanoutResult:
        """Aggregate results from all targets according to strategy.

        Args:
            target_results: Results from all target executions.
            total_latency_ms: Total execution time.
            original_message: Original input message.

        Returns:
            Aggregated FanoutResult.
        """
        successful_results = [r for r in target_results if r.success]
        failed_results = [r for r in target_results if not r.success]

        strategy = self._fanout_config.aggregation_strategy

        # Check if we should fail based on require_all_success
        if self._fanout_config.require_all_success and failed_results:
            error_details = "; ".join(
                f"{r.target_node}: {r.error}" for r in failed_results
            )
            return FanoutResult(
                success=False,
                target_results=target_results,
                strategy_used=strategy,
                total_latency_ms=total_latency_ms,
                successful_targets=len(successful_results),
                failed_targets=len(failed_results),
                error=f"Required all targets to succeed but {len(failed_results)} failed: {error_details}",
            )

        # Apply aggregation strategy
        if strategy == AggregationStrategy.FIRST_SUCCESS:
            return self._aggregate_first_success(
                target_results, successful_results, total_latency_ms, original_message
            )
        elif strategy == AggregationStrategy.ALL:
            return self._aggregate_all(
                target_results, successful_results, total_latency_ms, original_message
            )
        elif strategy == AggregationStrategy.MAJORITY_VOTE:
            return self._aggregate_majority_vote(
                target_results, successful_results, total_latency_ms, original_message
            )
        elif strategy == AggregationStrategy.BEST_SCORE:
            return self._aggregate_best_score(
                target_results, successful_results, total_latency_ms, original_message
            )
        else:
            return FanoutResult(
                success=False,
                target_results=target_results,
                strategy_used=strategy,
                total_latency_ms=total_latency_ms,
                successful_targets=len(successful_results),
                failed_targets=len(failed_results),
                error=f"Unknown aggregation strategy: {strategy}",
            )

    def _aggregate_first_success(
        self,
        target_results: List[FanoutTargetResult],
        successful_results: List[FanoutTargetResult],
        total_latency_ms: int,
        original_message: Message,
    ) -> FanoutResult:
        """Aggregate using FIRST_SUCCESS strategy."""
        if not successful_results:
            return FanoutResult(
                success=False,
                target_results=target_results,
                strategy_used=AggregationStrategy.FIRST_SUCCESS,
                total_latency_ms=total_latency_ms,
                successful_targets=0,
                failed_targets=len(target_results),
                error="No successful results from any target",
            )

        # Return first successful result
        first_success = successful_results[0]
        aggregated_message = first_success.message

        return FanoutResult(
            success=True,
            target_results=target_results,
            aggregated_message=aggregated_message,
            strategy_used=AggregationStrategy.FIRST_SUCCESS,
            total_latency_ms=total_latency_ms,
            successful_targets=len(successful_results),
            failed_targets=len(target_results) - len(successful_results),
        )

    def _aggregate_all(
        self,
        target_results: List[FanoutTargetResult],
        successful_results: List[FanoutTargetResult],
        total_latency_ms: int,
        original_message: Message,
    ) -> FanoutResult:
        """Aggregate using ALL strategy."""
        if not successful_results:
            return FanoutResult(
                success=False,
                target_results=target_results,
                strategy_used=AggregationStrategy.ALL,
                total_latency_ms=total_latency_ms,
                successful_targets=0,
                failed_targets=len(target_results),
                error="No successful results from any target",
            )

        # Combine all successful messages into one
        all_contents = []
        all_metadata = {}

        for result in successful_results:
            if result.message and result.message.payload.content:
                all_contents.append(
                    f"[{result.target_node}]: {result.message.payload.content}"
                )
            if result.metadata:
                all_metadata[result.target_node] = result.metadata

        combined_content = "\n\n".join(all_contents)

        aggregated_message = original_message.create_child(
            source_node=self.id,
            payload=MessagePayload(
                task=original_message.payload.task,
                content=combined_content,
                metadata={
                    **original_message.payload.metadata,
                    "fanout_all_results": all_metadata,
                    "target_count": len(successful_results),
                },
            ),
        )

        return FanoutResult(
            success=True,
            target_results=target_results,
            aggregated_message=aggregated_message,
            strategy_used=AggregationStrategy.ALL,
            total_latency_ms=total_latency_ms,
            successful_targets=len(successful_results),
            failed_targets=len(target_results) - len(successful_results),
        )

    def _aggregate_majority_vote(
        self,
        target_results: List[FanoutTargetResult],
        successful_results: List[FanoutTargetResult],
        total_latency_ms: int,
        original_message: Message,
    ) -> FanoutResult:
        """Aggregate using MAJORITY_VOTE strategy."""
        if not successful_results:
            return FanoutResult(
                success=False,
                target_results=target_results,
                strategy_used=AggregationStrategy.MAJORITY_VOTE,
                total_latency_ms=total_latency_ms,
                successful_targets=0,
                failed_targets=len(target_results),
                error="No successful results from any target",
            )

        # Collect all content strings and count occurrences
        contents = []
        for result in successful_results:
            if result.message and result.message.payload.content:
                # Normalize content for comparison (strip whitespace, lowercase)
                normalized = result.message.payload.content.strip().lower()
                contents.append((normalized, result.message.payload.content))

        if not contents:
            # Fall back to first result if no content
            return self._aggregate_first_success(
                target_results, successful_results, total_latency_ms, original_message
            )

        # Count occurrences of each normalized content
        content_counter = Counter(c[0] for c in contents)
        most_common_normalized, count = content_counter.most_common(1)[0]

        # Find the original (non-normalized) content
        majority_content = next(
            c[1] for c in contents if c[0] == most_common_normalized
        )

        aggregated_message = original_message.create_child(
            source_node=self.id,
            payload=MessagePayload(
                task=original_message.payload.task,
                content=majority_content,
                metadata={
                    **original_message.payload.metadata,
                    "fanout_vote_count": count,
                    "fanout_total_votes": len(successful_results),
                },
            ),
        )

        return FanoutResult(
            success=True,
            target_results=target_results,
            aggregated_message=aggregated_message,
            strategy_used=AggregationStrategy.MAJORITY_VOTE,
            total_latency_ms=total_latency_ms,
            successful_targets=len(successful_results),
            failed_targets=len(target_results) - len(successful_results),
        )

    def _aggregate_best_score(
        self,
        target_results: List[FanoutTargetResult],
        successful_results: List[FanoutTargetResult],
        total_latency_ms: int,
        original_message: Message,
    ) -> FanoutResult:
        """Aggregate using BEST_SCORE strategy."""
        if not successful_results:
            return FanoutResult(
                success=False,
                target_results=target_results,
                strategy_used=AggregationStrategy.BEST_SCORE,
                total_latency_ms=total_latency_ms,
                successful_targets=0,
                failed_targets=len(target_results),
                error="No successful results from any target",
            )

        # Find result with highest score
        score_field = self._fanout_config.score_field
        best_result = None
        best_score = float("-inf")

        for result in successful_results:
            if not result.message:
                continue

            # Try to get score from metadata
            score = None
            if result.message.metadata:
                score = getattr(result.message.metadata, score_field, None)

            # Also check payload metadata
            if score is None and result.message.payload.metadata:
                score = result.message.payload.metadata.get(score_field)

            if score is not None:
                try:
                    score_value = float(score)
                    if score_value > best_score:
                        best_score = score_value
                        best_result = result
                except (ValueError, TypeError):
                    continue

        # If no scores found, fall back to first result
        if best_result is None:
            return self._aggregate_first_success(
                target_results, successful_results, total_latency_ms, original_message
            )

        aggregated_message = best_result.message

        return FanoutResult(
            success=True,
            target_results=target_results,
            aggregated_message=aggregated_message,
            strategy_used=AggregationStrategy.BEST_SCORE,
            total_latency_ms=total_latency_ms,
            successful_targets=len(successful_results),
            failed_targets=len(target_results) - len(successful_results),
        )
