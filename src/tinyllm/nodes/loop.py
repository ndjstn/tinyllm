"""Loop node implementation.

Enables iterative workflows with condition-based termination.
"""

import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import ErrorInfo, Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class LoopCondition(str, Enum):
    """Types of loop termination conditions."""

    FIXED_COUNT = "fixed_count"
    UNTIL_SUCCESS = "until_success"
    UNTIL_CONDITION = "until_condition"
    WHILE_CONDITION = "while_condition"


class LoopConfig(NodeConfig):
    """Configuration for loop nodes."""

    model_config = ConfigDict(strict=False, frozen=True, extra="forbid")

    body_node: str = Field(
        description="Node ID to execute in loop body",
        pattern=r"^[a-z][a-z0-9_\.]*$",
    )
    condition_type: LoopCondition = Field(
        default=LoopCondition.FIXED_COUNT,
        description="Type of loop termination condition",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of iterations (safety limit)",
    )
    timeout_ms: int = Field(
        default=60000,
        ge=1000,
        le=300000,
        description="Maximum execution time in milliseconds",
    )
    fixed_count: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Number of iterations for FIXED_COUNT mode",
    )
    condition_expression: Optional[str] = Field(
        default=None,
        description="Python expression for UNTIL_CONDITION or WHILE_CONDITION modes",
    )
    collect_results: bool = Field(
        default=True,
        description="Whether to collect all iteration results",
    )
    continue_on_error: bool = Field(
        default=False,
        description="Whether to continue loop on iteration errors",
    )
    pass_iteration_number: bool = Field(
        default=True,
        description="Whether to pass iteration number in message metadata",
    )


class LoopState(BaseModel):
    """State tracking for loop execution."""

    model_config = ConfigDict(strict=False, frozen=True, extra="forbid")

    iteration_count: int = Field(default=0, ge=0, description="Current iteration number")
    accumulated_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from all iterations",
    )
    elapsed_time_ms: int = Field(default=0, ge=0, description="Total elapsed time")
    success_count: int = Field(default=0, ge=0, description="Number of successful iterations")
    failure_count: int = Field(default=0, ge=0, description="Number of failed iterations")
    last_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Result from last iteration",
    )
    terminated_by: Optional[str] = Field(
        default=None,
        description="Reason for termination",
    )


class LoopResult(BaseModel):
    """Final result from loop execution."""

    model_config = ConfigDict(strict=False, frozen=True, extra="forbid")

    success: bool = Field(description="Whether loop completed successfully")
    iterations_executed: int = Field(ge=0, description="Number of iterations executed")
    all_iterations: List[Dict[str, Any]] = Field(
        description="Data from all iterations",
    )
    final_output: Optional[str] = Field(
        default=None,
        description="Final output content",
    )
    termination_reason: str = Field(description="Why the loop terminated")
    total_elapsed_ms: int = Field(ge=0, description="Total loop execution time")
    success_rate: float = Field(ge=0.0, le=1.0, description="Ratio of successful iterations")


@NodeRegistry.register(NodeType.LOOP)
class LoopNode(BaseNode):
    """Loop node for iterative workflows.

    The LoopNode executes a body node repeatedly until a termination
    condition is met. It supports multiple termination strategies:
    - FIXED_COUNT: Run exactly N times
    - UNTIL_SUCCESS: Run until body succeeds
    - UNTIL_CONDITION: Run until condition expression evaluates true
    - WHILE_CONDITION: Run while condition is true

    The loop always respects max_iterations and timeout limits for safety.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize loop node."""
        super().__init__(definition)
        # Validate and freeze configuration
        self._loop_config = LoopConfig(**definition.config)

    @property
    def loop_config(self) -> LoopConfig:
        """Get loop-specific configuration."""
        return self._loop_config

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute the loop.

        Args:
            message: Input message to process.
            context: Execution context.

        Returns:
            NodeResult with loop execution results.
        """
        start_time = time.time()
        state = LoopState()

        # Validate configuration based on condition type
        validation_error = self._validate_config()
        if validation_error:
            return NodeResult.failure_result(
                error=validation_error,
                metadata={"state": state.model_dump()},
            )

        try:
            # Execute loop iterations
            state = await self._execute_loop(message, context, state, start_time)

            # Build final result
            loop_result = self._build_loop_result(state)

            # Create output message with loop results
            output_payload = MessagePayload(
                task=message.payload.task,
                content=loop_result.final_output,
                metadata={
                    **message.payload.metadata,
                    "loop_result": loop_result.model_dump(),
                    "iterations": loop_result.iterations_executed,
                    "termination": loop_result.termination_reason,
                },
            )

            output_message = message.create_child(
                source_node=self.id,
                payload=output_payload,
            )

            return NodeResult.success_result(
                output_messages=[output_message],
                next_nodes=[],  # Determined by executor from edges
                metadata={
                    "loop_result": loop_result.model_dump(),
                    "iterations": loop_result.iterations_executed,
                    "success_rate": loop_result.success_rate,
                },
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return NodeResult.failure_result(
                error=f"Loop execution failed: {str(e)}",
                metadata={
                    "state": state.model_dump(),
                    "elapsed_ms": elapsed_ms,
                },
            )

    async def _execute_loop(
        self,
        message: Message,
        context: "ExecutionContext",
        state: LoopState,
        start_time: float,
    ) -> LoopState:
        """Execute loop iterations.

        Args:
            message: Input message.
            context: Execution context.
            state: Current loop state.
            start_time: Loop start time.

        Returns:
            Updated loop state.
        """
        iteration_results = []
        success_count = 0
        failure_count = 0
        last_result = None
        terminated_by = None

        while True:
            # Check safety limits
            elapsed_ms = int((time.time() - start_time) * 1000)

            if len(iteration_results) >= self._loop_config.max_iterations:
                terminated_by = "max_iterations_reached"
                break

            if elapsed_ms >= self._loop_config.timeout_ms:
                terminated_by = "timeout_exceeded"
                break

            # Check termination condition before iteration
            if self._should_terminate_before(
                iteration_results, success_count, failure_count, last_result, context
            ):
                terminated_by = self._get_termination_reason(
                    iteration_results, success_count
                )
                break

            # Execute iteration
            iteration_num = len(iteration_results) + 1
            iteration_result = await self._execute_iteration(
                message, context, iteration_num, elapsed_ms
            )

            # Track result
            iteration_results.append(iteration_result)
            last_result = iteration_result

            if iteration_result.get("success", False):
                success_count += 1
            else:
                failure_count += 1

                # Stop if error and not continuing
                if not self._loop_config.continue_on_error:
                    terminated_by = "iteration_failed"
                    break

            # Check termination condition after iteration
            if self._should_terminate_after(
                iteration_results, success_count, failure_count, last_result, context
            ):
                terminated_by = self._get_termination_reason(
                    iteration_results, success_count
                )
                break

        # Build final state
        final_elapsed_ms = int((time.time() - start_time) * 1000)

        return LoopState(
            iteration_count=len(iteration_results),
            accumulated_results=iteration_results if self._loop_config.collect_results else [],
            elapsed_time_ms=final_elapsed_ms,
            success_count=success_count,
            failure_count=failure_count,
            last_result=last_result,
            terminated_by=terminated_by or "condition_met",
        )

    async def _execute_iteration(
        self,
        message: Message,
        context: "ExecutionContext",
        iteration_num: int,
        elapsed_ms: int,
    ) -> Dict[str, Any]:
        """Execute a single loop iteration.

        Args:
            message: Input message.
            context: Execution context.
            iteration_num: Current iteration number (1-indexed).
            elapsed_ms: Time elapsed since loop start.

        Returns:
            Iteration result dictionary.
        """
        iteration_start = time.time()

        try:
            # Create iteration message with metadata
            iteration_payload = message.payload.model_copy()

            if self._loop_config.pass_iteration_number:
                iteration_payload.metadata["iteration"] = iteration_num
                iteration_payload.metadata["loop_elapsed_ms"] = elapsed_ms

            iteration_message = message.create_child(
                source_node=self.id,
                target_node=self._loop_config.body_node,
                payload=iteration_payload,
            )

            # Get body node from context and execute
            # Note: In real execution, this would be handled by the executor
            # For now, we simulate the body execution
            body_result = await self._simulate_body_execution(
                iteration_message, context
            )

            iteration_time = int((time.time() - iteration_start) * 1000)

            return {
                "iteration": iteration_num,
                "success": body_result.get("success", False),
                "output": body_result.get("output"),
                "metadata": body_result.get("metadata", {}),
                "elapsed_ms": iteration_time,
                "error": body_result.get("error"),
            }

        except Exception as e:
            iteration_time = int((time.time() - iteration_start) * 1000)
            return {
                "iteration": iteration_num,
                "success": False,
                "output": None,
                "error": str(e),
                "elapsed_ms": iteration_time,
            }

    async def _simulate_body_execution(
        self, message: Message, context: "ExecutionContext"
    ) -> Dict[str, Any]:
        """Simulate body node execution.

        In production, the executor would handle this by routing to the body node.
        This is a placeholder for testing.

        Args:
            message: Message to process.
            context: Execution context.

        Returns:
            Simulated execution result.
        """
        # This is a placeholder - in real execution, the executor
        # would route to the body node and return its result
        return {
            "success": True,
            "output": f"Iteration {message.payload.metadata.get('iteration', 0)} completed",
            "metadata": {},
        }

    def _should_terminate_before(
        self,
        results: List[Dict[str, Any]],
        success_count: int,
        failure_count: int,
        last_result: Optional[Dict[str, Any]],
        context: "ExecutionContext",
    ) -> bool:
        """Check if loop should terminate before next iteration.

        Args:
            results: All iteration results so far.
            success_count: Number of successful iterations.
            failure_count: Number of failed iterations.
            last_result: Last iteration result.
            context: Execution context.

        Returns:
            True if loop should terminate.
        """
        # FIXED_COUNT checks after iteration, not before
        if self._loop_config.condition_type == LoopCondition.FIXED_COUNT:
            return False

        # WHILE_CONDITION checks before iteration
        if self._loop_config.condition_type == LoopCondition.WHILE_CONDITION:
            return not self._evaluate_condition(results, last_result, context)

        return False

    def _should_terminate_after(
        self,
        results: List[Dict[str, Any]],
        success_count: int,
        failure_count: int,
        last_result: Optional[Dict[str, Any]],
        context: "ExecutionContext",
    ) -> bool:
        """Check if loop should terminate after current iteration.

        Args:
            results: All iteration results so far.
            success_count: Number of successful iterations.
            failure_count: Number of failed iterations.
            last_result: Last iteration result.
            context: Execution context.

        Returns:
            True if loop should terminate.
        """
        condition_type = self._loop_config.condition_type

        # FIXED_COUNT: Check if we've reached target
        if condition_type == LoopCondition.FIXED_COUNT:
            target = self._loop_config.fixed_count or self._loop_config.max_iterations
            return len(results) >= target

        # UNTIL_SUCCESS: Check if last iteration succeeded
        if condition_type == LoopCondition.UNTIL_SUCCESS:
            return last_result is not None and last_result.get("success", False)

        # UNTIL_CONDITION: Check if condition is now true
        if condition_type == LoopCondition.UNTIL_CONDITION:
            return self._evaluate_condition(results, last_result, context)

        # WHILE_CONDITION checks before iteration, not after
        if condition_type == LoopCondition.WHILE_CONDITION:
            return False

        return False

    def _evaluate_condition(
        self,
        results: List[Dict[str, Any]],
        last_result: Optional[Dict[str, Any]],
        context: "ExecutionContext",
    ) -> bool:
        """Evaluate condition expression.

        Args:
            results: All iteration results.
            last_result: Last iteration result.
            context: Execution context.

        Returns:
            True if condition is met.
        """
        if not self._loop_config.condition_expression:
            return False

        # Build evaluation context
        eval_context = self._build_eval_context(results, last_result, context)

        try:
            # Safely evaluate expression
            result = eval(
                self._loop_config.condition_expression,
                {"__builtins__": {}},
                eval_context,
            )
            return bool(result)
        except Exception:
            # On evaluation error, don't terminate
            return False

    def _build_eval_context(
        self,
        results: List[Dict[str, Any]],
        last_result: Optional[Dict[str, Any]],
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """Build evaluation context for condition expressions.

        Args:
            results: All iteration results.
            last_result: Last iteration result.
            context: Execution context.

        Returns:
            Evaluation context dictionary.
        """
        return {
            # Loop state
            "iteration": len(results),
            "results": results,
            "last_result": last_result,
            "success_count": sum(1 for r in results if r.get("success", False)),
            "failure_count": sum(1 for r in results if not r.get("success", False)),
            # Last result fields for convenience
            "last_success": last_result.get("success", False) if last_result else False,
            "last_output": last_result.get("output") if last_result else None,
            "last_error": last_result.get("error") if last_result else None,
            # Execution context
            "variables": context.variables,
            # Utility functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "any": any,
            "all": all,
            "sum": sum,
            "min": min,
            "max": max,
        }

    def _get_termination_reason(
        self, results: List[Dict[str, Any]], success_count: int
    ) -> str:
        """Get human-readable termination reason.

        Args:
            results: All iteration results.
            success_count: Number of successful iterations.

        Returns:
            Termination reason string.
        """
        condition_type = self._loop_config.condition_type

        if condition_type == LoopCondition.FIXED_COUNT:
            return "fixed_count_reached"
        elif condition_type == LoopCondition.UNTIL_SUCCESS:
            return "success_achieved"
        elif condition_type == LoopCondition.UNTIL_CONDITION:
            return "condition_met"
        elif condition_type == LoopCondition.WHILE_CONDITION:
            return "condition_false"

        return "unknown"

    def _build_loop_result(self, state: LoopState) -> LoopResult:
        """Build final loop result from state.

        Args:
            state: Final loop state.

        Returns:
            LoopResult object.
        """
        # Get final output from last successful iteration
        final_output = None
        for result in reversed(state.accumulated_results):
            if result.get("success") and result.get("output"):
                final_output = result["output"]
                break

        # Calculate success rate
        total = state.iteration_count
        success_rate = state.success_count / total if total > 0 else 0.0

        return LoopResult(
            success=state.success_count > 0,
            iterations_executed=state.iteration_count,
            all_iterations=state.accumulated_results,
            final_output=final_output,
            termination_reason=state.terminated_by or "completed",
            total_elapsed_ms=state.elapsed_time_ms,
            success_rate=success_rate,
        )

    def _validate_config(self) -> Optional[str]:
        """Validate loop configuration.

        Returns:
            Error message if invalid, None if valid.
        """
        condition_type = self._loop_config.condition_type

        # FIXED_COUNT requires fixed_count
        if condition_type == LoopCondition.FIXED_COUNT:
            if self._loop_config.fixed_count is None:
                return "FIXED_COUNT condition requires 'fixed_count' parameter"
            if self._loop_config.fixed_count > self._loop_config.max_iterations:
                return f"fixed_count ({self._loop_config.fixed_count}) exceeds max_iterations ({self._loop_config.max_iterations})"

        # UNTIL_CONDITION and WHILE_CONDITION require condition_expression
        if condition_type in [LoopCondition.UNTIL_CONDITION, LoopCondition.WHILE_CONDITION]:
            if not self._loop_config.condition_expression:
                return f"{condition_type} requires 'condition_expression' parameter"

        return None
