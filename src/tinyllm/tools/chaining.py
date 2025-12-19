"""Tool chaining for TinyLLM.

This module provides tool chaining capabilities where
tools can be executed in sequence with data flowing between them.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel


class ChainStopReason(str, Enum):
    """Reasons for stopping a chain."""

    COMPLETED = "completed"
    ERROR = "error"
    CONDITION_NOT_MET = "condition_not_met"
    MAX_STEPS_REACHED = "max_steps_reached"
    CANCELLED = "cancelled"


@dataclass
class ChainStep:
    """A step in a tool chain."""

    tool: Any
    name: Optional[str] = None
    transform_input: Optional[Callable[[Any], Any]] = None
    transform_output: Optional[Callable[[Any], Any]] = None
    condition: Optional[Callable[[Any], bool]] = None
    on_error: Optional[Callable[[Exception], Any]] = None
    timeout_ms: Optional[int] = None


@dataclass
class StepResult:
    """Result from a chain step."""

    step_name: str
    tool_id: str
    input: Any
    output: Any
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class ChainResult:
    """Result from executing a tool chain."""

    success: bool
    final_output: Any
    stop_reason: ChainStopReason
    steps: List[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0
    error: Optional[str] = None


class ToolChain:
    """Executes tools in sequence."""

    def __init__(
        self,
        name: str = "chain",
        max_steps: int = 100,
        stop_on_error: bool = True,
    ):
        """Initialize chain.

        Args:
            name: Chain name.
            max_steps: Maximum steps to execute.
            stop_on_error: Whether to stop on first error.
        """
        self.name = name
        self.max_steps = max_steps
        self.stop_on_error = stop_on_error
        self._steps: List[ChainStep] = []

    def add_step(
        self,
        tool: Any,
        name: Optional[str] = None,
        transform_input: Optional[Callable[[Any], Any]] = None,
        transform_output: Optional[Callable[[Any], Any]] = None,
        condition: Optional[Callable[[Any], bool]] = None,
        on_error: Optional[Callable[[Exception], Any]] = None,
        timeout_ms: Optional[int] = None,
    ) -> "ToolChain":
        """Add a step to the chain.

        Args:
            tool: Tool to execute.
            name: Step name (defaults to tool ID).
            transform_input: Function to transform input before execution.
            transform_output: Function to transform output after execution.
            condition: Condition to check before executing (receives previous output).
            on_error: Error handler.
            timeout_ms: Step timeout.

        Returns:
            Self for chaining.
        """
        self._steps.append(
            ChainStep(
                tool=tool,
                name=name or getattr(tool.metadata, "id", f"step_{len(self._steps)}"),
                transform_input=transform_input,
                transform_output=transform_output,
                condition=condition,
                on_error=on_error,
                timeout_ms=timeout_ms,
            )
        )
        return self

    def then(
        self,
        tool: Any,
        **kwargs,
    ) -> "ToolChain":
        """Alias for add_step with fluent syntax.

        Args:
            tool: Tool to execute.
            **kwargs: Additional step options.

        Returns:
            Self for chaining.
        """
        return self.add_step(tool, **kwargs)

    async def execute(self, initial_input: Any) -> ChainResult:
        """Execute the chain.

        Args:
            initial_input: Initial input to the first step.

        Returns:
            ChainResult with all step results.
        """
        import time

        start_time = time.monotonic()
        step_results: List[StepResult] = []
        current_output = initial_input

        for i, step in enumerate(self._steps):
            if i >= self.max_steps:
                return ChainResult(
                    success=False,
                    final_output=current_output,
                    stop_reason=ChainStopReason.MAX_STEPS_REACHED,
                    steps=step_results,
                    total_duration_ms=(time.monotonic() - start_time) * 1000,
                )

            # Check condition
            if step.condition and not step.condition(current_output):
                return ChainResult(
                    success=True,
                    final_output=current_output,
                    stop_reason=ChainStopReason.CONDITION_NOT_MET,
                    steps=step_results,
                    total_duration_ms=(time.monotonic() - start_time) * 1000,
                )

            # Transform input
            step_input = current_output
            if step.transform_input:
                step_input = step.transform_input(step_input)

            # Execute step
            step_start = time.monotonic()
            try:
                if step.timeout_ms:
                    output = await asyncio.wait_for(
                        step.tool.execute(step_input),
                        timeout=step.timeout_ms / 1000,
                    )
                else:
                    output = await step.tool.execute(step_input)

                # Transform output
                if step.transform_output:
                    output = step.transform_output(output)

                step_results.append(
                    StepResult(
                        step_name=step.name,
                        tool_id=step.tool.metadata.id,
                        input=step_input,
                        output=output,
                        success=True,
                        duration_ms=(time.monotonic() - step_start) * 1000,
                    )
                )
                current_output = output

            except Exception as e:
                if step.on_error:
                    output = step.on_error(e)
                    step_results.append(
                        StepResult(
                            step_name=step.name,
                            tool_id=step.tool.metadata.id,
                            input=step_input,
                            output=output,
                            success=False,
                            error=str(e),
                            duration_ms=(time.monotonic() - step_start) * 1000,
                        )
                    )
                    current_output = output
                elif self.stop_on_error:
                    step_results.append(
                        StepResult(
                            step_name=step.name,
                            tool_id=step.tool.metadata.id,
                            input=step_input,
                            output=None,
                            success=False,
                            error=str(e),
                            duration_ms=(time.monotonic() - step_start) * 1000,
                        )
                    )
                    return ChainResult(
                        success=False,
                        final_output=None,
                        stop_reason=ChainStopReason.ERROR,
                        steps=step_results,
                        total_duration_ms=(time.monotonic() - start_time) * 1000,
                        error=str(e),
                    )
                else:
                    step_results.append(
                        StepResult(
                            step_name=step.name,
                            tool_id=step.tool.metadata.id,
                            input=step_input,
                            output=None,
                            success=False,
                            error=str(e),
                            duration_ms=(time.monotonic() - step_start) * 1000,
                        )
                    )

        return ChainResult(
            success=True,
            final_output=current_output,
            stop_reason=ChainStopReason.COMPLETED,
            steps=step_results,
            total_duration_ms=(time.monotonic() - start_time) * 1000,
        )


class ConditionalChain:
    """Chain with conditional branching."""

    def __init__(self, name: str = "conditional_chain"):
        """Initialize conditional chain.

        Args:
            name: Chain name.
        """
        self.name = name
        self._branches: List[tuple] = []
        self._default: Optional[ToolChain] = None

    def when(
        self,
        condition: Callable[[Any], bool],
        chain: ToolChain,
    ) -> "ConditionalChain":
        """Add a conditional branch.

        Args:
            condition: Condition function.
            chain: Chain to execute if condition is true.

        Returns:
            Self for chaining.
        """
        self._branches.append((condition, chain))
        return self

    def otherwise(self, chain: ToolChain) -> "ConditionalChain":
        """Set default chain.

        Args:
            chain: Default chain.

        Returns:
            Self for chaining.
        """
        self._default = chain
        return self

    async def execute(self, input: Any) -> ChainResult:
        """Execute the appropriate branch.

        Args:
            input: Input data.

        Returns:
            ChainResult from the executed branch.
        """
        for condition, chain in self._branches:
            if condition(input):
                return await chain.execute(input)

        if self._default:
            return await self._default.execute(input)

        return ChainResult(
            success=True,
            final_output=input,
            stop_reason=ChainStopReason.CONDITION_NOT_MET,
        )


class ParallelChain:
    """Execute multiple chains in parallel."""

    def __init__(
        self,
        chains: List[ToolChain],
        merge_strategy: str = "list",  # list, dict, first, last
    ):
        """Initialize parallel chain.

        Args:
            chains: Chains to execute in parallel.
            merge_strategy: How to merge results.
        """
        self.chains = chains
        self.merge_strategy = merge_strategy

    async def execute(self, input: Any) -> ChainResult:
        """Execute all chains in parallel.

        Args:
            input: Input data (sent to all chains).

        Returns:
            ChainResult with merged output.
        """
        import time

        start_time = time.monotonic()
        tasks = [chain.execute(input) for chain in self.chains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        outputs = []
        all_steps = []
        all_success = True
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_success = False
                errors.append(str(result))
                outputs.append(None)
            else:
                all_steps.extend(result.steps)
                outputs.append(result.final_output)
                if not result.success:
                    all_success = False
                    if result.error:
                        errors.append(result.error)

        # Merge outputs
        if self.merge_strategy == "list":
            merged = outputs
        elif self.merge_strategy == "dict":
            merged = {
                chain.name: output for chain, output in zip(self.chains, outputs)
            }
        elif self.merge_strategy == "first":
            merged = outputs[0] if outputs else None
        elif self.merge_strategy == "last":
            merged = outputs[-1] if outputs else None
        else:
            merged = outputs

        return ChainResult(
            success=all_success,
            final_output=merged,
            stop_reason=ChainStopReason.COMPLETED if all_success else ChainStopReason.ERROR,
            steps=all_steps,
            total_duration_ms=(time.monotonic() - start_time) * 1000,
            error="; ".join(errors) if errors else None,
        )


class ChainBuilder:
    """Fluent builder for tool chains."""

    def __init__(self, name: str = "chain"):
        """Initialize builder.

        Args:
            name: Chain name.
        """
        self._chain = ToolChain(name=name)

    def with_max_steps(self, max_steps: int) -> "ChainBuilder":
        """Set max steps.

        Args:
            max_steps: Maximum steps.

        Returns:
            Self for chaining.
        """
        self._chain.max_steps = max_steps
        return self

    def stop_on_error(self, stop: bool = True) -> "ChainBuilder":
        """Set stop on error behavior.

        Args:
            stop: Whether to stop on error.

        Returns:
            Self for chaining.
        """
        self._chain.stop_on_error = stop
        return self

    def add(self, tool: Any, **kwargs) -> "ChainBuilder":
        """Add a step.

        Args:
            tool: Tool to add.
            **kwargs: Step options.

        Returns:
            Self for chaining.
        """
        self._chain.add_step(tool, **kwargs)
        return self

    def build(self) -> ToolChain:
        """Build the chain.

        Returns:
            Configured ToolChain.
        """
        return self._chain


def chain(*tools: Any, **kwargs) -> ToolChain:
    """Create a simple chain from tools.

    Args:
        *tools: Tools to chain.
        **kwargs: Chain options.

    Returns:
        Configured ToolChain.
    """
    tc = ToolChain(**kwargs)
    for tool in tools:
        tc.add_step(tool)
    return tc
