"""Tool composition for TinyLLM.

This module provides composition patterns for combining
multiple tools into new composite tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)
IntermediateT = TypeVar("IntermediateT", bound=BaseModel)


class CompositeInput(BaseModel):
    """Generic composite tool input."""

    data: Any = None
    context: Dict[str, Any] = {}


class CompositeOutput(BaseModel):
    """Generic composite tool output."""

    success: bool = True
    error: str | None = None
    result: Any = None
    metadata: Dict[str, Any] = {}


class SequentialComposite(BaseTool[CompositeInput, CompositeOutput]):
    """Composite tool that executes tools in sequence."""

    def __init__(
        self,
        tools: List[BaseTool],
        name: str = "sequential_composite",
        description: str = "Executes tools in sequence",
        transformers: Optional[List[Callable[[Any], Any]]] = None,
    ):
        """Initialize composite.

        Args:
            tools: Tools to execute.
            name: Composite name.
            description: Composite description.
            transformers: Optional transformers between tools.
        """
        super().__init__()
        self._tools = tools
        self._transformers = transformers or []
        self.metadata = ToolMetadata(
            id=f"composite_{name}",
            name=name,
            description=description,
            category="utility",
        )

    input_type = CompositeInput
    output_type = CompositeOutput

    async def execute(self, input: CompositeInput) -> CompositeOutput:
        """Execute tools in sequence."""
        current = input.data

        for i, tool in enumerate(self._tools):
            try:
                # Apply transformer if available
                if i < len(self._transformers) and self._transformers[i]:
                    current = self._transformers[i](current)

                # Execute tool
                if isinstance(current, BaseModel):
                    result = await tool.execute(current)
                else:
                    # Wrap in tool's input type
                    tool_input = tool.input_type.model_validate({"data": current} if hasattr(tool.input_type.model_fields, "data") else current)
                    result = await tool.execute(tool_input)

                current = result
            except Exception as e:
                return CompositeOutput(
                    success=False,
                    error=f"Tool {tool.metadata.id} failed: {e}",
                )

        return CompositeOutput(success=True, result=current)


class ParallelComposite(BaseTool[CompositeInput, CompositeOutput]):
    """Composite tool that executes tools in parallel."""

    def __init__(
        self,
        tools: List[BaseTool],
        name: str = "parallel_composite",
        description: str = "Executes tools in parallel",
        aggregator: Optional[Callable[[List[Any]], Any]] = None,
    ):
        """Initialize composite.

        Args:
            tools: Tools to execute.
            name: Composite name.
            description: Composite description.
            aggregator: Function to aggregate results.
        """
        super().__init__()
        self._tools = tools
        self._aggregator = aggregator or (lambda results: results)
        self.metadata = ToolMetadata(
            id=f"composite_{name}",
            name=name,
            description=description,
            category="utility",
        )

    input_type = CompositeInput
    output_type = CompositeOutput

    async def execute(self, input: CompositeInput) -> CompositeOutput:
        """Execute tools in parallel."""
        import asyncio

        tasks = []
        for tool in self._tools:
            if isinstance(input.data, BaseModel):
                tasks.append(tool.execute(input.data))
            else:
                tool_input = tool.input_type.model_validate(input.data) if isinstance(input.data, dict) else tool.input_type()
                tasks.append(tool.execute(tool_input))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        outputs = []
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"{self._tools[i].metadata.id}: {result}")
            else:
                outputs.append(result)

        if errors:
            return CompositeOutput(
                success=False,
                error="; ".join(errors),
                result=outputs,
            )

        aggregated = self._aggregator(outputs)
        return CompositeOutput(success=True, result=aggregated)


class ConditionalComposite(BaseTool[CompositeInput, CompositeOutput]):
    """Composite that selects a tool based on condition."""

    def __init__(
        self,
        branches: List[tuple],  # [(condition, tool), ...]
        default: Optional[BaseTool] = None,
        name: str = "conditional_composite",
        description: str = "Selects tool based on condition",
    ):
        """Initialize composite.

        Args:
            branches: List of (condition, tool) tuples.
            default: Default tool if no condition matches.
            name: Composite name.
            description: Composite description.
        """
        super().__init__()
        self._branches = branches
        self._default = default
        self.metadata = ToolMetadata(
            id=f"composite_{name}",
            name=name,
            description=description,
            category="utility",
        )

    input_type = CompositeInput
    output_type = CompositeOutput

    async def execute(self, input: CompositeInput) -> CompositeOutput:
        """Execute selected tool."""
        for condition, tool in self._branches:
            if condition(input.data):
                try:
                    if isinstance(input.data, BaseModel):
                        result = await tool.execute(input.data)
                    else:
                        tool_input = tool.input_type.model_validate(input.data) if isinstance(input.data, dict) else tool.input_type()
                        result = await tool.execute(tool_input)
                    return CompositeOutput(success=True, result=result)
                except Exception as e:
                    return CompositeOutput(success=False, error=str(e))

        if self._default:
            try:
                if isinstance(input.data, BaseModel):
                    result = await self._default.execute(input.data)
                else:
                    tool_input = self._default.input_type.model_validate(input.data) if isinstance(input.data, dict) else self._default.input_type()
                    result = await self._default.execute(tool_input)
                return CompositeOutput(success=True, result=result)
            except Exception as e:
                return CompositeOutput(success=False, error=str(e))

        return CompositeOutput(
            success=False,
            error="No matching condition and no default tool",
        )


class MapComposite(BaseTool[CompositeInput, CompositeOutput]):
    """Composite that maps a tool over a list of inputs."""

    def __init__(
        self,
        tool: BaseTool,
        name: str = "map_composite",
        description: str = "Maps tool over inputs",
        parallel: bool = True,
    ):
        """Initialize composite.

        Args:
            tool: Tool to apply.
            name: Composite name.
            description: Composite description.
            parallel: Whether to execute in parallel.
        """
        super().__init__()
        self._tool = tool
        self._parallel = parallel
        self.metadata = ToolMetadata(
            id=f"composite_{name}",
            name=name,
            description=description,
            category="utility",
        )

    input_type = CompositeInput
    output_type = CompositeOutput

    async def execute(self, input: CompositeInput) -> CompositeOutput:
        """Map tool over inputs."""
        import asyncio

        if not isinstance(input.data, list):
            return CompositeOutput(
                success=False,
                error="Input data must be a list",
            )

        if self._parallel:
            tasks = []
            for item in input.data:
                if isinstance(item, BaseModel):
                    tasks.append(self._tool.execute(item))
                else:
                    tool_input = self._tool.input_type.model_validate(item) if isinstance(item, dict) else self._tool.input_type()
                    tasks.append(self._tool.execute(tool_input))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            outputs = []
            errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Item {i}: {result}")
                else:
                    outputs.append(result)

            if errors:
                return CompositeOutput(
                    success=False,
                    error="; ".join(errors),
                    result=outputs,
                )

            return CompositeOutput(success=True, result=outputs)
        else:
            outputs = []
            for i, item in enumerate(input.data):
                try:
                    if isinstance(item, BaseModel):
                        result = await self._tool.execute(item)
                    else:
                        tool_input = self._tool.input_type.model_validate(item) if isinstance(item, dict) else self._tool.input_type()
                        result = await self._tool.execute(tool_input)
                    outputs.append(result)
                except Exception as e:
                    return CompositeOutput(
                        success=False,
                        error=f"Item {i}: {e}",
                        result=outputs,
                    )

            return CompositeOutput(success=True, result=outputs)


class ReduceComposite(BaseTool[CompositeInput, CompositeOutput]):
    """Composite that reduces a list using a tool."""

    def __init__(
        self,
        tool: BaseTool,
        name: str = "reduce_composite",
        description: str = "Reduces list using tool",
    ):
        """Initialize composite.

        Args:
            tool: Tool to use for reduction.
            name: Composite name.
            description: Composite description.
        """
        super().__init__()
        self._tool = tool
        self.metadata = ToolMetadata(
            id=f"composite_{name}",
            name=name,
            description=description,
            category="utility",
        )

    input_type = CompositeInput
    output_type = CompositeOutput

    async def execute(self, input: CompositeInput) -> CompositeOutput:
        """Reduce list using tool."""
        if not isinstance(input.data, list) or len(input.data) < 2:
            return CompositeOutput(
                success=False,
                error="Input must be a list with at least 2 items",
            )

        accumulator = input.data[0]

        for item in input.data[1:]:
            try:
                # Create input with accumulator and current item
                reduce_input = self._tool.input_type.model_validate({
                    "left": accumulator,
                    "right": item,
                })
                result = await self._tool.execute(reduce_input)
                accumulator = result
            except Exception as e:
                return CompositeOutput(success=False, error=str(e))

        return CompositeOutput(success=True, result=accumulator)


def compose(*tools: BaseTool, name: str = "composed") -> SequentialComposite:
    """Create a sequential composite from tools.

    Args:
        *tools: Tools to compose.
        name: Composite name.

    Returns:
        Sequential composite tool.
    """
    return SequentialComposite(
        tools=list(tools),
        name=name,
        description=f"Composed from: {', '.join(t.metadata.id for t in tools)}",
    )


def parallel(*tools: BaseTool, name: str = "parallel") -> ParallelComposite:
    """Create a parallel composite from tools.

    Args:
        *tools: Tools to run in parallel.
        name: Composite name.

    Returns:
        Parallel composite tool.
    """
    return ParallelComposite(
        tools=list(tools),
        name=name,
        description=f"Parallel: {', '.join(t.metadata.id for t in tools)}",
    )


def conditional(
    *branches: tuple,
    default: Optional[BaseTool] = None,
    name: str = "conditional",
) -> ConditionalComposite:
    """Create a conditional composite.

    Args:
        *branches: (condition, tool) tuples.
        default: Default tool.
        name: Composite name.

    Returns:
        Conditional composite tool.
    """
    return ConditionalComposite(
        branches=list(branches),
        default=default,
        name=name,
    )


def map_tool(
    tool: BaseTool,
    name: str = "mapped",
    parallel: bool = True,
) -> MapComposite:
    """Create a map composite.

    Args:
        tool: Tool to map.
        name: Composite name.
        parallel: Whether to execute in parallel.

    Returns:
        Map composite tool.
    """
    return MapComposite(tool=tool, name=name, parallel=parallel)
