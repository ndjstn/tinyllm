"""Tests for tool composition."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.composition import (
    CompositeInput,
    CompositeOutput,
    ConditionalComposite,
    MapComposite,
    ParallelComposite,
    ReduceComposite,
    SequentialComposite,
    compose,
    conditional,
    map_tool,
    parallel,
)


class NumInput(BaseModel):
    """Numeric input."""

    value: int = 0


class NumOutput(BaseModel):
    """Numeric output."""

    value: int = 0
    success: bool = True
    error: str | None = None


class AddTool(BaseTool[NumInput, NumOutput]):
    """Tool that adds."""

    metadata = ToolMetadata(
        id="add_tool",
        name="Add Tool",
        description="Adds a value",
        category="computation",
    )
    input_type = NumInput
    output_type = NumOutput

    def __init__(self, amount: int = 1):
        super().__init__()
        self.amount = amount

    async def execute(self, input: NumInput) -> NumOutput:
        return NumOutput(value=input.value + self.amount)


class DoubleTool(BaseTool[NumInput, NumOutput]):
    """Tool that doubles."""

    metadata = ToolMetadata(
        id="double_tool",
        name="Double Tool",
        description="Doubles value",
        category="computation",
    )
    input_type = NumInput
    output_type = NumOutput

    async def execute(self, input: NumInput) -> NumOutput:
        return NumOutput(value=input.value * 2)


class FailTool(BaseTool[NumInput, NumOutput]):
    """Tool that fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = NumInput
    output_type = NumOutput

    async def execute(self, input: NumInput) -> NumOutput:
        raise ValueError("Intentional failure")


class TestSequentialComposite:
    """Tests for SequentialComposite."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential execution."""
        composite = SequentialComposite(
            tools=[AddTool(5), DoubleTool()],
            name="add_then_double",
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert result.success
        # (10 + 5) * 2 = 30
        assert result.result.value == 30

    @pytest.mark.asyncio
    async def test_sequential_with_transformer(self):
        """Test sequential with transformer."""
        composite = SequentialComposite(
            tools=[AddTool(5), DoubleTool()],
            transformers=[
                lambda x: NumInput(value=x.value * 2),  # Double first
                None,  # No transform before double
            ],
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=5)))

        # 5 -> transform to 10 -> add 5 = 15 -> double = 30
        assert result.success
        assert result.result.value == 30

    @pytest.mark.asyncio
    async def test_sequential_with_error(self):
        """Test sequential stops on error."""
        composite = SequentialComposite(
            tools=[AddTool(5), FailTool(), DoubleTool()],
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert not result.success
        assert "fail_tool" in result.error.lower()


class TestParallelComposite:
    """Tests for ParallelComposite."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution."""
        composite = ParallelComposite(
            tools=[AddTool(5), DoubleTool()],
            name="add_and_double",
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert result.success
        assert len(result.result) == 2

    @pytest.mark.asyncio
    async def test_parallel_with_aggregator(self):
        """Test parallel with aggregator."""
        composite = ParallelComposite(
            tools=[AddTool(5), AddTool(10)],
            aggregator=lambda results: sum(r.value for r in results),
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert result.success
        # 15 + 20 = 35
        assert result.result == 35

    @pytest.mark.asyncio
    async def test_parallel_with_error(self):
        """Test parallel handles errors."""
        composite = ParallelComposite(
            tools=[AddTool(5), FailTool()],
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert not result.success
        assert result.error is not None


class TestConditionalComposite:
    """Tests for ConditionalComposite."""

    @pytest.mark.asyncio
    async def test_condition_matches(self):
        """Test condition matching."""
        composite = ConditionalComposite(
            branches=[
                (lambda x: x.value > 50, AddTool(100)),
                (lambda x: x.value > 10, DoubleTool()),
            ],
            default=AddTool(1),
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=20)))

        assert result.success
        assert result.result.value == 40  # 20 * 2

    @pytest.mark.asyncio
    async def test_default_used(self):
        """Test default branch used."""
        composite = ConditionalComposite(
            branches=[
                (lambda x: x.value > 100, AddTool(100)),
            ],
            default=AddTool(1),
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=5)))

        assert result.success
        assert result.result.value == 6  # 5 + 1

    @pytest.mark.asyncio
    async def test_no_match_no_default(self):
        """Test no match and no default."""
        composite = ConditionalComposite(
            branches=[
                (lambda x: x.value > 100, AddTool(100)),
            ],
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=5)))

        assert not result.success


class TestMapComposite:
    """Tests for MapComposite."""

    @pytest.mark.asyncio
    async def test_map_parallel(self):
        """Test parallel map."""
        composite = MapComposite(
            tool=DoubleTool(),
            parallel=True,
        )

        result = await composite.execute(
            CompositeInput(
                data=[NumInput(value=1), NumInput(value=2), NumInput(value=3)]
            )
        )

        assert result.success
        assert len(result.result) == 3
        assert result.result[0].value == 2
        assert result.result[1].value == 4
        assert result.result[2].value == 6

    @pytest.mark.asyncio
    async def test_map_sequential(self):
        """Test sequential map."""
        composite = MapComposite(
            tool=DoubleTool(),
            parallel=False,
        )

        result = await composite.execute(
            CompositeInput(
                data=[NumInput(value=1), NumInput(value=2)]
            )
        )

        assert result.success
        assert len(result.result) == 2

    @pytest.mark.asyncio
    async def test_map_invalid_input(self):
        """Test map with non-list input."""
        composite = MapComposite(tool=DoubleTool())

        result = await composite.execute(CompositeInput(data=NumInput(value=5)))

        assert not result.success
        assert "list" in result.error.lower()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_compose(self):
        """Test compose function."""
        composite = compose(AddTool(5), DoubleTool(), name="my_compose")

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert result.success
        assert result.result.value == 30

    @pytest.mark.asyncio
    async def test_parallel_function(self):
        """Test parallel function."""
        composite = parallel(AddTool(5), DoubleTool(), name="my_parallel")

        result = await composite.execute(CompositeInput(data=NumInput(value=10)))

        assert result.success
        assert len(result.result) == 2

    @pytest.mark.asyncio
    async def test_conditional_function(self):
        """Test conditional function."""
        composite = conditional(
            (lambda x: x.value > 10, DoubleTool()),
            default=AddTool(1),
            name="my_conditional",
        )

        result = await composite.execute(CompositeInput(data=NumInput(value=20)))

        assert result.success
        assert result.result.value == 40

    @pytest.mark.asyncio
    async def test_map_tool_function(self):
        """Test map_tool function."""
        composite = map_tool(DoubleTool(), name="my_map")

        result = await composite.execute(
            CompositeInput(data=[NumInput(value=1), NumInput(value=2)])
        )

        assert result.success
        assert len(result.result) == 2
