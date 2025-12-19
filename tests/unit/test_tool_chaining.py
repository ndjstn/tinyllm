"""Tests for tool chaining."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.chaining import (
    ChainBuilder,
    ChainResult,
    ChainStopReason,
    ConditionalChain,
    ParallelChain,
    ToolChain,
    chain,
)


class ChainInput(BaseModel):
    """Input for chain test tools."""

    value: int = 0


class ChainOutput(BaseModel):
    """Output for chain test tools."""

    value: int = 0
    success: bool = True
    error: str | None = None


class AddTool(BaseTool[ChainInput, ChainOutput]):
    """Tool that adds a value."""

    metadata = ToolMetadata(
        id="add_tool",
        name="Add Tool",
        description="Adds a value",
        category="computation",
    )
    input_type = ChainInput
    output_type = ChainOutput

    def __init__(self, amount: int = 1):
        super().__init__()
        self.amount = amount

    async def execute(self, input: ChainInput) -> ChainOutput:
        return ChainOutput(value=input.value + self.amount)


class MultiplyTool(BaseTool[ChainInput, ChainOutput]):
    """Tool that multiplies a value."""

    metadata = ToolMetadata(
        id="multiply_tool",
        name="Multiply Tool",
        description="Multiplies a value",
        category="computation",
    )
    input_type = ChainInput
    output_type = ChainOutput

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    async def execute(self, input: ChainInput) -> ChainOutput:
        return ChainOutput(value=input.value * self.factor)


class FailingTool(BaseTool[ChainInput, ChainOutput]):
    """Tool that always fails."""

    metadata = ToolMetadata(
        id="failing_tool",
        name="Failing Tool",
        description="Always fails",
        category="utility",
    )
    input_type = ChainInput
    output_type = ChainOutput

    async def execute(self, input: ChainInput) -> ChainOutput:
        raise ValueError("Intentional failure")


class TestToolChain:
    """Tests for ToolChain."""

    @pytest.mark.asyncio
    async def test_simple_chain(self):
        """Test simple chain execution."""
        chain = ToolChain()
        chain.add_step(AddTool(5))
        chain.add_step(MultiplyTool(2))

        result = await chain.execute(ChainInput(value=10))

        assert result.success
        assert result.stop_reason == ChainStopReason.COMPLETED
        assert result.final_output.value == 30  # (10 + 5) * 2

    @pytest.mark.asyncio
    async def test_chain_with_transform(self):
        """Test chain with input/output transforms."""
        chain = ToolChain()
        chain.add_step(
            AddTool(10),
            transform_input=lambda x: ChainInput(value=x.value * 2),  # Double first
            transform_output=lambda x: ChainOutput(value=x.value + 1),  # Add 1 at end
        )

        result = await chain.execute(ChainInput(value=5))

        # Input: 5 -> transform: 10 -> add 10: 20 -> transform: 21
        assert result.final_output.value == 21

    @pytest.mark.asyncio
    async def test_chain_with_condition(self):
        """Test chain with condition."""
        chain = ToolChain()
        chain.add_step(AddTool(5))
        chain.add_step(
            MultiplyTool(2),
            condition=lambda x: x.value > 20,  # Only multiply if > 20
        )

        result = await chain.execute(ChainInput(value=10))

        # 10 + 5 = 15, condition not met
        assert result.success
        assert result.stop_reason == ChainStopReason.CONDITION_NOT_MET
        assert result.final_output.value == 15

    @pytest.mark.asyncio
    async def test_chain_error_stops(self):
        """Test that error stops chain."""
        chain = ToolChain(stop_on_error=True)
        chain.add_step(AddTool(5))
        chain.add_step(FailingTool())
        chain.add_step(MultiplyTool(2))

        result = await chain.execute(ChainInput(value=10))

        assert not result.success
        assert result.stop_reason == ChainStopReason.ERROR
        assert len(result.steps) == 2  # Only 2 steps executed
        assert "Intentional failure" in result.error

    @pytest.mark.asyncio
    async def test_chain_error_continues(self):
        """Test chain continues on error when configured."""
        chain = ToolChain(stop_on_error=False)
        chain.add_step(AddTool(5))
        chain.add_step(FailingTool())
        chain.add_step(AddTool(10))

        result = await chain.execute(ChainInput(value=10))

        assert result.success  # Final step succeeded
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_chain_error_handler(self):
        """Test chain with error handler."""
        chain = ToolChain()
        chain.add_step(
            FailingTool(),
            on_error=lambda e: ChainOutput(value=0),  # Return 0 on error
        )
        chain.add_step(AddTool(5))

        result = await chain.execute(ChainInput(value=10))

        assert result.success
        assert result.final_output.value == 5  # 0 + 5

    @pytest.mark.asyncio
    async def test_chain_max_steps(self):
        """Test chain max steps limit."""
        chain = ToolChain(max_steps=2)
        for _ in range(5):
            chain.add_step(AddTool(1))

        result = await chain.execute(ChainInput(value=0))

        assert not result.success
        assert result.stop_reason == ChainStopReason.MAX_STEPS_REACHED
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_chain_fluent_syntax(self):
        """Test fluent chain syntax."""
        tc = ToolChain().then(AddTool(5)).then(MultiplyTool(3))

        result = await tc.execute(ChainInput(value=10))

        assert result.final_output.value == 45  # (10 + 5) * 3

    @pytest.mark.asyncio
    async def test_chain_step_results(self):
        """Test that step results are recorded."""
        tc = ToolChain()
        tc.add_step(AddTool(5), name="add_step")
        tc.add_step(MultiplyTool(2), name="multiply_step")

        result = await tc.execute(ChainInput(value=10))

        assert len(result.steps) == 2
        assert result.steps[0].step_name == "add_step"
        assert result.steps[0].success
        assert result.steps[1].step_name == "multiply_step"


class TestConditionalChain:
    """Tests for ConditionalChain."""

    @pytest.mark.asyncio
    async def test_conditional_branch_taken(self):
        """Test conditional branch is taken."""
        add_chain = ToolChain().then(AddTool(10))
        multiply_chain = ToolChain().then(MultiplyTool(10))

        conditional = (
            ConditionalChain()
            .when(lambda x: x.value > 50, add_chain)
            .when(lambda x: x.value > 10, multiply_chain)
            .otherwise(ToolChain().then(AddTool(1)))
        )

        result = await conditional.execute(ChainInput(value=20))

        assert result.success
        assert result.final_output.value == 200  # 20 * 10

    @pytest.mark.asyncio
    async def test_conditional_default(self):
        """Test conditional default branch."""
        conditional = (
            ConditionalChain()
            .when(lambda x: x.value > 100, ToolChain().then(AddTool(100)))
            .otherwise(ToolChain().then(AddTool(1)))
        )

        result = await conditional.execute(ChainInput(value=10))

        assert result.final_output.value == 11  # 10 + 1

    @pytest.mark.asyncio
    async def test_conditional_no_match(self):
        """Test conditional with no matching branch."""
        conditional = ConditionalChain().when(
            lambda x: x.value > 100, ToolChain().then(AddTool(100))
        )

        result = await conditional.execute(ChainInput(value=10))

        assert result.success
        assert result.stop_reason == ChainStopReason.CONDITION_NOT_MET


class TestParallelChain:
    """Tests for ParallelChain."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution."""
        chain1 = ToolChain(name="add").then(AddTool(10))
        chain2 = ToolChain(name="multiply").then(MultiplyTool(10))

        parallel = ParallelChain([chain1, chain2], merge_strategy="list")

        result = await parallel.execute(ChainInput(value=5))

        assert result.success
        assert len(result.final_output) == 2
        assert result.final_output[0].value == 15  # 5 + 10
        assert result.final_output[1].value == 50  # 5 * 10

    @pytest.mark.asyncio
    async def test_parallel_dict_merge(self):
        """Test parallel with dict merge strategy."""
        chain1 = ToolChain(name="add").then(AddTool(10))
        chain2 = ToolChain(name="multiply").then(MultiplyTool(10))

        parallel = ParallelChain([chain1, chain2], merge_strategy="dict")

        result = await parallel.execute(ChainInput(value=5))

        assert "add" in result.final_output
        assert "multiply" in result.final_output

    @pytest.mark.asyncio
    async def test_parallel_with_error(self):
        """Test parallel with one failing chain."""
        chain1 = ToolChain(name="success").then(AddTool(10))
        chain2 = ToolChain(name="fail").then(FailingTool())

        parallel = ParallelChain([chain1, chain2])

        result = await parallel.execute(ChainInput(value=5))

        assert not result.success
        assert result.error is not None


class TestChainBuilder:
    """Tests for ChainBuilder."""

    @pytest.mark.asyncio
    async def test_builder(self):
        """Test chain builder."""
        tc = (
            ChainBuilder("my_chain")
            .with_max_steps(10)
            .stop_on_error(True)
            .add(AddTool(5))
            .add(MultiplyTool(2))
            .build()
        )

        assert tc.name == "my_chain"
        assert tc.max_steps == 10
        assert tc.stop_on_error

        result = await tc.execute(ChainInput(value=10))
        assert result.final_output.value == 30


class TestChainFunction:
    """Tests for chain() convenience function."""

    @pytest.mark.asyncio
    async def test_chain_function(self):
        """Test chain function."""
        tc = chain(AddTool(5), MultiplyTool(2), name="quick_chain")

        result = await tc.execute(ChainInput(value=10))

        assert result.success
        assert result.final_output.value == 30
