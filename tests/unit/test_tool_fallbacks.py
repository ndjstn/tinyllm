"""Tests for tool fallbacks."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.fallbacks import (
    CircuitBreaker,
    CircuitBreakerFallback,
    FallbackChain,
    FallbackResult,
    ToolFallback,
    with_circuit_breaker,
    with_fallback,
)


class FallbackInput(BaseModel):
    """Input for fallback tests."""

    value: int = 0


class FallbackOutput(BaseModel):
    """Output for fallback tests."""

    value: int = 0
    success: bool = True
    error: str | None = None


class SuccessTool(BaseTool[FallbackInput, FallbackOutput]):
    """Tool that succeeds."""

    metadata = ToolMetadata(
        id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        category="utility",
    )
    input_type = FallbackInput
    output_type = FallbackOutput

    async def execute(self, input: FallbackInput) -> FallbackOutput:
        return FallbackOutput(value=input.value * 2)


class FailTool(BaseTool[FallbackInput, FallbackOutput]):
    """Tool that fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = FallbackInput
    output_type = FallbackOutput

    async def execute(self, input: FallbackInput) -> FallbackOutput:
        raise ValueError("Intentional failure")


class SoftFailTool(BaseTool[FallbackInput, FallbackOutput]):
    """Tool that returns failure in output."""

    metadata = ToolMetadata(
        id="soft_fail_tool",
        name="Soft Fail Tool",
        description="Returns failure",
        category="utility",
    )
    input_type = FallbackInput
    output_type = FallbackOutput

    async def execute(self, input: FallbackInput) -> FallbackOutput:
        return FallbackOutput(success=False, error="Soft failure")


class TestToolFallback:
    """Tests for ToolFallback."""

    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        """Test primary tool success."""
        fallback = ToolFallback(
            primary=SuccessTool(),
            fallbacks=[SuccessTool()],
        )

        result = await fallback.execute(FallbackInput(value=5))

        assert result.success
        assert result.tool_id == "success_tool"
        assert result.attempts == 1
        assert result.output.value == 10

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self):
        """Test fallback on exception."""
        fallback = ToolFallback(
            primary=FailTool(),
            fallbacks=[SuccessTool()],
        )

        result = await fallback.execute(FallbackInput(value=5))

        assert result.success
        assert result.tool_id == "success_tool"
        assert result.attempts == 2
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_fallback_on_soft_failure(self):
        """Test fallback on soft failure."""
        fallback = ToolFallback(
            primary=SoftFailTool(),
            fallbacks=[SuccessTool()],
        )

        result = await fallback.execute(FallbackInput(value=5))

        assert result.success
        assert result.tool_id == "success_tool"

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """Test all tools fail."""
        fallback = ToolFallback(
            primary=FailTool(),
            fallbacks=[FailTool()],
        )

        result = await fallback.execute(FallbackInput(value=5))

        assert not result.success
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_on_fallback_callback(self):
        """Test fallback callback is called."""
        callbacks = []

        def on_fallback(tool_id, error):
            callbacks.append(tool_id)

        fallback = ToolFallback(
            primary=FailTool(),
            fallbacks=[SuccessTool()],
            on_fallback=on_fallback,
        )

        await fallback.execute(FallbackInput(value=5))

        assert "fail_tool" in callbacks


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state(self):
        """Test initial state is closed."""
        breaker = CircuitBreaker()

        assert breaker.state == "closed"
        assert breaker.is_available()

    def test_opens_on_failures(self):
        """Test circuit opens on failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_available()

        breaker.record_failure()
        assert not breaker.is_available()
        assert breaker.state == "open"

    def test_success_resets_failures(self):
        """Test success resets failure count."""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()

        assert breaker._failures == 0
        assert breaker.is_available()

    def test_reset(self):
        """Test reset method."""
        breaker = CircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_available()

        breaker.reset()
        assert breaker.is_available()
        assert breaker.state == "closed"


class TestCircuitBreakerFallback:
    """Tests for CircuitBreakerFallback."""

    @pytest.mark.asyncio
    async def test_primary_success(self):
        """Test primary succeeds."""
        cb_fallback = CircuitBreakerFallback(
            primary=SuccessTool(),
            fallback=SuccessTool(),
        )

        result = await cb_fallback.execute(FallbackInput(value=5))

        assert result.success
        assert result.tool_id == "success_tool"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test fallback when primary fails."""
        cb_fallback = CircuitBreakerFallback(
            primary=FailTool(),
            fallback=SuccessTool(),
        )

        result = await cb_fallback.execute(FallbackInput(value=5))

        assert result.success
        assert result.tool_id == "success_tool"
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_circuit_opens(self):
        """Test circuit opens after failures."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=60)
        cb_fallback = CircuitBreakerFallback(
            primary=FailTool(),
            fallback=SuccessTool(),
            breaker=breaker,
        )

        # First two failures open the circuit
        await cb_fallback.execute(FallbackInput(value=5))
        await cb_fallback.execute(FallbackInput(value=5))

        assert not breaker.is_available()

        # Third call should skip primary
        result = await cb_fallback.execute(FallbackInput(value=5))

        # Should have "Circuit open" in errors
        assert any("Circuit open" in e for e in result.errors)


class TestFallbackChain:
    """Tests for FallbackChain."""

    @pytest.mark.asyncio
    async def test_first_succeeds(self):
        """Test first tool succeeds."""
        chain = FallbackChain(tools=[SuccessTool(), SuccessTool()])

        result = await chain.execute(FallbackInput(value=5))

        assert result.success
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_fallback_in_chain(self):
        """Test fallback through chain."""
        chain = FallbackChain(tools=[FailTool(), SuccessTool()])

        result = await chain.execute(FallbackInput(value=5))

        assert result.success
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_tool_marked_unavailable(self):
        """Test tool marked unavailable after failures."""
        chain = FallbackChain(tools=[FailTool(), SuccessTool()])

        # Execute 3 times to hit threshold
        for _ in range(3):
            await chain.execute(FallbackInput(value=5))

        available = chain.get_available_tools()
        assert len(available) == 1
        assert available[0].metadata.id == "success_tool"

    def test_reset_health(self):
        """Test health reset."""
        chain = FallbackChain(tools=[SuccessTool()])
        chain.mark_failed("success_tool")
        chain.mark_failed("success_tool")
        chain.mark_failed("success_tool")

        assert len(chain.get_available_tools()) == 0

        chain.reset_health()
        assert len(chain.get_available_tools()) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_fallback(self):
        """Test with_fallback function."""
        fallback = with_fallback(FailTool(), SuccessTool())

        result = await fallback.execute(FallbackInput(value=5))

        assert result.success

    @pytest.mark.asyncio
    async def test_with_circuit_breaker(self):
        """Test with_circuit_breaker function."""
        cb = with_circuit_breaker(
            FailTool(),
            SuccessTool(),
            failure_threshold=2,
        )

        result = await cb.execute(FallbackInput(value=5))

        assert result.success
