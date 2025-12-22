"""Tests for tool fallback mechanisms."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinyllm.tools.fallbacks import (
    CircuitBreaker,
    CircuitBreakerFallback,
    FallbackChain,
    FallbackResult,
    FallbackStrategy,
    ToolFallback,
    with_circuit_breaker,
    with_fallback,
)


# Mock Tool Classes


class MockTool:
    """Mock tool for testing."""

    def __init__(self, tool_id: str, should_fail: bool = False, failure_message: str = ""):
        """Initialize mock tool.

        Args:
            tool_id: Tool identifier.
            should_fail: Whether tool should fail.
            failure_message: Failure message if should_fail.
        """
        self.metadata = MagicMock()
        self.metadata.id = tool_id
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.call_count = 0

    async def execute(self, input: str):
        """Execute mock tool.

        Args:
            input: Tool input.

        Returns:
            Success result or raises exception.
        """
        self.call_count += 1

        if self.should_fail:
            raise RuntimeError(self.failure_message or f"{self.metadata.id} failed")

        return MagicMock(success=True, output=f"Result from {self.metadata.id}")


class MockToolWithResult:
    """Mock tool that returns structured results."""

    def __init__(self, tool_id: str, success: bool = True, error: str = ""):
        """Initialize mock tool.

        Args:
            tool_id: Tool identifier.
            success: Whether result should indicate success.
            error: Error message if success is False.
        """
        self.metadata = MagicMock()
        self.metadata.id = tool_id
        self.success_value = success
        self.error_value = error
        self.call_count = 0

    async def execute(self, input: str):
        """Execute mock tool.

        Args:
            input: Tool input.

        Returns:
            Result object with success/error fields.
        """
        self.call_count += 1

        result = MagicMock()
        result.success = self.success_value
        result.error = self.error_value
        result.output = f"Result from {self.metadata.id}"

        return result


# Tests for ToolFallback


class TestToolFallback:
    """Tests for ToolFallback."""

    @pytest.mark.asyncio
    async def test_primary_success(self):
        """Test successful primary tool execution."""
        primary = MockTool("primary")
        fallback1 = MockTool("fallback1")

        fb = ToolFallback(primary=primary, fallbacks=[fallback1])
        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "primary"
        assert result.attempts == 1
        assert primary.call_count == 1
        assert fallback1.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Test fallback when primary fails."""
        primary = MockTool("primary", should_fail=True)
        fallback1 = MockTool("fallback1")

        fb = ToolFallback(primary=primary, fallbacks=[fallback1])
        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "fallback1"
        assert result.attempts == 2
        assert primary.call_count == 1
        assert fallback1.call_count == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self):
        """Test multiple fallback tools."""
        primary = MockTool("primary", should_fail=True)
        fallback1 = MockTool("fallback1", should_fail=True)
        fallback2 = MockTool("fallback2")

        fb = ToolFallback(primary=primary, fallbacks=[fallback1, fallback2])
        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "fallback2"
        assert result.attempts == 3
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """Test when all tools fail."""
        primary = MockTool("primary", should_fail=True, failure_message="Primary error")
        fallback1 = MockTool("fallback1", should_fail=True, failure_message="Fallback error")

        fb = ToolFallback(primary=primary, fallbacks=[fallback1])
        result = await fb.execute("test input")

        assert result.success is False
        assert result.output is None
        assert result.attempts == 2
        assert len(result.errors) == 2
        assert "Primary error" in result.errors[0]
        assert "Fallback error" in result.errors[1]

    @pytest.mark.asyncio
    async def test_fallback_callback(self):
        """Test fallback callback is called."""
        primary = MockTool("primary", should_fail=True)
        fallback1 = MockTool("fallback1")

        callback_called = []

        def on_fallback(tool_id, error):
            callback_called.append((tool_id, str(error)))

        fb = ToolFallback(primary=primary, fallbacks=[fallback1], on_fallback=on_fallback)
        result = await fb.execute("test input")

        assert result.success is True
        assert len(callback_called) == 1
        assert callback_called[0][0] == "primary"

    @pytest.mark.asyncio
    async def test_result_indicates_failure(self):
        """Test when tool result indicates failure."""
        primary = MockToolWithResult("primary", success=False, error="Tool returned failure")
        fallback1 = MockToolWithResult("fallback1", success=True)

        fb = ToolFallback(primary=primary, fallbacks=[fallback1])
        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "fallback1"
        assert len(result.errors) == 1


# Tests for CircuitBreaker


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == "closed"
        assert breaker.is_available() is True

    def test_opens_after_threshold(self):
        """Test circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == "open"
        assert breaker.is_available() is False

    def test_half_open_after_timeout(self):
        """Test circuit goes half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        # Wait for timeout
        import time

        time.sleep(0.15)

        assert breaker.state == "half_open"
        assert breaker.is_available() is True

    def test_closes_on_success_in_half_open(self):
        """Test circuit closes on success in half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        # Wait for half-open
        import time

        time.sleep(0.15)
        assert breaker.state == "half_open"

        # Record success
        breaker.record_success()
        assert breaker.state == "closed"
        assert breaker.is_available() is True

    def test_reset(self):
        """Test circuit breaker reset."""
        breaker = CircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        breaker.reset()
        assert breaker.state == "closed"
        assert breaker.is_available() is True


# Tests for CircuitBreakerFallback


class TestCircuitBreakerFallback:
    """Tests for CircuitBreakerFallback."""

    @pytest.mark.asyncio
    async def test_primary_success_records_success(self):
        """Test successful primary execution records success."""
        primary = MockTool("primary")
        fallback = MockTool("fallback")
        breaker = CircuitBreaker(failure_threshold=2)

        fb = CircuitBreakerFallback(primary=primary, fallback=fallback, breaker=breaker)
        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "primary"
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_primary_failure_uses_fallback(self):
        """Test primary failure uses fallback."""
        primary = MockTool("primary", should_fail=True)
        fallback = MockTool("fallback")
        breaker = CircuitBreaker(failure_threshold=2)

        fb = CircuitBreakerFallback(primary=primary, fallback=fallback, breaker=breaker)
        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "fallback"
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test circuit opens and uses fallback."""
        primary = MockTool("primary", should_fail=True)
        fallback = MockTool("fallback")
        breaker = CircuitBreaker(failure_threshold=2)

        fb = CircuitBreakerFallback(primary=primary, fallback=fallback, breaker=breaker)

        # First two failures
        await fb.execute("test1")
        await fb.execute("test2")

        assert breaker.state == "open"

        # Third attempt should skip primary due to open circuit
        result = await fb.execute("test3")

        assert result.success is True
        assert result.tool_id == "fallback"
        assert primary.call_count == 2  # Not called the third time
        assert fallback.call_count == 3  # Called all three times

    @pytest.mark.asyncio
    async def test_both_fail(self):
        """Test when both primary and fallback fail."""
        primary = MockTool("primary", should_fail=True)
        fallback = MockTool("fallback", should_fail=True)

        fb = CircuitBreakerFallback(primary=primary, fallback=fallback)
        result = await fb.execute("test input")

        assert result.success is False
        assert len(result.errors) == 2


# Tests for FallbackChain


class TestFallbackChain:
    """Tests for FallbackChain."""

    @pytest.mark.asyncio
    async def test_uses_first_available_tool(self):
        """Test uses first available tool."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        tool3 = MockTool("tool3")

        chain = FallbackChain(tools=[tool1, tool2, tool3])
        result = await chain.execute("test input")

        assert result.success is True
        assert result.tool_id == "tool1"
        assert tool1.call_count == 1
        assert tool2.call_count == 0

    @pytest.mark.asyncio
    async def test_marks_failed_tool_unavailable(self):
        """Test marks failed tool as unavailable."""
        tool1 = MockTool("tool1", should_fail=True)
        tool2 = MockTool("tool2")

        chain = FallbackChain(tools=[tool1, tool2])

        # First execution
        result1 = await chain.execute("test1")
        assert result1.success is True
        assert result1.tool_id == "tool2"

        # tool1 should have 1 failure
        assert chain._health["tool1"]["failures"] == 1

    @pytest.mark.asyncio
    async def test_marks_tool_unavailable_after_threshold(self):
        """Test marks tool unavailable after threshold."""
        tool1 = MockTool("tool1", should_fail=True)
        tool2 = MockTool("tool2")

        chain = FallbackChain(tools=[tool1, tool2])

        # Execute 3 times to exceed default threshold
        for i in range(3):
            await chain.execute(f"test{i}")

        # tool1 should be marked unavailable
        assert chain._health["tool1"]["available"] is False

        # Next execution should skip tool1
        tool1.call_count = 0  # Reset counter
        await chain.execute("test4")

        assert tool1.call_count == 0  # Not called
        assert tool2.call_count == 4  # Called all 4 times

    @pytest.mark.asyncio
    async def test_reset_health(self):
        """Test resetting health status."""
        tool1 = MockTool("tool1", should_fail=True)

        chain = FallbackChain(tools=[tool1])

        # Mark as failed
        for i in range(3):
            await chain.execute(f"test{i}")

        assert chain._health["tool1"]["available"] is False

        # Reset
        chain.reset_health("tool1")

        assert chain._health["tool1"]["available"] is True
        assert chain._health["tool1"]["failures"] == 0

    @pytest.mark.asyncio
    async def test_resets_all_when_all_unavailable(self):
        """Test resets all tools when all are unavailable."""
        tool1 = MockTool("tool1", should_fail=True)
        tool2 = MockTool("tool2", should_fail=True)

        chain = FallbackChain(tools=[tool1, tool2])

        # Mark both as failed
        for i in range(3):
            await chain.execute(f"test{i}")

        # Both should be unavailable
        assert chain._health["tool1"]["available"] is False
        assert chain._health["tool2"]["available"] is False

        # Next execution should reset all and try again
        tool1.should_fail = False  # Make tool1 succeed
        result = await chain.execute("test_reset")

        assert result.success is True
        assert result.tool_id == "tool1"


# Tests for helper functions


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.asyncio
    async def test_with_fallback(self):
        """Test with_fallback helper."""
        primary = MockTool("primary", should_fail=True)
        fallback1 = MockTool("fallback1")
        fallback2 = MockTool("fallback2")

        fb = with_fallback(primary, fallback1, fallback2)

        assert isinstance(fb, ToolFallback)

        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "fallback1"

    @pytest.mark.asyncio
    async def test_with_circuit_breaker(self):
        """Test with_circuit_breaker helper."""
        primary = MockTool("primary")
        fallback = MockTool("fallback")

        fb = with_circuit_breaker(
            primary,
            fallback,
            failure_threshold=3,
            reset_timeout=60.0,
        )

        assert isinstance(fb, CircuitBreakerFallback)
        assert fb.breaker.failure_threshold == 3
        assert fb.breaker.reset_timeout == 60.0

        result = await fb.execute("test input")

        assert result.success is True
        assert result.tool_id == "primary"
