"""Tool fallbacks for TinyLLM.

This module provides fallback mechanisms for tools,
allowing graceful degradation when tools fail.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Fallback strategies."""

    FIRST_SUCCESS = "first_success"  # Stop at first success
    ALL_FAIL = "all_fail"  # Try all, return first success or all errors
    PRIORITY = "priority"  # Try in priority order


@dataclass
class FallbackResult:
    """Result from fallback execution."""

    success: bool
    output: Any
    tool_id: str
    attempts: int
    errors: List[str] = field(default_factory=list)


class ToolFallback:
    """Provides fallback capabilities for tools."""

    def __init__(
        self,
        primary: Any,
        fallbacks: Optional[List[Any]] = None,
        strategy: FallbackStrategy = FallbackStrategy.FIRST_SUCCESS,
        on_fallback: Optional[Callable[[str, Exception], None]] = None,
    ):
        """Initialize fallback handler.

        Args:
            primary: Primary tool to try first.
            fallbacks: List of fallback tools.
            strategy: Fallback strategy.
            on_fallback: Callback when falling back.
        """
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.strategy = strategy
        self.on_fallback = on_fallback

    async def execute(self, input: Any) -> FallbackResult:
        """Execute with fallback support.

        Args:
            input: Tool input.

        Returns:
            FallbackResult with outcome.
        """
        all_tools = [self.primary] + self.fallbacks
        errors = []

        for i, tool in enumerate(all_tools):
            try:
                output = await tool.execute(input)

                # Check if output indicates success
                if hasattr(output, "success") and not output.success:
                    error_msg = getattr(output, "error", "Tool returned failure")
                    errors.append(f"{tool.metadata.id}: {error_msg}")

                    if i < len(all_tools) - 1 and self.on_fallback:
                        self.on_fallback(tool.metadata.id, ValueError(error_msg))
                    continue

                return FallbackResult(
                    success=True,
                    output=output,
                    tool_id=tool.metadata.id,
                    attempts=i + 1,
                    errors=errors,
                )
            except Exception as e:
                errors.append(f"{tool.metadata.id}: {e}")
                logger.debug(f"Tool {tool.metadata.id} failed: {e}")

                if i < len(all_tools) - 1 and self.on_fallback:
                    self.on_fallback(tool.metadata.id, e)

        return FallbackResult(
            success=False,
            output=None,
            tool_id=all_tools[-1].metadata.id if all_tools else "",
            attempts=len(all_tools),
            errors=errors,
        )


class CircuitBreaker:
    """Circuit breaker for tool fallbacks."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_requests: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening.
            reset_timeout: Seconds before trying again.
            half_open_requests: Requests to try in half-open state.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_requests = half_open_requests

        self._failures: int = 0
        self._last_failure_time: float = 0
        self._state: str = "closed"
        self._half_open_successes: int = 0

    @property
    def state(self) -> str:
        """Get current state."""
        import time

        if self._state == "open":
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = "half_open"
                self._half_open_successes = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == "half_open":
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_requests:
                self._state = "closed"
                self._failures = 0
        else:
            self._failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        import time

        self._failures += 1
        self._last_failure_time = time.monotonic()

        if self._failures >= self.failure_threshold:
            self._state = "open"

    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        return self.state != "open"

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self._failures = 0
        self._state = "closed"
        self._half_open_successes = 0


class CircuitBreakerFallback:
    """Fallback with circuit breaker pattern."""

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        breaker: Optional[CircuitBreaker] = None,
    ):
        """Initialize circuit breaker fallback.

        Args:
            primary: Primary tool.
            fallback: Fallback tool.
            breaker: Optional circuit breaker instance.
        """
        self.primary = primary
        self.fallback = fallback
        self.breaker = breaker or CircuitBreaker()

    async def execute(self, input: Any) -> FallbackResult:
        """Execute with circuit breaker logic.

        Args:
            input: Tool input.

        Returns:
            FallbackResult.
        """
        errors = []

        if self.breaker.is_available():
            try:
                output = await self.primary.execute(input)

                if hasattr(output, "success") and not output.success:
                    self.breaker.record_failure()
                    errors.append(f"{self.primary.metadata.id}: {getattr(output, 'error', 'Failed')}")
                else:
                    self.breaker.record_success()
                    return FallbackResult(
                        success=True,
                        output=output,
                        tool_id=self.primary.metadata.id,
                        attempts=1,
                    )
            except Exception as e:
                self.breaker.record_failure()
                errors.append(f"{self.primary.metadata.id}: {e}")
        else:
            errors.append(f"{self.primary.metadata.id}: Circuit open")

        # Use fallback
        try:
            output = await self.fallback.execute(input)
            return FallbackResult(
                success=True,
                output=output,
                tool_id=self.fallback.metadata.id,
                attempts=2,
                errors=errors,
            )
        except Exception as e:
            errors.append(f"{self.fallback.metadata.id}: {e}")
            return FallbackResult(
                success=False,
                output=None,
                tool_id=self.fallback.metadata.id,
                attempts=2,
                errors=errors,
            )


class FallbackChain:
    """Chain of fallbacks with health tracking."""

    def __init__(self, tools: List[Any]):
        """Initialize fallback chain.

        Args:
            tools: Tools in priority order.
        """
        self.tools = tools
        self._health: dict = {t.metadata.id: {"failures": 0, "available": True} for t in tools}

    def get_available_tools(self) -> List[Any]:
        """Get currently available tools."""
        return [t for t in self.tools if self._health[t.metadata.id]["available"]]

    def mark_failed(self, tool_id: str, threshold: int = 3) -> None:
        """Mark a tool as failed.

        Args:
            tool_id: Tool identifier.
            threshold: Failures before marking unavailable.
        """
        if tool_id in self._health:
            self._health[tool_id]["failures"] += 1
            if self._health[tool_id]["failures"] >= threshold:
                self._health[tool_id]["available"] = False
                logger.warning(f"Tool {tool_id} marked unavailable")

    def mark_success(self, tool_id: str) -> None:
        """Mark a tool as successful.

        Args:
            tool_id: Tool identifier.
        """
        if tool_id in self._health:
            self._health[tool_id]["failures"] = 0
            self._health[tool_id]["available"] = True

    def reset_health(self, tool_id: Optional[str] = None) -> None:
        """Reset health status.

        Args:
            tool_id: Specific tool or None for all.
        """
        if tool_id:
            if tool_id in self._health:
                self._health[tool_id] = {"failures": 0, "available": True}
        else:
            for tid in self._health:
                self._health[tid] = {"failures": 0, "available": True}

    async def execute(self, input: Any) -> FallbackResult:
        """Execute with fallback chain.

        Args:
            input: Tool input.

        Returns:
            FallbackResult.
        """
        available = self.get_available_tools()
        if not available:
            # Reset all and try again
            self.reset_health()
            available = self.tools

        errors = []
        for i, tool in enumerate(available):
            try:
                output = await tool.execute(input)

                if hasattr(output, "success") and not output.success:
                    self.mark_failed(tool.metadata.id)
                    errors.append(f"{tool.metadata.id}: {getattr(output, 'error', 'Failed')}")
                    continue

                self.mark_success(tool.metadata.id)
                return FallbackResult(
                    success=True,
                    output=output,
                    tool_id=tool.metadata.id,
                    attempts=i + 1,
                    errors=errors,
                )
            except Exception as e:
                self.mark_failed(tool.metadata.id)
                errors.append(f"{tool.metadata.id}: {e}")

        return FallbackResult(
            success=False,
            output=None,
            tool_id=available[-1].metadata.id if available else "",
            attempts=len(available),
            errors=errors,
        )


def with_fallback(primary: Any, *fallbacks: Any) -> ToolFallback:
    """Create a tool with fallbacks.

    Args:
        primary: Primary tool.
        *fallbacks: Fallback tools.

    Returns:
        ToolFallback instance.
    """
    return ToolFallback(primary=primary, fallbacks=list(fallbacks))


def with_circuit_breaker(
    primary: Any,
    fallback: Any,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
) -> CircuitBreakerFallback:
    """Create a circuit breaker fallback.

    Args:
        primary: Primary tool.
        fallback: Fallback tool.
        failure_threshold: Failures before opening.
        reset_timeout: Seconds before retrying.

    Returns:
        CircuitBreakerFallback instance.
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        reset_timeout=reset_timeout,
    )
    return CircuitBreakerFallback(primary=primary, fallback=fallback, breaker=breaker)
