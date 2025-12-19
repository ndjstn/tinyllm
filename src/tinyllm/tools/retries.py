"""Tool retries for TinyLLM.

This module provides retry mechanisms for tools with
various backoff strategies.
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategies."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Configuration for retries."""

    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    multiplier: float = 2.0
    jitter: float = 0.1  # 10% jitter
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {Exception}
    )


@dataclass
class RetryResult:
    """Result from retry execution."""

    success: bool
    output: Any
    attempts: int
    total_delay: float
    errors: List[str] = field(default_factory=list)


class BackoffCalculator(ABC):
    """Abstract backoff calculator."""

    @abstractmethod
    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for attempt.

        Args:
            attempt: Current attempt number (1-based).
            config: Retry configuration.

        Returns:
            Delay in seconds.
        """
        pass


class FixedBackoff(BackoffCalculator):
    """Fixed delay backoff."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        return config.initial_delay


class LinearBackoff(BackoffCalculator):
    """Linear increasing delay."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        delay = config.initial_delay * attempt
        return min(delay, config.max_delay)


class ExponentialBackoff(BackoffCalculator):
    """Exponential backoff."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        delay = config.initial_delay * (config.multiplier ** (attempt - 1))
        return min(delay, config.max_delay)


class ExponentialJitterBackoff(BackoffCalculator):
    """Exponential backoff with jitter."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        base_delay = config.initial_delay * (config.multiplier ** (attempt - 1))
        jitter_range = base_delay * config.jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = base_delay + jitter
        return max(0, min(delay, config.max_delay))


def get_backoff_calculator(strategy: RetryStrategy) -> BackoffCalculator:
    """Get backoff calculator for strategy.

    Args:
        strategy: Retry strategy.

    Returns:
        BackoffCalculator instance.
    """
    calculators = {
        RetryStrategy.FIXED: FixedBackoff(),
        RetryStrategy.LINEAR: LinearBackoff(),
        RetryStrategy.EXPONENTIAL: ExponentialBackoff(),
        RetryStrategy.EXPONENTIAL_JITTER: ExponentialJitterBackoff(),
    }
    return calculators[strategy]


class ToolRetry:
    """Provides retry capabilities for tools."""

    def __init__(
        self,
        tool: Any,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    ):
        """Initialize retry handler.

        Args:
            tool: Tool to retry.
            config: Retry configuration.
            on_retry: Callback on each retry (attempt, error, delay).
        """
        self.tool = tool
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self._backoff = get_backoff_calculator(self.config.strategy)

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        return any(
            isinstance(error, exc_type)
            for exc_type in self.config.retryable_exceptions
        )

    async def execute(self, input: Any) -> RetryResult:
        """Execute with retry logic.

        Args:
            input: Tool input.

        Returns:
            RetryResult with outcome.
        """
        errors = []
        total_delay = 0.0

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                output = await self.tool.execute(input)

                # Check for soft failures
                if hasattr(output, "success") and not output.success:
                    error_msg = getattr(output, "error", "Tool returned failure")
                    errors.append(f"Attempt {attempt}: {error_msg}")

                    if attempt < self.config.max_attempts:
                        delay = self._backoff.calculate(attempt, self.config)
                        total_delay += delay

                        if self.on_retry:
                            self.on_retry(attempt, ValueError(error_msg), delay)

                        await asyncio.sleep(delay)
                        continue

                    return RetryResult(
                        success=False,
                        output=output,
                        attempts=attempt,
                        total_delay=total_delay,
                        errors=errors,
                    )

                return RetryResult(
                    success=True,
                    output=output,
                    attempts=attempt,
                    total_delay=total_delay,
                    errors=errors,
                )

            except Exception as e:
                errors.append(f"Attempt {attempt}: {e}")
                logger.debug(f"Retry attempt {attempt} failed: {e}")

                if not self._is_retryable(e):
                    return RetryResult(
                        success=False,
                        output=None,
                        attempts=attempt,
                        total_delay=total_delay,
                        errors=errors,
                    )

                if attempt < self.config.max_attempts:
                    delay = self._backoff.calculate(attempt, self.config)
                    total_delay += delay

                    if self.on_retry:
                        self.on_retry(attempt, e, delay)

                    await asyncio.sleep(delay)

        return RetryResult(
            success=False,
            output=None,
            attempts=self.config.max_attempts,
            total_delay=total_delay,
            errors=errors,
        )


class RetryableToolWrapper:
    """Wrapper that adds retry to any tool."""

    def __init__(
        self,
        tool: Any,
        max_attempts: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        initial_delay: float = 1.0,
        **kwargs,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            max_attempts: Maximum retry attempts.
            strategy: Backoff strategy.
            initial_delay: Initial delay.
            **kwargs: Additional config options.
        """
        self.tool = tool
        self._retry = ToolRetry(
            tool,
            config=RetryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                initial_delay=initial_delay,
                **kwargs,
            ),
        )

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input: Any) -> Any:
        """Execute with retries.

        Args:
            input: Tool input.

        Returns:
            Tool output.

        Raises:
            Exception: If all retries fail.
        """
        result = await self._retry.execute(input)

        if result.success:
            return result.output

        # Raise last error
        raise RuntimeError(
            f"All {result.attempts} attempts failed: {result.errors[-1]}"
        )


class ConditionalRetry:
    """Retry based on output conditions."""

    def __init__(
        self,
        tool: Any,
        should_retry: Callable[[Any], bool],
        config: Optional[RetryConfig] = None,
    ):
        """Initialize conditional retry.

        Args:
            tool: Tool to retry.
            should_retry: Function to check if should retry.
            config: Retry configuration.
        """
        self.tool = tool
        self.should_retry = should_retry
        self.config = config or RetryConfig()
        self._backoff = get_backoff_calculator(self.config.strategy)

    async def execute(self, input: Any) -> RetryResult:
        """Execute with conditional retry.

        Args:
            input: Tool input.

        Returns:
            RetryResult.
        """
        errors = []
        total_delay = 0.0

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                output = await self.tool.execute(input)

                if not self.should_retry(output):
                    return RetryResult(
                        success=True,
                        output=output,
                        attempts=attempt,
                        total_delay=total_delay,
                        errors=errors,
                    )

                errors.append(f"Attempt {attempt}: Retry condition met")

                if attempt < self.config.max_attempts:
                    delay = self._backoff.calculate(attempt, self.config)
                    total_delay += delay
                    await asyncio.sleep(delay)

            except Exception as e:
                errors.append(f"Attempt {attempt}: {e}")

                if attempt < self.config.max_attempts:
                    delay = self._backoff.calculate(attempt, self.config)
                    total_delay += delay
                    await asyncio.sleep(delay)

        return RetryResult(
            success=False,
            output=None,
            attempts=self.config.max_attempts,
            total_delay=total_delay,
            errors=errors,
        )


def with_retry(
    tool: Any,
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    **kwargs,
) -> ToolRetry:
    """Create a retryable tool.

    Args:
        tool: Tool to wrap.
        max_attempts: Maximum attempts.
        strategy: Backoff strategy.
        **kwargs: Additional config.

    Returns:
        ToolRetry instance.
    """
    return ToolRetry(
        tool,
        config=RetryConfig(
            max_attempts=max_attempts,
            strategy=strategy,
            **kwargs,
        ),
    )


def retry_on(
    *exceptions: Type[Exception],
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
) -> Callable[[Any], ToolRetry]:
    """Decorator factory for retry on specific exceptions.

    Args:
        *exceptions: Exception types to retry on.
        max_attempts: Maximum attempts.
        strategy: Backoff strategy.

    Returns:
        Decorator function.
    """

    def decorator(tool: Any) -> ToolRetry:
        return ToolRetry(
            tool,
            config=RetryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                retryable_exceptions=set(exceptions),
            ),
        )

    return decorator
