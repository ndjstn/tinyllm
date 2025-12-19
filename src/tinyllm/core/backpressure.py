"""Backpressure handling for TinyLLM graph execution.

This module provides backpressure mechanisms to prevent resource exhaustion
during graph execution, including flow control, rate limiting, and adaptive
throttling.
"""

import asyncio
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="backpressure")


class BackpressureConfig(BaseModel):
    """Configuration for backpressure handling."""

    model_config = {"extra": "forbid"}

    max_queue_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum queue size before applying backpressure",
    )
    high_watermark: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Queue utilization threshold to trigger backpressure (0.0-1.0)",
    )
    low_watermark: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Queue utilization threshold to release backpressure (0.0-1.0)",
    )
    max_rate_per_second: int = Field(
        default=1000,
        ge=1,
        le=1000000,
        description="Maximum processing rate per second",
    )
    enable_adaptive_throttling: bool = Field(
        default=True,
        description="Enable adaptive throttling based on system load",
    )
    throttle_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=10.0,
        description="Multiplier for throttling delays when under pressure",
    )


class BackpressureState(BaseModel):
    """Current state of backpressure system."""

    model_config = {"extra": "forbid"}

    is_active: bool = Field(default=False, description="Whether backpressure is active")
    queue_size: int = Field(default=0, ge=0, description="Current queue size")
    queue_utilization: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Queue utilization ratio"
    )
    throttle_factor: float = Field(
        default=1.0, ge=1.0, description="Current throttle factor"
    )
    total_throttled: int = Field(
        default=0, ge=0, description="Total number of throttled operations"
    )
    total_processed: int = Field(
        default=0, ge=0, description="Total operations processed"
    )


class BackpressureController:
    """Controller for managing backpressure in execution pipelines.

    Implements adaptive backpressure using watermarks, rate limiting,
    and dynamic throttling to prevent resource exhaustion.
    """

    def __init__(self, config: Optional[BackpressureConfig] = None):
        """Initialize backpressure controller.

        Args:
            config: Backpressure configuration.
        """
        self.config = config or BackpressureConfig()
        self._state = BackpressureState()
        self._queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self._rate_limiter = RateLimiter(self.config.max_rate_per_second)
        self._lock = asyncio.Lock()
        self._last_check_time = time.perf_counter()

    async def enqueue(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Enqueue an item with backpressure handling.

        Args:
            item: Item to enqueue.
            timeout: Optional timeout in seconds.

        Returns:
            True if enqueued successfully, False if timed out.
        """
        async with self._lock:
            # Update state
            self._update_state()

            # Check if backpressure should activate
            if not self._state.is_active and self._should_activate():
                await self._activate_backpressure()

            # Apply rate limiting
            await self._rate_limiter.acquire()

        try:
            # Try to put item in queue
            if timeout:
                await asyncio.wait_for(
                    self._queue.put(item),
                    timeout=timeout,
                )
            else:
                await self._queue.put(item)

            async with self._lock:
                self._state.total_processed += 1

            return True

        except asyncio.TimeoutError:
            logger.warning(
                "backpressure_enqueue_timeout",
                queue_size=self._state.queue_size,
                utilization=self._state.queue_utilization,
            )
            return False

    async def dequeue(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Dequeue an item, releasing backpressure if needed.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            Dequeued item or None if timed out.
        """
        try:
            if timeout:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
            else:
                item = await self._queue.get()

            async with self._lock:
                # Update state
                self._update_state()

                # Check if backpressure should deactivate
                if self._state.is_active and self._should_deactivate():
                    await self._deactivate_backpressure()

            return item

        except asyncio.TimeoutError:
            return None

    def _update_state(self) -> None:
        """Update internal state metrics."""
        self._state.queue_size = self._queue.qsize()
        self._state.queue_utilization = (
            self._state.queue_size / self.config.max_queue_size
        )

    def _should_activate(self) -> bool:
        """Check if backpressure should be activated.

        Returns:
            True if backpressure should activate.
        """
        return self._state.queue_utilization >= self.config.high_watermark

    def _should_deactivate(self) -> bool:
        """Check if backpressure should be deactivated.

        Returns:
            True if backpressure should deactivate.
        """
        return self._state.queue_utilization <= self.config.low_watermark

    async def _activate_backpressure(self) -> None:
        """Activate backpressure."""
        self._state.is_active = True

        if self.config.enable_adaptive_throttling:
            # Increase throttle factor
            self._state.throttle_factor = self.config.throttle_multiplier

        logger.warning(
            "backpressure_activated",
            queue_size=self._state.queue_size,
            utilization=self._state.queue_utilization,
            throttle_factor=self._state.throttle_factor,
        )

    async def _deactivate_backpressure(self) -> None:
        """Deactivate backpressure."""
        self._state.is_active = False
        self._state.throttle_factor = 1.0

        logger.info(
            "backpressure_deactivated",
            queue_size=self._state.queue_size,
            utilization=self._state.queue_utilization,
        )

    async def throttle(self) -> None:
        """Apply throttling delay if backpressure is active."""
        if self._state.is_active:
            # Calculate throttle delay
            base_delay = 1.0 / self.config.max_rate_per_second
            delay = base_delay * self._state.throttle_factor

            await asyncio.sleep(delay)

            async with self._lock:
                self._state.total_throttled += 1

            logger.debug(
                "backpressure_throttled",
                delay_s=delay,
                throttle_factor=self._state.throttle_factor,
            )

    def get_state(self) -> BackpressureState:
        """Get current backpressure state.

        Returns:
            Current backpressure state.
        """
        self._update_state()
        return self._state.model_copy()

    def is_under_pressure(self) -> bool:
        """Check if system is currently under backpressure.

        Returns:
            True if backpressure is active.
        """
        return self._state.is_active

    def get_metrics(self) -> dict[str, Any]:
        """Get backpressure metrics.

        Returns:
            Dictionary of metrics.
        """
        self._update_state()
        return {
            "is_active": self._state.is_active,
            "queue_size": self._state.queue_size,
            "queue_utilization": self._state.queue_utilization,
            "throttle_factor": self._state.throttle_factor,
            "total_throttled": self._state.total_throttled,
            "total_processed": self._state.total_processed,
            "throttle_rate": (
                self._state.total_throttled / self._state.total_processed
                if self._state.total_processed > 0
                else 0.0
            ),
        }

    async def reset(self) -> None:
        """Reset backpressure controller state."""
        async with self._lock:
            # Clear queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Reset state
            self._state = BackpressureState()
            self._rate_limiter.reset()

            logger.info("backpressure_controller_reset")


class RateLimiter:
    """Token bucket rate limiter for controlling execution rate."""

    def __init__(self, max_rate_per_second: int):
        """Initialize rate limiter.

        Args:
            max_rate_per_second: Maximum rate of operations per second.
        """
        self.max_rate = max_rate_per_second
        self.tokens = float(max_rate_per_second)
        self.last_update = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to proceed (blocks if rate limit exceeded)."""
        async with self._lock:
            now = time.perf_counter()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.max_rate,
                self.tokens + (elapsed * self.max_rate),
            )
            self.last_update = now

            # If no tokens available, wait
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.max_rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

    def reset(self) -> None:
        """Reset rate limiter state."""
        self.tokens = float(self.max_rate)
        self.last_update = time.perf_counter()


class AdaptiveBackpressure:
    """Adaptive backpressure that adjusts based on system metrics.

    Dynamically adjusts watermarks and throttling based on observed
    performance metrics like latency and error rates.
    """

    def __init__(
        self,
        config: Optional[BackpressureConfig] = None,
        latency_threshold_ms: float = 1000.0,
    ):
        """Initialize adaptive backpressure.

        Args:
            config: Base backpressure configuration.
            latency_threshold_ms: Latency threshold for adaptation.
        """
        self.controller = BackpressureController(config)
        self.latency_threshold = latency_threshold_ms
        self._latency_samples: list[float] = []
        self._max_samples = 100
        self._adaptation_interval = 10  # Adapt every N operations

    async def enqueue_with_monitoring(
        self,
        item: Any,
        latency_ms: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Enqueue with latency monitoring and adaptation.

        Args:
            item: Item to enqueue.
            latency_ms: Optional observed latency.
            timeout: Optional timeout.

        Returns:
            True if enqueued successfully.
        """
        # Record latency sample
        if latency_ms is not None:
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > self._max_samples:
                self._latency_samples.pop(0)

            # Adapt if we have enough samples
            if len(self._latency_samples) >= self._adaptation_interval:
                await self._adapt()

        return await self.controller.enqueue(item, timeout)

    async def _adapt(self) -> None:
        """Adapt backpressure parameters based on metrics."""
        if not self._latency_samples:
            return

        avg_latency = sum(self._latency_samples) / len(self._latency_samples)

        # If latency is high, make backpressure more aggressive
        if avg_latency > self.latency_threshold:
            # Lower high watermark to trigger backpressure sooner
            new_high = max(0.6, self.controller.config.high_watermark - 0.1)
            self.controller.config.high_watermark = new_high

            logger.info(
                "backpressure_adapted_aggressive",
                avg_latency_ms=avg_latency,
                new_high_watermark=new_high,
            )
        # If latency is low, relax backpressure
        elif avg_latency < self.latency_threshold * 0.5:
            # Raise high watermark to delay backpressure
            new_high = min(0.9, self.controller.config.high_watermark + 0.05)
            self.controller.config.high_watermark = new_high

            logger.info(
                "backpressure_adapted_relaxed",
                avg_latency_ms=avg_latency,
                new_high_watermark=new_high,
            )


async def with_backpressure(
    controller: BackpressureController,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a function with backpressure control.

    Args:
        controller: Backpressure controller.
        func: Function to execute.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func execution.
    """
    # Apply throttling before execution
    await controller.throttle()

    # Execute function
    if asyncio.iscoroutinefunction(func):
        result = await func(*args, **kwargs)
    else:
        result = func(*args, **kwargs)

    return result
