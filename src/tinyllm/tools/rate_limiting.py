"""Tool rate limiting for TinyLLM.

This module provides rate limiting capabilities for tools
including token bucket, sliding window, and fixed window algorithms.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        remaining: int = 0,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    burst_size: int = 20
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET


@dataclass
class RateLimitState:
    """Current state of rate limiter."""

    remaining: int
    limit: int
    reset_at: float
    retry_after: Optional[float] = None


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Try to acquire tokens for a request.

        Args:
            key: Rate limit key (e.g., tool_id, user_id).
            tokens: Number of tokens to acquire.

        Returns:
            True if acquired, False if rate limited.
        """
        pass

    @abstractmethod
    def get_state(self, key: str = "default") -> RateLimitState:
        """Get current rate limit state.

        Args:
            key: Rate limit key.

        Returns:
            Current rate limit state.
        """
        pass

    @abstractmethod
    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit state.

        Args:
            key: Specific key to reset, or None for all.
        """
        pass


@dataclass
class TokenBucketState:
    """State for token bucket algorithm."""

    tokens: float
    last_update: float
    capacity: int
    refill_rate: float


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum capacity.
    Each request consumes tokens.
    """

    def __init__(
        self,
        capacity: int = 20,
        refill_rate: float = 10.0,  # tokens per second
    ):
        """Initialize token bucket.

        Args:
            capacity: Maximum tokens (burst size).
            refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: Dict[str, TokenBucketState] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, key: str) -> TokenBucketState:
        """Get or create bucket for key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucketState(
                tokens=float(self.capacity),
                last_update=time.monotonic(),
                capacity=self.capacity,
                refill_rate=self.refill_rate,
            )
        return self._buckets[key]

    def _refill(self, bucket: TokenBucketState) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - bucket.last_update
        bucket.tokens = min(
            bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate
        )
        bucket.last_update = now

    async def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        async with self._lock:
            bucket = self._get_bucket(key)
            self._refill(bucket)

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True
            return False

    def get_state(self, key: str = "default") -> RateLimitState:
        """Get current state."""
        bucket = self._get_bucket(key)
        self._refill(bucket)

        remaining = int(bucket.tokens)
        retry_after = None

        if remaining <= 0:
            # Calculate time until we have at least 1 token
            retry_after = (1 - bucket.tokens) / bucket.refill_rate

        return RateLimitState(
            remaining=remaining,
            limit=bucket.capacity,
            reset_at=time.time() + (bucket.capacity - bucket.tokens) / bucket.refill_rate,
            retry_after=retry_after,
        )

    def reset(self, key: Optional[str] = None) -> None:
        """Reset buckets."""
        if key:
            if key in self._buckets:
                del self._buckets[key]
        else:
            self._buckets.clear()


@dataclass
class SlidingWindowState:
    """State for sliding window algorithm."""

    requests: list = field(default_factory=list)
    window_size: float = 60.0  # seconds
    max_requests: int = 100


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter.

    Tracks requests within a moving time window.
    """

    def __init__(
        self,
        window_size: float = 60.0,  # seconds
        max_requests: int = 100,
    ):
        """Initialize sliding window.

        Args:
            window_size: Window duration in seconds.
            max_requests: Maximum requests in window.
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self._windows: Dict[str, SlidingWindowState] = {}
        self._lock = asyncio.Lock()

    def _get_window(self, key: str) -> SlidingWindowState:
        """Get or create window for key."""
        if key not in self._windows:
            self._windows[key] = SlidingWindowState(
                window_size=self.window_size,
                max_requests=self.max_requests,
            )
        return self._windows[key]

    def _cleanup(self, window: SlidingWindowState) -> None:
        """Remove expired requests from window."""
        now = time.monotonic()
        cutoff = now - window.window_size
        window.requests = [t for t in window.requests if t > cutoff]

    async def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Try to acquire request slot."""
        async with self._lock:
            window = self._get_window(key)
            self._cleanup(window)

            if len(window.requests) + tokens <= window.max_requests:
                now = time.monotonic()
                for _ in range(tokens):
                    window.requests.append(now)
                return True
            return False

    def get_state(self, key: str = "default") -> RateLimitState:
        """Get current state."""
        window = self._get_window(key)
        self._cleanup(window)

        remaining = window.max_requests - len(window.requests)
        retry_after = None

        if remaining <= 0 and window.requests:
            # Time until oldest request expires
            oldest = min(window.requests)
            retry_after = window.window_size - (time.monotonic() - oldest)

        reset_at = time.time() + window.window_size

        return RateLimitState(
            remaining=max(0, remaining),
            limit=window.max_requests,
            reset_at=reset_at,
            retry_after=retry_after if retry_after and retry_after > 0 else None,
        )

    def reset(self, key: Optional[str] = None) -> None:
        """Reset windows."""
        if key:
            if key in self._windows:
                del self._windows[key]
        else:
            self._windows.clear()


@dataclass
class FixedWindowState:
    """State for fixed window algorithm."""

    count: int = 0
    window_start: float = 0.0
    window_size: float = 60.0
    max_requests: int = 100


class FixedWindowLimiter(RateLimiter):
    """Fixed window rate limiter.

    Counts requests within fixed time intervals.
    """

    def __init__(
        self,
        window_size: float = 60.0,  # seconds
        max_requests: int = 100,
    ):
        """Initialize fixed window.

        Args:
            window_size: Window duration in seconds.
            max_requests: Maximum requests in window.
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self._windows: Dict[str, FixedWindowState] = {}
        self._lock = asyncio.Lock()

    def _get_window(self, key: str) -> FixedWindowState:
        """Get or create window for key."""
        now = time.monotonic()
        if key not in self._windows:
            self._windows[key] = FixedWindowState(
                window_start=now,
                window_size=self.window_size,
                max_requests=self.max_requests,
            )
        return self._windows[key]

    def _maybe_reset(self, window: FixedWindowState) -> None:
        """Reset window if expired."""
        now = time.monotonic()
        if now - window.window_start >= window.window_size:
            window.count = 0
            window.window_start = now

    async def acquire(self, key: str = "default", tokens: int = 1) -> bool:
        """Try to acquire request slot."""
        async with self._lock:
            window = self._get_window(key)
            self._maybe_reset(window)

            if window.count + tokens <= window.max_requests:
                window.count += tokens
                return True
            return False

    def get_state(self, key: str = "default") -> RateLimitState:
        """Get current state."""
        window = self._get_window(key)
        self._maybe_reset(window)

        remaining = window.max_requests - window.count
        window_end = window.window_start + window.window_size
        reset_at = time.time() + (window_end - time.monotonic())

        retry_after = None
        if remaining <= 0:
            retry_after = window_end - time.monotonic()

        return RateLimitState(
            remaining=max(0, remaining),
            limit=window.max_requests,
            reset_at=reset_at,
            retry_after=retry_after if retry_after and retry_after > 0 else None,
        )

    def reset(self, key: Optional[str] = None) -> None:
        """Reset windows."""
        if key:
            if key in self._windows:
                del self._windows[key]
        else:
            self._windows.clear()


class ToolRateLimiter:
    """Rate limiter specifically for tools."""

    def __init__(
        self,
        limiter: Optional[RateLimiter] = None,
        default_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize tool rate limiter.

        Args:
            limiter: Underlying rate limiter implementation.
            default_config: Default rate limit configuration.
        """
        self.config = default_config or RateLimitConfig()
        self._limiter = limiter or self._create_limiter()
        self._tool_configs: Dict[str, RateLimitConfig] = {}

    def _create_limiter(self) -> RateLimiter:
        """Create limiter based on config algorithm."""
        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketLimiter(
                capacity=self.config.burst_size,
                refill_rate=self.config.requests_per_second,
            )
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowLimiter(
                window_size=60.0,
                max_requests=int(self.config.requests_per_minute),
            )
        else:
            return FixedWindowLimiter(
                window_size=60.0,
                max_requests=int(self.config.requests_per_minute),
            )

    def set_tool_config(self, tool_id: str, config: RateLimitConfig) -> None:
        """Set rate limit config for a specific tool.

        Args:
            tool_id: Tool identifier.
            config: Rate limit configuration.
        """
        self._tool_configs[tool_id] = config

    async def check(self, tool_id: str, raise_on_limit: bool = True) -> bool:
        """Check if a tool call is allowed.

        Args:
            tool_id: Tool identifier.
            raise_on_limit: Whether to raise exception if limited.

        Returns:
            True if allowed, False if rate limited.

        Raises:
            RateLimitExceeded: If rate limited and raise_on_limit is True.
        """
        allowed = await self._limiter.acquire(tool_id)

        if not allowed and raise_on_limit:
            state = self._limiter.get_state(tool_id)
            raise RateLimitExceeded(
                f"Rate limit exceeded for tool {tool_id}",
                retry_after=state.retry_after,
                limit=state.limit,
                remaining=state.remaining,
            )

        return allowed

    def get_state(self, tool_id: str) -> RateLimitState:
        """Get rate limit state for a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            Current rate limit state.
        """
        return self._limiter.get_state(tool_id)

    def reset(self, tool_id: Optional[str] = None) -> None:
        """Reset rate limits.

        Args:
            tool_id: Specific tool to reset, or None for all.
        """
        self._limiter.reset(tool_id)


class RateLimitedToolWrapper:
    """Wrapper that adds rate limiting to tool execution."""

    def __init__(
        self,
        tool: Any,
        limiter: Optional[ToolRateLimiter] = None,
        wait_on_limit: bool = False,
        max_wait: float = 30.0,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            limiter: Rate limiter instance.
            wait_on_limit: Whether to wait and retry when limited.
            max_wait: Maximum time to wait in seconds.
        """
        self.tool = tool
        self.limiter = limiter or ToolRateLimiter()
        self.wait_on_limit = wait_on_limit
        self.max_wait = max_wait

    async def execute(self, input: Any) -> Any:
        """Execute tool with rate limiting.

        Args:
            input: Tool input.

        Returns:
            Tool output.

        Raises:
            RateLimitExceeded: If rate limited and not waiting.
        """
        tool_id = self.tool.metadata.id
        start = time.monotonic()

        while True:
            try:
                await self.limiter.check(tool_id)
                return await self.tool.execute(input)
            except RateLimitExceeded as e:
                if not self.wait_on_limit:
                    raise

                elapsed = time.monotonic() - start
                if elapsed >= self.max_wait:
                    raise

                wait_time = min(
                    e.retry_after or 1.0,
                    self.max_wait - elapsed,
                )
                if wait_time > 0:
                    await asyncio.sleep(wait_time)


def create_rate_limiter(
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    **kwargs,
) -> RateLimiter:
    """Create a rate limiter with the specified algorithm.

    Args:
        algorithm: Algorithm to use.
        **kwargs: Algorithm-specific parameters.

    Returns:
        Rate limiter instance.
    """
    if algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
        return TokenBucketLimiter(
            capacity=kwargs.get("capacity", 20),
            refill_rate=kwargs.get("refill_rate", 10.0),
        )
    elif algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
        return SlidingWindowLimiter(
            window_size=kwargs.get("window_size", 60.0),
            max_requests=kwargs.get("max_requests", 100),
        )
    else:
        return FixedWindowLimiter(
            window_size=kwargs.get("window_size", 60.0),
            max_requests=kwargs.get("max_requests", 100),
        )
