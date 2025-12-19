"""Tests for tool rate limiting."""

import asyncio
import time

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.rate_limiting import (
    FixedWindowLimiter,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitedToolWrapper,
    RateLimitExceeded,
    RateLimitState,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    ToolRateLimiter,
    create_rate_limiter,
)


class RateLimitInput(BaseModel):
    """Input for rate limit test tool."""

    data: str = ""


class RateLimitOutput(BaseModel):
    """Output for rate limit test tool."""

    success: bool = True
    error: str | None = None


class RateLimitTool(BaseTool[RateLimitInput, RateLimitOutput]):
    """Tool for rate limiting tests."""

    metadata = ToolMetadata(
        id="rate_limit_tool",
        name="Rate Limit Tool",
        description="A tool for rate limit tests",
        category="utility",
    )
    input_type = RateLimitInput
    output_type = RateLimitOutput

    async def execute(self, input: RateLimitInput) -> RateLimitOutput:
        return RateLimitOutput()


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_exception_attributes(self):
        """Test exception has correct attributes."""
        exc = RateLimitExceeded(
            "Rate limit exceeded",
            retry_after=5.0,
            limit=100,
            remaining=0,
        )

        assert str(exc) == "Rate limit exceeded"
        assert exc.retry_after == 5.0
        assert exc.limit == 100
        assert exc.remaining == 0


class TestTokenBucketLimiter:
    """Tests for TokenBucketLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful token acquisition."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=10.0)

        assert await limiter.acquire()
        assert await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_exhausted(self):
        """Test exhausting tokens."""
        limiter = TokenBucketLimiter(capacity=3, refill_rate=0.1)

        assert await limiter.acquire()
        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self):
        """Test acquiring multiple tokens at once."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=10.0)

        assert await limiter.acquire(tokens=5)
        assert await limiter.acquire(tokens=5)
        assert not await limiter.acquire(tokens=5)

    @pytest.mark.asyncio
    async def test_refill(self):
        """Test token refill over time."""
        limiter = TokenBucketLimiter(capacity=5, refill_rate=100.0)

        # Exhaust tokens
        for _ in range(5):
            await limiter.acquire()

        assert not await limiter.acquire()

        # Wait for refill
        await asyncio.sleep(0.05)  # 5 tokens at 100/sec
        assert await limiter.acquire()

    def test_get_state(self):
        """Test getting limiter state."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=10.0)
        state = limiter.get_state()

        assert state.limit == 10
        assert state.remaining == 10

    def test_reset(self):
        """Test resetting limiter."""
        limiter = TokenBucketLimiter(capacity=10, refill_rate=10.0)

        asyncio.get_event_loop().run_until_complete(limiter.acquire("key1"))
        asyncio.get_event_loop().run_until_complete(limiter.acquire("key2"))

        limiter.reset("key1")
        state = limiter.get_state("key1")
        assert state.remaining == 10

        limiter.reset()
        state = limiter.get_state("key2")
        assert state.remaining == 10


class TestSlidingWindowLimiter:
    """Tests for SlidingWindowLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful acquisition."""
        limiter = SlidingWindowLimiter(window_size=60.0, max_requests=10)

        assert await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_exhausted(self):
        """Test exhausting requests."""
        limiter = SlidingWindowLimiter(window_size=60.0, max_requests=3)

        assert await limiter.acquire()
        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()

    def test_get_state(self):
        """Test getting limiter state."""
        limiter = SlidingWindowLimiter(window_size=60.0, max_requests=10)
        state = limiter.get_state()

        assert state.limit == 10
        assert state.remaining == 10

    def test_reset(self):
        """Test resetting limiter."""
        limiter = SlidingWindowLimiter(window_size=60.0, max_requests=10)

        asyncio.get_event_loop().run_until_complete(limiter.acquire())
        limiter.reset()

        state = limiter.get_state()
        assert state.remaining == 10


class TestFixedWindowLimiter:
    """Tests for FixedWindowLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful acquisition."""
        limiter = FixedWindowLimiter(window_size=60.0, max_requests=10)

        assert await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_exhausted(self):
        """Test exhausting requests."""
        limiter = FixedWindowLimiter(window_size=60.0, max_requests=3)

        assert await limiter.acquire()
        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()

    @pytest.mark.asyncio
    async def test_window_reset(self):
        """Test window resetting after time."""
        limiter = FixedWindowLimiter(window_size=0.1, max_requests=2)

        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()

        # Wait for window to reset
        await asyncio.sleep(0.15)
        assert await limiter.acquire()

    def test_get_state(self):
        """Test getting limiter state."""
        limiter = FixedWindowLimiter(window_size=60.0, max_requests=10)
        state = limiter.get_state()

        assert state.limit == 10
        assert state.remaining == 10


class TestToolRateLimiter:
    """Tests for ToolRateLimiter."""

    @pytest.mark.asyncio
    async def test_check_allowed(self):
        """Test allowed check."""
        limiter = ToolRateLimiter()

        assert await limiter.check("test_tool", raise_on_limit=False)

    @pytest.mark.asyncio
    async def test_check_exceeded_raises(self):
        """Test that exceeded check raises exception."""
        config = RateLimitConfig(
            requests_per_second=100.0,
            burst_size=2,
        )
        limiter = ToolRateLimiter(default_config=config)

        await limiter.check("test_tool")
        await limiter.check("test_tool")

        with pytest.raises(RateLimitExceeded):
            await limiter.check("test_tool")

    @pytest.mark.asyncio
    async def test_check_exceeded_no_raise(self):
        """Test exceeded check without raising."""
        config = RateLimitConfig(
            requests_per_second=100.0,
            burst_size=2,
        )
        limiter = ToolRateLimiter(default_config=config)

        await limiter.check("test_tool", raise_on_limit=False)
        await limiter.check("test_tool", raise_on_limit=False)

        result = await limiter.check("test_tool", raise_on_limit=False)
        assert not result

    def test_get_state(self):
        """Test getting state."""
        limiter = ToolRateLimiter()
        state = limiter.get_state("test_tool")

        assert isinstance(state, RateLimitState)

    def test_reset(self):
        """Test resetting."""
        limiter = ToolRateLimiter()

        asyncio.get_event_loop().run_until_complete(limiter.check("test_tool"))
        limiter.reset("test_tool")

        state = limiter.get_state("test_tool")
        assert state.remaining == 20  # Default burst size


class TestRateLimitedToolWrapper:
    """Tests for RateLimitedToolWrapper."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        tool = RateLimitTool()
        wrapper = RateLimitedToolWrapper(tool)

        result = await wrapper.execute(RateLimitInput())
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_rate_limited(self):
        """Test rate limited execution."""
        config = RateLimitConfig(burst_size=1)
        limiter = ToolRateLimiter(default_config=config)
        tool = RateLimitTool()
        wrapper = RateLimitedToolWrapper(tool, limiter=limiter)

        await wrapper.execute(RateLimitInput())

        with pytest.raises(RateLimitExceeded):
            await wrapper.execute(RateLimitInput())

    @pytest.mark.asyncio
    async def test_execute_wait_on_limit(self):
        """Test waiting when rate limited."""
        config = RateLimitConfig(
            requests_per_second=100.0,  # Fast refill
            burst_size=1,
        )
        limiter = ToolRateLimiter(default_config=config)
        tool = RateLimitTool()
        wrapper = RateLimitedToolWrapper(
            tool, limiter=limiter, wait_on_limit=True, max_wait=1.0
        )

        await wrapper.execute(RateLimitInput())

        # This should wait and succeed
        start = time.monotonic()
        result = await wrapper.execute(RateLimitInput())
        elapsed = time.monotonic() - start

        assert result.success
        assert elapsed > 0  # Had to wait


class TestCreateRateLimiter:
    """Tests for create_rate_limiter."""

    def test_create_token_bucket(self):
        """Test creating token bucket limiter."""
        limiter = create_rate_limiter(
            RateLimitAlgorithm.TOKEN_BUCKET,
            capacity=50,
            refill_rate=20.0,
        )

        assert isinstance(limiter, TokenBucketLimiter)

    def test_create_sliding_window(self):
        """Test creating sliding window limiter."""
        limiter = create_rate_limiter(
            RateLimitAlgorithm.SLIDING_WINDOW,
            window_size=120.0,
            max_requests=200,
        )

        assert isinstance(limiter, SlidingWindowLimiter)

    def test_create_fixed_window(self):
        """Test creating fixed window limiter."""
        limiter = create_rate_limiter(
            RateLimitAlgorithm.FIXED_WINDOW,
            window_size=120.0,
            max_requests=200,
        )

        assert isinstance(limiter, FixedWindowLimiter)
