"""Tests for tool caching."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.caching import (
    CacheDecorator,
    CacheEntry,
    CacheStats,
    CachedToolWrapper,
    MemoryCache,
    ToolCache,
    cached,
    with_cache,
)


class CacheInput(BaseModel):
    """Input for cache tests."""

    query: str = ""


class CacheOutput(BaseModel):
    """Output for cache tests."""

    result: str = ""
    success: bool = True
    error: str | None = None


class CountingTool(BaseTool[CacheInput, CacheOutput]):
    """Tool that counts executions."""

    metadata = ToolMetadata(
        id="counting_tool",
        name="Counting Tool",
        description="Counts executions",
        category="utility",
    )
    input_type = CacheInput
    output_type = CacheOutput

    def __init__(self):
        super().__init__()
        self.execution_count = 0

    async def execute(self, input: CacheInput) -> CacheOutput:
        self.execution_count += 1
        return CacheOutput(result=f"Query: {input.query}, Count: {self.execution_count}")


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_expiration(self):
        """Test entry expiration."""
        import time

        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time() - 10,
            ttl=5,
        )

        assert entry.is_expired

    def test_not_expired(self):
        """Test entry not expired."""
        import time

        entry = CacheEntry(
            key="test",
            value="data",
            created_at=time.time(),
            ttl=300,
        )

        assert not entry.is_expired


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)

        assert stats.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        """Test hit rate with no requests."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0


class TestMemoryCache:
    """Tests for MemoryCache."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test setting and getting values."""
        cache = MemoryCache()

        await cache.set("key1", "value1", ttl=300)
        entry = await cache.get("key1")

        assert entry is not None
        assert entry.value == "value1"

    @pytest.mark.asyncio
    async def test_get_missing(self):
        """Test getting missing key."""
        cache = MemoryCache()

        entry = await cache.get("nonexistent")

        assert entry is None

    @pytest.mark.asyncio
    async def test_get_expired(self):
        """Test getting expired entry."""
        cache = MemoryCache()

        await cache.set("key1", "value1", ttl=0.01)
        import asyncio
        await asyncio.sleep(0.02)

        entry = await cache.get("key1")

        assert entry is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting entry."""
        cache = MemoryCache()

        await cache.set("key1", "value1", ttl=300)
        deleted = await cache.delete("key1")

        assert deleted
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing cache."""
        cache = MemoryCache()

        await cache.set("key1", "value1", ttl=300)
        await cache.set("key2", "value2", ttl=300)
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_eviction_at_capacity(self):
        """Test eviction when at capacity."""
        cache = MemoryCache(max_size=2)

        await cache.set("key1", "value1", ttl=300)
        await cache.set("key2", "value2", ttl=300)
        await cache.set("key3", "value3", ttl=300)

        stats = cache.stats()
        assert stats.size == 2
        assert stats.evictions == 1

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test cache statistics."""
        cache = MemoryCache()

        await cache.set("key1", "value1", ttl=300)
        await cache.get("key1")  # Hit
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1


class TestToolCache:
    """Tests for ToolCache."""

    @pytest.mark.asyncio
    async def test_get_and_set(self):
        """Test caching tool results."""
        cache = ToolCache()

        await cache.set("tool1", CacheInput(query="test"), "result1")
        result = await cache.get("tool1", CacheInput(query="test"))

        assert result == "result1"

    @pytest.mark.asyncio
    async def test_different_inputs_different_keys(self):
        """Test different inputs generate different keys."""
        cache = ToolCache()

        await cache.set("tool1", CacheInput(query="query1"), "result1")
        await cache.set("tool1", CacheInput(query="query2"), "result2")

        assert await cache.get("tool1", CacheInput(query="query1")) == "result1"
        assert await cache.get("tool1", CacheInput(query="query2")) == "result2"

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test invalidating cache entry."""
        cache = ToolCache()

        await cache.set("tool1", CacheInput(query="test"), "result1")
        invalidated = await cache.invalidate("tool1", CacheInput(query="test"))

        assert invalidated
        assert await cache.get("tool1", CacheInput(query="test")) is None


class TestCachedToolWrapper:
    """Tests for CachedToolWrapper."""

    @pytest.mark.asyncio
    async def test_caches_result(self):
        """Test result is cached."""
        tool = CountingTool()
        cached_tool = CachedToolWrapper(tool)

        # First call
        result1 = await cached_tool.execute(CacheInput(query="test"))
        # Second call (should be cached)
        result2 = await cached_tool.execute(CacheInput(query="test"))

        assert tool.execution_count == 1  # Only executed once
        assert result1.result == result2.result

    @pytest.mark.asyncio
    async def test_different_inputs_not_cached(self):
        """Test different inputs are not cached together."""
        tool = CountingTool()
        cached_tool = CachedToolWrapper(tool)

        await cached_tool.execute(CacheInput(query="query1"))
        await cached_tool.execute(CacheInput(query="query2"))

        assert tool.execution_count == 2

    @pytest.mark.asyncio
    async def test_cache_condition(self):
        """Test cache condition."""
        tool = CountingTool()
        cached_tool = CachedToolWrapper(
            tool,
            cache_condition=lambda x: x.success,  # Only cache successes
        )

        await cached_tool.execute(CacheInput(query="test"))
        await cached_tool.execute(CacheInput(query="test"))

        assert tool.execution_count == 1  # Should be cached

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """Test invalidating cached result."""
        tool = CountingTool()
        cached_tool = CachedToolWrapper(tool)

        await cached_tool.execute(CacheInput(query="test"))
        await cached_tool.invalidate(CacheInput(query="test"))
        await cached_tool.execute(CacheInput(query="test"))

        assert tool.execution_count == 2


class TestCacheDecorator:
    """Tests for CacheDecorator."""

    @pytest.mark.asyncio
    async def test_decorator(self):
        """Test cache decorator."""
        tool = CountingTool()
        decorator = CacheDecorator(ttl=300)
        cached_tool = decorator(tool)

        await cached_tool.execute(CacheInput(query="test"))
        await cached_tool.execute(CacheInput(query="test"))

        assert tool.execution_count == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_cache(self):
        """Test with_cache function."""
        tool = CountingTool()
        cached_tool = with_cache(tool, ttl=300)

        await cached_tool.execute(CacheInput(query="test"))
        await cached_tool.execute(CacheInput(query="test"))

        assert tool.execution_count == 1

    @pytest.mark.asyncio
    async def test_cached_decorator_factory(self):
        """Test cached decorator factory."""
        tool = CountingTool()
        decorator = cached(ttl=300)
        cached_tool = decorator(tool)

        await cached_tool.execute(CacheInput(query="test"))
        await cached_tool.execute(CacheInput(query="test"))

        assert tool.execution_count == 1
