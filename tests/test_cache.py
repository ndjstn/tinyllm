"""Tests for response caching."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinyllm.cache import (
    CacheEntry,
    CacheMetrics,
    CachedOllamaClient,
    InMemoryBackend,
    ResponseCache,
    create_cached_client,
    create_memory_cache,
)
from tinyllm.models.client import GenerateResponse, OllamaClient


@pytest.fixture
def sample_response():
    """Create a sample GenerateResponse."""
    return GenerateResponse(
        model="test-model",
        created_at="2024-01-01T00:00:00Z",
        response="This is a test response",
        done=True,
        total_duration=1000000,
        eval_count=10,
    )


@pytest.fixture
def cache_entry(sample_response):
    """Create a sample CacheEntry."""
    return CacheEntry(
        response=sample_response,
        created_at=time.monotonic(),
        ttl=60,
    )


class TestCacheMetrics:
    """Tests for CacheMetrics."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.evictions == 0
        assert metrics.errors == 0
        assert metrics.hit_rate == 0.0
        assert metrics.total_requests == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics(hits=7, misses=3)
        assert metrics.hit_rate == 0.7
        assert metrics.total_requests == 10

    def test_to_dict(self):
        """Test metrics to dict conversion."""
        metrics = CacheMetrics(hits=10, misses=5, sets=15, evictions=2)
        data = metrics.to_dict()
        assert data["hits"] == 10
        assert data["misses"] == 5
        assert data["sets"] == 15
        assert data["evictions"] == 2
        assert data["hit_rate"] == pytest.approx(0.667, rel=0.01)
        assert data["total_requests"] == 15


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_is_expired_no_ttl(self, cache_entry):
        """Test entry without TTL never expires."""
        cache_entry.ttl = None
        assert not cache_entry.is_expired()

    def test_is_expired_with_ttl(self, cache_entry):
        """Test entry with TTL expiration."""
        # Set very short TTL
        cache_entry.ttl = 0.01
        cache_entry.created_at = time.monotonic() - 1
        assert cache_entry.is_expired()

        # Set long TTL
        cache_entry.ttl = 3600
        cache_entry.created_at = time.monotonic()
        assert not cache_entry.is_expired()

    def test_touch(self, cache_entry):
        """Test access tracking."""
        initial_count = cache_entry.access_count
        initial_time = cache_entry.last_accessed

        time.sleep(0.01)
        cache_entry.touch()

        assert cache_entry.access_count == initial_count + 1
        assert cache_entry.last_accessed > initial_time


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.mark.asyncio
    async def test_get_set(self, cache_entry):
        """Test basic get/set operations."""
        backend = InMemoryBackend(max_size=10)

        # Get non-existent key
        result = await backend.get("nonexistent")
        assert result is None

        # Set and get
        await backend.set("test-key", cache_entry)
        result = await backend.get("test-key")
        assert result is not None
        assert result.response == cache_entry.response

    @pytest.mark.asyncio
    async def test_lru_eviction(self, sample_response):
        """Test LRU eviction when cache is full."""
        backend = InMemoryBackend(max_size=3)

        # Fill cache
        for i in range(3):
            entry = CacheEntry(
                response=sample_response,
                created_at=time.monotonic(),
                ttl=None,
            )
            await backend.set(f"key-{i}", entry)

        # Verify all entries exist
        assert await backend.size() == 3

        # Add one more (should evict oldest)
        entry = CacheEntry(
            response=sample_response,
            created_at=time.monotonic(),
            ttl=None,
        )
        await backend.set("key-3", entry)

        # Verify oldest was evicted
        assert await backend.size() == 3
        assert await backend.get("key-0") is None
        assert await backend.get("key-1") is not None
        assert backend.eviction_count == 1

    @pytest.mark.asyncio
    async def test_expired_entry_cleanup(self, sample_response):
        """Test expired entries are removed on access."""
        backend = InMemoryBackend(max_size=10)

        # Add entry with very short TTL
        entry = CacheEntry(
            response=sample_response,
            created_at=time.monotonic() - 10,
            ttl=0.01,
        )
        await backend.set("expired-key", entry)

        # Try to get expired entry
        result = await backend.get("expired-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache_entry):
        """Test entry deletion."""
        backend = InMemoryBackend(max_size=10)

        await backend.set("test-key", cache_entry)
        assert await backend.get("test-key") is not None

        await backend.delete("test-key")
        assert await backend.get("test-key") is None

    @pytest.mark.asyncio
    async def test_clear(self, cache_entry):
        """Test clearing all entries."""
        backend = InMemoryBackend(max_size=10)

        # Add multiple entries
        for i in range(5):
            await backend.set(f"key-{i}", cache_entry)

        assert await backend.size() == 5

        await backend.clear()
        assert await backend.size() == 0


class TestResponseCache:
    """Tests for ResponseCache."""

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test deterministic cache key generation."""
        backend = InMemoryBackend()
        cache = ResponseCache(backend)

        key1 = cache.generate_cache_key(
            model="test",
            prompt="hello",
            temperature=0.3,
            max_tokens=100,
        )
        key2 = cache.generate_cache_key(
            model="test",
            prompt="hello",
            temperature=0.3,
            max_tokens=100,
        )

        # Same parameters should produce same key
        assert key1 == key2

        # Different parameters should produce different key
        key3 = cache.generate_cache_key(
            model="test",
            prompt="hello",
            temperature=0.5,  # Different temperature
            max_tokens=100,
        )
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_get_set_response(self, sample_response):
        """Test getting and setting responses."""
        backend = InMemoryBackend()
        cache = ResponseCache(backend, enable_metrics=True)

        key = cache.generate_cache_key(model="test", prompt="hello")

        # Cache miss
        result = await cache.get(key)
        assert result is None
        assert cache.metrics.misses == 1

        # Set response
        await cache.set(key, sample_response)
        assert cache.metrics.sets == 1

        # Cache hit
        result = await cache.get(key)
        assert result is not None
        assert result.response == sample_response.response
        assert cache.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, sample_response):
        """Test cache metrics are tracked correctly."""
        backend = InMemoryBackend()
        cache = ResponseCache(backend, enable_metrics=True)

        key = cache.generate_cache_key(model="test", prompt="hello")

        # Generate some cache activity
        await cache.get(key)  # miss
        await cache.set(key, sample_response)  # set
        await cache.get(key)  # hit
        await cache.get(key)  # hit

        metrics = cache.get_metrics()
        assert metrics.hits == 2
        assert metrics.misses == 1
        assert metrics.sets == 1
        assert metrics.hit_rate == pytest.approx(0.667, rel=0.01)

    @pytest.mark.asyncio
    async def test_reset_metrics(self, sample_response):
        """Test metrics reset."""
        backend = InMemoryBackend()
        cache = ResponseCache(backend, enable_metrics=True)

        key = cache.generate_cache_key(model="test", prompt="hello")
        await cache.set(key, sample_response)
        await cache.get(key)

        assert cache.metrics.hits > 0

        cache.reset_metrics()
        assert cache.metrics.hits == 0
        assert cache.metrics.misses == 0


class TestCachedOllamaClient:
    """Tests for CachedOllamaClient."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, sample_response):
        """Test response served from cache."""
        # Mock OllamaClient
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)

        # Create cached client
        cache = create_memory_cache()
        cached_client = CachedOllamaClient(
            client=mock_client,
            cache=cache,
            enable_cache=True,
        )

        # First call - cache miss
        response1 = await cached_client.generate(
            model="test",
            prompt="hello",
        )
        assert response1 == sample_response
        assert mock_client.generate.call_count == 1

        # Second call - cache hit (should not call client)
        response2 = await cached_client.generate(
            model="test",
            prompt="hello",
        )
        assert response2 == sample_response
        assert mock_client.generate.call_count == 1  # Not incremented

        # Verify metrics
        metrics = cache.get_metrics()
        assert metrics.hits == 1
        assert metrics.misses == 1

    @pytest.mark.asyncio
    async def test_force_refresh(self, sample_response):
        """Test force refresh bypasses cache."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)

        cache = create_memory_cache()
        cached_client = CachedOllamaClient(
            client=mock_client,
            cache=cache,
            enable_cache=True,
        )

        # First call
        await cached_client.generate(model="test", prompt="hello")
        assert mock_client.generate.call_count == 1

        # Force refresh - should call client again
        await cached_client.generate(
            model="test",
            prompt="hello",
            force_refresh=True,
        )
        assert mock_client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_disabled(self, sample_response):
        """Test behavior when cache is disabled."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)

        cache = create_memory_cache()
        cached_client = CachedOllamaClient(
            client=mock_client,
            cache=cache,
            enable_cache=False,
        )

        # Multiple calls should all hit client
        await cached_client.generate(model="test", prompt="hello")
        await cached_client.generate(model="test", prompt="hello")

        assert mock_client.generate.call_count == 2

        # Cache should be empty
        assert cache.get_metrics().sets == 0

    @pytest.mark.asyncio
    async def test_different_params_different_cache(self, sample_response):
        """Test different parameters use different cache entries."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)

        cache = create_memory_cache()
        cached_client = CachedOllamaClient(
            client=mock_client,
            cache=cache,
            enable_cache=True,
        )

        # Call with different prompts
        await cached_client.generate(model="test", prompt="hello")
        await cached_client.generate(model="test", prompt="world")

        # Both should be cache misses (different keys)
        assert mock_client.generate.call_count == 2
        assert cache.get_metrics().misses == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, sample_response):
        """Test getting combined client and cache stats."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)
        mock_client.get_stats = MagicMock(
            return_value={
                "request_count": 5,
                "total_tokens": 100,
            }
        )

        cache = create_memory_cache()
        cached_client = CachedOllamaClient(
            client=mock_client,
            cache=cache,
            enable_cache=True,
        )

        # Generate some activity
        await cached_client.generate(model="test", prompt="hello")

        # Get stats
        stats = cached_client.get_stats()
        assert "client" in stats
        assert "cache" in stats
        assert stats["cache_enabled"] is True
        assert stats["cache"]["misses"] == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_memory_cache(self):
        """Test creating memory cache."""
        cache = create_memory_cache(max_size=100, default_ttl=3600)
        assert isinstance(cache.backend, InMemoryBackend)
        assert cache.backend.max_size == 100
        assert cache.default_ttl == 3600

    @pytest.mark.asyncio
    async def test_create_cached_client_memory(self):
        """Test creating cached client with memory backend."""
        mock_client = AsyncMock(spec=OllamaClient)
        cached_client = await create_cached_client(
            client=mock_client,
            backend="memory",
            max_size=100,
        )
        assert isinstance(cached_client, CachedOllamaClient)
        assert isinstance(cached_client.cache.backend, InMemoryBackend)

    @pytest.mark.asyncio
    async def test_create_cached_client_invalid_backend(self):
        """Test error on invalid backend."""
        mock_client = AsyncMock(spec=OllamaClient)
        with pytest.raises(ValueError, match="Invalid cache backend"):
            await create_cached_client(
                client=mock_client,
                backend="invalid",
            )
