"""Tests for advanced caching features (Tasks 91-100)."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from tinyllm.cache import (
    CacheEntry,
    CacheMetrics,
    CacheTier,
    CompressionAlgorithm,
    InMemoryBackend,
    ResponseCache,
    create_memory_cache,
)
from tinyllm.cache_advanced import (
    AdaptiveCache,
    CacheCoherence,
    CacheInvalidator,
    CacheWarmer,
    CompressedBackend,
    CostModel,
    RedisClusterBackend,
    SemanticCache,
    TieredCache,
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


# Task 91-92: Semantic Similarity Caching


class TestSemanticCache:
    """Tests for semantic similarity caching."""

    @pytest.mark.asyncio
    async def test_embedding_computation(self):
        """Test embedding computation is deterministic."""
        cache = create_memory_cache()
        semantic = SemanticCache(cache, similarity_threshold=0.85)

        emb1 = semantic._compute_embedding("hello world")
        emb2 = semantic._compute_embedding("hello world")
        emb3 = semantic._compute_embedding("goodbye world")

        assert len(emb1) == 64
        assert emb1 == emb2
        assert emb1 != emb3

    @pytest.mark.asyncio
    async def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        cache = create_memory_cache()
        semantic = SemanticCache(cache)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        sim_identical = semantic._cosine_similarity(vec1, vec2)
        assert sim_identical == pytest.approx(1.0)

        sim_orthogonal = semantic._cosine_similarity(vec1, vec3)
        assert sim_orthogonal == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_find_similar(self, sample_response):
        """Test finding similar cached responses."""
        cache = create_memory_cache()
        semantic = SemanticCache(cache, similarity_threshold=0.5)

        await semantic.set_with_embedding(
            key="key1",
            prompt="What is Python?",
            response=sample_response,
        )

        result = await semantic.find_similar("What is Python?")
        assert result is not None
        key, response, similarity = result
        assert response == sample_response
        assert similarity > 0.5

    @pytest.mark.asyncio
    async def test_similarity_metrics(self, sample_response):
        """Test similarity hit/miss tracking."""
        cache = create_memory_cache()
        semantic = SemanticCache(cache, similarity_threshold=0.85)

        await semantic.set_with_embedding(
            key="key1",
            prompt="test",
            response=sample_response,
        )

        await semantic.find_similar("test")
        assert cache.metrics.similarity_hits >= 0


# Task 93: Cache Warming Strategies


class TestCacheWarmer:
    """Tests for cache warming."""

    @pytest.mark.asyncio
    async def test_warm_from_queries(self, sample_response):
        """Test warming cache from query list."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)

        cache = create_memory_cache()
        warmer = CacheWarmer(cache, mock_client)

        queries = [
            {"prompt": "What is AI?"},
            {"prompt": "Explain ML", "system": "You are a teacher"},
        ]

        warmed = await warmer.warm_from_queries(queries)
        assert warmed == 2
        assert mock_client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_warm_skip_existing(self, sample_response):
        """Test warming skips existing entries."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=sample_response)

        cache = create_memory_cache()
        warmer = CacheWarmer(cache, mock_client)

        queries = [{"prompt": "test"}]

        await warmer.warm_from_queries(queries)
        warmed = await warmer.warm_from_queries(queries)

        assert warmed == 0


# Task 94: Cache Invalidation Patterns


class TestCacheInvalidator:
    """Tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_tag_entry(self):
        """Test tagging cache entries."""
        cache = create_memory_cache()
        invalidator = CacheInvalidator(cache)

        invalidator.tag_entry("key1", "user:123", "session:abc")
        invalidator.tag_entry("key2", "user:123")

        assert "user:123" in invalidator._tags
        assert "key1" in invalidator._tags["user:123"]

    @pytest.mark.asyncio
    async def test_invalidate_by_tag(self, sample_response):
        """Test invalidating entries by tag."""
        cache = create_memory_cache()
        invalidator = CacheInvalidator(cache)

        await cache.set("key1", sample_response)
        await cache.set("key2", sample_response)

        invalidator.tag_entry("key1", "user:123")
        invalidator.tag_entry("key2", "user:123")

        count = await invalidator.invalidate_by_tag("user:123")
        assert count == 2


# Task 95: Distributed Cache (Redis Cluster)


class TestRedisClusterBackend:
    """Tests for Redis cluster backend."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test cluster backend initialization."""
        backend = RedisClusterBackend(
            nodes=[("localhost", 6379), ("localhost", 6380)],
            key_prefix="test:",
        )

        assert len(backend.nodes) == 2
        assert backend.key_prefix == "test:"


# Task 96: Cache Coherence Protocols


class TestCacheCoherence:
    """Tests for cache coherence."""

    @pytest.mark.asyncio
    async def test_broadcast_set(self, sample_response):
        """Test broadcasting set to multiple caches."""
        cache1 = create_memory_cache()
        cache2 = create_memory_cache()
        coherence = CacheCoherence([cache1, cache2])

        await coherence.broadcast_set("key1", sample_response)

        assert await cache1.get("key1") == sample_response
        assert await cache2.get("key1") == sample_response

    @pytest.mark.asyncio
    async def test_invalidate_all(self, sample_response):
        """Test invalidating key in all caches."""
        cache1 = create_memory_cache()
        cache2 = create_memory_cache()

        await cache1.set("key1", sample_response)
        await cache2.set("key1", sample_response)

        coherence = CacheCoherence([cache1, cache2])
        await coherence.invalidate_all("key1")

        assert await cache1.get("key1") is None
        assert await cache2.get("key1") is None


# Task 97: Cache Compression


class TestCompressedBackend:
    """Tests for cache compression."""

    @pytest.mark.asyncio
    async def test_compression_algorithms(self, cache_entry):
        """Test different compression algorithms."""
        for algo in [CompressionAlgorithm.GZIP, CompressionAlgorithm.ZLIB]:
            base_backend = InMemoryBackend()
            compressed = CompressedBackend(base_backend, algorithm=algo)

            await compressed.set("key1", cache_entry)
            result = await compressed.get("key1")

            assert result is not None
            assert result.response == cache_entry.response

    @pytest.mark.asyncio
    async def test_compression_stats(self, cache_entry):
        """Test compression statistics tracking."""
        base_backend = InMemoryBackend()
        compressed = CompressedBackend(base_backend, algorithm=CompressionAlgorithm.ZLIB)

        await compressed.set("key1", cache_entry)

        stats = compressed.compression_stats
        assert stats["compressed_bytes"] > 0
        assert stats["uncompressed_bytes"] > 0


# Task 98: Cache Tiering (L1/L2/L3)


class TestTieredCache:
    """Tests for multi-tier caching."""

    @pytest.mark.asyncio
    async def test_l1_hit(self, cache_entry):
        """Test L1 cache hit."""
        l1 = InMemoryBackend(max_size=10)
        l2 = InMemoryBackend(max_size=100)
        tiered = TieredCache(l1=l1, l2=l2)

        await tiered.set("key1", cache_entry)
        result = await tiered.get("key1")

        assert result is not None
        stats = tiered.get_tier_stats()
        assert stats["l1_hits"] == 1

    @pytest.mark.asyncio
    async def test_l2_promotion(self, cache_entry):
        """Test promoting L2 hits to L1."""
        l1 = InMemoryBackend(max_size=10)
        l2 = InMemoryBackend(max_size=100)
        tiered = TieredCache(l1=l1, l2=l2)

        await l2.set("key1", cache_entry)

        result = await tiered.get("key1")
        assert result is not None

        stats = tiered.get_tier_stats()
        assert stats["l2_hits"] == 1

        assert await l1.get("key1") is not None


# Task 99: Cache Hit Rate Optimization


class TestAdaptiveCache:
    """Tests for adaptive caching."""

    @pytest.mark.asyncio
    async def test_access_tracking(self, cache_entry):
        """Test access pattern tracking."""
        backend = InMemoryBackend()
        adaptive = AdaptiveCache(backend, optimization_interval=10)

        await adaptive.set("key1", cache_entry)

        for _ in range(5):
            await adaptive.get("key1")

        assert adaptive._request_count == 5
        assert "key1" in adaptive._access_frequency

    @pytest.mark.asyncio
    async def test_optimization_trigger(self, cache_entry):
        """Test optimization is triggered periodically."""
        backend = InMemoryBackend()
        adaptive = AdaptiveCache(backend, optimization_interval=10)

        await adaptive.set("key1", cache_entry)

        for i in range(11):
            await adaptive.get(f"key{i}")

        assert len(adaptive._access_frequency) < 11


# Task 100: Cache Cost Modeling


class TestCostModel:
    """Tests for cache cost modeling."""

    def test_api_cost_calculation(self):
        """Test API cost calculation."""
        model = CostModel(api_cost_per_1k_tokens=0.01, avg_tokens_per_request=500)

        cost = model.calculate_api_cost(10)
        assert cost == pytest.approx(0.05)

    def test_cache_cost_calculation(self):
        """Test cache operation cost calculation."""
        model = CostModel(cache_get_cost=0.00001, cache_set_cost=0.0001)

        cost = model.calculate_cache_cost(hits=50, misses=10, sets=10)
        assert cost == pytest.approx(0.0016)

    def test_savings_calculation(self):
        """Test savings calculation."""
        model = CostModel()
        metrics = CacheMetrics(hits=80, misses=20, sets=20)

        savings = model.calculate_savings(metrics)

        assert "cost_without_cache" in savings
        assert "cost_with_cache" in savings
        assert "savings" in savings
        assert savings["savings"] > 0

    def test_high_hit_rate_savings(self):
        """Test high hit rate produces significant savings."""
        model = CostModel()

        metrics_high = CacheMetrics(hits=95, misses=5, sets=5)
        savings_high = model.calculate_savings(metrics_high)

        metrics_low = CacheMetrics(hits=50, misses=50, sets=50)
        savings_low = model.calculate_savings(metrics_low)

        assert savings_high["savings"] > savings_low["savings"]
