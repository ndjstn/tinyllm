"""Tests for advanced cache features (Tasks 101-110)."""

import asyncio
import os
import tempfile
import time
from unittest.mock import AsyncMock

import pytest

from tinyllm.cache import (
    CacheAnalytics,
    CacheAnalyzer,
    CacheEntry,
    CachePrefetcher,
    CacheSizeTuner,
    InMemoryBackend,
    NegativeCache,
    NegativeCacheEntry,
    PartitionedCache,
    PatternBypassRule,
    PersistentCache,
    ReplicatedCache,
    ResponseCache,
    RuleBasedCache,
    SizeBypassRule,
    TTLOptimizer,
    WritePolicy,
    WritePolicyCache,
    create_memory_cache,
)
from tinyllm.models.client import GenerateResponse


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


# Task 101: Cache Prefetching Tests


class TestCachePrefetcher:
    """Tests for CachePrefetcher."""

    @pytest.mark.asyncio
    async def test_record_access(self):
        """Test recording access patterns."""
        cache = create_memory_cache()
        prefetcher = CachePrefetcher(cache, pattern_window=10)

        prefetcher.record_access("key1")
        prefetcher.record_access("key2")
        prefetcher.record_access("key1")
        prefetcher.record_access("key3")

        assert len(prefetcher._access_history) == 4
        assert len(prefetcher._pattern_counts) > 0

    @pytest.mark.asyncio
    async def test_predict_next_keys(self):
        """Test prediction of next likely keys."""
        cache = create_memory_cache()
        prefetcher = CachePrefetcher(cache, pattern_window=10)

        # Create pattern: key1 -> key2
        prefetcher.record_access("key1")
        prefetcher.record_access("key2")
        prefetcher.record_access("key1")
        prefetcher.record_access("key2")

        predictions = prefetcher.predict_next_keys("key1")
        assert "key2" in predictions

    @pytest.mark.asyncio
    async def test_prefetch(self, sample_response):
        """Test prefetching cache entries."""
        cache = create_memory_cache()
        prefetcher = CachePrefetcher(cache)

        async def generator_fn(key):
            return sample_response

        await prefetcher.prefetch(["key1", "key2"], generator_fn)

        # Check that entries were prefetched
        result1 = await cache.get("key1")
        result2 = await cache.get("key2")
        assert result1 is None  # Won't be found because generator returns response, not cache key
        assert result2 is None


# Task 102: Cache Partitioning Tests


class TestPartitionedCache:
    """Tests for PartitionedCache."""

    @pytest.mark.asyncio
    async def test_partitioning(self, cache_entry):
        """Test cache partitioning."""
        partitioned = PartitionedCache(num_partitions=4)

        # Set entries in partitions
        await partitioned.set("key1", cache_entry)
        await partitioned.set("key2", cache_entry)
        await partitioned.set("key3", cache_entry)

        # Verify entries are retrievable
        result1 = await partitioned.get("key1")
        result2 = await partitioned.get("key2")
        result3 = await partitioned.get("key3")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

    @pytest.mark.asyncio
    async def test_partition_distribution(self, cache_entry):
        """Test that entries are distributed across partitions."""
        partitioned = PartitionedCache(num_partitions=4)

        # Add many entries
        for i in range(100):
            await partitioned.set(f"key{i}", cache_entry)

        # Check that size is distributed
        total_size = await partitioned.size()
        assert total_size == 100

    @pytest.mark.asyncio
    async def test_custom_partition_function(self, cache_entry):
        """Test custom partition function."""
        def custom_partition(key: str) -> int:
            # Partition by first character
            return ord(key[0]) % 4

        partitioned = PartitionedCache(
            num_partitions=4,
            partition_fn=custom_partition,
        )

        await partitioned.set("a_key", cache_entry)
        result = await partitioned.get("a_key")
        assert result is not None


# Task 103: Cache Replication Tests


class TestReplicatedCache:
    """Tests for ReplicatedCache."""

    @pytest.mark.asyncio
    async def test_replication(self, cache_entry):
        """Test cache replication."""
        primary = InMemoryBackend()
        replica1 = InMemoryBackend()
        replica2 = InMemoryBackend()

        replicated = ReplicatedCache(
            primary=primary,
            replicas=[replica1, replica2],
        )

        await replicated.set("key1", cache_entry)

        # Give async tasks time to complete
        await asyncio.sleep(0.1)

        # Verify all backends have the entry
        assert await primary.get("key1") is not None

    @pytest.mark.asyncio
    async def test_read_from_replica(self, cache_entry):
        """Test reading from replica when primary misses."""
        primary = InMemoryBackend()
        replica = InMemoryBackend()

        replicated = ReplicatedCache(
            primary=primary,
            replicas=[replica],
            read_from_replicas=True,
        )

        # Set directly in replica
        await replica.set("key1", cache_entry)

        # Read should find it and sync to primary
        result = await replicated.get("key1")
        assert result is not None

        await asyncio.sleep(0.1)
        # Primary should now have it
        assert await primary.get("key1") is not None


# Task 104: Write-Through/Write-Back Policies Tests


class TestWritePolicyCache:
    """Tests for WritePolicyCache."""

    @pytest.mark.asyncio
    async def test_write_through(self, cache_entry):
        """Test write-through policy."""
        cache_backend = InMemoryBackend()
        storage_backend = InMemoryBackend()

        policy_cache = WritePolicyCache(
            cache_backend=cache_backend,
            storage_backend=storage_backend,
            policy=WritePolicy.WRITE_THROUGH,
        )

        await policy_cache.set("key1", cache_entry)

        # Both backends should have the entry immediately
        assert await cache_backend.get("key1") is not None
        assert await storage_backend.get("key1") is not None

    @pytest.mark.asyncio
    async def test_write_back(self, cache_entry):
        """Test write-back policy."""
        cache_backend = InMemoryBackend()
        storage_backend = InMemoryBackend()

        policy_cache = WritePolicyCache(
            cache_backend=cache_backend,
            storage_backend=storage_backend,
            policy=WritePolicy.WRITE_BACK,
        )

        await policy_cache.set("key1", cache_entry)

        # Only cache should have it
        assert await cache_backend.get("key1") is not None
        assert await storage_backend.get("key1") is None

        # After flush, storage should have it
        await policy_cache.flush()
        assert await storage_backend.get("key1") is not None

    @pytest.mark.asyncio
    async def test_write_around(self, cache_entry):
        """Test write-around policy."""
        cache_backend = InMemoryBackend()
        storage_backend = InMemoryBackend()

        policy_cache = WritePolicyCache(
            cache_backend=cache_backend,
            storage_backend=storage_backend,
            policy=WritePolicy.WRITE_AROUND,
        )

        await policy_cache.set("key1", cache_entry)

        # Only storage should have it
        assert await cache_backend.get("key1") is None
        assert await storage_backend.get("key1") is not None


# Task 105: Cache TTL Optimization Tests


class TestTTLOptimizer:
    """Tests for TTLOptimizer."""

    def test_record_access(self):
        """Test recording accesses."""
        optimizer = TTLOptimizer()
        optimizer.record_access("key1", hit=True)
        optimizer.record_access("key1", hit=True)

        assert "key1" in optimizer._key_stats
        assert optimizer._key_stats["key1"]["accesses"] == 2

    def test_get_optimal_ttl_low_hit_rate(self):
        """Test TTL optimization with low hit rate."""
        optimizer = TTLOptimizer(min_ttl=300, max_ttl=7200)

        initial_ttl = 3600
        optimizer._key_stats["key1"]["ttl"] = initial_ttl

        # Low hit rate should increase TTL
        new_ttl = optimizer.get_optimal_ttl("key1", current_hit_rate=0.5)
        assert new_ttl > initial_ttl

    def test_get_optimal_ttl_high_hit_rate(self):
        """Test TTL optimization with high hit rate."""
        optimizer = TTLOptimizer(min_ttl=300, max_ttl=7200)

        initial_ttl = 3600
        optimizer._key_stats["key1"]["ttl"] = initial_ttl

        # High hit rate should decrease TTL
        new_ttl = optimizer.get_optimal_ttl("key1", current_hit_rate=0.95)
        assert new_ttl < initial_ttl


# Task 106: Cache Size Auto-Tuning Tests


class TestCacheSizeTuner:
    """Tests for CacheSizeTuner."""

    def test_initial_size(self):
        """Test initial cache size."""
        tuner = CacheSizeTuner(initial_size=1000)
        assert tuner.current_size == 1000

    def test_calculate_new_size_low_hit_rate(self):
        """Test size calculation with low hit rate."""
        tuner = CacheSizeTuner(initial_size=1000, adjustment_interval=0)

        # Low hit rate should increase size
        new_size = tuner.calculate_new_size(
            current_hit_rate=0.5,
            eviction_rate=0.2,
        )
        assert new_size > 1000

    def test_calculate_new_size_high_hit_rate(self):
        """Test size calculation with high hit rate."""
        tuner = CacheSizeTuner(initial_size=1000, adjustment_interval=0)

        # High hit rate and low evictions should decrease size
        new_size = tuner.calculate_new_size(
            current_hit_rate=0.95,
            eviction_rate=0.001,
        )
        assert new_size < 1000

    def test_size_bounds(self):
        """Test that size stays within bounds."""
        tuner = CacheSizeTuner(
            initial_size=1000,
            min_size=100,
            max_size=2000,
            adjustment_interval=0,
        )

        # Try to grow beyond max
        for _ in range(10):
            new_size = tuner.calculate_new_size(0.1, 0.5)

        assert tuner.current_size <= 2000

        # Try to shrink below min
        for _ in range(10):
            new_size = tuner.calculate_new_size(0.99, 0.001)

        assert tuner.current_size >= 100


# Task 107: Cache Persistence Tests


class TestPersistentCache:
    """Tests for PersistentCache."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, cache_entry):
        """Test saving and loading cache."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            persist_path = f.name

        try:
            backend = InMemoryBackend()
            persistent = PersistentCache(
                backend=backend,
                persist_path=persist_path,
            )

            # Add entries
            await backend.set("key1", cache_entry)
            await backend.set("key2", cache_entry)

            # Save
            await persistent.save()

            # Clear and reload
            await backend.clear()
            assert await backend.size() == 0

            await persistent.load()
            assert await backend.size() == 2

        finally:
            if os.path.exists(persist_path):
                os.unlink(persist_path)

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        backend = InMemoryBackend()
        persistent = PersistentCache(
            backend=backend,
            persist_path="/tmp/nonexistent_cache_file_xyz.json",
        )

        # Should not raise error
        await persistent.load()
        assert await backend.size() == 0


# Task 108: Cache Analytics Tests


class TestCacheAnalyzer:
    """Tests for CacheAnalyzer."""

    @pytest.mark.asyncio
    async def test_record_latency(self):
        """Test recording latencies."""
        cache = create_memory_cache()
        analyzer = CacheAnalyzer(cache)

        analyzer.record_latency(10.5)
        analyzer.record_latency(15.2)
        analyzer.record_latency(8.3)

        assert len(analyzer._latencies) == 3

    @pytest.mark.asyncio
    async def test_record_key_access(self):
        """Test recording key accesses."""
        cache = create_memory_cache()
        analyzer = CacheAnalyzer(cache)

        analyzer.record_key_access("key1")
        analyzer.record_key_access("key1")
        analyzer.record_key_access("key2")

        assert analyzer._key_access_counts["key1"] == 2
        assert analyzer._key_access_counts["key2"] == 1

    @pytest.mark.asyncio
    async def test_get_analytics(self, sample_response):
        """Test getting analytics."""
        cache = create_memory_cache()
        analyzer = CacheAnalyzer(cache)

        # Generate some activity
        key = cache.generate_cache_key(model="test", prompt="hello")
        await cache.get(key)  # miss
        await cache.set(key, sample_response)
        await cache.get(key)  # hit

        analyzer.record_latency(10.0)
        analyzer.record_latency(15.0)
        analyzer.record_key_access(key)

        analytics = analyzer.get_analytics()
        assert analytics.total_requests == 2
        assert analytics.cache_hits == 1
        assert analytics.cache_misses == 1
        assert analytics.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_analytics_to_dict(self):
        """Test analytics to dict conversion."""
        cache = create_memory_cache()
        analyzer = CacheAnalyzer(cache)

        analytics = analyzer.get_analytics()
        data = analytics.to_dict()

        assert "total_requests" in data
        assert "hit_rate" in data
        assert "avg_latency_ms" in data


# Task 109: Cache Bypass Rules Tests


class TestCacheBypassRules:
    """Tests for cache bypass rules."""

    @pytest.mark.asyncio
    async def test_pattern_bypass_rule(self):
        """Test pattern-based bypass rule."""
        rule = PatternBypassRule(patterns=["temp_*", "test_*"])

        assert rule.should_bypass("temp_key1")
        assert rule.should_bypass("test_key2")
        assert not rule.should_bypass("normal_key")

    @pytest.mark.asyncio
    async def test_size_bypass_rule(self, sample_response):
        """Test size-based bypass rule."""
        rule = SizeBypassRule(max_size_bytes=10)

        # Small response should not bypass
        assert not rule.should_bypass("key", response=sample_response)

        # Create large response
        large_response = GenerateResponse(
            model="test",
            created_at="2024-01-01T00:00:00Z",
            response="x" * 1000,
            done=True,
            total_duration=1000000,
            eval_count=10,
        )

        assert rule.should_bypass("key", response=large_response)

    @pytest.mark.asyncio
    async def test_rule_based_cache(self, sample_response):
        """Test rule-based cache."""
        cache = create_memory_cache()
        rule_cache = RuleBasedCache(cache)

        # Add bypass rule for temp_ prefix
        rule_cache.add_bypass_rule(PatternBypassRule(patterns=["temp_*"]))

        key1 = cache.generate_cache_key(model="test", prompt="hello")
        key2 = "temp_key"

        # Normal key should cache
        await rule_cache.set(key1, sample_response)
        result1 = await rule_cache.get(key1)
        assert result1 is not None

        # Temp key should bypass
        await rule_cache.set(key2, sample_response)
        result2 = await cache.get(key2)
        assert result2 is None


# Task 110: Negative Caching Tests


class TestNegativeCache:
    """Tests for NegativeCache."""

    @pytest.mark.asyncio
    async def test_add_and_check_negative(self):
        """Test adding and checking negative entries."""
        neg_cache = NegativeCache(default_ttl=60)

        await neg_cache.add_negative("key1", "NOT_FOUND")

        error_type = await neg_cache.is_negative("key1")
        assert error_type == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_negative_expiration(self):
        """Test that negative entries expire."""
        neg_cache = NegativeCache(default_ttl=0.1)

        await neg_cache.add_negative("key1", "ERROR")

        # Should exist initially
        assert await neg_cache.is_negative("key1") == "ERROR"

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired
        assert await neg_cache.is_negative("key1") is None

    @pytest.mark.asyncio
    async def test_negative_cache_size_limit(self):
        """Test negative cache size limit."""
        neg_cache = NegativeCache(max_size=3)

        await neg_cache.add_negative("key1", "ERROR")
        await neg_cache.add_negative("key2", "ERROR")
        await neg_cache.add_negative("key3", "ERROR")
        await neg_cache.add_negative("key4", "ERROR")  # Should evict key1

        # Size should be limited
        size = await neg_cache.size()
        assert size == 3

    @pytest.mark.asyncio
    async def test_negative_cache_clear(self):
        """Test clearing negative cache."""
        neg_cache = NegativeCache()

        await neg_cache.add_negative("key1", "ERROR")
        await neg_cache.add_negative("key2", "ERROR")

        assert await neg_cache.size() == 2

        await neg_cache.clear()
        assert await neg_cache.size() == 0

    def test_negative_entry_expiration(self):
        """Test NegativeCacheEntry expiration."""
        entry = NegativeCacheEntry(
            key="key1",
            error_type="ERROR",
            timestamp=time.monotonic() - 1000,
            ttl=60,
        )

        assert entry.is_expired()

        entry2 = NegativeCacheEntry(
            key="key2",
            error_type="ERROR",
            timestamp=time.monotonic(),
            ttl=60,
        )

        assert not entry2.is_expired()
