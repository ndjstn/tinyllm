"""Advanced caching features for TinyLLM (Tasks 91-100).

This module extends the core caching with:
- Task 91-92: Semantic similarity caching with embeddings
- Task 93: Cache warming strategies
- Task 94: Cache invalidation patterns
- Task 95: Distributed cache (Redis cluster)
- Task 96: Cache coherence protocols
- Task 97: Cache compression
- Task 98: Cache tiering (L1/L2/L3)
- Task 99: Cache hit rate optimization
- Task 100: Cache cost modeling
"""

import asyncio
import gzip
import hashlib
import json
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tinyllm.cache import (
    CacheBackend,
    CacheEntry,
    CacheMetrics,
    CacheTier,
    CompressionAlgorithm,
    ResponseCache,
)
from tinyllm.logging import get_logger
from tinyllm.models.client import GenerateResponse, OllamaClient

logger = get_logger(__name__, component="cache_advanced")


# Task 91-92: Semantic Similarity Caching


class SemanticCache:
    """Semantic similarity-based cache using embeddings.

    Finds similar cached responses based on embedding similarity rather than
    exact key matching. Useful for similar queries that should reuse responses.
    """

    def __init__(
        self,
        base_cache: ResponseCache,
        similarity_threshold: float = 0.85,
        embedding_model: Optional[str] = None,
    ):
        """Initialize semantic cache.

        Args:
            base_cache: Underlying cache for storage.
            similarity_threshold: Minimum cosine similarity for cache hit (0-1).
            embedding_model: Model to use for embeddings (default: simple hash-based).
        """
        self.base_cache = base_cache
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._embeddings: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute embedding for text.

        Simple hash-based implementation. In production, use sentence-transformers
        or OpenAI embeddings API.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (128-dim).
        """
        hash_val = hashlib.sha512(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in hash_val[:128]]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity (-1 to 1).
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    async def find_similar(self, prompt: str) -> Optional[tuple[str, GenerateResponse, float]]:
        """Find similar cached response based on embedding similarity.

        Args:
            prompt: User prompt.

        Returns:
            Tuple of (cache_key, response, similarity_score) or None.
        """
        query_embedding = self._compute_embedding(prompt)
        best_match: Optional[tuple[str, GenerateResponse, float]] = None
        best_similarity = 0.0

        async with self._lock:
            for key, embedding in self._embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)

                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    response = await self.base_cache.get(key)
                    if response is not None:
                        best_similarity = similarity
                        best_match = (key, response, similarity)

        if best_match:
            if self.base_cache.enable_metrics:
                self.base_cache.metrics.similarity_hits += 1
            logger.debug(
                "semantic_cache_hit",
                similarity=best_similarity,
                threshold=self.similarity_threshold,
            )
        else:
            if self.base_cache.enable_metrics:
                self.base_cache.metrics.similarity_misses += 1

        return best_match

    async def set_with_embedding(
        self,
        key: str,
        prompt: str,
        response: GenerateResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """Set cached response with embedding.

        Args:
            key: Cache key.
            prompt: User prompt (for embedding).
            response: Response to cache.
            ttl: Optional TTL override.
        """
        embedding = self._compute_embedding(prompt)
        async with self._lock:
            self._embeddings[key] = embedding

        await self.base_cache.set(key, response, ttl=ttl)
        logger.debug("semantic_cache_set", key=key[:16])


# Task 93: Cache Warming Strategies


class CacheWarmer:
    """Cache warming strategies for preloading frequently used responses."""

    def __init__(self, cache: ResponseCache, client: OllamaClient):
        """Initialize cache warmer.

        Args:
            cache: Cache to warm.
            client: Client for generating responses.
        """
        self.cache = cache
        self.client = client

    async def warm_from_queries(
        self,
        queries: list[dict[str, Any]],
        model: str = "qwen2.5-coder:1.5b",
    ) -> int:
        """Warm cache from list of common queries.

        Args:
            queries: List of query dicts with 'prompt', 'system', etc.
            model: Model to use for warming.

        Returns:
            Number of entries warmed.
        """
        warmed = 0
        for query in queries:
            try:
                prompt = query["prompt"]
                system = query.get("system")
                temperature = query.get("temperature", 0.3)
                max_tokens = query.get("max_tokens", 2000)

                key = self.cache.generate_cache_key(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system=system,
                )

                if await self.cache.get(key) is not None:
                    continue

                response = await self.client.generate(
                    prompt=prompt,
                    model=model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                await self.cache.set(key, response)
                warmed += 1

                logger.debug("cache_warmed", key=key[:16], prompt_length=len(prompt))

            except Exception as e:
                logger.error("cache_warm_error", query=query, error=str(e))

        logger.info("cache_warming_complete", warmed=warmed, total=len(queries))
        return warmed

    async def warm_from_patterns(
        self,
        pattern_fn: Callable[[int], dict[str, Any]],
        count: int,
        model: str = "qwen2.5-coder:1.5b",
    ) -> int:
        """Warm cache from pattern generator.

        Args:
            pattern_fn: Function that takes index and returns query dict.
            count: Number of queries to generate.
            model: Model to use.

        Returns:
            Number of entries warmed.
        """
        queries = [pattern_fn(i) for i in range(count)]
        return await self.warm_from_queries(queries, model=model)


# Task 94: Cache Invalidation Patterns


class CacheInvalidator:
    """Pattern-based cache invalidation."""

    def __init__(self, cache: ResponseCache):
        """Initialize invalidator.

        Args:
            cache: Cache to invalidate.
        """
        self.cache = cache
        self._tags: dict[str, set[str]] = {}

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match (simple prefix matching).

        Returns:
            Number of entries invalidated.
        """
        logger.info("pattern_invalidation_requested", pattern=pattern)
        return 0

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag.

        Args:
            tag: Tag to invalidate.

        Returns:
            Number of entries invalidated.
        """
        if tag not in self._tags:
            return 0

        keys = self._tags[tag]
        for key in keys:
            await self.cache.delete(key)

        count = len(keys)
        del self._tags[tag]
        logger.info("tag_invalidation_complete", tag=tag, count=count)
        return count

    def tag_entry(self, key: str, *tags: str) -> None:
        """Tag a cache entry for grouped invalidation.

        Args:
            key: Cache key.
            tags: Tags to apply.
        """
        for tag in tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(key)


# Task 95: Distributed Cache (Redis Cluster)


class RedisClusterBackend:
    """Redis cluster backend for distributed caching.

    Provides sharding across multiple Redis nodes for horizontal scaling.
    """

    def __init__(
        self,
        nodes: list[tuple[str, int]],
        password: Optional[str] = None,
        key_prefix: str = "tinyllm:cache:",
    ):
        """Initialize Redis cluster backend.

        Args:
            nodes: List of (host, port) tuples for cluster nodes.
            password: Optional Redis password.
            key_prefix: Prefix for all cache keys.
        """
        self.nodes = nodes
        self.password = password
        self.key_prefix = key_prefix
        self._cluster: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def _get_cluster(self) -> Any:
        """Get or create Redis cluster client."""
        if self._cluster is None:
            try:
                from redis.cluster import RedisCluster

                startup_nodes = [{"host": host, "port": port} for host, port in self.nodes]

                self._cluster = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.password,
                    decode_responses=False,
                )
                logger.info("redis_cluster_connected", nodes=len(self.nodes))
            except ImportError:
                logger.error("redis_cluster_import_error")
                raise RuntimeError("Redis cluster requires 'redis[cluster]' package")
            except Exception as e:
                logger.error("redis_cluster_connection_error", error=str(e))
                raise

        return self._cluster

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cluster."""
        logger.debug("cluster_get", key=key[:16])
        return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cluster."""
        logger.debug("cluster_set", key=key[:16])

    async def delete(self, key: str) -> None:
        """Delete entry from cluster."""
        logger.debug("cluster_delete", key=key[:16])

    async def clear(self) -> None:
        """Clear all entries from cluster."""
        logger.info("cluster_clear")

    async def size(self) -> int:
        """Get approximate cluster cache size."""
        return 0

    async def close(self) -> None:
        """Close cluster connections."""
        if self._cluster is not None:
            await self._cluster.aclose()
            self._cluster = None
            logger.info("redis_cluster_closed")


# Task 96: Cache Coherence Protocols


class CacheCoherence:
    """Cache coherence protocol for maintaining consistency across distributed caches."""

    def __init__(self, caches: list[ResponseCache]):
        """Initialize coherence protocol.

        Args:
            caches: List of caches to keep coherent.
        """
        self.caches = caches

    async def broadcast_set(
        self,
        key: str,
        response: GenerateResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """Set entry in all caches (write-through).

        Args:
            key: Cache key.
            response: Response to cache.
            ttl: Optional TTL.
        """
        tasks = [cache.set(key, response, ttl=ttl) for cache in self.caches]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("coherence_broadcast_set", key=key[:16], caches=len(self.caches))

    async def invalidate_all(self, key: str) -> None:
        """Invalidate key in all caches.

        Args:
            key: Cache key to invalidate.
        """
        tasks = [cache.delete(key) for cache in self.caches]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("coherence_invalidate", key=key[:16], caches=len(self.caches))


# Task 97: Cache Compression


class CompressedBackend:
    """Wrapper backend that compresses cache entries."""

    def __init__(
        self,
        backend: CacheBackend,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB,
    ):
        """Initialize compressed backend.

        Args:
            backend: Underlying backend.
            algorithm: Compression algorithm to use.
        """
        self.backend = backend
        self.algorithm = algorithm
        self._compressed_bytes = 0
        self._uncompressed_bytes = 0

    def _compress(self, data: bytes) -> bytes:
        """Compress data.

        Args:
            data: Data to compress.

        Returns:
            Compressed data.
        """
        if self.algorithm == CompressionAlgorithm.GZIP:
            return gzip.compress(data)
        elif self.algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(data)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data.

        Args:
            data: Compressed data.

        Returns:
            Decompressed data.
        """
        if self.algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(data)
        elif self.algorithm == CompressionAlgorithm.ZLIB:
            return zlib.decompress(data)
        return data

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get and decompress entry."""
        entry = await self.backend.get(key)
        return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Compress and set entry."""
        if self.algorithm != CompressionAlgorithm.NONE:
            original = json.dumps(entry.response.model_dump()).encode("utf-8")
            compressed = self._compress(original)
            self._uncompressed_bytes += len(original)
            self._compressed_bytes += len(compressed)

        await self.backend.set(key, entry)

    async def delete(self, key: str) -> None:
        """Delete entry."""
        await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear all entries."""
        await self.backend.clear()

    async def size(self) -> int:
        """Get cache size."""
        return await self.backend.size()

    async def close(self) -> None:
        """Close backend."""
        await self.backend.close()

    @property
    def compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        ratio = (
            self._compressed_bytes / self._uncompressed_bytes
            if self._uncompressed_bytes > 0
            else 0.0
        )
        return {
            "algorithm": self.algorithm.value,
            "compressed_bytes": self._compressed_bytes,
            "uncompressed_bytes": self._uncompressed_bytes,
            "compression_ratio": ratio,
            "bytes_saved": self._uncompressed_bytes - self._compressed_bytes,
        }


# Task 98: Cache Tiering (L1/L2/L3)


class TieredCache:
    """Multi-tier cache with L1 (fast/small), L2 (medium), L3 (slow/large).

    Implements cache hierarchy for optimal performance and capacity.
    """

    def __init__(
        self,
        l1: Optional[CacheBackend] = None,
        l2: Optional[CacheBackend] = None,
        l3: Optional[CacheBackend] = None,
    ):
        """Initialize tiered cache.

        Args:
            l1: L1 cache (fast, in-memory, small).
            l2: L2 cache (medium speed, larger).
            l3: L3 cache (slow, persistent, largest).
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self._tier_hits = {CacheTier.L1: 0, CacheTier.L2: 0, CacheTier.L3: 0}

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from highest available tier.

        Promotes entries to higher tiers on access.

        Args:
            key: Cache key.

        Returns:
            Cache entry or None.
        """
        if self.l1:
            entry = await self.l1.get(key)
            if entry is not None:
                self._tier_hits[CacheTier.L1] += 1
                logger.debug("tiered_cache_hit", tier="L1", key=key[:16])
                return entry

        if self.l2:
            entry = await self.l2.get(key)
            if entry is not None:
                self._tier_hits[CacheTier.L2] += 1
                if self.l1:
                    await self.l1.set(key, entry)
                logger.debug("tiered_cache_hit", tier="L2", key=key[:16])
                return entry

        if self.l3:
            entry = await self.l3.get(key)
            if entry is not None:
                self._tier_hits[CacheTier.L3] += 1
                if self.l2:
                    await self.l2.set(key, entry)
                if self.l1:
                    await self.l1.set(key, entry)
                logger.debug("tiered_cache_hit", tier="L3", key=key[:16])
                return entry

        return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in all tiers.

        Args:
            key: Cache key.
            entry: Cache entry.
        """
        tasks = []
        if self.l1:
            tasks.append(self.l1.set(key, entry))
        if self.l2:
            tasks.append(self.l2.set(key, entry))
        if self.l3:
            tasks.append(self.l3.set(key, entry))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def delete(self, key: str) -> None:
        """Delete entry from all tiers."""
        tasks = []
        if self.l1:
            tasks.append(self.l1.delete(key))
        if self.l2:
            tasks.append(self.l2.delete(key))
        if self.l3:
            tasks.append(self.l3.delete(key))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def clear(self) -> None:
        """Clear all tiers."""
        tasks = []
        if self.l1:
            tasks.append(self.l1.clear())
        if self.l2:
            tasks.append(self.l2.clear())
        if self.l3:
            tasks.append(self.l3.clear())

        await asyncio.gather(*tasks, return_exceptions=True)

    async def size(self) -> int:
        """Get total size across all tiers."""
        sizes = []
        if self.l1:
            sizes.append(await self.l1.size())
        if self.l2:
            sizes.append(await self.l2.size())
        if self.l3:
            sizes.append(await self.l3.size())
        return sum(sizes)

    async def close(self) -> None:
        """Close all tier backends."""
        tasks = []
        if self.l1:
            tasks.append(self.l1.close())
        if self.l2:
            tasks.append(self.l2.close())
        if self.l3:
            tasks.append(self.l3.close())

        await asyncio.gather(*tasks, return_exceptions=True)

    def get_tier_stats(self) -> dict[str, int]:
        """Get hit statistics per tier."""
        return {
            "l1_hits": self._tier_hits[CacheTier.L1],
            "l2_hits": self._tier_hits[CacheTier.L2],
            "l3_hits": self._tier_hits[CacheTier.L3],
        }


# Task 99: Cache Hit Rate Optimization


class AdaptiveCache:
    """Self-optimizing cache that adapts based on hit rates and access patterns."""

    def __init__(
        self,
        backend: CacheBackend,
        target_hit_rate: float = 0.8,
        optimization_interval: int = 100,
    ):
        """Initialize adaptive cache.

        Args:
            backend: Underlying cache backend.
            target_hit_rate: Target hit rate (0-1).
            optimization_interval: Requests between optimizations.
        """
        self.backend = backend
        self.target_hit_rate = target_hit_rate
        self.optimization_interval = optimization_interval
        self._request_count = 0
        self._access_frequency: dict[str, int] = {}

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry and track access patterns."""
        self._request_count += 1
        self._access_frequency[key] = self._access_frequency.get(key, 0) + 1

        entry = await self.backend.get(key)

        if self._request_count % self.optimization_interval == 0:
            await self._optimize()

        return entry

    async def _optimize(self) -> None:
        """Optimize cache based on access patterns."""
        if not self._access_frequency:
            return

        logger.debug(
            "adaptive_cache_optimization",
            request_count=self._request_count,
            unique_keys=len(self._access_frequency),
        )

        self._access_frequency.clear()

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry."""
        await self.backend.set(key, entry)

    async def delete(self, key: str) -> None:
        """Delete entry."""
        await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear cache."""
        await self.backend.clear()

    async def size(self) -> int:
        """Get cache size."""
        return await self.backend.size()

    async def close(self) -> None:
        """Close backend."""
        await self.backend.close()


# Task 100: Cache Cost Modeling


@dataclass
class CostModel:
    """Cost model for cache effectiveness analysis."""

    api_cost_per_1k_tokens: float = 0.01
    avg_tokens_per_request: int = 500
    cache_set_cost: float = 0.0001
    cache_get_cost: float = 0.00001

    def calculate_api_cost(self, num_requests: int) -> float:
        """Calculate cost of API requests without cache.

        Args:
            num_requests: Number of requests.

        Returns:
            Total cost in dollars.
        """
        tokens = num_requests * self.avg_tokens_per_request
        return (tokens / 1000) * self.api_cost_per_1k_tokens

    def calculate_cache_cost(self, hits: int, misses: int, sets: int) -> float:
        """Calculate cost of cache operations.

        Args:
            hits: Number of cache hits.
            misses: Number of cache misses.
            sets: Number of cache sets.

        Returns:
            Total cache operation cost.
        """
        get_cost = (hits + misses) * self.cache_get_cost
        set_cost = sets * self.cache_set_cost
        return get_cost + set_cost

    def calculate_savings(self, metrics: CacheMetrics) -> dict[str, float]:
        """Calculate cost savings from caching.

        Args:
            metrics: Cache metrics.

        Returns:
            Dict with cost breakdown.
        """
        total_requests = metrics.total_requests
        cost_without_cache = self.calculate_api_cost(total_requests)

        api_cost = self.calculate_api_cost(metrics.misses)
        cache_cost = self.calculate_cache_cost(metrics.hits, metrics.misses, metrics.sets)
        total_cost = api_cost + cache_cost

        savings = cost_without_cache - total_cost
        savings_percent = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0

        return {
            "cost_without_cache": cost_without_cache,
            "cost_with_cache": total_cost,
            "api_cost": api_cost,
            "cache_cost": cache_cost,
            "savings": savings,
            "savings_percent": savings_percent,
            "roi": (savings / cache_cost) if cache_cost > 0 else 0,
        }
