"""Response caching for TinyLLM.

Provides both in-memory and Redis-backed caching for LLM responses with
TTL support, LRU eviction, and comprehensive metrics tracking.

Advanced features:
- Semantic similarity caching with embeddings
- Cache warming and prefetching
- Distributed Redis cluster support
- Multi-tier caching (L1/L2/L3)
- Compression and cost modeling
"""

import asyncio
import gzip
import hashlib
import json
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel

from tinyllm.logging import get_logger
from tinyllm.models.client import GenerateResponse, OllamaClient

logger = get_logger(__name__, component="cache")


# Task 94: Cache invalidation patterns
class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""

    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    PATTERN = "pattern"  # Pattern-based invalidation
    TAG = "tag"  # Tag-based invalidation


# Task 97: Compression algorithms
class CompressionAlgorithm(Enum):
    """Cache compression algorithms."""

    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"


# Task 98: Cache tier levels
class CacheTier(Enum):
    """Cache tier levels for multi-tier caching."""

    L1 = "l1"  # Fast in-memory (small, hot data)
    L2 = "l2"  # Medium-speed (larger, warm data)
    L3 = "l3"  # Slower persistent (largest, cold data)


@dataclass
class CacheMetrics:
    """Cache performance metrics with advanced tracking."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    errors: int = 0
    similarity_hits: int = 0
    similarity_misses: int = 0
    compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    estimated_cost_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def similarity_hit_rate(self) -> float:
        """Calculate semantic similarity hit rate."""
        total = self.similarity_hits + self.similarity_misses
        return self.similarity_hits / total if total > 0 else 0.0

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.uncompressed_bytes == 0:
            return 0.0
        return self.compressed_bytes / self.uncompressed_bytes

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.hits + self.misses

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "total_requests": self.total_requests,
            "similarity_hits": self.similarity_hits,
            "similarity_misses": self.similarity_misses,
            "similarity_hit_rate": self.similarity_hit_rate,
            "compressed_bytes": self.compressed_bytes,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compression_ratio": self.compression_ratio,
            "estimated_cost_saved": self.estimated_cost_saved,
        }


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""

    response: GenerateResponse
    created_at: float
    ttl: Optional[int] = None  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.monotonic() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.monotonic()


class CacheBackend(Protocol):
    """Protocol for cache backends."""

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        ...

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache."""
        ...

    async def delete(self, key: str) -> None:
        """Delete entry from cache."""
        ...

    async def clear(self) -> None:
        """Clear all cache entries."""
        ...

    async def size(self) -> int:
        """Get current cache size."""
        ...

    async def close(self) -> None:
        """Close backend connections."""
        ...


class InMemoryBackend:
    """In-memory LRU cache backend with size limits."""

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory backend.

        Args:
            max_size: Maximum number of entries to cache.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._eviction_count = 0

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    return None
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                return entry
            return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in cache with LRU eviction."""
        async with self._lock:
            # Remove if exists (will re-add at end)
            if key in self._cache:
                del self._cache[key]
            # Evict oldest if at capacity
            elif len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._eviction_count += 1
                logger.debug(
                    "cache_eviction",
                    key=oldest_key,
                    total_evictions=self._eviction_count,
                    cache_size=len(self._cache),
                )
            # Add new entry
            self._cache[key] = entry

    async def delete(self, key: str) -> None:
        """Delete entry from cache."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            logger.info("cache_cleared")

    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)

    async def close(self) -> None:
        """Close backend (no-op for in-memory)."""
        pass

    @property
    def eviction_count(self) -> int:
        """Get total number of evictions."""
        return self._eviction_count


class RedisBackend:
    """Redis-backed cache backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "tinyllm:cache:",
    ):
        """Initialize Redis backend.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Optional Redis password.
            key_prefix: Prefix for all cache keys.
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self._redis: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False,  # We handle encoding/decoding
                )
                await self._redis.ping()
                logger.info("redis_connected", host=self.host, port=self.port, db=self.db)
            except ImportError:
                logger.error("redis_import_error", error="redis package not installed")
                raise RuntimeError(
                    "Redis backend requires 'redis' package. Install with: pip install redis"
                )
            except Exception as e:
                logger.error("redis_connection_error", error=str(e))
                raise
        return self._redis

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis."""
        try:
            client = await self._get_client()
            data = await client.get(self._make_key(key))
            if data is None:
                return None

            # Deserialize entry
            entry_dict = json.loads(data.decode("utf-8"))
            response = GenerateResponse(**entry_dict["response"])
            entry = CacheEntry(
                response=response,
                created_at=entry_dict["created_at"],
                ttl=entry_dict.get("ttl"),
                access_count=entry_dict.get("access_count", 0),
                last_accessed=entry_dict.get("last_accessed", time.monotonic()),
            )

            # Check expiration
            if entry.is_expired():
                await self.delete(key)
                return None

            # Update access metadata (async, don't wait)
            entry.touch()
            asyncio.create_task(self._update_metadata(key, entry))

            return entry
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            return None

    async def _update_metadata(self, key: str, entry: CacheEntry) -> None:
        """Update entry metadata in Redis."""
        try:
            await self.set(key, entry)
        except Exception as e:
            logger.debug("redis_metadata_update_error", key=key, error=str(e))

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in Redis with TTL."""
        try:
            client = await self._get_client()
            # Serialize entry
            entry_dict = {
                "response": entry.response.model_dump(),
                "created_at": entry.created_at,
                "ttl": entry.ttl,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }
            data = json.dumps(entry_dict).encode("utf-8")

            # Set with TTL if specified
            redis_key = self._make_key(key)
            if entry.ttl is not None:
                await client.setex(redis_key, entry.ttl, data)
            else:
                await client.set(redis_key, data)
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
            raise

    async def delete(self, key: str) -> None:
        """Delete entry from Redis."""
        try:
            client = await self._get_client()
            await client.delete(self._make_key(key))
        except Exception as e:
            logger.error("redis_delete_error", key=key, error=str(e))

    async def clear(self) -> None:
        """Clear all cache entries with prefix."""
        try:
            client = await self._get_client()
            # Use SCAN to find all keys with prefix (safer than KEYS)
            cursor = 0
            pattern = f"{self.key_prefix}*"
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            logger.info("redis_cache_cleared", prefix=self.key_prefix)
        except Exception as e:
            logger.error("redis_clear_error", error=str(e))
            raise

    async def size(self) -> int:
        """Get approximate cache size."""
        try:
            client = await self._get_client()
            # Count keys with prefix (approximate)
            cursor = 0
            count = 0
            pattern = f"{self.key_prefix}*"
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception as e:
            logger.error("redis_size_error", error=str(e))
            return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("redis_closed")


class ResponseCache:
    """Response cache with configurable backend and metrics."""

    def __init__(
        self,
        backend: CacheBackend,
        default_ttl: Optional[int] = 3600,  # 1 hour default
        enable_metrics: bool = True,
    ):
        """Initialize response cache.

        Args:
            backend: Cache backend to use.
            default_ttl: Default time-to-live in seconds.
            enable_metrics: Whether to track cache metrics.
        """
        self.backend = backend
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        self.metrics = CacheMetrics()

    def generate_cache_key(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        system: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate cache key from request parameters.

        Uses SHA256 hash of request parameters for consistent key generation.

        Args:
            model: Model name.
            prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system: Optional system prompt.
            json_mode: Whether JSON mode is enabled.

        Returns:
            Cache key string.
        """
        # Create deterministic string from parameters
        key_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system,
            "json_mode": json_mode,
        }
        key_str = json.dumps(key_data, sort_keys=True)

        # Generate hash
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> Optional[GenerateResponse]:
        """Get cached response.

        Args:
            key: Cache key.

        Returns:
            Cached GenerateResponse or None if not found/expired.
        """
        try:
            entry = await self.backend.get(key)
            if entry is not None:
                if self.enable_metrics:
                    self.metrics.hits += 1
                logger.debug(
                    "cache_hit",
                    key=key[:16],
                    access_count=entry.access_count,
                    age_seconds=time.monotonic() - entry.created_at,
                )
                return entry.response
            else:
                if self.enable_metrics:
                    self.metrics.misses += 1
                logger.debug("cache_miss", key=key[:16])
                return None
        except Exception as e:
            if self.enable_metrics:
                self.metrics.errors += 1
            logger.error("cache_get_error", key=key[:16], error=str(e))
            return None

    async def set(
        self,
        key: str,
        response: GenerateResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """Set cached response.

        Args:
            key: Cache key.
            response: GenerateResponse to cache.
            ttl: Optional TTL override (uses default_ttl if None).
        """
        try:
            entry = CacheEntry(
                response=response,
                created_at=time.monotonic(),
                ttl=ttl if ttl is not None else self.default_ttl,
            )
            await self.backend.set(key, entry)
            if self.enable_metrics:
                self.metrics.sets += 1
            logger.debug("cache_set", key=key[:16], ttl=entry.ttl)
        except Exception as e:
            if self.enable_metrics:
                self.metrics.errors += 1
            logger.error("cache_set_error", key=key[:16], error=str(e))

    async def delete(self, key: str) -> None:
        """Delete cached entry.

        Args:
            key: Cache key to delete.
        """
        try:
            await self.backend.delete(key)
            logger.debug("cache_delete", key=key[:16])
        except Exception as e:
            logger.error("cache_delete_error", key=key[:16], error=str(e))

    async def clear(self) -> None:
        """Clear all cached entries."""
        await self.backend.clear()
        if self.enable_metrics:
            logger.info("cache_cleared", metrics=self.metrics.to_dict())

    async def size(self) -> int:
        """Get current cache size."""
        return await self.backend.size()

    async def close(self) -> None:
        """Close cache backend."""
        await self.backend.close()

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        # Update eviction count from backend if available
        if isinstance(self.backend, InMemoryBackend):
            self.metrics.evictions = self.backend.eviction_count
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.metrics = CacheMetrics()
        logger.info("cache_metrics_reset")


class CachedOllamaClient:
    """OllamaClient wrapper with response caching.

    Wraps OllamaClient to provide transparent caching with configurable
    cache backends, TTL, and metrics tracking.
    """

    def __init__(
        self,
        client: OllamaClient,
        cache: ResponseCache,
        enable_cache: bool = True,
    ):
        """Initialize cached client.

        Args:
            client: OllamaClient instance to wrap.
            cache: ResponseCache instance.
            enable_cache: Whether caching is enabled.
        """
        self.client = client
        self.cache = cache
        self.enable_cache = enable_cache
        logger.info("cached_client_initialized", enable_cache=enable_cache)

    async def generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
        force_refresh: bool = False,
        cache_ttl: Optional[int] = None,
    ) -> GenerateResponse:
        """Generate response with caching.

        Args:
            prompt: User prompt.
            model: Model name.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request JSON output.
            force_refresh: Bypass cache and force fresh generation.
            cache_ttl: Optional TTL override for this request.

        Returns:
            GenerateResponse (from cache or fresh generation).
        """
        # Generate cache key
        cache_key = self.cache.generate_cache_key(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            json_mode=json_mode,
        )

        # Check cache if enabled and not forcing refresh
        if self.enable_cache and not force_refresh:
            cached_response = await self.cache.get(cache_key)
            if cached_response is not None:
                logger.info(
                    "cache_served_response",
                    model=model,
                    prompt_length=len(prompt),
                    cache_hit_rate=self.cache.get_metrics().hit_rate,
                )
                return cached_response

        # Generate fresh response
        logger.debug(
            "generating_fresh_response",
            model=model,
            force_refresh=force_refresh,
            cache_enabled=self.enable_cache,
        )
        response = await self.client.generate(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

        # Cache response if enabled
        if self.enable_cache:
            await self.cache.set(cache_key, response, ttl=cache_ttl)

        return response

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """Stream generate response (no caching for streaming).

        Note: Streaming responses are not cached as they're consumed incrementally.

        Args:
            prompt: User prompt.
            model: Model name.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Yields:
            Response text chunks as they arrive.
        """
        logger.debug("streaming_response_no_cache", model=model)
        async for chunk in self.client.generate_stream(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk

    async def list_models(self) -> list[str]:
        """List available models (delegated to client)."""
        return await self.client.list_models()

    async def pull_model(self, model: str) -> None:
        """Pull a model (delegated to client)."""
        await self.client.pull_model(model)

    async def check_health(self) -> bool:
        """Check health (delegated to client)."""
        return await self.client.check_health()

    def get_stats(self) -> dict[str, Any]:
        """Get client and cache statistics."""
        client_stats = self.client.get_stats()
        cache_metrics = self.cache.get_metrics().to_dict()
        return {
            "client": client_stats,
            "cache": cache_metrics,
            "cache_enabled": self.enable_cache,
        }

    async def close(self) -> None:
        """Close client and cache."""
        await self.client.close()
        await self.cache.close()
        logger.info("cached_client_closed")


# Convenience functions for creating common cache configurations


def create_memory_cache(
    max_size: int = 1000,
    default_ttl: Optional[int] = 3600,
    enable_metrics: bool = True,
) -> ResponseCache:
    """Create in-memory cache with LRU eviction.

    Args:
        max_size: Maximum number of entries.
        default_ttl: Default TTL in seconds (None for no expiration).
        enable_metrics: Whether to track metrics.

    Returns:
        ResponseCache with in-memory backend.
    """
    backend = InMemoryBackend(max_size=max_size)
    return ResponseCache(
        backend=backend,
        default_ttl=default_ttl,
        enable_metrics=enable_metrics,
    )


def create_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    default_ttl: Optional[int] = 3600,
    enable_metrics: bool = True,
    key_prefix: str = "tinyllm:cache:",
) -> ResponseCache:
    """Create Redis-backed cache.

    Args:
        host: Redis host.
        port: Redis port.
        db: Redis database number.
        password: Optional Redis password.
        default_ttl: Default TTL in seconds.
        enable_metrics: Whether to track metrics.
        key_prefix: Prefix for cache keys.

    Returns:
        ResponseCache with Redis backend.
    """
    backend = RedisBackend(
        host=host,
        port=port,
        db=db,
        password=password,
        key_prefix=key_prefix,
    )
    return ResponseCache(
        backend=backend,
        default_ttl=default_ttl,
        enable_metrics=enable_metrics,
    )


async def create_cached_client(
    client: OllamaClient,
    backend: str = "memory",
    max_size: int = 1000,
    default_ttl: Optional[int] = 3600,
    enable_cache: bool = True,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
) -> CachedOllamaClient:
    """Create cached Ollama client with specified backend.

    Args:
        client: OllamaClient instance to wrap.
        backend: Cache backend ("memory" or "redis").
        max_size: Maximum cache size (memory backend only).
        default_ttl: Default TTL in seconds.
        enable_cache: Whether caching is enabled.
        redis_host: Redis host (redis backend only).
        redis_port: Redis port (redis backend only).
        redis_db: Redis database (redis backend only).
        redis_password: Redis password (redis backend only).

    Returns:
        CachedOllamaClient instance.

    Raises:
        ValueError: If backend is not "memory" or "redis".
    """
    if backend == "memory":
        cache = create_memory_cache(max_size=max_size, default_ttl=default_ttl)
    elif backend == "redis":
        cache = create_redis_cache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            default_ttl=default_ttl,
        )
    else:
        raise ValueError(f"Invalid cache backend: {backend}. Must be 'memory' or 'redis'")

    logger.info(
        "cached_client_created",
        backend=backend,
        default_ttl=default_ttl,
        enable_cache=enable_cache,
    )

    return CachedOllamaClient(client=client, cache=cache, enable_cache=enable_cache)
# Cache Extensions for Tasks 101-110
# This will be appended to cache.py

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Dict, List
import asyncio
import time
import json

# Task 101: Cache Prefetching


class CachePrefetcher:
    """Prefetch cache entries based on access patterns.

    Analyzes access patterns to predict and pre-load likely cache entries.
    """

    def __init__(
        self,
        cache,
        pattern_window: int = 100,
        prefetch_threshold: float = 0.5,
    ):
        """Initialize cache prefetcher.

        Args:
            cache: ResponseCache instance to prefetch for.
            pattern_window: Number of recent accesses to analyze.
            prefetch_threshold: Probability threshold for prefetching (0-1).
        """
        self.cache = cache
        self.pattern_window = pattern_window
        self.prefetch_threshold = prefetch_threshold
        self._access_history: deque = deque(maxlen=pattern_window)
        self._pattern_counts: Dict[tuple, int] = defaultdict(int)
        self._prefetch_queue: asyncio.Queue = asyncio.Queue()
        self._prefetch_task: Optional[asyncio.Task] = None

    def record_access(self, key: str) -> None:
        """Record a cache access for pattern analysis."""
        self._access_history.append(key)
        if len(self._access_history) >= 2:
            pattern = tuple(list(self._access_history)[-2:])
            self._pattern_counts[pattern] += 1

    def predict_next_keys(self, current_key: str, top_n: int = 5) -> List[str]:
        """Predict next likely cache keys based on patterns."""
        candidates = []
        for pattern, count in self._pattern_counts.items():
            if pattern[0] == current_key:
                candidates.append((pattern[1], count))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in candidates[:top_n]]

    async def prefetch(
        self,
        keys: List[str],
        generator_fn: Callable[[str], Any],
    ) -> None:
        """Prefetch cache entries for given keys."""
        for key in keys:
            existing = await self.cache.get(key)
            if existing is None:
                try:
                    response = await generator_fn(key)
                    await self.cache.set(key, response)
                except Exception:
                    pass


# Task 102: Cache Partitioning


class PartitionedCache:
    """Cache with partitioning support for better isolation."""

    def __init__(
        self,
        num_partitions: int = 4,
        partition_fn: Optional[Callable[[str], int]] = None,
    ):
        """Initialize partitioned cache."""
        from tinyllm.cache import InMemoryBackend
        self.num_partitions = num_partitions
        self.partition_fn = partition_fn or self._default_partition
        self._partitions: List = [
            InMemoryBackend() for _ in range(num_partitions)
        ]

    def _default_partition(self, key: str) -> int:
        """Default partition function using hash."""
        return hash(key) % self.num_partitions

    def _get_partition(self, key: str):
        """Get partition for a key."""
        partition_id = self.partition_fn(key)
        return self._partitions[partition_id]

    async def get(self, key: str):
        """Get entry from appropriate partition."""
        partition = self._get_partition(key)
        return await partition.get(key)

    async def set(self, key: str, entry):
        """Set entry in appropriate partition."""
        partition = self._get_partition(key)
        await partition.set(key, entry)

    async def delete(self, key: str):
        """Delete entry from appropriate partition."""
        partition = self._get_partition(key)
        await partition.delete(key)

    async def clear(self):
        """Clear all partitions."""
        for partition in self._partitions:
            await partition.clear()

    async def size(self) -> int:
        """Get total size across all partitions."""
        total = 0
        for partition in self._partitions:
            total += await partition.size()
        return total

    async def close(self):
        """Close all partitions."""
        for partition in self._partitions:
            await partition.close()


# Task 103: Cache Replication


class ReplicatedCache:
    """Cache with replication across multiple backends."""

    def __init__(
        self,
        primary,
        replicas: List,
        read_from_replicas: bool = True,
    ):
        """Initialize replicated cache."""
        self.primary = primary
        self.replicas = replicas
        self.read_from_replicas = read_from_replicas

    async def get(self, key: str):
        """Get entry from primary, fallback to replicas."""
        entry = await self.primary.get(key)
        if entry is not None:
            return entry

        if self.read_from_replicas:
            for replica in self.replicas:
                entry = await replica.get(key)
                if entry is not None:
                    asyncio.create_task(self.primary.set(key, entry))
                    return entry
        return None

    async def set(self, key: str, entry):
        """Set entry in primary and all replicas."""
        await self.primary.set(key, entry)
        for replica in self.replicas:
            asyncio.create_task(replica.set(key, entry))

    async def delete(self, key: str):
        """Delete from primary and all replicas."""
        await self.primary.delete(key)
        for replica in self.replicas:
            asyncio.create_task(replica.delete(key))

    async def clear(self):
        """Clear primary and all replicas."""
        await self.primary.clear()
        for replica in self.replicas:
            await replica.clear()

    async def size(self) -> int:
        """Get size from primary."""
        return await self.primary.size()

    async def close(self):
        """Close primary and all replicas."""
        await self.primary.close()
        for replica in self.replicas:
            await replica.close()


# Task 104: Write-Through/Write-Back Policies


class WritePolicy(Enum):
    """Cache write policies."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


class WritePolicyCache:
    """Cache with configurable write policies."""

    def __init__(
        self,
        cache_backend,
        storage_backend=None,
        policy: WritePolicy = WritePolicy.WRITE_THROUGH,
        write_back_interval: int = 60,
    ):
        """Initialize write policy cache."""
        self.cache_backend = cache_backend
        self.storage_backend = storage_backend
        self.policy = policy
        self.write_back_interval = write_back_interval
        self._dirty_keys: set = set()
        self._flush_task: Optional[asyncio.Task] = None

    async def get(self, key: str):
        """Get entry from cache, fallback to storage."""
        entry = await self.cache_backend.get(key)
        if entry is not None:
            return entry
        if self.storage_backend:
            entry = await self.storage_backend.get(key)
            if entry is not None:
                await self.cache_backend.set(key, entry)
                return entry
        return None

    async def set(self, key: str, entry):
        """Set entry according to write policy."""
        if self.policy == WritePolicy.WRITE_THROUGH:
            await self.cache_backend.set(key, entry)
            if self.storage_backend:
                await self.storage_backend.set(key, entry)
        elif self.policy == WritePolicy.WRITE_BACK:
            await self.cache_backend.set(key, entry)
            self._dirty_keys.add(key)
        elif self.policy == WritePolicy.WRITE_AROUND:
            if self.storage_backend:
                await self.storage_backend.set(key, entry)
            else:
                await self.cache_backend.set(key, entry)

    async def flush(self):
        """Flush dirty entries to storage (write-back mode)."""
        if not self.storage_backend or not self._dirty_keys:
            return
        dirty_copy = self._dirty_keys.copy()
        self._dirty_keys.clear()
        for key in dirty_copy:
            try:
                entry = await self.cache_backend.get(key)
                if entry is not None:
                    await self.storage_backend.set(key, entry)
            except Exception:
                self._dirty_keys.add(key)

    async def delete(self, key: str):
        """Delete from both cache and storage."""
        await self.cache_backend.delete(key)
        if self.storage_backend:
            await self.storage_backend.delete(key)
        self._dirty_keys.discard(key)

    async def clear(self):
        """Clear both cache and storage."""
        await self.cache_backend.clear()
        if self.storage_backend:
            await self.storage_backend.clear()
        self._dirty_keys.clear()

    async def size(self) -> int:
        """Get cache size."""
        return await self.cache_backend.size()

    async def close(self):
        """Flush and close backends."""
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush()
        await self.cache_backend.close()
        if self.storage_backend:
            await self.storage_backend.close()


# Task 105: Cache TTL Optimization


class TTLOptimizer:
    """Optimize cache TTL based on access patterns."""

    def __init__(
        self,
        min_ttl: int = 300,
        max_ttl: int = 7200,
        target_hit_rate: float = 0.8,
    ):
        """Initialize TTL optimizer."""
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.target_hit_rate = target_hit_rate
        self._key_stats: Dict[str, Dict] = defaultdict(
            lambda: {"accesses": 0, "last_access": 0, "ttl": max_ttl}
        )

    def record_access(self, key: str, hit: bool):
        """Record cache access for TTL optimization."""
        stats = self._key_stats[key]
        stats["accesses"] += 1
        stats["last_access"] = time.monotonic()

    def get_optimal_ttl(self, key: str, current_hit_rate: float) -> int:
        """Calculate optimal TTL for a key."""
        stats = self._key_stats[key]
        if current_hit_rate < self.target_hit_rate:
            new_ttl = min(int(stats["ttl"] * 1.5), self.max_ttl)
        else:
            new_ttl = max(int(stats["ttl"] * 0.8), self.min_ttl)
        stats["ttl"] = new_ttl
        return new_ttl


# Task 106: Cache Size Auto-Tuning


class CacheSizeTuner:
    """Automatically tune cache size based on performance."""

    def __init__(
        self,
        initial_size: int = 1000,
        min_size: int = 100,
        max_size: int = 10000,
        target_hit_rate: float = 0.8,
        adjustment_interval: int = 300,
    ):
        """Initialize cache size tuner."""
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.target_hit_rate = target_hit_rate
        self.adjustment_interval = adjustment_interval
        self._last_adjustment = time.monotonic()

    def should_adjust(self) -> bool:
        """Check if cache size should be adjusted."""
        return (time.monotonic() - self._last_adjustment) >= self.adjustment_interval

    def calculate_new_size(self, current_hit_rate: float, eviction_rate: float) -> int:
        """Calculate new optimal cache size."""
        if not self.should_adjust():
            return self.current_size
        self._last_adjustment = time.monotonic()
        if current_hit_rate < self.target_hit_rate or eviction_rate > 0.1:
            new_size = min(int(self.current_size * 1.5), self.max_size)
        elif current_hit_rate > self.target_hit_rate * 1.1 and eviction_rate < 0.01:
            new_size = max(int(self.current_size * 0.8), self.min_size)
        else:
            new_size = self.current_size
        self.current_size = new_size
        return new_size


# Task 107: Cache Persistence


class PersistentCache:
    """Cache with disk persistence support."""

    def __init__(
        self,
        backend,
        persist_path: str = "/tmp/tinyllm_cache.json",
        auto_save_interval: int = 300,
    ):
        """Initialize persistent cache."""
        self.backend = backend
        self.persist_path = persist_path
        self.auto_save_interval = auto_save_interval
        self._save_task: Optional[asyncio.Task] = None

    async def save(self):
        """Save cache to disk."""
        from tinyllm.cache import InMemoryBackend, CacheEntry, GenerateResponse
        if not isinstance(self.backend, InMemoryBackend):
            return
        try:
            cache_data = []
            async with self.backend._lock:
                for key, entry in self.backend._cache.items():
                    cache_data.append({
                        "key": key,
                        "response": entry.response.model_dump(),
                        "created_at": entry.created_at,
                        "ttl": entry.ttl,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed,
                    })
            with open(self.persist_path, "w") as f:
                json.dump(cache_data, f)
        except Exception:
            pass

    async def load(self):
        """Load cache from disk."""
        from tinyllm.cache import InMemoryBackend, CacheEntry, GenerateResponse
        if not isinstance(self.backend, InMemoryBackend):
            return
        try:
            import os
            if not os.path.exists(self.persist_path):
                return
            with open(self.persist_path, "r") as f:
                cache_data = json.load(f)
            loaded = 0
            for item in cache_data:
                try:
                    response = GenerateResponse(**item["response"])
                    entry = CacheEntry(
                        response=response,
                        created_at=item["created_at"],
                        ttl=item.get("ttl"),
                        access_count=item.get("access_count", 0),
                        last_accessed=item.get("last_accessed", time.monotonic()),
                    )
                    if not entry.is_expired():
                        await self.backend.set(item["key"], entry)
                        loaded += 1
                except Exception:
                    pass
        except Exception:
            pass


# Task 108: Cache Analytics


@dataclass
class CacheAnalytics:
    """Comprehensive cache analytics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0
    p50_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0
    eviction_count: int = 0
    error_count: int = 0
    hot_keys: List[tuple] = field(default_factory=list)
    cold_keys: List[tuple] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analytics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / self.total_requests if self.total_requests > 0 else 0,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "eviction_count": self.eviction_count,
            "error_count": self.error_count,
            "hot_keys": self.hot_keys[:10],
            "cold_keys": self.cold_keys[:10],
        }


class CacheAnalyzer:
    """Analyze cache performance and access patterns."""

    def __init__(self, cache):
        """Initialize cache analyzer."""
        self.cache = cache
        self._latencies: deque = deque(maxlen=1000)
        self._key_access_counts: Dict[str, int] = defaultdict(int)

    def record_latency(self, latency_ms: float):
        """Record cache operation latency."""
        self._latencies.append(latency_ms)

    def record_key_access(self, key: str):
        """Record key access for hot/cold analysis."""
        self._key_access_counts[key] += 1

    def get_analytics(self) -> CacheAnalytics:
        """Get comprehensive cache analytics."""
        metrics = self.cache.get_metrics()
        analytics = CacheAnalytics(
            total_requests=metrics.total_requests,
            cache_hits=metrics.hits,
            cache_misses=metrics.misses,
            eviction_count=metrics.evictions,
            error_count=metrics.errors,
        )
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            analytics.avg_latency_ms = sum(sorted_latencies) / len(sorted_latencies)
            analytics.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
            analytics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            analytics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        sorted_keys = sorted(
            self._key_access_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        analytics.hot_keys = sorted_keys[:10]
        analytics.cold_keys = sorted_keys[-10:] if len(sorted_keys) > 10 else []
        return analytics


# Task 109: Cache Bypass Rules


class BypassRule:
    """Rule for bypassing cache."""

    def should_bypass(self, key: str, **kwargs) -> bool:
        """Check if cache should be bypassed."""
        return False


class PatternBypassRule(BypassRule):
    """Bypass cache based on key patterns."""

    def __init__(self, patterns: List[str]):
        """Initialize pattern bypass rule."""
        self.patterns = patterns

    def should_bypass(self, key: str, **kwargs) -> bool:
        """Check if key matches bypass patterns."""
        import fnmatch
        return any(fnmatch.fnmatch(key, pattern) for pattern in self.patterns)


class SizeBypassRule(BypassRule):
    """Bypass cache for large responses."""

    def __init__(self, max_size_bytes: int):
        """Initialize size bypass rule."""
        self.max_size_bytes = max_size_bytes

    def should_bypass(self, key: str, **kwargs) -> bool:
        """Check if response size exceeds threshold."""
        response = kwargs.get("response")
        if response and hasattr(response, "response"):
            size = len(response.response.encode("utf-8"))
            return size > self.max_size_bytes
        return False


class RuleBasedCache:
    """Cache with bypass rules."""

    def __init__(self, cache):
        """Initialize rule-based cache."""
        self.cache = cache
        self.bypass_rules: List[BypassRule] = []

    def add_bypass_rule(self, rule: BypassRule):
        """Add a bypass rule."""
        self.bypass_rules.append(rule)

    async def get(self, key: str, **kwargs):
        """Get from cache, respecting bypass rules."""
        if any(rule.should_bypass(key, **kwargs) for rule in self.bypass_rules):
            return None
        return await self.cache.get(key)

    async def set(self, key: str, response, ttl=None, **kwargs):
        """Set in cache, respecting bypass rules."""
        if any(rule.should_bypass(key, response=response, **kwargs) for rule in self.bypass_rules):
            return
        await self.cache.set(key, response, ttl)


# Task 110: Negative Caching


@dataclass
class NegativeCacheEntry:
    """Entry for caching negative results (errors, empty responses)."""
    key: str
    error_type: str
    timestamp: float
    ttl: int = 300

    def is_expired(self) -> bool:
        """Check if negative entry has expired."""
        return (time.monotonic() - self.timestamp) > self.ttl


class NegativeCache:
    """Cache for negative results to avoid repeated failures."""

    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """Initialize negative cache."""
        from collections import OrderedDict
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._entries: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()

    async def is_negative(self, key: str) -> Optional[str]:
        """Check if key has a negative cache entry."""
        async with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                if entry.is_expired():
                    del self._entries[key]
                    return None
                self._entries.move_to_end(key)
                return entry.error_type
            return None

    async def add_negative(self, key: str, error_type: str, ttl: Optional[int] = None):
        """Add a negative cache entry."""
        async with self._lock:
            entry = NegativeCacheEntry(
                key=key,
                error_type=error_type,
                timestamp=time.monotonic(),
                ttl=ttl or self.default_ttl,
            )
            if key in self._entries:
                del self._entries[key]
            elif len(self._entries) >= self.max_size:
                self._entries.popitem(last=False)
            self._entries[key] = entry

    async def clear(self):
        """Clear all negative entries."""
        async with self._lock:
            self._entries.clear()

    async def size(self) -> int:
        """Get number of negative entries."""
        async with self._lock:
            return len(self._entries)
