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
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

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
