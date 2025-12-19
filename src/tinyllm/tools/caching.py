"""Tool caching for TinyLLM.

This module provides caching capabilities for tool results.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry."""

    key: str
    value: Any
    created_at: float
    ttl: float
    hit_count: int = 0

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: float) -> None:
        """Set entry in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""

    def __init__(self, max_size: int = 1000):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of entries.
        """
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size = len(self._cache)
                return None

            entry.hit_count += 1
            self._stats.hits += 1
            return entry

    async def set(self, key: str, value: Any, ttl: float) -> None:
        """Set entry in cache."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_one()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=ttl,
            )
            self._stats.size = len(self._cache)

    async def _evict_one(self) -> None:
        """Evict one entry (LRU-ish: oldest first)."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats.evictions += 1

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=len(self._cache),
        )


class ToolCache:
    """Cache for tool results."""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: float = 300.0,  # 5 minutes
        key_generator: Optional[Callable[[str, Any], str]] = None,
    ):
        """Initialize tool cache.

        Args:
            backend: Cache backend to use.
            default_ttl: Default TTL in seconds.
            key_generator: Custom key generator function.
        """
        self.backend = backend or MemoryCache()
        self.default_ttl = default_ttl
        self.key_generator = key_generator or self._default_key_generator

    def _default_key_generator(self, tool_id: str, input: Any) -> str:
        """Generate cache key from tool ID and input."""
        if isinstance(input, BaseModel):
            input_str = input.model_dump_json()
        elif isinstance(input, dict):
            input_str = json.dumps(input, sort_keys=True, default=str)
        else:
            input_str = str(input)

        key_data = f"{tool_id}:{input_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def get(self, tool_id: str, input: Any) -> Optional[Any]:
        """Get cached result.

        Args:
            tool_id: Tool identifier.
            input: Tool input.

        Returns:
            Cached result or None.
        """
        key = self.key_generator(tool_id, input)
        entry = await self.backend.get(key)
        return entry.value if entry else None

    async def set(
        self,
        tool_id: str,
        input: Any,
        result: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a result.

        Args:
            tool_id: Tool identifier.
            input: Tool input.
            result: Tool result.
            ttl: Optional TTL override.
        """
        key = self.key_generator(tool_id, input)
        await self.backend.set(key, result, ttl or self.default_ttl)

    async def invalidate(self, tool_id: str, input: Any) -> bool:
        """Invalidate cached result.

        Args:
            tool_id: Tool identifier.
            input: Tool input.

        Returns:
            True if entry was invalidated.
        """
        key = self.key_generator(tool_id, input)
        return await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear all cached results."""
        await self.backend.clear()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.backend.stats()


class CachedToolWrapper:
    """Wrapper that adds caching to tool execution."""

    def __init__(
        self,
        tool: Any,
        cache: Optional[ToolCache] = None,
        ttl: Optional[float] = None,
        cache_condition: Optional[Callable[[Any], bool]] = None,
    ):
        """Initialize cached tool wrapper.

        Args:
            tool: Tool to wrap.
            cache: Cache instance.
            ttl: Cache TTL.
            cache_condition: Condition for caching result.
        """
        self.tool = tool
        self.cache = cache or ToolCache()
        self.ttl = ttl
        self.cache_condition = cache_condition or (lambda x: True)

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input: Any) -> Any:
        """Execute with caching.

        Args:
            input: Tool input.

        Returns:
            Tool output (possibly cached).
        """
        tool_id = self.tool.metadata.id

        # Check cache
        cached = await self.cache.get(tool_id, input)
        if cached is not None:
            logger.debug(f"Cache hit for {tool_id}")
            return cached

        # Execute tool
        result = await self.tool.execute(input)

        # Cache if condition met
        if self.cache_condition(result):
            await self.cache.set(tool_id, input, result, self.ttl)

        return result

    async def invalidate(self, input: Any) -> bool:
        """Invalidate cached result for input.

        Args:
            input: Tool input.

        Returns:
            True if invalidated.
        """
        return await self.cache.invalidate(self.tool.metadata.id, input)


class CacheDecorator:
    """Decorator for adding caching to tools."""

    def __init__(
        self,
        ttl: float = 300.0,
        cache: Optional[ToolCache] = None,
        condition: Optional[Callable[[Any], bool]] = None,
    ):
        """Initialize decorator.

        Args:
            ttl: Cache TTL.
            cache: Shared cache instance.
            condition: Caching condition.
        """
        self.ttl = ttl
        self.cache = cache or ToolCache()
        self.condition = condition

    def __call__(self, tool: Any) -> CachedToolWrapper:
        """Apply caching to tool.

        Args:
            tool: Tool to wrap.

        Returns:
            Cached tool wrapper.
        """
        return CachedToolWrapper(
            tool=tool,
            cache=self.cache,
            ttl=self.ttl,
            cache_condition=self.condition,
        )


def with_cache(
    tool: Any,
    ttl: float = 300.0,
    cache: Optional[ToolCache] = None,
) -> CachedToolWrapper:
    """Add caching to a tool.

    Args:
        tool: Tool to wrap.
        ttl: Cache TTL.
        cache: Optional shared cache.

    Returns:
        Cached tool wrapper.
    """
    return CachedToolWrapper(tool=tool, cache=cache, ttl=ttl)


def cached(
    ttl: float = 300.0,
    cache: Optional[ToolCache] = None,
) -> CacheDecorator:
    """Decorator factory for caching.

    Args:
        ttl: Cache TTL.
        cache: Optional shared cache.

    Returns:
        Cache decorator.
    """
    return CacheDecorator(ttl=ttl, cache=cache)
