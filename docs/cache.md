# Response Caching for TinyLLM

This document describes the response caching system for TinyLLM, which provides transparent caching of LLM responses with support for multiple backends, TTL management, and comprehensive metrics.

## Overview

The caching system reduces latency and API costs by storing and reusing LLM responses for identical requests. It supports both in-memory (LRU) and Redis-backed caching with configurable TTL and automatic eviction.

## Features

- **Multiple Backends**: In-memory (LRU) and Redis support
- **TTL Support**: Configurable time-to-live for cache entries
- **LRU Eviction**: Automatic eviction of least recently used entries when cache is full
- **Cache Key Generation**: SHA256-based deterministic cache keys from request parameters
- **Metrics Tracking**: Comprehensive hit/miss/eviction metrics
- **Thread-Safe**: Async-safe implementation with proper locking
- **Transparent Integration**: Wraps OllamaClient with minimal code changes

## Architecture

```
┌─────────────────────┐
│ CachedOllamaClient  │
└──────────┬──────────┘
           │
           ├──> ResponseCache
           │    └──> CacheBackend (Interface)
           │         ├──> InMemoryBackend (LRU)
           │         └──> RedisBackend
           │
           └──> OllamaClient (wrapped)
```

## Usage

### Basic In-Memory Caching

```python
from tinyllm.models import OllamaClient
from tinyllm.cache import create_cached_client

# Create client with caching
client = OllamaClient()
cached_client = await create_cached_client(
    client=client,
    backend="memory",
    max_size=1000,      # Maximum 1000 entries
    default_ttl=3600,   # 1 hour TTL
)

# Use as normal - responses are automatically cached
response = await cached_client.generate(
    model="qwen2.5:0.5b",
    prompt="What is the capital of France?",
)

# Subsequent identical requests hit the cache
response2 = await cached_client.generate(
    model="qwen2.5:0.5b",
    prompt="What is the capital of France?",
)  # Returns cached response
```

### Redis Backend

```python
from tinyllm.cache import create_cached_client

cached_client = await create_cached_client(
    client=client,
    backend="redis",
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    default_ttl=3600,
)
```

### Manual Cache Control

```python
# Force refresh (bypass cache)
response = await cached_client.generate(
    model="qwen2.5:0.5b",
    prompt="What is 2+2?",
    force_refresh=True,  # Always generates fresh response
)

# Custom TTL for specific request
response = await cached_client.generate(
    model="qwen2.5:0.5b",
    prompt="What is the weather?",
    cache_ttl=300,  # Cache for 5 minutes only
)
```

### Cache Metrics

```python
# Get statistics
stats = cached_client.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
print(f"Total hits: {stats['cache']['hits']}")
print(f"Total misses: {stats['cache']['misses']}")
```

## CLI Usage

### Enable Caching for Run Command

```bash
# Enable caching (default: in-memory, 1 hour TTL)
tinyllm run "What is AI?" --cache

# Disable caching
tinyllm run "What is AI?" --no-cache

# Use Redis backend
tinyllm run "What is AI?" --cache --cache-backend redis

# Custom TTL (in seconds)
tinyllm run "What is AI?" --cache --cache-ttl 1800  # 30 minutes
```

### Cache Statistics

```bash
# Show cache statistics
tinyllm cache-stats

# Show Redis cache statistics
tinyllm cache-stats --backend redis --redis-host localhost
```

### Clear Cache

```bash
# Clear in-memory cache
tinyllm cache-clear --backend memory

# Clear Redis cache
tinyllm cache-clear --backend redis --redis-host localhost

# Skip confirmation
tinyllm cache-clear --no-confirm
```

## Cache Key Generation

Cache keys are generated deterministically from request parameters:

```python
cache_key = SHA256({
    "model": "qwen2.5:0.5b",
    "prompt": "What is AI?",
    "temperature": 0.3,
    "max_tokens": 2000,
    "system": None,
    "json_mode": False,
})
```

Any change to these parameters results in a different cache key, ensuring correct cache behavior.

## Cache Backends

### InMemoryBackend

**Pros:**
- Fast access (no network overhead)
- Simple setup (no external dependencies)
- LRU eviction automatically manages memory

**Cons:**
- Not shared across processes
- Lost on restart
- Limited by available memory

**Best for:**
- Development
- Single-process applications
- Small to medium cache sizes

### RedisBackend

**Pros:**
- Shared across processes/instances
- Persists across restarts (with Redis persistence)
- Scalable to large datasets
- Can use Redis eviction policies

**Cons:**
- Network latency overhead
- Requires Redis server
- More complex setup

**Best for:**
- Production deployments
- Multi-process/distributed systems
- Large cache sizes
- Persistent caching needs

## Configuration

### In-Memory Backend

```python
from tinyllm.cache import create_memory_cache

cache = create_memory_cache(
    max_size=1000,         # Maximum entries (LRU eviction after)
    default_ttl=3600,      # Default TTL in seconds (None = no expiration)
    enable_metrics=True,   # Track cache statistics
)
```

### Redis Backend

```python
from tinyllm.cache import create_redis_cache

cache = create_redis_cache(
    host="localhost",
    port=6379,
    db=0,
    password=None,          # Optional Redis password
    default_ttl=3600,
    enable_metrics=True,
    key_prefix="tinyllm:cache:",  # Redis key prefix
)
```

## Performance Considerations

### Cache Hit Rate

A good cache hit rate depends on your use case:
- **High repetition** (FAQ, common queries): 60-90% hit rate
- **Medium repetition** (similar queries): 30-60% hit rate
- **Low repetition** (unique queries): <30% hit rate

### Memory Usage (In-Memory Backend)

Approximate memory per cached response:
- Small response (100 tokens): ~1-2 KB
- Medium response (500 tokens): ~5-10 KB
- Large response (2000 tokens): ~20-40 KB

For 1000 cached responses:
- Average case: ~10-20 MB
- Worst case: ~40 MB

### TTL Selection

Choose TTL based on data freshness requirements:
- **Static content** (definitions, facts): 3600-86400s (1-24 hours)
- **Semi-static** (common knowledge): 1800-3600s (30-60 min)
- **Dynamic** (weather, news): 300-900s (5-15 min)
- **Highly dynamic** (real-time data): Don't cache or use very short TTL

## Monitoring

### Cache Metrics

The cache tracks the following metrics:

```python
metrics = cache.get_metrics()

# Metrics available:
# - hits: Number of cache hits
# - misses: Number of cache misses
# - sets: Number of cache sets
# - evictions: Number of entries evicted (in-memory only)
# - errors: Number of cache errors
# - hit_rate: Cache hit rate (0.0-1.0)
# - total_requests: Total cache requests
```

### Logging

The cache logs important events:

```python
# Cache operations (DEBUG level)
logger.debug("cache_hit", key=key[:16], access_count=3)
logger.debug("cache_miss", key=key[:16])
logger.debug("cache_set", key=key[:16], ttl=3600)

# Cache evictions (DEBUG level)
logger.debug("cache_eviction", key=key, total_evictions=10)

# Cache errors (ERROR level)
logger.error("cache_get_error", key=key[:16], error=str(e))
```

## Best Practices

1. **Choose the Right Backend**
   - Use in-memory for development and testing
   - Use Redis for production with multiple instances

2. **Set Appropriate TTL**
   - Longer TTL for static content
   - Shorter TTL for dynamic content
   - Use `force_refresh` for critical updates

3. **Monitor Hit Rate**
   - Track cache metrics regularly
   - Optimize cache size based on hit rate
   - Adjust TTL if hit rate is too low

4. **Handle Cache Failures Gracefully**
   - Cache errors don't break functionality
   - Requests fall through to LLM on cache errors
   - Monitor error metrics for backend issues

5. **Consider Cache Warming**
   - Pre-populate cache with common queries
   - Reduces cold start latency
   - Improves user experience

## Example: Production Setup

```python
import asyncio
from tinyllm.models import OllamaClient
from tinyllm.cache import create_cached_client

async def setup_production_client():
    """Setup production client with Redis caching."""

    # Create base client
    client = OllamaClient(
        host="http://ollama:11434",
        timeout_ms=30000,
        max_retries=3,
    )

    # Wrap with caching
    cached_client = await create_cached_client(
        client=client,
        backend="redis",
        redis_host="redis",
        redis_port=6379,
        redis_db=0,
        default_ttl=3600,  # 1 hour default
        enable_cache=True,
    )

    return cached_client

async def main():
    client = await setup_production_client()

    # Use cached client
    response = await client.generate(
        model="qwen2.5:0.5b",
        prompt="Explain caching",
    )

    # Check stats
    stats = client.get_stats()
    print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Cache Not Working

1. **Verify cache is enabled:**
   ```python
   assert cached_client.enable_cache == True
   ```

2. **Check cache key generation:**
   ```python
   key = cache.generate_cache_key(model="test", prompt="hello")
   print(f"Cache key: {key}")
   ```

3. **Monitor metrics:**
   ```python
   metrics = cache.get_metrics()
   print(f"Hits: {metrics.hits}, Misses: {metrics.misses}")
   ```

### High Miss Rate

- **Different request parameters**: Even small changes create new keys
- **Short TTL**: Entries expiring too quickly
- **Cache too small**: LRU eviction removing entries
- **Unique queries**: Each query is different

### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
docker logs redis

# Verify Redis is accessible
telnet redis 6379
```

### Memory Usage Too High (In-Memory)

- Reduce `max_size`
- Decrease `default_ttl`
- Switch to Redis backend
- Monitor eviction metrics

## API Reference

See [src/tinyllm/cache.py](/home/uri/Desktop/tinyllm/src/tinyllm/cache.py) for complete API documentation.

### Key Classes

- **ResponseCache**: Main cache interface
- **CachedOllamaClient**: Cached wrapper for OllamaClient
- **InMemoryBackend**: LRU in-memory cache backend
- **RedisBackend**: Redis-backed cache backend
- **CacheMetrics**: Cache performance metrics
- **CacheEntry**: Individual cache entry with metadata

### Key Functions

- **create_memory_cache()**: Create in-memory cache
- **create_redis_cache()**: Create Redis cache
- **create_cached_client()**: Create cached Ollama client
