#!/usr/bin/env python
"""Example demonstrating TinyLLM response caching.

This example shows how to use the caching system to reduce latency
and API costs by reusing responses for identical requests.
"""

import asyncio
import time
from tinyllm.models import OllamaClient
from tinyllm.cache import create_cached_client


async def basic_caching_example():
    """Basic example of response caching."""
    print("=" * 60)
    print("Basic Caching Example")
    print("=" * 60)

    # Create base Ollama client
    client = OllamaClient(
        host="http://localhost:11434",
        timeout_ms=30000,
    )

    # Wrap with caching (in-memory backend)
    cached_client = await create_cached_client(
        client=client,
        backend="memory",
        max_size=100,
        default_ttl=3600,  # 1 hour
    )

    # First request - will be a cache miss
    print("\n1. First request (cache miss):")
    start = time.time()
    response1 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is the capital of France?",
        temperature=0.3,
    )
    elapsed1 = time.time() - start
    print(f"   Response: {response1.response}")
    print(f"   Time: {elapsed1:.2f}s")

    # Second identical request - should hit cache
    print("\n2. Second identical request (cache hit):")
    start = time.time()
    response2 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is the capital of France?",
        temperature=0.3,
    )
    elapsed2 = time.time() - start
    print(f"   Response: {response2.response}")
    print(f"   Time: {elapsed2:.2f}s")
    print(f"   Speedup: {elapsed1/elapsed2:.1f}x faster")

    # Show cache metrics
    print("\n3. Cache metrics:")
    stats = cached_client.get_stats()
    cache_stats = stats['cache']
    print(f"   Hits: {cache_stats['hits']}")
    print(f"   Misses: {cache_stats['misses']}")
    print(f"   Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Total Requests: {cache_stats['total_requests']}")

    await cached_client.close()


async def force_refresh_example():
    """Example of bypassing cache with force_refresh."""
    print("\n" + "=" * 60)
    print("Force Refresh Example")
    print("=" * 60)

    client = OllamaClient()
    cached_client = await create_cached_client(
        client=client,
        backend="memory",
    )

    # First request
    print("\n1. Initial request:")
    response1 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is 2 + 2?",
    )
    print(f"   Response: {response1.response[:50]}...")

    # Second request with force_refresh=True
    print("\n2. Request with force_refresh=True:")
    response2 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is 2 + 2?",
        force_refresh=True,  # Bypass cache
    )
    print(f"   Response: {response2.response[:50]}...")
    print("   Note: This bypassed the cache and generated a fresh response")

    # Check metrics
    stats = cached_client.get_stats()
    cache_stats = stats['cache']
    print(f"\n3. Cache metrics:")
    print(f"   Misses: {cache_stats['misses']} (includes force_refresh)")

    await cached_client.close()


async def custom_ttl_example():
    """Example of using custom TTL for different content types."""
    print("\n" + "=" * 60)
    print("Custom TTL Example")
    print("=" * 60)

    client = OllamaClient()
    cached_client = await create_cached_client(
        client=client,
        backend="memory",
        default_ttl=3600,  # 1 hour default
    )

    # Static content - long TTL
    print("\n1. Static content (long TTL):")
    response1 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is the speed of light?",
        cache_ttl=86400,  # 24 hours
    )
    print(f"   TTL: 24 hours (static fact)")
    print(f"   Response: {response1.response[:60]}...")

    # Dynamic content - short TTL
    print("\n2. Dynamic content (short TTL):")
    response2 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What's a random number?",
        cache_ttl=60,  # 1 minute
    )
    print(f"   TTL: 1 minute (dynamic content)")
    print(f"   Response: {response2.response[:60]}...")

    # Default TTL
    print("\n3. Default TTL:")
    response3 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="Explain photosynthesis",
        # Uses default_ttl=3600 (1 hour)
    )
    print(f"   TTL: 1 hour (default)")
    print(f"   Response: {response3.response[:60]}...")

    await cached_client.close()


async def cache_disabled_example():
    """Example of disabling cache."""
    print("\n" + "=" * 60)
    print("Cache Disabled Example")
    print("=" * 60)

    client = OllamaClient()

    # Create cached client with caching disabled
    cached_client = await create_cached_client(
        client=client,
        backend="memory",
        enable_cache=False,  # Disable caching
    )

    print("\n1. Multiple identical requests (cache disabled):")
    for i in range(3):
        response = await cached_client.generate(
            model="qwen2.5:0.5b",
            prompt="What is AI?",
        )
        print(f"   Request {i+1}: Generated fresh response")

    # Check metrics
    stats = cached_client.get_stats()
    print(f"\n2. Cache metrics:")
    print(f"   Enabled: {stats['cache_enabled']}")
    print(f"   Sets: {stats['cache']['sets']} (no caching occurred)")

    await cached_client.close()


async def parameter_sensitivity_example():
    """Example showing how different parameters create different cache keys."""
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Example")
    print("=" * 60)

    client = OllamaClient()
    cached_client = await create_cached_client(
        client=client,
        backend="memory",
    )

    prompt = "Explain machine learning"

    print("\n1. Same prompt, different temperatures:")

    # Temperature 0.3
    response1 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt=prompt,
        temperature=0.3,
    )
    print(f"   T=0.3: {response1.response[:50]}...")

    # Temperature 0.7 (different cache key)
    response2 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt=prompt,
        temperature=0.7,
    )
    print(f"   T=0.7: {response2.response[:50]}...")

    # Temperature 0.3 again (cache hit)
    response3 = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt=prompt,
        temperature=0.3,
    )
    print(f"   T=0.3 (again): {response3.response[:50]}...")

    # Check metrics
    stats = cached_client.get_stats()
    cache_stats = stats['cache']
    print(f"\n2. Cache metrics:")
    print(f"   Hits: {cache_stats['hits']} (third request)")
    print(f"   Misses: {cache_stats['misses']} (first two requests)")
    print("   Note: Different temperatures use different cache keys")

    await cached_client.close()


async def lru_eviction_example():
    """Example demonstrating LRU eviction."""
    print("\n" + "=" * 60)
    print("LRU Eviction Example")
    print("=" * 60)

    client = OllamaClient()

    # Small cache to demonstrate eviction
    cached_client = await create_cached_client(
        client=client,
        backend="memory",
        max_size=3,  # Only 3 entries
    )

    print("\n1. Filling cache (max_size=3):")

    # Fill cache
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    for prompt in prompts:
        await cached_client.generate(
            model="qwen2.5:0.5b",
            prompt=prompt,
        )
        print(f"   Cached: {prompt}")

    cache_size = await cached_client.cache.size()
    print(f"   Cache size: {cache_size}/3")

    # Add one more (will evict oldest)
    print("\n2. Adding 4th entry (triggers eviction):")
    await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is Go?",
    )
    print(f"   Cached: What is Go?")

    cache_size = await cached_client.cache.size()
    print(f"   Cache size: {cache_size}/3")

    # Try to access first entry (should be evicted)
    print("\n3. Checking if first entry was evicted:")
    start = time.time()
    response = await cached_client.generate(
        model="qwen2.5:0.5b",
        prompt="What is Python?",  # First entry
    )
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    if elapsed > 0.1:  # Took time = cache miss
        print("   ✓ Entry was evicted (cache miss)")
    else:
        print("   ✗ Entry still in cache (unexpected)")

    # Check eviction metrics
    stats = cached_client.get_stats()
    print(f"\n4. Eviction metrics:")
    print(f"   Evictions: {stats['cache']['evictions']}")

    await cached_client.close()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TinyLLM Response Caching Examples")
    print("=" * 60)
    print("\nNote: These examples require Ollama running locally")
    print("with the qwen2.5:0.5b model installed.\n")

    try:
        # Run examples
        await basic_caching_example()
        await force_refresh_example()
        await custom_ttl_example()
        await cache_disabled_example()
        await parameter_sensitivity_example()
        await lru_eviction_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ollama is running and qwen2.5:0.5b is installed:")
        print("  ollama pull qwen2.5:0.5b")


if __name__ == "__main__":
    asyncio.run(main())
