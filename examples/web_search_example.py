#!/usr/bin/env python3
"""Example usage of WebSearchTool.

This demonstrates how to use the web search tool with different configurations
and backends.
"""

import asyncio
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinyllm.tools.web_search import (
    WebSearchConfig,
    WebSearchInput,
    WebSearchTool,
)


async def basic_search_example():
    """Basic search example with DuckDuckGo (no API key needed)."""
    print("=" * 60)
    print("Example 1: Basic Search with DuckDuckGo")
    print("=" * 60)

    # Create tool with default config (DuckDuckGo enabled by default)
    config = WebSearchConfig(
        enable_duckduckgo=True,
        enable_searxng=False,
        enable_brave=False,
        max_results=5,
    )
    tool = WebSearchTool(config)

    # Perform search
    input = WebSearchInput(query="Python async programming")

    print(f"\nSearching for: {input.query}")
    result = await tool.execute(input)

    if result.success:
        print(f"\nFound {result.total_results} results using {result.provider_used}")
        print(f"Cached: {result.cached}\n")

        for i, res in enumerate(result.results, 1):
            print(f"{i}. {res.title}")
            print(f"   URL: {res.url}")
            print(f"   Snippet: {res.snippet[:100]}...")
            print(f"   Score: {res.score:.2f}\n")
    else:
        print(f"Search failed: {result.error}")

    await tool.cleanup()


async def searxng_example():
    """Example using SearXNG (requires instance URL)."""
    print("=" * 60)
    print("Example 2: Search with SearXNG")
    print("=" * 60)

    # Note: You need to set SEARXNG_URL environment variable or provide it here
    searxng_url = os.getenv("SEARXNG_URL")

    if not searxng_url:
        print("\nSkipping: Set SEARXNG_URL environment variable to use SearXNG")
        print("Example: export SEARXNG_URL=https://search.example.com")
        return

    config = WebSearchConfig(
        searxng_url=searxng_url,
        enable_searxng=True,
        enable_duckduckgo=False,
        max_results=10,
    )
    tool = WebSearchTool(config)

    input = WebSearchInput(
        query="machine learning tutorial",
        language="en",
        time_range="month",  # Only results from last month
    )

    print(f"\nSearching for: {input.query}")
    print(f"Time range: {input.time_range}")

    result = await tool.execute(input)

    if result.success:
        print(f"\nFound {result.total_results} results\n")
        for i, res in enumerate(result.results[:5], 1):
            print(f"{i}. {res.title}")
            print(f"   {res.url}\n")
    else:
        print(f"Search failed: {result.error}")

    await tool.cleanup()


async def brave_search_example():
    """Example using Brave Search API (requires API key)."""
    print("=" * 60)
    print("Example 3: Search with Brave Search API")
    print("=" * 60)

    # Note: You need to set BRAVE_API_KEY environment variable
    brave_api_key = os.getenv("BRAVE_API_KEY")

    if not brave_api_key:
        print("\nSkipping: Set BRAVE_API_KEY environment variable to use Brave")
        print("Get API key from: https://brave.com/search/api/")
        return

    config = WebSearchConfig(
        brave_api_key=brave_api_key,
        enable_brave=True,
        enable_searxng=False,
        enable_duckduckgo=False,
        max_results=10,
    )
    tool = WebSearchTool(config)

    input = WebSearchInput(query="climate change research 2024")

    print(f"\nSearching for: {input.query}")
    result = await tool.execute(input)

    if result.success:
        print(f"\nFound {result.total_results} results\n")
        for i, res in enumerate(result.results[:5], 1):
            print(f"{i}. {res.title}")
            print(f"   {res.url}")
            if res.published_date:
                print(f"   Published: {res.published_date}")
            print()
    else:
        print(f"Search failed: {result.error}")

    await tool.cleanup()


async def multi_backend_fallback_example():
    """Example with multiple backends and fallback."""
    print("=" * 60)
    print("Example 4: Multi-Backend with Fallback")
    print("=" * 60)

    # Configure multiple backends - will try in order
    config = WebSearchConfig(
        searxng_url=os.getenv("SEARXNG_URL"),
        brave_api_key=os.getenv("BRAVE_API_KEY"),
        enable_searxng=True,
        enable_duckduckgo=True,
        enable_brave=True,
        fallback_enabled=True,  # Fall back to next provider on error
        max_results=5,
    )
    tool = WebSearchTool(config)

    input = WebSearchInput(query="artificial intelligence")

    print(f"\nSearching for: {input.query}")
    print("Configured backends: SearXNG, DuckDuckGo, Brave")
    print("Will try each until one succeeds\n")

    result = await tool.execute(input)

    if result.success:
        print(f"Success with provider: {result.provider_used}")
        print(f"Found {result.total_results} results\n")
        for i, res in enumerate(result.results, 1):
            print(f"{i}. {res.title}")
            print(f"   Source: {res.source}\n")
    else:
        print(f"All backends failed: {result.error}")

    await tool.cleanup()


async def caching_example():
    """Example demonstrating result caching."""
    print("=" * 60)
    print("Example 5: Result Caching")
    print("=" * 60)

    config = WebSearchConfig(
        enable_duckduckgo=True,
        cache_ttl_seconds=3600,  # Cache for 1 hour
        max_results=5,
    )
    tool = WebSearchTool(config)

    input = WebSearchInput(query="Python web scraping")

    # First search
    print("\nFirst search (will hit API)...")
    result1 = await tool.execute(input)
    print(f"Cached: {result1.cached}")
    print(f"Found {result1.total_results} results")

    # Second search with same query
    print("\nSecond search (should use cache)...")
    result2 = await tool.execute(input)
    print(f"Cached: {result2.cached}")
    print(f"Found {result2.total_results} results")

    await tool.cleanup()


async def pagination_example():
    """Example showing pagination."""
    print("=" * 60)
    print("Example 6: Pagination")
    print("=" * 60)

    config = WebSearchConfig(
        enable_duckduckgo=True,
        max_results=3,
    )
    tool = WebSearchTool(config)

    query = "neural networks"

    # Get first page
    print(f"\nSearching for: {query}")
    print("\nPage 1:")
    result1 = await tool.execute(WebSearchInput(query=query, page=1))
    if result1.success:
        for i, res in enumerate(result1.results, 1):
            print(f"  {i}. {res.title}")

    # Get second page
    print("\nPage 2:")
    result2 = await tool.execute(WebSearchInput(query=query, page=2))
    if result2.success:
        for i, res in enumerate(result2.results, 1):
            print(f"  {i}. {res.title}")

    await tool.cleanup()


async def content_fetching_example():
    """Example with page content fetching."""
    print("=" * 60)
    print("Example 7: Fetching Page Content")
    print("=" * 60)

    config = WebSearchConfig(
        enable_duckduckgo=True,
        fetch_page_content=True,  # Fetch content from top results
        max_results=3,
    )
    tool = WebSearchTool(config)

    input = WebSearchInput(query="Python documentation")

    print(f"\nSearching for: {input.query}")
    print("Will also fetch page content from top results...\n")

    result = await tool.execute(input)

    if result.success:
        print(f"Found {result.total_results} results\n")

        if result.page_contents:
            print(f"Fetched content from {len(result.page_contents)} pages:\n")
            for i, content in enumerate(result.page_contents, 1):
                print(f"{i}. {content.title}")
                print(f"   URL: {content.url}")
                print(f"   Word count: {content.word_count}")
                print(f"   Excerpt: {content.excerpt[:100]}...")
                print(f"   Fetch time: {content.fetch_time:.2f}s\n")
        else:
            print("No page content was fetched")
    else:
        print(f"Search failed: {result.error}")

    await tool.cleanup()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("WebSearchTool Examples")
    print("=" * 60 + "\n")

    # Run examples
    await basic_search_example()
    print("\n")

    await searxng_example()
    print("\n")

    await brave_search_example()
    print("\n")

    await multi_backend_fallback_example()
    print("\n")

    await caching_example()
    print("\n")

    await pagination_example()
    print("\n")

    await content_fetching_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
