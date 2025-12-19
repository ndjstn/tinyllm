"""Example usage of the WebScraperTool.

This script demonstrates how to use the web scraper tool with different modes
and configurations.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tinyllm.tools.web_scraper import (
    WebScraperTool,
    ScraperInput,
    ScraperConfig,
    ScrapeMode,
    OutputFormat,
)


async def example_basic_scraping():
    """Example: Basic page scraping."""
    print("\n=== Example: Basic Page Scraping ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://example.com",
        mode=ScrapeMode.FULL_PAGE,
        output_format=OutputFormat.PLAIN_TEXT,
    )

    result = await scraper.execute(input)

    if result.success:
        print(f"Title: {result.content.title}")
        print(f"Word count: {result.content.word_count}")
        print(f"Content preview: {result.content.content[:200]}...")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_css_selector():
    """Example: Scraping with CSS selector."""
    print("\n=== Example: CSS Selector Scraping ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://news.ycombinator.com",
        mode=ScrapeMode.CSS_SELECTOR,
        selector=".title",
        output_format=OutputFormat.PLAIN_TEXT,
    )

    result = await scraper.execute(input)

    if result.success:
        print(f"Selected content: {result.content.content[:500]}...")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_extract_links():
    """Example: Extract all links from a page."""
    print("\n=== Example: Extract Links ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://example.com",
        mode=ScrapeMode.LINKS,
    )

    result = await scraper.execute(input)

    if result.success:
        print(f"Found {len(result.content.links)} links:")
        for link in result.content.links[:5]:  # Show first 5
            print(f"  - {link.text}: {link.url}")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_extract_images():
    """Example: Extract all images from a page."""
    print("\n=== Example: Extract Images ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://example.com",
        mode=ScrapeMode.IMAGES,
    )

    result = await scraper.execute(input)

    if result.success:
        print(f"Found {len(result.content.images)} images:")
        for image in result.content.images[:5]:  # Show first 5
            print(f"  - {image.alt or 'No alt text'}: {image.url}")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_extract_tables():
    """Example: Extract all tables from a page."""
    print("\n=== Example: Extract Tables ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://en.wikipedia.org/wiki/List_of_countries_by_population",
        mode=ScrapeMode.TABLES,
    )

    result = await scraper.execute(input)

    if result.success:
        print(f"Found {len(result.content.tables)} tables:")
        for i, table in enumerate(result.content.tables[:3], 1):  # Show first 3
            print(f"\nTable {i}:")
            if table.caption:
                print(f"  Caption: {table.caption}")
            print(f"  Headers: {table.headers}")
            print(f"  Rows: {len(table.rows)}")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_markdown_output():
    """Example: Get content in Markdown format."""
    print("\n=== Example: Markdown Output ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://example.com",
        mode=ScrapeMode.FULL_PAGE,
        output_format=OutputFormat.MARKDOWN,
    )

    result = await scraper.execute(input)

    if result.success:
        print("Markdown content:")
        print(result.content.content[:500])
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_with_extraction_options():
    """Example: Scrape page and extract links, images, and tables."""
    print("\n=== Example: Full Extraction ===")

    scraper = WebScraperTool()
    input = ScraperInput(
        url="https://example.com",
        mode=ScrapeMode.FULL_PAGE,
        output_format=OutputFormat.PLAIN_TEXT,
        extract_links=True,
        extract_images=True,
        extract_tables=True,
    )

    result = await scraper.execute(input)

    if result.success:
        print(f"Title: {result.content.title}")
        print(f"Content length: {len(result.content.content)} chars")
        print(f"Links found: {len(result.content.links)}")
        print(f"Images found: {len(result.content.images)}")
        print(f"Tables found: {len(result.content.tables)}")
        print(f"Fetch time: {result.content.fetch_time:.2f}s")
        print(f"Cached: {result.cached}")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_custom_config():
    """Example: Use custom configuration."""
    print("\n=== Example: Custom Configuration ===")

    config = ScraperConfig(
        timeout_ms=10000,
        max_retries=5,
        rate_limit_delay_ms=2000,
        cache_ttl_seconds=3600,
        respect_robots_txt=True,
        user_agent="MyBot/1.0",
        max_content_length=500_000,
    )

    scraper = WebScraperTool(config=config)
    input = ScraperInput(url="https://example.com")

    result = await scraper.execute(input)

    if result.success:
        print(f"Scraped successfully with custom config")
        print(f"Title: {result.content.title}")
    else:
        print(f"Error: {result.error}")

    await scraper.cleanup()


async def example_caching():
    """Example: Demonstrate caching."""
    print("\n=== Example: Caching ===")

    scraper = WebScraperTool()
    input = ScraperInput(url="https://example.com")

    # First request
    print("First request (not cached):")
    result1 = await scraper.execute(input)
    print(f"  Success: {result1.success}, Cached: {result1.cached}")

    # Second request (should be cached)
    print("Second request (should be cached):")
    result2 = await scraper.execute(input)
    print(f"  Success: {result2.success}, Cached: {result2.cached}")

    await scraper.cleanup()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Web Scraper Tool Examples")
    print("=" * 60)

    # Note: Most examples will fail without actual network access
    # These are demonstration examples showing the API

    try:
        # Basic examples that work with any URL
        print("\nNote: These examples demonstrate the API.")
        print("Some may fail depending on network access and target sites.\n")

        # Uncomment examples to run them:
        # await example_basic_scraping()
        # await example_css_selector()
        # await example_extract_links()
        # await example_extract_images()
        # await example_extract_tables()
        # await example_markdown_output()
        # await example_with_extraction_options()
        # await example_custom_config()
        # await example_caching()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
