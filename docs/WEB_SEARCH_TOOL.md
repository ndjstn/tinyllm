# Web Search Tool Documentation

The WebSearchTool provides a unified interface for searching the web using multiple backend providers. It includes features like result ranking, deduplication, caching, rate limiting, and optional page content fetching.

## Features

- **Multiple Search Backends**:
  - SearXNG (self-hosted, privacy-focused)
  - DuckDuckGo (no API key required)
  - Brave Search API (optional, requires API key)

- **Automatic Fallback**: If one provider fails, automatically tries the next available provider

- **Result Processing**:
  - Deduplication of URLs
  - Score-based ranking
  - Configurable result limits

- **Performance Optimizations**:
  - Built-in caching with TTL
  - Rate limiting
  - Async/await throughout

- **Page Content Fetching**: Optional extraction of page content with readability processing

## Installation

The web search tool is included with TinyLLM. Ensure you have the required dependencies:

```bash
pip install tinyllm
```

The tool uses `httpx` which is already included in TinyLLM's dependencies.

## Quick Start

### Basic Usage (DuckDuckGo)

```python
import asyncio
from tinyllm.tools import WebSearchTool, WebSearchInput, WebSearchConfig

async def main():
    # Create tool with DuckDuckGo (no API key needed)
    config = WebSearchConfig(
        enable_duckduckgo=True,
        max_results=10,
    )
    tool = WebSearchTool(config)

    # Search
    input = WebSearchInput(query="Python async programming")
    result = await tool.execute(input)

    if result.success:
        for res in result.results:
            print(f"{res.title}")
            print(f"  {res.url}")
            print(f"  {res.snippet}\n")

    await tool.cleanup()

asyncio.run(main())
```

### Using SearXNG

```python
from tinyllm.tools import WebSearchTool, WebSearchConfig

config = WebSearchConfig(
    searxng_url="https://your-searxng-instance.com",
    enable_searxng=True,
    enable_duckduckgo=False,
)
tool = WebSearchTool(config)
```

You can also set the `SEARXNG_URL` environment variable:

```bash
export SEARXNG_URL=https://your-searxng-instance.com
```

### Using Brave Search API

```python
from tinyllm.tools import WebSearchTool, WebSearchConfig

config = WebSearchConfig(
    brave_api_key="your-brave-api-key",
    enable_brave=True,
    enable_duckduckgo=False,
)
tool = WebSearchTool(config)
```

Or use the `BRAVE_API_KEY` environment variable:

```bash
export BRAVE_API_KEY=your-brave-api-key
```

Get your Brave API key at: https://brave.com/search/api/

## Configuration

### WebSearchConfig

```python
from tinyllm.tools import WebSearchConfig

config = WebSearchConfig(
    # Result limits
    max_results=10,              # Maximum results to return
    timeout_ms=30000,            # Request timeout in milliseconds

    # Caching
    cache_ttl_seconds=3600,      # Cache TTL (0 to disable)

    # Rate limiting
    rate_limit_per_minute=20,    # Max requests per minute

    # Processing
    deduplicate_results=True,    # Remove duplicate URLs
    fetch_page_content=False,    # Fetch and extract page content
    fallback_enabled=True,       # Fall back to next provider on error

    # Provider configuration
    searxng_url=None,            # SearXNG instance URL
    brave_api_key=None,          # Brave Search API key

    # Provider toggles
    enable_searxng=True,
    enable_duckduckgo=True,
    enable_brave=False,
)
```

### WebSearchInput

```python
from tinyllm.tools import WebSearchInput

input = WebSearchInput(
    query="search query",        # Required: search query
    max_results=None,            # Override config max_results
    page=1,                      # Page number for pagination
    language="en",               # Language code
    time_range=None,             # Filter: "day", "week", "month", "year"
)
```

## Advanced Features

### Multi-Backend with Fallback

Configure multiple providers - the tool will try them in order until one succeeds:

```python
config = WebSearchConfig(
    searxng_url="https://search.example.com",
    brave_api_key="your-key",
    enable_searxng=True,
    enable_duckduckgo=True,
    enable_brave=True,
    fallback_enabled=True,
)
```

### Result Caching

Results are automatically cached based on query, page, and other parameters:

```python
config = WebSearchConfig(
    cache_ttl_seconds=3600,  # Cache for 1 hour
)

# First call hits API
result1 = await tool.execute(input)
assert result1.cached == False

# Second call uses cache
result2 = await tool.execute(input)
assert result2.cached == True
```

### Rate Limiting

Built-in token bucket rate limiter prevents overwhelming APIs:

```python
config = WebSearchConfig(
    rate_limit_per_minute=20,  # Max 20 requests per minute
)
```

### Page Content Fetching

Optionally fetch and extract content from result URLs:

```python
config = WebSearchConfig(
    fetch_page_content=True,  # Fetch content from top 3 results
)

result = await tool.execute(input)

if result.page_contents:
    for content in result.page_contents:
        print(f"Title: {content.title}")
        print(f"Word count: {content.word_count}")
        print(f"Excerpt: {content.excerpt}")
```

### Pagination

Get additional result pages:

```python
# Page 1
result1 = await tool.execute(WebSearchInput(query="AI", page=1))

# Page 2
result2 = await tool.execute(WebSearchInput(query="AI", page=2))
```

### Time Range Filtering

Filter results by recency (support varies by provider):

```python
input = WebSearchInput(
    query="latest news",
    time_range="day",  # Results from last day
)
```

## Output Format

### WebSearchOutput

```python
{
    "success": True,              # Whether search succeeded
    "results": [...],             # List of SearchResult objects
    "total_results": 10,          # Number of results returned
    "page": 1,                    # Page number
    "query": "search query",      # Original query
    "provider_used": "searxng",   # Which provider was used
    "cached": False,              # Whether result was cached
    "error": None,                # Error message if failed
    "page_contents": None,        # Optional list of PageContent
}
```

### SearchResult

```python
{
    "title": "Result Title",
    "url": "https://example.com",
    "snippet": "Description of the result...",
    "score": 0.95,               # Relevance score (0.0-1.0)
    "source": "searxng",         # Provider that returned this
    "published_date": None,      # Optional publication date
}
```

### PageContent

```python
{
    "url": "https://example.com",
    "title": "Page Title",
    "content": "<html>...",      # Raw HTML (truncated)
    "text_content": "...",       # Plain text extracted
    "excerpt": "...",            # First 500 chars
    "word_count": 1234,          # Number of words
    "fetch_time": 0.5,           # Seconds to fetch
}
```

## Integration with TinyLLM

The web search tool is automatically registered when you call `register_default_tools()`:

```python
from tinyllm import register_default_tools, ToolRegistry

# Register all default tools
register_default_tools()

# Get the web search tool
tool = ToolRegistry.get("web_search")

# Or import directly
from tinyllm import WebSearchTool
```

## Custom Search Providers

You can implement custom search providers by extending `SearchProvider`:

```python
from tinyllm.tools.web_search import SearchProvider, SearchResult, WebSearchInput

class CustomProvider(SearchProvider):
    @property
    def name(self) -> str:
        return "custom"

    @property
    def available(self) -> bool:
        # Check if provider is configured
        return True

    async def search(self, input: WebSearchInput) -> list[SearchResult]:
        # Implement your search logic
        results = []
        # ... fetch from your API
        return results

# Use your custom provider
tool = WebSearchTool(config)
tool.providers.append(CustomProvider(config))
```

## Testing

The web search tool includes comprehensive unit tests using mocks:

```bash
pytest tests/unit/test_web_search.py -v
```

Key test coverage:
- All search providers
- Result deduplication
- Rate limiting
- Caching
- Content fetching
- Error handling and fallback
- Input validation

## Examples

See `examples/web_search_example.py` for complete examples including:
- Basic search with DuckDuckGo
- SearXNG configuration
- Brave Search API usage
- Multi-backend fallback
- Caching demonstration
- Pagination
- Page content fetching

Run the examples:

```bash
python examples/web_search_example.py
```

## Best Practices

1. **Use Caching**: Enable caching to reduce API calls and improve response time

2. **Configure Fallback**: Enable multiple providers for reliability

3. **Rate Limiting**: Respect API rate limits by configuring appropriate limits

4. **Deduplication**: Keep deduplication enabled to avoid showing duplicate results

5. **Cleanup**: Always call `await tool.cleanup()` when done to close HTTP clients

6. **Environment Variables**: Use environment variables for API keys and URLs

7. **Error Handling**: Always check `result.success` before processing results

## Troubleshooting

### SearXNG Not Working

- Verify your SearXNG instance URL is correct
- Ensure the instance allows JSON format responses
- Check firewall/network access

### Brave Search API Errors

- Verify your API key is valid
- Check your API quota/limits
- Ensure you're using the correct API endpoint

### Rate Limiting Issues

- Increase `rate_limit_per_minute` if you're under your API limits
- Add delays between searches if needed
- Consider using caching to reduce requests

### No Results Found

- Try different providers
- Check your query syntax
- Verify provider availability with `provider.available`

## Security Considerations

- API keys should be stored in environment variables, not code
- Use HTTPS for all connections
- SearXNG should be self-hosted or from a trusted provider
- Content fetching can expose you to malicious websites - use with caution
- Rate limiting helps prevent abuse

## Performance Tips

- Enable caching for frequently searched queries
- Use `max_results` to limit the number of results
- Disable `fetch_page_content` unless needed
- Configure appropriate timeouts
- Use pagination instead of fetching all results at once

## API Reference

For detailed API documentation, see:
- `src/tinyllm/tools/web_search.py` - Full implementation
- `tests/unit/test_web_search.py` - Usage examples in tests
- `examples/web_search_example.py` - Complete examples
