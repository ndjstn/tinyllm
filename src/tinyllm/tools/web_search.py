"""Web search tool for TinyLLM agents.

This module provides a unified interface for multiple search backends:
- SearXNG (self-hosted, privacy-focused)
- DuckDuckGo (via duckduckgo-search package)
- Brave Search API (optional, requires API key)

Features:
- Abstract base class for search providers
- Result ranking and deduplication
- URL content fetching with readability extraction
- Rate limiting
- Caching of results
"""

import asyncio
import hashlib
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, field_validator

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata


class SearchResult(BaseModel):
    """A single search result."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    snippet: str = Field(description="Short snippet/description of the result")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")
    source: str = Field(default="unknown", description="Search provider that returned this result")
    published_date: Optional[str] = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {v}")
        return v


class PageContent(BaseModel):
    """Extracted content from a web page."""

    url: str
    title: str
    content: str
    text_content: str = Field(description="Plain text content without HTML")
    excerpt: str = Field(description="First 500 chars of text content")
    word_count: int
    fetch_time: float = Field(description="Time taken to fetch in seconds")


class WebSearchConfig(ToolConfig):
    """Configuration for web search tool."""

    max_results: int = Field(default=10, ge=1, le=100)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    cache_ttl_seconds: int = Field(default=3600, ge=0, le=86400)
    rate_limit_per_minute: int = Field(default=20, ge=1, le=1000)
    deduplicate_results: bool = Field(default=True)
    fetch_page_content: bool = Field(default=False)
    fallback_enabled: bool = Field(default=True)

    # Provider-specific settings
    searxng_url: Optional[str] = Field(default=None)
    brave_api_key: Optional[str] = Field(default=None)
    enable_searxng: bool = Field(default=True)
    enable_duckduckgo: bool = Field(default=True)
    enable_brave: bool = Field(default=False)


class WebSearchInput(BaseModel):
    """Input for web search tool."""

    query: str = Field(
        description="Search query",
        min_length=1,
        max_length=500,
        examples=["Python async programming", "climate change 2024"],
    )
    max_results: Optional[int] = Field(default=None, ge=1, le=100)
    page: int = Field(default=1, ge=1, le=10, description="Result page number")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es')")
    time_range: Optional[str] = Field(
        default=None,
        description="Time range filter (day, week, month, year)",
        pattern=r"^(day|week|month|year)?$",
    )


class WebSearchOutput(BaseModel):
    """Output from web search tool."""

    success: bool
    results: list[SearchResult] = Field(default_factory=list)
    total_results: int = 0
    page: int = 1
    query: str = ""
    provider_used: str = ""
    cached: bool = False
    error: Optional[str] = None
    page_contents: Optional[list[PageContent]] = None


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, max_requests: int, per_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed.
            per_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests: list[float] = []

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        now = time.time()
        # Remove old requests outside the window
        cutoff = now - self.per_seconds
        self.requests = [ts for ts in self.requests if ts > cutoff]

        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest = self.requests[0]
            wait_time = self.per_seconds - (now - oldest)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Retry
                await self.acquire()
        else:
            self.requests.append(now)


class SearchCache:
    """Simple in-memory cache for search results."""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[WebSearchOutput, float]] = {}

    def get(self, key: str) -> Optional[WebSearchOutput]:
        """Get cached result if valid.

        Args:
            key: Cache key.

        Returns:
            Cached result or None if not found/expired.
        """
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                result.cached = True
                return result
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: WebSearchOutput) -> None:
        """Store result in cache.

        Args:
            key: Cache key.
            value: Result to cache.
        """
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    @staticmethod
    def make_key(query: str, page: int = 1, **kwargs: Any) -> str:
        """Create cache key from query parameters.

        Args:
            query: Search query.
            page: Page number.
            **kwargs: Additional parameters.

        Returns:
            Cache key.
        """
        key_parts = [query, str(page)]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    def __init__(self, config: WebSearchConfig):
        """Initialize provider.

        Args:
            config: Search configuration.
        """
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_ms / 1000),
            follow_redirects=True,
        )

    @abstractmethod
    async def search(self, input: WebSearchInput) -> list[SearchResult]:
        """Execute search query.

        Args:
            input: Search parameters.

        Returns:
            List of search results.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    def available(self) -> bool:
        """Check if provider is available/configured."""
        return True

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


class SearXNGProvider(SearchProvider):
    """SearXNG search provider (self-hosted, privacy-focused)."""

    @property
    def name(self) -> str:
        """Provider name."""
        return "searxng"

    @property
    def available(self) -> bool:
        """Check if SearXNG is configured."""
        return (
            self.config.enable_searxng
            and self.config.searxng_url is not None
            and len(self.config.searxng_url) > 0
        )

    async def search(self, input: WebSearchInput) -> list[SearchResult]:
        """Search using SearXNG.

        Args:
            input: Search parameters.

        Returns:
            List of search results.
        """
        if not self.available:
            raise ValueError("SearXNG not configured")

        params = {
            "q": input.query,
            "format": "json",
            "pageno": input.page,
            "language": input.language,
        }

        if input.time_range:
            params["time_range"] = input.time_range

        response = await self.client.get(
            f"{self.config.searxng_url}/search",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    score=item.get("score", 1.0),
                    source="searxng",
                    published_date=item.get("publishedDate"),
                )
            )

        return results


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider using duckduckgo-search package."""

    @property
    def name(self) -> str:
        """Provider name."""
        return "duckduckgo"

    @property
    def available(self) -> bool:
        """Check if DuckDuckGo is enabled."""
        return self.config.enable_duckduckgo

    async def search(self, input: WebSearchInput) -> list[SearchResult]:
        """Search using DuckDuckGo.

        Args:
            input: Search parameters.

        Returns:
            List of search results.
        """
        if not self.available:
            raise ValueError("DuckDuckGo not enabled")

        try:
            from ddgs import DDGS
        except ImportError:
            raise ValueError(
                "ddgs package not installed. "
                "Install with: pip install ddgs"
            )

        results = []
        max_results = input.max_results or self.config.max_results

        # Run the synchronous DDGS in a thread pool to not block async
        loop = asyncio.get_event_loop()

        def do_search():
            ddgs = DDGS()
            # Map time_range to DuckDuckGo format
            timelimit = None
            if input.time_range:
                timelimit_map = {
                    "day": "d",
                    "week": "w",
                    "month": "m",
                    "year": "y",
                }
                timelimit = timelimit_map.get(input.time_range)

            # Use 'wt-wt' for worldwide or map language to region
            region_map = {
                "en": "wt-wt",  # worldwide
                "us": "us-en",
                "uk": "uk-en",
                "de": "de-de",
                "fr": "fr-fr",
                "es": "es-es",
            }
            region = region_map.get(input.language, "wt-wt")

            return list(ddgs.text(
                input.query,
                region=region,
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max_results,
            ))

        search_results = await loop.run_in_executor(None, do_search)

        for idx, item in enumerate(search_results):
            try:
                # Calculate score based on position
                score = max(0.5, 1.0 - (idx * 0.05))

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("href", item.get("link", "")),
                        snippet=item.get("body", item.get("snippet", "")),
                        score=score,
                        source="duckduckgo",
                    )
                )
            except Exception:
                # Skip invalid results
                continue

        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider (requires API key)."""

    @property
    def name(self) -> str:
        """Provider name."""
        return "brave"

    @property
    def available(self) -> bool:
        """Check if Brave Search is configured."""
        return (
            self.config.enable_brave
            and self.config.brave_api_key is not None
            and len(self.config.brave_api_key) > 0
        )

    async def search(self, input: WebSearchInput) -> list[SearchResult]:
        """Search using Brave Search API.

        Args:
            input: Search parameters.

        Returns:
            List of search results.
        """
        if not self.available:
            raise ValueError("Brave Search not configured")

        headers = {
            "X-Subscription-Token": self.config.brave_api_key or "",
        }

        params = {
            "q": input.query,
            "count": min(input.max_results or 10, 20),
            "offset": (input.page - 1) * (input.max_results or 10),
        }

        if input.time_range:
            params["freshness"] = input.time_range

        response = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    score=1.0,
                    source="brave",
                    published_date=item.get("age"),
                )
            )

        return results


class ResultDeduplicator:
    """Deduplicates and ranks search results."""

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for comparison.

        Args:
            url: URL to normalize.

        Returns:
            Normalized URL.
        """
        parsed = urlparse(url)
        # Remove www, trailing slashes, common tracking params
        domain = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.rstrip("/")
        return f"{domain}{path}"

    @staticmethod
    def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate URLs and merge scores.

        Args:
            results: List of search results.

        Returns:
            Deduplicated list.
        """
        seen: dict[str, SearchResult] = {}

        for result in results:
            normalized = ResultDeduplicator.normalize_url(result.url)

            if normalized in seen:
                # Keep result with higher score
                existing = seen[normalized]
                if result.score > existing.score:
                    seen[normalized] = result
                else:
                    # Boost score for duplicate finding
                    existing.score = min(1.0, existing.score + 0.1)
            else:
                seen[normalized] = result

        return list(seen.values())

    @staticmethod
    def rank(results: list[SearchResult]) -> list[SearchResult]:
        """Sort results by score (descending).

        Args:
            results: List of search results.

        Returns:
            Sorted list.
        """
        return sorted(results, key=lambda r: r.score, reverse=True)


class ContentFetcher:
    """Fetches and extracts content from URLs."""

    def __init__(self, timeout: float = 10.0):
        """Initialize content fetcher.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

    async def fetch(self, url: str) -> PageContent:
        """Fetch and extract content from URL.

        Args:
            url: URL to fetch.

        Returns:
            Extracted page content.
        """
        start_time = time.time()

        response = await self.client.get(url)
        response.raise_for_status()

        html = response.text
        text = self._extract_text(html)
        title = self._extract_title(html)

        fetch_time = time.time() - start_time

        return PageContent(
            url=url,
            title=title,
            content=html[:10000],  # Limit stored HTML
            text_content=text,
            excerpt=text[:500],
            word_count=len(text.split()),
            fetch_time=fetch_time,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract title from HTML.

        Args:
            html: HTML content.

        Returns:
            Page title.
        """
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_text(html: str) -> str:
        """Extract plain text from HTML (simple approach).

        Args:
            html: HTML content.

        Returns:
            Plain text.
        """
        # Remove script and style tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text


class WebSearchTool(BaseTool[WebSearchInput, WebSearchOutput]):
    """Web search tool with multiple backend support."""

    metadata = ToolMetadata(
        id="web_search",
        name="Web Search",
        description="Search the web using multiple backends (SearXNG, DuckDuckGo, Brave). "
        "Returns ranked and deduplicated results. Optionally fetches page content.",
        category="search",
        sandbox_required=False,
    )
    input_type = WebSearchInput
    output_type = WebSearchOutput

    def __init__(self, config: WebSearchConfig | None = None):
        """Initialize web search tool.

        Args:
            config: Search configuration.
        """
        # Load config from environment if not provided
        if config is None:
            config = WebSearchConfig(
                searxng_url=os.getenv("SEARXNG_URL"),
                brave_api_key=os.getenv("BRAVE_API_KEY"),
            )

        super().__init__(config)
        self.config: WebSearchConfig = config  # type: ignore

        # Initialize components
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_per_minute,
            per_seconds=60,
        )
        self.cache = SearchCache(self.config.cache_ttl_seconds)
        self.content_fetcher = ContentFetcher(timeout=self.config.timeout_ms / 1000)

        # Initialize providers
        self.providers: list[SearchProvider] = [
            SearXNGProvider(self.config),
            DuckDuckGoProvider(self.config),
            BraveSearchProvider(self.config),
        ]

    async def execute(self, input: WebSearchInput) -> WebSearchOutput:
        """Execute web search.

        Args:
            input: Search parameters.

        Returns:
            Search results.
        """
        # Check cache
        cache_key = SearchCache.make_key(
            input.query,
            input.page,
            language=input.language,
            time_range=input.time_range,
        )

        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        # Rate limiting
        await self.rate_limiter.acquire()

        # Try providers in order with fallback
        results: list[SearchResult] = []
        last_error: Optional[str] = None
        provider_used = "none"

        for provider in self.providers:
            if not provider.available:
                continue

            try:
                results = await provider.search(input)
                provider_used = provider.name
                break
            except Exception as e:
                last_error = f"{provider.name}: {str(e)}"
                if not self.config.fallback_enabled:
                    break
                # Continue to next provider
                continue

        if not results and last_error:
            return WebSearchOutput(
                success=False,
                query=input.query,
                page=input.page,
                error=f"All providers failed. Last error: {last_error}",
            )

        # Deduplicate and rank
        if self.config.deduplicate_results:
            results = ResultDeduplicator.deduplicate(results)

        results = ResultDeduplicator.rank(results)

        # Limit results
        max_results = input.max_results or self.config.max_results
        results = results[:max_results]

        # Fetch page content if requested
        page_contents: Optional[list[PageContent]] = None
        if self.config.fetch_page_content:
            page_contents = []
            for result in results[:3]:  # Limit to top 3
                try:
                    content = await self.content_fetcher.fetch(result.url)
                    page_contents.append(content)
                except Exception:
                    # Skip failed fetches
                    continue

        output = WebSearchOutput(
            success=True,
            results=results,
            total_results=len(results),
            page=input.page,
            query=input.query,
            provider_used=provider_used,
            page_contents=page_contents,
        )

        # Cache result
        self.cache.set(cache_key, output)

        return output

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.content_fetcher.close()
        for provider in self.providers:
            await provider.close()
