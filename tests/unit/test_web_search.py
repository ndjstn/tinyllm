"""Tests for web search tool."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.tools.web_search import (
    BraveSearchProvider,
    ContentFetcher,
    DuckDuckGoProvider,
    PageContent,
    RateLimiter,
    ResultDeduplicator,
    SearchCache,
    SearchResult,
    SearXNGProvider,
    WebSearchConfig,
    WebSearchInput,
    WebSearchOutput,
    WebSearchTool,
)


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_valid_search_result(self):
        """Test creating a valid search result."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.score == 1.0
        assert result.source == "unknown"

    def test_search_result_with_custom_score(self):
        """Test search result with custom score."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
            score=0.75,
            source="test_source",
        )
        assert result.score == 0.75
        assert result.source == "test_source"

    def test_invalid_url(self):
        """Test that invalid URLs are rejected."""
        with pytest.raises(ValueError, match="Invalid URL"):
            SearchResult(
                title="Test",
                url="not-a-url",
                snippet="Test",
            )

    def test_valid_http_url(self):
        """Test that http URLs are accepted."""
        result = SearchResult(
            title="Test",
            url="http://example.com",
            snippet="Test",
        )
        assert result.url == "http://example.com"


class TestPageContent:
    """Tests for PageContent model."""

    def test_page_content_creation(self):
        """Test creating page content."""
        content = PageContent(
            url="https://example.com",
            title="Example Page",
            content="<html>test</html>",
            text_content="This is test content.",
            excerpt="This is test",
            word_count=4,
            fetch_time=0.5,
        )
        assert content.url == "https://example.com"
        assert content.title == "Example Page"
        assert content.word_count == 4
        assert content.fetch_time == 0.5


class TestWebSearchConfig:
    """Tests for WebSearchConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WebSearchConfig()
        assert config.max_results == 10
        assert config.timeout_ms == 30000
        assert config.cache_ttl_seconds == 3600
        assert config.deduplicate_results is True
        assert config.fallback_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = WebSearchConfig(
            max_results=20,
            cache_ttl_seconds=1800,
            searxng_url="https://search.example.com",
            brave_api_key="test_key",
        )
        assert config.max_results == 20
        assert config.cache_ttl_seconds == 1800
        assert config.searxng_url == "https://search.example.com"
        assert config.brave_api_key == "test_key"


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=5, per_seconds=1)

        start = time.time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_rate_limiter_delays_excess(self):
        """Test that rate limiter delays excess requests."""
        limiter = RateLimiter(max_requests=2, per_seconds=1)

        start = time.time()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.time() - start

        # Third request should be delayed
        assert elapsed >= 1.0

    @pytest.mark.asyncio
    async def test_rate_limiter_resets(self):
        """Test that rate limiter resets after time window."""
        limiter = RateLimiter(max_requests=2, per_seconds=1)

        # Use up quota
        await limiter.acquire()
        await limiter.acquire()

        # Wait for reset
        await asyncio.sleep(1.1)

        # Should allow more requests
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.5


class TestSearchCache:
    """Tests for SearchCache."""

    def test_cache_miss(self):
        """Test cache miss."""
        cache = SearchCache(ttl_seconds=3600)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_hit(self):
        """Test cache hit."""
        cache = SearchCache(ttl_seconds=3600)
        output = WebSearchOutput(
            success=True,
            results=[],
            query="test",
            page=1,
        )

        cache.set("test_key", output)
        result = cache.get("test_key")

        assert result is not None
        assert result.cached is True
        assert result.query == "test"

    def test_cache_expiration(self):
        """Test that cache entries expire."""
        cache = SearchCache(ttl_seconds=1)
        output = WebSearchOutput(
            success=True,
            results=[],
            query="test",
            page=1,
        )

        cache.set("test_key", output)
        time.sleep(1.5)

        result = cache.get("test_key")
        assert result is None

    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = SearchCache.make_key("test query", page=1)
        key2 = SearchCache.make_key("test query", page=1)
        key3 = SearchCache.make_key("test query", page=2)
        key4 = SearchCache.make_key("different query", page=1)

        # Same query and page should generate same key
        assert key1 == key2

        # Different page should generate different key
        assert key1 != key3

        # Different query should generate different key
        assert key1 != key4

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = SearchCache()
        output = WebSearchOutput(success=True, results=[], query="test", page=1)

        cache.set("key1", output)
        cache.set("key2", output)
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestResultDeduplicator:
    """Tests for ResultDeduplicator."""

    def test_normalize_url(self):
        """Test URL normalization."""
        url1 = "https://www.example.com/path/"
        url2 = "https://example.com/path"
        url3 = "http://www.example.com/path"

        norm1 = ResultDeduplicator.normalize_url(url1)
        norm2 = ResultDeduplicator.normalize_url(url2)
        norm3 = ResultDeduplicator.normalize_url(url3)

        # Should all normalize to the same
        assert norm1 == norm2 == norm3

    def test_deduplicate_results(self):
        """Test deduplication of results."""
        results = [
            SearchResult(
                title="Result 1",
                url="https://example.com/page",
                snippet="First",
                score=0.9,
            ),
            SearchResult(
                title="Result 2",
                url="https://www.example.com/page/",
                snippet="Second",
                score=0.8,
            ),
            SearchResult(
                title="Result 3",
                url="https://different.com/page",
                snippet="Third",
                score=0.7,
            ),
        ]

        deduped = ResultDeduplicator.deduplicate(results)

        # Should have 2 unique results (first two are duplicates)
        assert len(deduped) == 2

        # Should keep the one with higher score
        urls = [r.url for r in deduped]
        assert "https://example.com/page" in urls
        assert "https://different.com/page" in urls

    def test_rank_results(self):
        """Test ranking by score."""
        results = [
            SearchResult(title="Low", url="https://a.com", snippet="A", score=0.3),
            SearchResult(title="High", url="https://b.com", snippet="B", score=0.9),
            SearchResult(title="Mid", url="https://c.com", snippet="C", score=0.6),
        ]

        ranked = ResultDeduplicator.rank(results)

        assert ranked[0].score == 0.9
        assert ranked[1].score == 0.6
        assert ranked[2].score == 0.3


class TestContentFetcher:
    """Tests for ContentFetcher."""

    @pytest.mark.asyncio
    async def test_extract_title(self):
        """Test title extraction."""
        html = "<html><head><title>Test Title</title></head><body>Content</body></html>"
        title = ContentFetcher._extract_title(html)
        assert title == "Test Title"

    @pytest.mark.asyncio
    async def test_extract_title_no_title(self):
        """Test title extraction when no title tag."""
        html = "<html><body>Content</body></html>"
        title = ContentFetcher._extract_title(html)
        assert title == ""

    @pytest.mark.asyncio
    async def test_extract_text(self):
        """Test text extraction from HTML."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <script>var x = 1;</script>
                <p>This is content.</p>
                <style>.class { color: red; }</style>
                <div>More content here.</div>
            </body>
        </html>
        """
        text = ContentFetcher._extract_text(html)

        assert "This is content." in text
        assert "More content here." in text
        assert "var x = 1" not in text
        assert "color: red" not in text

    @pytest.mark.asyncio
    async def test_extract_text_with_entities(self):
        """Test HTML entity decoding."""
        html = "<p>Test &amp; more &lt;test&gt; &quot;quoted&quot;</p>"
        text = ContentFetcher._extract_text(html)

        assert "&" in text
        assert "<test>" in text
        assert '"quoted"' in text

    @pytest.mark.asyncio
    async def test_fetch_content(self):
        """Test fetching content from URL."""
        fetcher = ContentFetcher(timeout=5.0)

        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body><p>Test content here.</p></body>
        </html>
        """
        mock_response.raise_for_status = MagicMock()

        with patch.object(fetcher.client, "get", new=AsyncMock(return_value=mock_response)):
            content = await fetcher.fetch("https://example.com")

            assert content.url == "https://example.com"
            assert content.title == "Test Page"
            assert "Test content here." in content.text_content
            assert len(content.excerpt) <= 500
            assert content.word_count > 0

        await fetcher.close()


class TestSearXNGProvider:
    """Tests for SearXNGProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebSearchConfig(
            searxng_url="https://search.example.com",
            enable_searxng=True,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        return SearXNGProvider(config)

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.name == "searxng"

    def test_available_when_configured(self, provider):
        """Test provider is available when configured."""
        assert provider.available is True

    def test_not_available_when_not_configured(self):
        """Test provider is not available when not configured."""
        config = WebSearchConfig(searxng_url=None)
        provider = SearXNGProvider(config)
        assert provider.available is False

    @pytest.mark.asyncio
    async def test_search_success(self, provider):
        """Test successful search."""
        input = WebSearchInput(query="test query")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "First result snippet",
                    "score": 0.95,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Second result snippet",
                    "score": 0.85,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "get", new=AsyncMock(return_value=mock_response)):
            results = await provider.search(input)

            assert len(results) == 2
            assert results[0].title == "Test Result 1"
            assert results[0].url == "https://example.com/1"
            assert results[0].score == 0.95
            assert results[0].source == "searxng"

        await provider.close()

    @pytest.mark.asyncio
    async def test_search_not_configured(self):
        """Test search fails when not configured."""
        config = WebSearchConfig(searxng_url=None)
        provider = SearXNGProvider(config)
        input = WebSearchInput(query="test")

        with pytest.raises(ValueError, match="not configured"):
            await provider.search(input)

        await provider.close()


class TestDuckDuckGoProvider:
    """Tests for DuckDuckGoProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebSearchConfig(enable_duckduckgo=True)

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        return DuckDuckGoProvider(config)

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.name == "duckduckgo"

    def test_available_when_enabled(self, provider):
        """Test provider is available when enabled."""
        assert provider.available is True

    @pytest.mark.asyncio
    async def test_search_success(self, provider):
        """Test successful search."""
        input = WebSearchInput(query="test query")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "Test abstract content",
            "AbstractURL": "https://example.com/abstract",
            "Heading": "Test Topic",
            "RelatedTopics": [
                {
                    "Text": "Related topic 1 description",
                    "FirstURL": "https://example.com/1",
                },
                {
                    "Text": "Related topic 2 description",
                    "FirstURL": "https://example.com/2",
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "get", new=AsyncMock(return_value=mock_response)):
            results = await provider.search(input)

            # Should have abstract + 2 related topics
            assert len(results) >= 1
            assert results[0].source == "duckduckgo"
            assert any("Test abstract" in r.snippet for r in results)

        await provider.close()


class TestBraveSearchProvider:
    """Tests for BraveSearchProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebSearchConfig(
            brave_api_key="test_api_key",
            enable_brave=True,
        )

    @pytest.fixture
    def provider(self, config):
        """Create provider instance."""
        return BraveSearchProvider(config)

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.name == "brave"

    def test_available_when_configured(self, provider):
        """Test provider is available when configured."""
        assert provider.available is True

    def test_not_available_without_api_key(self):
        """Test provider is not available without API key."""
        config = WebSearchConfig(brave_api_key=None)
        provider = BraveSearchProvider(config)
        assert provider.available is False

    @pytest.mark.asyncio
    async def test_search_success(self, provider):
        """Test successful search."""
        input = WebSearchInput(query="test query")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Brave Result 1",
                        "url": "https://example.com/1",
                        "description": "First result from Brave",
                    },
                    {
                        "title": "Brave Result 2",
                        "url": "https://example.com/2",
                        "description": "Second result from Brave",
                        "age": "2024-01-15",
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider.client, "get", new=AsyncMock(return_value=mock_response)):
            results = await provider.search(input)

            assert len(results) == 2
            assert results[0].title == "Brave Result 1"
            assert results[0].source == "brave"
            assert results[1].published_date == "2024-01-15"

        await provider.close()


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WebSearchConfig(
            searxng_url="https://search.example.com",
            brave_api_key="test_key",
            enable_searxng=True,
            enable_duckduckgo=True,
            enable_brave=False,
            cache_ttl_seconds=0,  # Disable caching for tests
        )

    @pytest.fixture
    def tool(self, config):
        """Create tool instance."""
        return WebSearchTool(config)

    def test_metadata(self, tool):
        """Test tool metadata."""
        assert tool.metadata.id == "web_search"
        assert tool.metadata.name == "Web Search"
        assert tool.metadata.category == "search"
        assert tool.metadata.sandbox_required is False

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful search execution."""
        input = WebSearchInput(query="test query")

        # Mock SearXNG provider
        mock_results = [
            SearchResult(
                title="Result 1",
                url="https://example.com/1",
                snippet="Test snippet 1",
                score=0.9,
                source="searxng",
            ),
            SearchResult(
                title="Result 2",
                url="https://example.com/2",
                snippet="Test snippet 2",
                score=0.8,
                source="searxng",
            ),
        ]

        with patch.object(
            tool.providers[0], "search", new=AsyncMock(return_value=mock_results)
        ):
            output = await tool.execute(input)

            assert output.success is True
            assert len(output.results) == 2
            assert output.query == "test query"
            assert output.provider_used == "searxng"
            assert output.error is None

    @pytest.mark.asyncio
    async def test_execute_with_deduplication(self, tool):
        """Test search with deduplication."""
        input = WebSearchInput(query="test query")

        # Mock results with duplicates
        mock_results = [
            SearchResult(
                title="Result 1",
                url="https://example.com/page",
                snippet="Test",
                score=0.9,
            ),
            SearchResult(
                title="Result 2",
                url="https://www.example.com/page/",
                snippet="Test",
                score=0.8,
            ),
            SearchResult(
                title="Result 3",
                url="https://different.com",
                snippet="Test",
                score=0.7,
            ),
        ]

        with patch.object(
            tool.providers[0], "search", new=AsyncMock(return_value=mock_results)
        ):
            output = await tool.execute(input)

            # Should deduplicate to 2 results
            assert output.success is True
            assert len(output.results) == 2

    @pytest.mark.asyncio
    async def test_execute_with_max_results(self, tool):
        """Test limiting results."""
        input = WebSearchInput(query="test query", max_results=2)

        mock_results = [
            SearchResult(title=f"Result {i}", url=f"https://example.com/{i}", snippet="Test")
            for i in range(10)
        ]

        with patch.object(
            tool.providers[0], "search", new=AsyncMock(return_value=mock_results)
        ):
            output = await tool.execute(input)

            assert output.success is True
            assert len(output.results) == 2

    @pytest.mark.asyncio
    async def test_execute_fallback_to_second_provider(self, tool):
        """Test fallback when first provider fails."""
        input = WebSearchInput(query="test query")

        mock_results = [
            SearchResult(
                title="DDG Result",
                url="https://example.com",
                snippet="From DuckDuckGo",
                source="duckduckgo",
            )
        ]

        # First provider fails, second succeeds
        with patch.object(
            tool.providers[0], "search", new=AsyncMock(side_effect=Exception("API Error"))
        ):
            with patch.object(
                tool.providers[1], "search", new=AsyncMock(return_value=mock_results)
            ):
                output = await tool.execute(input)

                assert output.success is True
                assert output.provider_used == "duckduckgo"

    @pytest.mark.asyncio
    async def test_execute_all_providers_fail(self, tool):
        """Test when all providers fail."""
        input = WebSearchInput(query="test query")

        # All providers fail
        for provider in tool.providers:
            provider.search = AsyncMock(side_effect=Exception("API Error"))

        output = await tool.execute(input)

        assert output.success is False
        assert output.error is not None
        assert "failed" in output.error.lower()

    @pytest.mark.asyncio
    async def test_execute_caching(self):
        """Test result caching."""
        config = WebSearchConfig(
            searxng_url="https://search.example.com",
            cache_ttl_seconds=3600,  # Enable caching
        )
        tool = WebSearchTool(config)
        input = WebSearchInput(query="test query")

        mock_results = [
            SearchResult(
                title="Result 1",
                url="https://example.com",
                snippet="Test",
                source="searxng",
            )
        ]

        with patch.object(
            tool.providers[0], "search", new=AsyncMock(return_value=mock_results)
        ) as mock_search:
            # First call
            output1 = await tool.execute(input)
            assert output1.cached is False
            assert mock_search.call_count == 1

            # Second call should use cache
            output2 = await tool.execute(input)
            assert output2.cached is True
            assert mock_search.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_execute_with_page_content(self):
        """Test fetching page content."""
        config = WebSearchConfig(
            searxng_url="https://search.example.com",
            fetch_page_content=True,
        )
        tool = WebSearchTool(config)
        input = WebSearchInput(query="test query")

        mock_results = [
            SearchResult(
                title="Result 1",
                url="https://example.com",
                snippet="Test",
            )
        ]

        mock_content = PageContent(
            url="https://example.com",
            title="Page Title",
            content="<html>test</html>",
            text_content="Test content",
            excerpt="Test",
            word_count=2,
            fetch_time=0.1,
        )

        with patch.object(tool.providers[0], "search", new=AsyncMock(return_value=mock_results)):
            with patch.object(
                tool.content_fetcher, "fetch", new=AsyncMock(return_value=mock_content)
            ):
                output = await tool.execute(input)

                assert output.success is True
                assert output.page_contents is not None
                assert len(output.page_contents) == 1
                assert output.page_contents[0].title == "Page Title"

    @pytest.mark.asyncio
    async def test_rate_limiting(self, tool):
        """Test that rate limiting is applied."""
        input = WebSearchInput(query="test query")

        mock_results = [
            SearchResult(title="Result", url="https://example.com", snippet="Test")
        ]

        with patch.object(tool.providers[0], "search", new=AsyncMock(return_value=mock_results)):
            with patch.object(tool.rate_limiter, "acquire", new=AsyncMock()) as mock_acquire:
                await tool.execute(input)
                mock_acquire.assert_called_once()


class TestWebSearchInput:
    """Tests for WebSearchInput validation."""

    def test_valid_input(self):
        """Test valid input."""
        input = WebSearchInput(query="test query")
        assert input.query == "test query"
        assert input.page == 1
        assert input.language == "en"

    def test_input_with_options(self):
        """Test input with all options."""
        input = WebSearchInput(
            query="test query",
            max_results=20,
            page=2,
            language="es",
            time_range="week",
        )
        assert input.max_results == 20
        assert input.page == 2
        assert input.language == "es"
        assert input.time_range == "week"

    def test_invalid_time_range(self):
        """Test invalid time range."""
        with pytest.raises(ValueError):
            WebSearchInput(query="test", time_range="invalid")


@pytest.mark.parametrize(
    "query,expected_results",
    [
        ("simple query", True),
        ("query with special chars !@#", True),
        ("very " + "long " * 50 + "query", True),
    ],
)
@pytest.mark.asyncio
async def test_various_queries(query, expected_results):
    """Test tool with various query types."""
    config = WebSearchConfig(searxng_url="https://search.example.com")
    tool = WebSearchTool(config)

    input = WebSearchInput(query=query)

    mock_results = [
        SearchResult(title="Result", url="https://example.com", snippet="Test")
    ]

    with patch.object(tool.providers[0], "search", new=AsyncMock(return_value=mock_results)):
        output = await tool.execute(input)
        assert output.success == expected_results
