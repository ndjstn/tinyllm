"""Tests for web scraper tool."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.tools.web_scraper import (
    ContentExtractor,
    ImageInfo,
    LinkInfo,
    OutputFormat,
    RobotsTxtChecker,
    ScrapeMode,
    ScrapedContent,
    ScraperCache,
    ScraperConfig,
    ScraperInput,
    ScraperResult,
    TableData,
    WebScraperTool,
)


# Sample HTML for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="Test page description">
    <meta property="og:title" content="OG Title">
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a <b>test</b> paragraph with <a href="/relative" title="Relative Link">a link</a>.</p>
    <p>Another paragraph with <a href="https://example.com">external link</a>.</p>

    <div class="content">
        <h2>Section Content</h2>
        <p>Content inside a div with class.</p>
    </div>

    <div id="special">
        <p>Content with ID selector.</p>
    </div>

    <img src="/image.jpg" alt="Test Image" width="100" height="200">
    <img src="https://example.com/photo.png" alt="External Image" title="Photo">

    <table>
        <caption>Sample Table</caption>
        <thead>
            <tr>
                <th>Header 1</th>
                <th>Header 2</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Cell 1-1</td>
                <td>Cell 1-2</td>
            </tr>
            <tr>
                <td>Cell 2-1</td>
                <td>Cell 2-2</td>
            </tr>
        </tbody>
    </table>

    <script>console.log('test');</script>
    <style>.test { color: red; }</style>
</body>
</html>
"""


@pytest.fixture
def scraper_config():
    """Create a scraper configuration for testing."""
    return ScraperConfig(
        timeout_ms=5000,
        max_retries=2,
        rate_limit_delay_ms=0,  # No delay for tests
        cache_ttl_seconds=10,
        respect_robots_txt=False,  # Disable for tests
    )


@pytest.fixture
def web_scraper(scraper_config):
    """Create a web scraper tool instance."""
    return WebScraperTool(config=scraper_config)


@pytest.fixture
def content_extractor(scraper_config):
    """Create a content extractor instance."""
    return ContentExtractor(config=scraper_config)


class TestScraperCache:
    """Tests for scraper cache."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = ScraperCache.make_key("https://example.com", "full_page")
        key2 = ScraperCache.make_key("https://example.com", "full_page")
        key3 = ScraperCache.make_key("https://example.com", "css_selector", ".content")

        assert key1 == key2
        assert key1 != key3

    def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        cache = ScraperCache(ttl_seconds=10)

        result = ScraperResult(success=True, content=ScrapedContent(
            url="https://example.com",
            title="Test",
            content="Test content",
            fetch_time=0.5,
        ))

        key = "test_key"
        cache.set(key, result)

        cached = cache.get(key)
        assert cached is not None
        assert cached.success is True
        assert cached.cached is True
        assert cached.content.title == "Test"

    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = ScraperCache(ttl_seconds=1)

        result = ScraperResult(success=True)
        key = "test_key"
        cache.set(key, result)

        # Should be cached
        assert cache.get(key) is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get(key) is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ScraperCache(ttl_seconds=10)

        result = ScraperResult(success=True)
        cache.set("key1", result)
        cache.set("key2", result)

        assert cache.get("key1") is not None
        assert cache.get("key2") is not None

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestContentExtractor:
    """Tests for content extraction."""

    def test_extract_text(self, content_extractor):
        """Test plain text extraction."""
        text = content_extractor.extract_text(SAMPLE_HTML)

        assert "Main Heading" in text
        assert "test paragraph" in text
        assert "<script>" not in text
        assert "<style>" not in text
        assert "console.log" not in text

    def test_extract_title(self, content_extractor):
        """Test title extraction."""
        title = content_extractor.extract_title(SAMPLE_HTML)
        assert title == "Test Page"

    def test_extract_metadata(self, content_extractor):
        """Test metadata extraction."""
        metadata = content_extractor.extract_metadata(SAMPLE_HTML)

        assert "description" in metadata
        assert metadata["description"] == "Test page description"
        assert "og:title" in metadata
        assert metadata["og:title"] == "OG Title"

    def test_extract_links(self, content_extractor):
        """Test link extraction."""
        links = content_extractor.extract_links(SAMPLE_HTML, "https://example.com")

        assert len(links) == 2

        # Check relative link resolution
        relative_link = next(l for l in links if l.text == "a link")
        assert relative_link.url == "https://example.com/relative"
        assert relative_link.title == "Relative Link"

        # Check absolute link
        external_link = next(l for l in links if l.text == "external link")
        assert external_link.url == "https://example.com"

    def test_extract_images(self, content_extractor):
        """Test image extraction."""
        images = content_extractor.extract_images(SAMPLE_HTML, "https://example.com")

        assert len(images) == 2

        # Check relative image resolution
        relative_img = next(i for i in images if i.alt == "Test Image")
        assert relative_img.url == "https://example.com/image.jpg"
        assert relative_img.width == "100"
        assert relative_img.height == "200"

        # Check absolute image
        external_img = next(i for i in images if i.alt == "External Image")
        assert external_img.url == "https://example.com/photo.png"
        assert external_img.title == "Photo"

    def test_extract_tables(self, content_extractor):
        """Test table extraction."""
        tables = content_extractor.extract_tables(SAMPLE_HTML)

        assert len(tables) == 1

        table = tables[0]
        assert table.caption == "Sample Table"
        assert table.headers == ["Header 1", "Header 2"]
        assert len(table.rows) == 2
        assert table.rows[0] == ["Cell 1-1", "Cell 1-2"]
        assert table.rows[1] == ["Cell 2-1", "Cell 2-2"]

    def test_extract_by_css_selector_class(self, content_extractor):
        """Test CSS selector extraction with class."""
        content = content_extractor.extract_by_css_selector(SAMPLE_HTML, ".content")

        assert "Section Content" in content
        assert "Content inside a div with class" in content

    def test_extract_by_css_selector_id(self, content_extractor):
        """Test CSS selector extraction with ID."""
        content = content_extractor.extract_by_css_selector(SAMPLE_HTML, "#special")

        assert "Content with ID selector" in content

    def test_extract_by_css_selector_tag(self, content_extractor):
        """Test CSS selector extraction with tag."""
        content = content_extractor.extract_by_css_selector(SAMPLE_HTML, "h1")

        assert "Main Heading" in content

    def test_to_markdown(self, content_extractor):
        """Test HTML to Markdown conversion."""
        html = "<h1>Title</h1><p>Paragraph with <b>bold</b> and <i>italic</i>.</p>"
        markdown = content_extractor.to_markdown(html)

        assert "# Title" in markdown
        assert "**bold**" in markdown
        assert "*italic*" in markdown


class TestRobotsTxtChecker:
    """Tests for robots.txt checking."""

    @pytest.mark.asyncio
    async def test_can_fetch_allowed(self):
        """Test robots.txt allowing access."""
        checker = RobotsTxtChecker("TestBot")

        # Mock the RobotFileParser
        with patch('tinyllm.tools.web_scraper.RobotFileParser') as MockParser:
            mock_parser = MagicMock()
            mock_parser.can_fetch.return_value = True
            MockParser.return_value = mock_parser

            allowed = await checker.can_fetch("https://example.com/page")

            assert allowed is True

    @pytest.mark.asyncio
    async def test_can_fetch_disallowed(self):
        """Test robots.txt disallowing access."""
        checker = RobotsTxtChecker("TestBot")

        with patch('tinyllm.tools.web_scraper.RobotFileParser') as MockParser:
            mock_parser = MagicMock()
            mock_parser.can_fetch.return_value = False
            MockParser.return_value = mock_parser

            allowed = await checker.can_fetch("https://example.com/private")

            assert allowed is False

    @pytest.mark.asyncio
    async def test_robots_txt_cache(self):
        """Test robots.txt caching."""
        checker = RobotsTxtChecker("TestBot")

        with patch('tinyllm.tools.web_scraper.RobotFileParser') as MockParser:
            mock_parser = MagicMock()
            mock_parser.can_fetch.return_value = True
            MockParser.return_value = mock_parser

            # First call should fetch
            await checker.can_fetch("https://example.com/page1")

            # Second call to same domain should use cache
            await checker.can_fetch("https://example.com/page2")

            # Should only create one parser per domain
            assert len(checker.cache) == 1


class TestScraperInput:
    """Tests for scraper input validation."""

    def test_valid_input(self):
        """Test valid scraper input."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.FULL_PAGE,
            output_format=OutputFormat.PLAIN_TEXT,
        )

        assert input.url == "https://example.com"
        assert input.mode == ScrapeMode.FULL_PAGE
        assert input.output_format == OutputFormat.PLAIN_TEXT

    def test_invalid_url(self):
        """Test invalid URL validation."""
        with pytest.raises(ValueError, match="URL must start with"):
            ScraperInput(url="invalid-url")

    def test_default_values(self):
        """Test default input values."""
        input = ScraperInput(url="https://example.com")

        assert input.mode == ScrapeMode.FULL_PAGE
        assert input.output_format == OutputFormat.PLAIN_TEXT
        assert input.extract_links is False
        assert input.extract_images is False
        assert input.extract_tables is False
        assert input.use_javascript is False


@pytest.mark.asyncio
class TestWebScraperTool:
    """Tests for web scraper tool."""

    async def test_scrape_full_page_success(self, web_scraper):
        """Test successful full page scraping."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.FULL_PAGE,
            output_format=OutputFormat.PLAIN_TEXT,
        )

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response) as mock_get:
            mock_get.return_value = mock_response

            result = await web_scraper.execute(input)

            assert result.success is True
            assert result.content is not None
            assert result.content.title == "Test Page"
            assert "Main Heading" in result.content.content
            assert result.content.word_count > 0
            assert result.cached is False

    async def test_scrape_with_css_selector(self, web_scraper):
        """Test scraping with CSS selector."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.CSS_SELECTOR,
            selector=".content",
            output_format=OutputFormat.PLAIN_TEXT,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert "Section Content" in result.content.content

    async def test_scrape_links_mode(self, web_scraper):
        """Test scraping in links mode."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.LINKS,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert len(result.content.links) == 2
            assert "Found 2 links" in result.content.content

    async def test_scrape_images_mode(self, web_scraper):
        """Test scraping in images mode."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.IMAGES,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert len(result.content.images) == 2
            assert "Found 2 images" in result.content.content

    async def test_scrape_tables_mode(self, web_scraper):
        """Test scraping in tables mode."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.TABLES,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert len(result.content.tables) == 1
            assert "Found 1 tables" in result.content.content

    async def test_scrape_with_extract_options(self, web_scraper):
        """Test scraping with extract options enabled."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.FULL_PAGE,
            extract_links=True,
            extract_images=True,
            extract_tables=True,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert len(result.content.links) == 2
            assert len(result.content.images) == 2
            assert len(result.content.tables) == 1

    async def test_scrape_markdown_output(self, web_scraper):
        """Test scraping with markdown output."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.FULL_PAGE,
            output_format=OutputFormat.MARKDOWN,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert "# Main Heading" in result.content.content

    async def test_scrape_html_output(self, web_scraper):
        """Test scraping with HTML output."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.FULL_PAGE,
            output_format=OutputFormat.HTML,
        )

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert "<h1>" in result.content.content

    async def test_missing_selector_error(self, web_scraper):
        """Test error when selector is missing."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.CSS_SELECTOR,
            # No selector provided
        )

        result = await web_scraper.execute(input)

        assert result.success is False
        assert "Selector required" in result.error

    async def test_http_error_handling(self, web_scraper):
        """Test HTTP error handling."""
        input = ScraperInput(url="https://example.com")

        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        error = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=mock_response
        )

        with patch.object(web_scraper.client, 'get', side_effect=error):
            result = await web_scraper.execute(input)

            assert result.success is False
            assert "HTTP 404" in result.error

    async def test_retry_on_403(self, web_scraper):
        """Test retry behavior on 403 errors."""
        input = ScraperInput(url="https://example.com")

        # Mock 403 error followed by success
        mock_response_error = MagicMock()
        mock_response_error.status_code = 403
        mock_response_error.text = "Forbidden"

        mock_response_success = MagicMock()
        mock_response_success.text = SAMPLE_HTML
        mock_response_success.url = httpx.URL("https://example.com")
        mock_response_success.raise_for_status = MagicMock()

        error = httpx.HTTPStatusError(
            "403 Forbidden",
            request=MagicMock(),
            response=mock_response_error
        )

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return mock_response_success

        with patch.object(web_scraper.client, 'get', side_effect=mock_get):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert call_count == 2  # First failed, second succeeded

    async def test_caching(self, web_scraper):
        """Test result caching."""
        input = ScraperInput(url="https://example.com")

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response) as mock_get:
            # First call
            result1 = await web_scraper.execute(input)
            assert result1.success is True
            assert result1.cached is False

            # Second call should use cache
            result2 = await web_scraper.execute(input)
            assert result2.success is True
            assert result2.cached is True

            # HTTP client should only be called once
            assert mock_get.call_count == 1

    async def test_robots_txt_blocking(self):
        """Test robots.txt blocking."""
        config = ScraperConfig(respect_robots_txt=True)
        scraper = WebScraperTool(config=config)

        input = ScraperInput(url="https://example.com")

        # Mock robots.txt disallowing access
        with patch.object(scraper.robots_checker, 'can_fetch', return_value=False):
            result = await scraper.execute(input)

            assert result.success is False
            assert result.robots_allowed is False
            assert "robots.txt" in result.error

        await scraper.cleanup()

    async def test_content_length_limiting(self, web_scraper):
        """Test content length limiting."""
        input = ScraperInput(url="https://example.com")

        # Create large HTML content
        large_html = SAMPLE_HTML * 10000

        mock_response = MagicMock()
        mock_response.text = large_html
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            # Should still succeed but content should be truncated
            assert result.success is True
            assert len(result.content.content) <= web_scraper.config.max_content_length

    async def test_redirect_tracking(self, web_scraper):
        """Test redirect URL tracking."""
        input = ScraperInput(url="https://example.com")

        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.url = httpx.URL("https://example.com/redirected")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert result.redirected_url == "https://example.com/redirected"

    async def test_cleanup(self, web_scraper):
        """Test resource cleanup."""
        # Mock the client's aclose method
        with patch.object(web_scraper.client, 'aclose', new_callable=AsyncMock) as mock_close:
            await web_scraper.cleanup()
            mock_close.assert_called_once()


@pytest.mark.asyncio
class TestEdgeCases:
    """Tests for edge cases."""

    async def test_empty_html(self, web_scraper):
        """Test handling of empty HTML."""
        input = ScraperInput(url="https://example.com")

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert result.content.content == ""

    async def test_malformed_html(self, web_scraper):
        """Test handling of malformed HTML."""
        input = ScraperInput(url="https://example.com")

        malformed_html = "<html><body><p>Unclosed paragraph<div>Mixed tags</p></div>"

        mock_response = MagicMock()
        mock_response.text = malformed_html
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            # Should not crash, extract what it can
            assert result.success is True

    async def test_no_links_found(self, web_scraper):
        """Test when no links are found."""
        input = ScraperInput(
            url="https://example.com",
            mode=ScrapeMode.LINKS,
        )

        html = "<html><body><p>No links here</p></body></html>"

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert len(result.content.links) == 0
            assert "Found 0 links" in result.content.content

    async def test_special_characters_in_content(self, web_scraper):
        """Test handling of special characters."""
        input = ScraperInput(url="https://example.com")

        html = """
        <html><body>
        <p>Special chars: &lt;tag&gt; &amp; &quot;quotes&quot; &apos;apostrophe&apos;</p>
        <p>Unicode: 日本語 emoji 🚀</p>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.url = httpx.URL("https://example.com")
        mock_response.raise_for_status = MagicMock()

        with patch.object(web_scraper.client, 'get', return_value=mock_response):
            result = await web_scraper.execute(input)

            assert result.success is True
            assert "<tag>" in result.content.content
            assert "&" in result.content.content
            assert "日本語" in result.content.content
