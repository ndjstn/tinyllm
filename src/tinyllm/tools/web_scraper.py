"""Web scraping tool for TinyLLM agents.

This module provides a comprehensive web scraping tool with:
- Multiple extraction modes (full page, CSS selectors, XPath, links, images, tables)
- JavaScript rendering support (optional, when playwright available)
- Rate limiting to avoid overwhelming sites
- Robots.txt checking for respectful scraping
- Content caching
- Anti-bot measure handling
- Content cleaning and sanitization
- Multiple output formats (markdown, plain text, HTML)
"""

import asyncio
import hashlib
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from pydantic import BaseModel, Field, field_validator

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata


class ScrapeMode(str, Enum):
    """Scraping mode."""

    FULL_PAGE = "full_page"
    CSS_SELECTOR = "css_selector"
    XPATH = "xpath"
    LINKS = "links"
    IMAGES = "images"
    TABLES = "tables"


class OutputFormat(str, Enum):
    """Output format for scraped content."""

    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    JSON = "json"


class LinkInfo(BaseModel):
    """Information about a link found on the page."""

    url: str = Field(description="Absolute URL of the link")
    text: str = Field(description="Link text/anchor text")
    title: Optional[str] = Field(default=None, description="Link title attribute")


class ImageInfo(BaseModel):
    """Information about an image found on the page."""

    url: str = Field(description="Absolute URL of the image")
    alt: Optional[str] = Field(default=None, description="Alt text")
    title: Optional[str] = Field(default=None, description="Title attribute")
    width: Optional[str] = Field(default=None, description="Width attribute")
    height: Optional[str] = Field(default=None, description="Height attribute")


class TableData(BaseModel):
    """Extracted table data."""

    headers: list[str] = Field(default_factory=list, description="Table headers")
    rows: list[list[str]] = Field(default_factory=list, description="Table rows")
    caption: Optional[str] = Field(default=None, description="Table caption if present")


class ScrapedContent(BaseModel):
    """Content extracted from a web page."""

    url: str
    title: str
    content: str = Field(description="Main content based on scraping mode")
    links: list[LinkInfo] = Field(default_factory=list)
    images: list[ImageInfo] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    word_count: int = 0
    fetch_time: float = Field(description="Time taken to fetch in seconds")


class ScraperConfig(ToolConfig):
    """Configuration for web scraper tool."""

    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    max_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_delay_ms: int = Field(
        default=1000, ge=0, le=10000, description="Delay between requests in ms"
    )
    cache_ttl_seconds: int = Field(default=3600, ge=0, le=86400)
    respect_robots_txt: bool = Field(default=True)
    max_redirects: int = Field(default=5, ge=0, le=20)
    user_agent: str = Field(
        default="TinyLLM-WebScraper/1.0 (+https://github.com/tinyllm/tinyllm)"
    )

    # JavaScript rendering (requires playwright)
    enable_javascript: bool = Field(default=False)
    javascript_wait_ms: int = Field(default=2000, ge=0, le=30000)

    # Content extraction
    max_content_length: int = Field(
        default=1_000_000, ge=1000, le=10_000_000, description="Max content size in bytes"
    )
    remove_scripts: bool = Field(default=True)
    remove_styles: bool = Field(default=True)

    # Anti-bot handling
    retry_on_403: bool = Field(default=True)
    retry_on_429: bool = Field(default=True)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)


class ScraperInput(BaseModel):
    """Input for web scraper tool."""

    url: str = Field(
        description="URL to scrape",
        examples=["https://example.com", "https://news.ycombinator.com"],
    )
    mode: ScrapeMode = Field(
        default=ScrapeMode.FULL_PAGE,
        description="Scraping mode",
    )
    selector: Optional[str] = Field(
        default=None,
        description="CSS selector or XPath expression (required for CSS_SELECTOR and XPATH modes)",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PLAIN_TEXT,
        description="Output format for content",
    )
    extract_links: bool = Field(default=False, description="Extract all links from page")
    extract_images: bool = Field(default=False, description="Extract all images from page")
    extract_tables: bool = Field(default=False, description="Extract all tables from page")
    use_javascript: bool = Field(
        default=False, description="Use JavaScript rendering (requires playwright)"
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://: {v}")
        return v

    @field_validator("selector")
    @classmethod
    def validate_selector(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate selector is provided when required."""
        # Note: Pydantic v2 changed validation context access
        # We'll do mode-based validation in the tool itself
        return v


class ScraperResult(BaseModel):
    """Output from web scraper tool."""

    success: bool
    content: Optional[ScrapedContent] = None
    error: Optional[str] = None
    cached: bool = False
    robots_allowed: bool = True
    redirected_url: Optional[str] = None


class RobotsTxtChecker:
    """Checks robots.txt for URL access permissions."""

    def __init__(self, user_agent: str):
        """Initialize robots.txt checker.

        Args:
            user_agent: User agent string to check permissions for.
        """
        self.user_agent = user_agent
        self.cache: dict[str, RobotFileParser] = {}

    async def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt.

        Args:
            url: URL to check.

        Returns:
            True if allowed, False otherwise.
        """
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self.cache:
            parser = RobotFileParser()
            robots_url = f"{base_url}/robots.txt"
            parser.set_url(robots_url)

            try:
                # Run in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, parser.read
                )
                self.cache[base_url] = parser
            except Exception:
                # If robots.txt can't be fetched, assume allowed
                # Create a permissive parser
                parser = RobotFileParser()
                parser.parse([])
                self.cache[base_url] = parser

        parser = self.cache[base_url]
        return parser.can_fetch(self.user_agent, url)


class ScraperCache:
    """Simple in-memory cache for scraped content."""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[ScraperResult, float]] = {}

    def get(self, key: str) -> Optional[ScraperResult]:
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

    def set(self, key: str, value: ScraperResult) -> None:
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
    def make_key(url: str, mode: str, selector: Optional[str] = None) -> str:
        """Create cache key from scraping parameters.

        Args:
            url: URL being scraped.
            mode: Scraping mode.
            selector: Optional selector.

        Returns:
            Cache key.
        """
        key_parts = [url, mode]
        if selector:
            key_parts.append(selector)
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()


class ContentExtractor:
    """Extracts and processes content from HTML."""

    def __init__(self, config: ScraperConfig):
        """Initialize content extractor.

        Args:
            config: Scraper configuration.
        """
        self.config = config

    def extract_text(self, html: str) -> str:
        """Extract plain text from HTML.

        Args:
            html: HTML content.

        Returns:
            Plain text.
        """
        text = html

        # Remove script and style tags if configured
        if self.config.remove_scripts:
            text = re.sub(
                r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
            )
        if self.config.remove_styles:
            text = re.sub(
                r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
            )

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode common HTML entities
        entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&apos;": "'",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def extract_title(self, html: str) -> str:
        """Extract title from HTML.

        Args:
            html: HTML content.

        Returns:
            Page title.
        """
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return self.extract_text(match.group(1))
        return ""

    def extract_metadata(self, html: str) -> dict[str, Any]:
        """Extract metadata from HTML.

        Args:
            html: HTML content.

        Returns:
            Dictionary of metadata.
        """
        metadata: dict[str, Any] = {}

        # Extract meta tags
        meta_pattern = r'<meta\s+([^>]*)>'
        for match in re.finditer(meta_pattern, html, re.IGNORECASE):
            attrs_str = match.group(1)

            # Parse attributes
            name_match = re.search(r'(?:name|property)=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            content_match = re.search(r'content=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)

            if name_match and content_match:
                metadata[name_match.group(1)] = content_match.group(1)

        return metadata

    def extract_links(self, html: str, base_url: str) -> list[LinkInfo]:
        """Extract all links from HTML.

        Args:
            html: HTML content.
            base_url: Base URL for resolving relative links.

        Returns:
            List of link information.
        """
        links: list[LinkInfo] = []
        link_pattern = r'<a\s+([^>]*)>(.*?)</a>'

        for match in re.finditer(link_pattern, html, re.IGNORECASE | re.DOTALL):
            attrs_str = match.group(1)
            text = self.extract_text(match.group(2))

            # Extract href
            href_match = re.search(r'href=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            if not href_match:
                continue

            href = href_match.group(1)
            absolute_url = urljoin(base_url, href)

            # Extract title
            title_match = re.search(r'title=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            title = title_match.group(1) if title_match else None

            links.append(LinkInfo(url=absolute_url, text=text, title=title))

        return links

    def extract_images(self, html: str, base_url: str) -> list[ImageInfo]:
        """Extract all images from HTML.

        Args:
            html: HTML content.
            base_url: Base URL for resolving relative URLs.

        Returns:
            List of image information.
        """
        images: list[ImageInfo] = []
        img_pattern = r'<img\s+([^>]*)>'

        for match in re.finditer(img_pattern, html, re.IGNORECASE):
            attrs_str = match.group(1)

            # Extract src
            src_match = re.search(r'src=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            if not src_match:
                continue

            src = src_match.group(1)
            absolute_url = urljoin(base_url, src)

            # Extract other attributes
            alt_match = re.search(r'alt=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            title_match = re.search(r'title=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            width_match = re.search(r'width=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)
            height_match = re.search(r'height=["\']([^"\']+)["\']', attrs_str, re.IGNORECASE)

            images.append(
                ImageInfo(
                    url=absolute_url,
                    alt=alt_match.group(1) if alt_match else None,
                    title=title_match.group(1) if title_match else None,
                    width=width_match.group(1) if width_match else None,
                    height=height_match.group(1) if height_match else None,
                )
            )

        return images

    def extract_tables(self, html: str) -> list[TableData]:
        """Extract all tables from HTML.

        Args:
            html: HTML content.

        Returns:
            List of table data.
        """
        tables: list[TableData] = []
        table_pattern = r'<table[^>]*>(.*?)</table>'

        for table_match in re.finditer(table_pattern, html, re.IGNORECASE | re.DOTALL):
            table_html = table_match.group(1)

            # Extract caption
            caption_match = re.search(
                r'<caption[^>]*>(.*?)</caption>', table_html, re.IGNORECASE | re.DOTALL
            )
            caption = self.extract_text(caption_match.group(1)) if caption_match else None

            # Extract headers
            headers: list[str] = []
            thead_match = re.search(
                r'<thead[^>]*>(.*?)</thead>', table_html, re.IGNORECASE | re.DOTALL
            )
            if thead_match:
                th_pattern = r'<th[^>]*>(.*?)</th>'
                for th_match in re.finditer(th_pattern, thead_match.group(1), re.IGNORECASE | re.DOTALL):
                    headers.append(self.extract_text(th_match.group(1)))

            # Extract rows
            rows: list[list[str]] = []
            tbody_match = re.search(
                r'<tbody[^>]*>(.*?)</tbody>', table_html, re.IGNORECASE | re.DOTALL
            )

            # If no tbody, use entire table
            rows_html = tbody_match.group(1) if tbody_match else table_html

            tr_pattern = r'<tr[^>]*>(.*?)</tr>'
            for tr_match in re.finditer(tr_pattern, rows_html, re.IGNORECASE | re.DOTALL):
                row_html = tr_match.group(1)
                row: list[str] = []

                td_pattern = r'<td[^>]*>(.*?)</td>'
                for td_match in re.finditer(td_pattern, row_html, re.IGNORECASE | re.DOTALL):
                    row.append(self.extract_text(td_match.group(1)))

                if row:  # Only add non-empty rows
                    rows.append(row)

            if headers or rows:  # Only add tables with content
                tables.append(TableData(headers=headers, rows=rows, caption=caption))

        return tables

    def extract_by_css_selector(self, html: str, selector: str) -> str:
        """Extract content matching CSS selector.

        Note: This is a simple implementation. For production use,
        consider using a proper HTML parser like BeautifulSoup or lxml.

        Args:
            html: HTML content.
            selector: CSS selector.

        Returns:
            Extracted content.
        """
        # Simple CSS selector support (class and id only)
        if selector.startswith('.'):
            # Class selector
            class_name = selector[1:]
            pattern = rf'<[^>]+class=["\'][^"\']*\b{re.escape(class_name)}\b[^"\']*["\'][^>]*>(.*?)</[^>]+>'
        elif selector.startswith('#'):
            # ID selector
            id_name = selector[1:]
            pattern = rf'<[^>]+id=["\']({re.escape(id_name)})["\'][^>]*>(.*?)</[^>]+>'
        else:
            # Tag selector
            pattern = rf'<{re.escape(selector)}[^>]*>(.*?)</{re.escape(selector)}>'

        matches = re.finditer(pattern, html, re.IGNORECASE | re.DOTALL)
        content_parts = []

        for match in matches:
            # Get the last group which should be the content
            content_parts.append(match.group(match.lastindex or 1))

        return '\n\n'.join(content_parts)

    def extract_by_xpath(self, html: str, xpath: str) -> str:
        """Extract content matching XPath expression.

        Note: This is a placeholder. XPath requires a proper XML/HTML parser.
        For production use, use lxml or similar library.

        Args:
            html: HTML content.
            xpath: XPath expression.

        Returns:
            Extracted content (empty for now, needs lxml).
        """
        # XPath extraction requires lxml or similar
        # This is a placeholder that returns empty string
        return ""

    def to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown.

        Simple conversion - for better results, use a library like html2text.

        Args:
            html: HTML content.

        Returns:
            Markdown content.
        """
        md = html

        # Headers
        for i in range(6, 0, -1):
            md = re.sub(
                rf'<h{i}[^>]*>(.*?)</h{i}>',
                lambda m: f"{'#' * i} {self.extract_text(m.group(1))}\n\n",
                md,
                flags=re.IGNORECASE | re.DOTALL
            )

        # Bold
        md = re.sub(
            r'<(b|strong)[^>]*>(.*?)</\1>',
            lambda m: f"**{self.extract_text(m.group(2))}**",
            md,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Italic
        md = re.sub(
            r'<(i|em)[^>]*>(.*?)</\1>',
            lambda m: f"*{self.extract_text(m.group(2))}*",
            md,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Links
        md = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            lambda m: f"[{self.extract_text(m.group(2))}]({m.group(1)})",
            md,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Images
        md = re.sub(
            r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*alt=["\']([^"\']*)["\'][^>]*>',
            lambda m: f"![{m.group(2)}]({m.group(1)})",
            md,
            flags=re.IGNORECASE
        )

        # Paragraphs
        md = re.sub(
            r'<p[^>]*>(.*?)</p>',
            lambda m: f"{self.extract_text(m.group(1))}\n\n",
            md,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Line breaks
        md = re.sub(r'<br\s*/?>', '\n', md, flags=re.IGNORECASE)

        # Remove remaining HTML
        md = self.extract_text(md)

        return md


class WebScraperTool(BaseTool[ScraperInput, ScraperResult]):
    """Web scraping tool with multiple extraction modes."""

    metadata = ToolMetadata(
        id="web_scraper",
        name="Web Scraper",
        description="Scrapes web pages with support for multiple extraction modes "
        "(full page, CSS selectors, XPath, links, images, tables). "
        "Includes rate limiting, robots.txt checking, caching, and anti-bot handling.",
        category="search",
        sandbox_required=False,
    )
    input_type = ScraperInput
    output_type = ScraperResult

    def __init__(self, config: ScraperConfig | None = None):
        """Initialize web scraper tool.

        Args:
            config: Scraper configuration.
        """
        if config is None:
            config = ScraperConfig()

        super().__init__(config)
        self.config: ScraperConfig = config  # type: ignore

        # Initialize components
        self.cache = ScraperCache(self.config.cache_ttl_seconds)
        self.robots_checker = RobotsTxtChecker(self.config.user_agent)
        self.content_extractor = ContentExtractor(self.config)

        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_ms / 1000),
            follow_redirects=True,
            max_redirects=self.config.max_redirects,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
            },
        )

        # Playwright browser (lazy loaded)
        self._browser = None
        self._playwright = None

    async def execute(self, input: ScraperInput) -> ScraperResult:
        """Execute web scraping.

        Args:
            input: Scraper input parameters.

        Returns:
            Scraping results.
        """
        # Validate selector for modes that require it
        if input.mode in (ScrapeMode.CSS_SELECTOR, ScrapeMode.XPATH):
            if not input.selector:
                return ScraperResult(
                    success=False,
                    error=f"Selector required for {input.mode.value} mode",
                )

        # Check cache
        cache_key = ScraperCache.make_key(input.url, input.mode.value, input.selector)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        # Check robots.txt if enabled
        robots_allowed = True
        if self.config.respect_robots_txt:
            try:
                robots_allowed = await self.robots_checker.can_fetch(input.url)
                if not robots_allowed:
                    return ScraperResult(
                        success=False,
                        error="Access disallowed by robots.txt",
                        robots_allowed=False,
                    )
            except Exception as e:
                # If robots.txt check fails, log but continue
                pass

        # Fetch content with retries
        html: Optional[str] = None
        final_url = input.url
        last_error: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Rate limiting delay
                if attempt > 0 or self.config.rate_limit_delay_ms > 0:
                    delay = self.config.rate_limit_delay_ms / 1000
                    if attempt > 0:
                        delay *= (self.config.backoff_multiplier ** attempt)
                    await asyncio.sleep(delay)

                # Use JavaScript rendering if requested and available
                if input.use_javascript and self.config.enable_javascript:
                    html, final_url = await self._fetch_with_javascript(input.url)
                else:
                    html, final_url = await self._fetch_with_http(input.url)

                break  # Success

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Handle specific status codes
                if status_code == 403 and self.config.retry_on_403:
                    last_error = f"HTTP 403 Forbidden (attempt {attempt + 1}/{self.config.max_retries + 1})"
                    continue
                elif status_code == 429 and self.config.retry_on_429:
                    last_error = f"HTTP 429 Too Many Requests (attempt {attempt + 1}/{self.config.max_retries + 1})"
                    continue
                else:
                    last_error = f"HTTP {status_code}: {e.response.text[:200]}"
                    break

            except Exception as e:
                last_error = str(e)
                if attempt == self.config.max_retries:
                    break
                continue

        if html is None:
            return ScraperResult(
                success=False,
                error=f"Failed to fetch URL: {last_error}",
            )

        # Check content length
        if len(html) > self.config.max_content_length:
            html = html[:self.config.max_content_length]

        # Extract content based on mode
        start_time = time.time()

        try:
            content = await self._extract_content(html, final_url, input)
            fetch_time = time.time() - start_time

            result = ScraperResult(
                success=True,
                content=ScrapedContent(
                    url=final_url,
                    title=content.get("title", ""),
                    content=content.get("main", ""),
                    links=content.get("links", []),
                    images=content.get("images", []),
                    tables=content.get("tables", []),
                    metadata=content.get("metadata", {}),
                    word_count=len(content.get("main", "").split()),
                    fetch_time=fetch_time,
                ),
                redirected_url=final_url if final_url != input.url else None,
            )

            # Cache result
            self.cache.set(cache_key, result)

            return result

        except Exception as e:
            return ScraperResult(
                success=False,
                error=f"Content extraction failed: {str(e)}",
            )

    async def _fetch_with_http(self, url: str) -> tuple[str, str]:
        """Fetch URL using HTTP client.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (HTML content, final URL after redirects).
        """
        response = await self.client.get(url)
        response.raise_for_status()

        # Get final URL after redirects
        final_url = str(response.url)

        return response.text, final_url

    async def _fetch_with_javascript(self, url: str) -> tuple[str, str]:
        """Fetch URL with JavaScript rendering using Playwright.

        Args:
            url: URL to fetch.

        Returns:
            Tuple of (HTML content, final URL).
        """
        try:
            # Lazy import playwright
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for JavaScript rendering. "
                "Install with: pip install playwright && playwright install"
            )

        # Initialize playwright if needed
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch()

        # Create page and navigate
        page = await self._browser.new_page()

        try:
            await page.goto(url, wait_until="networkidle")

            # Wait for JavaScript execution
            if self.config.javascript_wait_ms > 0:
                await asyncio.sleep(self.config.javascript_wait_ms / 1000)

            # Get content and URL
            html = await page.content()
            final_url = page.url

            return html, final_url

        finally:
            await page.close()

    async def _extract_content(
        self, html: str, url: str, input: ScraperInput
    ) -> dict[str, Any]:
        """Extract content based on scraping mode.

        Args:
            html: HTML content.
            url: Page URL.
            input: Scraper input.

        Returns:
            Dictionary with extracted content.
        """
        result: dict[str, Any] = {
            "title": self.content_extractor.extract_title(html),
            "metadata": self.content_extractor.extract_metadata(html),
            "links": [],
            "images": [],
            "tables": [],
            "main": "",
        }

        # Extract based on mode
        if input.mode == ScrapeMode.FULL_PAGE:
            main_html = html
        elif input.mode == ScrapeMode.CSS_SELECTOR:
            main_html = self.content_extractor.extract_by_css_selector(
                html, input.selector or ""
            )
        elif input.mode == ScrapeMode.XPATH:
            main_html = self.content_extractor.extract_by_xpath(
                html, input.selector or ""
            )
        elif input.mode == ScrapeMode.LINKS:
            result["links"] = self.content_extractor.extract_links(html, url)
            result["main"] = f"Found {len(result['links'])} links"
            return result
        elif input.mode == ScrapeMode.IMAGES:
            result["images"] = self.content_extractor.extract_images(html, url)
            result["main"] = f"Found {len(result['images'])} images"
            return result
        elif input.mode == ScrapeMode.TABLES:
            result["tables"] = self.content_extractor.extract_tables(html)
            result["main"] = f"Found {len(result['tables'])} tables"
            return result
        else:
            main_html = html

        # Format output
        if input.output_format == OutputFormat.HTML:
            result["main"] = main_html
        elif input.output_format == OutputFormat.MARKDOWN:
            result["main"] = self.content_extractor.to_markdown(main_html)
        elif input.output_format == OutputFormat.PLAIN_TEXT:
            result["main"] = self.content_extractor.extract_text(main_html)
        elif input.output_format == OutputFormat.JSON:
            # For JSON, return structured data
            result["main"] = self.content_extractor.extract_text(main_html)

        # Extract additional elements if requested
        if input.extract_links:
            result["links"] = self.content_extractor.extract_links(html, url)
        if input.extract_images:
            result["images"] = self.content_extractor.extract_images(html, url)
        if input.extract_tables:
            result["tables"] = self.content_extractor.extract_tables(html)

        return result

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.client.aclose()

        # Cleanup playwright if used
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
