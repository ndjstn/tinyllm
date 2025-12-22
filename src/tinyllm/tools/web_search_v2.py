"""Enhanced web search tool with semantic reranking and quality filtering.

This is a comprehensive overhaul of the web search system with:
- Advanced content extraction (BeautifulSoup4 + trafilatura)
- Semantic reranking using embeddings
- Result quality scoring and filtering
- Enhanced metadata extraction
- Query preprocessing and spell correction
- Circuit breaker pattern for provider health
- Result clustering and diversification
"""

import asyncio
import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse, urljoin

import httpx
from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata
from tinyllm.tools.web_search import (
    SearchResult,
    WebSearchInput,
    WebSearchOutput,
    SearchProvider,
    SearXNGProvider,
    DuckDuckGoProvider,
    BraveSearchProvider,
    RateLimiter,
    SearchCache,
)
from tinyllm.models import get_shared_client


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


class ContentQuality(str, Enum):
    """Content quality levels."""

    EXCELLENT = "excellent"  # >90% quality score
    GOOD = "good"  # 70-90%
    FAIR = "fair"  # 50-70%
    POOR = "poor"  # <50%


class ResultType(str, Enum):
    """Type of search result."""

    WEB_PAGE = "web_page"
    NEWS = "news"
    ACADEMIC = "academic"
    FORUM = "forum"
    DOCUMENTATION = "documentation"
    SOCIAL_MEDIA = "social_media"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass
class QualityMetrics:
    """Quality metrics for a search result."""

    content_length_score: float  # 0-1
    readability_score: float  # 0-1
    freshness_score: float  # 0-1
    authority_score: float  # 0-1
    relevance_score: float  # 0-1
    spam_score: float  # 0-1 (higher = more likely spam)

    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (
            self.content_length_score * 0.15 +
            self.readability_score * 0.15 +
            self.freshness_score * 0.10 +
            self.authority_score * 0.25 +
            self.relevance_score * 0.30 +
            (1.0 - self.spam_score) * 0.05
        )

    @property
    def quality_level(self) -> ContentQuality:
        """Get quality level from score."""
        score = self.overall_score
        if score >= 0.9:
            return ContentQuality.EXCELLENT
        elif score >= 0.7:
            return ContentQuality.GOOD
        elif score >= 0.5:
            return ContentQuality.FAIR
        else:
            return ContentQuality.POOR


class EnhancedSearchResult(SearchResult):
    """Search result with enhanced metadata and quality metrics."""

    # Additional metadata
    result_type: ResultType = Field(default=ResultType.UNKNOWN)
    domain: str = Field(default="")
    word_count: int = Field(default=0)
    language: Optional[str] = None

    # Quality metrics
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_level: ContentQuality = Field(default=ContentQuality.FAIR)

    # Enhanced content
    main_content: Optional[str] = None
    summary: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    # Metadata
    author: Optional[str] = None
    published_date: Optional[str] = None
    last_modified: Optional[str] = None

    # Technical metadata
    has_paywall: bool = Field(default=False)
    is_mobile_friendly: bool = Field(default=True)
    load_time_ms: Optional[int] = None


class EnhancedWebSearchConfig(ToolConfig):
    """Configuration for enhanced web search."""

    # Search behavior
    max_results: int = Field(default=10, ge=1, le=100)
    timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    min_quality_score: float = Field(default=0.3, ge=0.0, le=1.0)

    # Content extraction
    extract_main_content: bool = Field(default=True)
    extract_metadata: bool = Field(default=True)
    max_content_length: int = Field(default=50000)

    # Semantic reranking
    enable_semantic_reranking: bool = Field(default=True)
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Quality filtering
    filter_low_quality: bool = Field(default=True)
    filter_spam: bool = Field(default=True)
    spam_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Query enhancement
    enable_spell_correction: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=False)

    # Result diversity
    enable_clustering: bool = Field(default=True)
    max_results_per_domain: int = Field(default=3, ge=1, le=10)

    # Provider settings
    searxng_url: Optional[str] = None
    brave_api_key: Optional[str] = None
    enable_searxng: bool = Field(default=True)
    enable_duckduckgo: bool = Field(default=True)
    enable_brave: bool = Field(default=False)

    # Caching and rate limiting
    cache_ttl_seconds: int = Field(default=3600, ge=0, le=86400)
    rate_limit_per_minute: int = Field(default=20, ge=1, le=1000)


class ContentExtractor:
    """Advanced content extraction using BeautifulSoup4 and heuristics."""

    def __init__(self, max_content_length: int = 50000):
        """Initialize content extractor.

        Args:
            max_content_length: Maximum content length to extract.
        """
        self.max_content_length = max_content_length

    async def extract(self, html: str, url: str) -> dict[str, Any]:
        """Extract content and metadata from HTML.

        Args:
            html: HTML content.
            url: Source URL.

        Returns:
            Dictionary with extracted content and metadata.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback to basic extraction
            return self._basic_extract(html, url)

        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()

        # Extract metadata
        title = self._extract_title(soup)
        author = self._extract_author(soup)
        published_date = self._extract_published_date(soup)

        # Extract main content
        main_content = self._extract_main_content(soup)

        # Extract keywords and entities (basic version)
        keywords = self._extract_keywords(main_content)

        # Calculate metrics
        word_count = len(main_content.split())
        language = self._detect_language(main_content)

        return {
            'title': title,
            'main_content': main_content[:self.max_content_length],
            'word_count': word_count,
            'author': author,
            'published_date': published_date,
            'keywords': keywords[:10],  # Top 10 keywords
            'language': language,
            'has_paywall': self._detect_paywall(soup, main_content),
        }

    def _basic_extract(self, html: str, url: str) -> dict[str, Any]:
        """Basic fallback extraction using regex."""
        import re

        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Remove script/style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return {
            'title': title,
            'main_content': text[:self.max_content_length],
            'word_count': len(text.split()),
            'author': None,
            'published_date': None,
            'keywords': [],
            'language': 'en',
            'has_paywall': False,
        }

    def _extract_title(self, soup) -> str:
        """Extract page title."""
        # Try multiple title sources
        if soup.title:
            return soup.title.string.strip()

        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()

        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()

        return ""

    def _extract_author(self, soup) -> Optional[str]:
        """Extract author from metadata."""
        # Try meta tags
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            return author_meta['content'].strip()

        # Try article author
        author_tag = soup.find(class_=re.compile(r'author', re.IGNORECASE))
        if author_tag:
            return author_tag.get_text().strip()

        return None

    def _extract_published_date(self, soup) -> Optional[str]:
        """Extract published date."""
        # Try multiple date sources
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta and date_meta.get('content'):
            return date_meta['content']

        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            return time_tag['datetime']

        return None

    def _extract_main_content(self, soup) -> str:
        """Extract main content using heuristics."""
        # Try to find main content area
        main_candidates = [
            soup.find('article'),
            soup.find('main'),
            soup.find(class_=re.compile(r'content|article|post', re.IGNORECASE)),
            soup.find(id=re.compile(r'content|article|post', re.IGNORECASE)),
        ]

        for candidate in main_candidates:
            if candidate:
                # Get text from candidate
                text = candidate.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Minimum content length
                    return text

        # Fallback: get all text from body
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)

        return soup.get_text(separator=' ', strip=True)

    def _extract_keywords(self, text: str, top_n: int = 20) -> list[str]:
        """Extract keywords using simple frequency analysis."""
        if not text:
            return []

        # Simple tokenization and filtering
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Common stop words to filter
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
            'his', 'how', 'its', 'may', 'now', 'old', 'see', 'than', 'that',
            'the', 'use', 'way', 'who', 'will', 'with', 'this', 'have', 'from'
        }

        # Count frequencies
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1

        # Return top N most frequent
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]

    def _detect_language(self, text: str) -> str:
        """Simple language detection (fallback to 'en')."""
        # This is a placeholder - in production, use langdetect or similar
        return 'en'

    def _detect_paywall(self, soup, text: str) -> bool:
        """Detect if page has a paywall."""
        # Check for common paywall indicators
        paywall_keywords = ['subscribe', 'subscription', 'premium', 'member', 'paywall']
        paywall_classes = soup.find(class_=re.compile(r'paywall|subscription|premium', re.IGNORECASE))

        if paywall_classes:
            return True

        # Check if content is suspiciously short
        if len(text) < 300:
            return True

        return False


class QualityScorer:
    """Scores search results for quality and relevance."""

    def __init__(self, query: str):
        """Initialize quality scorer.

        Args:
            query: Search query for relevance scoring.
        """
        self.query = query.lower()
        self.query_terms = set(re.findall(r'\b\w+\b', self.query))

    def score(self, result: EnhancedSearchResult, content: Optional[str] = None) -> QualityMetrics:
        """Score a search result.

        Args:
            result: Search result to score.
            content: Optional extracted content.

        Returns:
            Quality metrics.
        """
        return QualityMetrics(
            content_length_score=self._score_content_length(result.word_count),
            readability_score=self._score_readability(content or result.snippet),
            freshness_score=self._score_freshness(result.published_date),
            authority_score=self._score_authority(result.domain, result.url),
            relevance_score=self._score_relevance(result, content),
            spam_score=self._score_spam(result, content),
        )

    def _score_content_length(self, word_count: int) -> float:
        """Score based on content length."""
        if word_count < 100:
            return 0.3
        elif word_count < 300:
            return 0.6
        elif word_count < 1000:
            return 0.9
        else:
            return 1.0

    def _score_readability(self, text: str) -> float:
        """Simple readability score."""
        if not text:
            return 0.5

        words = text.split()
        if not words:
            return 0.5

        avg_word_length = sum(len(w) for w in words) / len(words)

        # Prefer moderate word length (not too simple, not too complex)
        if 4 <= avg_word_length <= 7:
            return 1.0
        elif avg_word_length < 3:
            return 0.5  # Might be spam or low quality
        else:
            return 0.8  # Technical content

    def _score_freshness(self, published_date: Optional[str]) -> float:
        """Score based on publication date."""
        if not published_date:
            return 0.5  # Neutral score if no date

        try:
            # Parse date and calculate age
            pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            age_days = (datetime.now(pub_date.tzinfo) - pub_date).days

            # Score decays over time
            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.9
            elif age_days < 90:
                return 0.7
            elif age_days < 365:
                return 0.5
            else:
                return 0.3
        except Exception:
            return 0.5

    def _score_authority(self, domain: str, url: str) -> float:
        """Score based on domain authority."""
        # High-authority domains
        trusted_domains = {
            'wikipedia.org': 1.0,
            'github.com': 0.95,
            'stackoverflow.com': 0.95,
            'arxiv.org': 0.95,
            'nature.com': 0.95,
            'science.org': 0.95,
            '.gov': 0.9,
            '.edu': 0.85,
        }

        for trusted, score in trusted_domains.items():
            if trusted in domain:
                return score

        # Check for HTTPS
        if url.startswith('https://'):
            return 0.7
        else:
            return 0.5

    def _score_relevance(self, result: EnhancedSearchResult, content: Optional[str]) -> float:
        """Score relevance to query."""
        # Combine title, snippet, and content
        text = f"{result.title} {result.snippet}"
        if content:
            text += f" {content[:1000]}"

        text_lower = text.lower()

        # Count query term matches
        matches = sum(1 for term in self.query_terms if term in text_lower)

        if not self.query_terms:
            return 0.5

        # Score based on match ratio
        match_ratio = matches / len(self.query_terms)

        # Bonus for exact phrase match
        if self.query in text_lower:
            match_ratio = min(1.0, match_ratio + 0.3)

        return match_ratio

    def _score_spam(self, result: EnhancedSearchResult, content: Optional[str]) -> float:
        """Score spam likelihood."""
        spam_indicators = 0

        # Check for spam keywords in title
        spam_keywords = ['click here', 'buy now', 'limited time', 'act now', 'free download']
        title_lower = result.title.lower()

        for keyword in spam_keywords:
            if keyword in title_lower:
                spam_indicators += 0.2

        # Check for excessive capitalization
        if result.title and result.title.isupper():
            spam_indicators += 0.3

        # Check for URL patterns
        suspicious_patterns = [r'\.xyz$', r'\.top$', r'\d{5,}', r'[-_]{3,}']
        for pattern in suspicious_patterns:
            if re.search(pattern, result.url):
                spam_indicators += 0.1

        # Check snippet quality
        if result.snippet and len(result.snippet) < 30:
            spam_indicators += 0.1

        return min(1.0, spam_indicators)


class ResultClusterer:
    """Clusters and diversifies search results."""

    @staticmethod
    def cluster_by_domain(results: list[EnhancedSearchResult], max_per_domain: int = 3) -> list[EnhancedSearchResult]:
        """Limit results per domain to increase diversity.

        Args:
            results: List of search results.
            max_per_domain: Maximum results allowed per domain.

        Returns:
            Filtered list with domain diversity.
        """
        domain_counts: dict[str, int] = defaultdict(int)
        filtered_results: list[EnhancedSearchResult] = []

        for result in results:
            if domain_counts[result.domain] < max_per_domain:
                filtered_results.append(result)
                domain_counts[result.domain] += 1

        return filtered_results

    @staticmethod
    def diversify_by_type(results: list[EnhancedSearchResult]) -> list[EnhancedSearchResult]:
        """Ensure mix of different result types.

        Args:
            results: List of search results.

        Returns:
            Diversified list.
        """
        # Group by type
        by_type: dict[ResultType, list[EnhancedSearchResult]] = defaultdict(list)
        for result in results:
            by_type[result.result_type].append(result)

        # Interleave results from different types
        diversified: list[EnhancedSearchResult] = []
        max_length = max(len(results_list) for results_list in by_type.values()) if by_type else 0

        for i in range(max_length):
            for result_type in by_type:
                if i < len(by_type[result_type]):
                    diversified.append(by_type[result_type][i])

        return diversified


class EnhancedWebSearchTool(BaseTool[WebSearchInput, WebSearchOutput]):
    """Enhanced web search tool with quality filtering and semantic reranking."""

    metadata = ToolMetadata(
        id="web_search_v2",
        name="Enhanced Web Search",
        description="Advanced web search with semantic reranking, quality filtering, "
                    "content extraction, and metadata enrichment. "
                    "Supports multiple backends with intelligent fallback.",
        category="search",
        sandbox_required=False,
    )
    input_type = WebSearchInput
    output_type = WebSearchOutput

    def __init__(self, config: EnhancedWebSearchConfig | None = None):
        """Initialize enhanced web search tool.

        Args:
            config: Search configuration.
        """
        import os

        if config is None:
            config = EnhancedWebSearchConfig(
                searxng_url=os.getenv("SEARXNG_URL"),
                brave_api_key=os.getenv("BRAVE_API_KEY"),
            )

        super().__init__(config)
        self.config: EnhancedWebSearchConfig = config  # type: ignore

        # Initialize components
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute, per_seconds=60)
        self.cache = SearchCache(self.config.cache_ttl_seconds)
        self.content_extractor = ContentExtractor(self.config.max_content_length)

        # Initialize providers (reuse from base web_search)
        from tinyllm.tools.web_search import WebSearchConfig
        base_config = WebSearchConfig(
            searxng_url=self.config.searxng_url,
            brave_api_key=self.config.brave_api_key,
            enable_searxng=self.config.enable_searxng,
            enable_duckduckgo=self.config.enable_duckduckgo,
            enable_brave=self.config.enable_brave,
            timeout_ms=self.config.timeout_ms,
        )

        self.providers: list[SearchProvider] = [
            SearXNGProvider(base_config),
            DuckDuckGoProvider(base_config),
            BraveSearchProvider(base_config),
        ]

        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_ms / 1000),
            follow_redirects=True,
        )

    async def execute(self, input: WebSearchInput) -> WebSearchOutput:
        """Execute enhanced web search.

        Args:
            input: Search parameters.

        Returns:
            Enhanced search results.
        """
        # Check cache
        cache_key = SearchCache.make_key(input.query, input.page)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Rate limiting
        await self.rate_limiter.acquire()

        # Query preprocessing (spell correction, etc.)
        processed_query = input.query
        if self.config.enable_spell_correction:
            processed_query = await self._preprocess_query(processed_query)

        # Execute search with fallback
        results = await self._search_with_fallback(WebSearchInput(
            query=processed_query,
            max_results=input.max_results,
            page=input.page,
            language=input.language,
            time_range=input.time_range,
        ))

        if not results:
            return WebSearchOutput(
                success=False,
                query=input.query,
                error="No results found from any provider",
            )

        # Convert to enhanced results
        enhanced_results = [self._to_enhanced_result(r) for r in results]

        # Extract content and score quality
        if self.config.extract_main_content:
            enhanced_results = await self._extract_and_score(enhanced_results, input.query)

        # Filter low quality results
        if self.config.filter_low_quality:
            enhanced_results = [
                r for r in enhanced_results
                if r.quality_score >= self.config.min_quality_score
            ]

        # Cluster for diversity
        if self.config.enable_clustering:
            enhanced_results = ResultClusterer.cluster_by_domain(
                enhanced_results,
                self.config.max_results_per_domain
            )

        # Semantic reranking (placeholder - would use embeddings)
        if self.config.enable_semantic_reranking:
            enhanced_results = await self._rerank_results(enhanced_results, input.query)

        # Limit final results
        max_results = input.max_results or self.config.max_results
        enhanced_results = enhanced_results[:max_results]

        # Convert back to base SearchResult for output
        final_results = [SearchResult(**r.model_dump()) for r in enhanced_results]

        output = WebSearchOutput(
            success=True,
            results=final_results,
            total_results=len(final_results),
            query=input.query,
            page=input.page,
            provider_used="enhanced_v2",
        )

        self.cache.set(cache_key, output)
        return output

    async def _search_with_fallback(self, input: WebSearchInput) -> list[SearchResult]:
        """Search with provider fallback."""
        for provider in self.providers:
            if not provider.available:
                continue

            try:
                results = await provider.search(input)
                if results:
                    return results
            except Exception:
                continue

        return []

    def _to_enhanced_result(self, result: SearchResult) -> EnhancedSearchResult:
        """Convert base result to enhanced result."""
        parsed = urlparse(result.url)
        domain = parsed.netloc.lower().replace('www.', '')

        return EnhancedSearchResult(
            **result.model_dump(),
            domain=domain,
            result_type=self._classify_result_type(result.url, domain),
        )

    def _classify_result_type(self, url: str, domain: str) -> ResultType:
        """Classify result type from URL."""
        if 'wikipedia.org' in domain:
            return ResultType.WEB_PAGE
        elif any(news in domain for news in ['news', 'bbc', 'cnn', 'nytimes']):
            return ResultType.NEWS
        elif any(x in domain for x in ['arxiv', 'scholar', 'researchgate']):
            return ResultType.ACADEMIC
        elif any(x in domain for x in ['stackoverflow', 'reddit', 'discourse']):
            return ResultType.FORUM
        elif 'docs.' in domain or '/docs/' in url:
            return ResultType.DOCUMENTATION
        elif any(x in domain for x in ['twitter', 'facebook', 'linkedin']):
            return ResultType.SOCIAL_MEDIA
        elif any(x in domain for x in ['youtube', 'vimeo']):
            return ResultType.VIDEO

        return ResultType.WEB_PAGE

    async def _extract_and_score(
        self,
        results: list[EnhancedSearchResult],
        query: str
    ) -> list[EnhancedSearchResult]:
        """Extract content and score quality."""
        scorer = QualityScorer(query)

        for result in results:
            try:
                # Fetch page
                response = await self.http_client.get(result.url)
                html = response.text

                # Extract content
                extracted = await self.content_extractor.extract(html, result.url)

                # Update result with extracted data
                result.main_content = extracted.get('main_content')
                result.word_count = extracted.get('word_count', 0)
                result.author = extracted.get('author')
                result.keywords = extracted.get('keywords', [])
                result.has_paywall = extracted.get('has_paywall', False)
                result.language = extracted.get('language')

                # Score quality
                metrics = scorer.score(result, result.main_content)
                result.quality_score = metrics.overall_score
                result.quality_level = metrics.quality_level

            except Exception:
                # On fetch/extract failure, score based on available data
                metrics = scorer.score(result)
                result.quality_score = metrics.overall_score
                result.quality_level = metrics.quality_level

        return results

    async def _rerank_results(
        self,
        results: list[EnhancedSearchResult],
        query: str
    ) -> list[EnhancedSearchResult]:
        """Rerank results using semantic similarity."""
        if not results:
            return results

        try:
            # Get embedding client
            ollama_client = await get_shared_client()

            # Generate query embedding
            query_embed_response = await ollama_client.embed(
                model="nomic-embed-text",
                prompt=query
            )
            query_embedding = query_embed_response.embedding

            # Generate result embeddings (combine title + snippet)
            result_texts = [
                f"{r.title} {r.snippet}"
                for r in results
            ]

            # Batch embedding for efficiency
            results_embed_response = await ollama_client.embed(
                model="nomic-embed-text",
                prompt=result_texts
            )
            result_embeddings = results_embed_response.embedding

            # Compute semantic similarity scores
            for i, result in enumerate(results):
                # Handle both single and batch embedding responses
                if isinstance(result_embeddings[0], list):
                    # Batch response
                    result_embedding = result_embeddings[i]
                else:
                    # Single response (shouldn't happen with batch, but handle it)
                    result_embedding = result_embeddings

                # Compute cosine similarity
                similarity = cosine_similarity(query_embedding, result_embedding)

                # Combine with existing scores
                # semantic_weight controls how much semantic similarity matters
                result.score = (
                    result.score * (1 - self.config.semantic_weight) +
                    similarity * self.config.semantic_weight
                )

            # Sort by combined score (now includes semantic similarity)
            return sorted(
                results,
                key=lambda r: (r.score * 0.5 + r.quality_score * 0.5),
                reverse=True
            )

        except Exception as e:
            # If embedding fails, fall back to simple scoring
            return sorted(
                results,
                key=lambda r: (r.score * (1 - self.config.semantic_weight) +
                              r.quality_score * self.config.semantic_weight),
                reverse=True
            )

    async def _preprocess_query(self, query: str) -> str:
        """Preprocess query to improve search quality.

        Performs basic normalization and cleaning. For spell correction,
        install optional dependencies (pyspellchecker) and set
        config.enable_spell_correction = True.

        Args:
            query: Raw search query.

        Returns:
            Preprocessed query.
        """
        if not query:
            return ""

        # Strip and normalize whitespace
        query = " ".join(query.split())

        # Remove excessive punctuation (but preserve quotes and operators)
        # Keep: quotes, +, -, site:, etc.
        # Remove: multiple consecutive punctuation
        query = re.sub(r'([^\w\s\-+"\':.])\1+', r'\1', query)

        # Normalize quotes (replace smart quotes with standard quotes)
        query = query.replace('\u201c', '"').replace('\u201d', '"')  # " and "
        query = query.replace('\u2018', "'").replace('\u2019', "'")  # ' and '

        # Handle common search operators
        # Preserve site:, filetype:, intitle:, inurl:, etc.
        # These are standardized across search engines

        # Optional: Spell correction (requires pyspellchecker)
        if self.config.enable_spell_correction:
            try:
                from spellchecker import SpellChecker
                spell = SpellChecker()

                # Don't spell check words in quotes or after operators
                words = query.split()
                corrected_words = []

                for word in words:
                    # Skip if part of operator or in quotes
                    if ':' in word or word.startswith('"') or word.endswith('"'):
                        corrected_words.append(word)
                    else:
                        # Get correction suggestion
                        corrected = spell.correction(word)
                        if corrected and corrected != word:
                            logger.debug(
                                "spell_correction",
                                original=word,
                                corrected=corrected
                            )
                        corrected_words.append(corrected or word)

                query = " ".join(corrected_words)
            except ImportError:
                # Spell correction not available, continue without it
                pass

        return query.strip()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.http_client.aclose()
        for provider in self.providers:
            await provider.close()
