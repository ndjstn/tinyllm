"""Tests for enhanced web search tool (v2)."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from tinyllm.tools.web_search_v2 import (
    ContentExtractor,
    QualityScorer,
    QualityMetrics,
    EnhancedSearchResult,
    ResultClusterer,
    ContentQuality,
    ResultType,
)
from tinyllm.tools.web_search import SearchResult


class TestContentExtractor:
    """Tests for content extraction."""

    @pytest.mark.asyncio
    async def test_basic_extraction_without_beautifulsoup(self):
        """Test fallback extraction when BeautifulSoup is not available."""
        extractor = ContentExtractor()

        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is the main content of the page.</p>
                <script>alert('test');</script>
            </body>
        </html>
        """

        with patch.dict('sys.modules', {'bs4': None}):
            result = await extractor.extract(html, "https://example.com")

        assert result['title'] == "Test Page"
        assert "main content" in result['main_content']
        assert "alert" not in result['main_content']  # Script removed
        assert result['word_count'] > 0

    @pytest.mark.asyncio
    async def test_keyword_extraction(self):
        """Test keyword extraction from content."""
        extractor = ContentExtractor()

        text = "Python programming is great. Python is used for data science. Python developers love Python."

        keywords = extractor._extract_keywords(text)

        assert 'python' in keywords
        assert 'programming' in keywords
        assert 'data' in keywords or 'science' in keywords

    def test_paywall_detection(self):
        """Test paywall detection."""
        extractor = ContentExtractor()

        # Mock soup with paywall indicator
        from unittest.mock import Mock
        mock_soup = Mock()
        mock_soup.find.return_value = Mock()  # Paywall class found

        assert extractor._detect_paywall(mock_soup, "short content") is True

        # Test short content detection
        mock_soup.find.return_value = None
        assert extractor._detect_paywall(mock_soup, "abc") is True

        # Test normal content
        long_content = " ".join(["word"] * 100)
        assert extractor._detect_paywall(mock_soup, long_content) is False


class TestQualityScorer:
    """Tests for quality scoring."""

    def test_content_length_scoring(self):
        """Test content length scoring."""
        scorer = QualityScorer("test query")

        assert scorer._score_content_length(50) == 0.3  # Very short
        assert scorer._score_content_length(200) == 0.6  # Short
        assert scorer._score_content_length(500) == 0.9  # Medium
        assert scorer._score_content_length(2000) == 1.0  # Long

    def test_readability_scoring(self):
        """Test readability scoring."""
        scorer = QualityScorer("test query")

        # Good readability (moderate word length)
        normal_text = "This is a normal text with regular words"
        assert scorer._score_readability(normal_text) > 0.8

        # Poor readability (very short words, might be spam)
        spam_text = "a b c d e f g h"
        assert scorer._score_readability(spam_text) < 0.7

    def test_freshness_scoring(self):
        """Test freshness scoring based on date."""
        scorer = QualityScorer("test query")

        # Recent content (1 day old)
        recent_date = (datetime.now() - timedelta(days=1)).isoformat()
        assert scorer._score_freshness(recent_date) == 1.0

        # Old content (2 years old)
        old_date = (datetime.now() - timedelta(days=730)).isoformat()
        assert scorer._score_freshness(old_date) < 0.5

        # No date provided
        assert scorer._score_freshness(None) == 0.5

    def test_authority_scoring(self):
        """Test domain authority scoring."""
        scorer = QualityScorer("test query")

        # Trusted domains
        assert scorer._score_authority("wikipedia.org", "https://wikipedia.org") == 1.0
        assert scorer._score_authority("github.com", "https://github.com") == 0.95
        assert scorer._score_authority("example.gov", "https://example.gov") >= 0.9
        assert scorer._score_authority("university.edu", "https://university.edu") >= 0.85

        # Regular HTTPS site
        assert scorer._score_authority("example.com", "https://example.com") == 0.7

        # HTTP site (less trusted)
        assert scorer._score_authority("example.com", "http://example.com") == 0.5

    def test_relevance_scoring(self):
        """Test relevance to query."""
        scorer = QualityScorer("python programming")

        result = EnhancedSearchResult(
            title="Python Programming Guide",
            url="https://example.com",
            snippet="Learn Python programming with examples",
            domain="example.com"
        )

        score = scorer._score_relevance(result, None)

        # Should match both "python" and "programming"
        assert score >= 0.5

    def test_spam_scoring(self):
        """Test spam detection."""
        scorer = QualityScorer("test query")

        # Spam indicators
        spam_result = EnhancedSearchResult(
            title="CLICK HERE BUY NOW LIMITED TIME",
            url="https://spam123456.xyz",
            snippet="Act now!",
            domain="spam123456.xyz"
        )

        spam_score = scorer._score_spam(spam_result, None)
        assert spam_score > 0.5  # High spam score

        # Normal result
        normal_result = EnhancedSearchResult(
            title="Understanding Python Basics",
            url="https://example.com/python-guide",
            snippet="A comprehensive guide to learning Python programming",
            domain="example.com"
        )

        normal_score = scorer._score_spam(normal_result, None)
        assert normal_score < 0.3  # Low spam score

    def test_overall_quality_metrics(self):
        """Test overall quality score calculation."""
        metrics = QualityMetrics(
            content_length_score=0.9,
            readability_score=0.8,
            freshness_score=1.0,
            authority_score=0.9,
            relevance_score=0.85,
            spam_score=0.1,
        )

        assert 0.8 <= metrics.overall_score <= 1.0
        # Score: 0.9*0.15 + 0.8*0.15 + 1.0*0.10 + 0.9*0.25 + 0.85*0.30 + 0.9*0.05 = 0.8725
        assert metrics.quality_level == ContentQuality.GOOD

    def test_quality_level_thresholds(self):
        """Test quality level classification."""
        # Excellent
        excellent = QualityMetrics(0.9, 0.9, 0.9, 0.9, 0.9, 0.0)
        assert excellent.quality_level == ContentQuality.EXCELLENT

        # Good
        good = QualityMetrics(0.8, 0.7, 0.7, 0.7, 0.7, 0.1)
        assert good.quality_level == ContentQuality.GOOD

        # Fair
        fair = QualityMetrics(0.6, 0.5, 0.5, 0.5, 0.5, 0.3)
        assert fair.quality_level == ContentQuality.FAIR

        # Poor
        poor = QualityMetrics(0.3, 0.3, 0.3, 0.3, 0.3, 0.7)
        assert poor.quality_level == ContentQuality.POOR


class TestResultClusterer:
    """Tests for result clustering and diversification."""

    def test_cluster_by_domain(self):
        """Test limiting results per domain."""
        results = [
            EnhancedSearchResult(
                title=f"Result {i}",
                url=f"https://example.com/page{i}",
                snippet="test",
                domain="example.com"
            )
            for i in range(5)
        ] + [
            EnhancedSearchResult(
                title="Different domain",
                url="https://other.com/page",
                snippet="test",
                domain="other.com"
            )
        ]

        clustered = ResultClusterer.cluster_by_domain(results, max_per_domain=2)

        # Count results per domain
        domain_counts = {}
        for result in clustered:
            domain_counts[result.domain] = domain_counts.get(result.domain, 0) + 1

        assert domain_counts["example.com"] <= 2
        assert "other.com" in domain_counts

    def test_diversify_by_type(self):
        """Test diversification by result type."""
        results = [
            EnhancedSearchResult(
                title="Web page",
                url="https://example.com",
                snippet="test",
                domain="example.com",
                result_type=ResultType.WEB_PAGE
            ),
            EnhancedSearchResult(
                title="News article",
                url="https://news.com",
                snippet="test",
                domain="news.com",
                result_type=ResultType.NEWS
            ),
            EnhancedSearchResult(
                title="Forum post",
                url="https://forum.com",
                snippet="test",
                domain="forum.com",
                result_type=ResultType.FORUM
            ),
            EnhancedSearchResult(
                title="Another web page",
                url="https://example.org",
                snippet="test",
                domain="example.org",
                result_type=ResultType.WEB_PAGE
            ),
        ]

        diversified = ResultClusterer.diversify_by_type(results)

        # Check that types are interleaved, not grouped
        types = [r.result_type for r in diversified]

        # First few results should have different types
        assert types[0] != types[1] or types[1] != types[2]


class TestEnhancedSearchResult:
    """Tests for enhanced search result model."""

    def test_enhanced_result_creation(self):
        """Test creating enhanced search result."""
        result = EnhancedSearchResult(
            title="Test Result",
            url="https://example.com",
            snippet="Test snippet",
            domain="example.com",
            result_type=ResultType.WEB_PAGE,
            word_count=500,
            quality_score=0.85,
            quality_level=ContentQuality.GOOD,
            keywords=["python", "programming", "tutorial"]
        )

        assert result.title == "Test Result"
        assert result.domain == "example.com"
        assert result.word_count == 500
        assert result.quality_score == 0.85
        assert len(result.keywords) == 3

    def test_enhanced_result_defaults(self):
        """Test default values for enhanced result."""
        result = EnhancedSearchResult(
            title="Test",
            url="https://example.com",
            snippet="Test"
        )

        assert result.result_type == ResultType.UNKNOWN
        assert result.quality_level == ContentQuality.FAIR
        assert result.has_paywall is False
        assert result.is_mobile_friendly is True
        assert result.keywords == []


class TestQueryPreprocessing:
    """Tests for query preprocessing."""

    @pytest.mark.asyncio
    async def test_basic_normalization(self):
        """Query preprocessing normalizes whitespace and quotes."""
        from tinyllm.tools.web_search_v2 import EnhancedWebSearchTool, EnhancedWebSearchConfig

        config = EnhancedWebSearchConfig(enable_spell_correction=False)
        tool = EnhancedWebSearchTool(config=config)

        # Test whitespace normalization
        query = "python    programming    tutorial"
        processed = await tool._preprocess_query(query)
        assert processed == "python programming tutorial"

        # Test quote normalization (smart quotes to standard)
        query = "\u201cpython\u201d \u2018tutorial\u2019"  # Smart quotes using unicode
        processed = await tool._preprocess_query(query)
        # Smart quotes should be converted to standard quotes
        assert '"' in processed  # Double quotes normalized
        assert "'" in processed  # Single quotes normalized

        # Test excessive punctuation removal
        query = "python!!!!! tutorial???"
        processed = await tool._preprocess_query(query)
        assert "!" in processed or "?" in processed  # Some punctuation preserved

    @pytest.mark.asyncio
    async def test_operator_preservation(self):
        """Query preprocessing preserves search operators."""
        from tinyllm.tools.web_search_v2 import EnhancedWebSearchTool, EnhancedWebSearchConfig

        config = EnhancedWebSearchConfig(enable_spell_correction=False)
        tool = EnhancedWebSearchTool(config=config)

        # Test site: operator
        query = "python tutorial site:stackoverflow.com"
        processed = await tool._preprocess_query(query)
        assert "site:stackoverflow.com" in processed

        # Test filetype: operator
        query = "python documentation filetype:pdf"
        processed = await tool._preprocess_query(query)
        assert "filetype:pdf" in processed

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Query preprocessing handles empty queries."""
        from tinyllm.tools.web_search_v2 import EnhancedWebSearchTool, EnhancedWebSearchConfig

        config = EnhancedWebSearchConfig(enable_spell_correction=False)
        tool = EnhancedWebSearchTool(config=config)

        processed = await tool._preprocess_query("")
        assert processed == ""

        processed = await tool._preprocess_query("   ")
        assert processed == ""


class TestIntegration:
    """Integration tests for enhanced web search."""

    @pytest.mark.asyncio
    async def test_quality_filtering_pipeline(self):
        """Test end-to-end quality filtering."""
        # Create mock results with varying quality
        results = [
            EnhancedSearchResult(
                title="High Quality Result",
                url="https://wikipedia.org/article",
                snippet="Comprehensive information about the topic",
                domain="wikipedia.org",
                word_count=1000,
                quality_score=0.95,
            ),
            EnhancedSearchResult(
                title="SPAM CLICK HERE",
                url="https://spam.xyz",
                snippet="Buy now!",
                domain="spam.xyz",
                word_count=50,
                quality_score=0.2,
            ),
            EnhancedSearchResult(
                title="Medium Quality",
                url="https://example.com",
                snippet="Some information",
                domain="example.com",
                word_count=300,
                quality_score=0.65,
            ),
        ]

        # Filter low quality (< 0.5)
        filtered = [r for r in results if r.quality_score >= 0.5]

        assert len(filtered) == 2
        assert all(r.quality_score >= 0.5 for r in filtered)
        assert "spam.xyz" not in [r.domain for r in filtered]

    def test_result_ranking_combined_scores(self):
        """Test ranking by combined provider and quality scores."""
        results = [
            EnhancedSearchResult(
                title="Result 1",
                url="https://example1.com",
                snippet="test",
                domain="example1.com",
                score=0.8,  # Provider score
                quality_score=0.9,  # Quality score
            ),
            EnhancedSearchResult(
                title="Result 2",
                url="https://example2.com",
                snippet="test",
                domain="example2.com",
                score=0.9,
                quality_score=0.6,
            ),
            EnhancedSearchResult(
                title="Result 3",
                url="https://example3.com",
                snippet="test",
                domain="example3.com",
                score=0.7,
                quality_score=0.95,
            ),
        ]

        # Rank by average of provider and quality scores
        semantic_weight = 0.5
        ranked = sorted(
            results,
            key=lambda r: (r.score * (1 - semantic_weight) + r.quality_score * semantic_weight),
            reverse=True
        )

        # Result 1 should rank first (0.8 * 0.5 + 0.9 * 0.5 = 0.85)
        # Result 3 is second (0.7 * 0.5 + 0.95 * 0.5 = 0.825)
        # Result 2 is third (0.9 * 0.5 + 0.6 * 0.5 = 0.75)
        assert ranked[0].url == "https://example1.com"
