"""Tests for Ollama embedding functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinyllm.models.client import OllamaClient, EmbedRequest, EmbedResponse
from tinyllm.tools.web_search_v2 import cosine_similarity


class TestCosineSimilarity:
    """Tests for cosine similarity utility."""

    def test_identical_vectors(self):
        """Identical vectors have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0, 4.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity of 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity of -1.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors have positive similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec1, vec2)
        assert 0.99 < similarity <= 1.0

    def test_dissimilar_vectors(self):
        """Dissimilar vectors have lower similarity."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.1, 0.9, 0.1]
        similarity = cosine_similarity(vec1, vec2)
        assert 0.0 < similarity < 0.5

    def test_zero_vector(self):
        """Zero vectors return 0.0 similarity."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_different_length_vectors(self):
        """Different length vectors raise ValueError."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        with pytest.raises(ValueError, match="Vectors must have same length"):
            cosine_similarity(vec1, vec2)


class TestEmbedRequest:
    """Tests for EmbedRequest model."""

    def test_single_prompt(self):
        """Can create request with single prompt."""
        request = EmbedRequest(
            model="nomic-embed-text",
            prompt="test text"
        )
        assert request.model == "nomic-embed-text"
        assert request.prompt == "test text"

    def test_batch_prompt(self):
        """Can create request with batch prompts."""
        request = EmbedRequest(
            model="nomic-embed-text",
            prompt=["text 1", "text 2", "text 3"]
        )
        assert isinstance(request.prompt, list)
        assert len(request.prompt) == 3


class TestEmbedResponse:
    """Tests for EmbedResponse model."""

    def test_single_embedding(self):
        """Can parse single embedding response."""
        response = EmbedResponse(
            embedding=[0.1, 0.2, 0.3, 0.4]
        )
        assert len(response.embedding) == 4
        assert response.embedding[0] == 0.1

    def test_batch_embeddings(self):
        """Can parse batch embeddings response."""
        response = EmbedResponse(
            embedding=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        assert len(response.embedding) == 3
        assert len(response.embedding[0]) == 3


class TestOllamaClientEmbed:
    """Tests for OllamaClient embed method."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Can embed a single text."""
        client = OllamaClient()

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        client._client = mock_http_client

        # Call embed
        response = await client.embed(
            model="nomic-embed-text",
            prompt="test text"
        )

        # Verify response
        assert isinstance(response, EmbedResponse)
        assert len(response.embedding) == 5
        assert response.embedding[0] == 0.1

        # Verify API call
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "/api/embeddings"
        assert "test text" in str(call_args[1]["json"])

    @pytest.mark.asyncio
    async def test_embed_batch_texts(self):
        """Can embed multiple texts in a batch."""
        client = OllamaClient()

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        client._client = mock_http_client

        # Call embed with batch
        response = await client.embed(
            model="nomic-embed-text",
            prompt=["text 1", "text 2"]
        )

        # Verify response
        assert isinstance(response, EmbedResponse)
        assert len(response.embedding) == 2
        assert len(response.embedding[0]) == 3

    @pytest.mark.asyncio
    async def test_embed_with_retry(self):
        """Embed retries on failure."""
        client = OllamaClient(max_retries=2)

        # Mock HTTP client to fail twice then succeed
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3]
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()

        # Fail twice, then succeed
        import httpx
        mock_http_client.post = AsyncMock(
            side_effect=[
                httpx.HTTPError("Network error"),
                httpx.HTTPError("Network error"),
                mock_response
            ]
        )

        client._client = mock_http_client

        # Call embed
        response = await client.embed(
            model="nomic-embed-text",
            prompt="test"
        )

        # Should succeed after retries
        assert isinstance(response, EmbedResponse)
        assert mock_http_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_max_retries_exceeded(self):
        """Embed raises after max retries."""
        client = OllamaClient(max_retries=1)

        # Mock HTTP client to always fail
        mock_http_client = AsyncMock()
        import httpx
        mock_http_client.post = AsyncMock(
            side_effect=httpx.HTTPError("Network error")
        )

        client._client = mock_http_client

        # Should raise after max retries
        with pytest.raises(httpx.HTTPError):
            await client.embed(
                model="nomic-embed-text",
                prompt="test"
            )

        # Should have tried max_retries + 1 times
        assert mock_http_client.post.call_count == 2


class TestSemanticReranking:
    """Integration tests for semantic reranking."""

    @pytest.mark.asyncio
    async def test_reranking_with_embeddings(self):
        """Results are reranked using semantic similarity."""
        from tinyllm.tools.web_search_v2 import EnhancedSearchResult, EnhancedWebSearchConfig

        # Mock results
        results = [
            EnhancedSearchResult(
                title="Python Tutorial",
                url="https://example.com/python",
                snippet="Learn Python programming basics",
                domain="example.com",
                score=0.5,
                quality_score=0.7,
            ),
            EnhancedSearchResult(
                title="JavaScript Guide",
                url="https://example.com/js",
                snippet="JavaScript programming tutorial",
                domain="example.com",
                score=0.9,  # Higher provider score
                quality_score=0.8,
            ),
        ]

        # Query is about Python, so Python result should rank higher
        # even though JS has higher provider score
        query = "python programming tutorial"

        # Mock embedding responses
        mock_query_embed = [0.9, 0.1, 0.1]  # Python-like embedding
        mock_result1_embed = [0.9, 0.1, 0.1]  # Python result (very similar)
        mock_result2_embed = [0.1, 0.9, 0.1]  # JS result (less similar)

        with patch('tinyllm.tools.web_search_v2.get_shared_client') as mock_get_client:
            mock_client = AsyncMock()

            # Mock query embedding
            mock_client.embed = AsyncMock(
                side_effect=[
                    # First call: query embedding
                    EmbedResponse(embedding=mock_query_embed),
                    # Second call: batch result embeddings
                    EmbedResponse(embedding=[mock_result1_embed, mock_result2_embed]),
                ]
            )
            mock_get_client.return_value = mock_client

            # Create config and rerank
            from tinyllm.tools.web_search_v2 import EnhancedWebSearchTool
            config = EnhancedWebSearchConfig(
                enable_semantic_reranking=True,
                semantic_weight=0.7  # High weight on semantic similarity
            )
            tool = EnhancedWebSearchTool(config=config)

            reranked = await tool._rerank_results(results, query)

            # Python result should rank first due to high semantic similarity
            # even though JS had higher provider score initially
            assert reranked[0].title == "Python Tutorial"
