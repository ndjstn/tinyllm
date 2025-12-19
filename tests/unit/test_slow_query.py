"""Tests for slow query detection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinyllm.models.client import GenerateResponse, OllamaClient


class TestSlowQueryDetection:
    """Test slow query detection in OllamaClient."""

    def test_init_with_threshold(self):
        """Test initializing client with slow query threshold."""
        client = OllamaClient(slow_query_threshold_ms=3000)
        assert client.slow_query_threshold_ms == 3000
        assert len(client._slow_queries) == 0

    def test_init_default_threshold(self):
        """Test default slow query threshold."""
        client = OllamaClient()
        assert client.slow_query_threshold_ms == 5000  # Default

    @pytest.mark.asyncio
    async def test_fast_query_not_logged(self):
        """Test that fast queries are not logged as slow."""
        client = OllamaClient(slow_query_threshold_ms=1000)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:model",
            "created_at": "2025-01-01T00:00:00Z",
            "response": "Fast response",
            "done": True,
            "total_duration": 500000000,  # 0.5 seconds in nanoseconds
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response

        client._client = mock_http_client
        client._current_model = "test:model"

        # Generate response (should be fast)
        result = await client.generate("test prompt")

        assert result.response == "Fast response"
        assert len(client._slow_queries) == 0

    @pytest.mark.asyncio
    async def test_slow_query_logged(self):
        """Test that slow queries are logged."""
        client = OllamaClient(slow_query_threshold_ms=100)  # Low threshold for testing

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:model",
            "created_at": "2025-01-01T00:00:00Z",
            "response": "Slow response",
            "done": True,
            "total_duration": 5000000000,  # 5 seconds
            "eval_count": 100,
            "prompt_eval_count": 50,
        }
        mock_response.raise_for_status = MagicMock()

        async def slow_post(*args, **kwargs):
            await asyncio.sleep(0.15)  # Simulate slow response
            return mock_response

        mock_http_client = AsyncMock()
        mock_http_client.post = slow_post

        client._client = mock_http_client
        client._current_model = "test:model"

        # Generate response (should be slow)
        with patch("tinyllm.models.client.logger") as mock_logger:
            result = await client.generate("test prompt")

            assert result.response == "Slow response"
            assert len(client._slow_queries) == 1

            # Check slow query was logged
            slow_query = client._slow_queries[0]
            assert slow_query["model"] == "test:model"
            assert slow_query["duration_ms"] >= 100
            assert slow_query["prompt_length"] == len("test prompt")
            assert slow_query["input_tokens"] == 50
            assert slow_query["output_tokens"] == 100

            # Check warning was logged
            mock_logger.warning.assert_called()

    def test_get_slow_queries(self):
        """Test retrieving slow queries."""
        client = OllamaClient()

        # Add some test slow queries
        client._slow_queries = [
            {
                "timestamp": 1000,
                "model": "model1",
                "duration_ms": 6000,
                "prompt_length": 100,
                "input_tokens": 50,
                "output_tokens": 100,
                "graph": "test",
            },
            {
                "timestamp": 2000,
                "model": "model2",
                "duration_ms": 8000,
                "prompt_length": 200,
                "input_tokens": 100,
                "output_tokens": 200,
                "graph": "test",
            },
            {
                "timestamp": 3000,
                "model": "model3",
                "duration_ms": 12000,
                "prompt_length": 300,
                "input_tokens": 150,
                "output_tokens": 300,
                "graph": "test",
            },
        ]

        # Get all queries (should be sorted by timestamp, newest first)
        queries = client.get_slow_queries()
        assert len(queries) == 3
        assert queries[0]["timestamp"] == 3000  # Most recent first

    def test_get_slow_queries_with_limit(self):
        """Test retrieving slow queries with limit."""
        client = OllamaClient()

        client._slow_queries = [
            {"timestamp": i * 1000, "duration_ms": 6000, "model": f"model{i}"}
            for i in range(10)
        ]

        queries = client.get_slow_queries(limit=5)
        assert len(queries) == 5
        # Should get most recent
        assert queries[0]["timestamp"] == 9000

    def test_get_slow_queries_with_min_duration(self):
        """Test filtering slow queries by minimum duration."""
        client = OllamaClient()

        client._slow_queries = [
            {"timestamp": 1000, "duration_ms": 6000, "model": "model1"},
            {"timestamp": 2000, "duration_ms": 8000, "model": "model2"},
            {"timestamp": 3000, "duration_ms": 12000, "model": "model3"},
        ]

        queries = client.get_slow_queries(min_duration_ms=10000)
        assert len(queries) == 1
        assert queries[0]["duration_ms"] == 12000

    def test_clear_slow_queries(self):
        """Test clearing slow query history."""
        client = OllamaClient()

        client._slow_queries = [
            {"timestamp": 1000, "duration_ms": 6000, "model": "model1"},
            {"timestamp": 2000, "duration_ms": 8000, "model": "model2"},
        ]

        assert len(client._slow_queries) == 2

        client.clear_slow_queries()
        assert len(client._slow_queries) == 0

    def test_get_stats_includes_slow_queries(self):
        """Test that stats include slow query information."""
        client = OllamaClient(slow_query_threshold_ms=3000)

        client._slow_queries = [
            {"timestamp": 1000, "duration_ms": 6000, "model": "model1"},
            {"timestamp": 2000, "duration_ms": 8000, "model": "model2"},
        ]

        stats = client.get_stats()
        assert stats["slow_query_count"] == 2
        assert stats["slow_query_threshold_ms"] == 3000

    @pytest.mark.asyncio
    async def test_multiple_slow_queries(self):
        """Test logging multiple slow queries."""
        client = OllamaClient(slow_query_threshold_ms=50)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "test:model",
            "created_at": "2025-01-01T00:00:00Z",
            "response": "Response",
            "done": True,
            "total_duration": 5000000000,
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        mock_response.raise_for_status = MagicMock()

        async def slow_post(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_response

        mock_http_client = AsyncMock()
        mock_http_client.post = slow_post

        client._client = mock_http_client
        client._current_model = "test:model"

        # Generate multiple slow queries
        with patch("tinyllm.models.client.logger"):
            await client.generate("query 1")
            await client.generate("query 2")
            await client.generate("query 3")

        assert len(client._slow_queries) == 3

        # Verify they're sorted by timestamp
        queries = client.get_slow_queries()
        assert len(queries) == 3
        # Should be in reverse chronological order
        for i in range(len(queries) - 1):
            assert queries[i]["timestamp"] >= queries[i + 1]["timestamp"]
