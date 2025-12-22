"""Tests for llama.cpp client."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.llamacpp_client import (
    CircuitBreaker,
    CompletionRequest,
    CompletionResponse,
    LlamaCppClient,
    LlamaCppConfig,
    RateLimiter,
    close_all_llamacpp_clients,
    get_shared_llamacpp_client,
)


@pytest.fixture
def sample_completion_response():
    """Create a sample CompletionResponse."""
    return CompletionResponse(
        content="Hello! How can I help you today?",
        stop=True,
        model="llama-2-7b",
        tokens_evaluated=50,
        tokens_predicted=20,
        stopped_eos=True,
    )


@pytest.fixture
def llamacpp_client():
    """Create a LlamaCppClient instance."""
    return LlamaCppClient(
        host="http://localhost:8080",
        timeout_ms=5000,
        max_retries=2,
    )


@pytest.fixture
async def mock_http_client():
    """Create a mock HTTP client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


# Rate Limiter Tests


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst(self):
        """Test rate limiter allows burst requests."""
        limiter = RateLimiter(rate=10.0, burst=5)

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_rate(self):
        """Test rate limiter enforces rate limit."""
        limiter = RateLimiter(rate=10.0, burst=1)

        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should take ~0.2s for 3 requests at 10/s (after first burst token)
        # Allowing for some timing variation
        assert elapsed >= 0.08


# Circuit Breaker Tests


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_success(self):
        """Test circuit breaker allows successful calls."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)

        async def failing_func():
            raise RuntimeError("failure")

        # Cause 3 failures
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_func)

        # Circuit should be open
        assert breaker.state == "open"
        assert breaker.failures >= 3

        # Next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call(failing_func)


# Client Tests


class TestLlamaCppClient:
    """Tests for LlamaCppClient."""

    def test_client_initialization(self, llamacpp_client):
        """Test client initializes with correct defaults."""
        assert llamacpp_client.host == "http://localhost:8080"
        assert llamacpp_client.max_retries == 2
        assert llamacpp_client._request_count == 0
        assert llamacpp_client._total_tokens == 0

    def test_set_get_model(self, llamacpp_client):
        """Test model setter/getter."""
        llamacpp_client.set_model("llama-2-13b")
        assert llamacpp_client.get_model() == "llama-2-13b"

    def test_set_graph_context(self, llamacpp_client):
        """Test setting graph context."""
        llamacpp_client.set_graph_context("test-graph")
        assert llamacpp_client._current_graph == "test-graph"

    @pytest.mark.asyncio
    async def test_generate_success(self, llamacpp_client, sample_completion_response):
        """Test successful generation."""
        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # Changed from AsyncMock
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_completion_response.model_dump())
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await llamacpp_client.generate("Hello")

            assert isinstance(result, CompletionResponse)
            assert result.content == "Hello! How can I help you today?"
            assert llamacpp_client._request_count == 1
            assert llamacpp_client._total_tokens == 20

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llamacpp_client, sample_completion_response):
        """Test generation with system prompt."""
        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_completion_response.model_dump())
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await llamacpp_client.generate("Hello", system="You are a helpful assistant")

            # Verify system prompt was prepended
            call_args = mock_client.post.call_args
            request_json = call_args[1]["json"]
            assert "You are a helpful assistant" in request_json["prompt"]
            assert isinstance(result, CompletionResponse)

    @pytest.mark.asyncio
    async def test_health_check_success(self, llamacpp_client):
        """Test successful health check."""
        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            is_healthy = await llamacpp_client.check_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_fallback(self, llamacpp_client):
        """Test health check fallback to /props endpoint."""
        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()

            # First call to /health fails
            health_response = AsyncMock()
            health_response.status_code = 404
            health_response.raise_for_status.side_effect = httpx.HTTPError("Not found")

            # Second call to /props succeeds
            props_response = AsyncMock()
            props_response.status_code = 200

            mock_client.get.side_effect = [httpx.HTTPError("Not found"), props_response]
            mock_get_client.return_value = mock_client

            is_healthy = await llamacpp_client.check_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llamacpp_client):
        """Test failed health check."""
        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_get_client.return_value = mock_client

            is_healthy = await llamacpp_client.check_health()
            assert is_healthy is False

    def test_get_stats(self, llamacpp_client):
        """Test getting client statistics."""
        llamacpp_client._request_count = 10
        llamacpp_client._total_tokens = 500

        stats = llamacpp_client.get_stats()

        assert stats["request_count"] == 10
        assert stats["total_tokens"] == 500
        assert "circuit_breaker_state" in stats
        assert "slow_queries_count" in stats

    @pytest.mark.asyncio
    async def test_slow_query_detection(self, llamacpp_client, sample_completion_response):
        """Test slow query detection and logging."""
        llamacpp_client.slow_query_threshold_ms = 100  # 100ms threshold

        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_completion_response.model_dump())
            mock_response.raise_for_status = MagicMock()

            # Simulate slow response
            async def slow_post(*args, **kwargs):
                await asyncio.sleep(0.15)  # 150ms
                return mock_response

            mock_client.post = slow_post
            mock_get_client.return_value = mock_client

            await llamacpp_client.generate("Hello")

            # Check slow query was logged
            slow_queries = llamacpp_client.get_slow_queries()
            assert len(slow_queries) == 1
            assert slow_queries[0]["duration_ms"] > 100

    @pytest.mark.asyncio
    async def test_close(self, llamacpp_client):
        """Test client cleanup."""
        # Create a real mock client
        llamacpp_client._client = AsyncMock(spec=httpx.AsyncClient)

        await llamacpp_client.close()

        assert llamacpp_client._client is None

    @pytest.mark.asyncio
    async def test_get_props(self, llamacpp_client):
        """Test getting server properties."""
        with patch.object(llamacpp_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value={
                "total_slots": 1,
                "n_ctx": 2048,
                "n_predict": 512,
                "model": "llama-2-7b",
            })
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            props = await llamacpp_client.get_props()

            assert props["model"] == "llama-2-7b"
            assert props["n_ctx"] == 2048


# Connection Pool Tests


class TestConnectionPool:
    """Tests for connection pooling."""

    @pytest.mark.asyncio
    async def test_shared_client_creation(self):
        """Test shared client is created once."""
        # Clean pool first
        await close_all_llamacpp_clients()

        client1 = await get_shared_llamacpp_client(host="http://localhost:8080")
        client2 = await get_shared_llamacpp_client(host="http://localhost:8080")

        # Should be same instance
        assert client1 is client2

        await close_all_llamacpp_clients()

    @pytest.mark.asyncio
    async def test_different_hosts_different_clients(self):
        """Test different hosts get different clients."""
        await close_all_llamacpp_clients()

        client1 = await get_shared_llamacpp_client(host="http://localhost:8080")
        client2 = await get_shared_llamacpp_client(host="http://localhost:8081")

        # Should be different instances
        assert client1 is not client2

        await close_all_llamacpp_clients()

    @pytest.mark.asyncio
    async def test_close_all_clients(self):
        """Test closing all pooled clients."""
        await close_all_llamacpp_clients()

        # Create some clients
        await get_shared_llamacpp_client(host="http://localhost:8080")
        await get_shared_llamacpp_client(host="http://localhost:8081")

        # Close all
        await close_all_llamacpp_clients()

        # Pool should be empty - new client should be created
        client = await get_shared_llamacpp_client(host="http://localhost:8080")
        assert client is not None

        await close_all_llamacpp_clients()


# Configuration Tests


class TestLlamaCppConfig:
    """Tests for LlamaCppConfig."""

    def test_config_defaults(self):
        """Test config uses correct defaults."""
        config = LlamaCppConfig()

        assert config.host == "http://localhost:8080"
        assert config.timeout_ms == 30000
        assert config.max_retries == 3
        assert config.rate_limit_rps == 10.0
        assert config.circuit_breaker_threshold == 5

    def test_config_validation(self):
        """Test config validates values."""
        # Valid config
        config = LlamaCppConfig(
            host="http://example.com:9000",
            timeout_ms=60000,
            max_retries=5,
            rate_limit_rps=20.0,
        )

        assert config.host == "http://example.com:9000"
        assert config.timeout_ms == 60000

        # Invalid values should fail validation
        with pytest.raises(Exception):  # Pydantic ValidationError
            LlamaCppConfig(timeout_ms=500)  # Below minimum

        with pytest.raises(Exception):
            LlamaCppConfig(rate_limit_rps=0.0)  # Below minimum
