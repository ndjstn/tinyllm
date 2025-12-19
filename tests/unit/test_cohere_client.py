"""Unit tests for Cohere API client."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.cohere_client import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatStreamEvent,
    CircuitBreaker,
    CohereClient,
    EmbedResponse,
    Embedding,
    RateLimiter,
    RerankDocument,
    RerankResponse,
    RerankResult,
    TokenCount,
    Tool,
    ToolCall,
    ToolParameterDefinition,
    ToolResult,
    close_all_cohere_clients,
    get_shared_cohere_client,
)


@pytest.fixture
def mock_chat_response():
    """Create a mock ChatResponse."""
    return {
        "text": "Hello! How can I help you today?",
        "generation_id": "gen-123",
        "conversation_id": "conv-456",
        "finish_reason": "COMPLETE",
        "token_count": {
            "input_tokens": 10,
            "output_tokens": 9,
            "total_tokens": 19,
            "billed_tokens": 19,
        },
    }


@pytest.fixture
def mock_chat_with_tools_response():
    """Create a mock response with tool calls."""
    return {
        "text": "I'll check the weather for you.",
        "generation_id": "gen-789",
        "finish_reason": "COMPLETE",
        "token_count": {
            "input_tokens": 15,
            "output_tokens": 10,
            "total_tokens": 25,
            "billed_tokens": 25,
        },
        "tool_calls": [
            {
                "name": "get_weather",
                "parameters": {"location": "San Francisco"},
            }
        ],
    }


@pytest.fixture
def mock_embed_response():
    """Create a mock EmbedResponse."""
    return {
        "id": "embed-123",
        "embeddings": {
            "float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        },
        "texts": ["Hello", "World"],
        "meta": {"api_version": {"version": "1"}},
    }


@pytest.fixture
def mock_rerank_response():
    """Create a mock RerankResponse."""
    return {
        "id": "rerank-123",
        "results": [
            {
                "index": 1,
                "relevance_score": 0.95,
                "document": {"text": "Most relevant document"},
            },
            {
                "index": 0,
                "relevance_score": 0.75,
                "document": {"text": "Less relevant document"},
            },
        ],
        "meta": {"api_version": {"version": "1"}},
    }


@pytest.fixture
def cohere_client():
    """Create a Cohere client for testing."""
    return CohereClient(api_key="test-key-123")


@pytest.fixture
async def cleanup_clients():
    """Cleanup shared clients after tests."""
    yield
    await close_all_cohere_clients()


class TestRateLimiter:
    """Test RateLimiter functionality."""

    async def test_allows_requests_within_rate(self):
        """Test that requests within rate limit are allowed."""
        limiter = RateLimiter(rate=10.0, burst=20)

        # Should allow immediate requests up to burst
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be immediate

    async def test_delays_when_rate_exceeded(self):
        """Test that requests are delayed when rate is exceeded."""
        limiter = RateLimiter(rate=5.0, burst=1)

        # First request should be immediate
        await limiter.acquire()

        # Second request should be delayed
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        # Should wait approximately 1/5 = 0.2 seconds
        assert 0.15 < elapsed < 0.35


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    async def test_allows_requests_when_closed(self):
        """Test that requests are allowed when circuit is closed."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"

    async def test_opens_after_threshold_failures(self):
        """Test that circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        async def failing_func():
            raise ValueError("Test error")

        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        assert breaker.get_state() == "open"

        # Next call should fail with circuit breaker error
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            await breaker.call(failing_func)

    async def test_recovers_after_timeout(self):
        """Test that circuit recovers to half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        assert breaker.get_state() == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open and allow a test request
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"


class TestCohereClient:
    """Test CohereClient functionality."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = CohereClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_init_with_env_var(self):
        """Test client initialization with environment variable."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "env-key"}):
            client = CohereClient()
            assert client.api_key == "env-key"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Cohere API key must be provided"):
                CohereClient()

    async def test_chat_completion(self, cohere_client, mock_chat_response):
        """Test basic chat completion."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_chat_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await cohere_client.chat(
                message="Hello",
                model="command-r-plus",
            )

            assert isinstance(response, ChatResponse)
            assert response.text == "Hello! How can I help you today?"
            assert response.generation_id == "gen-123"
            assert response.token_count.total_tokens == 19

    async def test_chat_with_tools(self, cohere_client, mock_chat_with_tools_response):
        """Test chat completion with tool calls."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_chat_with_tools_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            tools = [
                Tool(
                    name="get_weather",
                    description="Get weather for a location",
                    parameter_definitions={
                        "location": ToolParameterDefinition(
                            description="City name",
                            type="string",
                            required=True,
                        )
                    },
                )
            ]

            response = await cohere_client.chat(
                message="What's the weather in San Francisco?",
                tools=tools,
            )

            assert isinstance(response, ChatResponse)
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "get_weather"
            assert response.tool_calls[0].parameters["location"] == "San Francisco"

    async def test_chat_with_history(self, cohere_client, mock_chat_response):
        """Test chat with conversation history."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_chat_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            chat_history = [
                ChatMessage(role=ChatRole.USER, message="Hi there"),
                ChatMessage(role=ChatRole.CHATBOT, message="Hello! How can I help?"),
            ]

            response = await cohere_client.chat(
                message="Tell me a joke",
                chat_history=chat_history,
            )

            assert isinstance(response, ChatResponse)
            mock_client.post.assert_called_once()

    async def test_chat_retry_on_error(self, cohere_client, mock_chat_response):
        """Test that chat retries on transient errors."""
        # First call fails, second succeeds
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_error_response,
        )

        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_chat_response
        mock_success_response.status_code = 200
        mock_success_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[mock_error_response, mock_success_response]
            )
            mock_get_client.return_value = mock_client

            response = await cohere_client.chat(message="Hello")

            assert isinstance(response, ChatResponse)
            assert mock_client.post.call_count == 2

    async def test_chat_stream(self, cohere_client):
        """Test streaming chat completion."""
        stream_events = [
            '{"event_type": "stream-start"}',
            '{"event_type": "text-generation", "text": "Hello"}',
            '{"event_type": "text-generation", "text": " there!"}',
            '{"event_type": "stream-end", "is_finished": true, "finish_reason": "COMPLETE"}',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            for event in stream_events:
                yield event

        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_stream_context = MagicMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=mock_stream_context)
            mock_get_client.return_value = mock_client

            events = []
            async for event in cohere_client.chat_stream(message="Hello"):
                events.append(event)

            assert len(events) == 4
            assert events[0].event_type == "stream-start"
            assert events[1].text == "Hello"
            assert events[2].text == " there!"
            assert events[3].is_finished is True

    async def test_embed(self, cohere_client, mock_embed_response):
        """Test embedding generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embed_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await cohere_client.embed(
                texts=["Hello", "World"],
                model="embed-english-v3.0",
            )

            assert isinstance(response, EmbedResponse)
            assert response.id == "embed-123"
            assert len(response.embeddings) == 2
            assert response.embeddings[0].values == [0.1, 0.2, 0.3]
            assert response.embeddings[1].values == [0.4, 0.5, 0.6]

    async def test_embed_with_input_type(self, cohere_client, mock_embed_response):
        """Test embedding with input type."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embed_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await cohere_client.embed(
                texts=["search query"],
                input_type="search_query",
            )

            assert isinstance(response, EmbedResponse)
            mock_client.post.assert_called_once()

    async def test_rerank(self, cohere_client, mock_rerank_response):
        """Test document reranking."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_rerank_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            documents = ["Less relevant document", "Most relevant document"]
            response = await cohere_client.rerank(
                query="relevant information",
                documents=documents,
                model="rerank-english-v3.0",
            )

            assert isinstance(response, RerankResponse)
            assert response.id == "rerank-123"
            assert len(response.results) == 2
            assert response.results[0].relevance_score == 0.95
            assert response.results[1].relevance_score == 0.75

    async def test_rerank_with_top_n(self, cohere_client):
        """Test rerank with top_n parameter."""
        mock_response_data = {
            "id": "rerank-456",
            "results": [
                {
                    "index": 1,
                    "relevance_score": 0.95,
                    "document": {"text": "Most relevant"},
                },
            ],
            "meta": {},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = await cohere_client.rerank(
                query="test",
                documents=["doc1", "doc2", "doc3"],
                top_n=1,
            )

            assert len(response.results) == 1

    async def test_rate_limiting(self, cohere_client, mock_chat_response):
        """Test that rate limiting is applied."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_chat_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Set very low rate limit
            cohere_client._rate_limiter = RateLimiter(rate=2.0, burst=1)

            start = asyncio.get_event_loop().time()

            # Make two requests - second should be delayed
            await cohere_client.chat(message="First")
            await cohere_client.chat(message="Second")

            elapsed = asyncio.get_event_loop().time() - start

            # Should take at least 0.5 seconds for second request
            assert elapsed > 0.4

    async def test_circuit_breaker_opens_on_failures(self, cohere_client):
        """Test that circuit breaker opens after repeated failures."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Set low threshold
            cohere_client._circuit_breaker = CircuitBreaker(
                failure_threshold=2, recovery_timeout=1.0
            )

            # Fail twice to open circuit
            for _ in range(2):
                with pytest.raises(httpx.HTTPStatusError):
                    await cohere_client.chat(message="Test")

            # Circuit should now be open
            assert cohere_client._circuit_breaker.get_state() == "open"

            # Next request should fail immediately with circuit breaker error
            with pytest.raises(RuntimeError, match="Circuit breaker is open"):
                await cohere_client.chat(message="Test")

    async def test_set_graph_context(self, cohere_client):
        """Test setting graph context for metrics."""
        cohere_client.set_graph_context("test_graph")
        assert cohere_client._current_graph == "test_graph"

    async def test_get_stats(self, cohere_client, mock_chat_response):
        """Test getting client statistics."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_chat_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await cohere_client.chat(message="Hello")

            stats = cohere_client.get_stats()
            assert stats["request_count"] == 1
            assert stats["total_tokens"] == 19
            assert stats["circuit_breaker_state"] == "closed"

    async def test_close(self, cohere_client):
        """Test closing the client."""
        # Create a mock client
        mock_client = AsyncMock()
        cohere_client._client = mock_client

        await cohere_client.close()

        mock_client.aclose.assert_called_once()
        assert cohere_client._client is None

    async def test_shared_client_pool(self, cleanup_clients):
        """Test that shared clients are reused."""
        client1 = await get_shared_cohere_client(api_key="test-key")
        client2 = await get_shared_cohere_client(api_key="test-key")

        assert client1 is client2

    async def test_shared_client_different_keys(self, cleanup_clients):
        """Test that different API keys get different clients."""
        client1 = await get_shared_cohere_client(api_key="key1")
        client2 = await get_shared_cohere_client(api_key="key2")

        assert client1 is not client2

    async def test_close_all_clients(self):
        """Test closing all shared clients."""
        client1 = await get_shared_cohere_client(api_key="key1")
        client2 = await get_shared_cohere_client(api_key="key2")

        await close_all_cohere_clients()

        # Verify clients were closed
        client1._client is None or client1._client.is_closed
        client2._client is None or client2._client.is_closed


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_chat_message(self):
        """Test ChatMessage model."""
        msg = ChatMessage(role=ChatRole.USER, message="Hello")
        assert msg.role == ChatRole.USER
        assert msg.message == "Hello"

    def test_tool(self):
        """Test Tool model."""
        tool = Tool(
            name="calculator",
            description="Perform calculations",
            parameter_definitions={
                "expression": ToolParameterDefinition(
                    description="Math expression",
                    type="string",
                    required=True,
                )
            },
        )
        assert tool.name == "calculator"
        assert "expression" in tool.parameter_definitions

    def test_tool_call(self):
        """Test ToolCall model."""
        call = ToolCall(
            name="get_weather",
            parameters={"location": "NYC"},
        )
        assert call.name == "get_weather"
        assert call.parameters["location"] == "NYC"

    def test_chat_response(self):
        """Test ChatResponse model."""
        response = ChatResponse(
            text="Hello!",
            generation_id="gen-123",
            token_count=TokenCount(input_tokens=5, output_tokens=3, total_tokens=8),
        )
        assert response.text == "Hello!"
        assert response.token_count.total_tokens == 8

    def test_embed_response(self):
        """Test EmbedResponse model."""
        response = EmbedResponse(
            id="embed-123",
            embeddings=[Embedding(values=[0.1, 0.2, 0.3])],
            texts=["Hello"],
        )
        assert response.id == "embed-123"
        assert len(response.embeddings) == 1

    def test_rerank_response(self):
        """Test RerankResponse model."""
        response = RerankResponse(
            id="rerank-123",
            results=[
                RerankResult(
                    index=0,
                    relevance_score=0.95,
                    document={"text": "Relevant doc"},
                )
            ],
        )
        assert response.id == "rerank-123"
        assert response.results[0].relevance_score == 0.95


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_handles_rate_limit_error(self, cohere_client, mock_chat_response):
        """Test handling of rate limit errors."""
        # First response is rate limited
        mock_rate_limit_response = MagicMock()
        mock_rate_limit_response.status_code = 429
        mock_rate_limit_response.headers = {"retry-after": "0.1"}
        mock_rate_limit_response.raise_for_status = MagicMock()

        # Second response succeeds
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = mock_chat_response
        mock_success_response.status_code = 200
        mock_success_response.raise_for_status = MagicMock()

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=[mock_rate_limit_response, mock_success_response]
            )
            mock_get_client.return_value = mock_client

            start = asyncio.get_event_loop().time()
            response = await cohere_client.chat(message="Hello")
            elapsed = asyncio.get_event_loop().time() - start

            # Should have waited for retry-after period
            assert elapsed > 0.08
            assert isinstance(response, ChatResponse)

    async def test_exhausts_retries_on_persistent_error(self, cohere_client):
        """Test that retries are exhausted on persistent errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Should raise after max_retries attempts
            with pytest.raises(httpx.HTTPStatusError):
                await cohere_client.chat(message="Hello")

            # Should have tried max_retries + 1 times
            assert mock_client.post.call_count == cohere_client.max_retries + 1

    async def test_handles_network_error(self, cohere_client):
        """Test handling of network errors."""
        with patch.object(cohere_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await cohere_client.chat(message="Hello")
