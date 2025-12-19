"""Unit tests for OpenAI API client."""

import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.openai_client import (
    ChatCompletionResponse,
    ChatMessage,
    CircuitBreaker,
    EmbeddingResponse,
    ImageUrlDetail,
    MessageRole,
    OpenAIClient,
    OpenAIConfig,
    RateLimiter,
    ToolCall,
    ToolDefinition,
    FunctionDefinition,
    Usage,
    get_shared_openai_client,
    close_all_openai_clients,
)


@pytest.fixture
def mock_chat_response():
    """Create a mock ChatCompletionResponse."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "total_tokens": 19,
        },
    }


@pytest.fixture
def mock_embedding_response():
    """Create a mock EmbeddingResponse."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 0,
            "total_tokens": 5,
        },
    }


@pytest.fixture
def mock_tool_response():
    """Create a mock response with tool calls."""
    return {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25,
        },
    }


@pytest.fixture
def openai_client():
    """Create an OpenAI client for testing."""
    return OpenAIClient(api_key="test-key-123")


@pytest.fixture
async def cleanup_clients():
    """Cleanup shared clients after tests."""
    yield
    await close_all_openai_clients()


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

        # Should transition to half-open and succeed
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"


class TestOpenAIClient:
    """Test OpenAIClient functionality."""

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = OpenAIClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.default_model == "gpt-4o-mini"

    def test_initialization_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            client = OpenAIClient()
            assert client.api_key == "env-key"

    def test_initialization_without_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key must be provided"):
                OpenAIClient()

    async def test_close(self, openai_client):
        """Test client cleanup."""
        # Initialize client
        await openai_client._get_client()
        assert openai_client._client is not None

        # Close it
        await openai_client.close()
        assert openai_client._client is None

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_success(self, mock_post, openai_client, mock_chat_response):
        """Test successful chat completion."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_chat_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = await openai_client.chat_completion(messages=messages)

        # Verify
        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-123"
        assert result.choices[0].message.content == "Hello! How can I help you today?"
        assert result.usage.total_tokens == 19

        # Verify stats
        stats = openai_client.get_stats()
        assert stats["request_count"] == 1
        assert stats["total_tokens"] == 19

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_with_tools(
        self, mock_post, openai_client, mock_tool_response
    ):
        """Test chat completion with tool calls."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Define tool
        tools = [
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                ),
            )
        ]

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="What's the weather?")]
        result = await openai_client.chat_completion(messages=messages, tools=tools)

        # Verify
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_retry_on_error(self, mock_post, openai_client):
        """Test retry logic on transient errors."""
        # Setup mock to fail twice then succeed
        mock_response_error = MagicMock()
        mock_response_error.status_code = 500
        mock_response_error.headers = {}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Success"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response_success.raise_for_status = MagicMock()

        # First two calls fail, third succeeds
        def side_effect(*args, **kwargs):
            if mock_post.call_count <= 2:
                raise httpx.HTTPStatusError(
                    "Server error", request=MagicMock(), response=mock_response_error
                )
            return mock_response_success

        mock_post.side_effect = side_effect

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = await openai_client.chat_completion(messages=messages)

        # Verify it eventually succeeded
        assert result.id == "chatcmpl-123"
        assert mock_post.call_count == 3

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_fails_on_client_error(self, mock_post, openai_client):
        """Test that client errors (4xx) are not retried."""
        # Setup mock to return 400 error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        mock_post.side_effect = httpx.HTTPStatusError(
            "Bad request", request=MagicMock(), response=mock_response
        )

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

        with pytest.raises(httpx.HTTPStatusError):
            await openai_client.chat_completion(messages=messages)

        # Should not retry on 4xx errors
        assert mock_post.call_count == 1

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_respects_retry_after(self, mock_post, openai_client):
        """Test that Retry-After header is respected."""
        # Setup mock to return 429 with Retry-After
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {"Retry-After": "0.1"}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Success"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response_success.raise_for_status = MagicMock()

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "Rate limited", request=MagicMock(), response=mock_response_error
                )
            return mock_response_success

        mock_post.side_effect = side_effect

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = await openai_client.chat_completion(messages=messages)

        # Verify it succeeded after retry
        assert result.id == "chatcmpl-123"

    @patch("httpx.AsyncClient.stream")
    async def test_chat_completion_streaming(self, mock_stream, openai_client):
        """Test streaming chat completion."""
        # Setup mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            lines = [
                'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"}}]}',
                'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" world"}}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Make streaming request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        chunks = []
        async for chunk in openai_client.chat_completion_stream(messages=messages):
            chunks.append(chunk)

        # Verify
        assert chunks == ["Hello", " world"]

    @patch("httpx.AsyncClient.post")
    async def test_create_embedding_success(
        self, mock_post, openai_client, mock_embedding_response
    ):
        """Test successful embedding creation."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_embedding_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        result = await openai_client.create_embedding("Hello world")

        # Verify
        assert isinstance(result, EmbeddingResponse)
        assert result.model == "text-embedding-3-small"
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]

    @patch("httpx.AsyncClient.get")
    async def test_list_models(self, mock_get, openai_client):
        """Test listing available models."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
                {"id": "gpt-3.5-turbo"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Make request
        models = await openai_client.list_models()

        # Verify
        assert models == ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

    def test_encode_image_to_url(self, openai_client, tmp_path):
        """Test encoding image to base64 URL."""
        # Create a test image file
        image_path = tmp_path / "test.png"
        image_data = b"fake-image-data"
        image_path.write_bytes(image_data)

        # Encode it
        url = openai_client.encode_image_to_url(str(image_path))

        # Verify
        assert url.startswith("data:image/png;base64,")
        encoded_data = url.split(",")[1]
        assert base64.b64decode(encoded_data) == image_data

    def test_create_image_message(self, openai_client):
        """Test creating a message with images."""
        message = openai_client.create_image_message(
            text="What's in this image?",
            image_urls=["https://example.com/image.jpg"],
            detail=ImageUrlDetail.HIGH,
        )

        # Verify
        assert message.role == MessageRole.USER
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0].type == "text"
        assert message.content[0].text == "What's in this image?"
        assert message.content[1].type == "image_url"
        assert message.content[1].image_url.url == "https://example.com/image.jpg"
        assert message.content[1].image_url.detail == ImageUrlDetail.HIGH

    def test_set_graph_context(self, openai_client):
        """Test setting graph context."""
        openai_client.set_graph_context("test-graph")
        assert openai_client._current_graph == "test-graph"

    def test_get_stats(self, openai_client):
        """Test getting client statistics."""
        stats = openai_client.get_stats()

        assert "request_count" in stats
        assert "total_tokens" in stats
        assert "circuit_breaker_state" in stats
        assert "circuit_breaker_failures" in stats


class TestSharedClient:
    """Test shared client pool functionality."""

    async def test_get_shared_client(self, cleanup_clients):
        """Test getting a shared client."""
        client1 = await get_shared_openai_client(api_key="test-key")
        client2 = await get_shared_openai_client(api_key="test-key")

        # Should return the same instance
        assert client1 is client2

    async def test_different_configs_create_different_clients(self, cleanup_clients):
        """Test that different configs create different clients."""
        client1 = await get_shared_openai_client(api_key="key1")
        client2 = await get_shared_openai_client(api_key="key2")

        # Should be different instances
        assert client1 is not client2

    async def test_close_all_clients(self, cleanup_clients):
        """Test closing all shared clients."""
        # Create some clients
        await get_shared_openai_client(api_key="key1")
        await get_shared_openai_client(api_key="key2")

        # Close all
        await close_all_openai_clients()

        # Pool should be empty
        from tinyllm.providers.openai_client import _client_pool

        assert len(_client_pool) == 0


class TestOpenAIConfig:
    """Test OpenAIConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OpenAIConfig()

        assert config.base_url == "https://api.openai.com/v1"
        assert config.timeout_ms == 60000
        assert config.max_retries == 3
        assert config.rate_limit_rps == 10.0
        assert config.default_model == "gpt-4o-mini"
        assert config.circuit_breaker_threshold == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OpenAIConfig(
            api_key="custom-key",
            base_url="https://custom.api.com/v1",
            timeout_ms=30000,
            max_retries=5,
            rate_limit_rps=20.0,
            default_model="gpt-4o",
            circuit_breaker_threshold=10,
        )

        assert config.api_key == "custom-key"
        assert config.base_url == "https://custom.api.com/v1"
        assert config.timeout_ms == 30000
        assert config.max_retries == 5
        assert config.rate_limit_rps == 20.0
        assert config.default_model == "gpt-4o"
        assert config.circuit_breaker_threshold == 10


class TestPydanticModels:
    """Test Pydantic model validation and serialization."""

    def test_chat_message_simple(self):
        """Test simple chat message creation."""
        msg = ChatMessage(role=MessageRole.USER, content="Hello")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_chat_message_with_tools(self):
        """Test chat message with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function={"name": "test", "arguments": "{}"},
        )
        msg = ChatMessage(
            role=MessageRole.ASSISTANT, content=None, tool_calls=[tool_call]
        )

        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_123"

    def test_tool_definition(self):
        """Test tool definition creation."""
        tool = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            ),
        )

        assert tool.function.name == "get_weather"
        assert tool.function.description == "Get weather"

    def test_usage_model(self):
        """Test usage model."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
