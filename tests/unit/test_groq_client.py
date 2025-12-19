"""Unit tests for Groq API client."""

import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.groq_client import (
    ChatCompletionResponse,
    ChatMessage,
    CircuitBreaker,
    FunctionDefinition,
    GroqClient,
    GroqConfig,
    ImageUrlDetail,
    MessageRole,
    RateLimiter,
    ToolCall,
    ToolDefinition,
    Usage,
    close_all_groq_clients,
    get_shared_groq_client,
)


@pytest.fixture
def mock_chat_response():
    """Create a mock ChatCompletionResponse."""
    return {
        "id": "chatcmpl-groq-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-3.3-70b-versatile",
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
            "prompt_time": 0.001,
            "completion_time": 0.050,
            "total_time": 0.051,
        },
        "x_groq": {
            "id": "req_123",
        },
    }


@pytest.fixture
def mock_tool_response():
    """Create a mock response with tool calls."""
    return {
        "id": "chatcmpl-groq-456",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "llama-3.3-70b-versatile",
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
            "prompt_time": 0.002,
            "completion_time": 0.040,
            "total_time": 0.042,
        },
    }


@pytest.fixture
def groq_client():
    """Create a Groq client for testing."""
    return GroqClient(api_key="test-groq-key-123")


@pytest.fixture
async def cleanup_clients():
    """Cleanup shared clients after tests."""
    yield
    await close_all_groq_clients()


class TestRateLimiter:
    """Test RateLimiter functionality."""

    async def test_allows_requests_within_rate(self):
        """Test that requests within rate limit are allowed."""
        limiter = RateLimiter(rate=30.0, burst=60)

        # Should allow immediate requests up to burst
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be immediate

    async def test_delays_when_rate_exceeded(self):
        """Test that requests are delayed when rate is exceeded."""
        limiter = RateLimiter(rate=10.0, burst=1)

        # First request should be immediate
        await limiter.acquire()

        # Second request should be delayed
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        # Should wait approximately 1/10 = 0.1 seconds
        assert 0.05 < elapsed < 0.20


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


class TestGroqClient:
    """Test GroqClient functionality."""

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = GroqClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.default_model == "llama-3.3-70b-versatile"

    def test_initialization_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}):
            client = GroqClient()
            assert client.api_key == "env-key"

    def test_initialization_without_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Groq API key must be provided"):
                GroqClient()

    async def test_close(self, groq_client):
        """Test client cleanup."""
        # Initialize client
        await groq_client._get_client()
        assert groq_client._client is not None

        # Close it
        await groq_client.close()
        assert groq_client._client is None

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_success(self, mock_post, groq_client, mock_chat_response):
        """Test successful chat completion."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_chat_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = await groq_client.chat_completion(messages=messages)

        # Verify
        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-groq-123"
        assert result.choices[0].message.content == "Hello! How can I help you today?"
        assert result.usage.total_tokens == 19

        # Verify Groq-specific timing
        assert result.usage.prompt_time == 0.001
        assert result.usage.completion_time == 0.050
        assert result.usage.total_time == 0.051

        # Verify stats
        stats = groq_client.get_stats()
        assert stats["request_count"] == 1
        assert stats["total_tokens"] == 19
        assert stats["total_prompt_time"] == 0.001
        assert stats["total_completion_time"] == 0.050

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_with_tools(
        self, mock_post, groq_client, mock_tool_response
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
        result = await groq_client.chat_completion(messages=messages, tools=tools)

        # Verify
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].function.name == "get_weather"

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_retry_on_error(self, mock_post, groq_client):
        """Test retry logic on transient errors."""
        # Setup mock to fail twice then succeed
        mock_response_error = MagicMock()
        mock_response_error.status_code = 500
        mock_response_error.headers = {}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "id": "chatcmpl-groq-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Success"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
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
        result = await groq_client.chat_completion(messages=messages)

        # Verify it eventually succeeded
        assert result.id == "chatcmpl-groq-123"
        assert mock_post.call_count == 3

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_fails_on_client_error(self, mock_post, groq_client):
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
            await groq_client.chat_completion(messages=messages)

        # Should not retry on 4xx errors
        assert mock_post.call_count == 1

    @patch("httpx.AsyncClient.post")
    async def test_chat_completion_respects_retry_after(self, mock_post, groq_client):
        """Test that Retry-After header is respected."""
        # Setup mock to return 429 with Retry-After
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {"Retry-After": "0.1"}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "id": "chatcmpl-groq-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Success"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
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
        result = await groq_client.chat_completion(messages=messages)

        # Verify it succeeded after retry
        assert result.id == "chatcmpl-groq-123"

    @patch("httpx.AsyncClient.stream")
    async def test_chat_completion_streaming(self, mock_stream, groq_client):
        """Test streaming chat completion."""
        # Setup mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            lines = [
                'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{"content":"Hello"}}]}',
                'data: {"id":"1","object":"chat.completion.chunk","created":1234,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{"content":" world"}}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Make streaming request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        chunks = []
        async for chunk in groq_client.chat_completion_stream(messages=messages):
            chunks.append(chunk)

        # Verify
        assert chunks == ["Hello", " world"]

    @patch("httpx.AsyncClient.get")
    async def test_list_models(self, mock_get, groq_client):
        """Test listing available models."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "llama-3.3-70b-versatile"},
                {"id": "llama-3.1-70b-versatile"},
                {"id": "llama-3.1-8b-instant"},
                {"id": "mixtral-8x7b-32768"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Make request
        models = await groq_client.list_models()

        # Verify
        assert "llama-3.3-70b-versatile" in models
        assert "mixtral-8x7b-32768" in models
        assert len(models) == 4

    def test_encode_image_to_url(self, groq_client, tmp_path):
        """Test encoding image to base64 URL."""
        # Create a test image file
        image_path = tmp_path / "test.png"
        image_data = b"fake-image-data"
        image_path.write_bytes(image_data)

        # Encode it
        url = groq_client.encode_image_to_url(str(image_path))

        # Verify
        assert url.startswith("data:image/png;base64,")
        encoded_data = url.split(",")[1]
        assert base64.b64decode(encoded_data) == image_data

    def test_create_image_message(self, groq_client):
        """Test creating a message with images."""
        message = groq_client.create_image_message(
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

    def test_set_graph_context(self, groq_client):
        """Test setting graph context."""
        groq_client.set_graph_context("test-graph")
        assert groq_client._current_graph == "test-graph"

    def test_get_stats(self, groq_client):
        """Test getting client statistics."""
        stats = groq_client.get_stats()

        assert "request_count" in stats
        assert "total_tokens" in stats
        assert "circuit_breaker_state" in stats
        assert "circuit_breaker_failures" in stats
        assert "total_prompt_time" in stats
        assert "total_completion_time" in stats
        assert "avg_tokens_per_second" in stats

    @patch("httpx.AsyncClient.post")
    async def test_inference_speed_tracking(self, mock_post, groq_client, mock_chat_response):
        """Test that Groq's high-speed inference metrics are tracked."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_chat_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = await groq_client.chat_completion(messages=messages)

        # Verify inference speed metrics
        stats = groq_client.get_stats()
        assert stats["total_completion_time"] > 0
        assert stats["avg_tokens_per_second"] > 0

        # Calculate expected tokens per second
        expected_tps = result.usage.completion_tokens / result.usage.completion_time
        assert abs(stats["avg_tokens_per_second"] - expected_tps) < 0.01


class TestSharedClient:
    """Test shared client pool functionality."""

    async def test_get_shared_client(self, cleanup_clients):
        """Test getting a shared client."""
        client1 = await get_shared_groq_client(api_key="test-key")
        client2 = await get_shared_groq_client(api_key="test-key")

        # Should return the same instance
        assert client1 is client2

    async def test_different_configs_create_different_clients(self, cleanup_clients):
        """Test that different configs create different clients."""
        client1 = await get_shared_groq_client(api_key="key1")
        client2 = await get_shared_groq_client(api_key="key2")

        # Should be different instances
        assert client1 is not client2

    async def test_close_all_clients(self, cleanup_clients):
        """Test closing all shared clients."""
        # Create some clients
        await get_shared_groq_client(api_key="key1")
        await get_shared_groq_client(api_key="key2")

        # Close all
        await close_all_groq_clients()

        # Pool should be empty
        from tinyllm.providers.groq_client import _client_pool

        assert len(_client_pool) == 0


class TestGroqConfig:
    """Test GroqConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GroqConfig()

        assert config.base_url == "https://api.groq.com/openai/v1"
        assert config.timeout_ms == 30000
        assert config.max_retries == 3
        assert config.rate_limit_rps == 30.0
        assert config.default_model == "llama-3.3-70b-versatile"
        assert config.circuit_breaker_threshold == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GroqConfig(
            api_key="custom-key",
            base_url="https://custom.groq.com/v1",
            timeout_ms=60000,
            max_retries=5,
            rate_limit_rps=50.0,
            default_model="llama-3.1-70b-versatile",
            circuit_breaker_threshold=10,
        )

        assert config.api_key == "custom-key"
        assert config.base_url == "https://custom.groq.com/v1"
        assert config.timeout_ms == 60000
        assert config.max_retries == 5
        assert config.rate_limit_rps == 50.0
        assert config.default_model == "llama-3.1-70b-versatile"
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
        """Test usage model with Groq-specific timing."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_time=0.001,
            completion_time=0.050,
            total_time=0.051,
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        assert usage.prompt_time == 0.001
        assert usage.completion_time == 0.050
        assert usage.total_time == 0.051

    def test_usage_model_without_timing(self):
        """Test usage model without timing information."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        assert usage.prompt_time is None
        assert usage.completion_time is None
        assert usage.total_time is None
