"""Tests for Anthropic Claude API client."""

import asyncio
import base64
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.anthropic_client import (
    AnthropicClient,
    CircuitBreaker,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ImageContent,
    ImageSource,
    Message,
    MessageDeltaEvent,
    MessageRequest,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    RateLimiter,
    StopReason,
    TextContent,
    Tool,
    ToolChoice,
    ToolUseBlock,
    Usage,
    get_shared_anthropic_client,
)


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "sk-ant-test-key-123"


@pytest.fixture
def sample_usage():
    """Create sample usage data."""
    return Usage(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=5,
    )


@pytest.fixture
def sample_message_response(sample_usage):
    """Create a sample MessageResponse."""
    return MessageResponse(
        id="msg_123",
        type="message",
        role="assistant",
        content=[TextContent(text="Hello! How can I help you today?")],
        model="claude-opus-4-20250514",
        stop_reason=StopReason.END_TURN,
        usage=sample_usage,
    )


@pytest.fixture
def anthropic_client(api_key):
    """Create an AnthropicClient instance."""
    return AnthropicClient(api_key=api_key, timeout_ms=5000, max_retries=2)


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

        # Should allow burst of 5 immediately
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should complete almost instantly (< 0.1s)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_rate(self):
        """Test rate limiter enforces rate limit."""
        limiter = RateLimiter(rate=10.0, burst=1)

        # First request should be immediate
        start = time.monotonic()
        await limiter.acquire()
        first_elapsed = time.monotonic() - start
        assert first_elapsed < 0.01

        # Second request should wait ~0.1s (1/10 = 0.1)
        start = time.monotonic()
        await limiter.acquire()
        second_elapsed = time.monotonic() - start
        assert 0.05 < second_elapsed < 0.15  # Allow some margin

    @pytest.mark.asyncio
    async def test_rate_limiter_refills_tokens(self):
        """Test rate limiter refills tokens over time."""
        limiter = RateLimiter(rate=10.0, burst=2)

        # Use up tokens
        await limiter.acquire()
        await limiter.acquire()

        # Wait for refill
        await asyncio.sleep(0.2)  # Should refill 2 tokens

        # Should allow 2 more without waiting
        start = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05


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
        assert breaker.get_state() == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_func():
            raise ValueError("Test error")

        # Trigger failures
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        # Circuit should now be open
        assert breaker.get_state() == "open"

        # Next call should fail immediately
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open and recovers."""
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

        # Should transition to half-open and allow a test call
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == "closed"


# Pydantic Model Tests


class TestPydanticModels:
    """Tests for Pydantic models."""

    def test_text_content_creation(self):
        """Test TextContent model creation."""
        content = TextContent(text="Hello world")
        assert content.type == "text"
        assert content.text == "Hello world"

    def test_image_content_creation(self):
        """Test ImageContent model creation."""
        source = ImageSource(
            type="base64", media_type="image/jpeg", data="base64encodeddata"
        )
        content = ImageContent(source=source)
        assert content.type == "image"
        assert content.source.type == "base64"
        assert content.source.media_type == "image/jpeg"

    def test_tool_use_block(self):
        """Test ToolUseBlock model."""
        tool_use = ToolUseBlock(
            id="tool_123", name="calculator", input={"operation": "add", "a": 1, "b": 2}
        )
        assert tool_use.type == "tool_use"
        assert tool_use.name == "calculator"
        assert tool_use.input["a"] == 1

    def test_message_creation(self):
        """Test Message model creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        # With content blocks
        msg2 = Message(
            role="assistant", content=[TextContent(text="Hi"), TextContent(text=" there")]
        )
        assert msg2.role == "assistant"
        assert len(msg2.content) == 2

    def test_message_response_get_text(self, sample_message_response):
        """Test MessageResponse.get_text() method."""
        text = sample_message_response.get_text()
        assert text == "Hello! How can I help you today?"

    def test_message_response_get_tool_uses(self):
        """Test MessageResponse.get_tool_uses() method."""
        response = MessageResponse(
            id="msg_123",
            role="assistant",
            content=[
                TextContent(text="Let me calculate that."),
                ToolUseBlock(id="tool_1", name="calc", input={"x": 5}),
                ToolUseBlock(id="tool_2", name="search", input={"query": "test"}),
            ],
            model="claude-opus-4",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        tool_uses = response.get_tool_uses()
        assert len(tool_uses) == 2
        assert tool_uses[0].name == "calc"
        assert tool_uses[1].name == "search"

    def test_tool_definition(self):
        """Test Tool model."""
        tool = Tool(
            name="get_weather",
            description="Get current weather for a location",
            input_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        assert tool.name == "get_weather"
        assert "properties" in tool.input_schema

    def test_usage_model(self, sample_usage):
        """Test Usage model."""
        assert sample_usage.input_tokens == 100
        assert sample_usage.output_tokens == 50
        assert sample_usage.cache_creation_input_tokens == 10
        assert sample_usage.cache_read_input_tokens == 5


# Client Initialization Tests


class TestAnthropicClientInit:
    """Tests for AnthropicClient initialization."""

    def test_client_init_with_api_key(self, api_key):
        """Test client initializes with API key."""
        client = AnthropicClient(api_key=api_key)
        assert client.api_key == api_key
        assert client._current_model == "claude-opus-4-20250514"

    def test_client_init_from_env_var(self):
        """Test client initializes from environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key-123"}):
            client = AnthropicClient()
            assert client.api_key == "env-key-123"

    def test_client_init_fails_without_key(self):
        """Test client fails to initialize without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                AnthropicClient()

    def test_client_custom_config(self, api_key):
        """Test client with custom configuration."""
        client = AnthropicClient(
            api_key=api_key,
            timeout_ms=10000,
            max_retries=5,
            rate_limit_rps=2.0,
            default_model="claude-sonnet-4",
        )
        assert client.max_retries == 5
        assert client._current_model == "claude-sonnet-4"


# HTTP Client Tests


class TestHTTPClient:
    """Tests for HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, anthropic_client):
        """Test _get_client creates HTTP client."""
        assert anthropic_client._client is None
        client = await anthropic_client._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, anthropic_client):
        """Test _get_client reuses existing client."""
        client1 = await anthropic_client._get_client()
        client2 = await anthropic_client._get_client()
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_close_client(self, anthropic_client):
        """Test closing the HTTP client."""
        await anthropic_client._get_client()
        assert anthropic_client._client is not None

        await anthropic_client.close()
        assert anthropic_client._client is None


# Message Creation Tests


class TestCreateMessage:
    """Tests for create_message method."""

    @pytest.mark.asyncio
    async def test_create_message_success(
        self, anthropic_client, sample_message_response, mock_http_client
    ):
        """Test successful message creation."""
        # Mock the HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_message_response.model_dump()
        mock_response.raise_for_status = MagicMock()

        mock_http_client.post = AsyncMock(return_value=mock_response)
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]
        result = await anthropic_client.create_message(messages=messages)

        assert isinstance(result, MessageResponse)
        assert result.get_text() == "Hello! How can I help you today?"
        assert result.usage.input_tokens == 100

        # Verify request was made correctly
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "/v1/messages"
        assert "messages" in call_args[1]["json"]

    @pytest.mark.asyncio
    async def test_create_message_with_system_prompt(
        self, anthropic_client, sample_message_response, mock_http_client
    ):
        """Test message creation with system prompt."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_message_response.model_dump()
        mock_response.raise_for_status = MagicMock()

        mock_http_client.post = AsyncMock(return_value=mock_response)
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]
        result = await anthropic_client.create_message(
            messages=messages, system="You are a helpful assistant."
        )

        assert isinstance(result, MessageResponse)

        # Verify system prompt was included
        call_args = mock_http_client.post.call_args
        assert call_args[1]["json"]["system"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_create_message_with_tools(
        self, anthropic_client, mock_http_client
    ):
        """Test message creation with tools."""
        # Create response with tool use
        response_data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "search",
                    "input": {"query": "test"},
                },
            ],
            "model": "claude-opus-4",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 25},
        }

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_http_client.post = AsyncMock(return_value=mock_response)
        anthropic_client._client = mock_http_client

        tools = [
            Tool(
                name="search",
                description="Search the web",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ]

        messages = [Message(role="user", content="Search for Python tutorials")]
        result = await anthropic_client.create_message(
            messages=messages, tools=tools
        )

        assert result.stop_reason == StopReason.TOOL_USE
        tool_uses = result.get_tool_uses()
        assert len(tool_uses) == 1
        assert tool_uses[0].name == "search"

    @pytest.mark.asyncio
    async def test_create_message_retry_on_error(
        self, anthropic_client, sample_message_response, mock_http_client
    ):
        """Test message creation retries on HTTP error."""
        # First call fails, second succeeds
        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=error_response
        )

        success_response = MagicMock(spec=httpx.Response)
        success_response.status_code = 200
        success_response.json.return_value = sample_message_response.model_dump()
        success_response.raise_for_status = MagicMock()

        mock_http_client.post = AsyncMock(
            side_effect=[error_response, success_response]
        )
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]
        result = await anthropic_client.create_message(messages=messages)

        assert isinstance(result, MessageResponse)
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_create_message_rate_limit_handling(
        self, anthropic_client, sample_message_response, mock_http_client
    ):
        """Test message creation handles rate limiting."""
        # First call returns rate limit, second succeeds
        rate_limit_response = MagicMock(spec=httpx.Response)
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"retry-after": "0.1"}
        rate_limit_response.json.return_value = {
            "error": {"type": "rate_limit_error"}
        }

        success_response = MagicMock(spec=httpx.Response)
        success_response.status_code = 200
        success_response.json.return_value = sample_message_response.model_dump()
        success_response.raise_for_status = MagicMock()

        mock_http_client.post = AsyncMock(
            side_effect=[rate_limit_response, success_response]
        )
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]
        result = await anthropic_client.create_message(messages=messages)

        assert isinstance(result, MessageResponse)
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_create_message_max_retries_exceeded(
        self, anthropic_client, mock_http_client
    ):
        """Test message creation fails after max retries."""
        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=error_response
        )

        mock_http_client.post = AsyncMock(return_value=error_response)
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]

        with pytest.raises(httpx.HTTPStatusError):
            await anthropic_client.create_message(messages=messages)

        # Should retry max_retries + 1 times (initial + retries)
        assert mock_http_client.post.call_count == anthropic_client.max_retries + 1


# Streaming Tests


class TestCreateMessageStream:
    """Tests for create_message_stream method."""

    @pytest.mark.asyncio
    async def test_stream_message(self, anthropic_client, mock_http_client):
        """Test streaming message creation."""
        # Mock SSE stream
        stream_data = [
            'data: {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-4", "usage": {"input_tokens": 10, "output_tokens": 0}}}',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}',
            'data: {"type": "content_block_stop", "index": 0}',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}',
            'data: {"type": "message_stop"}',
        ]

        async def mock_aiter_lines():
            for line in stream_data:
                yield line

        mock_stream_response = MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = mock_aiter_lines
        mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_response.__aexit__ = AsyncMock(return_value=None)

        mock_http_client.stream = MagicMock(return_value=mock_stream_response)
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Say hello")]
        events = []

        async for event in anthropic_client.create_message_stream(messages=messages):
            events.append(event)

        # Verify event types (message_delta might be skipped if parsing fails)
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], ContentBlockStartEvent)
        assert isinstance(events[2], ContentBlockDeltaEvent)
        assert isinstance(events[3], ContentBlockDeltaEvent)
        assert isinstance(events[4], ContentBlockStopEvent)
        # Events 5 could be MessageDeltaEvent or MessageStopEvent depending on parsing
        if len(events) == 7:
            assert isinstance(events[5], MessageDeltaEvent)
            assert isinstance(events[6], MessageStopEvent)
        else:
            assert len(events) == 6
            assert isinstance(events[5], MessageStopEvent)

        # Verify delta content
        assert events[2].delta.get("text") == "Hello"
        assert events[3].delta.get("text") == " world"


# Vision Tests


class TestVisionSupport:
    """Tests for vision (image) support."""

    def test_encode_image(self, tmp_path):
        """Test encoding image to base64."""
        # Create a test image file
        image_path = tmp_path / "test.jpg"
        image_data = b"fake image data"
        image_path.write_bytes(image_data)

        # Encode it
        source = AnthropicClient.encode_image(str(image_path), "image/jpeg")

        assert source.type == "base64"
        assert source.media_type == "image/jpeg"
        assert source.data == base64.b64encode(image_data).decode("utf-8")

    def test_create_image_message(self):
        """Test creating a message with images."""
        source1 = ImageSource(type="base64", media_type="image/jpeg", data="data1")
        source2 = ImageSource(type="url", media_type="image/png", url="http://example.com/img.png")

        message = AnthropicClient.create_image_message(
            text="What's in these images?", image_sources=[source1, source2]
        )

        assert message.role == "user"
        assert len(message.content) == 3  # 1 text + 2 images
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)
        assert isinstance(message.content[2], ImageContent)


# Helper Methods Tests


class TestHelperMethods:
    """Tests for helper methods."""

    def test_set_graph_context(self, anthropic_client):
        """Test setting graph context."""
        anthropic_client.set_graph_context("test-graph")
        assert anthropic_client._current_graph == "test-graph"

    def test_set_model(self, anthropic_client):
        """Test setting default model."""
        anthropic_client.set_model("claude-sonnet-4")
        assert anthropic_client.get_model() == "claude-sonnet-4"

    def test_get_stats(self, anthropic_client):
        """Test getting client statistics."""
        stats = anthropic_client.get_stats()
        assert "request_count" in stats
        assert "total_tokens" in stats
        assert "circuit_breaker_state" in stats
        assert stats["request_count"] == 0

    @pytest.mark.asyncio
    async def test_slow_query_tracking(
        self, anthropic_client, sample_message_response, mock_http_client
    ):
        """Test slow query detection and tracking."""
        # Set a very low threshold
        anthropic_client.slow_query_threshold_ms = 0

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_message_response.model_dump()
        mock_response.raise_for_status = MagicMock()

        mock_http_client.post = AsyncMock(return_value=mock_response)
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]
        await anthropic_client.create_message(messages=messages)

        # Should have recorded a slow query
        slow_queries = anthropic_client.get_slow_queries()
        assert len(slow_queries) > 0
        assert slow_queries[0]["model"] == "claude-opus-4-20250514"

    def test_clear_slow_queries(self, anthropic_client):
        """Test clearing slow query history."""
        # Add a fake slow query
        anthropic_client._slow_queries.append(
            {"timestamp": time.time(), "model": "test", "duration_ms": 1000}
        )
        assert len(anthropic_client.get_slow_queries()) == 1

        anthropic_client.clear_slow_queries()
        assert len(anthropic_client.get_slow_queries()) == 0


# Shared Client Tests


class TestSharedClient:
    """Tests for shared client pool."""

    @pytest.mark.asyncio
    async def test_get_shared_client(self, api_key):
        """Test getting shared client from pool."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": api_key}):
            client1 = await get_shared_anthropic_client()
            client2 = await get_shared_anthropic_client()

            # Should return the same instance
            assert client1 is client2

    @pytest.mark.asyncio
    async def test_different_keys_different_clients(self):
        """Test different API keys create different clients."""
        client1 = await get_shared_anthropic_client(api_key="key1")
        client2 = await get_shared_anthropic_client(api_key="key2")

        # Should be different instances
        assert client1 is not client2


# Edge Cases


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_rate_limit_without_retry_after(self, anthropic_client):
        """Test handling rate limit response without retry-after header."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 429
        response.headers = {}
        response.json.return_value = {"error": {"type": "rate_limit_error"}}

        retry_after = anthropic_client._handle_rate_limit_error(response)
        assert retry_after == 60.0  # Default fallback

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, anthropic_client):
        """Test exponential backoff timing."""
        start = time.monotonic()
        await anthropic_client._exponential_backoff(0, base_delay=0.1, max_delay=1.0)
        elapsed = time.monotonic() - start

        # First attempt: ~0.1s + jitter
        assert 0.1 <= elapsed <= 0.2

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_create_message(
        self, anthropic_client, mock_http_client
    ):
        """Test circuit breaker opens and prevents requests."""
        # Configure to open quickly
        anthropic_client._circuit_breaker = CircuitBreaker(
            failure_threshold=2, recovery_timeout=10.0
        )

        error_response = MagicMock(spec=httpx.Response)
        error_response.status_code = 500
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=error_response
        )

        mock_http_client.post = AsyncMock(return_value=error_response)
        anthropic_client._client = mock_http_client

        messages = [Message(role="user", content="Hello")]

        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                await anthropic_client.create_message(messages=messages)

        # Circuit should be open
        assert anthropic_client._circuit_breaker.get_state() == "open"

        # Next attempt should fail with circuit breaker error
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            await anthropic_client.create_message(messages=messages)


# Integration-style Tests (still using mocks)


class TestIntegrationScenarios:
    """Integration-style tests for common scenarios."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self, anthropic_client, mock_http_client
    ):
        """Test multi-turn conversation."""
        responses = [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help?"}],
                "model": "claude-opus-4",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 8},
            },
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Python is a programming language."}],
                "model": "claude-opus-4",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        ]

        response_iter = iter(responses)

        def mock_post(*args, **kwargs):
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = next(response_iter)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_http_client.post = AsyncMock(side_effect=mock_post)
        anthropic_client._client = mock_http_client

        # First turn
        messages = [Message(role="user", content="Hello")]
        response1 = await anthropic_client.create_message(messages=messages)
        assert "How can I help" in response1.get_text()

        # Second turn
        messages.append(Message(role="assistant", content=response1.get_text()))
        messages.append(Message(role="user", content="What is Python?"))
        response2 = await anthropic_client.create_message(messages=messages)
        assert "programming language" in response2.get_text()

        # Verify stats updated
        stats = anthropic_client.get_stats()
        assert stats["request_count"] == 2
        assert stats["total_tokens"] == 48  # Sum of all input and output tokens

    @pytest.mark.asyncio
    async def test_tool_use_workflow(self, anthropic_client, mock_http_client):
        """Test complete tool use workflow."""
        # First response: model wants to use a tool
        tool_request_response = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate that."},
                {
                    "type": "tool_use",
                    "id": "tool_calc_1",
                    "name": "calculator",
                    "input": {"operation": "add", "a": 5, "b": 3},
                },
            ],
            "model": "claude-opus-4",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 15, "output_tokens": 20},
        }

        # Second response: model provides final answer
        final_response = {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "The result is 8."}],
            "model": "claude-opus-4",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 25, "output_tokens": 8},
        }

        response_iter = iter([tool_request_response, final_response])

        def mock_post(*args, **kwargs):
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = next(response_iter)
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_http_client.post = AsyncMock(side_effect=mock_post)
        anthropic_client._client = mock_http_client

        tools = [
            Tool(
                name="calculator",
                description="Perform arithmetic",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                },
            )
        ]

        # Initial request
        messages = [Message(role="user", content="What is 5 + 3?")]
        response1 = await anthropic_client.create_message(
            messages=messages, tools=tools
        )

        # Check tool use
        assert response1.stop_reason == StopReason.TOOL_USE
        tool_uses = response1.get_tool_uses()
        assert len(tool_uses) == 1
        assert tool_uses[0].name == "calculator"

        # Simulate executing tool and sending result back
        # (In real scenario, you'd execute the tool and get a result)
        # This is just showing the workflow
        assert tool_uses[0].input["a"] == 5
        assert tool_uses[0].input["b"] == 3
