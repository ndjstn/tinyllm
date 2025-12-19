"""Unit tests for Google Gemini API client."""

import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.gemini_client import (
    BatchEmbedContentsResponse,
    Blob,
    Candidate,
    CircuitBreaker,
    Content,
    ContentEmbedding,
    EmbedContentResponse,
    FinishReason,
    FunctionCall,
    FunctionCallPart,
    FunctionDeclaration,
    GenerateContentResponse,
    GenerationConfig,
    GeminiClient,
    HarmBlockThreshold,
    HarmCategory,
    InlineData,
    RateLimiter,
    SafetySetting,
    TextPart,
    Tool,
    UsageMetadata,
    close_all_gemini_clients,
    get_shared_gemini_client,
)


@pytest.fixture
def mock_generate_response():
    """Create a mock GenerateContentResponse."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I help you today?"}],
                    "role": "model",
                },
                "finish_reason": "STOP",
                "index": 0,
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 9,
            "total_token_count": 19,
        },
    }


@pytest.fixture
def mock_embedding_response():
    """Create a mock EmbedContentResponse."""
    return {
        "embedding": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    }


@pytest.fixture
def mock_batch_embedding_response():
    """Create a mock BatchEmbedContentsResponse."""
    return {
        "embeddings": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.4, 0.5, 0.6]},
        ]
    }


@pytest.fixture
def mock_function_call_response():
    """Create a mock response with function calls."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_weather",
                                "args": {"location": "San Francisco"},
                            }
                        }
                    ],
                    "role": "model",
                },
                "finish_reason": "STOP",
                "index": 0,
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 15,
            "candidates_token_count": 10,
            "total_token_count": 25,
        },
    }


@pytest.fixture
def gemini_client():
    """Create a Gemini client for testing."""
    return GeminiClient(api_key="test-key-123")


@pytest.fixture
async def cleanup_clients():
    """Cleanup shared clients after tests."""
    yield
    await close_all_gemini_clients()


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


class TestGeminiClient:
    """Test GeminiClient functionality."""

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = GeminiClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client._current_model == "gemini-1.5-pro"

    def test_initialization_from_env(self):
        """Test client initialization from environment variable."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            client = GeminiClient()
            assert client.api_key == "env-key"

    def test_initialization_without_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key must be provided"):
                GeminiClient()

    async def test_close(self, gemini_client):
        """Test client cleanup."""
        # Initialize client
        await gemini_client._get_client()
        assert gemini_client._client is not None

        # Close it
        await gemini_client.close()
        assert gemini_client._client is None

    @patch("httpx.AsyncClient.post")
    async def test_generate_content_success(
        self, mock_post, gemini_client, mock_generate_response
    ):
        """Test successful content generation."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_generate_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        contents = [Content(parts=[TextPart(text="Hello")], role="user")]
        result = await gemini_client.generate_content(contents=contents)

        # Verify
        assert isinstance(result, GenerateContentResponse)
        assert len(result.candidates) == 1
        assert result.get_text() == "Hello! How can I help you today?"
        assert result.usage_metadata.total_token_count == 19

        # Verify stats
        stats = gemini_client.get_stats()
        assert stats["request_count"] == 1
        assert stats["total_tokens"] == 19

    @patch("httpx.AsyncClient.post")
    async def test_generate_content_with_tools(
        self, mock_post, gemini_client, mock_function_call_response
    ):
        """Test content generation with function calls."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_function_call_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Define tool
        tools = [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="get_weather",
                        description="Get weather for a location",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    )
                ]
            )
        ]

        # Make request
        contents = [Content(parts=[TextPart(text="What's the weather?")], role="user")]
        result = await gemini_client.generate_content(contents=contents, tools=tools)

        # Verify
        function_calls = result.get_function_calls()
        assert len(function_calls) == 1
        assert function_calls[0].name == "get_weather"
        assert function_calls[0].args["location"] == "San Francisco"

    @patch("httpx.AsyncClient.post")
    async def test_generate_content_retry_on_error(self, mock_post, gemini_client):
        """Test retry logic on transient errors."""
        # Setup mock to fail twice then succeed
        mock_response_error = MagicMock()
        mock_response_error.status_code = 500
        mock_response_error.headers = {}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Success"}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
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
        contents = [Content(parts=[TextPart(text="Hello")], role="user")]
        result = await gemini_client.generate_content(contents=contents)

        # Verify it eventually succeeded
        assert result.get_text() == "Success"
        assert mock_post.call_count == 3

    @patch("httpx.AsyncClient.post")
    async def test_generate_content_respects_retry_after(self, mock_post, gemini_client):
        """Test that Retry-After header is respected."""
        # Setup mock to return 429 with Retry-After
        mock_response_error = MagicMock()
        mock_response_error.status_code = 429
        mock_response_error.headers = {"retry-after": "0.1"}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Success"}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
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
        contents = [Content(parts=[TextPart(text="Hello")], role="user")]
        result = await gemini_client.generate_content(contents=contents)

        # Verify it succeeded after retry
        assert result.get_text() == "Success"

    @patch("httpx.AsyncClient.stream")
    async def test_generate_content_streaming(self, mock_stream, gemini_client):
        """Test streaming content generation."""
        # Setup mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            lines = [
                'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}',
                'data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"index":0}]}',
                "data: [DONE]",
            ]
            for line in lines:
                yield line

        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Make streaming request
        contents = [Content(parts=[TextPart(text="Hello")], role="user")]
        chunks = []
        async for chunk in gemini_client.generate_content_stream(contents=contents):
            chunks.append(chunk.get_text())

        # Verify
        assert chunks == ["Hello", " world"]

    @patch("httpx.AsyncClient.post")
    async def test_embed_content_success(
        self, mock_post, gemini_client, mock_embedding_response
    ):
        """Test successful embedding creation."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_embedding_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        result = await gemini_client.embed_content("Hello world")

        # Verify
        assert isinstance(result, EmbedContentResponse)
        assert len(result.embedding.values) == 5
        assert result.embedding.values == [0.1, 0.2, 0.3, 0.4, 0.5]

    @patch("httpx.AsyncClient.post")
    async def test_batch_embed_contents_success(
        self, mock_post, gemini_client, mock_batch_embedding_response
    ):
        """Test successful batch embedding creation."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_batch_embedding_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Make request
        result = await gemini_client.batch_embed_contents(["Hello", "World"])

        # Verify
        assert isinstance(result, BatchEmbedContentsResponse)
        assert len(result.embeddings) == 2
        assert result.embeddings[0].values == [0.1, 0.2, 0.3]
        assert result.embeddings[1].values == [0.4, 0.5, 0.6]

    @patch("httpx.AsyncClient.get")
    async def test_list_models(self, mock_get, gemini_client):
        """Test listing available models."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "models/gemini-1.5-pro"},
                {"name": "models/gemini-1.5-flash"},
                {"name": "models/gemini-pro"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Make request
        models = await gemini_client.list_models()

        # Verify
        assert models == [
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash",
            "models/gemini-pro",
        ]

    def test_encode_image(self, gemini_client, tmp_path):
        """Test encoding image to base64."""
        # Create a test image file
        image_path = tmp_path / "test.png"
        image_data = b"fake-image-data"
        image_path.write_bytes(image_data)

        # Encode it
        inline_data = GeminiClient.encode_image(str(image_path), mime_type="image/png")

        # Verify
        assert isinstance(inline_data, InlineData)
        assert inline_data.inline_data.mime_type == "image/png"
        assert base64.b64decode(inline_data.inline_data.data) == image_data

    def test_create_image_content(self, gemini_client, tmp_path):
        """Test creating content with images."""
        # Create a test image
        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"test-image")

        # Encode image
        image_part = GeminiClient.encode_image(str(image_path), mime_type="image/jpeg")

        # Create content
        content = GeminiClient.create_image_content(
            text="What's in this image?",
            image_parts=[image_part],
        )

        # Verify
        assert content.role == "user"
        assert len(content.parts) == 2
        assert isinstance(content.parts[0], TextPart)
        assert content.parts[0].text == "What's in this image?"
        assert isinstance(content.parts[1], InlineData)

    def test_create_text_content(self, gemini_client):
        """Test creating simple text content."""
        content = GeminiClient.create_text_content("Hello world", role="user")

        # Verify
        assert content.role == "user"
        assert len(content.parts) == 1
        assert isinstance(content.parts[0], TextPart)
        assert content.parts[0].text == "Hello world"

    def test_set_graph_context(self, gemini_client):
        """Test setting graph context."""
        gemini_client.set_graph_context("test-graph")
        assert gemini_client._current_graph == "test-graph"

    def test_set_model(self, gemini_client):
        """Test setting model."""
        gemini_client.set_model("gemini-1.5-flash")
        assert gemini_client.get_model() == "gemini-1.5-flash"

    def test_get_stats(self, gemini_client):
        """Test getting client statistics."""
        stats = gemini_client.get_stats()

        assert "request_count" in stats
        assert "total_tokens" in stats
        assert "circuit_breaker_state" in stats
        assert "circuit_breaker_failures" in stats


class TestSharedClient:
    """Test shared client pool functionality."""

    async def test_get_shared_client(self, cleanup_clients):
        """Test getting a shared client."""
        client1 = await get_shared_gemini_client(api_key="test-key")
        client2 = await get_shared_gemini_client(api_key="test-key")

        # Should return the same instance
        assert client1 is client2

    async def test_different_configs_create_different_clients(self, cleanup_clients):
        """Test that different configs create different clients."""
        client1 = await get_shared_gemini_client(api_key="key1")
        client2 = await get_shared_gemini_client(api_key="key2")

        # Should be different instances
        assert client1 is not client2

    async def test_close_all_clients(self, cleanup_clients):
        """Test closing all shared clients."""
        # Create some clients
        await get_shared_gemini_client(api_key="key1")
        await get_shared_gemini_client(api_key="key2")

        # Close all
        await close_all_gemini_clients()

        # Pool should be empty
        from tinyllm.providers.gemini_client import _client_pool

        assert len(_client_pool) == 0


class TestPydanticModels:
    """Test Pydantic model validation and serialization."""

    def test_content_simple(self):
        """Test simple content creation."""
        content = Content(parts=[TextPart(text="Hello")], role="user")

        assert content.role == "user"
        assert len(content.parts) == 1
        assert isinstance(content.parts[0], TextPart)
        assert content.parts[0].text == "Hello"

    def test_content_with_image(self):
        """Test content with image."""
        blob = Blob(mime_type="image/jpeg", data="base64data")
        inline_data = InlineData(inline_data=blob)
        content = Content(parts=[TextPart(text="Image:"), inline_data], role="user")

        assert len(content.parts) == 2
        assert isinstance(content.parts[0], TextPart)
        assert isinstance(content.parts[1], InlineData)

    def test_function_call(self):
        """Test function call model."""
        func_call = FunctionCall(
            name="get_weather",
            args={"location": "San Francisco", "units": "celsius"},
        )

        assert func_call.name == "get_weather"
        assert func_call.args["location"] == "San Francisco"
        assert func_call.args["units"] == "celsius"

    def test_tool_definition(self):
        """Test tool definition creation."""
        tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                )
            ]
        )

        assert len(tool.function_declarations) == 1
        assert tool.function_declarations[0].name == "get_weather"
        assert tool.function_declarations[0].description == "Get weather"

    def test_generation_config(self):
        """Test generation config."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
        )

        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.top_k == 40
        assert config.max_output_tokens == 1024

    def test_safety_setting(self):
        """Test safety setting."""
        setting = SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        )

        assert setting.category == HarmCategory.HARM_CATEGORY_HARASSMENT
        assert setting.threshold == HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE

    def test_usage_metadata(self):
        """Test usage metadata."""
        usage = UsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        )

        assert usage.prompt_token_count == 10
        assert usage.candidates_token_count == 20
        assert usage.total_token_count == 30

    def test_generate_content_response_get_text(self):
        """Test extracting text from response."""
        response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[TextPart(text="Hello"), TextPart(text=" world")],
                        role="model",
                    ),
                    index=0,
                )
            ]
        )

        assert response.get_text() == "Hello world"

    def test_generate_content_response_get_function_calls(self):
        """Test extracting function calls from response."""
        func_call = FunctionCall(name="test_func", args={"arg": "value"})
        response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[FunctionCallPart(function_call=func_call)],
                        role="model",
                    ),
                    index=0,
                )
            ]
        )

        calls = response.get_function_calls()
        assert len(calls) == 1
        assert calls[0].name == "test_func"
        assert calls[0].args["arg"] == "value"
