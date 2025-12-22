"""Tests for vLLM client."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.vllm_client import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    VLLMClient,
    VLLMConfig,
    close_all_vllm_clients,
    get_shared_vllm_client,
)


@pytest.fixture
def sample_chat_completion_response():
    """Create a sample ChatCompletionResponse."""
    return ChatCompletionResponse(
        id="chatcmpl-123",
        object="chat.completion",
        created=1677652288,
        model="meta-llama/Llama-2-7b-hf",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello! How can I help you today?"),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


@pytest.fixture
def vllm_client():
    """Create a VLLMClient instance."""
    return VLLMClient(
        host="http://localhost:8000",
        timeout_ms=5000,
        max_retries=2,
        default_model="meta-llama/Llama-2-7b-hf",
    )


# Client Tests


class TestVLLMClient:
    """Tests for VLLMClient."""

    def test_client_initialization(self, vllm_client):
        """Test client initializes with correct defaults."""
        assert vllm_client.host == "http://localhost:8000"
        assert vllm_client.max_retries == 2
        assert vllm_client._request_count == 0
        assert vllm_client._total_tokens == 0
        assert vllm_client._current_model == "meta-llama/Llama-2-7b-hf"

    def test_set_get_model(self, vllm_client):
        """Test model setter/getter."""
        vllm_client.set_model("meta-llama/Llama-2-13b-hf")
        assert vllm_client.get_model() == "meta-llama/Llama-2-13b-hf"

    def test_set_graph_context(self, vllm_client):
        """Test setting graph context."""
        vllm_client.set_graph_context("test-graph")
        assert vllm_client._current_graph == "test-graph"

    @pytest.mark.asyncio
    async def test_generate_success(self, vllm_client, sample_chat_completion_response):
        """Test successful generation."""
        with patch.object(vllm_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_chat_completion_response.model_dump())
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await vllm_client.generate("Hello")

            assert isinstance(result, ChatCompletionResponse)
            assert result.choices[0].message.content == "Hello! How can I help you today?"
            assert vllm_client._request_count == 1
            assert vllm_client._total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, vllm_client, sample_chat_completion_response):
        """Test generation with system prompt."""
        with patch.object(vllm_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_chat_completion_response.model_dump())
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await vllm_client.generate("Hello", system="You are a helpful assistant")

            # Verify system message was included
            call_args = mock_client.post.call_args
            request_json = call_args[1]["json"]
            messages = request_json["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant"
            assert messages[1]["role"] == "user"
            assert isinstance(result, ChatCompletionResponse)

    @pytest.mark.asyncio
    async def test_generate_no_model_raises(self):
        """Test generation without model raises error."""
        client = VLLMClient(default_model=None)

        with pytest.raises(ValueError, match="Model must be specified"):
            await client.generate("Hello")

    @pytest.mark.asyncio
    async def test_health_check_success(self, vllm_client):
        """Test successful health check."""
        with patch.object(vllm_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            is_healthy = await vllm_client.check_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, vllm_client):
        """Test failed health check."""
        with patch.object(vllm_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_get_client.return_value = mock_client

            is_healthy = await vllm_client.check_health()
            assert is_healthy is False

    def test_get_stats(self, vllm_client):
        """Test getting client statistics."""
        vllm_client._request_count = 10
        vllm_client._total_tokens = 500

        stats = vllm_client.get_stats()

        assert stats["request_count"] == 10
        assert stats["total_tokens"] == 500
        assert "circuit_breaker_state" in stats
        assert "slow_queries_count" in stats

    @pytest.mark.asyncio
    async def test_close(self, vllm_client):
        """Test client cleanup."""
        vllm_client._client = AsyncMock(spec=httpx.AsyncClient)

        await vllm_client.close()

        assert vllm_client._client is None

    @pytest.mark.asyncio
    async def test_list_models(self, vllm_client):
        """Test listing available models."""
        with patch.object(vllm_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(
                return_value={
                    "data": [
                        {"id": "meta-llama/Llama-2-7b-hf"},
                        {"id": "meta-llama/Llama-2-13b-hf"},
                    ]
                }
            )
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            models = await vllm_client.list_models()

            assert len(models) == 2
            assert "meta-llama/Llama-2-7b-hf" in models
            assert "meta-llama/Llama-2-13b-hf" in models


# Connection Pool Tests


class TestConnectionPool:
    """Tests for connection pooling."""

    @pytest.mark.asyncio
    async def test_shared_client_creation(self):
        """Test shared client is created once."""
        await close_all_vllm_clients()

        client1 = await get_shared_vllm_client(host="http://localhost:8000")
        client2 = await get_shared_vllm_client(host="http://localhost:8000")

        assert client1 is client2

        await close_all_vllm_clients()

    @pytest.mark.asyncio
    async def test_different_hosts_different_clients(self):
        """Test different hosts get different clients."""
        await close_all_vllm_clients()

        client1 = await get_shared_vllm_client(host="http://localhost:8000")
        client2 = await get_shared_vllm_client(host="http://localhost:8001")

        assert client1 is not client2

        await close_all_vllm_clients()

    @pytest.mark.asyncio
    async def test_close_all_clients(self):
        """Test closing all pooled clients."""
        await close_all_vllm_clients()

        await get_shared_vllm_client(host="http://localhost:8000")
        await get_shared_vllm_client(host="http://localhost:8001")

        await close_all_vllm_clients()

        client = await get_shared_vllm_client(host="http://localhost:8000")
        assert client is not None

        await close_all_vllm_clients()


# Configuration Tests


class TestVLLMConfig:
    """Tests for VLLMConfig."""

    def test_config_defaults(self):
        """Test config uses correct defaults."""
        config = VLLMConfig()

        assert config.host == "http://localhost:8000"
        assert config.timeout_ms == 30000
        assert config.max_retries == 3
        assert config.rate_limit_rps == 10.0
        assert config.circuit_breaker_threshold == 5

    def test_config_validation(self):
        """Test config validates values."""
        config = VLLMConfig(
            host="http://example.com:9000",
            timeout_ms=60000,
            max_retries=5,
            rate_limit_rps=20.0,
            default_model="meta-llama/Llama-2-7b-hf",
        )

        assert config.host == "http://example.com:9000"
        assert config.default_model == "meta-llama/Llama-2-7b-hf"

        with pytest.raises(Exception):
            VLLMConfig(timeout_ms=500)

        with pytest.raises(Exception):
            VLLMConfig(rate_limit_rps=0.0)


# Request/Response Model Tests


class TestModels:
    """Tests for Pydantic models."""

    def test_chat_message_creation(self):
        """Test creating chat message."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request_creation(self):
        """Test creating chat completion request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.8,
            max_tokens=100,
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.temperature == 0.8
        assert request.max_tokens == 100

    def test_chat_completion_response_parsing(self, sample_chat_completion_response):
        """Test parsing chat completion response."""
        assert sample_chat_completion_response.id == "chatcmpl-123"
        assert sample_chat_completion_response.model == "meta-llama/Llama-2-7b-hf"
        assert len(sample_chat_completion_response.choices) == 1
        assert sample_chat_completion_response.usage.total_tokens == 30
