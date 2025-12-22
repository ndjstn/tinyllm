"""Tests for custom model server client."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tinyllm.providers.custom_client import (
    CustomModelClient,
    CustomModelConfig,
    CustomModelResponse,
    RequestTemplate,
    ResponseExtractor,
    close_all_custom_clients,
    get_shared_custom_client,
)


@pytest.fixture
def openai_style_config():
    """Config for OpenAI-style API."""
    return CustomModelConfig(
        name="openai-style",
        base_url="http://localhost:8000",
        request=RequestTemplate(
            method="POST",
            endpoint="/v1/completions",
            body_template="""{
                "model": "${model}",
                "prompt": "${prompt}",
                "temperature": ${temperature},
                "max_tokens": ${max_tokens}
            }""",
            headers={"Content-Type": "application/json"},
        ),
        response=ResponseExtractor(
            text_path="choices.0.text",
            input_tokens_path="usage.prompt_tokens",
            output_tokens_path="usage.completion_tokens",
            finish_reason_path="choices.0.finish_reason",
            model_path="model",
        ),
        default_model="test-model",
    )


@pytest.fixture
def custom_style_config():
    """Config for custom API format."""
    return CustomModelConfig(
        name="custom-style",
        base_url="http://localhost:9000",
        request=RequestTemplate(
            method="POST",
            endpoint="/generate",
            body_template="""{
                "input": "${prompt}",
                "params": {
                    "temp": ${temperature},
                    "max_len": ${max_tokens}
                }
            }""",
        ),
        response=ResponseExtractor(
            text_path="output.text",
            input_tokens_path="stats.input_tokens",
            output_tokens_path="stats.output_tokens",
        ),
    )


@pytest.fixture
def sample_openai_response():
    """Sample OpenAI-style response."""
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [
            {
                "text": "Hello! How can I help you today?",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def sample_custom_response():
    """Sample custom format response."""
    return {
        "output": {
            "text": "This is a custom response.",
            "confidence": 0.95,
        },
        "stats": {
            "input_tokens": 15,
            "output_tokens": 10,
        },
    }


# Client Tests


class TestCustomModelClient:
    """Tests for CustomModelClient."""

    def test_client_initialization(self, openai_style_config):
        """Test client initializes with correct config."""
        client = CustomModelClient(config=openai_style_config)

        assert client.name == "openai-style"
        assert client.base_url == "http://localhost:8000"
        assert client.max_retries == 3
        assert client._request_count == 0
        assert client._total_tokens == 0
        assert client._current_model == "test-model"

    def test_set_get_model(self, openai_style_config):
        """Test model setter/getter."""
        client = CustomModelClient(config=openai_style_config)

        client.set_model("new-model")
        assert client.get_model() == "new-model"

    def test_set_graph_context(self, openai_style_config):
        """Test setting graph context."""
        client = CustomModelClient(config=openai_style_config)

        client.set_graph_context("test-graph")
        assert client._current_graph == "test-graph"

    def test_extract_value_simple(self, openai_style_config):
        """Test extracting simple nested values."""
        client = CustomModelClient(config=openai_style_config)

        data = {
            "level1": {
                "level2": {
                    "value": "found"
                }
            }
        }

        result = client._extract_value(data, "level1.level2.value")
        assert result == "found"

    def test_extract_value_array(self, openai_style_config):
        """Test extracting from array."""
        client = CustomModelClient(config=openai_style_config)

        data = {
            "items": [
                {"name": "first"},
                {"name": "second"},
            ]
        }

        result = client._extract_value(data, "items.1.name")
        assert result == "second"

    def test_extract_value_missing_path(self, openai_style_config):
        """Test extracting from missing path returns None."""
        client = CustomModelClient(config=openai_style_config)

        data = {"key": "value"}

        result = client._extract_value(data, "missing.path.here")
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_openai_style(self, openai_style_config, sample_openai_response):
        """Test generation with OpenAI-style API."""
        client = CustomModelClient(config=openai_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_openai_response)
            mock_response.raise_for_status = MagicMock()
            mock_http_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.generate("Hello")

            assert isinstance(result, CustomModelResponse)
            assert result.text == "Hello! How can I help you today?"
            assert result.input_tokens == 10
            assert result.output_tokens == 20
            assert result.total_tokens == 30
            assert result.finish_reason == "stop"
            assert result.model == "test-model"
            assert client._request_count == 1
            assert client._total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_custom_style(self, custom_style_config, sample_custom_response):
        """Test generation with custom API format."""
        client = CustomModelClient(config=custom_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_custom_response)
            mock_response.raise_for_status = MagicMock()
            mock_http_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.generate("Test prompt")

            assert isinstance(result, CustomModelResponse)
            assert result.text == "This is a custom response."
            assert result.input_tokens == 15
            assert result.output_tokens == 10
            assert result.total_tokens == 25

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, openai_style_config, sample_openai_response):
        """Test generation with custom parameters."""
        client = CustomModelClient(config=openai_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=sample_openai_response)
            mock_response.raise_for_status = MagicMock()
            mock_http_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.generate(
                "Hello",
                model="custom-model",
                temperature=0.8,
                max_tokens=100,
            )

            # Verify request was made with correct parameters
            call_args = mock_http_client.request.call_args
            request_json = call_args[1]["json"]

            assert request_json["model"] == "custom-model"
            assert request_json["temperature"] == 0.8
            assert request_json["max_tokens"] == 100
            assert isinstance(result, CustomModelResponse)

    @pytest.mark.asyncio
    async def test_generate_missing_text_path_raises(self, openai_style_config):
        """Test generation fails if text path not found."""
        client = CustomModelClient(config=openai_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Response missing the text field
            mock_response.json = MagicMock(return_value={"invalid": "response"})
            mock_response.raise_for_status = MagicMock()
            mock_http_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            # Should fail after retries with RuntimeError wrapping ValueError
            with pytest.raises(RuntimeError, match="Failed after .* retries"):
                await client.generate("Hello")

    @pytest.mark.asyncio
    async def test_health_check_success(self, openai_style_config):
        """Test successful health check."""
        client = CustomModelClient(config=openai_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_http_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            is_healthy = await client.check_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_fallback(self, openai_style_config):
        """Test health check fallback to root endpoint."""
        client = CustomModelClient(config=openai_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()

            # First call to /health fails
            health_response = AsyncMock()
            health_response.status_code = 404

            # Second call to / succeeds
            root_response = AsyncMock()
            root_response.status_code = 200

            mock_http_client.get.side_effect = [
                httpx.HTTPError("Not found"),
                root_response,
            ]
            mock_get_client.return_value = mock_http_client

            is_healthy = await client.check_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, openai_style_config):
        """Test failed health check."""
        client = CustomModelClient(config=openai_style_config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_get_client.return_value = mock_http_client

            is_healthy = await client.check_health()
            assert is_healthy is False

    def test_get_stats(self, openai_style_config):
        """Test getting client statistics."""
        client = CustomModelClient(config=openai_style_config)
        client._request_count = 10
        client._total_tokens = 500

        stats = client.get_stats()

        assert stats["name"] == "openai-style"
        assert stats["base_url"] == "http://localhost:8000"
        assert stats["request_count"] == 10
        assert stats["total_tokens"] == 500
        assert "circuit_breaker_state" in stats
        assert "slow_queries_count" in stats

    @pytest.mark.asyncio
    async def test_close(self, openai_style_config):
        """Test client cleanup."""
        client = CustomModelClient(config=openai_style_config)
        client._client = AsyncMock(spec=httpx.AsyncClient)

        await client.close()

        assert client._client is None

    @pytest.mark.asyncio
    async def test_authentication_header(self):
        """Test authentication header is added."""
        config = CustomModelConfig(
            name="auth-test",
            base_url="http://localhost:8000",
            api_key="test-key-123",
            auth_header="X-API-Key",
            auth_prefix="Key",
            request=RequestTemplate(
                method="POST",
                endpoint="/generate",
                body_template='{"prompt": "${prompt}"}',
            ),
            response=ResponseExtractor(text_path="text"),
        )

        client = CustomModelClient(config=config)
        http_client = await client._get_client()

        # Check that auth header was added
        assert "X-API-Key" in http_client.headers
        assert http_client.headers["X-API-Key"] == "Key test-key-123"

        await client.close()


# Connection Pool Tests


class TestConnectionPool:
    """Tests for connection pooling."""

    @pytest.mark.asyncio
    async def test_shared_client_creation(self, openai_style_config):
        """Test shared client is created once."""
        await close_all_custom_clients()

        client1 = await get_shared_custom_client(config=openai_style_config)
        client2 = await get_shared_custom_client(config=openai_style_config)

        assert client1 is client2

        await close_all_custom_clients()

    @pytest.mark.asyncio
    async def test_different_configs_different_clients(self, openai_style_config, custom_style_config):
        """Test different configs get different clients."""
        await close_all_custom_clients()

        client1 = await get_shared_custom_client(config=openai_style_config)
        client2 = await get_shared_custom_client(config=custom_style_config)

        assert client1 is not client2
        assert client1.name == "openai-style"
        assert client2.name == "custom-style"

        await close_all_custom_clients()

    @pytest.mark.asyncio
    async def test_close_all_clients(self, openai_style_config, custom_style_config):
        """Test closing all pooled clients."""
        await close_all_custom_clients()

        await get_shared_custom_client(config=openai_style_config)
        await get_shared_custom_client(config=custom_style_config)

        await close_all_custom_clients()

        # Pool should be empty - new client should be created
        client = await get_shared_custom_client(config=openai_style_config)
        assert client is not None

        await close_all_custom_clients()


# Configuration Tests


class TestCustomModelConfig:
    """Tests for CustomModelConfig."""

    def test_config_creation(self):
        """Test creating valid config."""
        config = CustomModelConfig(
            name="test-server",
            base_url="http://localhost:8000",
            request=RequestTemplate(
                method="POST",
                endpoint="/api/generate",
                body_template='{"prompt": "${prompt}"}',
            ),
            response=ResponseExtractor(text_path="result.text"),
        )

        assert config.name == "test-server"
        assert config.base_url == "http://localhost:8000"
        assert config.timeout_ms == 30000
        assert config.max_retries == 3

    def test_config_with_auth(self):
        """Test config with authentication."""
        config = CustomModelConfig(
            name="auth-server",
            base_url="http://localhost:8000",
            api_key="secret-key",
            auth_header="Authorization",
            auth_prefix="Bearer",
            request=RequestTemplate(
                method="POST",
                endpoint="/generate",
                body_template='{"prompt": "${prompt}"}',
            ),
            response=ResponseExtractor(text_path="text"),
        )

        assert config.api_key == "secret-key"
        assert config.auth_header == "Authorization"
        assert config.auth_prefix == "Bearer"

    def test_config_validation(self):
        """Test config validates values."""
        with pytest.raises(Exception):
            CustomModelConfig(
                name="test",
                base_url="http://localhost:8000",
                timeout_ms=500,  # Below minimum
                request=RequestTemplate(
                    method="POST",
                    endpoint="/generate",
                    body_template='{"prompt": "${prompt}"}',
                ),
                response=ResponseExtractor(text_path="text"),
            )

        with pytest.raises(Exception):
            CustomModelConfig(
                name="test",
                base_url="http://localhost:8000",
                rate_limit_rps=0.0,  # Below minimum
                request=RequestTemplate(
                    method="POST",
                    endpoint="/generate",
                    body_template='{"prompt": "${prompt}"}',
                ),
                response=ResponseExtractor(text_path="text"),
            )


# Model Tests


class TestModels:
    """Tests for Pydantic models."""

    def test_request_template_creation(self):
        """Test creating request template."""
        template = RequestTemplate(
            method="POST",
            endpoint="/v1/completions",
            body_template='{"model": "${model}", "prompt": "${prompt}"}',
            headers={"Content-Type": "application/json"},
        )

        assert template.method == "POST"
        assert template.endpoint == "/v1/completions"
        assert "${model}" in template.body_template

    def test_response_extractor_creation(self):
        """Test creating response extractor."""
        extractor = ResponseExtractor(
            text_path="choices.0.text",
            input_tokens_path="usage.prompt_tokens",
            output_tokens_path="usage.completion_tokens",
            finish_reason_path="choices.0.finish_reason",
        )

        assert extractor.text_path == "choices.0.text"
        assert extractor.input_tokens_path == "usage.prompt_tokens"

    def test_custom_model_response_creation(self):
        """Test creating custom model response."""
        response = CustomModelResponse(
            text="Generated text",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            model="test-model",
        )

        assert response.text == "Generated text"
        assert response.total_tokens == 30
        assert response.finish_reason == "stop"
