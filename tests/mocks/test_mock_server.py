"""Tests for mock Ollama server."""

import pytest

from tinyllm.models.client import OllamaClient
from tests.mocks.mock_server import MockOllamaServer


@pytest.fixture
async def mock_server():
    """Create and start a mock Ollama server."""
    server = MockOllamaServer(port=11435)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
def mock_client(mock_server):
    """Create an Ollama client configured for the mock server."""
    return OllamaClient(host=mock_server.url, default_model="qwen2.5:3b")


class TestMockOllamaServer:
    """Tests for MockOllamaServer."""

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        """Test that server starts and stops cleanly."""
        server = MockOllamaServer(port=11436)
        await server.start()
        assert server.url == "http://127.0.0.1:11436"
        await server.stop()

    @pytest.mark.asyncio
    async def test_health_check(self, mock_client):
        """Test health check endpoint."""
        is_healthy = await mock_client.check_health()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_list_models(self, mock_client):
        """Test listing available models."""
        models = await mock_client.list_models()
        assert len(models) > 0
        assert "qwen2.5:3b" in models
        assert "granite-code:3b" in models

    @pytest.mark.asyncio
    async def test_generate_default_response(self, mock_client):
        """Test generating with default response."""
        response = await mock_client.generate(
            prompt="Hello, how are you?",
            model="qwen2.5:3b",
        )

        assert response.model == "qwen2.5:3b"
        assert len(response.response) > 0
        assert response.done is True
        assert response.eval_count is not None
        assert response.eval_count > 0

    @pytest.mark.asyncio
    async def test_generate_custom_response(self, mock_server, mock_client):
        """Test generating with custom response."""
        custom_prompt = "What is 2+2?"
        custom_response = "The answer is 4."

        mock_server.set_response(custom_prompt, custom_response)

        response = await mock_client.generate(
            prompt=custom_prompt,
            model="qwen2.5:3b",
        )

        assert response.response == custom_response

    @pytest.mark.asyncio
    async def test_generate_math_response(self, mock_client):
        """Test math prompt generates appropriate response."""
        response = await mock_client.generate(
            prompt="Calculate 10 + 20",
            model="qwen2.5:3b",
        )

        assert "42" in response.response

    @pytest.mark.asyncio
    async def test_generate_code_response(self, mock_client):
        """Test code prompt generates appropriate response."""
        response = await mock_client.generate(
            prompt="Write a Python function",
            model="granite-code:3b",
        )

        assert "def" in response.response
        assert "python" in response.response.lower()

    @pytest.mark.asyncio
    async def test_error_response(self, mock_server, mock_client):
        """Test simulated error response."""
        error_prompt = "This should fail"
        mock_server.set_error_response(error_prompt, 500, "Internal Server Error")

        with pytest.raises(Exception):
            await mock_client.generate(
                prompt=error_prompt,
                model="qwen2.5:3b",
            )

    @pytest.mark.asyncio
    async def test_request_tracking(self, mock_server, mock_client):
        """Test that requests are tracked."""
        initial_count = mock_server.request_count

        await mock_client.generate(
            prompt="Test prompt",
            model="qwen2.5:3b",
        )

        assert mock_server.request_count == initial_count + 1
        assert mock_server.get_request_count("/api/generate") > 0

        last_request = mock_server.get_last_request("/api/generate")
        assert last_request is not None
        assert last_request["data"]["prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_multiple_models(self, mock_server, mock_client):
        """Test generating with different models."""
        for model in ["qwen2.5:3b", "granite-code:3b", "phi3:mini"]:
            response = await mock_client.generate(
                prompt="Test",
                model=model,
            )
            assert response.model == model

    @pytest.mark.asyncio
    async def test_add_model(self, mock_server, mock_client):
        """Test adding a new model."""
        new_model = "custom-model:1b"
        mock_server.add_model(new_model)

        models = await mock_client.list_models()
        assert new_model in models

    @pytest.mark.asyncio
    async def test_remove_model(self, mock_server, mock_client):
        """Test removing a model."""
        model_to_remove = "phi3:mini"
        mock_server.remove_model(model_to_remove)

        models = await mock_client.list_models()
        assert model_to_remove not in models

    @pytest.mark.asyncio
    async def test_latency_simulation(self, mock_server, mock_client):
        """Test that latency is simulated."""
        import time

        mock_server.latency_ms = 200

        start = time.time()
        await mock_client.generate(
            prompt="Test",
            model="qwen2.5:3b",
        )
        duration_ms = (time.time() - start) * 1000

        # Should take at least the simulated latency
        assert duration_ms >= 200

    @pytest.mark.asyncio
    async def test_failure_simulation(self, mock_server):
        """Test simulated failures."""
        # Create a client with no retries for this test
        client = OllamaClient(
            host=mock_server.url,
            default_model="qwen2.5:3b",
            max_retries=0,
        )

        # Enable unlimited failures
        mock_server.enable_failures(max_fails=0)

        # Should fail
        with pytest.raises(Exception):
            await client.generate(
                prompt="Test",
                model="qwen2.5:3b",
            )

        # Disable failures
        mock_server.disable_failures()

        # Should succeed now
        response = await client.generate(
            prompt="Test",
            model="qwen2.5:3b",
        )
        assert response.done is True

        await client.close()

    @pytest.mark.asyncio
    async def test_clear_requests(self, mock_server, mock_client):
        """Test clearing request history."""
        await mock_client.generate(prompt="Test", model="qwen2.5:3b")
        assert mock_server.request_count > 0

        mock_server.clear_requests()
        assert mock_server.request_count == 0
        assert len(mock_server.requests) == 0

    @pytest.mark.asyncio
    async def test_system_prompt(self, mock_client):
        """Test with system prompt."""
        response = await mock_client.generate(
            prompt="What is your purpose?",
            system="You are a helpful assistant.",
            model="qwen2.5:3b",
        )

        assert response.response is not None
        assert len(response.response) > 0

    @pytest.mark.asyncio
    async def test_temperature_parameter(self, mock_client):
        """Test with different temperature values."""
        for temp in [0.1, 0.7, 1.5]:
            response = await mock_client.generate(
                prompt="Generate text",
                model="qwen2.5:3b",
                temperature=temp,
            )
            assert response.done is True

    @pytest.mark.asyncio
    async def test_max_tokens_parameter(self, mock_client):
        """Test with max_tokens parameter."""
        response = await mock_client.generate(
            prompt="Generate text",
            model="qwen2.5:3b",
            max_tokens=100,
        )
        assert response.done is True

    @pytest.mark.asyncio
    async def test_token_counts(self, mock_client):
        """Test that token counts are returned."""
        response = await mock_client.generate(
            prompt="This is a test prompt with multiple words.",
            model="qwen2.5:3b",
        )

        assert response.prompt_eval_count is not None
        assert response.prompt_eval_count > 0
        assert response.eval_count is not None
        assert response.eval_count > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_client):
        """Test handling concurrent requests."""
        import asyncio

        async def make_request(i):
            return await mock_client.generate(
                prompt=f"Request {i}",
                model="qwen2.5:3b",
            )

        # Make 5 concurrent requests
        responses = await asyncio.gather(*[make_request(i) for i in range(5)])

        assert len(responses) == 5
        for response in responses:
            assert response.done is True

    @pytest.mark.asyncio
    async def test_pull_model(self, mock_server, mock_client):
        """Test pulling a model."""
        new_model = "llama2:7b"
        await mock_client.pull_model(new_model)

        # Model should now be available
        models = await mock_client.list_models()
        assert new_model in models

    @pytest.mark.asyncio
    async def test_json_mode(self, mock_server, mock_client):
        """Test JSON mode generation."""
        mock_server.set_response(
            "Generate JSON",
            '{"name": "test", "value": 42}'
        )

        response = await mock_client.generate(
            prompt="Generate JSON",
            model="qwen2.5:3b",
            json_mode=True,
        )

        assert response.response.startswith("{")
        assert response.response.endswith("}")
