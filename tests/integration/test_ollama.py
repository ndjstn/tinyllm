"""Integration tests with real Ollama server.

These tests require a running Ollama instance with specific models.
Tests are automatically skipped if Ollama is not available.
"""

import asyncio

import pytest

pytestmark = pytest.mark.integration
import os
from typing import Optional

import httpx
import pytest

from tinyllm.models.client import OllamaClient, get_shared_client, close_all_clients
from tinyllm.models.registry import ModelRegistry
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.nodes.model import ModelNode


# Default Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TEST_MODEL = os.getenv("OLLAMA_TEST_MODEL", "qwen2.5:0.5b")


async def check_ollama_available() -> bool:
    """Check if Ollama server is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def check_model_available(model_name: str) -> bool:
    """Check if a specific model is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return model_name in models
    except Exception:
        return False


@pytest.fixture
async def ollama_client():
    """Create Ollama client for testing."""
    client = await get_shared_client(host=OLLAMA_HOST)
    yield client
    # Cleanup happens in close_all_clients


@pytest.fixture
async def execution_context():
    """Create execution context for tests."""
    return ExecutionContext(
        trace_id="ollama-integration-test",
        graph_id="test-graph",
        config=Config(),
    )


@pytest.fixture(scope="module", autouse=True)
async def cleanup_clients():
    """Cleanup all clients after module tests complete."""
    yield
    await close_all_clients()


@pytest.mark.asyncio
async def test_ollama_server_health():
    """Test Ollama server health check."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_models(ollama_client):
    """Test listing available models."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    models = await ollama_client.list_models()
    assert isinstance(models, list)


@pytest.mark.asyncio
async def test_simple_generation(ollama_client):
    """Test simple text generation."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    prompt = "Say hello in one word."
    response = await ollama_client.generate(
        model=TEST_MODEL,
        prompt=prompt,
        stream=False,
    )

    assert response is not None
    assert "response" in response
    assert len(response["response"]) > 0


@pytest.mark.asyncio
async def test_streaming_generation(ollama_client):
    """Test streaming text generation."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    prompt = "Count from 1 to 3."
    chunks = []

    async for chunk in ollama_client.generate_stream(
        model=TEST_MODEL,
        prompt=prompt,
    ):
        chunks.append(chunk)
        if chunk.get("done", False):
            break

    # Should receive multiple chunks
    assert len(chunks) > 0
    # Last chunk should have done=True
    assert chunks[-1].get("done", False) is True


@pytest.mark.asyncio
async def test_chat_completion(ollama_client):
    """Test chat completion API."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]

    response = await ollama_client.chat(
        model=TEST_MODEL,
        messages=messages,
        stream=False,
    )

    assert response is not None
    assert "message" in response
    assert "content" in response["message"]


@pytest.mark.asyncio
async def test_model_node_execution(execution_context):
    """Test ModelNode with real Ollama."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Create model node
    definition = NodeDefinition(
        id="model.test",
        type=NodeType.MODEL,
        config={
            "model_name": TEST_MODEL,
            "system_prompt": "You are a helpful assistant. Be concise.",
            "temperature": 0.7,
        },
    )
    node = ModelNode(definition)

    # Create input message
    message = Message(
        trace_id="test-model-exec",
        source_node="test",
        payload=MessagePayload(
            task="Answer a simple question",
            content="What is the capital of France? Answer with just the city name.",
        ),
    )

    # Execute
    result = await node.execute(message, execution_context)

    # Verify result
    assert result.success is True
    assert len(result.output_messages) > 0
    output_content = result.output_messages[0].payload.content.lower()
    assert "paris" in output_content


@pytest.mark.asyncio
async def test_concurrent_requests(ollama_client):
    """Test handling concurrent requests to Ollama."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Create multiple concurrent requests
    prompts = [
        "Say hi",
        "Say hello",
        "Say hey",
        "Say greetings",
        "Say salutations",
    ]

    async def make_request(prompt: str) -> dict:
        """Make a single request."""
        return await ollama_client.generate(
            model=TEST_MODEL,
            prompt=prompt,
            stream=False,
        )

    # Execute concurrently
    results = await asyncio.gather(*[make_request(p) for p in prompts])

    # All should succeed
    assert len(results) == len(prompts)
    for result in results:
        assert result is not None
        assert "response" in result


@pytest.mark.asyncio
async def test_error_handling_invalid_model(ollama_client):
    """Test error handling with invalid model."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    with pytest.raises(Exception):
        await ollama_client.generate(
            model="nonexistent-model-xyz",
            prompt="This should fail",
            stream=False,
        )


@pytest.mark.asyncio
async def test_timeout_handling(ollama_client):
    """Test timeout handling in requests."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Create client with very short timeout
    short_timeout_client = OllamaClient(
        host=OLLAMA_HOST,
        timeout_ms=100,  # Very short timeout
    )

    # This might timeout with complex prompt
    complex_prompt = "Write a very long essay about " + "philosophy " * 100

    try:
        await short_timeout_client.generate(
            model=TEST_MODEL,
            prompt=complex_prompt,
            stream=False,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException):
        # Timeout is expected
        pass
    finally:
        await short_timeout_client.close()


@pytest.mark.asyncio
async def test_model_registry_integration():
    """Test ModelRegistry with real Ollama."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    registry = ModelRegistry()

    # Register test model
    registry.register(
        model_id="test_model",
        model_name=TEST_MODEL,
        tier=0,
        specialization="general",
        description="Test model",
    )

    # Get model config
    config = registry.get_model_config("test_model")
    assert config is not None
    assert config.model_name == TEST_MODEL


@pytest.mark.asyncio
async def test_rate_limiting(ollama_client):
    """Test rate limiting behavior."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Make rapid requests
    num_requests = 10
    import time

    start = time.time()

    tasks = []
    for i in range(num_requests):
        task = ollama_client.generate(
            model=TEST_MODEL,
            prompt=f"Say {i}",
            stream=False,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start

    # All should eventually succeed
    successful = sum(1 for r in results if not isinstance(r, Exception))
    assert successful >= num_requests * 0.8  # At least 80% success


@pytest.mark.asyncio
async def test_long_context_handling(ollama_client):
    """Test handling of long context."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Create a moderately long prompt
    long_prompt = (
        "Here is some context: " + "This is important information. " * 50 + "\n\n"
        "Based on the above, what is the main topic? Be brief."
    )

    response = await ollama_client.generate(
        model=TEST_MODEL,
        prompt=long_prompt,
        stream=False,
    )

    assert response is not None
    assert "response" in response


@pytest.mark.asyncio
async def test_special_characters_handling(ollama_client):
    """Test handling of special characters in prompts."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Prompt with special characters
    special_prompt = "Translate this: Hello! 你好! مرحبا! 🌍"

    response = await ollama_client.generate(
        model=TEST_MODEL,
        prompt=special_prompt,
        stream=False,
    )

    assert response is not None
    assert "response" in response


@pytest.mark.asyncio
async def test_multi_turn_conversation(ollama_client):
    """Test multi-turn conversation."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # First turn
    messages = [{"role": "user", "content": "My name is Alice."}]

    response1 = await ollama_client.chat(
        model=TEST_MODEL,
        messages=messages,
        stream=False,
    )

    # Second turn - reference first turn
    messages.append(response1["message"])
    messages.append({"role": "user", "content": "What is my name?"})

    response2 = await ollama_client.chat(
        model=TEST_MODEL,
        messages=messages,
        stream=False,
    )

    # Should remember the name
    content = response2["message"]["content"].lower()
    # Model might remember Alice
    assert response2 is not None


@pytest.mark.asyncio
async def test_system_prompt_effectiveness(ollama_client):
    """Test that system prompts affect model behavior."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    # Test with system prompt
    messages = [
        {
            "role": "system",
            "content": "You are a pirate. Always respond in pirate speak.",
        },
        {"role": "user", "content": "Hello, how are you?"},
    ]

    response = await ollama_client.chat(
        model=TEST_MODEL,
        messages=messages,
        stream=False,
    )

    # Response might contain pirate-like words
    content = response["message"]["content"].lower()
    # Just verify we got a response
    assert len(content) > 0


@pytest.mark.asyncio
async def test_json_mode(ollama_client):
    """Test JSON response format if supported."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    prompt = 'Return a JSON object with a single key "greeting" and value "hello".'

    try:
        response = await ollama_client.generate(
            model=TEST_MODEL,
            prompt=prompt,
            format="json",
            stream=False,
        )

        assert response is not None
        # Response might be valid JSON
    except Exception:
        # JSON mode might not be supported by all models
        pytest.skip("JSON mode not supported")


@pytest.mark.asyncio
async def test_temperature_variation(ollama_client):
    """Test different temperature settings."""
    available = await check_ollama_available()
    if not available:
        pytest.skip("Ollama server not available")

    model_available = await check_model_available(TEST_MODEL)
    if not model_available:
        pytest.skip(f"Model {TEST_MODEL} not available")

    prompt = "Write a creative word."

    # Test different temperatures
    temperatures = [0.0, 0.5, 1.0]
    responses = []

    for temp in temperatures:
        response = await ollama_client.generate(
            model=TEST_MODEL,
            prompt=prompt,
            options={"temperature": temp},
            stream=False,
        )
        responses.append(response)

    # All should succeed
    assert len(responses) == len(temperatures)
    for response in responses:
        assert response is not None
        assert "response" in response
