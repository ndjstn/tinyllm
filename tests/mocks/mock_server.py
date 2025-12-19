"""Mock Ollama server for testing.

This module provides a mock HTTP server that simulates the Ollama API,
allowing tests to run without requiring a real Ollama installation.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from aiohttp import web

from tinyllm.models.client import GenerateResponse


class MockOllamaServer:
    """Mock Ollama server for testing.

    Provides a lightweight HTTP server that implements the Ollama API
    endpoints needed for testing, with configurable responses and error simulation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11435,  # Different from real Ollama (11434)
        default_model: str = "qwen2.5:3b",
        latency_ms: int = 100,
    ):
        """Initialize mock server.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            default_model: Default model name.
            latency_ms: Simulated latency in milliseconds.
        """
        self.host = host
        self.port = port
        self.default_model = default_model
        self.latency_ms = latency_ms

        # Server state
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Request tracking
        self.requests: List[Dict[str, Any]] = []
        self.request_count = 0

        # Response configuration
        self.responses: Dict[str, str] = {}
        self.error_responses: Dict[str, tuple[int, str]] = {}
        self.models: List[str] = [
            "qwen2.5:0.5b",
            "qwen2.5:3b",
            "granite-code:3b",
            "phi3:mini",
        ]

        # Behavior flags
        self.should_fail = False
        self.fail_count = 0
        self.max_fails = 0
        self.stream_enabled = False

    async def start(self) -> None:
        """Start the mock server."""
        self.app = web.Application()
        self.app.router.add_post("/api/generate", self.handle_generate)
        self.app.router.add_get("/api/tags", self.handle_list_models)
        self.app.router.add_post("/api/pull", self.handle_pull)
        self.app.router.add_get("/", self.handle_health)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

    async def stop(self) -> None:
        """Stop the mock server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        # Reset state
        self.requests.clear()
        self.request_count = 0
        self.should_fail = False
        self.fail_count = 0

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"

    async def handle_generate(self, request: web.Request) -> web.Response:
        """Handle /api/generate endpoint."""
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Parse request
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")

        # Track request
        self.request_count += 1
        self.requests.append({
            "endpoint": "/api/generate",
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Check for simulated failure
        if self.should_fail:
            if self.max_fails == 0 or self.fail_count < self.max_fails:
                self.fail_count += 1
                return web.Response(status=503, text="Service temporarily unavailable")

        # Get parameters
        model = data.get("model", self.default_model)
        prompt = data.get("prompt", "")
        system = data.get("system")
        stream = data.get("stream", False)
        options = data.get("options", {})

        # Check for custom response
        response_key = f"{model}:{prompt[:50]}"
        if response_key in self.responses:
            response_text = self.responses[response_key]
        elif prompt in self.responses:
            response_text = self.responses[prompt]
        else:
            # Generate default response based on model
            response_text = self._generate_default_response(model, prompt, system)

        # Check for error response
        if prompt in self.error_responses:
            status, text = self.error_responses[prompt]
            return web.Response(status=status, text=text)

        # Build response
        response_obj = GenerateResponse(
            model=model,
            created_at=datetime.utcnow().isoformat(),
            response=response_text,
            done=True,
            total_duration=self.latency_ms * 1_000_000,  # nanoseconds
            load_duration=10 * 1_000_000,
            prompt_eval_count=len(prompt.split()),
            prompt_eval_duration=50 * 1_000_000,
            eval_count=len(response_text.split()),
            eval_duration=(self.latency_ms - 60) * 1_000_000,
        )

        if stream:
            # Return streaming response
            return await self._stream_response(response_obj)
        else:
            # Return complete response
            return web.json_response(response_obj.model_dump())

    async def _stream_response(self, response: GenerateResponse) -> web.StreamResponse:
        """Stream a response in chunks."""
        stream_response = web.StreamResponse()
        stream_response.content_type = "application/x-ndjson"
        await stream_response.prepare()

        # Split response into words and stream them
        words = response.response.split()
        for i, word in enumerate(words):
            chunk = {
                "model": response.model,
                "created_at": response.created_at,
                "response": word + (" " if i < len(words) - 1 else ""),
                "done": False,
            }
            await stream_response.write((json.dumps(chunk) + "\n").encode())
            await asyncio.sleep(0.01)  # Small delay between chunks

        # Send final chunk with stats
        final_chunk = response.model_dump()
        final_chunk["response"] = ""
        final_chunk["done"] = True
        await stream_response.write((json.dumps(final_chunk) + "\n").encode())

        await stream_response.write_eof()
        return stream_response

    async def handle_list_models(self, request: web.Request) -> web.Response:
        """Handle /api/tags endpoint."""
        self.request_count += 1
        self.requests.append({
            "endpoint": "/api/tags",
            "timestamp": datetime.utcnow().isoformat(),
        })

        models_data = {
            "models": [
                {
                    "name": model,
                    "modified_at": datetime.utcnow().isoformat(),
                    "size": 1_000_000_000,  # 1GB
                    "digest": f"sha256:{'0' * 64}",
                }
                for model in self.models
            ]
        }

        return web.json_response(models_data)

    async def handle_pull(self, request: web.Request) -> web.Response:
        """Handle /api/pull endpoint."""
        data = await request.json()
        model = data.get("name")

        self.request_count += 1
        self.requests.append({
            "endpoint": "/api/pull",
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Add model to available models if not present
        if model and model not in self.models:
            self.models.append(model)

        # Return success response
        response_data = {
            "status": "success",
            "digest": f"sha256:{'0' * 64}",
            "total": 1_000_000_000,
        }

        return web.json_response(response_data)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle / (health check) endpoint."""
        return web.Response(text="Ollama is running")

    def _generate_default_response(
        self, model: str, prompt: str, system: Optional[str]
    ) -> str:
        """Generate a default response based on model and prompt."""
        # Simple pattern-based responses
        prompt_lower = prompt.lower()

        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I help you today?"

        if "math" in prompt_lower or any(op in prompt for op in ["+", "-", "*", "/"]):
            return "The result is 42."

        if "code" in prompt_lower or "function" in prompt_lower:
            return "```python\ndef example():\n    return 'Hello, World!'\n```"

        if "json" in prompt_lower:
            return '{"status": "success", "message": "This is a JSON response"}'

        # Default response
        return f"This is a mock response from {model} to your prompt: {prompt[:50]}..."

    def set_response(self, prompt: str, response: str) -> None:
        """Set a custom response for a specific prompt.

        Args:
            prompt: The prompt to match.
            response: The response to return.
        """
        self.responses[prompt] = response

    def set_error_response(self, prompt: str, status: int, message: str) -> None:
        """Set an error response for a specific prompt.

        Args:
            prompt: The prompt to match.
            status: HTTP status code.
            message: Error message.
        """
        self.error_responses[prompt] = (status, message)

    def enable_failures(self, max_fails: int = 0) -> None:
        """Enable simulated failures.

        Args:
            max_fails: Maximum number of failures before succeeding (0 = unlimited).
        """
        self.should_fail = True
        self.max_fails = max_fails
        self.fail_count = 0

    def disable_failures(self) -> None:
        """Disable simulated failures."""
        self.should_fail = False
        self.fail_count = 0
        self.max_fails = 0

    def add_model(self, model_name: str) -> None:
        """Add a model to the available models list.

        Args:
            model_name: Name of the model to add.
        """
        if model_name not in self.models:
            self.models.append(model_name)

    def remove_model(self, model_name: str) -> None:
        """Remove a model from the available models list.

        Args:
            model_name: Name of the model to remove.
        """
        if model_name in self.models:
            self.models.remove(model_name)

    def clear_requests(self) -> None:
        """Clear the request history."""
        self.requests.clear()
        self.request_count = 0

    def get_request_count(self, endpoint: Optional[str] = None) -> int:
        """Get the number of requests made.

        Args:
            endpoint: Optional endpoint to filter by.

        Returns:
            Number of requests.
        """
        if endpoint is None:
            return self.request_count
        return sum(1 for r in self.requests if r.get("endpoint") == endpoint)

    def get_last_request(self, endpoint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the last request made.

        Args:
            endpoint: Optional endpoint to filter by.

        Returns:
            Last request data or None.
        """
        if endpoint is None:
            return self.requests[-1] if self.requests else None

        for request in reversed(self.requests):
            if request.get("endpoint") == endpoint:
                return request
        return None


async def create_mock_server(**kwargs) -> MockOllamaServer:
    """Create and start a mock Ollama server.

    Args:
        **kwargs: Arguments to pass to MockOllamaServer constructor.

    Returns:
        Started MockOllamaServer instance.
    """
    server = MockOllamaServer(**kwargs)
    await server.start()
    return server
