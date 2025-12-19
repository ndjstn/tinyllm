"""Async Ollama client for TinyLLM.

Provides async interface to Ollama API with connection pooling,
retry logic, and structured output parsing.
"""

import asyncio
from typing import Any, AsyncIterator, Optional

import httpx
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request for Ollama generate endpoint."""

    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[list[int]] = None
    stream: bool = False
    raw: bool = False

    # Generation options
    options: Optional[dict[str, Any]] = None

    # Format
    format: Optional[str] = None  # "json" for JSON mode


class GenerateResponse(BaseModel):
    """Response from Ollama generate endpoint."""

    model: str
    created_at: str
    response: str
    done: bool

    # Stats (only in final response)
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout_ms: int = 30000,
        max_retries: int = 3,
    ):
        """Initialize Ollama client.

        Args:
            host: Ollama server URL.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
        """
        self.host = host.rstrip("/")
        self.timeout = httpx.Timeout(timeout_ms / 1000)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> GenerateResponse:
        """Generate a response from Ollama.

        Args:
            prompt: User prompt.
            model: Model name (e.g., "qwen2.5:0.5b").
            system: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request JSON output.

        Returns:
            GenerateResponse with the generated text.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
        """
        client = await self._get_client()

        request = GenerateRequest(
            model=model,
            prompt=prompt,
            system=system,
            stream=False,
            format="json" if json_mode else None,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(
                    "/api/generate",
                    json=request.model_dump(exclude_none=True),
                )
                response.raise_for_status()
                return GenerateResponse(**response.json())

            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        raise last_error  # type: ignore

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream generate response tokens.

        Args:
            prompt: User prompt.
            model: Model name.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Yields:
            Response text chunks as they arrive.
        """
        client = await self._get_client()

        request = GenerateRequest(
            model=model,
            prompt=prompt,
            system=system,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        async with client.stream(
            "POST",
            "/api/generate",
            json=request.model_dump(exclude_none=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names.
        """
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    async def pull_model(self, model: str) -> None:
        """Pull a model from Ollama registry.

        Args:
            model: Model name to pull.
        """
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/api/pull",
            json={"name": model},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                pass  # Consume stream

    async def check_health(self) -> bool:
        """Check if Ollama is healthy.

        Returns:
            True if Ollama is responding.
        """
        try:
            client = await self._get_client()
            response = await client.get("/")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
