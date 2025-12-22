"""Async vLLM client for TinyLLM.

Provides async interface to vLLM server with OpenAI-compatible API.
vLLM is a high-throughput inference engine optimized for local LLM serving.

API Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
"""

import asyncio
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="vllm_client")
metrics = get_metrics_collector()

# Global connection pool
_client_pool: dict[str, "VLLMClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_vllm_client(
    host: str = "http://localhost:8000",
    timeout_ms: int = 30000,
    max_retries: int = 3,
    rate_limit_rps: float = 10.0,
) -> "VLLMClient":
    """Get or create a shared VLLMClient from the pool."""
    async with _pool_lock:
        if host not in _client_pool:
            logger.info("creating_shared_vllm_client", host=host)
            _client_pool[host] = VLLMClient(
                host=host,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[host]


async def close_all_vllm_clients() -> None:
    """Close all pooled vLLM clients."""
    async with _pool_lock:
        for client in _client_pool.values():
            await client.close()
        _client_pool.clear()
        logger.info("closed_all_vllm_clients", count=len(_client_pool))


# Reuse utilities from llamacpp_client
from tinyllm.providers.llamacpp_client import CircuitBreaker, RateLimiter


# ============================================================================
# Pydantic Models for vLLM API (OpenAI-compatible)
# ============================================================================


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    model_config = {"extra": "forbid"}

    role: str = Field(description="Message role (system, user, assistant)")
    content: str = Field(description="Message content")


class ChatCompletionRequest(BaseModel):
    """Request for vLLM chat completion endpoint."""

    model_config = {"extra": "forbid"}

    model: str = Field(description="Model name to use")
    messages: list[ChatMessage] = Field(description="Chat messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    n: int = Field(default=1, ge=1, description="Number of completions")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to generate")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Enable streaming")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[dict[str, float]] = Field(default=None, description="Logit bias")
    user: Optional[str] = Field(default=None, description="User identifier")


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    model_config = {"extra": "allow"}

    index: int = Field(description="Choice index")
    message: ChatMessage = Field(description="Generated message")
    finish_reason: str = Field(description="Why generation stopped")


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    model_config = {"extra": "allow"}

    prompt_tokens: int = Field(description="Tokens in prompt")
    completion_tokens: int = Field(description="Tokens in completion")
    total_tokens: int = Field(description="Total tokens")


class ChatCompletionResponse(BaseModel):
    """Response from vLLM chat completion endpoint."""

    model_config = {"extra": "allow"}

    id: str = Field(description="Completion ID")
    object: str = Field(description="Object type (chat.completion)")
    created: int = Field(description="Unix timestamp")
    model: str = Field(description="Model used")
    choices: list[ChatCompletionChoice] = Field(description="Completion choices")
    usage: ChatCompletionUsage = Field(description="Token usage")


class VLLMConfig(BaseModel):
    """Configuration for vLLM client."""

    model_config = {"extra": "forbid"}

    host: str = Field(default="http://localhost:8000", description="vLLM server URL")
    timeout_ms: int = Field(default=30000, ge=1000, description="Request timeout in ms")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    rate_limit_rps: float = Field(default=10.0, ge=0.1, description="Rate limit (req/sec)")
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Circuit breaker failure threshold"
    )
    default_model: Optional[str] = Field(default=None, description="Default model name")
    slow_query_threshold_ms: int = Field(
        default=5000, ge=0, description="Slow query threshold in ms"
    )


# ============================================================================
# vLLM Client
# ============================================================================


class VLLMClient:
    """Async client for vLLM server with OpenAI-compatible API.

    Provides structured interface to vLLM HTTP server with:
    - OpenAI-compatible chat completions API
    - Connection pooling
    - Rate limiting
    - Circuit breaker
    - Retry logic
    - Metrics tracking
    """

    def __init__(
        self,
        host: str = "http://localhost:8000",
        timeout_ms: int = 30000,
        max_retries: int = 3,
        rate_limit_rps: float = 10.0,
        circuit_breaker_threshold: int = 5,
        default_model: Optional[str] = None,
        slow_query_threshold_ms: int = 5000,
    ):
        """Initialize vLLM client."""
        self.host = host.rstrip("/")
        self.timeout = httpx.Timeout(timeout_ms / 1000)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(rate=rate_limit_rps, burst=int(rate_limit_rps * 2))
        self._circuit_breaker = CircuitBreaker(failure_threshold=circuit_breaker_threshold)
        self._request_count = 0
        self._total_tokens = 0
        self._current_graph = "default"
        self._current_model = default_model
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self._slow_queries: list[dict[str, Any]] = []

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
            logger.info("vllm_client_closed", host=self.host)

    def set_graph_context(self, graph: str) -> None:
        """Set current graph context for metrics tracking."""
        self._current_graph = graph

    def set_model(self, model: str) -> None:
        """Set the default model."""
        self._current_model = model

    def get_model(self) -> Optional[str]:
        """Get the current default model."""
        return self._current_model

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Generate a completion from vLLM server.

        Args:
            prompt: Input prompt.
            model: Model name to use.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional vLLM parameters.

        Returns:
            ChatCompletionResponse with generated text.
        """
        # Build messages
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))

        # Determine model
        model_name = model or self._current_model
        if not model_name:
            raise ValueError("Model must be specified or default_model must be set")

        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model_name, graph=self._current_graph):
            metrics.increment_request_count(
                model=model_name, graph=self._current_graph, request_type="generate"
            )

            async def _do_generate() -> ChatCompletionResponse:
                client = await self._get_client()

                # Build request
                request_params = {
                    "model": model_name,
                    "messages": [msg.model_dump() for msg in messages],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                    **kwargs,
                }

                request = ChatCompletionRequest(**request_params)

                last_error: Optional[Exception] = None
                query_start_time = time.monotonic()

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/v1/chat/completions",
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        result = ChatCompletionResponse(**response.json())

                        # Track query duration
                        duration_ms = (time.monotonic() - query_start_time) * 1000

                        # Track metrics
                        self._request_count += 1
                        self._total_tokens += result.usage.total_tokens

                        metrics.record_tokens(
                            input_tokens=result.usage.prompt_tokens,
                            output_tokens=result.usage.completion_tokens,
                            model=model_name,
                            graph=self._current_graph,
                        )

                        # Log slow queries
                        if duration_ms > self.slow_query_threshold_ms:
                            slow_query = {
                                "prompt": prompt[:100],
                                "duration_ms": duration_ms,
                                "tokens": result.usage.total_tokens,
                                "timestamp": time.time(),
                            }
                            self._slow_queries.append(slow_query)
                            logger.warning(
                                "slow_vllm_query",
                                duration_ms=duration_ms,
                                threshold_ms=self.slow_query_threshold_ms,
                            )

                        logger.debug(
                            "vllm_generate_success",
                            tokens=result.usage.total_tokens,
                            duration_ms=duration_ms,
                        )

                        return result

                    except httpx.HTTPError as e:
                        last_error = e
                        logger.warning(
                            "vllm_request_failed",
                            attempt=attempt,
                            max_retries=self.max_retries,
                            error=str(e),
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(2**attempt)

                raise RuntimeError(f"Failed after {self.max_retries} retries") from last_error

            return await self._circuit_breaker.call(_do_generate)

    async def check_health(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            logger.warning("vllm_health_check_failed", host=self.host)
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "circuit_breaker_state": self._circuit_breaker.state,
            "circuit_breaker_failures": self._circuit_breaker.failures,
            "slow_queries_count": len(self._slow_queries),
        }

    def get_slow_queries(self) -> list[dict[str, Any]]:
        """Get list of slow queries."""
        return self._slow_queries.copy()

    def clear_slow_queries(self) -> None:
        """Clear slow query history."""
        self._slow_queries.clear()

    async def list_models(self) -> list[str]:
        """List available models on vLLM server."""
        client = await self._get_client()
        response = await client.get("/v1/models")
        response.raise_for_status()
        data = response.json()
        return [model["id"] for model in data.get("data", [])]
