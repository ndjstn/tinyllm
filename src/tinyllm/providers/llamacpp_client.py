"""Async llama.cpp client for TinyLLM.

Provides async interface to llama.cpp server with connection pooling,
retry logic, rate limiting, and structured output parsing.

Llama.cpp is a high-performance C++ implementation of LLaMA models
that can run locally without dependencies on Python ML frameworks.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="llamacpp_client")
metrics = get_metrics_collector()

# Global connection pool
_client_pool: dict[str, "LlamaCppClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_llamacpp_client(
    host: str = "http://localhost:8080",
    timeout_ms: int = 30000,
    max_retries: int = 3,
    rate_limit_rps: float = 10.0,
) -> "LlamaCppClient":
    """Get or create a shared LlamaCppClient from the pool.

    Args:
        host: Llama.cpp server URL.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum retry attempts.
        rate_limit_rps: Rate limit in requests per second.

    Returns:
        Shared LlamaCppClient instance.
    """
    async with _pool_lock:
        if host not in _client_pool:
            logger.info("creating_shared_llamacpp_client", host=host)
            _client_pool[host] = LlamaCppClient(
                host=host,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[host]


async def close_all_llamacpp_clients() -> None:
    """Close all pooled llama.cpp clients."""
    async with _pool_lock:
        for client in _client_pool.values():
            await client.close()
        _client_pool.clear()
        logger.info("closed_all_llamacpp_clients", count=len(_client_pool))


# ============================================================================
# Rate Limiter (Token Bucket Algorithm)
# ============================================================================


class RateLimiter:
    """Token bucket rate limiter for request throttling."""

    def __init__(self, rate: float, burst: int):
        """Initialize rate limiter.

        Args:
            rate: Requests per second.
            burst: Maximum burst size.
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            timeout: Seconds to wait before attempting to close circuit.
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit should transition from open to half_open
            if self.state == "open":
                if (
                    self.last_failure_time
                    and time.monotonic() - self.last_failure_time > self.timeout
                ):
                    self.state = "half_open"
                    logger.info("circuit_breaker_half_open")
                else:
                    raise Exception(f"Circuit breaker is open (failures: {self.failures})")

        # Attempt the call
        try:
            result = await func(*args, **kwargs)

            # Success - reset or close circuit
            async with self._lock:
                if self.state == "half_open":
                    self.state = "closed"
                    self.failures = 0
                    logger.info("circuit_breaker_closed")

            return result

        except Exception as e:
            # Failure - increment counter and potentially open circuit
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.monotonic()

                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(
                        "circuit_breaker_opened",
                        failures=self.failures,
                        threshold=self.failure_threshold,
                    )

            raise e


# ============================================================================
# Pydantic Models for llama.cpp API
# ============================================================================


class CompletionRequest(BaseModel):
    """Request for llama.cpp completion endpoint."""

    model_config = {"extra": "forbid"}

    prompt: str = Field(description="Input prompt")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=40, ge=0, description="Top-k sampling")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    n_predict: int = Field(default=128, ge=-1, description="Max tokens to generate (-1 = unlimited)")
    stop: list[str] = Field(default_factory=list, description="Stop sequences")
    stream: bool = Field(default=False, description="Enable streaming")
    n_keep: int = Field(default=0, ge=0, description="Tokens to keep from prompt")
    tfs_z: float = Field(default=1.0, ge=0.0, description="Tail free sampling")
    typical_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Typical sampling")
    repeat_penalty: float = Field(default=1.1, ge=0.0, description="Repeat penalty")
    repeat_last_n: int = Field(default=64, ge=0, description="Last n tokens to penalize")
    penalize_nl: bool = Field(default=True, description="Penalize newlines")
    mirostat: int = Field(default=0, ge=0, le=2, description="Mirostat sampling mode")
    mirostat_tau: float = Field(default=5.0, ge=0.0, description="Mirostat target entropy")
    mirostat_eta: float = Field(default=0.1, ge=0.0, description="Mirostat learning rate")
    grammar: str = Field(default="", description="GBNF grammar")
    seed: int = Field(default=-1, description="RNG seed (-1 = random)")
    ignore_eos: bool = Field(default=False, description="Ignore end of sequence token")
    logit_bias: list[list[int | float]] = Field(default_factory=list, description="Logit bias")
    n_probs: int = Field(default=0, ge=0, description="Return top n_probs probabilities")


class CompletionResponse(BaseModel):
    """Response from llama.cpp completion endpoint."""

    model_config = {"extra": "allow"}

    content: str = Field(description="Generated text")
    stop: bool = Field(description="Whether generation stopped")
    generation_settings: Optional[dict[str, Any]] = Field(
        default=None, description="Generation settings used"
    )
    model: Optional[str] = Field(default=None, description="Model name")
    prompt: Optional[str] = Field(default=None, description="Input prompt")
    stopped_eos: bool = Field(default=False, description="Stopped at EOS token")
    stopped_limit: bool = Field(default=False, description="Stopped at token limit")
    stopped_word: bool = Field(default=False, description="Stopped at stop word")
    stopping_word: str = Field(default="", description="Stop word that triggered stop")
    timings: Optional[dict[str, Any]] = Field(default=None, description="Timing information")
    tokens_cached: int = Field(default=0, description="Tokens from cache")
    tokens_evaluated: int = Field(default=0, description="Tokens evaluated")
    tokens_predicted: int = Field(default=0, description="Tokens predicted")
    truncated: bool = Field(default=False, description="Whether prompt was truncated")


class LlamaCppConfig(BaseModel):
    """Configuration for LlamaCpp client."""

    model_config = {"extra": "forbid"}

    host: str = Field(default="http://localhost:8080", description="Llama.cpp server URL")
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
# Llama.cpp Client
# ============================================================================


class LlamaCppClient:
    """Async client for llama.cpp server.

    Provides structured interface to llama.cpp HTTP server with:
    - Connection pooling
    - Rate limiting
    - Circuit breaker
    - Retry logic
    - Metrics tracking
    - Slow query detection
    """

    def __init__(
        self,
        host: str = "http://localhost:8080",
        timeout_ms: int = 30000,
        max_retries: int = 3,
        rate_limit_rps: float = 10.0,
        circuit_breaker_threshold: int = 5,
        default_model: Optional[str] = None,
        slow_query_threshold_ms: int = 5000,
    ):
        """Initialize llama.cpp client.

        Args:
            host: Llama.cpp server URL.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
            rate_limit_rps: Rate limit in requests per second.
            circuit_breaker_threshold: Failures before opening circuit.
            default_model: Default model name.
            slow_query_threshold_ms: Threshold for slow query logging.
        """
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
            logger.info("llamacpp_client_closed", host=self.host)

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
        json_mode: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion from llama.cpp server.

        Args:
            prompt: Input prompt.
            model: Model name (unused - llama.cpp uses loaded model).
            system: System prompt to prepend.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Enable JSON mode via grammar (if supported).
            **kwargs: Additional llama.cpp parameters.

        Returns:
            CompletionResponse with generated text.
        """
        # Combine system prompt if provided
        if system:
            prompt = f"{system}\n\n{prompt}"

        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        model_name = model or self._current_model or "llamacpp"
        with metrics.track_request_latency(model=model_name, graph=self._current_graph):
            metrics.increment_request_count(
                model=model_name, graph=self._current_graph, request_type="generate"
            )

            async def _do_generate() -> CompletionResponse:
                client = await self._get_client()

                # Build request
                request_params = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "n_predict": max_tokens,
                    "stream": False,
                    **kwargs,
                }

                # Add JSON grammar if requested (requires llama.cpp with JSON support)
                if json_mode:
                    request_params["grammar"] = "json"  # Simplified - may need proper GBNF

                request = CompletionRequest(**request_params)

                last_error: Optional[Exception] = None
                query_start_time = time.monotonic()

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/completion",
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        result = CompletionResponse(**response.json())

                        # Track query duration
                        duration_ms = (time.monotonic() - query_start_time) * 1000

                        # Track metrics
                        self._request_count += 1
                        if result.tokens_predicted:
                            self._total_tokens += result.tokens_predicted

                        metrics.record_tokens(
                            input_tokens=result.tokens_evaluated or 0,
                            output_tokens=result.tokens_predicted or 0,
                            model=model_name,
                            graph=self._current_graph,
                        )

                        # Log slow queries
                        if duration_ms > self.slow_query_threshold_ms:
                            slow_query = {
                                "prompt": prompt[:100],
                                "duration_ms": duration_ms,
                                "tokens": result.tokens_predicted,
                                "timestamp": time.time(),
                            }
                            self._slow_queries.append(slow_query)
                            logger.warning(
                                "slow_llamacpp_query",
                                duration_ms=duration_ms,
                                threshold_ms=self.slow_query_threshold_ms,
                                tokens=result.tokens_predicted,
                            )

                        logger.debug(
                            "llamacpp_generate_success",
                            tokens=result.tokens_predicted,
                            duration_ms=duration_ms,
                        )

                        return result

                    except httpx.HTTPError as e:
                        last_error = e
                        logger.warning(
                            "llamacpp_request_failed",
                            attempt=attempt,
                            max_retries=self.max_retries,
                            error=str(e),
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(2**attempt)  # Exponential backoff

                raise RuntimeError(f"Failed after {self.max_retries} retries") from last_error

            return await self._circuit_breaker.call(_do_generate)

    async def check_health(self) -> bool:
        """Check if llama.cpp server is healthy.

        Returns:
            True if server is responding, False otherwise.
        """
        try:
            client = await self._get_client()
            # Try health endpoint first, fall back to props
            try:
                response = await client.get("/health")
                return response.status_code == 200
            except httpx.HTTPError:
                # Fallback to /props endpoint which exists in llama.cpp
                response = await client.get("/props")
                return response.status_code == 200
        except httpx.HTTPError:
            logger.warning("llamacpp_health_check_failed", host=self.host)
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics.

        Returns:
            Dictionary with request count, tokens, circuit breaker state, etc.
        """
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

    async def get_props(self) -> dict[str, Any]:
        """Get server properties and loaded model info.

        Returns:
            Server properties including model info, context size, etc.
        """
        client = await self._get_client()
        response = await client.get("/props")
        response.raise_for_status()
        return response.json()
