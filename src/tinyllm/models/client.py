"""Async Ollama client for TinyLLM.

Provides async interface to Ollama API with connection pooling,
retry logic, rate limiting, and structured output parsing.
"""

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Optional
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector
from tinyllm.telemetry import set_span_attribute, trace_llm_request

logger = get_logger(__name__, component="ollama_client")
metrics = get_metrics_collector()


# Global connection pool for Ollama client reuse
_client_pool: dict[str, "OllamaClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_client(
    host: str = "http://localhost:11434",
    timeout_ms: int = 30000,
    max_retries: int = 3,
    rate_limit_rps: float = 10.0,
) -> "OllamaClient":
    """Get or create a shared OllamaClient for the given host.

    This enables connection pooling across the application by reusing
    clients instead of creating new ones for each request.

    Args:
        host: Ollama server URL.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum retry attempts.
        rate_limit_rps: Rate limit in requests per second.

    Returns:
        Shared OllamaClient instance.
    """
    async with _pool_lock:
        if host not in _client_pool:
            logger.info(
                "creating_shared_client",
                host=host,
                timeout_ms=timeout_ms,
                rate_limit_rps=rate_limit_rps,
            )
            _client_pool[host] = OllamaClient(
                host=host,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[host]


async def close_all_clients() -> None:
    """Close all pooled clients. Call during application shutdown."""
    logger.info("closing_all_clients", client_count=len(_client_pool))
    async with _pool_lock:
        for client in _client_pool.values():
            await client.close()
        _client_pool.clear()


class RateLimiter:
    """Token bucket rate limiter for controlling request rate."""

    def __init__(self, rate: float = 10.0, burst: int = 20):
        """Initialize rate limiter.

        Args:
            rate: Requests per second.
            burst: Maximum burst size.
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class CircuitBreaker:
    """Circuit breaker for handling service failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before trying again.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if (
                    self.last_failure_time
                    and time.monotonic() - self.last_failure_time > self.recovery_timeout
                ):
                    logger.info("circuit_breaker_half_open")
                    self.state = "half-open"
                else:
                    logger.warning("circuit_breaker_open", failures=self.failures)
                    raise RuntimeError("Circuit breaker is open - service unavailable")

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "half-open":
                    self.state = "closed"
                    self.failures = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.monotonic()
                if self.failures >= self.failure_threshold:
                    logger.error(
                        "circuit_breaker_opened",
                        failures=self.failures,
                        threshold=self.failure_threshold,
                    )
                    self.state = "open"
                    metrics.increment_circuit_breaker_failures(model="ollama")
            raise

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


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
    """Async client for Ollama API with connection pooling, rate limiting, and circuit breaker."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout_ms: int = 30000,
        max_retries: int = 3,
        rate_limit_rps: float = 10.0,
        circuit_breaker_threshold: int = 5,
        default_model: Optional[str] = None,
        slow_query_threshold_ms: int = 5000,
    ):
        """Initialize Ollama client.

        Args:
            host: Ollama server URL.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
            rate_limit_rps: Rate limit in requests per second.
            circuit_breaker_threshold: Failures before circuit opens.
            default_model: Default model to use for generate calls.
            slow_query_threshold_ms: Threshold in ms for slow query detection.
        """
        self.host = host.rstrip("/")
        self.timeout = httpx.Timeout(timeout_ms / 1000)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(rate=rate_limit_rps, burst=int(rate_limit_rps * 2))
        self._circuit_breaker = CircuitBreaker(failure_threshold=circuit_breaker_threshold)
        self._request_count = 0
        self._total_tokens = 0
        self._current_graph = "default"  # Can be set by executor
        self._current_model = default_model  # Current/default model
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

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> GenerateResponse:
        """Generate a response from Ollama.

        Args:
            prompt: User prompt.
            model: Model name (e.g., "qwen2.5:0.5b"). If None, uses default model.
            system: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request JSON output.

        Returns:
            GenerateResponse with the generated text.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
            RuntimeError: If circuit breaker is open.
            ValueError: If no model specified and no default model set.
        """
        # Use default model if none specified
        if model is None:
            if self._current_model is None:
                raise ValueError("No model specified and no default model set")
            model = self._current_model

        # Update circuit breaker state metric
        metrics.update_circuit_breaker_state(
            self._circuit_breaker.get_state(), model=model
        )

        # Apply rate limiting and track wait time
        rate_limit_start = time.monotonic()
        await self._rate_limiter.acquire()
        rate_limit_wait = time.monotonic() - rate_limit_start
        if rate_limit_wait > 0.001:  # Only record if significant
            metrics.record_rate_limit_wait(rate_limit_wait, model=model)

        # Track request with metrics and tracing
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            with trace_llm_request(
                model=model,
                prompt_length=len(prompt),
                temperature=temperature,
            ):
                metrics.increment_request_count(
                    model=model, graph=self._current_graph, request_type="generate"
                )

                async def _do_generate() -> GenerateResponse:
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
                    query_start_time = time.monotonic()

                    for attempt in range(self.max_retries + 1):
                        try:
                            response = await client.post(
                                "/api/generate",
                                json=request.model_dump(exclude_none=True),
                            )
                            response.raise_for_status()
                            result = GenerateResponse(**response.json())

                            # Calculate total query time
                            query_duration_ms = (time.monotonic() - query_start_time) * 1000

                            # Detect slow queries
                            if query_duration_ms > self.slow_query_threshold_ms:
                                slow_query_info = {
                                    "timestamp": time.time(),
                                    "model": model,
                                    "duration_ms": query_duration_ms,
                                    "prompt_length": len(prompt),
                                    "input_tokens": result.prompt_eval_count or 0,
                                    "output_tokens": result.eval_count or 0,
                                    "graph": self._current_graph,
                                }
                                self._slow_queries.append(slow_query_info)

                                logger.warning(
                                    "slow_query_detected",
                                    model=model,
                                    duration_ms=query_duration_ms,
                                    threshold_ms=self.slow_query_threshold_ms,
                                    prompt_length=len(prompt),
                                    input_tokens=result.prompt_eval_count or 0,
                                    output_tokens=result.eval_count or 0,
                                )

                            # Track metrics
                            self._request_count += 1
                            if result.eval_count:
                                self._total_tokens += result.eval_count

                            # Record token usage
                            input_tokens = result.prompt_eval_count or 0
                            output_tokens = result.eval_count or 0
                            metrics.record_tokens(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                model=model,
                                graph=self._current_graph,
                            )

                            # Add token counts to trace span
                            set_span_attribute("llm.input_tokens", input_tokens)
                            set_span_attribute("llm.output_tokens", output_tokens)
                            set_span_attribute("llm.total_tokens", input_tokens + output_tokens)

                            return result

                        except httpx.HTTPError as e:
                            last_error = e
                            error_type = type(e).__name__
                            metrics.increment_error_count(
                                error_type=error_type,
                                model=model,
                                graph=self._current_graph,
                            )
                            if attempt < self.max_retries:
                                # Exponential backoff with jitter
                                import random
                                base_delay = 2 ** attempt
                                jitter = random.uniform(0, base_delay * 0.5)
                                await asyncio.sleep(base_delay + jitter)

                    raise last_error  # type: ignore

            # Execute with circuit breaker protection
            try:
                return await self._circuit_breaker.call(_do_generate)
            except RuntimeError as e:
                if "Circuit breaker is open" in str(e):
                    metrics.increment_error_count(
                        error_type="circuit_breaker_open",
                        model=model,
                        graph=self._current_graph,
                    )
                raise

    def set_graph_context(self, graph: str) -> None:
        """Set the current graph context for metrics tracking.

        Args:
            graph: Graph name to use in metrics labels.
        """
        self._current_graph = graph

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics.

        Returns:
            Dict with request_count, total_tokens, circuit_breaker_state.
        """
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "circuit_breaker_state": self._circuit_breaker.state,
            "circuit_breaker_failures": self._circuit_breaker.failures,
            "slow_query_count": len(self._slow_queries),
            "slow_query_threshold_ms": self.slow_query_threshold_ms,
        }

    def get_slow_queries(
        self,
        limit: Optional[int] = None,
        min_duration_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get slow query information.

        Args:
            limit: Maximum number of queries to return (most recent first).
            min_duration_ms: Filter queries slower than this threshold.

        Returns:
            List of slow query info dictionaries.
        """
        queries = self._slow_queries

        # Filter by duration if specified
        if min_duration_ms is not None:
            queries = [q for q in queries if q["duration_ms"] >= min_duration_ms]

        # Sort by timestamp (most recent first)
        queries = sorted(queries, key=lambda q: q["timestamp"], reverse=True)

        # Apply limit
        if limit is not None:
            queries = queries[:limit]

        return queries

    def clear_slow_queries(self) -> None:
        """Clear slow query history."""
        self._slow_queries.clear()
        logger.info("slow_queries_cleared")

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream generate response tokens.

        Args:
            prompt: User prompt.
            model: Model name. If None, uses default model.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Yields:
            Response text chunks as they arrive.

        Raises:
            ValueError: If no model specified and no default model set.
        """
        # Use default model if none specified
        if model is None:
            if self._current_model is None:
                raise ValueError("No model specified and no default model set")
            model = self._current_model

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

    def set_model(self, model: str) -> None:
        """Set the default model for this client.

        This allows switching models mid-conversation without creating a new client.
        The conversation context is preserved across model switches.

        Args:
            model: Model name to use (e.g., "qwen2.5:3b").
        """
        logger.info("model_switched", from_model=self._current_model, to_model=model)
        self._current_model = model

    def get_model(self) -> Optional[str]:
        """Get the current default model.

        Returns:
            Current model name if set, None otherwise.
        """
        return self._current_model

    async def check_model_compatibility(
        self, model: str, required_features: Optional[list[str]] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if a model is available and compatible.

        Args:
            model: Model name to check.
            required_features: Optional list of required features (e.g., ["vision"]).

        Returns:
            Tuple of (is_compatible, warning_message).
        """
        # Check if model exists
        available_models = await self.list_models()
        if model not in available_models:
            return False, f"Model '{model}' is not available. Run: ollama pull {model}"

        # For now, we don't have feature detection, so just return True
        # This can be enhanced later with model info API
        return True, None

    async def switch_model(
        self, model: str, check_compatibility: bool = True
    ) -> tuple[bool, Optional[str]]:
        """Switch to a different model with optional compatibility check.

        Args:
            model: Model name to switch to.
            check_compatibility: Whether to check model availability first.

        Returns:
            Tuple of (success, warning_message).
        """
        if check_compatibility:
            compatible, warning = await self.check_model_compatibility(model)
            if not compatible:
                return False, warning

        self.set_model(model)
        return True, None
