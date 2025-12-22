"""Custom model server client for TinyLLM.

Provides a flexible, configuration-driven HTTP client for connecting to
arbitrary local model servers with custom APIs. Allows users to define
request templates and response extractors for any HTTP-based model server.

This enables TinyLLM to work with custom local model servers without
requiring dedicated client implementations.
"""

import asyncio
import json
import time
from string import Template
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="custom_client")
metrics = get_metrics_collector()

# Global connection pool
_client_pool: dict[str, "CustomModelClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_custom_client(config: "CustomModelConfig") -> "CustomModelClient":
    """Get or create a shared CustomModelClient from the pool."""
    pool_key = f"{config.base_url}:{config.name}"
    async with _pool_lock:
        if pool_key not in _client_pool:
            logger.info("creating_shared_custom_client", name=config.name, base_url=config.base_url)
            _client_pool[pool_key] = CustomModelClient(config=config)
        return _client_pool[pool_key]


async def close_all_custom_clients() -> None:
    """Close all pooled custom clients."""
    async with _pool_lock:
        for client in _client_pool.values():
            await client.close()
        _client_pool.clear()
        logger.info("closed_all_custom_clients", count=len(_client_pool))


# Reuse utilities from llamacpp_client
from tinyllm.providers.llamacpp_client import CircuitBreaker, RateLimiter


# ============================================================================
# Configuration Models
# ============================================================================


class RequestTemplate(BaseModel):
    """Template for building HTTP requests to custom model servers.

    Supports string templates with placeholders:
    - ${prompt} - User prompt
    - ${system} - System prompt (optional)
    - ${temperature} - Sampling temperature
    - ${max_tokens} - Maximum tokens to generate
    - ${model} - Model name

    Example:
        {
            "model": "${model}",
            "prompt": "${prompt}",
            "temperature": ${temperature},
            "max_tokens": ${max_tokens}
        }
    """

    model_config = {"extra": "forbid"}

    method: str = Field(default="POST", description="HTTP method (GET, POST, etc)")
    endpoint: str = Field(default="/v1/completions", description="API endpoint path")
    body_template: str = Field(description="JSON template for request body")
    headers: dict[str, str] = Field(
        default_factory=dict, description="Additional HTTP headers"
    )


class ResponseExtractor(BaseModel):
    """Configuration for extracting data from custom server responses.

    Uses dot notation for nested fields:
    - "choices.0.text" -> response["choices"][0]["text"]
    - "data.content" -> response["data"]["content"]

    Supports JSONPath-like syntax if jsonpath-ng is available.
    """

    model_config = {"extra": "forbid"}

    text_path: str = Field(description="Path to generated text in response")
    input_tokens_path: Optional[str] = Field(
        default=None, description="Path to input token count"
    )
    output_tokens_path: Optional[str] = Field(
        default=None, description="Path to output token count"
    )
    finish_reason_path: Optional[str] = Field(
        default=None, description="Path to finish reason"
    )
    model_path: Optional[str] = Field(default=None, description="Path to model name")


class CustomModelConfig(BaseModel):
    """Configuration for custom model server client."""

    model_config = {"extra": "forbid"}

    name: str = Field(description="Name for this custom model server")
    base_url: str = Field(description="Base URL of the model server")
    request: RequestTemplate = Field(description="Request template configuration")
    response: ResponseExtractor = Field(description="Response extraction configuration")

    # Optional authentication
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    auth_header: str = Field(
        default="Authorization", description="Header name for auth token"
    )
    auth_prefix: str = Field(default="Bearer", description="Prefix for auth token")

    # Client settings
    timeout_ms: int = Field(default=30000, ge=1000, description="Request timeout in ms")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    rate_limit_rps: float = Field(default=10.0, ge=0.1, description="Rate limit (req/sec)")
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Circuit breaker failure threshold"
    )
    slow_query_threshold_ms: int = Field(
        default=5000, ge=0, description="Slow query threshold in ms"
    )

    # Default generation parameters
    default_model: Optional[str] = Field(default=None, description="Default model name")
    default_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default temperature"
    )
    default_max_tokens: int = Field(default=2000, ge=1, description="Default max tokens")


# ============================================================================
# Response Model
# ============================================================================


class CustomModelResponse(BaseModel):
    """Response from custom model server."""

    model_config = {"extra": "allow"}

    text: str = Field(description="Generated text")
    input_tokens: Optional[int] = Field(default=None, description="Input token count")
    output_tokens: Optional[int] = Field(default=None, description="Output token count")
    total_tokens: Optional[int] = Field(default=None, description="Total token count")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")
    model: Optional[str] = Field(default=None, description="Model used")
    raw_response: Optional[dict[str, Any]] = Field(
        default=None, description="Raw server response"
    )


# ============================================================================
# Custom Model Client
# ============================================================================


class CustomModelClient:
    """Flexible client for custom model servers.

    Supports arbitrary HTTP-based model servers through configuration-driven
    request/response handling. Users provide templates for requests and
    extractors for responses.

    Features:
    - Template-based request building
    - Flexible response parsing
    - Authentication support
    - Rate limiting
    - Circuit breaker
    - Metrics tracking
    - Health checks
    """

    def __init__(self, config: CustomModelConfig):
        """Initialize custom model client.

        Args:
            config: Configuration for the custom model server.
        """
        self.config = config
        self.name = config.name
        self.base_url = config.base_url.rstrip("/")
        self.timeout = httpx.Timeout(config.timeout_ms / 1000)
        self.max_retries = config.max_retries

        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(
            rate=config.rate_limit_rps, burst=int(config.rate_limit_rps * 2)
        )
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold
        )

        # Metrics
        self._request_count = 0
        self._total_tokens = 0
        self._current_graph = "default"
        self._current_model = config.default_model
        self.slow_query_threshold_ms = config.slow_query_threshold_ms
        self._slow_queries: list[dict[str, Any]] = []

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = self.config.request.headers.copy()

            # Add authentication if configured
            if self.config.api_key:
                auth_value = f"{self.config.auth_prefix} {self.config.api_key}"
                headers[self.config.auth_header] = auth_value

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("custom_client_closed", name=self.name, base_url=self.base_url)

    def set_graph_context(self, graph: str) -> None:
        """Set current graph context for metrics tracking."""
        self._current_graph = graph

    def set_model(self, model: str) -> None:
        """Set the default model."""
        self._current_model = model

    def get_model(self) -> Optional[str]:
        """Get the current default model."""
        return self._current_model

    def _extract_value(self, response_data: dict[str, Any], path: Optional[str]) -> Any:
        """Extract value from response using dot notation path.

        Examples:
            "choices.0.text" -> response_data["choices"][0]["text"]
            "data.content" -> response_data["data"]["content"]

        Args:
            response_data: Response JSON data.
            path: Dot-notation path to value.

        Returns:
            Extracted value or None if path not found.
        """
        if path is None:
            return None

        try:
            value = response_data
            for part in path.split("."):
                # Check if it's an array index
                if part.isdigit():
                    value = value[int(part)]
                else:
                    value = value[part]
            return value
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(
                "failed_to_extract_value",
                path=path,
                error=str(e),
                response_keys=list(response_data.keys()),
            )
            return None

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CustomModelResponse:
        """Generate a completion from custom model server.

        Args:
            prompt: Input prompt.
            model: Model name to use.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters for the template.

        Returns:
            CustomModelResponse with generated text.
        """
        # Determine parameters
        model_name = model or self._current_model or self.config.default_model or "unknown"
        temp = temperature if temperature is not None else self.config.default_temperature
        max_tok = max_tokens if max_tokens is not None else self.config.default_max_tokens

        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model_name, graph=self._current_graph):
            metrics.increment_request_count(
                model=model_name, graph=self._current_graph, request_type="generate"
            )

            async def _do_generate() -> CustomModelResponse:
                client = await self._get_client()

                # Build request body from template
                template_vars = {
                    "prompt": prompt,
                    "system": system or "",
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "model": model_name,
                    **kwargs,
                }

                # Replace template variables
                body_str = Template(self.config.request.body_template).safe_substitute(
                    template_vars
                )

                # Parse as JSON
                try:
                    request_body = json.loads(body_str)
                except json.JSONDecodeError as e:
                    logger.error("invalid_request_template", error=str(e), body=body_str)
                    raise ValueError(f"Invalid request template: {e}")

                last_error: Optional[Exception] = None
                query_start_time = time.monotonic()

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.request(
                            method=self.config.request.method,
                            url=self.config.request.endpoint,
                            json=request_body,
                        )
                        response.raise_for_status()
                        response_data = response.json()

                        # Extract fields from response
                        text = self._extract_value(response_data, self.config.response.text_path)
                        if text is None:
                            raise ValueError(
                                f"Could not extract text from response using path: "
                                f"{self.config.response.text_path}"
                            )

                        input_tokens = self._extract_value(
                            response_data, self.config.response.input_tokens_path
                        )
                        output_tokens = self._extract_value(
                            response_data, self.config.response.output_tokens_path
                        )
                        finish_reason = self._extract_value(
                            response_data, self.config.response.finish_reason_path
                        )
                        response_model = self._extract_value(
                            response_data, self.config.response.model_path
                        )

                        # Calculate total tokens
                        total_tokens = None
                        if input_tokens is not None and output_tokens is not None:
                            total_tokens = input_tokens + output_tokens

                        result = CustomModelResponse(
                            text=text,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            finish_reason=finish_reason,
                            model=response_model,
                            raw_response=response_data,
                        )

                        # Track query duration
                        duration_ms = (time.monotonic() - query_start_time) * 1000

                        # Track metrics
                        self._request_count += 1
                        if total_tokens:
                            self._total_tokens += total_tokens

                        if input_tokens is not None and output_tokens is not None:
                            metrics.record_tokens(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                model=model_name,
                                graph=self._current_graph,
                            )

                        # Log slow queries
                        if duration_ms > self.slow_query_threshold_ms:
                            slow_query = {
                                "prompt": prompt[:100],
                                "duration_ms": duration_ms,
                                "tokens": total_tokens,
                                "timestamp": time.time(),
                            }
                            self._slow_queries.append(slow_query)
                            logger.warning(
                                "slow_custom_query",
                                name=self.name,
                                duration_ms=duration_ms,
                                threshold_ms=self.slow_query_threshold_ms,
                            )

                        logger.debug(
                            "custom_generate_success",
                            name=self.name,
                            tokens=total_tokens,
                            duration_ms=duration_ms,
                        )

                        return result

                    except (httpx.HTTPError, ValueError, KeyError) as e:
                        last_error = e
                        logger.warning(
                            "custom_request_failed",
                            name=self.name,
                            attempt=attempt,
                            max_retries=self.max_retries,
                            error=str(e),
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(2**attempt)

                raise RuntimeError(
                    f"Failed after {self.max_retries} retries for {self.name}"
                ) from last_error

            return await self._circuit_breaker.call(_do_generate)

    async def check_health(self) -> bool:
        """Check if custom model server is healthy.

        Returns:
            True if server is responding, False otherwise.
        """
        try:
            client = await self._get_client()
            # Try a simple GET to the base URL
            response = await client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            # Fallback: try the generation endpoint with a minimal request
            try:
                response = await client.get("/")
                return response.status_code < 500
            except httpx.HTTPError:
                logger.warning("custom_health_check_failed", name=self.name)
                return False

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics.

        Returns:
            Dictionary with request count, tokens, circuit breaker state, etc.
        """
        return {
            "name": self.name,
            "base_url": self.base_url,
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
