"""Async OpenAI client for TinyLLM.

Provides async interface to OpenAI API with connection pooling,
retry logic, rate limiting, and structured output parsing.
"""

import asyncio
import base64
import os
import time
from enum import Enum
from typing import Any, AsyncIterator, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="openai_client")
metrics = get_metrics_collector()


# Global connection pool for OpenAI client reuse
_client_pool: dict[str, "OpenAIClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_openai_client(
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    timeout_ms: int = 60000,
    max_retries: int = 3,
    rate_limit_rps: float = 10.0,
) -> "OpenAIClient":
    """Get or create a shared OpenAIClient for the given configuration.

    This enables connection pooling across the application by reusing
    clients instead of creating new ones for each request.

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        base_url: API base URL.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum retry attempts.
        rate_limit_rps: Rate limit in requests per second.

    Returns:
        Shared OpenAIClient instance.

    Raises:
        ValueError: If no API key provided and OPENAI_API_KEY env var not set.
    """
    async with _pool_lock:
        cache_key = f"{base_url}:{api_key or 'env'}"
        if cache_key not in _client_pool:
            logger.info(
                "creating_shared_openai_client",
                base_url=base_url,
                timeout_ms=timeout_ms,
                rate_limit_rps=rate_limit_rps,
            )
            _client_pool[cache_key] = OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[cache_key]


async def close_all_openai_clients() -> None:
    """Close all pooled clients. Call during application shutdown."""
    logger.info("closing_all_openai_clients", client_count=len(_client_pool))
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

    async def call(self, func, *args, **kwargs):
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
                    metrics.increment_circuit_breaker_failures(model="openai")
            raise

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


# Pydantic Models for API


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class ImageUrlDetail(str, Enum):
    """Detail level for image processing."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class ImageUrl(BaseModel):
    """Image URL content."""

    url: str
    detail: ImageUrlDetail = ImageUrlDetail.AUTO


class ImageContent(BaseModel):
    """Image content in a message."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class TextContent(BaseModel):
    """Text content in a message."""

    type: Literal["text"] = "text"
    text: str


class ChatMessage(BaseModel):
    """A message in the chat conversation."""

    role: MessageRole
    content: Optional[Union[str, list[Union[TextContent, ImageContent]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None
    tool_call_id: Optional[str] = None


class FunctionCall(BaseModel):
    """Function call information."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call information."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoice(BaseModel):
    """Tool choice configuration."""

    type: Literal["function"] = "function"
    function: dict[str, str]


class ChatCompletionRequest(BaseModel):
    """Request for OpenAI chat completions endpoint."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None

    # Function/tool calling
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None

    # Response format
    response_format: Optional[dict[str, str]] = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """A choice in the chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response from OpenAI chat completions endpoint."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk from chat completions."""

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    system_fingerprint: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """Request for OpenAI embeddings endpoint."""

    model: str
    input: Union[str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    """Embedding data."""

    object: str
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response from OpenAI embeddings endpoint."""

    object: str
    data: list[EmbeddingData]
    model: str
    usage: Usage


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI client."""

    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout_ms: int = 60000
    max_retries: int = 3
    rate_limit_rps: float = 10.0
    default_model: str = "gpt-4o-mini"
    circuit_breaker_threshold: int = 5


class OpenAIClient:
    """Async client for OpenAI API with connection pooling, rate limiting, and circuit breaker."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_ms: int = 60000,
        max_retries: int = 3,
        rate_limit_rps: float = 10.0,
        circuit_breaker_threshold: int = 5,
        default_model: str = "gpt-4o-mini",
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: API base URL.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
            rate_limit_rps: Rate limit in requests per second.
            circuit_breaker_threshold: Failures before circuit opens.
            default_model: Default model to use for completions.

        Raises:
            ValueError: If no API key provided and OPENAI_API_KEY env var not set.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY environment variable"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_ms / 1000)
        self.max_retries = max_retries
        self.default_model = default_model
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(rate=rate_limit_rps, burst=int(rate_limit_rps * 2))
        self._circuit_breaker = CircuitBreaker(failure_threshold=circuit_breaker_threshold)
        self._request_count = 0
        self._total_tokens = 0
        self._current_graph = "default"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _prepare_headers(self) -> dict[str, str]:
        """Prepare headers for API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[ToolDefinition]] = None,
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        response_format: Optional[dict[str, str]] = None,
        stream: bool = False,
    ) -> ChatCompletionResponse:
        """Create a chat completion.

        Args:
            messages: List of messages in the conversation.
            model: Model to use. If None, uses default_model.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            tools: List of tools available to the model.
            tool_choice: Controls which (if any) tool is called.
            response_format: Format of the response (e.g., {"type": "json_object"}).
            stream: Whether to stream the response.

        Returns:
            ChatCompletionResponse with the model's response.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
            RuntimeError: If circuit breaker is open.
        """
        model = model or self.default_model

        # Update circuit breaker state metric
        metrics.update_circuit_breaker_state(
            self._circuit_breaker.get_state(), model=model
        )

        # Apply rate limiting
        rate_limit_start = time.monotonic()
        await self._rate_limiter.acquire()
        rate_limit_wait = time.monotonic() - rate_limit_start
        if rate_limit_wait > 0.001:
            metrics.record_rate_limit_wait(rate_limit_wait, model=model)

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="chat_completion"
            )

            async def _do_completion() -> ChatCompletionResponse:
                client = await self._get_client()

                request = ChatCompletionRequest(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    stream=stream,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/chat/completions",
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        result = ChatCompletionResponse(**response.json())

                        # Track metrics
                        self._request_count += 1
                        if result.usage:
                            self._total_tokens += result.usage.total_tokens

                            # Record token usage
                            metrics.record_tokens(
                                input_tokens=result.usage.prompt_tokens,
                                output_tokens=result.usage.completion_tokens,
                                model=model,
                                graph=self._current_graph,
                            )

                        return result

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code

                        # Don't retry on client errors (4xx) except rate limits
                        if 400 <= status_code < 500 and status_code != 429:
                            error_type = f"http_{status_code}"
                            metrics.increment_error_count(
                                error_type=error_type,
                                model=model,
                                graph=self._current_graph,
                            )
                            raise

                        # Log and retry
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
                            delay = base_delay + jitter

                            # Respect Retry-After header if present
                            retry_after = e.response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    delay = max(delay, float(retry_after))
                                except ValueError:
                                    pass

                            logger.warning(
                                "openai_request_retry",
                                attempt=attempt + 1,
                                max_retries=self.max_retries,
                                delay=delay,
                                status_code=status_code,
                            )
                            await asyncio.sleep(delay)

                    except httpx.HTTPError as e:
                        last_error = e
                        error_type = type(e).__name__
                        metrics.increment_error_count(
                            error_type=error_type,
                            model=model,
                            graph=self._current_graph,
                        )
                        if attempt < self.max_retries:
                            import random
                            base_delay = 2 ** attempt
                            jitter = random.uniform(0, base_delay * 0.5)
                            await asyncio.sleep(base_delay + jitter)

                raise last_error  # type: ignore

            # Execute with circuit breaker protection
            try:
                return await self._circuit_breaker.call(_do_completion)
            except RuntimeError as e:
                if "Circuit breaker is open" in str(e):
                    metrics.increment_error_count(
                        error_type="circuit_breaker_open",
                        model=model,
                        graph=self._current_graph,
                    )
                raise

    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[ToolDefinition]] = None,
        tool_choice: Optional[Union[str, ToolChoice]] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion response.

        Args:
            messages: List of messages in the conversation.
            model: Model to use. If None, uses default_model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: List of tools available to the model.
            tool_choice: Controls which (if any) tool is called.

        Yields:
            Response text chunks as they arrive.
        """
        model = model or self.default_model
        client = await self._get_client()

        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        async with client.stream(
            "POST",
            "/chat/completions",
            json=request.model_dump(exclude_none=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def create_embedding(
        self,
        input_text: Union[str, list[str]],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ) -> EmbeddingResponse:
        """Create embeddings for the given input.

        Args:
            input_text: Text or list of texts to embed.
            model: Embedding model to use.
            dimensions: Number of dimensions for the embedding (model-specific).

        Returns:
            EmbeddingResponse with the embeddings.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
        """
        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="embedding"
            )

            async def _do_embedding() -> EmbeddingResponse:
                client = await self._get_client()

                request = EmbeddingRequest(
                    model=model,
                    input=input_text,
                    dimensions=dimensions,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/embeddings",
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        result = EmbeddingResponse(**response.json())

                        # Track metrics
                        self._request_count += 1
                        if result.usage:
                            self._total_tokens += result.usage.total_tokens
                            metrics.record_tokens(
                                input_tokens=result.usage.prompt_tokens,
                                output_tokens=0,
                                model=model,
                                graph=self._current_graph,
                            )

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
                            import random
                            base_delay = 2 ** attempt
                            jitter = random.uniform(0, base_delay * 0.5)
                            await asyncio.sleep(base_delay + jitter)

                raise last_error  # type: ignore

            return await self._circuit_breaker.call(_do_embedding)

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
        }

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model IDs.
        """
        client = await self._get_client()
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json()
        return [m["id"] for m in data.get("data", [])]

    def encode_image_to_url(self, image_path: str) -> str:
        """Encode a local image file to a base64 data URL.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64-encoded data URL.
        """
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Detect image type from extension
        ext = image_path.lower().split(".")[-1]
        mime_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(ext, "image/jpeg")

        return f"data:{mime_type};base64,{image_data}"

    def create_image_message(
        self,
        text: str,
        image_urls: list[str],
        detail: ImageUrlDetail = ImageUrlDetail.AUTO,
    ) -> ChatMessage:
        """Create a message with text and images.

        Args:
            text: Text content.
            image_urls: List of image URLs (can be http(s):// or data: URLs).
            detail: Detail level for image processing.

        Returns:
            ChatMessage with text and image content.
        """
        content: list[Union[TextContent, ImageContent]] = [
            TextContent(type="text", text=text)
        ]

        for url in image_urls:
            content.append(
                ImageContent(
                    type="image_url",
                    image_url=ImageUrl(url=url, detail=detail),
                )
            )

        return ChatMessage(role=MessageRole.USER, content=content)
