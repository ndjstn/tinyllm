"""Async Anthropic Claude API client for TinyLLM.

Provides async interface to Anthropic's Claude API with connection pooling,
retry logic, rate limiting, and support for all Claude features including:
- Messages API (streaming and non-streaming)
- Tool use (function calling)
- Vision (image inputs)
- Extended thinking
- Prompt caching
"""

import asyncio
import base64
import os
import random
import time
from enum import Enum
from typing import Any, AsyncIterator, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector
from tinyllm.telemetry import set_span_attribute, trace_llm_request

logger = get_logger(__name__, component="anthropic_client")
metrics = get_metrics_collector()


# Global connection pool for Anthropic client reuse
_client_pool: dict[str, "AnthropicClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_anthropic_client(
    api_key: Optional[str] = None,
    timeout_ms: int = 60000,
    max_retries: int = 3,
    rate_limit_rps: float = 5.0,
) -> "AnthropicClient":
    """Get or create a shared AnthropicClient.

    This enables connection pooling across the application by reusing
    clients instead of creating new ones for each request.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum retry attempts.
        rate_limit_rps: Rate limit in requests per second.

    Returns:
        Shared AnthropicClient instance.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "default")
    async with _pool_lock:
        if key not in _client_pool:
            logger.info(
                "creating_shared_anthropic_client",
                timeout_ms=timeout_ms,
                rate_limit_rps=rate_limit_rps,
            )
            _client_pool[key] = AnthropicClient(
                api_key=api_key,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[key]


async def close_all_anthropic_clients() -> None:
    """Close all pooled clients. Call during application shutdown."""
    logger.info("closing_all_anthropic_clients", client_count=len(_client_pool))
    async with _pool_lock:
        for client in _client_pool.values():
            await client.close()
        _client_pool.clear()


class RateLimiter:
    """Token bucket rate limiter for controlling request rate."""

    def __init__(self, rate: float = 5.0, burst: int = 10):
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
        recovery_timeout: float = 60.0,
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
                    metrics.increment_circuit_breaker_failures(model="claude")
            raise

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


# Pydantic models for API types


class ImageSource(BaseModel):
    """Image source for vision inputs."""

    type: Literal["base64", "url"]
    media_type: str = "image/jpeg"  # image/jpeg, image/png, image/gif, image/webp
    data: Optional[str] = None  # base64 encoded data
    url: Optional[str] = None  # URL to image


class TextContent(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Image content block for vision."""

    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseBlock(BaseModel):
    """Tool use request from the model."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result to send back to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Union[TextContent, ImageContent, ToolUseBlock, ToolResultBlock]


class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: Union[str, list[ContentBlock]]


class Tool(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema


class ToolChoice(BaseModel):
    """Tool choice configuration."""

    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None  # Required if type is "tool"


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class StopReason(str, Enum):
    """Reason the model stopped generating."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class MessageResponse(BaseModel):
    """Response from the messages API."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None
    usage: Usage

    def get_text(self) -> str:
        """Extract text from response content."""
        text_parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
        return "".join(text_parts)

    def get_tool_uses(self) -> list[ToolUseBlock]:
        """Extract tool use blocks from response."""
        return [block for block in self.content if isinstance(block, ToolUseBlock)]


class StreamEvent(BaseModel):
    """Base class for streaming events."""

    type: str


class MessageStartEvent(StreamEvent):
    """Message start event."""

    type: Literal["message_start"] = "message_start"
    message: MessageResponse


class ContentBlockStartEvent(StreamEvent):
    """Content block start event."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class ContentBlockDeltaEvent(StreamEvent):
    """Content block delta event."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: dict[str, Any]


class ContentBlockStopEvent(StreamEvent):
    """Content block stop event."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(StreamEvent):
    """Message delta event."""

    type: Literal["message_delta"] = "message_delta"
    delta: dict[str, Any]
    usage: Optional[dict[str, Any]] = None  # Partial usage data in streaming


class MessageStopEvent(StreamEvent):
    """Message stop event."""

    type: Literal["message_stop"] = "message_stop"


class MessageRequest(BaseModel):
    """Request for the messages API."""

    model: str
    messages: list[Message]
    max_tokens: int = 4096
    system: Optional[str] = None
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    stream: bool = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    metadata: Optional[dict[str, Any]] = None

    # Extended thinking (Claude Opus 4+)
    thinking: Optional[dict[str, Any]] = None


class AnthropicClient:
    """Async client for Anthropic Claude API with advanced features."""

    API_VERSION = "2023-06-01"
    BASE_URL = "https://api.anthropic.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_ms: int = 60000,
        max_retries: int = 3,
        rate_limit_rps: float = 5.0,
        circuit_breaker_threshold: int = 5,
        default_model: str = "claude-opus-4-20250514",
        slow_query_threshold_ms: int = 10000,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
            rate_limit_rps: Rate limit in requests per second.
            circuit_breaker_threshold: Failures before circuit opens.
            default_model: Default model to use.
            slow_query_threshold_ms: Threshold in ms for slow query detection.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY env var"
            )

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
                base_url=self.BASE_URL,
                timeout=self.timeout,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.API_VERSION,
                    "content-type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _handle_rate_limit_error(self, response: httpx.Response) -> Optional[float]:
        """Extract retry-after from rate limit response.

        Args:
            response: HTTP response.

        Returns:
            Retry-after delay in seconds, or None if not a rate limit error.
        """
        if response.status_code == 429:
            # Check for Retry-After header
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

            # Check response body for retry info
            try:
                error_data = response.json()
                if "error" in error_data and "retry_after" in error_data["error"]:
                    return float(error_data["error"]["retry_after"])
            except Exception:
                pass

            # Default rate limit backoff
            return 60.0

        return None

    async def _exponential_backoff(
        self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> None:
        """Perform exponential backoff with jitter.

        Args:
            attempt: Current retry attempt number (0-indexed).
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.
        """
        delay = min(base_delay * (2**attempt), max_delay)
        jitter = random.uniform(0, delay * 0.1)
        total_delay = delay + jitter
        logger.info("exponential_backoff", attempt=attempt, delay_seconds=total_delay)
        await asyncio.sleep(total_delay)

    async def create_message(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[ToolChoice] = None,
        thinking: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MessageResponse:
        """Create a message using Claude's Messages API.

        Args:
            messages: List of messages in the conversation.
            model: Model to use. If None, uses default model.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            temperature: Sampling temperature (0.0-1.0).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stop_sequences: Sequences that stop generation.
            tools: List of tools available for the model to use.
            tool_choice: How the model should choose tools.
            thinking: Extended thinking configuration for Claude Opus 4+.
            metadata: Additional metadata to send with the request.

        Returns:
            MessageResponse with the model's reply.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
            RuntimeError: If circuit breaker is open.
            ValueError: If API returns an error.
        """
        model = model or self._current_model

        # Update circuit breaker state metric
        metrics.update_circuit_breaker_state(self._circuit_breaker.get_state(), model=model)

        # Apply rate limiting
        rate_limit_start = time.monotonic()
        await self._rate_limiter.acquire()
        rate_limit_wait = time.monotonic() - rate_limit_start
        if rate_limit_wait > 0.001:
            metrics.record_rate_limit_wait(rate_limit_wait, model=model)

        # Track request with metrics and tracing
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            with trace_llm_request(
                model=model,
                prompt_length=sum(
                    len(str(m.content)) for m in messages
                ),  # Approximate prompt length
                temperature=temperature,
            ):
                metrics.increment_request_count(
                    model=model, graph=self._current_graph, request_type="message"
                )

                async def _do_create_message() -> MessageResponse:
                    client = await self._get_client()

                    request = MessageRequest(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        system=system,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=stop_sequences,
                        stream=False,
                        tools=tools,
                        tool_choice=tool_choice,
                        thinking=thinking,
                        metadata=metadata,
                    )

                    last_error: Optional[Exception] = None
                    query_start_time = time.monotonic()

                    for attempt in range(self.max_retries + 1):
                        try:
                            response = await client.post(
                                "/v1/messages",
                                json=request.model_dump(exclude_none=True),
                            )

                            # Handle rate limiting
                            retry_after = self._handle_rate_limit_error(response)
                            if retry_after is not None:
                                if attempt < self.max_retries:
                                    logger.warning(
                                        "rate_limit_hit",
                                        retry_after=retry_after,
                                        attempt=attempt,
                                    )
                                    await asyncio.sleep(retry_after)
                                    continue
                                else:
                                    response.raise_for_status()

                            response.raise_for_status()
                            data = response.json()

                            # Parse response
                            result = MessageResponse(**data)

                            # Calculate query duration
                            query_duration_ms = (time.monotonic() - query_start_time) * 1000

                            # Detect slow queries
                            if query_duration_ms > self.slow_query_threshold_ms:
                                slow_query_info = {
                                    "timestamp": time.time(),
                                    "model": model,
                                    "duration_ms": query_duration_ms,
                                    "message_count": len(messages),
                                    "input_tokens": result.usage.input_tokens,
                                    "output_tokens": result.usage.output_tokens,
                                    "graph": self._current_graph,
                                }
                                self._slow_queries.append(slow_query_info)

                                logger.warning(
                                    "slow_query_detected",
                                    model=model,
                                    duration_ms=query_duration_ms,
                                    threshold_ms=self.slow_query_threshold_ms,
                                    input_tokens=result.usage.input_tokens,
                                    output_tokens=result.usage.output_tokens,
                                )

                            # Track metrics
                            self._request_count += 1
                            self._total_tokens += (
                                result.usage.input_tokens + result.usage.output_tokens
                            )

                            # Record token usage
                            metrics.record_tokens(
                                input_tokens=result.usage.input_tokens,
                                output_tokens=result.usage.output_tokens,
                                model=model,
                                graph=self._current_graph,
                            )

                            # Add token counts to trace span
                            set_span_attribute("llm.input_tokens", result.usage.input_tokens)
                            set_span_attribute("llm.output_tokens", result.usage.output_tokens)
                            set_span_attribute(
                                "llm.total_tokens",
                                result.usage.input_tokens + result.usage.output_tokens,
                            )

                            # Add cache metrics if available
                            if result.usage.cache_creation_input_tokens:
                                set_span_attribute(
                                    "llm.cache_creation_tokens",
                                    result.usage.cache_creation_input_tokens,
                                )
                            if result.usage.cache_read_input_tokens:
                                set_span_attribute(
                                    "llm.cache_read_tokens",
                                    result.usage.cache_read_input_tokens,
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
                                await self._exponential_backoff(attempt)
                            else:
                                raise

                    raise last_error  # type: ignore

                # Execute with circuit breaker protection
                try:
                    return await self._circuit_breaker.call(_do_create_message)
                except RuntimeError as e:
                    if "Circuit breaker is open" in str(e):
                        metrics.increment_error_count(
                            error_type="circuit_breaker_open",
                            model=model,
                            graph=self._current_graph,
                        )
                    raise

    async def create_message_stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[ToolChoice] = None,
        thinking: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a message response using Claude's Messages API.

        Args:
            messages: List of messages in the conversation.
            model: Model to use. If None, uses default model.
            max_tokens: Maximum tokens to generate.
            system: System prompt.
            temperature: Sampling temperature (0.0-1.0).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            stop_sequences: Sequences that stop generation.
            tools: List of tools available for the model to use.
            tool_choice: How the model should choose tools.
            thinking: Extended thinking configuration for Claude Opus 4+.
            metadata: Additional metadata to send with the request.

        Yields:
            StreamEvent objects as they arrive.

        Raises:
            ValueError: If API returns an error.
        """
        model = model or self._current_model
        client = await self._get_client()

        request = MessageRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            stream=True,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            metadata=metadata,
        )

        async with client.stream(
            "POST",
            "/v1/messages",
            json=request.model_dump(exclude_none=True),
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Server-sent events format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        import json

                        data = json.loads(data_str)

                        # Parse different event types
                        event_type = data.get("type")

                        if event_type == "message_start":
                            yield MessageStartEvent(**data)
                        elif event_type == "content_block_start":
                            yield ContentBlockStartEvent(**data)
                        elif event_type == "content_block_delta":
                            yield ContentBlockDeltaEvent(**data)
                        elif event_type == "content_block_stop":
                            yield ContentBlockStopEvent(**data)
                        elif event_type == "message_delta":
                            yield MessageDeltaEvent(**data)
                        elif event_type == "message_stop":
                            yield MessageStopEvent(**data)

                    except Exception as e:
                        logger.error("stream_parse_error", error=str(e), line=line)
                        continue

    def set_graph_context(self, graph: str) -> None:
        """Set the current graph context for metrics tracking.

        Args:
            graph: Graph name to use in metrics labels.
        """
        self._current_graph = graph

    def set_model(self, model: str) -> None:
        """Set the default model for this client.

        Args:
            model: Model name to use (e.g., "claude-opus-4-20250514").
        """
        logger.info("model_switched", from_model=self._current_model, to_model=model)
        self._current_model = model

    def get_model(self) -> str:
        """Get the current default model.

        Returns:
            Current model name.
        """
        return self._current_model

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics.

        Returns:
            Dict with request_count, total_tokens, circuit_breaker_state, etc.
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

        if min_duration_ms is not None:
            queries = [q for q in queries if q["duration_ms"] >= min_duration_ms]

        queries = sorted(queries, key=lambda q: q["timestamp"], reverse=True)

        if limit is not None:
            queries = queries[:limit]

        return queries

    def clear_slow_queries(self) -> None:
        """Clear slow query history."""
        self._slow_queries.clear()
        logger.info("slow_queries_cleared")

    @staticmethod
    def encode_image(image_path: str, media_type: str = "image/jpeg") -> ImageSource:
        """Encode an image file to base64 for vision requests.

        Args:
            image_path: Path to the image file.
            media_type: MIME type of the image.

        Returns:
            ImageSource with base64 encoded image data.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return ImageSource(type="base64", media_type=media_type, data=image_data)

    @staticmethod
    def create_image_message(
        text: str,
        image_sources: list[ImageSource],
    ) -> Message:
        """Create a user message with text and images for vision requests.

        Args:
            text: Text prompt.
            image_sources: List of image sources.

        Returns:
            Message with text and image content.
        """
        content: list[ContentBlock] = []

        # Add text first
        if text:
            content.append(TextContent(text=text))

        # Add images
        for source in image_sources:
            content.append(ImageContent(source=source))

        return Message(role="user", content=content)
