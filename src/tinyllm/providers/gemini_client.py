"""Async Google Gemini API client for TinyLLM.

Provides async interface to Google's Gemini API with connection pooling,
retry logic, rate limiting, and support for all Gemini features including:
- generateContent API (chat/completion)
- Streaming responses
- Tool calling (function calling)
- Vision (image inputs)
- Embeddings
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

logger = get_logger(__name__, component="gemini_client")
metrics = get_metrics_collector()


# Global connection pool for Gemini client reuse
_client_pool: dict[str, "GeminiClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_gemini_client(
    api_key: Optional[str] = None,
    timeout_ms: int = 60000,
    max_retries: int = 3,
    rate_limit_rps: float = 5.0,
) -> "GeminiClient":
    """Get or create a shared GeminiClient.

    This enables connection pooling across the application by reusing
    clients instead of creating new ones for each request.

    Args:
        api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum retry attempts.
        rate_limit_rps: Rate limit in requests per second.

    Returns:
        Shared GeminiClient instance.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "default")
    async with _pool_lock:
        if key not in _client_pool:
            logger.info(
                "creating_shared_gemini_client",
                timeout_ms=timeout_ms,
                rate_limit_rps=rate_limit_rps,
            )
            _client_pool[key] = GeminiClient(
                api_key=api_key,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[key]


async def close_all_gemini_clients() -> None:
    """Close all pooled clients. Call during application shutdown."""
    logger.info("closing_all_gemini_clients", client_count=len(_client_pool))
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
                    metrics.increment_circuit_breaker_failures(model="gemini")
            raise

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


# Pydantic models for Gemini API types


class HarmCategory(str, Enum):
    """Harm categories for safety settings."""

    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"


class HarmBlockThreshold(str, Enum):
    """Threshold for blocking harmful content."""

    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"


class SafetySetting(BaseModel):
    """Safety setting for content filtering."""

    category: HarmCategory
    threshold: HarmBlockThreshold


class Blob(BaseModel):
    """Binary data blob."""

    mime_type: str
    data: str  # Base64 encoded data


class FileData(BaseModel):
    """File data reference."""

    mime_type: str
    file_uri: str


class InlineData(BaseModel):
    """Inline data for images/files."""

    inline_data: Blob


class FileDataPart(BaseModel):
    """File data part."""

    file_data: FileData


class TextPart(BaseModel):
    """Text part of content."""

    text: str


class FunctionCall(BaseModel):
    """Function call from model."""

    name: str
    args: dict[str, Any]


class FunctionCallPart(BaseModel):
    """Function call part."""

    function_call: FunctionCall


class FunctionResponse(BaseModel):
    """Function response to send back."""

    name: str
    response: dict[str, Any]


class FunctionResponsePart(BaseModel):
    """Function response part."""

    function_response: FunctionResponse


ContentPart = Union[TextPart, InlineData, FileDataPart, FunctionCallPart, FunctionResponsePart]


class Content(BaseModel):
    """Content in a message."""

    parts: list[ContentPart]
    role: Optional[Literal["user", "model"]] = None


class FunctionDeclaration(BaseModel):
    """Function declaration for tool calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class Tool(BaseModel):
    """Tool definition."""

    function_declarations: list[FunctionDeclaration]


class ToolConfig(BaseModel):
    """Tool configuration."""

    function_calling_config: Optional[dict[str, Any]] = None


class GenerationConfig(BaseModel):
    """Configuration for content generation."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None


class GenerateContentRequest(BaseModel):
    """Request for generateContent API."""

    contents: list[Content]
    generation_config: Optional[GenerationConfig] = None
    safety_settings: Optional[list[SafetySetting]] = None
    tools: Optional[list[Tool]] = None
    tool_config: Optional[ToolConfig] = None
    system_instruction: Optional[Content] = None


class FinishReason(str, Enum):
    """Reason generation finished."""

    FINISH_REASON_UNSPECIFIED = "FINISH_REASON_UNSPECIFIED"
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    OTHER = "OTHER"


class SafetyRating(BaseModel):
    """Safety rating for content."""

    category: HarmCategory
    probability: str


class CitationSource(BaseModel):
    """Citation source."""

    start_index: Optional[int] = None
    end_index: Optional[int] = None
    uri: Optional[str] = None
    license: Optional[str] = None


class CitationMetadata(BaseModel):
    """Citation metadata."""

    citation_sources: list[CitationSource] = Field(default_factory=list)


class Candidate(BaseModel):
    """Response candidate."""

    content: Content
    finish_reason: Optional[FinishReason] = None
    safety_ratings: Optional[list[SafetyRating]] = None
    citation_metadata: Optional[CitationMetadata] = None
    index: int = 0


class UsageMetadata(BaseModel):
    """Token usage metadata."""

    prompt_token_count: int
    candidates_token_count: int = 0
    total_token_count: int


class GenerateContentResponse(BaseModel):
    """Response from generateContent API."""

    candidates: list[Candidate]
    usage_metadata: Optional[UsageMetadata] = None
    prompt_feedback: Optional[dict[str, Any]] = None

    def get_text(self) -> str:
        """Extract text from response."""
        if not self.candidates:
            return ""

        text_parts = []
        for part in self.candidates[0].content.parts:
            if isinstance(part, TextPart):
                text_parts.append(part.text)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])

        return "".join(text_parts)

    def get_function_calls(self) -> list[FunctionCall]:
        """Extract function calls from response."""
        if not self.candidates:
            return []

        calls = []
        for part in self.candidates[0].content.parts:
            if isinstance(part, FunctionCallPart):
                calls.append(part.function_call)
            elif isinstance(part, dict) and "function_call" in part:
                calls.append(FunctionCall(**part["function_call"]))

        return calls


class EmbedContentRequest(BaseModel):
    """Request for embedding content."""

    model: str
    content: Content
    task_type: Optional[str] = None  # RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.
    title: Optional[str] = None


class ContentEmbedding(BaseModel):
    """Content embedding."""

    values: list[float]


class EmbedContentResponse(BaseModel):
    """Response from embedding API."""

    embedding: ContentEmbedding


class BatchEmbedContentsRequest(BaseModel):
    """Request for batch embedding."""

    requests: list[EmbedContentRequest]


class BatchEmbedContentsResponse(BaseModel):
    """Response from batch embedding API."""

    embeddings: list[ContentEmbedding]


class GeminiClient:
    """Async client for Google Gemini API with advanced features."""

    BASE_URL = "https://generativelanguage.googleapis.com"
    API_VERSION = "v1beta"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_ms: int = 60000,
        max_retries: int = 3,
        rate_limit_rps: float = 5.0,
        circuit_breaker_threshold: int = 5,
        default_model: str = "gemini-1.5-pro",
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
            rate_limit_rps: Rate limit in requests per second.
            circuit_breaker_threshold: Failures before circuit opens.
            default_model: Default model to use.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key must be provided or set in GOOGLE_API_KEY env var"
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

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"{self.BASE_URL}/{self.API_VERSION}",
                timeout=self.timeout,
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

    async def generate_content(
        self,
        contents: list[Content],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[list[SafetySetting]] = None,
        tools: Optional[list[Tool]] = None,
        tool_config: Optional[ToolConfig] = None,
        system_instruction: Optional[Content] = None,
    ) -> GenerateContentResponse:
        """Generate content using Gemini API.

        Args:
            contents: List of content messages.
            model: Model to use. If None, uses default model.
            generation_config: Generation configuration.
            safety_settings: Safety settings.
            tools: List of tools available for the model.
            tool_config: Tool configuration.
            system_instruction: System instruction.

        Returns:
            GenerateContentResponse with the model's reply.

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

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="generate_content"
            )

            async def _do_generate_content() -> GenerateContentResponse:
                client = await self._get_client()

                request = GenerateContentRequest(
                    contents=contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    tools=tools,
                    tool_config=tool_config,
                    system_instruction=system_instruction,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            f"/models/{model}:generateContent",
                            params={"key": self.api_key},
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
                        result = GenerateContentResponse(**data)

                        # Track metrics
                        self._request_count += 1
                        if result.usage_metadata:
                            self._total_tokens += result.usage_metadata.total_token_count

                            # Record token usage
                            metrics.record_tokens(
                                input_tokens=result.usage_metadata.prompt_token_count,
                                output_tokens=result.usage_metadata.candidates_token_count,
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
                            await self._exponential_backoff(attempt)
                        else:
                            raise

                raise last_error  # type: ignore

            # Execute with circuit breaker protection
            try:
                return await self._circuit_breaker.call(_do_generate_content)
            except RuntimeError as e:
                if "Circuit breaker is open" in str(e):
                    metrics.increment_error_count(
                        error_type="circuit_breaker_open",
                        model=model,
                        graph=self._current_graph,
                    )
                raise

    async def generate_content_stream(
        self,
        contents: list[Content],
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[list[SafetySetting]] = None,
        tools: Optional[list[Tool]] = None,
        tool_config: Optional[ToolConfig] = None,
        system_instruction: Optional[Content] = None,
    ) -> AsyncIterator[GenerateContentResponse]:
        """Stream content generation using Gemini API.

        Args:
            contents: List of content messages.
            model: Model to use. If None, uses default model.
            generation_config: Generation configuration.
            safety_settings: Safety settings.
            tools: List of tools available for the model.
            tool_config: Tool configuration.
            system_instruction: System instruction.

        Yields:
            GenerateContentResponse chunks as they arrive.

        Raises:
            ValueError: If API returns an error.
        """
        model = model or self._current_model
        client = await self._get_client()

        request = GenerateContentRequest(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
        )

        async with client.stream(
            "POST",
            f"/models/{model}:streamGenerateContent",
            params={"key": self.api_key, "alt": "sse"},
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
                        yield GenerateContentResponse(**data)

                    except Exception as e:
                        logger.error("stream_parse_error", error=str(e), line=line)
                        continue

    async def embed_content(
        self,
        content: Union[str, Content],
        model: str = "models/embedding-001",
        task_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> EmbedContentResponse:
        """Generate embeddings for content.

        Args:
            content: Text string or Content object to embed.
            model: Embedding model to use.
            task_type: Task type (e.g., "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT").
            title: Optional title for the content.

        Returns:
            EmbedContentResponse with the embedding.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
        """
        # Convert string to Content if needed
        if isinstance(content, str):
            content = Content(parts=[TextPart(text=content)])

        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="embedding"
            )

            async def _do_embed() -> EmbedContentResponse:
                client = await self._get_client()

                request = EmbedContentRequest(
                    model=model,
                    content=content,
                    task_type=task_type,
                    title=title,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            f"/{model}:embedContent",
                            params={"key": self.api_key},
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        result = EmbedContentResponse(**response.json())

                        # Track metrics
                        self._request_count += 1

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

                raise last_error  # type: ignore

            return await self._circuit_breaker.call(_do_embed)

    async def batch_embed_contents(
        self,
        contents: list[Union[str, Content]],
        model: str = "models/embedding-001",
        task_type: Optional[str] = None,
    ) -> BatchEmbedContentsResponse:
        """Generate embeddings for multiple contents in a batch.

        Args:
            contents: List of text strings or Content objects to embed.
            model: Embedding model to use.
            task_type: Task type for all contents.

        Returns:
            BatchEmbedContentsResponse with the embeddings.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
        """
        # Convert strings to Content if needed
        embed_requests = []
        for content in contents:
            if isinstance(content, str):
                content = Content(parts=[TextPart(text=content)])
            embed_requests.append(
                EmbedContentRequest(
                    model=model,
                    content=content,
                    task_type=task_type,
                )
            )

        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="batch_embedding"
            )

            async def _do_batch_embed() -> BatchEmbedContentsResponse:
                client = await self._get_client()

                request = BatchEmbedContentsRequest(requests=embed_requests)

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            f"/{model}:batchEmbedContents",
                            params={"key": self.api_key},
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        result = BatchEmbedContentsResponse(**response.json())

                        # Track metrics
                        self._request_count += 1

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

                raise last_error  # type: ignore

            return await self._circuit_breaker.call(_do_batch_embed)

    def set_graph_context(self, graph: str) -> None:
        """Set the current graph context for metrics tracking.

        Args:
            graph: Graph name to use in metrics labels.
        """
        self._current_graph = graph

    def set_model(self, model: str) -> None:
        """Set the default model for this client.

        Args:
            model: Model name to use (e.g., "gemini-1.5-pro").
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
        }

    @staticmethod
    def encode_image(image_path: str, mime_type: str = "image/jpeg") -> InlineData:
        """Encode an image file to base64 for vision requests.

        Args:
            image_path: Path to the image file.
            mime_type: MIME type of the image.

        Returns:
            InlineData with base64 encoded image data.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return InlineData(
            inline_data=Blob(mime_type=mime_type, data=image_data)
        )

    @staticmethod
    def create_image_content(
        text: str,
        image_parts: list[InlineData],
    ) -> Content:
        """Create a content message with text and images for vision requests.

        Args:
            text: Text prompt.
            image_parts: List of inline image data.

        Returns:
            Content with text and image parts.
        """
        parts: list[ContentPart] = []

        # Add text first
        if text:
            parts.append(TextPart(text=text))

        # Add images
        parts.extend(image_parts)

        return Content(parts=parts, role="user")

    @staticmethod
    def create_text_content(text: str, role: Literal["user", "model"] = "user") -> Content:
        """Create a simple text content message.

        Args:
            text: Text content.
            role: Role of the message (user or model).

        Returns:
            Content with text.
        """
        return Content(parts=[TextPart(text=text)], role=role)

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names.
        """
        client = await self._get_client()
        response = await client.get("/models", params={"key": self.api_key})
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
