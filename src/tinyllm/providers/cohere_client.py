"""Async Cohere API client for TinyLLM.

Provides async interface to Cohere API with connection pooling,
retry logic, rate limiting, and support for all Cohere features including:
- Chat API (chat completions)
- Streaming responses
- Tool use (connectors)
- Embeddings (embed endpoint)
- Rerank endpoint
"""

import asyncio
import os
import random
import time
from enum import Enum
from typing import Any, AsyncIterator, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="cohere_client")
metrics = get_metrics_collector()


# Global connection pool for Cohere client reuse
_client_pool: dict[str, "CohereClient"] = {}
_pool_lock = asyncio.Lock()


async def get_shared_cohere_client(
    api_key: Optional[str] = None,
    timeout_ms: int = 60000,
    max_retries: int = 3,
    rate_limit_rps: float = 10.0,
) -> "CohereClient":
    """Get or create a shared CohereClient.

    This enables connection pooling across the application by reusing
    clients instead of creating new ones for each request.

    Args:
        api_key: Cohere API key. If None, reads from COHERE_API_KEY env var.
        timeout_ms: Request timeout in milliseconds.
        max_retries: Maximum retry attempts.
        rate_limit_rps: Rate limit in requests per second.

    Returns:
        Shared CohereClient instance.

    Raises:
        ValueError: If no API key provided and COHERE_API_KEY env var not set.
    """
    key = api_key or os.environ.get("COHERE_API_KEY", "default")
    async with _pool_lock:
        if key not in _client_pool:
            logger.info(
                "creating_shared_cohere_client",
                timeout_ms=timeout_ms,
                rate_limit_rps=rate_limit_rps,
            )
            _client_pool[key] = CohereClient(
                api_key=api_key,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                rate_limit_rps=rate_limit_rps,
            )
        return _client_pool[key]


async def close_all_cohere_clients() -> None:
    """Close all pooled clients. Call during application shutdown."""
    logger.info("closing_all_cohere_clients", client_count=len(_client_pool))
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
                    metrics.increment_circuit_breaker_failures(model="cohere")
            raise

    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state


# Pydantic models for API types


class ChatRole(str, Enum):
    """Role of a message in the chat."""

    USER = "USER"
    CHATBOT = "CHATBOT"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"


class ChatMessage(BaseModel):
    """A message in the chat conversation."""

    role: ChatRole
    message: str


class ToolParameterDefinition(BaseModel):
    """Tool parameter definition."""

    description: str
    type: str
    required: bool = False


class Tool(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    parameter_definitions: dict[str, ToolParameterDefinition]


class ToolCall(BaseModel):
    """Tool call from the model."""

    name: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """Tool result to send back to the model."""

    call: ToolCall
    outputs: list[dict[str, Any]]


class ChatRequest(BaseModel):
    """Request for Cohere chat endpoint."""

    message: str
    model: str = "command-r-plus"
    chat_history: Optional[list[ChatMessage]] = None
    conversation_id: Optional[str] = None
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    k: int = 0  # top-k sampling
    p: float = 0.75  # top-p sampling
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[list[str]] = None
    stream: bool = False
    tools: Optional[list[Tool]] = None
    tool_results: Optional[list[ToolResult]] = None
    preamble: Optional[str] = None  # System prompt
    connectors: Optional[list[dict[str, Any]]] = None  # Connector definitions


class TokenCount(BaseModel):
    """Token count information."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    billed_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    """Response from Cohere chat endpoint."""

    text: str
    generation_id: Optional[str] = None
    conversation_id: Optional[str] = None
    finish_reason: Optional[str] = None
    token_count: Optional[TokenCount] = None
    tool_calls: Optional[list[ToolCall]] = None
    citations: Optional[list[dict[str, Any]]] = None
    documents: Optional[list[dict[str, Any]]] = None
    search_queries: Optional[list[dict[str, Any]]] = None
    search_results: Optional[list[dict[str, Any]]] = None


class ChatStreamEvent(BaseModel):
    """Streaming event from chat endpoint."""

    event_type: str
    text: Optional[str] = None
    is_finished: bool = False
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class EmbedRequest(BaseModel):
    """Request for Cohere embed endpoint."""

    texts: list[str]
    model: str = "embed-english-v3.0"
    input_type: Optional[Literal["search_document", "search_query", "classification", "clustering"]] = None
    embedding_types: Optional[list[Literal["float", "int8", "uint8", "binary", "ubinary"]]] = None
    truncate: Literal["NONE", "START", "END"] = "END"


class Embedding(BaseModel):
    """Embedding data."""

    values: list[float]


class EmbedResponse(BaseModel):
    """Response from Cohere embed endpoint."""

    id: str
    embeddings: list[Embedding]
    texts: list[str]
    meta: Optional[dict[str, Any]] = None


class RerankDocument(BaseModel):
    """Document to rerank."""

    text: str


class RerankRequest(BaseModel):
    """Request for Cohere rerank endpoint."""

    query: str
    documents: list[Union[str, RerankDocument]]
    model: str = "rerank-english-v3.0"
    top_n: Optional[int] = None
    max_chunks_per_doc: Optional[int] = None
    return_documents: bool = True


class RerankResult(BaseModel):
    """Reranked result."""

    index: int
    relevance_score: float
    document: Optional[dict[str, Any]] = None


class RerankResponse(BaseModel):
    """Response from Cohere rerank endpoint."""

    id: str
    results: list[RerankResult]
    meta: Optional[dict[str, Any]] = None


class CohereClient:
    """Async client for Cohere API with advanced features."""

    BASE_URL = "https://api.cohere.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_ms: int = 60000,
        max_retries: int = 3,
        rate_limit_rps: float = 10.0,
        circuit_breaker_threshold: int = 5,
        default_model: str = "command-r-plus",
    ):
        """Initialize Cohere client.

        Args:
            api_key: Cohere API key. If None, reads from COHERE_API_KEY env var.
            timeout_ms: Request timeout in milliseconds.
            max_retries: Maximum retry attempts.
            rate_limit_rps: Rate limit in requests per second.
            circuit_breaker_threshold: Failures before circuit opens.
            default_model: Default model to use for chat.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key must be provided or set in COHERE_API_KEY env var"
            )

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
                base_url=self.BASE_URL,
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
                if "retry_after" in error_data:
                    return float(error_data["retry_after"])
            except Exception:
                pass

            # Default rate limit backoff
            return 60.0

        return None

    async def chat(
        self,
        message: str,
        model: Optional[str] = None,
        chat_history: Optional[list[ChatMessage]] = None,
        conversation_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tools: Optional[list[Tool]] = None,
        tool_results: Optional[list[ToolResult]] = None,
        preamble: Optional[str] = None,
        connectors: Optional[list[dict[str, Any]]] = None,
        stop_sequences: Optional[list[str]] = None,
        stream: bool = False,
    ) -> ChatResponse:
        """Create a chat completion.

        Args:
            message: User message.
            model: Model to use. If None, uses default_model.
            chat_history: Previous messages in the conversation.
            conversation_id: Conversation ID for tracking.
            temperature: Sampling temperature (0.0-5.0).
            max_tokens: Maximum tokens to generate.
            tools: List of tools available to the model.
            tool_results: Results from previous tool calls.
            preamble: System prompt to guide the model.
            connectors: Connector definitions for web search, etc.
            stop_sequences: Sequences that stop generation.
            stream: Whether to stream the response.

        Returns:
            ChatResponse with the model's response.

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
                model=model, graph=self._current_graph, request_type="chat"
            )

            async def _do_chat() -> ChatResponse:
                client = await self._get_client()

                request = ChatRequest(
                    message=message,
                    model=model,
                    chat_history=chat_history,
                    conversation_id=conversation_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_results=tool_results,
                    preamble=preamble,
                    connectors=connectors,
                    stop_sequences=stop_sequences,
                    stream=stream,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/chat",
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
                        result = ChatResponse(**data)

                        # Track metrics
                        self._request_count += 1
                        if result.token_count:
                            if result.token_count.total_tokens:
                                self._total_tokens += result.token_count.total_tokens

                            # Record token usage
                            metrics.record_tokens(
                                input_tokens=result.token_count.input_tokens or 0,
                                output_tokens=result.token_count.output_tokens or 0,
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
                return await self._circuit_breaker.call(_do_chat)
            except RuntimeError as e:
                if "Circuit breaker is open" in str(e):
                    metrics.increment_error_count(
                        error_type="circuit_breaker_open",
                        model=model,
                        graph=self._current_graph,
                    )
                raise

    async def chat_stream(
        self,
        message: str,
        model: Optional[str] = None,
        chat_history: Optional[list[ChatMessage]] = None,
        conversation_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tools: Optional[list[Tool]] = None,
        tool_results: Optional[list[ToolResult]] = None,
        preamble: Optional[str] = None,
        connectors: Optional[list[dict[str, Any]]] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream chat completion response.

        Args:
            message: User message.
            model: Model to use. If None, uses default_model.
            chat_history: Previous messages in the conversation.
            conversation_id: Conversation ID for tracking.
            temperature: Sampling temperature (0.0-5.0).
            max_tokens: Maximum tokens to generate.
            tools: List of tools available to the model.
            tool_results: Results from previous tool calls.
            preamble: System prompt to guide the model.
            connectors: Connector definitions for web search, etc.
            stop_sequences: Sequences that stop generation.

        Yields:
            ChatStreamEvent objects as they arrive.
        """
        model = model or self.default_model
        client = await self._get_client()

        request = ChatRequest(
            message=message,
            model=model,
            chat_history=chat_history,
            conversation_id=conversation_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_results=tool_results,
            preamble=preamble,
            connectors=connectors,
            stop_sequences=stop_sequences,
            stream=True,
        )

        async with client.stream(
            "POST",
            "/chat",
            json=request.model_dump(exclude_none=True),
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    import json

                    data = json.loads(line)
                    event_type = data.get("event_type", "")

                    # Create stream event
                    event = ChatStreamEvent(
                        event_type=event_type,
                        text=data.get("text"),
                        is_finished=data.get("is_finished", False),
                        finish_reason=data.get("finish_reason"),
                        tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
                    )

                    yield event

                    if event.is_finished:
                        break

                except Exception as e:
                    logger.error("stream_parse_error", error=str(e), line=line)
                    continue

    async def embed(
        self,
        texts: list[str],
        model: str = "embed-english-v3.0",
        input_type: Optional[Literal["search_document", "search_query", "classification", "clustering"]] = None,
        embedding_types: Optional[list[Literal["float", "int8", "uint8", "binary", "ubinary"]]] = None,
        truncate: Literal["NONE", "START", "END"] = "END",
    ) -> EmbedResponse:
        """Create embeddings for the given texts.

        Args:
            texts: List of texts to embed.
            model: Embedding model to use.
            input_type: Type of input for specialized embeddings.
            embedding_types: Types of embeddings to return.
            truncate: How to truncate texts that exceed model's context length.

        Returns:
            EmbedResponse with the embeddings.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
        """
        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="embed"
            )

            async def _do_embed() -> EmbedResponse:
                client = await self._get_client()

                request = EmbedRequest(
                    texts=texts,
                    model=model,
                    input_type=input_type,
                    embedding_types=embedding_types,
                    truncate=truncate,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/embed",
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        data = response.json()

                        # Parse response - handle different response formats
                        if "embeddings" in data:
                            # New format: embeddings are nested
                            embeddings_data = data["embeddings"]
                            if isinstance(embeddings_data, dict):
                                # Handle dict format with embedding type keys
                                embedding_list = embeddings_data.get("float", [])
                                embeddings = [Embedding(values=e) for e in embedding_list]
                            else:
                                # Handle list format
                                embeddings = [Embedding(values=e) for e in embeddings_data]
                        else:
                            embeddings = []

                        result = EmbedResponse(
                            id=data.get("id", ""),
                            embeddings=embeddings,
                            texts=texts,
                            meta=data.get("meta"),
                        )

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

    async def rerank(
        self,
        query: str,
        documents: list[Union[str, RerankDocument]],
        model: str = "rerank-english-v3.0",
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: bool = True,
    ) -> RerankResponse:
        """Rerank documents based on relevance to a query.

        Args:
            query: Query to rank documents against.
            documents: List of documents to rerank.
            model: Rerank model to use.
            top_n: Number of top results to return.
            max_chunks_per_doc: Maximum chunks per document.
            return_documents: Whether to return document text in results.

        Returns:
            RerankResponse with reranked results.

        Raises:
            httpx.HTTPError: On network/API errors after retries.
        """
        # Apply rate limiting
        await self._rate_limiter.acquire()

        # Track request with metrics
        with metrics.track_request_latency(model=model, graph=self._current_graph):
            metrics.increment_request_count(
                model=model, graph=self._current_graph, request_type="rerank"
            )

            async def _do_rerank() -> RerankResponse:
                client = await self._get_client()

                request = RerankRequest(
                    query=query,
                    documents=documents,
                    model=model,
                    top_n=top_n,
                    max_chunks_per_doc=max_chunks_per_doc,
                    return_documents=return_documents,
                )

                last_error: Optional[Exception] = None

                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.post(
                            "/rerank",
                            json=request.model_dump(exclude_none=True),
                        )
                        response.raise_for_status()
                        data = response.json()

                        result = RerankResponse(**data)

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

            return await self._circuit_breaker.call(_do_rerank)

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
