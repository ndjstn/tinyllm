"""Async generator streaming for TinyLLM graph execution.

This module provides streaming capabilities for graph execution,
allowing results to be consumed as they are produced via async generators.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, NodeResult
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="streaming")


class StreamChunk(BaseModel):
    """A chunk of streaming data from graph execution.

    Represents a discrete piece of output that can be yielded
    during graph execution, including partial results, node outputs,
    and metadata.
    """

    model_config = {"extra": "forbid"}

    chunk_id: int = Field(ge=0, description="Sequential chunk identifier")
    node_id: str = Field(description="Node that produced this chunk")
    content: str = Field(description="Chunk content")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional chunk metadata"
    )


class StreamConfig(BaseModel):
    """Configuration for streaming execution."""

    model_config = {"extra": "forbid"}

    buffer_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum chunks to buffer before backpressure",
    )
    chunk_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Timeout for chunk production in milliseconds",
    )
    enable_partial_results: bool = Field(
        default=True,
        description="Whether to stream partial results from nodes",
    )
    stream_intermediate_nodes: bool = Field(
        default=True,
        description="Whether to stream output from intermediate nodes",
    )


class StreamingExecutor:
    """Executor that provides streaming output via async generators.

    Wraps standard execution to produce results incrementally as nodes
    complete, allowing consumers to process output as it becomes available.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize streaming executor.

        Args:
            config: Streaming configuration.
        """
        self.config = config or StreamConfig()
        self._chunk_counter = 0
        self._stream_buffer: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue(
            maxsize=self.config.buffer_size
        )

    async def stream_execution(
        self,
        node_results: AsyncGenerator[NodeResult, None],
        context: ExecutionContext,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream execution results as chunks.

        Converts node execution results into streamable chunks,
        handling buffering and backpressure automatically.

        Args:
            node_results: Async generator of node results.
            context: Execution context.

        Yields:
            StreamChunk instances as nodes complete.
        """
        logger.info(
            "streaming_execution_started",
            trace_id=context.trace_id,
            buffer_size=self.config.buffer_size,
        )

        self._chunk_counter = 0

        try:
            async for result in node_results:
                # Convert node result to stream chunk(s)
                chunks = await self._result_to_chunks(result, context)

                for chunk in chunks:
                    yield chunk
                    logger.debug(
                        "stream_chunk_yielded",
                        trace_id=context.trace_id,
                        chunk_id=chunk.chunk_id,
                        node_id=chunk.node_id,
                        is_final=chunk.is_final,
                    )

            # Send final chunk
            final_chunk = StreamChunk(
                chunk_id=self._chunk_counter,
                node_id="executor",
                content="",
                is_final=True,
                metadata={"completed": True},
            )
            yield final_chunk

            logger.info(
                "streaming_execution_completed",
                trace_id=context.trace_id,
                total_chunks=self._chunk_counter + 1,
            )

        except asyncio.CancelledError:
            logger.warning(
                "streaming_execution_cancelled",
                trace_id=context.trace_id,
                chunks_produced=self._chunk_counter,
            )
            raise

        except Exception as e:
            logger.error(
                "streaming_execution_failed",
                trace_id=context.trace_id,
                error=str(e),
                chunks_produced=self._chunk_counter,
            )
            # Send error chunk
            error_chunk = StreamChunk(
                chunk_id=self._chunk_counter,
                node_id="executor",
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True, "error_message": str(e)},
            )
            yield error_chunk
            raise

    async def _result_to_chunks(
        self,
        result: NodeResult,
        context: ExecutionContext,
    ) -> List[StreamChunk]:
        """Convert a node result into stream chunks.

        Args:
            result: Node execution result.
            context: Execution context.

        Returns:
            List of stream chunks.
        """
        chunks: List[StreamChunk] = []

        # Extract content from result
        if result.output_messages:
            for msg in result.output_messages:
                content = msg.payload.content or ""

                # Skip empty content unless configured to include
                if not content and not self.config.enable_partial_results:
                    continue

                chunk = StreamChunk(
                    chunk_id=self._chunk_counter,
                    node_id=msg.source_node,
                    content=content,
                    is_final=False,
                    metadata={
                        "message_id": msg.message_id,
                        "success": result.success,
                        "latency_ms": result.latency_ms,
                    },
                )
                chunks.append(chunk)
                self._chunk_counter += 1

        # If no messages but result has output, create chunk
        elif result.metadata:
            content = str(result.metadata.get("output", ""))
            if content:
                chunk = StreamChunk(
                    chunk_id=self._chunk_counter,
                    node_id=context.current_node or "unknown",
                    content=content,
                    is_final=False,
                    metadata={
                        "success": result.success,
                        "latency_ms": result.latency_ms,
                    },
                )
                chunks.append(chunk)
                self._chunk_counter += 1

        return chunks

    async def stream_with_buffering(
        self,
        source: AsyncGenerator[NodeResult, None],
        context: ExecutionContext,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream with buffering to handle backpressure.

        Uses a bounded queue to buffer chunks, applying backpressure
        to producers when buffer is full.

        Args:
            source: Source async generator of node results.
            context: Execution context.

        Yields:
            Buffered stream chunks.
        """
        producer_task = asyncio.create_task(
            self._produce_chunks(source, context)
        )

        try:
            while True:
                # Wait for next chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        self._stream_buffer.get(),
                        timeout=self.config.chunk_timeout_ms / 1000,
                    )

                    if chunk is None:
                        # Producer signaled completion
                        break

                    yield chunk

                except asyncio.TimeoutError:
                    logger.warning(
                        "stream_chunk_timeout",
                        trace_id=context.trace_id,
                        timeout_ms=self.config.chunk_timeout_ms,
                    )
                    # Check if producer is still running
                    if producer_task.done():
                        break
                    continue

        finally:
            # Ensure producer is cancelled
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

    async def _produce_chunks(
        self,
        source: AsyncGenerator[NodeResult, None],
        context: ExecutionContext,
    ) -> None:
        """Producer that feeds chunks into the buffer.

        Args:
            source: Source async generator of node results.
            context: Execution context.
        """
        try:
            async for result in source:
                chunks = await self._result_to_chunks(result, context)

                for chunk in chunks:
                    # Put chunk in buffer (blocks if full - backpressure)
                    await self._stream_buffer.put(chunk)

            # Signal completion
            await self._stream_buffer.put(None)

        except Exception as e:
            logger.error(
                "chunk_production_failed",
                trace_id=context.trace_id,
                error=str(e),
            )
            # Put error chunk
            error_chunk = StreamChunk(
                chunk_id=self._chunk_counter,
                node_id="producer",
                content=f"Production error: {str(e)}",
                is_final=True,
                metadata={"error": True},
            )
            await self._stream_buffer.put(error_chunk)
            await self._stream_buffer.put(None)

    def reset(self) -> None:
        """Reset the streaming executor state."""
        self._chunk_counter = 0
        # Clear the buffer
        while not self._stream_buffer.empty():
            try:
                self._stream_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.debug("streaming_executor_reset")


async def stream_node_outputs(
    messages: List[Message],
    chunk_size: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Stream message content in chunks.

    Helper function to stream message content, optionally breaking
    into smaller chunks for more granular streaming.

    Args:
        messages: List of messages to stream.
        chunk_size: Optional size for content chunking (in characters).

    Yields:
        Content chunks from messages.
    """
    for message in messages:
        content = message.payload.content or ""

        if chunk_size and len(content) > chunk_size:
            # Break content into chunks
            for i in range(0, len(content), chunk_size):
                chunk = content[i : i + chunk_size]
                yield chunk
                # Small delay to simulate streaming
                await asyncio.sleep(0.001)
        else:
            yield content
