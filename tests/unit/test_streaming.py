"""Tests for async generator streaming."""

import asyncio

import pytest

from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult
from tinyllm.core.streaming import (
    StreamChunk,
    StreamConfig,
    StreamingExecutor,
    stream_node_outputs,
)


async def mock_node_results() -> list[NodeResult]:
    """Generate mock node results for testing."""
    results = []
    for i in range(3):
        msg = Message(
            trace_id="test-trace",
            source_node=f"node-{i}",
            target_node="next",
            payload=MessagePayload(content=f"Result {i}"),
        )
        result = NodeResult(
            success=True,
            output_messages=[msg],
            latency_ms=100,
        )
        results.append(result)
    return results


async def async_result_generator(
    results: list[NodeResult],
) -> NodeResult:
    """Async generator for node results."""
    for result in results:
        await asyncio.sleep(0.01)  # Simulate processing time
        yield result


@pytest.mark.asyncio
async def test_streaming_basic():
    """Test basic streaming execution."""
    config = StreamConfig(buffer_size=10)
    executor = StreamingExecutor(config)

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    results = await mock_node_results()
    chunks = []

    async for chunk in executor.stream_execution(
        async_result_generator(results), context
    ):
        chunks.append(chunk)

    # Should have 3 result chunks + 1 final chunk
    assert len(chunks) == 4
    assert chunks[-1].is_final is True
    assert chunks[0].content == "Result 0"
    assert chunks[1].content == "Result 1"
    assert chunks[2].content == "Result 2"


@pytest.mark.asyncio
async def test_streaming_chunk_metadata():
    """Test that chunks include proper metadata."""
    executor = StreamingExecutor()

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    results = await mock_node_results()

    async for chunk in executor.stream_execution(
        async_result_generator(results), context
    ):
        if not chunk.is_final:
            assert chunk.chunk_id >= 0
            assert chunk.node_id.startswith("node-")
            assert "success" in chunk.metadata
            assert "latency_ms" in chunk.metadata


@pytest.mark.asyncio
async def test_streaming_with_buffering():
    """Test streaming with buffering."""
    config = StreamConfig(buffer_size=5, chunk_timeout_ms=1000)
    executor = StreamingExecutor(config)

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    results = await mock_node_results()
    chunks = []

    async for chunk in executor.stream_with_buffering(
        async_result_generator(results), context
    ):
        chunks.append(chunk)

    # Should have buffered all chunks
    assert len(chunks) >= 3


@pytest.mark.asyncio
async def test_streaming_empty_results():
    """Test streaming with empty results."""
    executor = StreamingExecutor()

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    async def empty_generator():
        return
        yield  # Make it a generator

    chunks = []
    async for chunk in executor.stream_execution(empty_generator(), context):
        chunks.append(chunk)

    # Should only have final chunk
    assert len(chunks) == 1
    assert chunks[0].is_final is True


@pytest.mark.asyncio
async def test_streaming_error_handling():
    """Test streaming handles errors gracefully."""
    executor = StreamingExecutor()

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    async def error_generator():
        yield NodeResult(
            success=True,
            output_messages=[
                Message(
                    trace_id="test",
                    source_node="node1",
                    target_node="node2",
                    payload=MessagePayload(content="Before error"),
                )
            ],
        )
        raise ValueError("Test error")

    chunks = []
    with pytest.raises(ValueError):
        async for chunk in executor.stream_execution(error_generator(), context):
            chunks.append(chunk)

    # Should have at least the successful chunk and error chunk
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_stream_node_outputs():
    """Test streaming node outputs helper."""
    messages = [
        Message(
            trace_id="test",
            source_node="node1",
            target_node="node2",
            payload=MessagePayload(content=f"Message {i}"),
        )
        for i in range(3)
    ]

    chunks = []
    async for chunk in stream_node_outputs(messages):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0] == "Message 0"
    assert chunks[1] == "Message 1"
    assert chunks[2] == "Message 2"


@pytest.mark.asyncio
async def test_stream_node_outputs_chunked():
    """Test streaming node outputs with chunking."""
    messages = [
        Message(
            trace_id="test",
            source_node="node1",
            target_node="node2",
            payload=MessagePayload(content="A" * 100),
        )
    ]

    chunks = []
    async for chunk in stream_node_outputs(messages, chunk_size=10):
        chunks.append(chunk)

    # 100 characters / 10 per chunk = 10 chunks
    assert len(chunks) == 10
    assert all(len(c) == 10 for c in chunks)


@pytest.mark.asyncio
async def test_streaming_reset():
    """Test streaming executor reset."""
    executor = StreamingExecutor()
    executor._chunk_counter = 42

    executor.reset()

    assert executor._chunk_counter == 0
    assert executor._stream_buffer.empty()


@pytest.mark.asyncio
async def test_stream_config_validation():
    """Test stream config validation."""
    config = StreamConfig(
        buffer_size=50,
        chunk_timeout_ms=2000,
        enable_partial_results=False,
    )

    assert config.buffer_size == 50
    assert config.chunk_timeout_ms == 2000
    assert config.enable_partial_results is False


@pytest.mark.asyncio
async def test_streaming_partial_results_disabled():
    """Test streaming with partial results disabled."""
    config = StreamConfig(enable_partial_results=False)
    executor = StreamingExecutor(config)

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    # Result with empty content
    async def partial_generator():
        yield NodeResult(
            success=True,
            output_messages=[
                Message(
                    trace_id="test",
                    source_node="node1",
                    target_node="node2",
                    payload=MessagePayload(content=""),  # Empty
                )
            ],
        )
        yield NodeResult(
            success=True,
            output_messages=[
                Message(
                    trace_id="test",
                    source_node="node1",
                    target_node="node2",
                    payload=MessagePayload(content="Real content"),
                )
            ],
        )

    chunks = []
    async for chunk in executor.stream_execution(partial_generator(), context):
        chunks.append(chunk)

    # Should skip empty content, only have real content + final
    assert len(chunks) == 2
    assert chunks[0].content == "Real content"
    assert chunks[1].is_final is True


@pytest.mark.asyncio
async def test_streaming_backpressure():
    """Test that buffering applies backpressure."""
    config = StreamConfig(buffer_size=2)  # Very small buffer
    executor = StreamingExecutor(config)

    context = ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )

    # Create many results to test backpressure
    async def many_results():
        for i in range(10):
            yield NodeResult(
                success=True,
                output_messages=[
                    Message(
                        trace_id="test",
                        source_node=f"node-{i}",
                        target_node="next",
                        payload=MessagePayload(content=f"Result {i}"),
                    )
                ],
            )
            await asyncio.sleep(0.001)

    chunks = []
    # Slow consumer to trigger backpressure
    async for chunk in executor.stream_with_buffering(many_results(), context):
        chunks.append(chunk)
        await asyncio.sleep(0.01)  # Slow consumption

    # Should still get all chunks despite backpressure
    assert len(chunks) >= 10
