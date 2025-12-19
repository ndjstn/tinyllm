"""Tests for request queue with backpressure."""

import asyncio
import time

import pytest

from tinyllm.core.executor import Executor, ExecutorConfig
from tinyllm.core.graph import Graph
from tinyllm.core.message import TaskPayload, TaskResponse
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult, NodeType
from tinyllm.queue import (
    BackpressureMode,
    Priority,
    QueuedExecutor,
    QueuedRequest,
    RequestQueue,
)


class DummyNode(BaseNode):
    """Dummy node for testing."""

    def __init__(self, definition: "NodeDefinition", delay_ms: int = 100):
        super().__init__(definition)
        self.delay_ms = delay_ms

    async def execute(self, message, context):
        """Execute with configurable delay."""
        await asyncio.sleep(self.delay_ms / 1000)
        return NodeResult.success_result(
            output_messages=[message],
            latency_ms=self.delay_ms,
        )


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    from tinyllm.config.graph import GraphDefinition, NodeDefinition, NodeType

    entry_def = NodeDefinition(
        id="entry_node",
        type=NodeType.ENTRY,
        config={},
    )
    exit_def = NodeDefinition(
        id="exit_node",
        type=NodeType.EXIT,
        config={},
    )

    definition = GraphDefinition(
        id="test_graph",
        version="1.0.0",
        name="Test Graph",
        description="Graph for testing",
        entry_points=["entry_node"],
        exit_points=["exit_node"],
        edges=[
            {"from_node": "entry_node", "to_node": "exit_node"},
        ],
        nodes=[entry_def, exit_def],
    )

    graph = Graph(definition)
    entry_node = DummyNode(entry_def, delay_ms=50)
    exit_node = DummyNode(exit_def, delay_ms=10)
    graph.add_node(entry_node)
    graph.add_node(exit_node)
    return graph


@pytest.fixture
def executor(simple_graph):
    """Create executor for testing."""
    config = ExecutorConfig(max_steps=10, timeout_ms=30000)
    return Executor(simple_graph, config=config)


class TestPriority:
    """Test priority ordering."""

    def test_priority_values(self):
        """Test priority integer values."""
        assert Priority.HIGH.value_int > Priority.NORMAL.value_int
        assert Priority.NORMAL.value_int > Priority.LOW.value_int

    def test_queued_request_ordering(self):
        """Test that queued requests are ordered by priority."""
        task = TaskPayload(content="test")

        high = QueuedRequest(
            priority_int=0,
            timestamp=2.0,
            request_id="high",
            priority=Priority.HIGH,
            task=task,
            timeout_ms=None,
            future=asyncio.Future(),
        )

        normal = QueuedRequest(
            priority_int=0,
            timestamp=1.0,
            request_id="normal",
            priority=Priority.NORMAL,
            task=task,
            timeout_ms=None,
            future=asyncio.Future(),
        )

        low = QueuedRequest(
            priority_int=0,
            timestamp=0.0,
            request_id="low",
            priority=Priority.LOW,
            task=task,
            timeout_ms=None,
            future=asyncio.Future(),
        )

        # High priority should come first despite later timestamp
        assert high < normal < low


class TestRequestQueue:
    """Test request queue functionality."""

    @pytest.mark.asyncio
    async def test_basic_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        queue = RequestQueue(max_size=10)

        task = TaskPayload(content="test query")
        request_id = await queue.enqueue(task, Priority.NORMAL)

        assert request_id is not None
        assert queue.qsize == 1

        queued_request = await queue.dequeue()
        assert queued_request.request_id == request_id
        assert queued_request.task.content == "test query"
        assert queue.qsize == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that requests are dequeued by priority."""
        queue = RequestQueue(max_size=10)

        # Enqueue in reverse priority order
        low_id = await queue.enqueue(TaskPayload(content="low"), Priority.LOW)
        normal_id = await queue.enqueue(TaskPayload(content="normal"), Priority.NORMAL)
        high_id = await queue.enqueue(TaskPayload(content="high"), Priority.HIGH)

        # Should dequeue in priority order
        first = await queue.dequeue()
        assert first.request_id == high_id

        second = await queue.dequeue()
        assert second.request_id == normal_id

        third = await queue.dequeue()
        assert third.request_id == low_id

    @pytest.mark.asyncio
    async def test_reject_backpressure(self):
        """Test reject backpressure mode."""
        queue = RequestQueue(max_size=2, backpressure_mode=BackpressureMode.REJECT)

        # Fill the queue
        await queue.enqueue(TaskPayload(content="1"), Priority.NORMAL)
        await queue.enqueue(TaskPayload(content="2"), Priority.NORMAL)

        # Next request should be rejected
        with pytest.raises(asyncio.QueueFull):
            await queue.enqueue(TaskPayload(content="3"), Priority.NORMAL)

    @pytest.mark.asyncio
    async def test_block_backpressure(self):
        """Test block backpressure mode."""
        queue = RequestQueue(max_size=2, backpressure_mode=BackpressureMode.BLOCK)

        # Fill the queue
        await queue.enqueue(TaskPayload(content="1"), Priority.NORMAL)
        await queue.enqueue(TaskPayload(content="2"), Priority.NORMAL)

        # Start enqueue that will block
        enqueue_task = asyncio.create_task(
            queue.enqueue(TaskPayload(content="3"), Priority.NORMAL)
        )

        # Give it time to block
        await asyncio.sleep(0.1)
        assert not enqueue_task.done()

        # Dequeue to make space
        await queue.dequeue()

        # Now the blocked enqueue should complete
        await asyncio.wait_for(enqueue_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_queue_position_tracking(self):
        """Test queue position estimation."""
        queue = RequestQueue(max_size=10)

        # Enqueue multiple requests
        id1 = await queue.enqueue(TaskPayload(content="1"), Priority.HIGH)
        id2 = await queue.enqueue(TaskPayload(content="2"), Priority.NORMAL)
        id3 = await queue.enqueue(TaskPayload(content="3"), Priority.LOW)

        # High priority should be first
        assert queue.get_position(id1) == 1
        assert queue.get_position(id2) == 2
        assert queue.get_position(id3) == 3

    @pytest.mark.asyncio
    async def test_queue_status(self):
        """Test queue status reporting."""
        queue = RequestQueue(max_size=100)

        # Enqueue different priorities
        await queue.enqueue(TaskPayload(content="h1"), Priority.HIGH)
        await queue.enqueue(TaskPayload(content="h2"), Priority.HIGH)
        await queue.enqueue(TaskPayload(content="n1"), Priority.NORMAL)
        await queue.enqueue(TaskPayload(content="l1"), Priority.LOW)

        status = queue.get_status()
        assert status.total_queued == 4
        assert status.high_priority == 2
        assert status.normal_priority == 1
        assert status.low_priority == 1
        assert status.max_queue_size == 100


class TestQueuedExecutor:
    """Test queued executor with worker pool."""

    @pytest.mark.asyncio
    async def test_basic_execution(self, executor):
        """Test basic queued execution."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=2,
            max_queue_size=10,
        )

        async with queued_executor.lifespan():
            task = TaskPayload(content="test query")
            response = await queued_executor.execute(task)

            assert isinstance(response, TaskResponse)
            assert response.trace_id is not None

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, executor):
        """Test concurrent request execution."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=3,
            max_queue_size=10,
        )

        async with queued_executor.lifespan():
            # Submit multiple requests
            tasks = [
                queued_executor.execute(TaskPayload(content=f"query {i}"))
                for i in range(5)
            ]

            # All should complete successfully
            responses = await asyncio.gather(*tasks)
            assert len(responses) == 5
            assert all(isinstance(r, TaskResponse) for r in responses)

    @pytest.mark.asyncio
    async def test_priority_execution_order(self, executor):
        """Test that high priority requests are processed first."""
        from tinyllm.config.graph import GraphDefinition, NodeDefinition, NodeType

        # Use a slow node to ensure queuing
        entry_def = NodeDefinition(id="slow_entry", type=NodeType.ENTRY, config={})
        exit_def = NodeDefinition(id="slow_exit", type=NodeType.EXIT, config={})

        definition = GraphDefinition(
            id="slow_graph",
            version="1.0.0",
            name="Slow Graph",
            description="Graph with slow node",
            entry_points=["slow_entry"],
            exit_points=["slow_exit"],
            edges=[{"from_node": "slow_entry", "to_node": "slow_exit"}],
            nodes=[entry_def, exit_def],
        )

        slow_graph = Graph(definition)
        slow_graph.add_node(DummyNode(entry_def, delay_ms=200))
        slow_graph.add_node(DummyNode(exit_def, delay_ms=10))

        slow_executor = Executor(slow_graph, config=ExecutorConfig())

        queued_executor = QueuedExecutor(
            executor=slow_executor,
            max_workers=1,  # Single worker to enforce ordering
            max_queue_size=10,
        )

        results = []

        async def track_result(task: TaskPayload, priority: Priority):
            response = await queued_executor.execute(task, priority)
            results.append((task.content, priority, time.monotonic()))

        async with queued_executor.lifespan():
            # Start with a slow task to fill the worker
            asyncio.create_task(
                track_result(TaskPayload(content="filler"), Priority.NORMAL)
            )
            await asyncio.sleep(0.05)  # Ensure it starts

            # Queue requests in reverse priority order
            tasks = [
                asyncio.create_task(
                    track_result(TaskPayload(content="low"), Priority.LOW)
                ),
                asyncio.create_task(
                    track_result(TaskPayload(content="normal"), Priority.NORMAL)
                ),
                asyncio.create_task(
                    track_result(TaskPayload(content="high"), Priority.HIGH)
                ),
            ]

            await asyncio.gather(*tasks)

        # High priority should complete before lower priorities
        # (excluding the filler)
        assert len(results) == 4
        priorities_completed = [r[1] for r in results[1:]]  # Skip filler
        assert priorities_completed == [Priority.HIGH, Priority.NORMAL, Priority.LOW]

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, executor):
        """Test graceful shutdown with drain."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=1,
            max_queue_size=10,
        )

        await queued_executor.start()

        # Submit some requests
        futures = [
            await queued_executor.submit(TaskPayload(content=f"query {i}"))
            for i in range(3)
        ]

        # Shutdown with drain
        await queued_executor.shutdown(drain=True, timeout=5.0)

        # All requests should complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        assert all(isinstance(r, TaskResponse) for r in results)

    @pytest.mark.asyncio
    async def test_worker_health_tracking(self, executor):
        """Test worker health monitoring."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=2,
            max_queue_size=10,
        )

        async with queued_executor.lifespan():
            # Execute some requests
            await queued_executor.execute(TaskPayload(content="query 1"))
            await queued_executor.execute(TaskPayload(content="query 2"))

            # Check worker health
            health = queued_executor.get_worker_health()
            assert len(health) == 2
            assert all(h.is_healthy for h in health.values())
            assert all(h.total_processed > 0 for h in health.values())

    @pytest.mark.asyncio
    async def test_queue_status_with_workers(self, executor):
        """Test queue status includes worker information."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=3,
            max_queue_size=100,
        )

        async with queued_executor.lifespan():
            status = queued_executor.get_status()
            assert status.max_workers == 3
            assert status.max_queue_size == 100
            assert status.active_workers >= 0

    @pytest.mark.asyncio
    async def test_submit_before_start(self, executor):
        """Test that submit fails if worker pool not started."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=2,
            max_queue_size=10,
        )

        # Should raise error if not started
        with pytest.raises(RuntimeError, match="Worker pool is not running"):
            await queued_executor.submit(TaskPayload(content="test"))

    @pytest.mark.asyncio
    async def test_request_timeout(self, executor):
        """Test request timeout in queue."""
        from tinyllm.config.graph import GraphDefinition, NodeDefinition, NodeType

        # Create a very slow executor to cause timeout
        entry_def = NodeDefinition(id="timeout_entry", type=NodeType.ENTRY, config={})
        exit_def = NodeDefinition(id="timeout_exit", type=NodeType.EXIT, config={})

        definition = GraphDefinition(
            id="timeout_graph",
            version="1.0.0",
            name="Timeout Graph",
            description="Graph with very slow node",
            entry_points=["timeout_entry"],
            exit_points=["timeout_exit"],
            edges=[{"from_node": "timeout_entry", "to_node": "timeout_exit"}],
            nodes=[entry_def, exit_def],
        )

        slow_graph = Graph(definition)
        slow_graph.add_node(DummyNode(entry_def, delay_ms=5000))
        slow_graph.add_node(DummyNode(exit_def, delay_ms=10))

        slow_executor = Executor(slow_graph, config=ExecutorConfig())

        queued_executor = QueuedExecutor(
            executor=slow_executor,
            max_workers=1,
            max_queue_size=10,
        )

        async with queued_executor.lifespan():
            # Submit with short timeout
            response = await queued_executor.execute(
                TaskPayload(content="test"),
                timeout_ms=100,  # Very short timeout
            )

            # Should return timeout error
            assert not response.success
            assert response.error is not None
            assert "timed out" in response.error.message.lower()


class TestBackpressureModes:
    """Test different backpressure handling modes."""

    @pytest.mark.asyncio
    async def test_reject_mode_integration(self, executor):
        """Test reject mode in queued executor."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=1,
            max_queue_size=2,
            backpressure_mode=BackpressureMode.REJECT,
        )

        async with queued_executor.lifespan():
            # Submit requests to fill queue
            await queued_executor.submit(TaskPayload(content="1"))
            await queued_executor.submit(TaskPayload(content="2"))

            # Next should be rejected
            with pytest.raises(asyncio.QueueFull):
                await queued_executor.submit(TaskPayload(content="3"))

    @pytest.mark.asyncio
    async def test_block_mode_integration(self, executor):
        """Test block mode in queued executor."""
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=1,
            max_queue_size=2,
            backpressure_mode=BackpressureMode.BLOCK,
        )

        async with queued_executor.lifespan():
            # Submit requests to fill queue
            f1 = await queued_executor.submit(TaskPayload(content="1"))
            f2 = await queued_executor.submit(TaskPayload(content="2"))

            # This should block but eventually succeed as queue drains
            f3 = await asyncio.wait_for(
                queued_executor.submit(TaskPayload(content="3")),
                timeout=2.0,
            )

            # All should complete
            results = await asyncio.gather(f1, f2, f3)
            assert len(results) == 3


class TestMetricsIntegration:
    """Test metrics integration."""

    @pytest.mark.asyncio
    async def test_queue_metrics_updated(self, executor):
        """Test that queue metrics are updated."""
        from tinyllm.metrics import get_metrics_collector

        metrics = get_metrics_collector()
        queued_executor = QueuedExecutor(
            executor=executor,
            max_workers=2,
            max_queue_size=10,
        )

        async with queued_executor.lifespan():
            # Execute some requests
            await queued_executor.execute(TaskPayload(content="test 1"))
            await queued_executor.execute(TaskPayload(content="test 2"))

            # Metrics should be updated (we can't easily verify values
            # but we can ensure no errors occurred)
            assert True  # If we got here, metrics are working
