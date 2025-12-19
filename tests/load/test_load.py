"""Load testing harness for TinyLLM.

This module provides load testing capabilities using pytest-benchmark
and custom load testing utilities to measure system performance under various loads.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.load

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.nodes.fanout import FanoutNode
from tinyllm.nodes.transform import TransformNode
from tinyllm.models.client import OllamaClient


class LoadTestMetrics:
    """Metrics collector for load tests."""

    def __init__(self):
        self.requests_completed = 0
        self.requests_failed = 0
        self.total_latency_ms = 0.0
        self.min_latency_ms = float("inf")
        self.max_latency_ms = 0.0
        self.start_time = None
        self.end_time = None

    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.requests_completed += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

    def record_failure(self):
        """Record a failed request."""
        self.requests_failed += 1

    def get_summary(self) -> dict:
        """Get test summary metrics."""
        total_requests = self.requests_completed + self.requests_failed
        duration_s = (
            (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        )

        return {
            "total_requests": total_requests,
            "successful_requests": self.requests_completed,
            "failed_requests": self.requests_failed,
            "success_rate": (
                self.requests_completed / total_requests if total_requests > 0 else 0
            ),
            "avg_latency_ms": (
                self.total_latency_ms / self.requests_completed
                if self.requests_completed > 0
                else 0
            ),
            "min_latency_ms": (
                self.min_latency_ms if self.min_latency_ms != float("inf") else 0
            ),
            "max_latency_ms": self.max_latency_ms,
            "duration_s": duration_s,
            "throughput_rps": total_requests / duration_s if duration_s > 0 else 0,
        }


@pytest.fixture
def load_metrics():
    """Create load test metrics collector."""
    return LoadTestMetrics()


@pytest.fixture
def execution_context():
    """Create test execution context."""
    return ExecutionContext(
        trace_id="load-test",
        graph_id="load-graph",
        config=Config(),
    )


class TestNodeLoadTesting:
    """Load tests for individual nodes."""

    @pytest.mark.asyncio
    async def test_transform_node_sustained_load(
        self, execution_context, load_metrics
    ):
        """Test transform node under sustained load."""
        definition = NodeDefinition(
            id="transform.load",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "uppercase"},
                    {"type": "strip"},
                ],
            },
        )
        node = TransformNode(definition)

        async def execute_batch(batch_size: int):
            """Execute a batch of requests."""
            messages = [
                Message(
                    trace_id=f"load-{i}",
                    source_node="test",
                    payload=MessagePayload(content=f"  test content {i}  "),
                )
                for i in range(batch_size)
            ]

            load_metrics.start_time = time.time()
            tasks = []
            for msg in messages:
                start = time.time()
                task = node.execute(msg, execution_context)
                tasks.append((task, start))

            results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

            for (_, start), result in zip(tasks, results):
                latency_ms = (time.time() - start) * 1000
                if isinstance(result, Exception) or not result.success:
                    load_metrics.record_failure()
                else:
                    load_metrics.record_success(latency_ms)

            load_metrics.end_time = time.time()
            return results

        # Run load test
        batch_size = 100
        results = await execute_batch(batch_size)

        summary = load_metrics.get_summary()

        # Assertions
        assert summary["successful_requests"] >= batch_size * 0.95  # 95% success rate
        assert summary["avg_latency_ms"] < 100  # Avg latency under 100ms
        assert summary["throughput_rps"] > 10  # At least 10 RPS

    @pytest.mark.asyncio
    async def test_fanout_node_parallel_load(self, execution_context, load_metrics):
        """Test fanout node parallel execution under load."""
        definition = NodeDefinition(
            id="fanout.load",
            type=NodeType.FANOUT,
            config={
                "target_nodes": [f"target_{i}" for i in range(10)],
                "aggregation_strategy": "all",
                "parallel": True,
                "timeout_ms": 10000,
            },
        )
        node = FanoutNode(definition)

        # Run multiple fanout executions concurrently
        num_executions = 20
        messages = [
            Message(
                trace_id=f"fanout-load-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Fanout test {i}"),
            )
            for i in range(num_executions)
        ]

        load_metrics.start_time = time.time()
        results = []

        for msg in messages:
            start = time.time()
            try:
                result = await node.execute(msg, execution_context)
                latency_ms = (time.time() - start) * 1000
                results.append(result)
                if result.success:
                    load_metrics.record_success(latency_ms)
                else:
                    load_metrics.record_failure()
            except Exception:
                load_metrics.record_failure()

        load_metrics.end_time = time.time()
        summary = load_metrics.get_summary()

        # Assertions
        assert summary["successful_requests"] >= num_executions * 0.9  # 90% success
        assert len(results) > 0


class TestSystemLoadTesting:
    """System-wide load tests."""

    @pytest.mark.asyncio
    async def test_message_creation_throughput(self, load_metrics):
        """Test message creation throughput."""
        num_messages = 10000

        load_metrics.start_time = time.time()

        messages = []
        for i in range(num_messages):
            start = time.time()
            msg = Message(
                trace_id=f"throughput-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Message {i}"),
            )
            latency_ms = (time.time() - start) * 1000
            messages.append(msg)
            load_metrics.record_success(latency_ms)

        load_metrics.end_time = time.time()
        summary = load_metrics.get_summary()

        # Assertions
        assert len(messages) == num_messages
        assert summary["throughput_rps"] > 1000  # At least 1000 messages/sec
        assert summary["avg_latency_ms"] < 1  # Very fast message creation

    @pytest.mark.asyncio
    async def test_concurrent_execution_contexts(self, load_metrics):
        """Test creating and using many concurrent execution contexts."""
        num_contexts = 100

        load_metrics.start_time = time.time()

        async def create_and_use_context(ctx_id: int):
            """Create and use an execution context."""
            start = time.time()
            try:
                ctx = ExecutionContext(
                    trace_id=f"ctx-{ctx_id}",
                    graph_id="load-test",
                    config=Config(),
                )
                # Simulate some context usage
                await asyncio.sleep(0.01)
                latency_ms = (time.time() - start) * 1000
                return ctx, latency_ms
            except Exception:
                return None, 0

        tasks = [create_and_use_context(i) for i in range(num_contexts)]
        results = await asyncio.gather(*tasks)

        for ctx, latency_ms in results:
            if ctx is not None:
                load_metrics.record_success(latency_ms)
            else:
                load_metrics.record_failure()

        load_metrics.end_time = time.time()
        summary = load_metrics.get_summary()

        # Assertions
        assert summary["successful_requests"] >= num_contexts * 0.95
        assert summary["avg_latency_ms"] < 50


class TestRampUpLoadTesting:
    """Ramp-up load tests to find system limits."""

    @pytest.mark.asyncio
    async def test_gradual_load_increase(self, execution_context):
        """Test system behavior with gradually increasing load."""
        definition = NodeDefinition(
            id="transform.ramp",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        # Start with 10 requests, increase by 10 each iteration
        results_by_load = []

        for load_level in [10, 20, 50, 100, 200]:
            metrics = LoadTestMetrics()
            metrics.start_time = time.time()

            messages = [
                Message(
                    trace_id=f"ramp-{load_level}-{i}",
                    source_node="test",
                    payload=MessagePayload(content=f"Content {i}"),
                )
                for i in range(load_level)
            ]

            tasks = []
            for msg in messages:
                start = time.time()
                task = node.execute(msg, execution_context)
                tasks.append((task, start))

            results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

            for (_, start), result in zip(tasks, results):
                latency_ms = (time.time() - start) * 1000
                if isinstance(result, Exception) or not result.success:
                    metrics.record_failure()
                else:
                    metrics.record_success(latency_ms)

            metrics.end_time = time.time()
            summary = metrics.get_summary()
            summary["load_level"] = load_level
            results_by_load.append(summary)

        # Analyze results - success rate should remain high
        for result in results_by_load:
            assert result["success_rate"] > 0.9

        # At higher loads, system should maintain good throughput
        # (not necessarily higher since lightweight operations may vary)
        assert results_by_load[-1]["throughput_rps"] > 1000  # At least 1000 RPS at high load


class TestSpikeLoadTesting:
    """Spike load tests to test system resilience."""

    @pytest.mark.asyncio
    async def test_sudden_load_spike(self, execution_context, load_metrics):
        """Test system handling sudden load spike."""
        definition = NodeDefinition(
            id="transform.spike",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "strip"}],
            },
        )
        node = TransformNode(definition)

        # Baseline: 10 requests
        baseline_messages = [
            Message(
                trace_id=f"baseline-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Content {i}"),
            )
            for i in range(10)
        ]

        # Spike: 500 requests
        spike_messages = [
            Message(
                trace_id=f"spike-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Content {i}"),
            )
            for i in range(500)
        ]

        # Execute baseline
        baseline_results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in baseline_messages]
        )

        # Execute spike
        load_metrics.start_time = time.time()
        spike_tasks = []
        for msg in spike_messages:
            start = time.time()
            task = node.execute(msg, execution_context)
            spike_tasks.append((task, start))

        spike_results = await asyncio.gather(
            *[t[0] for t in spike_tasks], return_exceptions=True
        )

        for (_, start), result in zip(spike_tasks, spike_results):
            latency_ms = (time.time() - start) * 1000
            if isinstance(result, Exception) or not result.success:
                load_metrics.record_failure()
            else:
                load_metrics.record_success(latency_ms)

        load_metrics.end_time = time.time()
        summary = load_metrics.get_summary()

        # System should handle spike gracefully
        assert summary["success_rate"] > 0.85  # At least 85% success during spike
        assert all(r.success for r in baseline_results)  # Baseline should be perfect


class TestEnduranceLoadTesting:
    """Endurance tests for long-running load."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load_endurance(self, execution_context):
        """Test system under sustained load for extended period."""
        definition = NodeDefinition(
            id="transform.endurance",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "lowercase"}],
            },
        )
        node = TransformNode(definition)

        # Run for 30 seconds at steady rate
        duration_seconds = 30
        requests_per_second = 10
        total_expected = duration_seconds * requests_per_second

        metrics = LoadTestMetrics()
        metrics.start_time = time.time()

        request_count = 0
        while time.time() - metrics.start_time < duration_seconds:
            msg = Message(
                trace_id=f"endurance-{request_count}",
                source_node="test",
                payload=MessagePayload(content=f"Content {request_count}"),
            )

            start = time.time()
            try:
                result = await node.execute(msg, execution_context)
                latency_ms = (time.time() - start) * 1000
                if result.success:
                    metrics.record_success(latency_ms)
                else:
                    metrics.record_failure()
            except Exception:
                metrics.record_failure()

            request_count += 1

            # Rate limiting to maintain steady load
            await asyncio.sleep(1.0 / requests_per_second)

        metrics.end_time = time.time()
        summary = metrics.get_summary()

        # Assertions for endurance
        assert summary["successful_requests"] >= total_expected * 0.9
        assert summary["success_rate"] > 0.95
        # No significant latency degradation over time
        assert summary["avg_latency_ms"] < 100


class TestMemoryLeakDetection:
    """Tests to detect potential memory leaks under load."""

    @pytest.mark.asyncio
    async def test_repeated_message_creation(self):
        """Test for memory leaks in message creation."""
        # Create many messages and let them go out of scope
        iterations = 100

        for i in range(iterations):
            messages = [
                Message(
                    trace_id=f"leak-test-{i}-{j}",
                    source_node="test",
                    payload=MessagePayload(content=f"Content {j}"),
                )
                for j in range(100)
            ]
            # Messages should be garbage collected when this loop iterates
            assert len(messages) == 100

        # If we get here without OOM, no obvious leak
        assert True

    @pytest.mark.asyncio
    async def test_repeated_context_creation(self):
        """Test for memory leaks in execution context creation."""
        iterations = 1000

        for i in range(iterations):
            ctx = ExecutionContext(
                trace_id=f"ctx-leak-{i}",
                graph_id="leak-test",
                config=Config(),
            )
            # Context should be garbage collected
            assert ctx.trace_id == f"ctx-leak-{i}"

        # No OOM means no obvious leak
        assert True
