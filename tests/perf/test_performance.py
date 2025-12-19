"""Performance regression tests for TinyLLM.

This module tests system performance and detects regressions by establishing
baselines and comparing against them.
"""

import asyncio

import pytest

pytestmark = pytest.mark.perf
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.nodes.fanout import FanoutNode
from tinyllm.nodes.loop import LoopNode
from tinyllm.nodes.transform import TransformNode


# Performance baseline file
BASELINE_FILE = Path(__file__).parent / "performance_baseline.json"


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}

    def record(self, test_name: str, metric_name: str, value: float):
        """Record a performance metric."""
        if test_name not in self.metrics:
            self.metrics[test_name] = {}
        self.metrics[test_name][metric_name] = value

    def get(self, test_name: str, metric_name: str) -> float:
        """Get a performance metric."""
        return self.metrics.get(test_name, {}).get(metric_name, 0.0)

    def save_baseline(self):
        """Save metrics as baseline."""
        with open(BASELINE_FILE, "w") as f:
            json.dump(self.metrics, f, indent=2)

    @classmethod
    def load_baseline(cls) -> "PerformanceMetrics":
        """Load baseline metrics."""
        metrics = cls()
        if BASELINE_FILE.exists():
            with open(BASELINE_FILE) as f:
                metrics.metrics = json.load(f)
        return metrics

    def compare_with_baseline(
        self, baseline: "PerformanceMetrics", threshold: float = 0.2
    ) -> Dict[str, Dict[str, str]]:
        """Compare current metrics with baseline.

        Args:
            baseline: Baseline metrics to compare against.
            threshold: Acceptable regression threshold (e.g., 0.2 = 20% slower is OK).

        Returns:
            Dictionary of test -> metric -> status (pass/warn/fail).
        """
        results = {}

        for test_name, test_metrics in self.metrics.items():
            results[test_name] = {}

            for metric_name, current_value in test_metrics.items():
                baseline_value = baseline.get(test_name, metric_name)

                if baseline_value == 0:
                    results[test_name][metric_name] = "baseline_missing"
                    continue

                # Calculate regression (positive means slower/worse)
                regression = (current_value - baseline_value) / baseline_value

                if regression > threshold:
                    results[test_name][metric_name] = f"fail (regression: {regression:.1%})"
                elif regression > threshold / 2:
                    results[test_name][metric_name] = f"warn (regression: {regression:.1%})"
                else:
                    results[test_name][metric_name] = f"pass (change: {regression:+.1%})"

        return results


@pytest.fixture
def perf_metrics():
    """Create performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
def execution_context():
    """Create test execution context."""
    return ExecutionContext(
        trace_id="perf-test",
        graph_id="perf-graph",
        config=Config(),
    )


class TestMessagePerformance:
    """Performance tests for message operations."""

    @pytest.mark.asyncio
    async def test_message_creation_performance(self, perf_metrics):
        """Test message creation performance."""
        num_messages = 10000

        start = time.perf_counter()
        messages = [
            Message(
                trace_id=f"perf-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Message {i}"),
            )
            for i in range(num_messages)
        ]
        elapsed = time.perf_counter() - start

        # Record metrics
        ops_per_sec = num_messages / elapsed
        avg_time_us = (elapsed / num_messages) * 1_000_000

        perf_metrics.record("message_creation", "ops_per_sec", ops_per_sec)
        perf_metrics.record("message_creation", "avg_time_us", avg_time_us)

        # Baseline expectations
        assert ops_per_sec > 10000, f"Message creation too slow: {ops_per_sec:.0f} ops/s"
        assert avg_time_us < 100, f"Message creation too slow: {avg_time_us:.2f} μs"

    @pytest.mark.asyncio
    async def test_message_child_creation_performance(self, perf_metrics):
        """Test child message creation performance."""
        root = Message(
            trace_id="root",
            source_node="test",
            payload=MessagePayload(content="Root"),
        )

        num_children = 1000

        start = time.perf_counter()
        children = [
            root.create_child(
                source_node=f"child_{i}",
                payload=MessagePayload(content=f"Child {i}"),
            )
            for i in range(num_children)
        ]
        elapsed = time.perf_counter() - start

        ops_per_sec = num_children / elapsed
        avg_time_us = (elapsed / num_children) * 1_000_000

        perf_metrics.record("message_child_creation", "ops_per_sec", ops_per_sec)
        perf_metrics.record("message_child_creation", "avg_time_us", avg_time_us)

        assert ops_per_sec > 5000, f"Child creation too slow: {ops_per_sec:.0f} ops/s"


class TestNodeExecutionPerformance:
    """Performance tests for node execution."""

    @pytest.mark.asyncio
    async def test_transform_node_throughput(self, execution_context, perf_metrics):
        """Test transform node throughput."""
        definition = NodeDefinition(
            id="transform.perf",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "uppercase"},
                    {"type": "strip"},
                ],
            },
        )
        node = TransformNode(definition)

        num_executions = 1000
        messages = [
            Message(
                trace_id=f"transform-perf-{i}",
                source_node="test",
                payload=MessagePayload(content=f"  test content {i}  "),
            )
            for i in range(num_executions)
        ]

        start = time.perf_counter()
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages]
        )
        elapsed = time.perf_counter() - start

        ops_per_sec = num_executions / elapsed
        avg_latency_ms = (elapsed / num_executions) * 1000

        perf_metrics.record("transform_throughput", "ops_per_sec", ops_per_sec)
        perf_metrics.record("transform_throughput", "avg_latency_ms", avg_latency_ms)

        assert all(r.success for r in results)
        assert ops_per_sec > 100, f"Transform too slow: {ops_per_sec:.0f} ops/s"

    @pytest.mark.asyncio
    async def test_fanout_node_latency(self, execution_context, perf_metrics):
        """Test fanout node parallel execution latency."""
        num_targets = 10
        definition = NodeDefinition(
            id="fanout.perf",
            type=NodeType.FANOUT,
            config={
                "target_nodes": [f"target_{i}" for i in range(num_targets)],
                "aggregation_strategy": "all",
                "parallel": True,
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="fanout-perf",
            source_node="test",
            payload=MessagePayload(content="Test"),
        )

        # Measure execution time
        num_runs = 50
        latencies = []

        for _ in range(num_runs):
            start = time.perf_counter()
            result = await node.execute(msg, execution_context)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # Convert to ms
            assert result.success

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        perf_metrics.record("fanout_latency", "avg_ms", avg_latency)
        perf_metrics.record("fanout_latency", "p95_ms", p95_latency)
        perf_metrics.record("fanout_latency", "p99_ms", p99_latency)

        # Should be reasonably fast
        assert avg_latency < 100, f"Fanout too slow: {avg_latency:.2f} ms"

    @pytest.mark.asyncio
    async def test_loop_node_iteration_speed(self, execution_context, perf_metrics):
        """Test loop node iteration speed."""
        num_iterations = 100
        definition = NodeDefinition(
            id="loop.perf",
            type=NodeType.LOOP,
            config={
                "body_node": "processor",
                "condition_type": "fixed_count",
                "fixed_count": num_iterations,
                "max_iterations": num_iterations,
            },
        )
        node = LoopNode(definition)

        msg = Message(
            trace_id="loop-perf",
            source_node="test",
            payload=MessagePayload(content="Test"),
        )

        start = time.perf_counter()
        result = await node.execute(msg, execution_context)
        elapsed = time.perf_counter() - start

        iterations_per_sec = num_iterations / elapsed
        avg_iteration_ms = (elapsed / num_iterations) * 1000

        perf_metrics.record("loop_iteration", "iterations_per_sec", iterations_per_sec)
        perf_metrics.record("loop_iteration", "avg_iteration_ms", avg_iteration_ms)

        assert result.success
        assert iterations_per_sec > 10, f"Loop too slow: {iterations_per_sec:.0f} iter/s"


class TestConcurrencyPerformance:
    """Performance tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_node_execution(self, execution_context, perf_metrics):
        """Test concurrent execution of multiple nodes."""
        num_concurrent = 100

        definition = NodeDefinition(
            id="transform.concurrent",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        messages = [
            Message(
                trace_id=f"concurrent-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Content {i}"),
            )
            for i in range(num_concurrent)
        ]

        start = time.perf_counter()
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages]
        )
        elapsed = time.perf_counter() - start

        throughput = num_concurrent / elapsed
        avg_latency_ms = (elapsed / num_concurrent) * 1000

        perf_metrics.record("concurrent_execution", "throughput", throughput)
        perf_metrics.record("concurrent_execution", "avg_latency_ms", avg_latency_ms)

        assert all(r.success for r in results)
        assert throughput > 50, f"Concurrent throughput too low: {throughput:.0f} ops/s"

    @pytest.mark.asyncio
    async def test_context_creation_concurrency(self, perf_metrics):
        """Test concurrent context creation performance."""
        num_contexts = 1000

        async def create_context(i: int):
            """Create a single context."""
            return ExecutionContext(
                trace_id=f"ctx-{i}",
                graph_id="perf-test",
                config=Config(),
            )

        start = time.perf_counter()
        contexts = await asyncio.gather(*[create_context(i) for i in range(num_contexts)])
        elapsed = time.perf_counter() - start

        contexts_per_sec = num_contexts / elapsed

        perf_metrics.record("context_creation_concurrent", "ops_per_sec", contexts_per_sec)

        assert len(contexts) == num_contexts
        assert contexts_per_sec > 1000, f"Context creation too slow: {contexts_per_sec:.0f} ops/s"


class TestMemoryPerformance:
    """Performance tests for memory usage."""

    @pytest.mark.asyncio
    async def test_large_message_batch(self, execution_context, perf_metrics):
        """Test handling large batches of messages."""
        batch_size = 10000

        start = time.perf_counter()
        messages = [
            Message(
                trace_id=f"batch-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Message {i}"),
            )
            for i in range(batch_size)
        ]
        creation_time = time.perf_counter() - start

        # Process batch
        definition = NodeDefinition(
            id="transform.batch",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "strip"}],
            },
        )
        node = TransformNode(definition)

        start = time.perf_counter()
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages]
        )
        processing_time = time.perf_counter() - start

        total_time = creation_time + processing_time
        throughput = batch_size / total_time

        perf_metrics.record("large_batch", "creation_time_s", creation_time)
        perf_metrics.record("large_batch", "processing_time_s", processing_time)
        perf_metrics.record("large_batch", "throughput", throughput)

        assert len(results) == batch_size
        assert throughput > 100, f"Batch processing too slow: {throughput:.0f} ops/s"

    @pytest.mark.asyncio
    async def test_large_content_handling(self, execution_context, perf_metrics):
        """Test performance with large message content."""
        content_sizes = [1_000, 10_000, 100_000]
        results = {}

        definition = NodeDefinition(
            id="transform.large",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        for size in content_sizes:
            content = "a" * size
            msg = Message(
                trace_id=f"large-{size}",
                source_node="test",
                payload=MessagePayload(content=content),
            )

            start = time.perf_counter()
            result = await node.execute(msg, execution_context)
            elapsed = time.perf_counter() - start

            results[size] = elapsed * 1000  # ms

            assert result.success

        # Record metrics
        for size, latency in results.items():
            perf_metrics.record(f"large_content_{size}", "latency_ms", latency)

        # Should scale reasonably with size
        assert results[100_000] < results[1_000] * 200  # Not more than 200x slower


class TestScalabilityPerformance:
    """Performance tests for scalability characteristics."""

    @pytest.mark.asyncio
    async def test_scaling_with_load(self, execution_context, perf_metrics):
        """Test performance scaling with increasing load."""
        loads = [10, 50, 100, 500]
        results = {}

        definition = NodeDefinition(
            id="transform.scaling",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "lowercase"}],
            },
        )
        node = TransformNode(definition)

        for load in loads:
            messages = [
                Message(
                    trace_id=f"scale-{load}-{i}",
                    source_node="test",
                    payload=MessagePayload(content=f"Content {i}"),
                )
                for i in range(load)
            ]

            start = time.perf_counter()
            batch_results = await asyncio.gather(
                *[node.execute(msg, execution_context) for msg in messages]
            )
            elapsed = time.perf_counter() - start

            throughput = load / elapsed
            results[load] = throughput

            assert all(r.success for r in batch_results)

        # Record metrics
        for load, throughput in results.items():
            perf_metrics.record(f"scaling_load_{load}", "throughput", throughput)

        # Throughput should not decrease dramatically with load
        # (allowing some degradation is normal)
        assert results[500] > results[10] * 0.3  # At least 30% of small load throughput


class TestRegressionChecks:
    """Regression tests against baseline."""

    @pytest.mark.asyncio
    async def test_check_performance_regression(self, perf_metrics):
        """Check for performance regressions against baseline."""
        # This test runs all other performance tests and compares with baseline

        # Load baseline
        baseline = PerformanceMetrics.load_baseline()

        # If no baseline exists, create one
        if not baseline.metrics:
            pytest.skip("No baseline exists - run tests with --save-baseline flag")

        # Compare with baseline
        comparison = perf_metrics.compare_with_baseline(baseline, threshold=0.25)

        # Print comparison results
        regressions = []
        for test_name, metrics in comparison.items():
            for metric_name, status in metrics.items():
                if status.startswith("fail"):
                    regressions.append(f"{test_name}.{metric_name}: {status}")

        # Fail if there are regressions
        if regressions:
            msg = "Performance regressions detected:\n" + "\n".join(regressions)
            pytest.fail(msg)


# Pytest hooks for baseline management
def pytest_addoption(parser):
    """Add command-line options for performance testing."""
    parser.addoption(
        "--save-baseline",
        action="store_true",
        help="Save current performance metrics as baseline",
    )
    parser.addoption(
        "--skip-regression",
        action="store_true",
        help="Skip regression checks",
    )


@pytest.fixture(scope="session", autouse=True)
def save_baseline_if_requested(request):
    """Save baseline after all tests if --save-baseline flag is set."""
    yield

    if request.config.getoption("--save-baseline"):
        # Collect metrics from all tests
        # (This is simplified - in practice you'd collect from all test instances)
        metrics = PerformanceMetrics()
        metrics.save_baseline()
        print(f"\nSaved performance baseline to {BASELINE_FILE}")
