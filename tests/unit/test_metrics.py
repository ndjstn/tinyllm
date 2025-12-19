"""Tests for the Prometheus metrics module."""

import time
from threading import Thread
from typing import Any, Dict, Generator

import pytest
from prometheus_client import REGISTRY

from tinyllm.metrics import (
    MetricsCollector,
    get_metrics_collector,
    start_metrics_server,
)


def get_label_value(labels: Dict[Any, Any], key: str) -> Any:
    """Extract label value, handling both string and enum keys.

    Args:
        labels: Labels dictionary from a metric sample.
        key: Label key to look up (as string).

    Returns:
        Label value or None if not found.
    """
    # Try direct string lookup
    if key in labels:
        return labels[key]

    # Try enum lookup - convert all keys to strings and search
    for label_key, label_value in labels.items():
        if str(label_key) == key or (hasattr(label_key, 'value') and label_key.value == key):
            return label_value

    return None


@pytest.fixture(autouse=True)
def metrics_collector(isolated_metrics_collector: MetricsCollector) -> MetricsCollector:
    """Provide a fresh metrics collector for each test.

    Uses the isolated_metrics_collector fixture from conftest to ensure
    clean state between tests.

    This fixture is autouse to ensure ALL tests in this module get
    isolated metrics, preventing state pollution between tests.
    """
    return isolated_metrics_collector


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_singleton_pattern(self, metrics_collector: MetricsCollector) -> None:
        """Test that MetricsCollector follows singleton pattern within a session."""
        # Within the same test, multiple calls return the same instance
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        collector3 = get_metrics_collector()

        assert collector1 is collector2
        assert collector2 is collector3
        # Also verify it's the same as our fixture instance
        assert collector1 is metrics_collector

    def test_increment_request_count(self, metrics_collector: MetricsCollector) -> None:
        """Test request counter increments."""
        # Increment counter
        metrics_collector.increment_request_count(
            model="test_model", graph="test_graph", request_type="generate"
        )

        # Check that metric was recorded with exact value (tests isolation)
        found = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_requests":
                for sample in metric.samples:
                    if (get_label_value(sample.labels, "model") == "test_model"
                        and "_total" in sample.name):
                        assert sample.value == 1, f"Expected 1, got {sample.value} (state pollution?)"
                        found = True
                        break

        assert found, "Metric tinyllm_requests with model=test_model not found"

    def test_track_request_latency(self, metrics_collector: MetricsCollector) -> None:
        """Test request latency tracking."""
        # Track a request with some delay
        with metrics_collector.track_request_latency(
            model="test_model", graph="test_graph"
        ):
            time.sleep(0.01)  # 10ms delay

        # Check that latency was recorded with exact count (tests isolation)
        latency_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_request_latency_seconds":
                for sample in metric.samples:
                    if (
                        get_label_value(sample.labels, "model") == "test_model"
                        and "_count" in sample.name
                    ):
                        assert sample.value == 1, f"Expected 1 request, got {sample.value} (state pollution?)"
                        latency_recorded = True
                        break

        assert latency_recorded, "Request latency not recorded"

    def test_active_requests_gauge(self, metrics_collector: MetricsCollector) -> None:
        """Test active requests gauge increments and decrements."""
        import asyncio

        async def simulate_request():
            with metrics_collector.track_request_latency(
                model="gauge_test", graph="test"
            ):
                await asyncio.sleep(0.05)

        # Run request
        asyncio.run(simulate_request())

        # After completion, active requests should be 0 for this model
        active_count = 0
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_active_requests":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "model") == "gauge_test":
                        active_count = sample.value
                        break

        assert active_count == 0  # Should be back to 0 after request completes

    def test_record_tokens(self, metrics_collector: MetricsCollector) -> None:
        """Test token counting."""
        metrics_collector.record_tokens(
            input_tokens=100,
            output_tokens=50,
            model="token_test",
            graph="test_graph",
        )

        # Check that tokens were recorded with exact values (tests isolation)
        input_recorded = False
        output_recorded = False

        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_tokens_input":
                for sample in metric.samples:
                    if (get_label_value(sample.labels, "model") == "token_test"
                        and "_total" in sample.name):
                        assert sample.value == 100, f"Expected 100 input tokens, got {sample.value}"
                        input_recorded = True
            elif metric.name == "tinyllm_tokens_output":
                for sample in metric.samples:
                    if (get_label_value(sample.labels, "model") == "token_test"
                        and "_total" in sample.name):
                        assert sample.value == 50, f"Expected 50 output tokens, got {sample.value}"
                        output_recorded = True

        assert input_recorded, "Input tokens not recorded"
        assert output_recorded, "Output tokens not recorded"

    def test_increment_error_count(self, metrics_collector: MetricsCollector) -> None:
        """Test error counter."""
        metrics_collector.increment_error_count(
            error_type="timeout", model="error_test", graph="test_graph"
        )

        # Check error was recorded
        error_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_errors":
                for sample in metric.samples:
                    if (
                        get_label_value(sample.labels, "error_type") == "timeout"
                        and get_label_value(sample.labels, "model") == "error_test"
                        and sample.value >= 1
                    ):
                        error_recorded = True
                        break

        assert error_recorded, "Error count not recorded"

    def test_circuit_breaker_state(self, metrics_collector: MetricsCollector) -> None:
        """Test circuit breaker state tracking."""
        states = ["closed", "half-open", "open"]
        expected_values = [0, 1, 2]

        for state, expected in zip(states, expected_values):
            metrics_collector.update_circuit_breaker_state(
                state=state, model="breaker_test"
            )

            # Check state was set
            current_value = None
            for metric in REGISTRY.collect():
                if metric.name == "tinyllm_circuit_breaker_state":
                    for sample in metric.samples:
                        if get_label_value(sample.labels, "model") == "breaker_test":
                            current_value = sample.value
                            break

            assert current_value == expected

    def test_circuit_breaker_failures(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test circuit breaker failure counter."""
        metrics_collector.increment_circuit_breaker_failures(model="failure_test")

        found = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_circuit_breaker_failures":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "model") == "failure_test" and sample.value >= 1:
                        found = True
                        break

        assert found, "Circuit breaker failures not recorded"

    def test_track_model_load(self, metrics_collector: MetricsCollector) -> None:
        """Test model load duration tracking."""
        with metrics_collector.track_model_load(model="load_test"):
            time.sleep(0.01)

        # Check load duration was recorded
        load_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_model_load_duration_seconds":
                for sample in metric.samples:
                    if (
                        get_label_value(sample.labels, "model") == "load_test"
                        and "_count" in sample.name
                        and sample.value > 0
                    ):
                        load_recorded = True
                        break

        assert load_recorded

    def test_node_execution_tracking(self, metrics_collector: MetricsCollector) -> None:
        """Test node execution tracking."""
        with metrics_collector.track_node_execution(node="test_node", graph="test"):
            time.sleep(0.01)

        # Check node execution was recorded
        exec_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_node_executions":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "node") == "test_node" and sample.value >= 1:
                        exec_recorded = True
                        break

        assert exec_recorded, "Node execution not recorded"

    def test_node_error_count(self, metrics_collector: MetricsCollector) -> None:
        """Test node error counting."""
        metrics_collector.increment_node_error_count(
            node="error_node", error_type="validation", graph="test"
        )

        error_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_node_errors":
                for sample in metric.samples:
                    if (
                        get_label_value(sample.labels, "node") == "error_node"
                        and get_label_value(sample.labels, "error_type") == "validation"
                        and sample.value >= 1
                    ):
                        error_recorded = True
                        break

        assert error_recorded, "Node error count not recorded"

    def test_graph_execution_tracking(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test graph execution tracking."""
        with metrics_collector.track_graph_execution(graph="test_graph"):
            time.sleep(0.01)

        exec_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_graph_executions":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "graph") == "test_graph" and sample.value >= 1:
                        exec_recorded = True
                        break

        assert exec_recorded, "Graph execution not recorded"

    def test_cache_metrics(self, metrics_collector: MetricsCollector) -> None:
        """Test cache hit/miss tracking."""
        metrics_collector.increment_cache_hit(cache_type="memory")
        metrics_collector.increment_cache_miss(cache_type="memory")

        hits_recorded = False
        misses_recorded = False

        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_cache_hits":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "cache_type") == "memory" and sample.value >= 1:
                        hits_recorded = True
            elif metric.name == "tinyllm_cache_misses":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "cache_type") == "memory" and sample.value >= 1:
                        misses_recorded = True

        assert hits_recorded, "Cache hits not recorded"
        assert misses_recorded, "Cache misses not recorded"

    def test_rate_limit_wait(self, metrics_collector: MetricsCollector) -> None:
        """Test rate limit wait time recording."""
        metrics_collector.record_rate_limit_wait(wait_time=0.05, model="wait_test")

        wait_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_rate_limit_wait_seconds":
                for sample in metric.samples:
                    if (
                        get_label_value(sample.labels, "model") == "wait_test"
                        and "_count" in sample.name
                    ):
                        wait_recorded = True
                        break

        assert wait_recorded

    def test_memory_operations(self, metrics_collector: MetricsCollector) -> None:
        """Test memory operation tracking."""
        metrics_collector.increment_memory_operation(operation_type="add")
        metrics_collector.increment_memory_operation(operation_type="get")

        ops_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_memory_operations":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "operation_type") in ["add", "get"] and sample.value >= 1:
                        ops_recorded = True
                        break

        assert ops_recorded, "Memory operations not recorded"

    def test_get_metrics_summary(self, metrics_collector: MetricsCollector) -> None:
        """Test metrics summary retrieval."""
        summary = metrics_collector.get_metrics_summary()

        assert "collector" in summary
        assert summary["collector"] == "prometheus"
        assert "registry" in summary
        assert "metrics_count" in summary
        assert summary["metrics_count"] > 0


class TestMetricsServer:
    """Test metrics server functionality."""

    def test_start_metrics_server(self) -> None:
        """Test starting metrics server on available port."""
        # Use a high port number to avoid conflicts
        port = 19090

        # Start server in thread
        server_thread = Thread(
            target=start_metrics_server, args=(port, "127.0.0.1"), daemon=True
        )
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        # Try to fetch metrics
        import urllib.request

        try:
            response = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics")
            content = response.read().decode("utf-8")

            # Check for expected metric names
            assert "tinyllm_requests" in content
            assert "tinyllm_request_latency_seconds" in content
            assert "tinyllm_tokens_input" in content
            assert "tinyllm_tokens_output" in content
            assert "tinyllm_system_info" in content

        except Exception as e:
            pytest.fail(f"Failed to fetch metrics: {e}")

    def test_server_already_running(self) -> None:
        """Test error handling when port is already in use."""
        port = 19091

        # Start first server
        server_thread1 = Thread(
            target=start_metrics_server, args=(port, "127.0.0.1"), daemon=True
        )
        server_thread1.start()
        time.sleep(0.5)

        # Try to start second server on same port - should log warning but not crash
        try:
            start_metrics_server(port=port, addr="127.0.0.1")
            # Should not raise exception, just log warning
        except OSError:
            # This is expected - port is in use
            pass


class TestMetricsIntegration:
    """Integration tests for metrics with other components."""

    def test_metrics_with_context_manager(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test that context managers work correctly."""
        # Nested context managers
        with metrics_collector.track_graph_execution(graph="outer"):
            with metrics_collector.track_node_execution(node="inner", graph="outer"):
                time.sleep(0.01)

        # Both should be recorded
        graph_recorded = False
        node_recorded = False

        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_graph_executions":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "graph") == "outer" and sample.value >= 1:
                        graph_recorded = True
            elif metric.name == "tinyllm_node_executions":
                for sample in metric.samples:
                    if get_label_value(sample.labels, "node") == "inner" and sample.value >= 1:
                        node_recorded = True

        assert graph_recorded, "Graph execution not recorded"
        assert node_recorded, "Node execution not recorded"

    def test_metrics_labels(self, metrics_collector: MetricsCollector) -> None:
        """Test that metric labels are properly applied."""
        metrics_collector.increment_request_count(
            model="qwen2.5:0.5b", graph="multi_domain", request_type="generate"
        )

        found_metric = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_requests":
                for sample in metric.samples:
                    labels = sample.labels
                    if (
                        get_label_value(labels, "model") == "qwen2.5:0.5b"
                        and get_label_value(labels, "graph") == "multi_domain"
                        and get_label_value(labels, "request_type") == "generate"
                        and sample.value >= 1
                    ):
                        found_metric = True
                        break

        assert found_metric, "Metric with correct labels not found"


class TestCardinalityControls:
    """Test cardinality controls and limits."""

    def test_cardinality_tracker_init(self, metrics_collector: MetricsCollector) -> None:
        """Test cardinality tracker initialization."""
        tracker = metrics_collector.cardinality_tracker
        assert tracker.max_cardinality == 1000
        assert len(tracker.label_sets) == 0
        assert len(tracker.dropped_counts) == 0

    def test_cardinality_check_and_add(self, metrics_collector: MetricsCollector) -> None:
        """Test adding labels within cardinality limit."""
        tracker = metrics_collector.cardinality_tracker

        # First label combination should be accepted
        assert tracker.check_and_add("test_metric", ("model1", "graph1"))
        assert tracker.get_cardinality("test_metric") == 1

        # Same combination should be accepted again
        assert tracker.check_and_add("test_metric", ("model1", "graph1"))
        assert tracker.get_cardinality("test_metric") == 1

        # Different combination should be accepted
        assert tracker.check_and_add("test_metric", ("model2", "graph2"))
        assert tracker.get_cardinality("test_metric") == 2

    def test_cardinality_limit_exceeded(self, metrics_collector: MetricsCollector) -> None:
        """Test cardinality limit enforcement."""
        from tinyllm.metrics import CardinalityTracker

        # Create tracker with low limit
        tracker = CardinalityTracker(max_cardinality=5)

        # Add up to limit
        for i in range(5):
            assert tracker.check_and_add("limited_metric", (f"label_{i}",))

        assert tracker.get_cardinality("limited_metric") == 5

        # Next addition should be rejected
        assert not tracker.check_and_add("limited_metric", ("label_6",))
        assert tracker.get_cardinality("limited_metric") == 5
        assert tracker.dropped_counts["limited_metric"] == 1

    def test_cardinality_stats(self, metrics_collector: MetricsCollector) -> None:
        """Test cardinality statistics retrieval."""
        tracker = metrics_collector.cardinality_tracker
        tracker.check_and_add("metric1", ("a", "b"))
        tracker.check_and_add("metric1", ("c", "d"))
        tracker.check_and_add("metric2", ("x", "y"))

        stats = tracker.get_stats()

        assert "metrics" in stats
        assert "total_label_combinations" in stats
        assert "max_cardinality" in stats
        assert stats["max_cardinality"] == 1000
        assert stats["total_label_combinations"] == 3
        assert "metric1" in stats["metrics"]
        assert "metric2" in stats["metrics"]
        assert stats["metrics"]["metric1"]["cardinality"] == 2
        assert stats["metrics"]["metric2"]["cardinality"] == 1

    def test_cardinality_reset(self, metrics_collector: MetricsCollector) -> None:
        """Test cardinality tracker reset."""
        tracker = metrics_collector.cardinality_tracker
        tracker.check_and_add("test", ("a",))

        assert tracker.get_cardinality("test") == 1

        tracker.reset()

        assert tracker.get_cardinality("test") == 0
        assert len(tracker.label_sets) == 0
        assert len(tracker.dropped_counts) == 0

    def test_cardinality_with_metrics(self, metrics_collector: MetricsCollector) -> None:
        """Test cardinality controls integrate with metrics collection."""
        from tinyllm.metrics import CardinalityTracker

        # Create collector with low cardinality limit
        test_collector = MetricsCollector(max_cardinality=3)

        # Add requests up to limit
        test_collector.increment_request_count(model="m1", graph="g1")
        test_collector.increment_request_count(model="m2", graph="g2")
        test_collector.increment_request_count(model="m3", graph="g3")

        # Fourth request should use fallback labels
        test_collector.increment_request_count(model="m4", graph="g4")

        # Check that fallback "other" label was used
        found_other = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_requests":
                for sample in metric.samples:
                    if (get_label_value(sample.labels, "model") == "other"
                        and get_label_value(sample.labels, "graph") == "other"):
                        found_other = True
                        break

        assert found_other, "Fallback 'other' label not found when cardinality exceeded"

    def test_get_cardinality_stats_from_collector(self, metrics_collector: MetricsCollector) -> None:
        """Test getting cardinality stats from collector."""
        metrics_collector.increment_request_count(model="test1", graph="g1")
        metrics_collector.increment_request_count(model="test2", graph="g2")

        stats = metrics_collector.get_cardinality_stats()

        assert "metrics" in stats
        assert "total_label_combinations" in stats
        assert stats["total_label_combinations"] >= 2
