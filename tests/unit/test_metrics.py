"""Tests for the Prometheus metrics module."""

import time
from threading import Thread
from typing import Generator

import pytest
from prometheus_client import REGISTRY

from tinyllm.metrics import (
    MetricsCollector,
    get_metrics_collector,
    start_metrics_server,
)


@pytest.fixture
def metrics_collector() -> Generator[MetricsCollector, None, None]:
    """Provide a fresh metrics collector for each test."""
    # Note: MetricsCollector is a singleton, so we get the same instance
    collector = get_metrics_collector()
    yield collector
    # Metrics persist across tests due to singleton pattern


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    def test_singleton_pattern(self) -> None:
        """Test that MetricsCollector follows singleton pattern."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        collector3 = get_metrics_collector()

        assert collector1 is collector2
        assert collector2 is collector3

    def test_increment_request_count(self, metrics_collector: MetricsCollector) -> None:
        """Test request counter increments."""
        # Get initial value
        before = sum(
            sample.value
            for metric in REGISTRY.collect()
            if metric.name == "tinyllm_requests_total"
            for sample in metric.samples
            if sample.labels.get("model") == "test_model"
        )

        # Increment counter
        metrics_collector.increment_request_count(
            model="test_model", graph="test_graph", request_type="generate"
        )

        # Check it increased
        after = sum(
            sample.value
            for metric in REGISTRY.collect()
            if metric.name == "tinyllm_requests_total"
            for sample in metric.samples
            if sample.labels.get("model") == "test_model"
        )

        assert after > before

    def test_track_request_latency(self, metrics_collector: MetricsCollector) -> None:
        """Test request latency tracking."""
        # Track a request with some delay
        with metrics_collector.track_request_latency(
            model="test_model", graph="test_graph"
        ):
            time.sleep(0.01)  # 10ms delay

        # Check that latency was recorded
        latency_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_request_latency_seconds":
                for sample in metric.samples:
                    if (
                        sample.labels.get("model") == "test_model"
                        and "_count" in sample.name
                        and sample.value > 0
                    ):
                        latency_recorded = True
                        break

        assert latency_recorded

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
                    if sample.labels.get("model") == "gauge_test":
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

        # Check that tokens were recorded
        input_recorded = False
        output_recorded = False

        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_tokens_input_total":
                for sample in metric.samples:
                    if (
                        sample.labels.get("model") == "token_test"
                        and sample.value >= 100
                    ):
                        input_recorded = True
            elif metric.name == "tinyllm_tokens_output_total":
                for sample in metric.samples:
                    if (
                        sample.labels.get("model") == "token_test"
                        and sample.value >= 50
                    ):
                        output_recorded = True

        assert input_recorded and output_recorded

    def test_increment_error_count(self, metrics_collector: MetricsCollector) -> None:
        """Test error counter."""
        metrics_collector.increment_error_count(
            error_type="timeout", model="error_test", graph="test_graph"
        )

        # Check error was recorded
        error_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_errors_total":
                for sample in metric.samples:
                    if (
                        sample.labels.get("error_type") == "timeout"
                        and sample.labels.get("model") == "error_test"
                    ):
                        error_recorded = True
                        break

        assert error_recorded

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
                        if sample.labels.get("model") == "breaker_test":
                            current_value = sample.value
                            break

            assert current_value == expected

    def test_circuit_breaker_failures(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test circuit breaker failure counter."""
        before = 0
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_circuit_breaker_failures_total":
                for sample in metric.samples:
                    if sample.labels.get("model") == "failure_test":
                        before = sample.value

        metrics_collector.increment_circuit_breaker_failures(model="failure_test")

        after = 0
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_circuit_breaker_failures_total":
                for sample in metric.samples:
                    if sample.labels.get("model") == "failure_test":
                        after = sample.value

        assert after > before

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
                        sample.labels.get("model") == "load_test"
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
            if metric.name == "tinyllm_node_executions_total":
                for sample in metric.samples:
                    if sample.labels.get("node") == "test_node":
                        exec_recorded = True
                        break

        assert exec_recorded

    def test_node_error_count(self, metrics_collector: MetricsCollector) -> None:
        """Test node error counting."""
        metrics_collector.increment_node_error_count(
            node="error_node", error_type="validation", graph="test"
        )

        error_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_node_errors_total":
                for sample in metric.samples:
                    if (
                        sample.labels.get("node") == "error_node"
                        and sample.labels.get("error_type") == "validation"
                    ):
                        error_recorded = True
                        break

        assert error_recorded

    def test_graph_execution_tracking(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test graph execution tracking."""
        with metrics_collector.track_graph_execution(graph="test_graph"):
            time.sleep(0.01)

        exec_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_graph_executions_total":
                for sample in metric.samples:
                    if sample.labels.get("graph") == "test_graph":
                        exec_recorded = True
                        break

        assert exec_recorded

    def test_cache_metrics(self, metrics_collector: MetricsCollector) -> None:
        """Test cache hit/miss tracking."""
        metrics_collector.increment_cache_hit(cache_type="memory")
        metrics_collector.increment_cache_miss(cache_type="memory")

        hits_recorded = False
        misses_recorded = False

        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_cache_hits_total":
                for sample in metric.samples:
                    if sample.labels.get("cache_type") == "memory":
                        hits_recorded = True
            elif metric.name == "tinyllm_cache_misses_total":
                for sample in metric.samples:
                    if sample.labels.get("cache_type") == "memory":
                        misses_recorded = True

        assert hits_recorded and misses_recorded

    def test_rate_limit_wait(self, metrics_collector: MetricsCollector) -> None:
        """Test rate limit wait time recording."""
        metrics_collector.record_rate_limit_wait(wait_time=0.05, model="wait_test")

        wait_recorded = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_rate_limit_wait_seconds":
                for sample in metric.samples:
                    if (
                        sample.labels.get("model") == "wait_test"
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
            if metric.name == "tinyllm_memory_operations_total":
                for sample in metric.samples:
                    if sample.labels.get("operation_type") in ["add", "get"]:
                        ops_recorded = True
                        break

        assert ops_recorded

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
            assert "tinyllm_requests_total" in content
            assert "tinyllm_request_latency_seconds" in content
            assert "tinyllm_tokens_input_total" in content
            assert "tinyllm_tokens_output_total" in content
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
            if metric.name == "tinyllm_graph_executions_total":
                for sample in metric.samples:
                    if sample.labels.get("graph") == "outer":
                        graph_recorded = True
            elif metric.name == "tinyllm_node_executions_total":
                for sample in metric.samples:
                    if sample.labels.get("node") == "inner":
                        node_recorded = True

        assert graph_recorded and node_recorded

    def test_metrics_labels(self, metrics_collector: MetricsCollector) -> None:
        """Test that metric labels are properly applied."""
        metrics_collector.increment_request_count(
            model="qwen2.5:0.5b", graph="multi_domain", request_type="generate"
        )

        found_metric = False
        for metric in REGISTRY.collect():
            if metric.name == "tinyllm_requests_total":
                for sample in metric.samples:
                    labels = sample.labels
                    if (
                        labels.get("model") == "qwen2.5:0.5b"
                        and labels.get("graph") == "multi_domain"
                        and labels.get("request_type") == "generate"
                    ):
                        found_metric = True
                        break

        assert found_metric
