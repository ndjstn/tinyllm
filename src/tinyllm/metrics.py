"""Prometheus metrics collection for TinyLLM.

Provides comprehensive metrics tracking for monitoring TinyLLM performance,
resource usage, and operational health. Exports metrics via HTTP for Prometheus scraping.
"""

import time
from contextlib import contextmanager
from enum import Enum
from threading import Lock
from typing import Any, Dict, Iterator, Optional

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="metrics")


class MetricLabels(str, Enum):
    """Standard metric label names."""

    MODEL = "model"
    GRAPH = "graph"
    NODE = "node"
    ERROR_TYPE = "error_type"
    CIRCUIT_STATE = "circuit_state"
    REQUEST_TYPE = "request_type"


class MetricsCollector:
    """Centralized metrics collector for TinyLLM.

    This class provides a singleton interface for collecting metrics across
    the TinyLLM system. All metrics are registered with Prometheus and can
    be scraped via the HTTP endpoint.

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.increment_request_count(model="qwen2.5:0.5b", graph="multi_domain")
        >>> with metrics.track_request_latency(model="qwen2.5:0.5b"):
        ...     # Your request handling code
        ...     pass
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = Lock()

    def __new__(cls) -> "MetricsCollector":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics collector."""
        if self._initialized:
            return

        logger.info("initializing_metrics_collector")

        # System info
        self.system_info = Info("tinyllm_system", "TinyLLM system information")
        self.system_info.info(
            {
                "version": "0.1.0",
                "app": "tinyllm",
            }
        )

        # Request metrics
        self.request_total = Counter(
            "tinyllm_requests_total",
            "Total number of requests processed",
            [MetricLabels.MODEL, MetricLabels.GRAPH, MetricLabels.REQUEST_TYPE],
        )

        self.request_latency = Histogram(
            "tinyllm_request_latency_seconds",
            "Request latency in seconds",
            [MetricLabels.MODEL, MetricLabels.GRAPH],
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
            ),
        )

        # Token metrics
        self.tokens_input = Counter(
            "tinyllm_tokens_input_total",
            "Total number of input tokens processed",
            [MetricLabels.MODEL, MetricLabels.GRAPH],
        )

        self.tokens_output = Counter(
            "tinyllm_tokens_output_total",
            "Total number of output tokens generated",
            [MetricLabels.MODEL, MetricLabels.GRAPH],
        )

        # Error metrics
        self.errors_total = Counter(
            "tinyllm_errors_total",
            "Total number of errors by type",
            [MetricLabels.ERROR_TYPE, MetricLabels.MODEL, MetricLabels.GRAPH],
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            "tinyllm_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            [MetricLabels.MODEL],
        )

        self.circuit_breaker_failures = Counter(
            "tinyllm_circuit_breaker_failures_total",
            "Total circuit breaker failures",
            [MetricLabels.MODEL],
        )

        # Active requests
        self.active_requests = Gauge(
            "tinyllm_active_requests",
            "Number of requests currently being processed",
            [MetricLabels.MODEL, MetricLabels.GRAPH],
        )

        # Model load time
        self.model_load_duration = Histogram(
            "tinyllm_model_load_duration_seconds",
            "Time taken to load a model",
            [MetricLabels.MODEL],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        # Node execution metrics
        self.node_executions_total = Counter(
            "tinyllm_node_executions_total",
            "Total number of node executions",
            [MetricLabels.NODE, MetricLabels.GRAPH],
        )

        self.node_execution_duration = Histogram(
            "tinyllm_node_execution_duration_seconds",
            "Node execution duration in seconds",
            [MetricLabels.NODE, MetricLabels.GRAPH],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        self.node_errors_total = Counter(
            "tinyllm_node_errors_total",
            "Total number of node execution errors",
            [MetricLabels.NODE, MetricLabels.GRAPH, MetricLabels.ERROR_TYPE],
        )

        # Graph execution metrics
        self.graph_executions_total = Counter(
            "tinyllm_graph_executions_total",
            "Total number of graph executions",
            [MetricLabels.GRAPH],
        )

        self.graph_execution_duration = Histogram(
            "tinyllm_graph_execution_duration_seconds",
            "Graph execution duration in seconds",
            [MetricLabels.GRAPH],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
        )

        # Cache metrics
        self.cache_hits = Counter(
            "tinyllm_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"],
        )

        self.cache_misses = Counter(
            "tinyllm_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"],
        )

        # Rate limiter metrics
        self.rate_limit_wait_seconds = Histogram(
            "tinyllm_rate_limit_wait_seconds",
            "Time spent waiting for rate limiter",
            [MetricLabels.MODEL],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        )

        # Memory metrics
        self.memory_operations_total = Counter(
            "tinyllm_memory_operations_total",
            "Total number of memory operations",
            ["operation_type"],
        )

        # Queue metrics
        self.queue_size = Gauge(
            "tinyllm_queue_size",
            "Current queue size",
            ["priority"],
        )

        self.queue_wait_time = Histogram(
            "tinyllm_queue_wait_time_seconds",
            "Time spent waiting in queue",
            ["priority"],
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
        )

        self.queue_requests_total = Counter(
            "tinyllm_queue_requests_total",
            "Total requests submitted to queue",
            ["priority"],
        )

        self.queue_requests_rejected = Counter(
            "tinyllm_queue_requests_rejected_total",
            "Total requests rejected due to full queue",
        )

        self.queue_active_workers = Gauge(
            "tinyllm_queue_active_workers",
            "Number of active worker threads",
        )

        self._initialized = True
        logger.info("metrics_collector_initialized")

    # Request tracking methods

    def increment_request_count(
        self,
        model: str = "unknown",
        graph: str = "unknown",
        request_type: str = "generate",
    ) -> None:
        """Increment the total request counter.

        Args:
            model: Model name.
            graph: Graph name.
            request_type: Type of request (generate, stream, etc).
        """
        self.request_total.labels(
            model=model, graph=graph, request_type=request_type
        ).inc()

    @contextmanager
    def track_request_latency(
        self,
        model: str = "unknown",
        graph: str = "unknown",
    ) -> Iterator[None]:
        """Context manager to track request latency.

        Args:
            model: Model name.
            graph: Graph name.

        Yields:
            None

        Example:
            >>> metrics = MetricsCollector()
            >>> with metrics.track_request_latency(model="qwen2.5:0.5b"):
            ...     # Request processing code
            ...     pass
        """
        start_time = time.monotonic()
        self.active_requests.labels(model=model, graph=graph).inc()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            self.request_latency.labels(model=model, graph=graph).observe(duration)
            self.active_requests.labels(model=model, graph=graph).dec()

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown",
        graph: str = "unknown",
    ) -> None:
        """Record token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            graph: Graph name.
        """
        self.tokens_input.labels(model=model, graph=graph).inc(input_tokens)
        self.tokens_output.labels(model=model, graph=graph).inc(output_tokens)

    # Error tracking methods

    def increment_error_count(
        self,
        error_type: str,
        model: str = "unknown",
        graph: str = "unknown",
    ) -> None:
        """Increment the error counter.

        Args:
            error_type: Type of error (e.g., timeout, connection, validation).
            model: Model name.
            graph: Graph name.
        """
        self.errors_total.labels(
            error_type=error_type, model=model, graph=graph
        ).inc()

    # Circuit breaker methods

    def update_circuit_breaker_state(self, state: str, model: str = "unknown") -> None:
        """Update circuit breaker state.

        Args:
            state: Circuit breaker state (closed, half-open, open).
            model: Model name.
        """
        state_value = {"closed": 0, "half-open": 1, "open": 2}.get(state, 0)
        self.circuit_breaker_state.labels(model=model).set(state_value)

    def increment_circuit_breaker_failures(self, model: str = "unknown") -> None:
        """Increment circuit breaker failure counter.

        Args:
            model: Model name.
        """
        self.circuit_breaker_failures.labels(model=model).inc()

    # Model loading methods

    @contextmanager
    def track_model_load(self, model: str) -> Iterator[None]:
        """Context manager to track model load duration.

        Args:
            model: Model name.

        Yields:
            None
        """
        start_time = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            self.model_load_duration.labels(model=model).observe(duration)

    # Node execution methods

    def increment_node_execution_count(
        self,
        node: str,
        graph: str = "unknown",
    ) -> None:
        """Increment node execution counter.

        Args:
            node: Node name.
            graph: Graph name.
        """
        self.node_executions_total.labels(node=node, graph=graph).inc()

    @contextmanager
    def track_node_execution(
        self,
        node: str,
        graph: str = "unknown",
    ) -> Iterator[None]:
        """Context manager to track node execution duration.

        Args:
            node: Node name.
            graph: Graph name.

        Yields:
            None
        """
        start_time = time.monotonic()
        self.increment_node_execution_count(node=node, graph=graph)
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            self.node_execution_duration.labels(node=node, graph=graph).observe(
                duration
            )

    def increment_node_error_count(
        self,
        node: str,
        error_type: str,
        graph: str = "unknown",
    ) -> None:
        """Increment node error counter.

        Args:
            node: Node name.
            error_type: Type of error.
            graph: Graph name.
        """
        self.node_errors_total.labels(
            node=node, graph=graph, error_type=error_type
        ).inc()

    # Graph execution methods

    def increment_graph_execution_count(self, graph: str) -> None:
        """Increment graph execution counter.

        Args:
            graph: Graph name.
        """
        self.graph_executions_total.labels(graph=graph).inc()

    @contextmanager
    def track_graph_execution(self, graph: str) -> Iterator[None]:
        """Context manager to track graph execution duration.

        Args:
            graph: Graph name.

        Yields:
            None
        """
        start_time = time.monotonic()
        self.increment_graph_execution_count(graph=graph)
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            self.graph_execution_duration.labels(graph=graph).observe(duration)

    # Cache methods

    def increment_cache_hit(self, cache_type: str = "general") -> None:
        """Increment cache hit counter.

        Args:
            cache_type: Type of cache (e.g., memory, disk, redis).
        """
        self.cache_hits.labels(cache_type=cache_type).inc()

    def increment_cache_miss(self, cache_type: str = "general") -> None:
        """Increment cache miss counter.

        Args:
            cache_type: Type of cache (e.g., memory, disk, redis).
        """
        self.cache_misses.labels(cache_type=cache_type).inc()

    # Rate limiter methods

    def record_rate_limit_wait(self, wait_time: float, model: str = "unknown") -> None:
        """Record rate limit wait time.

        Args:
            wait_time: Time spent waiting in seconds.
            model: Model name.
        """
        self.rate_limit_wait_seconds.labels(model=model).observe(wait_time)

    # Memory methods

    def increment_memory_operation(self, operation_type: str) -> None:
        """Increment memory operation counter.

        Args:
            operation_type: Type of operation (add, get, set, delete).
        """
        self.memory_operations_total.labels(operation_type=operation_type).inc()

    # Queue methods

    def update_queue_size(self, size: int, priority: str = "all") -> None:
        """Update queue size gauge.

        Args:
            size: Current queue size.
            priority: Priority level (high, normal, low, all).
        """
        self.queue_size.labels(priority=priority).set(size)

    def record_queue_wait_time(self, wait_time: float, priority: str = "normal") -> None:
        """Record time spent waiting in queue.

        Args:
            wait_time: Wait time in seconds.
            priority: Priority level.
        """
        self.queue_wait_time.labels(priority=priority).observe(wait_time)

    def increment_queue_request(self, priority: str = "normal") -> None:
        """Increment queue request counter.

        Args:
            priority: Priority level.
        """
        self.queue_requests_total.labels(priority=priority).inc()

    def increment_queue_rejected(self) -> None:
        """Increment queue rejection counter."""
        self.queue_requests_rejected.inc()

    def update_active_workers(self, count: int) -> None:
        """Update active worker count.

        Args:
            count: Number of active workers.
        """
        self.queue_active_workers.set(count)

    # Utility methods

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics.

        Returns:
            Dictionary containing metric summaries.
        """
        # Note: This is a simplified summary. For full metrics, use Prometheus queries.
        return {
            "collector": "prometheus",
            "registry": "default",
            "metrics_count": len(list(REGISTRY.collect())),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics to zero.

        Warning: This should only be used in testing. Production metrics
        should never be reset as it breaks historical data continuity.
        """
        logger.warning("resetting_all_metrics")
        # Note: Prometheus metrics don't support direct reset.
        # This is intentional - metrics should be cumulative.
        # For testing, recreate the MetricsCollector instance.


def start_metrics_server(
    port: int = 9090,
    addr: str = "0.0.0.0",
) -> None:
    """Start the Prometheus metrics HTTP server.

    This starts an HTTP server that exposes the /metrics endpoint for
    Prometheus to scrape. The server runs in a separate thread.

    Args:
        port: Port to listen on (default: 9090).
        addr: Address to bind to (default: 0.0.0.0 for all interfaces).

    Example:
        >>> start_metrics_server(port=9090)
        >>> # Metrics now available at http://localhost:9090/metrics
    """
    logger.info("starting_metrics_server", port=port, addr=addr)
    try:
        start_http_server(port=port, addr=addr)
        logger.info("metrics_server_started", port=port, addr=addr)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(
                "metrics_server_already_running",
                port=port,
                addr=addr,
            )
        else:
            logger.error(
                "metrics_server_start_failed",
                port=port,
                addr=addr,
                error=str(e),
            )
            raise


# Create global instance for convenience
_global_metrics: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector singleton.

    Example:
        >>> metrics = get_metrics_collector()
        >>> metrics.increment_request_count(model="qwen2.5:0.5b")
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
