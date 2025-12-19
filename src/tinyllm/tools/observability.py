"""Tool observability for TinyLLM.

This module provides observability capabilities for tools,
including metrics, tracing, and logging.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SpanStatus(str, Enum):
    """Status of a trace span."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Context for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    @classmethod
    def create(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Create a new span context.

        Args:
            parent: Optional parent context.

        Returns:
            New SpanContext.
        """
        return cls(
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id if parent else None,
        )


@dataclass
class Span:
    """A trace span representing a unit of work."""

    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute.

        Args:
            key: Attribute key.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Event attributes.

        Returns:
            Self for chaining.
        """
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )
        return self

    def end(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None) -> None:
        """End the span.

        Args:
            status: Span status.
            error: Error message if status is ERROR.
        """
        self.end_time = time.time()
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error": self.error,
            "attributes": self.attributes,
            "events": self.events,
        }


class SpanExporter(ABC):
    """Abstract base for span exporters."""

    @abstractmethod
    def export(self, spans: List[Span]) -> None:
        """Export spans.

        Args:
            spans: List of spans to export.
        """
        pass


class ConsoleExporter(SpanExporter):
    """Exports spans to console."""

    def export(self, spans: List[Span]) -> None:
        """Export spans to console."""
        for span in spans:
            status_symbol = "✓" if span.status == SpanStatus.OK else "✗"
            duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "N/A"
            print(f"[{status_symbol}] {span.name} ({duration})")


class LoggingExporter(SpanExporter):
    """Exports spans to logging."""

    def __init__(self, logger_name: str = "tinyllm.traces"):
        """Initialize logging exporter.

        Args:
            logger_name: Logger name to use.
        """
        self.logger = logging.getLogger(logger_name)

    def export(self, spans: List[Span]) -> None:
        """Export spans to logging."""
        for span in spans:
            span_data = span.to_dict()
            # Rename 'name' to avoid conflict with LogRecord
            extra = {"span_" + k: v for k, v in span_data.items()}
            if span.status == SpanStatus.OK:
                self.logger.info(f"Span completed: {span.name}", extra=extra)
            else:
                self.logger.warning(f"Span failed: {span.name}", extra=extra)


class InMemoryExporter(SpanExporter):
    """Exports spans to memory for testing."""

    def __init__(self, max_spans: int = 1000):
        """Initialize in-memory exporter.

        Args:
            max_spans: Maximum spans to keep.
        """
        self.spans: List[Span] = []
        self.max_spans = max_spans

    def export(self, spans: List[Span]) -> None:
        """Export spans to memory."""
        self.spans.extend(spans)
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans :]

    def get_spans(self, name: Optional[str] = None) -> List[Span]:
        """Get stored spans.

        Args:
            name: Optional name filter.

        Returns:
            List of matching spans.
        """
        if name:
            return [s for s in self.spans if s.name == name]
        return list(self.spans)

    def clear(self) -> None:
        """Clear stored spans."""
        self.spans.clear()


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A metric with aggregated values."""

    name: str
    type: MetricType
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    _values: List[float] = field(default_factory=list)
    _count: int = 0
    _sum: float = 0.0

    def record(self, value: float) -> None:
        """Record a value.

        Args:
            value: Value to record.
        """
        self._values.append(value)
        self._count += 1
        self._sum += value

    @property
    def count(self) -> int:
        """Get count of recorded values."""
        return self._count

    @property
    def sum(self) -> float:
        """Get sum of recorded values."""
        return self._sum

    @property
    def avg(self) -> float:
        """Get average of recorded values."""
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def min(self) -> float:
        """Get minimum value."""
        return min(self._values) if self._values else 0.0

    @property
    def max(self) -> float:
        """Get maximum value."""
        return max(self._values) if self._values else 0.0


class MetricsCollector:
    """Collects and manages metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._metrics: Dict[str, Metric] = {}

    def counter(self, name: str, description: str = "") -> Metric:
        """Get or create a counter metric.

        Args:
            name: Metric name.
            description: Metric description.

        Returns:
            Counter metric.
        """
        if name not in self._metrics:
            self._metrics[name] = Metric(
                name=name,
                type=MetricType.COUNTER,
                description=description,
            )
        return self._metrics[name]

    def gauge(self, name: str, description: str = "") -> Metric:
        """Get or create a gauge metric.

        Args:
            name: Metric name.
            description: Metric description.

        Returns:
            Gauge metric.
        """
        if name not in self._metrics:
            self._metrics[name] = Metric(
                name=name,
                type=MetricType.GAUGE,
                description=description,
            )
        return self._metrics[name]

    def histogram(self, name: str, description: str = "") -> Metric:
        """Get or create a histogram metric.

        Args:
            name: Metric name.
            description: Metric description.

        Returns:
            Histogram metric.
        """
        if name not in self._metrics:
            self._metrics[name] = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                description=description,
            )
        return self._metrics[name]

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name.

        Args:
            name: Metric name.

        Returns:
            Metric or None.
        """
        return self._metrics.get(name)

    def all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionary.

        Returns:
            Dictionary of metric data.
        """
        return {
            name: {
                "type": m.type.value,
                "description": m.description,
                "count": m.count,
                "sum": m.sum,
                "avg": m.avg,
                "min": m.min,
                "max": m.max,
            }
            for name, m in self._metrics.items()
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


class Tracer:
    """Tracer for creating and managing spans."""

    def __init__(
        self,
        service_name: str = "tinyllm",
        exporter: Optional[SpanExporter] = None,
    ):
        """Initialize tracer.

        Args:
            service_name: Name of the service.
            exporter: Span exporter to use.
        """
        self.service_name = service_name
        self.exporter = exporter or LoggingExporter()
        self._current_context: Optional[SpanContext] = None

    def start_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name.
            parent: Optional parent context.

        Returns:
            New span.
        """
        context = SpanContext.create(parent or self._current_context)
        span = Span(name=name, context=context)
        span.set_attribute("service.name", self.service_name)
        return span

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing.

        Args:
            name: Span name.
            attributes: Initial attributes.

        Yields:
            Active span.
        """
        span = self.start_span(name)

        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        old_context = self._current_context
        self._current_context = span.context

        try:
            yield span
            span.end(SpanStatus.OK)
        except Exception as e:
            span.end(SpanStatus.ERROR, str(e))
            raise
        finally:
            self._current_context = old_context
            self.exporter.export([span])


class ToolObserver:
    """Main observability interface for tools."""

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize tool observer.

        Args:
            tracer: Tracer instance.
            metrics: Metrics collector.
        """
        self.tracer = tracer or Tracer()
        self.metrics = metrics or MetricsCollector()

        # Pre-create common metrics
        self._execution_count = self.metrics.counter(
            "tool.executions.total",
            "Total number of tool executions",
        )
        self._execution_duration = self.metrics.histogram(
            "tool.execution.duration_ms",
            "Tool execution duration in milliseconds",
        )
        self._error_count = self.metrics.counter(
            "tool.errors.total",
            "Total number of tool errors",
        )

    def record_execution(
        self,
        tool_id: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool execution.

        Args:
            tool_id: Tool identifier.
            duration_ms: Execution duration in ms.
            success: Whether execution succeeded.
            error: Error message if failed.
        """
        self._execution_count.record(1)
        self._execution_duration.record(duration_ms)

        if not success:
            self._error_count.record(1)

        logger.info(
            f"Tool {tool_id} executed in {duration_ms:.2f}ms",
            extra={
                "tool_id": tool_id,
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
            },
        )

    @asynccontextmanager
    async def observe(
        self,
        tool_id: str,
        input_data: Any = None,
    ):
        """Context manager for observing tool execution.

        Args:
            tool_id: Tool identifier.
            input_data: Optional input data.

        Yields:
            Active span.
        """
        start = time.time()

        async with self.tracer.trace(
            f"tool.execute.{tool_id}",
            attributes={"tool.id": tool_id},
        ) as span:
            try:
                yield span
                duration_ms = (time.time() - start) * 1000
                self.record_execution(tool_id, duration_ms, True)

            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                self.record_execution(tool_id, duration_ms, False, str(e))
                raise

    def get_stats(self) -> Dict[str, Any]:
        """Get observability statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "metrics": self.metrics.all_metrics(),
            "total_executions": self._execution_count.count,
            "total_errors": self._error_count.count,
            "avg_duration_ms": self._execution_duration.avg,
        }


class ObservableToolWrapper:
    """Wrapper that adds observability to tools."""

    def __init__(
        self,
        tool: Any,
        observer: Optional[ToolObserver] = None,
        capture_input: bool = False,
        capture_output: bool = False,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            observer: Tool observer.
            capture_input: Whether to capture input in traces.
            capture_output: Whether to capture output in traces.
        """
        self.tool = tool
        self.observer = observer or ToolObserver()
        self.capture_input = capture_input
        self.capture_output = capture_output

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input_data: Any) -> Any:
        """Execute tool with observability.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        tool_id = self.tool.metadata.id

        async with self.observer.observe(tool_id, input_data) as span:
            if self.capture_input:
                span.set_attribute("tool.input", str(input_data)[:1000])

            span.add_event("tool.start", {"tool_id": tool_id})

            result = await self.tool.execute(input_data)

            if self.capture_output:
                span.set_attribute("tool.output", str(result)[:1000])

            span.add_event("tool.complete", {"tool_id": tool_id})

            return result


# Convenience functions


def with_observability(
    tool: Any,
    observer: Optional[ToolObserver] = None,
    capture_input: bool = False,
    capture_output: bool = False,
) -> ObservableToolWrapper:
    """Add observability to a tool.

    Args:
        tool: Tool to wrap.
        observer: Tool observer.
        capture_input: Whether to capture input.
        capture_output: Whether to capture output.

    Returns:
        ObservableToolWrapper.
    """
    return ObservableToolWrapper(
        tool,
        observer=observer,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def create_observer(
    service_name: str = "tinyllm",
    exporter: Optional[SpanExporter] = None,
) -> ToolObserver:
    """Create a new tool observer.

    Args:
        service_name: Service name.
        exporter: Span exporter.

    Returns:
        ToolObserver instance.
    """
    tracer = Tracer(service_name=service_name, exporter=exporter)
    return ToolObserver(tracer=tracer)
