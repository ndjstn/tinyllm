"""Tests for tool observability."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.observability import (
    ConsoleExporter,
    InMemoryExporter,
    LoggingExporter,
    Metric,
    MetricsCollector,
    MetricType,
    ObservableToolWrapper,
    Span,
    SpanContext,
    SpanStatus,
    ToolObserver,
    Tracer,
    create_observer,
    with_observability,
)


class ObsInput(BaseModel):
    """Input for observability tests."""

    value: int = 0


class ObsOutput(BaseModel):
    """Output for observability tests."""

    value: int = 0
    success: bool = True


class SuccessTool(BaseTool[ObsInput, ObsOutput]):
    """Tool that succeeds."""

    metadata = ToolMetadata(
        id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        category="utility",
    )
    input_type = ObsInput
    output_type = ObsOutput

    async def execute(self, input: ObsInput) -> ObsOutput:
        return ObsOutput(value=input.value * 2)


class FailTool(BaseTool[ObsInput, ObsOutput]):
    """Tool that fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = ObsInput
    output_type = ObsOutput

    async def execute(self, input: ObsInput) -> ObsOutput:
        raise ValueError("Intentional failure")


class SlowTool(BaseTool[ObsInput, ObsOutput]):
    """Tool that takes some time."""

    metadata = ToolMetadata(
        id="slow_tool",
        name="Slow Tool",
        description="Takes time",
        category="utility",
    )
    input_type = ObsInput
    output_type = ObsOutput

    async def execute(self, input: ObsInput) -> ObsOutput:
        import asyncio

        await asyncio.sleep(0.05)
        return ObsOutput(value=input.value * 2)


class TestSpanContext:
    """Tests for SpanContext."""

    def test_create_root_context(self):
        """Test creating root context."""
        context = SpanContext.create()

        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.parent_span_id is None

    def test_create_child_context(self):
        """Test creating child context."""
        parent = SpanContext.create()
        child = SpanContext.create(parent)

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id


class TestSpan:
    """Tests for Span."""

    def test_creation(self):
        """Test span creation."""
        context = SpanContext.create()
        span = Span(name="test_span", context=context)

        assert span.name == "test_span"
        assert span.status == SpanStatus.OK
        assert span.end_time is None

    def test_duration(self):
        """Test span duration calculation."""
        import time

        context = SpanContext.create()
        span = Span(name="test_span", context=context)

        time.sleep(0.01)
        span.end()

        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_set_attribute(self):
        """Test setting attributes."""
        context = SpanContext.create()
        span = Span(name="test", context=context)

        span.set_attribute("key", "value")

        assert span.attributes["key"] == "value"

    def test_add_event(self):
        """Test adding events."""
        context = SpanContext.create()
        span = Span(name="test", context=context)

        span.add_event("my_event", {"foo": "bar"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "my_event"
        assert span.events[0]["attributes"]["foo"] == "bar"

    def test_end_with_error(self):
        """Test ending span with error."""
        context = SpanContext.create()
        span = Span(name="test", context=context)

        span.end(SpanStatus.ERROR, "Something went wrong")

        assert span.status == SpanStatus.ERROR
        assert span.error == "Something went wrong"

    def test_to_dict(self):
        """Test converting to dictionary."""
        context = SpanContext.create()
        span = Span(name="test", context=context)
        span.end()

        d = span.to_dict()

        assert d["name"] == "test"
        assert d["status"] == "ok"
        assert "duration_ms" in d

    def test_chaining(self):
        """Test method chaining."""
        context = SpanContext.create()
        span = (
            Span(name="test", context=context)
            .set_attribute("key1", "value1")
            .set_attribute("key2", "value2")
            .add_event("event1")
        )

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
        assert len(span.events) == 1


class TestSpanExporters:
    """Tests for span exporters."""

    def test_console_exporter(self, capsys):
        """Test console exporter."""
        exporter = ConsoleExporter()
        context = SpanContext.create()
        span = Span(name="test_span", context=context)
        span.end()

        exporter.export([span])

        captured = capsys.readouterr()
        assert "test_span" in captured.out
        assert "✓" in captured.out

    def test_logging_exporter(self, caplog):
        """Test logging exporter."""
        import logging

        with caplog.at_level(logging.INFO, logger="tinyllm.traces"):
            exporter = LoggingExporter()
            context = SpanContext.create()
            span = Span(name="test_span", context=context)
            span.end()

            exporter.export([span])

            assert "test_span" in caplog.text

    def test_in_memory_exporter(self):
        """Test in-memory exporter."""
        exporter = InMemoryExporter()
        context = SpanContext.create()
        span = Span(name="test_span", context=context)
        span.end()

        exporter.export([span])

        assert len(exporter.get_spans()) == 1
        assert exporter.get_spans("test_span")[0].name == "test_span"

    def test_in_memory_exporter_clear(self):
        """Test clearing in-memory exporter."""
        exporter = InMemoryExporter()
        context = SpanContext.create()
        span = Span(name="test", context=context)
        exporter.export([span])

        exporter.clear()

        assert len(exporter.get_spans()) == 0

    def test_in_memory_exporter_max_spans(self):
        """Test max spans limit."""
        exporter = InMemoryExporter(max_spans=5)

        for i in range(10):
            context = SpanContext.create()
            span = Span(name=f"span_{i}", context=context)
            exporter.export([span])

        assert len(exporter.get_spans()) == 5


class TestMetric:
    """Tests for Metric."""

    def test_record(self):
        """Test recording values."""
        metric = Metric(name="test", type=MetricType.COUNTER)

        metric.record(10)
        metric.record(20)
        metric.record(30)

        assert metric.count == 3
        assert metric.sum == 60

    def test_avg(self):
        """Test average calculation."""
        metric = Metric(name="test", type=MetricType.HISTOGRAM)

        metric.record(10)
        metric.record(20)
        metric.record(30)

        assert metric.avg == 20

    def test_min_max(self):
        """Test min/max values."""
        metric = Metric(name="test", type=MetricType.HISTOGRAM)

        metric.record(10)
        metric.record(5)
        metric.record(20)

        assert metric.min == 5
        assert metric.max == 20

    def test_empty_metric(self):
        """Test empty metric values."""
        metric = Metric(name="test", type=MetricType.COUNTER)

        assert metric.count == 0
        assert metric.sum == 0
        assert metric.avg == 0


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_counter(self):
        """Test counter creation."""
        collector = MetricsCollector()

        counter = collector.counter("my_counter", "A counter")

        assert counter.type == MetricType.COUNTER
        assert counter.description == "A counter"

    def test_gauge(self):
        """Test gauge creation."""
        collector = MetricsCollector()

        gauge = collector.gauge("my_gauge")

        assert gauge.type == MetricType.GAUGE

    def test_histogram(self):
        """Test histogram creation."""
        collector = MetricsCollector()

        histogram = collector.histogram("my_histogram")

        assert histogram.type == MetricType.HISTOGRAM

    def test_get_metric(self):
        """Test getting metric by name."""
        collector = MetricsCollector()
        collector.counter("my_counter")

        metric = collector.get_metric("my_counter")

        assert metric is not None
        assert metric.name == "my_counter"

    def test_get_nonexistent_metric(self):
        """Test getting nonexistent metric."""
        collector = MetricsCollector()

        metric = collector.get_metric("nonexistent")

        assert metric is None

    def test_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        collector.counter("counter1").record(10)
        collector.histogram("hist1").record(100)

        all_metrics = collector.all_metrics()

        assert "counter1" in all_metrics
        assert "hist1" in all_metrics
        assert all_metrics["counter1"]["count"] == 1

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.counter("test").record(10)

        collector.reset()

        assert collector.get_metric("test") is None


class TestTracer:
    """Tests for Tracer."""

    def test_start_span(self):
        """Test starting a span."""
        tracer = Tracer(service_name="test_service")

        span = tracer.start_span("my_span")

        assert span.name == "my_span"
        assert span.attributes["service.name"] == "test_service"

    @pytest.mark.asyncio
    async def test_trace_context_manager(self):
        """Test trace context manager."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        async with tracer.trace("test_span") as span:
            span.set_attribute("key", "value")

        assert len(exporter.get_spans()) == 1
        assert exporter.get_spans()[0].status == SpanStatus.OK

    @pytest.mark.asyncio
    async def test_trace_with_error(self):
        """Test trace with exception."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)

        with pytest.raises(ValueError):
            async with tracer.trace("test_span"):
                raise ValueError("test error")

        assert exporter.get_spans()[0].status == SpanStatus.ERROR


class TestToolObserver:
    """Tests for ToolObserver."""

    def test_record_execution(self):
        """Test recording execution."""
        observer = ToolObserver()

        observer.record_execution("tool1", 100.0, True)
        observer.record_execution("tool1", 200.0, True)

        stats = observer.get_stats()
        assert stats["total_executions"] == 2
        assert stats["avg_duration_ms"] == 150.0

    def test_record_error(self):
        """Test recording error."""
        observer = ToolObserver()

        observer.record_execution("tool1", 50.0, False, "Error message")

        stats = observer.get_stats()
        assert stats["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_observe_success(self):
        """Test observing successful execution."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)

        async with observer.observe("test_tool"):
            pass

        stats = observer.get_stats()
        assert stats["total_executions"] == 1
        assert stats["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_observe_failure(self):
        """Test observing failed execution."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)

        with pytest.raises(ValueError):
            async with observer.observe("test_tool"):
                raise ValueError("test")

        stats = observer.get_stats()
        assert stats["total_errors"] == 1


class TestObservableToolWrapper:
    """Tests for ObservableToolWrapper."""

    @pytest.mark.asyncio
    async def test_success_execution(self):
        """Test successful execution with observability."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)
        wrapper = ObservableToolWrapper(SuccessTool(), observer=observer)

        result = await wrapper.execute(ObsInput(value=5))

        assert result.value == 10
        stats = observer.get_stats()
        assert stats["total_executions"] == 1

    @pytest.mark.asyncio
    async def test_failure_execution(self):
        """Test failed execution with observability."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)
        wrapper = ObservableToolWrapper(FailTool(), observer=observer)

        with pytest.raises(ValueError):
            await wrapper.execute(ObsInput(value=5))

        stats = observer.get_stats()
        assert stats["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_capture_input(self):
        """Test capturing input."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)
        wrapper = ObservableToolWrapper(
            SuccessTool(),
            observer=observer,
            capture_input=True,
        )

        await wrapper.execute(ObsInput(value=5))

        span = exporter.get_spans()[0]
        assert "tool.input" in span.attributes

    @pytest.mark.asyncio
    async def test_capture_output(self):
        """Test capturing output."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)
        wrapper = ObservableToolWrapper(
            SuccessTool(),
            observer=observer,
            capture_output=True,
        )

        await wrapper.execute(ObsInput(value=5))

        span = exporter.get_spans()[0]
        assert "tool.output" in span.attributes

    @pytest.mark.asyncio
    async def test_events_recorded(self):
        """Test events are recorded."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        observer = ToolObserver(tracer=tracer)
        wrapper = ObservableToolWrapper(SuccessTool(), observer=observer)

        await wrapper.execute(ObsInput(value=5))

        span = exporter.get_spans()[0]
        event_names = [e["name"] for e in span.events]
        assert "tool.start" in event_names
        assert "tool.complete" in event_names

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        wrapper = ObservableToolWrapper(SuccessTool())

        assert wrapper.metadata.id == "success_tool"

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        """Test duration is recorded."""
        observer = ToolObserver()
        wrapper = ObservableToolWrapper(SlowTool(), observer=observer)

        await wrapper.execute(ObsInput(value=5))

        stats = observer.get_stats()
        assert stats["avg_duration_ms"] >= 50


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_observability(self):
        """Test with_observability function."""
        wrapper = with_observability(SuccessTool())

        result = await wrapper.execute(ObsInput(value=5))

        assert result.value == 10

    def test_create_observer(self):
        """Test create_observer function."""
        observer = create_observer(service_name="my_service")

        assert observer.tracer.service_name == "my_service"
