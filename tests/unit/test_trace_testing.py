"""Tests for trace-based testing utilities."""

import pytest

from tinyllm.trace_testing import CapturedSpan, InMemorySpanExporter, TraceTestContext


class TestCapturedSpan:
    """Test CapturedSpan dataclass."""

    def test_captured_span_creation(self):
        """Test creating a captured span."""
        span = CapturedSpan(
            name="test.span",
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            parent_span_id=None,
            attributes={"key": "value"},
        )

        assert span.name == "test.span"
        assert span.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert span.span_id == "b7ad6b7169203331"
        assert span.attributes["key"] == "value"

    def test_has_attribute_existence(self):
        """Test checking attribute existence."""
        span = CapturedSpan(
            name="test.span",
            trace_id="trace123",
            span_id="span123",
            parent_span_id=None,
            attributes={"key1": "value1", "key2": "value2"},
        )

        assert span.has_attribute("key1") is True
        assert span.has_attribute("key2") is True
        assert span.has_attribute("key3") is False

    def test_has_attribute_with_value(self):
        """Test checking attribute with expected value."""
        span = CapturedSpan(
            name="test.span",
            trace_id="trace123",
            span_id="span123",
            parent_span_id=None,
            attributes={"status": "success", "count": "42"},
        )

        assert span.has_attribute("status", "success") is True
        assert span.has_attribute("status", "failed") is False
        assert span.has_attribute("count", "42") is True
        assert span.has_attribute("count", 42) is True  # String comparison

    def test_has_event(self):
        """Test checking for events."""
        span = CapturedSpan(
            name="test.span",
            trace_id="trace123",
            span_id="span123",
            parent_span_id=None,
            events=[
                {"name": "event1", "attributes": {}},
                {"name": "event2", "attributes": {"detail": "info"}},
            ],
        )

        assert span.has_event("event1") is True
        assert span.has_event("event2") is True
        assert span.has_event("event3") is False

    def test_is_error(self):
        """Test error status checking."""
        error_span = CapturedSpan(
            name="error.span",
            trace_id="trace123",
            span_id="span123",
            parent_span_id=None,
            status_code="ERROR",
        )

        success_span = CapturedSpan(
            name="success.span",
            trace_id="trace123",
            span_id="span456",
            parent_span_id=None,
            status_code="OK",
        )

        assert error_span.is_error() is True
        assert success_span.is_error() is False

    def test_duration_ms(self):
        """Test duration calculation."""
        span = CapturedSpan(
            name="test.span",
            trace_id="trace123",
            span_id="span123",
            parent_span_id=None,
            start_time=1000000000,  # nanoseconds
            end_time=1001000000,  # 1ms later
        )

        assert span.duration_ms == 1.0


class TestInMemorySpanExporter:
    """Test InMemorySpanExporter."""

    def test_exporter_creation(self):
        """Test creating an exporter."""
        exporter = InMemorySpanExporter()

        assert exporter.spans == []
        assert exporter._exported_count == 0

    def test_clear_spans(self):
        """Test clearing captured spans."""
        exporter = InMemorySpanExporter()

        # Manually add some spans
        exporter.spans.append(
            CapturedSpan(
                name="test.span",
                trace_id="trace123",
                span_id="span123",
                parent_span_id=None,
            )
        )

        assert len(exporter.spans) == 1

        exporter.clear()

        assert len(exporter.spans) == 0
        assert exporter._exported_count == 0

    def test_get_spans_no_filter(self):
        """Test getting all spans without filters."""
        exporter = InMemorySpanExporter()

        span1 = CapturedSpan(
            name="span1", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        span2 = CapturedSpan(
            name="span2", trace_id="trace1", span_id="span2", parent_span_id=None
        )

        exporter.spans = [span1, span2]

        spans = exporter.get_spans()
        assert len(spans) == 2

    def test_get_spans_filter_by_name(self):
        """Test filtering spans by name."""
        exporter = InMemorySpanExporter()

        span1 = CapturedSpan(
            name="span1", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        span2 = CapturedSpan(
            name="span2", trace_id="trace1", span_id="span2", parent_span_id=None
        )

        exporter.spans = [span1, span2]

        spans = exporter.get_spans(name="span1")
        assert len(spans) == 1
        assert spans[0].name == "span1"

    def test_get_spans_filter_by_trace_id(self):
        """Test filtering spans by trace ID."""
        exporter = InMemorySpanExporter()

        span1 = CapturedSpan(
            name="span1", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        span2 = CapturedSpan(
            name="span2", trace_id="trace2", span_id="span2", parent_span_id=None
        )

        exporter.spans = [span1, span2]

        spans = exporter.get_spans(trace_id="trace1")
        assert len(spans) == 1
        assert spans[0].trace_id == "trace1"

    def test_get_span(self):
        """Test getting first span by name."""
        exporter = InMemorySpanExporter()

        span1 = CapturedSpan(
            name="test.span", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        span2 = CapturedSpan(
            name="other.span", trace_id="trace1", span_id="span2", parent_span_id=None
        )

        exporter.spans = [span1, span2]

        span = exporter.get_span("test.span")
        assert span is not None
        assert span.name == "test.span"

        span = exporter.get_span("nonexistent")
        assert span is None

    def test_assert_span_exists_success(self):
        """Test asserting span exists (success case)."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        exporter.spans = [span]

        # Should not raise
        exporter.assert_span_exists("test.span")

    def test_assert_span_exists_failure(self):
        """Test asserting span exists (failure case)."""
        exporter = InMemorySpanExporter()

        with pytest.raises(AssertionError, match="Expected span.*not found"):
            exporter.assert_span_exists("nonexistent.span")

    def test_assert_span_count_success(self):
        """Test asserting span count (success case)."""
        exporter = InMemorySpanExporter()

        span1 = CapturedSpan(
            name="span1", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        span2 = CapturedSpan(
            name="span2", trace_id="trace1", span_id="span2", parent_span_id=None
        )

        exporter.spans = [span1, span2]

        # Should not raise
        exporter.assert_span_count(2)

    def test_assert_span_count_failure(self):
        """Test asserting span count (failure case)."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="span1", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        exporter.spans = [span]

        with pytest.raises(AssertionError, match="Expected 2 spans, got 1"):
            exporter.assert_span_count(2)

    def test_assert_span_attribute_success(self):
        """Test asserting span attribute (success case)."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            attributes={"key": "value", "count": "42"},
        )
        exporter.spans = [span]

        # Should not raise
        exporter.assert_span_attribute("test.span", "key", "value")
        exporter.assert_span_attribute("test.span", "count", "42")
        exporter.assert_span_attribute("test.span", "key")  # Just existence

    def test_assert_span_attribute_failure_missing_span(self):
        """Test asserting attribute on missing span."""
        exporter = InMemorySpanExporter()

        with pytest.raises(AssertionError, match="Span.*not found"):
            exporter.assert_span_attribute("nonexistent", "key")

    def test_assert_span_attribute_failure_missing_attribute(self):
        """Test asserting missing attribute."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            attributes={"key": "value"},
        )
        exporter.spans = [span]

        with pytest.raises(AssertionError, match="missing attribute"):
            exporter.assert_span_attribute("test.span", "nonexistent")

    def test_assert_span_attribute_failure_wrong_value(self):
        """Test asserting attribute with wrong value."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            attributes={"key": "value"},
        )
        exporter.spans = [span]

        with pytest.raises(AssertionError, match="expected"):
            exporter.assert_span_attribute("test.span", "key", "wrong")

    def test_assert_span_event_success(self):
        """Test asserting span event (success case)."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            events=[{"name": "test.event", "attributes": {}}],
        )
        exporter.spans = [span]

        # Should not raise
        exporter.assert_span_event("test.span", "test.event")

    def test_assert_span_event_failure(self):
        """Test asserting span event (failure case)."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            events=[],
        )
        exporter.spans = [span]

        with pytest.raises(AssertionError, match="missing event"):
            exporter.assert_span_event("test.span", "nonexistent.event")

    def test_assert_no_errors_success(self):
        """Test asserting no errors (success case)."""
        exporter = InMemorySpanExporter()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            status_code="OK",
        )
        exporter.spans = [span]

        # Should not raise
        exporter.assert_no_errors()

    def test_assert_no_errors_failure(self):
        """Test asserting no errors (failure case)."""
        exporter = InMemorySpanExporter()

        error_span = CapturedSpan(
            name="error.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            status_code="ERROR",
        )
        exporter.spans = [error_span]

        with pytest.raises(AssertionError, match="error spans"):
            exporter.assert_no_errors()


class TestTraceTestContext:
    """Test TraceTestContext."""

    def test_context_creation(self):
        """Test creating a trace test context."""
        ctx = TraceTestContext()

        assert ctx.exporter is not None
        assert isinstance(ctx.exporter, InMemorySpanExporter)

    def test_setup_teardown(self):
        """Test setup and teardown."""
        ctx = TraceTestContext()

        # Should not raise
        ctx.setup()
        ctx.teardown()

    def test_clear(self):
        """Test clearing captured spans."""
        ctx = TraceTestContext()

        # Manually add span
        ctx.exporter.spans.append(
            CapturedSpan(
                name="test.span",
                trace_id="trace1",
                span_id="span1",
                parent_span_id=None,
            )
        )

        assert len(ctx.exporter.spans) == 1

        ctx.clear()

        assert len(ctx.exporter.spans) == 0

    def test_get_spans(self):
        """Test getting spans through context."""
        ctx = TraceTestContext()

        span = CapturedSpan(
            name="test.span", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        ctx.exporter.spans = [span]

        spans = ctx.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.span"

    def test_get_span(self):
        """Test getting single span through context."""
        ctx = TraceTestContext()

        span = CapturedSpan(
            name="test.span", trace_id="trace1", span_id="span1", parent_span_id=None
        )
        ctx.exporter.spans = [span]

        result = ctx.get_span("test.span")
        assert result is not None
        assert result.name == "test.span"

    def test_assert_methods(self):
        """Test assertion methods through context."""
        ctx = TraceTestContext()

        span = CapturedSpan(
            name="test.span",
            trace_id="trace1",
            span_id="span1",
            parent_span_id=None,
            attributes={"key": "value"},
            events=[{"name": "test.event", "attributes": {}}],
            status_code="OK",
        )
        ctx.exporter.spans = [span]

        # Should not raise
        ctx.assert_span_exists("test.span")
        ctx.assert_span_count(1)
        ctx.assert_span_attribute("test.span", "key", "value")
        ctx.assert_span_event("test.span", "test.event")
        ctx.assert_no_errors()


class TestTraceTestingIntegration:
    """Integration tests for trace-based testing."""

    def test_full_workflow(self):
        """Test complete trace testing workflow."""
        ctx = TraceTestContext()
        ctx.setup()

        try:
            # Simulate span capture
            span = CapturedSpan(
                name="graph.execute",
                trace_id="trace123",
                span_id="span123",
                parent_span_id=None,
                attributes={
                    "graph.id": "test-graph",
                    "execution.step": "1",
                },
                events=[{"name": "node.started", "attributes": {}}],
                status_code="OK",
            )

            ctx.exporter.spans.append(span)

            # Perform assertions
            ctx.assert_span_exists("graph.execute")
            ctx.assert_span_count(1)
            ctx.assert_span_attribute("graph.execute", "graph.id", "test-graph")
            ctx.assert_span_event("graph.execute", "node.started")
            ctx.assert_no_errors()

        finally:
            ctx.teardown()
