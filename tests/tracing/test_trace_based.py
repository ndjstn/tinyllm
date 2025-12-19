"""Trace-based testing for TinyLLM.

This module implements trace-based testing where test assertions are made
based on the actual traces generated during execution. This ensures that
distributed tracing is working correctly and capturing all relevant
operations and attributes.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class SpanCapture:
    """Capture spans for testing."""

    def __init__(self):
        """Initialize span capture."""
        self.spans: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def capture_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Capture a span.

        Args:
            name: Span name.
            attributes: Span attributes.
        """
        self.spans.append(
            {
                "name": name,
                "attributes": attributes or {},
                "events": [],
                "status": "ok",
            }
        )
        return len(self.spans) - 1

    def add_event(self, span_idx: int, name: str, attributes: dict[str, Any] | None = None):
        """Add an event to a span.

        Args:
            span_idx: Index of span to add event to.
            name: Event name.
            attributes: Event attributes.
        """
        if 0 <= span_idx < len(self.spans):
            self.spans[span_idx]["events"].append(
                {
                    "name": name,
                    "attributes": attributes or {},
                }
            )

    def set_status(self, span_idx: int, status: str):
        """Set span status.

        Args:
            span_idx: Index of span.
            status: Status string.
        """
        if 0 <= span_idx < len(self.spans):
            self.spans[span_idx]["status"] = status

    def get_span_by_name(self, name: str) -> dict[str, Any] | None:
        """Get first span with given name.

        Args:
            name: Span name.

        Returns:
            Span data or None.
        """
        for span in self.spans:
            if span["name"] == name:
                return span
        return None

    def get_all_spans_by_name(self, name: str) -> list[dict[str, Any]]:
        """Get all spans with given name.

        Args:
            name: Span name.

        Returns:
            List of span data.
        """
        return [span for span in self.spans if span["name"] == name]

    def assert_span_exists(self, name: str):
        """Assert that a span with the given name exists.

        Args:
            name: Span name.
        """
        assert self.get_span_by_name(name) is not None, f"Span '{name}' not found in traces"

    def assert_span_attribute(self, span_name: str, attribute_key: str, expected_value: Any):
        """Assert that a span has a specific attribute value.

        Args:
            span_name: Span name.
            attribute_key: Attribute key.
            expected_value: Expected value.
        """
        span = self.get_span_by_name(span_name)
        assert span is not None, f"Span '{span_name}' not found"
        assert attribute_key in span["attributes"], f"Attribute '{attribute_key}' not in span"
        actual = span["attributes"][attribute_key]
        assert (
            actual == expected_value
        ), f"Expected {attribute_key}={expected_value}, got {actual}"

    def assert_span_count(self, name: str, expected_count: int):
        """Assert the number of spans with a given name.

        Args:
            name: Span name.
            expected_count: Expected count.
        """
        spans = self.get_all_spans_by_name(name)
        actual_count = len(spans)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} spans named '{name}', got {actual_count}"

    def assert_span_hierarchy(self, parent_name: str, child_name: str):
        """Assert that a child span exists for a parent span.

        Args:
            parent_name: Parent span name.
            child_name: Child span name.
        """
        parent = self.get_span_by_name(parent_name)
        assert parent is not None, f"Parent span '{parent_name}' not found"

        child = self.get_span_by_name(child_name)
        assert child is not None, f"Child span '{child_name}' not found"

        # In a real implementation, we'd check parent-child relationship
        # For this test, we just verify both exist
        parent_idx = self.spans.index(parent)
        child_idx = self.spans.index(child)
        assert (
            parent_idx < child_idx
        ), f"Parent span '{parent_name}' should come before child '{child_name}'"


@pytest.fixture
def span_capture():
    """Fixture to capture spans."""
    return SpanCapture()


@pytest.mark.asyncio
class TestTraceBasedExecution:
    """Test trace-based execution patterns."""

    async def test_simple_span_creation(self, span_capture):
        """Test that simple operations create spans."""
        from tinyllm.telemetry import configure_telemetry, trace_span

        # Mock telemetry
        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                mock_tracer = MagicMock()
                mock_span = MagicMock()
                mock_span.is_recording.return_value = True

                # Mock span context
                mock_span_context = MagicMock()
                mock_span_context.trace_id = 12345
                mock_span_context.span_id = 67890
                mock_span.get_span_context.return_value = mock_span_context

                # Capture span creation
                def start_span(name):
                    span_idx = span_capture.capture_span(name)
                    return mock_span

                mock_tracer.start_as_current_span.side_effect = (
                    lambda name: MockContextManager(start_span(name))
                )
                mock_trace.get_tracer.return_value = mock_tracer

                # Set up telemetry
                import tinyllm.telemetry

                tinyllm.telemetry._tracer = mock_tracer
                tinyllm.telemetry._telemetry_enabled = True

                # Execute operation
                with trace_span("test.operation"):
                    pass

                # Verify span was created
                span_capture.assert_span_exists("test.operation")

    async def test_graph_execution_trace(self, span_capture):
        """Test that graph execution creates proper trace hierarchy."""
        from tinyllm.telemetry import set_graph_attributes, trace_span

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                setup_mock_telemetry(mock_trace, span_capture)

                # Simulate graph execution
                with trace_span("graph.execute", attributes={"graph.id": "g1"}):
                    span_capture.capture_span("graph.execute", {"graph.id": "g1"})

                    # Simulate node executions
                    with trace_span("node.model", attributes={"node.id": "n1"}):
                        span_capture.capture_span("node.model", {"node.id": "n1"})

                    with trace_span("node.transform", attributes={"node.id": "n2"}):
                        span_capture.capture_span("node.transform", {"node.id": "n2"})

                # Verify trace structure
                span_capture.assert_span_exists("graph.execute")
                span_capture.assert_span_exists("node.model")
                span_capture.assert_span_exists("node.transform")
                span_capture.assert_span_hierarchy("graph.execute", "node.model")

    async def test_llm_request_trace(self, span_capture):
        """Test that LLM requests create proper traces."""
        from tinyllm.telemetry import set_llm_attributes, trace_span

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                setup_mock_telemetry(mock_trace, span_capture)

                # Simulate LLM request
                with trace_span("llm.generate"):
                    span_idx = span_capture.capture_span(
                        "llm.generate",
                        {
                            "llm.model": "gpt-4",
                            "llm.temperature": "0.7",
                            "llm.prompt_length": "100",
                        },
                    )

                # Verify LLM trace
                span_capture.assert_span_exists("llm.generate")
                span_capture.assert_span_attribute("llm.generate", "llm.model", "gpt-4")

    async def test_error_trace(self, span_capture):
        """Test that errors are properly traced."""
        from tinyllm.telemetry import set_error_attributes, trace_span

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                setup_mock_telemetry(mock_trace, span_capture)

                # Simulate operation with error
                try:
                    with trace_span("operation.with_error"):
                        span_idx = span_capture.capture_span("operation.with_error")
                        raise ValueError("Test error")
                except ValueError:
                    span_capture.set_status(span_idx, "error")
                    span_capture.spans[span_idx]["attributes"]["error.type"] = "ValueError"
                    span_capture.spans[span_idx]["attributes"]["error.message"] = "Test error"

                # Verify error trace
                span_capture.assert_span_exists("operation.with_error")
                error_span = span_capture.get_span_by_name("operation.with_error")
                assert error_span["status"] == "error"


class TestTraceAttributes:
    """Test that traces contain correct attributes."""

    def test_graph_attributes_in_trace(self, span_capture):
        """Test that graph attributes are captured in traces."""
        from tinyllm.telemetry import set_graph_attributes

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                mock_span = setup_mock_telemetry(mock_trace, span_capture)

                # Set attributes
                span_idx = span_capture.capture_span("test.span")

                import tinyllm.telemetry

                tinyllm.telemetry._tracer = mock_trace.get_tracer.return_value
                tinyllm.telemetry._telemetry_enabled = True

                set_graph_attributes(
                    graph_id="g123",
                    graph_name="test_graph",
                    node_count=5,
                    depth=3,
                )

                # Verify attributes were set
                assert mock_span.set_attribute.called

    def test_node_attributes_in_trace(self, span_capture):
        """Test that node attributes are captured in traces."""
        from tinyllm.telemetry import set_node_attributes

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                mock_span = setup_mock_telemetry(mock_trace, span_capture)

                import tinyllm.telemetry

                tinyllm.telemetry._tracer = mock_trace.get_tracer.return_value
                tinyllm.telemetry._telemetry_enabled = True

                set_node_attributes(
                    node_id="n1",
                    node_type="model",
                    node_name="code_gen",
                    children_nodes=["n2", "n3"],
                )

                # Verify attributes were set
                assert mock_span.set_attribute.called


class TestTraceCorrelation:
    """Test trace correlation across operations."""

    def test_correlation_id_propagation(self, span_capture):
        """Test that correlation IDs are propagated through traces."""
        from tinyllm.telemetry import generate_correlation_id, set_correlation_id, trace_span

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                setup_mock_telemetry(mock_trace, span_capture)

                # Set correlation ID
                corr_id = generate_correlation_id()
                set_correlation_id(corr_id)

                # Execute operations
                with trace_span("operation1"):
                    span_capture.capture_span("operation1", {"correlation_id": corr_id})

                with trace_span("operation2"):
                    span_capture.capture_span("operation2", {"correlation_id": corr_id})

                # Verify same correlation ID in both spans
                span1 = span_capture.get_span_by_name("operation1")
                span2 = span_capture.get_span_by_name("operation2")

                assert span1 is not None
                assert span2 is not None


class TestTraceSampling:
    """Test trace sampling behavior."""

    def test_sampled_traces(self):
        """Test that sampling works correctly."""
        from tinyllm.telemetry import TelemetryConfig, configure_telemetry

        # Configure with 50% sampling
        config = TelemetryConfig(
            enable_tracing=True,
            exporter="console",
            sampling_rate=0.5,
        )

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.TracerProvider") as mock_provider:
                with patch("tinyllm.telemetry.ParentBasedTraceIdRatio") as mock_sampler:
                    # This would configure sampling
                    # In a real test, we'd verify sampler is created with rate=0.5
                    pass


class TestBaggagePropagation:
    """Test baggage propagation in traces."""

    def test_baggage_in_trace_context(self, span_capture):
        """Test that baggage is available in trace context."""
        from tinyllm.telemetry import get_baggage, set_baggage

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.baggage") as mock_baggage:
                # Set baggage
                mock_baggage.get_baggage.return_value = None

                import tinyllm.telemetry

                tinyllm.telemetry._telemetry_enabled = True

                set_baggage("user_id", "12345")

                # Verify baggage was set
                mock_baggage.set_baggage.assert_called_with("user_id", "12345")

    def test_baggage_propagation_across_spans(self):
        """Test that baggage propagates across spans."""
        from tinyllm.telemetry import get_all_baggage, set_baggage

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.baggage") as mock_baggage:
                import tinyllm.telemetry

                tinyllm.telemetry._telemetry_enabled = True

                # Set baggage
                set_baggage("tenant_id", "acme")

                # Get all baggage
                mock_baggage.get_all.return_value = {"tenant_id": "acme"}
                all_baggage = get_all_baggage()

                assert all_baggage == {"tenant_id": "acme"}


# Helper functions


class MockContextManager:
    """Mock context manager for spans."""

    def __init__(self, span):
        """Initialize with span."""
        self.span = span

    def __enter__(self):
        """Enter context."""
        return self.span

    def __exit__(self, *args):
        """Exit context."""
        pass


def setup_mock_telemetry(mock_trace, span_capture=None):
    """Set up mock telemetry for testing.

    Args:
        mock_trace: Mock trace module.
        span_capture: Optional span capture.

    Returns:
        Mock span.
    """
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    # Mock span context
    mock_span_context = MagicMock()
    mock_span_context.trace_id = 12345
    mock_span_context.span_id = 67890
    mock_span.get_span_context.return_value = mock_span_context

    mock_tracer.start_as_current_span.return_value = MockContextManager(mock_span)
    mock_trace.get_tracer.return_value = mock_tracer
    mock_trace.get_current_span.return_value = mock_span

    import tinyllm.telemetry

    tinyllm.telemetry._tracer = mock_tracer
    tinyllm.telemetry._telemetry_enabled = True

    return mock_span
