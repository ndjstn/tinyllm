"""Trace-based testing utilities for TinyLLM.

This module provides utilities for testing distributed tracing behavior,
validating span creation, attributes, and trace propagation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tinyllm.logging import get_logger
from tinyllm.telemetry import is_telemetry_enabled

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    ReadableSpan = Any  # type: ignore
    TracerProvider = Any  # type: ignore
    SimpleSpanProcessor = Any  # type: ignore
    SpanExporter = Any  # type: ignore
    SpanExportResult = Any  # type: ignore

logger = get_logger(__name__, component="trace_testing")


@dataclass
class CapturedSpan:
    """A captured span for testing."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status_code: Optional[str] = None
    status_message: Optional[str] = None
    start_time: int = 0
    end_time: int = 0

    @property
    def duration_ms(self) -> float:
        """Calculate span duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) / 1_000_000  # nanoseconds to ms
        return 0.0

    def has_attribute(self, key: str, value: Optional[Any] = None) -> bool:
        """Check if span has an attribute.

        Args:
            key: Attribute key to check.
            value: Expected value (if None, just checks existence).

        Returns:
            True if attribute exists (and matches value if provided).
        """
        if key not in self.attributes:
            return False
        if value is not None:
            return str(self.attributes[key]) == str(value)
        return True

    def has_event(self, name: str) -> bool:
        """Check if span has an event with given name.

        Args:
            name: Event name to check.

        Returns:
            True if event exists.
        """
        return any(event.get("name") == name for event in self.events)

    def is_error(self) -> bool:
        """Check if span represents an error.

        Returns:
            True if span status is ERROR.
        """
        return self.status_code == "ERROR"


class InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for testing.

    This exporter captures spans in memory for later assertion
    in tests, rather than sending them to a backend.
    """

    def __init__(self):
        """Initialize the in-memory exporter."""
        self.spans: list[CapturedSpan] = []
        self._exported_count = 0

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        """Export spans to memory.

        Args:
            spans: List of spans to export.

        Returns:
            SpanExportResult.SUCCESS
        """
        if not OTEL_AVAILABLE:
            return SpanExportResult.FAILURE

        for span in spans:
            # Convert OpenTelemetry span to CapturedSpan
            span_context = span.get_span_context()

            captured = CapturedSpan(
                name=span.name,
                trace_id=format(span_context.trace_id, "032x"),
                span_id=format(span_context.span_id, "016x"),
                parent_span_id=(
                    format(span.parent.span_id, "016x") if span.parent else None
                ),
                attributes={k: str(v) for k, v in (span.attributes or {}).items()},
                events=[
                    {"name": event.name, "attributes": dict(event.attributes or {})}
                    for event in (span.events or [])
                ],
                status_code=span.status.status_code.name if span.status else None,
                status_message=span.status.description if span.status else None,
                start_time=span.start_time or 0,
                end_time=span.end_time or 0,
            )

            self.spans.append(captured)
            self._exported_count += 1

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if successful.
        """
        return True

    def clear(self):
        """Clear all captured spans."""
        self.spans.clear()
        self._exported_count = 0

    def get_spans(
        self,
        *,
        name: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> list[CapturedSpan]:
        """Get captured spans with optional filtering.

        Args:
            name: Filter by span name.
            trace_id: Filter by trace ID.

        Returns:
            List of matching captured spans.
        """
        spans = self.spans

        if name:
            spans = [s for s in spans if s.name == name]

        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]

        return spans

    def get_span(self, name: str) -> Optional[CapturedSpan]:
        """Get the first span with given name.

        Args:
            name: Span name to find.

        Returns:
            First matching span or None.
        """
        for span in self.spans:
            if span.name == name:
                return span
        return None

    def assert_span_exists(self, name: str, message: Optional[str] = None):
        """Assert that a span with given name exists.

        Args:
            name: Span name to check.
            message: Optional assertion message.

        Raises:
            AssertionError: If span doesn't exist.
        """
        if not self.get_span(name):
            msg = message or f"Expected span '{name}' not found. Available spans: {[s.name for s in self.spans]}"
            raise AssertionError(msg)

    def assert_span_count(self, expected: int, message: Optional[str] = None):
        """Assert expected number of spans.

        Args:
            expected: Expected span count.
            message: Optional assertion message.

        Raises:
            AssertionError: If count doesn't match.
        """
        actual = len(self.spans)
        if actual != expected:
            msg = message or f"Expected {expected} spans, got {actual}. Spans: {[s.name for s in self.spans]}"
            raise AssertionError(msg)

    def assert_span_attribute(
        self,
        span_name: str,
        attribute_key: str,
        expected_value: Optional[Any] = None,
        message: Optional[str] = None,
    ):
        """Assert span has expected attribute.

        Args:
            span_name: Name of span to check.
            attribute_key: Attribute key to check.
            expected_value: Expected attribute value (if None, just checks existence).
            message: Optional assertion message.

        Raises:
            AssertionError: If span or attribute doesn't match expectations.
        """
        span = self.get_span(span_name)
        if not span:
            raise AssertionError(f"Span '{span_name}' not found")

        if not span.has_attribute(attribute_key, expected_value):
            if expected_value is not None:
                msg = (
                    message
                    or f"Span '{span_name}' attribute '{attribute_key}' = '{span.attributes.get(attribute_key)}', expected '{expected_value}'"
                )
            else:
                msg = (
                    message
                    or f"Span '{span_name}' missing attribute '{attribute_key}'. Available: {list(span.attributes.keys())}"
                )
            raise AssertionError(msg)

    def assert_span_event(
        self,
        span_name: str,
        event_name: str,
        message: Optional[str] = None,
    ):
        """Assert span has expected event.

        Args:
            span_name: Name of span to check.
            event_name: Event name to check.
            message: Optional assertion message.

        Raises:
            AssertionError: If span or event doesn't exist.
        """
        span = self.get_span(span_name)
        if not span:
            raise AssertionError(f"Span '{span_name}' not found")

        if not span.has_event(event_name):
            msg = (
                message
                or f"Span '{span_name}' missing event '{event_name}'. Events: {[e['name'] for e in span.events]}"
            )
            raise AssertionError(msg)

    def assert_no_errors(self, message: Optional[str] = None):
        """Assert that no spans have error status.

        Args:
            message: Optional assertion message.

        Raises:
            AssertionError: If any span has error status.
        """
        error_spans = [s for s in self.spans if s.is_error()]
        if error_spans:
            msg = message or f"Found {len(error_spans)} error spans: {[s.name for s in error_spans]}"
            raise AssertionError(msg)


class TraceTestContext:
    """Context for trace-based testing.

    Provides utilities for capturing and asserting on traces during tests.
    """

    def __init__(self):
        """Initialize trace test context."""
        self.exporter = InMemorySpanExporter()
        self._original_provider = None
        self._test_provider = None

    def setup(self):
        """Set up trace testing (call in test setup/fixture)."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, trace testing disabled")
            return

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Create test tracer provider
        self._test_provider = TracerProvider()
        self._test_provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        # Save original provider and set test provider
        self._original_provider = trace.get_tracer_provider()
        trace.set_tracer_provider(self._test_provider)

        logger.debug("trace_test_context_setup")

    def teardown(self):
        """Tear down trace testing (call in test teardown/fixture)."""
        if not OTEL_AVAILABLE:
            return

        # Restore original provider
        if self._original_provider:
            trace.set_tracer_provider(self._original_provider)

        self.exporter.clear()
        logger.debug("trace_test_context_teardown")

    def get_spans(self, **filters) -> list[CapturedSpan]:
        """Get captured spans with optional filters.

        Args:
            **filters: Filtering criteria (name, trace_id, etc.).

        Returns:
            List of captured spans.
        """
        return self.exporter.get_spans(**filters)

    def get_span(self, name: str) -> Optional[CapturedSpan]:
        """Get first span with given name.

        Args:
            name: Span name.

        Returns:
            Captured span or None.
        """
        return self.exporter.get_span(name)

    def assert_span_exists(self, name: str):
        """Assert span exists.

        Args:
            name: Span name.
        """
        self.exporter.assert_span_exists(name)

    def assert_span_count(self, expected: int):
        """Assert expected span count.

        Args:
            expected: Expected count.
        """
        self.exporter.assert_span_count(expected)

    def assert_span_attribute(
        self,
        span_name: str,
        attribute_key: str,
        expected_value: Optional[Any] = None,
    ):
        """Assert span attribute.

        Args:
            span_name: Span name.
            attribute_key: Attribute key.
            expected_value: Expected value.
        """
        self.exporter.assert_span_attribute(span_name, attribute_key, expected_value)

    def assert_span_event(self, span_name: str, event_name: str):
        """Assert span event.

        Args:
            span_name: Span name.
            event_name: Event name.
        """
        self.exporter.assert_span_event(span_name, event_name)

    def assert_no_errors(self):
        """Assert no error spans."""
        self.exporter.assert_no_errors()

    def clear(self):
        """Clear captured spans."""
        self.exporter.clear()


# Pytest fixtures (if pytest is available)
try:
    import pytest

    @pytest.fixture
    def trace_test():
        """Pytest fixture for trace testing.

        Example:
            >>> def test_my_function(trace_test):
            ...     # Your code that creates spans
            ...     with trace_span("test.span"):
            ...         pass
            ...
            ...     # Assert on traces
            ...     trace_test.assert_span_exists("test.span")
        """
        ctx = TraceTestContext()
        ctx.setup()
        yield ctx
        ctx.teardown()

except ImportError:
    # pytest not available, skip fixture
    pass


__all__ = [
    "CapturedSpan",
    "InMemorySpanExporter",
    "TraceTestContext",
]
