"""OpenTelemetry distributed tracing for TinyLLM.

This module provides OpenTelemetry integration for distributed tracing across
TinyLLM's graph execution, node operations, and LLM API calls.
"""

import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, cast

from pydantic import BaseModel, Field

from tinyllm.logging import bind_context, get_logger, unbind_context

# OpenTelemetry imports (optional dependencies)
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio
    from opentelemetry.trace import Status, StatusCode, Tracer

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create stub types for type checking when OTel not installed
    trace = None  # type: ignore
    Tracer = Any  # type: ignore
    Status = Any  # type: ignore
    StatusCode = Any  # type: ignore

logger = get_logger(__name__, component="telemetry")

# Global tracer instance
_tracer: Optional[Tracer] = None
_telemetry_enabled = False


class TelemetryConfig(BaseModel):
    """Configuration for OpenTelemetry tracing."""

    model_config = {"extra": "forbid"}

    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    service_name: str = Field(default="tinyllm", description="Service name in traces")
    exporter: str = Field(
        default="console",
        description="Exporter type: console, otlp, or jaeger",
    )
    otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OTLP exporter endpoint (e.g., http://localhost:4317)",
    )
    sampling_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 to 1.0)",
    )


def configure_telemetry(config: TelemetryConfig) -> None:
    """Configure OpenTelemetry tracer provider and exporters.

    Args:
        config: Telemetry configuration.

    Raises:
        ImportError: If OpenTelemetry is not installed.
        ValueError: If invalid exporter type specified.
    """
    global _tracer, _telemetry_enabled

    if not config.enable_tracing:
        logger.info("telemetry_disabled")
        _telemetry_enabled = False
        return

    if not OTEL_AVAILABLE:
        logger.warning(
            "telemetry_unavailable",
            message="OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp",
        )
        _telemetry_enabled = False
        return

    logger.info(
        "telemetry_configuration_started",
        service_name=config.service_name,
        exporter=config.exporter,
        sampling_rate=config.sampling_rate,
    )

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": "0.1.0",
            "deployment.environment": "development",
        }
    )

    # Configure sampling
    sampler = ParentBasedTraceIdRatio(config.sampling_rate)

    # Create tracer provider
    provider = TracerProvider(resource=resource, sampler=sampler)

    # Configure exporter
    if config.exporter == "console":
        exporter = ConsoleSpanExporter()
        logger.info("telemetry_exporter_console")
    elif config.exporter == "otlp":
        if not config.otlp_endpoint:
            raise ValueError("otlp_endpoint required for OTLP exporter")
        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
        logger.info("telemetry_exporter_otlp", endpoint=config.otlp_endpoint)
    elif config.exporter == "jaeger":
        # Jaeger uses OTLP protocol now
        endpoint = config.otlp_endpoint or "http://localhost:4317"
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        logger.info("telemetry_exporter_jaeger", endpoint=endpoint)
    else:
        raise ValueError(f"Unknown exporter type: {config.exporter}")

    # Add span processor
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer instance
    _tracer = trace.get_tracer(__name__)
    _telemetry_enabled = True

    logger.info(
        "telemetry_configured",
        service_name=config.service_name,
        exporter=config.exporter,
        sampling_rate=config.sampling_rate,
    )


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled and available.

    Returns:
        True if telemetry is configured and available.
    """
    return _telemetry_enabled and OTEL_AVAILABLE


def get_tracer() -> Optional[Tracer]:
    """Get the configured tracer instance.

    Returns:
        Tracer instance if telemetry is enabled, None otherwise.
    """
    if not is_telemetry_enabled():
        return None
    return _tracer


@contextmanager
def trace_span(
    name: str,
    *,
    attributes: Optional[dict[str, Any]] = None,
    add_to_logs: bool = True,
):
    """Context manager for creating a trace span.

    Args:
        name: Span name.
        attributes: Optional span attributes.
        add_to_logs: Whether to add trace ID to log context.

    Yields:
        The active span (or None if tracing disabled).

    Example:
        >>> with trace_span("node_execution", attributes={"node_id": "entry"}):
        ...     # Your code here
        ...     pass
    """
    if not is_telemetry_enabled() or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))

        # Add trace ID to logs
        if add_to_logs:
            trace_id = format(span.get_span_context().trace_id, "032x")
            span_id = format(span.get_span_context().span_id, "016x")
            bind_context(trace_id=trace_id, span_id=span_id)

        try:
            yield span
        except Exception as e:
            # Record exception in span
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            # Clean up log context
            if add_to_logs:
                unbind_context("trace_id", "span_id")


def record_span_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Record an event in the current span.

    Args:
        name: Event name.
        attributes: Optional event attributes.
    """
    if not is_telemetry_enabled():
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current span.

    Args:
        key: Attribute key.
        value: Attribute value.
    """
    if not is_telemetry_enabled():
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, str(value))


def set_span_error(error: Exception) -> None:
    """Mark the current span as error and record exception.

    Args:
        error: Exception to record.
    """
    if not is_telemetry_enabled():
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)


# Type variable for generic decorator typing
F = TypeVar("F", bound=Callable[..., Any])


def traced(
    span_name: Optional[str] = None,
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to add tracing to async functions.

    Args:
        span_name: Optional span name (defaults to function name).
        attributes: Optional static attributes to add to span.

    Returns:
        Decorated function with tracing.

    Example:
        >>> @traced(attributes={"component": "executor"})
        ... async def execute_task(task_id: str):
        ...     pass
    """

    def decorator(func: F) -> F:
        name = span_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_telemetry_enabled():
                return await func(*args, **kwargs)

            with trace_span(name, attributes=attributes) as span:
                # Add function arguments as attributes
                if span and span.is_recording():
                    # Add positional args
                    for i, arg in enumerate(args):
                        if i < 3:  # Limit to first 3 args
                            span.set_attribute(f"arg.{i}", str(arg)[:100])

                    # Add keyword args
                    for key, value in kwargs.items():
                        span.set_attribute(f"arg.{key}", str(value)[:100])

                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if span and span.is_recording():
                    span.set_attribute("duration_ms", round(elapsed_ms, 2))

                return result

        return cast(F, wrapper)

    return decorator


def traced_method(
    span_name: Optional[str] = None,
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to add tracing to async class methods.

    Args:
        span_name: Optional span name (defaults to class.method name).
        attributes: Optional static attributes to add to span.

    Returns:
        Decorated method with tracing.

    Example:
        >>> class Executor:
        ...     @traced_method(attributes={"component": "executor"})
        ...     async def execute(self, task):
        ...         pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not is_telemetry_enabled():
                return await func(self, *args, **kwargs)

            # Build span name with class name
            name = span_name or f"{self.__class__.__name__}.{func.__name__}"

            with trace_span(name, attributes=attributes) as span:
                # Add method arguments as attributes
                if span and span.is_recording():
                    # Add positional args (skip self)
                    for i, arg in enumerate(args):
                        if i < 3:
                            span.set_attribute(f"arg.{i}", str(arg)[:100])

                    # Add keyword args
                    for key, value in kwargs.items():
                        span.set_attribute(f"arg.{key}", str(value)[:100])

                start_time = time.perf_counter()
                result = await func(self, *args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if span and span.is_recording():
                    span.set_attribute("duration_ms", round(elapsed_ms, 2))

                return result

        return cast(F, wrapper)

    return decorator


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID if available.

    Returns:
        Trace ID as hex string, or None if not in a trace context.
    """
    if not is_telemetry_enabled():
        return None

    span = trace.get_current_span()
    if span and span.is_recording():
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_current_span_id() -> Optional[str]:
    """Get the current span ID if available.

    Returns:
        Span ID as hex string, or None if not in a trace context.
    """
    if not is_telemetry_enabled():
        return None

    span = trace.get_current_span()
    if span and span.is_recording():
        return format(span.get_span_context().span_id, "016x")
    return None


# Convenience functions for common trace patterns

def trace_executor_execution(trace_id: str, graph_id: str, task_content: str):
    """Create a span for graph execution.

    Args:
        trace_id: Execution trace ID.
        graph_id: Graph ID being executed.
        task_content: Task content (truncated).

    Returns:
        Context manager for the span.
    """
    return trace_span(
        "graph.execute",
        attributes={
            "graph.id": graph_id,
            "execution.trace_id": trace_id,
            "task.content": task_content[:100],
        },
    )


def trace_node_execution(node_id: str, node_type: str, step: int):
    """Create a span for node execution.

    Args:
        node_id: Node ID being executed.
        node_type: Node type.
        step: Execution step number.

    Returns:
        Context manager for the span.
    """
    return trace_span(
        f"node.{node_type}",
        attributes={
            "node.id": node_id,
            "node.type": node_type,
            "execution.step": step,
        },
    )


def trace_llm_request(model: str, prompt_length: int, temperature: float):
    """Create a span for LLM API request.

    Args:
        model: Model name.
        prompt_length: Length of prompt in characters.
        temperature: Sampling temperature.

    Returns:
        Context manager for the span.
    """
    return trace_span(
        "llm.generate",
        attributes={
            "llm.model": model,
            "llm.prompt_length": prompt_length,
            "llm.temperature": temperature,
        },
    )


__all__ = [
    "TelemetryConfig",
    "configure_telemetry",
    "is_telemetry_enabled",
    "get_tracer",
    "trace_span",
    "traced",
    "traced_method",
    "record_span_event",
    "set_span_attribute",
    "set_span_error",
    "get_current_trace_id",
    "get_current_span_id",
    "trace_executor_execution",
    "trace_node_execution",
    "trace_llm_request",
]
