"""Service mesh integration for TinyLLM.

This module provides integration with service meshes like Istio and Linkerd
for distributed tracing, header propagation, and traffic management.
"""

from contextmanager import contextmanager
from typing import Any, Optional

from tinyllm.logging import get_logger
from tinyllm.telemetry import is_telemetry_enabled

try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore

logger = get_logger(__name__, component="service_mesh")


class ServiceMeshHeaders:
    """Standard headers for service mesh integration.

    These headers are used by service meshes like Istio and Linkerd
    for distributed tracing, routing, and traffic management.
    """

    # OpenTelemetry / W3C Trace Context
    TRACEPARENT = "traceparent"
    TRACESTATE = "tracestate"

    # Baggage
    BAGGAGE = "baggage"

    # Istio headers
    ISTIO_REQUEST_ID = "x-request-id"
    ISTIO_B3_TRACE_ID = "x-b3-traceid"
    ISTIO_B3_SPAN_ID = "x-b3-spanid"
    ISTIO_B3_PARENT_SPAN_ID = "x-b3-parentspanid"
    ISTIO_B3_SAMPLED = "x-b3-sampled"
    ISTIO_B3_FLAGS = "x-b3-flags"
    ISTIO_OT_SPAN_CONTEXT = "x-ot-span-context"

    # Linkerd headers
    LINKERD_CONTEXT_DEADLINE = "l5d-ctx-deadline"
    LINKERD_CONTEXT_TRACE = "l5d-ctx-trace"
    LINKERD_REQUEST_ID = "l5d-reqid"

    # All headers that should be propagated
    ALL_PROPAGATION_HEADERS = [
        TRACEPARENT,
        TRACESTATE,
        BAGGAGE,
        ISTIO_REQUEST_ID,
        ISTIO_B3_TRACE_ID,
        ISTIO_B3_SPAN_ID,
        ISTIO_B3_PARENT_SPAN_ID,
        ISTIO_B3_SAMPLED,
        ISTIO_B3_FLAGS,
        ISTIO_OT_SPAN_CONTEXT,
        LINKERD_CONTEXT_DEADLINE,
        LINKERD_CONTEXT_TRACE,
        LINKERD_REQUEST_ID,
    ]


def inject_service_mesh_headers(
    headers: Optional[dict[str, str]] = None,
    *,
    include_b3: bool = True,
    include_linkerd: bool = False,
) -> dict[str, str]:
    """Inject service mesh headers for distributed tracing.

    This function adds tracing headers compatible with Istio, Linkerd,
    and other service meshes. It propagates both W3C Trace Context
    and legacy formats like B3.

    Args:
        headers: Existing headers dictionary (optional).
        include_b3: Include B3 propagation headers for Istio (default: True).
        include_linkerd: Include Linkerd-specific headers (default: False).

    Returns:
        Headers with service mesh tracing headers injected.

    Example:
        >>> headers = inject_service_mesh_headers(
        ...     {"Content-Type": "application/json"},
        ...     include_b3=True
        ... )
        >>> # Make HTTP request with headers
    """
    if headers is None:
        headers = {}

    if not is_telemetry_enabled():
        return headers

    try:
        # Inject W3C Trace Context (traceparent, tracestate, baggage)
        from opentelemetry.propagate import inject

        inject(headers)

        span = trace.get_current_span()
        if span and span.is_recording():
            span_context = span.get_span_context()

            if span_context and span_context.is_valid:
                trace_id = format(span_context.trace_id, "032x")
                span_id = format(span_context.span_id, "016x")

                # Add B3 headers for Istio compatibility
                if include_b3:
                    headers[ServiceMeshHeaders.ISTIO_B3_TRACE_ID] = trace_id
                    headers[ServiceMeshHeaders.ISTIO_B3_SPAN_ID] = span_id

                    # B3 sampled flag (1 = sampled, 0 = not sampled)
                    sampled = "1" if span_context.trace_flags.sampled else "0"
                    headers[ServiceMeshHeaders.ISTIO_B3_SAMPLED] = sampled

                    # Request ID (use span ID)
                    headers[ServiceMeshHeaders.ISTIO_REQUEST_ID] = span_id

                # Add Linkerd headers if requested
                if include_linkerd:
                    # Linkerd trace context (simplified)
                    headers[ServiceMeshHeaders.LINKERD_CONTEXT_TRACE] = f"{trace_id}/{span_id}"
                    headers[ServiceMeshHeaders.LINKERD_REQUEST_ID] = span_id

        logger.debug(
            "service_mesh_headers_injected",
            header_count=len(headers),
            include_b3=include_b3,
            include_linkerd=include_linkerd,
        )

    except Exception as e:
        logger.warning("service_mesh_header_injection_failed", error=str(e))

    return headers


def extract_service_mesh_headers(
    headers: dict[str, str],
    *,
    set_as_current: bool = True,
) -> Optional[dict[str, Any]]:
    """Extract service mesh headers from incoming request.

    This function extracts tracing context from service mesh headers
    and optionally sets it as the current trace context.

    Args:
        headers: Incoming headers dictionary.
        set_as_current: Whether to set extracted context as current (default: True).

    Returns:
        Dictionary with extracted trace information, or None if not available.

    Example:
        >>> trace_info = extract_service_mesh_headers(request.headers)
        >>> if trace_info:
        ...     print(f"Trace ID: {trace_info['trace_id']}")
    """
    if not is_telemetry_enabled():
        return None

    try:
        # Normalize header keys to lowercase
        normalized_headers = {k.lower(): v for k, v in headers.items()}

        trace_info = {}

        # Extract W3C Trace Context
        from opentelemetry.propagate import extract

        if set_as_current:
            # This sets the context as current
            ctx = extract(headers)

        # Try to extract trace ID and span ID from various header formats

        # 1. Try W3C traceparent header (format: 00-trace_id-span_id-flags)
        if ServiceMeshHeaders.TRACEPARENT in normalized_headers:
            traceparent = normalized_headers[ServiceMeshHeaders.TRACEPARENT]
            parts = traceparent.split("-")
            if len(parts) >= 4:
                trace_info["trace_id"] = parts[1]
                trace_info["span_id"] = parts[2]
                trace_info["sampled"] = parts[3] == "01"
                trace_info["format"] = "w3c"

        # 2. Try B3 headers (Istio)
        elif ServiceMeshHeaders.ISTIO_B3_TRACE_ID in normalized_headers:
            trace_info["trace_id"] = normalized_headers[ServiceMeshHeaders.ISTIO_B3_TRACE_ID]
            trace_info["span_id"] = normalized_headers.get(ServiceMeshHeaders.ISTIO_B3_SPAN_ID, "")
            trace_info["parent_span_id"] = normalized_headers.get(
                ServiceMeshHeaders.ISTIO_B3_PARENT_SPAN_ID
            )
            trace_info["sampled"] = (
                normalized_headers.get(ServiceMeshHeaders.ISTIO_B3_SAMPLED) == "1"
            )
            trace_info["format"] = "b3"

        # 3. Try Linkerd headers
        elif ServiceMeshHeaders.LINKERD_CONTEXT_TRACE in normalized_headers:
            trace_context = normalized_headers[ServiceMeshHeaders.LINKERD_CONTEXT_TRACE]
            parts = trace_context.split("/")
            if len(parts) >= 2:
                trace_info["trace_id"] = parts[0]
                trace_info["span_id"] = parts[1]
                trace_info["format"] = "linkerd"

        # Extract request ID
        request_id = normalized_headers.get(
            ServiceMeshHeaders.ISTIO_REQUEST_ID
        ) or normalized_headers.get(ServiceMeshHeaders.LINKERD_REQUEST_ID)
        if request_id:
            trace_info["request_id"] = request_id

        if trace_info:
            logger.debug(
                "service_mesh_headers_extracted",
                trace_id=trace_info.get("trace_id", "")[:16],
                format=trace_info.get("format", "unknown"),
            )
            return trace_info

        return None

    except Exception as e:
        logger.warning("service_mesh_header_extraction_failed", error=str(e))
        return None


def propagate_service_mesh_context(
    source_headers: dict[str, str],
    target_headers: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Propagate service mesh context from source to target headers.

    This is useful for forwarding trace context when making downstream
    service calls in a service mesh.

    Args:
        source_headers: Headers from incoming request.
        target_headers: Headers for outgoing request (optional).

    Returns:
        Target headers with propagated context.

    Example:
        >>> # In a service handler
        >>> incoming_headers = request.headers
        >>>
        >>> # Making a downstream call
        >>> outgoing_headers = propagate_service_mesh_context(
        ...     incoming_headers,
        ...     {"Content-Type": "application/json"}
        ... )
        >>> response = await http_client.get(url, headers=outgoing_headers)
    """
    if target_headers is None:
        target_headers = {}

    # Normalize header keys
    normalized_source = {k.lower(): v for k, v in source_headers.items()}

    # Copy all propagation headers from source to target
    for header in ServiceMeshHeaders.ALL_PROPAGATION_HEADERS:
        if header in normalized_source:
            target_headers[header] = normalized_source[header]

    logger.debug(
        "service_mesh_context_propagated",
        propagated_header_count=sum(
            1 for h in ServiceMeshHeaders.ALL_PROPAGATION_HEADERS if h in normalized_source
        ),
    )

    return target_headers


@contextmanager
def service_mesh_context(headers: dict[str, str]):
    """Context manager for service mesh request handling.

    Extracts trace context from incoming headers and ensures proper
    propagation for downstream calls.

    Args:
        headers: Incoming request headers.

    Yields:
        Extracted trace information.

    Example:
        >>> with service_mesh_context(request.headers) as trace_info:
        ...     # Trace context is set for this block
        ...     if trace_info:
        ...         print(f"Processing request with trace ID: {trace_info['trace_id']}")
        ...
        ...     # Make downstream calls
        ...     downstream_headers = inject_service_mesh_headers()
        ...     await call_other_service(headers=downstream_headers)
    """
    trace_info = extract_service_mesh_headers(headers, set_as_current=True)

    try:
        yield trace_info
    finally:
        # Context cleanup is handled by OpenTelemetry
        pass


__all__ = [
    "ServiceMeshHeaders",
    "inject_service_mesh_headers",
    "extract_service_mesh_headers",
    "propagate_service_mesh_context",
    "service_mesh_context",
]
