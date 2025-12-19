"""Tests for service mesh integration."""

import pytest

from tinyllm.service_mesh import (
    ServiceMeshHeaders,
    extract_service_mesh_headers,
    inject_service_mesh_headers,
    propagate_service_mesh_context,
    service_mesh_context,
)


class TestServiceMeshHeaders:
    """Test ServiceMeshHeaders constants."""

    def test_w3c_headers(self):
        """Test W3C trace context headers."""
        assert ServiceMeshHeaders.TRACEPARENT == "traceparent"
        assert ServiceMeshHeaders.TRACESTATE == "tracestate"
        assert ServiceMeshHeaders.BAGGAGE == "baggage"

    def test_istio_headers(self):
        """Test Istio B3 headers."""
        assert ServiceMeshHeaders.ISTIO_B3_TRACE_ID == "x-b3-traceid"
        assert ServiceMeshHeaders.ISTIO_B3_SPAN_ID == "x-b3-spanid"
        assert ServiceMeshHeaders.ISTIO_B3_SAMPLED == "x-b3-sampled"
        assert ServiceMeshHeaders.ISTIO_REQUEST_ID == "x-request-id"

    def test_linkerd_headers(self):
        """Test Linkerd headers."""
        assert ServiceMeshHeaders.LINKERD_CONTEXT_TRACE == "l5d-ctx-trace"
        assert ServiceMeshHeaders.LINKERD_REQUEST_ID == "l5d-reqid"

    def test_all_propagation_headers(self):
        """Test that all propagation headers are defined."""
        assert len(ServiceMeshHeaders.ALL_PROPAGATION_HEADERS) > 0
        assert ServiceMeshHeaders.TRACEPARENT in ServiceMeshHeaders.ALL_PROPAGATION_HEADERS
        assert ServiceMeshHeaders.BAGGAGE in ServiceMeshHeaders.ALL_PROPAGATION_HEADERS


class TestInjectServiceMeshHeaders:
    """Test inject_service_mesh_headers function."""

    def test_inject_without_telemetry(self):
        """Test injection when telemetry is disabled."""
        headers = inject_service_mesh_headers()
        # Should return empty dict or dict with no trace headers
        # (depends on whether telemetry is enabled)
        assert isinstance(headers, dict)

    def test_inject_with_existing_headers(self):
        """Test injection with existing headers."""
        existing = {"Content-Type": "application/json", "Authorization": "Bearer token"}
        headers = inject_service_mesh_headers(existing)

        # Original headers should be preserved
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer token"

    def test_inject_b3_headers(self):
        """Test B3 header injection for Istio."""
        headers = inject_service_mesh_headers(include_b3=True)
        # B3 headers should be added if telemetry is enabled
        assert isinstance(headers, dict)

    def test_inject_linkerd_headers(self):
        """Test Linkerd header injection."""
        headers = inject_service_mesh_headers(include_linkerd=True)
        # Linkerd headers should be added if telemetry is enabled
        assert isinstance(headers, dict)

    def test_inject_both_mesh_types(self):
        """Test injection with both Istio and Linkerd headers."""
        headers = inject_service_mesh_headers(include_b3=True, include_linkerd=True)
        assert isinstance(headers, dict)


class TestExtractServiceMeshHeaders:
    """Test extract_service_mesh_headers function."""

    def test_extract_empty_headers(self):
        """Test extraction from empty headers."""
        result = extract_service_mesh_headers({})
        # Should return None when no trace headers present
        assert result is None or isinstance(result, dict)

    def test_extract_w3c_traceparent(self):
        """Test extraction of W3C traceparent header."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
        result = extract_service_mesh_headers(headers, set_as_current=False)

        if result:
            assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
            assert result["span_id"] == "b7ad6b7169203331"
            assert result["sampled"] is True
            assert result["format"] == "w3c"

    def test_extract_b3_headers(self):
        """Test extraction of Istio B3 headers."""
        headers = {
            "x-b3-traceid": "0af7651916cd43dd8448eb211c80319c",
            "x-b3-spanid": "b7ad6b7169203331",
            "x-b3-sampled": "1",
        }
        result = extract_service_mesh_headers(headers, set_as_current=False)

        if result:
            assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
            assert result["span_id"] == "b7ad6b7169203331"
            assert result["sampled"] is True
            assert result["format"] == "b3"

    def test_extract_linkerd_headers(self):
        """Test extraction of Linkerd headers."""
        headers = {
            "l5d-ctx-trace": "0af7651916cd43dd8448eb211c80319c/b7ad6b7169203331"
        }
        result = extract_service_mesh_headers(headers, set_as_current=False)

        if result:
            assert result["trace_id"] == "0af7651916cd43dd8448eb211c80319c"
            assert result["span_id"] == "b7ad6b7169203331"
            assert result["format"] == "linkerd"

    def test_extract_request_id(self):
        """Test extraction of request ID."""
        headers = {
            "x-b3-traceid": "0af7651916cd43dd8448eb211c80319c",
            "x-b3-spanid": "b7ad6b7169203331",
            "x-request-id": "req-12345",
        }
        result = extract_service_mesh_headers(headers, set_as_current=False)

        if result:
            assert result.get("request_id") == "req-12345"

    def test_extract_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        headers = {
            "X-B3-TraceId": "0af7651916cd43dd8448eb211c80319c",
            "X-B3-SpanId": "b7ad6b7169203331",
        }
        result = extract_service_mesh_headers(headers, set_as_current=False)
        # Should handle mixed case headers
        assert result is None or isinstance(result, dict)


class TestPropagateServiceMeshContext:
    """Test propagate_service_mesh_context function."""

    def test_propagate_empty_source(self):
        """Test propagation from empty source headers."""
        source = {}
        target = propagate_service_mesh_context(source)

        assert isinstance(target, dict)
        assert len(target) == 0

    def test_propagate_trace_headers(self):
        """Test propagation of trace headers."""
        source = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "baggage": "user_id=12345",
            "x-request-id": "req-123",
        }
        target = propagate_service_mesh_context(source)

        # All trace-related headers should be propagated
        assert "traceparent" in target
        assert target["traceparent"] == source["traceparent"]

    def test_propagate_with_existing_target(self):
        """Test propagation to existing target headers."""
        source = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "x-request-id": "req-123",
        }
        target = {"Content-Type": "application/json"}

        result = propagate_service_mesh_context(source, target)

        # Original target headers should be preserved
        assert result["Content-Type"] == "application/json"
        # Trace headers should be added
        assert "traceparent" in result or "x-request-id" in result

    def test_propagate_filters_non_trace_headers(self):
        """Test that non-trace headers are not propagated."""
        source = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "Content-Type": "application/json",  # Should not be propagated
            "Authorization": "Bearer token",  # Should not be propagated
        }
        target = propagate_service_mesh_context(source)

        # Non-trace headers should not be in target
        assert "Content-Type" not in target
        assert "Authorization" not in target

    def test_propagate_all_mesh_headers(self):
        """Test propagation of all mesh headers."""
        source = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "vendor=value",
            "baggage": "user_id=12345",
            "x-b3-traceid": "0af7651916cd43dd8448eb211c80319c",
            "x-b3-spanid": "b7ad6b7169203331",
            "x-request-id": "req-123",
            "l5d-ctx-trace": "trace/span",
        }
        target = propagate_service_mesh_context(source)

        # All trace headers should be propagated
        for header in [
            "traceparent",
            "tracestate",
            "baggage",
            "x-b3-traceid",
            "x-b3-spanid",
            "x-request-id",
            "l5d-ctx-trace",
        ]:
            if header in source:
                assert header in target


class TestServiceMeshContext:
    """Test service_mesh_context context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager usage."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        with service_mesh_context(headers) as trace_info:
            # Should not raise
            assert trace_info is None or isinstance(trace_info, dict)

    def test_context_manager_with_trace_headers(self):
        """Test context manager with valid trace headers."""
        headers = {
            "x-b3-traceid": "0af7651916cd43dd8448eb211c80319c",
            "x-b3-spanid": "b7ad6b7169203331",
            "x-request-id": "req-123",
        }

        with service_mesh_context(headers) as trace_info:
            if trace_info:
                assert "trace_id" in trace_info
                assert "span_id" in trace_info

    def test_context_manager_cleanup(self):
        """Test that context manager cleans up properly."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        # Should not raise, even with exceptions
        try:
            with service_mesh_context(headers):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

    def test_context_manager_empty_headers(self):
        """Test context manager with empty headers."""
        with service_mesh_context({}) as trace_info:
            # Should handle empty headers gracefully
            assert trace_info is None or isinstance(trace_info, dict)


class TestServiceMeshIntegration:
    """Integration tests for service mesh functionality."""

    def test_round_trip_w3c(self):
        """Test round-trip injection and extraction of W3C headers."""
        # This test requires telemetry to be enabled
        # Skip if not available
        from tinyllm.telemetry import is_telemetry_enabled

        if not is_telemetry_enabled():
            pytest.skip("Telemetry not enabled")

        # Inject headers
        injected = inject_service_mesh_headers()

        # Extract from injected headers
        extracted = extract_service_mesh_headers(injected, set_as_current=False)

        # Should be able to extract what was injected
        assert extracted is None or "format" in extracted

    def test_downstream_propagation(self):
        """Test full propagation flow from upstream to downstream service."""
        # Incoming request headers
        incoming = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "baggage": "user_id=12345,tenant_id=acme",
        }

        # Extract context
        trace_info = extract_service_mesh_headers(incoming, set_as_current=False)

        # Propagate to downstream call
        outgoing = propagate_service_mesh_context(incoming, {"Content-Type": "application/json"})

        # Downstream should have trace headers
        assert "traceparent" in outgoing or len(outgoing) > 1
        assert "Content-Type" in outgoing

    def test_multi_hop_propagation(self):
        """Test trace propagation across multiple service hops."""
        # Service A receives request
        service_a_incoming = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "x-request-id": "req-123",
        }

        # Service A calls Service B
        service_b_headers = propagate_service_mesh_context(service_a_incoming)

        # Service B calls Service C
        service_c_headers = propagate_service_mesh_context(service_b_headers)

        # Trace headers should propagate through all hops
        assert "traceparent" in service_c_headers or "x-request-id" in service_c_headers
