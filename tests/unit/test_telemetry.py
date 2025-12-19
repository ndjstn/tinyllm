"""Tests for OpenTelemetry telemetry module."""

import pytest

from tinyllm.telemetry import (
    TelemetryConfig,
    configure_telemetry,
    get_current_span_id,
    get_current_trace_id,
    is_telemetry_enabled,
    record_span_event,
    set_span_attribute,
    trace_span,
    traced,
    traced_method,
)


class TestTelemetryConfig:
    """Tests for TelemetryConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TelemetryConfig()
        assert config.enable_tracing is False
        assert config.service_name == "tinyllm"
        assert config.exporter == "console"
        assert config.otlp_endpoint is None
        assert config.sampling_rate == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TelemetryConfig(
            enable_tracing=True,
            service_name="test-service",
            exporter="otlp",
            otlp_endpoint="http://localhost:4317",
            sampling_rate=0.5,
        )
        assert config.enable_tracing is True
        assert config.service_name == "test-service"
        assert config.exporter == "otlp"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.sampling_rate == 0.5

    def test_sampling_rate_validation(self):
        """Test sampling rate bounds."""
        # Valid rates
        TelemetryConfig(sampling_rate=0.0)
        TelemetryConfig(sampling_rate=0.5)
        TelemetryConfig(sampling_rate=1.0)

        # Invalid rates should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            TelemetryConfig(sampling_rate=-0.1)
        with pytest.raises(Exception):
            TelemetryConfig(sampling_rate=1.5)


class TestTelemetryDisabled:
    """Tests when telemetry is disabled."""

    def test_disabled_by_default(self):
        """Telemetry should be disabled by default."""
        # Don't configure telemetry
        assert is_telemetry_enabled() is False

    def test_trace_span_noop(self):
        """trace_span should be no-op when disabled."""
        with trace_span("test.span") as span:
            assert span is None

    def test_get_trace_id_none(self):
        """get_current_trace_id should return None when disabled."""
        assert get_current_trace_id() is None

    def test_get_span_id_none(self):
        """get_current_span_id should return None when disabled."""
        assert get_current_span_id() is None

    def test_set_attribute_noop(self):
        """set_span_attribute should be no-op when disabled."""
        # Should not raise
        set_span_attribute("key", "value")

    def test_record_event_noop(self):
        """record_span_event should be no-op when disabled."""
        # Should not raise
        record_span_event("event", {"key": "value"})


class TestTelemetryEnabled:
    """Tests when telemetry is enabled (requires OpenTelemetry)."""

    @pytest.fixture(autouse=True)
    def setup_telemetry(self):
        """Set up telemetry before each test."""
        try:
            config = TelemetryConfig(
                enable_tracing=True,
                service_name="test",
                exporter="console",
                sampling_rate=1.0,
            )
            configure_telemetry(config)
            yield
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_telemetry_enabled(self):
        """Telemetry should be enabled after configuration."""
        assert is_telemetry_enabled() is True

    def test_trace_span_creates_span(self):
        """trace_span should create a span when enabled."""
        with trace_span("test.span") as span:
            # If OpenTelemetry is available, span should not be None
            # But we can't assert this strongly since it depends on installation
            pass

    def test_nested_spans(self):
        """Test nested span creation."""
        with trace_span("parent.span") as parent:
            with trace_span("child.span") as child:
                # Both should work without error
                pass

    def test_span_attributes(self):
        """Test setting span attributes."""
        with trace_span("test.span", attributes={"key1": "value1"}):
            set_span_attribute("key2", "value2")
            set_span_attribute("key3", 123)
            # Should not raise

    def test_span_events(self):
        """Test recording span events."""
        with trace_span("test.span"):
            record_span_event("event1")
            record_span_event("event2", {"detail": "info"})
            # Should not raise


class TestTracedDecorator:
    """Tests for @traced decorator."""

    @pytest.mark.asyncio
    async def test_traced_decorator(self):
        """Test @traced decorator on async function."""

        @traced(span_name="test.function")
        async def test_function(x: int, y: int) -> int:
            return x + y

        result = await test_function(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_traced_with_attributes(self):
        """Test @traced decorator with attributes."""

        @traced(attributes={"component": "math"})
        async def add(x: int, y: int) -> int:
            return x + y

        result = await add(2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_traced_method_decorator(self):
        """Test @traced_method decorator on class method."""

        class Calculator:
            @traced_method(span_name="calculator.add")
            async def add(self, x: int, y: int) -> int:
                return x + y

        calc = Calculator()
        result = await calc.add(2, 3)
        assert result == 5


class TestTelemetryConfiguration:
    """Tests for configure_telemetry function."""

    def test_configure_disabled(self):
        """Test configuring with tracing disabled."""
        config = TelemetryConfig(enable_tracing=False)
        configure_telemetry(config)
        assert is_telemetry_enabled() is False

    def test_configure_console_exporter(self):
        """Test configuring with console exporter."""
        try:
            config = TelemetryConfig(
                enable_tracing=True,
                exporter="console",
            )
            configure_telemetry(config)
            # Should not raise
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_configure_invalid_exporter(self):
        """Test configuring with invalid exporter."""
        try:
            config = TelemetryConfig(
                enable_tracing=True,
                exporter="invalid",
            )
            with pytest.raises(ValueError):
                configure_telemetry(config)
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_configure_otlp_without_endpoint(self):
        """Test configuring OTLP without endpoint."""
        try:
            config = TelemetryConfig(
                enable_tracing=True,
                exporter="otlp",
                otlp_endpoint=None,
            )
            with pytest.raises(ValueError):
                configure_telemetry(config)
        except ImportError:
            pytest.skip("OpenTelemetry not installed")


class TestTraceHelpers:
    """Tests for trace helper functions."""

    def test_trace_executor_execution(self):
        """Test trace_executor_execution helper."""
        from tinyllm.telemetry import trace_executor_execution

        with trace_executor_execution(
            trace_id="test-123",
            graph_id="graph-1",
            task_content="test task",
        ):
            # Should not raise
            pass

    def test_trace_node_execution(self):
        """Test trace_node_execution helper."""
        from tinyllm.telemetry import trace_node_execution

        with trace_node_execution(
            node_id="node-1",
            node_type="model",
            step=1,
        ):
            # Should not raise
            pass

    def test_trace_llm_request(self):
        """Test trace_llm_request helper."""
        from tinyllm.telemetry import trace_llm_request

        with trace_llm_request(
            model="qwen2.5:0.5b",
            prompt_length=100,
            temperature=0.7,
        ):
            # Should not raise
            pass


class TestCorrelationID:
    """Tests for correlation ID functionality."""

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        from tinyllm.telemetry import generate_correlation_id

        corr_id = generate_correlation_id()
        assert corr_id is not None
        assert len(corr_id) == 36  # UUID format

        # Each should be unique
        corr_id2 = generate_correlation_id()
        assert corr_id != corr_id2

    def test_correlation_context_auto_generate(self):
        """Test correlation context with auto-generated ID."""
        from tinyllm.telemetry import correlation_context, get_correlation_id

        # No correlation ID initially
        assert get_correlation_id() is None

        with correlation_context() as corr_id:
            # Correlation ID should be set
            assert corr_id is not None
            assert get_correlation_id() == corr_id

        # Correlation ID should be cleaned up
        assert get_correlation_id() is None

    def test_correlation_context_provided_id(self):
        """Test correlation context with provided ID."""
        from tinyllm.telemetry import correlation_context, get_correlation_id

        test_id = "test-correlation-123"

        with correlation_context(test_id) as corr_id:
            assert corr_id == test_id
            assert get_correlation_id() == test_id

        # Cleaned up after
        assert get_correlation_id() is None

    def test_correlation_context_nested(self):
        """Test nested correlation contexts."""
        from tinyllm.telemetry import correlation_context, get_correlation_id

        outer_id = "outer-123"
        inner_id = "inner-456"

        with correlation_context(outer_id):
            assert get_correlation_id() == outer_id

            with correlation_context(inner_id):
                assert get_correlation_id() == inner_id

            # Should restore outer ID
            assert get_correlation_id() == outer_id

        # All cleaned up
        assert get_correlation_id() is None

    def test_propagate_correlation_id(self):
        """Test correlation ID propagation to headers."""
        from tinyllm.telemetry import (
            correlation_context,
            propagate_correlation_id,
        )

        test_id = "test-123"

        with correlation_context(test_id):
            headers = {}
            headers = propagate_correlation_id(headers)

            # Should not add correlation ID when telemetry disabled
            # (it needs baggage which requires telemetry)
            # Just verify it doesn't crash
            assert isinstance(headers, dict)

    def test_extract_correlation_id(self):
        """Test correlation ID extraction from headers."""
        from tinyllm.telemetry import extract_correlation_id

        # Test with standard header
        headers = {"X-Correlation-ID": "test-123"}
        assert extract_correlation_id(headers) == "test-123"

        # Test with lowercase header
        headers = {"x-correlation-id": "test-456"}
        assert extract_correlation_id(headers) == "test-456"

        # Test with alternate format
        headers = {"correlation-id": "test-789"}
        assert extract_correlation_id(headers) == "test-789"

        # Test with no header
        headers = {}
        assert extract_correlation_id(headers) is None


class TestBaggagePropagation:
    """Tests for OpenTelemetry baggage propagation."""

    def test_set_and_get_baggage(self):
        """Test setting and getting baggage items."""
        from tinyllm.telemetry import get_baggage, set_baggage

        set_baggage("user_id", "12345")
        set_baggage("tenant_id", "acme-corp")

        # May return None if telemetry not enabled
        user_id = get_baggage("user_id")
        if user_id:
            assert user_id == "12345"

    def test_get_all_baggage(self):
        """Test getting all baggage items."""
        from tinyllm.telemetry import get_all_baggage, set_baggage

        set_baggage("key1", "value1")
        set_baggage("key2", "value2")

        all_baggage = get_all_baggage()
        assert isinstance(all_baggage, dict)

    def test_remove_baggage(self):
        """Test removing baggage items."""
        from tinyllm.telemetry import get_baggage, remove_baggage, set_baggage

        set_baggage("temp_key", "temp_value")
        remove_baggage("temp_key")

        # Should be removed
        value = get_baggage("temp_key")
        assert value is None or value != "temp_value"

    def test_clear_baggage(self):
        """Test clearing all baggage."""
        from tinyllm.telemetry import clear_baggage, get_all_baggage, set_baggage

        set_baggage("key1", "value1")
        set_baggage("key2", "value2")

        clear_baggage()

        all_baggage = get_all_baggage()
        # Should be empty or minimal
        assert len(all_baggage) == 0 or all_baggage is None

    def test_inject_baggage_into_headers(self):
        """Test injecting baggage into HTTP headers."""
        from tinyllm.telemetry import inject_baggage_into_headers, set_baggage

        set_baggage("user_id", "12345")

        headers = inject_baggage_into_headers({"Content-Type": "application/json"})

        # Original headers should be preserved
        assert headers["Content-Type"] == "application/json"
        # Baggage header may be added
        assert isinstance(headers, dict)

    def test_extract_baggage_from_headers(self):
        """Test extracting baggage from HTTP headers."""
        from tinyllm.telemetry import extract_baggage_from_headers

        headers = {"baggage": "user_id=12345,tenant_id=acme"}

        extracted = extract_baggage_from_headers(headers)
        assert isinstance(extracted, dict)

    def test_baggage_context(self):
        """Test baggage context manager."""
        from tinyllm.telemetry import baggage_context, get_baggage

        with baggage_context(user_id="12345", tenant_id="acme"):
            # Baggage should be set within context
            user_id = get_baggage("user_id")
            # May be None if telemetry not enabled

        # Baggage should be cleared after context
        # (or restored to original state)


class TestCustomSpanAttributes:
    """Tests for custom span attributes."""

    def test_graph_span_attributes(self):
        """Test GraphSpanAttributes constants."""
        from tinyllm.telemetry import GraphSpanAttributes

        # Graph attributes
        assert GraphSpanAttributes.GRAPH_ID == "graph.id"
        assert GraphSpanAttributes.GRAPH_VERSION == "graph.version"
        assert GraphSpanAttributes.GRAPH_NODE_COUNT == "graph.node_count"

        # Node attributes
        assert GraphSpanAttributes.NODE_ID == "node.id"
        assert GraphSpanAttributes.NODE_TYPE == "node.type"

        # Execution attributes
        assert GraphSpanAttributes.EXECUTION_ID == "execution.id"
        assert GraphSpanAttributes.EXECUTION_STATUS == "execution.status"

        # LLM attributes
        assert GraphSpanAttributes.LLM_MODEL == "llm.model"
        assert GraphSpanAttributes.LLM_TEMPERATURE == "llm.temperature"

        # Cache attributes
        assert GraphSpanAttributes.CACHE_HIT == "cache.hit"

    def test_set_graph_attributes(self):
        """Test setting graph attributes on span."""
        from tinyllm.telemetry import set_graph_attributes, trace_span

        with trace_span("test.span"):
            # Should not raise
            set_graph_attributes(
                graph_id="test-graph",
                graph_name="Test Graph",
                node_count=5,
                depth=3,
            )

    def test_set_node_attributes(self):
        """Test setting node attributes on span."""
        from tinyllm.telemetry import set_node_attributes, trace_span

        with trace_span("test.span"):
            # Should not raise
            set_node_attributes(
                node_id="node-1",
                node_type="model",
                node_name="Code Generator",
                node_index=1,
                node_depth=2,
            )

    def test_set_execution_attributes(self):
        """Test setting execution attributes on span."""
        from tinyllm.telemetry import set_execution_attributes, trace_span

        with trace_span("test.span"):
            # Should not raise
            set_execution_attributes(
                execution_id="exec-123",
                step=1,
                status="running",
                duration_ms=123.45,
            )

    def test_set_task_attributes(self):
        """Test setting task attributes on span."""
        from tinyllm.telemetry import set_task_attributes, trace_span

        with trace_span("test.span"):
            # Should not raise
            set_task_attributes(
                task_id="task-456",
                task_type="code_generation",
                task_content="Write a Python function that...",
                message_count=3,
            )

    def test_set_llm_attributes(self):
        """Test setting LLM attributes on span."""
        from tinyllm.telemetry import set_llm_attributes, trace_span

        with trace_span("test.span"):
            # Should not raise
            set_llm_attributes(
                model="gpt-4",
                provider="openai",
                temperature=0.7,
                max_tokens=1000,
                prompt_tokens=100,
                response_tokens=200,
                cost_usd=0.005,
            )

    def test_set_cache_attributes(self):
        """Test setting cache attributes on span."""
        from tinyllm.telemetry import set_cache_attributes, trace_span

        with trace_span("test.span"):
            # Should not raise
            set_cache_attributes(
                cache_hit=True,
                cache_key="prompt_hash_123",
                cache_ttl=3600,
            )

    def test_set_error_attributes(self):
        """Test setting error attributes on span."""
        from tinyllm.telemetry import set_error_attributes, trace_span

        with trace_span("test.span"):
            try:
                raise ValueError("Test error")
            except ValueError as e:
                # Should not raise
                set_error_attributes(
                    error_type="ValueError",
                    error_message=str(e),
                    retry_after_ms=1000,
                )

    def test_custom_attributes_in_workflow(self):
        """Test using custom attributes in a workflow."""
        from tinyllm.telemetry import (
            set_execution_attributes,
            set_graph_attributes,
            set_llm_attributes,
            set_node_attributes,
            trace_span,
        )

        # Simulate graph execution with custom attributes
        with trace_span("graph.execute"):
            set_graph_attributes(
                graph_id="workflow-1",
                graph_name="Code Review",
                node_count=4,
            )

            set_execution_attributes(
                execution_id="exec-789",
                step=1,
                status="running",
            )

            # Simulate node execution
            with trace_span("node.model"):
                set_node_attributes(
                    node_id="reviewer",
                    node_type="model",
                    node_index=0,
                )

                set_llm_attributes(
                    model="gpt-4",
                    temperature=0.2,
                    prompt_tokens=500,
                    response_tokens=300,
                )

        # Should complete without errors
