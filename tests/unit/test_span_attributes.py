"""Tests for custom span attributes."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_telemetry_enabled():
    """Mock telemetry as enabled."""
    with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=True):
        yield


@pytest.fixture
def mock_set_span_attribute():
    """Mock set_span_attribute function."""
    with patch("tinyllm.telemetry.set_span_attribute") as mock:
        yield mock


class TestGraphSpanAttributes:
    """Test GraphSpanAttributes class."""

    def test_attribute_names(self):
        """Test that attribute names follow conventions."""
        from tinyllm.telemetry import GraphSpanAttributes

        # Graph attributes
        assert GraphSpanAttributes.GRAPH_ID == "graph.id"
        assert GraphSpanAttributes.GRAPH_NAME == "graph.name"
        assert GraphSpanAttributes.NODE_TYPE == "node.type"

        # LLM attributes
        assert GraphSpanAttributes.LLM_MODEL == "llm.model"
        assert GraphSpanAttributes.LLM_TEMPERATURE == "llm.temperature"

        # Error attributes
        assert GraphSpanAttributes.ERROR_TYPE == "error.type"


class TestSetGraphAttributes:
    """Test set_graph_attributes function."""

    def test_set_graph_attributes_minimal(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting minimal graph attributes."""
        from tinyllm.telemetry import set_graph_attributes

        set_graph_attributes(graph_id="g123")

        # Should set graph_id
        assert mock_set_span_attribute.call_count == 1
        mock_set_span_attribute.assert_any_call("graph.id", "g123")

    def test_set_graph_attributes_full(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting all graph attributes."""
        from tinyllm.telemetry import set_graph_attributes

        set_graph_attributes(
            graph_id="g123",
            graph_name="code_gen",
            graph_version="1.0",
            graph_type="dag",
            node_count=5,
            edge_count=4,
            depth=3,
        )

        # Should set all attributes
        assert mock_set_span_attribute.call_count == 7
        mock_set_span_attribute.assert_any_call("graph.id", "g123")
        mock_set_span_attribute.assert_any_call("graph.name", "code_gen")
        mock_set_span_attribute.assert_any_call("graph.version", "1.0")


class TestSetNodeAttributes:
    """Test set_node_attributes function."""

    def test_set_node_attributes_minimal(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting minimal node attributes."""
        from tinyllm.telemetry import set_node_attributes

        set_node_attributes(node_id="n1", node_type="model")

        assert mock_set_span_attribute.call_count == 2
        mock_set_span_attribute.assert_any_call("node.id", "n1")
        mock_set_span_attribute.assert_any_call("node.type", "model")

    def test_set_node_attributes_with_children(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting node attributes with children."""
        from tinyllm.telemetry import set_node_attributes

        set_node_attributes(
            node_id="n1",
            node_type="fanout",
            children_nodes=["n2", "n3", "n4"],
        )

        # Check children are comma-separated
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        assert ("node.children", "n2,n3,n4") in calls


class TestSetExecutionAttributes:
    """Test set_execution_attributes function."""

    def test_set_execution_attributes(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting execution attributes."""
        from tinyllm.telemetry import set_execution_attributes

        set_execution_attributes(
            execution_id="e123",
            step=5,
            status="success",
            retry_count=2,
            duration_ms=1234.567,
        )

        # Check duration is rounded
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        assert ("execution.duration_ms", "1234.57") in calls


class TestSetTaskAttributes:
    """Test set_task_attributes function."""

    def test_set_task_attributes_truncates_content(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test that task content is truncated."""
        from tinyllm.telemetry import set_task_attributes

        long_content = "a" * 300
        set_task_attributes(task_content=long_content)

        # Check content is truncated
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        task_content_calls = [c for c in calls if c[0] == "task.content"]
        assert len(task_content_calls) == 1
        assert len(task_content_calls[0][1]) <= 203  # 200 + "..."

    def test_set_task_attributes_all_fields(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting all task attributes."""
        from tinyllm.telemetry import set_task_attributes

        set_task_attributes(
            task_id="t123",
            task_type="code_gen",
            task_content="Write a function",
            task_priority=1,
            message_count=5,
        )

        assert mock_set_span_attribute.call_count == 5


class TestSetLLMAttributes:
    """Test set_llm_attributes function."""

    def test_set_llm_attributes_minimal(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting minimal LLM attributes."""
        from tinyllm.telemetry import set_llm_attributes

        set_llm_attributes(model="gpt-4")

        assert mock_set_span_attribute.call_count == 1
        mock_set_span_attribute.assert_any_call("llm.model", "gpt-4")

    def test_set_llm_attributes_full(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting all LLM attributes."""
        from tinyllm.telemetry import set_llm_attributes

        set_llm_attributes(
            model="gpt-4",
            provider="openai",
            temperature=0.7,
            max_tokens=1000,
            prompt_length=500,
            response_length=250,
            prompt_tokens=125,
            response_tokens=63,
            cost_usd=0.005,
        )

        # Check cost is rounded to 6 decimal places
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        assert ("llm.cost_usd", "0.005") in calls


class TestSetCacheAttributes:
    """Test set_cache_attributes function."""

    def test_set_cache_attributes_hit(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting cache hit attributes."""
        from tinyllm.telemetry import set_cache_attributes

        set_cache_attributes(cache_hit=True, cache_key="key123", cache_ttl=3600)

        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        assert ("cache.hit", "True") in calls
        assert ("cache.key", "key123") in calls
        assert ("cache.ttl", "3600") in calls

    def test_set_cache_attributes_long_key(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test that long cache keys are hashed."""
        from tinyllm.telemetry import set_cache_attributes

        long_key = "a" * 200
        set_cache_attributes(cache_hit=False, cache_key=long_key)

        # Check key is hashed (should be 16 chars)
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        cache_key_calls = [c for c in calls if c[0] == "cache.key"]
        assert len(cache_key_calls) == 1
        assert len(cache_key_calls[0][1]) == 16


class TestSetErrorAttributes:
    """Test set_error_attributes function."""

    def test_set_error_attributes_basic(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test setting basic error attributes."""
        from tinyllm.telemetry import set_error_attributes

        set_error_attributes(error_type="ValueError", error_message="Invalid input")

        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        assert ("error.type", "ValueError") in calls
        assert ("error.message", "Invalid input") in calls

    def test_set_error_attributes_truncates_message(self, mock_telemetry_enabled, mock_set_span_attribute):
        """Test that error message is truncated."""
        from tinyllm.telemetry import set_error_attributes

        long_message = "x" * 1000
        set_error_attributes(error_type="Error", error_message=long_message)

        # Check message is truncated to 500 chars
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        message_calls = [c for c in calls if c[0] == "error.message"]
        assert len(message_calls) == 1
        assert len(message_calls[0][1]) == 500

    def test_set_error_attributes_truncates_stacktrace(
        self, mock_telemetry_enabled, mock_set_span_attribute
    ):
        """Test that stacktrace is truncated."""
        from tinyllm.telemetry import set_error_attributes

        long_stacktrace = "line\n" * 500
        set_error_attributes(
            error_type="Error",
            error_message="msg",
            error_stacktrace=long_stacktrace,
        )

        # Check stacktrace is truncated
        calls = [call[0] for call in mock_set_span_attribute.call_args_list]
        stacktrace_calls = [c for c in calls if c[0] == "error.stacktrace"]
        assert len(stacktrace_calls) == 1
        assert len(stacktrace_calls[0][1]) <= 1003  # 1000 + "..."


class TestIntegration:
    """Test integration of custom attributes with spans."""

    def test_attributes_in_span(self):
        """Test that attributes can be set in a span."""
        from tinyllm.telemetry import set_graph_attributes, set_node_attributes

        # Configure telemetry
        import tinyllm.telemetry

        with patch("tinyllm.telemetry.OTEL_AVAILABLE", True):
            with patch("tinyllm.telemetry.trace") as mock_trace:
                mock_tracer = MagicMock()
                mock_span = MagicMock()
                mock_span.is_recording.return_value = True

                # Mock the span context properly
                mock_span_context = MagicMock()
                mock_span_context.trace_id = 12345
                mock_span_context.span_id = 67890
                mock_span.get_span_context.return_value = mock_span_context

                mock_trace.get_current_span.return_value = mock_span

                # Manually set the tracer
                tinyllm.telemetry._tracer = mock_tracer
                tinyllm.telemetry._telemetry_enabled = True

                # Test setting attributes
                set_graph_attributes(graph_id="g1", node_count=5)
                set_node_attributes(node_id="n1", node_type="model")

                # Verify attributes were set
                assert mock_span.set_attribute.call_count >= 3
