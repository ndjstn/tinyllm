"""Snapshot tests for TinyLLM outputs.

These tests use pytest-syrupy to capture and verify complex outputs
remain stable across changes. Snapshots are stored in __snapshots__/ directory.
"""

import pytest
from datetime import datetime

from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult
from tinyllm.config.graph import (
    NodeDefinition,
    EdgeDefinition,
    GraphDefinition,
    NodeType,
    GraphMetadata,
)
from tinyllm.error_enrichment import EnrichedError, ErrorContext, ErrorCategory, ErrorSeverity
from tinyllm.error_aggregation import ErrorSignature


class TestMessageSnapshots:
    """Snapshot tests for message structures."""

    def test_message_payload_snapshot(self, snapshot):
        """Message payload serialization remains stable."""
        payload = MessagePayload(
            content="Calculate the factorial of 5",
            task="math_calculation",
            structured={
                "operation": "factorial",
                "operand": 5,
                "expected_type": "integer"
            }
        )

        # Snapshot the dict representation
        assert payload.model_dump() == snapshot

    def test_message_snapshot(self, snapshot):
        """Message serialization remains stable."""
        payload = MessagePayload(
            content="Write a function to sort a list",
            task="code_generation"
        )

        message = Message(
            trace_id="trace_123",
            source_node="router.main",
            target_node="model.code_specialist",
            payload=payload
        )

        # Snapshot without timestamp/message_id (they change)
        msg_dict = message.model_dump()
        msg_dict.pop('message_id', None)
        msg_dict.pop('created_at', None)

        assert msg_dict == snapshot

    def test_message_with_metadata_snapshot(self, snapshot):
        """Message with metadata serialization remains stable."""
        payload = MessagePayload(
            content="Summarize this text",
            task="summarization",
            metadata={
                "source": "document",
                "priority": "high",
                "max_length": 100
            }
        )

        message = Message(
            trace_id="trace_456",
            source_node="entry.main",
            payload=payload
        )

        msg_dict = message.model_dump()
        msg_dict.pop('message_id', None)
        msg_dict.pop('created_at', None)

        assert msg_dict == snapshot


class TestNodeResultSnapshots:
    """Snapshot tests for node execution results."""

    def test_successful_node_result_snapshot(self, snapshot):
        """Successful node result structure remains stable."""
        output_message = Message(
            trace_id="trace_789",
            source_node="model.specialist",
            target_node="exit.success",
            payload=MessagePayload(content="Result: 42")
        )

        result = NodeResult(
            success=True,
            output_messages=[output_message],
            next_nodes=["exit.success"],
            metadata={
                "execution_time_ms": 150,
                "tokens_used": 25,
                "model": "qwen2.5:3b"
            }
        )

        result_dict = result.model_dump()
        # Remove timestamps
        for msg in result_dict.get('output_messages', []):
            msg.pop('message_id', None)
            msg.pop('created_at', None)

        assert result_dict == snapshot

    def test_failed_node_result_snapshot(self, snapshot):
        """Failed node result structure remains stable."""
        result = NodeResult(
            success=False,
            output_messages=[],
            next_nodes=["exit.error"],
            metadata={
                "error": "Model timeout",
                "error_code": "TIMEOUT"
            }
        )

        assert result.model_dump() == snapshot


class TestGraphDefinitionSnapshots:
    """Snapshot tests for graph structures."""

    def test_simple_graph_snapshot(self, snapshot):
        """Simple linear graph structure remains stable."""
        entry = NodeDefinition(
            id="entry.main",
            type=NodeType.ENTRY,
            name="Main Entry",
            config={}
        )

        model = NodeDefinition(
            id="model.qwen",
            type=NodeType.MODEL,
            name="Qwen Model",
            config={"model": "qwen2.5:3b"}
        )

        exit_node = NodeDefinition(
            id="exit.success",
            type=NodeType.EXIT,
            name="Success Exit",
            config={"status": "success"}
        )

        edges = [
            EdgeDefinition(from_node="entry.main", to_node="model.qwen", weight=1.0),
            EdgeDefinition(from_node="model.qwen", to_node="exit.success", weight=1.0),
        ]

        graph = GraphDefinition(
            id="graph.simple",
            version="1.0.0",
            name="Simple Linear Graph",
            description="Entry -> Model -> Exit",
            metadata=GraphMetadata(
                created_at="2024-01-01T00:00:00",
                author="test",
                tags=["test", "simple"]
            ),
            nodes=[entry, model, exit_node],
            edges=edges,
            entry_points=["entry.main"],
            exit_points=["exit.success"]
        )

        assert graph.model_dump() == snapshot

    def test_router_graph_snapshot(self, snapshot):
        """Router graph with multiple paths remains stable."""
        entry = NodeDefinition(
            id="entry.main",
            type=NodeType.ENTRY,
            name="Main Entry",
            config={}
        )

        router = NodeDefinition(
            id="router.main",
            type=NodeType.ROUTER,
            name="Main Router",
            config={
                "model": "qwen2.5:0.5b",
                "routes": [
                    {"name": "code", "description": "Code-related tasks", "target": "model.code"},
                    {"name": "math", "description": "Math tasks", "target": "model.math"}
                ]
            }
        )

        code_model = NodeDefinition(
            id="model.code",
            type=NodeType.MODEL,
            name="Code Specialist",
            config={"model": "granite-code:3b"}
        )

        math_model = NodeDefinition(
            id="model.math",
            type=NodeType.MODEL,
            name="Math Specialist",
            config={"model": "phi3:mini"}
        )

        exit_node = NodeDefinition(
            id="exit.success",
            type=NodeType.EXIT,
            name="Success",
            config={"status": "success"}
        )

        edges = [
            EdgeDefinition(from_node="entry.main", to_node="router.main", weight=1.0),
            EdgeDefinition(from_node="router.main", to_node="model.code", weight=1.0, condition="route == 'code'"),
            EdgeDefinition(from_node="router.main", to_node="model.math", weight=1.0, condition="route == 'math'"),
            EdgeDefinition(from_node="model.code", to_node="exit.success", weight=1.0),
            EdgeDefinition(from_node="model.math", to_node="exit.success", weight=1.0),
        ]

        graph = GraphDefinition(
            id="graph.router",
            version="1.0.0",
            name="Router Graph",
            description="Router with code and math specialists",
            metadata=GraphMetadata(
                created_at="2024-01-01T00:00:00",
                author="test"
            ),
            nodes=[entry, router, code_model, math_model, exit_node],
            edges=edges,
            entry_points=["entry.main"],
            exit_points=["exit.success"]
        )

        assert graph.model_dump() == snapshot


class TestErrorSnapshots:
    """Snapshot tests for error structures."""

    def test_error_signature_snapshot(self, snapshot):
        """Error signature normalization remains stable."""
        context = ErrorContext(
            stack_trace="Traceback (most recent call last):\n  File test.py line 10",
            exception_type="ValueError",
            exception_message="Invalid input",
            node_id="model.specialist",
            graph_id="graph.test"
        )

        error = EnrichedError(
            error_id="err_001",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            message="Connection timeout after 5000 milliseconds",
            context=context,
            is_retryable=True,
            suggested_actions=["Retry with exponential backoff", "Check network connectivity"]
        )

        signature = ErrorSignature.from_enriched_error(error)

        # Snapshot the signature (hash will be deterministic for same input)
        sig_dict = signature.model_dump()

        assert sig_dict == snapshot

    def test_normalized_message_patterns_snapshot(self, snapshot):
        """Message normalization patterns remain stable."""
        test_messages = [
            "Connection timeout after 5000 milliseconds",
            "Error at 2024-01-15 10:30:45",
            "Server listening on :8080",
            "Failed to connect to 192.168.1.1:3306",
            "Request 550e8400-e29b-41d4-a716-446655440000 failed",
            "File not found: /var/log/app/file1234.txt"
        ]

        normalized = {
            msg: ErrorSignature._normalize_message(msg)
            for msg in test_messages
        }

        assert normalized == snapshot


class TestConfigurationSnapshots:
    """Snapshot tests for configuration structures."""

    def test_node_type_enum_snapshot(self, snapshot):
        """NodeType enum values remain stable."""
        node_types = {nt.value: nt.name for nt in NodeType}
        assert node_types == snapshot

    def test_error_severity_enum_snapshot(self, snapshot):
        """ErrorSeverity enum values remain stable."""
        severities = {sev.value: sev.name for sev in ErrorSeverity}
        assert severities == snapshot

    def test_error_category_enum_snapshot(self, snapshot):
        """ErrorCategory enum values remain stable."""
        categories = {cat.value: cat.name for cat in ErrorCategory}
        assert categories == snapshot
