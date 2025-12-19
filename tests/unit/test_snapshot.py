"""Snapshot tests for TinyLLM graph outputs.

Snapshot testing captures the output of graph executions and compares them
against previously recorded "snapshots". This helps detect unintended changes
in behavior across code changes.
"""

from syrupy import SnapshotAssertion

from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult


class TestMessageSnapshots:
    """Snapshot tests for Message serialization."""

    def test_message_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Message serialization produces consistent output."""
        message = Message(
            trace_id="test_trace_123",
            source_node="node_a",
            target_node="node_b",
            payload=MessagePayload(
                content="Test message content",
                task="test_task",
                structured={"key1": "value1", "key2": 42},
            ),
        )

        # Serialize to dict (excluding dynamic fields like message_id, created_at)
        msg_dict = message.model_dump(exclude={"message_id", "created_at", "metadata"})

        assert msg_dict == snapshot

    def test_payload_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """MessagePayload serialization is consistent."""
        payload = MessagePayload(
            content="Sample content for snapshot test",
            task="analysis",
            structured={
                "items": ["item1", "item2", "item3"],
                "count": 3,
                "enabled": True,
            },
        )

        payload_dict = payload.model_dump()
        assert payload_dict == snapshot


class TestNodeResultSnapshots:
    """Snapshot tests for NodeResult outputs."""

    def test_node_result_success_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Successful NodeResult has consistent structure."""
        message = Message(
            trace_id="trace_456",
            source_node="processor",
            target_node="output",
            payload=MessagePayload(content="Processed output"),
        )

        result = NodeResult(
            success=True,
            output_messages=[message],
            next_nodes=["next_node_1", "next_node_2"],
            metadata={"processing_time": 0.5, "model": "test_model"},
        )

        # Serialize excluding dynamic fields
        result_dict = result.model_dump(exclude={"latency_ms"})
        # Also exclude dynamic message fields
        for msg in result_dict.get("output_messages", []):
            msg.pop("message_id", None)
            msg.pop("created_at", None)
            if "metadata" in msg:
                msg.pop("metadata", None)

        assert result_dict == snapshot

    def test_node_result_error_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Failed NodeResult has consistent error structure."""
        result = NodeResult(
            success=False,
            output_messages=[],
            next_nodes=[],
            error="Validation failed: missing required field",
            metadata={"error_code": "VALIDATION_ERROR"},
        )

        result_dict = result.model_dump(exclude={"latency_ms"})
        assert result_dict == snapshot


class TestComplexStructureSnapshots:
    """Snapshot tests for complex data structures."""

    def test_nested_payload_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Deeply nested payload structures are captured correctly."""
        payload = MessagePayload(
            content="Complex nested structure",
            structured={
                "level1": {
                    "level2": {
                        "level3": {"values": [1, 2, 3], "enabled": True}
                    },
                    "items": ["a", "b", "c"],
                },
                "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00Z"},
            },
        )

        payload_dict = payload.model_dump()
        assert payload_dict == snapshot

    def test_batch_messages_snapshot(self, snapshot: SnapshotAssertion) -> None:
        """Batch of messages produces consistent output."""
        messages = []
        for i in range(3):
            msg = Message(
                trace_id=f"batch_trace_{i}",
                source_node=f"node_{i}",
                target_node=f"node_{i+1}",
                payload=MessagePayload(content=f"Message {i}", structured={"index": i}),
            )
            messages.append(msg)

        # Serialize batch
        batch_dict = [
            msg.model_dump(exclude={"message_id", "created_at", "metadata"})
            for msg in messages
        ]

        assert batch_dict == snapshot


# Note: Snapshots are stored in __snapshots__/ directory
# To update snapshots when behavior intentionally changes, run:
#   pytest tests/unit/test_snapshot.py --snapshot-update
#
# To review snapshot changes:
#   git diff tests/unit/__snapshots__/
