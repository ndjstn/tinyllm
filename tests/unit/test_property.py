"""Property-based tests using Hypothesis for TinyLLM core components.

These tests use Hypothesis to generate random inputs and verify that certain
properties always hold, helping to find edge cases and ensure robustness.
"""

import asyncio
from typing import Any, Dict

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessageMetadata, MessagePayload
from tinyllm.core.node import BaseNode, NodeResult


# Custom strategies for our domain
@st.composite
def message_payloads(draw: Any) -> MessagePayload:
    """Generate valid MessagePayload instances."""
    content = draw(st.text(min_size=0, max_size=1000))
    task = draw(st.one_of(st.none(), st.text(min_size=1, max_size=100)))
    structured = draw(
        st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=1, max_size=50),
                st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans()),
                max_size=10,
            ),
        )
    )

    return MessagePayload(content=content, task=task, structured=structured)


@st.composite
def messages(draw: Any) -> Message:
    """Generate valid Message instances."""
    trace_id = draw(st.text(min_size=1, max_size=100))
    source_node = draw(st.text(min_size=1, max_size=50))
    target_node = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    payload = draw(message_payloads())

    return Message(
        trace_id=trace_id, source_node=source_node, target_node=target_node, payload=payload
    )


@st.composite
def execution_contexts(draw: Any) -> ExecutionContext:
    """Generate valid ExecutionContext instances."""
    from tinyllm.config.loader import Config

    trace_id = draw(st.text(min_size=1, max_size=100))
    graph_id = draw(st.text(min_size=1, max_size=100))

    # Use a simple config
    config = Config()

    return ExecutionContext(trace_id=trace_id, graph_id=graph_id, config=config)


class TestMessageProperties:
    """Property-based tests for Message class."""

    @given(messages())
    def test_message_roundtrip_dict(self, message: Message) -> None:
        """Messages can be converted to dict and back without loss."""
        # Convert to dict
        message_dict = message.model_dump()

        # Reconstruct from dict
        reconstructed = Message(**message_dict)

        # Should be equal
        assert reconstructed.trace_id == message.trace_id
        assert reconstructed.source_node == message.source_node
        assert reconstructed.target_node == message.target_node
        assert reconstructed.payload.content == message.payload.content
        assert reconstructed.payload.task == message.payload.task

    @given(messages())
    def test_message_immutability_id(self, message: Message) -> None:
        """Message IDs are stable and don't change."""
        id1 = message.message_id
        id2 = message.message_id

        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) > 0

    @given(messages(), st.text(min_size=1, max_size=100))
    def test_message_trace_id_valid(self, message: Message, new_trace_id: str) -> None:
        """Messages can be created with any non-empty trace_id."""
        # Create a new message with different trace_id
        new_message = Message(
            trace_id=new_trace_id,
            source_node=message.source_node,
            target_node=message.target_node,
            payload=message.payload,
        )

        assert new_message.trace_id == new_trace_id


class TestMessagePayloadProperties:
    """Property-based tests for MessagePayload class."""

    @given(message_payloads())
    def test_payload_serializable(self, payload: MessagePayload) -> None:
        """Payloads can be serialized to dict."""
        payload_dict = payload.model_dump()

        assert isinstance(payload_dict, dict)
        assert "content" in payload_dict

    @given(st.text(min_size=0, max_size=1000))
    def test_payload_content_preserved(self, content: str) -> None:
        """Payload content is preserved exactly as provided."""
        payload = MessagePayload(content=content)

        assert payload.content == content

    @given(
        st.text(min_size=0, max_size=1000),
        st.dictionaries(st.text(min_size=1, max_size=50), st.text(min_size=0, max_size=100)),
    )
    def test_payload_structured_data(self, content: str, structured: Dict[str, str]) -> None:
        """Structured data in payload is preserved."""
        payload = MessagePayload(content=content, structured=structured)

        assert payload.content == content
        assert payload.structured == structured


class TestExecutionContextProperties:
    """Property-based tests for ExecutionContext class."""

    @given(execution_contexts())
    def test_context_trace_id_immutable(self, context: ExecutionContext) -> None:
        """Context trace_id doesn't change after creation."""
        trace_id_1 = context.trace_id
        trace_id_2 = context.trace_id

        assert trace_id_1 == trace_id_2

    @given(execution_contexts(), st.text(min_size=1, max_size=50), st.text(min_size=0))
    def test_context_memory_operations(
        self, context: ExecutionContext, key: str, value: str
    ) -> None:
        """Context variables operations work correctly."""
        # Add to variables
        context.variables[key] = value

        # Should be retrievable
        assert context.variables[key] == value

    @given(execution_contexts())
    def test_context_memory_independence(self, context: ExecutionContext) -> None:
        """Context variables changes don't affect original dict."""
        original_keys = set(context.variables.keys())

        # Add new key
        context.variables["new_key"] = "new_value"

        # Original keys should still be there
        assert all(key in context.variables for key in original_keys)


class TestNodeResultProperties:
    """Property-based tests for NodeResult class."""

    @given(
        st.booleans(),
        st.lists(messages(), max_size=10),
        st.lists(st.text(min_size=1, max_size=50), max_size=5),
    )
    def test_node_result_creation(
        self, success: bool, output_messages: list[Message], next_nodes: list[str]
    ) -> None:
        """NodeResult can be created with various combinations."""
        result = NodeResult(
            success=success, output_messages=output_messages, next_nodes=next_nodes
        )

        assert result.success == success
        assert result.output_messages == output_messages
        assert result.next_nodes == next_nodes

    @given(st.lists(messages(), min_size=1, max_size=10))
    def test_node_result_messages_preserved(self, messages_list: list[Message]) -> None:
        """NodeResult preserves all messages."""
        result = NodeResult(success=True, output_messages=messages_list, next_nodes=[])

        assert len(result.output_messages) == len(messages_list)
        for i, msg in enumerate(messages_list):
            assert result.output_messages[i].message_id == msg.message_id

    @given(
        st.booleans(),
        st.lists(messages(), max_size=10),
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(), st.integers(), st.booleans()),
            max_size=10,
        ),
    )
    def test_node_result_metadata(
        self, success: bool, messages_list: list[Message], metadata: Dict[str, Any]
    ) -> None:
        """NodeResult can carry arbitrary metadata."""
        result = NodeResult(
            success=success, output_messages=messages_list, next_nodes=[], metadata=metadata
        )

        assert result.metadata == metadata


class TestMessageTransformProperties:
    """Property-based tests for message transformations."""

    @given(messages(), st.text(min_size=0, max_size=1000))
    def test_message_content_update_idempotent(
        self, message: Message, new_content: str
    ) -> None:
        """Updating message content multiple times with same value is idempotent."""
        # Create new message with updated content
        updated1 = Message(
            trace_id=message.trace_id,
            source_node=message.source_node,
            target_node=message.target_node,
            payload=MessagePayload(
                content=new_content,
                task=message.payload.task,
                structured=message.payload.structured,
            ),
        )

        updated2 = Message(
            trace_id=updated1.trace_id,
            source_node=updated1.source_node,
            target_node=updated1.target_node,
            payload=MessagePayload(
                content=new_content,
                task=updated1.payload.task,
                structured=updated1.payload.structured,
            ),
        )

        assert updated1.payload.content == updated2.payload.content

    @given(messages(), messages())
    def test_different_messages_different_ids(self, msg1: Message, msg2: Message) -> None:
        """Different messages should have different IDs (with high probability)."""
        # Assume messages have different content to avoid false positives
        assume(
            msg1.source_node != msg2.source_node
            or msg1.target_node != msg2.target_node
            or msg1.payload.content != msg2.payload.content
        )

        # IDs should be different (though technically could collide, very unlikely)
        # We check that they're not trivially the same
        assert msg1.id != msg2.id or (
            msg1.source_node == msg2.source_node
            and msg1.target_node == msg2.target_node
            and msg1.payload.content == msg2.payload.content
        )


class TestEdgeCases:
    """Property-based tests for edge cases."""

    @given(st.text(min_size=0, max_size=0))
    def test_empty_content_messages(self, empty: str) -> None:
        """Messages can have empty content."""
        payload = MessagePayload(content=empty)
        message = Message(
            trace_id="test", source_node="source", target_node="target", payload=payload
        )

        assert message.payload.content == ""

    @given(st.dictionaries(st.text(min_size=1), st.text(), max_size=0))
    def test_empty_memory_context(self, empty_dict: Dict[str, str]) -> None:
        """Context can have empty memory."""
        context = ExecutionContext(trace_id="test", memory=empty_dict)

        assert len(context.memory) == 0

    @given(st.text(min_size=1, max_size=10000))
    def test_large_content_handling(self, large_content: str) -> None:
        """System can handle large content strings."""
        payload = MessagePayload(content=large_content)

        assert len(payload.content) == len(large_content)
        assert payload.content == large_content


# Configure hypothesis settings for CI/local development
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=2)

# Use CI profile in CI, dev profile otherwise
import os

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
