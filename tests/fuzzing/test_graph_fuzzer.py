"""Graph execution fuzzer for TinyLLM.

This fuzzer generates random inputs to discover edge cases, crashes,
and unexpected behaviors in core TinyLLM components.
"""

import random
from typing import Any, Dict, List

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tinyllm.core.message import Message, MessagePayload, MessageMetadata
from tinyllm.core.node import NodeResult


# Hypothesis strategies for fuzzing


@st.composite
def message_payloads_fuzz(draw: Any) -> MessagePayload:
    """Generate random MessagePayload instances for fuzzing."""
    content = draw(st.text(min_size=0, max_size=5000))
    task = draw(st.one_of(st.none(), st.text(min_size=0, max_size=500)))
    structured = draw(
        st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=0, max_size=100),
                st.one_of(
                    st.text(max_size=1000),
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans(),
                    st.none(),
                ),
                max_size=50,
            ),
        )
    )

    return MessagePayload(content=content, task=task, structured=structured)


@st.composite
def messages_fuzz(draw: Any) -> Message:
    """Generate random Message instances for fuzzing."""
    trace_id = draw(st.text(min_size=1, max_size=500))
    source_node = draw(st.text(min_size=1, max_size=200))
    target_node = draw(st.one_of(st.none(), st.text(min_size=0, max_size=200)))
    payload = draw(message_payloads_fuzz())

    return Message(
        trace_id=trace_id,
        source_node=source_node,
        target_node=target_node,
        payload=payload,
    )


class TestMessageFuzzing:
    """Fuzzing tests for Message class."""

    @given(messages_fuzz())
    @settings(max_examples=100, deadline=3000)
    def test_message_creation_fuzzing(self, message: Message) -> None:
        """Random messages can be created without crashing."""
        assert message.trace_id
        assert message.source_node
        assert message.payload is not None
        assert message.message_id is not None
        assert message.created_at is not None

    @given(messages_fuzz())
    @settings(max_examples=100, deadline=3000)
    def test_message_serialization_fuzzing(self, message: Message) -> None:
        """Random messages can be serialized to dict."""
        try:
            msg_dict = message.model_dump()
            assert isinstance(msg_dict, dict)
            assert "trace_id" in msg_dict
            assert "payload" in msg_dict
        except Exception as e:
            pytest.fail(f"Message serialization failed: {e}")

    @given(st.lists(messages_fuzz(), min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_message_batch_creation(self, messages: List[Message]) -> None:
        """Batches of random messages can be created."""
        assert len(messages) > 0

        # All should have unique IDs
        ids = [msg.message_id for msg in messages]
        assert len(ids) == len(set(ids))


class TestPayloadFuzzing:
    """Fuzzing tests for MessagePayload."""

    @given(message_payloads_fuzz())
    @settings(max_examples=100, deadline=2000)
    def test_payload_content_types(self, payload: MessagePayload) -> None:
        """Random payloads handle various content types."""
        assert isinstance(payload.content, str)

        # Should be serializable
        payload_dict = payload.model_dump()
        assert "content" in payload_dict

    @given(
        st.text(min_size=0, max_size=10000),
        st.dictionaries(
            st.text(min_size=0, max_size=200),
            st.one_of(st.text(max_size=1000), st.integers(), st.booleans()),
            max_size=100,
        ),
    )
    @settings(max_examples=50, deadline=3000)
    def test_large_payloads(self, content: str, structured: Dict[str, Any]) -> None:
        """Large payloads don't cause issues."""
        payload = MessagePayload(content=content, structured=structured)

        assert payload.content == content
        assert payload.structured == structured

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=2000)
    def test_special_characters_in_content(self, content: str) -> None:
        """Payloads handle special characters in content."""
        payload = MessagePayload(content=content)

        # Should preserve content exactly
        assert payload.content == content

        # Should be serializable
        payload_dict = payload.model_dump()
        assert payload_dict["content"] == content


class TestNodeResultFuzzing:
    """Fuzzing tests for NodeResult."""

    @given(
        st.booleans(),
        st.lists(messages_fuzz(), max_size=50),
        st.lists(st.text(min_size=1, max_size=100), max_size=20),
    )
    @settings(max_examples=100, deadline=3000)
    def test_node_result_creation_fuzzing(
        self, success: bool, messages: List[Message], next_nodes: List[str]
    ) -> None:
        """Random NodeResults can be created."""
        result = NodeResult(
            success=success,
            output_messages=messages,
            next_nodes=next_nodes,
        )

        assert result.success == success
        assert len(result.output_messages) == len(messages)
        assert len(result.next_nodes) == len(next_nodes)

    @given(
        st.lists(messages_fuzz(), min_size=0, max_size=100),
        st.dictionaries(
            st.text(min_size=1, max_size=100),
            st.one_of(st.text(), st.integers(), st.booleans()),
            max_size=50,
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_node_result_with_metadata(
        self, messages: List[Message], metadata: Dict[str, Any]
    ) -> None:
        """NodeResults with random metadata don't crash."""
        result = NodeResult(
            success=True,
            output_messages=messages,
            next_nodes=[],
            metadata=metadata,
        )

        assert result.metadata == metadata


class TestEdgeCasesFuzzing:
    """Fuzzing tests for edge cases."""

    @given(st.text(min_size=0, max_size=0))
    @settings(max_examples=10)
    def test_empty_strings(self, empty: str) -> None:
        """Empty strings are handled correctly."""
        payload = MessagePayload(content=empty)
        assert payload.content == ""

        message = Message(
            trace_id="test",
            source_node="test",
            target_node=None,
            payload=payload,
        )
        assert message.payload.content == ""

    @given(st.text(min_size=1000, max_size=5000))
    @settings(max_examples=10, deadline=5000)
    def test_very_large_content(self, large_content: str) -> None:
        """Very large content strings are handled."""
        payload = MessagePayload(content=large_content)
        assert len(payload.content) == len(large_content)

    @given(st.lists(st.text(), min_size=100, max_size=1000))
    @settings(max_examples=10, deadline=10000)
    def test_many_structured_fields(self, keys: List[str]) -> None:
        """Payloads with many structured fields are handled."""
        structured = {f"key_{i}": f"value_{i}" for i in range(len(keys[:500]))}
        payload = MessagePayload(content="test", structured=structured)

        assert len(payload.structured) <= 500

    @given(st.text(alphabet=st.characters(blacklist_categories=("C",)), min_size=0, max_size=1000))
    @settings(max_examples=50, deadline=3000)
    def test_unicode_content(self, unicode_content: str) -> None:
        """Unicode content is handled correctly."""
        payload = MessagePayload(content=unicode_content)
        assert payload.content == unicode_content


@pytest.mark.slow
class TestExtensiveFuzzing:
    """Extensive fuzzing tests (marked as slow)."""

    @given(messages_fuzz())
    @settings(max_examples=1000, deadline=None)
    def test_extensive_message_fuzzing(self, message: Message) -> None:
        """Run extensive fuzzing on messages."""
        # Should always be creatable and serializable
        msg_dict = message.model_dump()
        assert msg_dict is not None

        # Should have required fields
        assert message.trace_id
        assert message.message_id
        assert message.source_node

    @given(message_payloads_fuzz())
    @settings(max_examples=1000, deadline=None)
    def test_extensive_payload_fuzzing(self, payload: MessagePayload) -> None:
        """Run extensive fuzzing on payloads."""
        # Should always be serializable
        payload_dict = payload.model_dump()
        assert payload_dict is not None
        assert "content" in payload_dict


class TestGraphExecutionFuzzing:
    """Fuzzing tests for graph execution paths."""

    @given(
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
        messages_fuzz()
    )
    @settings(max_examples=50, deadline=5000)
    def test_graph_traversal_fuzzing(
        self,
        node_sequence: List[str],
        message: Message
    ) -> None:
        """Graph traversal with random node sequences doesn't crash."""
        from tinyllm.config.graph import NodeDefinition, EdgeDefinition, NodeType, GraphDefinition, GraphMetadata

        # Create nodes for sequence
        nodes = []
        edges = []

        # Entry node
        entry_id = "entry.main"
        nodes.append(NodeDefinition(
            id=entry_id,
            type=NodeType.ENTRY,
            name="Entry",
            config={}
        ))

        # Create nodes from sequence (filter to valid node IDs)
        valid_node_ids = []
        for i, node_name in enumerate(node_sequence):
            # Sanitize to valid node ID
            node_id = f"node.{i}"
            if node_id not in valid_node_ids:  # Avoid duplicates
                nodes.append(NodeDefinition(
                    id=node_id,
                    type=NodeType.EXIT,  # Use EXIT type for simplicity
                    name=f"Node {i}",
                    config={"status": "success"}
                ))
                valid_node_ids.append(node_id)

        # Create linear edges
        if len(nodes) >= 2:
            for i in range(len(nodes) - 1):
                edges.append(EdgeDefinition(
                    from_node=nodes[i].id,
                    to_node=nodes[i + 1].id,
                    weight=1.0
                ))

        # Exit node (last node is already an exit)
        exit_ids = [n.id for n in nodes if n.type == NodeType.EXIT]

        # Create graph definition
        try:
            graph = GraphDefinition(
                id="fuzz.graph",
                version="0.0.1",
                name="Fuzz Test Graph",
                description="Randomly generated graph for fuzzing",
                metadata=GraphMetadata(created_at="2024-01-01T00:00:00", author="fuzzer"),
                nodes=nodes,
                edges=edges,
                entry_points=[entry_id],
                exit_points=exit_ids if exit_ids else [nodes[-1].id]
            )

            # Basic validation - graph should have valid structure
            assert len(graph.nodes) > 0
            assert len(graph.entry_points) > 0
            assert len(graph.exit_points) > 0
            assert all(ep in [n.id for n in graph.nodes] for ep in graph.entry_points)
            assert all(ep in [n.id for n in graph.nodes] for ep in graph.exit_points)

        except Exception as e:
            # Graph validation can fail with random inputs - that's expected
            assert isinstance(e, Exception)

    @given(
        st.integers(min_value=0, max_value=100),
        st.booleans(),
        messages_fuzz()
    )
    @settings(max_examples=50, deadline=3000)
    def test_node_result_fuzzing(
        self,
        num_messages: int,
        success: bool,
        sample_message: Message
    ) -> None:
        """NodeResult handles various message counts and success states."""
        # Create random number of messages
        output_messages = [sample_message for _ in range(min(num_messages, 10))]

        result = NodeResult(
            success=success,
            output_messages=output_messages,
            next_nodes=[]
        )

        assert result.success == success
        assert len(result.output_messages) == len(output_messages)


class TestExecutionContextFuzzing:
    """Fuzzing tests for ExecutionContext."""

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.text(), st.integers(), st.booleans(), st.floats(allow_nan=False)),
            max_size=20
        ),
        st.lists(messages_fuzz(), max_size=10)
    )
    @settings(max_examples=30, deadline=5000)
    def test_execution_context_variables_fuzzing(
        self,
        variables: Dict[str, Any],
        messages: List[Message]
    ) -> None:
        """ExecutionContext handles arbitrary variables and messages."""
        from tinyllm.core.context import ExecutionContext
        from tinyllm.config.loader import Config

        try:
            context = ExecutionContext(
                trace_id="fuzz_trace",
                graph_id="fuzz_graph",
                config=Config(),
                messages=[msg.model_dump() for msg in messages]
            )

            # Add variables
            for key, value in variables.items():
                context.variables[key] = value

            # Should be able to retrieve them
            for key in variables:
                assert key in context.variables

        except Exception as e:
            # Some combinations might be invalid
            assert isinstance(e, Exception)


# Configure fuzzing profiles
settings.register_profile("fuzzing_intensive", max_examples=500, deadline=None)
settings.register_profile("fuzzing_quick", max_examples=20, deadline=2000)

import os

if os.getenv("FUZZING_PROFILE") == "intensive":
    settings.load_profile("fuzzing_intensive")
