"""Tests for test data generators.

These tests verify that the factory-based generators work correctly
and produce valid test data.
"""

import pytest
from hypothesis import given

from tinyllm.config.graph import GraphDefinition, NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tests.fixtures.generators import (
    EdgeDefinitionFactory,
    GraphDefinitionFactory,
    MessageFactory,
    MessagePayloadFactory,
    NodeDefinitionFactory,
    generate_graph,
    generate_message,
    generate_messages,
    generate_node,
    generate_nodes,
)

# Import hypothesis strategies if available
try:
    from tests.fixtures.generators import (
        message_strategy,
        node_definition_strategy,
    )
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


class TestMessageFactories:
    """Tests for message factories."""

    def test_message_payload_factory(self):
        """Test MessagePayloadFactory generates valid payloads."""
        payload = MessagePayloadFactory()

        assert isinstance(payload, MessagePayload)
        assert isinstance(payload.task, str)
        assert isinstance(payload.content, str)
        assert isinstance(payload.metadata, dict)
        assert len(payload.task) > 0
        assert len(payload.content) > 0

    def test_message_payload_with_code_task(self):
        """Test generating payload with code task."""
        payload = MessagePayloadFactory.with_code_task()

        assert isinstance(payload, MessagePayload)
        assert len(payload.task) > 0

    def test_message_payload_with_math_task(self):
        """Test generating payload with math task."""
        payload = MessagePayloadFactory.with_math_task()

        assert isinstance(payload, MessagePayload)
        assert len(payload.task) > 0

    def test_message_factory(self):
        """Test MessageFactory generates valid messages."""
        message = MessageFactory()

        assert isinstance(message, Message)
        assert isinstance(message.trace_id, str)
        assert isinstance(message.source_node, str)
        assert isinstance(message.payload, MessagePayload)
        assert len(message.trace_id) > 0
        assert len(message.source_node) > 0

    def test_message_with_parent(self):
        """Test generating message with parent relationship."""
        parent = MessageFactory()
        child = MessageFactory.with_parent(parent, source_node="child")

        assert child.trace_id == parent.trace_id
        assert child.parent_id == parent.message_id
        assert child.source_node == "child"

    def test_generate_message_convenience(self):
        """Test generate_message convenience function."""
        message = generate_message()

        assert isinstance(message, Message)

    def test_generate_messages_convenience(self):
        """Test generate_messages convenience function."""
        messages = generate_messages(count=3)

        assert len(messages) == 3
        assert all(isinstance(m, Message) for m in messages)
        # Each should have unique trace_id
        trace_ids = [m.trace_id for m in messages]
        assert len(set(trace_ids)) == 3


class TestNodeFactories:
    """Tests for node factories."""

    def test_node_definition_factory(self):
        """Test NodeDefinitionFactory generates valid nodes."""
        node = NodeDefinitionFactory()

        assert isinstance(node, NodeDefinition)
        assert isinstance(node.id, str)
        assert isinstance(node.type, NodeType)
        assert isinstance(node.config, dict)
        assert len(node.id) > 0

    def test_entry_node_factory(self):
        """Test entry node factory method."""
        node = NodeDefinitionFactory.entry_node()

        assert node.id == "entry.main"
        assert node.type == NodeType.ENTRY
        assert isinstance(node.config, dict)

    def test_exit_node_factory(self):
        """Test exit node factory method."""
        node = NodeDefinitionFactory.exit_node(status="success")

        assert node.id == "exit.success"
        assert node.type == NodeType.EXIT
        assert node.config["status"] == "success"

    def test_model_node_factory(self):
        """Test model node factory method."""
        node = NodeDefinitionFactory.model_node(model="qwen2.5:3b")

        assert node.type == NodeType.MODEL
        assert node.config["model"] == "qwen2.5:3b"

    def test_router_node_factory(self):
        """Test router node factory method."""
        node = NodeDefinitionFactory.router_node(routes=["code", "math"])

        assert node.type == NodeType.ROUTER
        assert len(node.config["routes"]) == 2
        assert node.config["routes"][0]["name"] == "code"
        assert node.config["routes"][1]["name"] == "math"

    def test_tool_node_factory(self):
        """Test tool node factory method."""
        node = NodeDefinitionFactory.tool_node(tool_id="calculator")

        assert node.type == NodeType.TOOL
        assert node.config["tool_id"] == "calculator"

    def test_gate_node_factory(self):
        """Test gate node factory method."""
        node = NodeDefinitionFactory.gate_node(threshold=0.8)

        assert node.type == NodeType.GATE
        assert node.config["pass_threshold"] == 0.8

    def test_generate_node_convenience(self):
        """Test generate_node convenience function."""
        node = generate_node(node_type=NodeType.MODEL)

        assert isinstance(node, NodeDefinition)
        assert node.type == NodeType.MODEL

    def test_generate_nodes_convenience(self):
        """Test generate_nodes convenience function."""
        nodes = generate_nodes(count=5)

        assert len(nodes) == 5
        assert all(isinstance(n, NodeDefinition) for n in nodes)


class TestEdgeFactories:
    """Tests for edge factories."""

    def test_edge_definition_factory(self):
        """Test EdgeDefinitionFactory generates valid edges."""
        edge = EdgeDefinitionFactory()

        assert isinstance(edge.from_node, str)
        assert isinstance(edge.to_node, str)
        assert isinstance(edge.weight, float)
        assert 0.0 <= edge.weight <= 1.0

    def test_simple_edge_factory(self):
        """Test simple edge factory method."""
        edge = EdgeDefinitionFactory.simple("node1", "node2")

        assert edge.from_node == "node1"
        assert edge.to_node == "node2"
        assert edge.weight == 1.0
        assert edge.condition is None

    def test_conditional_edge_factory(self):
        """Test conditional edge factory method."""
        edge = EdgeDefinitionFactory.conditional(
            "router", "code", "route == 'code'"
        )

        assert edge.from_node == "router"
        assert edge.to_node == "code"
        assert edge.condition == "route == 'code'"


class TestGraphFactories:
    """Tests for graph factories."""

    def test_graph_definition_factory(self):
        """Test GraphDefinitionFactory generates valid graphs."""
        graph = GraphDefinitionFactory()

        assert isinstance(graph, GraphDefinition)
        assert isinstance(graph.id, str)
        assert isinstance(graph.version, str)
        assert len(graph.nodes) >= 2  # At least entry and exit
        assert len(graph.entry_points) > 0

    def test_simple_graph_factory(self):
        """Test simple graph factory method."""
        graph = GraphDefinitionFactory.simple_graph()

        assert len(graph.nodes) == 3  # entry, model, exit
        assert len(graph.edges) == 2
        assert len(graph.entry_points) == 1

        # Verify structure
        node_ids = [n.id for n in graph.nodes]
        assert "entry.main" in node_ids
        assert "exit.success" in node_ids

    def test_router_graph_factory(self):
        """Test router graph factory method."""
        graph = GraphDefinitionFactory.router_graph()

        assert len(graph.nodes) == 6  # entry, router, 3 specialists, exit
        assert len(graph.edges) == 7

        # Verify router and specialists exist
        node_types = [n.type for n in graph.nodes]
        assert NodeType.ROUTER in node_types
        assert node_types.count(NodeType.MODEL) == 3

    def test_complex_graph_factory(self):
        """Test complex graph factory method."""
        graph = GraphDefinitionFactory.complex_graph()

        assert len(graph.nodes) >= 6
        assert len(graph.edges) >= 6

        # Verify complex structure has tools and gates
        node_types = [n.type for n in graph.nodes]
        assert NodeType.TOOL in node_types
        assert NodeType.GATE in node_types

    def test_generate_graph_convenience_simple(self):
        """Test generate_graph convenience function with simple complexity."""
        graph = generate_graph(complexity="simple")

        assert isinstance(graph, GraphDefinition)
        assert len(graph.nodes) == 3

    def test_generate_graph_convenience_router(self):
        """Test generate_graph convenience function with router complexity."""
        graph = generate_graph(complexity="router")

        assert isinstance(graph, GraphDefinition)
        assert len(graph.nodes) == 6

    def test_generate_graph_convenience_complex(self):
        """Test generate_graph convenience function with complex complexity."""
        graph = generate_graph(complexity="complex")

        assert isinstance(graph, GraphDefinition)
        assert len(graph.nodes) >= 6


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
class TestHypothesisStrategies:
    """Tests for Hypothesis property-based testing strategies."""

    @given(message_strategy)
    def test_message_strategy(self, message):
        """Test message strategy generates valid messages."""
        assert isinstance(message, Message)
        assert len(message.trace_id) > 0
        assert len(message.source_node) > 0
        assert isinstance(message.payload, MessagePayload)

    @given(node_definition_strategy)
    def test_node_definition_strategy(self, node):
        """Test node definition strategy generates valid nodes."""
        assert isinstance(node, NodeDefinition)
        assert len(node.id) > 0
        assert isinstance(node.type, NodeType)
        assert isinstance(node.config, dict)

    @given(message_strategy)
    def test_message_trace_id_format(self, message):
        """Property: trace_id should be a valid UUID string."""
        # Should be a valid UUID format
        parts = message.trace_id.split("-")
        assert len(parts) == 5
        assert all(part for part in parts)

    @given(node_definition_strategy)
    def test_node_id_format(self, node):
        """Property: node.id should start with lowercase letter."""
        assert node.id[0].islower() or node.id[0].isdigit()
        # Should not contain invalid characters
        assert all(c.isalnum() or c in "._" for c in node.id)


class TestFactoryReproducibility:
    """Tests for factory reproducibility and consistency."""

    def test_multiple_messages_have_unique_ids(self):
        """Test that multiple messages get unique IDs."""
        messages = [MessageFactory() for _ in range(10)]
        message_ids = [m.message_id for m in messages]

        assert len(set(message_ids)) == 10  # All unique

    def test_multiple_nodes_have_unique_ids(self):
        """Test that multiple nodes get sequential unique IDs."""
        nodes = [NodeDefinitionFactory() for _ in range(10)]
        node_ids = [n.id for n in nodes]

        # All IDs should be unique
        assert len(set(node_ids)) == 10
        # Should all start with "node."
        assert all(nid.startswith("node.") for nid in node_ids)

    def test_custom_attributes_override_defaults(self):
        """Test that custom attributes override factory defaults."""
        custom_task = "Custom task content"
        message = MessageFactory(
            payload=MessagePayloadFactory(task=custom_task)
        )

        assert message.payload.task == custom_task
