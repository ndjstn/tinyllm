"""Test data generators using factory_boy and faker.

This module provides factories for generating realistic test data for TinyLLM
tests, including messages, nodes, graphs, and other domain objects.
"""

import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import factory
from factory import Factory, Faker, LazyAttribute, LazyFunction, Sequence, SubFactory

from tinyllm.config.graph import (
    EdgeDefinition,
    GraphDefinition,
    GraphMetadata,
    NodeDefinition,
    NodeType,
)
from tinyllm.core.message import Message, MessagePayload


class MessagePayloadFactory(Factory):
    """Factory for generating MessagePayload instances."""

    class Meta:
        model = MessagePayload

    task = Faker("sentence", nb_words=10)
    content = Faker("paragraph", nb_sentences=3)
    metadata = LazyFunction(dict)

    @classmethod
    def with_code_task(cls, **kwargs):
        """Generate payload with code-related task."""
        return cls(
            task=Faker("sentence", variable_nb_words=True, ext_word_list=[
                "Write", "a", "Python", "function", "to", "implement",
                "algorithm", "class", "method", "database", "query"
            ]).generate(),
            **kwargs
        )

    @classmethod
    def with_math_task(cls, **kwargs):
        """Generate payload with math-related task."""
        return cls(
            task=Faker("sentence", variable_nb_words=True, ext_word_list=[
                "Calculate", "the", "sum", "product", "derivative", "integral",
                "equation", "solve", "find", "prove", "theorem"
            ]).generate(),
            **kwargs
        )


class MessageFactory(Factory):
    """Factory for generating Message instances."""

    class Meta:
        model = Message

    trace_id = LazyFunction(lambda: str(uuid.uuid4()))
    source_node = Faker("word")
    payload = SubFactory(MessagePayloadFactory)
    parent_id = None

    @classmethod
    def with_parent(cls, parent: Message, **kwargs):
        """Generate message with parent relationship."""
        return cls(
            trace_id=parent.trace_id,
            parent_id=parent.message_id,
            **kwargs
        )


class NodeDefinitionFactory(Factory):
    """Factory for generating NodeDefinition instances."""

    class Meta:
        model = NodeDefinition

    id = Sequence(lambda n: f"node.{n}")
    type = LazyFunction(lambda: random.choice(list(NodeType)))
    name = Faker("word")
    description = Faker("sentence")
    config = LazyFunction(dict)

    @classmethod
    def entry_node(cls, **kwargs):
        """Generate entry node definition."""
        return cls(
            id="entry.main",
            type=NodeType.ENTRY,
            name="Main Entry",
            config={},
            **kwargs
        )

    @classmethod
    def exit_node(cls, status: str = "success", **kwargs):
        """Generate exit node definition."""
        return cls(
            id=f"exit.{status}",
            type=NodeType.EXIT,
            name=f"Exit {status.title()}",
            config={"status": status},
            **kwargs
        )

    @classmethod
    def model_node(cls, model: str = "qwen2.5:3b", **kwargs):
        """Generate model node definition."""
        return cls(
            id=kwargs.get("id", f"model.{model.replace(':', '_')}"),
            type=NodeType.MODEL,
            name=f"Model {model}",
            config={"model": model},
            **kwargs
        )

    @classmethod
    def router_node(cls, routes: Optional[List[str]] = None, **kwargs):
        """Generate router node definition."""
        if routes is None:
            routes = ["code", "math", "general"]

        route_definitions = [
            {
                "name": route,
                "description": f"{route.title()}-related tasks",
                "target": f"{route}.specialist",
            }
            for route in routes
        ]

        return cls(
            id="router.main",
            type=NodeType.ROUTER,
            name="Main Router",
            config={
                "model": "qwen2.5:0.5b",
                "routes": route_definitions,
            },
            **kwargs
        )

    @classmethod
    def tool_node(cls, tool_id: str = "calculator", **kwargs):
        """Generate tool node definition."""
        return cls(
            id=f"tool.{tool_id}",
            type=NodeType.TOOL,
            name=f"Tool {tool_id.title()}",
            config={"tool_id": tool_id},
            **kwargs
        )

    @classmethod
    def gate_node(cls, threshold: float = 0.7, **kwargs):
        """Generate gate node definition."""
        return cls(
            id="gate.quality",
            type=NodeType.GATE,
            name="Quality Gate",
            config={
                "model": "qwen2.5:3b",
                "pass_threshold": threshold,
            },
            **kwargs
        )


class EdgeDefinitionFactory(Factory):
    """Factory for generating EdgeDefinition instances."""

    class Meta:
        model = EdgeDefinition

    from_node = Faker("word")
    to_node = Faker("word")
    weight = LazyFunction(lambda: round(random.uniform(0.1, 1.0), 2))
    condition = None

    @classmethod
    def simple(cls, from_node: str, to_node: str, **kwargs):
        """Generate simple edge."""
        return cls(
            from_node=from_node,
            to_node=to_node,
            weight=1.0,
            **kwargs
        )

    @classmethod
    def conditional(cls, from_node: str, to_node: str, condition: str, **kwargs):
        """Generate conditional edge."""
        return cls(
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            **kwargs
        )


class GraphMetadataFactory(Factory):
    """Factory for generating GraphMetadata instances."""

    class Meta:
        model = GraphMetadata

    created_at = LazyFunction(lambda: datetime.utcnow().isoformat())
    description = Faker("sentence")
    author = Faker("name")
    tags = LazyFunction(lambda: random.sample(
        ["test", "example", "demo", "production", "development"],
        k=random.randint(1, 3)
    ))


class GraphDefinitionFactory(Factory):
    """Factory for generating GraphDefinition instances."""

    class Meta:
        model = GraphDefinition

    id = Sequence(lambda n: f"graph.test{n}")
    version = LazyFunction(lambda: f"{random.randint(0,2)}.{random.randint(0,9)}.{random.randint(0,9)}")
    name = Faker("sentence", nb_words=3)
    description = Faker("paragraph")
    metadata = SubFactory(GraphMetadataFactory)
    nodes = LazyFunction(lambda: [
        NodeDefinitionFactory.entry_node(),
        NodeDefinitionFactory.exit_node(),
    ])
    edges = LazyFunction(lambda: [])
    entry_points = LazyAttribute(lambda obj: [obj.nodes[0].id])
    exit_points = LazyAttribute(lambda obj: [n.id for n in obj.nodes if n.type == NodeType.EXIT])

    @classmethod
    def simple_graph(cls, **kwargs):
        """Generate a simple linear graph: entry -> model -> exit."""
        entry = NodeDefinitionFactory.entry_node()
        model = NodeDefinitionFactory.model_node()
        exit_node = NodeDefinitionFactory.exit_node()

        edges = [
            EdgeDefinitionFactory.simple(entry.id, model.id),
            EdgeDefinitionFactory.simple(model.id, exit_node.id),
        ]

        return cls(
            nodes=[entry, model, exit_node],
            edges=edges,
            entry_points=[entry.id],
            **kwargs
        )

    @classmethod
    def router_graph(cls, **kwargs):
        """Generate a graph with router and specialists."""
        entry = NodeDefinitionFactory.entry_node()
        router = NodeDefinitionFactory.router_node()

        # Create specialist nodes
        code_node = NodeDefinitionFactory.model_node(
            id="code.specialist",
            model="granite-code:3b"
        )
        math_node = NodeDefinitionFactory.model_node(
            id="math.specialist",
            model="phi3:mini"
        )
        general_node = NodeDefinitionFactory.model_node(
            id="general.specialist",
            model="qwen2.5:3b"
        )

        exit_node = NodeDefinitionFactory.exit_node()

        # Create edges
        edges = [
            EdgeDefinitionFactory.simple(entry.id, router.id),
            EdgeDefinitionFactory.conditional(
                router.id, code_node.id, "route == 'code'"
            ),
            EdgeDefinitionFactory.conditional(
                router.id, math_node.id, "route == 'math'"
            ),
            EdgeDefinitionFactory.conditional(
                router.id, general_node.id, "route == 'general'"
            ),
            EdgeDefinitionFactory.simple(code_node.id, exit_node.id),
            EdgeDefinitionFactory.simple(math_node.id, exit_node.id),
            EdgeDefinitionFactory.simple(general_node.id, exit_node.id),
        ]

        return cls(
            nodes=[entry, router, code_node, math_node, general_node, exit_node],
            edges=edges,
            entry_points=[entry.id],
            **kwargs
        )

    @classmethod
    def complex_graph(cls, **kwargs):
        """Generate a complex graph with multiple layers."""
        entry = NodeDefinitionFactory.entry_node()
        router = NodeDefinitionFactory.router_node()

        # Specialists
        code_node = NodeDefinitionFactory.model_node(
            id="code.specialist",
            model="granite-code:3b"
        )
        math_node = NodeDefinitionFactory.model_node(
            id="math.specialist",
            model="phi3:mini"
        )

        # Tool nodes
        calc_tool = NodeDefinitionFactory.tool_node("calculator")

        # Quality gate
        gate = NodeDefinitionFactory.gate_node()

        exit_success = NodeDefinitionFactory.exit_node("success")
        exit_fallback = NodeDefinitionFactory.exit_node("fallback")

        edges = [
            EdgeDefinitionFactory.simple(entry.id, router.id),
            EdgeDefinitionFactory.conditional(
                router.id, code_node.id, "route == 'code'"
            ),
            EdgeDefinitionFactory.conditional(
                router.id, math_node.id, "route == 'math'"
            ),
            EdgeDefinitionFactory.simple(math_node.id, calc_tool.id),
            EdgeDefinitionFactory.simple(calc_tool.id, gate.id),
            EdgeDefinitionFactory.simple(code_node.id, gate.id),
            EdgeDefinitionFactory.conditional(
                gate.id, exit_success.id, "passed == true"
            ),
            EdgeDefinitionFactory.conditional(
                gate.id, router.id, "passed == false"
            ),
        ]

        return cls(
            nodes=[
                entry, router, code_node, math_node,
                calc_tool, gate, exit_success, exit_fallback
            ],
            edges=edges,
            entry_points=[entry.id],
            **kwargs
        )


# Convenience functions for quick generation


def generate_message(**kwargs) -> Message:
    """Generate a single test message.

    Args:
        **kwargs: Override default factory attributes.

    Returns:
        Message instance.
    """
    return MessageFactory(**kwargs)


def generate_messages(count: int = 5, **kwargs) -> List[Message]:
    """Generate multiple test messages.

    Args:
        count: Number of messages to generate.
        **kwargs: Override default factory attributes.

    Returns:
        List of Message instances.
    """
    return [MessageFactory(**kwargs) for _ in range(count)]


def generate_node(node_type: Optional[NodeType] = None, **kwargs) -> NodeDefinition:
    """Generate a single test node.

    Args:
        node_type: Specific node type to generate.
        **kwargs: Override default factory attributes.

    Returns:
        NodeDefinition instance.
    """
    if node_type:
        kwargs["type"] = node_type
    return NodeDefinitionFactory(**kwargs)


def generate_nodes(count: int = 5, **kwargs) -> List[NodeDefinition]:
    """Generate multiple test nodes.

    Args:
        count: Number of nodes to generate.
        **kwargs: Override default factory attributes.

    Returns:
        List of NodeDefinition instances.
    """
    return [NodeDefinitionFactory(**kwargs) for _ in range(count)]


def generate_graph(complexity: str = "simple", **kwargs) -> GraphDefinition:
    """Generate a test graph.

    Args:
        complexity: Graph complexity ('simple', 'router', or 'complex').
        **kwargs: Override default factory attributes.

    Returns:
        GraphDefinition instance.
    """
    if complexity == "simple":
        return GraphDefinitionFactory.simple_graph(**kwargs)
    elif complexity == "router":
        return GraphDefinitionFactory.router_graph(**kwargs)
    elif complexity == "complex":
        return GraphDefinitionFactory.complex_graph(**kwargs)
    else:
        return GraphDefinitionFactory(**kwargs)


# Hypothesis strategies for property-based testing

try:
    from hypothesis import strategies as st

    # Basic types
    node_id_strategy = st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="_"),
        min_size=1,
        max_size=50
    ).filter(lambda s: s[0].isalpha())

    # Node type strategy
    node_type_strategy = st.sampled_from(list(NodeType))

    # Message payload strategy
    message_payload_strategy = st.builds(
        MessagePayload,
        task=st.text(min_size=10, max_size=200),
        content=st.text(min_size=10, max_size=500),
        metadata=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False))
        )
    )

    # Message strategy
    message_strategy = st.builds(
        Message,
        trace_id=st.uuids().map(str),
        source_node=node_id_strategy,
        payload=message_payload_strategy,
    )

    # Node definition strategy
    node_definition_strategy = st.builds(
        NodeDefinition,
        id=node_id_strategy,
        type=node_type_strategy,
        name=st.text(min_size=1, max_size=50),
        description=st.one_of(st.none(), st.text(max_size=200)),
        config=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans())
        )
    )

except ImportError:
    # Hypothesis not installed
    pass
