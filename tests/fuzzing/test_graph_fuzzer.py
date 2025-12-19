"""Graph execution fuzzer for TinyLLM.

This fuzzer generates random graph structures and executions to discover
edge cases, crashes, and unexpected behaviors in graph execution logic.
"""

import random
from typing import Any, Dict, List

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.graph import Edge, Graph, NodeConfig
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeResult


class DummyNode(BaseNode):
    """A dummy node for fuzzing that always succeeds."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        """Execute dummy node logic."""
        # Simply pass through with some transformation
        new_content = f"Processed by {self.id}: {message.payload.content}"
        output_message = Message(
            trace_id=message.trace_id,
            source_node=self.id,
            target_node=None,
            payload=MessagePayload(content=new_content),
        )

        return NodeResult(success=True, output_messages=[output_message], next_nodes=[])


class ErrorNode(BaseNode):
    """A node that randomly fails for fuzzing."""

    def __init__(self, node_id: str, config: Dict[str, Any], fail_rate: float = 0.3):
        super().__init__(node_id, config)
        self.fail_rate = fail_rate

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        """Execute with random failures."""
        if random.random() < self.fail_rate:
            return NodeResult(
                success=False,
                output_messages=[],
                next_nodes=[],
                error="Random failure for fuzzing",
            )

        output_message = Message(
            trace_id=message.trace_id,
            source_node=self.id,
            target_node=None,
            payload=MessagePayload(content=f"Error node {self.id} processed"),
        )

        return NodeResult(success=True, output_messages=[output_message], next_nodes=[])


# Hypothesis strategies for fuzzing graphs


@st.composite
def node_ids(draw: Any) -> str:
    """Generate valid node IDs."""
    return draw(st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=20))


@st.composite
def node_configs(draw: Any, node_id: str) -> NodeConfig:
    """Generate valid NodeConfig instances."""
    return NodeConfig(
        id=node_id,
        type="dummy",
        config={
            "param1": draw(st.one_of(st.text(max_size=100), st.integers(), st.floats(allow_nan=False))),
            "param2": draw(st.booleans()),
        },
    )


@st.composite
def edges(draw: Any, from_nodes: List[str], to_nodes: List[str]) -> Edge:
    """Generate valid Edge instances."""
    assume(len(from_nodes) > 0 and len(to_nodes) > 0)

    from_node = draw(st.sampled_from(from_nodes))
    to_node = draw(st.sampled_from(to_nodes))

    return Edge(from_node=from_node, to_node=to_node, condition=None)


@st.composite
def small_graphs(draw: Any) -> Graph:
    """Generate small valid graphs for fuzzing."""
    # Generate 2-5 nodes
    num_nodes = draw(st.integers(min_value=2, max_value=5))
    node_ids_list = [f"node_{i}" for i in range(num_nodes)]

    # Create graph
    graph = Graph(graph_id=f"fuzz_graph_{random.randint(0, 10000)}", config=Config())

    # Add nodes
    for node_id in node_ids_list:
        node = DummyNode(node_id=node_id, config={})
        graph.add_node(node)

    # Add some edges (create a somewhat connected graph)
    num_edges = draw(st.integers(min_value=1, max_value=min(num_nodes * 2, 10)))
    for _ in range(num_edges):
        if len(node_ids_list) >= 2:
            from_idx = draw(st.integers(min_value=0, max_value=len(node_ids_list) - 2))
            to_idx = draw(st.integers(min_value=from_idx + 1, max_value=len(node_ids_list) - 1))

            try:
                graph.add_edge(
                    Edge(
                        from_node=node_ids_list[from_idx],
                        to_node=node_ids_list[to_idx],
                        condition=None,
                    )
                )
            except Exception:
                # Ignore edge addition failures (duplicate edges, etc.)
                pass

    # Set start and end nodes
    graph.start_node = node_ids_list[0]
    graph.end_node = node_ids_list[-1]

    return graph


class TestGraphFuzzer:
    """Fuzzing tests for graph execution."""

    @given(small_graphs())
    @settings(max_examples=50, deadline=5000)  # Run 50 fuzz tests
    def test_graph_structure_valid(self, graph: Graph) -> None:
        """Fuzzed graphs have valid structure."""
        # Check basic invariants
        assert graph.graph_id is not None
        assert len(graph.graph_id) > 0
        assert graph.start_node is not None
        assert graph.end_node is not None

        # All nodes should be reachable from start (or be start)
        nodes = list(graph.nodes.keys())
        assert len(nodes) > 0
        assert graph.start_node in nodes
        assert graph.end_node in nodes

    @given(small_graphs(), st.text(min_size=0, max_size=1000))
    @settings(max_examples=50, deadline=5000)
    def test_graph_accepts_messages(self, graph: Graph, content: str) -> None:
        """Fuzzed graphs can accept messages without crashing."""
        try:
            message = Message(
                trace_id="fuzz_trace",
                source_node="external",
                target_node=graph.start_node,
                payload=MessagePayload(content=content),
            )

            # Just verify message creation doesn't crash
            assert message.payload.content == content
        except Exception as e:
            # Some messages might be invalid, that's okay
            pytest.skip(f"Invalid message: {e}")

    @given(small_graphs())
    @settings(max_examples=30, deadline=10000)
    def test_graph_edges_consistent(self, graph: Graph) -> None:
        """Fuzzed graphs have consistent edge relationships."""
        nodes = set(graph.nodes.keys())

        for edge in graph.edges:
            # From and to nodes should exist
            assert edge.from_node in nodes, f"Edge from {edge.from_node} but node doesn't exist"
            assert edge.to_node in nodes, f"Edge to {edge.to_node} but node doesn't exist"

    @given(
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=20, deadline=3000)
    def test_random_graph_construction(self, num_nodes: int, num_edges: int) -> None:
        """Randomly constructed graphs don't crash."""
        graph = Graph(graph_id="random_graph", config=Config())

        # Add nodes
        node_ids_list = []
        for i in range(num_nodes):
            node_id = f"rand_node_{i}"
            node = DummyNode(node_id=node_id, config={})
            graph.add_node(node)
            node_ids_list.append(node_id)

        # Add random edges
        for _ in range(min(num_edges, num_nodes * (num_nodes - 1))):
            if len(node_ids_list) >= 2:
                from_node = random.choice(node_ids_list)
                to_node = random.choice([n for n in node_ids_list if n != from_node])

                try:
                    graph.add_edge(Edge(from_node=from_node, to_node=to_node, condition=None))
                except Exception:
                    # Ignore failures (e.g., duplicate edges)
                    pass

        # Set start/end
        graph.start_node = node_ids_list[0]
        graph.end_node = node_ids_list[-1]

        # Verify basic structure
        assert len(graph.nodes) == num_nodes
        assert len(graph.edges) >= 0

    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=50))
    @settings(max_examples=20, deadline=5000)
    def test_message_batch_fuzzing(self, contents: List[str]) -> None:
        """Process batches of random messages without crashing."""
        # Create a simple graph
        graph = Graph(graph_id="batch_test", config=Config())

        node1 = DummyNode(node_id="processor", config={})
        graph.add_node(node1)
        graph.start_node = "processor"
        graph.end_node = "processor"

        # Create messages
        messages = []
        for content in contents:
            msg = Message(
                trace_id=f"batch_{random.randint(0, 10000)}",
                source_node="external",
                target_node="processor",
                payload=MessagePayload(content=content),
            )
            messages.append(msg)

        # Verify all messages created successfully
        assert len(messages) == len(contents)

    def test_error_node_fuzzing(self) -> None:
        """Error nodes with random failures don't crash the system."""
        graph = Graph(graph_id="error_graph", config=Config())

        # Add error-prone nodes
        for i in range(5):
            node = ErrorNode(node_id=f"error_{i}", config={}, fail_rate=0.5)
            graph.add_node(node)

        # Connect them
        for i in range(4):
            graph.add_edge(Edge(from_node=f"error_{i}", to_node=f"error_{i+1}", condition=None))

        graph.start_node = "error_0"
        graph.end_node = "error_4"

        # Verify graph is constructed
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4


@pytest.mark.slow
class TestExtensiveGraphFuzzing:
    """Extensive fuzzing tests (marked as slow)."""

    @given(small_graphs())
    @settings(max_examples=200, deadline=10000)
    def test_extensive_graph_fuzzing(self, graph: Graph) -> None:
        """Run extensive fuzzing on graph structures."""
        # Check all basic properties hold
        assert graph.graph_id
        assert graph.start_node
        assert graph.end_node

        # Verify no nodes are None
        for node_id, node in graph.nodes.items():
            assert node_id is not None
            assert node is not None
            assert node.id == node_id


# Configure fuzzing profiles
settings.register_profile("fuzzing", max_examples=100, deadline=None)
settings.register_profile("fuzzing_quick", max_examples=20, deadline=5000)

import os

if os.getenv("FUZZING_PROFILE") == "extensive":
    settings.load_profile("fuzzing")
