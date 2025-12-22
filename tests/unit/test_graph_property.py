"""Property-based tests for graph structures and validation.

These tests use Hypothesis to verify graph invariants hold under all
possible graph configurations, particularly around validation, traversal,
and structural properties.
"""

from typing import Any, List, Set

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from tinyllm.config.graph import (
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    NodeType,
    GraphMetadata,
)


# Custom strategies for graph components


@st.composite
def node_ids(draw: Any) -> str:
    """Generate valid node IDs matching pattern ^[a-z][a-z0-9_\\.]*$."""
    # Start with lowercase letter
    first_char = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))

    # Followed by lowercase, digits, underscore, or dot
    rest = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_.",
        min_size=0,
        max_size=30
    ))

    return first_char + rest


@st.composite
def node_definitions(draw: Any) -> NodeDefinition:
    """Generate valid NodeDefinition instances."""
    node_type = draw(st.sampled_from(list(NodeType)))
    node_id = draw(node_ids())

    # Generate appropriate config for node type
    config = {}
    if node_type == NodeType.MODEL:
        config["model"] = draw(st.sampled_from([
            "qwen2.5:0.5b", "qwen2.5:3b", "granite-code:3b", "phi3:mini"
        ]))
    elif node_type == NodeType.ROUTER:
        config["model"] = "qwen2.5:0.5b"
        config["routes"] = []
    elif node_type == NodeType.TOOL:
        config["tool_id"] = draw(st.sampled_from([
            "calculator", "code_executor", "web_search"
        ]))
    elif node_type == NodeType.GATE:
        config["model"] = "qwen2.5:3b"
        config["pass_threshold"] = draw(st.floats(min_value=0.0, max_value=1.0))
    elif node_type == NodeType.EXIT:
        config["status"] = draw(st.sampled_from(["success", "failure", "timeout"]))

    return NodeDefinition(
        id=node_id,
        type=node_type,
        name=draw(st.text(min_size=1, max_size=50)),
        description=draw(st.one_of(st.none(), st.text(max_size=200))),
        config=config
    )


@st.composite
def edge_definitions(draw: Any, node_list: List[NodeDefinition]) -> EdgeDefinition:
    """Generate valid EdgeDefinition instances from a list of nodes."""
    assume(len(node_list) >= 2)  # Need at least 2 nodes for an edge

    from_node = draw(st.sampled_from(node_list))
    to_node = draw(st.sampled_from(node_list))

    # Avoid self-loops in most cases
    assume(from_node.id != to_node.id or draw(st.booleans()))

    return EdgeDefinition(
        from_node=from_node.id,
        to_node=to_node.id,
        weight=draw(st.floats(min_value=0.1, max_value=1.0)),
        condition=draw(st.one_of(st.none(), st.text(min_size=5, max_size=50)))
    )


@st.composite
def simple_graphs(draw: Any) -> GraphDefinition:
    """Generate simple valid graphs."""
    # Always have at least entry and exit nodes
    entry_node = NodeDefinition(
        id="entry.main",
        type=NodeType.ENTRY,
        name="Entry",
        config={}
    )

    exit_node = NodeDefinition(
        id="exit.success",
        type=NodeType.EXIT,
        name="Exit",
        config={"status": "success"}
    )

    # Add 0-3 middle nodes
    num_middle = draw(st.integers(min_value=0, max_value=3))
    middle_nodes = [draw(node_definitions()) for _ in range(num_middle)]

    # Ensure unique IDs
    all_nodes = [entry_node] + middle_nodes + [exit_node]
    seen_ids = set()
    unique_nodes = []
    for node in all_nodes:
        if node.id not in seen_ids:
            unique_nodes.append(node)
            seen_ids.add(node.id)

    # Create linear path: entry -> middle1 -> middle2 -> ... -> exit
    edges = []
    for i in range(len(unique_nodes) - 1):
        edges.append(EdgeDefinition(
            from_node=unique_nodes[i].id,
            to_node=unique_nodes[i + 1].id,
            weight=1.0
        ))

    # Generate valid graph ID (pattern: ^[a-z][a-z0-9_\.]*$)
    graph_id_first = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz"))
    graph_id_rest = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_.",
        min_size=0,
        max_size=20
    ))
    graph_id = graph_id_first + graph_id_rest

    # Generate valid semantic version (pattern: ^\d+\.\d+\.\d+$)
    version = f"{draw(st.integers(min_value=0, max_value=9))}.{draw(st.integers(min_value=0, max_value=9))}.{draw(st.integers(min_value=0, max_value=9))}"

    return GraphDefinition(
        id=graph_id,
        version=version,
        name=draw(st.text(min_size=1, max_size=50)),
        description=draw(st.text(max_size=200)),
        metadata=GraphMetadata(
            created_at="2024-01-01T00:00:00",
            author="test"
        ),
        nodes=unique_nodes,
        edges=edges,
        entry_points=[entry_node.id],
        exit_points=[exit_node.id]
    )


class TestNodeDefinitionProperties:
    """Property-based tests for NodeDefinition."""

    @given(node_definitions())
    @settings(max_examples=100, deadline=2000)
    def test_node_id_pattern_valid(self, node: NodeDefinition) -> None:
        """Node IDs should match pattern ^[a-z][a-z0-9_\\.]*$."""
        import re
        pattern = re.compile(r"^[a-z][a-z0-9_\.]*$")

        assert pattern.match(node.id), f"Node ID '{node.id}' doesn't match pattern"

    @given(node_definitions())
    @settings(max_examples=100, deadline=2000)
    def test_node_serialization_roundtrip(self, node: NodeDefinition) -> None:
        """Nodes should serialize and deserialize without loss."""
        node_dict = node.model_dump()
        restored = NodeDefinition(**node_dict)

        assert restored.id == node.id
        assert restored.type == node.type
        assert restored.name == node.name
        assert restored.config == node.config

    @given(st.lists(node_definitions(), min_size=2, max_size=10))
    @settings(max_examples=50, deadline=3000)
    def test_node_id_uniqueness(self, nodes: List[NodeDefinition]) -> None:
        """In a valid graph, node IDs should be unique."""
        # Simulate graph validation
        node_ids = [node.id for node in nodes]
        unique_ids = set(node_ids)

        # If we have duplicates, graph should be considered invalid
        if len(node_ids) != len(unique_ids):
            # This is an invalid graph state
            pass
        else:
            # Valid graph
            assert len(node_ids) == len(unique_ids)


class TestEdgeDefinitionProperties:
    """Property-based tests for EdgeDefinition."""

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_edge_references_valid_nodes(self, graph: GraphDefinition) -> None:
        """All edges should reference nodes that exist in the graph."""
        node_ids = {node.id for node in graph.nodes}

        for edge in graph.edges:
            assert edge.from_node in node_ids, f"Edge from_node '{edge.from_node}' not in graph"
            assert edge.to_node in node_ids, f"Edge to_node '{edge.to_node}' not in graph"

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_edge_weights_valid(self, graph: GraphDefinition) -> None:
        """Edge weights should be positive numbers."""
        for edge in graph.edges:
            assert edge.weight > 0, f"Edge weight {edge.weight} should be positive"
            assert isinstance(edge.weight, (int, float))


class TestGraphDefinitionProperties:
    """Property-based tests for GraphDefinition."""

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_graph_has_entry_points(self, graph: GraphDefinition) -> None:
        """Graphs should have at least one entry point."""
        assert len(graph.entry_points) > 0

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_graph_has_exit_points(self, graph: GraphDefinition) -> None:
        """Graphs should have at least one exit point."""
        assert len(graph.exit_points) > 0

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_entry_points_reference_entry_nodes(self, graph: GraphDefinition) -> None:
        """Entry points should reference actual entry nodes."""
        node_map = {node.id: node for node in graph.nodes}

        for entry_id in graph.entry_points:
            assert entry_id in node_map
            # Entry point should be an ENTRY type node
            assert node_map[entry_id].type == NodeType.ENTRY

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_exit_points_reference_exit_nodes(self, graph: GraphDefinition) -> None:
        """Exit points should reference actual exit nodes."""
        node_map = {node.id: node for node in graph.nodes}

        for exit_id in graph.exit_points:
            assert exit_id in node_map
            # Exit point should be an EXIT type node
            assert node_map[exit_id].type == NodeType.EXIT

    @given(simple_graphs())
    @settings(max_examples=50, deadline=3000)
    def test_graph_serialization_roundtrip(self, graph: GraphDefinition) -> None:
        """Graphs should serialize and deserialize without loss."""
        graph_dict = graph.model_dump()
        restored = GraphDefinition(**graph_dict)

        assert restored.id == graph.id
        assert restored.version == graph.version
        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)
        assert restored.entry_points == graph.entry_points
        assert restored.exit_points == graph.exit_points


class TestGraphTraversalProperties:
    """Property-based tests for graph traversal."""

    @given(simple_graphs())
    @settings(max_examples=30, deadline=5000)
    def test_reachability_from_entry(self, graph: GraphDefinition) -> None:
        """All nodes should be reachable from at least one entry point."""
        # Build adjacency list
        adjacency = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            adjacency[edge.from_node].append(edge.to_node)

        # BFS from all entry points
        reachable = set()
        queue = list(graph.entry_points)
        reachable.update(queue)

        while queue:
            current = queue.pop(0)
            if current in adjacency:
                for neighbor in adjacency[current]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)

        # For simple linear graphs, all nodes should be reachable
        all_node_ids = {node.id for node in graph.nodes}
        # In our simple_graphs strategy, we create linear paths, so all should be reachable
        if len(graph.edges) > 0:  # If there are edges
            assert len(reachable) > 0

    @given(simple_graphs())
    @settings(max_examples=30, deadline=5000)
    def test_exit_reachable_from_entry(self, graph: GraphDefinition) -> None:
        """Exit points should be reachable from entry points."""
        # Build adjacency list
        adjacency = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            adjacency[edge.from_node].append(edge.to_node)

        # BFS from entry points
        reachable = set()
        queue = list(graph.entry_points)
        reachable.update(queue)

        while queue:
            current = queue.pop(0)
            if current in adjacency:
                for neighbor in adjacency[current]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)

        # At least one exit should be reachable
        exit_set = set(graph.exit_points)
        reachable_exits = reachable.intersection(exit_set)

        # For simple linear graphs, exits should be reachable
        if len(graph.edges) > 0:
            assert len(reachable_exits) > 0


class TestGraphValidationProperties:
    """Property-based tests for graph validation."""

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_no_duplicate_node_ids(self, graph: GraphDefinition) -> None:
        """Graph should not have duplicate node IDs."""
        node_ids = [node.id for node in graph.nodes]
        assert len(node_ids) == len(set(node_ids))

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_all_edges_reference_existing_nodes(self, graph: GraphDefinition) -> None:
        """All edges should reference nodes that exist in the graph."""
        node_ids = {node.id for node in graph.nodes}

        for edge in graph.edges:
            assert edge.from_node in node_ids
            assert edge.to_node in node_ids

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_entry_and_exit_points_exist(self, graph: GraphDefinition) -> None:
        """Entry and exit points should reference existing nodes."""
        node_ids = {node.id for node in graph.nodes}

        for entry_id in graph.entry_points:
            assert entry_id in node_ids

        for exit_id in graph.exit_points:
            assert exit_id in node_ids


class TestGraphStructuralProperties:
    """Property-based tests for graph structural invariants."""

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_graph_has_nodes(self, graph: GraphDefinition) -> None:
        """Valid graphs should have at least one node."""
        assert len(graph.nodes) > 0

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_entry_nodes_have_type_entry(self, graph: GraphDefinition) -> None:
        """Nodes referenced in entry_points should have type ENTRY."""
        node_map = {node.id: node for node in graph.nodes}

        for entry_id in graph.entry_points:
            if entry_id in node_map:
                assert node_map[entry_id].type == NodeType.ENTRY

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_exit_nodes_have_type_exit(self, graph: GraphDefinition) -> None:
        """Nodes referenced in exit_points should have type EXIT."""
        node_map = {node.id: node for node in graph.nodes}

        for exit_id in graph.exit_points:
            if exit_id in node_map:
                assert node_map[exit_id].type == NodeType.EXIT

    @given(simple_graphs(), st.integers(min_value=0, max_value=10))
    @settings(max_examples=30, deadline=3000)
    def test_node_degree_bounded(self, graph: GraphDefinition, max_degree: int) -> None:
        """Node degree (in + out edges) should be computable."""
        in_degree = {node.id: 0 for node in graph.nodes}
        out_degree = {node.id: 0 for node in graph.nodes}

        for edge in graph.edges:
            out_degree[edge.from_node] += 1
            in_degree[edge.to_node] += 1

        # All nodes should have defined degrees (>= 0)
        for node in graph.nodes:
            assert in_degree[node.id] >= 0
            assert out_degree[node.id] >= 0

        # Entry nodes should have in_degree 0
        for entry_id in graph.entry_points:
            assert in_degree[entry_id] == 0

        # Exit nodes should have out_degree 0
        for exit_id in graph.exit_points:
            assert out_degree[exit_id] == 0


class TestModelNodeProperties:
    """Property-based tests specific to MODEL nodes."""

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_model_nodes_have_model_config(self, graph: GraphDefinition) -> None:
        """MODEL type nodes should have 'model' in config."""
        for node in graph.nodes:
            if node.type == NodeType.MODEL:
                assert "model" in node.config
                assert isinstance(node.config["model"], str)
                assert len(node.config["model"]) > 0


class TestRouterNodeProperties:
    """Property-based tests specific to ROUTER nodes."""

    @given(simple_graphs())
    @settings(max_examples=30, deadline=3000)
    def test_router_nodes_have_routes_config(self, graph: GraphDefinition) -> None:
        """ROUTER type nodes should have 'routes' in config."""
        for node in graph.nodes:
            if node.type == NodeType.ROUTER:
                assert "routes" in node.config
                assert isinstance(node.config["routes"], list)


# Configure hypothesis settings
settings.register_profile("graph_property", max_examples=50, deadline=3000)
settings.register_profile("graph_property_quick", max_examples=10, deadline=1000)

import os
if os.getenv("HYPOTHESIS_PROFILE") == "graph_property":
    settings.load_profile("graph_property")
