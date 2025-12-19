"""Tests for graph configuration and structure."""

import pytest
from pydantic import ValidationError

from tinyllm.config.graph import (
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    NodeType,
)


class TestNodeDefinition:
    """Tests for NodeDefinition model."""

    def test_valid_node(self):
        """Test creating a valid node definition."""
        node = NodeDefinition(
            id="entry.main",
            type=NodeType.ENTRY,
            name="Main Entry",
            config={"timeout_ms": 5000},
        )
        assert node.id == "entry.main"
        assert node.type == NodeType.ENTRY
        assert node.name == "Main Entry"

    def test_invalid_id_pattern(self):
        """Test that invalid ID patterns are rejected."""
        with pytest.raises(ValidationError):
            NodeDefinition(
                id="Invalid-ID",  # Contains uppercase and hyphen
                type=NodeType.ENTRY,
            )

    def test_node_types(self):
        """Test all node types are valid."""
        for node_type in NodeType:
            node = NodeDefinition(
                id=f"test.{node_type.value}",
                type=node_type,
            )
            assert node.type == node_type


class TestEdgeDefinition:
    """Tests for EdgeDefinition model."""

    def test_valid_edge(self):
        """Test creating a valid edge."""
        edge = EdgeDefinition(
            from_node="entry.main",
            to_node="router.task",
            weight=0.8,
            condition="route == 'code'",
        )
        assert edge.from_node == "entry.main"
        assert edge.to_node == "router.task"
        assert edge.weight == 0.8
        assert edge.condition == "route == 'code'"

    def test_default_weight(self):
        """Test default edge weight is 1.0."""
        edge = EdgeDefinition(from_node="a", to_node="b")
        assert edge.weight == 1.0

    def test_weight_bounds(self):
        """Test weight must be between 0 and 1."""
        with pytest.raises(ValidationError):
            EdgeDefinition(from_node="a", to_node="b", weight=1.5)
        with pytest.raises(ValidationError):
            EdgeDefinition(from_node="a", to_node="b", weight=-0.1)


class TestGraphDefinition:
    """Tests for GraphDefinition model."""

    def test_valid_graph(self):
        """Test creating a valid graph definition."""
        graph = GraphDefinition(
            id="graph.test",
            version="1.0.0",
            name="Test Graph",
            nodes=[
                NodeDefinition(id="entry.main", type=NodeType.ENTRY),
                NodeDefinition(id="exit.success", type=NodeType.EXIT),
            ],
            edges=[
                EdgeDefinition(from_node="entry.main", to_node="exit.success"),
            ],
            entry_points=["entry.main"],
            exit_points=["exit.success"],
        )
        assert graph.id == "graph.test"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_entry_point_must_exist(self):
        """Test that entry points must reference existing nodes."""
        with pytest.raises(ValidationError) as exc_info:
            GraphDefinition(
                id="graph.test",
                version="1.0.0",
                name="Test",
                nodes=[NodeDefinition(id="entry.main", type=NodeType.ENTRY)],
                entry_points=["nonexistent"],
                exit_points=["exit.success"],
            )
        assert "Entry point 'nonexistent' not found" in str(exc_info.value)

    def test_entry_must_be_entry_type(self):
        """Test that entry points must be ENTRY type nodes."""
        with pytest.raises(ValidationError) as exc_info:
            GraphDefinition(
                id="graph.test",
                version="1.0.0",
                name="Test",
                nodes=[
                    NodeDefinition(id="model.main", type=NodeType.MODEL),
                    NodeDefinition(id="exit.success", type=NodeType.EXIT),
                ],
                entry_points=["model.main"],
                exit_points=["exit.success"],
            )
        assert "must be of type ENTRY" in str(exc_info.value)

    def test_get_node(self):
        """Test get_node helper method."""
        graph = GraphDefinition(
            id="graph.test",
            version="1.0.0",
            name="Test",
            nodes=[
                NodeDefinition(id="entry.main", type=NodeType.ENTRY),
                NodeDefinition(id="exit.success", type=NodeType.EXIT),
            ],
            entry_points=["entry.main"],
            exit_points=["exit.success"],
        )
        node = graph.get_node("entry.main")
        assert node is not None
        assert node.id == "entry.main"

        assert graph.get_node("nonexistent") is None

    def test_get_outgoing_edges(self):
        """Test get_outgoing_edges helper method."""
        graph = GraphDefinition(
            id="graph.test",
            version="1.0.0",
            name="Test",
            nodes=[
                NodeDefinition(id="entry.main", type=NodeType.ENTRY),
                NodeDefinition(id="router.task", type=NodeType.ROUTER),
                NodeDefinition(id="exit.success", type=NodeType.EXIT),
            ],
            edges=[
                EdgeDefinition(from_node="entry.main", to_node="router.task"),
                EdgeDefinition(from_node="router.task", to_node="exit.success"),
            ],
            entry_points=["entry.main"],
            exit_points=["exit.success"],
        )

        edges = graph.get_outgoing_edges("entry.main")
        assert len(edges) == 1
        assert edges[0].to_node == "router.task"
