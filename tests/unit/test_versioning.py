"""Tests for graph versioning system."""

import tempfile
from pathlib import Path

import pytest

from tinyllm.config.graph import GraphDefinition, NodeDefinition, EdgeDefinition
from tinyllm.expansion.versioning import (
    GraphVersion,
    GraphVersionManager,
    VersionHistory,
)


def create_test_graph(name: str = "test_graph", nodes: list[str] = None) -> GraphDefinition:
    """Create a test graph definition."""
    if nodes is None:
        nodes = ["entry", "router", "exit"]

    node_defs = []
    for i, node_id in enumerate(nodes):
        if node_id == "entry":
            node_defs.append(NodeDefinition(
                id=node_id,
                type="entry",
                name="Entry",
                config={},
            ))
        elif node_id == "exit":
            node_defs.append(NodeDefinition(
                id=node_id,
                type="exit",
                name="Exit",
                config={},
            ))
        else:
            node_defs.append(NodeDefinition(
                id=node_id,
                type="router",
                name=node_id.title(),
                config={"routes": []},
            ))

    # Create edges connecting nodes sequentially
    edges = []
    for i in range(len(node_defs) - 1):
        edges.append(EdgeDefinition(
            from_node=node_defs[i].id,
            to_node=node_defs[i + 1].id,
        ))

    return GraphDefinition(
        id="test_graph",
        version="1.0.0",
        name=name,
        description="Test graph",
        nodes=node_defs,
        edges=edges,
        entry_points=["entry"],
        exit_points=["exit"],
    )


class TestGraphVersion:
    """Tests for GraphVersion model."""

    def test_create_version(self):
        """Should create a version with required fields."""
        version = GraphVersion(
            version="1.0.0",
            graph_id="test_graph",
            graph_hash="abc123",
            message="Initial version",
        )
        assert version.version == "1.0.0"
        assert version.graph_id == "test_graph"
        assert version.graph_hash == "abc123"
        assert version.message == "Initial version"
        assert version.parent_version is None
        assert version.changes == []

    def test_version_with_parent(self):
        """Should create version with parent reference."""
        version = GraphVersion(
            version="1.1.0",
            graph_id="test_graph",
            graph_hash="def456",
            parent_version="1.0.0",
            changes=["Added node X"],
        )
        assert version.parent_version == "1.0.0"
        assert len(version.changes) == 1


class TestVersionHistory:
    """Tests for VersionHistory model."""

    def test_empty_history(self):
        """Should create empty history."""
        history = VersionHistory(graph_id="test_graph")
        assert history.graph_id == "test_graph"
        assert history.versions == []
        assert history.current_version is None

    def test_history_with_versions(self):
        """Should track multiple versions."""
        v1 = GraphVersion(version="1.0.0", graph_id="test", graph_hash="a")
        v2 = GraphVersion(version="1.1.0", graph_id="test", graph_hash="b")

        history = VersionHistory(
            graph_id="test",
            versions=[v1, v2],
            current_version="1.1.0",
        )
        assert len(history.versions) == 2
        assert history.current_version == "1.1.0"


class TestGraphVersionManager:
    """Tests for GraphVersionManager."""

    def test_init_creates_directory(self):
        """Should create storage directory on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "versions"
            manager = GraphVersionManager(storage)
            assert storage.exists()

    def test_save_first_version(self):
        """Should save first version as 1.0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            version = manager.save_version(graph, "Initial version")

            assert version.version == "1.0.0"
            assert version.graph_id == "test_graph"
            assert version.message == "Initial version"
            assert version.parent_version is None

    def test_save_increments_version(self):
        """Should auto-increment version numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            v1 = manager.save_version(graph, "v1")
            v2 = manager.save_version(graph, "v2")
            v3 = manager.save_version(graph, "v3")

            assert v1.version == "1.0.0"
            assert v2.version == "1.0.1"
            assert v3.version == "1.0.2"

    def test_save_with_minor_bump(self):
        """Should support minor version bumps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            v1 = manager.save_version(graph, "v1")
            v2 = manager.save_version(graph, "v2", bump="minor")

            assert v1.version == "1.0.0"
            assert v2.version == "1.1.0"

    def test_save_with_major_bump(self):
        """Should support major version bumps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            v1 = manager.save_version(graph, "v1")
            v2 = manager.save_version(graph, "v2", bump="major")

            assert v1.version == "1.0.0"
            assert v2.version == "2.0.0"

    def test_save_explicit_version(self):
        """Should allow explicit version string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            version = manager.save_version(graph, "custom", version="3.5.0")
            assert version.version == "3.5.0"

    def test_duplicate_version_rejected(self):
        """Should reject duplicate version numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            manager.save_version(graph, "v1", version="1.0.0")

            with pytest.raises(ValueError, match="already exists"):
                manager.save_version(graph, "v2", version="1.0.0")

    def test_load_version(self):
        """Should load saved version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph("Original Name")

            manager.save_version(graph, "v1")
            loaded = manager.load_version("1.0.0")

            assert loaded is not None
            assert loaded.name == "Original Name"
            assert loaded.id == "test_graph"

    def test_load_nonexistent_version(self):
        """Should return None for nonexistent version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            loaded = manager.load_version("99.0.0")
            assert loaded is None

    def test_load_current(self):
        """Should load current version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph1 = create_test_graph("First")
            graph2 = create_test_graph("Second")

            manager.save_version(graph1, "v1")
            manager.save_version(graph2, "v2")

            current = manager.load_current()
            assert current is not None
            assert current.name == "Second"

    def test_list_versions(self):
        """Should list all versions newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            manager.save_version(graph, "v1")
            manager.save_version(graph, "v2")
            manager.save_version(graph, "v3")

            versions = manager.list_versions()

            assert len(versions) == 3
            # Newest first
            assert versions[0].version == "1.0.2"
            assert versions[1].version == "1.0.1"
            assert versions[2].version == "1.0.0"

    def test_get_version_info(self):
        """Should get metadata for specific version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            manager.save_version(graph, "Test message")
            info = manager.get_version_info("1.0.0")

            assert info is not None
            assert info.message == "Test message"

    def test_rollback(self):
        """Should rollback to previous version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph1 = create_test_graph("Original")
            graph2 = create_test_graph("Modified")

            manager.save_version(graph1, "original")
            manager.save_version(graph2, "modified")

            rolled_back = manager.rollback("1.0.0")

            assert rolled_back.name == "Original"
            assert manager.current_version == "1.0.2"

            # Current should now be the rolled-back version
            current = manager.load_current()
            assert current.name == "Original"

    def test_rollback_nonexistent(self):
        """Should raise error for nonexistent version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))

            with pytest.raises(ValueError, match="not found"):
                manager.rollback("99.0.0")

    def test_diff_nodes(self):
        """Should detect node changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph1 = create_test_graph(nodes=["entry", "router", "exit"])
            graph2 = create_test_graph(nodes=["entry", "router", "model", "exit"])

            manager.save_version(graph1, "v1")
            manager.save_version(graph2, "v2")

            diff = manager.diff("1.0.0", "1.0.1")

            # Should detect added model node
            assert any("Added node" in d for d in diff)

    def test_diff_edges(self):
        """Should detect edge changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))

            graph1 = create_test_graph(nodes=["entry", "exit"])
            graph2 = create_test_graph(nodes=["entry", "router", "exit"])

            manager.save_version(graph1, "v1")
            manager.save_version(graph2, "v2")

            diff = manager.diff("1.0.0", "1.0.1")

            # Should detect edge changes
            edge_changes = [d for d in diff if "edge" in d.lower()]
            assert len(edge_changes) > 0

    def test_diff_metadata(self):
        """Should detect name changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph1 = create_test_graph("Original Name")
            graph2 = create_test_graph("New Name")

            manager.save_version(graph1, "v1")
            manager.save_version(graph2, "v2")

            diff = manager.diff("1.0.0", "1.0.1")

            assert any("Renamed" in d for d in diff)

    def test_delete_version(self):
        """Should delete non-current version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            manager.save_version(graph, "v1")
            manager.save_version(graph, "v2")

            deleted = manager.delete_version("1.0.0")

            assert deleted is True
            assert manager.load_version("1.0.0") is None
            assert len(manager.list_versions()) == 1

    def test_delete_current_version_rejected(self):
        """Should not allow deleting current version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph = create_test_graph()

            manager.save_version(graph, "v1")

            with pytest.raises(ValueError, match="Cannot delete current"):
                manager.delete_version("1.0.0")

    def test_current_version_property(self):
        """Should track current version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))

            assert manager.current_version is None

            graph = create_test_graph()
            manager.save_version(graph, "v1")

            assert manager.current_version == "1.0.0"

    def test_tracks_changes_from_parent(self):
        """Should compute changes from parent version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GraphVersionManager(Path(tmpdir))
            graph1 = create_test_graph("First")
            graph2 = create_test_graph("Second")

            manager.save_version(graph1, "v1")
            v2 = manager.save_version(graph2, "v2")

            assert v2.parent_version == "1.0.0"
            assert any("Renamed" in c for c in v2.changes)
