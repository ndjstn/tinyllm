"""Graph versioning system for TinyLLM.

Provides version tracking, storage, and rollback capabilities for graph definitions.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

from tinyllm.config.graph import GraphDefinition


class GraphVersion(BaseModel):
    """A versioned snapshot of a graph definition."""

    version: str = Field(description="Semantic version string (e.g., '1.0.0')")
    graph_id: str = Field(description="ID of the graph")
    graph_hash: str = Field(description="SHA256 hash of graph content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_version: Optional[str] = Field(default=None, description="Parent version this was based on")
    message: str = Field(default="", description="Version commit message")
    changes: List[str] = Field(default_factory=list, description="List of changes from parent")


class VersionHistory(BaseModel):
    """Complete version history for a graph."""

    graph_id: str
    versions: List[GraphVersion] = Field(default_factory=list)
    current_version: Optional[str] = None


class GraphVersionManager:
    """Manages graph versions with storage and retrieval.

    Stores graph versions as YAML files in a versioned directory structure:
    storage_path/
      history.json          # Version history metadata
      v1.0.0/
        graph.yaml          # Graph definition
        metadata.json       # Version metadata
      v1.1.0/
        graph.yaml
        metadata.json
    """

    def __init__(self, storage_path: Path | str):
        """Initialize version manager.

        Args:
            storage_path: Directory to store versioned graphs.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._history_file = self.storage_path / "history.json"
        self._history: Optional[VersionHistory] = None

    def _load_history(self) -> VersionHistory:
        """Load version history from storage."""
        if self._history is not None:
            return self._history

        if self._history_file.exists():
            with open(self._history_file) as f:
                data = json.load(f)
                self._history = VersionHistory(**data)
        else:
            self._history = VersionHistory(graph_id="")

        return self._history

    def _save_history(self) -> None:
        """Save version history to storage."""
        if self._history is None:
            return

        with open(self._history_file, "w") as f:
            json.dump(self._history.model_dump(mode="json"), f, indent=2, default=str)

    def _compute_hash(self, graph: GraphDefinition) -> str:
        """Compute SHA256 hash of graph definition."""
        # Serialize to deterministic JSON
        content = graph.model_dump_json(indent=2)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _next_version(self, current: Optional[str], bump: str = "patch") -> str:
        """Calculate next version number.

        Args:
            current: Current version string.
            bump: Type of bump ('major', 'minor', 'patch').

        Returns:
            Next version string.
        """
        if current is None:
            return "1.0.0"

        parts = current.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if bump == "major":
            return f"{major + 1}.0.0"
        elif bump == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"

    def save_version(
        self,
        graph: GraphDefinition,
        message: str = "",
        version: Optional[str] = None,
        bump: str = "patch",
    ) -> GraphVersion:
        """Save a new version of the graph.

        Args:
            graph: Graph definition to save.
            message: Commit message describing changes.
            version: Explicit version string (auto-generated if not provided).
            bump: Version bump type if auto-generating ('major', 'minor', 'patch').

        Returns:
            Created GraphVersion.
        """
        history = self._load_history()

        # Determine version number
        if version is None:
            version = self._next_version(history.current_version, bump)

        # Check if version already exists
        existing = [v for v in history.versions if v.version == version]
        if existing:
            raise ValueError(f"Version {version} already exists")

        # Compute changes from parent
        changes = []
        if history.current_version:
            parent_graph = self.load_version(history.current_version)
            if parent_graph:
                changes = self._compute_changes(parent_graph, graph)

        # Create version record
        graph_version = GraphVersion(
            version=version,
            graph_id=graph.id,
            graph_hash=self._compute_hash(graph),
            parent_version=history.current_version,
            message=message,
            changes=changes,
        )

        # Save graph to versioned directory
        version_dir = self.storage_path / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save graph YAML
        graph_file = version_dir / "graph.yaml"
        with open(graph_file, "w") as f:
            yaml.dump(graph.model_dump(mode="json"), f, default_flow_style=False)

        # Save metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(graph_version.model_dump(mode="json"), f, indent=2, default=str)

        # Update history
        history.graph_id = graph.id
        history.versions.append(graph_version)
        history.current_version = version
        self._save_history()

        return graph_version

    def load_version(self, version: str) -> Optional[GraphDefinition]:
        """Load a specific version of the graph.

        Args:
            version: Version string to load.

        Returns:
            GraphDefinition or None if not found.
        """
        version_dir = self.storage_path / f"v{version}"
        graph_file = version_dir / "graph.yaml"

        if not graph_file.exists():
            return None

        with open(graph_file) as f:
            data = yaml.safe_load(f)

        return GraphDefinition(**data)

    def load_current(self) -> Optional[GraphDefinition]:
        """Load the current version of the graph.

        Returns:
            Current GraphDefinition or None.
        """
        history = self._load_history()
        if history.current_version:
            return self.load_version(history.current_version)
        return None

    def list_versions(self) -> List[GraphVersion]:
        """List all versions.

        Returns:
            List of GraphVersion records, newest first.
        """
        history = self._load_history()
        return list(reversed(history.versions))

    def get_version_info(self, version: str) -> Optional[GraphVersion]:
        """Get metadata for a specific version.

        Args:
            version: Version string.

        Returns:
            GraphVersion or None.
        """
        history = self._load_history()
        for v in history.versions:
            if v.version == version:
                return v
        return None

    def rollback(self, version: str) -> GraphDefinition:
        """Rollback to a previous version.

        Creates a new version based on the target version.

        Args:
            version: Version to rollback to.

        Returns:
            The rolled-back GraphDefinition.

        Raises:
            ValueError: If version not found.
        """
        graph = self.load_version(version)
        if graph is None:
            raise ValueError(f"Version {version} not found")

        # Save as new version with rollback message
        self.save_version(
            graph,
            message=f"Rollback to version {version}",
            bump="patch",
        )

        return graph

    def diff(self, version1: str, version2: str) -> List[str]:
        """Compare two versions and return differences.

        Args:
            version1: First version.
            version2: Second version.

        Returns:
            List of difference descriptions.
        """
        graph1 = self.load_version(version1)
        graph2 = self.load_version(version2)

        if graph1 is None:
            raise ValueError(f"Version {version1} not found")
        if graph2 is None:
            raise ValueError(f"Version {version2} not found")

        return self._compute_changes(graph1, graph2)

    def _compute_changes(
        self, old_graph: GraphDefinition, new_graph: GraphDefinition
    ) -> List[str]:
        """Compute list of changes between two graph versions."""
        changes = []

        # Compare nodes
        old_node_ids = {n.id for n in old_graph.nodes}
        new_node_ids = {n.id for n in new_graph.nodes}

        added_nodes = new_node_ids - old_node_ids
        removed_nodes = old_node_ids - new_node_ids

        for node_id in added_nodes:
            changes.append(f"Added node: {node_id}")
        for node_id in removed_nodes:
            changes.append(f"Removed node: {node_id}")

        # Compare edges
        old_edges = {(e.from_node, e.to_node) for e in old_graph.edges}
        new_edges = {(e.from_node, e.to_node) for e in new_graph.edges}

        added_edges = new_edges - old_edges
        removed_edges = old_edges - new_edges

        for from_n, to_n in added_edges:
            changes.append(f"Added edge: {from_n} -> {to_n}")
        for from_n, to_n in removed_edges:
            changes.append(f"Removed edge: {from_n} -> {to_n}")

        # Compare metadata
        if old_graph.name != new_graph.name:
            changes.append(f"Renamed: '{old_graph.name}' -> '{new_graph.name}'")
        if old_graph.description != new_graph.description:
            changes.append("Updated description")

        return changes

    def delete_version(self, version: str) -> bool:
        """Delete a specific version.

        Args:
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        history = self._load_history()

        # Cannot delete current version
        if history.current_version == version:
            raise ValueError("Cannot delete current version. Rollback first.")

        # Find and remove from history
        original_count = len(history.versions)
        history.versions = [v for v in history.versions if v.version != version]

        if len(history.versions) == original_count:
            return False

        # Delete directory
        version_dir = self.storage_path / f"v{version}"
        if version_dir.exists():
            shutil.rmtree(version_dir)

        self._save_history()
        return True

    @property
    def current_version(self) -> Optional[str]:
        """Get current version string."""
        return self._load_history().current_version
