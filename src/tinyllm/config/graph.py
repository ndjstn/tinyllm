"""Graph configuration models for TinyLLM.

This module defines the Pydantic models used to define and validate
graph structures loaded from YAML files.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    """Types of nodes in the graph."""

    ENTRY = "entry"
    EXIT = "exit"
    ROUTER = "router"
    MODEL = "model"
    TOOL = "tool"
    GATE = "gate"
    TRANSFORM = "transform"
    LOOP = "loop"
    FANOUT = "dynamic_fanout"


class EdgeDefinition(BaseModel):
    """Definition of an edge between nodes."""

    model_config = {"extra": "forbid"}

    from_node: str = Field(
        description="Source node ID", pattern=r"^[a-z][a-z0-9_\.]*$"
    )
    to_node: str = Field(
        description="Target node ID", pattern=r"^[a-z][a-z0-9_\.]*$"
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Edge weight for routing"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Optional condition expression (e.g., 'route == code')",
    )


class NodeDefinition(BaseModel):
    """Complete node definition for graph construction."""

    model_config = {"extra": "forbid"}

    id: str = Field(
        description="Unique node identifier", pattern=r"^[a-z][a-z0-9_\.]*$"
    )
    type: NodeType = Field(description="Node type")
    name: Optional[str] = Field(default=None, description="Human-readable name")
    description: Optional[str] = Field(default=None, description="Node description")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Node-specific configuration"
    )


class GraphMetadata(BaseModel):
    """Metadata about a graph."""

    model_config = {"extra": "allow"}

    created_at: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class GraphDefinition(BaseModel):
    """Complete graph definition for loading from YAML."""

    model_config = {"extra": "forbid"}

    id: str = Field(
        description="Unique graph identifier", pattern=r"^[a-z][a-z0-9_\.]*$"
    )
    version: str = Field(
        description="Semantic version", pattern=r"^\d+\.\d+\.\d+$"
    )
    name: str = Field(description="Human-readable graph name")
    description: Optional[str] = Field(default=None)

    metadata: GraphMetadata = Field(default_factory=GraphMetadata)

    nodes: List[NodeDefinition] = Field(
        description="List of node definitions", min_length=1
    )
    edges: List[EdgeDefinition] = Field(
        description="List of edge definitions", default_factory=list
    )

    entry_points: List[str] = Field(
        description="Node IDs that can be entry points", min_length=1
    )
    exit_points: List[str] = Field(
        description="Node IDs that are exit points", min_length=1
    )
    protected: List[str] = Field(
        default_factory=list,
        description="Node IDs that cannot be pruned during expansion",
    )

    @model_validator(mode="after")
    def validate_graph(self) -> "GraphDefinition":
        """Validate graph structure."""
        node_ids = {n.id for n in self.nodes}

        # Validate entry points exist
        for entry in self.entry_points:
            if entry not in node_ids:
                raise ValueError(f"Entry point '{entry}' not found in nodes")

        # Validate exit points exist
        for exit_point in self.exit_points:
            if exit_point not in node_ids:
                raise ValueError(f"Exit point '{exit_point}' not found in nodes")

        # Validate protected nodes exist
        for protected in self.protected:
            if protected not in node_ids:
                raise ValueError(f"Protected node '{protected}' not found in nodes")

        # Validate edges reference existing nodes
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(
                    f"Edge from_node '{edge.from_node}' not found in nodes"
                )
            if edge.to_node not in node_ids:
                raise ValueError(f"Edge to_node '{edge.to_node}' not found in nodes")

        # Validate entry nodes are of type ENTRY
        for entry in self.entry_points:
            node = next(n for n in self.nodes if n.id == entry)
            if node.type != NodeType.ENTRY:
                raise ValueError(
                    f"Entry point '{entry}' must be of type ENTRY, got {node.type}"
                )

        # Validate exit nodes are of type EXIT
        for exit_point in self.exit_points:
            node = next(n for n in self.nodes if n.id == exit_point)
            if node.type != NodeType.EXIT:
                raise ValueError(
                    f"Exit point '{exit_point}' must be of type EXIT, got {node.type}"
                )

        return self

    def get_node(self, node_id: str) -> Optional[NodeDefinition]:
        """Get a node definition by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_outgoing_edges(self, node_id: str) -> List[EdgeDefinition]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.from_node == node_id]

    def get_incoming_edges(self, node_id: str) -> List[EdgeDefinition]:
        """Get all edges terminating at a node."""
        return [e for e in self.edges if e.to_node == node_id]
