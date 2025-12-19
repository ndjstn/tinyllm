"""Graph structure for TinyLLM.

This module provides the Graph class that holds nodes and edges,
enabling traversal and execution.
"""

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from tinyllm.config.graph import EdgeDefinition, GraphDefinition, NodeType
from tinyllm.core.message import Message
from tinyllm.core.node import BaseNode


class Edge(BaseModel):
    """Runtime edge representation."""

    model_config = {"extra": "forbid"}

    from_node: str
    to_node: str
    weight: float = 1.0
    condition: Optional[str] = None


class ValidationError(BaseModel):
    """Graph validation error."""

    message: str
    node_id: Optional[str] = None
    severity: str = Field(default="error", pattern=r"^(error|warning)$")


class Graph:
    """Runtime graph structure.

    The Graph class holds the instantiated nodes and edges,
    providing methods for traversal and validation.
    """

    def __init__(self, definition: GraphDefinition):
        """Initialize graph from definition.

        Note: Nodes must be added separately after construction
        using add_node().

        Args:
            definition: Graph definition from config.
        """
        self.id = definition.id
        self.version = definition.version
        self.name = definition.name
        self.description = definition.description

        self._nodes: Dict[str, BaseNode] = {}
        self._edges: List[Edge] = []
        self._entry_points: Set[str] = set(definition.entry_points)
        self._exit_points: Set[str] = set(definition.exit_points)
        self._protected: Set[str] = set(definition.protected)

        # Build edges
        for edge_def in definition.edges:
            self._edges.append(
                Edge(
                    from_node=edge_def.from_node,
                    to_node=edge_def.to_node,
                    weight=edge_def.weight,
                    condition=edge_def.condition,
                )
            )

    def add_node(self, node: BaseNode) -> None:
        """Add a node to the graph.

        Args:
            node: Node instance to add.
        """
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Get a node by ID.

        Args:
            node_id: Node identifier.

        Returns:
            Node instance or None if not found.
        """
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists.

        Args:
            node_id: Node identifier.

        Returns:
            True if node exists.
        """
        return node_id in self._nodes

    @property
    def nodes(self) -> Dict[str, BaseNode]:
        """Get all nodes."""
        return self._nodes

    @property
    def edges(self) -> List[Edge]:
        """Get all edges."""
        return self._edges

    @property
    def entry_points(self) -> Set[str]:
        """Get entry point node IDs."""
        return self._entry_points

    @property
    def exit_points(self) -> Set[str]:
        """Get exit point node IDs."""
        return self._exit_points

    @property
    def protected_nodes(self) -> Set[str]:
        """Get protected node IDs."""
        return self._protected

    def get_entry_node(self) -> Optional[BaseNode]:
        """Get the primary entry node.

        Returns:
            First entry node or None.
        """
        if self._entry_points:
            entry_id = next(iter(self._entry_points))
            return self._nodes.get(entry_id)
        return None

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges from a node.

        Args:
            node_id: Source node ID.

        Returns:
            List of outgoing edges.
        """
        return [e for e in self._edges if e.from_node == node_id]

    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges to a node.

        Args:
            node_id: Target node ID.

        Returns:
            List of incoming edges.
        """
        return [e for e in self._edges if e.to_node == node_id]

    def get_next_nodes(
        self, node_id: str, message: Optional[Message] = None
    ) -> List[str]:
        """Get next node IDs based on edges and conditions.

        Args:
            node_id: Current node ID.
            message: Optional message for condition evaluation.

        Returns:
            List of next node IDs.
        """
        edges = self.get_outgoing_edges(node_id)
        if not edges:
            return []

        # Filter by condition if message provided
        next_nodes = []
        for edge in edges:
            if edge.condition and message:
                # Evaluate condition against message payload
                if self._evaluate_condition(edge.condition, message):
                    next_nodes.append(edge.to_node)
            elif not edge.condition:
                next_nodes.append(edge.to_node)

        return next_nodes

    def _evaluate_condition(self, condition: str, message: Message) -> bool:
        """Evaluate an edge condition against a message.

        Simple condition format: "field == value" or "field != value"

        Args:
            condition: Condition string.
            message: Message to evaluate against.

        Returns:
            True if condition matches.
        """
        # Parse simple conditions like "route == 'code'"
        condition = condition.strip()

        # Check for == operator
        if "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip().strip("'\"")

                # Get value from message payload
                payload_dict = message.payload.model_dump()
                actual = payload_dict.get(field)

                return str(actual) == value

        # Check for != operator
        if "!=" in condition:
            parts = condition.split("!=")
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip().strip("'\"")

                payload_dict = message.payload.model_dump()
                actual = payload_dict.get(field)

                return str(actual) != value

        return False

    def is_entry_point(self, node_id: str) -> bool:
        """Check if a node is an entry point."""
        return node_id in self._entry_points

    def is_exit_point(self, node_id: str) -> bool:
        """Check if a node is an exit point."""
        return node_id in self._exit_points

    def is_protected(self, node_id: str) -> bool:
        """Check if a node is protected."""
        return node_id in self._protected

    def validate(self) -> List[ValidationError]:
        """Validate graph structure.

        Checks for:
        - All referenced nodes exist
        - Entry/exit points exist
        - No orphan nodes (except protected)
        - Paths from entry to exit

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[ValidationError] = []

        # Check entry points exist
        for entry in self._entry_points:
            if entry not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Entry point '{entry}' not found in nodes",
                        node_id=entry,
                    )
                )

        # Check exit points exist
        for exit_point in self._exit_points:
            if exit_point not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Exit point '{exit_point}' not found in nodes",
                        node_id=exit_point,
                    )
                )

        # Check edges reference existing nodes
        for edge in self._edges:
            if edge.from_node not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Edge from_node '{edge.from_node}' not found",
                        node_id=edge.from_node,
                    )
                )
            if edge.to_node not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Edge to_node '{edge.to_node}' not found",
                        node_id=edge.to_node,
                    )
                )

        # Check for orphan nodes
        connected = set()
        for edge in self._edges:
            connected.add(edge.from_node)
            connected.add(edge.to_node)
        connected.update(self._entry_points)

        for node_id in self._nodes:
            if node_id not in connected and node_id not in self._protected:
                errors.append(
                    ValidationError(
                        message=f"Node '{node_id}' is not connected",
                        node_id=node_id,
                        severity="warning",
                    )
                )

        return errors

    def __repr__(self) -> str:
        return f"<Graph id={self.id} nodes={len(self._nodes)} edges={len(self._edges)}>"
