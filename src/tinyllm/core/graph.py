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
        self.allow_cycles = definition.allow_cycles

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

    def detect_cycles(self) -> List[List[str]]:
        """Detect all cycles in the graph using DFS.

        Returns:
            List of cycles, where each cycle is a list of node IDs.
            Empty list if the graph is acyclic.
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_id: str) -> None:
            """Depth-first search to detect cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            # Get all neighbors
            for edge in self.get_outgoing_edges(node_id):
                neighbor = edge.to_node

                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle - extract it from path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node_id)

        # Run DFS from each unvisited node
        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles

    def is_acyclic(self) -> bool:
        """Quick check if graph is acyclic (DAG).

        Returns:
            True if graph has no cycles, False otherwise.
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node_id: str) -> bool:
            """Check if there's a cycle starting from this node."""
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge in self.get_outgoing_edges(node_id):
                neighbor = edge.to_node
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        # Check all nodes
        for node_id in self._nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False

        return True

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order (execution order).

        Returns:
            List of node IDs in topological order.

        Raises:
            ValueError: If the graph contains cycles.
        """
        if not self.is_acyclic():
            raise ValueError("Cannot perform topological sort on a graph with cycles")

        visited: Set[str] = set()
        stack: List[str] = []

        def dfs(node_id: str) -> None:
            """DFS for topological sorting."""
            visited.add(node_id)

            for edge in self.get_outgoing_edges(node_id):
                neighbor = edge.to_node
                if neighbor not in visited:
                    dfs(neighbor)

            stack.append(node_id)

        # Visit all nodes
        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        # Return reversed stack (topological order)
        return list(reversed(stack))

    def _find_unreachable_nodes(self) -> Set[str]:
        """Find nodes that are not reachable from any entry point.

        Returns:
            Set of unreachable node IDs.
        """
        reachable: Set[str] = set()

        def dfs(node_id: str) -> None:
            """Mark all reachable nodes."""
            if node_id in reachable:
                return
            reachable.add(node_id)

            for edge in self.get_outgoing_edges(node_id):
                dfs(edge.to_node)

        # Start from all entry points
        for entry in self._entry_points:
            if entry in self._nodes:
                dfs(entry)

        # Find unreachable nodes
        unreachable = set(self._nodes.keys()) - reachable
        return unreachable

    def _find_dead_end_nodes(self) -> Set[str]:
        """Find non-exit nodes with no outgoing edges.

        Returns:
            Set of dead-end node IDs.
        """
        dead_ends: Set[str] = set()

        for node_id in self._nodes:
            # Skip exit points
            if node_id in self._exit_points:
                continue

            # Check if node has no outgoing edges
            if not self.get_outgoing_edges(node_id):
                dead_ends.add(node_id)

        return dead_ends

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram format.

        Returns:
            Mermaid diagram string.
        """
        lines = ["graph TD"]

        # Add nodes with styling
        for node_id, node in self._nodes.items():
            label = node.name if hasattr(node, "name") and node.name else node_id

            # Style based on node type
            if node_id in self._entry_points:
                lines.append(f"    {node_id}[[\"{label}\"]]")
            elif node_id in self._exit_points:
                lines.append(f"    {node_id}[[\"{label}\"]]")
            else:
                lines.append(f"    {node_id}[\"{label}\"]")

        # Add edges
        for edge in self._edges:
            from_node = edge.from_node
            to_node = edge.to_node

            # Add condition or weight as label
            if edge.condition:
                label = edge.condition
                lines.append(f"    {from_node} -->|{label}| {to_node}")
            elif edge.weight != 1.0:
                label = f"{edge.weight:.2f}"
                lines.append(f"    {from_node} -->|{label}| {to_node}")
            else:
                lines.append(f"    {from_node} --> {to_node}")

        # Add styling
        lines.append("")
        entry_nodes = ",".join(self._entry_points)
        exit_nodes = ",".join(self._exit_points)
        if entry_nodes:
            lines.append(f"    classDef entryClass fill:#90EE90,stroke:#333,stroke-width:2px")
            lines.append(f"    class {entry_nodes} entryClass")
        if exit_nodes:
            lines.append(f"    classDef exitClass fill:#FFB6C1,stroke:#333,stroke-width:2px")
            lines.append(f"    class {exit_nodes} exitClass")

        return "\n".join(lines)

    def to_dot(self) -> str:
        """Export graph as GraphViz DOT format.

        Returns:
            DOT format string.
        """
        lines = ["digraph G {", "    rankdir=TB;", "    node [shape=box];", ""]

        # Add nodes with styling
        for node_id, node in self._nodes.items():
            label = node.name if hasattr(node, "name") and node.name else node_id

            # Style based on node type
            style_attrs = []
            if node_id in self._entry_points:
                style_attrs.append('fillcolor=lightgreen')
                style_attrs.append('style=filled')
                style_attrs.append('shape=doubleoctagon')
            elif node_id in self._exit_points:
                style_attrs.append('fillcolor=lightpink')
                style_attrs.append('style=filled')
                style_attrs.append('shape=doubleoctagon')

            if style_attrs:
                attrs = ", ".join(style_attrs)
                lines.append(f'    {node_id} [label="{label}", {attrs}];')
            else:
                lines.append(f'    {node_id} [label="{label}"];')

        lines.append("")

        # Add edges
        for edge in self._edges:
            from_node = edge.from_node
            to_node = edge.to_node

            # Add condition or weight as label
            edge_attrs = []
            if edge.condition:
                edge_attrs.append(f'label="{edge.condition}"')
            elif edge.weight != 1.0:
                edge_attrs.append(f'label="{edge.weight:.2f}"')

            if edge_attrs:
                attrs = ", ".join(edge_attrs)
                lines.append(f"    {from_node} -> {to_node} [{attrs}];")
            else:
                lines.append(f"    {from_node} -> {to_node};")

        lines.append("}")
        return "\n".join(lines)

    def validate(self) -> List[ValidationError]:
        """Validate graph structure.

        Comprehensive validation checks:
        - All referenced nodes exist
        - Entry/exit points exist and are valid
        - Edges reference existing nodes
        - Cycle detection (warning)
        - Unreachable nodes detection
        - Dead-end nodes (non-exit nodes with no outgoing edges)
        - Orphan nodes (disconnected from the graph)

        Returns:
            List of validation errors and warnings (empty if valid).
        """
        errors: List[ValidationError] = []

        # Check entry points exist
        for entry in self._entry_points:
            if entry not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Entry point '{entry}' not found in nodes",
                        node_id=entry,
                        severity="error",
                    )
                )

        # Check exit points exist
        for exit_point in self._exit_points:
            if exit_point not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Exit point '{exit_point}' not found in nodes",
                        node_id=exit_point,
                        severity="error",
                    )
                )

        # Check edges reference existing nodes
        for edge in self._edges:
            if edge.from_node not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Edge from_node '{edge.from_node}' not found",
                        node_id=edge.from_node,
                        severity="error",
                    )
                )
            if edge.to_node not in self._nodes:
                errors.append(
                    ValidationError(
                        message=f"Edge to_node '{edge.to_node}' not found",
                        node_id=edge.to_node,
                        severity="error",
                    )
                )

        # Detect cycles
        cycles = self.detect_cycles()
        if cycles:
            # If cycles are not allowed, treat as error; otherwise warning
            severity = "error" if not self.allow_cycles else "warning"
            for cycle in cycles:
                cycle_path = " -> ".join(cycle)
                errors.append(
                    ValidationError(
                        message=f"Cycle detected: {cycle_path}",
                        node_id=cycle[0] if cycle else None,
                        severity=severity,
                    )
                )

        # Check for unreachable nodes
        unreachable = self._find_unreachable_nodes()
        for node_id in unreachable:
            if node_id not in self._protected:
                errors.append(
                    ValidationError(
                        message=f"Node '{node_id}' is unreachable from entry points",
                        node_id=node_id,
                        severity="warning",
                    )
                )

        # Check for dead-end nodes (non-exit nodes with no outgoing edges)
        dead_ends = self._find_dead_end_nodes()
        for node_id in dead_ends:
            errors.append(
                ValidationError(
                    message=f"Node '{node_id}' is a dead-end (no outgoing edges, not an exit point)",
                    node_id=node_id,
                    severity="warning",
                )
            )

        # Check for orphan nodes (completely disconnected)
        connected = set()
        for edge in self._edges:
            connected.add(edge.from_node)
            connected.add(edge.to_node)
        connected.update(self._entry_points)
        connected.update(self._exit_points)

        for node_id in self._nodes:
            if node_id not in connected and node_id not in self._protected:
                errors.append(
                    ValidationError(
                        message=f"Node '{node_id}' is orphaned (not connected to the graph)",
                        node_id=node_id,
                        severity="warning",
                    )
                )

        # Check that entry points have outgoing edges
        for entry in self._entry_points:
            if entry in self._nodes and not self.get_outgoing_edges(entry):
                errors.append(
                    ValidationError(
                        message=f"Entry point '{entry}' has no outgoing edges",
                        node_id=entry,
                        severity="error",
                    )
                )

        # Check that exit points are reachable
        for exit_point in self._exit_points:
            if exit_point in unreachable:
                errors.append(
                    ValidationError(
                        message=f"Exit point '{exit_point}' is unreachable from entry points",
                        node_id=exit_point,
                        severity="error",
                    )
                )

        return errors

    def __repr__(self) -> str:
        return f"<Graph id={self.id} nodes={len(self._nodes)} edges={len(self._edges)}>"
