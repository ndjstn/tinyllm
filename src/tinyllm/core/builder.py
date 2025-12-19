"""Graph builder for TinyLLM.

This module provides the GraphBuilder class that constructs Graph
instances from YAML definitions.
"""

from pathlib import Path
from typing import Optional

import yaml

from tinyllm.config.graph import GraphDefinition, NodeDefinition
from tinyllm.config.loader import Config
from tinyllm.core.graph import Graph
from tinyllm.core.node import BaseNode
from tinyllm.core.registry import NodeRegistry

# Import nodes to trigger registration via decorators
import tinyllm.nodes  # noqa: F401


class GraphBuilder:
    """Builds Graph instances from definitions.

    The GraphBuilder is responsible for loading graph definitions
    from YAML files and constructing runtime Graph objects with
    instantiated nodes.
    """

    def __init__(
        self,
        registry: Optional[NodeRegistry] = None,
        config: Optional[Config] = None,
    ):
        """Initialize graph builder.

        Args:
            registry: Node registry for creating node instances.
                     Uses the global registry if not provided.
            config: System configuration for nodes.
        """
        self._registry = registry or NodeRegistry
        self._config = config

    def build_from_yaml(self, path: Path | str) -> Graph:
        """Load and build a graph from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Constructed Graph instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the YAML is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty graph file: {path}")

        definition = GraphDefinition(**data)
        return self.build_from_definition(definition)

    def build_from_definition(self, definition: GraphDefinition) -> Graph:
        """Build a graph from a Pydantic definition.

        Args:
            definition: Graph definition model.

        Returns:
            Constructed Graph instance.

        Raises:
            ValueError: If a node type is not registered.
        """
        # Create the graph structure
        graph = Graph(definition)

        # Create and add nodes
        for node_def in definition.nodes:
            node = self._create_node(node_def)
            graph.add_node(node)

        # Validate the constructed graph
        errors = graph.validate()
        if errors:
            error_messages = [e.message for e in errors if e.severity == "error"]
            if error_messages:
                raise ValueError(
                    f"Graph validation failed: {'; '.join(error_messages)}"
                )

        return graph

    def _create_node(self, node_def: NodeDefinition) -> BaseNode:
        """Create a node instance from definition.

        Args:
            node_def: Node definition.

        Returns:
            Node instance.

        Raises:
            ValueError: If the node type is not registered.
        """
        return self._registry.create(node_def)


def load_graph(path: Path | str, config: Optional[Config] = None) -> Graph:
    """Convenience function to load a graph from a YAML file.

    Args:
        path: Path to the graph YAML file.
        config: Optional system configuration.

    Returns:
        Constructed Graph instance.
    """
    builder = GraphBuilder(config=config)
    return builder.build_from_yaml(path)
