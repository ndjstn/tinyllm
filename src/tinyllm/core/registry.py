"""Node registry for TinyLLM.

This module provides a registry for node types that allows dynamic
registration and creation of nodes.
"""

from typing import Callable, Dict, Optional, Type

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.node import BaseNode


class NodeRegistry:
    """Registry for node types and factories.

    The NodeRegistry allows node types to be registered and then
    instantiated from NodeDefinitions. This enables dynamic graph
    construction from YAML files.
    """

    _node_types: Dict[NodeType, Type[BaseNode]] = {}
    _factories: Dict[NodeType, Callable[[NodeDefinition], BaseNode]] = {}

    @classmethod
    def register(cls, node_type: NodeType) -> Callable[[Type[BaseNode]], Type[BaseNode]]:
        """Decorator to register a node type.

        Usage:
            @NodeRegistry.register(NodeType.MODEL)
            class ModelNode(BaseNode):
                ...

        Args:
            node_type: The node type to register.

        Returns:
            Decorator function.
        """

        def decorator(node_class: Type[BaseNode]) -> Type[BaseNode]:
            cls._node_types[node_type] = node_class
            return node_class

        return decorator

    @classmethod
    def register_factory(
        cls, node_type: NodeType
    ) -> Callable[[Callable[[NodeDefinition], BaseNode]], Callable[[NodeDefinition], BaseNode]]:
        """Decorator to register a factory function for a node type.

        Use this when node creation requires additional logic beyond
        simple instantiation.

        Args:
            node_type: The node type to register.

        Returns:
            Decorator function.
        """

        def decorator(
            factory: Callable[[NodeDefinition], BaseNode]
        ) -> Callable[[NodeDefinition], BaseNode]:
            cls._factories[node_type] = factory
            return factory

        return decorator

    @classmethod
    def create(cls, definition: NodeDefinition) -> BaseNode:
        """Create a node instance from definition.

        Args:
            definition: Node definition from graph config.

        Returns:
            Node instance.

        Raises:
            ValueError: If node type is not registered.
        """
        # Check for factory first
        if definition.type in cls._factories:
            return cls._factories[definition.type](definition)

        # Fall back to direct class instantiation
        node_class = cls._node_types.get(definition.type)
        if not node_class:
            raise ValueError(
                f"Unknown node type: {definition.type}. "
                f"Registered types: {list(cls._node_types.keys())}"
            )
        return node_class(definition)

    @classmethod
    def get(cls, node_type: NodeType) -> Optional[Type[BaseNode]]:
        """Get the class registered for a node type.

        Args:
            node_type: Node type to look up.

        Returns:
            Node class or None if not registered.
        """
        return cls._node_types.get(node_type)

    @classmethod
    def is_registered(cls, node_type: NodeType) -> bool:
        """Check if a node type is registered.

        Args:
            node_type: Node type to check.

        Returns:
            True if registered.
        """
        return node_type in cls._node_types or node_type in cls._factories

    @classmethod
    def list_types(cls) -> list[NodeType]:
        """List all registered node types.

        Returns:
            List of registered NodeTypes.
        """
        registered = set(cls._node_types.keys()) | set(cls._factories.keys())
        return sorted(registered, key=lambda x: x.value)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._node_types.clear()
        cls._factories.clear()
