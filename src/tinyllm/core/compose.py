"""Fluent workflow composition helpers for TinyLLM.

This module provides a builder pattern for constructing graphs
programmatically with a clean, chainable API.

Example:
    >>> workflow = (
    ...     WorkflowBuilder("my.workflow", "My Workflow")
    ...     .add_entry("entry.main")
    ...     .add_router("router.main", routes=[
    ...         Route("code", "Code tasks", "model.code"),
    ...         Route("general", "General tasks", "model.general"),
    ...     ])
    ...     .add_model("model.code", model="qwen2.5:3b")
    ...     .add_model("model.general", model="qwen2.5:0.5b")
    ...     .add_exit("exit.main")
    ...     .connect("entry.main", "router.main")
    ...     .connect("router.main", ["model.code", "model.general"])
    ...     .connect(["model.code", "model.general"], "exit.main")
    ...     .build()
    ... )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from tinyllm.config.graph import (
    EdgeDefinition,
    GraphDefinition,
    GraphMetadata,
    NodeDefinition,
    NodeType,
)
from tinyllm.core.builder import GraphBuilder
from tinyllm.core.graph import Graph


@dataclass
class Route:
    """Definition of a routing option."""

    name: str
    description: str
    target: str


@dataclass
class TransformOp:
    """Definition of a transform operation."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateCondition:
    """Definition of a gate condition."""

    name: str
    expression: str
    target: str


class WorkflowBuilder:
    """Fluent builder for constructing TinyLLM workflows.

    Provides a chainable API for building graphs without
    manually constructing Pydantic models.
    """

    def __init__(
        self,
        id: str,
        name: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
    ):
        """Initialize workflow builder.

        Args:
            id: Unique workflow identifier (lowercase, dots/underscores only)
            name: Human-readable workflow name
            version: Semantic version (default: 1.0.0)
            description: Optional workflow description
        """
        self._id = id
        self._name = name
        self._version = version
        self._description = description
        self._nodes: List[NodeDefinition] = []
        self._edges: List[EdgeDefinition] = []
        self._entry_points: List[str] = []
        self._exit_points: List[str] = []
        self._protected: List[str] = []
        self._metadata = GraphMetadata()

    def add_entry(
        self,
        node_id: str,
        required_fields: Optional[List[str]] = None,
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add an entry node.

        Args:
            node_id: Unique node identifier
            required_fields: Optional list of required input fields
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = dict(config)
        if required_fields:
            node_config["required_fields"] = required_fields

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.ENTRY,
                config=node_config,
            )
        )
        self._entry_points.append(node_id)
        return self

    def add_exit(
        self,
        node_id: str,
        status: str = "success",
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add an exit node.

        Args:
            node_id: Unique node identifier
            status: Exit status (success, error, fallback)
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {"status": status, **config}
        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.EXIT,
                config=node_config,
            )
        )
        self._exit_points.append(node_id)
        return self

    def add_router(
        self,
        node_id: str,
        routes: List[Route],
        model: str = "qwen2.5:0.5b",
        default_route: Optional[str] = None,
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add a router node.

        Args:
            node_id: Unique node identifier
            routes: List of Route definitions
            model: Model to use for classification
            default_route: Default route if none match
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {
            "model": model,
            "routes": [
                {"name": r.name, "description": r.description, "target": r.target}
                for r in routes
            ],
            **config,
        }
        if default_route:
            node_config["default_route"] = default_route

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.ROUTER,
                config=node_config,
            )
        )
        return self

    def add_model(
        self,
        node_id: str,
        model: str = "qwen2.5:3b",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add a model node.

        Args:
            node_id: Unique node identifier
            model: Model name to use
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {
            "model": model,
            "temperature": temperature,
            **config,
        }
        if system_prompt:
            node_config["system_prompt"] = system_prompt

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.MODEL,
                config=node_config,
            )
        )
        return self

    def add_gate(
        self,
        node_id: str,
        conditions: List[GateCondition],
        default_target: Optional[str] = None,
        mode: str = "expression",
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add a gate node.

        Args:
            node_id: Unique node identifier
            conditions: List of GateCondition definitions
            default_target: Target if no conditions match
            mode: Gate mode (expression, llm)
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {
            "mode": mode,
            "conditions": [
                {"name": c.name, "expression": c.expression, "target": c.target}
                for c in conditions
            ],
            **config,
        }
        if default_target:
            node_config["default_target"] = default_target

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.GATE,
                config=node_config,
            )
        )
        return self

    def add_transform(
        self,
        node_id: str,
        transforms: List[TransformOp],
        stop_on_error: bool = True,
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add a transform node.

        Args:
            node_id: Unique node identifier
            transforms: List of TransformOp definitions
            stop_on_error: Stop pipeline on transform error
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {
            "transforms": [{"type": t.type, "params": t.params} for t in transforms],
            "stop_on_error": stop_on_error,
            **config,
        }

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.TRANSFORM,
                config=node_config,
            )
        )
        return self

    def add_loop(
        self,
        node_id: str,
        body_node: str,
        condition_type: str = "fixed_count",
        fixed_count: int = 3,
        max_iterations: int = 10,
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add a loop node.

        Args:
            node_id: Unique node identifier
            body_node: Node to execute in loop body
            condition_type: Loop condition type (fixed_count, until_success, etc.)
            fixed_count: Number of iterations for fixed_count
            max_iterations: Maximum iterations allowed
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {
            "body_node": body_node,
            "condition_type": condition_type,
            "max_iterations": max_iterations,
            **config,
        }
        if condition_type == "fixed_count":
            node_config["fixed_count"] = fixed_count

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.LOOP,
                config=node_config,
            )
        )
        return self

    def add_fanout(
        self,
        node_id: str,
        target_nodes: List[str],
        aggregation_strategy: str = "all",
        parallel: bool = True,
        **config: Any,
    ) -> "WorkflowBuilder":
        """Add a fanout node.

        Args:
            node_id: Unique node identifier
            target_nodes: List of target node IDs
            aggregation_strategy: Strategy (all, first_success, majority_vote, best_score)
            parallel: Execute targets in parallel
            **config: Additional node configuration

        Returns:
            Self for chaining
        """
        node_config = {
            "target_nodes": target_nodes,
            "aggregation_strategy": aggregation_strategy,
            "parallel": parallel,
            **config,
        }

        self._nodes.append(
            NodeDefinition(
                id=node_id,
                type=NodeType.FANOUT,
                config=node_config,
            )
        )
        return self

    def connect(
        self,
        from_nodes: Union[str, List[str]],
        to_nodes: Union[str, List[str]],
        weight: float = 1.0,
        condition: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Connect nodes with edges.

        Supports many-to-many connections for convenience.

        Args:
            from_nodes: Source node ID or list of IDs
            to_nodes: Target node ID or list of IDs
            weight: Edge weight (0.0-1.0)
            condition: Optional condition expression

        Returns:
            Self for chaining
        """
        if isinstance(from_nodes, str):
            from_nodes = [from_nodes]
        if isinstance(to_nodes, str):
            to_nodes = [to_nodes]

        for from_node in from_nodes:
            for to_node in to_nodes:
                self._edges.append(
                    EdgeDefinition(
                        from_node=from_node,
                        to_node=to_node,
                        weight=weight,
                        condition=condition,
                    )
                )
        return self

    def protect(self, *node_ids: str) -> "WorkflowBuilder":
        """Mark nodes as protected from pruning.

        Args:
            *node_ids: Node IDs to protect

        Returns:
            Self for chaining
        """
        self._protected.extend(node_ids)
        return self

    def with_metadata(
        self,
        author: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **extra: Any,
    ) -> "WorkflowBuilder":
        """Add metadata to the workflow.

        Args:
            author: Workflow author
            description: Workflow description
            tags: List of tags
            **extra: Additional metadata fields

        Returns:
            Self for chaining
        """
        if author:
            self._metadata.author = author
        if description:
            self._metadata.description = description
        if tags:
            self._metadata.tags = tags
        # Handle extra fields
        for key, value in extra.items():
            setattr(self._metadata, key, value)
        return self

    def build_definition(self) -> GraphDefinition:
        """Build the GraphDefinition without creating nodes.

        Returns:
            Constructed GraphDefinition
        """
        return GraphDefinition(
            id=self._id,
            version=self._version,
            name=self._name,
            description=self._description,
            metadata=self._metadata,
            nodes=self._nodes,
            edges=self._edges,
            entry_points=self._entry_points,
            exit_points=self._exit_points,
            protected=self._protected,
        )

    def build(self) -> Graph:
        """Build the complete Graph with instantiated nodes.

        Returns:
            Constructed Graph with all nodes
        """
        definition = self.build_definition()
        builder = GraphBuilder()
        return builder.build_from_definition(definition)


# Convenience functions for common patterns


def simple_qa_workflow(
    workflow_id: str = "simple.qa",
    model: str = "qwen2.5:3b",
) -> Graph:
    """Create a simple question-answering workflow.

    Entry → Model → Exit

    Args:
        workflow_id: Workflow identifier
        model: Model to use for generation

    Returns:
        Constructed Graph
    """
    return (
        WorkflowBuilder(workflow_id, "Simple QA")
        .add_entry("entry.main")
        .add_model("model.qa", model=model, system_prompt="Answer the question clearly.")
        .add_exit("exit.main")
        .connect("entry.main", "model.qa")
        .connect("model.qa", "exit.main")
        .build()
    )


def routed_specialist_workflow(
    workflow_id: str = "routed.specialist",
    routes: Optional[List[Route]] = None,
) -> Graph:
    """Create a workflow with routing to specialist models.

    Entry → Router → [Specialists] → Exit

    Args:
        workflow_id: Workflow identifier
        routes: Custom routes (defaults to code/math/general)

    Returns:
        Constructed Graph
    """
    if routes is None:
        routes = [
            Route("code", "Code and programming tasks", "model.code"),
            Route("math", "Mathematical problems", "model.math"),
            Route("general", "General knowledge", "model.general"),
        ]

    builder = (
        WorkflowBuilder(workflow_id, "Routed Specialists")
        .add_entry("entry.main")
        .add_router("router.main", routes=routes, default_route="general")
    )

    # Add specialist models
    for route in routes:
        builder.add_model(
            route.target,
            model="qwen2.5:3b",
            system_prompt=f"You are an expert in {route.description}.",
        )

    builder.add_exit("exit.main")

    # Connect entry → router
    builder.connect("entry.main", "router.main")

    # Connect router → specialists → exit
    specialist_ids = [r.target for r in routes]
    builder.connect("router.main", specialist_ids)
    builder.connect(specialist_ids, "exit.main")

    return builder.build()


def parallel_consensus_workflow(
    workflow_id: str = "parallel.consensus",
    num_models: int = 3,
) -> Graph:
    """Create a workflow with parallel model execution and majority vote.

    Entry → Fanout → [Models] → Exit

    Args:
        workflow_id: Workflow identifier
        num_models: Number of parallel models

    Returns:
        Constructed Graph
    """
    target_nodes = [f"model.expert_{i}" for i in range(num_models)]

    builder = (
        WorkflowBuilder(workflow_id, "Parallel Consensus")
        .add_entry("entry.main")
        .add_fanout(
            "fanout.parallel",
            target_nodes=target_nodes,
            aggregation_strategy="majority_vote",
        )
    )

    # Add expert models
    for i, target in enumerate(target_nodes):
        builder.add_model(target, model="qwen2.5:3b")

    builder.add_exit("exit.main")
    builder.connect("entry.main", "fanout.parallel")
    builder.connect("fanout.parallel", "exit.main")

    return builder.build()


def transform_pipeline_workflow(
    workflow_id: str = "transform.pipeline",
    transforms: Optional[List[TransformOp]] = None,
) -> Graph:
    """Create a data transformation pipeline.

    Entry → Transform → Exit

    Args:
        workflow_id: Workflow identifier
        transforms: List of transform operations

    Returns:
        Constructed Graph
    """
    if transforms is None:
        transforms = [
            TransformOp("strip"),
            TransformOp("lowercase"),
            TransformOp("truncate", {"max_length": 500}),
        ]

    return (
        WorkflowBuilder(workflow_id, "Transform Pipeline")
        .add_entry("entry.main")
        .add_transform("transform.pipeline", transforms=transforms)
        .add_exit("exit.main")
        .connect("entry.main", "transform.pipeline")
        .connect("transform.pipeline", "exit.main")
        .build()
    )
