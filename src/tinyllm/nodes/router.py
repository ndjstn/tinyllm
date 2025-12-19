"""Router node implementation.

Routes messages to different paths based on LLM classification.
Supports multi-dimensional routing for cross-domain queries.
"""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.models.client import OllamaClient
from tinyllm.prompts.loader import PromptLoader

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class RouteDefinition(BaseModel):
    """Definition of a routing option."""

    name: str = Field(description="Route identifier")
    description: str = Field(description="When to use this route")
    target: str = Field(description="Target node ID")
    priority: int = Field(default=0, description="Route priority (higher = preferred)")
    domains: List[str] = Field(
        default_factory=list,
        description="Domain tags for multi-dimensional matching",
    )


class CompoundRoute(BaseModel):
    """Route for multi-domain combinations."""

    domains: Set[str] = Field(description="Set of domains that trigger this route")
    target: str = Field(description="Target node for this domain combination")
    priority: int = Field(default=0, description="Priority when multiple compound routes match")


class RouterNodeConfig(NodeConfig):
    """Configuration for router nodes."""

    model: str = Field(
        default="qwen2.5:0.5b", description="Model to use for routing"
    )
    prompt_id: Optional[str] = Field(
        default=None, description="Prompt ID for routing logic"
    )
    routes: List[RouteDefinition] = Field(
        default_factory=list, description="Available routes"
    )
    default_route: Optional[str] = Field(
        default=None, description="Default route if classification fails"
    )
    confidence_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence for routing"
    )
    # Multi-dimensional routing settings
    multi_label: bool = Field(
        default=False,
        description="Enable multi-label classification (query can match multiple domains)",
    )
    compound_routes: List[CompoundRoute] = Field(
        default_factory=list,
        description="Routes for specific domain combinations",
    )
    fanout_enabled: bool = Field(
        default=False,
        description="Fan out to multiple nodes when multi-label matches",
    )
    max_labels: int = Field(
        default=3, ge=1, le=5,
        description="Maximum number of labels in multi-label mode",
    )


@NodeRegistry.register(NodeType.ROUTER)
class RouterNode(BaseNode):
    """Routes messages based on LLM classification.

    The RouterNode uses a small, fast LLM to classify incoming messages
    and route them to appropriate downstream nodes.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize router node."""
        super().__init__(definition)
        self._router_config = RouterNodeConfig(**definition.config)
        self._client: Optional[OllamaClient] = None
        self._prompt_loader = PromptLoader()

    @property
    def router_config(self) -> RouterNodeConfig:
        """Get router-specific configuration."""
        return self._router_config

    def _get_client(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute routing logic.

        Classifies the message and determines the next node(s).
        Supports both single-label and multi-label classification.

        Args:
            message: Input message to route.
            context: Execution context.

        Returns:
            NodeResult with routing decision.
        """
        try:
            if self._router_config.multi_label:
                return await self._execute_multi_label(message, context)
            else:
                return await self._execute_single_label(message, context)
        except Exception as e:
            return NodeResult.failure_result(error=f"Routing failed: {str(e)}")

    async def _execute_single_label(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Standard single-label routing."""
        prompt = self._build_routing_prompt(message)
        client = self._get_client()

        response = await client.generate(
            model=self._router_config.model,
            prompt=prompt,
            system=self._get_system_prompt(),
        )

        route_name = self._parse_route(response.response)
        route = self._find_route(route_name)

        if route is None:
            if self._router_config.default_route:
                route = self._find_route(self._router_config.default_route)
            if route is None:
                return NodeResult.failure_result(
                    error=f"No matching route found for: {route_name}"
                )

        output_message = message.create_child(
            source_node=self.id,
            payload=message.payload.model_copy(
                update={"metadata": {**message.payload.metadata, "route": route.name}}
            ),
        )

        return NodeResult.success_result(
            output_messages=[output_message],
            next_nodes=[route.target],
            metadata={
                "route": route.name,
                "route_target": route.target,
                "raw_response": response.response,
            },
        )

    async def _execute_multi_label(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Multi-dimensional routing for cross-domain queries.

        Returns multiple labels when a query spans domains.
        Can either fanout to multiple nodes or use compound routes.
        """
        prompt = self._build_multi_label_prompt(message)
        client = self._get_client()

        response = await client.generate(
            model=self._router_config.model,
            prompt=prompt,
            system=self._get_multi_label_system_prompt(),
        )

        # Parse multiple labels from response
        labels = self._parse_multi_labels(response.response)

        if not labels:
            if self._router_config.default_route:
                route = self._find_route(self._router_config.default_route)
                if route:
                    output_message = message.create_child(
                        source_node=self.id,
                        payload=message.payload.model_copy(
                            update={"metadata": {**message.payload.metadata, "routes": ["default"]}}
                        ),
                    )
                    return NodeResult.success_result(
                        output_messages=[output_message],
                        next_nodes=[route.target],
                        metadata={"routes": ["default"], "raw_response": response.response},
                    )
            return NodeResult.failure_result(error="No labels matched from multi-label classification")

        # Check for compound route match first
        label_set = set(labels)
        compound_match = self._find_compound_route(label_set)

        if compound_match:
            output_message = message.create_child(
                source_node=self.id,
                payload=message.payload.model_copy(
                    update={"metadata": {
                        **message.payload.metadata,
                        "routes": labels,
                        "compound_route": True,
                    }}
                ),
            )
            return NodeResult.success_result(
                output_messages=[output_message],
                next_nodes=[compound_match.target],
                metadata={
                    "routes": labels,
                    "compound_domains": list(compound_match.domains),
                    "compound_target": compound_match.target,
                    "raw_response": response.response,
                },
            )

        # Either fanout or pick highest priority
        if self._router_config.fanout_enabled:
            return self._create_fanout_result(message, labels, response.response)
        else:
            return self._create_priority_result(message, labels, response.response)

    def _build_routing_prompt(self, message: Message) -> str:
        """Build the routing prompt from message content."""
        task = message.payload.task or message.payload.content or ""
        return f"Classify this request:\n{task}"

    def _build_multi_label_prompt(self, message: Message) -> str:
        """Build prompt for multi-label classification."""
        task = message.payload.task or message.payload.content or ""
        return f"""Analyze this request and identify ALL applicable domains (may be multiple):

{task}"""

    def _get_system_prompt(self) -> str:
        """Get the system prompt for single-label routing."""
        routes_text = "\n".join(
            f"- {r.name}: {r.description}" for r in self._router_config.routes
        )
        return f"""You are a task classifier. Analyze the request and determine the SINGLE best category.

Available categories:
{routes_text}

Respond with ONLY the category name, nothing else."""

    def _get_multi_label_system_prompt(self) -> str:
        """Get the system prompt for multi-label routing."""
        routes_text = "\n".join(
            f"- {r.name}: {r.description}" for r in self._router_config.routes
        )
        max_labels = self._router_config.max_labels
        return f"""You are a multi-domain task classifier. Analyze requests that may span multiple domains.

Available domains:
{routes_text}

IMPORTANT: Many requests span multiple domains. For example:
- "Write Python code to calculate compound interest" → code, math
- "Explain the algorithm behind neural networks" → code, general
- "Create a SQL query to compute statistics" → code, math

Respond with ALL applicable domains (up to {max_labels}), separated by commas.
Format: domain1, domain2, domain3
Only list domains that genuinely apply. If only one domain applies, list just that one."""

    def _parse_route(self, response: str) -> str:
        """Parse route name from LLM response."""
        route = response.strip().lower().split("\n")[0]
        route = "".join(c for c in route if c.isalnum() or c == "_")
        return route

    def _parse_multi_labels(self, response: str) -> List[str]:
        """Parse multiple labels from LLM response."""
        # Handle various response formats
        text = response.strip().lower()

        # Try JSON array format first
        if text.startswith("["):
            try:
                labels = json.loads(text)
                if isinstance(labels, list):
                    return [str(l).strip() for l in labels[:self._router_config.max_labels]]
            except json.JSONDecodeError:
                pass

        # Try comma-separated format
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            labels = []
            for part in parts[:self._router_config.max_labels]:
                # Clean each label
                clean = "".join(c for c in part if c.isalnum() or c == "_")
                if clean and self._find_route(clean):
                    labels.append(clean)
            if labels:
                return labels

        # Try newline-separated
        if "\n" in text:
            labels = []
            for line in text.split("\n")[:self._router_config.max_labels]:
                clean = "".join(c for c in line.strip() if c.isalnum() or c == "_")
                if clean and self._find_route(clean):
                    labels.append(clean)
            if labels:
                return labels

        # Fallback: single label
        single = self._parse_route(response)
        if self._find_route(single):
            return [single]

        return []

    def _find_route(self, route_name: str) -> Optional[RouteDefinition]:
        """Find route definition by name."""
        for route in self._router_config.routes:
            if route.name.lower() == route_name.lower():
                return route
        return None

    def _find_compound_route(self, labels: Set[str]) -> Optional[CompoundRoute]:
        """Find a compound route that matches the given label set.

        Returns the highest priority compound route where all its domains
        are present in the labels.
        """
        matching = []
        for compound in self._router_config.compound_routes:
            # Check if all compound domains are in the labels
            if compound.domains.issubset(labels):
                matching.append(compound)

        if not matching:
            return None

        # Return highest priority match, preferring more specific (more domains)
        return max(matching, key=lambda c: (len(c.domains), c.priority))

    def _create_fanout_result(
        self, message: Message, labels: List[str], raw_response: str
    ) -> NodeResult:
        """Create a fanout result that routes to multiple nodes."""
        targets = []
        output_messages = []

        for label in labels:
            route = self._find_route(label)
            if route and route.target not in targets:
                targets.append(route.target)
                # Create a message copy for each branch
                output_message = message.create_child(
                    source_node=self.id,
                    payload=message.payload.model_copy(
                        update={"metadata": {
                            **message.payload.metadata,
                            "routes": labels,
                            "fanout_branch": label,
                        }}
                    ),
                )
                output_messages.append(output_message)

        if not targets:
            return NodeResult.failure_result(error="No valid targets for fanout")

        return NodeResult.success_result(
            output_messages=output_messages,
            next_nodes=targets,
            metadata={
                "routes": labels,
                "fanout_targets": targets,
                "raw_response": raw_response,
            },
        )

    def _create_priority_result(
        self, message: Message, labels: List[str], raw_response: str
    ) -> NodeResult:
        """Create result routing to the highest priority matched route."""
        # Find highest priority route among matched labels
        best_route = None
        best_priority = -1

        for label in labels:
            route = self._find_route(label)
            if route and route.priority > best_priority:
                best_route = route
                best_priority = route.priority

        # If no priority set, use first match
        if best_route is None:
            for label in labels:
                route = self._find_route(label)
                if route:
                    best_route = route
                    break

        if best_route is None:
            return NodeResult.failure_result(error="No valid route found from labels")

        output_message = message.create_child(
            source_node=self.id,
            payload=message.payload.model_copy(
                update={"metadata": {
                    **message.payload.metadata,
                    "routes": labels,
                    "selected_route": best_route.name,
                }}
            ),
        )

        return NodeResult.success_result(
            output_messages=[output_message],
            next_nodes=[best_route.target],
            metadata={
                "routes": labels,
                "selected_route": best_route.name,
                "route_target": best_route.target,
                "raw_response": raw_response,
            },
        )
