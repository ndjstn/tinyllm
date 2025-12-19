"""Expansion strategy generation.

Generates and evaluates strategies for addressing failure patterns.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.expansion.models import (
    EdgeCreationSpec,
    ExpansionBenefit,
    ExpansionCost,
    ExpansionProposal,
    ExpansionStrategy,
    FailureCategory,
    FailurePattern,
    NodeCreationSpec,
    StrategyType,
)


class StrategyGeneratorConfig(BaseModel):
    """Configuration for strategy generation."""

    enable_prompt_refinement: bool = Field(default=True)
    enable_tool_augmentation: bool = Field(default=True)
    enable_model_upgrade: bool = Field(default=True)
    enable_sub_routing: bool = Field(default=True)

    # Model upgrade options
    upgrade_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "qwen2.5:0.5b": "qwen2.5:1.5b",
            "qwen2.5:1.5b": "qwen2.5:3b",
            "qwen2.5:3b": "qwen2.5:7b",
            "qwen3:0.6b": "qwen3:1.7b",
            "qwen3:1.7b": "qwen3:4b",
            "qwen3:4b": "qwen3:8b",
        }
    )

    # Tool recommendations by category
    tool_recommendations: Dict[FailureCategory, List[str]] = Field(
        default_factory=lambda: {
            FailureCategory.TOOL_MISSING: ["calculator", "search", "code_executor"],
            FailureCategory.CONTEXT_OVERFLOW: ["summarizer", "chunker"],
            FailureCategory.TASK_COMPLEXITY: ["planner", "decomposer"],
        }
    )

    # Min improvement threshold
    min_expected_improvement: float = Field(default=0.1, ge=0.0, le=1.0)


class StrategyGenerator:
    """Generates expansion strategies based on failure patterns.

    Analyzes patterns and proposes appropriate strategies:
    - Prompt refinement for instruction issues
    - Tool augmentation for missing capabilities
    - Model upgrade for complexity issues
    - Sub-routing for domain mismatch or diverse failures
    """

    def __init__(self, config: Optional[StrategyGeneratorConfig] = None):
        """Initialize the strategy generator.

        Args:
            config: Configuration options.
        """
        self.config = config or StrategyGeneratorConfig()

    def generate_strategies(
        self,
        node_id: str,
        patterns: List[FailurePattern],
        current_model: Optional[str] = None,
        current_tools: Optional[List[str]] = None,
        sub_domains: Optional[List[str]] = None,
    ) -> List[ExpansionStrategy]:
        """Generate candidate strategies for a failing node.

        Args:
            node_id: The failing node ID.
            patterns: Identified failure patterns.
            current_model: Current model (for upgrade strategy).
            current_tools: Current tools (for augmentation strategy).
            sub_domains: Identified sub-domains (for sub-routing).

        Returns:
            List of candidate strategies, sorted by score.
        """
        strategies: List[ExpansionStrategy] = []
        current_tools = current_tools or []
        sub_domains = sub_domains or []

        # Analyze pattern categories
        categories = {p.category for p in patterns}

        # Generate prompt refinement strategy
        if self.config.enable_prompt_refinement and self._should_refine_prompt(
            categories
        ):
            strategy = self._create_prompt_strategy(node_id, patterns)
            if strategy:
                strategies.append(strategy)

        # Generate tool augmentation strategy
        if self.config.enable_tool_augmentation and self._should_add_tools(
            categories, current_tools
        ):
            strategy = self._create_tool_strategy(node_id, patterns, current_tools)
            if strategy:
                strategies.append(strategy)

        # Generate model upgrade strategy
        if self.config.enable_model_upgrade and current_model:
            strategy = self._create_upgrade_strategy(
                node_id, patterns, current_model
            )
            if strategy:
                strategies.append(strategy)

        # Generate sub-routing strategy
        if self.config.enable_sub_routing and len(sub_domains) >= 2:
            strategy = self._create_routing_strategy(node_id, patterns, sub_domains)
            if strategy:
                strategies.append(strategy)

        # Sort by score (descending)
        strategies.sort(key=lambda s: s.score, reverse=True)

        return strategies

    def select_best_strategy(
        self, strategies: List[ExpansionStrategy]
    ) -> Optional[ExpansionStrategy]:
        """Select the best strategy from candidates.

        Args:
            strategies: Candidate strategies.

        Returns:
            Best strategy or None if none meet threshold.
        """
        if not strategies:
            return None

        # Already sorted by score
        best = strategies[0]

        # Check if improvement meets threshold
        if best.benefit.expected_improvement < self.config.min_expected_improvement:
            return None

        return best

    def create_proposal(
        self,
        strategy: ExpansionStrategy,
        node_config: Optional[Dict[str, Any]] = None,
    ) -> ExpansionProposal:
        """Create a concrete expansion proposal from a strategy.

        Args:
            strategy: The selected strategy.
            node_config: Current node configuration.

        Returns:
            Expansion proposal ready for approval/application.
        """
        from datetime import datetime

        proposal_id = f"proposal_{datetime.utcnow().timestamp():.0f}"
        node_config = node_config or {}

        if strategy.type == StrategyType.PROMPT_REFINEMENT:
            return self._create_prompt_proposal(proposal_id, strategy, node_config)
        elif strategy.type == StrategyType.TOOL_AUGMENTATION:
            return self._create_tool_proposal(proposal_id, strategy, node_config)
        elif strategy.type == StrategyType.MODEL_UPGRADE:
            return self._create_upgrade_proposal(proposal_id, strategy, node_config)
        elif strategy.type == StrategyType.SUB_ROUTING:
            return self._create_routing_proposal(proposal_id, strategy, node_config)
        else:
            # No-op proposal
            return ExpansionProposal(id=proposal_id, strategy=strategy)

    def _should_refine_prompt(self, categories: set) -> bool:
        """Check if prompt refinement might help."""
        return FailureCategory.INSTRUCTION_UNCLEAR in categories or len(categories) == 1

    def _should_add_tools(
        self, categories: set, current_tools: List[str]
    ) -> bool:
        """Check if tool augmentation might help."""
        if FailureCategory.TOOL_MISSING in categories:
            return True
        if FailureCategory.TASK_COMPLEXITY in categories and not current_tools:
            return True
        return False

    def _create_prompt_strategy(
        self, node_id: str, patterns: List[FailurePattern]
    ) -> Optional[ExpansionStrategy]:
        """Create a prompt refinement strategy."""
        # Extract common issues from patterns
        issues = []
        for pattern in patterns:
            issues.extend(pattern.sample_errors[:2])

        if not issues:
            return None

        # Generate prompt suggestions
        suggestions = [
            "Be more specific about the expected output format",
            "Include examples of good responses",
            "Clarify edge cases and error handling",
        ]

        return ExpansionStrategy(
            id=f"prompt_{node_id}",
            type=StrategyType.PROMPT_REFINEMENT,
            description=f"Refine system prompt to address: {issues[0][:50]}...",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(latency_ms=0, complexity=0.1, maintenance=0.1),
            benefit=ExpansionBenefit(
                expected_improvement=0.2,
                coverage_increase=0.0,
                reliability=0.6,
            ),
            implementation={
                "suggestions": suggestions,
                "identified_issues": issues[:3],
            },
        )

    def _create_tool_strategy(
        self,
        node_id: str,
        patterns: List[FailurePattern],
        current_tools: List[str],
    ) -> Optional[ExpansionStrategy]:
        """Create a tool augmentation strategy."""
        recommended_tools = set()

        for pattern in patterns:
            tools = self.config.tool_recommendations.get(pattern.category, [])
            for tool in tools:
                if tool not in current_tools:
                    recommended_tools.add(tool)

        if not recommended_tools:
            return None

        tools_list = list(recommended_tools)[:3]  # Max 3 tools

        return ExpansionStrategy(
            id=f"tools_{node_id}",
            type=StrategyType.TOOL_AUGMENTATION,
            description=f"Add tools: {', '.join(tools_list)}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(
                latency_ms=30 * len(tools_list),
                complexity=0.15 * len(tools_list),
                maintenance=0.1 * len(tools_list),
            ),
            benefit=ExpansionBenefit(
                expected_improvement=0.25,
                coverage_increase=0.15,
                reliability=0.75,
            ),
            implementation={"tools": tools_list},
        )

    def _create_upgrade_strategy(
        self,
        node_id: str,
        patterns: List[FailurePattern],
        current_model: str,
    ) -> Optional[ExpansionStrategy]:
        """Create a model upgrade strategy."""
        # Find upgrade path
        upgrade_model = self.config.upgrade_models.get(current_model)
        if not upgrade_model:
            return None

        return ExpansionStrategy(
            id=f"model_{node_id}",
            type=StrategyType.MODEL_UPGRADE,
            description=f"Upgrade from {current_model} to {upgrade_model}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(
                memory_mb=500,
                latency_ms=150,
                complexity=0.05,
                maintenance=0.05,
            ),
            benefit=ExpansionBenefit(
                expected_improvement=0.35,
                coverage_increase=0.1,
                reliability=0.85,
            ),
            implementation={
                "current_model": current_model,
                "new_model": upgrade_model,
            },
        )

    def _create_routing_strategy(
        self,
        node_id: str,
        patterns: List[FailurePattern],
        sub_domains: List[str],
    ) -> Optional[ExpansionStrategy]:
        """Create a sub-routing strategy."""
        if len(sub_domains) < 2:
            return None

        return ExpansionStrategy(
            id=f"route_{node_id}",
            type=StrategyType.SUB_ROUTING,
            description=f"Create sub-router with specialists: {', '.join(sub_domains)}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(
                memory_mb=150 * len(sub_domains),
                latency_ms=80,
                complexity=0.3,
                maintenance=0.2,
            ),
            benefit=ExpansionBenefit(
                expected_improvement=0.45,
                coverage_increase=0.35,
                reliability=0.8,
            ),
            implementation={"sub_domains": sub_domains},
        )

    def _create_prompt_proposal(
        self,
        proposal_id: str,
        strategy: ExpansionStrategy,
        node_config: Dict[str, Any],
    ) -> ExpansionProposal:
        """Create proposal for prompt refinement."""
        suggestions = strategy.implementation.get("suggestions", [])

        return ExpansionProposal(
            id=proposal_id,
            strategy=strategy,
            nodes_to_modify={
                strategy.target_node_id: {
                    "prompt_suggestions": suggestions,
                    "needs_prompt_review": True,
                }
            },
        )

    def _create_tool_proposal(
        self,
        proposal_id: str,
        strategy: ExpansionStrategy,
        node_config: Dict[str, Any],
    ) -> ExpansionProposal:
        """Create proposal for tool augmentation."""
        tools = strategy.implementation.get("tools", [])

        return ExpansionProposal(
            id=proposal_id,
            strategy=strategy,
            nodes_to_modify={
                strategy.target_node_id: {
                    "tools_to_add": tools,
                }
            },
        )

    def _create_upgrade_proposal(
        self,
        proposal_id: str,
        strategy: ExpansionStrategy,
        node_config: Dict[str, Any],
    ) -> ExpansionProposal:
        """Create proposal for model upgrade."""
        new_model = strategy.implementation.get("new_model")

        return ExpansionProposal(
            id=proposal_id,
            strategy=strategy,
            nodes_to_modify={
                strategy.target_node_id: {
                    "model": new_model,
                }
            },
        )

    def _create_routing_proposal(
        self,
        proposal_id: str,
        strategy: ExpansionStrategy,
        node_config: Dict[str, Any],
    ) -> ExpansionProposal:
        """Create proposal for sub-routing."""
        sub_domains = strategy.implementation.get("sub_domains", [])
        target_id = strategy.target_node_id

        # Create router node
        router_id = f"{target_id}_router"
        router_node = NodeCreationSpec(
            id=router_id,
            type="router",
            name=f"{target_id} Sub-Router",
            config={
                "domains": sub_domains,
                "parent_node": target_id,
            },
        )

        # Create specialist nodes
        specialist_nodes = []
        for domain in sub_domains:
            specialist_id = f"{target_id}_{domain}"
            specialist_nodes.append(
                NodeCreationSpec(
                    id=specialist_id,
                    type="model",
                    name=f"{domain} specialist",
                    config={
                        "domain": domain,
                        "parent_router": router_id,
                    },
                    model=node_config.get("model", "qwen2.5:1.5b"),
                    system_prompt=self._generate_specialist_prompt(domain),
                )
            )

        # Create edges
        edges = []
        # Edge from router to each specialist
        for node in specialist_nodes:
            edges.append(
                EdgeCreationSpec(
                    from_node=router_id,
                    to_node=node.id,
                    condition=node.config.get("domain"),
                )
            )

        return ExpansionProposal(
            id=proposal_id,
            strategy=strategy,
            nodes_to_create=[router_node] + specialist_nodes,
            edges_to_create=edges,
            nodes_to_modify={
                target_id: {
                    "replaced_by": router_id,
                    "status": "expanded",
                }
            },
            nodes_to_protect=[router_id] + [n.id for n in specialist_nodes],
        )

    def _generate_specialist_prompt(self, domain: str) -> str:
        """Generate a system prompt for a specialist node.

        Args:
            domain: The domain specialization.

        Returns:
            System prompt string.
        """
        prompts = {
            "arithmetic": "You are a specialist in arithmetic operations. Focus on accurate calculations.",
            "algebra": "You are an algebra specialist. Solve equations step by step.",
            "calculus": "You are a calculus expert. Handle derivatives, integrals, and limits.",
            "geometry": "You are a geometry specialist. Work with shapes, angles, and spatial reasoning.",
            "statistics": "You are a statistics expert. Handle probability and statistical analysis.",
            "code_python": "You are a Python expert. Write clean, efficient Python code.",
            "code_js": "You are a JavaScript specialist. Write modern, clean JavaScript.",
            "code_sql": "You are a SQL expert. Write efficient database queries.",
            "writing": "You are a writing specialist. Create clear, engaging content.",
            "reasoning": "You are a logical reasoning expert. Think step by step.",
        }

        return prompts.get(
            domain,
            f"You are a specialist in {domain}. Focus on this area of expertise.",
        )
