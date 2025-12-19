"""Expansion system data models.

Defines structures for failure patterns, expansion strategies,
and expansion proposals used by the self-improvement system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """Categories of failure patterns."""

    TASK_COMPLEXITY = "task_complexity"  # Task too complex for model
    DOMAIN_MISMATCH = "domain_mismatch"  # Wrong domain expertise
    CONTEXT_OVERFLOW = "context_overflow"  # Context too long
    TOOL_MISSING = "tool_missing"  # Required tool not available
    INSTRUCTION_UNCLEAR = "instruction_unclear"  # Prompt issues
    MODEL_LIMITATION = "model_limitation"  # Model capability limit
    UNKNOWN = "unknown"


class StrategyType(str, Enum):
    """Types of expansion strategies."""

    PROMPT_REFINEMENT = "prompt_refinement"  # Improve prompts
    TOOL_AUGMENTATION = "tool_augmentation"  # Add tools
    MODEL_UPGRADE = "model_upgrade"  # Use bigger model
    SUB_ROUTING = "sub_routing"  # Create sub-graph
    CONTEXT_SPLITTING = "context_splitting"  # Split into chunks
    NONE = "none"  # No action needed


class FailurePattern(BaseModel):
    """A cluster of similar failures identified through analysis."""

    id: str = Field(description="Unique identifier for this pattern")
    category: FailureCategory = Field(description="Primary failure category")
    description: str = Field(description="Human-readable description")
    sample_tasks: List[str] = Field(
        default_factory=list, description="Example tasks that triggered this failure"
    )
    sample_errors: List[str] = Field(
        default_factory=list, description="Example error messages or feedback"
    )
    occurrence_count: int = Field(default=1, ge=1, description="How often this occurs")
    node_id: str = Field(description="ID of the failing node")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in pattern identification"
    )
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def merge(self, other: "FailurePattern") -> "FailurePattern":
        """Merge another pattern into this one."""
        return FailurePattern(
            id=self.id,
            category=self.category,
            description=self.description,
            sample_tasks=self.sample_tasks[:5] + other.sample_tasks[:5],
            sample_errors=self.sample_errors[:5] + other.sample_errors[:5],
            occurrence_count=self.occurrence_count + other.occurrence_count,
            node_id=self.node_id,
            confidence=max(self.confidence, other.confidence),
            first_seen=min(self.first_seen, other.first_seen),
            last_seen=max(self.last_seen, other.last_seen),
            metadata={**self.metadata, **other.metadata},
        )


class ExpansionCost(BaseModel):
    """Cost estimate for an expansion strategy."""

    memory_mb: float = Field(default=0.0, ge=0.0, description="Additional memory needed")
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="Additional latency per request"
    )
    complexity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized complexity increase"
    )
    maintenance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Maintenance burden increase"
    )

    @property
    def total_score(self) -> float:
        """Calculate total cost score (lower is better)."""
        return (
            self.memory_mb / 1000  # Normalize to GB
            + self.latency_ms / 1000  # Normalize to seconds
            + self.complexity
            + self.maintenance
        )


class ExpansionBenefit(BaseModel):
    """Expected benefit from an expansion strategy."""

    expected_improvement: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Expected quality improvement"
    )
    coverage_increase: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Additional task coverage"
    )
    reliability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in improvement"
    )

    @property
    def total_score(self) -> float:
        """Calculate total benefit score (higher is better)."""
        return (
            self.expected_improvement * 0.5
            + self.coverage_increase * 0.3
            + self.reliability * 0.2
        )


class ExpansionStrategy(BaseModel):
    """A proposed strategy for addressing failure patterns."""

    id: str = Field(description="Unique identifier")
    type: StrategyType = Field(description="Type of strategy")
    description: str = Field(description="Human-readable description")
    target_patterns: List[str] = Field(
        default_factory=list, description="Pattern IDs this addresses"
    )
    target_node_id: str = Field(description="Node to expand")
    cost: ExpansionCost = Field(default_factory=ExpansionCost)
    benefit: ExpansionBenefit = Field(default_factory=ExpansionBenefit)
    implementation: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific implementation details"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def score(self) -> float:
        """Calculate strategy score (benefit - cost)."""
        return self.benefit.total_score - (self.cost.total_score * 0.5)

    @classmethod
    def create_prompt_refinement(
        cls,
        node_id: str,
        patterns: List[FailurePattern],
        new_prompt: str,
        expected_improvement: float = 0.2,
    ) -> "ExpansionStrategy":
        """Create a prompt refinement strategy."""
        return cls(
            id=f"prompt_{node_id}_{datetime.utcnow().timestamp():.0f}",
            type=StrategyType.PROMPT_REFINEMENT,
            description=f"Refine system prompt for node {node_id}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(latency_ms=0, complexity=0.1),
            benefit=ExpansionBenefit(
                expected_improvement=expected_improvement, reliability=0.7
            ),
            implementation={"new_prompt": new_prompt},
        )

    @classmethod
    def create_tool_augmentation(
        cls,
        node_id: str,
        patterns: List[FailurePattern],
        tools: List[str],
        expected_improvement: float = 0.3,
    ) -> "ExpansionStrategy":
        """Create a tool augmentation strategy."""
        return cls(
            id=f"tools_{node_id}_{datetime.utcnow().timestamp():.0f}",
            type=StrategyType.TOOL_AUGMENTATION,
            description=f"Add tools {tools} to node {node_id}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(latency_ms=50, complexity=0.2),
            benefit=ExpansionBenefit(
                expected_improvement=expected_improvement,
                coverage_increase=0.2,
                reliability=0.8,
            ),
            implementation={"tools": tools},
        )

    @classmethod
    def create_model_upgrade(
        cls,
        node_id: str,
        patterns: List[FailurePattern],
        new_model: str,
        expected_improvement: float = 0.4,
    ) -> "ExpansionStrategy":
        """Create a model upgrade strategy."""
        return cls(
            id=f"model_{node_id}_{datetime.utcnow().timestamp():.0f}",
            type=StrategyType.MODEL_UPGRADE,
            description=f"Upgrade node {node_id} to model {new_model}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(memory_mb=500, latency_ms=200, complexity=0.1),
            benefit=ExpansionBenefit(
                expected_improvement=expected_improvement, reliability=0.9
            ),
            implementation={"new_model": new_model},
        )

    @classmethod
    def create_sub_routing(
        cls,
        node_id: str,
        patterns: List[FailurePattern],
        sub_domains: List[str],
        expected_improvement: float = 0.5,
    ) -> "ExpansionStrategy":
        """Create a sub-routing strategy."""
        return cls(
            id=f"route_{node_id}_{datetime.utcnow().timestamp():.0f}",
            type=StrategyType.SUB_ROUTING,
            description=f"Create sub-router for {node_id} with specialists: {sub_domains}",
            target_patterns=[p.id for p in patterns],
            target_node_id=node_id,
            cost=ExpansionCost(
                memory_mb=200 * len(sub_domains), latency_ms=100, complexity=0.4
            ),
            benefit=ExpansionBenefit(
                expected_improvement=expected_improvement,
                coverage_increase=0.4,
                reliability=0.85,
            ),
            implementation={"sub_domains": sub_domains},
        )


class NodeCreationSpec(BaseModel):
    """Specification for a new node to create."""

    id: str = Field(description="ID for new node")
    type: str = Field(description="Node type (router, model, tool, gate)")
    name: str = Field(description="Human-readable name")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Node configuration"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for model nodes"
    )
    model: Optional[str] = Field(default=None, description="Model ID for model nodes")
    tools: List[str] = Field(default_factory=list, description="Tools for tool nodes")


class EdgeCreationSpec(BaseModel):
    """Specification for a new edge to create."""

    from_node: str = Field(description="Source node ID")
    to_node: str = Field(description="Target node ID")
    weight: float = Field(default=1.0, ge=0.0, description="Edge weight")
    condition: Optional[str] = Field(
        default=None, description="Routing condition for router edges"
    )


class ExpansionProposal(BaseModel):
    """A concrete proposal for expanding the graph."""

    id: str = Field(description="Unique identifier")
    strategy: ExpansionStrategy = Field(description="Strategy being implemented")
    nodes_to_create: List[NodeCreationSpec] = Field(
        default_factory=list, description="New nodes to add"
    )
    edges_to_create: List[EdgeCreationSpec] = Field(
        default_factory=list, description="New edges to add"
    )
    nodes_to_modify: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Node ID -> modifications"
    )
    edges_to_remove: List[str] = Field(
        default_factory=list, description="Edge IDs to remove"
    )
    nodes_to_protect: List[str] = Field(
        default_factory=list, description="Node IDs to protect from pruning"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = Field(default=False, description="Whether human approved")
    approved_at: Optional[datetime] = Field(default=None)
    approved_by: Optional[str] = Field(default=None)

    @property
    def is_safe(self) -> bool:
        """Check if proposal is safe to apply automatically."""
        # Modifications and removals are riskier
        if self.nodes_to_modify or self.edges_to_remove:
            return False
        # Creating many nodes is also risky
        if len(self.nodes_to_create) > 5:
            return False
        return True

    def approve(self, approver: str = "system") -> None:
        """Mark proposal as approved."""
        self.approved = True
        self.approved_at = datetime.utcnow()
        self.approved_by = approver


class ExpansionResult(BaseModel):
    """Result of applying an expansion proposal."""

    proposal_id: str = Field(description="ID of the applied proposal")
    success: bool = Field(description="Whether expansion succeeded")
    nodes_created: List[str] = Field(
        default_factory=list, description="IDs of created nodes"
    )
    edges_created: List[str] = Field(
        default_factory=list, description="IDs of created edges"
    )
    nodes_modified: List[str] = Field(
        default_factory=list, description="IDs of modified nodes"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    rollback_info: Dict[str, Any] = Field(
        default_factory=dict, description="Info needed to rollback"
    )

    @classmethod
    def success_result(
        cls,
        proposal_id: str,
        nodes_created: List[str],
        edges_created: List[str],
        rollback_info: Optional[Dict[str, Any]] = None,
    ) -> "ExpansionResult":
        """Create a successful result."""
        return cls(
            proposal_id=proposal_id,
            success=True,
            nodes_created=nodes_created,
            edges_created=edges_created,
            rollback_info=rollback_info or {},
        )

    @classmethod
    def failure_result(cls, proposal_id: str, error: str) -> "ExpansionResult":
        """Create a failed result."""
        return cls(
            proposal_id=proposal_id,
            success=False,
            error=error,
        )


class ExpansionConfig(BaseModel):
    """Configuration for the expansion engine."""

    min_evaluations: int = Field(
        default=10, ge=5, description="Minimum evaluations before considering expansion"
    )
    fail_threshold: float = Field(
        default=0.4, ge=0.1, le=0.9, description="Failure rate threshold for expansion"
    )
    max_depth: int = Field(
        default=3, ge=1, le=10, description="Maximum expansion depth"
    )
    auto_approve: bool = Field(
        default=False, description="Auto-approve safe expansions"
    )
    cooldown_seconds: int = Field(
        default=3600, ge=60, description="Cooldown between expansions for same node"
    )
    max_nodes_per_expansion: int = Field(
        default=5, ge=1, le=20, description="Maximum nodes to create per expansion"
    )
    prune_inactive_days: int = Field(
        default=30, ge=7, description="Days of inactivity before pruning"
    )
