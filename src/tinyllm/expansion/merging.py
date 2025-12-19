"""
Node merging system for TinyLLM.

Detects similar nodes and merges them to reduce redundancy and improve efficiency.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from tinyllm.core.node import BaseNode

from tinyllm.config.graph import NodeDefinition, NodeType


# =============================================================================
# ENUMS
# =============================================================================


class SimilarityMetric(str, Enum):
    """Metrics for measuring node similarity."""

    CONFIG_OVERLAP = "config_overlap"
    PROMPT_SIMILARITY = "prompt_similarity"
    PERFORMANCE_CORRELATION = "performance_correlation"
    ROUTING_OVERLAP = "routing_overlap"
    COMBINED = "combined"


class MergeStrategy(str, Enum):
    """Strategies for merging nodes."""

    KEEP_BETTER = "keep_better"  # Keep the better performing node
    COMBINE_PROMPTS = "combine_prompts"  # Combine system prompts
    WEIGHTED_AVERAGE = "weighted_average"  # Average configurations by performance
    A_B_TEST = "a_b_test"  # Keep both, route randomly and measure
    MANUAL = "manual"  # Require manual approval


class MergeStatus(str, Enum):
    """Status of a merge operation."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# =============================================================================
# MODELS
# =============================================================================


class SimilarityScore(BaseModel):
    """Score representing similarity between two nodes."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    metric: SimilarityMetric
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_similar(self) -> bool:
        """Check if score indicates similarity."""
        return self.score >= 0.7


class NodeSimilarityResult(BaseModel):
    """Result of comparing two nodes for similarity."""

    model_config = ConfigDict(strict=True, extra="forbid")

    node_a_id: str
    node_b_id: str
    scores: list[SimilarityScore]
    overall_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    merge_recommended: bool = False
    recommended_strategy: MergeStrategy | None = None
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_merge_candidate(self) -> bool:
        """Check if nodes are candidates for merging."""
        return self.overall_similarity >= 0.8


class MergeConfig(BaseModel):
    """Configuration for node merging."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    similarity_threshold: Annotated[float, Field(ge=0.5, le=1.0)] = 0.8
    auto_merge_threshold: Annotated[float, Field(ge=0.8, le=1.0)] = 0.95
    require_approval: bool = True
    preserve_history: bool = True
    rollback_on_regression: bool = True
    regression_threshold: Annotated[float, Field(ge=0.0, le=0.5)] = 0.1
    cooldown_ms: Annotated[int, Field(ge=0, le=86400000)] = 3600000  # 1 hour
    max_merges_per_day: Annotated[int, Field(ge=0, le=100)] = 10


class MergeProposal(BaseModel):
    """Proposal to merge two nodes."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"merge_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}")
    node_a_id: str
    node_b_id: str
    similarity: NodeSimilarityResult
    strategy: MergeStrategy
    merged_config: dict[str, Any] = Field(default_factory=dict)
    status: MergeStatus = MergeStatus.PROPOSED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    approved_at: datetime | None = None
    completed_at: datetime | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None

    def approve(self, approver: str = "system") -> MergeProposal:
        """Approve this merge proposal."""
        return MergeProposal(
            id=self.id,
            node_a_id=self.node_a_id,
            node_b_id=self.node_b_id,
            similarity=self.similarity,
            strategy=self.strategy,
            merged_config=self.merged_config,
            status=MergeStatus.APPROVED,
            created_at=self.created_at,
            approved_at=datetime.now(UTC),
            approved_by=approver,
        )

    def reject(self, reason: str) -> MergeProposal:
        """Reject this merge proposal."""
        return MergeProposal(
            id=self.id,
            node_a_id=self.node_a_id,
            node_b_id=self.node_b_id,
            similarity=self.similarity,
            strategy=self.strategy,
            merged_config=self.merged_config,
            status=MergeStatus.REJECTED,
            created_at=self.created_at,
            rejection_reason=reason,
        )


class MergeResult(BaseModel):
    """Result of a merge operation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    proposal_id: str
    success: bool
    merged_node_id: str | None = None
    removed_node_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    performance_before: dict[str, float] = Field(default_factory=dict)
    performance_after: dict[str, float] | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MergeHistory(BaseModel):
    """History of merge operations."""

    model_config = ConfigDict(strict=True, extra="forbid")

    merges: list[MergeResult] = Field(default_factory=list)
    rollbacks: list[str] = Field(default_factory=list)

    @property
    def successful_merges(self) -> int:
        """Count successful merges."""
        return sum(1 for m in self.merges if m.success)

    @property
    def failed_merges(self) -> int:
        """Count failed merges."""
        return sum(1 for m in self.merges if not m.success)


# =============================================================================
# SIMILARITY DETECTOR
# =============================================================================


class NodeSimilarityDetector:
    """Detects similarity between nodes."""

    def __init__(self, config: MergeConfig | None = None) -> None:
        self._config = config or MergeConfig()

    def compare_nodes(
        self,
        node_a: NodeDefinition,
        node_b: NodeDefinition,
    ) -> NodeSimilarityResult:
        """
        Compare two nodes for similarity.

        Args:
            node_a: First node definition
            node_b: Second node definition

        Returns:
            Similarity result with scores
        """
        scores: list[SimilarityScore] = []

        # Compare node types first
        if node_a.type != node_b.type:
            return NodeSimilarityResult(
                node_a_id=node_a.id,
                node_b_id=node_b.id,
                scores=[],
                overall_similarity=0.0,
                merge_recommended=False,
            )

        # Config overlap
        config_score = self._compute_config_similarity(node_a.config, node_b.config)
        scores.append(config_score)

        # Prompt similarity (if applicable)
        if "system_prompt" in node_a.config or "prompt_id" in node_a.config:
            prompt_score = self._compute_prompt_similarity(node_a.config, node_b.config)
            scores.append(prompt_score)

        # Combined score
        overall = sum(s.score for s in scores) / len(scores) if scores else 0.0

        # Determine recommended strategy
        strategy = None
        if overall >= self._config.similarity_threshold:
            strategy = self._recommend_strategy(node_a, node_b, scores)

        return NodeSimilarityResult(
            node_a_id=node_a.id,
            node_b_id=node_b.id,
            scores=scores,
            overall_similarity=overall,
            merge_recommended=overall >= self._config.similarity_threshold,
            recommended_strategy=strategy,
        )

    def _compute_config_similarity(
        self,
        config_a: dict[str, Any],
        config_b: dict[str, Any],
    ) -> SimilarityScore:
        """Compute configuration overlap."""
        # Get all keys
        all_keys = set(config_a.keys()) | set(config_b.keys())
        if not all_keys:
            return SimilarityScore(
                metric=SimilarityMetric.CONFIG_OVERLAP,
                score=1.0,
                details={"reason": "Both configs empty"},
            )

        # Count matching values
        matching = 0
        for key in all_keys:
            if key in config_a and key in config_b:
                if config_a[key] == config_b[key]:
                    matching += 1

        score = matching / len(all_keys)
        return SimilarityScore(
            metric=SimilarityMetric.CONFIG_OVERLAP,
            score=score,
            details={
                "total_keys": len(all_keys),
                "matching_keys": matching,
            },
        )

    def _compute_prompt_similarity(
        self,
        config_a: dict[str, Any],
        config_b: dict[str, Any],
    ) -> SimilarityScore:
        """Compute prompt similarity using simple text comparison."""
        prompt_a = config_a.get("system_prompt", "")
        prompt_b = config_b.get("system_prompt", "")

        if not prompt_a and not prompt_b:
            return SimilarityScore(
                metric=SimilarityMetric.PROMPT_SIMILARITY,
                score=1.0,
                details={"reason": "Both prompts empty"},
            )

        if not prompt_a or not prompt_b:
            return SimilarityScore(
                metric=SimilarityMetric.PROMPT_SIMILARITY,
                score=0.0,
                details={"reason": "One prompt empty"},
            )

        # Simple word-based Jaccard similarity
        words_a = set(re.findall(r"\w+", prompt_a.lower()))
        words_b = set(re.findall(r"\w+", prompt_b.lower()))

        if not words_a and not words_b:
            return SimilarityScore(
                metric=SimilarityMetric.PROMPT_SIMILARITY,
                score=1.0,
            )

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        score = intersection / union if union > 0 else 0.0

        return SimilarityScore(
            metric=SimilarityMetric.PROMPT_SIMILARITY,
            score=score,
            details={
                "words_a": len(words_a),
                "words_b": len(words_b),
                "intersection": intersection,
            },
        )

    def _recommend_strategy(
        self,
        node_a: NodeDefinition,
        node_b: NodeDefinition,
        scores: list[SimilarityScore],
    ) -> MergeStrategy:
        """Recommend a merge strategy based on similarity scores."""
        config_score = next(
            (s for s in scores if s.metric == SimilarityMetric.CONFIG_OVERLAP), None
        )
        prompt_score = next(
            (s for s in scores if s.metric == SimilarityMetric.PROMPT_SIMILARITY), None
        )

        # If configs are nearly identical, just keep the better one
        if config_score and config_score.score >= 0.95:
            return MergeStrategy.KEEP_BETTER

        # If prompts are different but configs similar, combine prompts
        if prompt_score and 0.5 <= prompt_score.score < 0.9:
            return MergeStrategy.COMBINE_PROMPTS

        # Default to weighted average
        return MergeStrategy.WEIGHTED_AVERAGE


# =============================================================================
# NODE MERGER
# =============================================================================


class NodeMerger:
    """Merges similar nodes based on strategies."""

    def __init__(self, config: MergeConfig | None = None) -> None:
        self._config = config or MergeConfig()
        self._detector = NodeSimilarityDetector(config)
        self._history = MergeHistory()
        self._pending_proposals: dict[str, MergeProposal] = {}
        self._last_merge: datetime | None = None
        self._merges_today = 0
        self._merge_day: str | None = None

    @property
    def config(self) -> MergeConfig:
        """Get merge configuration."""
        return self._config

    @property
    def history(self) -> MergeHistory:
        """Get merge history."""
        return self._history

    def find_merge_candidates(
        self,
        nodes: list[NodeDefinition],
    ) -> list[NodeSimilarityResult]:
        """
        Find all pairs of nodes that are candidates for merging.

        Args:
            nodes: List of node definitions to compare

        Returns:
            List of similarity results for merge candidates
        """
        candidates: list[NodeSimilarityResult] = []

        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                result = self._detector.compare_nodes(node_a, node_b)
                if result.is_merge_candidate:
                    candidates.append(result)

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x.overall_similarity, reverse=True)
        return candidates

    def create_proposal(
        self,
        node_a: NodeDefinition,
        node_b: NodeDefinition,
        strategy: MergeStrategy | None = None,
    ) -> MergeProposal:
        """
        Create a merge proposal for two nodes.

        Args:
            node_a: First node
            node_b: Second node
            strategy: Optional merge strategy (auto-detected if not provided)

        Returns:
            Merge proposal
        """
        similarity = self._detector.compare_nodes(node_a, node_b)
        strategy = strategy or similarity.recommended_strategy or MergeStrategy.KEEP_BETTER

        # Create merged config based on strategy
        merged_config = self._create_merged_config(node_a, node_b, strategy)

        proposal = MergeProposal(
            node_a_id=node_a.id,
            node_b_id=node_b.id,
            similarity=similarity,
            strategy=strategy,
            merged_config=merged_config,
        )

        self._pending_proposals[proposal.id] = proposal
        return proposal

    def approve_proposal(self, proposal_id: str, approver: str = "system") -> MergeProposal:
        """Approve a pending merge proposal."""
        if proposal_id not in self._pending_proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self._pending_proposals[proposal_id].approve(approver)
        self._pending_proposals[proposal_id] = proposal
        return proposal

    def reject_proposal(self, proposal_id: str, reason: str) -> MergeProposal:
        """Reject a pending merge proposal."""
        if proposal_id not in self._pending_proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self._pending_proposals[proposal_id].reject(reason)
        self._pending_proposals[proposal_id] = proposal
        return proposal

    def execute_merge(
        self,
        proposal_id: str,
        node_a: NodeDefinition,
        node_b: NodeDefinition,
    ) -> tuple[MergeResult, NodeDefinition]:
        """
        Execute an approved merge.

        Args:
            proposal_id: ID of approved proposal
            node_a: First node definition
            node_b: Second node definition

        Returns:
            Tuple of (result, merged_node_definition)
        """
        if proposal_id not in self._pending_proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self._pending_proposals[proposal_id]
        if proposal.status != MergeStatus.APPROVED:
            raise ValueError(f"Proposal not approved: {proposal.status}")

        # Check rate limits
        if not self._can_merge():
            return (
                MergeResult(
                    proposal_id=proposal_id,
                    success=False,
                    error="Rate limit exceeded or cooldown active",
                ),
                node_a,
            )

        try:
            # Create merged node
            merged_node = self._apply_merge(node_a, node_b, proposal)

            result = MergeResult(
                proposal_id=proposal_id,
                success=True,
                merged_node_id=merged_node.id,
                removed_node_ids=[node_b.id],  # Keep node_a's ID, remove node_b
            )

            self._history.merges.append(result)
            self._last_merge = datetime.now(UTC)
            self._update_merge_count()

            # Update proposal status
            proposal = MergeProposal(
                **{**proposal.model_dump(), "status": MergeStatus.COMPLETED, "completed_at": datetime.now(UTC)}
            )
            self._pending_proposals[proposal_id] = proposal

            return result, merged_node

        except Exception as e:
            result = MergeResult(
                proposal_id=proposal_id,
                success=False,
                error=str(e),
            )
            self._history.merges.append(result)
            return result, node_a

    def get_pending_proposals(self) -> list[MergeProposal]:
        """Get all pending proposals."""
        return [p for p in self._pending_proposals.values() if p.status == MergeStatus.PROPOSED]

    def _can_merge(self) -> bool:
        """Check if a merge can be performed (rate limiting)."""
        now = datetime.now(UTC)

        # Check cooldown
        if self._last_merge:
            elapsed_ms = (now - self._last_merge).total_seconds() * 1000
            if elapsed_ms < self._config.cooldown_ms:
                return False

        # Check daily limit
        today = now.strftime("%Y-%m-%d")
        if self._merge_day != today:
            self._merge_day = today
            self._merges_today = 0

        return self._merges_today < self._config.max_merges_per_day

    def _update_merge_count(self) -> None:
        """Update merge count for rate limiting."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._merge_day != today:
            self._merge_day = today
            self._merges_today = 0
        self._merges_today += 1

    def _create_merged_config(
        self,
        node_a: NodeDefinition,
        node_b: NodeDefinition,
        strategy: MergeStrategy,
    ) -> dict[str, Any]:
        """Create merged configuration based on strategy."""
        if strategy == MergeStrategy.KEEP_BETTER:
            # Just return the first node's config (caller should pick better)
            return dict(node_a.config)

        elif strategy == MergeStrategy.COMBINE_PROMPTS:
            merged = dict(node_a.config)
            prompt_a = node_a.config.get("system_prompt", "")
            prompt_b = node_b.config.get("system_prompt", "")
            if prompt_a and prompt_b and prompt_a != prompt_b:
                merged["system_prompt"] = f"{prompt_a}\n\nAdditionally:\n{prompt_b}"
            return merged

        elif strategy == MergeStrategy.WEIGHTED_AVERAGE:
            merged = {}
            all_keys = set(node_a.config.keys()) | set(node_b.config.keys())
            for key in all_keys:
                val_a = node_a.config.get(key)
                val_b = node_b.config.get(key)
                if val_a == val_b:
                    merged[key] = val_a
                elif val_a is not None and val_b is None:
                    merged[key] = val_a
                elif val_a is None and val_b is not None:
                    merged[key] = val_b
                elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    merged[key] = (val_a + val_b) / 2
                else:
                    merged[key] = val_a  # Default to first
            return merged

        else:
            return dict(node_a.config)

    def _apply_merge(
        self,
        node_a: NodeDefinition,
        node_b: NodeDefinition,
        proposal: MergeProposal,
    ) -> NodeDefinition:
        """Apply merge and create new node definition."""
        # Generate merged ID
        merged_id = node_a.id  # Keep original ID

        # Create merged description
        desc_a = node_a.description or ""
        desc_b = node_b.description or ""
        merged_desc = desc_a
        if desc_b and desc_b != desc_a:
            merged_desc = f"{desc_a} (merged with {node_b.id})"

        return NodeDefinition(
            id=merged_id,
            type=node_a.type,
            name=node_a.name or node_b.name,
            description=merged_desc,
            config=proposal.merged_config,
        )


__all__ = [
    "SimilarityMetric",
    "MergeStrategy",
    "MergeStatus",
    "SimilarityScore",
    "NodeSimilarityResult",
    "MergeConfig",
    "MergeProposal",
    "MergeResult",
    "MergeHistory",
    "NodeSimilarityDetector",
    "NodeMerger",
]
