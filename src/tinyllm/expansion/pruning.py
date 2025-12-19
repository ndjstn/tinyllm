"""
Adaptive pruning system for TinyLLM.

Automatically identifies and removes underperforming or inactive nodes
to optimize system efficiency.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from tinyllm.core.node import BaseNode, NodeStats

from tinyllm.config.graph import NodeDefinition


# =============================================================================
# ENUMS
# =============================================================================


class PruneReason(str, Enum):
    """Reasons for pruning a node."""

    LOW_PERFORMANCE = "low_performance"
    INACTIVITY = "inactivity"
    HIGH_ERROR_RATE = "high_error_rate"
    REDUNDANCY = "redundancy"
    RESOURCE_CONSTRAINT = "resource_constraint"
    MANUAL = "manual"


class PruneStatus(str, Enum):
    """Status of a prune operation."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"


class NodeHealth(str, Enum):
    """Health status of a node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# =============================================================================
# MODELS
# =============================================================================


class PruneConfig(BaseModel):
    """Configuration for node pruning."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    # Performance thresholds
    min_success_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    max_error_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    max_avg_latency_ms: Annotated[int, Field(ge=100, le=300000)] = 60000

    # Inactivity settings
    inactivity_threshold_hours: Annotated[int, Field(ge=1, le=720)] = 24  # 1 day
    min_executions_before_prune: Annotated[int, Field(ge=1, le=1000)] = 10

    # Approval settings
    require_approval: bool = True
    auto_prune_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1  # Very bad = auto

    # Safety settings
    protect_entry_exit: bool = True
    max_prunes_per_day: Annotated[int, Field(ge=0, le=100)] = 5
    cooldown_hours: Annotated[int, Field(ge=0, le=72)] = 4

    # Archive vs delete
    archive_instead_of_delete: bool = True


class NodeHealthReport(BaseModel):
    """Health assessment for a single node."""

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str
    health: NodeHealth
    success_rate: float = 0.0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_executions: int = 0
    last_execution: datetime | None = None
    hours_since_last_exec: float | None = None
    issues: list[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_prune_candidate(self) -> bool:
        """Check if node is candidate for pruning."""
        return self.health in {NodeHealth.UNHEALTHY, NodeHealth.CRITICAL}


class PruneProposal(BaseModel):
    """Proposal to prune a node."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(default_factory=lambda: f"prune_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}")
    node_id: str
    reason: PruneReason
    health_report: NodeHealthReport
    status: PruneStatus = PruneStatus.PROPOSED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    approved_at: datetime | None = None
    executed_at: datetime | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    archived_definition: dict[str, Any] | None = None

    def approve(self, approver: str = "system") -> PruneProposal:
        """Approve this prune proposal."""
        return PruneProposal(
            id=self.id,
            node_id=self.node_id,
            reason=self.reason,
            health_report=self.health_report,
            status=PruneStatus.APPROVED,
            created_at=self.created_at,
            approved_at=datetime.now(UTC),
            approved_by=approver,
        )

    def reject(self, reason: str) -> PruneProposal:
        """Reject this prune proposal."""
        return PruneProposal(
            id=self.id,
            node_id=self.node_id,
            reason=self.reason,
            health_report=self.health_report,
            status=PruneStatus.REJECTED,
            created_at=self.created_at,
            rejection_reason=reason,
        )


class PruneResult(BaseModel):
    """Result of a prune operation."""

    model_config = ConfigDict(strict=True, extra="forbid")

    proposal_id: str
    success: bool
    node_id: str
    archived: bool = False
    archive_path: str | None = None
    error: str | None = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PruneHistory(BaseModel):
    """History of prune operations."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prunes: list[PruneResult] = Field(default_factory=list)
    rollbacks: list[str] = Field(default_factory=list)

    @property
    def successful_prunes(self) -> int:
        """Count successful prunes."""
        return sum(1 for p in self.prunes if p.success)

    @property
    def failed_prunes(self) -> int:
        """Count failed prunes."""
        return sum(1 for p in self.prunes if not p.success)


# =============================================================================
# HEALTH ANALYZER
# =============================================================================


class NodeHealthAnalyzer:
    """Analyzes node health based on stats."""

    def __init__(self, config: PruneConfig | None = None) -> None:
        self._config = config or PruneConfig()

    def assess_health(
        self,
        node_id: str,
        stats: "NodeStats",
        is_protected: bool = False,
    ) -> NodeHealthReport:
        """
        Assess the health of a node based on its stats.

        Args:
            node_id: Node identifier
            stats: Node execution statistics
            is_protected: Whether node is protected from pruning

        Returns:
            Health assessment report
        """
        issues: list[str] = []
        health = NodeHealth.HEALTHY

        # Calculate metrics
        success_rate = stats.success_rate
        error_rate = stats.failure_rate
        avg_latency = stats.avg_latency_ms
        total_executions = stats.total_executions
        last_exec = stats.last_execution

        # Calculate hours since last execution
        hours_since_last = None
        if last_exec:
            delta = datetime.now(UTC) - last_exec.replace(tzinfo=UTC)
            hours_since_last = delta.total_seconds() / 3600

        # Check for issues
        if total_executions >= self._config.min_executions_before_prune:
            # Check success rate
            if success_rate < self._config.min_success_rate:
                issues.append(f"Low success rate: {success_rate:.1%}")
                health = NodeHealth.UNHEALTHY

            # Check error rate
            if error_rate > self._config.max_error_rate:
                issues.append(f"High error rate: {error_rate:.1%}")
                if health != NodeHealth.UNHEALTHY:
                    health = NodeHealth.DEGRADED

            # Check latency
            if avg_latency > self._config.max_avg_latency_ms:
                issues.append(f"High latency: {avg_latency:.0f}ms")
                if health == NodeHealth.HEALTHY:
                    health = NodeHealth.DEGRADED

        # Check inactivity
        if hours_since_last is not None:
            if hours_since_last > self._config.inactivity_threshold_hours:
                issues.append(f"Inactive for {hours_since_last:.1f} hours")
                if health == NodeHealth.HEALTHY:
                    health = NodeHealth.DEGRADED

        # Critical if multiple severe issues
        severe_issues = sum(1 for i in issues if "Low success" in i or "High error" in i)
        if severe_issues >= 2:
            health = NodeHealth.CRITICAL

        # Not enough data
        if total_executions < self._config.min_executions_before_prune:
            health = NodeHealth.UNKNOWN
            issues.append(f"Insufficient data: {total_executions} executions")

        # Protected nodes
        if is_protected:
            issues.append("Node is protected")

        return NodeHealthReport(
            node_id=node_id,
            health=health,
            success_rate=success_rate,
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            total_executions=total_executions,
            last_execution=last_exec,
            hours_since_last_exec=hours_since_last,
            issues=issues,
        )


# =============================================================================
# NODE PRUNER
# =============================================================================


class NodePruner:
    """Manages adaptive node pruning."""

    def __init__(self, config: PruneConfig | None = None) -> None:
        self._config = config or PruneConfig()
        self._analyzer = NodeHealthAnalyzer(config)
        self._history = PruneHistory()
        self._pending_proposals: dict[str, PruneProposal] = {}
        self._protected_nodes: set[str] = set()
        self._archived_nodes: dict[str, dict[str, Any]] = {}
        self._last_prune: datetime | None = None
        self._prunes_today = 0
        self._prune_day: str | None = None

    @property
    def config(self) -> PruneConfig:
        """Get prune configuration."""
        return self._config

    @property
    def history(self) -> PruneHistory:
        """Get prune history."""
        return self._history

    def protect_node(self, node_id: str) -> None:
        """Mark a node as protected from pruning."""
        self._protected_nodes.add(node_id)

    def unprotect_node(self, node_id: str) -> None:
        """Remove protection from a node."""
        self._protected_nodes.discard(node_id)

    def is_protected(self, node_id: str) -> bool:
        """Check if a node is protected."""
        return node_id in self._protected_nodes

    def analyze_nodes(
        self,
        nodes: dict[str, "NodeStats"],
    ) -> list[NodeHealthReport]:
        """
        Analyze health of multiple nodes.

        Args:
            nodes: Dictionary mapping node IDs to their stats

        Returns:
            List of health reports, sorted by health (worst first)
        """
        reports = []
        for node_id, stats in nodes.items():
            is_protected = self.is_protected(node_id)
            report = self._analyzer.assess_health(node_id, stats, is_protected)
            reports.append(report)

        # Sort by health (CRITICAL first, HEALTHY last)
        health_order = {
            NodeHealth.CRITICAL: 0,
            NodeHealth.UNHEALTHY: 1,
            NodeHealth.DEGRADED: 2,
            NodeHealth.UNKNOWN: 3,
            NodeHealth.HEALTHY: 4,
        }
        reports.sort(key=lambda r: health_order.get(r.health, 5))
        return reports

    def find_prune_candidates(
        self,
        nodes: dict[str, "NodeStats"],
    ) -> list[NodeHealthReport]:
        """
        Find nodes that are candidates for pruning.

        Args:
            nodes: Dictionary mapping node IDs to their stats

        Returns:
            List of health reports for prune candidates
        """
        reports = self.analyze_nodes(nodes)
        return [r for r in reports if r.is_prune_candidate and not self.is_protected(r.node_id)]

    def create_proposal(
        self,
        node_id: str,
        stats: "NodeStats",
        reason: PruneReason | None = None,
    ) -> PruneProposal:
        """
        Create a prune proposal for a node.

        Args:
            node_id: Node to prune
            stats: Node statistics
            reason: Reason for pruning (auto-detected if not provided)

        Returns:
            Prune proposal
        """
        if self.is_protected(node_id):
            raise ValueError(f"Cannot prune protected node: {node_id}")

        health_report = self._analyzer.assess_health(node_id, stats)

        # Auto-detect reason from health report
        if reason is None:
            if health_report.success_rate < self._config.min_success_rate:
                reason = PruneReason.LOW_PERFORMANCE
            elif health_report.error_rate > self._config.max_error_rate:
                reason = PruneReason.HIGH_ERROR_RATE
            elif (
                health_report.hours_since_last_exec
                and health_report.hours_since_last_exec > self._config.inactivity_threshold_hours
            ):
                reason = PruneReason.INACTIVITY
            else:
                reason = PruneReason.MANUAL

        proposal = PruneProposal(
            node_id=node_id,
            reason=reason,
            health_report=health_report,
        )

        self._pending_proposals[proposal.id] = proposal
        return proposal

    def approve_proposal(self, proposal_id: str, approver: str = "system") -> PruneProposal:
        """Approve a pending prune proposal."""
        if proposal_id not in self._pending_proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self._pending_proposals[proposal_id].approve(approver)
        self._pending_proposals[proposal_id] = proposal
        return proposal

    def reject_proposal(self, proposal_id: str, reason: str) -> PruneProposal:
        """Reject a pending prune proposal."""
        if proposal_id not in self._pending_proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self._pending_proposals[proposal_id].reject(reason)
        self._pending_proposals[proposal_id] = proposal
        return proposal

    def execute_prune(
        self,
        proposal_id: str,
        node_definition: NodeDefinition,
        on_prune: Callable[[str], bool] | None = None,
    ) -> PruneResult:
        """
        Execute an approved prune.

        Args:
            proposal_id: ID of approved proposal
            node_definition: Definition of node to prune (for archiving)
            on_prune: Optional callback to actually remove the node

        Returns:
            Prune result
        """
        if proposal_id not in self._pending_proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self._pending_proposals[proposal_id]
        if proposal.status != PruneStatus.APPROVED:
            raise ValueError(f"Proposal not approved: {proposal.status}")

        # Check rate limits
        if not self._can_prune():
            return PruneResult(
                proposal_id=proposal_id,
                success=False,
                node_id=proposal.node_id,
                error="Rate limit exceeded or cooldown active",
            )

        try:
            # Archive if configured
            archived = False
            archive_path = None
            if self._config.archive_instead_of_delete:
                self._archived_nodes[proposal.node_id] = node_definition.model_dump()
                archived = True
                archive_path = f"archive/{proposal.node_id}"

            # Execute actual prune via callback
            if on_prune:
                success = on_prune(proposal.node_id)
                if not success:
                    return PruneResult(
                        proposal_id=proposal_id,
                        success=False,
                        node_id=proposal.node_id,
                        error="Prune callback returned False",
                    )

            result = PruneResult(
                proposal_id=proposal_id,
                success=True,
                node_id=proposal.node_id,
                archived=archived,
                archive_path=archive_path,
            )

            self._history.prunes.append(result)
            self._last_prune = datetime.now(UTC)
            self._update_prune_count()

            # Update proposal status
            self._pending_proposals[proposal_id] = PruneProposal(
                **{**proposal.model_dump(), "status": PruneStatus.EXECUTED, "executed_at": datetime.now(UTC)}
            )

            return result

        except Exception as e:
            result = PruneResult(
                proposal_id=proposal_id,
                success=False,
                node_id=proposal.node_id,
                error=str(e),
            )
            self._history.prunes.append(result)
            return result

    def restore_node(self, node_id: str) -> NodeDefinition | None:
        """
        Restore an archived node.

        Args:
            node_id: ID of node to restore

        Returns:
            Restored node definition or None if not found
        """
        if node_id not in self._archived_nodes:
            return None

        archived_data = self._archived_nodes.pop(node_id)
        self._history.rollbacks.append(node_id)
        return NodeDefinition(**archived_data)

    def get_pending_proposals(self) -> list[PruneProposal]:
        """Get all pending proposals."""
        return [p for p in self._pending_proposals.values() if p.status == PruneStatus.PROPOSED]

    def get_archived_nodes(self) -> list[str]:
        """Get list of archived node IDs."""
        return list(self._archived_nodes.keys())

    def should_auto_prune(self, health_report: NodeHealthReport) -> bool:
        """
        Check if a node should be auto-pruned without approval.

        Args:
            health_report: Node health report

        Returns:
            True if auto-prune should occur
        """
        if self._config.require_approval:
            # Only auto-prune if success rate is below auto threshold
            if health_report.success_rate <= self._config.auto_prune_threshold:
                return True
            return False
        return health_report.is_prune_candidate

    def _can_prune(self) -> bool:
        """Check if a prune can be performed (rate limiting)."""
        now = datetime.now(UTC)

        # Check cooldown
        if self._last_prune:
            cooldown = timedelta(hours=self._config.cooldown_hours)
            if now - self._last_prune < cooldown:
                return False

        # Check daily limit
        today = now.strftime("%Y-%m-%d")
        if self._prune_day != today:
            self._prune_day = today
            self._prunes_today = 0

        return self._prunes_today < self._config.max_prunes_per_day

    def _update_prune_count(self) -> None:
        """Update prune count for rate limiting."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._prune_day != today:
            self._prune_day = today
            self._prunes_today = 0
        self._prunes_today += 1

    def get_stats(self) -> dict[str, Any]:
        """Get pruner statistics."""
        return {
            "config": self._config.model_dump(),
            "pending_proposals": len(self.get_pending_proposals()),
            "protected_nodes": len(self._protected_nodes),
            "archived_nodes": len(self._archived_nodes),
            "total_prunes": len(self._history.prunes),
            "successful_prunes": self._history.successful_prunes,
            "failed_prunes": self._history.failed_prunes,
            "rollbacks": len(self._history.rollbacks),
            "can_prune": self._can_prune(),
        }


__all__ = [
    "PruneReason",
    "PruneStatus",
    "NodeHealth",
    "PruneConfig",
    "NodeHealthReport",
    "PruneProposal",
    "PruneResult",
    "PruneHistory",
    "NodeHealthAnalyzer",
    "NodePruner",
]
