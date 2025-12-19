"""Expansion engine.

Main orchestrator for the self-improvement system.
Monitors node performance, analyzes failures, and proposes expansions.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.expansion.analyzer import PatternAnalyzer, PatternAnalyzerConfig
from tinyllm.expansion.models import (
    ExpansionConfig,
    ExpansionProposal,
    ExpansionResult,
    ExpansionStrategy,
    FailurePattern,
)
from tinyllm.expansion.strategies import StrategyGenerator, StrategyGeneratorConfig
from tinyllm.grading.metrics import MetricsTracker, NodeMetrics
from tinyllm.grading.models import GradingResult


class ExpansionEngine:
    """Main engine for self-improvement through graph expansion.

    Monitors node performance, analyzes failure patterns,
    generates expansion strategies, and applies approved expansions.

    Flow:
    1. Record grading results
    2. Identify failing nodes
    3. Analyze failure patterns
    4. Generate expansion strategies
    5. Create expansion proposals
    6. Apply approved proposals
    """

    def __init__(
        self,
        config: Optional[ExpansionConfig] = None,
        analyzer_config: Optional[PatternAnalyzerConfig] = None,
        generator_config: Optional[StrategyGeneratorConfig] = None,
    ):
        """Initialize the expansion engine.

        Args:
            config: Expansion configuration.
            analyzer_config: Pattern analyzer configuration.
            generator_config: Strategy generator configuration.
        """
        self.config = config or ExpansionConfig()
        self.metrics = MetricsTracker()
        self.analyzer = PatternAnalyzer(analyzer_config)
        self.generator = StrategyGenerator(generator_config)

        # Track expansion history
        self._expansion_history: Dict[str, List[ExpansionResult]] = {}
        self._pending_proposals: Dict[str, ExpansionProposal] = {}
        self._last_expansion: Dict[str, datetime] = {}
        self._protected_nodes: set = set()

        # Callbacks for graph modification
        self._apply_callback: Optional[Callable[[ExpansionProposal], ExpansionResult]] = None

    def record_result(self, result: GradingResult) -> None:
        """Record a grading result for monitoring.

        Args:
            result: The grading result to record.
        """
        self.metrics.record_result(result)

        # Also record failures for pattern analysis
        if not result.grade.is_passing:
            self.analyzer.record_failure(result)

    def check_for_expansion(
        self,
        node_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ExpansionProposal]:
        """Check if any nodes need expansion and create proposals.

        Args:
            node_configs: Current node configurations (node_id -> config).

        Returns:
            List of expansion proposals (not yet applied).
        """
        node_configs = node_configs or {}
        proposals = []

        # Get nodes that might need expansion
        candidates = self.metrics.get_expansion_candidates(
            min_evaluations=self.config.min_evaluations,
            fail_threshold=self.config.fail_threshold,
        )

        for node_metrics in candidates:
            node_id = node_metrics.node_id

            # Skip protected nodes
            if node_id in self._protected_nodes:
                continue

            # Check cooldown
            if self._is_in_cooldown(node_id):
                continue

            # Analyze failure patterns
            patterns = self.analyzer.analyze_node(node_id)
            if not patterns:
                continue

            # Get node configuration
            node_config = node_configs.get(node_id, {})

            # Generate proposal
            proposal = self._create_proposal_for_node(
                node_id, patterns, node_metrics, node_config
            )

            if proposal:
                # Auto-approve if configured and safe
                if self.config.auto_approve and proposal.is_safe:
                    proposal.approve("auto")

                self._pending_proposals[proposal.id] = proposal
                proposals.append(proposal)

        return proposals

    def get_pending_proposals(self) -> List[ExpansionProposal]:
        """Get all pending expansion proposals.

        Returns:
            List of pending proposals.
        """
        return list(self._pending_proposals.values())

    def approve_proposal(
        self, proposal_id: str, approver: str = "user"
    ) -> Optional[ExpansionProposal]:
        """Approve a pending proposal.

        Args:
            proposal_id: ID of the proposal to approve.
            approver: Who approved it.

        Returns:
            The approved proposal or None if not found.
        """
        proposal = self._pending_proposals.get(proposal_id)
        if proposal:
            proposal.approve(approver)
        return proposal

    def reject_proposal(self, proposal_id: str) -> bool:
        """Reject and remove a pending proposal.

        Args:
            proposal_id: ID of the proposal to reject.

        Returns:
            True if proposal was found and removed.
        """
        if proposal_id in self._pending_proposals:
            del self._pending_proposals[proposal_id]
            return True
        return False

    def apply_approved_proposals(self) -> List[ExpansionResult]:
        """Apply all approved proposals.

        Returns:
            List of expansion results.
        """
        results = []

        approved = [
            p for p in self._pending_proposals.values() if p.approved
        ]

        for proposal in approved:
            result = self._apply_proposal(proposal)
            results.append(result)

            # Track history
            node_id = proposal.strategy.target_node_id
            if node_id not in self._expansion_history:
                self._expansion_history[node_id] = []
            self._expansion_history[node_id].append(result)

            # Update cooldown
            if result.success:
                self._last_expansion[node_id] = datetime.utcnow()

                # Add protected nodes
                for protected_id in proposal.nodes_to_protect:
                    self._protected_nodes.add(protected_id)

            # Remove from pending
            del self._pending_proposals[proposal.id]

        return results

    def set_apply_callback(
        self,
        callback: Callable[[ExpansionProposal], ExpansionResult],
    ) -> None:
        """Set callback for applying proposals to the actual graph.

        Args:
            callback: Function that takes a proposal and returns a result.
        """
        self._apply_callback = callback

    def protect_node(self, node_id: str) -> None:
        """Protect a node from expansion.

        Args:
            node_id: Node to protect.
        """
        self._protected_nodes.add(node_id)

    def unprotect_node(self, node_id: str) -> None:
        """Remove protection from a node.

        Args:
            node_id: Node to unprotect.
        """
        self._protected_nodes.discard(node_id)

    def get_expansion_history(
        self, node_id: Optional[str] = None
    ) -> Dict[str, List[ExpansionResult]]:
        """Get expansion history.

        Args:
            node_id: Optional node ID to filter by.

        Returns:
            Expansion history by node.
        """
        if node_id:
            return {node_id: self._expansion_history.get(node_id, [])}
        return self._expansion_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary.
        """
        total_expansions = sum(
            len(results) for results in self._expansion_history.values()
        )
        successful_expansions = sum(
            sum(1 for r in results if r.success)
            for results in self._expansion_history.values()
        )

        return {
            "nodes_tracked": len(self.metrics._node_metrics),
            "protected_nodes": len(self._protected_nodes),
            "pending_proposals": len(self._pending_proposals),
            "total_expansions": total_expansions,
            "successful_expansions": successful_expansions,
            "analyzer_stats": self.analyzer.get_stats(),
            "metrics_summary": self.metrics.get_system_summary(),
        }

    def reset(self) -> None:
        """Reset the engine state."""
        self.metrics.reset()
        self.analyzer.clear_all()
        self._expansion_history.clear()
        self._pending_proposals.clear()
        self._last_expansion.clear()
        self._protected_nodes.clear()

    def _is_in_cooldown(self, node_id: str) -> bool:
        """Check if a node is in expansion cooldown.

        Args:
            node_id: The node to check.

        Returns:
            True if in cooldown.
        """
        last = self._last_expansion.get(node_id)
        if not last:
            return False

        cooldown = timedelta(seconds=self.config.cooldown_seconds)
        return datetime.utcnow() - last < cooldown

    def _create_proposal_for_node(
        self,
        node_id: str,
        patterns: List[FailurePattern],
        node_metrics: NodeMetrics,
        node_config: Dict[str, Any],
    ) -> Optional[ExpansionProposal]:
        """Create an expansion proposal for a failing node.

        Args:
            node_id: The node ID.
            patterns: Failure patterns.
            node_metrics: Node metrics.
            node_config: Current node configuration.

        Returns:
            Expansion proposal or None.
        """
        # Identify potential sub-domains
        sub_domains = self.analyzer.identify_sub_domains(node_id, patterns)

        # Generate strategies
        strategies = self.generator.generate_strategies(
            node_id=node_id,
            patterns=patterns,
            current_model=node_config.get("model"),
            current_tools=node_config.get("tools", []),
            sub_domains=sub_domains,
        )

        if not strategies:
            return None

        # Select best strategy
        best = self.generator.select_best_strategy(strategies)
        if not best:
            return None

        # Create proposal
        return self.generator.create_proposal(best, node_config)

    def _apply_proposal(self, proposal: ExpansionProposal) -> ExpansionResult:
        """Apply an expansion proposal.

        Args:
            proposal: The proposal to apply.

        Returns:
            Expansion result.
        """
        if self._apply_callback:
            try:
                return self._apply_callback(proposal)
            except Exception as e:
                return ExpansionResult.failure_result(
                    proposal.id, f"Callback error: {str(e)}"
                )

        # Default: mark as successful but no actual changes
        # (In real usage, _apply_callback should be set)
        return ExpansionResult.success_result(
            proposal_id=proposal.id,
            nodes_created=[n.id for n in proposal.nodes_to_create],
            edges_created=[],
            rollback_info={
                "nodes_created": [n.id for n in proposal.nodes_to_create],
                "edges_created": [],
                "nodes_modified": list(proposal.nodes_to_modify.keys()),
            },
        )


class ExpansionTrigger(BaseModel):
    """Configuration for automatic expansion triggers."""

    enabled: bool = Field(default=True, description="Whether trigger is enabled")
    check_interval_seconds: int = Field(
        default=300, ge=60, description="How often to check for expansion"
    )
    min_evaluations: int = Field(
        default=20, ge=5, description="Min evaluations before triggering"
    )
    fail_threshold: float = Field(
        default=0.4, ge=0.1, le=0.9, description="Failure rate threshold"
    )
    auto_apply: bool = Field(
        default=False, description="Auto-apply safe expansions"
    )


class AutoExpansionMonitor:
    """Background monitor for automatic expansion.

    Can be used with asyncio to periodically check
    for expansion opportunities.
    """

    def __init__(
        self,
        engine: ExpansionEngine,
        trigger: Optional[ExpansionTrigger] = None,
    ):
        """Initialize the monitor.

        Args:
            engine: The expansion engine.
            trigger: Trigger configuration.
        """
        self.engine = engine
        self.trigger = trigger or ExpansionTrigger()
        self._running = False
        self._last_check = datetime.min

    async def start(
        self,
        node_config_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
    ) -> None:
        """Start the monitoring loop.

        Args:
            node_config_provider: Callback to get current node configs.
        """
        import asyncio

        self._running = True
        while self._running:
            try:
                await self._check_cycle(node_config_provider)
            except Exception:
                pass  # Log error in real implementation

            await asyncio.sleep(self.trigger.check_interval_seconds)

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False

    async def _check_cycle(
        self,
        node_config_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]],
    ) -> None:
        """Run one check cycle.

        Args:
            node_config_provider: Callback to get configs.
        """
        if not self.trigger.enabled:
            return

        # Get node configs
        configs = {}
        if node_config_provider:
            configs = node_config_provider()

        # Override engine config for this check
        original_min = self.engine.config.min_evaluations
        original_thresh = self.engine.config.fail_threshold

        self.engine.config.min_evaluations = self.trigger.min_evaluations
        self.engine.config.fail_threshold = self.trigger.fail_threshold

        try:
            # Check for expansions
            proposals = self.engine.check_for_expansion(configs)

            if self.trigger.auto_apply:
                # Apply safe proposals automatically
                results = self.engine.apply_approved_proposals()
                # In real implementation, log/notify about results

        finally:
            # Restore original config
            self.engine.config.min_evaluations = original_min
            self.engine.config.fail_threshold = original_thresh

        self._last_check = datetime.utcnow()
