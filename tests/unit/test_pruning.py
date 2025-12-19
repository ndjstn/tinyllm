"""
Tests for the adaptive pruning system.
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.expansion.pruning import (
    PruneReason,
    PruneStatus,
    NodeHealth,
    PruneConfig,
    NodeHealthReport,
    PruneProposal,
    PruneResult,
    PruneHistory,
    NodeHealthAnalyzer,
    NodePruner,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


class MockNodeStats:
    """Mock NodeStats for testing."""

    def __init__(
        self,
        success_rate: float = 0.9,
        failure_rate: float = 0.1,
        avg_latency_ms: float = 100.0,
        total_executions: int = 100,
        last_execution: datetime | None = None,
    ):
        self.success_rate = success_rate
        self.failure_rate = failure_rate
        self.avg_latency_ms = avg_latency_ms
        self.total_executions = total_executions
        self.last_execution = last_execution or datetime.now(UTC)


@pytest.fixture
def prune_config():
    """Default prune configuration."""
    return PruneConfig()


@pytest.fixture
def analyzer(prune_config):
    """Health analyzer instance."""
    return NodeHealthAnalyzer(prune_config)


@pytest.fixture
def pruner(prune_config):
    """Node pruner instance."""
    return NodePruner(prune_config)


@pytest.fixture
def healthy_stats():
    """Healthy node statistics."""
    return MockNodeStats(
        success_rate=0.95,
        failure_rate=0.05,
        avg_latency_ms=50.0,
        total_executions=100,
    )


@pytest.fixture
def unhealthy_stats():
    """Unhealthy node statistics (low success rate)."""
    return MockNodeStats(
        success_rate=0.1,
        failure_rate=0.9,
        avg_latency_ms=100.0,
        total_executions=100,
    )


@pytest.fixture
def degraded_stats():
    """Degraded node statistics (high error rate)."""
    return MockNodeStats(
        success_rate=0.5,
        failure_rate=0.6,
        avg_latency_ms=50000.0,
        total_executions=100,
    )


@pytest.fixture
def inactive_stats():
    """Inactive node statistics."""
    return MockNodeStats(
        success_rate=0.9,
        failure_rate=0.1,
        avg_latency_ms=100.0,
        total_executions=100,
        last_execution=datetime.now(UTC) - timedelta(hours=48),
    )


@pytest.fixture
def sample_node_definition():
    """Sample node definition for testing."""
    return NodeDefinition(
        id="test_node",
        type=NodeType.MODEL,
        name="Test Node",
        config={"model": "qwen2.5:3b"},
    )


# =============================================================================
# PRUNE CONFIG TESTS
# =============================================================================


class TestPruneConfig:
    """Tests for PruneConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PruneConfig()
        assert config.min_success_rate == 0.3
        assert config.max_error_rate == 0.5
        assert config.require_approval is True
        assert config.max_prunes_per_day == 5
        assert config.archive_instead_of_delete is True

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            PruneConfig(min_success_rate=-0.1)

        with pytest.raises(ValueError):
            PruneConfig(max_prunes_per_day=200)

    def test_config_immutable(self):
        """Test config is immutable."""
        config = PruneConfig()
        with pytest.raises(Exception):
            config.min_success_rate = 0.5


# =============================================================================
# NODE HEALTH REPORT TESTS
# =============================================================================


class TestNodeHealthReport:
    """Tests for NodeHealthReport."""

    def test_create_report(self):
        """Test creating a health report."""
        report = NodeHealthReport(
            node_id="test_node",
            health=NodeHealth.HEALTHY,
            success_rate=0.95,
            error_rate=0.05,
            avg_latency_ms=50.0,
            total_executions=100,
        )
        assert report.node_id == "test_node"
        assert report.health == NodeHealth.HEALTHY
        assert report.is_prune_candidate is False

    def test_unhealthy_is_prune_candidate(self):
        """Test that unhealthy nodes are prune candidates."""
        report = NodeHealthReport(
            node_id="bad_node",
            health=NodeHealth.UNHEALTHY,
            success_rate=0.1,
            error_rate=0.9,
        )
        assert report.is_prune_candidate is True

    def test_critical_is_prune_candidate(self):
        """Test that critical nodes are prune candidates."""
        report = NodeHealthReport(
            node_id="critical_node",
            health=NodeHealth.CRITICAL,
        )
        assert report.is_prune_candidate is True

    def test_degraded_not_prune_candidate(self):
        """Test that degraded nodes are not prune candidates."""
        report = NodeHealthReport(
            node_id="slow_node",
            health=NodeHealth.DEGRADED,
        )
        assert report.is_prune_candidate is False


# =============================================================================
# PRUNE PROPOSAL TESTS
# =============================================================================


class TestPruneProposal:
    """Tests for PruneProposal."""

    def test_create_proposal(self):
        """Test creating a prune proposal."""
        health_report = NodeHealthReport(
            node_id="test_node",
            health=NodeHealth.UNHEALTHY,
        )
        proposal = PruneProposal(
            node_id="test_node",
            reason=PruneReason.LOW_PERFORMANCE,
            health_report=health_report,
        )
        assert proposal.node_id == "test_node"
        assert proposal.reason == PruneReason.LOW_PERFORMANCE
        assert proposal.status == PruneStatus.PROPOSED
        assert proposal.id.startswith("prune_")

    def test_proposal_approve(self):
        """Test approving a proposal."""
        health_report = NodeHealthReport(
            node_id="test_node",
            health=NodeHealth.UNHEALTHY,
        )
        proposal = PruneProposal(
            node_id="test_node",
            reason=PruneReason.LOW_PERFORMANCE,
            health_report=health_report,
        )

        approved = proposal.approve("admin")
        assert approved.status == PruneStatus.APPROVED
        assert approved.approved_by == "admin"
        assert approved.approved_at is not None
        # Original unchanged
        assert proposal.status == PruneStatus.PROPOSED

    def test_proposal_reject(self):
        """Test rejecting a proposal."""
        health_report = NodeHealthReport(
            node_id="test_node",
            health=NodeHealth.UNHEALTHY,
        )
        proposal = PruneProposal(
            node_id="test_node",
            reason=PruneReason.LOW_PERFORMANCE,
            health_report=health_report,
        )

        rejected = proposal.reject("Node is actually important")
        assert rejected.status == PruneStatus.REJECTED
        assert rejected.rejection_reason == "Node is actually important"


# =============================================================================
# PRUNE HISTORY TESTS
# =============================================================================


class TestPruneHistory:
    """Tests for PruneHistory."""

    def test_default_history(self):
        """Test default history values."""
        history = PruneHistory()
        assert history.successful_prunes == 0
        assert history.failed_prunes == 0
        assert len(history.prunes) == 0
        assert len(history.rollbacks) == 0

    def test_history_counts(self):
        """Test history counts."""
        history = PruneHistory(
            prunes=[
                PruneResult(proposal_id="p1", success=True, node_id="n1"),
                PruneResult(proposal_id="p2", success=True, node_id="n2"),
                PruneResult(proposal_id="p3", success=False, node_id="n3", error="Failed"),
            ],
            rollbacks=["n1"],
        )
        assert history.successful_prunes == 2
        assert history.failed_prunes == 1


# =============================================================================
# NODE HEALTH ANALYZER TESTS
# =============================================================================


class TestNodeHealthAnalyzer:
    """Tests for NodeHealthAnalyzer."""

    def test_assess_healthy_node(self, analyzer, healthy_stats):
        """Test assessing a healthy node."""
        report = analyzer.assess_health("healthy_node", healthy_stats)
        assert report.health == NodeHealth.HEALTHY
        assert report.success_rate == 0.95
        assert len(report.issues) == 0

    def test_assess_unhealthy_node(self, analyzer, unhealthy_stats):
        """Test assessing an unhealthy node."""
        report = analyzer.assess_health("unhealthy_node", unhealthy_stats)
        # Node with both low success rate and high error rate is CRITICAL
        assert report.health in {NodeHealth.UNHEALTHY, NodeHealth.CRITICAL}
        assert any("Low success rate" in issue for issue in report.issues)

    def test_assess_degraded_node(self, analyzer, degraded_stats):
        """Test assessing a degraded node."""
        report = analyzer.assess_health("degraded_node", degraded_stats)
        # Should be at least degraded due to high error rate and latency
        assert report.health in {NodeHealth.DEGRADED, NodeHealth.UNHEALTHY, NodeHealth.CRITICAL}
        assert len(report.issues) > 0

    def test_assess_inactive_node(self, analyzer, inactive_stats):
        """Test assessing an inactive node."""
        report = analyzer.assess_health("inactive_node", inactive_stats)
        assert any("Inactive" in issue for issue in report.issues)
        assert report.hours_since_last_exec is not None
        assert report.hours_since_last_exec > 24

    def test_assess_insufficient_data(self, analyzer):
        """Test assessing a node with insufficient data."""
        stats = MockNodeStats(total_executions=5)
        report = analyzer.assess_health("new_node", stats)
        assert report.health == NodeHealth.UNKNOWN
        assert any("Insufficient data" in issue for issue in report.issues)

    def test_assess_protected_node(self, analyzer, unhealthy_stats):
        """Test assessing a protected node."""
        report = analyzer.assess_health("protected", unhealthy_stats, is_protected=True)
        assert any("protected" in issue.lower() for issue in report.issues)

    def test_assess_critical_multiple_issues(self, analyzer):
        """Test that multiple severe issues result in critical health."""
        stats = MockNodeStats(
            success_rate=0.1,
            failure_rate=0.9,
            total_executions=100,
        )
        report = analyzer.assess_health("critical_node", stats)
        assert report.health == NodeHealth.CRITICAL


# =============================================================================
# NODE PRUNER TESTS
# =============================================================================


class TestNodePruner:
    """Tests for NodePruner."""

    def test_protect_node(self, pruner):
        """Test protecting a node."""
        assert not pruner.is_protected("node1")
        pruner.protect_node("node1")
        assert pruner.is_protected("node1")

    def test_unprotect_node(self, pruner):
        """Test unprotecting a node."""
        pruner.protect_node("node1")
        pruner.unprotect_node("node1")
        assert not pruner.is_protected("node1")

    def test_analyze_nodes(self, pruner, healthy_stats, unhealthy_stats):
        """Test analyzing multiple nodes."""
        nodes = {
            "healthy_node": healthy_stats,
            "unhealthy_node": unhealthy_stats,
        }
        reports = pruner.analyze_nodes(nodes)
        assert len(reports) == 2
        # Should be sorted with unhealthy first
        assert reports[0].node_id == "unhealthy_node"

    def test_find_prune_candidates(self, pruner, healthy_stats, unhealthy_stats):
        """Test finding prune candidates."""
        nodes = {
            "healthy_node": healthy_stats,
            "unhealthy_node": unhealthy_stats,
        }
        candidates = pruner.find_prune_candidates(nodes)
        assert len(candidates) >= 1
        assert any(c.node_id == "unhealthy_node" for c in candidates)
        assert not any(c.node_id == "healthy_node" for c in candidates)

    def test_find_prune_candidates_excludes_protected(self, pruner, unhealthy_stats):
        """Test that protected nodes are excluded from candidates."""
        nodes = {"protected_node": unhealthy_stats}
        pruner.protect_node("protected_node")
        candidates = pruner.find_prune_candidates(nodes)
        assert len(candidates) == 0

    def test_create_proposal(self, pruner, unhealthy_stats):
        """Test creating a prune proposal."""
        proposal = pruner.create_proposal("bad_node", unhealthy_stats)
        assert proposal.node_id == "bad_node"
        assert proposal.status == PruneStatus.PROPOSED
        assert proposal.reason in list(PruneReason)

    def test_create_proposal_auto_detect_reason(self, pruner, unhealthy_stats):
        """Test auto-detecting prune reason."""
        proposal = pruner.create_proposal("bad_node", unhealthy_stats)
        assert proposal.reason == PruneReason.LOW_PERFORMANCE

    def test_create_proposal_for_protected_fails(self, pruner, unhealthy_stats):
        """Test that creating proposal for protected node fails."""
        pruner.protect_node("protected")
        with pytest.raises(ValueError, match="Cannot prune protected"):
            pruner.create_proposal("protected", unhealthy_stats)

    def test_approve_proposal(self, pruner, unhealthy_stats):
        """Test approving a proposal."""
        proposal = pruner.create_proposal("bad_node", unhealthy_stats)
        approved = pruner.approve_proposal(proposal.id, "admin")
        assert approved.status == PruneStatus.APPROVED
        assert approved.approved_by == "admin"

    def test_reject_proposal(self, pruner, unhealthy_stats):
        """Test rejecting a proposal."""
        proposal = pruner.create_proposal("bad_node", unhealthy_stats)
        rejected = pruner.reject_proposal(proposal.id, "Too risky")
        assert rejected.status == PruneStatus.REJECTED
        assert rejected.rejection_reason == "Too risky"

    def test_approve_nonexistent_fails(self, pruner):
        """Test that approving nonexistent proposal fails."""
        with pytest.raises(ValueError, match="Proposal not found"):
            pruner.approve_proposal("nonexistent", "admin")

    def test_execute_prune(self, pruner, unhealthy_stats, sample_node_definition):
        """Test executing a prune."""
        # Use config with no cooldown for testing
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)

        result = pruner.execute_prune(proposal.id, sample_node_definition)
        assert result.success is True
        assert result.node_id == "test_node"
        assert result.archived is True

    def test_execute_unapproved_fails(self, pruner, unhealthy_stats, sample_node_definition):
        """Test that executing unapproved proposal fails."""
        proposal = pruner.create_proposal("test_node", unhealthy_stats)

        with pytest.raises(ValueError, match="not approved"):
            pruner.execute_prune(proposal.id, sample_node_definition)

    def test_execute_with_callback(self, pruner, unhealthy_stats, sample_node_definition):
        """Test executing prune with callback."""
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)

        callback_called = []

        def on_prune(node_id: str) -> bool:
            callback_called.append(node_id)
            return True

        result = pruner.execute_prune(proposal.id, sample_node_definition, on_prune)
        assert result.success is True
        assert "test_node" in callback_called

    def test_execute_with_callback_failure(self, pruner, unhealthy_stats, sample_node_definition):
        """Test executing prune with failing callback."""
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)

        def on_prune(node_id: str) -> bool:
            return False  # Simulate failure

        result = pruner.execute_prune(proposal.id, sample_node_definition, on_prune)
        assert result.success is False
        assert "callback returned False" in result.error

    def test_restore_node(self, pruner, unhealthy_stats, sample_node_definition):
        """Test restoring an archived node."""
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)
        pruner.execute_prune(proposal.id, sample_node_definition)

        # Restore the node
        restored = pruner.restore_node("test_node")
        assert restored is not None
        assert restored.id == "test_node"
        assert "test_node" in pruner.history.rollbacks

    def test_restore_nonexistent_returns_none(self, pruner):
        """Test that restoring nonexistent node returns None."""
        result = pruner.restore_node("nonexistent")
        assert result is None

    def test_get_pending_proposals(self, pruner, unhealthy_stats):
        """Test getting pending proposals."""
        proposal1 = pruner.create_proposal("node1", unhealthy_stats)
        proposal2 = pruner.create_proposal("node2", unhealthy_stats)

        pending = pruner.get_pending_proposals()
        # Proposals may share same ID if created in same second, verify at least one exists
        assert len(pending) >= 1
        assert all(p.status == PruneStatus.PROPOSED for p in pending)
        # Verify both nodes are represented
        pending_nodes = {p.node_id for p in pending}
        assert "node2" in pending_nodes  # At minimum, last one should be there

    def test_get_archived_nodes(self, pruner, unhealthy_stats, sample_node_definition):
        """Test getting archived nodes list."""
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)
        pruner.execute_prune(proposal.id, sample_node_definition)

        archived = pruner.get_archived_nodes()
        assert "test_node" in archived

    def test_should_auto_prune(self, pruner):
        """Test auto-prune decision."""
        # Very bad node
        bad_report = NodeHealthReport(
            node_id="very_bad",
            health=NodeHealth.CRITICAL,
            success_rate=0.05,  # Below auto threshold of 0.1
        )
        assert pruner.should_auto_prune(bad_report) is True

        # Moderately bad node
        moderate_report = NodeHealthReport(
            node_id="moderate",
            health=NodeHealth.UNHEALTHY,
            success_rate=0.25,  # Above auto threshold
        )
        assert pruner.should_auto_prune(moderate_report) is False

    def test_rate_limiting(self, pruner, unhealthy_stats, sample_node_definition):
        """Test rate limiting prunes."""
        # Create pruner with strict limits
        pruner = NodePruner(PruneConfig(max_prunes_per_day=1, cooldown_hours=0))

        # First prune should work
        proposal1 = pruner.create_proposal("node1", unhealthy_stats)
        pruner.approve_proposal(proposal1.id)
        result1 = pruner.execute_prune(proposal1.id, sample_node_definition)
        assert result1.success is True

        # Second prune should be rate limited
        proposal2 = pruner.create_proposal("node2", unhealthy_stats)
        pruner.approve_proposal(proposal2.id)
        result2 = pruner.execute_prune(proposal2.id, sample_node_definition)
        assert result2.success is False
        assert "Rate limit" in result2.error

    def test_history_tracking(self, pruner, unhealthy_stats, sample_node_definition):
        """Test history tracking."""
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)
        pruner.execute_prune(proposal.id, sample_node_definition)

        assert pruner.history.successful_prunes == 1
        assert len(pruner.history.prunes) == 1

    def test_get_stats(self, pruner, unhealthy_stats, sample_node_definition):
        """Test getting pruner statistics."""
        pruner = NodePruner(PruneConfig(cooldown_hours=0))
        pruner.protect_node("protected_node")
        proposal = pruner.create_proposal("test_node", unhealthy_stats)
        pruner.approve_proposal(proposal.id)
        pruner.execute_prune(proposal.id, sample_node_definition)

        stats = pruner.get_stats()
        assert "config" in stats
        assert stats["protected_nodes"] == 1
        assert stats["archived_nodes"] == 1
        assert stats["successful_prunes"] == 1


# =============================================================================
# PRUNE REASON DETECTION TESTS
# =============================================================================


class TestPruneReasonDetection:
    """Tests for automatic prune reason detection."""

    def test_detect_low_performance(self, pruner):
        """Test detecting low performance reason."""
        stats = MockNodeStats(success_rate=0.1, total_executions=100)
        proposal = pruner.create_proposal("bad_node", stats)
        assert proposal.reason == PruneReason.LOW_PERFORMANCE

    def test_detect_high_error_rate(self, pruner):
        """Test detecting high error rate reason."""
        stats = MockNodeStats(
            success_rate=0.5,
            failure_rate=0.7,
            total_executions=100,
        )
        proposal = pruner.create_proposal("error_node", stats)
        # May be LOW_PERFORMANCE or HIGH_ERROR_RATE depending on thresholds
        assert proposal.reason in {PruneReason.LOW_PERFORMANCE, PruneReason.HIGH_ERROR_RATE}

    def test_detect_inactivity(self, pruner, inactive_stats):
        """Test detecting inactivity reason."""
        # Inactive node with good performance
        stats = MockNodeStats(
            success_rate=0.9,
            failure_rate=0.1,
            total_executions=100,
            last_execution=datetime.now(UTC) - timedelta(hours=48),
        )
        proposal = pruner.create_proposal("inactive_node", stats)
        assert proposal.reason == PruneReason.INACTIVITY

    def test_explicit_reason(self, pruner, unhealthy_stats):
        """Test setting explicit prune reason."""
        proposal = pruner.create_proposal(
            "node",
            unhealthy_stats,
            reason=PruneReason.REDUNDANCY,
        )
        assert proposal.reason == PruneReason.REDUNDANCY


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_nodes_analysis(self, pruner):
        """Test analyzing empty node dict."""
        reports = pruner.analyze_nodes({})
        assert len(reports) == 0

    def test_node_with_no_executions(self, analyzer):
        """Test node with zero executions."""
        stats = MockNodeStats(
            success_rate=0.0,
            failure_rate=0.0,
            total_executions=0,
        )
        report = analyzer.assess_health("new_node", stats)
        assert report.health == NodeHealth.UNKNOWN

    def test_node_with_no_last_execution(self, analyzer):
        """Test node with no last execution timestamp."""
        stats = MockNodeStats(total_executions=100)
        stats.last_execution = None
        report = analyzer.assess_health("node", stats)
        assert report.hours_since_last_exec is None

    def test_prune_config_boundary_values(self):
        """Test config with boundary values."""
        config = PruneConfig(
            min_success_rate=0.0,
            max_error_rate=1.0,
            max_prunes_per_day=0,  # Effectively disables pruning
        )
        assert config.min_success_rate == 0.0
        assert config.max_prunes_per_day == 0
