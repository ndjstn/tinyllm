"""Tests for error handling and recovery system."""

import uuid
from datetime import datetime, timedelta

import pytest

from tinyllm.error_aggregation import (
    AggregatedError,
    ErrorAggregator,
    ErrorSignature,
    get_aggregator,
)
from tinyllm.error_branching import (
    BranchCondition,
    BranchStrategy,
    ErrorBranchManager,
    ErrorBranchRule,
    create_escalation_rule,
    create_fallback_rule,
    create_graceful_degradation_rule,
    create_retry_rule,
    get_branch_manager,
)
from tinyllm.error_enrichment import (
    EnrichedError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    enrich_error,
)
from tinyllm.error_impact import (
    ImpactLevel,
    ImpactScore,
    ImpactScorer,
    get_scorer,
)
from tinyllm.errors import (
    ConfigurationError,
    ExecutionError,
    ModelError,
    NetworkError,
    ResourceExhaustedError as ResourceError,
    TimeoutError,
    TinyLLMError,
    ValidationError,
)


# ============================================================================
# Error Context Tests
# ============================================================================


class TestErrorContext:
    """Test error context enrichment."""

    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            stack_trace="test stack trace",
            exception_type="ValueError",
            exception_message="test error",
            trace_id="trace-123",
            node_id="node-1",
            graph_id="graph-1",
        )

        assert context.exception_type == "ValueError"
        assert context.exception_message == "test error"
        assert context.trace_id == "trace-123"
        assert context.node_id == "node-1"
        assert context.graph_id == "graph-1"

    def test_error_context_with_state(self):
        """Test error context with execution state."""
        context = ErrorContext(
            stack_trace="test",
            exception_type="TestError",
            exception_message="test",
            context_variables={"var1": "value1"},
            node_config={"timeout": 5000},
            visited_nodes=["node1", "node2", "node3"],
            execution_step=5,
        )

        assert context.context_variables == {"var1": "value1"}
        assert context.node_config == {"timeout": 5000}
        assert len(context.visited_nodes) == 3
        assert context.execution_step == 5


class TestTinyLLMError:
    """Test TinyLLM error classes."""

    def test_base_error_creation(self):
        """Test creating base error."""
        error = TinyLLMError(
            "test error",
            code="EXECUTION_ERROR",
            details={"trace_id": "trace-123"},
            recoverable=False,
        )

        assert error.message == "test error"
        assert error.code == "EXECUTION_ERROR"
        assert error.details["trace_id"] == "trace-123"
        assert error.recoverable is False
        assert error.timestamp is not None

    def test_error_retryability(self):
        """Test error retryability classification."""
        timeout_error = TimeoutError("timeout")
        assert timeout_error._is_retryable()

        validation_error = ValidationError("invalid input")
        assert not validation_error._is_retryable()

    def test_error_suggested_actions(self):
        """Test error suggested actions."""
        timeout_error = TimeoutError("timeout")
        actions = timeout_error._get_suggested_actions()

        assert "Retry the operation" in actions
        assert "Increase timeout value" in actions

    def test_specific_error_types(self):
        """Test specific error type creation."""
        validation_err = ValidationError("invalid")
        assert validation_err.category == ErrorCategory.VALIDATION
        assert not validation_err._is_user_error()  # Should be True for validation

        model_err = ModelError("model failed")
        assert model_err.category == ErrorCategory.MODEL

        timeout_err = TimeoutError("timeout")
        assert timeout_err.category == ErrorCategory.TIMEOUT

        network_err = NetworkError("connection failed")
        assert network_err.category == ErrorCategory.NETWORK


# ============================================================================
# Error Aggregation Tests
# ============================================================================


class TestErrorSignature:
    """Test error signature generation."""

    def test_signature_creation(self):
        """Test creating error signature."""
        error = TinyLLMError(
            "test error",
            category=ErrorCategory.EXECUTION,
            trace_id="trace-123",
            node_id="node-1",
        )
        enriched = error.to_enriched_error("error-1")
        signature = ErrorSignature.from_enriched_error(enriched)

        assert signature.category == ErrorCategory.EXECUTION
        assert signature.node_id == "node-1"
        assert len(signature.signature_hash) == 16

    def test_message_normalization(self):
        """Test error message normalization."""
        # UUIDs should be normalized
        msg1 = "Error with UUID: 12345678-1234-1234-1234-123456789abc"
        msg2 = "Error with UUID: 87654321-4321-4321-4321-cba987654321"
        norm1 = ErrorSignature._normalize_message(msg1)
        norm2 = ErrorSignature._normalize_message(msg2)
        assert norm1 == norm2

        # Numeric IDs should be normalized
        msg3 = "Error with ID: 123456"
        msg4 = "Error with ID: 789012"
        norm3 = ErrorSignature._normalize_message(msg3)
        norm4 = ErrorSignature._normalize_message(msg4)
        assert norm3 == norm4

    def test_same_errors_same_signature(self):
        """Test that similar errors produce same signature."""
        error1 = ExecutionError("Failed at step 123", node_id="node-1")
        error2 = ExecutionError("Failed at step 456", node_id="node-1")

        enriched1 = error1.to_enriched_error("err-1")
        enriched2 = error2.to_enriched_error("err-2")

        sig1 = ErrorSignature.from_enriched_error(enriched1)
        sig2 = ErrorSignature.from_enriched_error(enriched2)

        assert sig1.signature_hash == sig2.signature_hash


class TestErrorAggregator:
    """Test error aggregation."""

    def test_aggregator_creation(self):
        """Test creating error aggregator."""
        aggregator = ErrorAggregator()
        assert aggregator.max_aggregations == 10000

    def test_add_single_error(self):
        """Test adding single error."""
        aggregator = ErrorAggregator()
        error = ExecutionError("test error", node_id="node-1")
        enriched = error.to_enriched_error("error-1")

        agg = aggregator.add_error(enriched)

        assert agg.count == 1
        assert len(agg.error_ids) == 1
        assert agg.signature.exception_type == "ExecutionError"

    def test_aggregate_similar_errors(self):
        """Test aggregating similar errors."""
        aggregator = ErrorAggregator()

        # Add multiple similar errors
        for i in range(5):
            error = ExecutionError(f"Failed at step {i}", node_id="node-1")
            enriched = error.to_enriched_error(f"error-{i}")
            aggregator.add_error(enriched)

        # Should be aggregated into one
        stats = aggregator.get_statistics()
        assert stats["total_errors_seen"] == 5
        assert stats["unique_signatures"] == 1

    def test_aggregate_different_errors(self):
        """Test that different errors don't aggregate."""
        aggregator = ErrorAggregator()

        # Add different error types
        error1 = ExecutionError("execution error", node_id="node-1")
        error2 = ValidationError("validation error", node_id="node-1")

        enriched1 = error1.to_enriched_error("error-1")
        enriched2 = error2.to_enriched_error("error-2")

        aggregator.add_error(enriched1)
        aggregator.add_error(enriched2)

        stats = aggregator.get_statistics()
        assert stats["unique_signatures"] == 2

    def test_get_top_errors(self):
        """Test getting top errors."""
        aggregator = ErrorAggregator()

        # Add multiple errors with different frequencies
        for i in range(10):
            error = ExecutionError("error A", node_id="node-1")
            enriched = error.to_enriched_error(f"error-a-{i}")
            aggregator.add_error(enriched)

        for i in range(3):
            error = ValidationError("error B", node_id="node-1")
            enriched = error.to_enriched_error(f"error-b-{i}")
            aggregator.add_error(enriched)

        top = aggregator.get_top_errors(limit=2)
        assert len(top) <= 2
        assert top[0].count >= top[1].count  # Sorted by count

    def test_get_recent_errors(self):
        """Test getting recent errors."""
        aggregator = ErrorAggregator()

        # Add error
        error = ExecutionError("recent error", node_id="node-1")
        enriched = error.to_enriched_error("error-1")
        aggregator.add_error(enriched)

        recent = aggregator.get_recent_errors(hours=1)
        assert len(recent) >= 1


# ============================================================================
# Error Impact Scoring Tests
# ============================================================================


class TestImpactScorer:
    """Test error impact scoring."""

    def test_scorer_creation(self):
        """Test creating impact scorer."""
        scorer = ImpactScorer()
        assert scorer.weights["severity"] > 0
        assert scorer.weights["frequency"] > 0

    def test_score_single_error(self):
        """Test scoring single error."""
        scorer = ImpactScorer()

        error = ExecutionError("test error", severity=ErrorSeverity.ERROR)
        enriched = error.to_enriched_error("error-1")

        score = scorer.score_error(enriched)

        assert 0 <= score.total_score <= 100
        assert score.severity_score > 0
        assert isinstance(score.impact_level, ImpactLevel)

    def test_severity_affects_score(self):
        """Test that severity affects score."""
        scorer = ImpactScorer()

        error_low = TinyLLMError(
            "test",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.WARNING,
        )
        error_high = TinyLLMError(
            "test",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
        )

        enriched_low = error_low.to_enriched_error("error-1")
        enriched_high = error_high.to_enriched_error("error-2")

        score_low = scorer.score_error(enriched_low)
        score_high = scorer.score_error(enriched_high)

        assert score_high.total_score > score_low.total_score

    def test_score_aggregated_error(self):
        """Test scoring aggregated error."""
        scorer = ImpactScorer()
        aggregator = ErrorAggregator()

        # Add multiple errors
        for i in range(10):
            error = ExecutionError("test error", node_id="node-1")
            enriched = error.to_enriched_error(f"error-{i}")
            aggregator.add_error(enriched)

        # Get aggregation
        top = aggregator.get_top_errors(limit=1)
        agg = top[0]

        score = scorer.score_aggregated_error(agg)

        assert score.frequency_score > 0  # Should have frequency
        assert score.total_score > 0

    def test_impact_level_classification(self):
        """Test impact level classification."""
        scorer = ImpactScorer()

        # Test different score ranges
        assert scorer._classify_impact(5.0) == ImpactLevel.NEGLIGIBLE
        assert scorer._classify_impact(20.0) == ImpactLevel.LOW
        assert scorer._classify_impact(40.0) == ImpactLevel.MEDIUM
        assert scorer._classify_impact(60.0) == ImpactLevel.HIGH
        assert scorer._classify_impact(80.0) == ImpactLevel.CRITICAL
        assert scorer._classify_impact(95.0) == ImpactLevel.CATASTROPHIC

    def test_recency_affects_score(self):
        """Test that recency affects score."""
        scorer = ImpactScorer()

        # Recent error
        recent_score = scorer._score_recency(datetime.utcnow())

        # Old error
        old_time = datetime.utcnow() - timedelta(hours=48)
        old_score = scorer._score_recency(old_time)

        assert recent_score > old_score


# ============================================================================
# Error Branching Tests
# ============================================================================


class TestErrorBranchRule:
    """Test error branch rules."""

    def test_rule_creation(self):
        """Test creating branch rule."""
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Fallback rule",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback-node",
        )

        assert rule.rule_id == "rule-1"
        assert rule.condition_type == BranchCondition.ON_ERROR
        assert rule.target_node_id == "fallback-node"

    def test_rule_matches_any_error(self):
        """Test rule matching any error."""
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Catch all",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        error = ExecutionError("test")
        assert rule.matches(error)

    def test_rule_matches_category(self):
        """Test rule matching specific category."""
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Match timeout",
            condition_type=BranchCondition.ON_CATEGORY,
            condition_config={"category": "timeout"},
            target_node_id="retry-node",
        )

        timeout_error = TimeoutError("timeout")
        execution_error = ExecutionError("other")

        assert rule.matches(timeout_error)
        assert not rule.matches(execution_error)

    def test_rule_matches_severity(self):
        """Test rule matching minimum severity."""
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Match critical",
            condition_type=BranchCondition.ON_SEVERITY,
            condition_config={"min_severity": "critical"},
            target_node_id="escalate",
        )

        critical_error = TinyLLMError(
            "critical",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
        )
        warning_error = TinyLLMError(
            "warning",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.WARNING,
        )

        assert rule.matches(critical_error)
        assert not rule.matches(warning_error)

    def test_rule_matches_retryable(self):
        """Test rule matching retryable errors."""
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Retry retryable",
            condition_type=BranchCondition.ON_RETRYABLE,
            target_node_id="retry",
        )

        timeout_error = TimeoutError("timeout")
        validation_error = ValidationError("invalid")

        assert rule.matches(timeout_error)
        assert not rule.matches(validation_error)

    def test_max_activations(self):
        """Test max activations limit."""
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Limited retry",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="retry",
            max_activations=3,
        )

        error = ExecutionError("test")

        # Should match first 3 times
        assert rule.matches(error)
        rule.increment_activation()
        assert rule.matches(error)
        rule.increment_activation()
        assert rule.matches(error)
        rule.increment_activation()

        # Should not match after max activations
        assert not rule.matches(error)


class TestErrorBranchManager:
    """Test error branch manager."""

    def test_manager_creation(self):
        """Test creating branch manager."""
        manager = ErrorBranchManager()
        assert manager is not None

    def test_add_global_rule(self):
        """Test adding global rule."""
        manager = ErrorBranchManager()
        rule = create_fallback_rule("rule-1", "fallback-node")

        manager.add_rule(rule)

        all_rules = manager.get_all_rules()
        assert len(all_rules) == 1

    def test_add_node_rule(self):
        """Test adding node-specific rule."""
        manager = ErrorBranchManager()
        rule = create_fallback_rule("rule-1", "fallback-node")

        manager.add_rule(rule, node_id="node-1")

        node_rules = manager.get_rules_for_node("node-1")
        assert len(node_rules) >= 1

    def test_evaluate_error_with_match(self):
        """Test evaluating error with matching rule."""
        manager = ErrorBranchManager()
        rule = create_fallback_rule("rule-1", "fallback-node")
        manager.add_rule(rule)

        error = ExecutionError("test")
        result = manager.evaluate_error(error)

        assert result.should_branch
        assert result.target_node_id == "fallback-node"
        assert result.strategy == BranchStrategy.FALLBACK

    def test_evaluate_error_without_match(self):
        """Test evaluating error without matching rule."""
        manager = ErrorBranchManager()

        # Add rule that won't match
        rule = ErrorBranchRule(
            rule_id="rule-1",
            name="Never match",
            condition_type=BranchCondition.ON_CATEGORY,
            condition_config={"category": "network"},
            target_node_id="fallback",
        )
        manager.add_rule(rule)

        error = ValidationError("test")
        result = manager.evaluate_error(error, node_id="node-1")

        assert not result.should_branch

    def test_rule_priority(self):
        """Test rule priority ordering."""
        manager = ErrorBranchManager()

        # Add rules with different priorities
        rule1 = create_fallback_rule("rule-1", "fallback-1", priority=50)
        rule2 = create_fallback_rule("rule-2", "fallback-2", priority=80)

        manager.add_rule(rule1)
        manager.add_rule(rule2)

        # Higher priority should match first
        error = ExecutionError("test")
        result = manager.evaluate_error(error)

        assert result.target_node_id == "fallback-2"  # Higher priority

    def test_helper_functions(self):
        """Test helper functions for creating rules."""
        fallback = create_fallback_rule("fb", "fallback-node")
        assert fallback.strategy == BranchStrategy.FALLBACK

        retry = create_retry_rule("rt", "retry-node", max_retries=3)
        assert retry.strategy == BranchStrategy.RETRY_WITH_ALTERNATIVE
        assert retry.max_activations == 3

        escalate = create_escalation_rule("esc", "escalate-node")
        assert escalate.strategy == BranchStrategy.ESCALATE

        degrade = create_graceful_degradation_rule("deg", "degraded-node")
        assert degrade.strategy == BranchStrategy.GRACEFUL_DEGRADATION

    def test_remove_rule(self):
        """Test removing rule."""
        manager = ErrorBranchManager()
        rule = create_fallback_rule("rule-1", "fallback")
        manager.add_rule(rule)

        assert len(manager.get_all_rules()) == 1

        removed = manager.remove_rule("rule-1")
        assert removed
        assert len(manager.get_all_rules()) == 0

    def test_get_statistics(self):
        """Test getting statistics."""
        manager = ErrorBranchManager()

        # Add various rules
        manager.add_rule(create_fallback_rule("fb1", "fallback1"))
        manager.add_rule(create_retry_rule("rt1", "retry1"))
        manager.add_rule(create_escalation_rule("esc1", "escalate1"))

        stats = manager.get_statistics()

        assert stats["total_rules"] == 3
        assert stats["enabled_rules"] == 3
        assert "by_strategy" in stats
        assert "by_condition" in stats


# ============================================================================
# Integration Tests
# ============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    def test_full_error_pipeline(self):
        """Test complete error handling pipeline."""
        # Create error
        error = ExecutionError(
            "Critical execution failure",
            severity=ErrorSeverity.CRITICAL,
            node_id="worker-node",
            trace_id="trace-123",
        )

        # Enrich error
        enriched = error.to_enriched_error(
            "error-1",
            context_variables={"attempt": 1},
            visited_nodes=["entry", "router", "worker"],
        )

        # Aggregate error
        aggregator = ErrorAggregator()
        agg = aggregator.add_error(enriched)

        # Score impact
        scorer = ImpactScorer()
        score = scorer.score_aggregated_error(agg)

        # Check branching
        manager = ErrorBranchManager()
        escalate_rule = create_escalation_rule(
            "escalate-critical",
            "supervisor-node",
            min_severity=ErrorSeverity.CRITICAL,
        )
        manager.add_rule(escalate_rule)

        branch_result = manager.evaluate_error(error, node_id="worker-node")

        # Verify pipeline
        assert agg.count == 1
        assert score.impact_level == ImpactLevel.CRITICAL
        assert branch_result.should_branch
        assert branch_result.target_node_id == "supervisor-node"

    def test_error_deduplication_with_scoring(self):
        """Test error deduplication combined with impact scoring."""
        aggregator = ErrorAggregator()
        scorer = ImpactScorer()

        # Add multiple similar errors
        for i in range(10):
            error = TimeoutError(
                f"Operation timed out after {i} seconds",
                node_id="api-node",
            )
            enriched = error.to_enriched_error(f"error-{i}")
            aggregator.add_error(enriched)

        # Should be deduplicated
        top = aggregator.get_top_errors(limit=1)
        assert len(top) == 1
        assert top[0].count == 10

        # Score should reflect frequency
        score = scorer.score_aggregated_error(top[0])
        assert score.frequency_score > 0
        assert score.impact_level in {ImpactLevel.MEDIUM, ImpactLevel.HIGH}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
