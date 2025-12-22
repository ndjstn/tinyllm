"""Tests for error-triggered graph branching."""

from unittest.mock import MagicMock

import pytest

from tinyllm.error_branching import (
    BranchCondition,
    BranchStrategy,
    ErrorBranchManager,
    ErrorBranchResult,
    ErrorBranchRule,
    create_escalation_rule,
    create_fallback_rule,
    create_graceful_degradation_rule,
    create_retry_rule,
    get_branch_manager,
)
from tinyllm.error_enrichment import ErrorCategory, ErrorSeverity
from tinyllm.errors import TinyLLMError


class TestErrorBranchRule:
    """Tests for ErrorBranchRule."""

    def test_rule_creation(self):
        """Test creating a branch rule."""
        rule = ErrorBranchRule(
            rule_id="rule_001",
            name="Fallback on timeout",
            condition_type=BranchCondition.ON_CATEGORY,
            condition_config={"category": "timeout"},
            target_node_id="fallback_node",
            strategy=BranchStrategy.FALLBACK,
            priority=80,
        )

        assert rule.rule_id == "rule_001"
        assert rule.condition_type == BranchCondition.ON_CATEGORY
        assert rule.target_node_id == "fallback_node"
        assert rule.enabled is True
        assert rule.activation_count == 0

    def test_matches_on_error_condition(self):
        """Test ON_ERROR condition matches any error."""
        rule = ErrorBranchRule(
            rule_id="catch_all",
            name="Catch all errors",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="error_handler",
        )

        error = RuntimeError("Any error")

        assert rule.matches(error) is True

    def test_matches_on_category_condition(self):
        """Test ON_CATEGORY condition matches specific category."""
        rule = ErrorBranchRule(
            rule_id="timeout_handler",
            name="Handle timeouts",
            condition_type=BranchCondition.ON_CATEGORY,
            condition_config={"category": "timeout"},
            target_node_id="timeout_fallback",
        )

        # Create TinyLLM error with timeout category
        timeout_error = TinyLLMError(message="Connection timeout", recoverable=True)
        timeout_error.category = ErrorCategory.TIMEOUT
        timeout_error.severity = ErrorSeverity.ERROR

        # Create TinyLLM error with different category
        validation_error = TinyLLMError(message="Validation failed", recoverable=True)
        validation_error.category = ErrorCategory.VALIDATION
        validation_error.severity = ErrorSeverity.WARNING

        assert rule.matches(timeout_error) is True
        assert rule.matches(validation_error) is False

    def test_matches_on_severity_condition(self):
        """Test ON_SEVERITY condition matches minimum severity."""
        rule = ErrorBranchRule(
            rule_id="critical_handler",
            name="Handle critical errors",
            condition_type=BranchCondition.ON_SEVERITY,
            condition_config={"min_severity": "error"},
            target_node_id="critical_fallback",
        )

        # Critical error (higher than ERROR)
        critical_error = TinyLLMError(message="Critical failure", recoverable=False)
        critical_error.category = ErrorCategory.INTERNAL
        critical_error.severity = ErrorSeverity.CRITICAL

        # Warning (lower than ERROR)
        warning = TinyLLMError(message="Warning message", recoverable=True)
        warning.category = ErrorCategory.VALIDATION
        warning.severity = ErrorSeverity.WARNING

        assert rule.matches(critical_error) is True
        assert rule.matches(warning) is False

    def test_matches_on_retryable_condition(self):
        """Test ON_RETRYABLE condition matches retryable errors."""
        rule = ErrorBranchRule(
            rule_id="retry_handler",
            name="Retry retryable errors",
            condition_type=BranchCondition.ON_RETRYABLE,
            target_node_id="retry_node",
        )

        # Retryable TinyLLM error
        retryable = TinyLLMError(message="Temporary failure", recoverable=True)
        retryable.category = ErrorCategory.NETWORK
        retryable.severity = ErrorSeverity.ERROR

        # Non-retryable TinyLLM error
        fatal = TinyLLMError(message="Fatal error", recoverable=False)
        fatal.category = ErrorCategory.INTERNAL
        fatal.severity = ErrorSeverity.FATAL

        assert rule.matches(retryable) is True
        assert rule.matches(fatal) is False

    def test_matches_on_retryable_non_tinyllm_error(self):
        """Test ON_RETRYABLE matches retryable patterns in standard errors."""
        rule = ErrorBranchRule(
            rule_id="retry_handler",
            name="Retry on timeout",
            condition_type=BranchCondition.ON_RETRYABLE,
            target_node_id="retry_node",
        )

        # Standard errors with retryable patterns
        timeout_error = TimeoutError("Connection timeout")
        connection_error = ConnectionError("Network connection failed")
        value_error = ValueError("Invalid value")

        assert rule.matches(timeout_error) is True
        assert rule.matches(connection_error) is True
        assert rule.matches(value_error) is False

    def test_matches_on_fatal_condition(self):
        """Test ON_FATAL condition matches non-recoverable errors."""
        rule = ErrorBranchRule(
            rule_id="fatal_handler",
            name="Handle fatal errors",
            condition_type=BranchCondition.ON_FATAL,
            target_node_id="fatal_fallback",
        )

        # Fatal TinyLLM error
        fatal = TinyLLMError(message="Fatal error", recoverable=False)
        fatal.category = ErrorCategory.INTERNAL
        fatal.severity = ErrorSeverity.FATAL

        # Recoverable TinyLLM error
        recoverable = TinyLLMError(message="Temporary error", recoverable=True)
        recoverable.category = ErrorCategory.NETWORK
        recoverable.severity = ErrorSeverity.ERROR

        assert rule.matches(fatal) is True
        assert rule.matches(recoverable) is False

    def test_matches_on_custom_condition(self):
        """Test ON_CUSTOM condition with custom function."""

        def custom_condition(error, context):
            # Match errors with "database" in message
            return "database" in str(error).lower()

        rule = ErrorBranchRule(
            rule_id="db_handler",
            name="Handle database errors",
            condition_type=BranchCondition.ON_CUSTOM,
            condition_config={"condition_func": custom_condition},
            target_node_id="db_fallback",
        )

        db_error = RuntimeError("Database connection failed")
        network_error = RuntimeError("Network timeout")

        assert rule.matches(db_error) is True
        assert rule.matches(network_error) is False

    def test_matches_respects_enabled_flag(self):
        """Test that disabled rules don't match."""
        rule = ErrorBranchRule(
            rule_id="disabled_rule",
            name="Disabled rule",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
            enabled=False,
        )

        error = RuntimeError("Any error")

        assert rule.matches(error) is False

    def test_matches_respects_max_activations(self):
        """Test that rules stop matching after max activations."""
        rule = ErrorBranchRule(
            rule_id="limited_rule",
            name="Limited activations",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
            max_activations=2,
        )

        error = RuntimeError("Error")

        # First two matches should succeed
        assert rule.matches(error) is True
        rule.increment_activation()
        assert rule.matches(error) is True
        rule.increment_activation()

        # Third match should fail (limit reached)
        assert rule.matches(error) is False

    def test_increment_activation(self):
        """Test incrementing activation counter."""
        rule = ErrorBranchRule(
            rule_id="test_rule",
            name="Test",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        assert rule.activation_count == 0

        rule.increment_activation()
        assert rule.activation_count == 1

        rule.increment_activation()
        assert rule.activation_count == 2


class TestErrorBranchManager:
    """Tests for ErrorBranchManager."""

    @pytest.fixture
    def manager(self):
        """Create fresh manager for each test."""
        mgr = ErrorBranchManager()
        yield mgr
        mgr.clear()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager._rules) == 0
        assert len(manager._global_rules) == 0
        assert len(manager._node_rules) == 0

    def test_add_global_rule(self, manager):
        """Test adding global rule."""
        rule = ErrorBranchRule(
            rule_id="global_001",
            name="Global fallback",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback_node",
        )

        manager.add_rule(rule)

        assert rule.rule_id in manager._rules
        assert rule in manager._global_rules

    def test_add_node_specific_rule(self, manager):
        """Test adding node-specific rule."""
        rule = ErrorBranchRule(
            rule_id="node_001",
            name="Node fallback",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback_node",
        )

        manager.add_rule(rule, node_id="test_node")

        assert rule.rule_id in manager._rules
        assert "test_node" in manager._node_rules
        assert rule in manager._node_rules["test_node"]

    def test_add_rule_sorts_by_priority(self, manager):
        """Test that rules are sorted by priority."""
        rule_low = ErrorBranchRule(
            rule_id="low_priority",
            name="Low",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
            priority=10,
        )

        rule_high = ErrorBranchRule(
            rule_id="high_priority",
            name="High",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
            priority=90,
        )

        manager.add_rule(rule_low)
        manager.add_rule(rule_high)

        # Higher priority should be first
        assert manager._global_rules[0].priority == 90
        assert manager._global_rules[1].priority == 10

    def test_remove_rule(self, manager):
        """Test removing a rule."""
        rule = ErrorBranchRule(
            rule_id="test_rule",
            name="Test",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        manager.add_rule(rule)
        assert rule.rule_id in manager._rules

        result = manager.remove_rule(rule.rule_id)

        assert result is True
        assert rule.rule_id not in manager._rules
        assert rule not in manager._global_rules

    def test_remove_nonexistent_rule(self, manager):
        """Test removing non-existent rule returns False."""
        result = manager.remove_rule("nonexistent")
        assert result is False

    def test_evaluate_error_no_rules(self, manager):
        """Test evaluating error with no rules."""
        error = RuntimeError("Test error")

        result = manager.evaluate_error(error)

        assert result.should_branch is False
        assert result.matched_rule is None

    def test_evaluate_error_matches_rule(self, manager):
        """Test evaluating error that matches a rule."""
        rule = ErrorBranchRule(
            rule_id="test_rule",
            name="Catch all",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback_node",
            strategy=BranchStrategy.FALLBACK,
        )

        manager.add_rule(rule)

        error = RuntimeError("Test error")
        result = manager.evaluate_error(error)

        assert result.should_branch is True
        assert result.matched_rule is not None
        assert result.matched_rule.rule_id == "test_rule"
        assert result.target_node_id == "fallback_node"
        assert result.strategy == BranchStrategy.FALLBACK

    def test_evaluate_error_priority_order(self, manager):
        """Test that higher priority rules are evaluated first."""
        rule_low = ErrorBranchRule(
            rule_id="low_priority",
            name="Low",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="low_fallback",
            priority=10,
        )

        rule_high = ErrorBranchRule(
            rule_id="high_priority",
            name="High",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="high_fallback",
            priority=90,
        )

        manager.add_rule(rule_low)
        manager.add_rule(rule_high)

        error = RuntimeError("Test error")
        result = manager.evaluate_error(error)

        # Should match high priority rule
        assert result.matched_rule.rule_id == "high_priority"
        assert result.target_node_id == "high_fallback"

    def test_evaluate_error_node_specific_first(self, manager):
        """Test that node-specific rules are evaluated before global."""
        global_rule = ErrorBranchRule(
            rule_id="global",
            name="Global",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="global_fallback",
            priority=50,
        )

        node_rule = ErrorBranchRule(
            rule_id="node",
            name="Node",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="node_fallback",
            priority=50,  # Same priority
        )

        manager.add_rule(global_rule)
        manager.add_rule(node_rule, node_id="test_node")

        error = RuntimeError("Test error")
        result = manager.evaluate_error(error, node_id="test_node")

        # Should match node-specific rule first
        assert result.matched_rule.rule_id == "node"
        assert result.target_node_id == "node_fallback"

    def test_evaluate_error_increments_activation(self, manager):
        """Test that matched rule increments activation count."""
        rule = ErrorBranchRule(
            rule_id="test_rule",
            name="Test",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        manager.add_rule(rule)

        error = RuntimeError("Test error")
        manager.evaluate_error(error)

        assert rule.activation_count == 1

    def test_get_rule(self, manager):
        """Test getting rule by ID."""
        rule = ErrorBranchRule(
            rule_id="test_rule",
            name="Test",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        manager.add_rule(rule)

        retrieved = manager.get_rule("test_rule")

        assert retrieved is not None
        assert retrieved.rule_id == "test_rule"

    def test_get_rule_not_found(self, manager):
        """Test getting non-existent rule returns None."""
        result = manager.get_rule("nonexistent")
        assert result is None

    def test_get_rules_for_node(self, manager):
        """Test getting all rules for a node."""
        global_rule = ErrorBranchRule(
            rule_id="global",
            name="Global",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        node_rule = ErrorBranchRule(
            rule_id="node",
            name="Node",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        manager.add_rule(global_rule)
        manager.add_rule(node_rule, node_id="test_node")

        rules = manager.get_rules_for_node("test_node")

        # Should include both node-specific and global rules
        assert len(rules) == 2
        rule_ids = {r.rule_id for r in rules}
        assert "global" in rule_ids
        assert "node" in rule_ids

    def test_get_all_rules(self, manager):
        """Test getting all rules."""
        rule1 = ErrorBranchRule(
            rule_id="rule1",
            name="Rule 1",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        rule2 = ErrorBranchRule(
            rule_id="rule2",
            name="Rule 2",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        manager.add_rule(rule1)
        manager.add_rule(rule2, node_id="test_node")

        all_rules = manager.get_all_rules()

        assert len(all_rules) == 2
        rule_ids = {r.rule_id for r in all_rules}
        assert "rule1" in rule_ids
        assert "rule2" in rule_ids

    def test_get_statistics(self, manager):
        """Test getting branching statistics."""
        rule1 = ErrorBranchRule(
            rule_id="rule1",
            name="Rule 1",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
            strategy=BranchStrategy.FALLBACK,
        )

        rule2 = ErrorBranchRule(
            rule_id="rule2",
            name="Rule 2",
            condition_type=BranchCondition.ON_CATEGORY,
            target_node_id="fallback",
            strategy=BranchStrategy.RETRY_WITH_ALTERNATIVE,
            enabled=False,
        )

        manager.add_rule(rule1)
        manager.add_rule(rule2, node_id="test_node")

        # Activate rule1
        error = RuntimeError("Test")
        manager.evaluate_error(error)

        stats = manager.get_statistics()

        assert stats["total_rules"] == 2
        assert stats["enabled_rules"] == 1
        assert stats["global_rules"] == 1
        assert stats["node_rules"] == 1
        assert stats["total_activations"] == 1
        assert "by_strategy" in stats
        assert "by_condition" in stats

    def test_clear(self, manager):
        """Test clearing all rules."""
        rule = ErrorBranchRule(
            rule_id="test_rule",
            name="Test",
            condition_type=BranchCondition.ON_ERROR,
            target_node_id="fallback",
        )

        manager.add_rule(rule)
        assert len(manager._rules) > 0

        manager.clear()

        assert len(manager._rules) == 0
        assert len(manager._global_rules) == 0
        assert len(manager._node_rules) == 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_fallback_rule(self):
        """Test creating fallback rule."""
        rule = create_fallback_rule(
            rule_id="fb_001",
            fallback_node_id="fallback_node",
            priority=60,
        )

        assert rule.rule_id == "fb_001"
        assert rule.target_node_id == "fallback_node"
        assert rule.strategy == BranchStrategy.FALLBACK
        assert rule.condition_type == BranchCondition.ON_ERROR
        assert rule.priority == 60

    def test_create_retry_rule(self):
        """Test creating retry rule."""
        rule = create_retry_rule(
            rule_id="retry_001",
            retry_node_id="retry_node",
            max_retries=5,
            priority=70,
        )

        assert rule.rule_id == "retry_001"
        assert rule.target_node_id == "retry_node"
        assert rule.strategy == BranchStrategy.RETRY_WITH_ALTERNATIVE
        assert rule.condition_type == BranchCondition.ON_RETRYABLE
        assert rule.max_activations == 5
        assert rule.priority == 70

    def test_create_escalation_rule(self):
        """Test creating escalation rule."""
        rule = create_escalation_rule(
            rule_id="esc_001",
            escalation_node_id="escalation_node",
            min_severity=ErrorSeverity.CRITICAL,
            priority=80,
        )

        assert rule.rule_id == "esc_001"
        assert rule.target_node_id == "escalation_node"
        assert rule.strategy == BranchStrategy.ESCALATE
        assert rule.condition_type == BranchCondition.ON_SEVERITY
        assert rule.condition_config["min_severity"] == "critical"
        assert rule.priority == 80

    def test_create_graceful_degradation_rule(self):
        """Test creating graceful degradation rule."""
        rule = create_graceful_degradation_rule(
            rule_id="deg_001",
            degraded_node_id="degraded_node",
            priority=40,
        )

        assert rule.rule_id == "deg_001"
        assert rule.target_node_id == "degraded_node"
        assert rule.strategy == BranchStrategy.GRACEFUL_DEGRADATION
        assert rule.condition_type == BranchCondition.ON_ERROR
        assert rule.priority == 40


class TestGlobalBranchManager:
    """Tests for global branch manager instance."""

    def test_get_global_manager(self):
        """Test getting global manager singleton."""
        mgr1 = get_branch_manager()
        mgr2 = get_branch_manager()

        assert mgr1 is mgr2  # Same instance
