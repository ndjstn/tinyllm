"""Error-triggered graph branching for TinyLLM.

This module enables dynamic graph branching based on error conditions,
allowing workflows to take alternative paths when errors occur.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.errors import ErrorCategory, ErrorSeverity, TinyLLMError
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="error_branching")


class BranchCondition(str, Enum):
    """Error branch condition types."""

    ON_ERROR = "on_error"  # Any error
    ON_CATEGORY = "on_category"  # Specific error category
    ON_SEVERITY = "on_severity"  # Minimum severity level
    ON_RETRYABLE = "on_retryable"  # Only retryable errors
    ON_FATAL = "on_fatal"  # Only fatal errors
    ON_CUSTOM = "on_custom"  # Custom condition function


class BranchStrategy(str, Enum):
    """Branch execution strategy."""

    FALLBACK = "fallback"  # Execute fallback path
    RETRY_WITH_ALTERNATIVE = "retry_with_alternative"  # Retry with different config
    ESCALATE = "escalate"  # Escalate to higher-capacity node
    PARALLEL_RECOVERY = "parallel_recovery"  # Try multiple recovery paths
    CIRCUIT_BREAK = "circuit_break"  # Open circuit and fail fast
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Return partial result


class ErrorBranchRule(BaseModel):
    """Rule for error-triggered branching.

    Defines conditions under which to branch and the target path.
    """

    model_config = {"extra": "forbid"}

    # Rule identification
    rule_id: str = Field(description="Unique rule identifier")
    name: str = Field(description="Human-readable rule name")
    enabled: bool = Field(default=True, description="Whether rule is active")

    # Condition
    condition_type: BranchCondition = Field(description="Branch condition type")
    condition_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Condition-specific configuration"
    )

    # Target
    target_node_id: str = Field(description="Node to branch to on error")
    strategy: BranchStrategy = Field(
        default=BranchStrategy.FALLBACK,
        description="Branch strategy"
    )

    # Metadata
    priority: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Rule priority (higher = evaluated first)"
    )
    max_activations: Optional[int] = Field(
        default=None,
        description="Maximum times this rule can activate"
    )
    activation_count: int = Field(
        default=0,
        ge=0,
        description="Number of times rule has activated"
    )

    # Recovery configuration
    recovery_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for recovery node"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def matches(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if error matches this rule's condition.

        Args:
            error: Exception to check.
            context: Optional execution context.

        Returns:
            True if error matches condition.
        """
        # Check if rule is enabled
        if not self.enabled:
            return False

        # Check max activations
        if self.max_activations and self.activation_count >= self.max_activations:
            return False

        # Evaluate condition
        if self.condition_type == BranchCondition.ON_ERROR:
            # Match any error
            return True

        elif self.condition_type == BranchCondition.ON_CATEGORY:
            # Match specific category
            if isinstance(error, TinyLLMError):
                category_str = self.condition_config.get("category", "")
                try:
                    target_category = ErrorCategory(category_str)
                    return error.category == target_category
                except ValueError:
                    logger.warning(
                        "invalid_category_in_rule",
                        rule_id=self.rule_id,
                        category=category_str,
                    )
                    return False
            return False

        elif self.condition_type == BranchCondition.ON_SEVERITY:
            # Match minimum severity
            if isinstance(error, TinyLLMError):
                severity_str = self.condition_config.get("min_severity", "error")
                try:
                    min_severity = ErrorSeverity(severity_str)
                    severity_values = {
                        ErrorSeverity.DEBUG: 0,
                        ErrorSeverity.INFO: 1,
                        ErrorSeverity.WARNING: 2,
                        ErrorSeverity.ERROR: 3,
                        ErrorSeverity.CRITICAL: 4,
                        ErrorSeverity.FATAL: 5,
                    }
                    return severity_values.get(error.severity, 0) >= severity_values.get(min_severity, 3)
                except ValueError:
                    return False
            return False

        elif self.condition_type == BranchCondition.ON_RETRYABLE:
            # Match retryable errors
            if isinstance(error, TinyLLMError):
                return error.recoverable
            # Check message patterns for non-TinyLLM errors
            error_str = str(error).lower()
            retryable_patterns = ["timeout", "connection", "rate limit", "network"]
            return any(pattern in error_str for pattern in retryable_patterns)

        elif self.condition_type == BranchCondition.ON_FATAL:
            # Match fatal errors
            if isinstance(error, TinyLLMError):
                return not error.recoverable
            return True  # Default to fatal for unknown errors

        elif self.condition_type == BranchCondition.ON_CUSTOM:
            # Custom condition function
            custom_func = self.condition_config.get("condition_func")
            if custom_func and callable(custom_func):
                try:
                    return custom_func(error, context)
                except Exception as e:
                    logger.error(
                        "custom_condition_error",
                        rule_id=self.rule_id,
                        error=str(e),
                    )
                    return False
            return False

        return False

    def increment_activation(self) -> None:
        """Increment activation counter."""
        self.activation_count += 1


class ErrorBranchResult(BaseModel):
    """Result of error branch evaluation."""

    model_config = {"extra": "forbid"}

    should_branch: bool = Field(description="Whether to branch")
    matched_rule: Optional[ErrorBranchRule] = Field(
        default=None,
        description="Rule that was matched"
    )
    target_node_id: Optional[str] = Field(
        default=None,
        description="Target node to branch to"
    )
    strategy: Optional[BranchStrategy] = Field(
        default=None,
        description="Branch strategy to use"
    )
    recovery_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Recovery configuration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ErrorBranchManager:
    """Manage error-triggered graph branching.

    Evaluates errors against configured rules and determines
    appropriate branching behavior.
    """

    def __init__(self):
        """Initialize error branch manager."""
        self._rules: Dict[str, ErrorBranchRule] = {}
        self._global_rules: List[ErrorBranchRule] = []
        self._node_rules: Dict[str, List[ErrorBranchRule]] = {}

        logger.info("error_branch_manager_initialized")

    def add_rule(
        self,
        rule: ErrorBranchRule,
        node_id: Optional[str] = None,
    ) -> None:
        """Add an error branching rule.

        Args:
            rule: Error branch rule to add.
            node_id: Optional node ID to scope rule to.
        """
        self._rules[rule.rule_id] = rule

        if node_id:
            # Node-specific rule
            if node_id not in self._node_rules:
                self._node_rules[node_id] = []
            self._node_rules[node_id].append(rule)

            logger.info(
                "node_error_rule_added",
                rule_id=rule.rule_id,
                node_id=node_id,
                condition=rule.condition_type.value,
                target=rule.target_node_id,
            )
        else:
            # Global rule
            self._global_rules.append(rule)

            logger.info(
                "global_error_rule_added",
                rule_id=rule.rule_id,
                condition=rule.condition_type.value,
                target=rule.target_node_id,
            )

        # Sort by priority
        self._global_rules.sort(key=lambda r: r.priority, reverse=True)
        for rules in self._node_rules.values():
            rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an error branching rule.

        Args:
            rule_id: Rule ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if rule_id not in self._rules:
            return False

        rule = self._rules[rule_id]
        del self._rules[rule_id]

        # Remove from global rules
        self._global_rules = [r for r in self._global_rules if r.rule_id != rule_id]

        # Remove from node rules
        for node_id in list(self._node_rules.keys()):
            self._node_rules[node_id] = [
                r for r in self._node_rules[node_id]
                if r.rule_id != rule_id
            ]
            if not self._node_rules[node_id]:
                del self._node_rules[node_id]

        logger.info("error_rule_removed", rule_id=rule_id)
        return True

    def evaluate_error(
        self,
        error: Exception,
        node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorBranchResult:
        """Evaluate error against branching rules.

        Args:
            error: Exception to evaluate.
            node_id: Optional current node ID.
            context: Optional execution context.

        Returns:
            ErrorBranchResult with branching decision.
        """
        # Collect applicable rules
        applicable_rules: List[ErrorBranchRule] = []

        # Add node-specific rules first
        if node_id and node_id in self._node_rules:
            applicable_rules.extend(self._node_rules[node_id])

        # Add global rules
        applicable_rules.extend(self._global_rules)

        # Evaluate rules in priority order
        for rule in applicable_rules:
            if rule.matches(error, context):
                # Found a matching rule
                rule.increment_activation()

                logger.info(
                    "error_branch_matched",
                    rule_id=rule.rule_id,
                    node_id=node_id,
                    target_node=rule.target_node_id,
                    strategy=rule.strategy.value,
                )

                return ErrorBranchResult(
                    should_branch=True,
                    matched_rule=rule,
                    target_node_id=rule.target_node_id,
                    strategy=rule.strategy,
                    recovery_config=rule.recovery_config,
                    metadata={
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "activation_count": rule.activation_count,
                    },
                )

        # No matching rule
        logger.debug(
            "error_branch_no_match",
            node_id=node_id,
            error_type=type(error).__name__,
        )

        return ErrorBranchResult(
            should_branch=False,
        )

    def get_rule(self, rule_id: str) -> Optional[ErrorBranchRule]:
        """Get a rule by ID.

        Args:
            rule_id: Rule ID to retrieve.

        Returns:
            ErrorBranchRule or None if not found.
        """
        return self._rules.get(rule_id)

    def get_rules_for_node(self, node_id: str) -> List[ErrorBranchRule]:
        """Get all rules for a specific node.

        Args:
            node_id: Node ID.

        Returns:
            List of applicable rules.
        """
        rules = []

        # Node-specific rules
        if node_id in self._node_rules:
            rules.extend(self._node_rules[node_id])

        # Global rules
        rules.extend(self._global_rules)

        # Sort by priority
        rules.sort(key=lambda r: r.priority, reverse=True)

        return rules

    def get_all_rules(self) -> List[ErrorBranchRule]:
        """Get all rules.

        Returns:
            List of all rules.
        """
        return list(self._rules.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get branching statistics.

        Returns:
            Dictionary of statistics.
        """
        total_rules = len(self._rules)
        enabled_rules = sum(1 for r in self._rules.values() if r.enabled)
        total_activations = sum(r.activation_count for r in self._rules.values())

        # Group by strategy
        by_strategy = {}
        for rule in self._rules.values():
            strategy = rule.strategy.value
            if strategy not in by_strategy:
                by_strategy[strategy] = 0
            by_strategy[strategy] += 1

        # Group by condition
        by_condition = {}
        for rule in self._rules.values():
            condition = rule.condition_type.value
            if condition not in by_condition:
                by_condition[condition] = 0
            by_condition[condition] += 1

        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "global_rules": len(self._global_rules),
            "node_rules": sum(len(rules) for rules in self._node_rules.values()),
            "total_activations": total_activations,
            "by_strategy": by_strategy,
            "by_condition": by_condition,
        }

    def clear(self) -> None:
        """Clear all rules."""
        self._rules.clear()
        self._global_rules.clear()
        self._node_rules.clear()

        logger.info("error_branch_rules_cleared")


# Global branch manager instance
_branch_manager: Optional[ErrorBranchManager] = None


def get_branch_manager() -> ErrorBranchManager:
    """Get global error branch manager instance.

    Returns:
        Global ErrorBranchManager instance.
    """
    global _branch_manager
    if _branch_manager is None:
        _branch_manager = ErrorBranchManager()
    return _branch_manager


# Helper functions for creating common rules

def create_fallback_rule(
    rule_id: str,
    fallback_node_id: str,
    condition_type: BranchCondition = BranchCondition.ON_ERROR,
    priority: int = 50,
) -> ErrorBranchRule:
    """Create a simple fallback rule.

    Args:
        rule_id: Rule identifier.
        fallback_node_id: Fallback node ID.
        condition_type: Condition type.
        priority: Rule priority.

    Returns:
        ErrorBranchRule configured for fallback.
    """
    return ErrorBranchRule(
        rule_id=rule_id,
        name=f"Fallback to {fallback_node_id}",
        condition_type=condition_type,
        target_node_id=fallback_node_id,
        strategy=BranchStrategy.FALLBACK,
        priority=priority,
    )


def create_retry_rule(
    rule_id: str,
    retry_node_id: str,
    max_retries: int = 3,
    priority: int = 60,
) -> ErrorBranchRule:
    """Create a retry with alternative rule.

    Args:
        rule_id: Rule identifier.
        retry_node_id: Alternative node ID.
        max_retries: Maximum retry attempts.
        priority: Rule priority.

    Returns:
        ErrorBranchRule configured for retry.
    """
    return ErrorBranchRule(
        rule_id=rule_id,
        name=f"Retry with {retry_node_id}",
        condition_type=BranchCondition.ON_RETRYABLE,
        target_node_id=retry_node_id,
        strategy=BranchStrategy.RETRY_WITH_ALTERNATIVE,
        priority=priority,
        max_activations=max_retries,
    )


def create_escalation_rule(
    rule_id: str,
    escalation_node_id: str,
    min_severity: ErrorSeverity = ErrorSeverity.ERROR,
    priority: int = 70,
) -> ErrorBranchRule:
    """Create an escalation rule.

    Args:
        rule_id: Rule identifier.
        escalation_node_id: Escalation node ID.
        min_severity: Minimum severity to escalate.
        priority: Rule priority.

    Returns:
        ErrorBranchRule configured for escalation.
    """
    return ErrorBranchRule(
        rule_id=rule_id,
        name=f"Escalate to {escalation_node_id}",
        condition_type=BranchCondition.ON_SEVERITY,
        condition_config={"min_severity": min_severity.value},
        target_node_id=escalation_node_id,
        strategy=BranchStrategy.ESCALATE,
        priority=priority,
    )


def create_graceful_degradation_rule(
    rule_id: str,
    degraded_node_id: str,
    priority: int = 40,
) -> ErrorBranchRule:
    """Create a graceful degradation rule.

    Args:
        rule_id: Rule identifier.
        degraded_node_id: Degraded mode node ID.
        priority: Rule priority.

    Returns:
        ErrorBranchRule configured for degradation.
    """
    return ErrorBranchRule(
        rule_id=rule_id,
        name=f"Degrade to {degraded_node_id}",
        condition_type=BranchCondition.ON_ERROR,
        target_node_id=degraded_node_id,
        strategy=BranchStrategy.GRACEFUL_DEGRADATION,
        priority=priority,
    )
