"""Tests for execution budget limits."""

import time

import pytest

from tinyllm.core.budget import (
    BudgetExceededError,
    BudgetManager,
    BudgetStatus,
    BudgetType,
    ExecutionBudget,
)


def test_budget_basic():
    """Test basic budget tracking."""
    budget = ExecutionBudget(
        max_tokens=1000,
        max_cost_usd=1.0,
        max_operations=10,
    )
    manager = BudgetManager(budget)

    # Track an operation
    manager.track_operation(tokens_in=100, tokens_out=50, cost_usd=0.05)

    assert manager.usage.tokens == 150
    assert manager.usage.cost_usd == 0.05
    assert manager.usage.operations == 1


def test_budget_status():
    """Test budget status checking."""
    budget = ExecutionBudget(max_tokens=100)
    manager = BudgetManager(budget)

    # Initially OK
    assert manager.check_budget(BudgetType.TOKENS) == BudgetStatus.OK

    # Use 85 tokens (85% - warning threshold)
    manager.track_operation(tokens_in=85)
    assert manager.check_budget(BudgetType.TOKENS) == BudgetStatus.WARNING

    # Use more to exceed
    manager.track_operation(tokens_in=20)
    assert manager.check_budget(BudgetType.TOKENS) == BudgetStatus.EXCEEDED


def test_budget_is_exceeded():
    """Test budget exceeded check."""
    budget = ExecutionBudget(max_tokens=100)
    manager = BudgetManager(budget)

    assert not manager.is_exceeded(BudgetType.TOKENS)

    manager.track_operation(tokens_in=99)
    assert not manager.is_exceeded(BudgetType.TOKENS)  # Just under limit

    manager.track_operation(tokens_in=2)
    assert manager.is_exceeded(BudgetType.TOKENS)


def test_budget_can_proceed():
    """Test can_proceed check."""
    budget = ExecutionBudget(max_tokens=100, max_cost_usd=1.0)
    manager = BudgetManager(budget)

    # Can proceed with small operation
    assert manager.can_proceed(required_tokens=50, required_cost=0.25)

    # Use some budget
    manager.track_operation(tokens_in=60, cost_usd=0.6)

    # Can't proceed with large operation
    assert not manager.can_proceed(required_tokens=50)
    assert not manager.can_proceed(required_cost=0.5)

    # Can proceed with small operation
    assert manager.can_proceed(required_tokens=30, required_cost=0.3)


def test_budget_remaining():
    """Test remaining budget calculation."""
    budget = ExecutionBudget(max_tokens=1000)
    manager = BudgetManager(budget)

    assert manager.get_remaining(BudgetType.TOKENS) == 1000

    manager.track_operation(tokens_in=300)
    assert manager.get_remaining(BudgetType.TOKENS) == 700


def test_budget_utilization():
    """Test utilization calculation."""
    budget = ExecutionBudget(max_tokens=1000)
    manager = BudgetManager(budget)

    assert manager.get_utilization(BudgetType.TOKENS) == 0.0

    manager.track_operation(tokens_in=500)
    assert manager.get_utilization(BudgetType.TOKENS) == 0.5

    manager.track_operation(tokens_in=250)
    assert manager.get_utilization(BudgetType.TOKENS) == 0.75


def test_budget_time_tracking():
    """Test time budget tracking."""
    budget = ExecutionBudget(max_time_seconds=1.0)
    manager = BudgetManager(budget)

    # Should be OK initially
    assert manager.check_budget(BudgetType.TIME) == BudgetStatus.OK

    # Wait to consume time budget
    time.sleep(0.5)
    manager.track_operation()  # Update time

    # Should be at ~50% utilization
    utilization = manager.get_utilization(BudgetType.TIME)
    assert utilization is not None
    assert 0.4 < utilization < 0.6


def test_budget_cost_tracking():
    """Test cost budget tracking."""
    budget = ExecutionBudget(max_cost_usd=10.0)
    manager = BudgetManager(budget)

    manager.track_operation(cost_usd=2.5)
    assert manager.usage.cost_usd == 2.5
    assert manager.get_utilization(BudgetType.COST) == 0.25

    manager.track_operation(cost_usd=3.0)
    assert manager.usage.cost_usd == 5.5


def test_budget_operations_tracking():
    """Test operations budget tracking."""
    budget = ExecutionBudget(max_operations=5)
    manager = BudgetManager(budget)

    for i in range(4):
        manager.track_operation()
        assert manager.usage.operations == i + 1

    assert manager.check_budget(BudgetType.OPERATIONS) == BudgetStatus.WARNING


def test_budget_memory_tracking():
    """Test memory budget tracking."""
    budget = ExecutionBudget(max_memory_bytes=1024 * 1024)  # 1 MB
    manager = BudgetManager(budget)

    manager.track_operation(memory_bytes=512 * 1024)  # 512 KB
    assert manager.usage.memory_bytes == 512 * 1024

    # Memory tracking keeps max
    manager.track_operation(memory_bytes=256 * 1024)  # 256 KB
    assert manager.usage.memory_bytes == 512 * 1024  # Still at max


def test_budget_no_limit():
    """Test budget with no limit set."""
    budget = ExecutionBudget()  # No limits
    manager = BudgetManager(budget)

    # Should always be OK
    manager.track_operation(tokens_in=999999)
    assert manager.check_budget() == BudgetStatus.OK


def test_budget_summary():
    """Test budget summary."""
    budget = ExecutionBudget(
        max_tokens=1000,
        max_cost_usd=10.0,
        max_operations=100,
    )
    manager = BudgetManager(budget)

    manager.track_operation(tokens_in=500, cost_usd=2.5)

    summary = manager.get_summary()
    assert summary["usage"]["tokens"] == 500
    assert summary["usage"]["cost_usd"] == 2.5
    assert summary["usage"]["operations"] == 1
    assert summary["limits"]["tokens"] == 1000
    assert summary["status"]["overall"] in [s.value for s in BudgetStatus]


def test_budget_reset():
    """Test budget reset."""
    budget = ExecutionBudget(max_tokens=1000)
    manager = BudgetManager(budget)

    manager.track_operation(tokens_in=500)
    assert manager.usage.tokens == 500

    manager.reset()
    assert manager.usage.tokens == 0
    assert manager.usage.operations == 0


def test_budget_warning_threshold():
    """Test custom warning threshold."""
    budget = ExecutionBudget(max_tokens=100, warning_threshold=0.5)
    manager = BudgetManager(budget)

    # Use 51% - should trigger warning
    manager.track_operation(tokens_in=51)
    assert manager.check_budget(BudgetType.TOKENS) == BudgetStatus.WARNING


def test_budget_exceeded_error():
    """Test BudgetExceededError exception."""
    error = BudgetExceededError(
        budget_type=BudgetType.TOKENS,
        usage=1000,
        limit=500,
    )

    assert error.budget_type == BudgetType.TOKENS
    assert error.usage == 1000
    assert error.limit == 500
    assert "tokens" in str(error).lower()


def test_budget_multiple_types():
    """Test checking multiple budget types."""
    budget = ExecutionBudget(
        max_tokens=100,
        max_cost_usd=1.0,
        max_operations=10,
    )
    manager = BudgetManager(budget)

    # Exceed tokens
    manager.track_operation(tokens_in=110)

    # Overall status should be EXCEEDED
    assert manager.check_budget() == BudgetStatus.EXCEEDED

    # Individual checks
    assert manager.check_budget(BudgetType.TOKENS) == BudgetStatus.EXCEEDED
    assert manager.check_budget(BudgetType.COST) == BudgetStatus.OK


def test_budget_remaining_no_limit():
    """Test remaining calculation with no limit."""
    budget = ExecutionBudget()  # No limits
    manager = BudgetManager(budget)

    assert manager.get_remaining(BudgetType.TOKENS) is None


def test_budget_utilization_no_limit():
    """Test utilization calculation with no limit."""
    budget = ExecutionBudget()  # No limits
    manager = BudgetManager(budget)

    assert manager.get_utilization(BudgetType.TOKENS) is None


def test_budget_can_proceed_no_limit():
    """Test can_proceed with no limits."""
    budget = ExecutionBudget()  # No limits
    manager = BudgetManager(budget)

    # Should always be able to proceed
    assert manager.can_proceed(required_tokens=999999)


def test_budget_mixed_usage():
    """Test mixed usage across different budget types."""
    budget = ExecutionBudget(
        max_tokens=1000,
        max_cost_usd=5.0,
        max_operations=20,
    )
    manager = BudgetManager(budget)

    # Track multiple operations
    manager.track_operation(tokens_in=100, tokens_out=50, cost_usd=0.5)
    manager.track_operation(tokens_in=200, tokens_out=100, cost_usd=1.0)
    manager.track_operation(tokens_in=150, cost_usd=0.75)

    assert manager.usage.tokens == 600
    assert manager.usage.cost_usd == 2.25
    assert manager.usage.operations == 3

    # Check individual statuses
    assert manager.check_budget(BudgetType.TOKENS) == BudgetStatus.OK
    assert manager.check_budget(BudgetType.COST) == BudgetStatus.OK
    assert manager.check_budget(BudgetType.OPERATIONS) == BudgetStatus.OK
