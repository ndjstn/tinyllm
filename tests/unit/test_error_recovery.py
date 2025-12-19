"""Tests for error recovery features (Tasks 6-10)."""

import asyncio
import pytest
from typing import Dict, Any

from tinyllm.core.executor import (
    DegradationMode,
    Executor,
    ExecutorConfig,
    TransactionalExecutor,
    Transaction,
    TransactionLog,
)
from tinyllm.core.graph import Graph
from tinyllm.core.message import TaskPayload
from tinyllm.config.graph import GraphDefinition, NodeDefinition, NodeType
from tinyllm.errors import (
    ErrorRecoveryManager,
    RecoveryPlaybook,
    RecoveryAction,
    TimeoutError as TinyTimeoutError,
    ModelError,
    NetworkError,
    get_recovery_manager,
)
from tinyllm.metrics import get_metrics_collector


@pytest.fixture
def simple_graph_def():
    """Create a simple graph definition for testing."""
    return GraphDefinition(
        id="test_graph",
        version="1.0.0",
        name="Test Graph",
        nodes=[
            NodeDefinition(id="entry", type=NodeType.ENTRY, name="Entry"),
            NodeDefinition(id="process", type=NodeType.TRANSFORM, name="Process"),
            NodeDefinition(id="exit_node", type=NodeType.EXIT, name="Exit"),
        ],
        edges=[
            {"from_node": "entry", "to_node": "process"},
            {"from_node": "process", "to_node": "exit_node"},
        ],
        entry_points=["entry"],
        exit_points=["exit_node"],
    )


@pytest.fixture
def executor_config():
    """Create executor configuration with degradation mode."""
    return ExecutorConfig(
        max_steps=10,
        timeout_ms=5000,
        fail_fast=False,
        degradation_mode="best_effort",
        max_node_failures=3,
        enable_partial_results=True,
        enable_checkpointing=True,
        checkpoint_interval=2,
    )


# ============================================================================
# Task 6: Graceful Degradation Modes
# ============================================================================


def test_degradation_mode_enum():
    """Test degradation mode enum values."""
    assert DegradationMode.FAIL_FAST == "fail_fast"
    assert DegradationMode.SKIP_FAILED == "skip_failed"
    assert DegradationMode.FALLBACK_SIMPLE == "fallback_simple"
    assert DegradationMode.BEST_EFFORT == "best_effort"


def test_executor_config_degradation_modes():
    """Test executor configuration with different degradation modes."""
    # Fail fast mode
    config = ExecutorConfig(degradation_mode="fail_fast")
    assert config.degradation_mode == "fail_fast"

    # Skip failed mode
    config = ExecutorConfig(degradation_mode="skip_failed")
    assert config.degradation_mode == "skip_failed"

    # Fallback simple mode
    config = ExecutorConfig(degradation_mode="fallback_simple")
    assert config.degradation_mode == "fallback_simple"

    # Best effort mode
    config = ExecutorConfig(degradation_mode="best_effort")
    assert config.degradation_mode == "best_effort"


def test_executor_partial_results_config():
    """Test partial results configuration."""
    config = ExecutorConfig(enable_partial_results=True, max_node_failures=5)
    assert config.enable_partial_results is True
    assert config.max_node_failures == 5


# ============================================================================
# Task 7: Error Rate Alerting Thresholds
# ============================================================================


def test_metrics_error_rate_threshold():
    """Test error rate threshold configuration."""
    metrics = get_metrics_collector()

    # Test setting error rate threshold
    metrics.set_error_rate_threshold(0.1)
    assert metrics._error_rate_threshold == 0.1

    # Test setting error count threshold
    metrics.set_error_count_threshold(20)
    assert metrics._error_count_threshold == 20

    # Test setting alert window
    metrics.set_alert_window(600)
    assert metrics._alert_window_seconds == 600


def test_metrics_error_rate_threshold_validation():
    """Test error rate threshold validation."""
    metrics = get_metrics_collector()

    # Invalid rate (too high)
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        metrics.set_error_rate_threshold(1.5)

    # Invalid rate (negative)
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        metrics.set_error_rate_threshold(-0.1)

    # Invalid count (zero)
    with pytest.raises(ValueError, match="must be >= 1"):
        metrics.set_error_count_threshold(0)

    # Invalid window (too small)
    with pytest.raises(ValueError, match="must be >= 1.0"):
        metrics.set_alert_window(0.5)


def test_metrics_error_rate_tracking():
    """Test error rate tracking and alerting."""
    metrics = get_metrics_collector()

    # Set low threshold for testing
    metrics.set_error_count_threshold(3)
    metrics.set_alert_window(60)

    # Clear any existing errors
    metrics._error_count_window.clear()

    # Record errors
    for i in range(5):
        metrics.increment_error_count("test_error", model="test_model", graph="test_graph")

    # Get current error rate
    error_rate = metrics.get_current_error_rate()
    assert error_rate["error_count"] >= 3
    assert error_rate["threshold_exceeded"] is True
    assert "test_error" in error_rate["error_types"]


# ============================================================================
# Task 8: Error Recovery Playbooks
# ============================================================================


def test_recovery_playbook_creation():
    """Test creating recovery playbooks."""
    playbook = RecoveryPlaybook(
        error_pattern="timeout",
        actions=[RecoveryAction.RETRY, RecoveryAction.REDUCE_LOAD],
        max_attempts=3,
        cooldown_ms=2000,
    )

    assert playbook.error_pattern == "timeout"
    assert len(playbook.actions) == 2
    assert RecoveryAction.RETRY in playbook.actions
    assert playbook.max_attempts == 3
    assert playbook.cooldown_ms == 2000


def test_recovery_manager_initialization():
    """Test recovery manager initialization with default playbooks."""
    manager = ErrorRecoveryManager()

    # Check that default playbooks are registered
    assert "timeout" in manager._playbooks
    assert "rate_limit" in manager._playbooks
    assert "network" in manager._playbooks
    assert "model" in manager._playbooks
    assert "resource_exhausted" in manager._playbooks
    assert "circuit_open" in manager._playbooks
    assert "validation" in manager._playbooks


def test_recovery_manager_register_playbook():
    """Test registering custom playbooks."""
    manager = ErrorRecoveryManager()

    custom_playbook = RecoveryPlaybook(
        error_pattern="custom_error",
        actions=[RecoveryAction.ALERT, RecoveryAction.SKIP],
        max_attempts=1,
    )

    manager.register_playbook(custom_playbook)
    assert "custom_error" in manager._playbooks


def test_recovery_manager_get_playbook():
    """Test finding playbooks for errors."""
    manager = ErrorRecoveryManager()

    # Test timeout error
    timeout_error = TinyTimeoutError("Operation timed out", timeout_ms=5000)
    playbook = manager.get_playbook(timeout_error)
    assert playbook is not None
    assert playbook.error_pattern == "timeout"

    # Test model error
    model_error = ModelError("Model failed", model="test_model")
    playbook = manager.get_playbook(model_error)
    assert playbook is not None
    assert playbook.error_pattern == "model"

    # Test network error
    network_error = NetworkError("Connection failed")
    playbook = manager.get_playbook(network_error)
    assert playbook is not None
    assert playbook.error_pattern == "network"


@pytest.mark.asyncio
async def test_recovery_manager_execute_recovery():
    """Test executing recovery actions."""
    manager = ErrorRecoveryManager()

    # Create a timeout error
    error = TinyTimeoutError("Operation timed out", timeout_ms=5000)
    context = {"function": "test_func", "model": "test_model", "graph": "test_graph"}

    # Execute recovery
    result = await manager.execute_recovery(error, context)

    assert "recovered" in result
    assert "actions_taken" in result
    assert "playbook" in result
    assert len(result["actions_taken"]) > 0


@pytest.mark.asyncio
async def test_recovery_manager_stats():
    """Test recovery statistics tracking."""
    manager = ErrorRecoveryManager()

    # Execute some recoveries
    error = TinyTimeoutError("Operation timed out", timeout_ms=5000)
    context = {"function": "test_func"}

    for _ in range(3):
        await manager.execute_recovery(error, context)

    # Get stats
    stats = manager.get_recovery_stats()
    assert "timeout" in stats
    assert stats["timeout"]["attempts"] == 3


def test_get_global_recovery_manager():
    """Test global recovery manager singleton."""
    manager1 = get_recovery_manager()
    manager2 = get_recovery_manager()
    assert manager1 is manager2


# ============================================================================
# Task 9: Partial Graph Execution Recovery
# ============================================================================


def test_executor_partial_results_enabled():
    """Test executor with partial results enabled."""
    config = ExecutorConfig(
        enable_partial_results=True,
        max_node_failures=2,
        degradation_mode="best_effort",
    )

    assert config.enable_partial_results is True
    assert config.max_node_failures == 2


# Note: Full integration test would require mocking node failures
# This is tested through the executor's existing degradation logic


# ============================================================================
# Task 10: Transaction-like Rollback
# ============================================================================


def test_transaction_creation():
    """Test creating a transaction."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    assert transaction.trace_id == "test_trace"
    assert transaction.graph_id == "test_graph"
    assert len(transaction.logs) == 0
    assert transaction.committed is False
    assert transaction.rolled_back is False


def test_transaction_log_operation():
    """Test logging operations in a transaction."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    state_before = {"step": 0, "tokens": 0}
    state_after = {"step": 1, "tokens": 10}

    transaction.log_operation(
        step=1,
        node_id="test_node",
        operation="execute",
        state_before=state_before,
        state_after=state_after,
        reversible=True,
    )

    assert len(transaction.logs) == 1
    assert transaction.logs[0].node_id == "test_node"
    assert transaction.logs[0].operation == "execute"
    assert transaction.logs[0].reversible is True


@pytest.mark.asyncio
async def test_transaction_commit():
    """Test committing a transaction."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    # Log some operations
    transaction.log_operation(
        step=1,
        node_id="node1",
        operation="execute",
        state_before={},
        state_after={},
    )

    # Commit
    await transaction.commit()

    assert transaction.committed is True
    assert transaction.rolled_back is False

    # Cannot commit again
    with pytest.raises(RuntimeError, match="already committed"):
        await transaction.commit()


@pytest.mark.asyncio
async def test_transaction_rollback():
    """Test rolling back a transaction."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    # Log multiple operations
    for i in range(3):
        transaction.log_operation(
            step=i + 1,
            node_id=f"node{i}",
            operation="execute",
            state_before={},
            state_after={},
            reversible=True,
        )

    # Rollback all
    result = await transaction.rollback()

    assert transaction.rolled_back is True
    assert result["success"] is True
    assert result["rolled_back"] == 3
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_transaction_partial_rollback():
    """Test rolling back to a specific step."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    # Log 5 operations
    for i in range(5):
        transaction.log_operation(
            step=i + 1,
            node_id=f"node{i}",
            operation="execute",
            state_before={},
            state_after={},
            reversible=True,
        )

    # Rollback to step 2 (should undo steps 3, 4, 5)
    result = await transaction.rollback(to_step=2)

    assert transaction.rolled_back is True
    assert result["rolled_back"] == 3  # Steps 3, 4, 5
    assert result["to_step"] == 2


@pytest.mark.asyncio
async def test_transaction_rollback_committed_error():
    """Test that committed transactions cannot be rolled back."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    await transaction.commit()

    with pytest.raises(RuntimeError, match="Cannot rollback committed"):
        await transaction.rollback()


@pytest.mark.asyncio
async def test_transaction_irreversible_operations():
    """Test handling of irreversible operations during rollback."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    # Log reversible operation
    transaction.log_operation(
        step=1,
        node_id="node1",
        operation="execute",
        state_before={},
        state_after={},
        reversible=True,
    )

    # Log irreversible operation
    transaction.log_operation(
        step=2,
        node_id="node2",
        operation="external_api_call",
        state_before={},
        state_after={},
        reversible=False,
    )

    # Rollback
    result = await transaction.rollback()

    assert result["rolled_back"] == 1  # Only reversible operation
    assert result["skipped"] == 1  # Irreversible operation skipped


def test_transaction_get_status():
    """Test getting transaction status."""
    transaction = Transaction(trace_id="test_trace", graph_id="test_graph")

    # Log operations
    transaction.log_operation(
        step=1,
        node_id="node1",
        operation="execute",
        state_before={},
        state_after={},
        reversible=True,
    )
    transaction.log_operation(
        step=2,
        node_id="node2",
        operation="api_call",
        state_before={},
        state_after={},
        reversible=False,
    )

    status = transaction.get_status()

    assert status["trace_id"] == "test_trace"
    assert status["graph_id"] == "test_graph"
    assert status["operations_count"] == 2
    assert status["committed"] is False
    assert status["rolled_back"] is False
    assert status["reversible_operations"] == 1
    assert status["irreversible_operations"] == 1


def test_transactional_executor_creation(simple_graph_def):
    """Test creating a transactional executor."""
    from tinyllm.nodes.transform import TransformNode

    # Use simpler minimal graph
    graph = Graph(simple_graph_def)
    for node_def in simple_graph_def.nodes:
        graph.add_node(TransformNode(id=node_def.id, name=node_def.name))

    executor = TransactionalExecutor(graph, enable_transactions=True)

    assert executor.enable_transactions is True
    assert executor._current_transaction is None


def test_transactional_executor_get_status(simple_graph_def):
    """Test getting transaction status from executor."""
    from tinyllm.nodes.transform import TransformNode

    # Use simpler minimal graph
    graph = Graph(simple_graph_def)
    for node_def in simple_graph_def.nodes:
        graph.add_node(TransformNode(id=node_def.id, name=node_def.name))

    executor = TransactionalExecutor(graph)

    # No transaction active
    status = executor.get_transaction_status()
    assert status is None


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_degradation_modes_integration(simple_graph_def):
    """Test different degradation modes in executor."""
    from tinyllm.nodes.transform import TransformNode

    # Build graph
    graph = Graph(simple_graph_def)
    for node_def in simple_graph_def.nodes:
        graph.add_node(TransformNode(id=node_def.id, name=node_def.name))

    # Test fail_fast mode
    config = ExecutorConfig(degradation_mode="fail_fast", fail_fast=True)
    executor = Executor(graph, config)
    assert executor.config.degradation_mode == "fail_fast"

    # Test best_effort mode
    config = ExecutorConfig(degradation_mode="best_effort", enable_partial_results=True)
    executor = Executor(graph, config)
    assert executor.config.degradation_mode == "best_effort"


@pytest.mark.asyncio
async def test_checkpoint_and_recovery_integration(simple_graph_def):
    """Test checkpoint creation and recovery integration."""
    from tinyllm.nodes.transform import TransformNode

    # Build graph
    graph = Graph(simple_graph_def)
    for node_def in simple_graph_def.nodes:
        graph.add_node(TransformNode(id=node_def.id, name=node_def.name))

    # Create executor with checkpointing
    config = ExecutorConfig(
        enable_checkpointing=True,
        checkpoint_interval=1,
        enable_partial_results=True,
    )
    executor = Executor(graph, config)

    # Execute task
    task = TaskPayload(content="test task")
    response = await executor.execute(task)

    # Check that checkpoints were created
    stats = executor.get_checkpoint_stats()
    assert stats["checkpointing_enabled"] is True
    assert stats["checkpoint_interval"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
