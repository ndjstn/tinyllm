"""Tests for transaction-like rollback support."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinyllm.core.transaction import (
    Operation,
    Transaction,
    TransactionManager,
    TransactionState,
    get_transaction_manager,
)


class TestOperation:
    """Tests for Operation model."""

    def test_operation_creation(self):
        """Test creating an operation."""
        rollback_fn = lambda: None

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            rollback_fn=rollback_fn,
            state_before={"messages": []},
            state_after={"messages": ["new"]},
        )

        assert operation.operation_id == "op_001"
        assert operation.node_id == "model_node"
        assert operation.operation_type == "generate"
        assert operation.rollback_fn is rollback_fn
        assert operation.success is True

    def test_operation_defaults(self):
        """Test operation default values."""
        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
        )

        assert operation.state_before is None
        assert operation.state_after is None
        assert operation.rollback_fn is None
        assert operation.success is True


class TestTransaction:
    """Tests for Transaction model."""

    def test_transaction_creation(self):
        """Test creating a transaction."""
        tx = Transaction(transaction_id="tx_001")

        assert tx.transaction_id == "tx_001"
        assert tx.state == TransactionState.PENDING
        assert len(tx.operations) == 0
        assert tx.started_at is None
        assert tx.completed_at is None

    def test_begin_transaction(self):
        """Test beginning a transaction."""
        tx = Transaction(transaction_id="tx_001")

        tx.begin()

        assert tx.state == TransactionState.ACTIVE
        assert tx.started_at is not None

    def test_begin_transaction_wrong_state(self):
        """Test begin fails from wrong state."""
        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        with pytest.raises(ValueError, match="Cannot begin"):
            tx.begin()

    def test_add_operation(self):
        """Test adding operation to transaction."""
        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
        )

        tx.add_operation(operation)

        assert len(tx.operations) == 1
        assert tx.operations[0].operation_id == "op_001"

    def test_add_operation_wrong_state(self):
        """Test cannot add operation in wrong state."""
        tx = Transaction(transaction_id="tx_001")

        # Can add in PENDING state
        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
        )
        tx.add_operation(operation)

        # Begin and commit
        tx.begin()
        asyncio.run(tx.commit())

        # Cannot add after commit
        operation2 = Operation(
            operation_id="op_002",
            node_id="model_node",
            operation_type="generate",
        )

        with pytest.raises(ValueError, match="Cannot add operation"):
            tx.add_operation(operation2)

    @pytest.mark.asyncio
    async def test_commit_success(self):
        """Test committing a successful transaction."""
        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            success=True,
        )
        tx.add_operation(operation)

        result = await tx.commit()

        assert result is True
        assert tx.state == TransactionState.COMMITTED
        assert tx.completed_at is not None

    @pytest.mark.asyncio
    async def test_commit_with_failed_operation(self):
        """Test commit rolls back when operation failed."""
        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        # Add failed operation
        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            success=False,
        )
        tx.add_operation(operation)

        result = await tx.commit()

        assert result is False
        assert tx.state == TransactionState.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_commit_wrong_state(self):
        """Test commit fails from wrong state."""
        tx = Transaction(transaction_id="tx_001")

        with pytest.raises(ValueError, match="Cannot commit"):
            await tx.commit()

    @pytest.mark.asyncio
    async def test_rollback(self):
        """Test rolling back a transaction."""
        rollback_called = []

        def rollback_fn():
            rollback_called.append(True)

        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            rollback_fn=rollback_fn,
        )
        tx.add_operation(operation)

        result = await tx.rollback()

        assert result is True
        assert tx.state == TransactionState.ROLLED_BACK
        assert len(rollback_called) == 1

    @pytest.mark.asyncio
    async def test_rollback_reverse_order(self):
        """Test rollback executes operations in reverse order."""
        call_order = []

        def rollback_1():
            call_order.append(1)

        def rollback_2():
            call_order.append(2)

        def rollback_3():
            call_order.append(3)

        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        tx.add_operation(
            Operation(
                operation_id="op_001",
                node_id="node1",
                operation_type="op",
                rollback_fn=rollback_1,
            )
        )
        tx.add_operation(
            Operation(
                operation_id="op_002",
                node_id="node2",
                operation_type="op",
                rollback_fn=rollback_2,
            )
        )
        tx.add_operation(
            Operation(
                operation_id="op_003",
                node_id="node3",
                operation_type="op",
                rollback_fn=rollback_3,
            )
        )

        await tx.rollback()

        assert call_order == [3, 2, 1]  # Reverse order

    @pytest.mark.asyncio
    async def test_rollback_async_functions(self):
        """Test rollback with async rollback functions."""
        rollback_called = []

        async def rollback_fn():
            rollback_called.append(True)

        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            rollback_fn=rollback_fn,
        )
        tx.add_operation(operation)

        result = await tx.rollback()

        assert result is True
        assert len(rollback_called) == 1

    @pytest.mark.asyncio
    async def test_rollback_failure(self):
        """Test rollback failure handling."""

        def failing_rollback():
            raise RuntimeError("Rollback failed")

        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            rollback_fn=failing_rollback,
        )
        tx.add_operation(operation)

        result = await tx.rollback()

        assert result is False
        assert tx.state == TransactionState.FAILED

    @pytest.mark.asyncio
    async def test_cannot_rollback_committed(self):
        """Test cannot rollback committed transaction."""
        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            success=True,
        )
        tx.add_operation(operation)

        await tx.commit()

        result = await tx.rollback()

        assert result is False
        assert tx.state == TransactionState.COMMITTED

    @pytest.mark.asyncio
    async def test_rollback_already_rolled_back(self):
        """Test rolling back already rolled back transaction."""
        tx = Transaction(transaction_id="tx_001")
        tx.begin()

        operation = Operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
        )
        tx.add_operation(operation)

        result1 = await tx.rollback()
        result2 = await tx.rollback()

        assert result1 is True
        assert result2 is True  # Idempotent
        assert tx.state == TransactionState.ROLLED_BACK


class TestTransactionManager:
    """Tests for TransactionManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = TransactionManager()

        assert len(manager._transactions) == 0
        assert manager._active_transaction is None

    def test_begin_transaction(self):
        """Test beginning a transaction."""
        manager = TransactionManager()

        tx = manager.begin_transaction("tx_001", metadata={"graph_id": "test_graph"})

        assert tx.transaction_id == "tx_001"
        assert tx.state == TransactionState.ACTIVE
        assert tx.metadata["graph_id"] == "test_graph"
        assert manager._active_transaction is tx

    def test_begin_transaction_duplicate_id(self):
        """Test cannot begin transaction with duplicate ID."""
        manager = TransactionManager()

        manager.begin_transaction("tx_001")

        with pytest.raises(ValueError, match="already exists"):
            manager.begin_transaction("tx_001")

    def test_get_transaction(self):
        """Test getting transaction by ID."""
        manager = TransactionManager()

        tx = manager.begin_transaction("tx_001")
        retrieved = manager.get_transaction("tx_001")

        assert retrieved is tx

    def test_get_transaction_not_found(self):
        """Test getting non-existent transaction returns None."""
        manager = TransactionManager()

        tx = manager.get_transaction("nonexistent")

        assert tx is None

    def test_get_active_transaction(self):
        """Test getting active transaction."""
        manager = TransactionManager()

        tx = manager.begin_transaction("tx_001")

        assert manager.get_active_transaction() is tx

    @pytest.mark.asyncio
    async def test_commit_transaction(self):
        """Test committing a transaction."""
        manager = TransactionManager()

        tx = manager.begin_transaction("tx_001")
        tx.add_operation(
            Operation(
                operation_id="op_001",
                node_id="model_node",
                operation_type="generate",
                success=True,
            )
        )

        result = await manager.commit_transaction("tx_001")

        assert result is True
        assert tx.state == TransactionState.COMMITTED
        assert manager._active_transaction is None

    @pytest.mark.asyncio
    async def test_commit_transaction_not_found(self):
        """Test commit fails for non-existent transaction."""
        manager = TransactionManager()

        with pytest.raises(ValueError, match="not found"):
            await manager.commit_transaction("nonexistent")

    @pytest.mark.asyncio
    async def test_rollback_transaction(self):
        """Test rolling back a transaction."""
        manager = TransactionManager()

        rollback_called = []

        def rollback_fn():
            rollback_called.append(True)

        tx = manager.begin_transaction("tx_001")
        tx.add_operation(
            Operation(
                operation_id="op_001",
                node_id="model_node",
                operation_type="generate",
                rollback_fn=rollback_fn,
            )
        )

        result = await manager.rollback_transaction("tx_001")

        assert result is True
        assert tx.state == TransactionState.ROLLED_BACK
        assert len(rollback_called) == 1
        assert manager._active_transaction is None

    @pytest.mark.asyncio
    async def test_rollback_transaction_not_found(self):
        """Test rollback fails for non-existent transaction."""
        manager = TransactionManager()

        with pytest.raises(ValueError, match="not found"):
            await manager.rollback_transaction("nonexistent")

    def test_add_operation_to_active_transaction(self):
        """Test adding operation to active transaction."""
        manager = TransactionManager()

        manager.begin_transaction("tx_001")

        operation = manager.add_operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
        )

        tx = manager.get_transaction("tx_001")
        assert len(tx.operations) == 1
        assert tx.operations[0].operation_id == "op_001"

    def test_add_operation_to_specific_transaction(self):
        """Test adding operation to specific transaction."""
        manager = TransactionManager()

        manager.begin_transaction("tx_001")
        manager.begin_transaction("tx_002")

        operation = manager.add_operation(
            operation_id="op_001",
            node_id="model_node",
            operation_type="generate",
            transaction_id="tx_001",
        )

        tx1 = manager.get_transaction("tx_001")
        tx2 = manager.get_transaction("tx_002")

        assert len(tx1.operations) == 1
        assert len(tx2.operations) == 0

    def test_add_operation_no_active_transaction(self):
        """Test adding operation fails with no active transaction."""
        manager = TransactionManager()

        with pytest.raises(ValueError, match="No active transaction"):
            manager.add_operation(
                operation_id="op_001",
                node_id="model_node",
                operation_type="generate",
            )

    def test_add_operation_transaction_not_found(self):
        """Test adding operation fails for non-existent transaction."""
        manager = TransactionManager()

        with pytest.raises(ValueError, match="not found"):
            manager.add_operation(
                operation_id="op_001",
                node_id="model_node",
                operation_type="generate",
                transaction_id="nonexistent",
            )

    def test_get_transaction_stats(self):
        """Test getting transaction statistics."""
        manager = TransactionManager()

        # Create transactions in different states
        tx1 = manager.begin_transaction("tx_001")
        asyncio.run(tx1.commit())

        tx2 = manager.begin_transaction("tx_002")
        asyncio.run(tx2.rollback())

        manager.begin_transaction("tx_003")  # Active

        stats = manager.get_transaction_stats()

        assert stats["total_transactions"] == 3
        assert stats["committed"] == 1
        assert stats["rolled_back"] == 1
        assert stats["active"] == 1
        assert stats["has_active_transaction"] is True

    def test_clear_completed_transactions(self):
        """Test clearing old completed transactions."""
        manager = TransactionManager()

        # Create and complete transactions
        tx1 = manager.begin_transaction("tx_001")
        asyncio.run(tx1.commit())

        tx2 = manager.begin_transaction("tx_002")
        asyncio.run(tx2.rollback())

        manager.begin_transaction("tx_003")  # Active

        # Clear with very long threshold (should clear nothing)
        cleared = manager.clear_completed_transactions(older_than_seconds=999999)

        assert cleared == 0
        assert len(manager._transactions) == 3

        # Clear with 0 threshold (should clear completed ones)
        cleared = manager.clear_completed_transactions(older_than_seconds=0)

        assert cleared == 2  # tx_001 and tx_002
        assert len(manager._transactions) == 1
        assert "tx_003" in manager._transactions


class TestGlobalTransactionManager:
    """Tests for global transaction manager."""

    def test_get_global_manager(self):
        """Test getting global transaction manager singleton."""
        manager1 = get_transaction_manager()
        manager2 = get_transaction_manager()

        assert manager1 is manager2  # Same instance

    def test_global_manager_isolated(self):
        """Test global manager is separate from local instances."""
        global_manager = get_transaction_manager()
        local_manager = TransactionManager()

        global_manager.begin_transaction("tx_001")

        assert global_manager.get_transaction("tx_001") is not None
        assert local_manager.get_transaction("tx_001") is None
