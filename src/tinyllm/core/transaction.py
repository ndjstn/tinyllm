"""Transaction-like rollback support for multi-node operations.

This module provides transactional semantics for graph execution,
allowing atomic operations across multiple nodes with rollback capabilities.
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.core.message import Message
from tinyllm.core.node import NodeResult
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="transaction")


class TransactionState(str, Enum):
    """States of a transaction."""

    PENDING = "pending"  # Transaction created but not started
    ACTIVE = "active"  # Transaction in progress
    COMMITTED = "committed"  # Transaction successfully committed
    ROLLED_BACK = "rolled_back"  # Transaction rolled back
    FAILED = "failed"  # Transaction failed and cannot rollback


class Operation(BaseModel):
    """Represents an operation in a transaction."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    operation_id: str = Field(description="Unique operation identifier")
    node_id: str = Field(description="Node that performed the operation")
    operation_type: str = Field(description="Type of operation")
    state_before: Optional[Dict[str, Any]] = Field(
        default=None, description="State before operation"
    )
    state_after: Optional[Dict[str, Any]] = Field(
        default=None, description="State after operation"
    )
    rollback_fn: Optional[Callable] = Field(
        default=None, description="Function to rollback this operation"
    )
    timestamp: float = Field(default_factory=time.time, description="Operation timestamp")
    success: bool = Field(default=True, description="Whether operation succeeded")


class Transaction(BaseModel):
    """Represents a transaction across multiple nodes."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    transaction_id: str = Field(description="Unique transaction identifier")
    state: TransactionState = Field(
        default=TransactionState.PENDING, description="Current transaction state"
    )
    operations: List[Operation] = Field(
        default_factory=list, description="Operations in this transaction"
    )
    started_at: Optional[float] = Field(default=None, description="Transaction start time")
    completed_at: Optional[float] = Field(default=None, description="Transaction completion time")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional transaction metadata"
    )

    def add_operation(self, operation: Operation) -> None:
        """Add an operation to the transaction.

        Args:
            operation: Operation to add.
        """
        if self.state not in [TransactionState.PENDING, TransactionState.ACTIVE]:
            raise ValueError(
                f"Cannot add operation to transaction in state {self.state}"
            )

        self.operations.append(operation)
        logger.debug(
            "operation_added",
            transaction_id=self.transaction_id,
            operation_id=operation.operation_id,
            node_id=operation.node_id,
        )

    def begin(self) -> None:
        """Begin the transaction."""
        if self.state != TransactionState.PENDING:
            raise ValueError(f"Cannot begin transaction in state {self.state}")

        self.state = TransactionState.ACTIVE
        self.started_at = time.time()
        logger.info("transaction_started", transaction_id=self.transaction_id)

    async def commit(self) -> bool:
        """Commit the transaction.

        Returns:
            True if committed successfully.
        """
        if self.state != TransactionState.ACTIVE:
            raise ValueError(f"Cannot commit transaction in state {self.state}")

        # Check if all operations succeeded
        failed_ops = [op for op in self.operations if not op.success]
        if failed_ops:
            logger.warning(
                "transaction_commit_failed_operations",
                transaction_id=self.transaction_id,
                failed_count=len(failed_ops),
            )
            await self.rollback()
            return False

        self.state = TransactionState.COMMITTED
        self.completed_at = time.time()
        logger.info(
            "transaction_committed",
            transaction_id=self.transaction_id,
            operation_count=len(self.operations),
            duration_ms=int((self.completed_at - (self.started_at or 0)) * 1000),
        )
        return True

    async def rollback(self) -> bool:
        """Rollback the transaction.

        Returns:
            True if rollback succeeded.
        """
        if self.state == TransactionState.COMMITTED:
            logger.warning(
                "cannot_rollback_committed",
                transaction_id=self.transaction_id,
            )
            return False

        if self.state == TransactionState.ROLLED_BACK:
            logger.info(
                "already_rolled_back",
                transaction_id=self.transaction_id,
            )
            return True

        logger.info(
            "transaction_rollback_started",
            transaction_id=self.transaction_id,
            operation_count=len(self.operations),
        )

        # Rollback operations in reverse order
        rollback_failures = []
        for operation in reversed(self.operations):
            if operation.rollback_fn:
                try:
                    logger.debug(
                        "rolling_back_operation",
                        transaction_id=self.transaction_id,
                        operation_id=operation.operation_id,
                        node_id=operation.node_id,
                    )

                    # Execute rollback function
                    if asyncio.iscoroutinefunction(operation.rollback_fn):
                        await operation.rollback_fn()
                    else:
                        operation.rollback_fn()

                    logger.debug(
                        "operation_rolled_back",
                        transaction_id=self.transaction_id,
                        operation_id=operation.operation_id,
                    )

                except Exception as e:
                    logger.error(
                        "operation_rollback_failed",
                        transaction_id=self.transaction_id,
                        operation_id=operation.operation_id,
                        error=str(e),
                    )
                    rollback_failures.append(
                        {"operation_id": operation.operation_id, "error": str(e)}
                    )

        if rollback_failures:
            self.state = TransactionState.FAILED
            logger.error(
                "transaction_rollback_failed",
                transaction_id=self.transaction_id,
                failures=rollback_failures,
            )
            return False

        self.state = TransactionState.ROLLED_BACK
        self.completed_at = time.time()
        logger.info(
            "transaction_rolled_back",
            transaction_id=self.transaction_id,
            duration_ms=int((self.completed_at - (self.started_at or 0)) * 1000),
        )
        return True


class TransactionManager:
    """Manages transactions for graph execution.

    Provides transactional semantics for multi-node operations,
    ensuring atomicity and rollback capabilities.

    Example:
        >>> tm = TransactionManager()
        >>> tx = tm.begin_transaction("tx_001")
        >>> tx.add_operation(Operation(
        ...     operation_id="op_1",
        ...     node_id="model_node",
        ...     operation_type="generate",
        ...     rollback_fn=lambda: cleanup_model()
        ... ))
        >>> await tx.commit()
    """

    def __init__(self):
        """Initialize transaction manager."""
        self._transactions: Dict[str, Transaction] = {}
        self._active_transaction: Optional[Transaction] = None

    def begin_transaction(
        self, transaction_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """Begin a new transaction.

        Args:
            transaction_id: Unique transaction identifier.
            metadata: Optional metadata for the transaction.

        Returns:
            New Transaction object.

        Raises:
            ValueError: If transaction ID already exists.
        """
        if transaction_id in self._transactions:
            raise ValueError(f"Transaction {transaction_id} already exists")

        transaction = Transaction(
            transaction_id=transaction_id,
            metadata=metadata or {},
        )

        self._transactions[transaction_id] = transaction
        self._active_transaction = transaction
        transaction.begin()

        logger.info(
            "transaction_created",
            transaction_id=transaction_id,
        )

        return transaction

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID.

        Args:
            transaction_id: Transaction identifier.

        Returns:
            Transaction or None if not found.
        """
        return self._transactions.get(transaction_id)

    def get_active_transaction(self) -> Optional[Transaction]:
        """Get the currently active transaction.

        Returns:
            Active transaction or None.
        """
        return self._active_transaction

    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction.

        Args:
            transaction_id: Transaction to commit.

        Returns:
            True if committed successfully.
        """
        transaction = self.get_transaction(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")

        success = await transaction.commit()

        if self._active_transaction and self._active_transaction.transaction_id == transaction_id:
            self._active_transaction = None

        return success

    async def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback a transaction.

        Args:
            transaction_id: Transaction to rollback.

        Returns:
            True if rollback succeeded.
        """
        transaction = self.get_transaction(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")

        success = await transaction.rollback()

        if self._active_transaction and self._active_transaction.transaction_id == transaction_id:
            self._active_transaction = None

        return success

    def add_operation(
        self,
        operation_id: str,
        node_id: str,
        operation_type: str,
        rollback_fn: Optional[Callable] = None,
        state_before: Optional[Dict[str, Any]] = None,
        state_after: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> Operation:
        """Add an operation to a transaction.

        Args:
            operation_id: Unique operation identifier.
            node_id: Node performing the operation.
            operation_type: Type of operation.
            rollback_fn: Function to rollback this operation.
            state_before: State before operation.
            state_after: State after operation.
            transaction_id: Transaction to add to (uses active if None).

        Returns:
            Created Operation.

        Raises:
            ValueError: If no active transaction.
        """
        if transaction_id:
            transaction = self.get_transaction(transaction_id)
            if not transaction:
                raise ValueError(f"Transaction {transaction_id} not found")
        else:
            transaction = self._active_transaction
            if not transaction:
                raise ValueError("No active transaction")

        operation = Operation(
            operation_id=operation_id,
            node_id=node_id,
            operation_type=operation_type,
            rollback_fn=rollback_fn,
            state_before=state_before,
            state_after=state_after,
        )

        transaction.add_operation(operation)
        return operation

    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get statistics about transactions.

        Returns:
            Dictionary with transaction statistics.
        """
        total = len(self._transactions)
        committed = sum(
            1 for tx in self._transactions.values() if tx.state == TransactionState.COMMITTED
        )
        rolled_back = sum(
            1 for tx in self._transactions.values() if tx.state == TransactionState.ROLLED_BACK
        )
        active = sum(
            1 for tx in self._transactions.values() if tx.state == TransactionState.ACTIVE
        )
        failed = sum(
            1 for tx in self._transactions.values() if tx.state == TransactionState.FAILED
        )

        return {
            "total_transactions": total,
            "committed": committed,
            "rolled_back": rolled_back,
            "active": active,
            "failed": failed,
            "has_active_transaction": self._active_transaction is not None,
        }

    def clear_completed_transactions(self, older_than_seconds: float = 3600) -> int:
        """Clear completed transactions older than specified time.

        Args:
            older_than_seconds: Clear transactions completed before this many seconds ago.

        Returns:
            Number of transactions cleared.
        """
        cutoff_time = time.time() - older_than_seconds
        cleared = 0

        to_remove = []
        for tx_id, tx in self._transactions.items():
            if tx.state in [TransactionState.COMMITTED, TransactionState.ROLLED_BACK, TransactionState.FAILED]:
                if tx.completed_at and tx.completed_at < cutoff_time:
                    to_remove.append(tx_id)
                    cleared += 1

        for tx_id in to_remove:
            del self._transactions[tx_id]

        if cleared > 0:
            logger.info(
                "transactions_cleared",
                cleared_count=cleared,
                older_than_seconds=older_than_seconds,
            )

        return cleared


# Need to import asyncio for the rollback function check
import asyncio


# Global transaction manager instance
_global_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """Get global transaction manager instance.

    Returns:
        Global TransactionManager singleton.
    """
    global _global_transaction_manager
    if _global_transaction_manager is None:
        _global_transaction_manager = TransactionManager()
    return _global_transaction_manager
