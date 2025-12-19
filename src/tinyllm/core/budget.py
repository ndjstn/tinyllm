"""Execution budget limits for TinyLLM graph execution.

This module provides budget-based execution control, limiting resource
consumption in terms of time, tokens, cost, and operations.
"""

import time
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="budget")


class BudgetType(str, Enum):
    """Types of execution budgets."""

    TIME = "time"  # Time budget in seconds
    TOKENS = "tokens"  # Token budget
    COST = "cost"  # Cost budget in USD
    OPERATIONS = "operations"  # Number of operations
    MEMORY = "memory"  # Memory budget in bytes


class BudgetStatus(str, Enum):
    """Status of budget consumption."""

    OK = "ok"  # Within budget
    WARNING = "warning"  # Approaching limit (>80%)
    EXCEEDED = "exceeded"  # Budget exceeded
    EXHAUSTED = "exhausted"  # Completely exhausted


class ExecutionBudget(BaseModel):
    """Budget specification for execution resources."""

    model_config = {"extra": "forbid"}

    max_time_seconds: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="Maximum execution time in seconds",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens (input + output)",
    )
    max_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum cost in USD",
    )
    max_operations: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of operations",
    )
    max_memory_bytes: Optional[int] = Field(
        default=None,
        ge=1024,
        description="Maximum memory usage in bytes",
    )
    warning_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for warnings (0.0-1.0)",
    )


class BudgetUsage(BaseModel):
    """Current usage of execution budgets."""

    model_config = {"extra": "forbid"}

    time_seconds: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    operations: int = Field(default=0, ge=0)
    memory_bytes: int = Field(default=0, ge=0)


class BudgetManager:
    """Manages execution budgets and tracks resource consumption.

    Enforces budget limits across multiple dimensions (time, tokens, cost, etc.)
    and provides warnings when approaching limits.
    """

    def __init__(self, budget: ExecutionBudget):
        """Initialize budget manager.

        Args:
            budget: Budget limits to enforce.
        """
        self.budget = budget
        self.usage = BudgetUsage()
        self.start_time = time.perf_counter()
        self._warned_budgets: set[BudgetType] = set()

    def track_operation(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        memory_bytes: int = 0,
    ) -> None:
        """Track resource consumption from an operation.

        Args:
            tokens_in: Input tokens consumed.
            tokens_out: Output tokens consumed.
            cost_usd: Cost incurred.
            memory_bytes: Memory consumed.
        """
        # Update usage
        self.usage.tokens += tokens_in + tokens_out
        self.usage.cost_usd += cost_usd
        self.usage.operations += 1
        self.usage.memory_bytes = max(self.usage.memory_bytes, memory_bytes)

        # Update time
        self.usage.time_seconds = time.perf_counter() - self.start_time

        # Log tracking
        logger.debug(
            "budget_usage_tracked",
            tokens_delta=tokens_in + tokens_out,
            cost_delta=cost_usd,
            total_tokens=self.usage.tokens,
            total_cost=self.usage.cost_usd,
            operations=self.usage.operations,
        )

        # Check for warnings
        self._check_warnings()

    def check_budget(self, budget_type: Optional[BudgetType] = None) -> BudgetStatus:
        """Check budget status for a specific type or all types.

        Args:
            budget_type: Specific budget type to check, or None for all.

        Returns:
            Budget status (OK, WARNING, or EXCEEDED).
        """
        if budget_type:
            return self._check_single_budget(budget_type)

        # Check all budgets, return worst status
        statuses = [
            self._check_single_budget(bt)
            for bt in [
                BudgetType.TIME,
                BudgetType.TOKENS,
                BudgetType.COST,
                BudgetType.OPERATIONS,
                BudgetType.MEMORY,
            ]
        ]

        # Priority: EXCEEDED > WARNING > OK
        if BudgetStatus.EXCEEDED in statuses:
            return BudgetStatus.EXCEEDED
        if BudgetStatus.WARNING in statuses:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def _check_single_budget(self, budget_type: BudgetType) -> BudgetStatus:
        """Check a single budget type.

        Args:
            budget_type: Budget type to check.

        Returns:
            Budget status for this type.
        """
        if budget_type == BudgetType.TIME:
            limit = self.budget.max_time_seconds
            usage = self.usage.time_seconds
        elif budget_type == BudgetType.TOKENS:
            limit = self.budget.max_tokens
            usage = self.usage.tokens
        elif budget_type == BudgetType.COST:
            limit = self.budget.max_cost_usd
            usage = self.usage.cost_usd
        elif budget_type == BudgetType.OPERATIONS:
            limit = self.budget.max_operations
            usage = self.usage.operations
        elif budget_type == BudgetType.MEMORY:
            limit = self.budget.max_memory_bytes
            usage = self.usage.memory_bytes
        else:
            return BudgetStatus.OK

        # If no limit set, always OK
        if limit is None:
            return BudgetStatus.OK

        # Calculate utilization
        utilization = usage / limit if limit > 0 else 0.0

        # Determine status
        if utilization >= 1.0:
            return BudgetStatus.EXCEEDED
        if utilization >= self.budget.warning_threshold:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def is_exceeded(self, budget_type: Optional[BudgetType] = None) -> bool:
        """Check if budget is exceeded.

        Args:
            budget_type: Specific budget type, or None to check all.

        Returns:
            True if budget is exceeded.
        """
        return self.check_budget(budget_type) == BudgetStatus.EXCEEDED

    def can_proceed(self, required_tokens: int = 0, required_cost: float = 0.0) -> bool:
        """Check if operation can proceed within budget.

        Args:
            required_tokens: Tokens needed for operation.
            required_cost: Cost needed for operation.

        Returns:
            True if operation can proceed without exceeding budget.
        """
        # Check if current budget allows the operation
        if self.budget.max_tokens:
            if self.usage.tokens + required_tokens > self.budget.max_tokens:
                return False

        if self.budget.max_cost_usd:
            if self.usage.cost_usd + required_cost > self.budget.max_cost_usd:
                return False

        if self.budget.max_operations:
            if self.usage.operations + 1 > self.budget.max_operations:
                return False

        return True

    def get_remaining(self, budget_type: BudgetType) -> Optional[float]:
        """Get remaining budget for a type.

        Args:
            budget_type: Budget type to check.

        Returns:
            Remaining budget, or None if no limit set.
        """
        if budget_type == BudgetType.TIME:
            limit = self.budget.max_time_seconds
            usage = self.usage.time_seconds
        elif budget_type == BudgetType.TOKENS:
            limit = self.budget.max_tokens
            usage = self.usage.tokens
        elif budget_type == BudgetType.COST:
            limit = self.budget.max_cost_usd
            usage = self.usage.cost_usd
        elif budget_type == BudgetType.OPERATIONS:
            limit = self.budget.max_operations
            usage = self.usage.operations
        elif budget_type == BudgetType.MEMORY:
            limit = self.budget.max_memory_bytes
            usage = self.usage.memory_bytes
        else:
            return None

        if limit is None:
            return None

        return max(0.0, limit - usage)

    def get_utilization(self, budget_type: BudgetType) -> Optional[float]:
        """Get budget utilization as a percentage.

        Args:
            budget_type: Budget type to check.

        Returns:
            Utilization (0.0-1.0), or None if no limit set.
        """
        if budget_type == BudgetType.TIME:
            limit = self.budget.max_time_seconds
            usage = self.usage.time_seconds
        elif budget_type == BudgetType.TOKENS:
            limit = self.budget.max_tokens
            usage = self.usage.tokens
        elif budget_type == BudgetType.COST:
            limit = self.budget.max_cost_usd
            usage = self.usage.cost_usd
        elif budget_type == BudgetType.OPERATIONS:
            limit = self.budget.max_operations
            usage = self.usage.operations
        elif budget_type == BudgetType.MEMORY:
            limit = self.budget.max_memory_bytes
            usage = self.usage.memory_bytes
        else:
            return None

        if limit is None or limit == 0:
            return None

        return usage / limit

    def _check_warnings(self) -> None:
        """Check and log warnings for budgets approaching limits."""
        for budget_type in [
            BudgetType.TIME,
            BudgetType.TOKENS,
            BudgetType.COST,
            BudgetType.OPERATIONS,
            BudgetType.MEMORY,
        ]:
            status = self._check_single_budget(budget_type)

            if status == BudgetStatus.WARNING and budget_type not in self._warned_budgets:
                utilization = self.get_utilization(budget_type)
                remaining = self.get_remaining(budget_type)

                logger.warning(
                    "budget_warning",
                    budget_type=budget_type.value,
                    utilization_pct=int(utilization * 100) if utilization else 0,
                    remaining=remaining,
                )

                self._warned_budgets.add(budget_type)

            elif status == BudgetStatus.EXCEEDED:
                logger.error(
                    "budget_exceeded",
                    budget_type=budget_type.value,
                    usage=self._get_usage_value(budget_type),
                    limit=self._get_limit_value(budget_type),
                )

    def _get_usage_value(self, budget_type: BudgetType) -> Any:
        """Get current usage value for a budget type."""
        if budget_type == BudgetType.TIME:
            return self.usage.time_seconds
        elif budget_type == BudgetType.TOKENS:
            return self.usage.tokens
        elif budget_type == BudgetType.COST:
            return self.usage.cost_usd
        elif budget_type == BudgetType.OPERATIONS:
            return self.usage.operations
        elif budget_type == BudgetType.MEMORY:
            return self.usage.memory_bytes
        return None

    def _get_limit_value(self, budget_type: BudgetType) -> Any:
        """Get limit value for a budget type."""
        if budget_type == BudgetType.TIME:
            return self.budget.max_time_seconds
        elif budget_type == BudgetType.TOKENS:
            return self.budget.max_tokens
        elif budget_type == BudgetType.COST:
            return self.budget.max_cost_usd
        elif budget_type == BudgetType.OPERATIONS:
            return self.budget.max_operations
        elif budget_type == BudgetType.MEMORY:
            return self.budget.max_memory_bytes
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get budget summary with all metrics.

        Returns:
            Dictionary with budget usage and status.
        """
        return {
            "usage": {
                "time_seconds": self.usage.time_seconds,
                "tokens": self.usage.tokens,
                "cost_usd": self.usage.cost_usd,
                "operations": self.usage.operations,
                "memory_bytes": self.usage.memory_bytes,
            },
            "limits": {
                "time_seconds": self.budget.max_time_seconds,
                "tokens": self.budget.max_tokens,
                "cost_usd": self.budget.max_cost_usd,
                "operations": self.budget.max_operations,
                "memory_bytes": self.budget.max_memory_bytes,
            },
            "status": {
                "overall": self.check_budget().value,
                "time": self._check_single_budget(BudgetType.TIME).value,
                "tokens": self._check_single_budget(BudgetType.TOKENS).value,
                "cost": self._check_single_budget(BudgetType.COST).value,
                "operations": self._check_single_budget(BudgetType.OPERATIONS).value,
                "memory": self._check_single_budget(BudgetType.MEMORY).value,
            },
            "utilization": {
                "time": self.get_utilization(BudgetType.TIME),
                "tokens": self.get_utilization(BudgetType.TOKENS),
                "cost": self.get_utilization(BudgetType.COST),
                "operations": self.get_utilization(BudgetType.OPERATIONS),
                "memory": self.get_utilization(BudgetType.MEMORY),
            },
        }

    def reset(self) -> None:
        """Reset budget usage counters."""
        self.usage = BudgetUsage()
        self.start_time = time.perf_counter()
        self._warned_budgets.clear()

        logger.info("budget_reset")


class BudgetExceededError(Exception):
    """Exception raised when execution budget is exceeded."""

    def __init__(self, budget_type: BudgetType, usage: Any, limit: Any):
        """Initialize exception.

        Args:
            budget_type: Type of budget exceeded.
            usage: Current usage.
            limit: Budget limit.
        """
        self.budget_type = budget_type
        self.usage = usage
        self.limit = limit
        super().__init__(
            f"Budget exceeded: {budget_type.value} usage {usage} exceeds limit {limit}"
        )
