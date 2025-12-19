"""Automated error recovery playbooks for TinyLLM.

This module provides automated recovery strategies for common failure scenarios,
enabling the system to self-heal and maintain availability.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="recovery")
metrics = get_metrics_collector()


class RecoveryAction(str, Enum):
    """Types of recovery actions."""

    RETRY = "retry"  # Retry the failed operation
    RESTART = "restart"  # Restart the affected component
    FAILOVER = "failover"  # Switch to backup/fallback
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker
    DEGRADE = "degrade"  # Activate degraded mode
    ALERT = "alert"  # Send alert to operators
    ROLLBACK = "rollback"  # Rollback to previous state


class ErrorCategory(str, Enum):
    """Categories of errors for recovery."""

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INVALID_STATE = "invalid_state"
    VALIDATION = "validation"
    MODEL_ERROR = "model_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(BaseModel):
    """Configuration for a recovery strategy."""

    model_config = {"extra": "forbid"}

    error_category: ErrorCategory = Field(description="Category of error this handles")
    actions: List[RecoveryAction] = Field(description="Ordered list of recovery actions")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_ms: int = Field(default=1000, ge=0, description="Delay between retries")
    backoff_multiplier: float = Field(
        default=2.0, ge=1.0, description="Backoff multiplier for exponential delays"
    )
    max_delay_ms: int = Field(
        default=30000, ge=1000, description="Maximum delay between retries"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Failures before circuit break"
    )
    circuit_breaker_timeout_ms: int = Field(
        default=60000, ge=1000, description="Circuit breaker timeout"
    )


class RecoveryResult(BaseModel):
    """Result of a recovery attempt."""

    model_config = {"extra": "forbid"}

    success: bool = Field(description="Whether recovery succeeded")
    actions_taken: List[RecoveryAction] = Field(
        default_factory=list, description="Recovery actions taken"
    )
    retries: int = Field(default=0, ge=0, description="Number of retries performed")
    elapsed_ms: int = Field(default=0, ge=0, description="Total recovery time")
    error: Optional[str] = Field(default=None, description="Error if recovery failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional recovery metadata"
    )


class RecoveryPlaybook:
    """Automated recovery playbook executor.

    Implements automated recovery strategies for different error categories,
    including retries, failover, circuit breaking, and degradation.

    Example:
        >>> playbook = RecoveryPlaybook()
        >>> result = await playbook.recover(
        ...     error_category=ErrorCategory.TIMEOUT,
        ...     operation=my_async_func,
        ...     context={"node_id": "model_1"}
        ... )
    """

    def __init__(self):
        """Initialize recovery playbook."""
        self._strategies: Dict[ErrorCategory, RecoveryStrategy] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_strategies()

    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies for common errors."""

        # Timeout recovery: retry with backoff, then failover
        self._strategies[ErrorCategory.TIMEOUT] = RecoveryStrategy(
            error_category=ErrorCategory.TIMEOUT,
            actions=[RecoveryAction.RETRY, RecoveryAction.FAILOVER, RecoveryAction.DEGRADE],
            max_retries=3,
            retry_delay_ms=1000,
            backoff_multiplier=2.0,
        )

        # Connection errors: retry quickly, then circuit break
        self._strategies[ErrorCategory.CONNECTION] = RecoveryStrategy(
            error_category=ErrorCategory.CONNECTION,
            actions=[
                RecoveryAction.RETRY,
                RecoveryAction.CIRCUIT_BREAK,
                RecoveryAction.FAILOVER,
            ],
            max_retries=5,
            retry_delay_ms=500,
            backoff_multiplier=1.5,
            circuit_breaker_threshold=3,
        )

        # Rate limit: wait and retry with exponential backoff
        self._strategies[ErrorCategory.RATE_LIMIT] = RecoveryStrategy(
            error_category=ErrorCategory.RATE_LIMIT,
            actions=[RecoveryAction.RETRY, RecoveryAction.FAILOVER],
            max_retries=3,
            retry_delay_ms=2000,
            backoff_multiplier=3.0,
            max_delay_ms=30000,
        )

        # Resource exhaustion: circuit break and alert
        self._strategies[ErrorCategory.RESOURCE_EXHAUSTION] = RecoveryStrategy(
            error_category=ErrorCategory.RESOURCE_EXHAUSTION,
            actions=[RecoveryAction.CIRCUIT_BREAK, RecoveryAction.ALERT, RecoveryAction.DEGRADE],
            max_retries=1,
            circuit_breaker_threshold=2,
        )

        # Validation errors: no retry, just alert
        self._strategies[ErrorCategory.VALIDATION] = RecoveryStrategy(
            error_category=ErrorCategory.VALIDATION,
            actions=[RecoveryAction.ALERT],
            max_retries=0,
        )

        # Model errors: retry once, then failover
        self._strategies[ErrorCategory.MODEL_ERROR] = RecoveryStrategy(
            error_category=ErrorCategory.MODEL_ERROR,
            actions=[RecoveryAction.RETRY, RecoveryAction.FAILOVER, RecoveryAction.DEGRADE],
            max_retries=2,
            retry_delay_ms=1000,
        )

    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a custom recovery strategy.

        Args:
            strategy: Recovery strategy to register.
        """
        self._strategies[strategy.error_category] = strategy
        logger.info(
            "recovery_strategy_registered",
            error_category=strategy.error_category.value,
            actions=[a.value for a in strategy.actions],
        )

    def get_strategy(self, error_category: ErrorCategory) -> Optional[RecoveryStrategy]:
        """Get recovery strategy for an error category.

        Args:
            error_category: Category of error.

        Returns:
            Recovery strategy or None if not found.
        """
        return self._strategies.get(error_category)

    async def recover(
        self,
        error_category: ErrorCategory,
        operation: Callable,
        context: Optional[Dict[str, Any]] = None,
        fallback_operation: Optional[Callable] = None,
    ) -> RecoveryResult:
        """Execute recovery playbook for a failed operation.

        Args:
            error_category: Category of error that occurred.
            operation: The operation to retry/recover.
            context: Additional context about the failure.
            fallback_operation: Optional fallback operation to try.

        Returns:
            RecoveryResult with outcome of recovery attempt.
        """
        start_time = time.time()
        context = context or {}
        actions_taken = []

        logger.info(
            "recovery_started",
            error_category=error_category.value,
            context=context,
        )

        # Get strategy for this error category
        strategy = self.get_strategy(error_category)
        if not strategy:
            logger.warning(
                "no_recovery_strategy",
                error_category=error_category.value,
            )
            return RecoveryResult(
                success=False,
                error="No recovery strategy found for error category",
                elapsed_ms=int((time.time() - start_time) * 1000),
            )

        # Check circuit breaker
        component_id = context.get("component_id", "default")
        if self._is_circuit_open(component_id):
            logger.warning(
                "circuit_breaker_open",
                component_id=component_id,
            )
            return RecoveryResult(
                success=False,
                error="Circuit breaker is open",
                actions_taken=[RecoveryAction.CIRCUIT_BREAK],
                elapsed_ms=int((time.time() - start_time) * 1000),
            )

        # Execute recovery actions in order
        for action in strategy.actions:
            if action == RecoveryAction.RETRY:
                success, retries = await self._retry_with_backoff(
                    operation, strategy, context
                )
                actions_taken.append(RecoveryAction.RETRY)
                if success:
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    self._record_success(component_id)
                    return RecoveryResult(
                        success=True,
                        actions_taken=actions_taken,
                        retries=retries,
                        elapsed_ms=elapsed_ms,
                        metadata={"recovery_method": "retry"},
                    )

            elif action == RecoveryAction.FAILOVER and fallback_operation:
                logger.info("attempting_failover", context=context)
                try:
                    result = await fallback_operation()
                    actions_taken.append(RecoveryAction.FAILOVER)
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    self._record_success(component_id)
                    return RecoveryResult(
                        success=True,
                        actions_taken=actions_taken,
                        elapsed_ms=elapsed_ms,
                        metadata={"recovery_method": "failover"},
                    )
                except Exception as e:
                    logger.warning("failover_failed", error=str(e))

            elif action == RecoveryAction.CIRCUIT_BREAK:
                actions_taken.append(RecoveryAction.CIRCUIT_BREAK)
                self._open_circuit(component_id, strategy)
                logger.warning("circuit_breaker_opened", component_id=component_id)

            elif action == RecoveryAction.DEGRADE:
                actions_taken.append(RecoveryAction.DEGRADE)
                logger.warning("degraded_mode_activated", context=context)
                # Degradation is handled by the caller

            elif action == RecoveryAction.ALERT:
                actions_taken.append(RecoveryAction.ALERT)
                self._send_alert(error_category, context)

        elapsed_ms = int((time.time() - start_time) * 1000)
        self._record_failure(component_id)

        return RecoveryResult(
            success=False,
            actions_taken=actions_taken,
            error="All recovery actions exhausted",
            elapsed_ms=elapsed_ms,
        )

    async def _retry_with_backoff(
        self,
        operation: Callable,
        strategy: RecoveryStrategy,
        context: Dict[str, Any],
    ) -> tuple[bool, int]:
        """Retry operation with exponential backoff.

        Args:
            operation: Operation to retry.
            strategy: Recovery strategy with retry configuration.
            context: Context information.

        Returns:
            Tuple of (success, retry_count).
        """
        delay_ms = strategy.retry_delay_ms
        retries = 0

        for attempt in range(strategy.max_retries):
            retries = attempt + 1

            try:
                logger.info(
                    "retry_attempt",
                    attempt=retries,
                    max_retries=strategy.max_retries,
                    delay_ms=delay_ms,
                )

                # Wait before retry (except first attempt)
                if attempt > 0:
                    await asyncio.sleep(delay_ms / 1000.0)

                # Try the operation
                result = await operation() if asyncio.iscoroutinefunction(operation) else operation()

                logger.info("retry_succeeded", attempt=retries)
                return True, retries

            except Exception as e:
                logger.warning(
                    "retry_failed",
                    attempt=retries,
                    error=str(e),
                )

                # Calculate next delay with exponential backoff
                delay_ms = min(
                    int(delay_ms * strategy.backoff_multiplier),
                    strategy.max_delay_ms,
                )

        return False, retries

    def _is_circuit_open(self, component_id: str) -> bool:
        """Check if circuit breaker is open for a component.

        Args:
            component_id: Component identifier.

        Returns:
            True if circuit is open.
        """
        if component_id not in self._circuit_breakers:
            return False

        circuit = self._circuit_breakers[component_id]
        if not circuit.get("is_open", False):
            return False

        # Check if timeout has elapsed
        opened_at = circuit.get("opened_at", 0)
        timeout_ms = circuit.get("timeout_ms", 60000)
        elapsed_ms = (time.time() - opened_at) * 1000

        if elapsed_ms >= timeout_ms:
            # Try to close circuit (half-open state)
            logger.info("circuit_breaker_half_open", component_id=component_id)
            circuit["is_open"] = False
            circuit["half_open"] = True
            return False

        return True

    def _open_circuit(self, component_id: str, strategy: RecoveryStrategy) -> None:
        """Open circuit breaker for a component.

        Args:
            component_id: Component identifier.
            strategy: Recovery strategy with circuit breaker config.
        """
        self._circuit_breakers[component_id] = {
            "is_open": True,
            "half_open": False,
            "opened_at": time.time(),
            "timeout_ms": strategy.circuit_breaker_timeout_ms,
            "failure_count": 0,
        }

        metrics.update_circuit_breaker_state("open", model=component_id)

    def _record_success(self, component_id: str) -> None:
        """Record successful operation for a component.

        Args:
            component_id: Component identifier.
        """
        if component_id in self._circuit_breakers:
            circuit = self._circuit_breakers[component_id]
            if circuit.get("half_open", False):
                # Close circuit on success in half-open state
                logger.info("circuit_breaker_closed", component_id=component_id)
                del self._circuit_breakers[component_id]
                metrics.update_circuit_breaker_state("closed", model=component_id)

    def _record_failure(self, component_id: str) -> None:
        """Record failed operation for a component.

        Args:
            component_id: Component identifier.
        """
        if component_id in self._circuit_breakers:
            circuit = self._circuit_breakers[component_id]
            if circuit.get("half_open", False):
                # Re-open circuit on failure in half-open state
                logger.warning("circuit_breaker_reopened", component_id=component_id)
                circuit["is_open"] = True
                circuit["half_open"] = False
                circuit["opened_at"] = time.time()
                metrics.update_circuit_breaker_state("open", model=component_id)

        metrics.increment_circuit_breaker_failures(model=component_id)

    def _send_alert(self, error_category: ErrorCategory, context: Dict[str, Any]) -> None:
        """Send alert for error requiring attention.

        Args:
            error_category: Category of error.
            context: Error context.
        """
        logger.error(
            "recovery_alert",
            error_category=error_category.value,
            context=context,
            message=f"Recovery alert: {error_category.value} error requires attention",
        )

        # Could integrate with alerting systems like PagerDuty, Slack, etc.

    def get_circuit_breaker_status(self, component_id: str) -> Dict[str, Any]:
        """Get circuit breaker status for a component.

        Args:
            component_id: Component identifier.

        Returns:
            Circuit breaker status.
        """
        if component_id not in self._circuit_breakers:
            return {"state": "closed", "is_open": False}

        circuit = self._circuit_breakers[component_id]
        elapsed_ms = (time.time() - circuit.get("opened_at", 0)) * 1000

        return {
            "state": "open" if circuit.get("is_open") else "half-open",
            "is_open": circuit.get("is_open", False),
            "opened_at": circuit.get("opened_at"),
            "elapsed_ms": int(elapsed_ms),
            "timeout_ms": circuit.get("timeout_ms"),
            "failure_count": circuit.get("failure_count", 0),
        }

    def reset_circuit_breaker(self, component_id: str) -> None:
        """Manually reset circuit breaker for a component.

        Args:
            component_id: Component identifier.
        """
        if component_id in self._circuit_breakers:
            del self._circuit_breakers[component_id]
            logger.info("circuit_breaker_reset", component_id=component_id)
            metrics.update_circuit_breaker_state("closed", model=component_id)


# Global recovery playbook instance
_global_playbook: Optional[RecoveryPlaybook] = None


def get_recovery_playbook() -> RecoveryPlaybook:
    """Get global recovery playbook instance.

    Returns:
        Global RecoveryPlaybook singleton.
    """
    global _global_playbook
    if _global_playbook is None:
        _global_playbook = RecoveryPlaybook()
    return _global_playbook
