"""Tests for automated error recovery playbooks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinyllm.recovery import (
    ErrorCategory,
    RecoveryAction,
    RecoveryPlaybook,
    RecoveryResult,
    RecoveryStrategy,
    get_recovery_playbook,
)


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy configuration."""

    def test_strategy_creation(self):
        """Test creating recovery strategy."""
        strategy = RecoveryStrategy(
            error_category=ErrorCategory.TIMEOUT,
            actions=[RecoveryAction.RETRY, RecoveryAction.FAILOVER],
            max_retries=3,
            retry_delay_ms=1000,
        )

        assert strategy.error_category == ErrorCategory.TIMEOUT
        assert RecoveryAction.RETRY in strategy.actions
        assert strategy.max_retries == 3
        assert strategy.retry_delay_ms == 1000

    def test_strategy_validation(self):
        """Test strategy validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            RecoveryStrategy(
                error_category=ErrorCategory.TIMEOUT,
                actions=[RecoveryAction.RETRY],
                max_retries=20,  # Exceeds max
            )


class TestRecoveryPlaybook:
    """Tests for RecoveryPlaybook."""

    def test_initialization(self):
        """Test playbook initialization."""
        playbook = RecoveryPlaybook()

        # Check default strategies are initialized
        assert ErrorCategory.TIMEOUT in playbook._strategies
        assert ErrorCategory.CONNECTION in playbook._strategies
        assert ErrorCategory.RATE_LIMIT in playbook._strategies

    def test_register_custom_strategy(self):
        """Test registering custom recovery strategy."""
        playbook = RecoveryPlaybook()

        custom_strategy = RecoveryStrategy(
            error_category=ErrorCategory.MODEL_ERROR,
            actions=[RecoveryAction.RESTART, RecoveryAction.ALERT],
            max_retries=1,
        )

        playbook.register_strategy(custom_strategy)

        retrieved = playbook.get_strategy(ErrorCategory.MODEL_ERROR)
        assert retrieved is not None
        assert RecoveryAction.RESTART in retrieved.actions

    def test_get_strategy(self):
        """Test getting strategy for error category."""
        playbook = RecoveryPlaybook()

        strategy = playbook.get_strategy(ErrorCategory.TIMEOUT)

        assert strategy is not None
        assert strategy.error_category == ErrorCategory.TIMEOUT
        assert RecoveryAction.RETRY in strategy.actions

    @pytest.mark.asyncio
    async def test_successful_retry(self):
        """Test successful recovery via retry."""
        playbook = RecoveryPlaybook()

        call_count = [0]

        async def operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("First attempt fails")
            return "success"

        result = await playbook.recover(
            error_category=ErrorCategory.TIMEOUT,
            operation=operation,
            context={"component_id": "test_component"},
        )

        assert result.success is True
        assert RecoveryAction.RETRY in result.actions_taken
        assert result.retries == 2
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test when retries are exhausted."""
        playbook = RecoveryPlaybook()

        async def operation():
            raise RuntimeError("Always fails")

        result = await playbook.recover(
            error_category=ErrorCategory.TIMEOUT,
            operation=operation,
            context={"component_id": "test_component"},
        )

        assert result.success is False
        assert RecoveryAction.RETRY in result.actions_taken
        # After retry fails, strategy continues to DEGRADE
        assert RecoveryAction.DEGRADE in result.actions_taken

    @pytest.mark.asyncio
    async def test_failover_after_retry_failure(self):
        """Test failover when retry fails."""
        playbook = RecoveryPlaybook()

        async def primary_operation():
            raise RuntimeError("Primary always fails")

        async def fallback_operation():
            return "fallback success"

        result = await playbook.recover(
            error_category=ErrorCategory.TIMEOUT,
            operation=primary_operation,
            fallback_operation=fallback_operation,
            context={"component_id": "test_component"},
        )

        assert result.success is True
        assert RecoveryAction.RETRY in result.actions_taken
        assert RecoveryAction.FAILOVER in result.actions_taken
        assert result.metadata["recovery_method"] == "failover"

    @pytest.mark.asyncio
    async def test_failover_failure(self):
        """Test when both primary and fallback fail."""
        playbook = RecoveryPlaybook()

        async def primary_operation():
            raise RuntimeError("Primary fails")

        async def fallback_operation():
            raise RuntimeError("Fallback fails")

        result = await playbook.recover(
            error_category=ErrorCategory.TIMEOUT,
            operation=primary_operation,
            fallback_operation=fallback_operation,
            context={"component_id": "test_component"},
        )

        assert result.success is False
        assert RecoveryAction.RETRY in result.actions_taken
        # Failover is attempted but not added to actions_taken when it fails
        # Strategy continues to DEGRADE
        assert RecoveryAction.DEGRADE in result.actions_taken

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold."""
        playbook = RecoveryPlaybook()

        async def operation():
            raise RuntimeError("Always fails")

        # Register strategy with low threshold
        playbook.register_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.CONNECTION,
                actions=[RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK],
                max_retries=2,
                circuit_breaker_threshold=2,
            )
        )

        # First attempt - retries then fails
        result1 = await playbook.recover(
            error_category=ErrorCategory.CONNECTION,
            operation=operation,
            context={"component_id": "test_service"},
        )

        assert result1.success is False
        assert RecoveryAction.CIRCUIT_BREAK in result1.actions_taken

        # Second attempt - circuit should be open
        result2 = await playbook.recover(
            error_category=ErrorCategory.CONNECTION,
            operation=operation,
            context={"component_id": "test_service"},
        )

        assert result2.success is False
        assert "Circuit breaker is open" in result2.error

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_after_timeout(self):
        """Test circuit breaker resets after timeout."""
        playbook = RecoveryPlaybook()

        call_count = [0]

        async def operation():
            call_count[0] += 1
            if call_count[0] < 2:  # Fail first time only
                raise RuntimeError("Fails initially")
            return "success"

        # Register strategy with short timeout (minimum 1000ms)
        playbook.register_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.CONNECTION,
                actions=[RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK],
                max_retries=1,
                circuit_breaker_threshold=1,
                circuit_breaker_timeout_ms=1000,  # Minimum valid timeout
            )
        )

        # First attempt - opens circuit after failure
        result1 = await playbook.recover(
            error_category=ErrorCategory.CONNECTION,
            operation=operation,
            context={"component_id": "test_service"},
        )

        assert result1.success is False
        assert RecoveryAction.CIRCUIT_BREAK in result1.actions_taken

        # Wait for circuit to transition to half-open
        await asyncio.sleep(1.1)

        # Second attempt - circuit should be half-open, then succeed and close
        result2 = await playbook.recover(
            error_category=ErrorCategory.CONNECTION,
            operation=operation,
            context={"component_id": "test_service"},
        )

        assert result2.success is True

    @pytest.mark.asyncio
    async def test_validation_error_no_retry(self):
        """Test validation errors don't retry."""
        playbook = RecoveryPlaybook()

        call_count = [0]

        async def operation():
            call_count[0] += 1
            raise ValueError("Validation error")

        result = await playbook.recover(
            error_category=ErrorCategory.VALIDATION,
            operation=operation,
            context={"component_id": "test_component"},
        )

        # Validation strategy should have max_retries=0
        assert call_count[0] == 0  # Not called because no retry action
        assert RecoveryAction.ALERT in result.actions_taken

    @pytest.mark.asyncio
    async def test_rate_limit_exponential_backoff(self):
        """Test rate limit uses exponential backoff."""
        playbook = RecoveryPlaybook()

        call_times = []

        async def operation():
            import time

            call_times.append(time.time())
            if len(call_times) < 3:
                raise RuntimeError("Rate limited")
            return "success"

        result = await playbook.recover(
            error_category=ErrorCategory.RATE_LIMIT,
            operation=operation,
            context={"component_id": "test_component"},
        )

        assert result.success is True

        # Check that delays increased (exponential backoff)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Second delay should be longer than first (exponential)
            assert delay2 > delay1

    @pytest.mark.asyncio
    async def test_no_strategy_found(self):
        """Test when no strategy exists for error category."""
        playbook = RecoveryPlaybook()

        async def operation():
            return "success"

        result = await playbook.recover(
            error_category=ErrorCategory.UNKNOWN,
            operation=operation,
            context={},
        )

        assert result.success is False
        assert "No recovery strategy found" in result.error

    @pytest.mark.asyncio
    async def test_synchronous_operation(self):
        """Test recovery with synchronous operation."""
        playbook = RecoveryPlaybook()

        call_count = [0]

        def sync_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("First attempt fails")
            return "success"

        result = await playbook.recover(
            error_category=ErrorCategory.TIMEOUT,
            operation=sync_operation,
            context={"component_id": "test_component"},
        )

        assert result.success is True
        assert call_count[0] == 2

    def test_circuit_breaker_status(self):
        """Test getting circuit breaker status."""
        playbook = RecoveryPlaybook()

        # Initially closed
        status = playbook.get_circuit_breaker_status("test_component")
        assert status["state"] == "closed"
        assert status["is_open"] is False

        # Open circuit manually
        strategy = playbook.get_strategy(ErrorCategory.TIMEOUT)
        playbook._open_circuit("test_component", strategy)

        status = playbook.get_circuit_breaker_status("test_component")
        assert status["state"] == "open"
        assert status["is_open"] is True

    def test_reset_circuit_breaker(self):
        """Test manually resetting circuit breaker."""
        playbook = RecoveryPlaybook()

        # Open circuit
        strategy = playbook.get_strategy(ErrorCategory.TIMEOUT)
        playbook._open_circuit("test_component", strategy)

        assert playbook._is_circuit_open("test_component") is True

        # Reset
        playbook.reset_circuit_breaker("test_component")

        assert playbook._is_circuit_open("test_component") is False

    @pytest.mark.asyncio
    async def test_degradation_mode(self):
        """Test degradation mode activation."""
        playbook = RecoveryPlaybook()

        # Register strategy that uses degradation
        playbook.register_strategy(
            RecoveryStrategy(
                error_category=ErrorCategory.RESOURCE_EXHAUSTION,
                actions=[RecoveryAction.DEGRADE, RecoveryAction.ALERT],
                max_retries=0,
            )
        )

        async def operation():
            raise RuntimeError("Resource exhausted")

        result = await playbook.recover(
            error_category=ErrorCategory.RESOURCE_EXHAUSTION,
            operation=operation,
            context={"component_id": "test_component"},
        )

        assert RecoveryAction.DEGRADE in result.actions_taken
        assert RecoveryAction.ALERT in result.actions_taken


class TestRecoveryResult:
    """Tests for RecoveryResult."""

    def test_result_creation(self):
        """Test creating recovery result."""
        result = RecoveryResult(
            success=True,
            actions_taken=[RecoveryAction.RETRY],
            retries=2,
            elapsed_ms=150,
            metadata={"recovery_method": "retry"},
        )

        assert result.success is True
        assert RecoveryAction.RETRY in result.actions_taken
        assert result.retries == 2
        assert result.metadata["recovery_method"] == "retry"

    def test_result_with_error(self):
        """Test result with error."""
        result = RecoveryResult(
            success=False,
            actions_taken=[RecoveryAction.RETRY, RecoveryAction.FAILOVER],
            retries=3,
            elapsed_ms=300,
            error="All recovery actions exhausted",
        )

        assert result.success is False
        assert result.error is not None


class TestGlobalPlaybook:
    """Tests for global playbook instance."""

    def test_get_global_playbook(self):
        """Test getting global playbook singleton."""
        playbook1 = get_recovery_playbook()
        playbook2 = get_recovery_playbook()

        assert playbook1 is playbook2  # Same instance

    def test_global_playbook_has_strategies(self):
        """Test global playbook is initialized with strategies."""
        playbook = get_recovery_playbook()

        assert ErrorCategory.TIMEOUT in playbook._strategies
        assert ErrorCategory.CONNECTION in playbook._strategies
        assert ErrorCategory.RATE_LIMIT in playbook._strategies
