"""Tests for error handling and recovery system."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from tinyllm.errors import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    DeadLetterQueue,
    ErrorClassifier,
    ExecutionError,
    FatalError,
    ModelError,
    NetworkError,
    RateLimitError,
    RetryableError,
    RetryConfig,
    TimeoutError,
    TinyLLMError,
    ValidationError,
    calculate_delay,
    global_exception_handler,
    with_retry,
)


class TestExceptionHierarchy:
    """Test exception hierarchy and classification."""

    def test_base_exception(self):
        """Test TinyLLMError base class."""
        error = TinyLLMError("test error", code="TEST", details={"key": "value"})
        assert error.message == "test error"
        assert error.code == "TEST"
        assert error.details == {"key": "value"}
        assert error.recoverable is False

    def test_retryable_error(self):
        """Test retryable error."""
        error = RetryableError("network timeout")
        assert error.recoverable is True
        assert error.code == "RETRYABLE"

    def test_fatal_error(self):
        """Test fatal error."""
        error = FatalError("invalid input")
        assert error.recoverable is False
        assert error.code == "FATAL"

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("invalid schema")
        assert error.recoverable is False
        assert error.code == "VALIDATION_ERROR"

    def test_model_error(self):
        """Test model error."""
        error = ModelError("model failed", model="gpt-4")
        assert error.recoverable is True
        assert error.code == "MODEL_ERROR"
        assert error.details["model"] == "gpt-4"

    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError("operation timeout", timeout_ms=5000)
        assert error.recoverable is True
        assert error.code == "TIMEOUT_ERROR"
        assert error.details["timeout_ms"] == 5000

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("rate limit exceeded", retry_after_ms=1000)
        assert error.recoverable is True
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.details["retry_after_ms"] == 1000

    def test_circuit_open_error(self):
        """Test circuit open error."""
        error = CircuitOpenError("circuit open", service="api")
        assert error.recoverable is True
        assert error.code == "CIRCUIT_OPEN"
        assert error.details["service"] == "api"

    def test_error_to_dict(self):
        """Test error serialization."""
        error = ModelError("test", model="gpt-4")
        error_dict = error.to_dict()
        assert error_dict["type"] == "ModelError"
        assert error_dict["message"] == "test"
        assert error_dict["code"] == "MODEL_ERROR"
        assert error_dict["recoverable"] is True
        assert "timestamp" in error_dict
        assert "stack_trace" in error_dict


class TestErrorClassifier:
    """Test error classification logic."""

    def test_classify_tinyllm_error(self):
        """Test classification of TinyLLM errors."""
        retryable = RetryableError("test")
        fatal = FatalError("test")

        assert ErrorClassifier.classify(retryable) is True
        assert ErrorClassifier.classify(fatal) is False

    def test_classify_retryable_patterns(self):
        """Test classification of retryable patterns."""
        retryable_errors = [
            Exception("connection timeout"),
            Exception("503 Service Unavailable"),
            Exception("network error"),
            Exception("rate limit exceeded"),
        ]

        for error in retryable_errors:
            assert ErrorClassifier.classify(error) is True

    def test_classify_fatal_patterns(self):
        """Test classification of fatal patterns."""
        fatal_errors = [
            Exception("validation failed"),
            Exception("400 Bad Request"),
            Exception("401 Unauthorized"),
            Exception("invalid input"),
        ]

        for error in fatal_errors:
            assert ErrorClassifier.classify(error) is False

    def test_classify_unknown_defaults_fatal(self):
        """Test unknown errors default to fatal."""
        unknown_error = Exception("some random error")
        assert ErrorClassifier.classify(unknown_error) is False


class TestRetryLogic:
    """Test retry with jitter functionality."""

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            base_delay_ms=1000,
            exponential_base=2.0,
            jitter=False,
        )

        assert calculate_delay(0, config) == 1.0  # 1000ms = 1s
        assert calculate_delay(1, config) == 2.0  # 2000ms = 2s
        assert calculate_delay(2, config) == 4.0  # 4000ms = 4s

    def test_calculate_delay_max_limit(self):
        """Test max delay limit."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=5000,
            exponential_base=2.0,
            jitter=False,
        )

        # Should cap at max_delay_ms
        assert calculate_delay(10, config) == 5.0  # Capped at 5000ms

    def test_calculate_delay_with_jitter(self):
        """Test delay with jitter."""
        config = RetryConfig(
            base_delay_ms=1000,
            jitter=True,
            jitter_ratio=0.1,
        )

        delay = calculate_delay(0, config)
        # Should be within 10% jitter range
        assert 0.9 <= delay <= 1.1

    @pytest.mark.asyncio
    async def test_retry_decorator_success_first_try(self):
        """Test retry decorator with immediate success."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3))
        async def succeeds_immediately():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeeds_immediately()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_success_after_retries(self):
        """Test retry decorator with success after retries."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, base_delay_ms=10))
        async def succeeds_on_second_try():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("temporary failure")
            return "success"

        result = await succeeds_on_second_try()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_fatal_error_no_retry(self):
        """Test retry decorator doesn't retry fatal errors."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3))
        async def raises_fatal():
            nonlocal call_count
            call_count += 1
            raise ValidationError("invalid input")

        with pytest.raises(ValidationError):
            await raises_fatal()

        # Should not retry fatal errors
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_exhausted(self):
        """Test retry decorator when all attempts exhausted."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, base_delay_ms=10))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("persistent failure")

        with pytest.raises(TimeoutError):
            await always_fails()

        assert call_count == 3

    def test_retry_decorator_sync(self):
        """Test retry decorator with sync function."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, base_delay_ms=10))
        def succeeds_on_second_try():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("temporary failure")
            return "success"

        result = succeeds_on_second_try()
        assert result == "success"
        assert call_count == 2


class TestDeadLetterQueue:
    """Test dead letter queue functionality."""

    @pytest.mark.asyncio
    async def test_dlq_initialization(self):
        """Test DLQ initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            await dlq.initialize()
            assert dlq._initialized is True
            await dlq.close()

    @pytest.mark.asyncio
    async def test_dlq_enqueue_and_get(self):
        """Test enqueuing and retrieving from DLQ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            await dlq.initialize()

            # Enqueue a failed message
            error = ModelError("model failed", model="gpt-4")
            await dlq.enqueue(
                message_id="msg-123",
                trace_id="trace-456",
                payload={"content": "test"},
                error=error,
                attempts=3,
            )

            # Retrieve the message
            msg = await dlq.get("msg-123")
            assert msg is not None
            assert msg.id == "msg-123"
            assert msg.trace_id == "trace-456"
            assert msg.payload == {"content": "test"}
            assert msg.error_type == "ModelError"
            assert msg.attempts == 3

            await dlq.close()

    @pytest.mark.asyncio
    async def test_dlq_list(self):
        """Test listing DLQ messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            await dlq.initialize()

            # Enqueue multiple messages
            for i in range(5):
                await dlq.enqueue(
                    message_id=f"msg-{i}",
                    trace_id=f"trace-{i}",
                    payload={"index": i},
                    error=TimeoutError("timeout"),
                    attempts=1,
                )

            # List all messages
            messages = await dlq.list(limit=10)
            assert len(messages) == 5

            # List with limit
            messages = await dlq.list(limit=3)
            assert len(messages) == 3

            await dlq.close()

    @pytest.mark.asyncio
    async def test_dlq_delete(self):
        """Test deleting DLQ messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            await dlq.initialize()

            await dlq.enqueue(
                message_id="msg-123",
                trace_id="trace-456",
                payload={},
                error=TimeoutError("timeout"),
                attempts=1,
            )

            # Delete the message
            deleted = await dlq.delete("msg-123")
            assert deleted is True

            # Verify it's gone
            msg = await dlq.get("msg-123")
            assert msg is None

            await dlq.close()

    @pytest.mark.asyncio
    async def test_dlq_count(self):
        """Test counting DLQ messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            await dlq.initialize()

            # Initially empty
            count = await dlq.count()
            assert count == 0

            # Add messages
            await dlq.enqueue("msg-1", "trace-1", {}, TimeoutError("timeout"), 1)
            await dlq.enqueue("msg-2", "trace-2", {}, ModelError("error"), 1)

            count = await dlq.count()
            assert count == 2

            # Filter by error type
            count = await dlq.count(error_type="TimeoutError")
            assert count == 1

            await dlq.close()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            circuit = CircuitBreaker(
                name="test-service",
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()
            assert circuit._initialized is True
            assert circuit._stats.state == CircuitState.CLOSED
            await circuit.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            circuit = CircuitBreaker(
                name="test-service",
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()

            async def successful_operation():
                return "success"

            result = await circuit.call(successful_operation)
            assert result == "success"
            assert circuit._stats.state == CircuitState.CLOSED
            assert circuit._stats.total_successes == 1

            await circuit.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CircuitBreakerConfig(failure_threshold=3)
            circuit = CircuitBreaker(
                name="test-service",
                config=config,
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()

            async def failing_operation():
                raise Exception("simulated failure")

            # Fail 3 times to open circuit
            for _ in range(3):
                with pytest.raises(Exception):
                    await circuit.call(failing_operation)

            # Circuit should now be open
            assert circuit._stats.state == CircuitState.OPEN
            assert circuit._stats.total_failures == 3

            # Next call should be rejected
            with pytest.raises(CircuitOpenError):
                await circuit.call(failing_operation)

            await circuit.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=2,
                timeout_ms=1000,  # Minimum timeout
            )
            circuit = CircuitBreaker(
                name="test-service",
                config=config,
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()

            async def operation(should_fail):
                if should_fail:
                    raise Exception("failure")
                return "success"

            # Fail twice to open circuit
            for _ in range(2):
                with pytest.raises(Exception):
                    await circuit.call(operation, True)

            assert circuit._stats.state == CircuitState.OPEN

            # Wait for timeout to transition to half-open
            await asyncio.sleep(1.1)

            # Succeed twice to close circuit
            for _ in range(2):
                result = await circuit.call(operation, False)
                assert result == "success"

            # Circuit should be closed now
            assert circuit._stats.state == CircuitState.CLOSED

            await circuit.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_persistence(self):
        """Test circuit breaker state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cb.db"
            config = CircuitBreakerConfig(failure_threshold=2)

            # Create circuit and fail twice
            circuit1 = CircuitBreaker(
                name="test-service",
                config=config,
                db_path=db_path,
            )
            await circuit1.initialize()

            async def failing_operation():
                raise Exception("failure")

            for _ in range(2):
                with pytest.raises(Exception):
                    await circuit1.call(failing_operation)

            assert circuit1._stats.state == CircuitState.OPEN
            await circuit1.close()

            # Create new circuit with same name and verify state loaded
            circuit2 = CircuitBreaker(
                name="test-service",
                config=config,
                db_path=db_path,
            )
            await circuit2.initialize()

            assert circuit2._stats.state == CircuitState.OPEN
            assert circuit2._stats.total_failures == 2

            await circuit2.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_context_manager(self):
        """Test circuit breaker context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            circuit = CircuitBreaker(
                name="test-service",
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()

            async with circuit.protect():
                result = "success"

            assert circuit._stats.total_successes == 1

            await circuit.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self):
        """Test circuit breaker manual reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CircuitBreakerConfig(failure_threshold=2)
            circuit = CircuitBreaker(
                name="test-service",
                config=config,
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()

            async def failing_operation():
                raise Exception("failure")

            # Fail twice to open circuit
            for _ in range(2):
                with pytest.raises(Exception):
                    await circuit.call(failing_operation)

            assert circuit._stats.state == CircuitState.OPEN

            # Reset circuit
            await circuit.reset()

            assert circuit._stats.state == CircuitState.CLOSED

            await circuit.close()


class TestGlobalExceptionHandler:
    """Test global exception handler decorator."""

    @pytest.mark.asyncio
    async def test_handler_catches_tinyllm_error(self):
        """Test handler catches TinyLLM errors."""
        @global_exception_handler()
        async def raises_tinyllm_error():
            raise ModelError("test error")

        with pytest.raises(ModelError):
            await raises_tinyllm_error()

    @pytest.mark.asyncio
    async def test_handler_catches_generic_exception(self):
        """Test handler catches generic exceptions."""
        @global_exception_handler()
        async def raises_generic_error():
            raise Exception("generic error")

        with pytest.raises(Exception):
            await raises_generic_error()

    @pytest.mark.asyncio
    async def test_handler_with_dlq(self):
        """Test handler sends to DLQ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            await dlq.initialize()

            @global_exception_handler(dlq=dlq, send_to_dlq=True)
            async def raises_retryable_error(message_id, trace_id, payload):
                raise TimeoutError("timeout")

            with pytest.raises(TimeoutError):
                await raises_retryable_error(
                    message_id="msg-1",
                    trace_id="trace-1",
                    payload={"test": "data"},
                )

            # Check DLQ
            count = await dlq.count()
            assert count == 1

            msg = await dlq.get("msg-1")
            assert msg is not None
            assert msg.error_type == "TimeoutError"

            await dlq.close()

    def test_handler_sync_function(self):
        """Test handler with sync function."""
        @global_exception_handler()
        def raises_error():
            raise ValidationError("validation failed")

        with pytest.raises(ValidationError):
            raises_error()


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry logic with circuit breaker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            circuit = CircuitBreaker(
                name="api",
                config=CircuitBreakerConfig(failure_threshold=5),
                db_path=Path(tmpdir) / "cb.db",
            )
            await circuit.initialize()

            call_count = 0

            @with_retry(RetryConfig(max_attempts=3, base_delay_ms=10))
            async def api_call_with_circuit():
                nonlocal call_count
                call_count += 1

                async def operation():
                    if call_count < 3:
                        raise TimeoutError("timeout")
                    return "success"

                return await circuit.call(operation)

            result = await api_call_with_circuit()
            assert result == "success"
            assert call_count == 3

            await circuit.close()

    @pytest.mark.asyncio
    async def test_full_error_recovery_pipeline(self):
        """Test complete error recovery pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dlq = DeadLetterQueue(Path(tmpdir) / "dlq.db")
            circuit = CircuitBreaker(
                name="service",
                db_path=Path(tmpdir) / "cb.db",
            )
            await dlq.initialize()
            await circuit.initialize()

            @global_exception_handler(dlq=dlq)
            @with_retry(RetryConfig(max_attempts=2, base_delay_ms=10))
            async def complex_operation(message_id, trace_id, payload):
                async def inner_operation():
                    # Simulate persistent failure
                    raise NetworkError("network down")

                return await circuit.call(inner_operation)

            # Should retry, then send to DLQ
            with pytest.raises(NetworkError):
                await complex_operation(
                    message_id="msg-1",
                    trace_id="trace-1",
                    payload={"data": "test"},
                )

            # Verify DLQ has the message
            count = await dlq.count()
            assert count == 1

            await dlq.close()
            await circuit.close()
