"""Demo of TinyLLM's comprehensive error handling system.

This example demonstrates all 5 error handling features:
1. Global exception handler with structured error types
2. Dead letter queue for failed messages
3. Automatic retry with exponential backoff and jitter
4. Error classification system (retryable vs fatal)
5. Circuit breaker with state persistence across restarts
"""

import asyncio
import tempfile
from pathlib import Path

from tinyllm.errors import (
    CircuitBreaker,
    CircuitBreakerConfig,
    DeadLetterQueue,
    ErrorClassifier,
    ModelError,
    NetworkError,
    RetryConfig,
    TimeoutError,
    ValidationError,
    global_exception_handler,
    with_retry,
)


async def main():
    print("=" * 60)
    print("TinyLLM Error Handling System Demo")
    print("=" * 60)

    # Create temporary directory for persistence
    tmpdir = tempfile.mkdtemp()
    dlq_path = Path(tmpdir) / "dlq.db"
    cb_path = Path(tmpdir) / "circuit_breaker.db"

    # ================================================================
    # Task 1: Global Exception Handler with Structured Error Types
    # ================================================================
    print("\n1. Structured Error Types")
    print("-" * 60)

    try:
        raise ModelError("Model inference failed", model="gpt-4")
    except ModelError as e:
        print(f"  Caught: {e.__class__.__name__}")
        print(f"  Code: {e.code}")
        print(f"  Message: {e.message}")
        print(f"  Recoverable: {e.recoverable}")
        print(f"  Details: {e.details}")

    # ================================================================
    # Task 2: Dead Letter Queue
    # ================================================================
    print("\n2. Dead Letter Queue")
    print("-" * 60)

    dlq = DeadLetterQueue(dlq_path)
    await dlq.initialize()

    # Send failed messages to DLQ
    error = NetworkError("Connection refused")
    await dlq.enqueue(
        message_id="msg-001",
        trace_id="trace-123",
        payload={"prompt": "Hello, world!"},
        error=error,
        attempts=3,
    )

    # List DLQ messages
    messages = await dlq.list(limit=10)
    print(f"  Messages in DLQ: {len(messages)}")
    for msg in messages:
        print(f"    - ID: {msg.id}, Error: {msg.error_type}, Attempts: {msg.attempts}")

    await dlq.close()

    # ================================================================
    # Task 3: Automatic Retry with Jitter
    # ================================================================
    print("\n3. Automatic Retry with Exponential Backoff + Jitter")
    print("-" * 60)

    call_count = 0

    @with_retry(RetryConfig(max_attempts=3, base_delay_ms=100, jitter=True))
    async def unreliable_api_call():
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}")

        if call_count < 3:
            raise TimeoutError("Request timeout")
        return "Success!"

    result = await unreliable_api_call()
    print(f"  Final result: {result}")
    print(f"  Total attempts: {call_count}")

    # ================================================================
    # Task 4: Error Classification System
    # ================================================================
    print("\n4. Error Classification (Retryable vs Fatal)")
    print("-" * 60)

    errors_to_classify = [
        TimeoutError("Connection timeout"),
        NetworkError("DNS resolution failed"),
        ValidationError("Invalid input schema"),
        ModelError("Model server unavailable"),
    ]

    for error in errors_to_classify:
        is_retryable = ErrorClassifier.classify(error)
        print(f"  {error.__class__.__name__:20s} -> {'RETRYABLE' if is_retryable else 'FATAL'}")

    # ================================================================
    # Task 5: Circuit Breaker with Persistence
    # ================================================================
    print("\n5. Circuit Breaker with State Persistence")
    print("-" * 60)

    circuit = CircuitBreaker(
        name="api-service",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_ms=2000,
        ),
        db_path=cb_path,
    )
    await circuit.initialize()

    # Simulate failures to open circuit
    async def failing_operation():
        raise Exception("Service down")

    print("  Simulating 3 failures to open circuit...")
    for i in range(3):
        try:
            await circuit.call(failing_operation)
        except Exception:
            print(f"    Failure {i+1}/3")

    stats = circuit.get_stats()
    print(f"  Circuit state: {stats.state}")
    print(f"  Total failures: {stats.total_failures}")

    # Demonstrate persistence
    await circuit.close()

    print("\n  Recreating circuit to demonstrate persistence...")
    circuit2 = CircuitBreaker(
        name="api-service",
        config=CircuitBreakerConfig(failure_threshold=3),
        db_path=cb_path,
    )
    await circuit2.initialize()

    stats2 = circuit2.get_stats()
    print(f"  Loaded state: {stats2.state}")
    print(f"  Loaded failures: {stats2.total_failures}")

    await circuit2.close()

    # ================================================================
    # Bonus: Complete Integration
    # ================================================================
    print("\n6. Complete Integration Example")
    print("-" * 60)

    dlq_integrated = DeadLetterQueue(Path(tmpdir) / "integrated_dlq.db")
    await dlq_integrated.initialize()

    circuit_integrated = CircuitBreaker(
        name="integrated-service",
        db_path=Path(tmpdir) / "integrated_cb.db",
    )
    await circuit_integrated.initialize()

    @global_exception_handler(dlq=dlq_integrated, send_to_dlq=True)
    @with_retry(RetryConfig(max_attempts=2, base_delay_ms=50))
    async def complex_operation(message_id, trace_id, payload):
        """Operation with full error handling stack."""
        
        async def inner():
            # Simulate retryable error
            raise NetworkError("Temporary network issue")
        
        return await circuit_integrated.call(inner)

    try:
        await complex_operation(
            message_id="complex-msg-001",
            trace_id="trace-456",
            payload={"data": "test"},
        )
    except NetworkError:
        print("  Operation failed after retries")

    # Check DLQ
    dlq_count = await dlq_integrated.count()
    print(f"  Messages sent to DLQ: {dlq_count}")

    await dlq_integrated.close()
    await circuit_integrated.close()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nAll 5 error handling features demonstrated:")
    print("  ✓ Structured error types")
    print("  ✓ Dead letter queue")
    print("  ✓ Automatic retry with jitter")
    print("  ✓ Error classification")
    print("  ✓ Circuit breaker with persistence")


if __name__ == "__main__":
    asyncio.run(main())
