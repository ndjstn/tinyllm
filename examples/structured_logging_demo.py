#!/usr/bin/env python3
"""Comprehensive demo of TinyLLM's structured logging capabilities.

This script demonstrates:
1. Console vs JSON logging
2. Different log levels
3. Structured context
4. Error handling with logging
5. Performance tracking
"""

import asyncio
import time
from pathlib import Path

from tinyllm import configure_logging, get_logger, bind_context, clear_context


class LoggingDemo:
    """Demo class showing logging best practices."""

    def __init__(self):
        self.logger = get_logger(__name__, component="demo")

    async def simulate_query_execution(self, query: str, trace_id: str):
        """Simulate a query execution with comprehensive logging."""
        # Bind trace context for all logs in this scope
        bind_context(trace_id=trace_id)

        start_time = time.perf_counter()

        self.logger.info(
            "query_received",
            query=query[:50],  # Truncate for logging
            query_length=len(query),
        )

        try:
            # Simulate validation
            self.logger.debug("validating_query", checks=["syntax", "length", "safety"])
            await asyncio.sleep(0.01)
            self.logger.debug("validation_passed")

            # Simulate routing
            self.logger.info("routing_query", router="domain_classifier")
            await asyncio.sleep(0.02)
            domain = "math" if "2+2" in query else "general"
            self.logger.info("routing_completed", domain=domain, confidence=0.95)

            # Simulate processing
            self.logger.info("processing_started", processor=f"{domain}_processor")
            await asyncio.sleep(0.05)

            # Simulate token generation
            tokens_generated = 42
            for i in range(0, tokens_generated, 10):
                self.logger.debug("tokens_generated", count=min(i + 10, tokens_generated))
                await asyncio.sleep(0.01)

            self.logger.info("processing_completed", tokens=tokens_generated)

            # Calculate metrics
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            self.logger.info(
                "query_completed",
                success=True,
                latency_ms=elapsed_ms,
                tokens=tokens_generated,
                throughput_tps=tokens_generated / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            )

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            self.logger.error(
                "query_failed",
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=elapsed_ms,
                exc_info=True,
            )
            raise
        finally:
            clear_context()

    async def demonstrate_error_handling(self):
        """Show how errors are logged with full context."""
        self.logger.info("error_demo_started")

        try:
            # Simulate various error conditions
            self.logger.debug("attempting_risky_operation")
            # Simulate a controlled failure
            raise ValueError("Simulated error for demo purposes")
        except ValueError as e:
            self.logger.error(
                "operation_failed",
                error=str(e),
                operation="risky_operation",
                recoverable=True,
                exc_info=True,  # Include stack trace
            )

        self.logger.info("error_demo_completed", handled=True)

    def demonstrate_performance_logging(self):
        """Show performance-related logging."""
        self.logger.info("performance_test_started")

        # Log system metrics
        self.logger.info(
            "system_metrics",
            cpu_count=4,
            memory_available_mb=8192,
            disk_space_gb=500,
        )

        # Log operation metrics
        operations = [
            ("database_query", 5.2),
            ("cache_lookup", 0.3),
            ("model_inference", 150.5),
            ("response_serialization", 2.1),
        ]

        for operation, duration_ms in operations:
            self.logger.info(
                "operation_timed",
                operation=operation,
                duration_ms=duration_ms,
            )

        total_ms = sum(d for _, d in operations)
        self.logger.info(
            "performance_test_completed",
            total_duration_ms=total_ms,
            operations_count=len(operations),
        )


async def main():
    print("\n" + "=" * 70)
    print("TinyLLM Structured Logging Demo")
    print("=" * 70 + "\n")

    demo = LoggingDemo()

    # Demo 1: Console logging with DEBUG level
    print("Demo 1: Console Logging (Development Mode)")
    print("-" * 70)
    configure_logging(log_level="DEBUG", log_format="console")

    await demo.simulate_query_execution("What is 2+2?", "trace-001")
    print()

    # Demo 2: JSON logging with INFO level
    print("\nDemo 2: JSON Logging (Production Mode)")
    print("-" * 70)
    configure_logging(log_level="INFO", log_format="json")

    await demo.simulate_query_execution("Explain quantum computing", "trace-002")
    print()

    # Demo 3: Error handling
    print("\nDemo 3: Error Handling with Logging")
    print("-" * 70)
    configure_logging(log_level="INFO", log_format="console")

    await demo.demonstrate_error_handling()
    print()

    # Demo 4: Performance logging
    print("\nDemo 4: Performance Metrics Logging")
    print("-" * 70)

    demo.demonstrate_performance_logging()
    print()

    # Demo 5: Context binding across async operations
    print("\nDemo 5: Context Binding with Async Operations")
    print("-" * 70)

    async def worker(worker_id: int):
        logger = get_logger("worker")
        bind_context(worker_id=worker_id, session_id="session-123")

        logger.info("worker_started")
        await asyncio.sleep(0.01)
        logger.info("worker_processing", items=10)
        await asyncio.sleep(0.01)
        logger.info("worker_completed")

        clear_context()

    # Run multiple workers concurrently
    await asyncio.gather(*[worker(i) for i in range(3)])

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Use structured fields (key=value) instead of string interpolation")
    print("2. Console format for development, JSON for production")
    print("3. Bind context (trace_id, user_id) for request tracking")
    print("4. Include exc_info=True for exceptions to get stack traces")
    print("5. Log performance metrics for monitoring and optimization")


if __name__ == "__main__":
    asyncio.run(main())
