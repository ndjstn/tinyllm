"""Example demonstrating request queue with backpressure.

This example shows how to use the QueuedExecutor to handle multiple
concurrent requests with priority-based queuing and backpressure control.
"""

import asyncio
import time

from tinyllm.core.builder import load_graph
from tinyllm.core.executor import Executor
from tinyllm.core.message import TaskPayload
from tinyllm.queue import BackpressureMode, Priority, QueuedExecutor
from tinyllm.metrics import start_metrics_server, get_metrics_collector


async def basic_queue_example():
    """Basic queue usage example."""
    print("=== Basic Queue Example ===\n")

    # Load graph
    graph = load_graph("graphs/multi_domain.yaml")
    executor = Executor(graph)

    # Create queued executor with 3 workers
    queued_executor = QueuedExecutor(
        executor=executor,
        max_workers=3,
        max_queue_size=100,
        backpressure_mode=BackpressureMode.BLOCK,
    )

    # Start the worker pool
    async with queued_executor.lifespan():
        # Submit a normal priority request
        task = TaskPayload(content="What is Python?")
        response = await queued_executor.execute(task, priority=Priority.NORMAL)

        print(f"Response: {response.content}")
        print(f"Latency: {response.total_latency_ms}ms")
        print(f"Nodes executed: {response.nodes_executed}")


async def priority_queue_example():
    """Demonstrate priority-based queuing."""
    print("\n=== Priority Queue Example ===\n")

    # Load graph
    graph = load_graph("graphs/multi_domain.yaml")
    executor = Executor(graph)

    # Create queued executor with single worker to show ordering
    queued_executor = QueuedExecutor(
        executor=executor,
        max_workers=1,
        max_queue_size=100,
    )

    async with queued_executor.lifespan():
        # Submit requests with different priorities
        print("Submitting requests with different priorities...")

        low_future = await queued_executor.submit(
            TaskPayload(content="Low priority task"),
            priority=Priority.LOW,
        )

        normal_future = await queued_executor.submit(
            TaskPayload(content="Normal priority task"),
            priority=Priority.NORMAL,
        )

        high_future = await queued_executor.submit(
            TaskPayload(content="High priority task"),
            priority=Priority.HIGH,
        )

        # High priority will be processed first
        print("\nProcessing requests (high priority first)...")

        high_response = await high_future
        print(f"✓ High priority completed: {high_response.total_latency_ms}ms")

        normal_response = await normal_future
        print(f"✓ Normal priority completed: {normal_response.total_latency_ms}ms")

        low_response = await low_future
        print(f"✓ Low priority completed: {low_response.total_latency_ms}ms")


async def concurrent_requests_example():
    """Demonstrate concurrent request processing."""
    print("\n=== Concurrent Requests Example ===\n")

    # Load graph
    graph = load_graph("graphs/multi_domain.yaml")
    executor = Executor(graph)

    # Create queued executor with multiple workers
    queued_executor = QueuedExecutor(
        executor=executor,
        max_workers=5,
        max_queue_size=100,
    )

    async with queued_executor.lifespan():
        # Submit many requests concurrently
        queries = [
            "What is machine learning?",
            "How does Python work?",
            "Explain quantum computing",
            "What is REST API?",
            "How to use Docker?",
            "What is Kubernetes?",
            "Explain microservices",
            "What is GraphQL?",
            "How does Redis work?",
            "What is PostgreSQL?",
        ]

        print(f"Submitting {len(queries)} requests concurrently...")
        start_time = time.monotonic()

        # Submit all requests
        tasks = [
            queued_executor.execute(TaskPayload(content=query))
            for query in queries
        ]

        # Wait for all to complete
        responses = await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start_time
        success_count = sum(1 for r in responses if r.success)

        print(f"\nCompleted {success_count}/{len(queries)} requests")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average time per request: {elapsed/len(queries):.2f}s")

        # Show queue status
        status = queued_executor.get_status()
        print(f"\nQueue Status:")
        print(f"  Total processed: {status.total_processed}")
        print(f"  Total rejected: {status.total_rejected}")
        print(f"  Average wait time: {status.average_wait_time_ms:.2f}ms")


async def backpressure_example():
    """Demonstrate backpressure handling."""
    print("\n=== Backpressure Example ===\n")

    # Load graph
    graph = load_graph("graphs/multi_domain.yaml")
    executor = Executor(graph)

    # Create queued executor with small queue (reject mode)
    queued_executor = QueuedExecutor(
        executor=executor,
        max_workers=2,
        max_queue_size=5,
        backpressure_mode=BackpressureMode.REJECT,
    )

    async with queued_executor.lifespan():
        print("Testing REJECT backpressure mode (queue size: 5)...")

        accepted = 0
        rejected = 0

        # Try to submit more requests than queue can hold
        for i in range(10):
            try:
                await queued_executor.submit(
                    TaskPayload(content=f"Request {i}")
                )
                accepted += 1
            except asyncio.QueueFull:
                rejected += 1
                print(f"  Request {i} rejected (queue full)")

        print(f"\nResults:")
        print(f"  Accepted: {accepted}")
        print(f"  Rejected: {rejected}")


async def worker_health_example():
    """Demonstrate worker health monitoring."""
    print("\n=== Worker Health Monitoring Example ===\n")

    # Load graph
    graph = load_graph("graphs/multi_domain.yaml")
    executor = Executor(graph)

    queued_executor = QueuedExecutor(
        executor=executor,
        max_workers=3,
        max_queue_size=100,
    )

    async with queued_executor.lifespan():
        # Submit some requests
        tasks = [
            queued_executor.execute(TaskPayload(content=f"Query {i}"))
            for i in range(5)
        ]

        await asyncio.gather(*tasks)

        # Check worker health
        health = queued_executor.get_worker_health()

        print("Worker Health Status:")
        for worker_id, status in health.items():
            print(f"\n  Worker {worker_id}:")
            print(f"    Healthy: {status.is_healthy}")
            print(f"    Total processed: {status.total_processed}")
            print(f"    Total errors: {status.total_errors}")

        # Check queue status
        status = queued_executor.get_status()
        print(f"\nQueue Status:")
        print(f"  Active workers: {status.active_workers}/{status.max_workers}")
        print(f"  Total processed: {status.total_processed}")


async def metrics_example():
    """Demonstrate metrics integration."""
    print("\n=== Metrics Integration Example ===\n")

    # Start metrics server
    print("Starting metrics server on port 9090...")
    start_metrics_server(port=9090)
    print("Metrics available at http://localhost:9090/metrics\n")

    # Load graph
    graph = load_graph("graphs/multi_domain.yaml")
    executor = Executor(graph)

    queued_executor = QueuedExecutor(
        executor=executor,
        max_workers=3,
        max_queue_size=100,
    )

    async with queued_executor.lifespan():
        # Execute some requests
        print("Processing requests (metrics being collected)...")

        for i in range(10):
            priority = [Priority.LOW, Priority.NORMAL, Priority.HIGH][i % 3]
            await queued_executor.execute(
                TaskPayload(content=f"Query {i}"),
                priority=priority,
            )

        print("\nMetrics collected! View at http://localhost:9090/metrics")
        print("Look for tinyllm_queue_* metrics")


async def main():
    """Run all examples."""
    print("TinyLLM Queue Examples\n")
    print("=" * 60)

    try:
        await basic_queue_example()
        await priority_queue_example()
        await concurrent_requests_example()
        await backpressure_example()
        await worker_health_example()
        # await metrics_example()  # Uncomment to test metrics

    except FileNotFoundError:
        print("\n[ERROR] Graph file not found. Make sure you have:")
        print("  graphs/multi_domain.yaml")
        print("\nOr update the examples to use a different graph.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
