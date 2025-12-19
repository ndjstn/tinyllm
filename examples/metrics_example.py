#!/usr/bin/env python
"""Example demonstrating TinyLLM metrics collection and Prometheus integration.

This script shows how to:
1. Start the metrics server
2. Track request metrics
3. Record token usage
4. Handle errors with metrics
5. Use context managers for automatic tracking

Run this script and navigate to http://localhost:9090/metrics to see the metrics.
"""

import asyncio
import time
from threading import Thread

from tinyllm.metrics import get_metrics_collector, start_metrics_server


async def simulate_requests():
    """Simulate various requests to demonstrate metrics collection."""
    metrics = get_metrics_collector()

    print("Starting request simulation...")

    # Simulate 10 successful requests
    for i in range(10):
        print(f"  Request {i + 1}/10...")

        # Track request with automatic latency measurement
        with metrics.track_request_latency(
            model="qwen2.5:0.5b", graph="multi_domain"
        ):
            # Increment request counter
            metrics.increment_request_count(
                model="qwen2.5:0.5b", graph="multi_domain", request_type="generate"
            )

            # Simulate processing time
            await asyncio.sleep(0.1)

            # Record token usage
            metrics.record_tokens(
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                model="qwen2.5:0.5b",
                graph="multi_domain",
            )

    # Simulate some errors
    print("  Simulating errors...")
    for error_type in ["timeout", "connection", "validation"]:
        metrics.increment_error_count(
            error_type=error_type, model="qwen2.5:0.5b", graph="multi_domain"
        )

    # Simulate circuit breaker states
    print("  Simulating circuit breaker state changes...")
    for state in ["closed", "half-open", "open", "closed"]:
        metrics.update_circuit_breaker_state(state=state, model="qwen2.5:0.5b")
        metrics.increment_circuit_breaker_failures(model="qwen2.5:0.5b")
        await asyncio.sleep(0.5)

    # Simulate node executions
    print("  Simulating node executions...")
    for node in ["router", "classifier", "generator", "validator"]:
        with metrics.track_node_execution(node=node, graph="multi_domain"):
            await asyncio.sleep(0.05)

    # Simulate graph execution
    print("  Simulating graph execution...")
    with metrics.track_graph_execution(graph="multi_domain"):
        await asyncio.sleep(0.5)

    # Simulate cache operations
    print("  Simulating cache operations...")
    for _ in range(20):
        if _ % 3 == 0:
            metrics.increment_cache_miss(cache_type="memory")
        else:
            metrics.increment_cache_hit(cache_type="memory")

    # Simulate model loading
    print("  Simulating model load...")
    with metrics.track_model_load(model="qwen2.5:0.5b"):
        await asyncio.sleep(2.0)

    # Simulate memory operations
    print("  Simulating memory operations...")
    for op in ["add", "get", "set", "delete"]:
        for _ in range(5):
            metrics.increment_memory_operation(operation_type=op)

    print("✓ Simulation complete!")


def main():
    """Main function to run the metrics example."""
    print("TinyLLM Metrics Example")
    print("=" * 60)

    # Start metrics server in background thread
    print("\n1. Starting metrics server on port 9090...")
    server_thread = Thread(
        target=start_metrics_server, args=(9090, "127.0.0.1"), daemon=True
    )
    server_thread.start()

    # Give server time to start
    time.sleep(1)

    print("✓ Metrics server started at http://localhost:9090/metrics")
    print("\nYou can view metrics by visiting:")
    print("  - http://localhost:9090/metrics (raw Prometheus format)")
    print("  - Use Prometheus/Grafana for visualization")

    # Run simulations
    print("\n2. Running request simulations...")
    asyncio.run(simulate_requests())

    # Show summary
    print("\n3. Metrics Summary:")
    metrics = get_metrics_collector()
    summary = metrics.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Keep server running
    print("\n" + "=" * 60)
    print("Metrics server is running. Press Ctrl+C to stop.")
    print("Visit http://localhost:9090/metrics to see all metrics.")
    print("=" * 60)

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")


if __name__ == "__main__":
    main()
