#!/usr/bin/env python3
"""Example demonstrating TinyLLM structured logging capabilities.

This example shows how to:
1. Configure logging for development vs production
2. Use structured logging with context
3. Log throughout the execution pipeline
"""

import asyncio
from pathlib import Path

from tinyllm import configure_logging, get_logger
from tinyllm.core.builder import load_graph
from tinyllm.core.executor import Executor
from tinyllm.core.message import TaskPayload


async def main():
    # Example 1: Development logging (colored console output)
    print("=== Example 1: Development Mode ===\n")
    configure_logging(log_level="DEBUG", log_format="console")

    logger = get_logger(__name__)
    logger.info("application_started", mode="development")

    # Create a simple task
    task = TaskPayload(content="What is 2 + 2?")

    # Load graph and execute (this will show all debug logs)
    try:
        graph_path = Path("graphs/simple_calc.yaml")
        if graph_path.exists():
            graph = load_graph(graph_path)
            executor = Executor(graph)

            logger.info("executing_query", query=task.content)
            response = await executor.execute(task)

            logger.info(
                "query_completed",
                success=response.success,
                latency_ms=response.total_latency_ms,
                nodes_executed=response.nodes_executed,
            )
        else:
            logger.warning("graph_not_found", path=str(graph_path))
    except Exception as e:
        logger.error("execution_failed", error=str(e), exc_info=True)

    print("\n" + "=" * 60 + "\n")

    # Example 2: Production logging (JSON output)
    print("=== Example 2: Production Mode ===\n")
    configure_logging(log_level="INFO", log_format="json")

    logger = get_logger(__name__)
    logger.info("application_started", mode="production", environment="prod")

    # Simulate some operations
    logger.info("database_connection", status="connected", pool_size=10)
    logger.info("cache_initialized", backend="redis", ttl_seconds=300)
    logger.info("api_server_ready", port=8000, workers=4)

    # Simulate request processing
    logger.info(
        "request_received",
        method="POST",
        path="/api/v1/execute",
        client_ip="192.168.1.100",
    )

    logger.info(
        "request_completed",
        method="POST",
        path="/api/v1/execute",
        status_code=200,
        duration_ms=145,
    )

    print("\n" + "=" * 60 + "\n")

    # Example 3: Using context binding for tracing
    print("=== Example 3: Context Binding ===\n")
    configure_logging(log_level="INFO", log_format="console")

    from tinyllm.logging import bind_context, clear_context

    logger = get_logger(__name__)

    # Simulate processing multiple requests with trace IDs
    for i in range(3):
        trace_id = f"trace-{i+1:03d}"
        bind_context(trace_id=trace_id, request_id=f"req-{i+1}")

        logger.info("request_started")
        logger.info("validation_passed", fields_validated=5)
        logger.info("processing_query")
        logger.info("request_completed", duration_ms=50 + i * 10)

        clear_context()
        print()


if __name__ == "__main__":
    asyncio.run(main())
