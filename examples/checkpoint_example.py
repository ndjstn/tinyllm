"""Example demonstrating checkpoint functionality for long-running graphs.

This example shows how to:
1. Create an executor with checkpointing enabled
2. Save checkpoints during execution
3. Resume execution from a checkpoint
"""

import asyncio
from pathlib import Path

from tinyllm.core.builder import load_graph
from tinyllm.core.executor import Executor, ExecutorConfig
from tinyllm.core.message import TaskPayload
from tinyllm.persistence.interface import StorageConfig
from tinyllm.persistence.sqlite_backend import SQLiteCheckpointStorage
from tinyllm.persistence.memory_backend import InMemoryCheckpointStorage


async def example_with_sqlite():
    """Example using SQLite checkpoint storage."""
    print("=== SQLite Checkpoint Example ===\n")

    # Create checkpoint storage
    config = StorageConfig(sqlite_path=str(Path.home() / ".tinyllm" / "data"))
    checkpoint_storage = SQLiteCheckpointStorage(config)
    await checkpoint_storage.initialize()

    try:
        # Load a graph
        graph_path = Path("graphs/multi_domain.yaml")
        if not graph_path.exists():
            print(f"Graph file not found: {graph_path}")
            return

        graph = load_graph(graph_path)

        # Create executor with checkpointing enabled
        executor_config = ExecutorConfig(
            checkpoint_interval_ms=5000,  # Checkpoint every 5 seconds
            checkpoint_after_each_node=True,  # Also checkpoint after each node
        )

        executor = Executor(
            graph=graph,
            config=executor_config,
            checkpoint_storage=checkpoint_storage,
        )

        # Execute a task
        task = TaskPayload(content="What is the capital of France?")
        print(f"Executing task: {task.content}")

        response = await executor.execute(task)

        print(f"\nTrace ID: {response.trace_id}")
        print(f"Success: {response.success}")
        print(f"Nodes executed: {response.nodes_executed}")
        print(f"Total latency: {response.total_latency_ms}ms")

        if response.success:
            print(f"Response: {response.content}")

            # List checkpoints created
            checkpoints = await checkpoint_storage.list_checkpoints(
                graph_id=graph.id,
                trace_id=response.trace_id,
            )
            print(f"\nCheckpoints created: {len(checkpoints)}")
            for cp in checkpoints:
                print(f"  - Step {cp.step}: {cp.node_id} ({cp.status})")

            # Demonstrate resume capability
            print(f"\n--- Resuming from checkpoint ---")
            print(f"Resuming trace: {response.trace_id}")

            resumed_response = await executor.resume_from_checkpoint(
                trace_id=response.trace_id
            )

            print(f"Resumed successfully: {resumed_response.success}")

        await executor.close()

    finally:
        await checkpoint_storage.close()


async def example_with_memory():
    """Example using in-memory checkpoint storage."""
    print("\n=== In-Memory Checkpoint Example ===\n")

    # Create in-memory checkpoint storage
    config = StorageConfig()
    checkpoint_storage = InMemoryCheckpointStorage(config)
    await checkpoint_storage.initialize()

    try:
        # Load a graph
        graph_path = Path("graphs/multi_domain.yaml")
        if not graph_path.exists():
            print(f"Graph file not found: {graph_path}")
            return

        graph = load_graph(graph_path)

        # Create executor with checkpointing
        executor_config = ExecutorConfig(
            checkpoint_interval_ms=0,  # Disable time-based checkpointing
            checkpoint_after_each_node=True,  # Only checkpoint after each node
        )

        executor = Executor(
            graph=graph,
            config=executor_config,
            checkpoint_storage=checkpoint_storage,
        )

        # Execute a task
        task = TaskPayload(content="Calculate 123 + 456")
        print(f"Executing task: {task.content}")

        response = await executor.execute(task)

        print(f"\nTrace ID: {response.trace_id}")
        print(f"Success: {response.success}")

        if response.success:
            # List checkpoints
            checkpoints = await checkpoint_storage.list_checkpoints(
                graph_id=graph.id,
                trace_id=response.trace_id,
            )
            print(f"Checkpoints in memory: {len(checkpoints)}")

        await executor.close()

    finally:
        await checkpoint_storage.close()


async def example_checkpoint_management():
    """Example showing checkpoint management operations."""
    print("\n=== Checkpoint Management Example ===\n")

    config = StorageConfig()
    checkpoint_storage = InMemoryCheckpointStorage(config)
    await checkpoint_storage.initialize()

    try:
        from tinyllm.core.checkpoint import CheckpointManager, CheckpointConfig
        from tinyllm.core.context import ExecutionContext
        from tinyllm.config.loader import Config

        # Create checkpoint manager
        checkpoint_config = CheckpointConfig(
            checkpoint_interval_ms=0,
            checkpoint_after_each_node=True,
            max_checkpoints_per_trace=5,
        )

        manager = CheckpointManager(checkpoint_storage, checkpoint_config)
        await manager.initialize()

        # Create some execution contexts and save checkpoints
        for i in range(10):
            context = ExecutionContext(
                trace_id=f"trace-{i % 2}",  # Two different traces
                graph_id="test-graph",
                config=Config(),
            )
            context.visit_node(f"node-{i}")
            context.set_variable("step", i)

            checkpoint = await manager.save_checkpoint(
                context=context,
                current_node_id=f"node-{i}",
                force=True,
            )
            print(f"Saved checkpoint {checkpoint.id[:8]}... for trace-{i % 2}, step {i}")

        # List all checkpoints
        all_checkpoints = await manager.list_checkpoints("test-graph")
        print(f"\nTotal checkpoints: {len(all_checkpoints)}")

        # List for specific trace (should be max 5 due to pruning)
        trace0_checkpoints = await manager.list_checkpoints("test-graph", "trace-0")
        print(f"Checkpoints for trace-0: {len(trace0_checkpoints)}")

        trace1_checkpoints = await manager.list_checkpoints("test-graph", "trace-1")
        print(f"Checkpoints for trace-1: {len(trace1_checkpoints)}")

        # Load and restore latest checkpoint
        latest = await manager.load_checkpoint("test-graph", "trace-0")
        if latest:
            print(f"\nLatest checkpoint for trace-0: step {latest.step}")

            # Restore to new context
            new_context = ExecutionContext(
                trace_id="trace-0",
                graph_id="test-graph",
                config=Config(),
            )
            await manager.restore_context(latest, new_context)
            print(f"Restored: step_count={new_context.step_count}, variable step={new_context.get_variable('step')}")

        # Clear checkpoints for a trace
        count = await manager.clear_checkpoints("test-graph", "trace-0")
        print(f"\nCleared {count} checkpoints for trace-0")

        await manager.close()

    finally:
        await checkpoint_storage.close()


async def main():
    """Run all examples."""
    # Note: These examples require a running Ollama instance and graph files.
    # Uncomment the examples you want to run:

    # await example_with_sqlite()
    # await example_with_memory()
    await example_checkpoint_management()


if __name__ == "__main__":
    asyncio.run(main())
