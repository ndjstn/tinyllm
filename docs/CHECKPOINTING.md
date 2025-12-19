# Checkpointing for Long-Running Graphs

TinyLLM now supports checkpointing for long-running graph executions, allowing you to save and resume execution state.

## Overview

Checkpointing enables:
- **Fault tolerance**: Resume execution after failures
- **Progress tracking**: Save execution state at regular intervals
- **Debugging**: Inspect intermediate execution states
- **Resource management**: Pause and resume long-running tasks

## Features

- Configurable checkpoint intervals (time-based or per-node)
- Support for multiple storage backends (SQLite, in-memory)
- Automatic checkpoint pruning to manage storage
- Full serialization of execution context, messages, and node state
- CLI commands for listing and resuming from checkpoints

## Storage Backends

### SQLite Storage (Recommended for Production)

Persistent storage using SQLite database:

```python
from tinyllm.persistence.interface import StorageConfig
from tinyllm.persistence.sqlite_backend import SQLiteCheckpointStorage

config = StorageConfig(
    sqlite_path="/path/to/checkpoint/directory"
)
checkpoint_storage = SQLiteCheckpointStorage(config)
await checkpoint_storage.initialize()
```

### In-Memory Storage (For Testing)

Non-persistent storage for development and testing:

```python
from tinyllm.persistence.memory_backend import InMemoryCheckpointStorage

config = StorageConfig()
checkpoint_storage = InMemoryCheckpointStorage(config)
await checkpoint_storage.initialize()
```

## Configuration

### Executor Configuration

Configure checkpointing when creating an Executor:

```python
from tinyllm.core.executor import Executor, ExecutorConfig

executor_config = ExecutorConfig(
    checkpoint_interval_ms=5000,        # Checkpoint every 5 seconds
    checkpoint_after_each_node=True,    # Also checkpoint after each node
)

executor = Executor(
    graph=graph,
    config=executor_config,
    checkpoint_storage=checkpoint_storage,
)
```

### CheckpointConfig Options

- `checkpoint_interval_ms`: Time interval between checkpoints in milliseconds (0 = disabled)
- `checkpoint_after_each_node`: Whether to checkpoint after each node execution
- `max_checkpoints_per_trace`: Maximum number of checkpoints to keep per trace (default: 100)

## Usage

### Basic Execution with Checkpointing

```python
from tinyllm.core.executor import Executor, ExecutorConfig
from tinyllm.core.message import TaskPayload
from tinyllm.persistence.sqlite_backend import SQLiteCheckpointStorage
from tinyllm.persistence.interface import StorageConfig

# Create checkpoint storage
config = StorageConfig(sqlite_path="~/.tinyllm/data")
checkpoint_storage = SQLiteCheckpointStorage(config)
await checkpoint_storage.initialize()

# Create executor with checkpointing
executor_config = ExecutorConfig(
    checkpoint_interval_ms=5000,
    checkpoint_after_each_node=True,
)

executor = Executor(
    graph=graph,
    config=executor_config,
    checkpoint_storage=checkpoint_storage,
)

# Execute task
task = TaskPayload(content="Process this long-running task")
response = await executor.execute(task)

# Checkpoints are automatically saved during execution
print(f"Trace ID: {response.trace_id}")

await executor.close()
await checkpoint_storage.close()
```

### Resuming from Checkpoint

```python
# Resume using the same executor
response = await executor.resume_from_checkpoint(
    trace_id="<trace-id-from-previous-execution>"
)

# Or execute with resume_trace_id parameter
response = await executor.execute(
    task=TaskPayload(content="Resume task"),
    resume_trace_id="<trace-id>"
)
```

### Using CheckpointManager Directly

For more control over checkpoint operations:

```python
from tinyllm.core.checkpoint import CheckpointManager, CheckpointConfig

checkpoint_config = CheckpointConfig(
    checkpoint_interval_ms=5000,
    checkpoint_after_each_node=True,
    max_checkpoints_per_trace=50,
)

manager = CheckpointManager(checkpoint_storage, checkpoint_config)
await manager.initialize()

# Save checkpoint
checkpoint = await manager.save_checkpoint(
    context=execution_context,
    current_node_id="node1",
    force=True,
)

# Load checkpoint
checkpoint = await manager.load_checkpoint(
    graph_id="my-graph",
    trace_id="trace-123",
)

# Restore context
await manager.restore_context(checkpoint, new_context)

# List checkpoints
checkpoints = await manager.list_checkpoints(
    graph_id="my-graph",
    trace_id="trace-123",
)

# Clear checkpoints
count = await manager.clear_checkpoints(
    graph_id="my-graph",
    trace_id="trace-123",
)

await manager.close()
```

## CLI Commands

### Resume from Checkpoint

```bash
tinyllm graph resume <trace-id> \
  --graph graphs/my-graph.yaml \
  --storage /path/to/checkpoints \
  --log-level INFO
```

### List Checkpoints

```bash
tinyllm graph checkpoints \
  --graph-id my-graph \
  --trace-id trace-123 \
  --storage /path/to/checkpoints
```

## Checkpoint Data Structure

Each checkpoint stores:
- **Trace metadata**: Graph ID, trace ID, step number
- **Execution state**: Current node, visited nodes, variables, counters
- **Messages**: All messages in the execution context
- **Node state**: Current node ID and status
- **Timestamps**: Creation and update times

Example checkpoint record:

```python
CheckpointRecord(
    id="checkpoint-uuid",
    graph_id="my-graph",
    trace_id="trace-123",
    step=5,
    state={
        "current_node": "node5",
        "visited_nodes": ["node1", "node2", "node3", "node4", "node5"],
        "variables": {"key": "value"},
        "step_count": 5,
        "total_tokens_in": 100,
        "total_tokens_out": 150,
        "start_time": "2025-12-19T10:00:00",
    },
    messages=[...],
    node_id="node5",
    status="pending",
)
```

## Best Practices

### 1. Choose Appropriate Checkpoint Interval

- For long-running tasks (>1 minute): Use time-based intervals (5-10 seconds)
- For critical workflows: Checkpoint after each node
- For testing: Use in-memory storage with per-node checkpointing

### 2. Storage Management

- Use SQLite storage for production deployments
- Configure `max_checkpoints_per_trace` to prevent unbounded growth
- Periodically clean up old checkpoints for completed executions

### 3. Error Handling

```python
try:
    response = await executor.resume_from_checkpoint(trace_id)
except Exception as e:
    logger.error(f"Failed to resume from checkpoint: {e}")
    # Fall back to new execution
    response = await executor.execute(task)
```

### 4. Monitoring

```python
# Check checkpoint status
checkpoints = await manager.list_checkpoints(graph_id, trace_id)
print(f"Total checkpoints: {len(checkpoints)}")

for cp in checkpoints:
    print(f"Step {cp.step}: {cp.node_id} ({cp.status})")
```

## Performance Considerations

### Checkpoint Overhead

- **Time-based checkpointing**: Minimal overhead, only checks time interval
- **Per-node checkpointing**: Small overhead per node execution (~1-5ms)
- **SQLite storage**: Fast writes with WAL mode enabled (default)
- **In-memory storage**: Negligible overhead for testing

### Optimization Tips

1. **Adjust checkpoint interval** based on task duration
2. **Use SQLite WAL mode** for better concurrency (enabled by default)
3. **Limit max checkpoints** to avoid database growth
4. **Disable checkpointing** for short-running graphs

## Troubleshooting

### Checkpoint Not Found

If resume fails with "no checkpoint found":
- Verify the trace ID is correct
- Check that checkpointing was enabled during execution
- Ensure checkpoint storage path is accessible

### Checkpoint Restore Fails

If context restore fails:
- Verify the graph definition hasn't changed
- Check that the checkpoint node still exists in the graph
- Review checkpoint data for corruption

### Performance Issues

If checkpointing slows execution:
- Increase checkpoint interval
- Disable per-node checkpointing
- Use in-memory storage for non-critical workflows

## Example Workflows

### Fault-Tolerant Data Processing

```python
async def process_large_dataset():
    checkpoint_storage = SQLiteCheckpointStorage(
        StorageConfig(sqlite_path="./checkpoints")
    )
    await checkpoint_storage.initialize()

    executor = Executor(
        graph=processing_graph,
        config=ExecutorConfig(
            checkpoint_interval_ms=10000,  # Checkpoint every 10s
            checkpoint_after_each_node=True,
        ),
        checkpoint_storage=checkpoint_storage,
    )

    try:
        response = await executor.execute(
            TaskPayload(content="Process 1M records")
        )
        return response
    except Exception as e:
        # Resume from last checkpoint on failure
        logger.error(f"Execution failed: {e}")
        latest = await checkpoint_storage.get_latest_checkpoint(
            graph_id=processing_graph.id,
            trace_id=response.trace_id,
        )
        if latest:
            return await executor.resume_from_checkpoint(latest.trace_id)
        raise
    finally:
        await executor.close()
        await checkpoint_storage.close()
```

### Multi-Stage Pipeline with Checkpoints

```python
async def multi_stage_pipeline():
    # Stage 1: Data ingestion
    response1 = await executor.execute(TaskPayload(content="Ingest data"))

    # Save checkpoint between stages
    checkpoint = await manager.save_checkpoint(
        context=executor._context,
        current_node_id="ingestion_complete",
        force=True,
    )

    # Stage 2: Processing (can be resumed if it fails)
    response2 = await executor.execute(
        TaskPayload(content="Process data"),
        resume_trace_id=response1.trace_id,
    )

    return response2
```

## API Reference

See the docstrings in:
- `/home/uri/Desktop/tinyllm/src/tinyllm/core/checkpoint.py` - CheckpointManager class
- `/home/uri/Desktop/tinyllm/src/tinyllm/core/executor.py` - Executor with checkpointing
- `/home/uri/Desktop/tinyllm/src/tinyllm/persistence/interface.py` - Storage interfaces
- `/home/uri/Desktop/tinyllm/src/tinyllm/persistence/sqlite_backend.py` - SQLite implementation
- `/home/uri/Desktop/tinyllm/src/tinyllm/persistence/memory_backend.py` - In-memory implementation
