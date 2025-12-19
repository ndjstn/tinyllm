# Request Queuing System with Backpressure

## Overview

This document describes the request queuing system implemented for TinyLLM, providing priority-based request handling with backpressure control, worker pool management, and comprehensive metrics.

## Files Created

1. **`/home/uri/Desktop/tinyllm/src/tinyllm/queue.py`** - Main implementation
2. **`/home/uri/Desktop/tinyllm/tests/unit/test_queue.py`** - Comprehensive test suite (19 tests)
3. **`/home/uri/Desktop/tinyllm/examples/queue_example.py`** - Usage examples

## Features Implemented

### 1. RequestQueue Class

Priority-based request queue with the following features:

- **Three Priority Levels**: HIGH, NORMAL, LOW
- **Configurable Max Queue Size**: Set via `max_size` parameter (0 for unlimited)
- **Backpressure Modes**:
  - `REJECT`: Reject requests when queue is full
  - `BLOCK`: Block until space is available
- **Fair Scheduling**: Priority-based ordering ensures high-priority requests are processed first
- **Queue Depth Metrics**: Track queue size by priority level
- **Position Tracking**: Get estimated queue position for any request

### 2. QueuedExecutor Wrapper

Wraps the graph Executor to provide queuing functionality:

- **Request Submission**: Submit requests via `submit()` (returns future) or `execute()` (awaits completion)
- **Priority Support**: Assign priority to each request
- **Timeout Support**: Optional timeout while waiting in queue
- **Queue Position Reporting**: Check position of queued requests
- **Worker Pool Management**: Configurable number of concurrent workers
- **Graceful Shutdown**: Drain queue before shutdown with optional timeout

### 3. Worker Pool

Multi-worker system for concurrent request processing:

- **Configurable Workers**: Set via `max_workers` parameter
- **Worker Health Monitoring**: Track total processed, errors, and current request
- **Concurrent Execution**: Process multiple requests simultaneously
- **Graceful Shutdown**: Option to drain remaining requests before stopping

### 4. Integration

#### CLI Integration

Added two new options to the `tinyllm run` command:

```bash
# Enable queuing with max queue size of 100 and 5 workers
tinyllm run "your query" --max-queue-size 100 --workers 5

# Metrics can be exposed on port 9090
tinyllm run "your query" --max-queue-size 100 --workers 5 --metrics-port 9090
```

Options:
- `--max-queue-size`: Maximum request queue size (0 for unlimited, enables queuing if > 0)
- `--workers`: Number of concurrent worker threads (default: 5, only used with queuing)

#### Metrics Integration

New Prometheus metrics added to `/home/uri/Desktop/tinyllm/src/tinyllm/metrics.py`:

```python
# Queue size by priority
tinyllm_queue_size{priority="high|normal|low|all"}

# Time spent waiting in queue
tinyllm_queue_wait_time_seconds{priority="high|normal|low"}

# Total requests submitted to queue
tinyllm_queue_requests_total{priority="high|normal|low"}

# Total requests rejected (queue full)
tinyllm_queue_requests_rejected_total

# Number of active workers
tinyllm_queue_active_workers
```

## Usage Examples

### Basic Usage

```python
from tinyllm.core.builder import load_graph
from tinyllm.core.executor import Executor
from tinyllm.core.message import TaskPayload
from tinyllm.queue import QueuedExecutor, Priority

# Load graph and create executor
graph = load_graph("graphs/multi_domain.yaml")
executor = Executor(graph)

# Create queued executor with 3 workers
queued_executor = QueuedExecutor(
    executor=executor,
    max_workers=3,
    max_queue_size=100,
)

# Execute with queue
async with queued_executor.lifespan():
    task = TaskPayload(content="What is Python?")
    response = await queued_executor.execute(task, priority=Priority.NORMAL)
    print(response.content)
```

### Priority-Based Execution

```python
# Submit requests with different priorities
high_future = await queued_executor.submit(
    TaskPayload(content="Urgent task"),
    priority=Priority.HIGH,
)

normal_future = await queued_executor.submit(
    TaskPayload(content="Normal task"),
    priority=Priority.NORMAL,
)

low_future = await queued_executor.submit(
    TaskPayload(content="Low priority task"),
    priority=Priority.LOW,
)

# High priority will be processed first
high_response = await high_future
normal_response = await normal_future
low_response = await low_future
```

### Concurrent Requests

```python
# Submit many requests concurrently
queries = [f"Query {i}" for i in range(10)]

tasks = [
    queued_executor.execute(TaskPayload(content=query))
    for query in queries
]

# Wait for all to complete
responses = await asyncio.gather(*tasks)
```

### Backpressure Handling

```python
from tinyllm.queue import BackpressureMode

# REJECT mode - reject when queue is full
queued_executor = QueuedExecutor(
    executor=executor,
    max_workers=2,
    max_queue_size=5,
    backpressure_mode=BackpressureMode.REJECT,
)

async with queued_executor.lifespan():
    try:
        await queued_executor.submit(TaskPayload(content="test"))
    except asyncio.QueueFull:
        print("Queue is full, request rejected")

# BLOCK mode - wait for space
queued_executor = QueuedExecutor(
    executor=executor,
    max_workers=2,
    max_queue_size=5,
    backpressure_mode=BackpressureMode.BLOCK,
)

async with queued_executor.lifespan():
    # This will block if queue is full
    future = await queued_executor.submit(TaskPayload(content="test"))
```

### Worker Health Monitoring

```python
# Check worker health
health = queued_executor.get_worker_health()

for worker_id, status in health.items():
    print(f"Worker {worker_id}:")
    print(f"  Healthy: {status.is_healthy}")
    print(f"  Total processed: {status.total_processed}")
    print(f"  Total errors: {status.total_errors}")

# Check queue status
status = queued_executor.get_status()
print(f"Active workers: {status.active_workers}/{status.max_workers}")
print(f"Total queued: {status.total_queued}")
print(f"Average wait time: {status.average_wait_time_ms:.2f}ms")
```

## Architecture

### Request Flow

1. **Submission**: Request submitted via `submit()` or `execute()`
2. **Queuing**: Request added to priority queue with assigned priority
3. **Dequeuing**: Worker pulls highest-priority request from queue
4. **Execution**: Worker executes request through graph executor
5. **Completion**: Future resolved with response, request removed from tracking

### Priority Ordering

Requests are ordered by:
1. **Priority** (descending): HIGH > NORMAL > LOW
2. **Timestamp** (ascending): Earlier requests first within same priority

### Backpressure

When queue is full:
- **REJECT mode**: Immediately raise `asyncio.QueueFull` exception
- **BLOCK mode**: Wait until space becomes available (respects timeout)

### Worker Pool

- Workers run in asyncio tasks
- Semaphore limits concurrent executions
- Health tracking per worker
- Graceful shutdown with optional drain

## Testing

Comprehensive test suite with 19 tests covering:

- Priority ordering
- Queue operations (enqueue, dequeue, position tracking)
- Backpressure modes (reject and block)
- Worker pool management
- Concurrent execution
- Graceful shutdown
- Health monitoring
- Timeout handling
- Metrics integration

Run tests:
```bash
pytest tests/unit/test_queue.py -v
```

All 19 tests pass successfully.

## Performance Considerations

1. **Queue Size**: Set appropriate `max_queue_size` based on memory constraints
2. **Worker Count**: More workers = higher concurrency but more resource usage
3. **Priority Distribution**: Avoid starving low-priority requests
4. **Timeouts**: Set reasonable timeouts to prevent indefinite waiting
5. **Metrics Overhead**: Minimal, but consider disabling detailed metrics in extreme load

## Future Enhancements

Potential improvements:
1. Dynamic worker scaling based on queue depth
2. Request deduplication
3. Rate limiting per priority level
4. Queue persistence for crash recovery
5. Dead letter queue for failed requests
6. Advanced scheduling algorithms (weighted fair queuing, etc.)

## Dependencies

- `asyncio`: For async queue and worker management
- `pydantic`: For data validation
- `prometheus_client`: For metrics (already in project)

No new dependencies required.
