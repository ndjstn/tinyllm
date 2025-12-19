# Memory Bounds Checking Implementation

## Overview

Added comprehensive memory bounds checking to `ExecutionContext` in `/home/uri/Desktop/tinyllm/src/tinyllm/core/context.py` to prevent unbounded growth and out-of-memory errors during graph execution.

## Components Added

### 1. Exception Classes

#### `BoundsExceededError`
Custom exception raised when memory limits are exceeded.

**Attributes:**
- `limit_type`: Type of limit exceeded (messages, message_size, total_size)
- `current_value`: Current value that exceeded the limit
- `limit_value`: The limit that was exceeded
- `details`: Additional context about the error

**Usage:**
```python
raise BoundsExceededError(
    "Message count 1000 exceeds limit of 1000",
    limit_type="message_count",
    current_value=1000,
    limit_value=1000,
    details={"trace_id": "abc-123"}
)
```

### 2. Data Models

#### `MemoryUsageStats`
Pydantic model providing detailed metrics about context memory consumption.

**Fields:**
- `message_count`: Current number of messages
- `total_size_bytes`: Total memory size in bytes
- `largest_message_bytes`: Size of largest message
- `max_messages`: Maximum allowed messages
- `max_message_size_bytes`: Maximum size per message
- `max_total_size_bytes`: Maximum total size
- `message_count_utilization`: Message count utilization ratio (0.0-1.0)
- `total_size_utilization`: Total size utilization ratio (0.0-1.0)
- `near_message_limit`: True if > 80% of message limit
- `near_size_limit`: True if > 80% of size limit

### 3. Configuration Fields

Added to `ExecutionContext`:

```python
max_messages: int = Field(
    default=1000,
    ge=1,
    description="Maximum number of messages allowed in context"
)

max_message_size_bytes: int = Field(
    default=1_048_576,  # 1 MB
    ge=1024,
    description="Maximum size per message in bytes"
)

max_total_size_bytes: int = Field(
    default=104_857_600,  # 100 MB
    ge=1024,
    description="Maximum total context size in bytes"
)
```

**Default Limits:**
- `max_messages`: 1000 messages
- `max_message_size_bytes`: 1 MB (1,048,576 bytes)
- `max_total_size_bytes`: 100 MB (104,857,600 bytes)

### 4. Public Methods

#### `check_bounds() -> None`
Verifies current state is within configured memory bounds.

**Raises:**
- `BoundsExceededError`: If any limit is exceeded

**Usage:**
```python
ctx.check_bounds()  # Validate current state
```

#### `get_memory_usage() -> MemoryUsageStats`
Returns current memory usage statistics and utilization metrics.

**Returns:**
- `MemoryUsageStats`: Complete usage and utilization data

**Usage:**
```python
stats = ctx.get_memory_usage()
print(f"Messages: {stats.message_count}/{stats.max_messages}")
print(f"Utilization: {stats.message_count_utilization:.1%}")
print(f"Near limit: {stats.near_message_limit}")
```

#### `prune_old_messages(keep_count: Optional[int] = None) -> int`
Removes oldest messages when at or approaching limit.

**Parameters:**
- `keep_count`: Number of messages to keep (default: max_messages // 2)

**Returns:**
- `int`: Number of messages pruned

**Raises:**
- `ValueError`: If keep_count is invalid

**Usage:**
```python
# Prune to keep only 100 most recent messages
removed = ctx.prune_old_messages(keep_count=100)

# Use default strategy (keeps half of max_messages)
removed = ctx.prune_old_messages()
```

### 5. Enhanced Methods

#### `add_message(message: Message) -> None`
Enhanced with automatic bounds checking.

**Behavior:**
1. Checks individual message size against `max_message_size_bytes`
2. Checks message count against `max_messages`
3. Adds message to context
4. Checks total size against `max_total_size_bytes`
5. Logs warning when approaching 80% of any limit
6. Logs debug metrics for every message addition

**Raises:**
- `BoundsExceededError`: If any limit would be exceeded

## Logging and Monitoring

### Error Logs
- `message_size_exceeded`: Individual message too large
- `message_count_exceeded`: Too many messages
- `total_size_exceeded`: Total context size too large
- `bounds_check_failed_*`: Bounds validation failed

### Warning Logs (80% threshold)
- `approaching_message_limit`: Message count ≥ 80% of limit
- `approaching_size_limit`: Total size ≥ 80% of limit

### Info Logs
- `messages_pruned`: Messages were pruned
- `memory_after_pruning`: Memory stats after pruning

### Debug Logs
- `message_added`: Metrics for each message addition
- `memory_usage_calculated`: Memory usage stats computed
- `bounds_check_passed`: Bounds validation successful

## Integration Example

```python
from tinyllm.core.context import ExecutionContext, BoundsExceededError
from tinyllm.config.loader import Config
from tinyllm.core.message import Message, MessagePayload

# Create context with custom limits
config = Config()
ctx = ExecutionContext(
    trace_id='trace-123',
    graph_id='my-graph',
    config=config,
    max_messages=500,           # 500 messages max
    max_message_size_bytes=2_097_152,    # 2 MB per message
    max_total_size_bytes=52_428_800      # 50 MB total
)

try:
    # Add messages with automatic bounds checking
    msg = Message(
        trace_id='trace-123',
        source_node='node-1',
        payload=MessagePayload(content='Hello world')
    )
    ctx.add_message(msg)

    # Check memory usage
    stats = ctx.get_memory_usage()
    if stats.near_message_limit:
        print(f"Warning: {stats.message_count_utilization:.0%} of message limit used")

    # Prune if needed
    if stats.message_count > 400:
        pruned = ctx.prune_old_messages(keep_count=250)
        print(f"Pruned {pruned} old messages")

except BoundsExceededError as e:
    print(f"Limit exceeded: {e.limit_type}")
    print(f"Current: {e.current_value}, Limit: {e.limit_value}")
```

## Production Considerations

### Memory Sizing Guidelines

**Small workloads (single queries):**
- `max_messages`: 100-500
- `max_message_size_bytes`: 1 MB
- `max_total_size_bytes`: 10-50 MB

**Medium workloads (conversations):**
- `max_messages`: 500-2000
- `max_message_size_bytes`: 2 MB
- `max_total_size_bytes`: 50-200 MB

**Large workloads (long-running agents):**
- `max_messages`: 2000-10000
- `max_message_size_bytes`: 5 MB
- `max_total_size_bytes`: 500 MB - 1 GB

### Automatic Pruning Strategy

For long-running executions, implement automatic pruning:

```python
# Check and prune periodically
if ctx.get_memory_usage().message_count_utilization > 0.9:
    ctx.prune_old_messages()  # Default: keep half
```

### Monitoring

Key metrics to monitor in production:
- `message_count_utilization`: Alert when > 80%
- `total_size_utilization`: Alert when > 80%
- `messages_pruned` events: Track pruning frequency
- `bounds_exceeded` errors: Track limit violations

## Testing

Comprehensive test suite available in `/home/uri/Desktop/tinyllm/test_memory_bounds.py`

Run with:
```bash
PYTHONPATH=src python test_memory_bounds.py
```

**Test Coverage:**
1. Memory usage statistics calculation
2. Bounds checking validation
3. Message count limit enforcement
4. Message size limit enforcement
5. Warning threshold (80%) logging
6. Message pruning with custom count
7. Default pruning strategy
8. Configurable limits

All tests pass successfully with proper logging output.

## Implementation Details

### Message Size Calculation

Uses `sys.getsizeof(message.model_dump_json())` for approximate size calculation:
- Includes message object and immediate content
- Provides conservative estimate on error
- Falls back to 1024 bytes if calculation fails

### Warning Threshold

- Warnings logged once when crossing 80% threshold
- Flags reset after pruning to allow re-warning
- Separate tracking for message count and total size warnings

### Thread Safety

Current implementation is not thread-safe. For concurrent access:
- Use separate ExecutionContext per execution/thread
- Or implement external synchronization if sharing contexts

## Future Enhancements

Potential improvements:
1. Configurable warning threshold (currently hard-coded at 80%)
2. Automatic pruning when approaching limits
3. Message compression for older messages
4. More sophisticated pruning strategies (importance-based, time-based)
5. Metrics export to monitoring systems (Prometheus, etc.)
6. Thread-safe implementation for concurrent access
7. Memory pooling to reduce allocation overhead
