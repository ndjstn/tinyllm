# Redis Messaging Backend for TinyLLM

## Overview

The `RedisMessageQueue` class provides a production-ready Redis-based implementation of the `MessageQueue` interface for inter-agent communication in TinyLLM. This backend supports reliable message delivery, priority queuing, message expiration, and both single Redis instance and Redis Cluster deployments.

## Features

### Core Capabilities

1. **Publish/Subscribe Messaging**: Real-time message delivery using Redis Streams and Pub/Sub
2. **Connection Pooling**: Efficient resource usage with configurable connection pools
3. **Redis Cluster Support**: Seamless support for both single-instance and cluster deployments
4. **Message Expiration**: Automatic TTL-based message cleanup
5. **JSON Serialization**: Automatic serialization/deserialization of message payloads
6. **Priority Queuing**: Messages delivered by priority (higher priority first)
7. **Acknowledgement Tracking**: Reliable delivery with message acknowledgement
8. **Pending Message Retrieval**: Query unprocessed messages for any agent

### Architecture

The Redis backend uses multiple Redis data structures for optimal performance:

- **Redis Streams**: Primary storage for messages (`{prefix}:stream:{channel}`)
- **Redis Pub/Sub**: Real-time notifications (`{prefix}:notify:{channel}`)
- **Redis Sorted Sets**: Pending messages index (`{prefix}:pending:{agent_id}`)
- **Redis Hashes**: Message metadata (`{prefix}:msg:{message_id}`)

## Installation

The Redis backend requires the `redis` package, which is already included in TinyLLM's dependencies:

```bash
pip install tinyllm
```

Or if installing from source:

```bash
pip install redis>=5.0.0
```

## Configuration

### Basic Configuration

```python
from tinyllm.persistence.interface import StorageConfig
from tinyllm.persistence.redis_backend import RedisMessageQueue

# Create configuration
config = StorageConfig(
    redis_url="redis://localhost:6379/0",
    redis_prefix="tinyllm",
    ttl_seconds=3600,  # 1 hour message expiration
    max_items=10000,   # Max items in stream
)

# Initialize backend
queue = RedisMessageQueue(config)
await queue.initialize()
```

### Redis Cluster Configuration

```python
config = StorageConfig(
    redis_url="redis-cluster://node1:6379,node2:6379,node3:6379",
    redis_prefix="tinyllm",
)

queue = RedisMessageQueue(config)
await queue.initialize()
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_url` | `str` | `redis://localhost:6379/0` | Redis connection URL |
| `redis_prefix` | `str` | `tinyllm` | Key prefix for Redis keys |
| `ttl_seconds` | `int` | `None` | Default message TTL (None = no expiration) |
| `max_items` | `int` | `10000` | Maximum items in each stream |
| `namespace` | `str` | `default` | Namespace for logical separation |

## Usage

### Publishing Messages

```python
# Publish a targeted message
message = await queue.publish(
    channel="agent_communication",
    payload={"task": "process_data", "data": [1, 2, 3]},
    source_agent="agent_1",
    target_agent="agent_2",  # Specific target
    priority=10,  # Higher priority
    ttl_seconds=300,  # 5 minutes
)

# Publish a broadcast message
broadcast = await queue.publish(
    channel="announcements",
    payload={"type": "system_update", "message": "Restarting in 5 minutes"},
    source_agent="system",
    target_agent=None,  # Broadcast to all subscribers
    priority=5,
)
```

### Subscribing to Messages

```python
# Subscribe to a channel
async for message in queue.subscribe(
    channel="agent_communication",
    agent_id="agent_2",
):
    print(f"Received: {message.payload}")

    # Process the message
    process_message(message)

    # Acknowledge receipt
    await queue.acknowledge(message.id)
```

### Retrieving Pending Messages

```python
# Get all pending messages for an agent
pending = await queue.get_pending(
    agent_id="agent_2",
    channel="agent_communication",  # Optional filter
    limit=100,
)

for msg in pending:
    print(f"Pending message from {msg.source_agent}: {msg.payload}")
    await queue.acknowledge(msg.id)
```

### Message Acknowledgement

```python
# Acknowledge a message
success = await queue.acknowledge(message_id)

if success:
    print("Message acknowledged")
else:
    print("Message not found")
```

### Retrieving Messages

```python
# Get a specific message by ID
message = await queue.get(message_id)

if message:
    print(f"Message: {message.payload}")
    print(f"Priority: {message.priority}")
    print(f"Acknowledged: {message.acknowledged}")
```

### Listing and Filtering Messages

```python
# List all messages
messages = await queue.list(limit=100, offset=0)

# Filter by channel
channel_messages = await queue.list(
    limit=50,
    filters={"channel": "agent_communication"}
)

# Filter by source agent
agent_messages = await queue.list(
    limit=50,
    filters={"source_agent": "agent_1"}
)

# Filter by acknowledgement status
unacked = await queue.list(
    limit=50,
    filters={"acknowledged": False}
)
```

### Counting Messages

```python
# Count all messages
total = await queue.count()

# Count messages with filters
pending_count = await queue.count(
    filters={"acknowledged": False}
)
```

### Deleting Messages

```python
# Delete a specific message
deleted = await queue.delete(message_id)

# Clear all messages
cleared_count = await queue.clear()
print(f"Cleared {cleared_count} messages")
```

### Cleanup

```python
# Always close the connection when done
await queue.close()
```

## Advanced Usage

### Custom TTL per Message

```python
# Short-lived message (30 seconds)
await queue.publish(
    channel="ephemeral",
    payload={"status": "online"},
    source_agent="agent_1",
    ttl_seconds=30,
)

# Long-lived message (24 hours)
await queue.publish(
    channel="persistent",
    payload={"report": "daily_summary"},
    source_agent="agent_1",
    ttl_seconds=86400,
)
```

### Priority-Based Processing

```python
# High priority urgent message
await queue.publish(
    channel="tasks",
    payload={"task": "urgent_fix"},
    source_agent="system",
    priority=100,  # Highest priority
)

# Normal priority message
await queue.publish(
    channel="tasks",
    payload={"task": "routine_check"},
    source_agent="system",
    priority=10,  # Normal priority
)

# Messages are delivered in priority order (100 before 10)
```

### Context Manager Pattern

```python
async with RedisMessageQueue(config) as queue:
    await queue.initialize()

    # Use the queue
    await queue.publish(...)

    # Automatically closed on exit
```

## Performance Considerations

### Connection Pooling

The Redis backend uses connection pooling by default with the following settings:

- **Max connections**: 50 (for single instance)
- **Socket timeout**: 5 seconds
- **Keep-alive**: Enabled
- **Health check interval**: 30 seconds

### Memory Management

- Use appropriate `max_items` to limit stream sizes
- Set `ttl_seconds` to automatically expire old messages
- Regularly acknowledge messages to remove them from pending sets
- Use `clear()` periodically in development/testing

### Scalability

For high-throughput scenarios:

1. **Use Redis Cluster**: Distributes load across multiple nodes
2. **Set appropriate TTLs**: Prevents unbounded memory growth
3. **Batch operations**: Process multiple messages per subscription iteration
4. **Monitor pending queues**: Track `get_pending()` sizes to detect bottlenecks

## Error Handling

The Redis backend raises `RedisError` for all Redis-related failures:

```python
from redis.exceptions import RedisError

try:
    await queue.publish(...)
except RedisError as e:
    logger.error(f"Redis operation failed: {e}")
    # Handle error (retry, fallback, etc.)
```

Common error scenarios:

- **Connection failures**: Redis server unreachable
- **Authentication errors**: Invalid credentials
- **Memory limits**: Redis out of memory
- **Timeout errors**: Operation took too long

## Logging

The Redis backend uses structured logging with the following events:

- `initializing_redis_single`: Single instance initialization
- `initializing_redis_cluster`: Cluster initialization
- `redis_initialized`: Successful initialization
- `redis_initialization_failed`: Initialization error
- `message_published`: Message published successfully
- `message_acknowledged`: Message acknowledged
- `pending_messages_retrieved`: Pending messages fetched
- `messages_cleared`: All messages cleared
- `closing_redis_connection`: Connection cleanup started
- `redis_connection_closed`: Connection cleanup complete

Example log output:

```
2025-12-19T10:30:15.123456Z [info] initializing_redis_single url=redis://localhost:6379/0
2025-12-19T10:30:15.234567Z [info] redis_initialized is_cluster=False
2025-12-19T10:30:16.345678Z [debug] message_published message_id=abc123 channel=tasks source=agent_1 target=agent_2 priority=10
```

## Testing

A test script is provided at `/home/uri/Desktop/tinyllm/test_redis_backend.py`:

```bash
# Make sure Redis is running
docker run -d -p 6379:6379 redis:latest

# Run the test
python test_redis_backend.py
```

The test covers:

- Backend initialization
- Message publishing (targeted and broadcast)
- Message retrieval
- Pending message queries
- Acknowledgement
- Message deletion
- Clearing all messages

## Integration Example

```python
import asyncio
from tinyllm.persistence.interface import StorageConfig
from tinyllm.persistence.redis_backend import RedisMessageQueue


async def agent_worker(agent_id: str, queue: RedisMessageQueue):
    """Example agent worker that processes messages."""

    # Process any pending messages first
    pending = await queue.get_pending(agent_id)
    for msg in pending:
        print(f"[{agent_id}] Processing pending: {msg.payload}")
        await queue.acknowledge(msg.id)

    # Subscribe to real-time messages
    async for msg in queue.subscribe("tasks", agent_id):
        print(f"[{agent_id}] Received: {msg.payload}")

        # Process the task
        result = await process_task(msg.payload)

        # Send response
        await queue.publish(
            channel="results",
            payload={"result": result},
            source_agent=agent_id,
            target_agent=msg.source_agent,
        )

        # Acknowledge completion
        await queue.acknowledge(msg.id)


async def main():
    config = StorageConfig(redis_url="redis://localhost:6379/0")
    queue = RedisMessageQueue(config)

    try:
        await queue.initialize()

        # Start multiple agent workers
        workers = [
            agent_worker("agent_1", queue),
            agent_worker("agent_2", queue),
            agent_worker("agent_3", queue),
        ]

        await asyncio.gather(*workers)

    finally:
        await queue.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Comparison with In-Memory Backend

| Feature | RedisMessageQueue | InMemoryBackend |
|---------|-------------------|-----------------|
| **Persistence** | Yes (survives restarts) | No (memory only) |
| **Distribution** | Yes (multi-process/host) | No (single process) |
| **Scalability** | High (Redis Cluster) | Limited (single machine) |
| **Pub/Sub** | Native Redis Pub/Sub | In-memory queues |
| **Performance** | Network latency | In-memory speed |
| **Use Case** | Production | Testing/Development |

## Troubleshooting

### Connection Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check Redis connection
redis_client = await queue._redis.ping()
```

### Message Not Received

1. Check if message expired (TTL)
2. Verify agent_id matches target_agent
3. Check if message was acknowledged already
4. Verify channel name is correct

### Memory Usage

Monitor Redis memory:

```bash
redis-cli INFO memory
```

Set max memory policy:

```bash
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## Best Practices

1. **Always use TTLs in production** to prevent memory leaks
2. **Acknowledge messages promptly** to keep pending queues small
3. **Use meaningful channel names** for organization
4. **Monitor Redis metrics** (memory, connections, throughput)
5. **Set max_items appropriately** based on message volume
6. **Use Redis Cluster** for high availability
7. **Implement retry logic** for transient failures
8. **Clean up test data** in development environments

## License

Part of TinyLLM - MIT License
