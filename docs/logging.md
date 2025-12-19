# Structured Logging in TinyLLM

TinyLLM uses [structlog](https://www.structlog.org/) for structured logging, providing powerful debugging capabilities in development and machine-readable JSON logs in production.

## Quick Start

```python
from tinyllm import configure_logging, get_logger

# Configure logging (call once at startup)
configure_logging(log_level="INFO", log_format="console")

# Get a logger instance
logger = get_logger(__name__)

# Log structured events
logger.info("user_authenticated", user_id="user123", method="oauth")
logger.error("database_error", table="users", error="connection timeout")
```

## Configuration

### Development Mode

For development, use colored console output:

```python
from tinyllm import configure_logging

configure_logging(
    log_level="DEBUG",      # Show all log levels
    log_format="console"    # Colored, human-readable output
)
```

### Production Mode

For production, use JSON output for log aggregation:

```python
from tinyllm import configure_logging

configure_logging(
    log_level="INFO",       # Only INFO and above
    log_format="json"       # Structured JSON output
)
```

### Environment Variables

You can also configure logging via environment variables:

```bash
export TINYLLM_LOG_LEVEL=DEBUG
export TINYLLM_LOG_FORMAT=json
```

## Log Levels

Standard Python log levels are supported:

- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: General informational messages
- `WARNING`: Warning messages for potentially harmful situations
- `ERROR`: Error messages for serious problems
- `CRITICAL`: Critical messages for very serious errors

## Using Loggers

### Basic Logging

```python
from tinyllm import get_logger

logger = get_logger(__name__)

# Simple events
logger.info("server_started")
logger.warning("high_memory_usage")
logger.error("request_failed")

# With structured data
logger.info(
    "query_processed",
    trace_id="abc123",
    duration_ms=150,
    tokens_used=42
)
```

### Logger with Initial Context

You can bind initial context when creating a logger:

```python
logger = get_logger(__name__, component="api", version="1.0")

# All logs from this logger will include component and version
logger.info("request_received")  # Includes component=api, version=1.0
```

### Context Binding

Bind context for the current execution scope (useful for request/trace IDs):

```python
from tinyllm import bind_context, clear_context, get_logger

logger = get_logger(__name__)

# Bind trace context
bind_context(trace_id="abc-123", user_id="user456")

# All subsequent logs will include the bound context
logger.info("step_1_started")
logger.info("step_2_completed")

# Clear context when done
clear_context()
```

## Logging in TinyLLM Components

### Executor Logging

The executor automatically logs:

- Execution start/completion
- Node execution start/completion
- Timeouts and errors
- Performance metrics

```python
from tinyllm.core.executor import Executor

executor = Executor(graph)
response = await executor.execute(task)

# Logs will show:
# - executor_initialized
# - execution_started
# - node_execution_started (for each node)
# - node_execution_completed (for each node)
# - execution_completed
```

### Node Logging

Custom nodes can add their own logging:

```python
from tinyllm import get_logger
from tinyllm.core.node import BaseNode

class MyCustomNode(BaseNode):
    def __init__(self, definition):
        super().__init__(definition)
        self.logger = get_logger(__name__, node_id=self.id)

    async def execute(self, message, context):
        self.logger.info("custom_processing_started", input_size=len(message.payload.content))

        # ... processing logic ...

        self.logger.info("custom_processing_completed", output_size=len(result))
        return result
```

### CLI Logging

The CLI supports logging configuration via flags:

```bash
# Development with debug logs
tinyllm run "query" --log-level DEBUG --log-format console

# Production with JSON logs
tinyllm run "query" --log-level INFO --log-format json
```

## Log Output Examples

### Console Format (Development)

```
2025-12-19T05:57:10.174Z [info    ] execution_started    [tinyllm.core.executor] trace_id=abc123 graph_id=main
2025-12-19T05:57:10.175Z [debug   ] node_execution_started [tinyllm.core.executor] node_id=router step=1
2025-12-19T05:57:10.220Z [debug   ] node_execution_completed [tinyllm.core.executor] node_id=router success=True latency_ms=45
```

### JSON Format (Production)

```json
{"event": "execution_started", "trace_id": "abc123", "graph_id": "main", "level": "info", "timestamp": "2025-12-19T05:57:10.174Z"}
{"event": "node_execution_started", "node_id": "router", "step": 1, "level": "debug", "timestamp": "2025-12-19T05:57:10.175Z"}
{"event": "node_execution_completed", "node_id": "router", "success": true, "latency_ms": 45, "level": "debug", "timestamp": "2025-12-19T05:57:10.220Z"}
```

## Best Practices

### 1. Use Structured Fields

Instead of:
```python
logger.info(f"User {user_id} completed action in {duration}ms")
```

Do:
```python
logger.info("action_completed", user_id=user_id, duration_ms=duration)
```

### 2. Use Consistent Event Names

Use snake_case event names that clearly describe what happened:

```python
logger.info("request_received")
logger.info("database_query_completed")
logger.error("authentication_failed")
```

### 3. Include Relevant Context

Always include enough context to debug issues:

```python
logger.error(
    "node_execution_failed",
    node_id=node.id,
    node_type=node.type,
    error=str(e),
    trace_id=context.trace_id
)
```

### 4. Use Appropriate Log Levels

- `DEBUG`: Internal details, verbose information
- `INFO`: Important state changes, business events
- `WARNING`: Potentially harmful situations that were handled
- `ERROR`: Errors that need attention

### 5. Log Exceptions Properly

Use `exc_info=True` to include stack traces:

```python
try:
    risky_operation()
except Exception as e:
    logger.error("operation_failed", error=str(e), exc_info=True)
```

## Performance Considerations

- Logging is optimized with lazy evaluation
- Logger instances are cached on first use
- Debug logs are only formatted if the log level is enabled
- JSON serialization is fast and efficient

## Integration with Log Aggregation

The JSON format is designed to work with common log aggregation systems:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Grafana Loki**: Time-series log aggregation
- **Datadog**: APM and log management
- **CloudWatch**: AWS log aggregation

Simply configure `log_format="json"` and pipe logs to your aggregation system.

## Debugging Tips

### Enable Debug Logging

```bash
export TINYLLM_LOG_LEVEL=DEBUG
tinyllm run "your query"
```

### Filter Logs by Component

```python
# In your aggregation tool, filter by:
component=executor
component=node
component=cli
```

### Trace Execution Flow

All logs include `trace_id` when executing queries, allowing you to follow the complete execution path.

## Environment Configuration

Create a `.env` file or export environment variables:

```bash
# .env file
TINYLLM_LOG_LEVEL=INFO
TINYLLM_LOG_FORMAT=json
```

Or use docker/kubernetes environment variables:

```yaml
env:
  - name: TINYLLM_LOG_LEVEL
    value: "INFO"
  - name: TINYLLM_LOG_FORMAT
    value: "json"
```
