# TinyLLM Structured Logging

TinyLLM now includes comprehensive structured logging using [structlog](https://www.structlog.org/). This provides powerful debugging capabilities in development and machine-readable JSON logs for production monitoring.

## Quick Start

```python
from tinyllm import configure_logging, get_logger

# Configure at startup
configure_logging(log_level="INFO", log_format="console")

# Get a logger
logger = get_logger(__name__)

# Log events with structured data
logger.info("query_executed", trace_id="abc123", latency_ms=150)
```

## Features

- **Structured Logging**: All logs include structured key-value pairs for easy filtering and analysis
- **Dual Modes**:
  - Console mode: Colored, human-readable output for development
  - JSON mode: Machine-readable logs for production and log aggregation
- **Context Binding**: Automatically attach context (trace IDs, user IDs) to all logs
- **Performance**: Lazy evaluation and caching for minimal overhead
- **Integration**: Logs added to all critical components (Executor, CLI, Nodes, Model Client)

## Configuration

### Development (Colored Console)

```bash
tinyllm run "query" --log-level DEBUG --log-format console
```

Or in code:
```python
configure_logging(log_level="DEBUG", log_format="console")
```

### Production (JSON)

```bash
export TINYLLM_LOG_LEVEL=INFO
export TINYLLM_LOG_FORMAT=json
tinyllm run "query"
```

Or in code:
```python
configure_logging(log_level="INFO", log_format="json")
```

## What Gets Logged

### Executor Events

- `executor_initialized` - When executor is created
- `execution_started` - Query execution begins
- `execution_completed` - Query execution finishes
- `execution_timeout` - Execution times out
- `execution_error` - Fatal execution error
- `node_execution_started` - Individual node starts
- `node_execution_completed` - Individual node completes
- `node_timeout` - Node times out
- `node_execution_exception` - Node throws exception

### CLI Events

- `system_check_started` - Health check begins
- `ollama_health_check` - Ollama connectivity status
- `query_execution_started` - CLI query starts
- `query_execution_success` - CLI query succeeds
- `query_execution_failed` - CLI query fails
- `graph_file_not_found` - Graph file missing
- `execution_exception` - Unexpected error

### Node Events

- `node_stats_update` - Periodic stats update (every 10 executions)
- `transform_node_initialized` - Transform node created
- `transform_pipeline_started` - Transform processing begins
- `transform_applied` - Individual transform applied
- `transform_pipeline_completed` - All transforms complete

### Model Client Events

- `creating_shared_client` - New client connection created
- `closing_all_clients` - Shutdown cleanup
- `circuit_breaker_half_open` - Circuit breaker testing recovery
- `circuit_breaker_open` - Circuit breaker blocks requests
- `circuit_breaker_opened` - Circuit breaker triggers

## Output Examples

### Console (Development)

```
2025-12-19T05:57:10.174Z [info    ] execution_started         [executor] trace_id=abc123 graph_id=main task_content=What is 2+2?
2025-12-19T05:57:10.175Z [debug   ] node_execution_started    [executor] trace_id=abc123 node_id=router node_type=router step=1
2025-12-19T05:57:10.220Z [debug   ] node_execution_completed  [executor] trace_id=abc123 node_id=router success=True latency_ms=45
2025-12-19T05:57:10.350Z [info    ] execution_completed       [executor] trace_id=abc123 success=True elapsed_ms=176 nodes_executed=3 tokens_used=42
```

### JSON (Production)

```json
{"event":"execution_started","trace_id":"abc123","graph_id":"main","task_content":"What is 2+2?","level":"info","component":"executor","timestamp":"2025-12-19T05:57:10.174Z"}
{"event":"node_execution_started","trace_id":"abc123","node_id":"router","node_type":"router","step":1,"level":"debug","component":"executor","timestamp":"2025-12-19T05:57:10.175Z"}
{"event":"node_execution_completed","trace_id":"abc123","node_id":"router","success":true,"latency_ms":45,"level":"debug","component":"executor","timestamp":"2025-12-19T05:57:10.220Z"}
{"event":"execution_completed","trace_id":"abc123","success":true,"elapsed_ms":176,"nodes_executed":3,"tokens_used":42,"level":"info","component":"executor","timestamp":"2025-12-19T05:57:10.350Z"}
```

## Context Binding

Attach context to all subsequent logs:

```python
from tinyllm.logging import bind_context, clear_context

bind_context(trace_id="abc123", user_id="user456")
logger.info("processing_request")  # Includes trace_id and user_id
logger.info("validation_complete") # Also includes trace_id and user_id
clear_context()
```

## Log Aggregation

The JSON format integrates with:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Grafana Loki**
- **Datadog**
- **AWS CloudWatch**
- **Google Cloud Logging**

Simply pipe JSON logs to your aggregation system:

```bash
tinyllm run "query" --log-format json | your-log-shipper
```

## Environment Variables

- `TINYLLM_LOG_LEVEL`: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `TINYLLM_LOG_FORMAT`: Set format (console or json)

## Documentation

See [docs/logging.md](docs/logging.md) for comprehensive documentation including:

- Best practices
- Custom logging in nodes
- Performance considerations
- Integration examples
- Debugging tips

## Testing

Run the test script to verify logging:

```bash
python test_logging.py
```

Or run the example:

```bash
python examples/logging_example.py
```
