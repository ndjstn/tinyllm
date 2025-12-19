# TinyLLM Observability Guide

Comprehensive guide to monitoring, debugging, and understanding TinyLLM systems.

## Table of Contents

- [Overview](#overview)
- [Three Pillars of Observability](#three-pillars-of-observability)
- [Metrics](#metrics)
- [Logging](#logging)
- [Tracing](#tracing)
- [Events](#events)
- [Debug Tools](#debug-tools)
- [Best Practices](#best-practices)
- [Integration Examples](#integration-examples)

## Overview

TinyLLM provides a comprehensive observability stack built on industry-standard tools:

- **Metrics**: Prometheus-compatible metrics for monitoring and alerting
- **Logging**: Structured logging with JSON/console output via structlog
- **Tracing**: Distributed tracing via OpenTelemetry
- **Events**: Structured event system for audit trails and business metrics

These systems work together to provide complete visibility into your LLM workflows.

## Three Pillars of Observability

### 1. Metrics (What is happening?)

Metrics answer questions like:
- How many requests per second?
- What's the error rate?
- How long do operations take?
- How much memory is used?

See [Metrics](#metrics) section for details.

### 2. Logs (Why is it happening?)

Logs answer questions like:
- What exactly failed?
- What was the request content?
- What path did execution take?
- What errors occurred?

See [Logging](#logging) section for details.

### 3. Traces (Where did time go?)

Traces answer questions like:
- Which node took the most time?
- What was the execution path?
- How long did each step take?
- What's the critical path?

See [Tracing](#tracing) section for details.

## Metrics

TinyLLM exposes Prometheus-compatible metrics for monitoring and alerting.

### Starting the Metrics Server

```bash
# Start metrics server on port 9090
tinyllm metrics --port 9090

# Metrics available at http://localhost:9090/metrics
```

### Key Metrics

#### Request Metrics

```
tinyllm_requests_total{model, graph, request_type}
```
Total requests processed by model and graph.

```
tinyllm_request_latency_seconds{model, graph}
```
Request latency histogram in seconds.

```
tinyllm_active_requests{model, graph}
```
Currently active requests.

#### Token Metrics

```
tinyllm_tokens_input_total{model, graph}
```
Total input tokens processed.

```
tinyllm_tokens_output_total{model, graph}
```
Total output tokens generated.

#### Error Metrics

```
tinyllm_errors_total{error_type, model, graph}
```
Total errors by type.

```
tinyllm_node_errors_total{node, graph, error_type}
```
Node-specific errors.

#### Performance Metrics

```
tinyllm_node_execution_duration_seconds{node, graph}
```
Node execution duration histogram.

```
tinyllm_graph_execution_duration_seconds{graph}
```
Graph execution duration histogram.

#### Cache Metrics

```
tinyllm_cache_hits_total{cache_type}
```
Cache hits by cache type.

```
tinyllm_cache_misses_total{cache_type}
```
Cache misses by cache type.

#### Queue Metrics

```
tinyllm_queue_size{priority}
```
Current queue size by priority.

```
tinyllm_queue_wait_time_seconds{priority}
```
Time spent waiting in queue.

### Cardinality Controls

TinyLLM includes built-in cardinality limits to prevent metric explosion:

```python
from tinyllm.metrics import MetricsCollector

# Create collector with cardinality limit
collector = MetricsCollector(max_cardinality=1000)

# Get cardinality stats
stats = collector.get_cardinality_stats()
print(stats)
```

When cardinality limits are exceeded, new label combinations are automatically mapped to fallback labels ("other") to prevent memory issues.

### Prometheus Configuration

Example `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'tinyllm'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
```

### Example Queries

```promql
# Request rate per second
rate(tinyllm_requests_total[5m])

# Average latency by model
rate(tinyllm_request_latency_seconds_sum[5m])
  / rate(tinyllm_request_latency_seconds_count[5m])

# Error rate
rate(tinyllm_errors_total[5m])
  / rate(tinyllm_requests_total[5m])

# P95 node execution time
histogram_quantile(0.95,
  rate(tinyllm_node_execution_duration_seconds_bucket[5m]))

# Cache hit rate
rate(tinyllm_cache_hits_total[5m])
  / (rate(tinyllm_cache_hits_total[5m])
     + rate(tinyllm_cache_misses_total[5m]))
```

## Logging

TinyLLM uses structured logging via `structlog` for both human-readable and machine-parseable logs.

### Configuration

```python
from tinyllm.logging import configure_logging

# Console output (development)
configure_logging(
    log_level="INFO",
    log_format="console"
)

# JSON output (production)
configure_logging(
    log_level="INFO",
    log_format="json",
    log_file="/var/log/tinyllm/app.log"
)
```

### Using Loggers

```python
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="executor")

# Simple log
logger.info("task_started", task_id="abc123")

# With context
logger.info(
    "node_executed",
    node_id="entry",
    duration_ms=123,
    tokens_used=45
)

# Error with exception
try:
    risky_operation()
except Exception as e:
    logger.error(
        "operation_failed",
        error=str(e),
        exc_info=True
    )
```

### Trace Correlation

Logs automatically include trace and span IDs when running within a trace context:

```python
from tinyllm.telemetry import trace_span
from tinyllm.logging import get_logger

logger = get_logger(__name__)

with trace_span("my_operation"):
    # This log will include trace_id and span_id
    logger.info("operation_started")
```

Output:
```json
{
  "event": "operation_started",
  "level": "info",
  "timestamp": "2025-12-19T06:45:00.123Z",
  "trace_id": "abc123def456...",
  "span_id": "789012ghi345...",
  "app": "tinyllm",
  "version": "0.1.0"
}
```

### Context Binding

Bind context for all logs in a scope:

```python
from tinyllm.logging import bind_context, unbind_context

# Bind context
bind_context(request_id="req-123", user_id="user-456")

logger.info("processing_request")  # Includes request_id and user_id

# Unbind when done
unbind_context("request_id", "user_id")
```

### Log Sampling

For high-volume scenarios, enable log sampling:

```python
from tinyllm.logging import configure_log_sampling

# Sample 10% of logs
configure_log_sampling(sample_rate=0.1)

# Or limit to 1000 logs per second
configure_log_sampling(max_per_second=1000)

# Combine both strategies
configure_log_sampling(
    sample_rate=0.5,
    max_per_second=500,
    hash_based=True  # Consistent sampling
)
```

### Sensitive Data Redaction

TinyLLM automatically redacts sensitive information:

```python
from tinyllm.logging import get_request_logger

req_logger = get_request_logger()

# Log request (API keys, passwords automatically redacted)
req_logger.log_request(
    request_id="req-123",
    method="POST",
    path="/api/generate",
    headers={"Authorization": "Bearer sk-abc123"},  # Will be redacted
    body={"prompt": "Hello", "api_key": "secret"}   # api_key redacted
)
```

## Tracing

TinyLLM supports distributed tracing via OpenTelemetry.

### Configuration

```python
from tinyllm.telemetry import configure_telemetry, TelemetryConfig

# Console exporter (development)
config = TelemetryConfig(
    enable_tracing=True,
    service_name="tinyllm",
    exporter="console",
    sampling_rate=1.0
)
configure_telemetry(config)

# OTLP exporter (production - Jaeger/Tempo)
config = TelemetryConfig(
    enable_tracing=True,
    service_name="tinyllm",
    exporter="otlp",
    otlp_endpoint="http://localhost:4317",
    sampling_rate=0.1  # Sample 10% of traces
)
configure_telemetry(config)
```

### Creating Spans

```python
from tinyllm.telemetry import trace_span

# Basic span
with trace_span("my_operation"):
    # Your code here
    process_task()

# Span with attributes
with trace_span(
    "node_execution",
    attributes={
        "node_id": "entry",
        "node_type": "router"
    }
):
    result = execute_node()
```

### Decorators for Async Functions

```python
from tinyllm.telemetry import traced

@traced(attributes={"component": "executor"})
async def execute_graph(graph_id: str):
    # Function execution is automatically traced
    return await graph.execute()
```

### Recording Events

```python
from tinyllm.telemetry import record_span_event, set_span_attribute

with trace_span("processing"):
    set_span_attribute("user_id", "user-123")

    record_span_event(
        "cache_miss",
        attributes={"cache_key": "abc"}
    )

    result = fetch_from_db()
```

### Error Recording

```python
from tinyllm.telemetry import set_span_error

with trace_span("risky_operation"):
    try:
        dangerous_code()
    except Exception as e:
        set_span_error(e)
        raise
```

### Jaeger Setup

Run Jaeger locally:

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest

# Access UI at http://localhost:16686
```

## Events

Structured events provide a third pillar for tracking important application occurrences.

### Event Categories

- **SYSTEM**: System-level events (startup, shutdown)
- **EXECUTION**: Graph/node execution events
- **MODEL**: Model-related events
- **CACHE**: Cache operations
- **SECURITY**: Security events
- **PERFORMANCE**: Performance anomalies
- **USER**: User actions
- **INTEGRATION**: External system integrations
- **DATA**: Data operations

### Event Severities

- **DEBUG**: Detailed debugging information
- **INFO**: Informational messages
- **WARNING**: Warning messages
- **ERROR**: Error events
- **CRITICAL**: Critical failures

### Emitting Events

```python
from tinyllm.events import (
    emit_event,
    EventCategory,
    EventSeverity
)

# Basic event
emit_event(
    event_type="node.execution.completed",
    category=EventCategory.EXECUTION,
    severity=EventSeverity.INFO,
    message="Node execution completed successfully",
    data={"node_id": "entry", "duration_ms": 123},
    tags=["execution", "success"]
)

# Error event with exception
from tinyllm.events import emit_error_event

try:
    risky_operation()
except Exception as e:
    emit_error_event(
        message="Operation failed",
        error=e,
        category=EventCategory.EXECUTION,
        data={"node_id": "entry"}
    )
```

### Event Handlers

Events are dispatched to multiple handlers:

```python
from tinyllm.events import (
    get_event_emitter,
    LogEventHandler,
    MetricEventHandler,
    BufferedEventHandler
)

emitter = get_event_emitter()

# Add custom handler
class CustomEventHandler(EventHandler):
    def handle(self, event: Event):
        # Send to external system
        send_to_slack(event)

emitter.add_handler(CustomEventHandler())

# Add buffered handler for testing
buffer = BufferedEventHandler(max_size=1000)
emitter.add_handler(buffer)

# Get buffered events
events = buffer.get_events(
    category=EventCategory.EXECUTION,
    severity=EventSeverity.ERROR,
    limit=10
)
```

### CLI Event Testing

```bash
# Emit test event
tinyllm debug emit-event \
  "test.event" \
  "This is a test event" \
  --severity info \
  --category system
```

## Debug Tools

TinyLLM provides powerful debug commands for inspecting the observability stack.

### Inspect Observability System

```bash
# Inspect all components
tinyllm debug inspect

# Inspect specific component
tinyllm debug inspect --component metrics
tinyllm debug inspect --component events
tinyllm debug inspect --component telemetry
tinyllm debug inspect --component logs

# JSON output
tinyllm debug inspect --json
```

Example output:

```
Metrics System:
  Collector: prometheus
  Registry: default
  Total metrics: 24

Cardinality Stats:
  Total combinations: 15
  Max cardinality: 1000

Per-Metric Cardinality:
    requests_total: 3 combinations, 0 dropped
    errors_total: 2 combinations, 0 dropped

Event System:
  Enabled: True
  Handlers: LogEventHandler, MetricEventHandler

Telemetry System:
  Enabled: False

Logging System:
  Configured: True
  Processors: 8
```

### Test Trace Context

```bash
# Test trace context injection
tinyllm debug test-trace
```

This creates a test trace span and emits logs to verify trace correlation is working.

### View Cache Stats

```bash
# Memory cache stats
tinyllm cache-stats --backend memory

# Redis cache stats
tinyllm cache-stats --backend redis --redis-host localhost
```

### View System Health

```bash
# Check system health
tinyllm health

# JSON output for monitoring
tinyllm health --json
```

## Best Practices

### 1. Use Structured Logging

**Good:**
```python
logger.info(
    "request_processed",
    request_id=req_id,
    duration_ms=duration,
    status="success"
)
```

**Bad:**
```python
logger.info(f"Request {req_id} processed in {duration}ms - success")
```

### 2. Add Context to Spans

```python
with trace_span("node_execution", attributes={
    "node_id": node.id,
    "node_type": node.type,
    "graph_id": graph.id
}):
    result = node.execute()
```

### 3. Use Events for Business Metrics

```python
# Log: Technical details
logger.info("task_completed", task_id=task.id)

# Event: Business significance
emit_event(
    "task.completed",
    category=EventCategory.EXECUTION,
    severity=EventSeverity.INFO,
    message=f"Task {task.id} completed successfully",
    data={"user_id": user.id, "completion_time": time.time()}
)
```

### 4. Monitor Cardinality

Regularly check metric cardinality to prevent memory issues:

```python
from tinyllm.metrics import get_metrics_collector

collector = get_metrics_collector()
stats = collector.get_cardinality_stats()

# Alert if any metric has high cardinality
for metric, info in stats["metrics"].items():
    if info["cardinality"] > 500:
        logger.warning(
            "high_metric_cardinality",
            metric=metric,
            cardinality=info["cardinality"]
        )
```

### 5. Use Sampling in Production

```python
# Sample 10% of traces in production
config = TelemetryConfig(
    enable_tracing=True,
    sampling_rate=0.1
)

# Sample high-volume logs
configure_log_sampling(
    sample_rate=0.1,
    max_per_second=1000
)
```

### 6. Redact Sensitive Data

Always use the request logger for HTTP requests:

```python
from tinyllm.logging import get_request_logger

req_logger = get_request_logger()
req_logger.log_request(request_id, method, path, headers, body)
```

### 7. Correlate Across Systems

Use trace IDs to correlate logs, metrics, and traces:

```python
with trace_span("operation") as span:
    trace_id = span.get_span_context().trace_id

    # Logs include trace_id automatically
    logger.info("step_1")

    # Add trace_id to events
    emit_event(
        "step.completed",
        category=EventCategory.EXECUTION,
        severity=EventSeverity.INFO,
        message="Step completed",
        trace_id=format(trace_id, "032x")
    )
```

## Integration Examples

### Grafana Dashboard

Example Grafana queries:

```promql
# Request rate panel
rate(tinyllm_requests_total[5m])

# Error rate panel
rate(tinyllm_errors_total[5m]) / rate(tinyllm_requests_total[5m])

# Latency heatmap
histogram_quantile(0.95,
  rate(tinyllm_request_latency_seconds_bucket[5m]))
```

### ELK Stack Integration

Send JSON logs to Elasticsearch:

```python
configure_logging(
    log_level="INFO",
    log_format="json",
    log_file="/var/log/tinyllm/app.json"
)
```

Filebeat config:

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/tinyllm/*.json
    json.keys_under_root: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "tinyllm-%{+yyyy.MM.dd}"
```

### Production Stack

Complete observability stack:

```yaml
# docker-compose.yml
version: '3.8'
services:
  # Metrics
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true

  # Tracing
  jaeger:
    image: jaegertracing/all-in-one
    ports:
      - "4317:4317"   # OTLP
      - "16686:16686" # UI

  # Logging
  elasticsearch:
    image: elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  kibana:
    image: kibana:8.5.0
    ports:
      - "5601:5601"
```

### Running with Full Observability

```bash
# Start TinyLLM with all observability features
tinyllm run "Analyze this text" \
  --tracing \
  --otlp-endpoint http://localhost:4317 \
  --metrics-port 9090 \
  --log-format json

# View traces: http://localhost:16686
# View metrics: http://localhost:9090
# View Grafana: http://localhost:3000
```

## Troubleshooting

### Metrics Not Appearing

1. Check metrics server is running:
```bash
curl http://localhost:9090/metrics
```

2. Verify Prometheus is scraping:
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Logs Missing Trace IDs

1. Ensure telemetry is configured:
```python
from tinyllm.telemetry import is_telemetry_enabled
print(is_telemetry_enabled())  # Should be True
```

2. Verify you're inside a trace span:
```python
from tinyllm.telemetry import get_current_trace_id
print(get_current_trace_id())  # Should not be None
```

### High Cardinality Warnings

1. Check cardinality stats:
```bash
tinyllm debug inspect --component metrics
```

2. Reduce label dimensions or increase limit:
```python
collector = MetricsCollector(max_cardinality=2000)
```

### Events Not Being Recorded

1. Check event emitter is enabled:
```python
from tinyllm.events import get_event_emitter
emitter = get_event_emitter()
print(emitter._enabled)  # Should be True
```

2. Verify handlers are registered:
```python
print([h.__class__.__name__ for h in emitter.handlers])
```

## Summary

TinyLLM provides enterprise-grade observability through:

- **Metrics**: Prometheus-compatible metrics with cardinality controls
- **Logging**: Structured logging with trace correlation and sensitive data redaction
- **Tracing**: OpenTelemetry distributed tracing
- **Events**: Structured event system for audit and business metrics
- **Debug Tools**: CLI commands for real-time inspection

This comprehensive stack enables you to monitor, debug, and optimize your LLM workflows with confidence.
