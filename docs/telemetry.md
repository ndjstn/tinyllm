# OpenTelemetry Distributed Tracing

TinyLLM includes built-in support for distributed tracing using OpenTelemetry, allowing you to monitor and debug the execution of your LLM workflows across multiple nodes and services.

## Overview

Distributed tracing provides visibility into:
- **Graph execution flow**: See how messages flow through your graph
- **Node performance**: Measure latency for each node execution
- **LLM API calls**: Track token usage, latency, and errors
- **Error diagnosis**: Pinpoint failures with detailed stack traces
- **Request correlation**: Follow a single request across all components

## Installation

Install the required OpenTelemetry packages:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

Or install TinyLLM with tracing support:

```bash
pip install tinyllm[tracing]  # When available
```

## Quick Start

### Console Output (Development)

The simplest way to get started is with console output:

```python
from tinyllm.telemetry import TelemetryConfig, configure_telemetry

# Configure telemetry
config = TelemetryConfig(
    enable_tracing=True,
    service_name="tinyllm",
    exporter="console",  # Print traces to console
    sampling_rate=1.0,   # Sample 100% of traces
)
configure_telemetry(config)
```

### OTLP Exporter (Production)

For production use, export traces to an observability platform:

```python
config = TelemetryConfig(
    enable_tracing=True,
    service_name="tinyllm",
    exporter="otlp",
    otlp_endpoint="http://localhost:4317",  # Your OTLP collector
    sampling_rate=0.1,  # Sample 10% of traces
)
configure_telemetry(config)
```

### CLI Usage

Enable tracing from the command line:

```bash
# Console output
tinyllm run "Your query" --tracing

# Export to OTLP/Jaeger
tinyllm run "Your query" --tracing --otlp-endpoint http://localhost:4317
```

## Supported Exporters

### 1. Console Exporter

Best for local development and debugging.

```python
TelemetryConfig(
    enable_tracing=True,
    exporter="console",
)
```

**Pros**: No additional infrastructure needed
**Cons**: Not suitable for production

### 2. OTLP Exporter

Export to any OTLP-compatible backend (Jaeger, Tempo, etc.).

```python
TelemetryConfig(
    enable_tracing=True,
    exporter="otlp",
    otlp_endpoint="http://localhost:4317",
)
```

**Pros**: Production-ready, vendor-neutral
**Cons**: Requires OTLP collector

### 3. Jaeger

Jaeger uses OTLP protocol now:

```python
TelemetryConfig(
    enable_tracing=True,
    exporter="jaeger",
    otlp_endpoint="http://localhost:4317",  # Jaeger OTLP endpoint
)
```

## Integration Points

TinyLLM automatically traces the following:

### 1. Graph Execution

Every graph execution creates a root span:

```python
# Automatically traced when using Executor
executor = Executor(graph)
response = await executor.execute(task)
```

**Span attributes**:
- `graph.id`: Graph identifier
- `execution.trace_id`: Internal trace ID
- `task.content`: Task content (truncated)
- `execution.nodes_executed`: Number of nodes executed
- `execution.tokens_used`: Total tokens consumed
- `execution.duration_ms`: Total execution time

### 2. Node Execution

Each node execution becomes a child span:

```python
# Automatically traced for all node types
# Span name: "node.{node_type}" (e.g., "node.model", "node.router")
```

**Span attributes**:
- `node.id`: Node identifier
- `node.type`: Node type
- `execution.step`: Step number in execution

### 3. LLM API Calls

All Ollama API calls are traced:

```python
# Automatically traced in OllamaClient
client = OllamaClient()
response = await client.generate(model="qwen2.5:0.5b", prompt="...")
```

**Span attributes**:
- `llm.model`: Model name
- `llm.prompt_length`: Prompt character count
- `llm.temperature`: Sampling temperature
- `llm.input_tokens`: Input token count
- `llm.output_tokens`: Output token count
- `llm.total_tokens`: Total tokens

## Custom Tracing

### Adding Custom Spans

Use the `trace_span` context manager:

```python
from tinyllm.telemetry import trace_span, set_span_attribute

with trace_span("custom.operation", attributes={"user": "demo"}):
    # Your code here
    result = do_something()
    set_span_attribute("result.size", len(result))
```

### Decorating Functions

Use the `@traced` decorator for async functions:

```python
from tinyllm.telemetry import traced

@traced(span_name="process.data", attributes={"version": "1.0"})
async def process_data(data: str):
    # Function is automatically traced
    return data.upper()
```

### Decorating Methods

Use `@traced_method` for class methods:

```python
from tinyllm.telemetry import traced_method

class DataProcessor:
    @traced_method(attributes={"component": "processor"})
    async def process(self, data: str):
        return data.upper()
```

### Recording Events

Add events to the current span:

```python
from tinyllm.telemetry import record_span_event

record_span_event("cache.miss", {"key": "user:123"})
```

### Setting Attributes

Add attributes to the current span:

```python
from tinyllm.telemetry import set_span_attribute

set_span_attribute("user.id", "123")
set_span_attribute("request.size", 1024)
```

### Error Tracking

Mark spans as errors:

```python
from tinyllm.telemetry import set_span_error

try:
    result = await risky_operation()
except Exception as e:
    set_span_error(e)  # Records exception in span
    raise
```

## Log Correlation

Traces are automatically correlated with logs through trace IDs:

```python
from tinyllm.telemetry import get_current_trace_id

trace_id = get_current_trace_id()
logger.info("processing_request", trace_id=trace_id, user_id=user_id)
```

When using `trace_span` with `add_to_logs=True` (default), trace IDs are automatically added to all log messages within that span.

## Sampling

Control trace sampling to reduce overhead:

```python
# Sample 10% of traces
config = TelemetryConfig(
    enable_tracing=True,
    sampling_rate=0.1,
)
```

**Sampling strategies**:
- `1.0`: Sample all traces (development)
- `0.1`: Sample 10% (moderate traffic)
- `0.01`: Sample 1% (high traffic)

## Visualization

### Jaeger

1. Start Jaeger:
```bash
docker run -d \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

2. Configure TinyLLM:
```python
config = TelemetryConfig(
    enable_tracing=True,
    exporter="otlp",
    otlp_endpoint="http://localhost:4317",
)
```

3. View traces at http://localhost:16686

### Grafana Tempo

1. Add to `docker-compose.yaml`:
```yaml
tempo:
  image: grafana/tempo:latest
  command: ["-config.file=/etc/tempo.yaml"]
  ports:
    - "4317:4317"  # OTLP gRPC
```

2. Configure TinyLLM:
```python
config = TelemetryConfig(
    enable_tracing=True,
    exporter="otlp",
    otlp_endpoint="http://localhost:4317",
)
```

## Best Practices

### 1. Use Descriptive Span Names

```python
# Good
with trace_span("node.router.classify_intent"):
    ...

# Bad
with trace_span("process"):
    ...
```

### 2. Add Meaningful Attributes

```python
# Good
set_span_attribute("model.name", "qwen2.5:0.5b")
set_span_attribute("model.tier", "t0")

# Bad
set_span_attribute("data", str(some_object))
```

### 3. Sample Appropriately

```python
# Development
sampling_rate=1.0  # All traces

# Production (moderate traffic)
sampling_rate=0.1  # 10% of traces

# Production (high traffic)
sampling_rate=0.01  # 1% of traces
```

### 4. Correlate with Logs

Always include trace IDs in logs:

```python
trace_id = get_current_trace_id()
logger.info("operation_completed", trace_id=trace_id, duration_ms=100)
```

### 5. Handle Errors Properly

```python
try:
    result = await operation()
except Exception as e:
    set_span_error(e)  # Record in trace
    logger.error("operation_failed", error=str(e))  # Record in logs
    raise
```

## Performance Considerations

### Overhead

Tracing adds minimal overhead:
- **Span creation**: ~0.1ms
- **Attribute addition**: ~0.01ms per attribute
- **Console export**: Synchronous, adds ~1-5ms
- **OTLP export**: Async, batched, negligible impact

### Optimization Tips

1. **Use sampling** in production to reduce data volume
2. **Batch exports** are automatic with OTLP exporter
3. **Limit attribute size** to avoid large spans
4. **Disable in high-perf scenarios** if needed:
   ```python
   config = TelemetryConfig(enable_tracing=False)
   ```

## Troubleshooting

### Traces Not Appearing

1. Check telemetry is enabled:
```python
from tinyllm.telemetry import is_telemetry_enabled
print(is_telemetry_enabled())  # Should be True
```

2. Verify OTLP endpoint is reachable:
```bash
curl http://localhost:4317
```

3. Check sampling rate:
```python
# Ensure it's not 0.0
config.sampling_rate = 1.0
```

### OpenTelemetry Not Installed

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

### Jaeger Connection Issues

1. Ensure Jaeger is running:
```bash
docker ps | grep jaeger
```

2. Check port 4317 is accessible:
```bash
netstat -an | grep 4317
```

## Examples

See `examples/tracing_example.py` for a complete working example.

## Configuration Reference

### TelemetryConfig

```python
class TelemetryConfig(BaseModel):
    enable_tracing: bool = False        # Enable/disable tracing
    service_name: str = "tinyllm"       # Service name in traces
    exporter: str = "console"           # Exporter type: console, otlp, jaeger
    otlp_endpoint: Optional[str] = None # OTLP endpoint URL
    sampling_rate: float = 1.0          # Sampling rate (0.0-1.0)
```

## Further Reading

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
