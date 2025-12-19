# TinyLLM Metrics

TinyLLM provides comprehensive Prometheus metrics for monitoring performance, resource usage, and operational health.

## Quick Start

### Starting the Metrics Server

Start a standalone metrics server:

```bash
tinyllm metrics --port 9090
```

Or enable metrics during query execution:

```bash
tinyllm run "What is 2+2?" --metrics-port 9090
```

The metrics endpoint will be available at `http://localhost:9090/metrics`.

### Configuring Prometheus

Add this scrape configuration to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'tinyllm'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

## Available Metrics

### System Information

- `tinyllm_system_info`: System metadata (version, app name)

### Request Metrics

- `tinyllm_requests_total`: Total requests processed
  - Labels: `model`, `graph`, `request_type`

- `tinyllm_request_latency_seconds`: Request latency histogram
  - Labels: `model`, `graph`
  - Buckets: 5ms to 60s

- `tinyllm_active_requests`: Currently active requests (gauge)
  - Labels: `model`, `graph`

### Token Metrics

- `tinyllm_tokens_input_total`: Total input tokens processed
  - Labels: `model`, `graph`

- `tinyllm_tokens_output_total`: Total output tokens generated
  - Labels: `model`, `graph`

### Error Metrics

- `tinyllm_errors_total`: Total errors by type
  - Labels: `error_type`, `model`, `graph`

### Circuit Breaker Metrics

- `tinyllm_circuit_breaker_state`: Circuit breaker state (0=closed, 1=half-open, 2=open)
  - Labels: `model`

- `tinyllm_circuit_breaker_failures_total`: Total circuit breaker failures
  - Labels: `model`

### Model Loading Metrics

- `tinyllm_model_load_duration_seconds`: Model load time histogram
  - Labels: `model`
  - Buckets: 0.1s to 60s

### Node Execution Metrics

- `tinyllm_node_executions_total`: Total node executions
  - Labels: `node`, `graph`

- `tinyllm_node_execution_duration_seconds`: Node execution duration
  - Labels: `node`, `graph`
  - Buckets: 10ms to 10s

- `tinyllm_node_errors_total`: Node execution errors
  - Labels: `node`, `graph`, `error_type`

### Graph Execution Metrics

- `tinyllm_graph_executions_total`: Total graph executions
  - Labels: `graph`

- `tinyllm_graph_execution_duration_seconds`: Graph execution duration
  - Labels: `graph`
  - Buckets: 0.1s to 120s

### Cache Metrics

- `tinyllm_cache_hits_total`: Cache hits
  - Labels: `cache_type`

- `tinyllm_cache_misses_total`: Cache misses
  - Labels: `cache_type`

### Rate Limiter Metrics

- `tinyllm_rate_limit_wait_seconds`: Time spent waiting for rate limiter
  - Labels: `model`
  - Buckets: 1ms to 5s

### Memory Metrics

- `tinyllm_memory_operations_total`: Memory operations
  - Labels: `operation_type`

### Queue Metrics

- `tinyllm_queue_size`: Current queue size (gauge)
  - Labels: `priority`

- `tinyllm_queue_wait_time_seconds`: Queue wait time
  - Labels: `priority`

- `tinyllm_queue_requests_total`: Total queue requests
  - Labels: `priority`

- `tinyllm_queue_requests_rejected_total`: Rejected requests (full queue)

- `tinyllm_queue_active_workers`: Active worker threads (gauge)

## Usage in Code

### Basic Usage

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Increment request count
metrics.increment_request_count(
    model="qwen2.5:0.5b",
    graph="multi_domain",
    request_type="generate"
)

# Record token usage
metrics.record_tokens(
    input_tokens=100,
    output_tokens=50,
    model="qwen2.5:0.5b",
    graph="multi_domain"
)
```

### Tracking Request Latency

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Track request latency automatically
with metrics.track_request_latency(model="qwen2.5:0.5b", graph="test"):
    # Your request handling code
    response = await client.generate(...)
```

### Tracking Node Execution

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Track node execution
with metrics.track_node_execution(node="router", graph="test"):
    result = await node.execute(message, context)
```

### Tracking Graph Execution

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Track entire graph execution
with metrics.track_graph_execution(graph="multi_domain"):
    response = await executor.execute(task)
```

### Error Tracking

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

try:
    response = await client.generate(...)
except TimeoutError:
    metrics.increment_error_count(
        error_type="timeout",
        model="qwen2.5:0.5b",
        graph="test"
    )
except Exception as e:
    metrics.increment_error_count(
        error_type=type(e).__name__,
        model="qwen2.5:0.5b",
        graph="test"
    )
```

### Circuit Breaker State

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Update circuit breaker state
metrics.update_circuit_breaker_state(
    state="open",  # closed, half-open, or open
    model="qwen2.5:0.5b"
)

# Increment failure count
metrics.increment_circuit_breaker_failures(model="qwen2.5:0.5b")
```

## Useful Prometheus Queries

### Request Rate

```promql
# Requests per second by model
rate(tinyllm_requests_total[5m])

# Requests per second by graph
sum(rate(tinyllm_requests_total[5m])) by (graph)
```

### Latency Percentiles

```promql
# 95th percentile latency
histogram_quantile(0.95, rate(tinyllm_request_latency_seconds_bucket[5m]))

# 99th percentile latency by model
histogram_quantile(0.99, rate(tinyllm_request_latency_seconds_bucket[5m])) by (model)
```

### Error Rate

```promql
# Error rate percentage
sum(rate(tinyllm_errors_total[5m])) / sum(rate(tinyllm_requests_total[5m])) * 100

# Errors by type
sum(rate(tinyllm_errors_total[5m])) by (error_type)
```

### Token Throughput

```promql
# Total tokens per second
sum(rate(tinyllm_tokens_input_total[5m]) + rate(tinyllm_tokens_output_total[5m]))

# Tokens per second by model
sum(rate(tinyllm_tokens_output_total[5m])) by (model)
```

### Circuit Breaker Health

```promql
# Models with open circuit breakers
tinyllm_circuit_breaker_state == 2

# Circuit breaker failure rate
rate(tinyllm_circuit_breaker_failures_total[5m])
```

### Cache Performance

```promql
# Cache hit rate
sum(rate(tinyllm_cache_hits_total[5m])) /
(sum(rate(tinyllm_cache_hits_total[5m])) + sum(rate(tinyllm_cache_misses_total[5m])))

# Cache hits vs misses
sum(rate(tinyllm_cache_hits_total[5m])) by (cache_type)
sum(rate(tinyllm_cache_misses_total[5m])) by (cache_type)
```

### Node Performance

```promql
# Node execution rate
sum(rate(tinyllm_node_executions_total[5m])) by (node)

# Slowest nodes (95th percentile)
histogram_quantile(0.95, rate(tinyllm_node_execution_duration_seconds_bucket[5m])) by (node)

# Node error rate
sum(rate(tinyllm_node_errors_total[5m])) by (node, error_type)
```

### Queue Metrics

```promql
# Current queue size
tinyllm_queue_size

# Average queue wait time
rate(tinyllm_queue_wait_time_seconds_sum[5m]) / rate(tinyllm_queue_wait_time_seconds_count[5m])

# Queue rejection rate
rate(tinyllm_queue_requests_rejected_total[5m])
```

## Grafana Dashboards

### Creating a Dashboard

1. Import the metrics as a Prometheus data source
2. Create panels using the queries above
3. Set up alerts for critical metrics

### Recommended Panels

1. **Request Rate**: Line graph of requests/second
2. **Latency**: Heatmap or line graph with p50, p95, p99
3. **Error Rate**: Gauge showing error percentage
4. **Token Throughput**: Area graph of tokens/second
5. **Circuit Breaker Status**: State graph with alerts
6. **Active Requests**: Gauge showing current load
7. **Cache Hit Rate**: Gauge showing cache efficiency
8. **Node Performance**: Table of node execution times

## Production Considerations

### Resource Usage

The metrics collector is lightweight and designed for production use:
- Singleton pattern ensures only one instance
- Metrics stored in-memory by Prometheus client
- HTTP server runs in a separate thread
- Minimal overhead per metric update

### Best Practices

1. **Label Cardinality**: Keep label values bounded (don't use trace IDs as labels)
2. **Scrape Interval**: Use 15-30s intervals for most use cases
3. **Retention**: Configure Prometheus retention based on your needs
4. **Alerts**: Set up alerts for error rates, latencies, and circuit breakers
5. **Dashboards**: Create role-specific dashboards (ops, dev, business)

### Security

- Bind to `127.0.0.1` for local-only access
- Use `0.0.0.0` only if behind a firewall/proxy
- Consider adding authentication via reverse proxy
- Don't expose metrics publicly without protection

### High Availability

For HA setups:
- Each TinyLLM instance exposes its own metrics
- Use Prometheus federation or service discovery
- Aggregate metrics in Grafana or Prometheus recording rules
- Monitor individual instances separately

## Troubleshooting

### Metrics Not Appearing

1. Check server is running: `curl http://localhost:9090/metrics`
2. Verify Prometheus can reach the endpoint
3. Check for firewall rules blocking access
4. Ensure metrics are being generated (run some queries)

### High Memory Usage

- Check metric label cardinality
- Review scrape interval (longer = less frequent)
- Set Prometheus retention limits
- Consider using recording rules for complex queries

### Missing Labels

- Ensure all code paths set proper labels
- Check for typos in label names
- Verify labels are consistent across metrics

## Example Integration

See `/home/uri/Desktop/tinyllm/src/tinyllm/models/client.py` for a complete example of metrics integration in the OllamaClient.

Key integration points:
- Request counting before execution
- Latency tracking with context manager
- Token counting after successful requests
- Error tracking on exceptions
- Circuit breaker state updates
