# TinyLLM Metrics Quick Reference

## Start Metrics Server

```bash
# Standalone
tinyllm metrics --port 9090

# With query
tinyllm run "query" --metrics-port 9090
```

## View Metrics
```bash
curl http://localhost:9090/metrics
```

## Python API

```python
from tinyllm.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Request tracking
metrics.increment_request_count(model="model", graph="graph", request_type="generate")
with metrics.track_request_latency(model="model", graph="graph"):
    # ... request code ...
    pass

# Tokens
metrics.record_tokens(input_tokens=100, output_tokens=50, model="model", graph="graph")

# Errors
metrics.increment_error_count(error_type="timeout", model="model", graph="graph")

# Circuit breaker
metrics.update_circuit_breaker_state(state="closed", model="model")  # closed, half-open, open
metrics.increment_circuit_breaker_failures(model="model")

# Nodes
with metrics.track_node_execution(node="node", graph="graph"):
    pass
metrics.increment_node_error_count(node="node", error_type="error", graph="graph")

# Graphs
with metrics.track_graph_execution(graph="graph"):
    pass

# Cache
metrics.increment_cache_hit(cache_type="memory")
metrics.increment_cache_miss(cache_type="memory")

# Model loading
with metrics.track_model_load(model="model"):
    pass

# Rate limiting
metrics.record_rate_limit_wait(wait_time=0.1, model="model")

# Memory
metrics.increment_memory_operation(operation_type="add")
```

## Key Metrics

| Metric | Type | What it tracks |
|--------|------|----------------|
| `tinyllm_requests_total` | Counter | Total requests |
| `tinyllm_request_latency_seconds` | Histogram | Request latency |
| `tinyllm_tokens_input_total` | Counter | Input tokens |
| `tinyllm_tokens_output_total` | Counter | Output tokens |
| `tinyllm_errors_total` | Counter | Errors by type |
| `tinyllm_circuit_breaker_state` | Gauge | Circuit breaker (0/1/2) |
| `tinyllm_active_requests` | Gauge | Active requests |
| `tinyllm_node_executions_total` | Counter | Node executions |
| `tinyllm_graph_executions_total` | Counter | Graph executions |
| `tinyllm_cache_hits_total` | Counter | Cache hits |

## Prometheus Queries

```promql
# Request rate
rate(tinyllm_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(tinyllm_request_latency_seconds_bucket[5m]))

# Error rate
sum(rate(tinyllm_errors_total[5m])) / sum(rate(tinyllm_requests_total[5m]))

# Tokens/second
sum(rate(tinyllm_tokens_output_total[5m]))

# Cache hit rate
sum(rate(tinyllm_cache_hits_total[5m])) / (sum(rate(tinyllm_cache_hits_total[5m])) + sum(rate(tinyllm_cache_misses_total[5m])))
```

## Files

- `/home/uri/Desktop/tinyllm/src/tinyllm/metrics.py` - Main module
- `/home/uri/Desktop/tinyllm/tests/unit/test_metrics.py` - Tests
- `/home/uri/Desktop/tinyllm/examples/metrics_example.py` - Example
- `/home/uri/Desktop/tinyllm/docs/metrics.md` - Full docs
