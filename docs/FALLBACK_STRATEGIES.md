# Fallback Model Strategies

A comprehensive guide to implementing resilient model execution in TinyLLM with automatic fallback to alternate models.

## Overview

The Fallback Model Strategy system provides automatic failover to alternate models when the primary model fails or is unavailable. It includes health tracking, multiple routing strategies, and comprehensive metrics for monitoring model performance.

## Features

- **Multiple Strategies**: Sequential, Fastest (racing), and Load-Balanced routing
- **Health Tracking**: Automatic monitoring of model success rates and latency
- **Smart Routing**: Dynamically select the best model based on real-time metrics
- **Comprehensive Metrics**: Track fallback frequency, per-model success rates, and latency
- **Workflow Integration**: Easy configuration in YAML workflow definitions
- **Circuit Breaking**: Automatically skip unhealthy models

## Quick Start

### Basic Configuration

```python
from tinyllm.models.fallback import FallbackClient, FallbackConfig

# Create configuration
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
)

# Create client
client = FallbackClient(config=config)

# Generate with automatic fallback
result = await client.generate(
    prompt="What is Python?",
    temperature=0.7,
)

print(f"Model used: {result.model_used}")
print(f"Response: {result.response.response}")
```

## Fallback Strategies

### 1. Sequential (Default)

Tries models in order, falling back only on failure.

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    strategy=FallbackStrategy.SEQUENTIAL,
)
```

**Use Cases:**
- Preferred model hierarchy (quality over speed)
- Cost optimization (try expensive model first)
- Predictable behavior

**Behavior:**
1. Try primary model
2. On failure, try first fallback
3. On failure, try second fallback
4. Continue until success or all models fail

### 2. Fastest (Racing)

Races all models simultaneously, uses first successful response.

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    strategy=FallbackStrategy.FASTEST,
)
```

**Use Cases:**
- Latency-critical applications
- When any working model is acceptable
- High availability requirements

**Behavior:**
1. Start all models simultaneously
2. First successful response wins
3. Cancel remaining requests
4. Higher resource usage but lower latency

### 3. Load Balanced

Selects models based on health metrics (success rate, latency).

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    strategy=FallbackStrategy.LOAD_BALANCED,
    enable_health_tracking=True,
)
```

**Use Cases:**
- Production systems with varying model performance
- Adaptive routing based on real-time conditions
- Automatic recovery from model degradation

**Behavior:**
1. Order models by health score (success rate + latency)
2. Try healthiest model first
3. Fall back to next healthiest on failure
4. Dynamically reorder based on performance

## Configuration Options

### FallbackConfig

```python
config = FallbackConfig(
    # Required
    primary_model="qwen2.5:3b",

    # Optional
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    retry_on_errors=["timeout", "connection", "rate_limit"],
    timeout_ms=30000,
    strategy=FallbackStrategy.SEQUENTIAL,
    max_retries_per_model=2,
    enable_health_tracking=True,
    health_check_interval_s=60.0,
)
```

**Parameters:**

- `primary_model` (str): Primary model to try first
- `fallback_models` (List[str]): Ordered list of fallback models
- `retry_on_errors` (List[str]): Error patterns to retry on
- `timeout_ms` (int): Per-model timeout in milliseconds
- `strategy` (FallbackStrategy): Routing strategy
- `max_retries_per_model` (int): Max retries per individual model
- `enable_health_tracking` (bool): Enable health metrics tracking
- `health_check_interval_s` (float): Health check interval

## Health Tracking

### How It Works

The health tracker monitors each model's performance:

```python
class ModelHealth:
    success_count: int
    failure_count: int
    total_latency_ms: float
    consecutive_failures: int
    is_healthy: bool
```

**Health Metrics:**
- Success rate (0.0-1.0)
- Average latency (ms)
- Consecutive failures
- Last success/failure time

**Unhealthy Status:**
- Model becomes unhealthy after 3 consecutive failures
- Unhealthy models are skipped during fallback
- Health resets after successful request

### Accessing Health Metrics

```python
# Get comprehensive health metrics
metrics = client.get_health_metrics()

print(f"Overall success rate: {metrics['overall']['overall_success_rate']}")

for model, stats in metrics['per_model'].items():
    print(f"{model}:")
    print(f"  Success rate: {stats['success_rate']}")
    print(f"  Avg latency: {stats['average_latency_ms']}ms")
    print(f"  Healthy: {stats['is_healthy']}")
```

### Fallback Statistics

```python
# Get fallback usage statistics
stats = client.get_fallback_statistics()

print(f"Fallback rate: {stats['fallback_rate']}")
print(f"Models used: {stats['models_used']}")
```

## Workflow Integration

### YAML Configuration

```yaml
nodes:
  - id: "analyzer"
    type: "model"
    config:
      # Standard config
      model: "qwen2.5:3b"
      temperature: 0.7
      max_tokens: 2000

      # Fallback config
      enable_fallback: true
      fallback_models:
        - "qwen2.5:1.5b"
        - "qwen2.5:0.5b"
      fallback_strategy: "sequential"
      fallback_timeout_ms: 30000
```

### Python API

```python
from tinyllm.nodes.model import ModelNode

# ModelNode automatically uses fallback if configured
node_def = NodeDefinition(
    id="analyzer",
    type=NodeType.MODEL,
    config={
        "model": "qwen2.5:3b",
        "enable_fallback": True,
        "fallback_models": ["qwen2.5:1.5b"],
        "fallback_strategy": "sequential",
    }
)

node = ModelNode(node_def)
result = await node.execute(message, context)

# Check which model was used
print(f"Model: {result.metadata['model_used']}")
print(f"Fallback occurred: {result.metadata.get('fallback_occurred')}")
```

## Error Handling

### Retryable Errors

Configure which errors trigger retry:

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b"],
    retry_on_errors=["timeout", "connection", "rate_limit"],
)
```

**Common retryable errors:**
- `timeout` - Request timed out
- `connection` - Network connection failed
- `rate_limit` - API rate limit exceeded
- `server_error` - Server internal error (5xx)

### Non-Retryable Errors

These immediately fall back to next model:
- Invalid model name
- Authentication failures
- Invalid request format

### All Models Fail

```python
try:
    result = await client.generate(prompt="test")
except RuntimeError as e:
    print(f"All models failed: {e}")
    # e.message contains list of attempted models
```

## Best Practices

### 1. Strategy Selection

**Use Sequential when:**
- You have a clear model preference hierarchy
- Cost optimization is important
- Predictable behavior is required

**Use Fastest when:**
- Latency is critical
- Any working model is acceptable
- Higher resource usage is acceptable

**Use Load Balanced when:**
- Running in production
- Models have varying availability
- You want adaptive performance

### 2. Model Chain Design

**Good Chain:**
```python
fallback_models=[
    "gpt-4",           # High quality, expensive
    "gpt-3.5-turbo",   # Balanced
    "qwen2.5:0.5b",    # Fast, cheap fallback
]
```

**Considerations:**
- Order by preference (quality, cost, speed)
- Include at least one reliable fallback
- Balance quality vs availability
- Consider total latency: timeout × num_models

### 3. Timeout Configuration

```python
# Fast models - shorter timeout
config = FallbackConfig(
    primary_model="qwen2.5:0.5b",
    timeout_ms=5000,
)

# Larger models - longer timeout
config = FallbackConfig(
    primary_model="qwen2.5:7b",
    timeout_ms=60000,
)
```

### 4. Health Tracking

**Enable in production:**
```python
config = FallbackConfig(
    enable_health_tracking=True,
    health_check_interval_s=60.0,  # Adjust based on traffic
)
```

**Monitor regularly:**
```python
# Check health every minute
async def monitor_health():
    while True:
        metrics = client.get_health_metrics()
        for model, stats in metrics['per_model'].items():
            if not stats['is_healthy']:
                alert(f"Model {model} is unhealthy!")
        await asyncio.sleep(60)
```

### 5. Metrics and Alerting

**Track key metrics:**
```python
stats = client.get_fallback_statistics()

# Alert if fallback rate is too high
if stats['fallback_rate'] > 0.5:
    alert("High fallback rate - primary model may be down")

# Alert if overall success rate is low
metrics = client.get_health_metrics()
if metrics['overall']['overall_success_rate'] < 0.9:
    alert("Low overall success rate")
```

## Examples

See the examples directory for comprehensive demonstrations:

- `examples/fallback_example.py` - Working code examples
- `examples/fallback_usage_guide.py` - Detailed usage guide
- `examples/workflows/fallback_workflow.yaml` - YAML workflow example

## API Reference

### Classes

#### `FallbackConfig`
Configuration for fallback model chains.

#### `FallbackClient`
Client implementing fallback strategies.

#### `FallbackResult`
Result from fallback execution with metadata.

#### `HealthTracker`
Tracks health metrics across all models.

#### `ModelHealth`
Health metrics for a single model.

### Enums

#### `FallbackStrategy`
- `SEQUENTIAL` - Try models in order
- `FASTEST` - Race all models
- `LOAD_BALANCED` - Use health metrics

## Testing

Run the test suite:

```bash
# Standalone test (no pytest required)
python test_fallback_standalone.py

# Full test suite
pytest tests/unit/test_fallback.py -v
```

## Performance Considerations

### Sequential Strategy
- **Latency**: Sum of timeouts until success
- **Resource Usage**: One model at a time
- **Cost**: Only pay for models actually used

### Fastest Strategy
- **Latency**: Minimum model latency
- **Resource Usage**: All models run simultaneously
- **Cost**: Pay for all attempted models

### Load Balanced Strategy
- **Latency**: Similar to sequential, but with smarter ordering
- **Resource Usage**: One model at a time
- **Cost**: Optimized by using healthiest models first

## Troubleshooting

### Issue: High fallback rate

**Cause:** Primary model frequently failing

**Solution:**
1. Check primary model health metrics
2. Increase timeout if timing out
3. Verify model is available
4. Consider promoting healthy fallback to primary

### Issue: All models failing

**Cause:** Systemic issue affecting all models

**Solution:**
1. Check Ollama server status
2. Verify network connectivity
3. Review error logs for patterns
4. Ensure models are pulled and available

### Issue: Slow response times

**Cause:** Sequential fallback through multiple models

**Solution:**
1. Use FASTEST strategy for lower latency
2. Reduce timeout values
3. Remove slow/unreliable models from chain
4. Enable health tracking to skip unhealthy models

## Contributing

Contributions welcome! Areas for improvement:

- Additional routing strategies
- Custom health scoring algorithms
- Integration with external monitoring systems
- Performance optimizations

## License

MIT License - See LICENSE file for details
