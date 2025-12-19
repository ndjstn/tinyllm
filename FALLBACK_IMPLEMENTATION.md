# Fallback Model Strategies Implementation

## Summary

This implementation adds comprehensive fallback model strategies to TinyLLM, enabling resilient model execution with automatic failover to alternate models.

## Files Created

### Core Implementation

1. **`src/tinyllm/models/fallback.py`** (850+ lines)
   - `FallbackConfig`: Configuration for fallback chains
   - `FallbackClient`: Main client implementing fallback strategies
   - `FallbackStrategy`: Enum for strategy types (Sequential, Fastest, Load Balanced)
   - `HealthTracker`: Tracks model health metrics
   - `ModelHealth`: Health metrics for individual models
   - `FallbackResult`: Result with metadata about which model was used

### Integration

2. **`src/tinyllm/models/__init__.py`** (Updated)
   - Exports all fallback classes

3. **`src/tinyllm/nodes/model.py`** (Updated)
   - Added fallback configuration to `ModelNodeConfig`
   - Integrated `FallbackClient` into model execution
   - Added fallback metadata to results

### Tests

4. **`tests/unit/test_fallback.py`** (400+ lines)
   - Test suite covering all fallback functionality
   - Tests for health tracking, strategies, and configuration

5. **`test_fallback_standalone.py`**
   - Standalone test script (no pytest required)
   - Validates core functionality

### Examples

6. **`examples/fallback_example.py`**
   - Working code examples for all strategies
   - Demonstrates health tracking and metrics
   - Shows error handling and recovery

7. **`examples/fallback_usage_guide.py`**
   - Comprehensive usage documentation
   - Best practices and recommendations
   - Real-world use cases

8. **`examples/workflows/fallback_workflow.yaml`**
   - YAML workflow demonstrating fallback configuration
   - Shows all three strategies in action

### Documentation

9. **`docs/FALLBACK_STRATEGIES.md`**
   - Complete documentation
   - API reference
   - Configuration guide
   - Best practices
   - Troubleshooting

## Features Implemented

### 1. FallbackConfig Class

```python
FallbackConfig(
    primary_model: str,
    fallback_models: List[str],
    retry_on_errors: List[str],
    timeout_ms: int,
    strategy: FallbackStrategy,
    max_retries_per_model: int,
    enable_health_tracking: bool,
    health_check_interval_s: float,
)
```

### 2. FallbackClient Class

**Strategies:**
- **SEQUENTIAL**: Try models in order
- **FASTEST**: Race all models, use first response
- **LOAD_BALANCED**: Distribute based on model health

**Key Methods:**
- `generate()`: Generate with fallback support
- `get_health_metrics()`: Get per-model health data
- `get_fallback_statistics()`: Get usage statistics

### 3. Health Tracking

**Metrics Tracked:**
- Success count and rate
- Failure count
- Average latency
- Consecutive failures
- Health status

**Behavior:**
- Models become unhealthy after 3 consecutive failures
- Unhealthy models are skipped during fallback
- Health resets after successful request
- Dynamic reordering based on performance

### 4. Integration with Nodes

**YAML Configuration:**
```yaml
nodes:
  - id: "analyzer"
    type: "model"
    config:
      model: "qwen2.5:3b"
      enable_fallback: true
      fallback_models:
        - "qwen2.5:1.5b"
        - "qwen2.5:0.5b"
      fallback_strategy: "sequential"
      fallback_timeout_ms: 30000
```

**Result Metadata:**
- `model_used`: Which model generated the response
- `fallback_occurred`: Whether fallback happened
- `attempts`: List of models attempted
- `total_latency_ms`: Total time including retries

### 5. Comprehensive Metrics

**Health Metrics:**
```python
{
    "per_model": {
        "qwen2.5:3b": {
            "success_count": 42,
            "failure_count": 3,
            "success_rate": 0.933,
            "average_latency_ms": 245.3,
            "is_healthy": true
        }
    },
    "overall": {
        "total_requests": 100,
        "overall_success_rate": 0.95
    }
}
```

**Fallback Statistics:**
```python
{
    "total_requests": 100,
    "fallback_requests": 15,
    "fallback_rate": 0.15,
    "models_used": {
        "qwen2.5:3b": 85,
        "qwen2.5:1.5b": 15
    }
}
```

## Usage Examples

### Basic Usage

```python
from tinyllm.models.fallback import FallbackClient, FallbackConfig

config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
)

client = FallbackClient(config=config)

result = await client.generate(
    prompt="What is Python?",
    temperature=0.7,
)

print(f"Model used: {result.model_used}")
print(f"Fallback occurred: {result.fallback_occurred}")
```

### Sequential Strategy (Default)

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    strategy=FallbackStrategy.SEQUENTIAL,
)
```

**Tries models in order until success.**

### Fastest Strategy

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    strategy=FallbackStrategy.FASTEST,
)
```

**Races all models, uses first response.**

### Load Balanced Strategy

```python
config = FallbackConfig(
    primary_model="qwen2.5:3b",
    fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    strategy=FallbackStrategy.LOAD_BALANCED,
    enable_health_tracking=True,
)
```

**Selects models based on health metrics.**

### Workflow Integration

```yaml
- id: "analyzer"
  type: "model"
  config:
    model: "qwen2.5:3b"
    enable_fallback: true
    fallback_models: ["qwen2.5:1.5b"]
    fallback_strategy: "sequential"
```

## Testing

### Run Tests

```bash
# Standalone test
python test_fallback_standalone.py

# Full test suite
pytest tests/unit/test_fallback.py -v
```

### Test Coverage

- ModelHealth tracking
- HealthTracker functionality
- FallbackConfig validation
- Sequential fallback strategy
- Fastest (racing) strategy
- Load balanced strategy
- Health-based model skipping
- Metrics and statistics

### Test Results

All tests passing:
- ✓ ModelHealth tests
- ✓ HealthTracker tests
- ✓ FallbackConfig tests
- ✓ Sequential strategy
- ✓ Fastest strategy
- ✓ Load balanced strategy
- ✓ Health recovery

## Architecture

### Class Hierarchy

```
FallbackClient
├── FallbackConfig (configuration)
├── HealthTracker (health monitoring)
│   └── ModelHealth (per-model metrics)
└── OllamaClient (underlying client)

FallbackResult (result with metadata)
```

### Strategy Pattern

Three strategies implemented:
1. `_sequential_fallback()`: Try in order
2. `_fastest_fallback()`: Race all models
3. `_load_balanced_fallback()`: Health-based routing

### Health Tracking Flow

```
Request → Try Model → Success/Failure
    ↓
Update Health Metrics
    ↓
Adjust Model Ordering (Load Balanced)
    ↓
Skip Unhealthy Models
```

## Integration Points

### 1. OllamaClient Integration

Uses existing `OllamaClient` and connection pooling via `get_shared_client()`.

### 2. Logging Integration

Uses TinyLLM's structured logging:
```python
from tinyllm.logging import get_logger
logger = get_logger(__name__, component="fallback")
```

### 3. Node Integration

`ModelNode` automatically uses fallback when configured:
```python
if self._model_config.enable_fallback:
    result = await fallback_client.generate(...)
```

## Performance Characteristics

### Sequential
- **Latency**: Sum of timeouts until success
- **Resources**: One model at a time
- **Best for**: Quality-first, cost optimization

### Fastest
- **Latency**: Minimum model latency
- **Resources**: All models run simultaneously
- **Best for**: Latency-critical applications

### Load Balanced
- **Latency**: Sequential but optimized order
- **Resources**: One model at a time
- **Best for**: Production systems, adaptive routing

## Best Practices

1. **Strategy Selection**
   - Sequential: Most common, predictable costs
   - Fastest: Low latency requirements
   - Load Balanced: Production systems

2. **Model Chain Design**
   - Order by preference (quality → speed)
   - Include reliable fallback
   - Consider total latency

3. **Timeout Configuration**
   - Set based on expected model latency
   - Account for total chain timeout

4. **Health Tracking**
   - Enable in production
   - Monitor metrics regularly
   - Alert on high fallback rates

5. **Error Handling**
   - Configure retryable errors
   - Handle all-models-fail scenario
   - Log failures for debugging

## Future Enhancements

Potential improvements:
- Custom health scoring algorithms
- External monitoring integration
- Cost-aware routing
- Geographic failover
- A/B testing support
- Caching layer
- Async health checks

## Files Summary

```
src/tinyllm/models/
├── fallback.py          (850 lines - core implementation)
└── __init__.py          (updated - exports)

src/tinyllm/nodes/
└── model.py             (updated - integration)

tests/unit/
└── test_fallback.py     (400 lines - test suite)

examples/
├── fallback_example.py       (300 lines - working examples)
├── fallback_usage_guide.py   (400 lines - usage guide)
└── workflows/
    └── fallback_workflow.yaml (100 lines - YAML example)

docs/
└── FALLBACK_STRATEGIES.md    (500 lines - documentation)

test_fallback_standalone.py   (100 lines - standalone test)
```

## Conclusion

This implementation provides a production-ready fallback system with:
- ✓ Three routing strategies
- ✓ Comprehensive health tracking
- ✓ Full workflow integration
- ✓ Extensive test coverage
- ✓ Complete documentation
- ✓ Working examples
- ✓ Best practices guide

The system is ready for immediate use in TinyLLM workflows and can be extended with additional strategies and features as needed.
