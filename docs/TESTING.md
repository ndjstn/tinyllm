# Testing Infrastructure

This document describes the comprehensive testing infrastructure for TinyLLM.

## Overview

TinyLLM has a multi-layered testing strategy covering:

1. **Unit Tests** - Fast, isolated component tests
2. **Integration Tests** - Tests with real services (Ollama)
3. **Load Tests** - Performance under sustained load
4. **Chaos Tests** - Resilience to failures
5. **Performance Tests** - Regression detection
6. **Stress Tests** - System behavior under extreme conditions

## Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (require external services)
├── stress/            # Stress tests (system limits)
├── load/              # Load testing (sustained throughput)
├── chaos/             # Chaos engineering tests (failure injection)
└── perf/              # Performance regression tests
```

## Running Tests

### Quick Commands

```bash
# Run all tests
make test

# Run specific test suites
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-stress        # Stress tests
make test-load          # Load tests
make test-chaos         # Chaos tests
make test-perf          # Performance tests

# Run with coverage
make test-cov           # Generate coverage report
make test-cov-gate      # Enforce 80% coverage requirement
```

### Direct pytest Commands

```bash
# Run tests with markers
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests
pytest -m "not slow"    # Skip slow tests

# Run specific test files
pytest tests/load/test_load.py -v
pytest tests/chaos/test_chaos.py -v

# Run with coverage
pytest --cov=src/tinyllm --cov-report=html --cov-report=term

# Parallel execution
pytest -n auto          # Use all CPU cores
pytest -n 4             # Use 4 workers

# Retry flaky tests
pytest --reruns 3 --reruns-delay 1
```

## Test Categories

### 1. Unit Tests

**Location**: `tests/unit/`

Fast, isolated tests for individual components.

**Running**:
```bash
make test-unit
# or
pytest tests/unit/ -v
```

**Coverage**: Unit tests should cover:
- Core message handling
- Node execution logic
- Configuration parsing
- Tool implementations
- Graph construction

### 2. Integration Tests

**Location**: `tests/integration/`

**Marker**: `@pytest.mark.integration`

Tests that require external services like Ollama.

**Running**:
```bash
make test-integration
# or
pytest -m integration
```

**Requirements**:
- Running Ollama server at `http://localhost:11434`
- At least one model installed (e.g., `qwen2.5:0.5b`)

**Configuration**:
```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_TEST_MODEL=qwen2.5:0.5b
```

**Auto-skipping**: Tests automatically skip if Ollama is not available.

### 3. Load Tests

**Location**: `tests/load/`

**Marker**: `@pytest.mark.load`

Tests system performance under sustained load.

**Running**:
```bash
make test-load
# or
pytest -m load
```

**Test Categories**:
- **Node Load Testing**: Individual node throughput
- **System Load Testing**: End-to-end performance
- **Ramp-up Testing**: Gradual load increase
- **Spike Testing**: Sudden load spikes
- **Endurance Testing**: Long-running load (marked as `slow`)
- **Memory Leak Detection**: Resource cleanup validation

**Metrics Collected**:
- Operations per second (ops/s)
- Average latency (ms)
- Success rate (%)
- Throughput (requests/s)

### 4. Chaos Tests

**Location**: `tests/chaos/`

**Marker**: `@pytest.mark.chaos`

Chaos engineering tests for system resilience.

**Running**:
```bash
make test-chaos
# or
pytest -m chaos
```

**Failure Scenarios**:
- **Network Chaos**: Random failures, timeouts, packet loss
- **Resource Chaos**: Memory pressure, CPU saturation
- **Data Chaos**: Corrupted payloads, malformed configs
- **Cascading Failures**: Failure propagation
- **Partitioning**: Network partitions
- **Clock Skew**: Timing anomalies
- **Recovery**: Post-failure recovery

**Chaos Injector**:
```python
from tests.chaos.test_chaos import ChaosInjector

# Create injector with 30% failure rate
chaos = ChaosInjector(failure_rate=0.3, latency_ms=100)

# Inject chaos
await chaos.maybe_fail("operation_name")

# Get statistics
stats = chaos.get_stats()
```

### 5. Performance Tests

**Location**: `tests/perf/`

**Marker**: `@pytest.mark.perf`

Performance regression detection with baseline tracking.

**Running**:
```bash
make test-perf
# or
pytest -m perf
```

**Creating Baseline**:
```bash
pytest tests/perf/ --save-baseline
```

**Checking for Regressions**:
```bash
pytest tests/perf/ -v
```

**Metrics Tracked**:
- Message creation speed
- Node execution latency
- Concurrent throughput
- Memory usage patterns
- Scaling characteristics

**Baseline File**: `tests/perf/performance_baseline.json`

**Regression Threshold**: 20% (configurable)

### 6. Stress Tests

**Location**: `tests/stress/`

Tests system behavior under extreme conditions.

**Running**:
```bash
make test-stress
# or
pytest tests/stress/ -v
```

**Scenarios**:
- Many parallel executions (100+)
- Large content processing (100KB+)
- Deep message hierarchies (50+ levels)
- High failure rates (70%+)
- Concurrent mixed workloads

## Coverage Requirements

TinyLLM enforces a minimum of **80% code coverage**.

### Checking Coverage

```bash
# Generate HTML report
make test-cov

# View in browser
open htmlcov/index.html

# Enforce coverage gate
make test-cov-gate
```

### Coverage Configuration

From `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/tinyllm"]
branch = true
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/site-packages/*",
]

[tool.coverage.report]
fail_under = 80.0
show_missing = true
precision = 2
```

### Excluded from Coverage

- Test files
- `__repr__` methods
- Abstract methods (`@abstract`)
- Type checking blocks (`if TYPE_CHECKING:`)
- Import error handlers
- Main execution blocks (`if __name__ == "__main__":`)

## Performance Benchmarking

### Load Test Metrics

Load tests collect detailed metrics:

```python
class LoadTestMetrics:
    def get_summary(self) -> dict:
        return {
            "total_requests": int,
            "successful_requests": int,
            "failed_requests": int,
            "success_rate": float,
            "avg_latency_ms": float,
            "min_latency_ms": float,
            "max_latency_ms": float,
            "duration_s": float,
            "throughput_rps": float,
        }
```

### Performance Baselines

Performance tests track metrics over time:

```python
# Record metric
perf_metrics.record("test_name", "metric_name", value)

# Compare with baseline
comparison = perf_metrics.compare_with_baseline(baseline, threshold=0.2)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev

      - name: Run unit tests
        run: make test-unit

      - name: Run coverage check
        run: make test-cov-gate

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

## Writing Tests

### Best Practices

1. **Use Markers**: Tag tests appropriately
   ```python
   @pytest.mark.integration
   @pytest.mark.slow
   async def test_long_running_operation():
       ...
   ```

2. **Use Fixtures**: Share common setup
   ```python
   @pytest.fixture
   def execution_context():
       return ExecutionContext(...)
   ```

3. **Mock External Services**: For unit tests
   ```python
   from unittest.mock import AsyncMock

   async def test_with_mock():
       mock_client = AsyncMock()
       mock_client.generate.return_value = {"response": "test"}
   ```

4. **Test Error Paths**: Don't just test happy paths
   ```python
   async def test_handles_timeout():
       with pytest.raises(asyncio.TimeoutError):
           await slow_operation()
   ```

5. **Use Descriptive Names**: Test names should describe behavior
   ```python
   async def test_fanout_succeeds_with_majority_strategy():
       ...
   ```

### Test Template

```python
"""Module description."""

import pytest

pytestmark = pytest.mark.unit  # or integration, load, chaos, perf


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return {...}


class TestFeature:
    """Test suite for Feature."""

    @pytest.mark.asyncio
    async def test_feature_behavior(self, sample_data):
        """Test that feature behaves correctly."""
        # Arrange
        ...

        # Act
        result = await feature.execute(...)

        # Assert
        assert result.success is True
        assert result.output == expected
```

## Troubleshooting

### Common Issues

**1. Tests timing out**
```bash
# Increase timeout (default: 300s)
pytest --timeout=600
```

**2. Ollama tests failing**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required model
ollama pull qwen2.5:0.5b

# Set environment variables
export OLLAMA_HOST=http://localhost:11434
```

**3. Coverage below 80%**
```bash
# Generate detailed report
pytest --cov=src/tinyllm --cov-report=term-missing

# View which lines are missing
open htmlcov/index.html
```

**4. Flaky tests**
```bash
# Retry failed tests
pytest --reruns 3 --reruns-delay 1

# Mark as flaky
@pytest.mark.flaky
async def test_sometimes_fails():
    ...
```

**5. Slow test suite**
```bash
# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Run only changed files
pytest --lf  # last failed
pytest --ff  # failed first
```

## Continuous Improvement

### Adding New Tests

1. Choose appropriate directory (`unit/`, `integration/`, etc.)
2. Create test file: `test_<feature>.py`
3. Add appropriate markers
4. Write tests following the template
5. Ensure tests pass: `pytest <test_file>.py -v`
6. Check coverage: `pytest --cov=src/tinyllm <test_file>.py`
7. Update documentation if needed

### Metrics to Track

- **Test Count**: Should grow with codebase
- **Coverage**: Maintain >80%
- **Test Speed**: Keep unit tests fast (<1s each)
- **Flakiness**: Track and fix flaky tests
- **Performance**: Monitor performance test trends

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Chaos Engineering](https://principlesofchaos.org/)
- [Performance Testing Best Practices](https://www.selenium.dev/documentation/test_practices/encouraged/performance/)
