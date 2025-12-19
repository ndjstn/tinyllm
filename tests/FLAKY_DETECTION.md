# Flaky Test Detection and Quarantine Guide

TinyLLM automatically detects flaky tests and provides tools to quarantine them, preventing them from blocking CI/CD pipelines while they're being fixed.

## Quick Start

### Detect Flaky Tests

Run your test suite normally. The flaky detection plugin automatically tracks test results:

```bash
pytest tests/
```

After several runs, the plugin will report detected flaky tests:

```
==================================================
FLAKY TESTS DETECTED: 3
==================================================
  tests/unit/test_network.py::test_api_request
    Failure rate: 40.0%
    Recent results: passed failed passed failed passed

  tests/integration/test_database.py::test_connection
    Failure rate: 30.0%
    Recent results: passed passed failed passed passed
```

### Quarantine Flaky Tests

Mark identified flaky tests with the `@pytest.mark.quarantine` decorator:

```python
import pytest

@pytest.mark.quarantine
def test_flaky_network_request():
    """This test is flaky due to network timing issues."""
    # Test code here
    pass
```

### Skip Quarantined Tests in CI

```bash
# Skip quarantined tests
pytest --skip-quarantine

# Or in CI environment
if [ "$CI" = "true" ]; then
    pytest --skip-quarantine
else
    pytest
fi
```

## How It Works

### Flaky Detection Algorithm

A test is considered flaky if:
1. It has both passes and failures in recent history (last 10 runs)
2. Its failure rate is ≥30% (configurable)
3. It has at least 3 recorded runs

### Quarantine Recommendation

A test is recommended for quarantine if:
1. It meets the flaky criteria
2. It has been run at least 5 times

### History Tracking

Test results are stored in `.pytest_flaky_history.json`:

```json
{
  "test_results": {
    "tests/unit/test_example.py::test_flaky": [
      "passed",
      "failed",
      "passed",
      "failed",
      "passed"
    ]
  },
  "last_updated": "2025-01-15T10:30:00"
}
```

## Configuration

### Command-Line Options

```bash
# Skip quarantined tests
pytest --skip-quarantine

# Auto-retry failed tests (requires pytest-rerunfailures)
pytest --reruns 3 --reruns-delay 1

# Generate detailed flaky test report
pytest --flaky-report
```

### Mark Individual Tests as Flaky

```python
# Mark with automatic retry
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_potentially_flaky():
    """This test may fail occasionally but should retry."""
    pass
```

### Configure Flaky Threshold

Edit `tests/conftest_flaky.py`:

```python
# Default threshold is 30%
tracker = FlakyTestTracker()
is_flaky = tracker.is_flaky(test_id, threshold=0.2)  # 20% threshold
```

## Common Flaky Test Patterns

### 1. Network/API Calls

**Problem**: External services may be slow or unavailable.

```python
# Bad: No retry logic
def test_api_call():
    response = requests.get("https://api.example.com/data")
    assert response.status_code == 200

# Good: Add retries and timeouts
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_api_call():
    response = requests.get(
        "https://api.example.com/data",
        timeout=5
    )
    assert response.status_code == 200
```

### 2. Timing/Race Conditions

**Problem**: Tests depend on timing or async operations.

```python
# Bad: Hard-coded sleep
def test_async_operation():
    start_operation()
    time.sleep(1)  # Might not be enough
    assert operation_complete()

# Good: Poll with timeout
def test_async_operation():
    start_operation()
    for _ in range(10):
        if operation_complete():
            break
        time.sleep(0.1)
    else:
        pytest.fail("Operation did not complete in time")
```

### 3. Shared State

**Problem**: Tests modify global state affecting other tests.

```python
# Bad: Global state
DATABASE = {}

def test_create_user():
    DATABASE["user1"] = {"name": "Alice"}
    assert "user1" in DATABASE

# Good: Isolated fixtures
@pytest.fixture
def db():
    return {}

def test_create_user(db):
    db["user1"] = {"name": "Alice"}
    assert "user1" in db
```

### 4. Resource Leaks

**Problem**: Tests don't clean up resources properly.

```python
# Bad: No cleanup
def test_file_operations():
    f = open("test.txt", "w")
    f.write("test")
    # File not closed!

# Good: Use context managers
def test_file_operations(temp_dir):
    file_path = temp_dir / "test.txt"
    with open(file_path, "w") as f:
        f.write("test")
    # Automatically closed
```

## Quarantine Workflow

### 1. Detect Flaky Tests

```bash
# Run tests multiple times to gather data
for i in {1..5}; do
    pytest tests/unit/
done

# Check for flaky tests
pytest --flaky-report
```

### 2. Mark for Quarantine

```python
# Add marker to flaky test
@pytest.mark.quarantine
@pytest.mark.flaky(reruns=3)
def test_intermittent_failure():
    # Flaky test code
    pass
```

### 3. Skip in CI

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest tests/ --skip-quarantine
```

### 4. Fix and Remove Quarantine

Once fixed, remove the quarantine marker:

```python
# After fixing root cause
def test_now_stable():
    # Fixed test code
    pass
```

## Integration with pytest-rerunfailures

The flaky detection plugin works seamlessly with `pytest-rerunfailures`:

```bash
# Auto-retry failed tests
pytest --reruns 3 --reruns-delay 1

# Combine with parallel execution
pytest -n auto --reruns 2

# Skip quarantine and retry others
pytest --skip-quarantine --reruns 3
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests (skip quarantined)
        run: |
          pytest --skip-quarantine --reruns 2 -n auto

      - name: Upload flaky test report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: flaky-history
          path: .pytest_flaky_history.json
```

### GitLab CI

```yaml
test:
  script:
    - pytest --skip-quarantine --reruns 2 -n 4
  artifacts:
    paths:
      - .pytest_flaky_history.json
    when: always
  allow_failure:
    exit_codes: [5]  # No tests collected (all quarantined)
```

## Best Practices

1. **Don't quarantine immediately**: Run tests multiple times first to confirm flakiness
2. **Document why**: Add a comment explaining why a test is quarantined
3. **Set a deadline**: Create a ticket to fix quarantined tests
4. **Regular review**: Review quarantined tests weekly
5. **Fix root causes**: Don't just retry - fix the underlying issue

## Monitoring Flaky Tests

### View Flaky Test Trends

```python
# In conftest.py
def pytest_sessionfinish(session, exitstatus):
    tracker = get_tracker()
    flaky_tests = tracker.get_flaky_tests()

    # Log to metrics system
    metrics.gauge("tests.flaky.count", len(flaky_tests))

    # Alert if too many flaky tests
    if len(flaky_tests) > 10:
        send_alert("Too many flaky tests detected!")
```

### Generate Report

```bash
# View detailed report
pytest --flaky-report > flaky_report.txt
```

## Advanced: Custom Flaky Detection

You can customize the flaky detection logic:

```python
# In tests/conftest_flaky.py
class CustomFlakyTracker(FlakyTestTracker):
    def is_flaky(self, test_id: str, threshold: float = 0.3) -> bool:
        """Custom flaky detection logic."""
        results = self.test_results.get(test_id, [])

        # Require more data points
        if len(results) < 5:
            return False

        # Custom threshold for specific tests
        if "integration" in test_id:
            threshold = 0.2  # Stricter for integration tests

        # ... rest of logic
```

## Troubleshooting

### Tests Not Being Tracked

Ensure `conftest_flaky.py` is loaded:

```bash
pytest --trace-config | grep conftest_flaky
```

### History File Growing Too Large

Clear old history:

```bash
rm .pytest_flaky_history.json
```

Or configure max history per test in `conftest_flaky.py`:

```python
# Keep only last 5 results instead of 10
if len(results) > 5:
    results.pop(0)
```

### False Positives

Increase the threshold or required runs:

```python
tracker.is_flaky(test_id, threshold=0.4)  # 40% failure rate
```

## References

- [pytest-rerunfailures](https://github.com/pytest-dev/pytest-rerunfailures)
- [Pytest fixtures guide](https://docs.pytest.org/en/stable/fixture.html)
- [Google's Test Flakiness blog](https://testing.googleblog.com/2016/05/flaky-tests-at-google-and-how-we.html)
