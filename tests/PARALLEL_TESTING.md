# Parallel Test Execution Guide

TinyLLM uses `pytest-xdist` for parallel test execution, which significantly reduces test runtime by distributing tests across multiple CPU cores.

## Quick Start

### Run tests in parallel using auto-detection

```bash
pytest -n auto
```

This automatically detects the number of available CPU cores and creates one worker per core.

### Run tests with a specific number of workers

```bash
pytest -n 4  # Use 4 workers
```

### Run specific test categories in parallel

```bash
# Unit tests only
pytest -n auto -m unit

# Integration tests only
pytest -n auto -m integration

# Exclude slow tests
pytest -n auto -m "not slow"
```

## Configuration

Parallel execution is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short --strict-markers"
markers = [
    "unit: unit tests that don't require external services",
    "integration: integration tests requiring external services",
    "slow: tests that take a long time to run",
    "flaky: potentially flaky tests",
    "quarantine: quarantined tests",
]
```

## Test Isolation

Tests are automatically isolated to prevent conflicts during parallel execution:

### Redis Database Isolation

Each worker gets its own Redis database number (0-15):

```python
@pytest.fixture(autouse=True)
def isolate_redis_db(monkeypatch):
    """Isolate Redis database for parallel test execution."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker_num = int(worker_id.replace("gw", "")) if worker_id != "master" else 0
    redis_db = worker_num % 16
    monkeypatch.setenv("REDIS_DB", str(redis_db))
    return redis_db
```

### Environment Isolation

Each worker has isolated environment variables:

```python
@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for test isolation."""
    monkeypatch.setenv("TINYLLM_ENV", "test")
    monkeypatch.setenv("TINYLLM_LOG_LEVEL", "ERROR")
```

### File System Isolation

Use the `temp_dir` fixture for temporary files:

```python
def test_something(temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("test")
```

## Load Distribution Strategies

### By Module (Default)

```bash
pytest -n auto --dist=loadscope
```

Distributes entire test modules to workers. Best for tests with expensive setup.

### By File

```bash
pytest -n auto --dist=loadfile
```

Distributes entire test files to workers.

### By Test

```bash
pytest -n auto --dist=loadgroup
```

Distributes individual tests. Best for fine-grained load balancing.

## Performance Tips

### 1. Run unit tests in parallel

Unit tests are fast and benefit most from parallelization:

```bash
pytest tests/unit/ -n auto
```

### 2. Run integration tests serially or with fewer workers

Integration tests may have external dependencies:

```bash
pytest tests/integration/ -n 2
```

### 3. Use test markers to control parallelization

```python
# Mark tests that should run serially
@pytest.mark.serial
def test_database_migration():
    ...

# Then exclude them from parallel runs
pytest -n auto -m "not serial"
```

### 4. Monitor worker load

```bash
pytest -n auto -v --dist=loadscope
```

The `-v` flag shows which worker runs each test.

## Troubleshooting

### Tests fail in parallel but pass serially

This usually indicates test isolation issues:

1. **Shared state**: Tests may be modifying global state
2. **Race conditions**: Tests may have timing dependencies
3. **Resource conflicts**: Tests may compete for the same resources

**Solution**: Use fixtures for test isolation and avoid global state.

### Inconsistent test results

Some tests may be flaky:

```python
# Mark flaky tests
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_network_request():
    ...
```

### Worker crashes

If workers crash frequently:

1. Reduce worker count: `pytest -n 2`
2. Increase timeouts: `pytest --timeout=300`
3. Check memory usage

## Benchmark Performance

Compare serial vs parallel execution:

```bash
# Serial execution
time pytest tests/unit/

# Parallel execution
time pytest tests/unit/ -n auto
```

Example speedup on 8-core system:
- Serial: 120 seconds
- Parallel (n=8): 25 seconds
- Speedup: 4.8x

## Best Practices

1. **Write isolated tests**: Each test should be independent
2. **Use fixtures**: Leverage pytest fixtures for setup/teardown
3. **Avoid global state**: Don't modify global variables or singletons
4. **Use temp directories**: Always use `temp_dir` fixture for file I/O
5. **Mark test categories**: Use markers to categorize tests
6. **Handle async properly**: Use `pytest-asyncio` with proper event loop handling

## CI/CD Integration

In GitHub Actions:

```yaml
- name: Run tests in parallel
  run: |
    pytest -n auto --dist=loadscope --maxfail=5
```

In GitLab CI:

```yaml
test:
  script:
    - pytest -n 4 --dist=loadfile
  parallel:
    matrix:
      - WORKER: [1, 2, 3, 4]
```

## Advanced: Custom Distribution

Create custom distribution logic:

```python
# In conftest.py
def pytest_xdist_make_scheduler(config, log):
    """Create custom scheduler for test distribution."""
    return MyCustomScheduler(config, log)
```

## References

- [pytest-xdist documentation](https://pytest-xdist.readthedocs.io/)
- [pytest fixtures guide](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers guide](https://docs.pytest.org/en/stable/example/markers.html)
