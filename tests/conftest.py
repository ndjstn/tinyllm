"""Pytest configuration and fixtures for TinyLLM tests.

This file contains global fixtures and pytest configuration for parallel
execution and test isolation.
"""

import asyncio
import os
from typing import Generator

import pytest
from prometheus_client import REGISTRY, CollectorRegistry


@pytest.fixture
def sample_task_content():
    """Sample task content for testing."""
    return "Write a Python function to check if a number is prime"


@pytest.fixture
def sample_math_expression():
    """Sample math expression for calculator testing."""
    return "2 + 2 * 3"


@pytest.fixture
def reset_metrics_state() -> Generator[None, None, None]:
    """Reset metrics state between tests to prevent pollution.

    This fixture can be used explicitly in tests that need to ensure
    metrics from one test don't affect another. It clears all metric
    collectors between tests.

    Note: Not autouse to avoid interfering with metrics tests that need
    to read their own metrics.
    """
    # Store initial collectors
    collectors_before = list(REGISTRY._collector_to_names.keys())

    yield

    # After test: unregister any new collectors that were added during the test
    collectors_after = list(REGISTRY._collector_to_names.keys())
    for collector in collectors_after:
        if collector not in collectors_before:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                # Collector might have already been unregistered
                pass


@pytest.fixture
def isolated_metrics_collector():
    """Provide an isolated metrics collector for tests that need clean state.

    This fixture creates a fresh MetricsCollector instance by ensuring any
    previous metrics are unregistered before creating new ones.

    Note: Due to the singleton pattern in MetricsCollector, this fixture
    works by resetting the singleton instance state before each test.
    After the test, metrics remain in the registry but are associated with
    the collector instance used in that test.
    """
    from tinyllm.metrics import MetricsCollector

    # Unregister all existing tinyllm metrics before creating new collector
    collectors_to_unregister = []
    for collector in list(REGISTRY._collector_to_names.keys()):
        names = REGISTRY._collector_to_names.get(collector, set())
        if any(name.startswith('tinyllm_') for name in names):
            collectors_to_unregister.append(collector)

    for collector in collectors_to_unregister:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    # Reset the singleton instance so __init__ runs again
    MetricsCollector._instance = None

    # Create fresh instance - this will register new metrics
    collector = MetricsCollector()

    # Reset cardinality tracker for clean test state
    if hasattr(collector, 'cardinality_tracker'):
        collector.cardinality_tracker.reset()

    yield collector

    # Note: We don't cleanup after the test because the test needs to read
    # the metrics it just created. Each test will cleanup before it starts
    # by unregistering previous metrics (see above).


# Pytest configuration hooks


def pytest_configure(config):
    """Configure pytest environment."""
    # Set test environment variable
    os.environ["TINYLLM_ENV"] = "test"

    # Register custom markers
    config.addinivalue_line("markers", "unit: unit tests that don't require external services")
    config.addinivalue_line("markers", "integration: integration tests requiring external services")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")
    config.addinivalue_line("markers", "flaky: potentially flaky tests")
    config.addinivalue_line("markers", "quarantine: quarantined tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests based on path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "stress" in str(item.fspath) or "load" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session.

    This fixture ensures that async tests have a consistent event loop
    across the entire test session, which is important for parallel execution.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests.

    Args:
        tmp_path: pytest's built-in tmp_path fixture.

    Returns:
        Path to temporary directory.
    """
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for test isolation.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
    """
    # Store original env vars
    original_env = dict(os.environ)

    # Set test-specific env vars
    monkeypatch.setenv("TINYLLM_ENV", "test")
    monkeypatch.setenv("TINYLLM_LOG_LEVEL", "ERROR")

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def isolate_redis_db(monkeypatch):
    """Isolate Redis database for parallel test execution.

    Each test worker gets its own Redis database number to prevent
    conflicts during parallel execution.
    """
    # Get worker ID from xdist
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")

    # Extract worker number (gw0 -> 0, gw1 -> 1, etc.)
    worker_num = int(worker_id.replace("gw", "")) if worker_id != "master" else 0

    # Use different Redis DB for each worker (0-15 are available)
    redis_db = worker_num % 16
    monkeypatch.setenv("REDIS_DB", str(redis_db))

    return redis_db


@pytest.fixture(scope="session")
def worker_id():
    """Get the current worker ID for parallel execution.

    Returns:
        Worker ID string (e.g., 'gw0', 'gw1') or 'master' for non-parallel runs.
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


# Hooks for parallel execution

# Note: pytest_xdist_setupnodes hook is only available when pytest-xdist is installed
# Uncomment when pytest-xdist is added to dependencies

# def pytest_xdist_setupnodes(config, specs):
#     """Called before distributing tests to workers.
#
#     Args:
#         config: pytest config object.
#         specs: list of worker specs.
#     """
#     print(f"\nDistributing tests across {len(specs)} workers")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to add custom test report information.

    This hook adds timing information and worker ID to test reports.
    """
    outcome = yield
    report = outcome.get_result()

    # Add worker information to report
    if hasattr(item.config, "workerinput"):
        report.worker_id = item.config.workerinput["workerid"]
    else:
        report.worker_id = "master"

    # Store test execution time
    if report.when == "call":
        report.test_duration = report.duration


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to test output.

    Args:
        terminalreporter: terminal reporter object.
        exitstatus: exit status code.
        config: pytest config object.
    """
    # Report parallel execution stats
    if hasattr(config, "workerinput"):
        terminalreporter.write_sep("=", "Parallel Execution Summary")
        terminalreporter.write_line(
            f"Worker ID: {config.workerinput['workerid']}"
        )
