"""Pytest configuration and fixtures."""

import pytest
from prometheus_client import REGISTRY, CollectorRegistry
from typing import Generator


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
