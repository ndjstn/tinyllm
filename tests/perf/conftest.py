"""Performance test configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    """Add command-line options for performance testing."""
    parser.addoption(
        "--save-baseline",
        action="store_true",
        help="Save current performance metrics as baseline",
    )
    parser.addoption(
        "--skip-regression",
        action="store_true",
        help="Skip regression checks",
    )


@pytest.fixture(scope="session", autouse=True)
def save_baseline_if_requested(request):
    """Save baseline after all tests if --save-baseline flag is set."""
    yield

    if request.config.getoption("--save-baseline"):
        from pathlib import Path
        from tinyllm.tests.perf.test_performance import PerformanceMetrics

        # Collect metrics from all tests
        # (This is simplified - in practice you'd collect from all test instances)
        metrics = PerformanceMetrics()
        baseline_file = Path(__file__).parent / "performance_baseline.json"

        # Note: In a real implementation, we would collect metrics from all tests
        # For now, this just creates an empty baseline
        metrics.save_baseline()
        print(f"\nSaved performance baseline to {baseline_file}")
