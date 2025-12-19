"""Pytest plugin for flaky test detection and quarantine.

This plugin automatically detects flaky tests and can quarantine them
to prevent them from blocking CI/CD pipelines.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class FlakyTestTracker:
    """Tracks test failures to detect flaky tests."""

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize the tracker.

        Args:
            history_file: Path to file storing test history.
        """
        self.history_file = history_file or Path(".pytest_flaky_history.json")
        self.test_results: Dict[str, List[str]] = defaultdict(list)
        self.load_history()

    def load_history(self) -> None:
        """Load test history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    self.test_results = defaultdict(list, data.get("test_results", {}))
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start fresh
                self.test_results = defaultdict(list)

    def save_history(self) -> None:
        """Save test history to file."""
        data = {
            "test_results": dict(self.test_results),
            "last_updated": datetime.utcnow().isoformat(),
        }
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_result(self, test_id: str, outcome: str) -> None:
        """Record a test result.

        Args:
            test_id: Unique test identifier.
            outcome: Test outcome ('passed', 'failed', 'skipped').
        """
        # Keep last 10 results
        results = self.test_results[test_id]
        results.append(outcome)
        if len(results) > 10:
            results.pop(0)

    def is_flaky(self, test_id: str, threshold: float = 0.3) -> bool:
        """Check if a test is flaky.

        A test is considered flaky if it has both passes and failures
        in its recent history, and the failure rate is above threshold.

        Args:
            test_id: Unique test identifier.
            threshold: Minimum failure rate to be considered flaky (0.0-1.0).

        Returns:
            True if test is flaky.
        """
        results = self.test_results.get(test_id, [])

        if len(results) < 3:
            # Not enough data
            return False

        has_pass = "passed" in results
        has_fail = "failed" in results

        if not (has_pass and has_fail):
            # Not flaky if always passes or always fails
            return False

        # Calculate failure rate
        total = len(results)
        failures = results.count("failed")
        failure_rate = failures / total

        return failure_rate >= threshold

    def should_quarantine(self, test_id: str) -> bool:
        """Check if a test should be quarantined.

        Args:
            test_id: Unique test identifier.

        Returns:
            True if test should be quarantined.
        """
        # Quarantine if flaky for 3+ runs
        return self.is_flaky(test_id) and len(self.test_results.get(test_id, [])) >= 5

    def get_flaky_tests(self) -> List[str]:
        """Get list of all flaky tests.

        Returns:
            List of test IDs that are flaky.
        """
        return [test_id for test_id in self.test_results if self.is_flaky(test_id)]

    def get_quarantined_tests(self) -> List[str]:
        """Get list of quarantined tests.

        Returns:
            List of test IDs that should be quarantined.
        """
        return [test_id for test_id in self.test_results if self.should_quarantine(test_id)]


# Global tracker instance
_tracker: Optional[FlakyTestTracker] = None


def get_tracker() -> FlakyTestTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = FlakyTestTracker()
    return _tracker


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to track test results for flaky detection."""
    outcome = yield
    report = outcome.get_result()

    # Only track test results, not setup/teardown
    if report.when == "call":
        tracker = get_tracker()
        test_id = item.nodeid

        # Record the result
        if report.passed:
            tracker.record_result(test_id, "passed")
        elif report.failed:
            tracker.record_result(test_id, "failed")
        elif report.skipped:
            tracker.record_result(test_id, "skipped")


def pytest_sessionfinish(session, exitstatus):
    """Hook called after all tests complete."""
    tracker = get_tracker()
    tracker.save_history()

    # Report flaky tests
    flaky_tests = tracker.get_flaky_tests()
    if flaky_tests:
        print("\n" + "=" * 70)
        print(f"FLAKY TESTS DETECTED: {len(flaky_tests)}")
        print("=" * 70)
        for test_id in flaky_tests[:10]:  # Show first 10
            results = tracker.test_results[test_id]
            failure_rate = results.count("failed") / len(results) * 100
            print(f"  {test_id}")
            print(f"    Failure rate: {failure_rate:.1f}%")
            print(f"    Recent results: {' '.join(results[-5:])}")
            print()

    # Report quarantined tests
    quarantined = tracker.get_quarantined_tests()
    if quarantined:
        print("=" * 70)
        print(f"TESTS RECOMMENDED FOR QUARANTINE: {len(quarantined)}")
        print("=" * 70)
        for test_id in quarantined:
            print(f"  @pytest.mark.quarantine")
            print(f"  {test_id}")
            print()


def pytest_collection_modifyitems(config, items):
    """Hook to skip quarantined tests if requested."""
    # Skip quarantined tests if --skip-quarantine is passed
    if config.getoption("--skip-quarantine", default=False):
        skip_quarantine = pytest.mark.skip(reason="Quarantined test")
        for item in items:
            if "quarantine" in item.keywords:
                item.add_marker(skip_quarantine)


def pytest_addoption(parser):
    """Add custom command-line options."""
    group = parser.getgroup("flaky", "Flaky test detection and quarantine")

    group.addoption(
        "--skip-quarantine",
        action="store_true",
        default=False,
        help="Skip tests marked as quarantine",
    )

    group.addoption(
        "--reruns",
        action="store",
        dest="reruns",
        type=int,
        default=0,
        help="Number of times to re-run failed tests (requires pytest-rerunfailures)",
    )

    group.addoption(
        "--reruns-delay",
        action="store",
        dest="reruns_delay",
        type=float,
        default=0,
        help="Delay in seconds between re-runs (requires pytest-rerunfailures)",
    )

    group.addoption(
        "--flaky-report",
        action="store_true",
        default=False,
        help="Generate detailed flaky test report",
    )


def pytest_configure(config):
    """Configure pytest for flaky test detection."""
    # Register markers
    config.addinivalue_line(
        "markers",
        "flaky(reruns=N, reruns_delay=S): mark test as potentially flaky with retry options",
    )
    config.addinivalue_line(
        "markers",
        "quarantine: mark test as quarantined due to flakiness",
    )


@pytest.fixture
def flaky_tracker():
    """Provide flaky test tracker in tests.

    Returns:
        FlakyTestTracker instance.
    """
    return get_tracker()
