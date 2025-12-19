"""Example tests demonstrating flaky test detection and quarantine.

These tests are for demonstration purposes and show how to use the
flaky test detection and quarantine features.
"""

import random
import time

import pytest


class TestFlakyDetectionExamples:
    """Examples of flaky test patterns and how to handle them."""

    def test_stable(self):
        """A stable test that always passes."""
        assert 1 + 1 == 2

    @pytest.mark.skip(reason="Example of artificially flaky test - skip to avoid noise")
    def test_artificially_flaky(self):
        """Example of a flaky test that fails 30% of the time.

        In real scenarios, this would be caused by timing issues,
        network failures, or race conditions.
        """
        # Simulate flakiness
        if random.random() < 0.3:
            pytest.fail("Simulated intermittent failure")

    @pytest.mark.quarantine
    @pytest.mark.skip(reason="Quarantined - known flaky test")
    def test_quarantined_example(self):
        """Example of a test that has been quarantined.

        This test was identified as flaky and marked with @pytest.mark.quarantine
        to prevent it from blocking CI/CD.

        Issue: #123
        TODO: Fix race condition in async handler
        """
        # Flaky test code
        if random.random() < 0.5:
            pytest.fail("Known flaky behavior - under investigation")

    @pytest.mark.flaky(reruns=3, reruns_delay=0.1)
    @pytest.mark.skip(reason="Example with retry logic - skip to avoid noise")
    def test_with_retry(self):
        """Example of a test with automatic retry on failure.

        The @pytest.mark.flaky decorator will retry this test up to 3 times
        with 0.1 second delay between attempts if it fails.
        """
        # Simulate occasional failure that resolves on retry
        if random.random() < 0.2:
            pytest.fail("Transient failure - will retry")

    def test_timing_dependent(self):
        """Example of a test that could be flaky due to timing.

        This demonstrates a GOOD pattern - using timeouts and retries
        instead of hard-coded sleeps.
        """
        # Simulate an async operation
        start_time = time.time()

        # Poll with timeout instead of fixed sleep
        timeout = 1.0
        while time.time() - start_time < timeout:
            # Check if operation complete
            if True:  # In real code, this would check actual state
                return
            time.sleep(0.01)

        pytest.fail("Operation timed out")

    def test_with_fixture_isolation(self, temp_dir):
        """Example of proper test isolation using fixtures.

        This test uses the temp_dir fixture to ensure file operations
        don't interfere with other tests.
        """
        test_file = temp_dir / "test.txt"
        test_file.write_text("isolated test data")

        assert test_file.read_text() == "isolated test data"

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_parametrized_stability(self, value):
        """Example of parametrized test that is stable."""
        assert value > 0
        assert isinstance(value, int)
