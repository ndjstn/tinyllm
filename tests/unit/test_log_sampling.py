"""Tests for log sampling functionality."""

import time

import pytest

from tinyllm.logging import LogSampler, configure_log_sampling


class TestLogSampler:
    """Test the LogSampler class."""

    def test_no_sampling_logs_all(self):
        """Test that no sampling configuration logs everything."""
        sampler = LogSampler()
        for _ in range(100):
            assert sampler.should_log({"event": "test"})

    def test_sample_rate_percentage(self):
        """Test percentage-based sampling."""
        sampler = LogSampler(sample_rate=0.5, hash_based=False)
        logged = sum(1 for _ in range(1000) if sampler.should_log({"event": "test"}))
        # Should be roughly 50% with some tolerance
        assert 400 < logged < 600

    def test_sample_rate_zero_logs_nothing(self):
        """Test that 0% sample rate logs nothing."""
        sampler = LogSampler(sample_rate=0.0)
        for _ in range(100):
            assert not sampler.should_log({"event": "test"})

    def test_sample_rate_one_logs_all(self):
        """Test that 100% sample rate logs everything."""
        sampler = LogSampler(sample_rate=1.0)
        for _ in range(100):
            assert sampler.should_log({"event": "test"})

    def test_hash_based_sampling_consistent(self):
        """Test that hash-based sampling is consistent."""
        sampler = LogSampler(sample_rate=0.5, hash_based=True)

        # Same message should always get same result
        event = {"event": "consistent_message"}
        first_result = sampler.should_log(event)
        for _ in range(10):
            assert sampler.should_log(event) == first_result

    def test_hash_based_sampling_different_messages(self):
        """Test that hash-based sampling varies across messages."""
        sampler = LogSampler(sample_rate=0.5, hash_based=True)

        results = []
        for i in range(100):
            results.append(sampler.should_log({"event": f"message_{i}"}))

        # Should have a mix of True and False
        logged = sum(results)
        assert 30 < logged < 70  # Should be roughly 50%

    def test_max_per_second_rate_limit(self):
        """Test rate limiting with max_per_second."""
        sampler = LogSampler(max_per_second=10)

        # First 10 should log
        for i in range(10):
            assert sampler.should_log({"event": f"test_{i}"})

        # Next logs should be dropped (same second)
        for i in range(10, 20):
            assert not sampler.should_log({"event": f"test_{i}"})

    def test_max_per_second_resets_window(self):
        """Test that rate limit resets after time window."""
        sampler = LogSampler(max_per_second=5)

        # First 5 should log
        for i in range(5):
            assert sampler.should_log({"event": f"test_{i}"})

        # Next should be dropped
        assert not sampler.should_log({"event": "dropped"})

        # Simulate time passing (manually reset the window)
        sampler._last_reset = time.monotonic() - 2.0

        # Should log again after window reset
        assert sampler.should_log({"event": "after_reset"})

    def test_combined_sampling_and_rate_limit(self):
        """Test combining percentage sampling with rate limiting."""
        sampler = LogSampler(sample_rate=1.0, max_per_second=5)

        # Even with 100% sampling, should be limited to 5 per second
        logged = sum(
            1 for i in range(10) if sampler.should_log({"event": f"test_{i}"})
        )
        assert logged == 5


class TestConfigureLogSampling:
    """Test the configure_log_sampling function."""

    def test_configure_percentage_sampling(self):
        """Test configuring percentage-based sampling."""
        configure_log_sampling(sample_rate=0.1)
        # Global sampler should be configured
        from tinyllm.logging import _log_sampler

        assert _log_sampler is not None
        assert _log_sampler.sample_rate == 0.1
        assert not _log_sampler.hash_based

    def test_configure_rate_limit(self):
        """Test configuring rate limiting."""
        configure_log_sampling(max_per_second=100)
        from tinyllm.logging import _log_sampler

        assert _log_sampler is not None
        assert _log_sampler.max_per_second == 100

    def test_configure_hash_based(self):
        """Test configuring hash-based sampling."""
        configure_log_sampling(sample_rate=0.5, hash_based=True)
        from tinyllm.logging import _log_sampler

        assert _log_sampler is not None
        assert _log_sampler.sample_rate == 0.5
        assert _log_sampler.hash_based

    def test_configure_combined(self):
        """Test configuring combined sampling strategies."""
        configure_log_sampling(sample_rate=0.5, max_per_second=1000, hash_based=True)
        from tinyllm.logging import _log_sampler

        assert _log_sampler is not None
        assert _log_sampler.sample_rate == 0.5
        assert _log_sampler.max_per_second == 1000
        assert _log_sampler.hash_based
