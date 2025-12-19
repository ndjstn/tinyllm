"""Tests for structured logging module."""

import time

import pytest
import structlog

from tinyllm.logging import (
    LogSampler,
    RequestResponseLogger,
    configure_log_sampling,
    configure_logging,
    get_logger,
    get_request_logger,
)


class TestLogSampler:
    """Tests for LogSampler class."""

    def test_no_sampling_by_default(self):
        """Test that all logs pass through when no sampling configured."""
        sampler = LogSampler()

        event_dict = {"event": "test_event"}
        for _ in range(100):
            assert sampler.should_log(event_dict) is True

    def test_sample_rate_percentage(self):
        """Test percentage-based sampling."""
        # Sample 50% of logs
        sampler = LogSampler(sample_rate=0.5)

        events = [{"event": f"test_{i}"} for i in range(1000)]
        sampled_count = sum(1 for event in events if sampler.should_log(event))

        # Should be approximately 500 (with some variance)
        # Use a wide range to avoid flakiness
        assert 400 <= sampled_count <= 600

    def test_sample_rate_zero(self):
        """Test that sample_rate=0.0 drops all logs."""
        sampler = LogSampler(sample_rate=0.0)

        event_dict = {"event": "test_event"}
        for _ in range(100):
            assert sampler.should_log(event_dict) is False

    def test_sample_rate_one(self):
        """Test that sample_rate=1.0 keeps all logs."""
        sampler = LogSampler(sample_rate=1.0)

        event_dict = {"event": "test_event"}
        for _ in range(100):
            assert sampler.should_log(event_dict) is True

    def test_max_per_second_rate_limiting(self):
        """Test rate limiting with max_per_second."""
        max_per_sec = 10
        sampler = LogSampler(max_per_second=max_per_sec)

        event_dict = {"event": "test_event"}

        # First 10 should pass
        passed = sum(1 for _ in range(15) if sampler.should_log(event_dict))
        assert passed == max_per_sec

        # Wait for window to reset
        time.sleep(1.1)

        # Next 10 should pass again
        passed = sum(1 for _ in range(15) if sampler.should_log(event_dict))
        assert passed == max_per_sec

    def test_hash_based_sampling_consistency(self):
        """Test that hash-based sampling is consistent for same message."""
        sampler = LogSampler(sample_rate=0.5, hash_based=True)

        # Same event should always get same result
        event_dict = {"event": "test_event"}
        first_result = sampler.should_log(event_dict)

        # Try multiple times
        for _ in range(10):
            assert sampler.should_log(event_dict) == first_result

    def test_hash_based_sampling_varies_by_message(self):
        """Test that different messages get different sampling decisions."""
        sampler = LogSampler(sample_rate=0.5, hash_based=True)

        # Different events should have varying results
        events = [{"event": f"test_{i}"} for i in range(1000)]
        sampled = [sampler.should_log(event) for event in events]

        # Should sample roughly 50%
        sampled_count = sum(sampled)
        assert 400 <= sampled_count <= 600

    def test_combined_sampling_and_rate_limiting(self):
        """Test combining sample_rate and max_per_second."""
        sampler = LogSampler(sample_rate=0.5, max_per_second=20)

        event_dict = {"event": "test_event"}

        # Even with 50% sampling, should stop at 20 per second
        passed = sum(1 for _ in range(100) if sampler.should_log(event_dict))
        assert passed <= 20


class TestLoggingConfiguration:
    """Tests for logging configuration."""

    def test_configure_logging_default(self):
        """Test default logging configuration."""
        configure_logging()
        logger = get_logger(__name__)
        assert logger is not None

    def test_configure_logging_json_format(self):
        """Test JSON format configuration."""
        configure_logging(log_format="json")
        logger = get_logger(__name__)
        assert logger is not None

    def test_configure_logging_with_sampling(self):
        """Test enabling sampling in configuration."""
        configure_logging(enable_sampling=True)
        logger = get_logger(__name__)
        assert logger is not None

    def test_configure_log_sampling(self):
        """Test log sampling configuration."""
        configure_log_sampling(sample_rate=0.5, max_per_second=100)
        # Should not raise

    def test_get_logger_with_context(self):
        """Test getting logger with initial context."""
        logger = get_logger(__name__, component="test", trace_id="123")
        assert logger is not None


class TestRequestResponseLogger:
    """Tests for RequestResponseLogger."""

    def test_redact_dict_sensitive_fields(self):
        """Test redaction of sensitive fields in dictionaries."""
        logger = RequestResponseLogger()

        data = {
            "username": "alice",
            "password": "secret123",
            "api_key": "sk-abc123",
            "normal_field": "public",
        }

        redacted = logger.redact_dict(data)

        assert redacted["username"] == "alice"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["normal_field"] == "public"

    def test_redact_dict_nested(self):
        """Test redaction of nested dictionaries."""
        logger = RequestResponseLogger()

        data = {
            "user": {
                "name": "alice",
                "password": "secret",
                "preferences": {
                    "theme": "dark",
                    "api_key": "sk-123",
                },
            }
        }

        redacted = logger.redact_dict(data)

        assert redacted["user"]["name"] == "alice"
        assert redacted["user"]["password"] == "[REDACTED]"
        assert redacted["user"]["preferences"]["theme"] == "dark"
        assert redacted["user"]["preferences"]["api_key"] == "[REDACTED]"

    def test_redact_string_api_keys(self):
        """Test redaction of API keys in strings."""
        logger = RequestResponseLogger()

        text = "Using api_key: sk-abc123def456 for authentication"
        redacted = logger.redact_string(text)

        assert "sk-abc123def456" not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_redact_string_passwords(self):
        """Test redaction of passwords in strings."""
        logger = RequestResponseLogger()

        text = "password=mysecret123 should be hidden"
        redacted = logger.redact_string(text)

        assert "mysecret123" not in redacted
        assert "[REDACTED_PASSWORD]" in redacted

    def test_redact_string_bearer_tokens(self):
        """Test redaction of bearer tokens."""
        logger = RequestResponseLogger()

        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = logger.redact_string(text)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "[REDACTED_BEARER_TOKEN]" in redacted

    def test_redact_string_emails(self):
        """Test redaction of email addresses (PII)."""
        logger = RequestResponseLogger()

        text = "Contact user@example.com for more info"
        redacted = logger.redact_string(text)

        assert "user@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_redact_string_credit_cards(self):
        """Test redaction of credit card numbers."""
        logger = RequestResponseLogger()

        text = "Card number is 4532-1234-5678-9010"
        redacted = logger.redact_string(text)

        assert "4532-1234-5678-9010" not in redacted
        assert "[REDACTED_CREDIT_CARD]" in redacted

    def test_redact_patterns_disabled(self):
        """Test that pattern redaction can be disabled."""
        logger = RequestResponseLogger(redact_patterns=False)

        text = "password=secret123"
        redacted = logger.redact_string(text)

        # Should not redact when disabled
        assert text == redacted

    def test_truncate_short_text(self):
        """Test that short text is not truncated."""
        logger = RequestResponseLogger(max_length=100)

        text = "Short message"
        truncated = logger.truncate(text)

        assert truncated == text

    def test_truncate_long_text(self):
        """Test that long text is truncated."""
        logger = RequestResponseLogger(max_length=50)

        text = "a" * 100
        truncated = logger.truncate(text)

        assert len(truncated) == 50 + len("... [truncated]")
        assert truncated.endswith("... [truncated]")

    def test_log_request(self):
        """Test request logging."""
        logger = RequestResponseLogger()

        # Should not raise
        logger.log_request(
            request_id="req-123",
            method="POST",
            path="/api/v1/test",
            headers={"Authorization": "Bearer token123"},
            body={"password": "secret", "data": "public"},
        )

    def test_log_response(self):
        """Test response logging."""
        logger = RequestResponseLogger()

        # Should not raise
        logger.log_response(
            request_id="req-123",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"result": "success", "api_key": "sk-123"},
            duration_ms=123.45,
        )

    def test_log_llm_request(self):
        """Test LLM request logging."""
        logger = RequestResponseLogger()

        # Should not raise
        logger.log_llm_request(
            request_id="llm-123",
            model="gpt-4",
            prompt="Tell me a secret password=abc123",
            parameters={"temperature": 0.7, "api_key": "sk-secret"},
        )

    def test_log_llm_response(self):
        """Test LLM response logging."""
        logger = RequestResponseLogger()

        # Should not raise
        logger.log_llm_response(
            request_id="llm-123",
            model="gpt-4",
            response="Here is the info you requested",
            tokens_used=150,
            duration_ms=1234.56,
        )

    def test_get_request_logger_singleton(self):
        """Test that get_request_logger returns same instance."""
        logger1 = get_request_logger()
        logger2 = get_request_logger()

        assert logger1 is logger2


class TestSamplingIntegration:
    """Integration tests for log sampling with structlog."""

    def test_sampling_drops_events(self):
        """Test that sampling actually drops log events."""
        # Configure with 0% sampling (drop all)
        configure_log_sampling(sample_rate=0.0)
        configure_logging(enable_sampling=True)

        logger = get_logger(__name__)

        # This should be dropped by sampling
        # We can't easily verify it was dropped without checking output,
        # but we can verify it doesn't raise
        logger.info("this_should_be_dropped")

    def test_no_sampling_keeps_events(self):
        """Test that disabling sampling keeps all events."""
        configure_log_sampling(sample_rate=1.0)
        configure_logging(enable_sampling=True)

        logger = get_logger(__name__)

        # This should not be dropped
        logger.info("this_should_pass")
