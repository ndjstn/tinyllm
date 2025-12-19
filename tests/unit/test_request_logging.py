"""Tests for request/response logging with redaction."""

import pytest

from tinyllm.logging import RequestResponseLogger, get_request_logger


class TestRequestResponseLogger:
    """Test the RequestResponseLogger class."""

    def test_redact_sensitive_fields(self):
        """Test redaction of sensitive field names."""
        logger = RequestResponseLogger()

        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "normal_field": "visible",
        }

        redacted = logger.redact_dict(data)

        assert redacted["username"] == "john"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["normal_field"] == "visible"

    def test_redact_nested_dicts(self):
        """Test redaction in nested dictionaries."""
        logger = RequestResponseLogger()

        data = {
            "user": {"name": "john", "password": "secret123"},
            "config": {"token": "abc123", "timeout": 30},
        }

        redacted = logger.redact_dict(data)

        assert redacted["user"]["name"] == "john"
        assert redacted["user"]["password"] == "[REDACTED]"
        assert redacted["config"]["token"] == "[REDACTED]"
        assert redacted["config"]["timeout"] == 30

    def test_redact_api_key_pattern(self):
        """Test pattern-based redaction of API keys."""
        logger = RequestResponseLogger()

        text = "Using api_key: sk-1234567890abcdef in request"
        redacted = logger.redact_string(text)

        assert "sk-1234567890abcdef" not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_redact_bearer_token(self):
        """Test redaction of Bearer tokens."""
        logger = RequestResponseLogger()

        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = logger.redact_string(text)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "[REDACTED_BEARER_TOKEN]" in redacted

    def test_redact_password_pattern(self):
        """Test pattern-based redaction of passwords."""
        logger = RequestResponseLogger()

        text = "Login with password: mySecretPass123"
        redacted = logger.redact_string(text)

        assert "mySecretPass123" not in redacted
        assert "[REDACTED_PASSWORD]" in redacted

    def test_redact_credit_card(self):
        """Test redaction of credit card numbers."""
        logger = RequestResponseLogger()

        text = "Card number: 4532-1234-5678-9010"
        redacted = logger.redact_string(text)

        assert "4532-1234-5678-9010" not in redacted
        assert "[REDACTED_CREDIT_CARD]" in redacted

    def test_redact_email(self):
        """Test redaction of email addresses."""
        logger = RequestResponseLogger()

        text = "Contact us at user@example.com for support"
        redacted = logger.redact_string(text)

        assert "user@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_redact_ssn(self):
        """Test redaction of SSN."""
        logger = RequestResponseLogger()

        text = "SSN: 123-45-6789"
        redacted = logger.redact_string(text)

        assert "123-45-6789" not in redacted
        assert "[REDACTED_SSN]" in redacted

    def test_redact_phone(self):
        """Test redaction of phone numbers."""
        logger = RequestResponseLogger()

        text = "Call 555-123-4567 for help"
        redacted = logger.redact_string(text)

        assert "555-123-4567" not in redacted
        assert "[REDACTED_PHONE]" in redacted

    def test_redact_aws_key(self):
        """Test redaction of AWS access keys."""
        logger = RequestResponseLogger()

        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        redacted = logger.redact_string(text)

        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED_AWS_KEY]" in redacted

    def test_redact_private_key(self):
        """Test redaction of private keys."""
        logger = RequestResponseLogger()

        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
        redacted = logger.redact_string(text)

        assert "-----BEGIN RSA PRIVATE KEY-----" not in redacted
        assert "[REDACTED_PRIVATE_KEY]" in redacted

    def test_disable_pattern_redaction(self):
        """Test disabling pattern-based redaction."""
        logger = RequestResponseLogger(redact_patterns=False)

        text = "api_key: sk-1234567890"
        redacted = logger.redact_string(text)

        # Should not redact when disabled
        assert text == redacted

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        logger = RequestResponseLogger(max_length=100)

        text = "a" * 200
        truncated = logger.truncate(text)

        assert len(truncated) <= 120  # 100 + "... [truncated]"
        assert truncated.endswith("... [truncated]")

    def test_no_truncate_short_text(self):
        """Test that short text is not truncated."""
        logger = RequestResponseLogger(max_length=100)

        text = "short text"
        truncated = logger.truncate(text)

        assert truncated == text

    def test_log_request(self):
        """Test logging of HTTP request."""
        logger = RequestResponseLogger()

        # Should not raise exception
        logger.log_request(
            request_id="req-123",
            method="POST",
            path="/api/chat",
            headers={"Authorization": "Bearer secret-token"},
            body={"message": "Hello", "api_key": "sk-secret"},
        )

    def test_log_response(self):
        """Test logging of HTTP response."""
        logger = RequestResponseLogger()

        # Should not raise exception
        logger.log_response(
            request_id="req-123",
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"result": "Success", "token": "secret-response-token"},
            duration_ms=123.45,
        )

    def test_log_llm_request(self):
        """Test logging of LLM request."""
        logger = RequestResponseLogger()

        # Should not raise exception
        logger.log_llm_request(
            request_id="req-123",
            model="gpt-4",
            prompt="What is my password: secret123?",
            parameters={"temperature": 0.7, "api_key": "sk-secret"},
        )

    def test_log_llm_response(self):
        """Test logging of LLM response."""
        logger = RequestResponseLogger()

        # Should not raise exception
        logger.log_llm_response(
            request_id="req-123",
            model="gpt-4",
            response="I cannot share passwords. Contact user@example.com",
            tokens_used=42,
            duration_ms=1500.0,
        )

    def test_get_request_logger_singleton(self):
        """Test that get_request_logger returns singleton."""
        logger1 = get_request_logger()
        logger2 = get_request_logger()

        assert logger1 is logger2

    def test_redact_lists_with_dicts(self):
        """Test redaction in lists containing dictionaries."""
        logger = RequestResponseLogger()

        data = {
            "users": [
                {"name": "alice", "password": "pass1"},
                {"name": "bob", "password": "pass2"},
            ]
        }

        redacted = logger.redact_dict(data)

        assert redacted["users"][0]["name"] == "alice"
        assert redacted["users"][0]["password"] == "[REDACTED]"
        assert redacted["users"][1]["name"] == "bob"
        assert redacted["users"][1]["password"] == "[REDACTED]"

    def test_case_insensitive_field_matching(self):
        """Test that field matching is case-insensitive."""
        logger = RequestResponseLogger()

        data = {
            "Password": "secret1",
            "API_KEY": "key123",
            "ApiKey": "key456",
            "api-key": "key789",
        }

        redacted = logger.redact_dict(data)

        assert redacted["Password"] == "[REDACTED]"
        assert redacted["API_KEY"] == "[REDACTED]"
        assert redacted["ApiKey"] == "[REDACTED]"
        assert redacted["api-key"] == "[REDACTED]"

    def test_multiple_patterns_in_single_string(self):
        """Test redaction of multiple sensitive patterns in one string."""
        logger = RequestResponseLogger()

        text = (
            "Login with api_key: sk-123 and password: secret. "
            "Email: user@test.com, Phone: 555-123-4567"
        )
        redacted = logger.redact_string(text)

        assert "sk-123" not in redacted
        assert "secret" not in redacted
        assert "user@test.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "[REDACTED_" in redacted
