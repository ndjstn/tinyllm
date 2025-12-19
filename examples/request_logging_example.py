"""Example of request/response logging with automatic redaction.

This example demonstrates TinyLLM's built-in request/response logger that
automatically redacts sensitive information like API keys, passwords, and PII.
"""

import time

from tinyllm.logging import configure_logging, get_request_logger


def main():
    """Demonstrate request/response logging with redaction."""
    # Configure logging
    configure_logging(log_format="console", log_level="INFO")

    # Get the global request/response logger
    req_logger = get_request_logger(max_length=500, redact_patterns=True)

    print("=== Request/Response Logging Example ===\n")

    # Example 1: HTTP Request with sensitive headers
    print("1. Logging HTTP request (with sensitive data):")
    req_logger.log_request(
        request_id="req-001",
        method="POST",
        path="/api/v1/generate",
        headers={
            "Authorization": "Bearer sk-abc123def456",  # Will be redacted
            "Content-Type": "application/json",
            "X-API-Key": "secret-key-789",  # Will be redacted
            "User-Agent": "TinyLLM-Client/1.0",
        },
        body={
            "prompt": "Tell me about user john.doe@example.com",  # Email redacted
            "temperature": 0.7,
            "api_key": "sk-secret",  # Will be redacted
            "user": {
                "email": "user@company.com",  # Will be redacted
                "password": "mysecret123",  # Will be redacted
                "preferences": {"theme": "dark"},
            },
        },
    )
    print()

    # Example 2: HTTP Response
    print("2. Logging HTTP response:")
    start = time.time()
    time.sleep(0.1)  # Simulate processing
    duration_ms = (time.time() - start) * 1000

    req_logger.log_response(
        request_id="req-001",
        status_code=200,
        headers={"Content-Type": "application/json"},
        body={
            "result": "The user information has been processed.",
            "api_key": "sk-returned",  # Will be redacted
            "metadata": {"processed_at": "2024-01-01T00:00:00Z"},
        },
        duration_ms=duration_ms,
    )
    print()

    # Example 3: LLM Request
    print("3. Logging LLM request:")
    req_logger.log_llm_request(
        request_id="llm-001",
        model="gpt-4",
        prompt="Generate code for authentication. Use password=admin123 for testing.",  # Password redacted
        parameters={
            "temperature": 0.7,
            "max_tokens": 1000,
            "api_key": "sk-openai-key",  # Will be redacted
        },
    )
    print()

    # Example 4: LLM Response
    print("4. Logging LLM response:")
    req_logger.log_llm_response(
        request_id="llm-001",
        model="gpt-4",
        response="Here's the authentication code. Contact admin@example.com for support.",  # Email redacted
        tokens_used=234,
        duration_ms=1234.56,
    )
    print()

    # Example 5: Custom redaction patterns
    print("5. Testing various sensitive data patterns:")

    test_data = {
        "credit_card": "4532-1234-5678-9010",  # Will be redacted
        "ssn": "123-45-6789",  # Will be redacted
        "phone": "555-123-4567",  # Will be redacted
        "aws_key": "AKIAIOSFODNN7EXAMPLE",  # Will be redacted
        "bearer_token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Will be redacted
        "normal_data": "This is public information",
    }

    req_logger.log_request(
        request_id="req-002",
        method="POST",
        path="/api/test",
        body=test_data,
    )
    print()

    # Example 6: Long content truncation
    print("6. Testing content truncation:")
    long_content = "x" * 2000  # Create long string

    req_logger.log_request(
        request_id="req-003",
        method="POST",
        path="/api/bulk",
        body={"data": long_content},
    )
    print()

    # Example 7: Nested sensitive data
    print("7. Testing nested redaction:")
    nested_data = {
        "user": {
            "profile": {
                "email": "nested@example.com",  # Will be redacted
                "settings": {
                    "api_key": "sk-nested-key",  # Will be redacted
                    "public_name": "John Doe",
                },
            },
            "credentials": {
                "password": "nested-secret",  # Will be redacted
                "token": "auth-token-123",  # Will be redacted
            },
        },
        "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
    }

    req_logger.log_request(
        request_id="req-004",
        method="POST",
        path="/api/user/update",
        body=nested_data,
    )
    print()

    print("=== Redaction Summary ===")
    print("All sensitive data has been automatically redacted:")
    print("  - API keys: [REDACTED_API_KEY]")
    print("  - Passwords: [REDACTED_PASSWORD]")
    print("  - Bearer tokens: [REDACTED_BEARER_TOKEN]")
    print("  - Emails (PII): [REDACTED_EMAIL]")
    print("  - Credit cards: [REDACTED_CREDIT_CARD]")
    print("  - SSN: [REDACTED_SSN]")
    print("  - Phone numbers: [REDACTED_PHONE]")
    print("  - AWS keys: [REDACTED_AWS_KEY]")
    print("  - Long content: Truncated with '... [truncated]'")
    print()
    print("Check the logs above to see redaction in action!")


if __name__ == "__main__":
    main()
