#!/usr/bin/env python3
"""Quick test to verify structured logging is working."""

from tinyllm.logging import configure_logging, get_logger

# Test 1: Console output (development mode)
print("=== Test 1: Console Output (Development) ===")
configure_logging(log_level="DEBUG", log_format="console")
logger = get_logger(__name__, component="test")

logger.debug("debug_message", extra_field="debug_value")
logger.info("info_message", user_id="user123", action="test")
logger.warning("warning_message", threshold=90)
logger.error("error_message", error_code="E001", details={"key": "value"})

print("\n")

# Test 2: JSON output (production mode)
print("=== Test 2: JSON Output (Production) ===")
configure_logging(log_level="INFO", log_format="json")
logger2 = get_logger("production_test", component="api")

logger2.info("request_received", method="POST", path="/api/v1/query")
logger2.info("query_processed", duration_ms=150, tokens=42)
logger2.error("validation_error", field="email", message="Invalid format")

print("\n")

# Test 3: Context binding
print("=== Test 3: Context Binding ===")
configure_logging(log_level="INFO", log_format="console")
from tinyllm.logging import bind_context, clear_context

logger3 = get_logger("context_test")

# Bind trace_id to all subsequent logs
bind_context(trace_id="abc-123", request_id="req-456")
logger3.info("first_event")
logger3.info("second_event", extra_data="test")

clear_context()
logger3.info("third_event_no_context")

print("\n=== All tests completed ===")
