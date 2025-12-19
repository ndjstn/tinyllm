"""Structured logging configuration for TinyLLM.

This module provides a centralized logging setup using structlog with support
for both development (colored console output) and production (JSON) modes.
"""

import hashlib
import logging
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Pattern

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log events."""
    event_dict["app"] = "tinyllm"
    event_dict["version"] = "0.1.0"
    return event_dict


def add_trace_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add trace and span IDs to log events if available.

    This processor automatically injects OpenTelemetry trace and span IDs
    into log records for correlation between logs and traces.
    """
    try:
        # Try to get trace context from OpenTelemetry if available
        from tinyllm.telemetry import get_current_trace_id, get_current_span_id

        trace_id = get_current_trace_id()
        span_id = get_current_span_id()

        if trace_id:
            event_dict["trace_id"] = trace_id
        if span_id:
            event_dict["span_id"] = span_id
    except ImportError:
        # OpenTelemetry not available, skip trace context
        pass
    except Exception:
        # Silently ignore errors in trace context injection
        # to avoid breaking logging
        pass

    return event_dict


# Log sampling configuration
class LogSampler:
    """Rate-based log sampler for high-volume scenarios.

    Implements multiple sampling strategies:
    - Rate-based: Sample X logs per second
    - Percentage: Sample X% of all logs
    - Hash-based: Consistent sampling based on message hash
    """

    def __init__(
        self,
        sample_rate: Optional[float] = None,
        max_per_second: Optional[int] = None,
        hash_based: bool = False,
    ):
        """Initialize log sampler.

        Args:
            sample_rate: Percentage of logs to sample (0.0 to 1.0).
            max_per_second: Maximum logs per second (rate limiting).
            hash_based: Use hash-based sampling for consistency.
        """
        self.sample_rate = sample_rate
        self.max_per_second = max_per_second
        self.hash_based = hash_based

        # Rate limiting state
        self._count = 0
        self._last_reset = time.monotonic()
        self._window = 1.0  # 1 second window

    def should_log(self, event_dict: EventDict) -> bool:
        """Determine if this log should be emitted.

        Args:
            event_dict: The log event dictionary.

        Returns:
            True if the log should be emitted, False to drop it.
        """
        # If no sampling configured, log everything
        if self.sample_rate is None and self.max_per_second is None:
            return True

        # Rate limiting (max per second)
        if self.max_per_second is not None:
            now = time.monotonic()
            if now - self._last_reset >= self._window:
                # Reset counter for new window
                self._count = 0
                self._last_reset = now

            if self._count >= self.max_per_second:
                return False

            self._count += 1

        # Percentage-based sampling
        if self.sample_rate is not None:
            if self.hash_based:
                # Hash-based sampling (consistent for same message)
                event_str = str(event_dict.get("event", ""))
                hash_val = int(hashlib.md5(event_str.encode()).hexdigest()[:8], 16)
                return (hash_val % 100) < (self.sample_rate * 100)
            else:
                # Random sampling
                return random.random() < self.sample_rate

        return True


# Global sampler instance
_log_sampler: Optional[LogSampler] = None


def configure_log_sampling(
    sample_rate: Optional[float] = None,
    max_per_second: Optional[int] = None,
    hash_based: bool = False,
) -> None:
    """Configure log sampling for high-volume scenarios.

    Args:
        sample_rate: Percentage of logs to sample (0.0 to 1.0).
        max_per_second: Maximum logs per second (rate limiting).
        hash_based: Use hash-based sampling for consistency.

    Example:
        >>> # Sample 10% of logs
        >>> configure_log_sampling(sample_rate=0.1)
        >>> # Limit to 100 logs per second
        >>> configure_log_sampling(max_per_second=100)
        >>> # Combine both strategies
        >>> configure_log_sampling(sample_rate=0.5, max_per_second=1000)
    """
    global _log_sampler
    _log_sampler = LogSampler(
        sample_rate=sample_rate,
        max_per_second=max_per_second,
        hash_based=hash_based,
    )


def add_log_sampling(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Processor that implements log sampling.

    Args:
        logger: Logger instance.
        method_name: Log method name.
        event_dict: Log event dictionary.

    Returns:
        Event dictionary or raises DropEvent to drop the log.

    Raises:
        structlog.DropEvent: If the log should be sampled out.
    """
    if _log_sampler is not None and not _log_sampler.should_log(event_dict):
        # Mark as sampled for metrics
        event_dict["_sampled"] = True
        raise structlog.DropEvent

    return event_dict


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_file: Optional[str] = None,
    enable_sampling: bool = False,
) -> None:
    """Configure structured logging for TinyLLM.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - "console" for colored development output,
                   "json" for structured production logging.
        log_file: Optional file path to write logs to. If None, logs to stdout.
        enable_sampling: Enable log sampling processor in the chain.
    """
    # Convert log level string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_app_context,
        add_trace_context,  # Inject trace/span IDs for correlation
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Add log sampling processor if enabled
    if enable_sampling:
        shared_processors.insert(0, add_log_sampling)

    if log_format == "json":
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Colored console output for development
        processors = shared_processors + [
            structlog.processors.ExceptionRenderer(),
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a logger instance with optional initial context.

    Args:
        name: Logger name, typically __name__ of the calling module.
        **initial_context: Additional context to bind to all log messages.

    Returns:
        A configured structlog logger instance.

    Example:
        >>> logger = get_logger(__name__, component="executor")
        >>> logger.info("starting_execution", trace_id="abc123")
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**context: Any) -> None:
    """Bind context to the current thread's logger.

    Useful for adding request-scoped or execution-scoped context.

    Args:
        **context: Key-value pairs to add to the logging context.

    Example:
        >>> bind_context(trace_id="abc123", user_id="user456")
    """
    structlog.contextvars.bind_contextvars(**context)


def unbind_context(*keys: str) -> None:
    """Remove keys from the current thread's logging context.

    Args:
        *keys: Keys to remove from context.
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context from the current thread."""
    structlog.contextvars.clear_contextvars()


# Request/Response logging with redaction
class RequestResponseLogger:
    """Logger for request/response data with sensitive data redaction.

    Automatically redacts sensitive information like API keys, passwords,
    tokens, and PII before logging.
    """

    # Patterns for sensitive data detection
    SENSITIVE_PATTERNS: List[tuple[str, Pattern]] = [
        # API keys and tokens
        ("api_key", re.compile(r"(api[_-]?key|apikey)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)),
        ("bearer_token", re.compile(r"bearer[\s]+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)),
        ("auth_token", re.compile(r"(auth[_-]?token|token)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)),
        # Passwords
        ("password", re.compile(r"(password|passwd|pwd)[\s:=]+['\"]?([^\s'\"]+)", re.IGNORECASE)),
        # Credit card numbers (basic pattern)
        ("credit_card", re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")),
        # Email addresses (PII)
        ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
        # SSN (US)
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        # Phone numbers (basic)
        ("phone", re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")),
        # AWS keys
        ("aws_key", re.compile(r"(AKIA[0-9A-Z]{16})", re.IGNORECASE)),
        # Private keys
        ("private_key", re.compile(r"-----BEGIN.*PRIVATE KEY-----", re.IGNORECASE)),
    ]

    # Sensitive field names (case-insensitive)
    SENSITIVE_FIELDS = {
        "password",
        "passwd",
        "pwd",
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "token",
        "auth",
        "authorization",
        "credit_card",
        "creditcard",
        "ssn",
        "social_security",
        "private_key",
        "privatekey",
    }

    def __init__(self, max_length: int = 1000, redact_patterns: bool = True):
        """Initialize request/response logger.

        Args:
            max_length: Maximum length of logged content before truncation.
            redact_patterns: Enable pattern-based redaction.
        """
        self.max_length = max_length
        self.redact_patterns = redact_patterns
        self.logger = get_logger(__name__, component="request_logger")

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from dictionary.

        Args:
            data: Dictionary to redact.

        Returns:
            Redacted dictionary.
        """
        redacted = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if field name is sensitive
            if any(sensitive in key_lower for sensitive in self.SENSITIVE_FIELDS):
                redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str):
                redacted[key] = self.redact_string(value)
            else:
                redacted[key] = value

        return redacted

    def redact_string(self, text: str) -> str:
        """Redact sensitive patterns from string.

        Args:
            text: String to redact.

        Returns:
            Redacted string.
        """
        if not self.redact_patterns:
            return text

        redacted = text
        for pattern_name, pattern in self.SENSITIVE_PATTERNS:
            redacted = pattern.sub(f"[REDACTED_{pattern_name.upper()}]", redacted)

        return redacted

    def truncate(self, text: str) -> str:
        """Truncate text to max length.

        Args:
            text: Text to truncate.

        Returns:
            Truncated text with ellipsis if needed.
        """
        if len(text) <= self.max_length:
            return text

        return text[: self.max_length] + "... [truncated]"

    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
    ) -> None:
        """Log incoming request with redaction.

        Args:
            request_id: Unique request identifier.
            method: HTTP method.
            path: Request path.
            headers: Request headers.
            body: Request body.
        """
        log_data: Dict[str, Any] = {
            "request_id": request_id,
            "method": method,
            "path": path,
        }

        # Redact headers
        if headers:
            log_data["headers"] = self.redact_dict(headers)

        # Redact and truncate body
        if body:
            if isinstance(body, dict):
                redacted_body = self.redact_dict(body)
                body_str = str(redacted_body)
            else:
                body_str = str(body)
                body_str = self.redact_string(body_str)

            log_data["body"] = self.truncate(body_str)

        self.logger.info("incoming_request", **log_data)

    def log_response(
        self,
        request_id: str,
        status_code: int,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log outgoing response with redaction.

        Args:
            request_id: Unique request identifier.
            status_code: HTTP status code.
            headers: Response headers.
            body: Response body.
            duration_ms: Request duration in milliseconds.
        """
        log_data: Dict[str, Any] = {
            "request_id": request_id,
            "status_code": status_code,
        }

        if duration_ms is not None:
            log_data["duration_ms"] = round(duration_ms, 2)

        # Redact headers
        if headers:
            log_data["headers"] = self.redact_dict(headers)

        # Redact and truncate body
        if body:
            if isinstance(body, dict):
                redacted_body = self.redact_dict(body)
                body_str = str(redacted_body)
            else:
                body_str = str(body)
                body_str = self.redact_string(body_str)

            log_data["body"] = self.truncate(body_str)

        self.logger.info("outgoing_response", **log_data)

    def log_llm_request(
        self,
        request_id: str,
        model: str,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log LLM API request with redaction.

        Args:
            request_id: Unique request identifier.
            model: Model name.
            prompt: User prompt (will be truncated).
            parameters: Model parameters.
        """
        log_data: Dict[str, Any] = {
            "request_id": request_id,
            "model": model,
            "prompt": self.truncate(self.redact_string(prompt)),
        }

        if parameters:
            log_data["parameters"] = self.redact_dict(parameters)

        self.logger.info("llm_request", **log_data)

    def log_llm_response(
        self,
        request_id: str,
        model: str,
        response: str,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log LLM API response with redaction.

        Args:
            request_id: Unique request identifier.
            model: Model name.
            response: Model response (will be truncated).
            tokens_used: Number of tokens used.
            duration_ms: Request duration in milliseconds.
        """
        log_data: Dict[str, Any] = {
            "request_id": request_id,
            "model": model,
            "response": self.truncate(self.redact_string(response)),
        }

        if tokens_used is not None:
            log_data["tokens_used"] = tokens_used

        if duration_ms is not None:
            log_data["duration_ms"] = round(duration_ms, 2)

        self.logger.info("llm_response", **log_data)


# Global request/response logger instance
_request_logger: Optional[RequestResponseLogger] = None


def get_request_logger(
    max_length: int = 1000,
    redact_patterns: bool = True,
) -> RequestResponseLogger:
    """Get or create the global request/response logger.

    Args:
        max_length: Maximum length of logged content.
        redact_patterns: Enable pattern-based redaction.

    Returns:
        RequestResponseLogger instance.
    """
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestResponseLogger(
            max_length=max_length,
            redact_patterns=redact_patterns,
        )
    return _request_logger


# Initialize with sensible defaults
configure_logging()
