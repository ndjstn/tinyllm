"""Structured logging configuration for TinyLLM.

This module provides a centralized logging setup using structlog with support
for both development (colored console output) and production (JSON) modes.
"""

import hashlib
import logging
import random
import sys
import time
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log events."""
    event_dict["app"] = "tinyllm"
    event_dict["version"] = "0.1.0"
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


# Initialize with sensible defaults
configure_logging()
