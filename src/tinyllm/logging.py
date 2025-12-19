"""Structured logging configuration for TinyLLM.

This module provides a centralized logging setup using structlog with support
for both development (colored console output) and production (JSON) modes.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log events."""
    event_dict["app"] = "tinyllm"
    event_dict["version"] = "0.1.0"
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_file: Optional[str] = None,
) -> None:
    """Configure structured logging for TinyLLM.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format - "console" for colored development output,
                   "json" for structured production logging.
        log_file: Optional file path to write logs to. If None, logs to stdout.
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
