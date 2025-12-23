"""Error handling and recovery system for TinyLLM.

Provides:
- Custom exception hierarchy for structured error handling
- Dead letter queue for failed messages
- Automatic retry with jitter for transient failures
- Error classification system (retryable vs fatal)
- Circuit breaker with state persistence across restarts
- Global exception handler decorator
"""

import asyncio
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import aiosqlite
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="errors")
metrics = get_metrics_collector()

T = TypeVar("T")


# ============================================================================
# Exception Hierarchy
# ============================================================================


class TinyLLMError(Exception):
    """Base exception for all TinyLLM errors."""

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()

        # Capture stack trace
        self.stack_trace = "".join(traceback.format_exception(*sys.exc_info()))
        if self.stack_trace.strip() == "":
            self.stack_trace = "".join(traceback.format_stack())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace,
        }


class RetryableError(TinyLLMError):
    """Errors that can be retried (transient failures)."""

    def __init__(self, message: str, code: str = "RETRYABLE", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code, details=details, recoverable=True)


class FatalError(TinyLLMError):
    """Errors that cannot be retried (permanent failures)."""

    def __init__(self, message: str, code: str = "FATAL", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code, details=details, recoverable=False)


class ValidationError(FatalError):
    """Input validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class ModelError(RetryableError):
    """Model execution errors (usually retryable)."""

    def __init__(self, message: str, model: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if model:
            details["model"] = model
        super().__init__(message, code="MODEL_ERROR", details=details)


class TimeoutError(RetryableError):
    """Operation timeout errors."""

    def __init__(self, message: str, timeout_ms: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if timeout_ms:
            details["timeout_ms"] = timeout_ms
        super().__init__(message, code="TIMEOUT_ERROR", details=details)


class RateLimitError(RetryableError):
    """Rate limit exceeded errors."""

    def __init__(self, message: str, retry_after_ms: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if retry_after_ms:
            details["retry_after_ms"] = retry_after_ms
        super().__init__(message, code="RATE_LIMIT_ERROR", details=details)


class CircuitOpenError(RetryableError):
    """Circuit breaker is open."""

    def __init__(self, message: str, service: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service"] = service
        super().__init__(message, code="CIRCUIT_OPEN", details=details)


class ResourceExhaustedError(RetryableError):
    """Resource exhaustion errors (memory, connections, etc)."""

    def __init__(self, message: str, resource: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if resource:
            details["resource"] = resource
        super().__init__(message, code="RESOURCE_EXHAUSTED", details=details)


class NetworkError(RetryableError):
    """Network connectivity errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="NETWORK_ERROR", details=details)


class ExecutionError(FatalError):
    """Execution errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="EXECUTION_ERROR", details=details)


class ConfigurationError(FatalError):
    """Configuration errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


# ============================================================================
# Node-specific Errors (for graph execution)
# ============================================================================


class RetryableNodeError(RetryableError):
    """Retryable node execution errors.

    Use this for transient failures in node execution that may succeed on retry.
    Examples: temporary model failures, transient network issues, resource contention.
    """

    def __init__(
        self,
        node_id: str,
        attempt: int = 1,
        max_retries: int = 3,
        reason: Optional[str] = None,
        **kwargs
    ):
        """Initialize retryable node error.

        Args:
            node_id: ID of the node that failed
            attempt: Current retry attempt number
            max_retries: Maximum number of retries allowed
            reason: Optional reason for the failure
            **kwargs: Additional details to include in error context
        """
        message = f"Node '{node_id}' failed (attempt {attempt}/{max_retries})"
        if reason:
            message += f": {reason}"

        details = {
            "node_id": node_id,
            "attempt": attempt,
            "max_retries": max_retries,
            **kwargs
        }
        if reason:
            details["reason"] = reason

        super().__init__(message, code="NODE_RETRY", details=details)
        self.node_id = node_id
        self.attempt = attempt
        self.max_retries = max_retries


class PermanentNodeError(FatalError):
    """Permanent node execution errors.

    Use this for permanent failures that will not succeed on retry.
    Examples: invalid input, missing configuration, unsupported operation.
    """

    def __init__(
        self,
        node_id: str,
        reason: str,
        **kwargs
    ):
        """Initialize permanent node error.

        Args:
            node_id: ID of the node that failed
            reason: Reason for the permanent failure
            **kwargs: Additional details to include in error context
        """
        message = f"Node '{node_id}' permanently failed: {reason}"

        details = {
            "node_id": node_id,
            "reason": reason,
            **kwargs
        }

        super().__init__(message, code="NODE_PERMANENT_FAILURE", details=details)
        self.node_id = node_id
        self.reason = reason


class NodeTimeoutError(TimeoutError):
    """Node execution timeout.

    Specialized timeout error for node execution with node-specific context.
    """

    def __init__(
        self,
        node_id: str,
        timeout_ms: int,
        **kwargs
    ):
        """Initialize node timeout error.

        Args:
            node_id: ID of the node that timed out
            timeout_ms: Timeout value in milliseconds
            **kwargs: Additional details to include in error context
        """
        message = f"Node '{node_id}' timed out after {timeout_ms}ms"

        details = {
            "node_id": node_id,
            **kwargs
        }

        super().__init__(message, timeout_ms=timeout_ms, details=details)
        self.node_id = node_id


class NodeValidationError(ValidationError):
    """Node input/output validation error.

    Raised when node input or output fails validation.
    """

    def __init__(
        self,
        node_id: str,
        validation_type: str,  # "input" or "output"
        errors: List[str],
        **kwargs
    ):
        """Initialize node validation error.

        Args:
            node_id: ID of the node with validation failure
            validation_type: Type of validation ("input" or "output")
            errors: List of validation error messages
            **kwargs: Additional details to include in error context
        """
        message = f"Node '{node_id}' {validation_type} validation failed: {'; '.join(errors)}"

        details = {
            "node_id": node_id,
            "validation_type": validation_type,
            "errors": errors,
            **kwargs
        }

        super().__init__(message, details=details)
        self.node_id = node_id
        self.validation_type = validation_type
        self.errors = errors


# ============================================================================
# Error Classification
# ============================================================================


class ErrorClassifier:
    """Classifies errors as retryable or fatal."""

    # Patterns that indicate retryable errors
    RETRYABLE_PATTERNS = [
        "timeout",
        "connection",
        "rate limit",
        "too many requests",
        "service unavailable",
        "temporarily unavailable",
        "503",
        "504",
        "429",
        "network",
        "socket",
        "circuit",
        "resource",
    ]

    # Patterns that indicate fatal errors
    FATAL_PATTERNS = [
        "validation",
        "invalid",
        "malformed",
        "unauthorized",
        "forbidden",
        "not found",
        "400",
        "401",
        "403",
        "404",
        "permission",
        "configuration",
    ]

    @classmethod
    def classify(cls, error: Exception) -> bool:
        """Classify if an error is retryable.

        Args:
            error: Exception to classify.

        Returns:
            True if retryable, False if fatal.
        """
        # Check if it's a TinyLLM error with explicit classification
        if isinstance(error, TinyLLMError):
            return error.recoverable

        # Check error message against patterns
        error_str = str(error).lower()

        # Check fatal patterns first (more specific)
        for pattern in cls.FATAL_PATTERNS:
            if pattern in error_str:
                logger.debug("error_classified_fatal", error=error_str, pattern=pattern)
                return False

        # Check retryable patterns
        for pattern in cls.RETRYABLE_PATTERNS:
            if pattern in error_str:
                logger.debug("error_classified_retryable", error=error_str, pattern=pattern)
                return True

        # Default to non-retryable for safety
        logger.warning("error_classification_unknown", error=error_str)
        return False


# ============================================================================
# Retry with Jitter
# ============================================================================


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    base_delay_ms: int = Field(default=1000, ge=0, description="Base delay in milliseconds")
    max_delay_ms: int = Field(default=60000, ge=0, description="Maximum delay in milliseconds")
    exponential_base: float = Field(default=2.0, ge=1.0, description="Exponential backoff base")
    jitter: bool = Field(default=True, description="Add random jitter to delays")
    jitter_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Jitter ratio (0-1)")


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate retry delay with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    # Exponential backoff
    delay_ms = min(
        config.base_delay_ms * (config.exponential_base ** attempt),
        config.max_delay_ms,
    )

    # Add jitter
    if config.jitter:
        jitter_amount = delay_ms * config.jitter_ratio
        delay_ms += random.uniform(-jitter_amount, jitter_amount)

    return max(0, delay_ms) / 1000.0  # Convert to seconds


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic with exponential backoff and jitter.

    Args:
        config: Retry configuration (uses defaults if None).

    Example:
        @with_retry(RetryConfig(max_attempts=5))
        async def make_api_call():
            ...
    """
    retry_config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error = None

            for attempt in range(retry_config.max_attempts):
                try:
                    result = await func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            "retry_succeeded",
                            function=func.__name__,
                            attempt=attempt + 1,
                            total_attempts=retry_config.max_attempts,
                        )
                        metrics.increment_retry_success()

                    return result

                except Exception as e:
                    last_error = e
                    is_retryable = ErrorClassifier.classify(e)

                    if not is_retryable or attempt == retry_config.max_attempts - 1:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempt=attempt + 1,
                            error=str(e),
                            retryable=is_retryable,
                        )
                        metrics.increment_retry_exhausted()
                        raise

                    delay = calculate_delay(attempt, retry_config)
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=retry_config.max_attempts,
                        delay_s=delay,
                        error=str(e),
                    )
                    metrics.increment_retry_attempt()

                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            raise last_error or RuntimeError("Retry logic failed unexpectedly")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_error = None

            for attempt in range(retry_config.max_attempts):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            "retry_succeeded",
                            function=func.__name__,
                            attempt=attempt + 1,
                            total_attempts=retry_config.max_attempts,
                        )
                        metrics.increment_retry_success()

                    return result

                except Exception as e:
                    last_error = e
                    is_retryable = ErrorClassifier.classify(e)

                    if not is_retryable or attempt == retry_config.max_attempts - 1:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempt=attempt + 1,
                            error=str(e),
                            retryable=is_retryable,
                        )
                        metrics.increment_retry_exhausted()
                        raise

                    delay = calculate_delay(attempt, retry_config)
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=retry_config.max_attempts,
                        delay_s=delay,
                        error=str(e),
                    )
                    metrics.increment_retry_attempt()

                    time.sleep(delay)

            raise last_error or RuntimeError("Retry logic failed unexpectedly")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# Dead Letter Queue
# ============================================================================


class DLQMessage(BaseModel):
    """Message stored in dead letter queue."""

    id: str = Field(description="Unique message ID")
    trace_id: str = Field(description="Trace ID for correlation")
    payload: Dict[str, Any] = Field(description="Original message payload")
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Error message")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    attempts: int = Field(description="Number of attempts made")
    first_failed_at: datetime = Field(description="First failure timestamp")
    last_failed_at: datetime = Field(description="Last failure timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DeadLetterQueue:
    """Dead letter queue for failed messages using SQLite.

    Stores failed messages for later inspection, retry, or manual intervention.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize dead letter queue.

        Args:
            db_path: Path to SQLite database (uses default if None).
        """
        if db_path:
            self.db_path = db_path
        else:
            base_path = Path.home() / ".tinyllm" / "data"
            base_path.mkdir(parents=True, exist_ok=True)
            self.db_path = base_path / "dlq.db"

        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._db = await aiosqlite.connect(str(self.db_path))
            await self._db.execute("PRAGMA journal_mode=WAL")

            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS dlq_messages (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_details TEXT DEFAULT '{}',
                    attempts INTEGER NOT NULL,
                    first_failed_at TEXT NOT NULL,
                    last_failed_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)

            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_trace_id
                ON dlq_messages(trace_id)
            """)

            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_error_type
                ON dlq_messages(error_type)
            """)

            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_dlq_last_failed
                ON dlq_messages(last_failed_at DESC)
            """)

            await self._db.commit()
            self._initialized = True

            logger.info("dlq_initialized", db_path=str(self.db_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        self._initialized = False

    async def enqueue(
        self,
        message_id: str,
        trace_id: str,
        payload: Dict[str, Any],
        error: Exception,
        attempts: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a failed message to the dead letter queue.

        Args:
            message_id: Unique message identifier.
            trace_id: Trace ID for correlation.
            payload: Original message payload.
            error: The exception that caused the failure.
            attempts: Number of attempts made.
            metadata: Additional metadata.
        """
        if not self._initialized:
            await self.initialize()

        now = datetime.utcnow().isoformat()
        error_dict = error.to_dict() if isinstance(error, TinyLLMError) else {
            "type": type(error).__name__,
            "message": str(error),
        }

        async with self._lock:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO dlq_messages
                (id, trace_id, payload, error_type, error_message, error_details,
                 attempts, first_failed_at, last_failed_at, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    trace_id,
                    json.dumps(payload),
                    error_dict.get("type", "Unknown"),
                    error_dict.get("message", str(error)),
                    json.dumps(error_dict.get("details", {})),
                    attempts,
                    now,  # first_failed_at
                    now,  # last_failed_at
                    json.dumps(metadata or {}),
                    now,
                ),
            )
            await self._db.commit()

        logger.warning(
            "message_sent_to_dlq",
            message_id=message_id,
            trace_id=trace_id,
            error_type=error_dict.get("type"),
            attempts=attempts,
        )
        metrics.increment_dlq_message()

    async def get(self, message_id: str) -> Optional[DLQMessage]:
        """Retrieve a message from the DLQ by ID.

        Args:
            message_id: Message identifier.

        Returns:
            DLQMessage or None if not found.
        """
        if not self._initialized:
            await self.initialize()

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM dlq_messages WHERE id = ?", (message_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return DLQMessage(
                    id=row["id"],
                    trace_id=row["trace_id"],
                    payload=json.loads(row["payload"]),
                    error_type=row["error_type"],
                    error_message=row["error_message"],
                    error_details=json.loads(row["error_details"]),
                    attempts=row["attempts"],
                    first_failed_at=datetime.fromisoformat(row["first_failed_at"]),
                    last_failed_at=datetime.fromisoformat(row["last_failed_at"]),
                    metadata=json.loads(row["metadata"]),
                )
        return None

    async def list(
        self,
        limit: int = 100,
        error_type: Optional[str] = None,
    ) -> List[DLQMessage]:
        """List messages in the DLQ.

        Args:
            limit: Maximum number of messages to return.
            error_type: Filter by error type.

        Returns:
            List of DLQ messages.
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM dlq_messages WHERE 1=1"
        params: List[Any] = []

        if error_type:
            query += " AND error_type = ?"
            params.append(error_type)

        query += " ORDER BY last_failed_at DESC LIMIT ?"
        params.append(limit)

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                DLQMessage(
                    id=row["id"],
                    trace_id=row["trace_id"],
                    payload=json.loads(row["payload"]),
                    error_type=row["error_type"],
                    error_message=row["error_message"],
                    error_details=json.loads(row["error_details"]),
                    attempts=row["attempts"],
                    first_failed_at=datetime.fromisoformat(row["first_failed_at"]),
                    last_failed_at=datetime.fromisoformat(row["last_failed_at"]),
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]

    async def delete(self, message_id: str) -> bool:
        """Delete a message from the DLQ.

        Args:
            message_id: Message identifier.

        Returns:
            True if deleted, False if not found.
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM dlq_messages WHERE id = ?", (message_id,)
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def count(self, error_type: Optional[str] = None) -> int:
        """Count messages in the DLQ.

        Args:
            error_type: Filter by error type.

        Returns:
            Number of messages.
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT COUNT(*) FROM dlq_messages WHERE 1=1"
        params: List[Any] = []

        if error_type:
            query += " AND error_type = ?"
            params.append(error_type)

        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


# ============================================================================
# Circuit Breaker
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""

    failure_threshold: int = Field(default=5, ge=1, description="Failures to open circuit")
    success_threshold: int = Field(default=2, ge=1, description="Successes to close from half-open")
    timeout_ms: int = Field(default=60000, ge=1000, description="Time before trying half-open")
    half_open_max_calls: int = Field(default=3, ge=1, description="Max calls in half-open state")


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """Circuit breaker with state persistence across restarts.

    Protects services from cascading failures by temporarily blocking
    requests when failure rate is high.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        db_path: Optional[Path] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Unique name for this circuit.
            config: Circuit breaker configuration.
            db_path: Path to SQLite database for persistence.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        if db_path:
            self.db_path = db_path
        else:
            base_path = Path.home() / ".tinyllm" / "data"
            base_path.mkdir(parents=True, exist_ok=True)
            self.db_path = base_path / "circuit_breaker.db"

        self._db: Optional[aiosqlite.Connection] = None
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database and load persisted state."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._db = await aiosqlite.connect(str(self.db_path))
            await self._db.execute("PRAGMA journal_mode=WAL")

            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS circuit_state (
                    name TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    failure_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    last_failure_time REAL,
                    last_state_change REAL NOT NULL,
                    total_calls INTEGER DEFAULT 0,
                    total_failures INTEGER DEFAULT 0,
                    total_successes INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)

            await self._db.commit()

            # Load persisted state
            await self._load_state()

            self._initialized = True
            logger.info(
                "circuit_breaker_initialized",
                name=self.name,
                state=self._stats.state,
            )

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._save_state()
            await self._db.close()
            self._db = None
        self._initialized = False

    async def _load_state(self) -> None:
        """Load circuit state from database."""
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM circuit_state WHERE name = ?", (self.name,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                self._stats = CircuitStats(
                    state=CircuitState(row["state"]),
                    failure_count=row["failure_count"],
                    success_count=row["success_count"],
                    last_failure_time=row["last_failure_time"],
                    last_state_change=row["last_state_change"],
                    total_calls=row["total_calls"],
                    total_failures=row["total_failures"],
                    total_successes=row["total_successes"],
                )
                logger.info(
                    "circuit_state_loaded",
                    name=self.name,
                    state=self._stats.state,
                    failure_count=self._stats.failure_count,
                )

    async def _save_state(self) -> None:
        """Persist circuit state to database."""
        if not self._db:
            return

        now = datetime.utcnow().isoformat()

        await self._db.execute(
            """
            INSERT OR REPLACE INTO circuit_state
            (name, state, failure_count, success_count, last_failure_time,
             last_state_change, total_calls, total_failures, total_successes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.name,
                self._stats.state.value,
                self._stats.failure_count,
                self._stats.success_count,
                self._stats.last_failure_time,
                self._stats.last_state_change,
                self._stats.total_calls,
                self._stats.total_failures,
                self._stats.total_successes,
                now,
            ),
        )
        await self._db.commit()

    async def _check_half_open_timeout(self) -> None:
        """Check if enough time has passed to try half-open state."""
        if self._stats.state != CircuitState.OPEN:
            return

        if not self._stats.last_failure_time:
            return

        elapsed_ms = (time.time() - self._stats.last_failure_time) * 1000
        if elapsed_ms >= self.config.timeout_ms:
            await self._transition_to(CircuitState.HALF_OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new circuit state.

        Args:
            new_state: Target state.
        """
        old_state = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._stats.success_count = 0
            self._stats.failure_count = 0
        elif new_state == CircuitState.CLOSED:
            self._stats.failure_count = 0
            self._stats.success_count = 0

        await self._save_state()

        logger.info(
            "circuit_state_changed",
            name=self.name,
            old_state=old_state,
            new_state=new_state,
        )
        metrics.record_circuit_state_change(self.name, new_state.value)

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            CircuitOpenError: If circuit is open.
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            self._stats.total_calls += 1
            await self._check_half_open_timeout()

            # Check if circuit is open
            if self._stats.state == CircuitState.OPEN:
                logger.warning(
                    "circuit_open_rejected",
                    name=self.name,
                    failure_count=self._stats.failure_count,
                )
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    service=self.name,
                )

            # Check half-open max calls
            if self._stats.state == CircuitState.HALF_OPEN:
                current_attempts = self._stats.success_count + self._stats.failure_count
                if current_attempts >= self.config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' half-open max calls reached",
                        service=self.name,
                    )

        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            async with self._lock:
                self._stats.success_count += 1
                self._stats.total_successes += 1

                if self._stats.state == CircuitState.HALF_OPEN:
                    if self._stats.success_count >= self.config.success_threshold:
                        await self._transition_to(CircuitState.CLOSED)

                await self._save_state()

            return result

        except Exception as e:
            # Record failure
            async with self._lock:
                self._stats.failure_count += 1
                self._stats.total_failures += 1
                self._stats.last_failure_time = time.time()

                if self._stats.state == CircuitState.HALF_OPEN:
                    # Any failure in half-open goes back to open
                    await self._transition_to(CircuitState.OPEN)
                elif self._stats.failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

                await self._save_state()

            raise

    @asynccontextmanager
    async def protect(self):
        """Context manager for circuit breaker protection.

        Example:
            async with circuit.protect():
                result = await risky_operation()
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            self._stats.total_calls += 1
            await self._check_half_open_timeout()

            if self._stats.state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    service=self.name,
                )

        try:
            yield

            # Record success
            async with self._lock:
                self._stats.success_count += 1
                self._stats.total_successes += 1

                if self._stats.state == CircuitState.HALF_OPEN:
                    if self._stats.success_count >= self.config.success_threshold:
                        await self._transition_to(CircuitState.CLOSED)

                await self._save_state()

        except Exception as e:
            # Record failure
            async with self._lock:
                self._stats.failure_count += 1
                self._stats.total_failures += 1
                self._stats.last_failure_time = time.time()

                if self._stats.state == CircuitState.HALF_OPEN:
                    await self._transition_to(CircuitState.OPEN)
                elif self._stats.failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

                await self._save_state()

            raise

    def get_stats(self) -> CircuitStats:
        """Get current circuit breaker statistics.

        Returns:
            Circuit statistics.
        """
        return self._stats

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            logger.info("circuit_breaker_reset", name=self.name)


# ============================================================================
# Error Recovery Playbooks
# ============================================================================


class RecoveryAction(str, Enum):
    """Available recovery actions."""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ALERT = "alert"
    CIRCUIT_OPEN = "circuit_open"
    REDUCE_LOAD = "reduce_load"
    SWITCH_MODEL = "switch_model"
    RESTART_SERVICE = "restart_service"


class RecoveryPlaybook(BaseModel):
    """Defines automated recovery steps for specific error types."""

    error_pattern: str = Field(description="Error type or pattern to match")
    actions: List[RecoveryAction] = Field(description="Recovery actions to take")
    max_attempts: int = Field(default=3, description="Max recovery attempts")
    cooldown_ms: int = Field(default=5000, description="Cooldown between attempts")
    escalate_after: int = Field(default=3, description="Escalate after N failures")


class ErrorRecoveryManager:
    """Manages automated error recovery using playbooks."""

    def __init__(self):
        """Initialize recovery manager with default playbooks."""
        self._playbooks: Dict[str, RecoveryPlaybook] = {}
        self._recovery_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0, "failures": 0}
        )
        self._lock = asyncio.Lock()

        # Register default playbooks
        self._register_default_playbooks()

    def _register_default_playbooks(self) -> None:
        """Register default recovery playbooks for common error types."""
        # Timeout errors: retry with backoff
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="timeout",
                actions=[RecoveryAction.RETRY, RecoveryAction.REDUCE_LOAD],
                max_attempts=3,
                cooldown_ms=2000,
            )
        )

        # Rate limit errors: wait and retry, then reduce load
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="rate_limit",
                actions=[RecoveryAction.RETRY, RecoveryAction.REDUCE_LOAD],
                max_attempts=5,
                cooldown_ms=10000,
            )
        )

        # Network errors: retry, then alert
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="network",
                actions=[RecoveryAction.RETRY, RecoveryAction.ALERT],
                max_attempts=3,
                cooldown_ms=5000,
            )
        )

        # Model errors: retry, fallback to simpler model, then alert
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="model",
                actions=[RecoveryAction.RETRY, RecoveryAction.SWITCH_MODEL, RecoveryAction.ALERT],
                max_attempts=2,
                cooldown_ms=3000,
            )
        )

        # Resource exhaustion: reduce load, open circuit
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="resource_exhausted",
                actions=[RecoveryAction.REDUCE_LOAD, RecoveryAction.CIRCUIT_OPEN],
                max_attempts=1,
                cooldown_ms=30000,
            )
        )

        # Circuit open: skip request, alert
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="circuit_open",
                actions=[RecoveryAction.SKIP, RecoveryAction.ALERT],
                max_attempts=1,
                cooldown_ms=60000,
            )
        )

        # Validation errors: skip (no retry)
        self.register_playbook(
            RecoveryPlaybook(
                error_pattern="validation",
                actions=[RecoveryAction.SKIP, RecoveryAction.ALERT],
                max_attempts=1,
                cooldown_ms=0,
            )
        )

    def register_playbook(self, playbook: RecoveryPlaybook) -> None:
        """Register a recovery playbook.

        Args:
            playbook: Recovery playbook to register.
        """
        self._playbooks[playbook.error_pattern] = playbook
        logger.info(
            "recovery_playbook_registered",
            pattern=playbook.error_pattern,
            actions=[a.value for a in playbook.actions],
        )

    def get_playbook(self, error: Exception) -> Optional[RecoveryPlaybook]:
        """Get recovery playbook for an error.

        Args:
            error: Exception to find playbook for.

        Returns:
            Matching playbook or None.
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Try exact match on error type
        if error_type in self._playbooks:
            return self._playbooks[error_type]

        # Try pattern matching on error message
        for pattern, playbook in self._playbooks.items():
            if pattern in error_str or pattern in error_type:
                return playbook

        return None

    async def execute_recovery(
        self,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute recovery actions for an error.

        Args:
            error: Exception that occurred.
            context: Execution context.

        Returns:
            Dictionary with recovery results.
        """
        playbook = self.get_playbook(error)
        if not playbook:
            logger.warning(
                "no_recovery_playbook",
                error_type=type(error).__name__,
                error=str(error),
            )
            return {"recovered": False, "actions_taken": []}

        async with self._lock:
            stats = self._recovery_stats[playbook.error_pattern]
            stats["attempts"] += 1

        logger.info(
            "executing_recovery_playbook",
            error_type=type(error).__name__,
            pattern=playbook.error_pattern,
            actions=[a.value for a in playbook.actions],
        )

        actions_taken = []
        recovered = False

        for action in playbook.actions:
            try:
                result = await self._execute_action(action, error, context, playbook)
                actions_taken.append({"action": action.value, "result": result})

                if result.get("success", False):
                    recovered = True
                    break

            except Exception as action_error:
                logger.error(
                    "recovery_action_failed",
                    action=action.value,
                    error=str(action_error),
                )
                actions_taken.append(
                    {"action": action.value, "error": str(action_error)}
                )

        async with self._lock:
            stats = self._recovery_stats[playbook.error_pattern]
            if recovered:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

        return {
            "recovered": recovered,
            "actions_taken": actions_taken,
            "playbook": playbook.error_pattern,
        }

    async def _execute_action(
        self,
        action: RecoveryAction,
        error: Exception,
        context: Dict[str, Any],
        playbook: RecoveryPlaybook,
    ) -> Dict[str, Any]:
        """Execute a single recovery action.

        Args:
            action: Action to execute.
            error: Original error.
            context: Execution context.
            playbook: Playbook being executed.

        Returns:
            Action result dictionary.
        """
        if action == RecoveryAction.RETRY:
            # Signal that retry should be attempted
            return {
                "success": True,
                "message": "Retry scheduled",
                "delay_ms": playbook.cooldown_ms,
            }

        elif action == RecoveryAction.FALLBACK:
            # Signal fallback mode should be used
            return {
                "success": True,
                "message": "Fallback mode activated",
            }

        elif action == RecoveryAction.SKIP:
            # Skip this operation
            return {
                "success": True,
                "message": "Operation skipped",
            }

        elif action == RecoveryAction.ALERT:
            # Log alert (could integrate with alerting systems)
            logger.error(
                "recovery_alert",
                error_type=type(error).__name__,
                error=str(error),
                context=context,
                playbook=playbook.error_pattern,
            )
            metrics.increment_error_count(
                error_type="recovery_alert",
                model=context.get("model", "unknown"),
                graph=context.get("graph", "unknown"),
            )
            return {
                "success": True,
                "message": "Alert sent",
            }

        elif action == RecoveryAction.CIRCUIT_OPEN:
            # Signal circuit breaker should open
            return {
                "success": True,
                "message": "Circuit breaker opened",
            }

        elif action == RecoveryAction.REDUCE_LOAD:
            # Signal load reduction (e.g., reject new requests)
            return {
                "success": True,
                "message": "Load reduction activated",
            }

        elif action == RecoveryAction.SWITCH_MODEL:
            # Signal model switch (implementation specific)
            return {
                "success": True,
                "message": "Model switch requested",
            }

        elif action == RecoveryAction.RESTART_SERVICE:
            # Log restart request (implementation specific)
            logger.error(
                "service_restart_requested",
                error_type=type(error).__name__,
                error=str(error),
            )
            return {
                "success": False,
                "message": "Service restart requires manual intervention",
            }

        return {"success": False, "message": f"Unknown action: {action}"}

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics.

        Returns:
            Dictionary with recovery stats per error pattern.
        """
        return dict(self._recovery_stats)


# Global recovery manager instance
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager.

    Returns:
        Global ErrorRecoveryManager instance.
    """
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


# ============================================================================
# Global Exception Handler
# ============================================================================


def global_exception_handler(
    dlq: Optional[DeadLetterQueue] = None,
    send_to_dlq: bool = True,
    enable_recovery: bool = True,
):
    """Decorator for global exception handling.

    Args:
        dlq: Dead letter queue instance (creates new if None).
        send_to_dlq: Whether to send failed messages to DLQ.
        enable_recovery: Whether to enable automated recovery.

    Example:
        @global_exception_handler()
        async def process_message(msg):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except TinyLLMError as e:
                logger.error(
                    "tinyllm_error",
                    function=func.__name__,
                    error_type=e.__class__.__name__,
                    error_code=e.code,
                    error_message=e.message,
                    recoverable=e.recoverable,
                    details=e.details,
                )
                metrics.increment_error(e.code)

                # Attempt automated recovery
                if enable_recovery and e.recoverable:
                    recovery_manager = get_recovery_manager()
                    context = {
                        "function": func.__name__,
                        "model": kwargs.get("model", "unknown"),
                        "graph": kwargs.get("graph", "unknown"),
                    }
                    recovery_result = await recovery_manager.execute_recovery(e, context)
                    logger.info(
                        "recovery_attempted",
                        recovered=recovery_result["recovered"],
                        actions=recovery_result["actions_taken"],
                    )

                # Send to DLQ if configured and retryable
                if send_to_dlq and dlq and e.recoverable:
                    message_id = kwargs.get("message_id", "unknown")
                    trace_id = kwargs.get("trace_id", "unknown")
                    payload = kwargs.get("payload", {})
                    await dlq.enqueue(message_id, trace_id, payload, e)

                raise

            except Exception as e:
                logger.error(
                    "unexpected_error",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                metrics.increment_error("UNKNOWN")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except TinyLLMError as e:
                logger.error(
                    "tinyllm_error",
                    function=func.__name__,
                    error_type=e.__class__.__name__,
                    error_code=e.code,
                    error_message=e.message,
                    recoverable=e.recoverable,
                    details=e.details,
                )
                metrics.increment_error(e.code)
                raise

            except Exception as e:
                logger.error(
                    "unexpected_error",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                metrics.increment_error("UNKNOWN")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
