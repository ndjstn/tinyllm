"""Tool error handling for TinyLLM.

This module provides comprehensive error handling for tools,
including error categorization, recovery strategies, and context.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories for tool errors."""

    VALIDATION = "validation"  # Input/output validation errors
    EXECUTION = "execution"  # Runtime execution errors
    TIMEOUT = "timeout"  # Timeout errors
    RESOURCE = "resource"  # Resource exhaustion (memory, disk, etc.)
    NETWORK = "network"  # Network-related errors
    PERMISSION = "permission"  # Permission/authorization errors
    CONFIGURATION = "configuration"  # Configuration errors
    DEPENDENCY = "dependency"  # Missing dependencies
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    INTERNAL = "internal"  # Internal/unexpected errors
    USER = "user"  # User-caused errors


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    CRITICAL = "critical"  # System-wide impact
    HIGH = "high"  # Major functionality affected
    MEDIUM = "medium"  # Some functionality affected
    LOW = "low"  # Minor impact
    INFO = "info"  # Informational only


@dataclass
class ErrorContext:
    """Context information for an error."""

    tool_id: str
    input_data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt: int = 1
    trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, key: str, value: Any) -> "ErrorContext":
        """Add metadata to context.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for chaining.
        """
        self.metadata[key] = value
        return self


@dataclass
class ToolError:
    """Represents a tool execution error."""

    message: str
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    original_exception: Optional[Exception] = None
    context: Optional[ErrorContext] = None
    recoverable: bool = True
    recovery_hint: Optional[str] = None

    @property
    def exception_type(self) -> Optional[str]:
        """Get the type of the original exception."""
        if self.original_exception:
            return type(self.original_exception).__name__
        return None

    @property
    def full_trace(self) -> Optional[str]:
        """Get full stack trace if available."""
        if self.context and self.context.trace:
            return self.context.trace
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "exception_type": self.exception_type,
            "recoverable": self.recoverable,
            "recovery_hint": self.recovery_hint,
            "tool_id": self.context.tool_id if self.context else None,
            "timestamp": self.context.timestamp.isoformat() if self.context else None,
        }


class ErrorClassifier:
    """Classifies exceptions into error categories."""

    def __init__(self):
        """Initialize classifier with default rules."""
        self._rules: List[tuple[Type[Exception], ErrorCategory, ErrorSeverity]] = [
            (ValueError, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            (TypeError, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            (TimeoutError, ErrorCategory.TIMEOUT, ErrorSeverity.HIGH),
            (PermissionError, ErrorCategory.PERMISSION, ErrorSeverity.HIGH),
            (FileNotFoundError, ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM),
            (MemoryError, ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
            (ConnectionError, ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            (OSError, ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM),
            (ImportError, ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH),
            (ModuleNotFoundError, ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH),
            (KeyError, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            (AttributeError, ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM),
            (RuntimeError, ErrorCategory.EXECUTION, ErrorSeverity.MEDIUM),
        ]

    def add_rule(
        self,
        exception_type: Type[Exception],
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ) -> "ErrorClassifier":
        """Add a classification rule.

        Args:
            exception_type: Exception type to match.
            category: Category to assign.
            severity: Severity to assign.

        Returns:
            Self for chaining.
        """
        # Insert at beginning to give precedence to newer rules
        self._rules.insert(0, (exception_type, category, severity))
        return self

    def classify(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify an exception.

        Args:
            exception: Exception to classify.

        Returns:
            Tuple of (category, severity).
        """
        for exc_type, category, severity in self._rules:
            if isinstance(exception, exc_type):
                return category, severity

        return ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM


class RecoveryStrategy(ABC):
    """Abstract base for recovery strategies."""

    @abstractmethod
    async def recover(
        self, error: ToolError, tool: Any, input_data: Any
    ) -> Optional[Any]:
        """Attempt to recover from error.

        Args:
            error: The error that occurred.
            tool: The tool that failed.
            input_data: Original input data.

        Returns:
            Recovered result or None if recovery failed.
        """
        pass


class RetryRecovery(RecoveryStrategy):
    """Retry the operation."""

    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        """Initialize retry recovery.

        Args:
            max_retries: Maximum retry attempts.
            delay: Delay between retries in seconds.
        """
        self.max_retries = max_retries
        self.delay = delay

    async def recover(
        self, error: ToolError, tool: Any, input_data: Any
    ) -> Optional[Any]:
        """Retry the tool execution."""
        import asyncio

        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.delay * (attempt + 1))
                return await tool.execute(input_data)
            except Exception:
                continue

        return None


class FallbackRecovery(RecoveryStrategy):
    """Use a fallback tool."""

    def __init__(self, fallback_tool: Any):
        """Initialize fallback recovery.

        Args:
            fallback_tool: Tool to use as fallback.
        """
        self.fallback_tool = fallback_tool

    async def recover(
        self, error: ToolError, tool: Any, input_data: Any
    ) -> Optional[Any]:
        """Execute fallback tool."""
        try:
            return await self.fallback_tool.execute(input_data)
        except Exception:
            return None


class DefaultValueRecovery(RecoveryStrategy):
    """Return a default value."""

    def __init__(self, default_value: Any):
        """Initialize default value recovery.

        Args:
            default_value: Value to return on error.
        """
        self.default_value = default_value

    async def recover(
        self, error: ToolError, tool: Any, input_data: Any
    ) -> Optional[Any]:
        """Return the default value."""
        if callable(self.default_value):
            return self.default_value(error, input_data)
        return self.default_value


class TransformRecovery(RecoveryStrategy):
    """Transform input and retry."""

    def __init__(self, transformer: Callable[[Any, ToolError], Any]):
        """Initialize transform recovery.

        Args:
            transformer: Function to transform input.
        """
        self.transformer = transformer

    async def recover(
        self, error: ToolError, tool: Any, input_data: Any
    ) -> Optional[Any]:
        """Transform input and retry."""
        try:
            transformed = self.transformer(input_data, error)
            return await tool.execute(transformed)
        except Exception:
            return None


class ErrorHandler:
    """Main error handler for tools."""

    def __init__(
        self,
        classifier: Optional[ErrorClassifier] = None,
        on_error: Optional[Callable[[ToolError], None]] = None,
    ):
        """Initialize error handler.

        Args:
            classifier: Error classifier to use.
            on_error: Callback for all errors.
        """
        self.classifier = classifier or ErrorClassifier()
        self.on_error = on_error
        self._recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {}
        self._error_history: List[ToolError] = []
        self._max_history = 100

    def add_recovery(
        self,
        category: ErrorCategory,
        strategy: RecoveryStrategy,
    ) -> "ErrorHandler":
        """Add a recovery strategy for a category.

        Args:
            category: Error category.
            strategy: Recovery strategy.

        Returns:
            Self for chaining.
        """
        if category not in self._recovery_strategies:
            self._recovery_strategies[category] = []
        self._recovery_strategies[category].append(strategy)
        return self

    def handle(
        self,
        exception: Exception,
        tool_id: str,
        input_data: Any = None,
        attempt: int = 1,
    ) -> ToolError:
        """Handle an exception and create a ToolError.

        Args:
            exception: The exception that occurred.
            tool_id: ID of the tool that failed.
            input_data: Original input data.
            attempt: Attempt number.

        Returns:
            ToolError with classification.
        """
        category, severity = self.classifier.classify(exception)

        context = ErrorContext(
            tool_id=tool_id,
            input_data=input_data,
            attempt=attempt,
            trace=traceback.format_exc(),
        )

        error = ToolError(
            message=str(exception),
            category=category,
            severity=severity,
            original_exception=exception,
            context=context,
            recoverable=category in self._recovery_strategies,
            recovery_hint=self._get_recovery_hint(category),
        )

        # Track history
        self._error_history.append(error)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        # Call error callback
        if self.on_error:
            try:
                self.on_error(error)
            except Exception:
                pass

        logger.warning(
            f"Tool {tool_id} error: [{category.value}] {exception}",
            extra={"error": error.to_dict()},
        )

        return error

    async def recover(
        self,
        error: ToolError,
        tool: Any,
        input_data: Any,
    ) -> Optional[Any]:
        """Attempt to recover from an error.

        Args:
            error: The error to recover from.
            tool: The tool that failed.
            input_data: Original input data.

        Returns:
            Recovered result or None.
        """
        strategies = self._recovery_strategies.get(error.category, [])

        for strategy in strategies:
            try:
                result = await strategy.recover(error, tool, input_data)
                if result is not None:
                    logger.info(
                        f"Recovered from {error.category.value} error using "
                        f"{type(strategy).__name__}"
                    )
                    return result
            except Exception as e:
                logger.debug(f"Recovery strategy failed: {e}")

        return None

    def _get_recovery_hint(self, category: ErrorCategory) -> Optional[str]:
        """Get recovery hint for a category."""
        hints = {
            ErrorCategory.VALIDATION: "Check input parameters and types",
            ErrorCategory.TIMEOUT: "Consider increasing timeout or simplifying request",
            ErrorCategory.NETWORK: "Check network connectivity and try again",
            ErrorCategory.PERMISSION: "Verify permissions and authentication",
            ErrorCategory.RATE_LIMIT: "Wait before retrying or reduce request rate",
            ErrorCategory.RESOURCE: "Check available resources",
            ErrorCategory.CONFIGURATION: "Review tool configuration",
            ErrorCategory.DEPENDENCY: "Install missing dependencies",
        }
        return hints.get(category)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Dictionary with error statistics.
        """
        if not self._error_history:
            return {"total": 0, "by_category": {}, "by_severity": {}}

        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for error in self._error_history:
            cat = error.category.value
            sev = error.severity.value

            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total": len(self._error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "most_common": max(by_category, key=by_category.get) if by_category else None,
        }

    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()


class ErrorHandlingToolWrapper:
    """Wrapper that adds error handling to tools."""

    def __init__(
        self,
        tool: Any,
        handler: Optional[ErrorHandler] = None,
        reraise: bool = False,
        error_result: Optional[Callable[[ToolError], Any]] = None,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            handler: Error handler to use.
            reraise: Whether to re-raise after handling.
            error_result: Function to create result from error.
        """
        self.tool = tool
        self.handler = handler or ErrorHandler()
        self.reraise = reraise
        self.error_result = error_result

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input_data: Any) -> Any:
        """Execute tool with error handling.

        Args:
            input_data: Tool input.

        Returns:
            Tool result or error result.

        Raises:
            Exception: If reraise is True.
        """
        try:
            return await self.tool.execute(input_data)

        except Exception as e:
            error = self.handler.handle(
                exception=e,
                tool_id=self.tool.metadata.id,
                input_data=input_data,
            )

            # Try recovery
            if error.recoverable:
                recovered = await self.handler.recover(error, self.tool, input_data)
                if recovered is not None:
                    return recovered

            # Generate error result
            if self.error_result:
                return self.error_result(error)

            # Re-raise if configured
            if self.reraise:
                raise

            # Return None as last resort
            return None


class ErrorBoundary:
    """Provides error boundary for tool execution."""

    def __init__(
        self,
        on_error: Optional[Callable[[ToolError], None]] = None,
        fallback: Optional[Any] = None,
    ):
        """Initialize error boundary.

        Args:
            on_error: Callback for errors.
            fallback: Fallback result on error.
        """
        self.on_error = on_error
        self.fallback = fallback
        self._handler = ErrorHandler(on_error=on_error)
        self._last_error: Optional[ToolError] = None

    @property
    def last_error(self) -> Optional[ToolError]:
        """Get the last error."""
        return self._last_error

    async def execute(self, tool: Any, input_data: Any) -> Any:
        """Execute tool within error boundary.

        Args:
            tool: Tool to execute.
            input_data: Tool input.

        Returns:
            Tool result or fallback.
        """
        try:
            return await tool.execute(input_data)

        except Exception as e:
            self._last_error = self._handler.handle(
                exception=e,
                tool_id=tool.metadata.id,
                input_data=input_data,
            )

            if callable(self.fallback):
                return self.fallback(self._last_error, input_data)
            return self.fallback


# Convenience functions


def with_error_handling(
    tool: Any,
    handler: Optional[ErrorHandler] = None,
    reraise: bool = False,
) -> ErrorHandlingToolWrapper:
    """Add error handling to a tool.

    Args:
        tool: Tool to wrap.
        handler: Error handler.
        reraise: Whether to re-raise errors.

    Returns:
        ErrorHandlingToolWrapper.
    """
    return ErrorHandlingToolWrapper(tool, handler, reraise)


def create_error_handler(
    on_error: Optional[Callable[[ToolError], None]] = None,
) -> ErrorHandler:
    """Create a new error handler.

    Args:
        on_error: Error callback.

    Returns:
        ErrorHandler instance.
    """
    return ErrorHandler(on_error=on_error)


def classify_error(exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
    """Classify an exception.

    Args:
        exception: Exception to classify.

    Returns:
        Tuple of (category, severity).
    """
    classifier = ErrorClassifier()
    return classifier.classify(exception)
