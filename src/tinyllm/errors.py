"""Enhanced error handling for TinyLLM.

This module provides rich error classes with context enrichment,
including stack traces, execution state, and input capture.
"""

import sys
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    NETWORK = "network"
    MODEL = "model"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    DATA = "data"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class ErrorContext(BaseModel):
    """Rich context information for errors.

    Captures execution state, stack traces, and input data
    to aid in debugging and error analysis.
    """

    model_config = {"extra": "forbid"}

    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When error occurred"
    )

    # Stack information
    stack_trace: str = Field(description="Full stack trace")
    exception_type: str = Field(description="Exception class name")
    exception_message: str = Field(description="Exception message")

    # Execution context
    trace_id: Optional[str] = Field(
        default=None, description="Trace ID if available"
    )
    node_id: Optional[str] = Field(
        default=None, description="Node ID where error occurred"
    )
    graph_id: Optional[str] = Field(
        default=None, description="Graph ID being executed"
    )
    execution_step: Optional[int] = Field(
        default=None, description="Step number in execution"
    )

    # State capture
    context_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Execution context variables"
    )
    node_config: Dict[str, Any] = Field(
        default_factory=dict, description="Node configuration"
    )
    visited_nodes: List[str] = Field(
        default_factory=list, description="Nodes visited before error"
    )

    # Input capture
    input_message: Optional[Dict[str, Any]] = Field(
        default=None, description="Input message that caused error"
    )
    input_payload: Optional[Dict[str, Any]] = Field(
        default=None, description="Payload data"
    )

    # Additional context
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # System information
    python_version: str = Field(
        default_factory=lambda: sys.version.split()[0],
        description="Python version"
    )
    process_id: int = Field(
        default_factory=lambda: sys.getpid(),
        description="Process ID"
    )


class EnrichedError(BaseModel):
    """Error with enriched context.

    Provides comprehensive error information including
    categorization, severity, and full context.
    """

    model_config = {"extra": "forbid"}

    # Core error information
    error_id: str = Field(description="Unique error identifier")
    category: ErrorCategory = Field(description="Error category")
    severity: ErrorSeverity = Field(description="Error severity")
    message: str = Field(description="Human-readable error message")

    # Context
    context: ErrorContext = Field(description="Rich error context")

    # Classification
    is_retryable: bool = Field(
        default=False, description="Whether error is retryable"
    )
    is_transient: bool = Field(
        default=False, description="Whether error is likely transient"
    )
    is_user_error: bool = Field(
        default=False, description="Whether caused by user input"
    )

    # Recovery suggestions
    suggested_actions: List[str] = Field(
        default_factory=list, description="Suggested recovery actions"
    )
    recovery_strategies: List[str] = Field(
        default_factory=list, description="Possible recovery strategies"
    )

    # Related errors
    caused_by: Optional[str] = Field(
        default=None, description="ID of error that caused this one"
    )
    related_errors: List[str] = Field(
        default_factory=list, description="IDs of related errors"
    )

    # Impact
    affected_components: List[str] = Field(
        default_factory=list, description="Components affected by this error"
    )

    # Metrics
    occurrence_count: int = Field(
        default=1, ge=1, description="Number of times this error occurred"
    )
    first_seen: datetime = Field(
        default_factory=datetime.utcnow, description="First occurrence"
    )
    last_seen: datetime = Field(
        default_factory=datetime.utcnow, description="Last occurrence"
    )


class TinyLLMError(Exception):
    """Base exception for TinyLLM with context enrichment.

    All TinyLLM exceptions should inherit from this class
    to benefit from automatic context capture and enrichment.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        trace_id: Optional[str] = None,
        node_id: Optional[str] = None,
        graph_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize error with context.

        Args:
            message: Error message.
            category: Error category.
            severity: Error severity.
            trace_id: Optional trace ID.
            node_id: Optional node ID.
            graph_id: Optional graph ID.
            metadata: Additional metadata.
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.trace_id = trace_id
        self.node_id = node_id
        self.graph_id = graph_id
        self.metadata = metadata or {}

        # Capture stack trace
        self._capture_stack()

    def _capture_stack(self) -> None:
        """Capture stack trace at error creation."""
        self.stack_trace = "".join(traceback.format_exception(*sys.exc_info()))
        if self.stack_trace.strip() == "":
            # If no exception in flight, capture current stack
            self.stack_trace = "".join(traceback.format_stack())

    def to_enriched_error(
        self,
        error_id: str,
        context_variables: Optional[Dict[str, Any]] = None,
        node_config: Optional[Dict[str, Any]] = None,
        visited_nodes: Optional[List[str]] = None,
        input_message: Optional[Dict[str, Any]] = None,
        input_payload: Optional[Dict[str, Any]] = None,
        execution_step: Optional[int] = None,
    ) -> EnrichedError:
        """Convert to enriched error with full context.

        Args:
            error_id: Unique error identifier.
            context_variables: Execution context variables.
            node_config: Node configuration.
            visited_nodes: Nodes visited before error.
            input_message: Input message.
            input_payload: Input payload.
            execution_step: Execution step number.

        Returns:
            EnrichedError with full context.
        """
        context = ErrorContext(
            timestamp=datetime.utcnow(),
            stack_trace=self.stack_trace,
            exception_type=self.__class__.__name__,
            exception_message=self.message,
            trace_id=self.trace_id,
            node_id=self.node_id,
            graph_id=self.graph_id,
            execution_step=execution_step,
            context_variables=context_variables or {},
            node_config=node_config or {},
            visited_nodes=visited_nodes or [],
            input_message=input_message,
            input_payload=input_payload,
            metadata=self.metadata,
        )

        return EnrichedError(
            error_id=error_id,
            category=self.category,
            severity=self.severity,
            message=self.message,
            context=context,
            is_retryable=self._is_retryable(),
            is_transient=self._is_transient(),
            is_user_error=self._is_user_error(),
            suggested_actions=self._get_suggested_actions(),
            recovery_strategies=self._get_recovery_strategies(),
            affected_components=self._get_affected_components(),
        )

    def _is_retryable(self) -> bool:
        """Determine if error is retryable.

        Returns:
            True if error might succeed on retry.
        """
        return self.category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RESOURCE,
        }

    def _is_transient(self) -> bool:
        """Determine if error is transient.

        Returns:
            True if error is likely temporary.
        """
        return self.category in {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.RESOURCE,
        }

    def _is_user_error(self) -> bool:
        """Determine if error is caused by user.

        Returns:
            True if error is due to user input.
        """
        return self.category in {
            ErrorCategory.VALIDATION,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.PERMISSION,
        }

    def _get_suggested_actions(self) -> List[str]:
        """Get suggested recovery actions.

        Returns:
            List of suggested actions.
        """
        actions = []

        if self._is_retryable():
            actions.append("Retry the operation")

        if self.category == ErrorCategory.TIMEOUT:
            actions.extend([
                "Increase timeout value",
                "Check if operation is hanging",
                "Review system resources",
            ])
        elif self.category == ErrorCategory.NETWORK:
            actions.extend([
                "Check network connectivity",
                "Verify service availability",
                "Review firewall settings",
            ])
        elif self.category == ErrorCategory.RESOURCE:
            actions.extend([
                "Check available memory",
                "Review resource limits",
                "Scale resources if needed",
            ])
        elif self.category == ErrorCategory.VALIDATION:
            actions.extend([
                "Check input format",
                "Review validation rules",
                "Verify input data",
            ])
        elif self.category == ErrorCategory.MODEL:
            actions.extend([
                "Verify model is loaded",
                "Check model configuration",
                "Review model compatibility",
            ])

        return actions

    def _get_recovery_strategies(self) -> List[str]:
        """Get recovery strategies.

        Returns:
            List of recovery strategies.
        """
        strategies = []

        if self._is_retryable():
            strategies.append("exponential_backoff")

        if self.category == ErrorCategory.TIMEOUT:
            strategies.extend(["increase_timeout", "fallback_node"])
        elif self.category == ErrorCategory.NETWORK:
            strategies.extend(["retry_with_backoff", "fallback_service"])
        elif self.category == ErrorCategory.RESOURCE:
            strategies.extend(["prune_memory", "scale_up"])
        elif self.category == ErrorCategory.MODEL:
            strategies.extend(["fallback_model", "reload_model"])

        return strategies

    def _get_affected_components(self) -> List[str]:
        """Get affected components.

        Returns:
            List of affected component identifiers.
        """
        components = []

        if self.node_id:
            components.append(f"node:{self.node_id}")
        if self.graph_id:
            components.append(f"graph:{self.graph_id}")
        if self.trace_id:
            components.append(f"trace:{self.trace_id}")

        return components


# Specific error types

class ValidationError(TinyLLMError):
    """Validation error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ExecutionError(TinyLLMError):
    """Execution error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class TimeoutError(TinyLLMError):
    """Timeout error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )


class ResourceError(TinyLLMError):
    """Resource error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class NetworkError(TinyLLMError):
    """Network error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )


class ModelError(TinyLLMError):
    """Model error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ConfigurationError(TinyLLMError):
    """Configuration error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
