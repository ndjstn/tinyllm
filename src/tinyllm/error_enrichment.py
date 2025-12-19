"""Error enrichment for TinyLLM.

This module extends the base error system with rich context,
impact scoring, aggregation, and notification capabilities.
"""

import sys
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.errors import TinyLLMError


class ErrorSeverity(str, Enum):
    """Error severity levels for enrichment."""

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


def enrich_error(
    error: Exception,
    error_id: str,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    trace_id: Optional[str] = None,
    node_id: Optional[str] = None,
    graph_id: Optional[str] = None,
    context_variables: Optional[Dict[str, Any]] = None,
    node_config: Optional[Dict[str, Any]] = None,
    visited_nodes: Optional[List[str]] = None,
    input_message: Optional[Dict[str, Any]] = None,
    input_payload: Optional[Dict[str, Any]] = None,
    execution_step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EnrichedError:
    """Enrich an exception with full context.

    Args:
        error: Exception to enrich.
        error_id: Unique error identifier.
        category: Error category.
        severity: Error severity.
        trace_id: Optional trace ID.
        node_id: Optional node ID.
        graph_id: Optional graph ID.
        context_variables: Execution context variables.
        node_config: Node configuration.
        visited_nodes: Nodes visited before error.
        input_message: Input message.
        input_payload: Input payload.
        execution_step: Execution step number.
        metadata: Additional metadata.

    Returns:
        EnrichedError with full context.
    """
    # Capture stack trace
    stack_trace = "".join(traceback.format_exception(*sys.exc_info()))
    if stack_trace.strip() == "":
        stack_trace = "".join(traceback.format_stack())

    # Create context
    context = ErrorContext(
        timestamp=datetime.utcnow(),
        stack_trace=stack_trace,
        exception_type=type(error).__name__,
        exception_message=str(error),
        trace_id=trace_id,
        node_id=node_id,
        graph_id=graph_id,
        execution_step=execution_step,
        context_variables=context_variables or {},
        node_config=node_config or {},
        visited_nodes=visited_nodes or [],
        input_message=input_message,
        input_payload=input_payload,
        metadata=metadata or {},
    )

    # Determine retryability
    is_retryable = isinstance(error, TinyLLMError) and error.recoverable

    # Build affected components
    affected_components = []
    if node_id:
        affected_components.append(f"node:{node_id}")
    if graph_id:
        affected_components.append(f"graph:{graph_id}")
    if trace_id:
        affected_components.append(f"trace:{trace_id}")

    return EnrichedError(
        error_id=error_id,
        category=category,
        severity=severity,
        message=str(error),
        context=context,
        is_retryable=is_retryable,
        is_transient=is_retryable,
        is_user_error=category == ErrorCategory.VALIDATION,
        affected_components=affected_components,
    )
