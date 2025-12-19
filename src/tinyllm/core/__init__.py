"""Core execution engine for TinyLLM."""

from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import (
    ErrorInfo,
    Message,
    MessageMetadata,
    MessagePayload,
    TaskPayload,
    TaskResponse,
    ToolCall,
    ToolResult,
)
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult, NodeStats
from tinyllm.core.registry import NodeRegistry
from tinyllm.core.graph import Graph, Edge, ValidationError
from tinyllm.core.builder import GraphBuilder, load_graph
from tinyllm.core.executor import Executor, ExecutorConfig, ExecutionError
from tinyllm.core.trace import ExecutionTrace, NodeExecution, TraceRecorder
from tinyllm.core.completion import (
    CompletionStatus,
    CompletionSignals,
    CompletionAnalysis,
    TaskCompletionDetector,
    is_task_complete,
)

__all__ = [
    # Message types
    "Message",
    "MessagePayload",
    "MessageMetadata",
    "ToolCall",
    "ToolResult",
    "ErrorInfo",
    "TaskPayload",
    "TaskResponse",
    # Node types
    "BaseNode",
    "NodeConfig",
    "NodeResult",
    "NodeStats",
    "NodeRegistry",
    # Graph
    "Graph",
    "Edge",
    "ValidationError",
    "GraphBuilder",
    "load_graph",
    # Execution
    "Executor",
    "ExecutorConfig",
    "ExecutionError",
    "ExecutionContext",
    # Tracing
    "ExecutionTrace",
    "NodeExecution",
    "TraceRecorder",
    # Completion detection
    "CompletionStatus",
    "CompletionSignals",
    "CompletionAnalysis",
    "TaskCompletionDetector",
    "is_task_complete",
]
