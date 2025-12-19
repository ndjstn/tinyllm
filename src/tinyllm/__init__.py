"""TinyLLM: A Neural Network of LLMs.

TinyLLM treats small language models as intelligent neurons in a larger
cognitive architecture, enabling self-improvement through recursive expansion.
"""

__version__ = "0.1.0"

# Core exports
from tinyllm.core import (
    BaseNode,
    Edge,
    ExecutionContext,
    ExecutionError,
    ExecutionTrace,
    Executor,
    ExecutorConfig,
    Graph,
    GraphBuilder,
    Message,
    MessageMetadata,
    MessagePayload,
    NodeConfig,
    NodeExecution,
    NodeRegistry,
    NodeResult,
    NodeStats,
    TaskPayload,
    TaskResponse,
    ToolCall,
    ToolResult,
    TraceRecorder,
    ValidationError,
    load_graph,
)

# Config exports
from tinyllm.config import (
    Config,
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    NodeType,
    load_config,
)

# Tool exports
from tinyllm.tools import (
    BaseTool,
    CalculatorTool,
    ToolConfig,
    ToolMetadata,
    ToolRegistry,
    register_default_tools,
)

__all__ = [
    "__version__",
    # Core
    "BaseNode",
    "Edge",
    "ExecutionContext",
    "ExecutionError",
    "ExecutionTrace",
    "Executor",
    "ExecutorConfig",
    "Graph",
    "GraphBuilder",
    "Message",
    "MessageMetadata",
    "MessagePayload",
    "NodeConfig",
    "NodeExecution",
    "NodeRegistry",
    "NodeResult",
    "NodeStats",
    "TaskPayload",
    "TaskResponse",
    "ToolCall",
    "ToolResult",
    "TraceRecorder",
    "ValidationError",
    "load_graph",
    # Config
    "Config",
    "EdgeDefinition",
    "GraphDefinition",
    "NodeDefinition",
    "NodeType",
    "load_config",
    # Tools
    "BaseTool",
    "CalculatorTool",
    "ToolConfig",
    "ToolMetadata",
    "ToolRegistry",
    "register_default_tools",
]
