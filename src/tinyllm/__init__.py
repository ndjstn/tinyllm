"""TinyLLM: A Neural Network of LLMs.

TinyLLM treats small language models as intelligent neurons in a larger
cognitive architecture, enabling self-improvement through recursive expansion.
"""

__version__ = "0.1.0"

# Logging exports
from tinyllm.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    unbind_context,
)

# Core exports
from tinyllm.core import (
    BaseNode,
    CompletionAnalysis,
    CompletionSignals,
    CompletionStatus,
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
    TaskCompletionDetector,
    TaskPayload,
    TaskResponse,
    ToolCall,
    ToolResult,
    TraceRecorder,
    ValidationError,
    is_task_complete,
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

# Metrics exports
from tinyllm.metrics import (
    MetricsCollector,
    get_metrics_collector,
    start_metrics_server,
)

# Prompts exports
from tinyllm.prompts import (
    ASSISTANT_IDENTITY,
    CHAT_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PromptConfig,
    get_chat_prompt,
    get_task_prompt,
    get_identity_correction,
    get_default_config,
    set_default_config,
)

# Cache exports (Tasks 91-100)
from tinyllm.cache import (
    CacheBackend,
    CacheEntry,
    CacheMetrics,
    CacheTier,
    CachedOllamaClient,
    CompressionAlgorithm,
    InMemoryBackend,
    InvalidationStrategy,
    RedisBackend,
    ResponseCache,
    create_cached_client,
    create_memory_cache,
    create_redis_cache,
)
from tinyllm.cache_advanced import (
    AdaptiveCache,
    CacheCoherence,
    CacheInvalidator,
    CacheWarmer,
    CompressedBackend,
    CostModel,
    RedisClusterBackend,
    SemanticCache,
    TieredCache,
)

__all__ = [
    "__version__",
    # Logging
    "configure_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "clear_context",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    "start_metrics_server",
    # Core
    "BaseNode",
    "CompletionAnalysis",
    "CompletionSignals",
    "CompletionStatus",
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
    "TaskCompletionDetector",
    "TaskPayload",
    "TaskResponse",
    "ToolCall",
    "ToolResult",
    "TraceRecorder",
    "ValidationError",
    "is_task_complete",
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
    # Prompts
    "ASSISTANT_IDENTITY",
    "CHAT_SYSTEM_PROMPT",
    "TASK_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "SPECIALIST_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "PromptConfig",
    "get_chat_prompt",
    "get_task_prompt",
    "get_identity_correction",
    "get_default_config",
    "set_default_config",
    # Cache
    "CacheBackend",
    "CacheEntry",
    "CacheMetrics",
    "CacheTier",
    "CachedOllamaClient",
    "CompressionAlgorithm",
    "InMemoryBackend",
    "InvalidationStrategy",
    "RedisBackend",
    "ResponseCache",
    "create_cached_client",
    "create_memory_cache",
    "create_redis_cache",
    # Advanced Cache (Tasks 91-100)
    "AdaptiveCache",
    "CacheCoherence",
    "CacheInvalidator",
    "CacheWarmer",
    "CompressedBackend",
    "CostModel",
    "RedisClusterBackend",
    "SemanticCache",
    "TieredCache",
]
