"""Provider clients for external LLM APIs."""

from tinyllm.providers.openai_client import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingRequest,
    EmbeddingResponse,
    FunctionCall,
    FunctionDefinition,
    ImageUrl,
    ImageUrlDetail,
    MessageRole,
    OpenAIClient,
    OpenAIConfig,
    ToolCall,
    ToolDefinition,
    close_all_openai_clients,
    get_shared_openai_client,
)
from tinyllm.providers.openai_client import ImageContent as OpenAIImageContent
from tinyllm.providers.openai_client import TextContent as OpenAITextContent
from tinyllm.providers.openai_client import ToolChoice as OpenAIToolChoice
from tinyllm.providers.openai_client import Usage as OpenAIUsage

# Anthropic client imports
from tinyllm.providers.anthropic_client import (
    AnthropicClient,
    CircuitBreaker,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ImageSource,
    Message as AnthropicMessage,
    MessageDeltaEvent,
    MessageRequest,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    RateLimiter,
    StopReason,
    StreamEvent,
    Tool as AnthropicTool,
    ToolResultBlock,
    ToolUseBlock,
    close_all_anthropic_clients,
    get_shared_anthropic_client,
)
from tinyllm.providers.anthropic_client import ImageContent as AnthropicImageContent
from tinyllm.providers.anthropic_client import TextContent as AnthropicTextContent
from tinyllm.providers.anthropic_client import ToolChoice as AnthropicToolChoice
from tinyllm.providers.anthropic_client import Usage as AnthropicUsage

__all__ = [
    # OpenAI
    "OpenAIClient",
    "OpenAIConfig",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "FunctionCall",
    "FunctionDefinition",
    "OpenAIImageContent",
    "ImageUrl",
    "ImageUrlDetail",
    "MessageRole",
    "OpenAITextContent",
    "ToolCall",
    "OpenAIToolChoice",
    "ToolDefinition",
    "OpenAIUsage",
    "get_shared_openai_client",
    "close_all_openai_clients",
    # Anthropic
    "AnthropicClient",
    "AnthropicMessage",
    "AnthropicTool",
    "AnthropicTextContent",
    "AnthropicImageContent",
    "AnthropicToolChoice",
    "AnthropicUsage",
    "CircuitBreaker",
    "ContentBlock",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "ImageSource",
    "MessageDeltaEvent",
    "MessageRequest",
    "MessageResponse",
    "MessageStartEvent",
    "MessageStopEvent",
    "RateLimiter",
    "StopReason",
    "StreamEvent",
    "ToolResultBlock",
    "ToolUseBlock",
    "get_shared_anthropic_client",
    "close_all_anthropic_clients",
]
