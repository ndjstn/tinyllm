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

# Mistral client imports
from tinyllm.providers.mistral_client import (
    MistralClient,
    MistralConfig,
    ChatMessage as MistralChatMessage,
    MessageRole as MistralMessageRole,
    ChatCompletionRequest as MistralChatCompletionRequest,
    ChatCompletionResponse as MistralChatCompletionResponse,
    EmbeddingRequest as MistralEmbeddingRequest,
    EmbeddingResponse as MistralEmbeddingResponse,
    FunctionCall as MistralFunctionCall,
    FunctionDefinition as MistralFunctionDefinition,
    ToolCall as MistralToolCall,
    ToolDefinition as MistralToolDefinition,
    ToolChoice as MistralToolChoice,
    SafeMode,
    ResponseFormat,
    get_shared_mistral_client,
    close_all_mistral_clients,
)
from tinyllm.providers.mistral_client import ImageContent as MistralImageContent
from tinyllm.providers.mistral_client import TextContent as MistralTextContent
from tinyllm.providers.mistral_client import Usage as MistralUsage
from tinyllm.providers.mistral_client import ImageUrl as MistralImageUrl
from tinyllm.providers.mistral_client import ImageUrlDetail as MistralImageUrlDetail
from tinyllm.providers.mistral_client import RateLimiter as MistralRateLimiter
from tinyllm.providers.mistral_client import CircuitBreaker as MistralCircuitBreaker

# Cohere client imports
from tinyllm.providers.cohere_client import (
    CohereClient,
    ChatMessage as CohereChatMessage,
    ChatRole as CohereChatRole,
    ChatRequest as CohereChatRequest,
    ChatResponse as CohereChatResponse,
    ChatStreamEvent as CohereChatStreamEvent,
    EmbedRequest as CohereEmbedRequest,
    EmbedResponse as CohereEmbedResponse,
    Embedding as CohereEmbedding,
    RerankRequest as CohereRerankRequest,
    RerankResponse as CohereRerankResponse,
    RerankResult as CohereRerankResult,
    RerankDocument as CohereRerankDocument,
    Tool as CohereTool,
    ToolCall as CohereToolCall,
    ToolResult as CohereToolResult,
    ToolParameterDefinition as CohereToolParameterDefinition,
    TokenCount as CohereTokenCount,
    get_shared_cohere_client,
    close_all_cohere_clients,
)
from tinyllm.providers.cohere_client import RateLimiter as CohereRateLimiter
from tinyllm.providers.cohere_client import CircuitBreaker as CohereCircuitBreaker

# Gemini client imports
from tinyllm.providers.gemini_client import (
    BatchEmbedContentsResponse,
    Blob,
    Candidate,
    Content as GeminiContent,
    ContentEmbedding,
    EmbedContentResponse,
    FileData,
    FileDataPart,
    FinishReason,
    FunctionCall as GeminiFunctionCall,
    FunctionCallPart,
    FunctionDeclaration as GeminiFunctionDeclaration,
    FunctionResponse,
    FunctionResponsePart,
    GenerateContentResponse,
    GenerationConfig,
    GeminiClient,
    HarmBlockThreshold,
    HarmCategory,
    InlineData,
    SafetyRating,
    SafetySetting,
    Tool as GeminiTool,
    ToolConfig,
    UsageMetadata,
    close_all_gemini_clients,
    get_shared_gemini_client,
)
from tinyllm.providers.gemini_client import TextPart as GeminiTextPart
from tinyllm.providers.gemini_client import RateLimiter as GeminiRateLimiter
from tinyllm.providers.gemini_client import CircuitBreaker as GeminiCircuitBreaker

# Groq client imports
from tinyllm.providers.groq_client import (
    GroqClient,
    GroqConfig,
    close_all_groq_clients,
    get_shared_groq_client,
)
from tinyllm.providers.groq_client import ChatMessage as GroqChatMessage
from tinyllm.providers.groq_client import ChatCompletionRequest as GroqChatCompletionRequest
from tinyllm.providers.groq_client import ChatCompletionResponse as GroqChatCompletionResponse
from tinyllm.providers.groq_client import ImageContent as GroqImageContent
from tinyllm.providers.groq_client import MessageRole as GroqMessageRole
from tinyllm.providers.groq_client import TextContent as GroqTextContent
from tinyllm.providers.groq_client import ToolDefinition as GroqToolDefinition
from tinyllm.providers.groq_client import ToolCall as GroqToolCall
from tinyllm.providers.groq_client import FunctionDefinition as GroqFunctionDefinition
from tinyllm.providers.groq_client import Usage as GroqUsage
from tinyllm.providers.groq_client import RateLimiter as GroqRateLimiter
from tinyllm.providers.groq_client import CircuitBreaker as GroqCircuitBreaker
from tinyllm.providers.groq_client import ImageUrl as GroqImageUrl
from tinyllm.providers.groq_client import ImageUrlDetail as GroqImageUrlDetail

# Llama.cpp client imports
from tinyllm.providers.llamacpp_client import (
    LlamaCppClient,
    LlamaCppConfig,
    CompletionRequest,
    CompletionResponse,
    get_shared_llamacpp_client,
    close_all_llamacpp_clients,
)
from tinyllm.providers.llamacpp_client import RateLimiter as LlamaCppRateLimiter
from tinyllm.providers.llamacpp_client import CircuitBreaker as LlamaCppCircuitBreaker

# vLLM client imports
from tinyllm.providers.vllm_client import (
    VLLMClient,
    VLLMConfig,
    ChatMessage as VLLMChatMessage,
    ChatCompletionRequest as VLLMChatCompletionRequest,
    ChatCompletionResponse as VLLMChatCompletionResponse,
    ChatCompletionChoice as VLLMChatCompletionChoice,
    ChatCompletionUsage as VLLMChatCompletionUsage,
    get_shared_vllm_client,
    close_all_vllm_clients,
)

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
    # Mistral
    "MistralClient",
    "MistralConfig",
    "MistralChatMessage",
    "MistralMessageRole",
    "MistralChatCompletionRequest",
    "MistralChatCompletionResponse",
    "MistralEmbeddingRequest",
    "MistralEmbeddingResponse",
    "MistralFunctionCall",
    "MistralFunctionDefinition",
    "MistralToolCall",
    "MistralToolDefinition",
    "MistralToolChoice",
    "MistralImageContent",
    "MistralTextContent",
    "MistralUsage",
    "MistralImageUrl",
    "MistralImageUrlDetail",
    "MistralRateLimiter",
    "MistralCircuitBreaker",
    "SafeMode",
    "ResponseFormat",
    "get_shared_mistral_client",
    "close_all_mistral_clients",
    # Cohere
    "CohereClient",
    "CohereChatMessage",
    "CohereChatRole",
    "CohereChatRequest",
    "CohereChatResponse",
    "CohereChatStreamEvent",
    "CohereEmbedRequest",
    "CohereEmbedResponse",
    "CohereEmbedding",
    "CohereRerankRequest",
    "CohereRerankResponse",
    "CohereRerankResult",
    "CohereRerankDocument",
    "CohereTool",
    "CohereToolCall",
    "CohereToolResult",
    "CohereToolParameterDefinition",
    "CohereTokenCount",
    "CohereRateLimiter",
    "CohereCircuitBreaker",
    "get_shared_cohere_client",
    "close_all_cohere_clients",
    # Gemini
    "GeminiClient",
    "GeminiContent",
    "GeminiTextPart",
    "GeminiFunctionCall",
    "GeminiFunctionDeclaration",
    "GeminiTool",
    "GenerateContentResponse",
    "GenerationConfig",
    "BatchEmbedContentsResponse",
    "Blob",
    "Candidate",
    "ContentEmbedding",
    "EmbedContentResponse",
    "FileData",
    "FileDataPart",
    "FinishReason",
    "FunctionCallPart",
    "FunctionResponse",
    "FunctionResponsePart",
    "HarmBlockThreshold",
    "HarmCategory",
    "InlineData",
    "SafetyRating",
    "SafetySetting",
    "ToolConfig",
    "UsageMetadata",
    "GeminiRateLimiter",
    "GeminiCircuitBreaker",
    "get_shared_gemini_client",
    "close_all_gemini_clients",
    # Groq
    "GroqClient",
    "GroqConfig",
    "GroqChatMessage",
    "GroqMessageRole",
    "GroqChatCompletionRequest",
    "GroqChatCompletionResponse",
    "GroqImageContent",
    "GroqTextContent",
    "GroqToolDefinition",
    "GroqToolCall",
    "GroqFunctionDefinition",
    "GroqUsage",
    "GroqRateLimiter",
    "GroqCircuitBreaker",
    "GroqImageUrl",
    "GroqImageUrlDetail",
    "get_shared_groq_client",
    "close_all_groq_clients",
    # Llama.cpp
    "LlamaCppClient",
    "LlamaCppConfig",
    "CompletionRequest",
    "CompletionResponse",
    "LlamaCppRateLimiter",
    "LlamaCppCircuitBreaker",
    "get_shared_llamacpp_client",
    "close_all_llamacpp_clients",
    # vLLM
    "VLLMClient",
    "VLLMConfig",
    "VLLMChatMessage",
    "VLLMChatCompletionRequest",
    "VLLMChatCompletionResponse",
    "VLLMChatCompletionChoice",
    "VLLMChatCompletionUsage",
    "get_shared_vllm_client",
    "close_all_vllm_clients",
]
