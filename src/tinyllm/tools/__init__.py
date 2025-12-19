"""Tool implementations for TinyLLM."""

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata
from tinyllm.tools.calculator import CalculatorInput, CalculatorOutput, CalculatorTool
from tinyllm.tools.code_executor import (
    CodeExecutorConfig,
    CodeExecutorInput,
    CodeExecutorOutput,
    CodeExecutorTool,
)
from tinyllm.tools.registry import ToolRegistry, register_default_tools
from tinyllm.tools.web_search import (
    BraveSearchProvider,
    ContentFetcher,
    DuckDuckGoProvider,
    PageContent,
    RateLimiter,
    ResultDeduplicator,
    SearchCache,
    SearchProvider,
    SearchResult,
    SearXNGProvider,
    WebSearchConfig,
    WebSearchInput,
    WebSearchOutput,
    WebSearchTool,
)

__all__ = [
    # Base
    "BaseTool",
    "ToolConfig",
    "ToolMetadata",
    # Registry
    "ToolRegistry",
    "register_default_tools",
    # Calculator
    "CalculatorTool",
    "CalculatorInput",
    "CalculatorOutput",
    # Code Executor
    "CodeExecutorTool",
    "CodeExecutorInput",
    "CodeExecutorOutput",
    "CodeExecutorConfig",
    # Web Search
    "WebSearchTool",
    "WebSearchInput",
    "WebSearchOutput",
    "WebSearchConfig",
    "SearchResult",
    "PageContent",
    "SearchProvider",
    "SearXNGProvider",
    "DuckDuckGoProvider",
    "BraveSearchProvider",
    "SearchCache",
    "RateLimiter",
    "ResultDeduplicator",
    "ContentFetcher",
]
