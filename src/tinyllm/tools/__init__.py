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
]
