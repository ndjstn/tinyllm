"""Base class for all tools."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel, Field

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolConfig(BaseModel):
    """Base configuration for tools."""

    timeout_ms: int = Field(default=5000, ge=100, le=60000)
    enabled: bool = Field(default=True)


class ToolMetadata(BaseModel):
    """Metadata about a tool."""

    id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = Field(pattern=r"^(computation|execution|search|memory|utility)$")
    sandbox_required: bool = False


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all tools."""

    metadata: ToolMetadata
    input_type: Type[InputT]
    output_type: Type[OutputT]

    def __init__(self, config: ToolConfig | None = None):
        """Initialize tool with optional configuration."""
        self.config = config or ToolConfig()

    @abstractmethod
    async def execute(self, input: InputT) -> OutputT:
        """Execute the tool with given input.

        Args:
            input: Validated input matching input_type.

        Returns:
            Output matching output_type.
        """
        pass

    def get_schema_description(self) -> str:
        """Get description for LLM prompt."""
        return f"""Tool: {self.metadata.name}
ID: {self.metadata.id}
Description: {self.metadata.description}

Input Schema:
{self.input_type.model_json_schema()}

Output Schema:
{self.output_type.model_json_schema()}
"""

    async def safe_execute(self, input: InputT) -> OutputT:
        """Execute with timeout and error handling.

        Args:
            input: Tool input.

        Returns:
            Tool output, with error set on failure.
        """
        try:
            result = await asyncio.wait_for(
                self.execute(input), timeout=self.config.timeout_ms / 1000
            )
            return result
        except asyncio.TimeoutError:
            # Create error response - subclasses should handle this properly
            return self.output_type(  # type: ignore
                success=False,
                error=f"Tool timed out after {self.config.timeout_ms}ms",
            )
        except Exception as e:
            return self.output_type(  # type: ignore
                success=False,
                error=str(e),
            )
