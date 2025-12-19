"""Tool registry for TinyLLM.

This module provides a registry for tools that allows dynamic
registration and lookup.
"""

from typing import Dict, List, Optional

from tinyllm.tools.base import BaseTool, ToolMetadata


class ToolRegistry:
    """Registry for available tools.

    The ToolRegistry provides a central location for registering
    and looking up tools. Tools can be registered either as instances
    or via the class decorator.
    """

    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> BaseTool:
        """Register a tool instance.

        Args:
            tool: Tool instance to register.

        Returns:
            The registered tool.
        """
        cls._tools[tool.metadata.id] = tool
        return tool

    @classmethod
    def get(cls, tool_id: str) -> Optional[BaseTool]:
        """Get a tool by ID.

        Args:
            tool_id: Tool identifier.

        Returns:
            Tool instance or None if not found.
        """
        return cls._tools.get(tool_id)

    @classmethod
    def has(cls, tool_id: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_id: Tool identifier.

        Returns:
            True if tool is registered.
        """
        return tool_id in cls._tools

    @classmethod
    def list_tools(cls) -> List[ToolMetadata]:
        """List all available tools.

        Returns:
            List of tool metadata.
        """
        return [t.metadata for t in cls._tools.values()]

    @classmethod
    def list_enabled_tools(cls) -> List[ToolMetadata]:
        """List only enabled tools.

        Returns:
            List of enabled tool metadata.
        """
        return [t.metadata for t in cls._tools.values() if t.config.enabled]

    @classmethod
    def get_tool_ids(cls) -> List[str]:
        """Get all registered tool IDs.

        Returns:
            List of tool IDs.
        """
        return list(cls._tools.keys())

    @classmethod
    def get_tool_descriptions(cls) -> str:
        """Get descriptions for all enabled tools.

        This is useful for including in prompts to tell
        the LLM what tools are available.

        Returns:
            Formatted string of tool descriptions.
        """
        descriptions = []
        for tool in cls._tools.values():
            if tool.config.enabled:
                descriptions.append(tool.get_schema_description())
        return "\n\n".join(descriptions)

    @classmethod
    def get_tools_by_category(cls, category: str) -> List[BaseTool]:
        """Get all tools in a category.

        Args:
            category: Category to filter by.

        Returns:
            List of tools in that category.
        """
        return [t for t in cls._tools.values() if t.metadata.category == category]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (for testing)."""
        cls._tools.clear()


def register_default_tools() -> None:
    """Register the default built-in tools.

    Call this at startup to register all built-in tools.
    """
    from tinyllm.tools.calculator import CalculatorTool
    from tinyllm.tools.code_executor import CodeExecutorTool

    ToolRegistry.register(CalculatorTool())
    ToolRegistry.register(CodeExecutorTool())
