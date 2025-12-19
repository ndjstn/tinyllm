"""Tests for tool registry."""

import pytest

from tinyllm.tools.base import ToolConfig
from tinyllm.tools.calculator import CalculatorTool
from tinyllm.tools.registry import ToolRegistry, register_default_tools


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ToolRegistry.clear()

    def test_register_tool(self):
        """Test registering a tool."""
        tool = CalculatorTool()
        ToolRegistry.register(tool)

        assert ToolRegistry.has("calculator")
        assert ToolRegistry.get("calculator") is tool

    def test_get_nonexistent(self):
        """Test getting a non-existent tool."""
        assert ToolRegistry.get("nonexistent") is None
        assert not ToolRegistry.has("nonexistent")

    def test_list_tools(self):
        """Test listing registered tools."""
        tool = CalculatorTool()
        ToolRegistry.register(tool)

        tools = ToolRegistry.list_tools()
        assert len(tools) == 1
        assert tools[0].id == "calculator"

    def test_get_tool_ids(self):
        """Test getting tool IDs."""
        tool = CalculatorTool()
        ToolRegistry.register(tool)

        ids = ToolRegistry.get_tool_ids()
        assert "calculator" in ids

    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        tool = CalculatorTool()
        ToolRegistry.register(tool)

        descriptions = ToolRegistry.get_tool_descriptions()
        assert "Calculator" in descriptions
        assert "calculator" in descriptions

    def test_list_enabled_tools(self):
        """Test listing only enabled tools."""
        # Register enabled tool
        enabled_tool = CalculatorTool()
        ToolRegistry.register(enabled_tool)

        enabled = ToolRegistry.list_enabled_tools()
        assert len(enabled) == 1
        assert enabled[0].id == "calculator"

    def test_get_tools_by_category(self):
        """Test filtering tools by category."""
        tool = CalculatorTool()
        ToolRegistry.register(tool)

        computation_tools = ToolRegistry.get_tools_by_category("computation")
        assert len(computation_tools) == 1

        execution_tools = ToolRegistry.get_tools_by_category("execution")
        assert len(execution_tools) == 0

    def test_register_default_tools(self):
        """Test registering default tools."""
        ToolRegistry.clear()  # Ensure clean state
        register_default_tools()

        assert ToolRegistry.has("calculator")
