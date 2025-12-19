"""Tests for tool registry and discovery."""

import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata
from tinyllm.tools.registry import (
    ToolDiscoveryError,
    ToolRegistry,
    register_default_tools,
)


class DummyInput(BaseModel):
    """Dummy input for testing."""

    value: str = ""


class DummyOutput(BaseModel):
    """Dummy output for testing."""

    success: bool = True
    error: str | None = None


class DummyTool(BaseTool[DummyInput, DummyOutput]):
    """Test tool for registry tests."""

    metadata = ToolMetadata(
        id="dummy_tool",
        name="Dummy Tool",
        description="A test tool",
        version="1.0.0",
        category="utility",
    )
    input_type = DummyInput
    output_type = DummyOutput

    async def execute(self, input: DummyInput) -> DummyOutput:
        return DummyOutput()


class AnotherDummyTool(BaseTool[DummyInput, DummyOutput]):
    """Another test tool."""

    metadata = ToolMetadata(
        id="another_dummy",
        name="Another Dummy",
        description="Another test tool for search tests",
        version="2.0.0",
        category="computation",
    )
    input_type = DummyInput
    output_type = DummyOutput

    async def execute(self, input: DummyInput) -> DummyOutput:
        return DummyOutput()


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_tool(self):
        """Test registering a tool."""
        tool = DummyTool()
        result = ToolRegistry.register(tool)

        assert result is tool
        assert ToolRegistry.has("dummy_tool")
        assert ToolRegistry.get("dummy_tool") is tool

    def test_register_class_decorator(self):
        """Test registering a tool class."""
        ToolRegistry.register_class(DummyTool)

        assert ToolRegistry.has("dummy_tool")

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        ToolRegistry.register(DummyTool())

        assert ToolRegistry.unregister("dummy_tool")
        assert not ToolRegistry.has("dummy_tool")

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent tool."""
        assert not ToolRegistry.unregister("nonexistent")


class TestToolLookup:
    """Tests for tool lookup and listing."""

    def test_get_tool(self):
        """Test getting a tool by ID."""
        tool = DummyTool()
        ToolRegistry.register(tool)

        assert ToolRegistry.get("dummy_tool") is tool
        assert ToolRegistry.get("nonexistent") is None

    def test_has_tool(self):
        """Test checking tool existence."""
        ToolRegistry.register(DummyTool())

        assert ToolRegistry.has("dummy_tool")
        assert not ToolRegistry.has("nonexistent")

    def test_list_tools(self):
        """Test listing all tools."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        tools = ToolRegistry.list_tools()
        assert len(tools) == 2
        ids = {t.id for t in tools}
        assert ids == {"dummy_tool", "another_dummy"}

    def test_list_enabled_tools(self):
        """Test listing enabled tools only."""
        tool1 = DummyTool(config=ToolConfig(enabled=True))
        tool2 = AnotherDummyTool(config=ToolConfig(enabled=False))

        ToolRegistry.register(tool1)
        ToolRegistry.register(tool2)

        enabled = ToolRegistry.list_enabled_tools()
        assert len(enabled) == 1
        assert enabled[0].id == "dummy_tool"

    def test_get_tool_ids(self):
        """Test getting all tool IDs."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        ids = ToolRegistry.get_tool_ids()
        assert set(ids) == {"dummy_tool", "another_dummy"}


class TestToolFiltering:
    """Tests for tool filtering and search."""

    def test_get_tools_by_category(self):
        """Test filtering by category."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        utility_tools = ToolRegistry.get_tools_by_category("utility")
        assert len(utility_tools) == 1
        assert utility_tools[0].metadata.id == "dummy_tool"

    def test_get_tools_by_version(self):
        """Test filtering by version prefix."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        v1_tools = ToolRegistry.get_tools_by_version("1.")
        assert len(v1_tools) == 1
        assert v1_tools[0].metadata.id == "dummy_tool"

    def test_search_tools_by_name(self):
        """Test searching tools by name."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        results = ToolRegistry.search_tools("dummy")
        assert len(results) == 2

    def test_search_tools_by_description(self):
        """Test searching tools by description."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        results = ToolRegistry.search_tools("search")
        assert len(results) == 1
        assert results[0].metadata.id == "another_dummy"

    def test_search_tools_with_category_filter(self):
        """Test searching with category filter."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        results = ToolRegistry.search_tools("dummy", categories=["computation"])
        assert len(results) == 1
        assert results[0].metadata.id == "another_dummy"

    def test_get_categories(self):
        """Test getting all categories."""
        ToolRegistry.register(DummyTool())
        ToolRegistry.register(AnotherDummyTool())

        categories = ToolRegistry.get_categories()
        assert categories == {"utility", "computation"}


class TestToolDescriptions:
    """Tests for tool description generation."""

    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        ToolRegistry.register(DummyTool())

        descriptions = ToolRegistry.get_tool_descriptions()
        assert "Dummy Tool" in descriptions
        assert "dummy_tool" in descriptions


class TestToolDiscovery:
    """Tests for tool discovery mechanisms."""

    def test_discover_from_module(self):
        """Test discovering tools from a module."""
        # Discovery from calculator module (known to have a tool)
        tools = ToolRegistry.discover_from_module("tinyllm.tools.calculator")

        assert len(tools) >= 1
        assert any(t.metadata.id == "calculator" for t in tools)

    def test_discover_from_module_error(self):
        """Test error on invalid module."""
        with pytest.raises(ToolDiscoveryError):
            ToolRegistry.discover_from_module("nonexistent.module")

    def test_discover_from_directory(self):
        """Test discovering tools from a directory."""
        # Create a temporary directory with a tool file
        with tempfile.TemporaryDirectory() as tmpdir:
            tool_file = Path(tmpdir) / "my_tool.py"
            tool_file.write_text(
                '''
from pydantic import BaseModel
from tinyllm.tools.base import BaseTool, ToolMetadata

class TempInput(BaseModel):
    value: str = ""

class TempOutput(BaseModel):
    success: bool = True
    error: str | None = None

class TempTool(BaseTool):
    metadata = ToolMetadata(
        id="temp_tool",
        name="Temp Tool",
        description="Temp",
        category="utility",
    )
    input_type = TempInput
    output_type = TempOutput

    async def execute(self, input):
        return TempOutput()
'''
            )

            tools = ToolRegistry.discover_from_directory(tmpdir)
            assert len(tools) == 1
            assert tools[0].metadata.id == "temp_tool"

    def test_discover_from_directory_error(self):
        """Test error on invalid directory."""
        with pytest.raises(ToolDiscoveryError):
            ToolRegistry.discover_from_directory("/nonexistent/path")

    def test_discover_from_entry_points(self):
        """Test entry point discovery (may find nothing in test env)."""
        # This should not raise, even with no entry points
        tools = ToolRegistry.discover_from_entry_points()
        assert isinstance(tools, list)


class TestToolReload:
    """Tests for tool reloading."""

    def test_get_tool_class(self):
        """Test getting tool class."""
        ToolRegistry.register(DummyTool())

        tool_class = ToolRegistry.get_tool_class("dummy_tool")
        assert tool_class is DummyTool

    def test_reload_tool(self):
        """Test reloading a tool."""
        original = DummyTool()
        ToolRegistry.register(original)

        reloaded = ToolRegistry.reload_tool("dummy_tool")
        assert reloaded is not None
        assert reloaded is not original
        assert reloaded.metadata.id == "dummy_tool"


class TestDiscoveryCallbacks:
    """Tests for discovery callbacks."""

    def test_add_discovery_callback(self):
        """Test adding a discovery callback."""

        def my_callback():
            return [DummyTool()]

        ToolRegistry.add_discovery_callback(my_callback)

        results = ToolRegistry.run_discovery()
        assert "callback_0" in results
        assert len(results["callback_0"]) == 1


class TestDefaultTools:
    """Tests for default tool registration."""

    def test_register_default_tools(self):
        """Test registering default tools."""
        register_default_tools()

        assert ToolRegistry.has("calculator")
        assert ToolRegistry.has("code_executor")
        assert ToolRegistry.has("web_search")
