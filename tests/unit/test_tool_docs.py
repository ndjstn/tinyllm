"""Tests for tool documentation generation."""

import json

import pytest
from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata
from tinyllm.tools.docs import (
    DocFormat,
    ToolDocCatalog,
    ToolDocGenerator,
    ToolDocumentation,
    generate_registry_docs,
    generate_tool_docs,
)


class DocInput(BaseModel):
    """Input for doc test tool."""

    query: str = Field(description="The search query")
    limit: int = Field(default=10, description="Max results", ge=1, le=100)


class DocOutput(BaseModel):
    """Output for doc test tool."""

    success: bool = True
    error: str | None = None
    results: list[str] = Field(default_factory=list, description="Search results")


class DocTool(BaseTool[DocInput, DocOutput]):
    """Tool for documentation tests."""

    metadata = ToolMetadata(
        id="doc_tool",
        name="Documentation Tool",
        description="A tool for testing documentation generation",
        version="1.2.3",
        category="search",
    )
    input_type = DocInput
    output_type = DocOutput

    async def execute(self, input: DocInput) -> DocOutput:
        return DocOutput(results=["result1", "result2"])


class AnotherDocTool(BaseTool[DocInput, DocOutput]):
    """Another tool for catalog tests."""

    metadata = ToolMetadata(
        id="another_doc_tool",
        name="Another Doc Tool",
        description="Another tool for docs",
        version="2.0.0",
        category="utility",
    )
    input_type = DocInput
    output_type = DocOutput

    async def execute(self, input: DocInput) -> DocOutput:
        return DocOutput()


class TestToolDocumentation:
    """Tests for ToolDocumentation dataclass."""

    def test_documentation_fields(self):
        """Test documentation has all fields."""
        doc = ToolDocumentation(
            tool_id="test",
            content="# Test",
            format=DocFormat.MARKDOWN,
            metadata={"name": "Test"},
        )

        assert doc.tool_id == "test"
        assert doc.content == "# Test"
        assert doc.format == DocFormat.MARKDOWN
        assert doc.metadata["name"] == "Test"


class TestToolDocGenerator:
    """Tests for ToolDocGenerator."""

    def test_generate_markdown(self):
        """Test generating Markdown docs."""
        generator = ToolDocGenerator()
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.MARKDOWN)

        assert doc.format == DocFormat.MARKDOWN
        assert "# Documentation Tool" in doc.content
        assert "doc_tool" in doc.content
        assert "1.2.3" in doc.content
        assert "search" in doc.content
        assert "Input Schema" in doc.content
        assert "Output Schema" in doc.content

    def test_generate_markdown_includes_fields(self):
        """Test Markdown includes field descriptions."""
        generator = ToolDocGenerator()
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.MARKDOWN)

        assert "query" in doc.content
        assert "limit" in doc.content
        assert "results" in doc.content

    def test_generate_markdown_includes_example(self):
        """Test Markdown includes usage example."""
        generator = ToolDocGenerator(include_examples=True)
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.MARKDOWN)

        assert "Usage Example" in doc.content
        assert "ToolRegistry.get" in doc.content

    def test_generate_markdown_no_example(self):
        """Test Markdown without examples."""
        generator = ToolDocGenerator(include_examples=False)
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.MARKDOWN)

        assert "Usage Example" not in doc.content

    def test_generate_json(self):
        """Test generating JSON docs."""
        generator = ToolDocGenerator()
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.JSON)

        assert doc.format == DocFormat.JSON

        data = json.loads(doc.content)
        assert data["id"] == "doc_tool"
        assert data["name"] == "Documentation Tool"
        assert data["version"] == "1.2.3"
        assert "input_schema" in data
        assert "output_schema" in data

    def test_generate_openapi(self):
        """Test generating OpenAPI spec."""
        generator = ToolDocGenerator()
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.OPENAPI)

        assert doc.format == DocFormat.OPENAPI

        spec = json.loads(doc.content)
        assert spec["openapi"] == "3.0.0"
        assert spec["info"]["title"] == "Documentation Tool"
        assert "/tools/doc_tool/execute" in spec["paths"]

    def test_generate_openapi_has_request_response(self):
        """Test OpenAPI has request and response schemas."""
        generator = ToolDocGenerator()
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.OPENAPI)
        spec = json.loads(doc.content)

        endpoint = spec["paths"]["/tools/doc_tool/execute"]["post"]
        assert "requestBody" in endpoint
        assert "responses" in endpoint
        assert "200" in endpoint["responses"]

    def test_generate_llm_prompt(self):
        """Test generating LLM prompt format."""
        generator = ToolDocGenerator()
        tool = DocTool()

        doc = generator.generate(tool, DocFormat.LLM_PROMPT)

        assert doc.format == DocFormat.LLM_PROMPT
        assert "Tool: Documentation Tool" in doc.content
        assert "Input Parameters:" in doc.content
        assert "query" in doc.content
        assert "(required)" in doc.content

    def test_generate_invalid_format(self):
        """Test generating with invalid format."""
        generator = ToolDocGenerator()
        tool = DocTool()

        # DocFormat enum won't allow invalid values
        # but we can test non-tool input
        with pytest.raises(ValueError):
            generator.generate("not a tool", DocFormat.MARKDOWN)


class TestToolDocCatalog:
    """Tests for ToolDocCatalog."""

    def test_generate_markdown_catalog(self):
        """Test generating Markdown catalog."""
        catalog = ToolDocCatalog()
        tools = [DocTool(), AnotherDocTool()]

        result = catalog.generate_catalog(tools, DocFormat.MARKDOWN)

        assert "# Tool Catalog" in result
        assert "Total tools: 2" in result
        assert "Documentation Tool" in result
        assert "Another Doc Tool" in result

    def test_generate_catalog_groups_by_category(self):
        """Test catalog groups by category."""
        catalog = ToolDocCatalog()
        tools = [DocTool(), AnotherDocTool()]

        result = catalog.generate_catalog(tools, DocFormat.MARKDOWN, group_by_category=True)

        assert "### Search" in result
        assert "### Utility" in result

    def test_generate_json_catalog(self):
        """Test generating JSON catalog."""
        catalog = ToolDocCatalog()
        tools = [DocTool(), AnotherDocTool()]

        result = catalog.generate_catalog(tools, DocFormat.JSON)

        data = json.loads(result)
        assert data["count"] == 2
        assert len(data["tools"]) == 2
        assert "search" in data["categories"]
        assert "utility" in data["categories"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_tool_docs(self):
        """Test generate_tool_docs function."""
        tool = DocTool()

        result = generate_tool_docs(tool, DocFormat.MARKDOWN)

        assert "Documentation Tool" in result

    def test_generate_tool_docs_json(self):
        """Test generate_tool_docs with JSON format."""
        tool = DocTool()

        result = generate_tool_docs(tool, DocFormat.JSON)

        data = json.loads(result)
        assert data["id"] == "doc_tool"


class TestGenerateRegistryDocs:
    """Tests for generate_registry_docs."""

    def test_generate_registry_docs(self):
        """Test generating docs for registered tools."""
        from tinyllm.tools.registry import ToolRegistry, register_default_tools

        ToolRegistry.clear()
        register_default_tools()

        result = generate_registry_docs(DocFormat.MARKDOWN)

        assert "# Tool Catalog" in result
        assert "calculator" in result.lower() or "Calculator" in result
