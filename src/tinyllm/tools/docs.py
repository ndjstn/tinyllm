"""Tool documentation generation for TinyLLM.

This module generates documentation for tools in various formats
including Markdown, JSON, and OpenAPI-compatible specs.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocFormat(str, Enum):
    """Supported documentation formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    OPENAPI = "openapi"
    LLM_PROMPT = "llm_prompt"


@dataclass
class ToolDocumentation:
    """Generated documentation for a tool."""

    tool_id: str
    content: str
    format: DocFormat
    metadata: Dict[str, Any]


class ToolDocGenerator:
    """Generates documentation for tools."""

    def __init__(self, include_examples: bool = True):
        """Initialize generator.

        Args:
            include_examples: Whether to include usage examples.
        """
        self.include_examples = include_examples

    def generate(
        self,
        tool: Any,
        format: DocFormat = DocFormat.MARKDOWN,
    ) -> ToolDocumentation:
        """Generate documentation for a tool.

        Args:
            tool: Tool instance to document.
            format: Output format.

        Returns:
            Generated documentation.
        """
        from tinyllm.tools.base import BaseTool

        if not isinstance(tool, BaseTool):
            raise ValueError(f"Expected BaseTool, got {type(tool).__name__}")

        if format == DocFormat.MARKDOWN:
            content = self._generate_markdown(tool)
        elif format == DocFormat.JSON:
            content = self._generate_json(tool)
        elif format == DocFormat.OPENAPI:
            content = self._generate_openapi(tool)
        elif format == DocFormat.LLM_PROMPT:
            content = self._generate_llm_prompt(tool)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return ToolDocumentation(
            tool_id=tool.metadata.id,
            content=content,
            format=format,
            metadata={
                "name": tool.metadata.name,
                "version": tool.metadata.version,
                "category": tool.metadata.category,
            },
        )

    def _generate_markdown(self, tool: Any) -> str:
        """Generate Markdown documentation."""
        lines = [
            f"# {tool.metadata.name}",
            "",
            f"**ID:** `{tool.metadata.id}`  ",
            f"**Version:** {tool.metadata.version}  ",
            f"**Category:** {tool.metadata.category}  ",
            f"**Sandbox Required:** {'Yes' if tool.metadata.sandbox_required else 'No'}",
            "",
            "## Description",
            "",
            tool.metadata.description,
            "",
            "## Input Schema",
            "",
            "```json",
            json.dumps(tool.input_type.model_json_schema(), indent=2),
            "```",
            "",
            "### Input Fields",
            "",
        ]

        # Document input fields
        input_schema = tool.input_type.model_json_schema()
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for field_name, field_info in properties.items():
            required_str = " *(required)*" if field_name in required else ""
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "No description")
            lines.append(f"- **{field_name}**{required_str}: `{field_type}` - {description}")

        lines.extend([
            "",
            "## Output Schema",
            "",
            "```json",
            json.dumps(tool.output_type.model_json_schema(), indent=2),
            "```",
            "",
            "### Output Fields",
            "",
        ])

        # Document output fields
        output_schema = tool.output_type.model_json_schema()
        properties = output_schema.get("properties", {})

        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "No description")
            lines.append(f"- **{field_name}**: `{field_type}` - {description}")

        if self.include_examples:
            lines.extend([
                "",
                "## Usage Example",
                "",
                "```python",
                f"from tinyllm.tools.registry import ToolRegistry",
                "",
                f'tool = ToolRegistry.get("{tool.metadata.id}")',
                f"result = await tool.execute(input_data)",
                "```",
            ])

        return "\n".join(lines)

    def _generate_json(self, tool: Any) -> str:
        """Generate JSON documentation."""
        doc = {
            "id": tool.metadata.id,
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "version": tool.metadata.version,
            "category": tool.metadata.category,
            "sandbox_required": tool.metadata.sandbox_required,
            "input_schema": tool.input_type.model_json_schema(),
            "output_schema": tool.output_type.model_json_schema(),
            "config": {
                "timeout_ms": tool.config.timeout_ms,
                "enabled": tool.config.enabled,
            },
        }
        return json.dumps(doc, indent=2)

    def _generate_openapi(self, tool: Any) -> str:
        """Generate OpenAPI-compatible specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": tool.metadata.name,
                "version": tool.metadata.version,
                "description": tool.metadata.description,
            },
            "paths": {
                f"/tools/{tool.metadata.id}/execute": {
                    "post": {
                        "summary": f"Execute {tool.metadata.name}",
                        "description": tool.metadata.description,
                        "operationId": f"execute_{tool.metadata.id}",
                        "tags": [tool.metadata.category],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": tool.input_type.model_json_schema(),
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Successful execution",
                                "content": {
                                    "application/json": {
                                        "schema": tool.output_type.model_json_schema(),
                                    }
                                },
                            },
                            "400": {"description": "Invalid input"},
                            "500": {"description": "Tool execution error"},
                        },
                    }
                }
            },
        }
        return json.dumps(spec, indent=2)

    def _generate_llm_prompt(self, tool: Any) -> str:
        """Generate LLM-friendly tool description."""
        lines = [
            f"Tool: {tool.metadata.name}",
            f"ID: {tool.metadata.id}",
            f"Description: {tool.metadata.description}",
            "",
            "Input Parameters:",
        ]

        input_schema = tool.input_type.model_json_schema()
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for field_name, field_info in properties.items():
            required_str = "(required)" if field_name in required else "(optional)"
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "")
            lines.append(f"  - {field_name} {required_str}: {field_type}")
            if description:
                lines.append(f"    {description}")

        lines.extend([
            "",
            "Output:",
        ])

        output_schema = tool.output_type.model_json_schema()
        properties = output_schema.get("properties", {})

        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "any")
            lines.append(f"  - {field_name}: {field_type}")

        return "\n".join(lines)


class ToolDocCatalog:
    """Generates documentation catalogs for multiple tools."""

    def __init__(self, generator: Optional[ToolDocGenerator] = None):
        """Initialize catalog generator.

        Args:
            generator: Optional doc generator instance.
        """
        self.generator = generator or ToolDocGenerator()

    def generate_catalog(
        self,
        tools: List[Any],
        format: DocFormat = DocFormat.MARKDOWN,
        group_by_category: bool = True,
    ) -> str:
        """Generate a catalog of tool documentation.

        Args:
            tools: List of tools to document.
            format: Output format.
            group_by_category: Whether to group by category.

        Returns:
            Combined documentation string.
        """
        if format == DocFormat.MARKDOWN:
            return self._generate_markdown_catalog(tools, group_by_category)
        elif format == DocFormat.JSON:
            return self._generate_json_catalog(tools)
        else:
            # For other formats, just concatenate
            docs = [self.generator.generate(t, format).content for t in tools]
            return "\n\n---\n\n".join(docs)

    def _generate_markdown_catalog(
        self, tools: List[Any], group_by_category: bool
    ) -> str:
        """Generate Markdown catalog."""
        lines = [
            "# Tool Catalog",
            "",
            f"Total tools: {len(tools)}",
            "",
            "## Table of Contents",
            "",
        ]

        # Group tools by category
        if group_by_category:
            categories: Dict[str, List[Any]] = {}
            for tool in tools:
                cat = tool.metadata.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(tool)

            for cat, cat_tools in sorted(categories.items()):
                lines.append(f"### {cat.title()}")
                for tool in cat_tools:
                    lines.append(f"- [{tool.metadata.name}](#{tool.metadata.id})")
                lines.append("")
        else:
            for tool in tools:
                lines.append(f"- [{tool.metadata.name}](#{tool.metadata.id})")
            lines.append("")

        # Generate individual docs
        lines.append("---")
        lines.append("")

        for tool in tools:
            doc = self.generator.generate(tool, DocFormat.MARKDOWN)
            lines.append(doc.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_json_catalog(self, tools: List[Any]) -> str:
        """Generate JSON catalog."""
        catalog = {
            "tools": [],
            "categories": {},
            "count": len(tools),
        }

        for tool in tools:
            doc = json.loads(self.generator.generate(tool, DocFormat.JSON).content)
            catalog["tools"].append(doc)

            cat = tool.metadata.category
            if cat not in catalog["categories"]:
                catalog["categories"][cat] = []
            catalog["categories"][cat].append(tool.metadata.id)

        return json.dumps(catalog, indent=2)


def generate_tool_docs(
    tool: Any,
    format: DocFormat = DocFormat.MARKDOWN,
) -> str:
    """Generate documentation for a single tool.

    Args:
        tool: Tool to document.
        format: Output format.

    Returns:
        Documentation string.
    """
    generator = ToolDocGenerator()
    return generator.generate(tool, format).content


def generate_registry_docs(
    format: DocFormat = DocFormat.MARKDOWN,
) -> str:
    """Generate documentation for all registered tools.

    Args:
        format: Output format.

    Returns:
        Documentation string.
    """
    from tinyllm.tools.registry import ToolRegistry

    tools = [ToolRegistry.get(tid) for tid in ToolRegistry.get_tool_ids()]
    tools = [t for t in tools if t is not None]

    catalog = ToolDocCatalog()
    return catalog.generate_catalog(tools, format)
