"""Tool use shim for small models.

Small models (0.5B-3B) often can't reliably use tools in the native way.
This module provides parsing and formatting utilities to enable tool use
with models that don't have native function calling support.

Strategies:
1. Output parsing - Extract tool calls from JSON/structured output
2. Few-shot prompting - Include examples in prompt
3. Retry with feedback - If parse fails, retry with error message
"""

import json
import re
from typing import Optional, Type

from pydantic import BaseModel, Field

from tinyllm.core.message import ToolCall
from tinyllm.tools.base import BaseTool


class ToolUseShim:
    """Enables tool use for models without native function calling."""

    # Patterns to extract tool calls from various output formats
    JSON_PATTERN = re.compile(
        r'\{[^{}]*"tool(?:_id|_call|_name)?"\s*:\s*"([^"]+)"[^{}]*\}',
        re.IGNORECASE | re.DOTALL,
    )

    TAGGED_PATTERN = re.compile(
        r"<tool(?:_call)?>\s*(.*?)\s*</tool(?:_call)?>",
        re.IGNORECASE | re.DOTALL,
    )

    CODE_BLOCK_PATTERN = re.compile(
        r"```(?:json|tool)?\s*(.*?)\s*```",
        re.DOTALL,
    )

    def __init__(self, tools: list[BaseTool]):
        """Initialize with available tools.

        Args:
            tools: List of available tools.
        """
        self.tools = {t.metadata.id: t for t in tools}

    def format_tools_for_prompt(self) -> str:
        """Format tool descriptions for inclusion in prompt.

        Returns:
            Formatted string describing available tools.
        """
        lines = ["Available tools:", ""]

        for tool in self.tools.values():
            lines.append(f"**{tool.metadata.name}** (`{tool.metadata.id}`)")
            lines.append(f"  {tool.metadata.description}")
            lines.append("")
            lines.append("  Input format:")
            lines.append(f"  ```json")
            lines.append(f'  {{"tool": "{tool.metadata.id}", "input": {{...}}}}')
            lines.append("  ```")
            lines.append("")

        lines.append("To use a tool, output ONLY a JSON object in this format:")
        lines.append('{"tool": "<tool_id>", "input": {<parameters>}}')
        lines.append("")

        return "\n".join(lines)

    def format_few_shot_examples(self) -> str:
        """Generate few-shot examples for tool use.

        Returns:
            Formatted examples string.
        """
        examples = []

        if "calculator" in self.tools:
            examples.append(
                """Example - Using calculator:
User: What is 847 * 392?
Assistant: {"tool": "calculator", "input": {"expression": "847 * 392"}}"""
            )

        if "code_executor" in self.tools:
            examples.append(
                """Example - Running code:
User: Run a Python script that prints hello world
Assistant: {"tool": "code_executor", "input": {"code": "print('Hello, World!')", "language": "python"}}"""
            )

        return "\n\n".join(examples)

    def extract_tool_call(self, response: str) -> Optional[ToolCall]:
        """Extract a tool call from model response.

        Tries multiple patterns to handle various output formats.

        Args:
            response: Raw model response text.

        Returns:
            Extracted ToolCall or None if no valid call found.
        """
        # Try direct JSON parsing first
        tool_call = self._try_json_parse(response)
        if tool_call:
            return tool_call

        # Try extracting from code blocks
        tool_call = self._try_code_block_extraction(response)
        if tool_call:
            return tool_call

        # Try tagged format
        tool_call = self._try_tagged_extraction(response)
        if tool_call:
            return tool_call

        # Try regex pattern matching
        tool_call = self._try_pattern_extraction(response)
        if tool_call:
            return tool_call

        return None

    def _try_json_parse(self, text: str) -> Optional[ToolCall]:
        """Try to parse the entire response as JSON."""
        text = text.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            data = json.loads(text)
            return self._validate_tool_call(data)
        except json.JSONDecodeError:
            return None

    def _try_code_block_extraction(self, text: str) -> Optional[ToolCall]:
        """Try to extract tool call from code block."""
        matches = self.CODE_BLOCK_PATTERN.findall(text)
        for match in matches:
            try:
                data = json.loads(match)
                tool_call = self._validate_tool_call(data)
                if tool_call:
                    return tool_call
            except json.JSONDecodeError:
                continue
        return None

    def _try_tagged_extraction(self, text: str) -> Optional[ToolCall]:
        """Try to extract from <tool>...</tool> tags."""
        matches = self.TAGGED_PATTERN.findall(text)
        for match in matches:
            try:
                data = json.loads(match)
                tool_call = self._validate_tool_call(data)
                if tool_call:
                    return tool_call
            except json.JSONDecodeError:
                continue
        return None

    def _try_pattern_extraction(self, text: str) -> Optional[ToolCall]:
        """Try regex pattern matching for JSON-like structures."""
        # Find all JSON-like objects
        json_pattern = re.compile(r"\{[^{}]*\}")
        matches = json_pattern.findall(text)

        for match in matches:
            try:
                data = json.loads(match)
                tool_call = self._validate_tool_call(data)
                if tool_call:
                    return tool_call
            except json.JSONDecodeError:
                continue

        return None

    def _validate_tool_call(self, data: dict) -> Optional[ToolCall]:
        """Validate and convert dict to ToolCall."""
        if not isinstance(data, dict):
            return None

        # Try different key names for tool ID
        tool_id = data.get("tool") or data.get("tool_id") or data.get("tool_name")
        if not tool_id:
            return None

        # Check if tool exists
        if tool_id not in self.tools:
            return None

        # Get input parameters
        tool_input = data.get("input") or data.get("parameters") or data.get("args") or {}

        return ToolCall(tool_id=tool_id, input=tool_input)

    def format_retry_prompt(self, original_prompt: str, error: str) -> str:
        """Format a retry prompt after failed tool call extraction.

        Args:
            original_prompt: The original user prompt.
            error: Description of what went wrong.

        Returns:
            Formatted retry prompt.
        """
        return f"""Your previous response could not be parsed as a tool call.

Error: {error}

Please try again. Output ONLY a valid JSON object in this format:
{{"tool": "<tool_id>", "input": {{<parameters>}}}}

Original request: {original_prompt}"""


class ToolCallParser(BaseModel):
    """Pydantic model for parsing tool calls."""

    tool: str = Field(description="Tool ID to invoke")
    input: dict = Field(default_factory=dict, description="Tool input parameters")

    def to_tool_call(self) -> ToolCall:
        """Convert to ToolCall message type."""
        return ToolCall(tool_id=self.tool, input=self.input)
