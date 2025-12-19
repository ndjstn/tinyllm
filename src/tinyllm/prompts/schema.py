"""Pydantic models for prompt definitions.

This module defines the schema for prompt YAML files used by TinyLLM.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptCategory(str, Enum):
    """Categories of prompts."""

    ROUTING = "routing"
    SPECIALIST = "specialist"
    THINKING = "thinking"
    TOOL = "tool"
    GRADING = "grading"
    META = "meta"
    MEMORY = "memory"


class OutputFormat(str, Enum):
    """Output format types."""

    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"
    STRUCTURED = "structured"


class PromptExample(BaseModel):
    """Few-shot example for prompts."""

    model_config = {"extra": "forbid"}

    input: str = Field(description="Example input")
    output: str = Field(description="Expected output")


class PromptDefinition(BaseModel):
    """Complete prompt definition.

    Prompts are stored as YAML files and loaded at runtime, allowing
    iteration without code changes.
    """

    model_config = {"extra": "forbid"}

    # Identity
    id: str = Field(
        description="Unique identifier", pattern=r"^[a-z][a-z0-9_\.]*$"
    )
    name: str = Field(description="Human-readable name")
    version: str = Field(
        description="Semantic version", pattern=r"^\d+\.\d+\.\d+$"
    )
    category: PromptCategory = Field(description="Prompt category")
    description: Optional[str] = Field(default=None, description="Description")

    # Compatibility
    compatible_models: List[str] = Field(
        min_length=1, description="List of compatible model names"
    )

    # Parameters
    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=1000, ge=1, le=32000, description="Maximum output tokens"
    )
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling")

    # Output
    output_format: OutputFormat = Field(
        default=OutputFormat.TEXT, description="Expected output format"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schema for output validation"
    )

    # Content
    system_prompt: str = Field(min_length=1, description="System prompt template")
    user_template: str = Field(min_length=1, description="User message template")

    # Examples
    examples: List[PromptExample] = Field(
        default_factory=list, description="Few-shot examples"
    )

    def render(self, **variables: Any) -> tuple[str, str]:
        """Render system prompt and user message with variables.

        Uses Jinja2 templating for variable substitution.

        Args:
            **variables: Variables to substitute in templates.

        Returns:
            Tuple of (system_prompt, user_message).
        """
        try:
            from jinja2 import Template
        except ImportError:
            # Fall back to simple string formatting if Jinja2 not available
            system = self.system_prompt
            user = self.user_template
            for key, value in variables.items():
                system = system.replace("{{" + key + "}}", str(value))
                system = system.replace("{{ " + key + " }}", str(value))
                user = user.replace("{{" + key + "}}", str(value))
                user = user.replace("{{ " + key + " }}", str(value))
            return system, user

        system = Template(self.system_prompt).render(**variables)
        user = Template(self.user_template).render(**variables)

        # Add few-shot examples if present
        if self.examples:
            examples_text = "\n\nExamples:\n"
            for ex in self.examples:
                examples_text += f"Input: {ex.input}\nOutput: {ex.output}\n\n"
            system = system + examples_text

        return system, user

    def get_generation_params(self) -> Dict[str, Any]:
        """Get parameters for model generation.

        Returns:
            Dictionary of generation parameters.
        """
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if self.output_format == OutputFormat.JSON:
            params["json_mode"] = True
        elif self.output_format == OutputFormat.JSON_SCHEMA and self.output_schema:
            params["json_mode"] = True
            params["output_schema"] = self.output_schema

        return params
