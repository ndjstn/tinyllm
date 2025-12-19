"""Tool schema validation for TinyLLM.

This module provides comprehensive validation for tool inputs and outputs,
including schema validation, type checking, and constraint enforcement.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found during schema validation."""

    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected_type: Optional[str] = None

    def __str__(self) -> str:
        if self.expected_type:
            return f"[{self.severity.value}] {self.field}: {self.message} (expected {self.expected_type})"
        return f"[{self.severity.value}] {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    issues: List[ValidationIssue]
    validated_data: Optional[BaseModel] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def __bool__(self) -> bool:
        return self.is_valid


class ToolSchemaValidator:
    """Validates tool inputs and outputs against their schemas."""

    def __init__(self, strict: bool = True):
        """Initialize validator.

        Args:
            strict: If True, raise errors on validation failure.
        """
        self.strict = strict

    def validate_input(
        self,
        data: Any,
        schema_type: Type[BaseModel],
    ) -> ValidationResult:
        """Validate input data against a schema.

        Args:
            data: Input data to validate (dict or BaseModel).
            schema_type: The Pydantic model class to validate against.

        Returns:
            ValidationResult with issues and validated data.
        """
        issues: List[ValidationIssue] = []

        # Handle different input types
        if isinstance(data, schema_type):
            return ValidationResult(is_valid=True, issues=[], validated_data=data)

        if isinstance(data, BaseModel):
            data = data.model_dump()

        if not isinstance(data, dict):
            issues.append(
                ValidationIssue(
                    field="$root",
                    message=f"Expected dict or {schema_type.__name__}, got {type(data).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=data,
                    expected_type="dict",
                )
            )
            return ValidationResult(is_valid=False, issues=issues)

        # Validate against schema
        try:
            validated = schema_type.model_validate(data)
            return ValidationResult(is_valid=True, issues=[], validated_data=validated)
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                issues.append(
                    ValidationIssue(
                        field=field,
                        message=error["msg"],
                        severity=ValidationSeverity.ERROR,
                        value=error.get("input"),
                        expected_type=error.get("type"),
                    )
                )
            return ValidationResult(is_valid=False, issues=issues)

    def validate_output(
        self,
        data: Any,
        schema_type: Type[BaseModel],
    ) -> ValidationResult:
        """Validate output data against a schema.

        Args:
            data: Output data to validate.
            schema_type: The Pydantic model class to validate against.

        Returns:
            ValidationResult with issues and validated data.
        """
        return self.validate_input(data, schema_type)

    def check_schema_compatibility(
        self,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
    ) -> List[ValidationIssue]:
        """Check if input/output schemas are well-formed.

        Args:
            input_schema: The input schema type.
            output_schema: The output schema type.

        Returns:
            List of validation issues found.
        """
        issues: List[ValidationIssue] = []

        # Check input schema
        try:
            input_schema.model_json_schema()
        except Exception as e:
            issues.append(
                ValidationIssue(
                    field="input_schema",
                    message=f"Invalid input schema: {e}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check output schema
        try:
            output_schema.model_json_schema()
        except Exception as e:
            issues.append(
                ValidationIssue(
                    field="output_schema",
                    message=f"Invalid output schema: {e}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check that output has standard fields
        output_fields = output_schema.model_fields
        if "success" not in output_fields:
            issues.append(
                ValidationIssue(
                    field="output_schema.success",
                    message="Output schema should have 'success' field",
                    severity=ValidationSeverity.WARNING,
                )
            )
        if "error" not in output_fields:
            issues.append(
                ValidationIssue(
                    field="output_schema.error",
                    message="Output schema should have 'error' field",
                    severity=ValidationSeverity.WARNING,
                )
            )

        return issues

    def validate_tool_definition(
        self,
        tool: Any,
    ) -> ValidationResult:
        """Validate a complete tool definition.

        Args:
            tool: A BaseTool instance.

        Returns:
            ValidationResult indicating if the tool is valid.
        """
        from tinyllm.tools.base import BaseTool, ToolMetadata

        issues: List[ValidationIssue] = []

        # Check it's a BaseTool
        if not isinstance(tool, BaseTool):
            issues.append(
                ValidationIssue(
                    field="$type",
                    message=f"Expected BaseTool, got {type(tool).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return ValidationResult(is_valid=False, issues=issues)

        # Check metadata
        if not hasattr(tool, "metadata"):
            issues.append(
                ValidationIssue(
                    field="metadata",
                    message="Tool missing metadata",
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif not isinstance(tool.metadata, ToolMetadata):
            issues.append(
                ValidationIssue(
                    field="metadata",
                    message="metadata must be ToolMetadata instance",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check input/output types
        if not hasattr(tool, "input_type"):
            issues.append(
                ValidationIssue(
                    field="input_type",
                    message="Tool missing input_type",
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif not issubclass(tool.input_type, BaseModel):
            issues.append(
                ValidationIssue(
                    field="input_type",
                    message="input_type must be a Pydantic BaseModel",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if not hasattr(tool, "output_type"):
            issues.append(
                ValidationIssue(
                    field="output_type",
                    message="Tool missing output_type",
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif not issubclass(tool.output_type, BaseModel):
            issues.append(
                ValidationIssue(
                    field="output_type",
                    message="output_type must be a Pydantic BaseModel",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check schema compatibility if types exist
        if hasattr(tool, "input_type") and hasattr(tool, "output_type"):
            if issubclass(tool.input_type, BaseModel) and issubclass(
                tool.output_type, BaseModel
            ):
                issues.extend(
                    self.check_schema_compatibility(tool.input_type, tool.output_type)
                )

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == ValidationSeverity.ERROR])
            == 0,
            issues=issues,
        )


class ValidatedToolWrapper:
    """Wrapper that adds validation to tool execution."""

    def __init__(
        self,
        tool: Any,
        validator: Optional[ToolSchemaValidator] = None,
        validate_input: bool = True,
        validate_output: bool = True,
    ):
        """Initialize wrapper.

        Args:
            tool: The tool to wrap.
            validator: Optional validator instance.
            validate_input: Whether to validate inputs.
            validate_output: Whether to validate outputs.
        """
        self.tool = tool
        self.validator = validator or ToolSchemaValidator()
        self.validate_input_enabled = validate_input
        self.validate_output_enabled = validate_output

    async def execute(self, input: Any) -> Any:
        """Execute with validation.

        Args:
            input: Tool input.

        Returns:
            Tool output.

        Raises:
            ValueError: If validation fails.
        """
        # Validate input
        if self.validate_input_enabled:
            result = self.validator.validate_input(input, self.tool.input_type)
            if not result.is_valid:
                error_msgs = "; ".join(str(i) for i in result.errors)
                raise ValueError(f"Input validation failed: {error_msgs}")
            input = result.validated_data

        # Execute tool
        output = await self.tool.execute(input)

        # Validate output
        if self.validate_output_enabled:
            result = self.validator.validate_output(output, self.tool.output_type)
            if not result.is_valid:
                logger.warning(
                    f"Output validation failed for {self.tool.metadata.id}: "
                    f"{'; '.join(str(i) for i in result.errors)}"
                )

        return output


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Validate data against a JSON schema.

    Args:
        data: Data to validate.
        schema: JSON schema to validate against.

    Returns:
        ValidationResult with issues.
    """
    issues: List[ValidationIssue] = []

    # Basic type checking
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(data, dict):
            issues.append(
                ValidationIssue(
                    field="$root",
                    message=f"Expected object, got {type(data).__name__}",
                    severity=ValidationSeverity.ERROR,
                    expected_type="object",
                )
            )
            return ValidationResult(is_valid=False, issues=issues)

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                issues.append(
                    ValidationIssue(
                        field=field,
                        message="Required field missing",
                        severity=ValidationSeverity.ERROR,
                    )
                )

        # Check property types
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                field_type = field_schema.get("type")
                if field_type and not _check_type(value, field_type):
                    issues.append(
                        ValidationIssue(
                            field=field,
                            message=f"Invalid type: expected {field_type}",
                            severity=ValidationSeverity.ERROR,
                            value=value,
                            expected_type=field_type,
                        )
                    )

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected JSON schema type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    expected = type_map.get(expected_type)
    if expected is None:
        return True  # Unknown type, allow
    return isinstance(value, expected)
