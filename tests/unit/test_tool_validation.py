"""Tests for tool schema validation."""

import pytest
from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata
from tinyllm.tools.validation import (
    ToolSchemaValidator,
    ValidatedToolWrapper,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_json_schema,
)


class ValidInput(BaseModel):
    """Valid input for testing."""

    name: str
    value: int = Field(ge=0, le=100)


class ValidOutput(BaseModel):
    """Valid output for testing."""

    success: bool = True
    error: str | None = None
    result: str = ""


class InvalidOutput(BaseModel):
    """Output without standard fields."""

    data: str = ""


class SampleTool(BaseTool[ValidInput, ValidOutput]):
    """Sample tool for validation tests."""

    metadata = ToolMetadata(
        id="sample_tool",
        name="Sample Tool",
        description="A sample tool",
        category="utility",
    )
    input_type = ValidInput
    output_type = ValidOutput

    async def execute(self, input: ValidInput) -> ValidOutput:
        return ValidOutput(result=f"Hello, {input.name}")


class TestValidationIssue:
    """Tests for ValidationIssue."""

    def test_issue_str_basic(self):
        """Test string representation."""
        issue = ValidationIssue(
            field="name",
            message="Field is required",
            severity=ValidationSeverity.ERROR,
        )
        assert "[error] name: Field is required" in str(issue)

    def test_issue_str_with_type(self):
        """Test string with expected type."""
        issue = ValidationIssue(
            field="value",
            message="Invalid type",
            severity=ValidationSeverity.ERROR,
            expected_type="integer",
        )
        assert "expected integer" in str(issue)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_valid(self):
        """Test valid result."""
        result = ValidationResult(is_valid=True, issues=[])
        assert result.is_valid
        assert bool(result)
        assert len(result.errors) == 0

    def test_result_with_errors(self):
        """Test result with errors."""
        issues = [
            ValidationIssue(
                field="name", message="Required", severity=ValidationSeverity.ERROR
            ),
            ValidationIssue(
                field="value", message="Too large", severity=ValidationSeverity.WARNING
            ),
        ]
        result = ValidationResult(is_valid=False, issues=issues)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


class TestToolSchemaValidator:
    """Tests for ToolSchemaValidator."""

    def test_validate_input_valid_dict(self):
        """Test validating valid dict input."""
        validator = ToolSchemaValidator()
        result = validator.validate_input(
            {"name": "test", "value": 50}, ValidInput
        )

        assert result.is_valid
        assert result.validated_data is not None
        assert result.validated_data.name == "test"

    def test_validate_input_valid_model(self):
        """Test validating input that's already a model."""
        validator = ToolSchemaValidator()
        input_model = ValidInput(name="test", value=50)
        result = validator.validate_input(input_model, ValidInput)

        assert result.is_valid
        assert result.validated_data is input_model

    def test_validate_input_missing_field(self):
        """Test validation with missing required field."""
        validator = ToolSchemaValidator()
        result = validator.validate_input({"value": 50}, ValidInput)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert "name" in result.errors[0].field

    def test_validate_input_invalid_type(self):
        """Test validation with wrong type."""
        validator = ToolSchemaValidator()
        result = validator.validate_input(
            {"name": "test", "value": "not_an_int"}, ValidInput
        )

        assert not result.is_valid
        assert len(result.errors) >= 1

    def test_validate_input_constraint_violation(self):
        """Test validation with constraint violation."""
        validator = ToolSchemaValidator()
        result = validator.validate_input(
            {"name": "test", "value": 200}, ValidInput  # value > 100
        )

        assert not result.is_valid
        assert len(result.errors) >= 1

    def test_validate_input_wrong_root_type(self):
        """Test validation with wrong root type."""
        validator = ToolSchemaValidator()
        result = validator.validate_input("not a dict", ValidInput)

        assert not result.is_valid
        assert result.errors[0].field == "$root"

    def test_validate_output(self):
        """Test output validation."""
        validator = ToolSchemaValidator()
        result = validator.validate_output(
            {"success": True, "error": None, "result": "done"}, ValidOutput
        )

        assert result.is_valid

    def test_check_schema_compatibility_valid(self):
        """Test schema compatibility check with valid schemas."""
        validator = ToolSchemaValidator()
        issues = validator.check_schema_compatibility(ValidInput, ValidOutput)

        # Should have no errors
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_check_schema_compatibility_missing_fields(self):
        """Test schema compatibility with missing standard fields."""
        validator = ToolSchemaValidator()
        issues = validator.check_schema_compatibility(ValidInput, InvalidOutput)

        # Should have warnings about missing success/error fields
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 2

    def test_validate_tool_definition_valid(self):
        """Test validating a valid tool definition."""
        validator = ToolSchemaValidator()
        tool = SampleTool()
        result = validator.validate_tool_definition(tool)

        assert result.is_valid

    def test_validate_tool_definition_not_basetool(self):
        """Test validating something that's not a tool."""
        validator = ToolSchemaValidator()
        result = validator.validate_tool_definition("not a tool")

        assert not result.is_valid
        assert any("BaseTool" in str(i) for i in result.errors)


class TestValidatedToolWrapper:
    """Tests for ValidatedToolWrapper."""

    @pytest.mark.asyncio
    async def test_execute_valid_input(self):
        """Test executing with valid input."""
        tool = SampleTool()
        wrapper = ValidatedToolWrapper(tool)

        result = await wrapper.execute({"name": "World", "value": 42})

        assert result.success
        assert result.result == "Hello, World"

    @pytest.mark.asyncio
    async def test_execute_invalid_input(self):
        """Test executing with invalid input."""
        tool = SampleTool()
        wrapper = ValidatedToolWrapper(tool)

        with pytest.raises(ValueError) as exc_info:
            await wrapper.execute({"value": 50})  # Missing name

        assert "validation failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_validation_disabled(self):
        """Test executing with validation disabled."""
        tool = SampleTool()
        wrapper = ValidatedToolWrapper(tool, validate_input=False)

        # This would fail validation but passes because it's disabled
        # (the tool itself may still fail)
        input_data = ValidInput(name="Test", value=10)
        result = await wrapper.execute(input_data)
        assert result.success


class TestValidateJsonSchema:
    """Tests for JSON schema validation."""

    def test_validate_object_valid(self):
        """Test validating valid object."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        }
        result = validate_json_schema({"name": "test", "value": 42}, schema)

        assert result.is_valid

    def test_validate_object_missing_required(self):
        """Test validating object with missing required field."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
            },
        }
        result = validate_json_schema({}, schema)

        assert not result.is_valid
        assert any(i.field == "name" for i in result.issues)

    def test_validate_object_wrong_type(self):
        """Test validating object with wrong field type."""
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
            },
        }
        result = validate_json_schema({"value": "not an int"}, schema)

        assert not result.is_valid
        assert any(i.field == "value" for i in result.issues)

    def test_validate_wrong_root_type(self):
        """Test validating wrong root type."""
        schema = {"type": "object"}
        result = validate_json_schema("not an object", schema)

        assert not result.is_valid
        assert result.issues[0].field == "$root"
