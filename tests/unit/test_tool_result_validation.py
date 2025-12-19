"""Tests for tool result validation."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.result_validation import (
    CompositeValidator,
    ConditionalValidator,
    NotEmptyValidator,
    RangeValidator,
    ResultValidator,
    RuleValidator,
    SchemaValidator,
    ToolResultValidator,
    TypeValidator,
    ValidatedToolWrapper,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_result,
    with_validation,
)


class SampleResultSchema(BaseModel):
    """Schema for result tests."""

    value: int
    message: str
    success: bool = True


class PartialResultSchema(BaseModel):
    """Partial schema for tests."""

    value: int


class ResultInput(BaseModel):
    """Input for result validation tests."""

    value: int = 0


class ResultOutput(BaseModel):
    """Output for result validation tests."""

    value: int = 0
    message: str = ""
    success: bool = True


class ValidResultTool(BaseTool[ResultInput, ResultOutput]):
    """Tool that returns valid results."""

    metadata = ToolMetadata(
        id="valid_tool",
        name="Valid Tool",
        description="Returns valid results",
        category="utility",
    )
    input_type = ResultInput
    output_type = ResultOutput

    async def execute(self, input: ResultInput) -> ResultOutput:
        return ResultOutput(value=input.value * 2, message="success")


class InvalidResultTool(BaseTool[ResultInput, ResultOutput]):
    """Tool that returns invalid results."""

    metadata = ToolMetadata(
        id="invalid_tool",
        name="Invalid Tool",
        description="Returns invalid results",
        category="utility",
    )
    input_type = ResultInput
    output_type = ResultOutput

    async def execute(self, input: ResultInput) -> ResultOutput:
        return ResultOutput(value=-1, message="")  # Invalid: negative and empty


class TestValidationIssue:
    """Tests for ValidationIssue."""

    def test_str_representation(self):
        """Test string representation."""
        issue = ValidationIssue(
            message="Value too large",
            severity=ValidationSeverity.ERROR,
            field="amount",
        )

        assert "[ERROR]" in str(issue)
        assert "amount" in str(issue)
        assert "Value too large" in str(issue)

    def test_warning_severity(self):
        """Test warning severity in string."""
        issue = ValidationIssue(
            message="Value near limit",
            severity=ValidationSeverity.WARNING,
        )

        assert "[WARNING]" in str(issue)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_errors_property(self):
        """Test errors property."""
        result = ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(message="Error 1", severity=ValidationSeverity.ERROR),
                ValidationIssue(message="Warning 1", severity=ValidationSeverity.WARNING),
                ValidationIssue(message="Error 2", severity=ValidationSeverity.ERROR),
            ],
        )

        assert len(result.errors) == 2
        assert len(result.warnings) == 1

    def test_error_messages(self):
        """Test error messages property."""
        result = ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(message="Error 1", severity=ValidationSeverity.ERROR),
            ],
        )

        messages = result.error_messages
        assert len(messages) == 1
        assert "Error 1" in messages[0]


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_valid_dict(self):
        """Test validating dict against schema."""
        validator = SchemaValidator(SampleResultSchema)

        result = validator.validate({"value": 10, "message": "test"})

        assert result.valid
        assert isinstance(result.validated_output, SampleResultSchema)

    def test_valid_model_instance(self):
        """Test validating existing model instance."""
        validator = SchemaValidator(SampleResultSchema)
        model = SampleResultSchema(value=10, message="test")

        result = validator.validate(model)

        assert result.valid

    def test_missing_required_field(self):
        """Test missing required field."""
        validator = SchemaValidator(SampleResultSchema)

        result = validator.validate({"value": 10})  # Missing 'message'

        assert not result.valid
        assert len(result.errors) > 0

    def test_invalid_type(self):
        """Test invalid field type."""
        validator = SchemaValidator(SampleResultSchema)

        result = validator.validate({"value": "not an int", "message": "test"})

        assert not result.valid


class TestTypeValidator:
    """Tests for TypeValidator."""

    def test_correct_type(self):
        """Test correct type validation."""
        validator = TypeValidator(dict)

        result = validator.validate({"key": "value"})

        assert result.valid

    def test_wrong_type(self):
        """Test wrong type validation."""
        validator = TypeValidator(str)

        result = validator.validate(123)

        assert not result.valid
        assert "Expected type str" in result.error_messages[0]

    def test_tuple_of_types(self):
        """Test validation against multiple types."""
        validator = TypeValidator((str, int))

        assert validator.validate("hello").valid
        assert validator.validate(123).valid
        assert not validator.validate([]).valid


class TestRuleValidator:
    """Tests for RuleValidator."""

    def test_passing_rule(self):
        """Test rule that passes."""
        validator = RuleValidator()
        validator.add_rule(lambda x: None if x > 0 else "Must be positive")

        result = validator.validate(10)

        assert result.valid

    def test_failing_rule(self):
        """Test rule that fails."""
        validator = RuleValidator()
        validator.add_rule(lambda x: None if x > 0 else "Must be positive")

        result = validator.validate(-5)

        assert not result.valid
        assert "Must be positive" in result.error_messages[0]

    def test_multiple_rules(self):
        """Test multiple rules."""
        validator = RuleValidator(
            rules=[
                lambda x: None if x > 0 else "Must be positive",
                lambda x: None if x < 100 else "Must be under 100",
            ]
        )

        assert validator.validate(50).valid
        assert not validator.validate(-5).valid
        assert not validator.validate(150).valid

    def test_chaining(self):
        """Test rule chaining."""
        validator = (
            RuleValidator()
            .add_rule(lambda x: None if x > 0 else "Must be positive")
            .add_rule(lambda x: None if x < 100 else "Must be under 100")
        )

        assert validator.validate(50).valid


class TestRangeValidator:
    """Tests for RangeValidator."""

    def test_in_range(self):
        """Test value in range."""
        validator = RangeValidator(min_value=0, max_value=100)

        assert validator.validate(50).valid

    def test_below_minimum(self):
        """Test value below minimum."""
        validator = RangeValidator(min_value=0, max_value=100)

        result = validator.validate(-5)

        assert not result.valid
        assert "below minimum" in result.error_messages[0]

    def test_above_maximum(self):
        """Test value above maximum."""
        validator = RangeValidator(min_value=0, max_value=100)

        result = validator.validate(150)

        assert not result.valid
        assert "exceeds maximum" in result.error_messages[0]

    def test_exclusive_bounds(self):
        """Test exclusive bounds."""
        validator = RangeValidator(min_value=0, max_value=100, inclusive=False)

        assert validator.validate(50).valid
        assert not validator.validate(0).valid
        assert not validator.validate(100).valid

    def test_field_validation(self):
        """Test validating a field."""
        validator = RangeValidator(field="value", min_value=0)

        result = validator.validate({"value": 10})
        assert result.valid

        result = validator.validate({"value": -5})
        assert not result.valid

    def test_none_value_allowed(self):
        """Test None value is allowed."""
        validator = RangeValidator(min_value=0)

        assert validator.validate(None).valid


class TestNotEmptyValidator:
    """Tests for NotEmptyValidator."""

    def test_not_empty_string(self):
        """Test non-empty string."""
        validator = NotEmptyValidator()

        assert validator.validate("hello").valid

    def test_empty_string(self):
        """Test empty string."""
        validator = NotEmptyValidator()

        result = validator.validate("")

        assert not result.valid
        assert "empty" in result.error_messages[0].lower()

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        validator = NotEmptyValidator()

        assert not validator.validate("   ").valid

    def test_whitespace_allowed(self):
        """Test whitespace allowed."""
        validator = NotEmptyValidator(allow_whitespace=True)

        assert validator.validate("   ").valid

    def test_none_value(self):
        """Test None value."""
        validator = NotEmptyValidator()

        result = validator.validate(None)

        assert not result.valid
        assert "None" in result.error_messages[0]

    def test_field_validation(self):
        """Test validating a field."""
        validator = NotEmptyValidator(field="message")

        assert validator.validate({"message": "hello"}).valid
        assert not validator.validate({"message": ""}).valid

    def test_empty_list(self):
        """Test empty list."""
        validator = NotEmptyValidator()

        assert not validator.validate([]).valid
        assert validator.validate([1, 2, 3]).valid


class TestCompositeValidator:
    """Tests for CompositeValidator."""

    def test_all_pass(self):
        """Test all validators pass."""
        validator = CompositeValidator(
            validators=[
                TypeValidator(dict),
                RuleValidator([lambda x: None if "key" in x else "Missing key"]),
            ]
        )

        result = validator.validate({"key": "value"})

        assert result.valid

    def test_one_fails(self):
        """Test one validator fails."""
        validator = CompositeValidator(
            validators=[
                TypeValidator(dict),
                RuleValidator([lambda x: None if "missing" in x else "Missing key"]),
            ]
        )

        result = validator.validate({"key": "value"})

        assert not result.valid

    def test_fail_fast(self):
        """Test fail-fast mode."""
        validator = CompositeValidator(
            validators=[
                RuleValidator([lambda x: "Error 1"]),
                RuleValidator([lambda x: "Error 2"]),
            ],
            fail_fast=True,
        )

        result = validator.validate("test")

        # Should only have one error due to fail-fast
        assert len(result.errors) == 1

    def test_chaining(self):
        """Test add method chaining."""
        validator = (
            CompositeValidator()
            .add(TypeValidator(int))
            .add(RangeValidator(min_value=0))
        )

        assert validator.validate(10).valid


class TestConditionalValidator:
    """Tests for ConditionalValidator."""

    def test_condition_true(self):
        """Test validation when condition is true."""
        validator = ConditionalValidator(
            condition=lambda x: isinstance(x, int),
            validator=RangeValidator(min_value=0),
        )

        assert validator.validate(10).valid
        assert not validator.validate(-5).valid

    def test_condition_false_no_else(self):
        """Test validation when condition is false without else."""
        validator = ConditionalValidator(
            condition=lambda x: isinstance(x, int),
            validator=RangeValidator(min_value=0),
        )

        # Condition false, no else validator, should pass
        result = validator.validate("not an int")

        assert result.valid

    def test_condition_false_with_else(self):
        """Test validation when condition is false with else."""
        validator = ConditionalValidator(
            condition=lambda x: isinstance(x, int),
            validator=RangeValidator(min_value=0),
            else_validator=NotEmptyValidator(),
        )

        assert validator.validate("hello").valid
        assert not validator.validate("").valid


class TestToolResultValidator:
    """Tests for ToolResultValidator."""

    def test_schema_validation(self):
        """Test schema validation."""
        validator = ToolResultValidator(schema=SampleResultSchema)

        result = validator.validate({"value": 10, "message": "test"})

        assert result.valid
        assert isinstance(result.validated_output, SampleResultSchema)

    def test_additional_validators(self):
        """Test additional validators."""
        validator = ToolResultValidator(
            schema=SampleResultSchema,
            validators=[
                RangeValidator(field="value", min_value=0),
            ],
        )

        assert validator.validate({"value": 10, "message": "test"}).valid
        assert not validator.validate({"value": -5, "message": "test"}).valid

    def test_on_validation_error_callback(self):
        """Test on_validation_error callback."""
        errors = []

        def on_error(result):
            errors.append(result)

        validator = ToolResultValidator(
            schema=SampleResultSchema,
            on_validation_error=on_error,
        )

        validator.validate({"value": "invalid"})

        assert len(errors) == 1

    def test_raise_on_error(self):
        """Test raise_on_error option."""
        validator = ToolResultValidator(schema=SampleResultSchema)

        with pytest.raises(ValueError) as exc_info:
            validator.validate({"value": "invalid"}, raise_on_error=True)

        assert "Validation failed" in str(exc_info.value)

    def test_add_validator(self):
        """Test add_validator method."""
        validator = ToolResultValidator()
        validator.add_validator(TypeValidator(dict))
        validator.add_validator(RuleValidator([lambda x: None if "key" in x else "Missing key"]))

        assert validator.validate({"key": "value"}).valid


class TestValidatedToolWrapper:
    """Tests for ValidatedToolWrapper."""

    @pytest.mark.asyncio
    async def test_valid_result(self):
        """Test valid result passes through."""
        tool = ValidResultTool()
        validator = ToolResultValidator(
            validators=[RangeValidator(field="value", min_value=0)]
        )
        wrapper = ValidatedToolWrapper(tool, validator)

        result = await wrapper.execute(ResultInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_invalid_result_raises(self):
        """Test invalid result raises error."""
        tool = InvalidResultTool()
        validator = ToolResultValidator(
            validators=[
                RangeValidator(field="value", min_value=0),
                NotEmptyValidator(field="message"),
            ]
        )
        wrapper = ValidatedToolWrapper(tool, validator)

        with pytest.raises(ValueError):
            await wrapper.execute(ResultInput(value=5))

    @pytest.mark.asyncio
    async def test_transform_on_error(self):
        """Test transform_on_error callback."""

        def transform(result, validation):
            return ResultOutput(value=0, message="fallback", success=False)

        tool = InvalidResultTool()
        validator = ToolResultValidator(
            validators=[RangeValidator(field="value", min_value=0)]
        )
        wrapper = ValidatedToolWrapper(tool, validator, transform_on_error=transform)

        result = await wrapper.execute(ResultInput(value=5))

        assert result.value == 0
        assert result.message == "fallback"

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        tool = ValidResultTool()
        validator = ToolResultValidator()
        wrapper = ValidatedToolWrapper(tool, validator)

        assert wrapper.metadata.id == "valid_tool"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_validation(self):
        """Test with_validation function."""
        tool = ValidResultTool()
        wrapper = with_validation(
            tool,
            validators=[RangeValidator(field="value", min_value=0)],
        )

        result = await wrapper.execute(ResultInput(value=5))

        assert result.value == 10

    def test_validate_result(self):
        """Test validate_result function."""
        result = validate_result(
            {"value": 10, "message": "test"},
            schema=SampleResultSchema,
        )

        assert result.valid
        assert isinstance(result.validated_output, SampleResultSchema)

    def test_validate_result_invalid(self):
        """Test validate_result with invalid data."""
        result = validate_result(
            {"value": -5, "message": "test"},
            schema=SampleResultSchema,
            validators=[RangeValidator(field="value", min_value=0)],
        )

        assert not result.valid
