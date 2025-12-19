"""Tool result validation for TinyLLM.

This module provides validation for tool execution results,
ensuring outputs conform to expected schemas and constraints.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field: Optional[str] = None
    value: Any = None
    rule: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}]"]
        if self.field:
            parts.append(f"Field '{self.field}':")
        parts.append(self.message)
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_output: Any = None

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def error_messages(self) -> List[str]:
        """Get error messages as strings."""
        return [str(e) for e in self.errors]


class ResultValidator(ABC):
    """Abstract base class for result validators."""

    @abstractmethod
    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a result.

        Args:
            result: The result to validate.
            context: Optional validation context.

        Returns:
            ValidationResult with issues if any.
        """
        pass


class SchemaValidator(ResultValidator):
    """Validates results against a Pydantic schema."""

    def __init__(self, schema: Type[BaseModel], strict: bool = True):
        """Initialize schema validator.

        Args:
            schema: Pydantic model for validation.
            strict: If True, extra fields cause errors.
        """
        self.schema = schema
        self.strict = strict

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate result against schema."""
        issues = []

        try:
            if isinstance(result, self.schema):
                # Already validated model
                validated = result
            elif isinstance(result, dict):
                validated = self.schema.model_validate(result)
            else:
                # Try to coerce
                validated = self.schema.model_validate(result)

            return ValidationResult(valid=True, validated_output=validated)

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                issues.append(
                    ValidationIssue(
                        message=error["msg"],
                        severity=ValidationSeverity.ERROR,
                        field=field_path,
                        value=error.get("input"),
                        rule="schema",
                    )
                )

            return ValidationResult(valid=False, issues=issues)

        except Exception as e:
            issues.append(
                ValidationIssue(
                    message=f"Schema validation error: {e}",
                    severity=ValidationSeverity.ERROR,
                    rule="schema",
                )
            )
            return ValidationResult(valid=False, issues=issues)


class TypeValidator(ResultValidator):
    """Validates result type."""

    def __init__(self, expected_type: Union[Type, tuple]):
        """Initialize type validator.

        Args:
            expected_type: Expected type or tuple of types.
        """
        self.expected_type = expected_type

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate result type."""
        if isinstance(result, self.expected_type):
            return ValidationResult(valid=True, validated_output=result)

        expected_name = (
            self.expected_type.__name__
            if hasattr(self.expected_type, "__name__")
            else str(self.expected_type)
        )

        return ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(
                    message=f"Expected type {expected_name}, got {type(result).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=result,
                    rule="type",
                )
            ],
        )


class RuleValidator(ResultValidator):
    """Validates results against custom rules."""

    def __init__(self, rules: List[Callable[[Any], Optional[str]]] = None):
        """Initialize rule validator.

        Args:
            rules: List of validation functions that return error message or None.
        """
        self.rules: List[Callable[[Any], Optional[str]]] = rules or []

    def add_rule(self, rule: Callable[[Any], Optional[str]]) -> "RuleValidator":
        """Add a validation rule.

        Args:
            rule: Function that returns error message or None.

        Returns:
            Self for chaining.
        """
        self.rules.append(rule)
        return self

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate against all rules."""
        issues = []

        for i, rule in enumerate(self.rules):
            try:
                error = rule(result)
                if error:
                    issues.append(
                        ValidationIssue(
                            message=error,
                            severity=ValidationSeverity.ERROR,
                            rule=f"rule_{i}",
                        )
                    )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        message=f"Rule execution error: {e}",
                        severity=ValidationSeverity.ERROR,
                        rule=f"rule_{i}",
                    )
                )

        return ValidationResult(
            valid=len(issues) == 0, issues=issues, validated_output=result if not issues else None
        )


class RangeValidator(ResultValidator):
    """Validates numeric values are within range."""

    def __init__(
        self,
        field: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
    ):
        """Initialize range validator.

        Args:
            field: Field to validate (None for direct value).
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            inclusive: Whether bounds are inclusive.
        """
        self.field = field
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate value is in range."""
        # Get the value to check
        if self.field:
            if isinstance(result, dict):
                value = result.get(self.field)
            elif hasattr(result, self.field):
                value = getattr(result, self.field)
            else:
                return ValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            message=f"Field '{self.field}' not found",
                            severity=ValidationSeverity.ERROR,
                            field=self.field,
                            rule="range",
                        )
                    ],
                )
        else:
            value = result

        if value is None:
            return ValidationResult(valid=True, validated_output=result)

        issues = []

        if self.min_value is not None:
            if self.inclusive:
                if value < self.min_value:
                    issues.append(
                        ValidationIssue(
                            message=f"Value {value} is below minimum {self.min_value}",
                            severity=ValidationSeverity.ERROR,
                            field=self.field,
                            value=value,
                            rule="range",
                        )
                    )
            else:
                if value <= self.min_value:
                    issues.append(
                        ValidationIssue(
                            message=f"Value {value} must be greater than {self.min_value}",
                            severity=ValidationSeverity.ERROR,
                            field=self.field,
                            value=value,
                            rule="range",
                        )
                    )

        if self.max_value is not None:
            if self.inclusive:
                if value > self.max_value:
                    issues.append(
                        ValidationIssue(
                            message=f"Value {value} exceeds maximum {self.max_value}",
                            severity=ValidationSeverity.ERROR,
                            field=self.field,
                            value=value,
                            rule="range",
                        )
                    )
            else:
                if value >= self.max_value:
                    issues.append(
                        ValidationIssue(
                            message=f"Value {value} must be less than {self.max_value}",
                            severity=ValidationSeverity.ERROR,
                            field=self.field,
                            value=value,
                            rule="range",
                        )
                    )

        return ValidationResult(
            valid=len(issues) == 0, issues=issues, validated_output=result if not issues else None
        )


class NotEmptyValidator(ResultValidator):
    """Validates that string/collection is not empty."""

    def __init__(self, field: Optional[str] = None, allow_whitespace: bool = False):
        """Initialize not-empty validator.

        Args:
            field: Field to validate (None for direct value).
            allow_whitespace: Whether whitespace-only strings are valid.
        """
        self.field = field
        self.allow_whitespace = allow_whitespace

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate value is not empty."""
        if self.field:
            if isinstance(result, dict):
                value = result.get(self.field)
            elif hasattr(result, self.field):
                value = getattr(result, self.field)
            else:
                return ValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            message=f"Field '{self.field}' not found",
                            severity=ValidationSeverity.ERROR,
                            field=self.field,
                            rule="not_empty",
                        )
                    ],
                )
        else:
            value = result

        if value is None:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        message="Value is None",
                        severity=ValidationSeverity.ERROR,
                        field=self.field,
                        rule="not_empty",
                    )
                ],
            )

        # Check for empty
        is_empty = False
        if isinstance(value, str):
            is_empty = len(value.strip() if not self.allow_whitespace else value) == 0
        elif hasattr(value, "__len__"):
            is_empty = len(value) == 0

        if is_empty:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        message="Value is empty",
                        severity=ValidationSeverity.ERROR,
                        field=self.field,
                        value=value,
                        rule="not_empty",
                    )
                ],
            )

        return ValidationResult(valid=True, validated_output=result)


class CompositeValidator(ResultValidator):
    """Composes multiple validators."""

    def __init__(
        self,
        validators: List[ResultValidator] = None,
        fail_fast: bool = False,
    ):
        """Initialize composite validator.

        Args:
            validators: List of validators to run.
            fail_fast: Stop on first error.
        """
        self.validators: List[ResultValidator] = validators or []
        self.fail_fast = fail_fast

    def add(self, validator: ResultValidator) -> "CompositeValidator":
        """Add a validator.

        Args:
            validator: Validator to add.

        Returns:
            Self for chaining.
        """
        self.validators.append(validator)
        return self

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Run all validators."""
        all_issues = []
        validated = result

        for validator in self.validators:
            vr = validator.validate(validated, context)
            all_issues.extend(vr.issues)

            if vr.valid and vr.validated_output is not None:
                validated = vr.validated_output

            if not vr.valid and self.fail_fast:
                break

        has_errors = any(i.severity == ValidationSeverity.ERROR for i in all_issues)

        return ValidationResult(
            valid=not has_errors,
            issues=all_issues,
            validated_output=validated if not has_errors else None,
        )


class ConditionalValidator(ResultValidator):
    """Applies validation conditionally."""

    def __init__(
        self,
        condition: Callable[[Any], bool],
        validator: ResultValidator,
        else_validator: Optional[ResultValidator] = None,
    ):
        """Initialize conditional validator.

        Args:
            condition: Function that returns True to apply validator.
            validator: Validator to apply if condition is True.
            else_validator: Optional validator for else case.
        """
        self.condition = condition
        self.validator = validator
        self.else_validator = else_validator

    def validate(self, result: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate conditionally."""
        try:
            if self.condition(result):
                return self.validator.validate(result, context)
            elif self.else_validator:
                return self.else_validator.validate(result, context)
            else:
                return ValidationResult(valid=True, validated_output=result)
        except Exception as e:
            return ValidationResult(
                valid=False,
                issues=[
                    ValidationIssue(
                        message=f"Condition evaluation error: {e}",
                        severity=ValidationSeverity.ERROR,
                        rule="conditional",
                    )
                ],
            )


class ToolResultValidator:
    """Main validator for tool results."""

    def __init__(
        self,
        schema: Optional[Type[BaseModel]] = None,
        validators: Optional[List[ResultValidator]] = None,
        on_validation_error: Optional[Callable[[ValidationResult], None]] = None,
    ):
        """Initialize tool result validator.

        Args:
            schema: Optional Pydantic schema for validation.
            validators: Additional validators to apply.
            on_validation_error: Callback on validation error.
        """
        self.validators: List[ResultValidator] = []

        if schema:
            self.validators.append(SchemaValidator(schema))

        if validators:
            self.validators.extend(validators)

        self.on_validation_error = on_validation_error

    def add_validator(self, validator: ResultValidator) -> "ToolResultValidator":
        """Add a validator.

        Args:
            validator: Validator to add.

        Returns:
            Self for chaining.
        """
        self.validators.append(validator)
        return self

    def validate(
        self,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """Validate a tool result.

        Args:
            result: Result to validate.
            context: Validation context.
            raise_on_error: Whether to raise on validation error.

        Returns:
            ValidationResult.

        Raises:
            ValueError: If raise_on_error and validation fails.
        """
        all_issues = []
        validated = result

        for validator in self.validators:
            vr = validator.validate(validated, context)
            all_issues.extend(vr.issues)

            if vr.valid and vr.validated_output is not None:
                validated = vr.validated_output

        has_errors = any(i.severity == ValidationSeverity.ERROR for i in all_issues)

        validation_result = ValidationResult(
            valid=not has_errors,
            issues=all_issues,
            validated_output=validated if not has_errors else None,
        )

        if not validation_result.valid:
            if self.on_validation_error:
                self.on_validation_error(validation_result)

            if raise_on_error:
                raise ValueError(
                    f"Validation failed: {'; '.join(validation_result.error_messages)}"
                )

        return validation_result


class ValidatedToolWrapper:
    """Wrapper that validates tool results."""

    def __init__(
        self,
        tool: Any,
        validator: ToolResultValidator,
        transform_on_error: Optional[Callable[[Any, ValidationResult], Any]] = None,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            validator: Result validator.
            transform_on_error: Optional transform on validation error.
        """
        self.tool = tool
        self.validator = validator
        self.transform_on_error = transform_on_error

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input: Any) -> Any:
        """Execute tool and validate result.

        Args:
            input: Tool input.

        Returns:
            Validated result.

        Raises:
            ValueError: If validation fails.
        """
        result = await self.tool.execute(input)

        validation = self.validator.validate(result)

        if validation.valid:
            return validation.validated_output or result

        logger.warning(
            f"Tool {self.tool.metadata.id} result validation failed: "
            f"{validation.error_messages}"
        )

        if self.transform_on_error:
            return self.transform_on_error(result, validation)

        raise ValueError(
            f"Result validation failed for {self.tool.metadata.id}: "
            f"{'; '.join(validation.error_messages)}"
        )


# Convenience functions


def with_validation(
    tool: Any,
    schema: Optional[Type[BaseModel]] = None,
    validators: Optional[List[ResultValidator]] = None,
) -> ValidatedToolWrapper:
    """Add validation to a tool.

    Args:
        tool: Tool to wrap.
        schema: Optional Pydantic schema.
        validators: Additional validators.

    Returns:
        ValidatedToolWrapper.
    """
    return ValidatedToolWrapper(
        tool=tool,
        validator=ToolResultValidator(schema=schema, validators=validators),
    )


def validate_result(
    result: Any,
    schema: Optional[Type[BaseModel]] = None,
    validators: Optional[List[ResultValidator]] = None,
) -> ValidationResult:
    """Validate a result directly.

    Args:
        result: Result to validate.
        schema: Optional Pydantic schema.
        validators: Additional validators.

    Returns:
        ValidationResult.
    """
    validator = ToolResultValidator(schema=schema, validators=validators)
    return validator.validate(result)
