"""Tests for tool error handling."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.error_handling import (
    DefaultValueRecovery,
    ErrorBoundary,
    ErrorCategory,
    ErrorClassifier,
    ErrorContext,
    ErrorHandler,
    ErrorHandlingToolWrapper,
    ErrorSeverity,
    FallbackRecovery,
    RetryRecovery,
    ToolError,
    TransformRecovery,
    classify_error,
    create_error_handler,
    with_error_handling,
)


class ErrorInput(BaseModel):
    """Input for error handling tests."""

    value: int = 0
    fail: bool = False


class ErrorOutput(BaseModel):
    """Output for error handling tests."""

    value: int = 0
    success: bool = True
    error: str | None = None


class SuccessTool(BaseTool[ErrorInput, ErrorOutput]):
    """Tool that succeeds."""

    metadata = ToolMetadata(
        id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        category="utility",
    )
    input_type = ErrorInput
    output_type = ErrorOutput

    async def execute(self, input: ErrorInput) -> ErrorOutput:
        return ErrorOutput(value=input.value * 2)


class FailTool(BaseTool[ErrorInput, ErrorOutput]):
    """Tool that always fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = ErrorInput
    output_type = ErrorOutput

    async def execute(self, input: ErrorInput) -> ErrorOutput:
        raise ValueError("Intentional failure")


class ConditionalFailTool(BaseTool[ErrorInput, ErrorOutput]):
    """Tool that fails conditionally."""

    metadata = ToolMetadata(
        id="conditional_fail_tool",
        name="Conditional Fail Tool",
        description="Fails when input.fail is True",
        category="utility",
    )
    input_type = ErrorInput
    output_type = ErrorOutput

    def __init__(self, fail_count: int = 0):
        super().__init__()
        self.fail_count = fail_count
        self.attempts = 0

    async def execute(self, input: ErrorInput) -> ErrorOutput:
        self.attempts += 1
        if input.fail or self.attempts <= self.fail_count:
            raise ValueError(f"Failure #{self.attempts}")
        return ErrorOutput(value=input.value * 2)


class TestErrorContext:
    """Tests for ErrorContext."""

    def test_creation(self):
        """Test context creation."""
        context = ErrorContext(tool_id="test_tool")

        assert context.tool_id == "test_tool"
        assert context.attempt == 1
        assert context.timestamp is not None

    def test_add_metadata(self):
        """Test adding metadata."""
        context = ErrorContext(tool_id="test_tool")
        context.add_metadata("key", "value")

        assert context.metadata["key"] == "value"

    def test_chaining(self):
        """Test method chaining."""
        context = (
            ErrorContext(tool_id="test_tool")
            .add_metadata("key1", "value1")
            .add_metadata("key2", "value2")
        )

        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == "value2"


class TestToolError:
    """Tests for ToolError."""

    def test_creation(self):
        """Test error creation."""
        error = ToolError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
        )

        assert error.message == "Test error"
        assert error.category == ErrorCategory.VALIDATION

    def test_exception_type(self):
        """Test exception type property."""
        exc = ValueError("Test")
        error = ToolError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            original_exception=exc,
        )

        assert error.exception_type == "ValueError"

    def test_to_dict(self):
        """Test conversion to dict."""
        context = ErrorContext(tool_id="test_tool")
        error = ToolError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=context,
        )

        d = error.to_dict()

        assert d["message"] == "Test error"
        assert d["category"] == "validation"
        assert d["severity"] == "high"
        assert d["tool_id"] == "test_tool"


class TestErrorClassifier:
    """Tests for ErrorClassifier."""

    def test_classify_value_error(self):
        """Test classifying ValueError."""
        classifier = ErrorClassifier()

        category, severity = classifier.classify(ValueError("test"))

        assert category == ErrorCategory.VALIDATION

    def test_classify_timeout_error(self):
        """Test classifying TimeoutError."""
        classifier = ErrorClassifier()

        category, severity = classifier.classify(TimeoutError())

        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.HIGH

    def test_classify_connection_error(self):
        """Test classifying ConnectionError."""
        classifier = ErrorClassifier()

        category, severity = classifier.classify(ConnectionError())

        assert category == ErrorCategory.NETWORK

    def test_classify_unknown(self):
        """Test classifying unknown exception."""
        classifier = ErrorClassifier()

        class CustomError(Exception):
            pass

        category, severity = classifier.classify(CustomError())

        assert category == ErrorCategory.INTERNAL

    def test_add_custom_rule(self):
        """Test adding custom classification rule."""
        classifier = ErrorClassifier()

        class RateLimitError(Exception):
            pass

        classifier.add_rule(RateLimitError, ErrorCategory.RATE_LIMIT, ErrorSeverity.HIGH)

        category, severity = classifier.classify(RateLimitError())

        assert category == ErrorCategory.RATE_LIMIT
        assert severity == ErrorSeverity.HIGH


class TestRecoveryStrategies:
    """Tests for recovery strategies."""

    @pytest.mark.asyncio
    async def test_retry_recovery(self):
        """Test retry recovery."""
        tool = ConditionalFailTool(fail_count=2)
        recovery = RetryRecovery(max_retries=3, delay=0.01)

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await recovery.recover(error, tool, ErrorInput(value=5))

        assert result is not None
        assert result.value == 10

    @pytest.mark.asyncio
    async def test_retry_recovery_all_fail(self):
        """Test retry recovery when all fail."""
        tool = FailTool()
        recovery = RetryRecovery(max_retries=2, delay=0.01)

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await recovery.recover(error, tool, ErrorInput(value=5))

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_recovery(self):
        """Test fallback recovery."""
        fallback_tool = SuccessTool()
        recovery = FallbackRecovery(fallback_tool)

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await recovery.recover(error, FailTool(), ErrorInput(value=5))

        assert result is not None
        assert result.value == 10

    @pytest.mark.asyncio
    async def test_default_value_recovery(self):
        """Test default value recovery."""
        default = ErrorOutput(value=0, success=False, error="Recovery fallback")
        recovery = DefaultValueRecovery(default)

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await recovery.recover(error, FailTool(), ErrorInput(value=5))

        assert result is not None
        assert result.success is False
        assert result.error == "Recovery fallback"

    @pytest.mark.asyncio
    async def test_default_value_callable(self):
        """Test default value recovery with callable."""

        def create_default(error, input_data):
            return ErrorOutput(value=input_data.value, success=False)

        recovery = DefaultValueRecovery(create_default)

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await recovery.recover(error, FailTool(), ErrorInput(value=7))

        assert result.value == 7

    @pytest.mark.asyncio
    async def test_transform_recovery(self):
        """Test transform recovery."""
        tool = ConditionalFailTool()

        def transformer(input_data, error):
            return ErrorInput(value=input_data.value, fail=False)

        recovery = TransformRecovery(transformer)

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await recovery.recover(error, tool, ErrorInput(value=5, fail=True))

        assert result is not None


class TestErrorHandler:
    """Tests for ErrorHandler."""

    def test_handle_exception(self):
        """Test handling exception."""
        handler = ErrorHandler()

        error = handler.handle(
            exception=ValueError("Test error"),
            tool_id="test_tool",
            input_data={"value": 5},
        )

        assert error.category == ErrorCategory.VALIDATION
        assert error.context.tool_id == "test_tool"

    def test_on_error_callback(self):
        """Test on_error callback."""
        errors = []
        handler = ErrorHandler(on_error=lambda e: errors.append(e))

        handler.handle(ValueError("Test"), "test_tool")

        assert len(errors) == 1

    def test_add_recovery(self):
        """Test adding recovery strategy."""
        handler = ErrorHandler()
        handler.add_recovery(ErrorCategory.VALIDATION, DefaultValueRecovery(None))

        error = handler.handle(ValueError("Test"), "test_tool")

        assert error.recoverable

    @pytest.mark.asyncio
    async def test_recover(self):
        """Test recovery execution."""
        handler = ErrorHandler()
        handler.add_recovery(
            ErrorCategory.EXECUTION,
            DefaultValueRecovery(ErrorOutput(value=0)),
        )

        error = ToolError(
            message="Test",
            category=ErrorCategory.EXECUTION,
        )

        result = await handler.recover(error, FailTool(), ErrorInput(value=5))

        assert result is not None

    def test_error_stats(self):
        """Test error statistics."""
        handler = ErrorHandler()

        handler.handle(ValueError("Error 1"), "tool1")
        handler.handle(TypeError("Error 2"), "tool1")
        handler.handle(ValueError("Error 3"), "tool2")

        stats = handler.get_error_stats()

        assert stats["total"] == 3
        assert stats["by_category"]["validation"] == 3

    def test_clear_history(self):
        """Test clearing error history."""
        handler = ErrorHandler()

        handler.handle(ValueError("Error"), "tool")
        handler.clear_history()

        stats = handler.get_error_stats()
        assert stats["total"] == 0


class TestErrorHandlingToolWrapper:
    """Tests for ErrorHandlingToolWrapper."""

    @pytest.mark.asyncio
    async def test_success_passthrough(self):
        """Test success passes through."""
        wrapper = ErrorHandlingToolWrapper(SuccessTool())

        result = await wrapper.execute(ErrorInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error is handled."""
        errors = []
        handler = ErrorHandler(on_error=lambda e: errors.append(e))
        wrapper = ErrorHandlingToolWrapper(FailTool(), handler=handler)

        result = await wrapper.execute(ErrorInput(value=5))

        assert result is None
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_reraise(self):
        """Test error re-raising."""
        wrapper = ErrorHandlingToolWrapper(FailTool(), reraise=True)

        with pytest.raises(ValueError):
            await wrapper.execute(ErrorInput(value=5))

    @pytest.mark.asyncio
    async def test_error_result(self):
        """Test custom error result."""

        def make_error_result(error):
            return ErrorOutput(success=False, error=error.message)

        wrapper = ErrorHandlingToolWrapper(
            FailTool(),
            error_result=make_error_result,
        )

        result = await wrapper.execute(ErrorInput(value=5))

        assert result.success is False
        assert "Intentional failure" in result.error

    @pytest.mark.asyncio
    async def test_recovery(self):
        """Test error recovery."""
        handler = ErrorHandler()
        handler.add_recovery(
            ErrorCategory.VALIDATION,
            DefaultValueRecovery(ErrorOutput(value=99)),
        )

        wrapper = ErrorHandlingToolWrapper(FailTool(), handler=handler)

        result = await wrapper.execute(ErrorInput(value=5))

        assert result.value == 99

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        wrapper = ErrorHandlingToolWrapper(SuccessTool())

        assert wrapper.metadata.id == "success_tool"


class TestErrorBoundary:
    """Tests for ErrorBoundary."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Test success within boundary."""
        boundary = ErrorBoundary()

        result = await boundary.execute(SuccessTool(), ErrorInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_error_fallback(self):
        """Test error with fallback."""
        fallback = ErrorOutput(value=0, success=False)
        boundary = ErrorBoundary(fallback=fallback)

        result = await boundary.execute(FailTool(), ErrorInput(value=5))

        assert result.success is False
        assert result.value == 0

    @pytest.mark.asyncio
    async def test_callable_fallback(self):
        """Test callable fallback."""

        def create_fallback(error, input_data):
            return ErrorOutput(value=input_data.value, success=False, error=error.message)

        boundary = ErrorBoundary(fallback=create_fallback)

        result = await boundary.execute(FailTool(), ErrorInput(value=7))

        assert result.value == 7
        assert "Intentional failure" in result.error

    @pytest.mark.asyncio
    async def test_last_error(self):
        """Test last_error property."""
        boundary = ErrorBoundary(fallback=None)

        await boundary.execute(FailTool(), ErrorInput(value=5))

        assert boundary.last_error is not None
        assert boundary.last_error.category == ErrorCategory.VALIDATION

    @pytest.mark.asyncio
    async def test_on_error_callback(self):
        """Test on_error callback."""
        errors = []
        boundary = ErrorBoundary(on_error=lambda e: errors.append(e))

        await boundary.execute(FailTool(), ErrorInput(value=5))

        assert len(errors) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_error_handling(self):
        """Test with_error_handling function."""
        wrapper = with_error_handling(SuccessTool())

        result = await wrapper.execute(ErrorInput(value=5))

        assert result.value == 10

    def test_create_error_handler(self):
        """Test create_error_handler function."""
        errors = []
        handler = create_error_handler(on_error=lambda e: errors.append(e))

        handler.handle(ValueError("Test"), "tool")

        assert len(errors) == 1

    def test_classify_error(self):
        """Test classify_error function."""
        category, severity = classify_error(TimeoutError())

        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.HIGH
