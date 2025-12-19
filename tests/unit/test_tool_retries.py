"""Tests for tool retries."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.retries import (
    ConditionalRetry,
    ExponentialBackoff,
    ExponentialJitterBackoff,
    FixedBackoff,
    LinearBackoff,
    RetryConfig,
    RetryStrategy,
    RetryableToolWrapper,
    ToolRetry,
    retry_on,
    with_retry,
)


class RetryInput(BaseModel):
    """Input for retry tests."""

    value: int = 0


class RetryOutput(BaseModel):
    """Output for retry tests."""

    value: int = 0
    success: bool = True
    error: str | None = None


class SuccessTool(BaseTool[RetryInput, RetryOutput]):
    """Tool that succeeds."""

    metadata = ToolMetadata(
        id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        category="utility",
    )
    input_type = RetryInput
    output_type = RetryOutput

    async def execute(self, input: RetryInput) -> RetryOutput:
        return RetryOutput(value=input.value * 2)


class FailTool(BaseTool[RetryInput, RetryOutput]):
    """Tool that always fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = RetryInput
    output_type = RetryOutput

    async def execute(self, input: RetryInput) -> RetryOutput:
        raise ValueError("Intentional failure")


class FlakeyTool(BaseTool[RetryInput, RetryOutput]):
    """Tool that fails initially then succeeds."""

    metadata = ToolMetadata(
        id="flakey_tool",
        name="Flakey Tool",
        description="Fails then succeeds",
        category="utility",
    )
    input_type = RetryInput
    output_type = RetryOutput

    def __init__(self, fail_count: int = 2):
        super().__init__()
        self.fail_count = fail_count
        self.attempts = 0

    async def execute(self, input: RetryInput) -> RetryOutput:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ValueError(f"Failure {self.attempts}")
        return RetryOutput(value=input.value * 2)


class SoftFailTool(BaseTool[RetryInput, RetryOutput]):
    """Tool that returns soft failure."""

    metadata = ToolMetadata(
        id="soft_fail_tool",
        name="Soft Fail Tool",
        description="Returns failure",
        category="utility",
    )
    input_type = RetryInput
    output_type = RetryOutput

    def __init__(self, fail_count: int = 2):
        super().__init__()
        self.fail_count = fail_count
        self.attempts = 0

    async def execute(self, input: RetryInput) -> RetryOutput:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            return RetryOutput(success=False, error=f"Soft failure {self.attempts}")
        return RetryOutput(value=input.value * 2)


class TestBackoffCalculators:
    """Tests for backoff calculators."""

    def test_fixed_backoff(self):
        """Test fixed backoff."""
        calc = FixedBackoff()
        config = RetryConfig(initial_delay=1.0)

        assert calc.calculate(1, config) == 1.0
        assert calc.calculate(2, config) == 1.0
        assert calc.calculate(3, config) == 1.0

    def test_linear_backoff(self):
        """Test linear backoff."""
        calc = LinearBackoff()
        config = RetryConfig(initial_delay=1.0, max_delay=10.0)

        assert calc.calculate(1, config) == 1.0
        assert calc.calculate(2, config) == 2.0
        assert calc.calculate(3, config) == 3.0
        assert calc.calculate(20, config) == 10.0  # Capped

    def test_exponential_backoff(self):
        """Test exponential backoff."""
        calc = ExponentialBackoff()
        config = RetryConfig(initial_delay=1.0, multiplier=2.0, max_delay=100.0)

        assert calc.calculate(1, config) == 1.0
        assert calc.calculate(2, config) == 2.0
        assert calc.calculate(3, config) == 4.0
        assert calc.calculate(4, config) == 8.0

    def test_exponential_jitter_backoff(self):
        """Test exponential with jitter."""
        calc = ExponentialJitterBackoff()
        config = RetryConfig(initial_delay=1.0, multiplier=2.0, jitter=0.1)

        # Should be around 1.0 +/- 10%
        delay = calc.calculate(1, config)
        assert 0.9 <= delay <= 1.1


class TestToolRetry:
    """Tests for ToolRetry."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test success without retry."""
        retry = ToolRetry(SuccessTool())

        result = await retry.execute(RetryInput(value=5))

        assert result.success
        assert result.attempts == 1
        assert result.output.value == 10

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """Test retry then success."""
        tool = FlakeyTool(fail_count=2)
        retry = ToolRetry(
            tool,
            config=RetryConfig(max_attempts=5, initial_delay=0.01),
        )

        result = await retry.execute(RetryInput(value=5))

        assert result.success
        assert result.attempts == 3
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        """Test all retries fail."""
        retry = ToolRetry(
            FailTool(),
            config=RetryConfig(max_attempts=3, initial_delay=0.01),
        )

        result = await retry.execute(RetryInput(value=5))

        assert not result.success
        assert result.attempts == 3
        assert len(result.errors) == 3

    @pytest.mark.asyncio
    async def test_soft_failure_retry(self):
        """Test retry on soft failure."""
        tool = SoftFailTool(fail_count=2)
        retry = ToolRetry(
            tool,
            config=RetryConfig(max_attempts=5, initial_delay=0.01),
        )

        result = await retry.execute(RetryInput(value=5))

        assert result.success
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback."""
        callbacks = []

        def on_retry(attempt, error, delay):
            callbacks.append((attempt, str(error)))

        tool = FlakeyTool(fail_count=2)
        retry = ToolRetry(
            tool,
            config=RetryConfig(max_attempts=5, initial_delay=0.01),
            on_retry=on_retry,
        )

        await retry.execute(RetryInput(value=5))

        assert len(callbacks) == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exception stops immediately."""
        retry = ToolRetry(
            FailTool(),
            config=RetryConfig(
                max_attempts=5,
                initial_delay=0.01,
                retryable_exceptions={TypeError},  # Only retry TypeError
            ),
        )

        result = await retry.execute(RetryInput(value=5))

        assert not result.success
        assert result.attempts == 1  # Stopped immediately


class TestRetryableToolWrapper:
    """Tests for RetryableToolWrapper."""

    @pytest.mark.asyncio
    async def test_wrapper_success(self):
        """Test wrapper success."""
        wrapper = RetryableToolWrapper(SuccessTool())

        result = await wrapper.execute(RetryInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_wrapper_retry(self):
        """Test wrapper with retry."""
        tool = FlakeyTool(fail_count=1)
        wrapper = RetryableToolWrapper(
            tool,
            max_attempts=3,
            initial_delay=0.01,
        )

        result = await wrapper.execute(RetryInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_wrapper_all_fail(self):
        """Test wrapper raises on all failures."""
        wrapper = RetryableToolWrapper(
            FailTool(),
            max_attempts=2,
            initial_delay=0.01,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await wrapper.execute(RetryInput(value=5))

        assert "2 attempts failed" in str(exc_info.value)


class TestConditionalRetry:
    """Tests for ConditionalRetry."""

    @pytest.mark.asyncio
    async def test_no_retry_when_condition_false(self):
        """Test no retry when condition is false."""
        retry = ConditionalRetry(
            SuccessTool(),
            should_retry=lambda x: False,  # Never retry
        )

        result = await retry.execute(RetryInput(value=5))

        assert result.success
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_retry_when_condition_true(self):
        """Test retry when condition is true."""
        attempts = [0]

        class CountingTool(BaseTool[RetryInput, RetryOutput]):
            metadata = ToolMetadata(
                id="counting", name="Counting", description="Counts", category="utility"
            )
            input_type = RetryInput
            output_type = RetryOutput

            async def execute(self, input: RetryInput) -> RetryOutput:
                attempts[0] += 1
                return RetryOutput(value=attempts[0])

        retry = ConditionalRetry(
            CountingTool(),
            should_retry=lambda x: x.value < 3,  # Retry until value >= 3
            config=RetryConfig(max_attempts=5, initial_delay=0.01),
        )

        result = await retry.execute(RetryInput())

        assert result.success
        assert result.attempts == 3


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_retry(self):
        """Test with_retry function."""
        tool = FlakeyTool(fail_count=1)
        retry = with_retry(tool, max_attempts=3, initial_delay=0.01)

        result = await retry.execute(RetryInput(value=5))

        assert result.success

    @pytest.mark.asyncio
    async def test_retry_on_decorator(self):
        """Test retry_on decorator."""
        tool = FailTool()
        decorator = retry_on(ValueError, max_attempts=2)
        retry = decorator(tool)

        result = await retry.execute(RetryInput(value=5))

        assert not result.success
        assert result.attempts == 2
