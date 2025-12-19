"""Tests for Plan-and-Execute agent implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, Field

from tinyllm.agents.plan_execute import (
    ExecutionResult,
    PlanExecuteAgent,
    PlanExecuteConfig,
    PlanExecuteError,
    PlanExecuteStatistics,
    PlanStep,
    StepStatus,
)
from tinyllm.tools.base import BaseTool, ToolMetadata


# Mock Tool for Testing
class MockToolInput(BaseModel):
    """Input for mock tool."""

    expression: str = Field(description="Expression to evaluate")


class MockToolOutput(BaseModel):
    """Output from mock tool."""

    success: bool
    value: float | None = None
    formatted: str | None = None
    error: str | None = None


class MockCalculatorTool(BaseTool[MockToolInput, MockToolOutput]):
    """Mock calculator tool for testing."""

    metadata = ToolMetadata(
        id="calculator",
        name="Calculator",
        description="Evaluates mathematical expressions",
        category="computation",
        sandbox_required=False,
    )
    input_type = MockToolInput
    output_type = MockToolOutput

    async def execute(self, input: MockToolInput) -> MockToolOutput:
        """Execute mock calculator."""
        # Simple mock implementation
        if input.expression == "15 * 3":
            return MockToolOutput(success=True, value=45.0, formatted="45")
        elif input.expression == "45 + 20":
            return MockToolOutput(success=True, value=65.0, formatted="65")
        elif input.expression == "10 + 5":
            return MockToolOutput(success=True, value=15.0, formatted="15")
        elif input.expression == "error":
            return MockToolOutput(success=False, error="Calculation error")
        else:
            # Default calculation
            try:
                result = eval(input.expression)
                return MockToolOutput(success=True, value=float(result), formatted=str(result))
            except Exception as e:
                return MockToolOutput(success=False, error=str(e))


class MockSearchInput(BaseModel):
    """Input for mock search tool."""

    query: str


class MockSearchOutput(BaseModel):
    """Output from mock search tool."""

    success: bool
    output: str
    error: str | None = None


class MockSearchTool(BaseTool[MockSearchInput, MockSearchOutput]):
    """Mock search tool for testing."""

    metadata = ToolMetadata(
        id="search",
        name="Search",
        description="Searches for information",
        category="search",
        sandbox_required=False,
    )
    input_type = MockSearchInput
    output_type = MockSearchOutput

    async def execute(self, input: MockSearchInput) -> MockSearchOutput:
        """Execute mock search."""
        if input.query == "weather in Paris":
            return MockSearchOutput(success=True, output="The weather in Paris is sunny, 22°C")
        else:
            return MockSearchOutput(success=True, output=f"Search result for: {input.query}")


# Mock LLM Response
class MockGenerateResponse:
    """Mock response from LLM generate call."""

    def __init__(self, response_text: str, tokens: int = 100):
        self.response = response_text
        self.eval_count = tokens
        self.prompt_eval_count = 50


# Mock LLM Client
class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize with predefined responses."""
        self.responses = responses or []
        self.call_count = 0

    async def generate(self, prompt: str, system: str = "", **kwargs) -> MockGenerateResponse:
        """Mock generate method."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return MockGenerateResponse(response)
        else:
            # Default response
            return MockGenerateResponse("Default response")


class TestPlanExecuteConfig:
    """Tests for PlanExecuteConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PlanExecuteConfig()
        assert config.max_plan_steps == 10
        assert config.max_replans == 2
        assert config.max_tokens == 50000
        assert config.temperature == 0.0
        assert config.replan_on_failure is True
        assert config.stop_on_step_failure is False
        assert config.verbose is True
        assert config.enable_streaming is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlanExecuteConfig(
            max_plan_steps=5,
            max_replans=1,
            max_tokens=10000,
            temperature=0.7,
            replan_on_failure=False,
            stop_on_step_failure=True,
            verbose=False,
            enable_streaming=True,
        )
        assert config.max_plan_steps == 5
        assert config.max_replans == 1
        assert config.max_tokens == 10000
        assert config.temperature == 0.7
        assert config.replan_on_failure is False
        assert config.stop_on_step_failure is True
        assert config.verbose is False
        assert config.enable_streaming is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Max plan steps must be >= 1
        with pytest.raises(ValueError):
            PlanExecuteConfig(max_plan_steps=0)

        # Temperature must be in valid range
        with pytest.raises(ValueError):
            PlanExecuteConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            PlanExecuteConfig(temperature=2.1)


class TestPlanStep:
    """Tests for PlanStep."""

    def test_basic_step(self):
        """Test creating a basic step."""
        step = PlanStep(
            step_number=1,
            description="Calculate 2 + 2",
        )
        assert step.step_number == 1
        assert step.description == "Calculate 2 + 2"
        assert step.tool_name is None
        assert step.tool_input is None
        assert step.dependencies == []
        assert step.status == StepStatus.PENDING

    def test_step_with_tool(self):
        """Test creating a step with tool."""
        step = PlanStep(
            step_number=1,
            description="Calculate 15 * 3",
            tool_name="calculator",
            tool_input="15 * 3",
        )
        assert step.tool_name == "calculator"
        assert step.tool_input == "15 * 3"

    def test_step_with_dependencies(self):
        """Test creating a step with dependencies."""
        step = PlanStep(
            step_number=2,
            description="Add to previous result",
            dependencies=[1],
        )
        assert step.dependencies == [1]

    def test_step_status_update(self):
        """Test updating step status."""
        step = PlanStep(step_number=1, description="Test")
        assert step.status == StepStatus.PENDING

        step.status = StepStatus.IN_PROGRESS
        assert step.status == StepStatus.IN_PROGRESS

        step.status = StepStatus.COMPLETED
        assert step.status == StepStatus.COMPLETED


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ExecutionResult(
            step_number=1,
            success=True,
            output="42",
        )
        assert result.step_number == 1
        assert result.success is True
        assert result.output == "42"
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = ExecutionResult(
            step_number=2,
            success=False,
            output="",
            error="Calculation failed",
        )
        assert result.success is False
        assert result.error == "Calculation failed"

    def test_result_with_tokens(self):
        """Test result with token tracking."""
        result = ExecutionResult(
            step_number=1,
            success=True,
            output="result",
            tokens_used=150,
        )
        assert result.tokens_used == 150


class TestPlanExecuteAgent:
    """Tests for PlanExecuteAgent."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def agent(self, mock_client):
        """Create agent instance."""
        config = PlanExecuteConfig(max_plan_steps=5, verbose=False)
        return PlanExecuteAgent(llm_client=mock_client, config=config)

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.llm_client is not None
        assert agent.config.max_plan_steps == 5
        assert agent.tools == {}
        assert agent.current_plan == []
        assert agent.execution_results == []
        assert agent.total_tokens == 0
        assert agent.replan_count == 0

    def test_register_tool(self, agent):
        """Test tool registration."""
        tool = MockCalculatorTool()
        agent.register_tool("calculator", tool)

        assert "calculator" in agent.tools
        assert agent.tools["calculator"] == tool

    def test_register_function(self, agent):
        """Test function registration."""
        def simple_func(input: str) -> str:
            return f"Result: {input}"

        agent.register_function("simple", simple_func, "A simple test function")

        assert "simple" in agent.tools
        assert agent.tools["simple"].metadata.name == "simple"

    def test_parse_plan_basic(self, agent):
        """Test parsing a basic plan."""
        plan_text = """
Step 1: Calculate 15 * 3
Tool: calculator
Input: 15 * 3
Dependencies: none

Step 2: Add 20 to the result
Tool: calculator
Input: <result_from_step_1> + 20
Dependencies: 1

END PLAN
"""
        steps = agent._parse_plan(plan_text)

        assert len(steps) == 2

        assert steps[0].step_number == 1
        assert "15 * 3" in steps[0].description
        assert steps[0].tool_name == "calculator"
        assert steps[0].tool_input == "15 * 3"
        assert steps[0].dependencies == []

        assert steps[1].step_number == 2
        assert steps[1].tool_name == "calculator"
        assert steps[1].dependencies == [1]

    def test_parse_plan_no_tools(self, agent):
        """Test parsing a plan with no tools."""
        plan_text = """
Step 1: Think about the answer
Tool: none
Input: none
Dependencies: none

END PLAN
"""
        steps = agent._parse_plan(plan_text)

        assert len(steps) == 1
        assert steps[0].tool_name is None
        assert steps[0].tool_input is None

    def test_parse_plan_multiple_dependencies(self, agent):
        """Test parsing a plan with multiple dependencies."""
        plan_text = """
Step 1: First step
Tool: none
Input: none
Dependencies: none

Step 2: Second step
Tool: none
Input: none
Dependencies: none

Step 3: Combine results
Tool: none
Input: none
Dependencies: 1, 2

END PLAN
"""
        steps = agent._parse_plan(plan_text)

        assert len(steps) == 3
        assert steps[2].dependencies == [1, 2]

    def test_parse_plan_invalid_dependency(self, agent):
        """Test parsing fails with invalid dependency."""
        plan_text = """
Step 1: First step
Tool: none
Input: none
Dependencies: 2

END PLAN
"""
        with pytest.raises(ValueError, match="invalid dependency"):
            agent._parse_plan(plan_text)

    def test_parse_plan_empty(self, agent):
        """Test parsing fails with empty plan."""
        plan_text = "No steps here"

        with pytest.raises(ValueError, match="Failed to parse any steps"):
            agent._parse_plan(plan_text)

    @pytest.mark.asyncio
    async def test_plan_generation(self):
        """Test plan generation."""
        plan_response = """
Step 1: Calculate 15 * 3
Tool: calculator
Input: 15 * 3
Dependencies: none

Step 2: Add 20 to result
Tool: calculator
Input: <result_from_step_1> + 20
Dependencies: 1

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))

        plan = await agent.plan("Calculate 15 * 3 and add 20")

        assert len(plan) == 2
        assert plan[0].tool_name == "calculator"
        assert plan[1].dependencies == [1]

    @pytest.mark.asyncio
    async def test_plan_with_context(self):
        """Test plan generation with context."""
        plan_response = """
Step 1: Try a different approach
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))

        context = "Previous attempt failed at step 1"
        plan = await agent.plan("Calculate something", context=context)

        assert len(plan) == 1

    @pytest.mark.asyncio
    async def test_execute_step_with_tool(self, agent):
        """Test executing a step with a tool."""
        agent.register_tool("calculator", MockCalculatorTool())

        step = PlanStep(
            step_number=1,
            description="Calculate 15 * 3",
            tool_name="calculator",
            tool_input="15 * 3",
        )

        result = await agent.execute_step(step, {})

        assert result.success is True
        assert result.output == "45"
        assert result.step_number == 1

    @pytest.mark.asyncio
    async def test_execute_step_with_dependency(self, agent):
        """Test executing a step with dependencies."""
        agent.register_tool("calculator", MockCalculatorTool())

        # First step result
        prev_result = ExecutionResult(
            step_number=1,
            success=True,
            output="45",
        )

        # Second step depends on first
        step = PlanStep(
            step_number=2,
            description="Add 20 to result",
            tool_name="calculator",
            tool_input="<result_from_step_1> + 20",
            dependencies=[1],
        )

        result = await agent.execute_step(step, {1: prev_result})

        assert result.success is True
        assert result.output == "65"

    @pytest.mark.asyncio
    async def test_execute_step_missing_dependency(self, agent):
        """Test executing a step with missing dependency."""
        step = PlanStep(
            step_number=2,
            description="Depends on step 1",
            dependencies=[1],
        )

        result = await agent.execute_step(step, {})

        assert result.success is False
        assert "Dependency step 1 not yet executed" in result.error

    @pytest.mark.asyncio
    async def test_execute_step_failed_dependency(self, agent):
        """Test executing a step with failed dependency."""
        # Failed dependency
        prev_result = ExecutionResult(
            step_number=1,
            success=False,
            output="",
            error="Failed",
        )

        step = PlanStep(
            step_number=2,
            description="Depends on step 1",
            dependencies=[1],
        )

        result = await agent.execute_step(step, {1: prev_result})

        assert result.success is False
        assert "Dependency step 1 failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_step_unknown_tool(self, agent):
        """Test executing a step with unknown tool."""
        step = PlanStep(
            step_number=1,
            description="Use unknown tool",
            tool_name="unknown",
            tool_input="test",
        )

        result = await agent.execute_step(step, {})

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_step_tool_error(self, agent):
        """Test executing a step when tool returns error."""
        agent.register_tool("calculator", MockCalculatorTool())

        step = PlanStep(
            step_number=1,
            description="Cause error",
            tool_name="calculator",
            tool_input="error",
        )

        result = await agent.execute_step(step, {})

        assert result.success is False
        assert "Calculation error" in result.error

    @pytest.mark.asyncio
    async def test_execute_step_no_tool(self):
        """Test executing a step without a tool (using LLM)."""
        llm_response = "This is a reasoned answer"
        client = MockLLMClient([llm_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))

        step = PlanStep(
            step_number=1,
            description="Think about the answer",
            tool_name=None,
        )

        result = await agent.execute_step(step, {})

        assert result.success is True
        assert result.output == llm_response

    @pytest.mark.asyncio
    async def test_run_simple_success(self):
        """Test successful run with simple plan."""
        plan_response = """
Step 1: Calculate 15 * 3
Tool: calculator
Input: 15 * 3
Dependencies: none

END PLAN
"""
        responses = [plan_response]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("What is 15 * 3?")

        assert result == "45"
        assert len(agent.current_plan) == 1
        assert len(agent.execution_results) == 1
        assert agent.statistics.total_steps_completed == 1

    @pytest.mark.asyncio
    async def test_run_multi_step_success(self):
        """Test successful run with multi-step plan."""
        plan_response = """
Step 1: Calculate 15 * 3
Tool: calculator
Input: 15 * 3
Dependencies: none

Step 2: Add 20 to the result
Tool: calculator
Input: <result_from_step_1> + 20
Dependencies: 1

END PLAN
"""
        responses = [plan_response]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("Calculate 15 * 3 and add 20")

        assert result == "65"
        assert len(agent.current_plan) == 2
        assert len(agent.execution_results) == 2
        assert agent.statistics.total_steps_completed == 2
        assert agent.statistics.total_tool_calls == 2

    @pytest.mark.asyncio
    async def test_run_with_replan(self):
        """Test run that requires re-planning."""
        # First plan with error
        first_plan = """
Step 1: This will fail
Tool: calculator
Input: error
Dependencies: none

END PLAN
"""
        # Second plan after replan
        second_plan = """
Step 1: Calculate correctly
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        responses = [first_plan, second_plan]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(
            llm_client=client,
            config=PlanExecuteConfig(
                verbose=False,
                replan_on_failure=True,
                max_replans=2,
            ),
        )
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("Calculate something")

        # Should have re-planned and succeeded
        assert result == "15"
        assert agent.statistics.total_replans == 1
        assert agent.statistics.total_steps_failed >= 1
        assert agent.statistics.total_steps_completed >= 1

    @pytest.mark.asyncio
    async def test_run_max_replans_exceeded(self):
        """Test run when max replans exceeded."""
        # Always fail
        failing_plan = """
Step 1: This will fail
Tool: calculator
Input: error
Dependencies: none

END PLAN
"""
        responses = [failing_plan] * 5

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(
            llm_client=client,
            config=PlanExecuteConfig(
                verbose=False,
                replan_on_failure=True,
                max_replans=1,
            ),
        )
        agent.register_tool("calculator", MockCalculatorTool())

        with pytest.raises(ValueError, match="Max replans"):
            await agent.run("Calculate something")

    @pytest.mark.asyncio
    async def test_run_stop_on_failure(self):
        """Test run with stop_on_step_failure enabled."""
        plan_response = """
Step 1: This will fail
Tool: calculator
Input: error
Dependencies: none

END PLAN
"""
        responses = [plan_response]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(
            llm_client=client,
            config=PlanExecuteConfig(
                verbose=False,
                stop_on_step_failure=True,
            ),
        )
        agent.register_tool("calculator", MockCalculatorTool())

        with pytest.raises(PlanExecuteError, match="Step 1 failed"):
            await agent.run("Calculate something")

    @pytest.mark.asyncio
    async def test_run_no_replan_on_failure(self):
        """Test run with replan_on_failure disabled."""
        plan_response = """
Step 1: This will fail
Tool: calculator
Input: error
Dependencies: none

Step 2: This won't execute
Tool: calculator
Input: 10 + 5
Dependencies: 1

END PLAN
"""
        responses = [plan_response]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(
            llm_client=client,
            config=PlanExecuteConfig(
                verbose=False,
                replan_on_failure=False,
                stop_on_step_failure=False,
            ),
        )
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("Calculate something")

        # Should complete with whatever it has
        assert agent.statistics.total_steps_failed >= 1

    @pytest.mark.asyncio
    async def test_get_plan(self):
        """Test getting the current plan."""
        plan_response = """
Step 1: Calculate
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("Calculate")

        plan = agent.get_plan()
        assert len(plan) == 1
        assert plan[0].description == "Calculate"

    @pytest.mark.asyncio
    async def test_get_results(self):
        """Test getting execution results."""
        plan_response = """
Step 1: Calculate
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("Calculate")

        results = agent.get_results()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "15"

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting execution statistics."""
        plan_response = """
Step 1: Calculate
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("Calculate")

        stats = agent.get_statistics()
        assert stats.total_steps_planned == 1
        assert stats.total_steps_executed == 1
        assert stats.total_steps_completed == 1
        assert stats.total_steps_failed == 0
        assert stats.total_tool_calls == 1
        assert stats.total_tokens_used > 0
        assert stats.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting agent state."""
        plan_response = """
Step 1: Calculate
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("Calculate")

        assert len(agent.current_plan) > 0
        assert len(agent.execution_results) > 0
        assert agent.total_tokens > 0

        agent.reset()

        assert len(agent.current_plan) == 0
        assert len(agent.execution_results) == 0
        assert agent.total_tokens == 0
        assert agent.replan_count == 0

    @pytest.mark.asyncio
    async def test_streaming_updates(self):
        """Test that streaming can be enabled."""
        plan_response = """
Step 1: Calculate
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(
            llm_client=client,
            config=PlanExecuteConfig(verbose=False, enable_streaming=True),
        )
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("Calculate")

        assert result == "15"
        assert agent.config.enable_streaming is True

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self):
        """Test token budget enforcement."""
        plan_response = """
Step 1: Calculate
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        client = MockLLMClient([plan_response])
        agent = PlanExecuteAgent(
            llm_client=client,
            config=PlanExecuteConfig(verbose=False, max_tokens=50),  # Very low budget
        )
        agent.register_tool("calculator", MockCalculatorTool())

        # Should fail due to token budget
        with pytest.raises(ValueError, match="Token budget exceeded"):
            await agent.run("Calculate")

    @pytest.mark.asyncio
    async def test_system_prompts_include_tools(self, agent):
        """Test that system prompts include tool descriptions."""
        agent.register_tool("calculator", MockCalculatorTool())

        planning_prompt = agent._build_planning_system_prompt()
        execution_prompt = agent._build_execution_system_prompt()

        assert "calculator" in planning_prompt.lower()
        assert "calculator" in execution_prompt.lower()

    @pytest.mark.asyncio
    async def test_multiple_tools_in_plan(self):
        """Test plan using multiple different tools."""
        plan_response = """
Step 1: Search for weather
Tool: search
Input: weather in Paris
Dependencies: none

Step 2: Calculate something
Tool: calculator
Input: 10 + 5
Dependencies: none

END PLAN
"""
        responses = [plan_response]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())
        agent.register_tool("search", MockSearchTool())

        result = await agent.run("Get weather and calculate")

        assert agent.statistics.total_steps_completed == 2
        assert agent.statistics.total_tool_calls == 2

    @pytest.mark.asyncio
    async def test_complex_dependencies(self):
        """Test plan with complex dependency chain."""
        plan_response = """
Step 1: First calculation
Tool: calculator
Input: 10 + 5
Dependencies: none

Step 2: Second calculation
Tool: calculator
Input: 2 * 3
Dependencies: none

Step 3: Combine results
Tool: none
Input: none
Dependencies: 1, 2

END PLAN
"""
        execution_response = "Combined result: 15 and 6"
        responses = [plan_response, execution_response]

        client = MockLLMClient(responses)
        agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("Complex task")

        assert agent.statistics.total_steps_completed == 3


class TestPlanExecuteStatistics:
    """Tests for PlanExecuteStatistics."""

    def test_default_statistics(self):
        """Test default statistics values."""
        stats = PlanExecuteStatistics()
        assert stats.total_steps_planned == 0
        assert stats.total_steps_executed == 0
        assert stats.total_steps_completed == 0
        assert stats.total_steps_failed == 0
        assert stats.total_steps_skipped == 0
        assert stats.total_replans == 0
        assert stats.total_tokens_used == 0
        assert stats.total_tool_calls == 0
        assert stats.execution_time_ms == 0.0

    def test_statistics_update(self):
        """Test updating statistics."""
        stats = PlanExecuteStatistics()
        stats.total_steps_planned = 5
        stats.total_steps_executed = 4
        stats.total_steps_completed = 3
        stats.total_steps_failed = 1
        stats.total_replans = 1
        stats.total_tokens_used = 1000
        stats.total_tool_calls = 3

        assert stats.total_steps_planned == 5
        assert stats.total_steps_executed == 4
        assert stats.total_steps_completed == 3
        assert stats.total_steps_failed == 1


@pytest.mark.asyncio
async def test_integration_realistic_scenario():
    """Integration test with realistic multi-step scenario."""
    plan_response = """
Step 1: Search for weather information
Tool: search
Input: weather in Paris
Dependencies: none

Step 2: Calculate temperature conversion
Tool: calculator
Input: 22 * 1.8 + 32
Dependencies: 1

END PLAN
"""
    responses = [plan_response]

    client = MockLLMClient(responses)
    agent = PlanExecuteAgent(
        llm_client=client,
        config=PlanExecuteConfig(max_plan_steps=5, verbose=False),
    )

    # Register tools
    agent.register_tool("search", MockSearchTool())
    agent.register_tool("calculator", MockCalculatorTool())

    # Run agent
    result = await agent.run("What's the weather in Paris in Fahrenheit?")

    # Verify execution
    assert agent.statistics.total_steps_completed == 2
    assert agent.statistics.total_tool_calls == 2
    assert len(agent.execution_results) == 2

    # Verify both tools were used
    assert any("Paris" in r.output for r in agent.execution_results if r.success)


@pytest.mark.asyncio
async def test_integration_with_registered_function():
    """Integration test using registered function."""
    def reverse_string(input: str) -> str:
        """Reverse a string."""
        return input[::-1]

    plan_response = """
Step 1: Reverse the string
Tool: reverse
Input: hello
Dependencies: none

END PLAN
"""
    responses = [plan_response]

    client = MockLLMClient(responses)
    agent = PlanExecuteAgent(llm_client=client, config=PlanExecuteConfig(verbose=False))

    # Register function
    agent.register_function("reverse", reverse_string, "Reverses a string")

    result = await agent.run("Reverse 'hello'")

    assert "olleh" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
