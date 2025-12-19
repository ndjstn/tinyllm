"""Tests for ReAct agent implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
from pydantic import BaseModel, Field

from tinyllm.agents.react import (
    ActionResult,
    ReActAgent,
    ReActConfig,
    ReActStep,
    ReActStepType,
    ToolExecutionError,
)
from tinyllm.tools.base import BaseTool, ToolConfig, ToolMetadata


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
        if input.expression == "2 + 2":
            return MockToolOutput(success=True, value=4.0, formatted="4")
        elif input.expression == "sqrt(144)":
            return MockToolOutput(success=True, value=12.0, formatted="12")
        elif input.expression == "error":
            return MockToolOutput(success=False, error="Calculation error")
        else:
            return MockToolOutput(success=True, value=42.0, formatted="42")


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
        if input.query == "capital of France":
            return MockSearchOutput(success=True, output="Paris is the capital of France")
        else:
            return MockSearchOutput(success=True, output="Search result for: " + input.query)


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
            # Default response if we run out
            return MockGenerateResponse("Final Answer: Default answer")


class TestReActConfig:
    """Tests for ReActConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReActConfig()
        assert config.max_iterations == 10
        assert config.max_tokens == 50000
        assert config.temperature == 0.0
        assert config.stop_on_error is False
        assert config.verbose is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReActConfig(
            max_iterations=5,
            max_tokens=10000,
            temperature=0.7,
            stop_on_error=True,
            verbose=False,
        )
        assert config.max_iterations == 5
        assert config.max_tokens == 10000
        assert config.temperature == 0.7
        assert config.stop_on_error is True
        assert config.verbose is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Max iterations must be >= 1
        with pytest.raises(ValueError):
            ReActConfig(max_iterations=0)

        # Temperature must be in valid range
        with pytest.raises(ValueError):
            ReActConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            ReActConfig(temperature=2.1)


class TestReActStep:
    """Tests for ReActStep."""

    def test_thought_step(self):
        """Test creating a thought step."""
        step = ReActStep(
            step_type=ReActStepType.THOUGHT,
            content="I need to calculate something",
        )
        assert step.step_type == ReActStepType.THOUGHT
        assert step.content == "I need to calculate something"
        assert step.metadata == {}

    def test_action_step(self):
        """Test creating an action step."""
        step = ReActStep(
            step_type=ReActStepType.ACTION,
            content="calculator: 2 + 2",
            metadata={"action": "calculator", "action_input": "2 + 2"},
        )
        assert step.step_type == ReActStepType.ACTION
        assert step.metadata["action"] == "calculator"


class TestReActAgent:
    """Tests for ReActAgent."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def agent(self, mock_client):
        """Create agent instance."""
        config = ReActConfig(max_iterations=5, verbose=False)
        return ReActAgent(llm_client=mock_client, config=config)

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.llm_client is not None
        assert agent.config.max_iterations == 5
        assert agent.tools == {}
        assert agent.steps == []
        assert agent.total_tokens == 0

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

    def test_parse_response_with_action(self, agent):
        """Test parsing response with action."""
        response = """Thought: I need to calculate 2 + 2
Action: calculator
Action Input: 2 + 2
"""
        thought, action, action_input, final_answer = agent._parse_response(response)

        assert thought == "I need to calculate 2 + 2"
        assert action == "calculator"
        assert action_input == "2 + 2"
        assert final_answer is None

    def test_parse_response_with_final_answer(self, agent):
        """Test parsing response with final answer."""
        response = """Thought: I have the answer now
Final Answer: The result is 4
"""
        thought, action, action_input, final_answer = agent._parse_response(response)

        assert thought == "I have the answer now"
        assert action is None
        assert action_input is None
        assert final_answer == "The result is 4"

    def test_parse_response_multiline_final_answer(self, agent):
        """Test parsing final answer with multiple lines."""
        response = """Thought: I have all the information
Final Answer: The answer is complex.
It has multiple parts.
First part: A
Second part: B"""
        thought, action, action_input, final_answer = agent._parse_response(response)

        assert "The answer is complex" in final_answer
        assert "multiple parts" in final_answer

    @pytest.mark.asyncio
    async def test_execute_action_success(self, agent):
        """Test successful action execution."""
        tool = MockCalculatorTool()
        agent.register_tool("calculator", tool)

        result = await agent._execute_action("calculator", "2 + 2")

        assert result.success is True
        assert result.output == "4"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_action_unknown_tool(self, agent):
        """Test execution with unknown tool."""
        result = await agent._execute_action("unknown", "input")

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_action_tool_error(self, agent):
        """Test execution when tool returns error."""
        tool = MockCalculatorTool()
        agent.register_tool("calculator", tool)

        result = await agent._execute_action("calculator", "error")

        assert result.success is False
        assert result.error == "Calculation error"

    @pytest.mark.asyncio
    async def test_execute_action_stop_on_error(self, agent):
        """Test stop_on_error behavior."""
        agent.config.stop_on_error = True
        tool = MockCalculatorTool()
        agent.register_tool("calculator", tool)

        with pytest.raises(ToolExecutionError):
            await agent._execute_action("calculator", "error")

    @pytest.mark.asyncio
    async def test_run_simple_success(self):
        """Test successful run with one iteration."""
        responses = [
            """Thought: I need to calculate 2 + 2
Action: calculator
Action Input: 2 + 2
""",
            """Thought: I have the result
Final Answer: The answer is 4""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("What is 2 + 2?")

        assert result == "The answer is 4"
        assert len(agent.steps) > 0

    @pytest.mark.asyncio
    async def test_run_immediate_answer(self):
        """Test run when LLM provides immediate answer."""
        responses = [
            """Thought: This is a simple question
Final Answer: The answer is obvious"""
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))

        result = await agent.run("What is 1 + 1?")

        assert result == "The answer is obvious"

    @pytest.mark.asyncio
    async def test_run_multiple_iterations(self):
        """Test run with multiple iterations."""
        responses = [
            """Thought: First, I need to search
Action: search
Action Input: capital of France
""",
            """Thought: Now I have the information
Final Answer: Paris is the capital of France""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("search", MockSearchTool())

        result = await agent.run("What is the capital of France?")

        assert "Paris" in result
        # Should have thought, action, observation, thought, final answer
        assert len(agent.steps) >= 4

    @pytest.mark.asyncio
    async def test_run_max_iterations_exceeded(self):
        """Test run exceeding max iterations."""
        # LLM never provides final answer
        responses = [
            """Thought: Let me think
Action: calculator
Action Input: 1 + 1
"""
        ] * 10  # More than max_iterations

        client = MockLLMClient(responses)
        agent = ReActAgent(
            llm_client=client,
            config=ReActConfig(max_iterations=3, verbose=False),
        )
        agent.register_tool("calculator", MockCalculatorTool())

        with pytest.raises(ValueError, match="Max iterations"):
            await agent.run("What is 1 + 1?")

    @pytest.mark.asyncio
    async def test_run_token_budget_exceeded(self):
        """Test run exceeding token budget."""
        responses = [
            """Thought: Thinking
Action: calculator
Action Input: 1 + 1
"""
        ] * 5

        client = MockLLMClient(responses)
        agent = ReActAgent(
            llm_client=client,
            config=ReActConfig(max_tokens=200, verbose=False),  # Very low budget
        )
        agent.register_tool("calculator", MockCalculatorTool())

        with pytest.raises(ValueError, match="Token budget exceeded"):
            await agent.run("What is 1 + 1?")

    @pytest.mark.asyncio
    async def test_get_trace(self):
        """Test getting the reasoning trace."""
        responses = [
            """Thought: I need to calculate
Action: calculator
Action Input: 2 + 2
""",
            """Thought: Got the result
Final Answer: 4""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("What is 2 + 2?")
        trace = agent.get_trace()

        assert len(trace) > 0
        # Should have thoughts, action, observation, final answer
        step_types = [step.step_type for step in trace]
        assert ReActStepType.THOUGHT in step_types
        assert ReActStepType.ACTION in step_types
        assert ReActStepType.OBSERVATION in step_types
        assert ReActStepType.FINAL_ANSWER in step_types

    @pytest.mark.asyncio
    async def test_get_trace_string(self):
        """Test getting trace as formatted string."""
        responses = [
            """Thought: Calculate
Action: calculator
Action Input: 2 + 2
""",
            """Final Answer: 4""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("What is 2 + 2?")
        trace_str = agent.get_trace_string()

        assert "Thought:" in trace_str
        assert "Action:" in trace_str
        assert "Observation:" in trace_str
        assert "Final Answer:" in trace_str

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting agent state."""
        responses = ["""Final Answer: 4"""]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))

        await agent.run("What is 2 + 2?")

        assert len(agent.steps) > 0
        assert agent.total_tokens > 0

        agent.reset()

        assert len(agent.steps) == 0
        assert agent.total_tokens == 0

    @pytest.mark.asyncio
    async def test_system_prompt_includes_tools(self, agent):
        """Test that system prompt includes tool descriptions."""
        agent.register_tool("calculator", MockCalculatorTool())
        agent.register_tool("search", MockSearchTool())

        system_prompt = agent._build_system_prompt()

        assert "calculator" in system_prompt.lower()
        assert "search" in system_prompt.lower()
        assert "Available Tools" in system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_no_tools(self, agent):
        """Test system prompt when no tools registered."""
        system_prompt = agent._build_system_prompt()

        assert "No tools available" in system_prompt

    @pytest.mark.asyncio
    async def test_malformed_response_handling(self):
        """Test handling of malformed LLM responses."""
        # Response without proper format
        responses = [
            """Just some random text without proper formatting""",
            """Final Answer: I'll answer anyway""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))

        result = await agent.run("Question?")

        # Should still complete with final answer
        assert result == "I'll answer anyway"

    @pytest.mark.asyncio
    async def test_partial_action_response(self):
        """Test handling of response with action but no input."""
        responses = [
            """Thought: I need to do something
Action: calculator
""",  # Missing Action Input
            """Final Answer: Done""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        result = await agent.run("Question?")

        # Should continue and complete
        assert result == "Done"

    @pytest.mark.asyncio
    async def test_multiple_tools_in_sequence(self):
        """Test using multiple different tools."""
        responses = [
            """Thought: First search
Action: search
Action Input: test query
""",
            """Thought: Now calculate
Action: calculator
Action Input: 2 + 2
""",
            """Thought: Done
Final Answer: Complete""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())
        agent.register_tool("search", MockSearchTool())

        result = await agent.run("Multi-step question")

        assert result == "Complete"

        # Verify both tools were used
        action_steps = [s for s in agent.steps if s.step_type == ReActStepType.ACTION]
        assert len(action_steps) == 2

    @pytest.mark.asyncio
    async def test_observation_recorded_correctly(self):
        """Test that observations are recorded correctly."""
        responses = [
            """Thought: Calculate
Action: calculator
Action Input: sqrt(144)
""",
            """Final Answer: 12""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("What is sqrt(144)?")

        observations = [s for s in agent.steps if s.step_type == ReActStepType.OBSERVATION]
        assert len(observations) == 1
        assert observations[0].content == "12"
        assert observations[0].metadata["success"] is True

    @pytest.mark.asyncio
    async def test_error_observation_recorded(self):
        """Test that error observations are recorded."""
        responses = [
            """Thought: Calculate
Action: calculator
Action Input: error
""",
            """Final Answer: Failed""",
        ]

        client = MockLLMClient(responses)
        agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))
        agent.register_tool("calculator", MockCalculatorTool())

        await agent.run("Question")

        observations = [s for s in agent.steps if s.step_type == ReActStepType.OBSERVATION]
        assert len(observations) == 1
        assert "Error:" in observations[0].content
        assert observations[0].metadata["success"] is False

    @pytest.mark.asyncio
    async def test_case_insensitive_parsing(self, agent):
        """Test that parsing works with different cases."""
        response = """thought: lowercase thought
action: calculator
action input: 2 + 2
"""
        thought, action, action_input, final_answer = agent._parse_response(response)

        assert thought == "lowercase thought"
        assert action == "calculator"
        assert action_input == "2 + 2"

    @pytest.mark.asyncio
    async def test_build_prompt_with_context(self, agent):
        """Test prompt building with context."""
        question = "What is 2 + 2?"
        context = "Previous conversation here\n"

        prompt = agent._build_prompt(question, context)

        assert context in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_without_context(self, agent):
        """Test prompt building without context."""
        question = "What is 2 + 2?"

        prompt = agent._build_prompt(question, "")

        assert "Question: What is 2 + 2?" in prompt


class TestActionResult:
    """Tests for ActionResult model."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ActionResult(success=True, output="42")

        assert result.success is True
        assert result.output == "42"
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = ActionResult(success=False, output="", error="Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = ActionResult(
            success=True,
            output="result",
            metadata={"tool_id": "calc", "duration": 100},
        )

        assert result.metadata["tool_id"] == "calc"
        assert result.metadata["duration"] == 100


class TestToolExecutionError:
    """Tests for ToolExecutionError exception."""

    def test_exception_creation(self):
        """Test creating the exception."""
        error = ToolExecutionError("Tool failed")
        assert str(error) == "Tool failed"

    def test_exception_raised(self):
        """Test raising the exception."""
        with pytest.raises(ToolExecutionError):
            raise ToolExecutionError("Test error")


@pytest.mark.asyncio
async def test_integration_realistic_scenario():
    """Integration test with realistic multi-step scenario."""
    responses = [
        # First, search for information
        """Thought: I need to find out what the capital of France is.
Action: search
Action Input: capital of France
""",
        # Then, verify with a calculation (contrived, but tests multiple tools)
        """Thought: I found that Paris is the capital. Let me verify I can count the letters.
Action: calculator
Action Input: 5
""",
        # Finally, provide answer
        """Thought: I have all the information I need.
Final Answer: The capital of France is Paris, which has 5 letters.""",
    ]

    client = MockLLMClient(responses)
    agent = ReActAgent(
        llm_client=client,
        config=ReActConfig(max_iterations=5, verbose=False),
    )

    # Register tools
    agent.register_tool("search", MockSearchTool())
    agent.register_tool("calculator", MockCalculatorTool())

    # Run agent
    result = await agent.run("What is the capital of France?")

    # Verify result
    assert "Paris" in result
    assert len(agent.steps) > 0

    # Verify trace structure
    trace = agent.get_trace()
    step_types = [s.step_type for s in trace]

    # Should have multiple thoughts
    assert step_types.count(ReActStepType.THOUGHT) >= 3
    # Should have two actions (search and calculator)
    assert step_types.count(ReActStepType.ACTION) == 2
    # Should have two observations
    assert step_types.count(ReActStepType.OBSERVATION) == 2
    # Should have one final answer
    assert step_types.count(ReActStepType.FINAL_ANSWER) == 1


@pytest.mark.asyncio
async def test_integration_with_registered_function():
    """Integration test using registered function instead of tool."""

    def reverse_string(input: str) -> str:
        """Reverse a string."""
        return input[::-1]

    responses = [
        """Thought: I need to reverse the string
Action: reverse
Action Input: hello
""",
        """Thought: I got the reversed string
Final Answer: The reversed string is olleh""",
    ]

    client = MockLLMClient(responses)
    agent = ReActAgent(llm_client=client, config=ReActConfig(verbose=False))

    # Register function
    agent.register_function("reverse", reverse_string, "Reverses a string")

    result = await agent.run("Reverse 'hello'")

    assert "olleh" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
