"""ReAct (Reasoning + Acting) agent implementation for TinyLLM.

This module implements the ReAct pattern from the paper:
"ReAct: Synergizing Reasoning and Acting in Language Models"
https://arxiv.org/abs/2210.03629

The ReAct pattern follows a simple loop:
1. Thought: The LLM reasons about what to do next
2. Action: The LLM decides on an action and its input
3. Observation: The tool executes and returns a result
4. Repeat until Final Answer or max iterations

Example ReAct trace:
    Question: What is the square root of 144?

    Thought: I need to calculate the square root of 144.
    Action: calculator
    Action Input: sqrt(144)
    Observation: 12

    Thought: I have the answer.
    Final Answer: The square root of 144 is 12.
"""

import re
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.tools.base import BaseTool

logger = get_logger(__name__, component="react_agent")


class ReActStepType(str, Enum):
    """Type of step in the ReAct loop."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


class ReActStep(BaseModel):
    """A single step in the ReAct reasoning trace."""

    step_type: ReActStepType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Result from executing an action."""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""

    pass


class ReActConfig(BaseModel):
    """Configuration for ReAct agent."""

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reasoning iterations before stopping",
    )
    max_tokens: int = Field(
        default=50000,
        ge=100,
        description="Maximum total tokens to use across all LLM calls",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    stop_on_error: bool = Field(
        default=False,
        description="Whether to stop on first tool error or continue",
    )
    verbose: bool = Field(
        default=True,
        description="Whether to log detailed execution information",
    )
    thought_pattern: str = Field(
        default=r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)",
        description="Regex pattern to extract thought",
    )
    action_pattern: str = Field(
        default=r"Action:\s*(\w+)",
        description="Regex pattern to extract action name",
    )
    action_input_pattern: str = Field(
        default=r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\nFinal Answer:|$)",
        description="Regex pattern to extract action input",
    )
    final_answer_pattern: str = Field(
        default=r"Final Answer:\s*(.+?)$",
        description="Regex pattern to extract final answer",
    )


class ReActAgent:
    """ReAct agent that orchestrates the think-act-observe loop.

    The ReAct agent iteratively:
    1. Prompts the LLM to think about what to do
    2. Parses the LLM output for actions to take
    3. Executes tools based on parsed actions
    4. Feeds observations back to the LLM
    5. Repeats until a final answer or max iterations

    Example usage:
        >>> from tinyllm.models.client import OllamaClient
        >>> from tinyllm.tools import CalculatorTool
        >>> from tinyllm.agents import ReActAgent, ReActConfig
        >>>
        >>> # Create LLM client
        >>> client = OllamaClient(default_model="qwen2.5:0.5b")
        >>>
        >>> # Create agent
        >>> agent = ReActAgent(
        ...     llm_client=client,
        ...     config=ReActConfig(max_iterations=5, verbose=True)
        ... )
        >>>
        >>> # Register tools
        >>> agent.register_tool("calculator", CalculatorTool())
        >>>
        >>> # Run agent
        >>> result = await agent.run("What is the square root of 144?")
        >>> print(result)
    """

    def __init__(
        self,
        llm_client: Any,
        config: Optional[ReActConfig] = None,
    ):
        """Initialize ReAct agent.

        Args:
            llm_client: LLM client with generate() method that returns response with
                       .response attribute containing the generated text.
            config: Agent configuration. If None, uses default config.
        """
        self.llm_client = llm_client
        self.config = config or ReActConfig()
        self.tools: dict[str, BaseTool] = {}
        self.steps: list[ReActStep] = []
        self.total_tokens = 0

        logger.info(
            "react_agent_initialized",
            max_iterations=self.config.max_iterations,
            max_tokens=self.config.max_tokens,
        )

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a tool for the agent to use.

        Args:
            name: Name to use when referencing the tool in actions.
            tool: Tool instance implementing BaseTool interface.
        """
        self.tools[name] = tool
        logger.info("tool_registered", tool_name=name, tool_id=tool.metadata.id)

    def register_function(
        self,
        name: str,
        func: Callable[[str], str],
        description: str,
    ) -> None:
        """Register a simple function as a tool.

        This is a convenience method for registering simple functions
        that don't need the full BaseTool interface.

        Args:
            name: Name to use when referencing the function.
            func: Function that takes a string input and returns string output.
            description: Description of what the function does.
        """
        # Create a wrapper tool for the function
        from pydantic import BaseModel

        class FuncInput(BaseModel):
            input: str

        class FuncOutput(BaseModel):
            success: bool
            output: str
            error: Optional[str] = None

        class FunctionTool(BaseTool[FuncInput, FuncOutput]):
            metadata = type(
                "Metadata",
                (),
                {
                    "id": name,
                    "name": name,
                    "description": description,
                    "category": "utility",
                    "sandbox_required": False,
                },
            )()
            input_type = FuncInput
            output_type = FuncOutput

            def __init__(self, func: Callable[[str], str]):
                super().__init__()
                self.func = func

            async def execute(self, input: FuncInput) -> FuncOutput:
                try:
                    result = self.func(input.input)
                    return FuncOutput(success=True, output=result)
                except Exception as e:
                    return FuncOutput(success=False, output="", error=str(e))

        self.tools[name] = FunctionTool(func)
        logger.info("function_registered", function_name=name)

    def _build_system_prompt(self) -> str:
        """Build system prompt with available tools."""
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(
                f"- {name}: {tool.metadata.description}\n"
                f"  Input: {tool.input_type.model_json_schema()}"
            )

        tools_section = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."

        return f"""You are a helpful assistant that uses the ReAct (Reasoning + Acting) pattern to solve problems.

You should follow this format EXACTLY:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [This will be filled in by the system]

After seeing the observation, you can continue with another Thought/Action/Observation cycle, or provide a final answer:

Final Answer: [Your final answer to the question]

Available Tools:
{tools_section}

IMPORTANT RULES:
1. Always start with "Thought:" to explain your reasoning
2. Use "Action:" to specify which tool to use (must be one of the available tools)
3. Use "Action Input:" to provide the input for the tool
4. Wait for "Observation:" before continuing (this will be provided automatically)
5. When you have enough information, provide "Final Answer:" with your complete answer
6. Do NOT make up observations - wait for the system to provide them
7. Each action must be on its own line
8. Be concise but thorough in your reasoning

Example:
Question: What is the sum of 15 and 27?

Thought: I need to add two numbers together. I should use the calculator tool.
Action: calculator
Action Input: 15 + 27
Observation: 42

Thought: I have the answer from the calculator.
Final Answer: The sum of 15 and 27 is 42.
"""

    def _build_prompt(self, question: str, context: str = "") -> str:
        """Build the prompt for the LLM.

        Args:
            question: User's question to answer.
            context: Conversation context from previous steps.

        Returns:
            Complete prompt string.
        """
        if context:
            return f"{context}\n\n"
        else:
            return f"Question: {question}\n\n"

    def _parse_response(self, response: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse LLM response for thought, action, action input, and final answer.

        Args:
            response: Raw LLM response text.

        Returns:
            Tuple of (thought, action, action_input, final_answer).
            Any component can be None if not found.
        """
        # Extract thought
        thought_match = re.search(
            self.config.thought_pattern, response, re.DOTALL | re.IGNORECASE
        )
        thought = thought_match.group(1).strip() if thought_match else None

        # Extract action
        action_match = re.search(self.config.action_pattern, response, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None

        # Extract action input
        action_input_match = re.search(
            self.config.action_input_pattern, response, re.DOTALL | re.IGNORECASE
        )
        action_input = action_input_match.group(1).strip() if action_input_match else None

        # Extract final answer
        final_answer_match = re.search(
            self.config.final_answer_pattern, response, re.DOTALL | re.IGNORECASE
        )
        final_answer = final_answer_match.group(1).strip() if final_answer_match else None

        return thought, action, action_input, final_answer

    async def _execute_action(self, action: str, action_input: str) -> ActionResult:
        """Execute a tool action.

        Args:
            action: Name of the tool to execute.
            action_input: Input to pass to the tool.

        Returns:
            ActionResult with execution results.

        Raises:
            ToolExecutionError: If stop_on_error is True and execution fails.
        """
        if action not in self.tools:
            error_msg = f"Unknown tool: {action}. Available tools: {list(self.tools.keys())}"
            logger.error("unknown_tool", action=action, available_tools=list(self.tools.keys()))

            if self.config.stop_on_error:
                raise ToolExecutionError(error_msg)

            return ActionResult(success=False, output="", error=error_msg)

        tool = self.tools[action]

        try:
            # Parse action input into tool's expected input format
            # For now, we'll try to match it to the input schema
            input_schema = tool.input_type

            # Try to intelligently construct input
            # If input schema has an 'expression' field (like Calculator), use it
            # If it has 'input' field, use it
            # Otherwise, try to parse as JSON or use as-is
            input_data = {}

            # Get field names from schema
            field_names = list(input_schema.model_fields.keys())

            if len(field_names) == 1:
                # Single field - use action_input directly
                input_data[field_names[0]] = action_input
            else:
                # Multiple fields - try to parse as JSON or use first field
                try:
                    import json

                    input_data = json.loads(action_input)
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, use for first field
                    if field_names:
                        input_data[field_names[0]] = action_input

            tool_input = input_schema(**input_data)

            # Execute tool
            logger.info(
                "executing_tool",
                tool_name=action,
                tool_id=tool.metadata.id,
                input_preview=action_input[:100],
            )

            result = await tool.safe_execute(tool_input)

            # Extract output from result
            if hasattr(result, "success") and not result.success:
                error = getattr(result, "error", "Unknown error")
                logger.error("tool_execution_failed", tool_name=action, error=error)

                if self.config.stop_on_error:
                    raise ToolExecutionError(f"Tool {action} failed: {error}")

                return ActionResult(success=False, output="", error=error)

            # Get output value
            output = ""
            if hasattr(result, "formatted"):
                output = str(result.formatted)
            elif hasattr(result, "value"):
                output = str(result.value)
            elif hasattr(result, "output"):
                output = str(result.output)
            else:
                output = str(result)

            logger.info(
                "tool_execution_success",
                tool_name=action,
                output_preview=output[:100],
            )

            return ActionResult(
                success=True,
                output=output,
                metadata={"tool_id": tool.metadata.id},
            )

        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error("tool_execution_exception", tool_name=action, error=str(e), exc_info=True)

            if self.config.stop_on_error:
                raise ToolExecutionError(error_msg)

            return ActionResult(success=False, output="", error=error_msg)

    async def run(self, question: str) -> str:
        """Run the ReAct agent to answer a question.

        Args:
            question: Question to answer.

        Returns:
            Final answer string.

        Raises:
            ToolExecutionError: If stop_on_error is True and a tool fails.
            ValueError: If max iterations or token budget exceeded without answer.
        """
        self.steps = []
        self.total_tokens = 0

        logger.info("react_agent_started", question=question[:100])

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Initialize context with question
        context = f"Question: {question}\n\n"

        for iteration in range(self.config.max_iterations):
            logger.info("react_iteration_start", iteration=iteration + 1)

            # Check token budget
            if self.total_tokens >= self.config.max_tokens:
                error_msg = f"Token budget exceeded: {self.total_tokens} >= {self.config.max_tokens}"
                logger.error("token_budget_exceeded", total_tokens=self.total_tokens)
                raise ValueError(error_msg)

            # Generate LLM response
            try:
                response = await self.llm_client.generate(
                    prompt=context,
                    system=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=min(2000, self.config.max_tokens - self.total_tokens),
                )

                # Track tokens
                if hasattr(response, "eval_count") and response.eval_count:
                    self.total_tokens += response.eval_count
                if hasattr(response, "prompt_eval_count") and response.prompt_eval_count:
                    self.total_tokens += response.prompt_eval_count

                llm_response = response.response

                if self.config.verbose:
                    logger.info("llm_response", iteration=iteration + 1, response=llm_response[:200])

            except Exception as e:
                logger.error("llm_generation_failed", error=str(e), exc_info=True)
                raise

            # Parse response
            thought, action, action_input, final_answer = self._parse_response(llm_response)

            # Record thought
            if thought:
                self.steps.append(
                    ReActStep(step_type=ReActStepType.THOUGHT, content=thought)
                )
                if self.config.verbose:
                    logger.info("thought", content=thought)

            # Check for final answer
            if final_answer:
                self.steps.append(
                    ReActStep(step_type=ReActStepType.FINAL_ANSWER, content=final_answer)
                )
                logger.info(
                    "react_agent_completed",
                    iterations=iteration + 1,
                    total_tokens=self.total_tokens,
                    answer_preview=final_answer[:100],
                )
                return final_answer

            # Execute action if present
            if action and action_input:
                # Record action
                self.steps.append(
                    ReActStep(
                        step_type=ReActStepType.ACTION,
                        content=f"{action}: {action_input}",
                        metadata={"action": action, "action_input": action_input},
                    )
                )

                if self.config.verbose:
                    logger.info("action", action=action, action_input=action_input)

                # Execute action
                result = await self._execute_action(action, action_input)

                # Get observation text
                observation = result.output if result.success else f"Error: {result.error}"

                # Record observation
                self.steps.append(
                    ReActStep(
                        step_type=ReActStepType.OBSERVATION,
                        content=observation,
                        metadata={"success": result.success},
                    )
                )

                if self.config.verbose:
                    logger.info("observation", content=observation)

                # Append to context
                context += llm_response
                if not llm_response.endswith("\n"):
                    context += "\n"
                context += f"Observation: {observation}\n\n"

            else:
                # No valid action/final answer - append response and continue
                # This handles partial responses or malformed output
                context += llm_response
                if not llm_response.endswith("\n"):
                    context += "\n"

                logger.warning(
                    "no_action_or_answer",
                    iteration=iteration + 1,
                    response_preview=llm_response[:200],
                )

        # Max iterations reached
        error_msg = f"Max iterations ({self.config.max_iterations}) reached without final answer"
        logger.error(
            "max_iterations_reached",
            max_iterations=self.config.max_iterations,
            total_steps=len(self.steps),
        )
        raise ValueError(error_msg)

    def get_trace(self) -> list[ReActStep]:
        """Get the reasoning trace from the last run.

        Returns:
            List of ReActStep objects showing the agent's reasoning process.
        """
        return self.steps

    def get_trace_string(self) -> str:
        """Get a human-readable string representation of the trace.

        Returns:
            Formatted trace string.
        """
        lines = []
        for step in self.steps:
            if step.step_type == ReActStepType.THOUGHT:
                lines.append(f"Thought: {step.content}")
            elif step.step_type == ReActStepType.ACTION:
                action = step.metadata.get("action", "")
                action_input = step.metadata.get("action_input", "")
                lines.append(f"Action: {action}")
                lines.append(f"Action Input: {action_input}")
            elif step.step_type == ReActStepType.OBSERVATION:
                lines.append(f"Observation: {step.content}")
            elif step.step_type == ReActStepType.FINAL_ANSWER:
                lines.append(f"Final Answer: {step.content}")
            lines.append("")  # Blank line between steps
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the agent state for a new run."""
        self.steps = []
        self.total_tokens = 0
        logger.info("react_agent_reset")
