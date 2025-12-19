"""Plan-and-Execute agent implementation for TinyLLM.

This module implements the Plan-and-Execute pattern where an agent:
1. First generates a complete plan of steps to accomplish a task
2. Executes each step in sequence
3. Optionally re-plans if steps fail or new information is discovered

The Plan-and-Execute pattern is useful for complex tasks that benefit from
upfront planning and structured execution, as opposed to the more dynamic
ReAct pattern.

Example usage:
    >>> from tinyllm.models.client import OllamaClient
    >>> from tinyllm.tools import CalculatorTool
    >>> from tinyllm.agents import PlanExecuteAgent, PlanExecuteConfig
    >>>
    >>> # Create LLM client
    >>> client = OllamaClient(default_model="qwen2.5:0.5b")
    >>>
    >>> # Create agent
    >>> agent = PlanExecuteAgent(
    ...     llm_client=client,
    ...     config=PlanExecuteConfig(max_replans=2, verbose=True)
    ... )
    >>>
    >>> # Register tools
    >>> agent.register_tool("calculator", CalculatorTool())
    >>>
    >>> # Run agent
    >>> result = await agent.run("Calculate 15 * 3 and then add 20")
    >>> print(result)
"""

import re
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.tools.base import BaseTool

logger = get_logger(__name__, component="plan_execute_agent")


class StepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    step_number: int = Field(ge=1, description="Step number in the plan")
    description: str = Field(description="What this step should accomplish")
    tool_name: Optional[str] = Field(
        default=None, description="Tool to use for this step, if any"
    )
    tool_input: Optional[str] = Field(
        default=None, description="Input for the tool, if any"
    )
    dependencies: list[int] = Field(
        default_factory=list, description="Step numbers this step depends on"
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING, description="Current status of the step"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result from executing a single plan step."""

    step_number: int
    success: bool
    output: str
    error: Optional[str] = None
    tokens_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanExecuteConfig(BaseModel):
    """Configuration for Plan-and-Execute agent."""

    max_plan_steps: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of steps allowed in a plan",
    )
    max_replans: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum number of times to re-plan after failures",
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
    replan_on_failure: bool = Field(
        default=True,
        description="Whether to automatically re-plan when a step fails",
    )
    stop_on_step_failure: bool = Field(
        default=False,
        description="Whether to stop execution on first step failure",
    )
    verbose: bool = Field(
        default=True,
        description="Whether to log detailed execution information",
    )
    enable_streaming: bool = Field(
        default=False,
        description="Whether to stream execution updates",
    )


class PlanExecuteStatistics(BaseModel):
    """Statistics from plan execution."""

    total_steps_planned: int = 0
    total_steps_executed: int = 0
    total_steps_completed: int = 0
    total_steps_failed: int = 0
    total_steps_skipped: int = 0
    total_replans: int = 0
    total_tokens_used: int = 0
    total_tool_calls: int = 0
    execution_time_ms: float = 0.0


class PlanExecuteError(Exception):
    """Raised when plan execution fails."""

    pass


class PlanExecuteAgent:
    """Plan-and-Execute agent that plans then executes steps sequentially.

    The agent works in two main phases:
    1. Planning: Generate a complete plan of steps to accomplish the task
    2. Execution: Execute each step in sequence, handling dependencies

    The agent can also re-plan dynamically if steps fail or new information
    is discovered during execution.

    Example usage:
        >>> agent = PlanExecuteAgent(llm_client=client)
        >>> agent.register_tool("calculator", CalculatorTool())
        >>> result = await agent.run("Calculate 10 * 5 and add 20")
    """

    def __init__(
        self,
        llm_client: Any,
        config: Optional[PlanExecuteConfig] = None,
    ):
        """Initialize Plan-and-Execute agent.

        Args:
            llm_client: LLM client with generate() method that returns response with
                       .response attribute containing the generated text.
            config: Agent configuration. If None, uses default config.
        """
        self.llm_client = llm_client
        self.config = config or PlanExecuteConfig()
        self.tools: dict[str, BaseTool] = {}
        self.current_plan: list[PlanStep] = []
        self.execution_results: list[ExecutionResult] = []
        self.statistics = PlanExecuteStatistics()
        self.total_tokens = 0
        self.replan_count = 0

        logger.info(
            "plan_execute_agent_initialized",
            max_plan_steps=self.config.max_plan_steps,
            max_replans=self.config.max_replans,
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

    def _build_planning_system_prompt(self) -> str:
        """Build system prompt for the planning phase."""
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(
                f"- {name}: {tool.metadata.description}\n"
                f"  Input: {tool.input_type.model_json_schema()}"
            )

        tools_section = (
            "\n".join(tool_descriptions)
            if tool_descriptions
            else "No tools available."
        )

        return f"""You are a planning assistant that creates detailed execution plans.

Your task is to break down a user's question into a sequence of concrete steps that can be executed to answer the question.

Available Tools:
{tools_section}

For each step, specify:
1. Step number (starting from 1)
2. Description of what the step should accomplish
3. Tool name (if a tool is needed)
4. Tool input (if a tool is needed)
5. Dependencies (which previous steps must complete first)

Format your plan EXACTLY like this:

Step 1: [description]
Tool: [tool_name or "none"]
Input: [tool_input or "none"]
Dependencies: [comma-separated step numbers or "none"]

Step 2: [description]
Tool: [tool_name or "none"]
Input: [tool_input or "none"]
Dependencies: [comma-separated step numbers or "none"]

...

IMPORTANT RULES:
1. Be specific and actionable in step descriptions
2. Keep the plan concise but complete
3. Use tools when needed for computations or lookups
4. Specify dependencies accurately (a step depends on another if it needs that step's output)
5. Number steps sequentially starting from 1
6. Don't create more than {self.config.max_plan_steps} steps
7. Each line must follow the format exactly
8. After the plan, add a line "END PLAN"

Example:
Question: What is 15 * 3, and then add 20 to that result?

Step 1: Calculate 15 * 3
Tool: calculator
Input: 15 * 3
Dependencies: none

Step 2: Add 20 to the result from step 1
Tool: calculator
Input: <result_from_step_1> + 20
Dependencies: 1

END PLAN
"""

    def _build_execution_system_prompt(self) -> str:
        """Build system prompt for the execution phase."""
        tool_descriptions = []
        for name, tool in self.tools.items():
            tool_descriptions.append(
                f"- {name}: {tool.metadata.description}\n"
                f"  Input: {tool.input_type.model_json_schema()}"
            )

        tools_section = (
            "\n".join(tool_descriptions)
            if tool_descriptions
            else "No tools available."
        )

        return f"""You are an execution assistant that carries out individual plan steps.

Available Tools:
{tools_section}

You will be given:
1. A step description
2. Previous step results (if any)
3. A tool to use (if specified)

Your task is to:
1. Execute the step using the specified tool (if any)
2. Use results from previous steps as needed
3. Provide a clear output for this step

If no tool is specified, provide a reasoned answer based on the step description and previous results.

Be concise and factual in your responses.
"""

    def _parse_plan(self, plan_text: str) -> list[PlanStep]:
        """Parse LLM-generated plan into PlanStep objects.

        Args:
            plan_text: Raw plan text from LLM.

        Returns:
            List of PlanStep objects.

        Raises:
            ValueError: If plan parsing fails.
        """
        steps = []

        # Split by "Step N:" pattern
        step_pattern = r"Step\s+(\d+):\s*(.+?)(?=Step\s+\d+:|END PLAN|$)"
        matches = re.finditer(step_pattern, plan_text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            step_num = int(match.group(1))
            step_content = match.group(2).strip()

            # Extract description (first line)
            lines = step_content.split("\n")
            description = lines[0].strip()

            # Extract tool name
            tool_match = re.search(
                r"Tool:\s*(.+?)(?:\n|$)", step_content, re.IGNORECASE
            )
            tool_name = None
            if tool_match:
                tool_text = tool_match.group(1).strip()
                if tool_text.lower() not in ["none", "n/a", ""]:
                    tool_name = tool_text

            # Extract tool input
            input_match = re.search(
                r"Input:\s*(.+?)(?:\n|$)", step_content, re.IGNORECASE
            )
            tool_input = None
            if input_match:
                input_text = input_match.group(1).strip()
                if input_text.lower() not in ["none", "n/a", ""]:
                    tool_input = input_text

            # Extract dependencies
            dep_match = re.search(
                r"Dependencies:\s*(.+?)(?:\n|$)", step_content, re.IGNORECASE
            )
            dependencies = []
            if dep_match:
                dep_text = dep_match.group(1).strip()
                if dep_text.lower() not in ["none", "n/a", ""]:
                    # Parse comma-separated numbers
                    dep_parts = [d.strip() for d in dep_text.split(",")]
                    for part in dep_parts:
                        if part.isdigit():
                            dependencies.append(int(part))

            steps.append(
                PlanStep(
                    step_number=step_num,
                    description=description,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    dependencies=dependencies,
                )
            )

        if not steps:
            raise ValueError("Failed to parse any steps from plan")

        # Validate dependencies
        for step in steps:
            for dep in step.dependencies:
                if dep >= step.step_number:
                    raise ValueError(
                        f"Step {step.step_number} has invalid dependency on step {dep}"
                    )
                if not any(s.step_number == dep for s in steps):
                    raise ValueError(
                        f"Step {step.step_number} depends on non-existent step {dep}"
                    )

        return steps

    async def plan(self, task: str, context: str = "") -> list[PlanStep]:
        """Generate a plan for the given task.

        Args:
            task: Task to plan for.
            context: Additional context (e.g., from previous failed attempts).

        Returns:
            List of PlanStep objects representing the plan.

        Raises:
            ValueError: If planning fails or token budget exceeded.
        """
        logger.info("planning_started", task=task[:100])

        # Check token budget
        if self.total_tokens >= self.config.max_tokens:
            raise ValueError(
                f"Token budget exceeded: {self.total_tokens} >= {self.config.max_tokens}"
            )

        # Build prompt
        system_prompt = self._build_planning_system_prompt()

        prompt = f"Question: {task}\n\n"
        if context:
            prompt += f"Context from previous attempt:\n{context}\n\n"
        prompt += "Please create a detailed execution plan:"

        # Generate plan
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=self.config.temperature,
                max_tokens=min(2000, self.config.max_tokens - self.total_tokens),
            )

            # Track tokens
            if hasattr(response, "eval_count") and response.eval_count:
                self.total_tokens += response.eval_count
                self.statistics.total_tokens_used += response.eval_count
            if hasattr(response, "prompt_eval_count") and response.prompt_eval_count:
                self.total_tokens += response.prompt_eval_count
                self.statistics.total_tokens_used += response.prompt_eval_count

            plan_text = response.response

            if self.config.verbose:
                logger.info("plan_generated", plan_preview=plan_text[:500])

        except Exception as e:
            logger.error("planning_failed", error=str(e), exc_info=True)
            raise

        # Parse plan
        try:
            steps = self._parse_plan(plan_text)

            if len(steps) > self.config.max_plan_steps:
                logger.warning(
                    "plan_too_long",
                    step_count=len(steps),
                    max_steps=self.config.max_plan_steps,
                )
                steps = steps[: self.config.max_plan_steps]

            self.current_plan = steps
            self.statistics.total_steps_planned = len(steps)

            logger.info("planning_completed", step_count=len(steps))

            return steps

        except Exception as e:
            logger.error("plan_parsing_failed", error=str(e), exc_info=True)
            raise ValueError(f"Failed to parse plan: {str(e)}")

    async def execute_step(
        self, step: PlanStep, previous_results: dict[int, ExecutionResult]
    ) -> ExecutionResult:
        """Execute a single plan step.

        Args:
            step: Step to execute.
            previous_results: Results from previously executed steps.

        Returns:
            ExecutionResult from executing the step.
        """
        logger.info("step_execution_started", step_number=step.step_number)

        step.status = StepStatus.IN_PROGRESS
        tokens_before = self.total_tokens

        try:
            # Check dependencies
            for dep in step.dependencies:
                if dep not in previous_results:
                    error_msg = f"Dependency step {dep} not yet executed"
                    logger.error(
                        "dependency_not_met",
                        step_number=step.step_number,
                        missing_dep=dep,
                    )
                    return ExecutionResult(
                        step_number=step.step_number,
                        success=False,
                        output="",
                        error=error_msg,
                    )

                dep_result = previous_results[dep]
                if not dep_result.success:
                    error_msg = f"Dependency step {dep} failed"
                    logger.error(
                        "dependency_failed",
                        step_number=step.step_number,
                        failed_dep=dep,
                    )
                    return ExecutionResult(
                        step_number=step.step_number,
                        success=False,
                        output="",
                        error=error_msg,
                    )

            # Build context from previous results
            context = ""
            if previous_results:
                context = "Previous step results:\n"
                for dep in step.dependencies:
                    if dep in previous_results:
                        result = previous_results[dep]
                        context += f"Step {dep}: {result.output}\n"
                context += "\n"

            # Execute based on whether tool is specified
            if step.tool_name:
                # Tool-based execution
                if step.tool_name not in self.tools:
                    error_msg = f"Unknown tool: {step.tool_name}"
                    logger.error(
                        "unknown_tool",
                        step_number=step.step_number,
                        tool_name=step.tool_name,
                    )
                    return ExecutionResult(
                        step_number=step.step_number,
                        success=False,
                        output="",
                        error=error_msg,
                    )

                # Replace placeholders in tool input with previous results
                tool_input = step.tool_input or ""
                for dep, result in previous_results.items():
                    # Replace patterns like <result_from_step_N>
                    tool_input = tool_input.replace(
                        f"<result_from_step_{dep}>", result.output
                    )

                # Execute tool
                result = await self._execute_tool(step.tool_name, tool_input)
                self.statistics.total_tool_calls += 1

                step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

                return ExecutionResult(
                    step_number=step.step_number,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    tokens_used=self.total_tokens - tokens_before,
                )

            else:
                # Reasoning-based execution using LLM
                system_prompt = self._build_execution_system_prompt()

                prompt = f"{context}Step {step.step_number}: {step.description}\n\n"
                prompt += "Please provide the output for this step:"

                # Check token budget
                if self.total_tokens >= self.config.max_tokens:
                    raise ValueError("Token budget exceeded")

                response = await self.llm_client.generate(
                    prompt=prompt,
                    system=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=min(1000, self.config.max_tokens - self.total_tokens),
                )

                # Track tokens
                if hasattr(response, "eval_count") and response.eval_count:
                    self.total_tokens += response.eval_count
                    self.statistics.total_tokens_used += response.eval_count
                if hasattr(response, "prompt_eval_count") and response.prompt_eval_count:
                    self.total_tokens += response.prompt_eval_count
                    self.statistics.total_tokens_used += response.prompt_eval_count

                output = response.response.strip()

                step.status = StepStatus.COMPLETED

                return ExecutionResult(
                    step_number=step.step_number,
                    success=True,
                    output=output,
                    tokens_used=self.total_tokens - tokens_before,
                )

        except Exception as e:
            logger.error(
                "step_execution_failed",
                step_number=step.step_number,
                error=str(e),
                exc_info=True,
            )
            step.status = StepStatus.FAILED
            return ExecutionResult(
                step_number=step.step_number,
                success=False,
                output="",
                error=str(e),
                tokens_used=self.total_tokens - tokens_before,
            )

    async def _execute_tool(self, tool_name: str, tool_input: str) -> ExecutionResult:
        """Execute a tool and return result.

        Args:
            tool_name: Name of tool to execute.
            tool_input: Input for the tool.

        Returns:
            ExecutionResult with tool output.
        """
        tool = self.tools[tool_name]

        try:
            # Parse tool input
            input_schema = tool.input_type
            field_names = list(input_schema.model_fields.keys())

            input_data = {}
            if len(field_names) == 1:
                input_data[field_names[0]] = tool_input
            else:
                try:
                    import json
                    input_data = json.loads(tool_input)
                except (json.JSONDecodeError, ValueError):
                    if field_names:
                        input_data[field_names[0]] = tool_input

            tool_input_obj = input_schema(**input_data)

            logger.info(
                "executing_tool",
                tool_name=tool_name,
                input_preview=tool_input[:100],
            )

            result = await tool.safe_execute(tool_input_obj)

            # Extract output
            if hasattr(result, "success") and not result.success:
                error = getattr(result, "error", "Unknown error")
                logger.error("tool_execution_failed", tool_name=tool_name, error=error)
                return ExecutionResult(
                    step_number=0,  # Will be set by caller
                    success=False,
                    output="",
                    error=error,
                )

            output = ""
            if hasattr(result, "formatted"):
                output = str(result.formatted)
            elif hasattr(result, "value"):
                output = str(result.value)
            elif hasattr(result, "output"):
                output = str(result.output)
            else:
                output = str(result)

            logger.info("tool_execution_success", tool_name=tool_name)

            return ExecutionResult(
                step_number=0,  # Will be set by caller
                success=True,
                output=output,
            )

        except Exception as e:
            logger.error(
                "tool_execution_exception",
                tool_name=tool_name,
                error=str(e),
                exc_info=True,
            )
            return ExecutionResult(
                step_number=0,  # Will be set by caller
                success=False,
                output="",
                error=str(e),
            )

    async def replan(
        self, task: str, failed_step: PlanStep, error: str
    ) -> list[PlanStep]:
        """Re-plan after a step failure.

        Args:
            task: Original task.
            failed_step: The step that failed.
            error: Error message from the failure.

        Returns:
            New list of PlanStep objects.

        Raises:
            ValueError: If re-planning fails or max replans exceeded.
        """
        if self.replan_count >= self.config.max_replans:
            raise ValueError(
                f"Max replans ({self.config.max_replans}) exceeded"
            )

        self.replan_count += 1
        self.statistics.total_replans += 1

        logger.info(
            "replanning_started",
            replan_count=self.replan_count,
            failed_step=failed_step.step_number,
        )

        # Build context about the failure
        context = f"""Previous plan failed at step {failed_step.step_number}.
Step description: {failed_step.description}
Error: {error}

Successful steps so far:
"""
        for result in self.execution_results:
            if result.success:
                context += f"Step {result.step_number}: {result.output}\n"

        context += "\nPlease create a new plan that avoids this error."

        # Generate new plan
        new_plan = await self.plan(task, context)

        return new_plan

    async def run(self, task: str) -> str:
        """Run the Plan-and-Execute agent to complete a task.

        Args:
            task: Task to complete.

        Returns:
            Final result string.

        Raises:
            PlanExecuteError: If execution fails.
            ValueError: If token budget exceeded.
        """
        import time

        start_time = time.time()

        self.current_plan = []
        self.execution_results = []
        self.statistics = PlanExecuteStatistics()
        self.total_tokens = 0
        self.replan_count = 0

        logger.info("plan_execute_agent_started", task=task[:100])

        try:
            # Phase 1: Planning
            plan = await self.plan(task)

            # Phase 2: Execution
            attempt = 0
            max_attempts = self.config.max_replans + 1

            while attempt < max_attempts:
                logger.info("execution_attempt_started", attempt=attempt + 1)

                previous_results: dict[int, ExecutionResult] = {}
                execution_failed = False
                failed_step = None
                failed_error = None

                for step in plan:
                    # Stream update if enabled
                    if self.config.enable_streaming:
                        logger.info(
                            "step_started",
                            step_number=step.step_number,
                            description=step.description,
                        )

                    # Execute step
                    result = await self.execute_step(step, previous_results)
                    self.execution_results.append(result)
                    previous_results[step.step_number] = result

                    self.statistics.total_steps_executed += 1

                    if result.success:
                        self.statistics.total_steps_completed += 1
                        if self.config.verbose:
                            logger.info(
                                "step_completed",
                                step_number=step.step_number,
                                output_preview=result.output[:100],
                            )
                    else:
                        self.statistics.total_steps_failed += 1
                        logger.error(
                            "step_failed",
                            step_number=step.step_number,
                            error=result.error,
                        )

                        if self.config.stop_on_step_failure:
                            raise PlanExecuteError(
                                f"Step {step.step_number} failed: {result.error}"
                            )

                        if self.config.replan_on_failure:
                            execution_failed = True
                            failed_step = step
                            failed_error = result.error or "Unknown error"
                            break

                # Check if we need to re-plan
                if execution_failed and failed_step and self.replan_count < self.config.max_replans:
                    logger.info("attempting_replan")
                    plan = await self.replan(task, failed_step, failed_error)
                    attempt += 1
                    # Reset for new attempt
                    previous_results = {}
                    continue
                else:
                    # Execution complete or no more replans
                    break

            # Generate final answer from all successful results
            final_output = self._generate_final_answer(task, previous_results)

            # Update statistics
            end_time = time.time()
            self.statistics.execution_time_ms = (end_time - start_time) * 1000

            logger.info(
                "plan_execute_agent_completed",
                total_steps=self.statistics.total_steps_executed,
                completed_steps=self.statistics.total_steps_completed,
                failed_steps=self.statistics.total_steps_failed,
                replans=self.statistics.total_replans,
                total_tokens=self.statistics.total_tokens_used,
            )

            return final_output

        except Exception as e:
            logger.error("plan_execute_agent_failed", error=str(e), exc_info=True)
            raise PlanExecuteError(f"Agent execution failed: {str(e)}")

    def _generate_final_answer(
        self, task: str, results: dict[int, ExecutionResult]
    ) -> str:
        """Generate final answer from execution results.

        Args:
            task: Original task.
            results: Execution results by step number.

        Returns:
            Final answer string.
        """
        if not results:
            return "No results were produced."

        # Get the last successful result
        last_step = max(results.keys())
        last_result = results[last_step]

        if last_result.success:
            return last_result.output

        # If last step failed, find the last successful one
        for step_num in sorted(results.keys(), reverse=True):
            if results[step_num].success:
                return results[step_num].output

        return "All steps failed."

    def get_plan(self) -> list[PlanStep]:
        """Get the current execution plan.

        Returns:
            List of PlanStep objects.
        """
        return self.current_plan

    def get_results(self) -> list[ExecutionResult]:
        """Get all execution results.

        Returns:
            List of ExecutionResult objects.
        """
        return self.execution_results

    def get_statistics(self) -> PlanExecuteStatistics:
        """Get execution statistics.

        Returns:
            PlanExecuteStatistics object.
        """
        return self.statistics

    def reset(self) -> None:
        """Reset the agent state for a new run."""
        self.current_plan = []
        self.execution_results = []
        self.statistics = PlanExecuteStatistics()
        self.total_tokens = 0
        self.replan_count = 0
        logger.info("plan_execute_agent_reset")
