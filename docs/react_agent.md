# ReAct Agent Framework

The ReAct (Reasoning + Acting) agent implements the ReAct pattern from the paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).

## Overview

The ReAct agent follows an iterative loop:

1. **Thought**: The LLM reasons about what to do next
2. **Action**: The LLM decides on a tool and provides input
3. **Observation**: The tool executes and returns a result
4. **Repeat**: Continue until a final answer is reached

This pattern allows the LLM to solve complex problems by breaking them down into steps, using tools, and reasoning about the results.

## Installation

The ReAct agent is included in TinyLLM:

```python
from tinyllm.agents import ReActAgent, ReActConfig
```

## Quick Start

```python
import asyncio
from tinyllm.agents import ReActAgent, ReActConfig
from tinyllm.models.client import OllamaClient
from tinyllm.tools.calculator import CalculatorTool

async def main():
    # Create LLM client
    client = OllamaClient(default_model="qwen2.5:0.5b")

    # Create agent
    agent = ReActAgent(
        llm_client=client,
        config=ReActConfig(max_iterations=5)
    )

    # Register tools
    agent.register_tool("calculator", CalculatorTool())

    # Run the agent
    result = await agent.run("What is the square root of 144?")
    print(result)

    # View the reasoning trace
    print(agent.get_trace_string())

    await client.close()

asyncio.run(main())
```

## Configuration

The `ReActConfig` class provides comprehensive configuration:

```python
config = ReActConfig(
    max_iterations=10,        # Maximum reasoning loops
    max_tokens=50000,         # Total token budget
    temperature=0.0,          # LLM temperature
    stop_on_error=False,      # Stop on first tool error
    verbose=True,             # Enable detailed logging
)
```

### Configuration Parameters

- **max_iterations** (default: 10): Maximum number of think-act-observe cycles before stopping
- **max_tokens** (default: 50000): Total token budget across all LLM calls
- **temperature** (default: 0.0): Controls LLM randomness (0.0 = deterministic)
- **stop_on_error** (default: False): Whether to stop on first tool error or continue
- **verbose** (default: True): Enable detailed structured logging

## Tool Registration

### Using Built-in Tools

```python
from tinyllm.tools import CalculatorTool, CodeExecutorTool

agent.register_tool("calculator", CalculatorTool())
agent.register_tool("python", CodeExecutorTool())
```

### Registering Custom Functions

For simple tools, you can register Python functions directly:

```python
def search(query: str) -> str:
    """Search for information."""
    # Your search implementation
    return f"Results for: {query}"

agent.register_function(
    "search",
    search,
    "Searches for information on the web"
)
```

### Creating Custom Tools

For more complex tools, implement the `BaseTool` interface:

```python
from pydantic import BaseModel, Field
from tinyllm.tools.base import BaseTool, ToolMetadata

class MyToolInput(BaseModel):
    query: str = Field(description="Search query")

class MyToolOutput(BaseModel):
    success: bool
    result: str
    error: str | None = None

class MyTool(BaseTool[MyToolInput, MyToolOutput]):
    metadata = ToolMetadata(
        id="my_tool",
        name="My Tool",
        description="Does something useful",
        category="utility",
    )
    input_type = MyToolInput
    output_type = MyToolOutput

    async def execute(self, input: MyToolInput) -> MyToolOutput:
        try:
            result = self.do_work(input.query)
            return MyToolOutput(success=True, result=result)
        except Exception as e:
            return MyToolOutput(success=False, result="", error=str(e))

agent.register_tool("my_tool", MyTool())
```

## Response Format

The agent expects LLM responses in this format:

```
Thought: [Reasoning about what to do]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [Filled by system]

Thought: [Further reasoning]
Action: [another_tool]
Action Input: [more input]
Observation: [Filled by system]

Thought: [Final reasoning]
Final Answer: [Complete answer to the question]
```

The agent uses regex patterns to parse these components. You can customize the patterns in `ReActConfig`:

```python
config = ReActConfig(
    thought_pattern=r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)",
    action_pattern=r"Action:\s*(\w+)",
    action_input_pattern=r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\nFinal Answer:|$)",
    final_answer_pattern=r"Final Answer:\s*(.+?)$",
)
```

## Reasoning Trace

After running the agent, you can inspect the reasoning trace:

```python
# Get structured trace
trace = agent.get_trace()
for step in trace:
    print(f"{step.step_type}: {step.content}")

# Get formatted string
trace_str = agent.get_trace_string()
print(trace_str)
```

The trace contains steps of type:
- `ReActStepType.THOUGHT`: Reasoning steps
- `ReActStepType.ACTION`: Tool invocations
- `ReActStepType.OBSERVATION`: Tool results
- `ReActStepType.FINAL_ANSWER`: Final answer

## Error Handling

The agent provides robust error handling:

### Soft Error Handling (Default)

By default, tool errors become observations that the LLM can reason about:

```python
config = ReActConfig(stop_on_error=False)
agent = ReActAgent(llm_client=client, config=config)

# If a tool fails, the agent sees:
# Observation: Error: [error message]
# And can reason about how to proceed
```

### Hard Error Handling

Stop immediately on any tool error:

```python
config = ReActConfig(stop_on_error=True)
agent = ReActAgent(llm_client=client, config=config)

try:
    result = await agent.run(question)
except ToolExecutionError as e:
    print(f"Tool failed: {e}")
```

### Iteration and Token Limits

The agent will raise `ValueError` if limits are exceeded:

```python
try:
    result = await agent.run(question)
except ValueError as e:
    if "Max iterations" in str(e):
        print("Agent ran out of iterations")
    elif "Token budget exceeded" in str(e):
        print(f"Used too many tokens: {agent.total_tokens}")
```

## Advanced Usage

### Token Budget Management

Track and limit token usage:

```python
config = ReActConfig(max_tokens=5000)
agent = ReActAgent(llm_client=client, config=config)

result = await agent.run(question)
print(f"Tokens used: {agent.total_tokens}")
```

### Resetting Agent State

Reset the agent between runs:

```python
result1 = await agent.run("First question")
agent.reset()  # Clear trace and token count
result2 = await agent.run("Second question")
```

### Custom Parsing

The agent provides access to its parsing logic:

```python
thought, action, action_input, final_answer = agent._parse_response(llm_output)
```

## LLM Client Requirements

The agent works with any LLM client that implements:

```python
async def generate(
    self,
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Response:
    """Generate a response.

    Returns:
        Object with .response attribute containing generated text
        Optional: .eval_count and .prompt_eval_count for token tracking
    """
```

The TinyLLM `OllamaClient` implements this interface. You can also use:
- Any client with a compatible `generate()` method
- Wrapped OpenAI/Anthropic clients
- Mock clients for testing

## Examples

See `/home/uri/Desktop/tinyllm/examples/react_agent_example.py` for complete examples:

1. **Basic Calculator Usage**: Simple math problems
2. **Custom Tool Registration**: Using custom functions
3. **Multi-Step Reasoning**: Complex problems requiring multiple tools
4. **Error Handling**: Dealing with tool failures
5. **Token Budget Management**: Limiting resource usage

## Testing

The ReAct agent includes comprehensive tests in `tests/unit/test_react_agent.py`:

```bash
# Note: Tests require fixing pytest-asyncio version compatibility
PYTHONPATH=src python3 -m pytest tests/unit/test_react_agent.py -v
```

Tests cover:
- Configuration validation
- Tool registration and execution
- Response parsing
- Error handling
- Token budget enforcement
- Trace generation
- Integration scenarios

## Structured Logging

The agent uses structured logging with `structlog`:

```python
# All agent operations are logged
[info] react_agent_started question='What is 2+2?'
[info] react_iteration_start iteration=1
[info] executing_tool tool_name='calculator' input_preview='2+2'
[info] tool_execution_success output_preview='4'
[info] react_agent_completed iterations=2 total_tokens=300
```

Configure logging:

```python
from tinyllm.logging import configure_logging

configure_logging(log_level="INFO", log_format="console")
```

## Performance Considerations

1. **Token Usage**: Each iteration consumes tokens. Use `max_tokens` to control costs.
2. **Iteration Limits**: Set appropriate `max_iterations` based on problem complexity.
3. **Temperature**: Lower temperature (0.0-0.3) gives more consistent reasoning.
4. **Tool Efficiency**: Optimize tool execution time as it affects overall latency.

## Comparison with Other Patterns

| Pattern | When to Use |
|---------|-------------|
| **ReAct** | Need step-by-step reasoning with tool access |
| **Chain-of-Thought** | Pure reasoning without external tools |
| **Function Calling** | Single tool invocation per query |
| **Planning** | Need upfront plan before execution |

## Limitations

1. **LLM Dependency**: Quality depends on the LLM's ability to follow format
2. **Tool Complexity**: Complex tool outputs may confuse the LLM
3. **Iteration Limits**: Very complex problems may need many iterations
4. **Format Sensitivity**: LLM must produce exact format for parsing

## Future Enhancements

Potential improvements:
- Memory/context management for long traces
- Parallel tool execution
- Tool recommendation/suggestion
- Automatic tool description generation
- Learning from failed attempts
- Multi-agent ReAct collaboration

## References

- [ReAct Paper](https://arxiv.org/abs/2210.03629): Original research paper
- [TinyLLM Documentation](https://github.com/ndjstn/tinyllm): Main project docs
- [Tool System](./tools.md): Tool development guide

## Support

For issues or questions:
- GitHub Issues: https://github.com/ndjstn/tinyllm/issues
- Documentation: https://github.com/ndjstn/tinyllm#readme
