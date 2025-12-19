# Tool Specification

## Overview

Tools extend the capabilities of LLMs by providing access to external computation (calculators, code execution, search, etc.). They shift computational burden off the models and provide deterministic, verifiable results.

## Dependencies

- `pydantic>=2.0.0`
- `asyncio` (standard library)
- Docker (for sandboxed code execution)

---

## Tool Interface

### Base Tool

```python
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Type
from pydantic import BaseModel, Field

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolConfig(BaseModel):
    """Base configuration for tools."""
    timeout_ms: int = Field(default=5000, ge=100, le=60000)
    enabled: bool = Field(default=True)


class ToolMetadata(BaseModel):
    """Metadata about a tool."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = Field(
        pattern=r"^(computation|execution|search|memory|utility)$"
    )
    sandbox_required: bool = False


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all tools."""

    metadata: ToolMetadata
    input_type: Type[InputT]
    output_type: Type[OutputT]
    config: ToolConfig

    def __init__(self, config: ToolConfig | None = None):
        self.config = config or ToolConfig()

    @abstractmethod
    async def execute(self, input: InputT) -> OutputT:
        """Execute the tool with given input."""
        pass

    def get_schema_description(self) -> str:
        """Get description for LLM prompt."""
        return f"""
Tool: {self.metadata.name}
Description: {self.metadata.description}
Input: {self.input_type.model_json_schema()}
Output: {self.output_type.model_json_schema()}
"""

    async def safe_execute(self, input: InputT) -> OutputT:
        """Execute with timeout and error handling."""
        import asyncio
        try:
            result = await asyncio.wait_for(
                self.execute(input),
                timeout=self.config.timeout_ms / 1000
            )
            return result
        except asyncio.TimeoutError:
            return self.output_type(
                success=False,
                error=f"Tool timed out after {self.config.timeout_ms}ms"
            )
        except Exception as e:
            return self.output_type(
                success=False,
                error=str(e)
            )
```

---

## Built-in Tools

### Calculator

Safe mathematical expression evaluation.

```python
import math
import operator
from typing import Optional
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(
        description="Mathematical expression to evaluate",
        examples=["2 + 2", "sqrt(16)", "sin(pi/2)", "2**10"],
        max_length=1000
    )


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""
    success: bool
    value: Optional[float] = None
    formatted: Optional[str] = None  # Human-readable result
    error: Optional[str] = None


class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    """Safely evaluates mathematical expressions."""

    metadata = ToolMetadata(
        id="calculator",
        name="Calculator",
        description="Evaluates mathematical expressions safely",
        category="computation",
        sandbox_required=False,
    )
    input_type = CalculatorInput
    output_type = CalculatorOutput

    # Safe operations whitelist
    SAFE_OPERATIONS = {
        # Basic operators
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "pow": operator.pow,
        "neg": operator.neg,

        # Math functions
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,

        # Constants
        "pi": math.pi,
        "e": math.e,
    }

    async def execute(self, input: CalculatorInput) -> CalculatorOutput:
        try:
            # Parse and evaluate safely
            result = self._safe_eval(input.expression)
            return CalculatorOutput(
                success=True,
                value=float(result),
                formatted=self._format_result(result),
            )
        except ZeroDivisionError:
            return CalculatorOutput(
                success=False,
                error="Division by zero"
            )
        except ValueError as e:
            return CalculatorOutput(
                success=False,
                error=f"Math error: {e}"
            )
        except Exception as e:
            return CalculatorOutput(
                success=False,
                error=f"Invalid expression: {e}"
            )

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate expression using AST parsing."""
        import ast

        # Parse expression to AST
        tree = ast.parse(expression, mode='eval')

        # Validate all nodes are safe
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls allowed")
                if node.func.id not in self.SAFE_OPERATIONS:
                    raise ValueError(f"Unknown function: {node.func.id}")

        # Compile and evaluate
        code = compile(tree, "<expression>", "eval")
        return eval(code, {"__builtins__": {}}, self.SAFE_OPERATIONS)

    def _format_result(self, result: float) -> str:
        """Format result for human readability."""
        if result == int(result):
            return str(int(result))
        return f"{result:.10g}"
```

**YAML Configuration:**
```yaml
# config/tools.yaml
tools:
  calculator:
    id: calculator
    enabled: true
    timeout_ms: 1000
```

---

### Code Executor

Sandboxed Python code execution.

```python
from typing import Optional, List
from pydantic import BaseModel, Field


class CodeExecutorInput(BaseModel):
    """Input for code executor tool."""
    code: str = Field(
        description="Python code to execute",
        max_length=50000
    )
    language: str = Field(
        default="python",
        pattern=r"^(python|javascript|bash)$"
    )
    timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=30
    )


class CodeExecutorOutput(BaseModel):
    """Output from code executor tool."""
    success: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class CodeExecutorTool(BaseTool[CodeExecutorInput, CodeExecutorOutput]):
    """Executes code in a sandboxed environment."""

    metadata = ToolMetadata(
        id="code_executor",
        name="Code Executor",
        description="Executes code in a secure sandbox",
        category="execution",
        sandbox_required=True,
    )
    input_type = CodeExecutorInput
    output_type = CodeExecutorOutput

    async def execute(self, input: CodeExecutorInput) -> CodeExecutorOutput:
        """Execute code in sandbox."""
        import asyncio
        import time

        start = time.perf_counter()

        try:
            if input.language == "python":
                result = await self._execute_python(input.code, input.timeout_seconds)
            else:
                return CodeExecutorOutput(
                    success=False,
                    error=f"Unsupported language: {input.language}"
                )

            elapsed = int((time.perf_counter() - start) * 1000)
            return CodeExecutorOutput(
                success=result["exit_code"] == 0,
                stdout=result["stdout"],
                stderr=result["stderr"],
                exit_code=result["exit_code"],
                execution_time_ms=elapsed,
            )

        except asyncio.TimeoutError:
            return CodeExecutorOutput(
                success=False,
                error=f"Execution timed out after {input.timeout_seconds}s"
            )
        except Exception as e:
            return CodeExecutorOutput(
                success=False,
                error=str(e)
            )

    async def _execute_python(self, code: str, timeout: int) -> dict:
        """Execute Python code in Docker sandbox."""
        import asyncio

        # Docker command with restrictions
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--network", "none",  # No network
            "--memory", "256m",   # Memory limit
            "--cpus", "1",        # CPU limit
            "--read-only",        # Read-only filesystem
            "--user", "nobody",   # Non-root user
            "python:3.11-slim",
            "python", "-c", code
        ]

        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )

        return {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "exit_code": proc.returncode,
        }
```

**YAML Configuration:**
```yaml
tools:
  code_executor:
    id: code_executor
    enabled: true
    timeout_ms: 30000
    sandbox:
      type: docker
      image: python:3.11-slim
      memory_limit: 256m
      network: none
      read_only: true
```

---

### Web Search

Web search capability (requires external API).

```python
from typing import Optional, List
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(
        description="Search query",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=10
    )


class SearchResult(BaseModel):
    """Single search result."""
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    """Output from web search tool."""
    success: bool
    results: List[SearchResult] = Field(default_factory=list)
    error: Optional[str] = None


class WebSearchTool(BaseTool[WebSearchInput, WebSearchOutput]):
    """Searches the web for information."""

    metadata = ToolMetadata(
        id="web_search",
        name="Web Search",
        description="Searches the web and returns relevant results",
        category="search",
        sandbox_required=False,
    )
    input_type = WebSearchInput
    output_type = WebSearchOutput

    async def execute(self, input: WebSearchInput) -> WebSearchOutput:
        # Implementation depends on search API (SearXNG, DuckDuckGo, etc.)
        # Placeholder for now
        return WebSearchOutput(
            success=False,
            error="Web search not yet implemented"
        )
```

---

## Tool Registry

```python
from typing import Dict, Type


class ToolRegistry:
    """Registry for available tools."""

    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """Register a tool instance."""
        cls._tools[tool.metadata.id] = tool

    @classmethod
    def get(cls, tool_id: str) -> BaseTool | None:
        """Get a tool by ID."""
        return cls._tools.get(tool_id)

    @classmethod
    def list_tools(cls) -> list[ToolMetadata]:
        """List all available tools."""
        return [t.metadata for t in cls._tools.values()]

    @classmethod
    def get_tool_descriptions(cls) -> str:
        """Get descriptions for all tools (for prompts)."""
        return "\n\n".join(
            tool.get_schema_description()
            for tool in cls._tools.values()
            if tool.config.enabled
        )
```

---

## Tool Invocation Format

LLMs should output tool calls in this format:

```json
{
  "tool_call": {
    "tool_id": "calculator",
    "input": {
      "expression": "2 + 2"
    }
  }
}
```

Or for multiple calls:

```json
{
  "tool_calls": [
    {"tool_id": "calculator", "input": {"expression": "sqrt(16)"}},
    {"tool_id": "calculator", "input": {"expression": "2**10"}}
  ]
}
```

---

## File Locations

| Component | File |
|-----------|------|
| Base class | `src/tinyllm/tools/base.py` |
| Registry | `src/tinyllm/tools/registry.py` |
| Calculator | `src/tinyllm/tools/calculator.py` |
| Code Executor | `src/tinyllm/tools/code_executor.py` |
| Web Search | `src/tinyllm/tools/web_search.py` |

---

## Test Cases

### Calculator Tests

| Input | Expected Output |
|-------|-----------------|
| `2 + 2` | `{success: true, value: 4}` |
| `sqrt(16)` | `{success: true, value: 4}` |
| `sin(0)` | `{success: true, value: 0}` |
| `1 / 0` | `{success: false, error: "Division by zero"}` |
| `import os` | `{success: false, error: "Invalid expression"}` |
| `__import__('os')` | `{success: false, error: "Invalid expression"}` |

### Code Executor Tests

| Input | Expected Output |
|-------|-----------------|
| `print("hello")` | `{success: true, stdout: "hello\n"}` |
| `1/0` | `{success: false, stderr: "ZeroDivisionError"}` |
| `while True: pass` | `{success: false, error: "timed out"}` |
| `import socket` | `{success: true}` (but network blocked) |
