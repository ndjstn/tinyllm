# TinyLLM Development Guide

> **Local-First Philosophy**: TinyLLM runs 100% offline using small, local models. No cloud APIs, no data leaves your machine.

## Quick Start

```bash
# 1. Install prerequisites
# Python 3.11+, uv, and Ollama required

# 2. Clone and install
git clone https://github.com/ndjstn/tinyllm.git
cd tinyllm
uv sync --dev

# 3. Pull local models
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b

# 4. Set up pre-commit hooks
.venv/bin/pre-commit install

# 5. Verify setup
uv run tinyllm doctor

# 6. Run tests
make test
```

## Prerequisites

### Required
- **Python 3.11+**: Modern Python with async/await support
- **[uv](https://github.com/astral-sh/uv)**: Fast Python package manager (`pip install uv`)
- **[Ollama](https://ollama.ai)**: Local LLM inference engine

### Recommended
- **16GB+ RAM**: For running multiple local models
- **8GB+ VRAM** (optional): GPU acceleration for faster inference
- **50GB disk space**: For model storage

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ndjstn/tinyllm.git
cd tinyllm
```

### 2. Install Dependencies

```bash
# Install all dependencies (including dev tools)
uv sync --dev

# Or install with optional extras
uv sync --dev --extras data        # CSV/JSON processing tools
uv sync --dev --extras all-tools   # All optional tools
```

### 3. Set Up Pre-commit Hooks

```bash
# Install hooks
.venv/bin/pre-commit install

# Test hooks (optional)
.venv/bin/pre-commit run --all-files
```

### 4. Pull Local Models

```bash
# Router (fast, small decisions)
ollama pull qwen2.5:0.5b     # 500MB

# General specialist
ollama pull qwen2.5:3b       # 1.9GB

# Code specialist
ollama pull granite-code:3b  # 1.9GB

# Verify models
ollama list
```

### 5. Verify Installation

```bash
# Run the doctor command
uv run tinyllm doctor

# Run a quick test
uv run tinyllm run "What is 2 + 2?"
```

## Project Structure

```
tinyllm/
├── src/tinyllm/
│   ├── core/              # Core engine (graph, executor, nodes)
│   │   ├── graph.py       # Graph definition and traversal
│   │   ├── executor.py    # Graph execution engine
│   │   ├── context.py     # Execution context
│   │   └── message.py     # Message passing
│   ├── config/            # Configuration models
│   │   └── loader.py      # Config loading and validation
│   ├── models/            # LLM clients (Ollama)
│   │   └── ollama.py      # Ollama client implementation
│   ├── nodes/             # Node implementations
│   │   ├── entry.py       # Entry nodes
│   │   ├── exit.py        # Exit nodes
│   │   ├── router.py      # Routing nodes
│   │   ├── model.py       # Model inference nodes
│   │   └── tool.py        # Tool execution nodes
│   ├── prompts/           # Prompt templates
│   │   └── loader.py      # Prompt loading
│   ├── tools/             # Tool implementations
│   │   ├── calculator.py  # Math calculations
│   │   ├── code_executor.py  # Code execution
│   │   ├── csv_tool.py    # CSV processing
│   │   ├── json_tool.py   # JSON processing
│   │   └── ...
│   ├── cli.py             # Command-line interface
│   ├── logging.py         # Structured logging
│   └── metrics.py         # Prometheus metrics
├── graphs/                # Graph YAML definitions
├── prompts/               # Prompt YAML files
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── conftest.py        # Pytest configuration
├── docs/                  # Documentation
├── docker/                # Docker configurations
├── .github/               # CI/CD workflows
├── Makefile               # Development commands
└── pyproject.toml         # Project configuration
```

## Development Workflow

### Running Tests

```bash
# All tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests (requires Ollama running)
make test-integration

# With coverage
make test-cov

# Coverage gate (requires >=80%)
make test-cov-gate

# Specific test file
.venv/bin/python -m pytest tests/unit/test_calculator.py -v

# Specific test
.venv/bin/python -m pytest tests/unit/test_calculator.py::test_addition -v

# Parallel execution (faster)
.venv/bin/python -m pytest tests/unit/ -n auto
```

### Code Quality

#### Formatting and Linting

```bash
# Format code (auto-fix)
.venv/bin/ruff format .

# Lint code
.venv/bin/ruff check .

# Auto-fix linting issues
.venv/bin/ruff check --fix .
```

#### Type Checking

```bash
# Type check the codebase
.venv/bin/mypy src/tinyllm/

# Type check specific module
.venv/bin/mypy src/tinyllm/core/
```

#### Pre-commit (runs all checks)

```bash
# Run all checks on staged files
.venv/bin/pre-commit run

# Run all checks on all files
.venv/bin/pre-commit run --all-files
```

### Running the CLI

```bash
# Initialize configuration
uv run tinyllm init

# Run a query
uv run tinyllm run "Your query here"

# Run with trace output
uv run tinyllm run --trace "Your query"

# Interactive chat mode
uv run tinyllm chat

# Agent mode with tools
uv run tinyllm chat --agent

# Use specific graph
uv run tinyllm run --graph multi_domain "Your query"

# Doctor (health check)
uv run tinyllm doctor
```

## Common Development Tasks

### Adding a New Node

1. Create the node class in `src/tinyllm/nodes/`
2. Inherit from `BaseNode`
3. Implement the `execute()` method
4. Add tests in `tests/unit/test_<node_name>.py`
5. Export from `src/tinyllm/nodes/__init__.py`

Example:

```python
from tinyllm.nodes.base import BaseNode
from tinyllm.core.message import Message
from tinyllm.core.context import ExecutionContext

class MyNode(BaseNode):
    async def execute(
        self,
        message: Message,
        context: ExecutionContext
    ) -> Message:
        # Your node logic here
        return message
```

### Adding a New Tool

1. Create the tool in `src/tinyllm/tools/`
2. Inherit from `BaseTool`
3. Define `metadata`, `input_type`, `output_type`
4. Implement the `execute()` method
5. Add tests in `tests/unit/test_<tool_name>.py`
6. Export from `src/tinyllm/tools/__init__.py`

Example:

```python
from pydantic import BaseModel
from tinyllm.tools.base import BaseTool, ToolMetadata

class MyInput(BaseModel):
    value: str

class MyOutput(BaseModel):
    result: str
    success: bool

class MyTool(BaseTool[MyInput, MyOutput]):
    metadata = ToolMetadata(
        id="my_tool",
        name="My Tool",
        description="Does something useful",
        category="utility",
    )
    input_type = MyInput
    output_type = MyOutput

    async def execute(self, input: MyInput) -> MyOutput:
        # Your tool logic here
        return MyOutput(result=input.value, success=True)
```

### Modifying Prompts

1. Edit the YAML file in `prompts/`
2. Test with `uv run tinyllm run` to verify changes
3. Add tests if changing behavior

Prompt structure:

```yaml
system: |
  You are a helpful assistant.

user: |
  {{ query }}

examples:
  - query: "What is 2+2?"
    response: "4"
```

### Adding a Graph

1. Create YAML definition in `graphs/`
2. Define nodes, edges, entry/exit points
3. Test the graph with `uv run tinyllm run --graph your_graph`

Graph structure:

```yaml
id: my_graph
version: "1.0"
name: My Custom Graph

nodes:
  - id: entry
    type: entry
    config: {}

  - id: router
    type: router
    config:
      model: qwen2.5:0.5b

  # ... more nodes

edges:
  - from_node: entry
    to_node: router
  # ... more edges

entry_points: [entry]
exit_points: [exit]
```

## Troubleshooting

### Test Failures

**Metrics State Pollution**:
```python
# Use the isolated_metrics_collector fixture
@pytest.fixture(autouse=True)
def metrics_collector(isolated_metrics_collector):
    return isolated_metrics_collector
```

**Async Test Issues**:
```python
# Mark async tests properly
@pytest.mark.asyncio
async def test_my_async_function():
    result = await my_async_function()
    assert result is not None
```

### Model Loading Issues

**Model not found**:
```bash
# Check available models
ollama list

# Pull the missing model
ollama pull qwen2.5:3b
```

**Ollama not running**:
```bash
# Start Ollama service
# macOS/Linux: Run Ollama app or `ollama serve`
# Check status
curl http://localhost:11434/api/tags
```

### Import Errors

**ModuleNotFoundError**:
```bash
# Reinstall in development mode
uv sync --dev

# Or use pip
.venv/bin/pip install -e ".[dev]"
```

### Coverage Too Low

```bash
# See uncovered lines
make test-cov

# Focus on specific module
.venv/bin/python -m pytest tests/unit/test_mymodule.py --cov=src/tinyllm/mymodule --cov-report=term-missing
```

## Project Philosophy

### Local-Only
- **No cloud APIs**: Everything runs on your machine
- **No telemetry**: Your data never leaves localhost
- **Offline capable**: Works without internet after setup

### Small & Fast
- **≤3B parameter models**: Fast inference on consumer hardware
- **Tool-augmented**: Let tools handle what they're good at
- **Efficient routing**: Tiny models route to specialists

### Developer Experience
- **Atomic issues**: 1-4 hour tasks, easy to contribute
- **Well-tested**: 80%+ coverage requirement
- **Documented**: Every public API has docstrings

## Tips & Best Practices

1. **Test First**: Write tests before implementation
2. **Keep it Simple**: Avoid over-engineering
3. **Follow Patterns**: Look at existing code for examples
4. **Use Type Hints**: Makes code self-documenting
5. **Document Why**: Explain non-obvious decisions
6. **Commit Often**: Small, atomic commits
7. **Run Pre-commit**: Before pushing

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/ndjstn/tinyllm/issues
- **Contributing**: See `docs/CONTRIBUTING.md`
- **Roadmap**: See `ROADMAP_500.md`

## Next Steps

- Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
- Check [CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines
- Browse [ROADMAP_500.md](ROADMAP_500.md) for future tasks
- Look at existing tests for examples

**Happy coding! 🚀**
