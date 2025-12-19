# Contributing to TinyLLM

Thank you for your interest in contributing to TinyLLM! This project is designed for **parallel, atomic contributions** - you can pick up a single issue and complete it independently.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Getting Started](#getting-started)
3. [Finding Issues](#finding-issues)
4. [Contribution Types](#contribution-types)
5. [Development Workflow](#development-workflow)
6. [Code Standards](#code-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Pull Request Process](#pull-request-process)
10. [Community](#community)

---

## Philosophy

### Atomic Issues

Every issue in this repository is designed to be:

- **Self-contained**: All context needed is in the issue
- **Completable in one session**: 1-4 hours of work
- **Independently testable**: Clear acceptance criteria
- **Well-specified**: Exact files to create/modify

This means you can:
- Pick up any issue without deep project knowledge
- Use AI assistants (Claude, GPT) to help implement
- Work in parallel with other contributors without conflicts

### AI-Assisted Development

We embrace AI-assisted development. Feel free to:
- Use Claude, GPT, or other AI tools to help implement features
- Ask AI to explain specifications or suggest approaches
- Generate boilerplate code with AI assistance

Just make sure you:
- Understand what the code does
- Test it properly
- Follow our code standards

---

## Getting Started

### Prerequisites

```bash
# Required
- Python 3.11+
- Git
- Ollama (https://ollama.ai)

# Recommended
- Docker (for sandbox testing)
- gh CLI (for issue management)
```

### Setup

```bash
# Fork and clone
gh repo fork ndjstn/tinyllm --clone
cd tinyllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Pull test models
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b

# Run tests to verify setup
pytest tests/
```

---

## Finding Issues

### Issue Labels

| Label | Meaning |
|-------|---------|
| `good-first-issue` | Great for newcomers |
| `help-wanted` | We need contributors |
| `phase-0` through `phase-5` | Implementation phase |
| `python` | Requires Python skills |
| `yaml` | Configuration work |
| `prompt-engineering` | Prompt writing |
| `testing` | Test cases |
| `documentation` | Docs work |
| `research` | Analysis/benchmarking |
| `bug` | Something broken |
| `enhancement` | New feature |

### Finding Your First Issue

```bash
# List good first issues
gh issue list --label "good-first-issue"

# List issues needing help
gh issue list --label "help-wanted"

# List issues for a specific skill
gh issue list --label "python"
gh issue list --label "prompt-engineering"

# List issues by phase
gh issue list --label "phase-0"
```

### Claiming an Issue

```bash
# Comment on the issue to claim it
gh issue comment <number> --body "I'd like to work on this!"

# The issue will be assigned to you
```

---

## Contribution Types

### 1. Code (Python)

Core implementation work:

```
src/tinyllm/
├── config/     # Configuration loading
├── core/       # Nodes, graphs, executor
├── models/     # Ollama client
├── tools/      # Calculator, code executor
├── grading/    # Evaluation system
├── expansion/  # Self-improvement
└── memory/     # STM/LTM
```

**Example Issue:**
> **Implement Calculator Tool**
>
> Create `src/tinyllm/tools/calculator.py` that:
> - Safely evaluates math expressions
> - Returns structured result
> - Handles errors gracefully
>
> Files: `src/tinyllm/tools/calculator.py`, `tests/unit/tools/test_calculator.py`
>
> Acceptance: All tests pass, mypy clean

### 2. Prompts (YAML)

Write and refine prompts:

```
prompts/
├── routing/      # Classification prompts
├── specialists/  # Task-specific prompts
├── thinking/     # Chain-of-thought, etc.
├── tools/        # Tool usage formats
├── grading/      # Evaluation prompts
└── meta/         # Self-improvement prompts
```

**Example Issue:**
> **Write Math Solver Prompt**
>
> Create `prompts/specialists/math_solver.yaml` that:
> - Solves arithmetic and algebra
> - Uses calculator tool when appropriate
> - Shows work step by step
>
> See [Prompt Specification](specs/prompts.md)

### 3. Testing

Write test cases:

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── seed/           # Gold standard test cases
    ├── routing/
    ├── code/
    ├── math/
    └── ...
```

**Example Issue:**
> **Add Math Routing Test Cases**
>
> Create 10 test cases in `tests/seed/routing/math.yaml`:
> - Simple arithmetic (3 cases)
> - Word problems (3 cases)
> - Algebra (2 cases)
> - Edge cases (2 cases)
>
> Format: See [Test Specification](specs/tests.md)

### 4. Configuration (YAML)

Define system configuration:

```
config/
├── models.yaml
├── tools.yaml
└── ...

graphs/
└── main.v1.0.yaml
```

### 5. Documentation

Improve docs:

```
docs/
├── concepts/   # Explanatory guides
├── specs/      # Technical specifications
├── guides/     # How-to guides
└── adr/        # Architecture decisions
```

---

## Development Workflow

### Branch Naming

```
<type>/<issue-number>-<short-description>

Examples:
feat/42-calculator-tool
fix/57-routing-timeout
docs/23-prompt-spec
test/89-math-cases
```

### Commit Messages

```
<type>(<scope>): <description>

[optional body]

[optional footer with issue reference]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

Examples:
```
feat(tools): implement calculator tool

- Add safe expression evaluation
- Add input validation
- Add comprehensive tests

Closes #42
```

### Workflow

```bash
# 1. Create branch
git checkout -b feat/42-calculator-tool

# 2. Make changes
# ... edit files ...

# 3. Run checks
ruff check src/ tests/
mypy src/
pytest tests/

# 4. Commit
git add .
git commit -m "feat(tools): implement calculator tool"

# 5. Push
git push -u origin feat/42-calculator-tool

# 6. Create PR
gh pr create --fill
```

---

## Code Standards

### Python Style

We use:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pydantic** for data validation

```python
# Good
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""

    expression: str = Field(
        description="Mathematical expression to evaluate",
        examples=["2 + 2", "sqrt(16)", "sin(pi/2)"],
    )


async def calculate(input: CalculatorInput) -> CalculatorOutput:
    """Evaluate a mathematical expression safely."""
    # Implementation
    ...
```

### File Structure

```python
"""Module docstring explaining purpose."""

# Standard library imports
from typing import Optional

# Third-party imports
from pydantic import BaseModel

# Local imports
from tinyllm.core import Message

# Constants
MAX_EXPRESSION_LENGTH = 1000

# Classes/Functions
class MyClass:
    ...

# Main execution (if applicable)
if __name__ == "__main__":
    ...
```

### Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator


class NodeConfig(BaseModel):
    """Configuration for a graph node."""

    model_config = {"extra": "forbid"}  # Catch typos

    id: str = Field(description="Unique node identifier")
    name: str = Field(description="Human-readable name")
    timeout_ms: int = Field(default=5000, ge=100, le=60000)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.replace("_", "").replace(".", "").isalnum():
            raise ValueError("ID must be alphanumeric with _ and .")
        return v
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/tools/test_calculator.py

# With coverage
pytest --cov=tinyllm tests/

# Only fast tests (no LLM calls)
pytest -m "not slow" tests/
```

### Writing Tests

```python
import pytest
from tinyllm.tools.calculator import Calculator, CalculatorInput


class TestCalculator:
    """Tests for calculator tool."""

    @pytest.fixture
    def calculator(self):
        return Calculator()

    def test_basic_arithmetic(self, calculator):
        """Test simple math operations."""
        result = calculator.execute(CalculatorInput(expression="2 + 2"))
        assert result.value == 4
        assert result.error is None

    def test_invalid_expression(self, calculator):
        """Test error handling for invalid input."""
        result = calculator.execute(CalculatorInput(expression="2 +"))
        assert result.error is not None

    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("1 + 1", 2),
            ("10 - 5", 5),
            ("3 * 4", 12),
            ("10 / 2", 5),
        ],
    )
    def test_operations(self, calculator, expr, expected):
        """Test various operations."""
        result = calculator.execute(CalculatorInput(expression=expr))
        assert result.value == expected
```

### Test Categories

| Marker | Purpose | Speed |
|--------|---------|-------|
| (none) | Unit tests | Fast |
| `@pytest.mark.integration` | Integration tests | Medium |
| `@pytest.mark.slow` | LLM-based tests | Slow |
| `@pytest.mark.e2e` | End-to-end tests | Slow |

---

## Documentation

### Writing Specs

Specifications must be complete enough for AI implementation:

```markdown
# Component: Calculator Tool

## Overview
Safely evaluates mathematical expressions.

## Interface

### Input
```python
class CalculatorInput(BaseModel):
    expression: str  # Math expression like "2 + 2"
```

### Output
```python
class CalculatorOutput(BaseModel):
    value: Optional[float]
    error: Optional[str]
```

## Behavior

### Happy Path
1. Parse expression
2. Validate safety (no code execution)
3. Evaluate with math library
4. Return result

### Error Cases
- Invalid syntax → error message
- Division by zero → error message
- Overflow → error message

## Test Cases
| Input | Expected Output |
|-------|-----------------|
| "2 + 2" | 4.0 |
| "sqrt(16)" | 4.0 |
| "2 +" | error: "Invalid syntax" |
```

### ADRs (Architecture Decision Records)

For significant decisions:

```markdown
# ADR-001: Use Pydantic for Configuration

## Status
Accepted

## Context
Need a way to validate configuration files.

## Decision
Use Pydantic v2 with YAML loading.

## Consequences
- Type safety for all config
- Good error messages
- Requires Pydantic knowledge
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`pytest tests/`)
- [ ] Linting passes (`ruff check src/ tests/`)
- [ ] Types check (`mypy src/`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention

### PR Template

PRs automatically use our template. Fill in:

- **What**: Brief description
- **Why**: Link to issue
- **How**: Implementation approach
- **Testing**: What tests were added

### Review Process

1. Automated checks run (CI)
2. Maintainer reviews code
3. Feedback addressed
4. Approval and merge

---

## Community

### Getting Help

- **Issues**: Ask questions on GitHub issues
- **Discussions**: Use GitHub Discussions for broader topics

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` - All contributors listed
- Release notes - Significant contributions highlighted
- README badges - Top contributors

### Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build something cool.

---

## Quick Reference

```bash
# Setup
gh repo fork ndjstn/tinyllm --clone
pip install -e ".[dev]"
pre-commit install

# Find work
gh issue list --label "good-first-issue"

# Develop
git checkout -b feat/42-feature
# ... make changes ...
ruff check src/ tests/
mypy src/
pytest tests/

# Submit
git commit -m "feat(scope): description"
git push -u origin feat/42-feature
gh pr create --fill
```

**Thank you for contributing!**
