# Prompt Specification

## Overview

Prompts are the instructions given to LLMs. They are stored as YAML files and loaded at runtime, allowing iteration without code changes.

## Dependencies

- `pyyaml>=6.0.0`
- `pydantic>=2.0.0`
- Jinja2 (for template rendering)

---

## Prompt Schema

### YAML Format

```yaml
# prompts/<category>/<name>.yaml

id: category.name.v1              # Unique identifier
name: "Human Readable Name"       # Display name
version: "1.0.0"                  # Semantic version
category: ROUTING                 # Category enum

# Documentation
description: |
  What this prompt does and when to use it.

# Model compatibility
compatible_models:
  - qwen2.5:0.5b
  - qwen2.5:3b
  - tinyllama

# Generation parameters
temperature: 0.1                  # 0.0 - 2.0
max_tokens: 500                   # Max output tokens
top_p: 0.9                        # Nucleus sampling

# Output format
output_format: JSON_SCHEMA        # TEXT, JSON, JSON_SCHEMA, STRUCTURED
output_schema:                    # Required if JSON_SCHEMA
  type: object
  properties:
    category:
      type: string
      enum: [code, math, factual, reasoning, creative]
    confidence:
      type: number
      minimum: 0
      maximum: 1
  required: [category, confidence]

# Prompt content
system_prompt: |
  You are a task classifier. Your ONLY job is to categorize incoming tasks.
  Respond with ONLY a JSON object. No explanation, no extra text.

user_template: |
  Classify this task:
  {{task_content}}

  Output JSON: {"category": "<category>", "confidence": <0.0-1.0>}

# Optional: Few-shot examples
examples:
  - input: "Write a Python function to sort a list"
    output: '{"category": "code", "confidence": 0.95}'
  - input: "What is 15% of 80?"
    output: '{"category": "math", "confidence": 0.9}'
```

### Pydantic Models

```python
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PromptCategory(str, Enum):
    """Categories of prompts."""
    ROUTING = "routing"
    SPECIALIST = "specialist"
    THINKING = "thinking"
    TOOL = "tool"
    GRADING = "grading"
    META = "meta"
    MEMORY = "memory"


class OutputFormat(str, Enum):
    """Output format types."""
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"
    STRUCTURED = "structured"


class PromptExample(BaseModel):
    """Few-shot example."""
    input: str
    output: str


class PromptDefinition(BaseModel):
    """Complete prompt definition."""

    model_config = {"extra": "forbid"}

    # Identity
    id: str = Field(pattern=r"^[a-z][a-z0-9_\.]*$")
    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    category: PromptCategory
    description: Optional[str] = None

    # Compatibility
    compatible_models: List[str] = Field(min_length=1)

    # Parameters
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=32000)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # Output
    output_format: OutputFormat = OutputFormat.TEXT
    output_schema: Optional[Dict[str, Any]] = None

    # Content
    system_prompt: str = Field(min_length=1)
    user_template: str = Field(min_length=1)

    # Examples
    examples: List[PromptExample] = Field(default_factory=list)

    def render(self, **variables: Any) -> tuple[str, str]:
        """Render system prompt and user message with variables."""
        from jinja2 import Template

        system = Template(self.system_prompt).render(**variables)
        user = Template(self.user_template).render(**variables)

        # Add few-shot examples if present
        if self.examples:
            examples_text = "\n\nExamples:\n"
            for ex in self.examples:
                examples_text += f"Input: {ex.input}\nOutput: {ex.output}\n\n"
            system = system + examples_text

        return system, user
```

---

## Prompt Categories

### 1. Routing Prompts

Classify tasks and route to appropriate handlers.

| Prompt ID | Purpose |
|-----------|---------|
| `router.task_classifier.v1` | Main task type classification |
| `router.complexity_estimator.v1` | Estimate task complexity |
| `router.tool_selector.v1` | Select appropriate tools |
| `router.language_detector.v1` | Detect code language |
| `router.subtask_router.v1` | Route within a category |

### 2. Specialist Prompts

Execute specific types of tasks.

| Prompt ID | Purpose |
|-----------|---------|
| `specialist.code_generator.v1` | Generate code |
| `specialist.code_explainer.v1` | Explain code |
| `specialist.code_debugger.v1` | Debug code |
| `specialist.math_solver.v1` | Solve math problems |
| `specialist.general_qa.v1` | General Q&A |
| `specialist.summarizer.v1` | Summarize text |

### 3. Thinking Prompts

Enable structured reasoning.

| Prompt ID | Purpose |
|-----------|---------|
| `thinking.chain_of_thought.v1` | Step-by-step reasoning |
| `thinking.decomposer.v1` | Break into subtasks |
| `thinking.self_critique.v1` | Evaluate own output |
| `thinking.react.v1` | Reason-Act-Observe loop |
| `thinking.reflection.v1` | Reflect on approach |

### 4. Tool Prompts

Format tool usage.

| Prompt ID | Purpose |
|-----------|---------|
| `tool.calculator_format.v1` | Format calculator calls |
| `tool.code_executor_format.v1` | Format code execution |
| `tool.tool_selection.v1` | Choose which tool to use |

### 5. Grading Prompts

Evaluate outputs.

| Prompt ID | Purpose |
|-----------|---------|
| `grading.output_judge.v1` | General output quality |
| `grading.code_judge.v1` | Code quality |
| `grading.correctness_judge.v1` | Factual correctness |
| `grading.failure_forensics.v1` | Analyze failures |

### 6. Meta Prompts

Self-improvement.

| Prompt ID | Purpose |
|-----------|---------|
| `meta.prompt_improver.v1` | Improve prompts |
| `meta.expansion_suggester.v1` | Suggest expansions |
| `meta.strategy_generator.v1` | Generate strategies |

---

## Example Prompts

### Task Classifier

```yaml
# prompts/routing/task_classifier.yaml
id: router.task_classifier.v1
name: "Task Type Classifier"
version: "1.0.0"
category: ROUTING

description: |
  Classifies incoming tasks into categories for routing.
  Used by the main router node to determine which specialist to use.

compatible_models:
  - qwen2.5:0.5b
  - tinyllama
  - granite3.1-moe:1b

temperature: 0.1
max_tokens: 100
output_format: JSON_SCHEMA
output_schema:
  type: object
  properties:
    category:
      type: string
      enum: [code, math, factual, reasoning, creative, extraction, conversation]
    confidence:
      type: number
      minimum: 0
      maximum: 1
    reasoning:
      type: string
  required: [category, confidence]

system_prompt: |
  You are a task classifier. Your ONLY job is to categorize incoming tasks.

  Categories:
  - "code": Programming, debugging, code explanation, technical implementation
  - "math": Calculations, equations, word problems, numerical analysis
  - "factual": Questions with definitive answers, definitions, facts
  - "reasoning": Logic puzzles, analysis, comparisons, decision-making
  - "creative": Writing, brainstorming, open-ended generation
  - "extraction": Pull structured data from unstructured text
  - "conversation": Greetings, chitchat, clarifications, meta-questions

  Respond with ONLY valid JSON. No markdown, no explanation.

user_template: |
  Classify this task:
  """
  {{task_content}}
  """

  JSON response:

examples:
  - input: "Write a function to sort a list in Python"
    output: '{"category": "code", "confidence": 0.95, "reasoning": "Explicitly asks for code"}'
  - input: "What is 15% of 200?"
    output: '{"category": "math", "confidence": 0.9, "reasoning": "Numerical calculation"}'
  - input: "Hello, how are you?"
    output: '{"category": "conversation", "confidence": 0.95, "reasoning": "Greeting"}'
```

### Code Generator

```yaml
# prompts/specialists/code_generator.yaml
id: specialist.code_generator.v1
name: "Code Generator"
version: "1.0.0"
category: SPECIALIST

description: |
  Generates code based on user requirements.
  Can use code_executor tool to test generated code.

compatible_models:
  - granite-code:3b
  - qwen2.5-coder:3b
  - deepseek-coder:1.3b

temperature: 0.2
max_tokens: 2000
output_format: TEXT

system_prompt: |
  You are an expert programmer. Write clean, efficient, well-documented code.

  Guidelines:
  1. Write idiomatic code for the requested language
  2. Include brief comments for complex logic
  3. Handle common edge cases
  4. Keep code concise but readable
  5. If you need to test code, use the code_executor tool

  Available tools:
  {{available_tools}}

user_template: |
  {{task_content}}

  {{#if context}}
  Additional context:
  {{context}}
  {{/if}}
```

### Chain of Thought

```yaml
# prompts/thinking/chain_of_thought.yaml
id: thinking.chain_of_thought.v1
name: "Chain of Thought"
version: "1.0.0"
category: THINKING

description: |
  Enables step-by-step reasoning for complex problems.

compatible_models:
  - qwen2.5:3b
  - qwen3:4b
  - phi3:mini

temperature: 0.3
max_tokens: 1500
output_format: STRUCTURED

system_prompt: |
  Think through problems step by step. For each step:
  1. State what you're doing
  2. Show your work
  3. Explain your reasoning

  Structure your response as:

  ## Thinking
  Step 1: [description]
  [work]

  Step 2: [description]
  [work]

  ...

  ## Answer
  [final answer]

user_template: |
  Solve this step by step:

  {{task_content}}
```

### Output Judge

```yaml
# prompts/grading/output_judge.yaml
id: grading.output_judge.v1
name: "Output Quality Judge"
version: "1.0.0"
category: GRADING

description: |
  Evaluates the quality of LLM outputs across multiple dimensions.
  Used for grading and feedback.

compatible_models:
  - qwen3:14b
  - gpt-oss:20b

temperature: 0.1
max_tokens: 500
output_format: JSON_SCHEMA
output_schema:
  type: object
  properties:
    overall_score:
      type: number
      minimum: 0
      maximum: 1
    dimensions:
      type: object
      properties:
        correctness:
          type: number
        completeness:
          type: number
        clarity:
          type: number
        format:
          type: number
    issues:
      type: array
      items:
        type: string
    suggestions:
      type: array
      items:
        type: string
  required: [overall_score, dimensions]

system_prompt: |
  You are a quality evaluator. Assess the given output objectively.

  Scoring guide (0.0 - 1.0):
  - 0.0-0.2: Completely wrong or harmful
  - 0.2-0.4: Major issues, mostly incorrect
  - 0.4-0.6: Partially correct, significant gaps
  - 0.6-0.8: Mostly correct, minor issues
  - 0.8-1.0: Excellent, fully addresses the task

  Be specific about issues. Respond with JSON only.

user_template: |
  Task:
  """
  {{original_task}}
  """

  Output to evaluate:
  """
  {{output}}
  """

  {{#if expected_output}}
  Expected output (reference):
  """
  {{expected_output}}
  """
  {{/if}}

  Evaluate this output:
```

---

## Prompt Loader

```python
from pathlib import Path
from typing import Dict
import yaml


class PromptLoader:
    """Loads and caches prompt definitions."""

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self._cache: Dict[str, PromptDefinition] = {}

    def load(self, prompt_id: str) -> PromptDefinition:
        """Load a prompt by ID."""
        if prompt_id in self._cache:
            return self._cache[prompt_id]

        # Convert ID to path: router.task_classifier.v1 -> routing/task_classifier.yaml
        parts = prompt_id.split(".")
        category = self._category_to_folder(parts[0])
        name = parts[1]

        path = self.prompts_dir / category / f"{name}.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_id} at {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        prompt = PromptDefinition(**data)
        self._cache[prompt_id] = prompt
        return prompt

    def _category_to_folder(self, category: str) -> str:
        """Map category prefix to folder name."""
        mapping = {
            "router": "routing",
            "specialist": "specialists",
            "thinking": "thinking",
            "tool": "tools",
            "grading": "grading",
            "meta": "meta",
            "memory": "memory",
        }
        return mapping.get(category, category)
```

---

## File Locations

| Component | File |
|-----------|------|
| Schema models | `src/tinyllm/prompts/schema.py` |
| Loader | `src/tinyllm/prompts/loader.py` |
| Routing prompts | `prompts/routing/*.yaml` |
| Specialist prompts | `prompts/specialists/*.yaml` |
| Thinking prompts | `prompts/thinking/*.yaml` |
| Tool prompts | `prompts/tools/*.yaml` |
| Grading prompts | `prompts/grading/*.yaml` |
| Meta prompts | `prompts/meta/*.yaml` |

---

## Test Cases

| Test | Input | Expected |
|------|-------|----------|
| Load valid prompt | `router.task_classifier.v1` | PromptDefinition instance |
| Load missing prompt | `nonexistent.v1` | FileNotFoundError |
| Render template | variables dict | Rendered strings |
| Invalid YAML | malformed file | Validation error |
| Version validation | `1.0.0` | Valid |
| Version validation | `1.0` | Invalid |
