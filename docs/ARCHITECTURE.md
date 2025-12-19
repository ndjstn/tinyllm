# TinyLLM Architecture

## Overview

TinyLLM is a **neural network of LLMs** - a system where small language models act as intelligent neurons in a larger cognitive architecture. This document describes the system design, core concepts, and implementation details.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Node Types](#node-types)
5. [Message Format](#message-format)
6. [Graph Structure](#graph-structure)
7. [Tool System](#tool-system)
8. [Grading & Evaluation](#grading--evaluation)
9. [Recursive Expansion](#recursive-expansion)
10. [Memory System](#memory-system)
11. [Configuration](#configuration)

---

## Core Concepts

### The Neural Network Metaphor

| Traditional NN | TinyLLM |
|----------------|---------|
| Neuron = activation function | Neuron = small LLM (0.5B-3B params) |
| Weight = learned parameter | Weight = routing probability + prompt |
| Layer = collection of neurons | Layer = pipeline stage |
| Forward pass = weighted sum + activation | Forward pass = route → process → validate |
| Backprop = gradient descent | Backprop = LLM-as-judge → expansion triggers |
| Training = batch updates | Training = continuous self-improvement |

### Design Principles

1. **Small Models, Big System**: Individual models are tiny (≤3B), but the system as a whole is powerful
2. **Tool-First**: Shift computation from models to tools (calculator > mental math)
3. **Fail Forward**: Failures trigger expansion and improvement, not just retries
4. **Observability**: Every execution produces a complete trace
5. **Deterministic When Possible**: Use low temperature, structured outputs, validation

### Model Tiers

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MODEL TIERS                                 │
├─────────┬────────────────┬───────────────────┬─────────────────────┤
│  Tier   │  Purpose       │  Models           │  VRAM/Latency       │
├─────────┼────────────────┼───────────────────┼─────────────────────┤
│  T0     │  Routers       │  qwen2.5:0.5b     │  ~500MB / <100ms    │
│         │  (classify)    │  tinyllama        │                     │
├─────────┼────────────────┼───────────────────┼─────────────────────┤
│  T1     │  Specialists   │  granite-code:3b  │  2-3GB / 200-500ms  │
│         │  (execute)     │  qwen2.5:3b       │                     │
│         │                │  phi3:mini        │                     │
├─────────┼────────────────┼───────────────────┼─────────────────────┤
│  T2     │  Workers       │  qwen3:8b         │  5-6GB / 500ms-1s   │
│         │  (complex)     │  granite3.1:8b    │                     │
├─────────┼────────────────┼───────────────────┼─────────────────────┤
│  T3     │  Judges        │  qwen3:14b        │  10-15GB / 1-3s     │
│         │  (evaluate)    │  gpt-oss:20b      │                     │
└─────────┴────────────────┴───────────────────┴─────────────────────┘
```

---

## System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         API LAYER                             │   │
│  │              FastAPI endpoints / CLI interface                │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │                      GRAPH EXECUTOR                           │   │
│  │         Orchestrates node execution, manages state            │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │                       GRAPH DEFINITION                        │   │
│  │              Nodes, edges, routing rules (YAML)               │   │
│  └─────────┬────────────────────┬─────────────────────┬─────────┘   │
│            │                    │                     │              │
│  ┌─────────▼─────────┐ ┌───────▼───────┐ ┌──────────▼──────────┐   │
│  │   MODEL NODES     │ │  TOOL NODES   │ │   CONTROL NODES     │   │
│  │  (LLM inference)  │ │  (calculat.)  │ │ (router/gate/loop)  │   │
│  └─────────┬─────────┘ └───────┬───────┘ └──────────┬──────────┘   │
│            │                   │                     │               │
│  ┌─────────▼───────────────────▼─────────────────────▼──────────┐   │
│  │                      OLLAMA CLIENT                            │   │
│  │              Async connection pool to Ollama                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      SUPPORT SYSTEMS                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │   │
│  │  │ Grading  │ │ Memory   │ │ Metrics  │ │    Expansion     │ │   │
│  │  │ System   │ │ (STM/LTM)│ │ Tracker  │ │     Engine       │ │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Request Lifecycle

```
1. INPUT VALIDATION
   ├─ Validate against TaskPayload schema
   ├─ Extract metadata (timestamp, trace_id)
   └─ Create initial Message

2. ROUTING
   ├─ Router node classifies task type
   ├─ Confidence score determines path
   └─ Low confidence → fallback or parallel

3. EXECUTION
   ├─ Specialist node processes task
   ├─ May invoke tools (calculator, code exec)
   ├─ May iterate (chain-of-thought, self-critique)
   └─ Produces output + metadata

4. QUALITY GATE
   ├─ Rule-based validation (length, format, schema)
   ├─ Optional LLM judge (sampled)
   └─ Decision: pass / retry / expand

5. OUTPUT
   ├─ Package response
   ├─ Record metrics
   └─ Store trace
```

### Message Flow Between Nodes

```
┌─────────────┐     Message      ┌─────────────┐
│   Node A    │ ───────────────► │   Node B    │
└─────────────┘                  └─────────────┘

Message = {
    trace_id: "uuid",
    parent_node: "node_a",
    payload: { ... },
    metadata: {
        timestamp: "...",
        latency_ms: 120,
        model_used: "qwen2.5:0.5b",
        tokens_in: 50,
        tokens_out: 20,
    }
}
```

---

## Node Types

### Entry Node
First node in the graph. Validates input and creates initial context.

```yaml
- id: entry.main
  type: entry
  config:
    input_schema: TaskPayload
    required_fields: ["content"]
```

### Exit Node
Terminal node. Packages final response.

```yaml
- id: exit.success
  type: exit
  config:
    output_schema: TaskResponse
```

### Router Node
Classifies input and routes to appropriate specialist.

```yaml
- id: router.task_type
  type: router
  config:
    model: qwen2.5:0.5b
    prompt_id: router.task_classifier.v1
    routes:
      - name: code
        target: specialist.code
        confidence_threshold: 0.7
      - name: math
        target: specialist.math
        confidence_threshold: 0.7
    default_route: specialist.general
```

### Model Node
Invokes an LLM for processing.

```yaml
- id: specialist.code
  type: model
  config:
    model: granite-code:3b
    prompt_id: specialist.code_generator.v1
    temperature: 0.2
    max_tokens: 2000
    tools: [code_executor]
```

### Tool Node
Invokes a tool (calculator, code executor, etc.).

```yaml
- id: tool.calculator
  type: tool
  config:
    tool_id: calculator
    timeout_ms: 1000
```

### Gate Node
Quality checkpoint. Pass or fail based on rules/LLM.

```yaml
- id: gate.quality
  type: gate
  config:
    evaluator:
      type: rule_based  # or "llm_judge"
      rules:
        - min_length: 10
        - max_length: 5000
        - required_fields: ["answer"]
    threshold: 0.6
    on_pass: exit.success
    on_fail:
      action: retry
      max_retries: 2
      fallback: exit.fallback
```

### Transform Node
Modifies message payload (summarize, format, extract).

```yaml
- id: transform.summarize
  type: transform
  config:
    operation: summarize
    max_length: 500
```

### Loop Node
Iterative processing (chain-of-thought, refinement).

```yaml
- id: loop.refine
  type: loop
  config:
    max_iterations: 3
    body_node: specialist.refiner
    condition: improvement_detected
```

### Fanout Node
Dynamic parallel execution.

```yaml
- id: fanout.parallel
  type: dynamic_fanout
  config:
    splitter: split_by_subtask
    targets: [specialist.code, specialist.research]
    aggregator: merge_results
```

---

## Message Format

### Pydantic Schema

```python
class Message(BaseModel):
    trace_id: str = Field(description="Unique trace identifier")
    message_id: str = Field(default_factory=uuid4, description="Message ID")
    parent_id: Optional[str] = Field(description="Parent message ID")
    source_node: str = Field(description="Node that created this message")
    target_node: Optional[str] = Field(description="Intended recipient")

    payload: MessagePayload = Field(description="Actual content")
    metadata: MessageMetadata = Field(description="Execution metadata")

    created_at: datetime = Field(default_factory=datetime.utcnow)

class MessagePayload(BaseModel):
    task: Optional[str] = None
    content: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None

class MessageMetadata(BaseModel):
    latency_ms: Optional[int] = None
    model_used: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
```

---

## Graph Structure

### YAML Definition

```yaml
id: graph.main
version: "1.0.0"
name: "Main Execution Graph"

metadata:
  created_at: "2024-01-01"
  description: "Day 1 minimal viable graph"

nodes:
  - id: entry.main
    type: entry
    # ...

edges:
  - from: entry.main
    to: router.task_type
    weight: 1.0

  - from: router.task_type
    to: specialist.code
    condition: "route == 'code'"

protected:
  - entry.main
  - exit.success

entry_points:
  - entry.main

exit_points:
  - exit.success
  - exit.fallback
```

### Graph Invariants

1. Must have exactly one entry point
2. All paths must lead to an exit point
3. No orphan nodes
4. No cycles (except explicit loops)
5. Protected nodes cannot be removed by pruning

---

## Tool System

### Tool Interface

```python
class BaseTool(ABC):
    """Base class for all tools."""

    id: str
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    @abstractmethod
    async def execute(self, input: BaseModel) -> BaseModel:
        """Execute the tool and return result."""
        pass
```

### Built-in Tools

| Tool | Purpose | Sandbox |
|------|---------|---------|
| `calculator` | Math expressions | None |
| `code_executor` | Run Python code | Container |
| `web_search` | Search the web | None |
| `file_reader` | Read files | Restricted |
| `memory_store` | Store/retrieve memory | None |

### Tool Invocation Flow

```
1. Model outputs tool_call in structured format
2. Parser extracts tool_call from response
3. Tool executor validates input against schema
4. Tool runs (possibly sandboxed)
5. Result packaged as tool_result message
6. Sent back to model for continuation
```

---

## Grading & Evaluation

### LLM-as-Judge

Large models evaluate outputs from small models:

```yaml
grading:
  default_judge: qwen3:14b
  sampling_rate: 0.1  # Grade 10% of requests

  rubric:
    correctness:
      weight: 0.4
      prompt_id: grading.correctness.v1
    completeness:
      weight: 0.3
      prompt_id: grading.completeness.v1
    format:
      weight: 0.2
      prompt_id: grading.format.v1
    efficiency:
      weight: 0.1
      prompt_id: grading.efficiency.v1
```

### Failure Forensics

Every failure is categorized:

```
WHERE did it fail?
├── router     (wrong classification)
├── specialist (wrong answer)
├── tool       (tool error)
└── gate       (validation failed)

WHAT went wrong?
├── incorrect   (factually wrong)
├── incomplete  (missing parts)
├── malformed   (bad format)
├── timeout     (too slow)
└── exception   (code error)

WHY did it happen?
├── ambiguous_input    (unclear request)
├── knowledge_gap      (model doesn't know)
├── reasoning_error    (logic mistake)
├── tool_misuse        (wrong tool call)
└── context_overflow   (too long)
```

---

## Recursive Expansion

### Trigger Conditions

A node expands when:

```python
def should_expand(node: Node) -> bool:
    return (
        node.stats.failure_rate > 0.3 and
        node.stats.total_executions > 50 and
        node.expansion_count < MAX_EXPANSIONS and
        not is_on_cooldown(node)
    )
```

### Expansion Process

```
1. FAILURE ANALYSIS
   └─ Use T3 judge to categorize failures

2. STRATEGY GENERATION
   └─ T3 model suggests sub-strategies

3. GRAPH MUTATION
   └─ Replace failing node with router + specialists

4. WEIGHT INITIALIZATION
   └─ New routes start with equal weights

5. CALIBRATION
   └─ Run against test set to tune weights

6. VALIDATION
   └─ Must improve overall success rate
```

### Pruning

Low-performing branches are removed:

```python
def should_prune(node: Node) -> bool:
    return (
        node.stats.selection_rate < 0.05 and  # Rarely selected
        node.stats.total_executions > 100 and
        node.age_days > 7 and
        node not in protected_nodes
    )
```

---

## Memory System

### Short-Term Memory (STM)

Within a single conversation:

```python
class STM:
    messages: List[Message]  # Recent messages
    context: Dict[str, Any]  # Extracted entities/facts
    max_messages: int = 20

    def add(self, message: Message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.summarize_and_prune()
```

### Long-Term Memory (LTM)

Persistent knowledge across sessions:

```python
class LTM:
    # Vector store for semantic search
    vector_store: VectorStore

    # Structured facts
    facts: Dict[str, Fact]

    def store(self, content: str, metadata: dict):
        embedding = embed(content)
        self.vector_store.add(embedding, content, metadata)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        embedding = embed(query)
        return self.vector_store.search(embedding, k)
```

---

## Configuration

### File Structure

```
config/
├── tinyllm.yaml           # Main config (includes others)
├── system.yaml            # Global settings
├── models.yaml            # Model tiers
├── tools.yaml             # Tool definitions
├── grading.yaml           # Evaluation settings
├── expansion.yaml         # Growth rules
├── memory.yaml            # Memory settings
└── environments/
    ├── development.yaml   # Dev overrides
    ├── production.yaml    # Prod settings
    └── testing.yaml       # Test settings
```

### Main Config

```yaml
# tinyllm.yaml
version: "1.0"

includes:
  - system.yaml
  - models.yaml
  - tools.yaml
  - grading.yaml
  - expansion.yaml
  - memory.yaml

environment: ${TINYLLM_ENV:-development}
environment_file: environments/${environment}.yaml

graph:
  file: ../graphs/current
```

See [Configuration Specification](specs/configuration.md) for full details.

---

## Next Steps

- [Node Specification](specs/nodes.md)
- [Message Specification](specs/messages.md)
- [Graph Specification](specs/graphs.md)
- [Tool Specification](specs/tools.md)
- [Prompt Specification](specs/prompts.md)
