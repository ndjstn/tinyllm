# Node Specification

## Overview

Nodes are the fundamental units of computation in TinyLLM. Each node represents a step in the processing pipeline - it receives a message, processes it, and produces output.

## Dependencies

- `pydantic>=2.0.0`
- `src/tinyllm/core/message.py` (Message types)

---

## Base Node Interface

### Pydantic Models

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class NodeType(str, Enum):
    """Types of nodes in the graph."""
    ENTRY = "entry"
    EXIT = "exit"
    ROUTER = "router"
    MODEL = "model"
    TOOL = "tool"
    GATE = "gate"
    TRANSFORM = "transform"
    LOOP = "loop"
    FANOUT = "dynamic_fanout"


class NodeStats(BaseModel):
    """Runtime statistics for a node."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_latency_ms: int = 0
    last_execution: Optional[datetime] = None
    expansion_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_latency_ms / self.total_executions


class NodeConfig(BaseModel):
    """Base configuration for all nodes."""
    model_config = {"extra": "forbid"}

    timeout_ms: int = Field(default=5000, ge=100, le=60000)
    retry_count: int = Field(default=0, ge=0, le=3)
    retry_delay_ms: int = Field(default=1000, ge=0, le=10000)


class NodeDefinition(BaseModel):
    """Complete node definition for graph construction."""
    id: str = Field(description="Unique identifier", pattern=r"^[a-z][a-z0-9_\.]*$")
    type: NodeType
    name: Optional[str] = Field(default=None, description="Human-readable name")
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    stats: NodeStats = Field(default_factory=NodeStats)
```

### Abstract Base Class

```python
class BaseNode(ABC):
    """Abstract base class for all nodes."""

    def __init__(self, definition: NodeDefinition):
        self.id = definition.id
        self.type = definition.type
        self.name = definition.name or definition.id
        self.config = definition.config
        self.stats = definition.stats

    @abstractmethod
    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        """
        Execute the node's logic.

        Args:
            message: Input message to process
            context: Execution context with graph state, memory, etc.

        Returns:
            NodeResult containing output message(s) and execution metadata
        """
        pass

    def update_stats(self, success: bool, latency_ms: int) -> None:
        """Update node statistics after execution."""
        self.stats.total_executions += 1
        if success:
            self.stats.successful_executions += 1
        else:
            self.stats.failed_executions += 1
        self.stats.total_latency_ms += latency_ms
        self.stats.last_execution = datetime.utcnow()


class NodeResult(BaseModel):
    """Result of node execution."""
    success: bool
    output_messages: List[Message] = Field(default_factory=list)
    next_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0
```

---

## Node Types

### Entry Node

First node in the graph. Validates input and initializes context.

```python
class EntryNodeConfig(NodeConfig):
    """Configuration for entry nodes."""
    input_schema: str = Field(description="Pydantic model name for input validation")
    required_fields: List[str] = Field(default_factory=list)


class EntryNode(BaseNode):
    """Entry point node for the graph."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # Validate input against schema
        # Initialize trace
        # Set up context
        # Return validated message to next node
        pass
```

**YAML Configuration:**
```yaml
- id: entry.main
  type: entry
  name: "Main Entry"
  config:
    input_schema: TaskPayload
    required_fields:
      - content
```

### Exit Node

Terminal node. Packages final response.

```python
class ExitNodeConfig(NodeConfig):
    """Configuration for exit nodes."""
    output_schema: str = Field(description="Pydantic model name for output")
    status: str = Field(default="success", pattern=r"^(success|fallback|error)$")


class ExitNode(BaseNode):
    """Exit point node for the graph."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # Package final response
        # Record completion in trace
        # Return with empty next_nodes (terminal)
        pass
```

**YAML Configuration:**
```yaml
- id: exit.success
  type: exit
  config:
    output_schema: TaskResponse
    status: success
```

### Router Node

Classifies input and routes to appropriate handler.

```python
class RouteDefinition(BaseModel):
    """Definition of a routing path."""
    name: str
    target: str  # Node ID to route to
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    description: Optional[str] = None


class RouterNodeConfig(NodeConfig):
    """Configuration for router nodes."""
    model: str = Field(description="Ollama model for classification")
    prompt_id: str = Field(description="Prompt template ID")
    routes: List[RouteDefinition]
    default_route: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class RouterNode(BaseNode):
    """Routes messages based on classification."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # 1. Format prompt with message content
        # 2. Call model for classification
        # 3. Parse JSON response for route + confidence
        # 4. Select route based on confidence threshold
        # 5. Return message routed to selected node
        pass
```

**YAML Configuration:**
```yaml
- id: router.task_type
  type: router
  config:
    model: qwen2.5:0.5b
    prompt_id: router.task_classifier.v1
    temperature: 0.1
    routes:
      - name: code
        target: specialist.code
        confidence_threshold: 0.7
        description: "Programming and code tasks"
      - name: math
        target: specialist.math
        confidence_threshold: 0.7
        description: "Mathematical problems"
      - name: general
        target: specialist.general
        confidence_threshold: 0.5
    default_route: specialist.general
```

### Model Node

Invokes an LLM for processing.

```python
class ModelNodeConfig(NodeConfig):
    """Configuration for model nodes."""
    model: str = Field(description="Ollama model name")
    prompt_id: str = Field(description="Prompt template ID")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=32000)
    tools: List[str] = Field(default_factory=list, description="Available tool IDs")
    output_format: str = Field(default="text", pattern=r"^(text|json|structured)$")


class ModelNode(BaseNode):
    """Processes messages using an LLM."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # 1. Load prompt template
        # 2. Format with message and context
        # 3. Call Ollama with model
        # 4. Parse response (handle tool calls if present)
        # 5. If tool call, execute tool and loop
        # 6. Return processed message
        pass
```

**YAML Configuration:**
```yaml
- id: specialist.code
  type: model
  name: "Code Specialist"
  config:
    model: granite-code:3b
    prompt_id: specialist.code_generator.v1
    temperature: 0.2
    max_tokens: 2000
    tools:
      - code_executor
    output_format: text
```

### Tool Node

Invokes a tool directly (without LLM).

```python
class ToolNodeConfig(NodeConfig):
    """Configuration for tool nodes."""
    tool_id: str = Field(description="ID of tool to invoke")
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map message fields to tool input"
    )


class ToolNode(BaseNode):
    """Invokes a tool directly."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # 1. Extract tool input from message
        # 2. Invoke tool
        # 3. Package result as message
        pass
```

### Gate Node

Quality checkpoint with pass/fail logic.

```python
class EvaluatorConfig(BaseModel):
    """Configuration for evaluator."""
    type: str = Field(pattern=r"^(rule_based|llm_judge)$")
    rules: Optional[List[Dict[str, Any]]] = None  # For rule_based
    judge_model: Optional[str] = None  # For llm_judge
    judge_prompt: Optional[str] = None


class OnFailAction(BaseModel):
    """What to do when gate fails."""
    action: str = Field(pattern=r"^(retry|fallback|expand)$")
    max_retries: int = Field(default=1, ge=0, le=5)
    fallback: Optional[str] = None  # Node ID for fallback


class GateNodeConfig(NodeConfig):
    """Configuration for gate nodes."""
    evaluator: EvaluatorConfig
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    on_pass: str = Field(description="Node ID when passed")
    on_fail: OnFailAction


class GateNode(BaseNode):
    """Quality gate that validates outputs."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # 1. Run evaluator (rules or LLM)
        # 2. Compare score to threshold
        # 3. Route to on_pass or handle on_fail
        pass
```

**YAML Configuration:**
```yaml
- id: gate.quality
  type: gate
  config:
    evaluator:
      type: rule_based
      rules:
        - check: min_length
          value: 10
        - check: max_length
          value: 5000
        - check: not_empty
    threshold: 0.6
    on_pass: exit.success
    on_fail:
      action: retry
      max_retries: 2
      fallback: exit.fallback
```

### Transform Node

Modifies message content.

```python
class TransformNodeConfig(NodeConfig):
    """Configuration for transform nodes."""
    operation: str = Field(
        description="Transform operation",
        pattern=r"^(summarize|extract|format|merge)$"
    )
    model: Optional[str] = None  # Some transforms need LLM
    prompt_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TransformNode(BaseNode):
    """Transforms message content."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # Apply transformation based on operation
        pass
```

### Loop Node

Iterative processing.

```python
class LoopNodeConfig(NodeConfig):
    """Configuration for loop nodes."""
    max_iterations: int = Field(default=3, ge=1, le=10)
    body_node: str = Field(description="Node to execute in loop")
    condition: str = Field(
        description="Condition to continue",
        pattern=r"^(improvement_detected|confidence_low|iteration_count)$"
    )
    condition_threshold: Optional[float] = None


class LoopNode(BaseNode):
    """Iterative processing node."""

    async def execute(self, message: Message, context: ExecutionContext) -> NodeResult:
        # 1. Initialize iteration counter
        # 2. Execute body_node
        # 3. Check condition
        # 4. Repeat or exit
        pass
```

---

## Node Registry

```python
class NodeRegistry:
    """Registry for node types and factories."""

    _node_types: Dict[NodeType, Type[BaseNode]] = {}

    @classmethod
    def register(cls, node_type: NodeType):
        """Decorator to register a node type."""
        def decorator(node_class: Type[BaseNode]):
            cls._node_types[node_type] = node_class
            return node_class
        return decorator

    @classmethod
    def create(cls, definition: NodeDefinition) -> BaseNode:
        """Create a node instance from definition."""
        node_class = cls._node_types.get(definition.type)
        if not node_class:
            raise ValueError(f"Unknown node type: {definition.type}")
        return node_class(definition)
```

---

## File Locations

| Component | File |
|-----------|------|
| Base classes | `src/tinyllm/core/node.py` |
| Node registry | `src/tinyllm/core/registry.py` |
| Entry/Exit nodes | `src/tinyllm/nodes/entry_exit.py` |
| Router node | `src/tinyllm/nodes/router.py` |
| Model node | `src/tinyllm/nodes/model.py` |
| Gate node | `src/tinyllm/nodes/gate.py` |
| Tool node | `src/tinyllm/nodes/tool.py` |
| Transform node | `src/tinyllm/nodes/transform.py` |
| Loop node | `src/tinyllm/nodes/loop.py` |

---

## Test Cases

### Base Node Tests

| Test | Input | Expected |
|------|-------|----------|
| Create node from definition | Valid NodeDefinition | Node instance |
| Stats update on success | success=True, latency=100 | stats.successful_executions += 1 |
| Stats update on failure | success=False | stats.failed_executions += 1 |
| Success rate calculation | 8 success, 2 fail | 0.8 |

### Router Node Tests

| Test | Input | Expected |
|------|-------|----------|
| Route to code | "write a function" | next_nodes=["specialist.code"] |
| Route to math | "calculate 2+2" | next_nodes=["specialist.math"] |
| Default route | "hello" | next_nodes=["specialist.general"] |
| Low confidence | ambiguous input | Uses default_route |

### Gate Node Tests

| Test | Input | Expected |
|------|-------|----------|
| Pass on valid | long, complete response | next_nodes=["exit.success"] |
| Fail on too short | "ok" | triggers on_fail action |
| Retry on fail | first attempt fails | retries body_node |
