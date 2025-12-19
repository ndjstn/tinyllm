# LoopNode Usage Guide

The `LoopNode` enables iterative workflows with flexible termination conditions in TinyLLM graphs.

## Overview

The LoopNode executes a body node repeatedly until a termination condition is met. It supports multiple termination strategies and includes safety limits (max iterations and timeout) to prevent infinite loops.

## Key Classes

### LoopCondition (Enum)

Defines the type of loop termination:

- `FIXED_COUNT`: Run exactly N times
- `UNTIL_SUCCESS`: Run until the body node succeeds
- `UNTIL_CONDITION`: Run until a condition expression evaluates to true
- `WHILE_CONDITION`: Run while a condition expression is true

### LoopConfig

Configuration for the loop node:

```python
class LoopConfig(NodeConfig):
    body_node: str                           # Node ID to execute in loop body
    condition_type: LoopCondition            # Type of termination condition
    max_iterations: int = 10                 # Safety limit (1-1000)
    timeout_ms: int = 60000                  # Timeout in milliseconds (1000-300000)

    # Condition-specific parameters
    fixed_count: Optional[int] = None        # For FIXED_COUNT mode
    condition_expression: Optional[str] = None  # For UNTIL/WHILE_CONDITION modes

    # Behavior options
    collect_results: bool = True             # Collect all iteration results
    continue_on_error: bool = False          # Continue on iteration errors
    pass_iteration_number: bool = True       # Pass iteration # in metadata
```

### LoopState

Tracks loop execution state:

```python
class LoopState(BaseModel):
    iteration_count: int                     # Current iteration number
    accumulated_results: List[Dict[str, Any]]  # Results from all iterations
    elapsed_time_ms: int                     # Total elapsed time
    success_count: int                       # Number of successful iterations
    failure_count: int                       # Number of failed iterations
    last_result: Optional[Dict[str, Any]]    # Result from last iteration
    terminated_by: Optional[str]             # Termination reason
```

### LoopResult

Final result from loop execution:

```python
class LoopResult(BaseModel):
    success: bool                            # Whether loop completed successfully
    iterations_executed: int                 # Number of iterations executed
    all_iterations: List[Dict[str, Any]]     # Data from all iterations
    final_output: Optional[str]              # Final output content
    termination_reason: str                  # Why the loop terminated
    total_elapsed_ms: int                    # Total loop execution time
    success_rate: float                      # Ratio of successful iterations
```

## Usage Examples

### Example 1: Fixed Count Loop

Run exactly 5 iterations:

```yaml
- id: loop.fixed
  type: loop
  config:
    body_node: model.process
    condition_type: fixed_count
    fixed_count: 5
    max_iterations: 10
```

### Example 2: Retry Until Success

Retry a task up to 10 times until it succeeds:

```yaml
- id: loop.retry
  type: loop
  config:
    body_node: model.api_call
    condition_type: until_success
    max_iterations: 10
    timeout_ms: 120000
    continue_on_error: true
```

### Example 3: Until Condition

Run until 3 successful iterations are achieved:

```yaml
- id: loop.until_threshold
  type: loop
  config:
    body_node: model.validate
    condition_type: until_condition
    condition_expression: "success_count >= 3"
    max_iterations: 20
    continue_on_error: true
```

### Example 4: While Condition

Run while iteration count is below threshold:

```yaml
- id: loop.while_counter
  type: loop
  config:
    body_node: model.process
    condition_type: while_condition
    condition_expression: "iteration < 10"
    max_iterations: 15
```

### Example 5: Complex Condition

Run until output contains specific marker:

```yaml
- id: loop.until_complete
  type: loop
  config:
    body_node: model.worker
    condition_type: until_condition
    condition_expression: "last_success and 'DONE' in str(last_output)"
    max_iterations: 50
    timeout_ms: 300000
```

### Example 6: Using Context Variables

Use execution context variables in condition:

```yaml
- id: loop.with_vars
  type: loop
  config:
    body_node: model.compute
    condition_type: until_condition
    condition_expression: "iteration >= variables.get('target_count', 5)"
    max_iterations: 20
```

## Condition Expression Context

When using `UNTIL_CONDITION` or `WHILE_CONDITION`, expressions have access to:

### Loop State Variables
- `iteration`: Current iteration number (starting from 0)
- `results`: List of all iteration results
- `last_result`: Dictionary containing last iteration result
- `success_count`: Number of successful iterations
- `failure_count`: Number of failed iterations

### Last Result Fields (for convenience)
- `last_success`: Boolean indicating if last iteration succeeded
- `last_output`: Output from last iteration
- `last_error`: Error from last iteration (if any)

### Execution Context
- `variables`: Dictionary of execution context variables

### Utility Functions
- `len`, `str`, `int`, `float`, `bool`
- `any`, `all`, `sum`, `min`, `max`

## Termination Behavior

### FIXED_COUNT
- Runs exactly `fixed_count` iterations
- Terminates after completing the specified count
- Requires `fixed_count` parameter

### UNTIL_SUCCESS
- Runs until an iteration succeeds (`success=True`)
- Useful for retry logic
- Respects `max_iterations` limit

### UNTIL_CONDITION
- Evaluates condition **after** each iteration
- Terminates when condition becomes true
- Requires `condition_expression` parameter

### WHILE_CONDITION
- Evaluates condition **before** each iteration
- Terminates when condition becomes false
- Requires `condition_expression` parameter

## Safety Limits

All loops respect two safety limits:

1. **max_iterations**: Maximum number of iterations (1-1000)
   - Loop terminates with `termination_reason: "max_iterations_reached"`

2. **timeout_ms**: Maximum execution time in milliseconds (1000-300000)
   - Loop terminates with `termination_reason: "timeout_exceeded"`

These limits prevent infinite loops and runaway execution.

## Error Handling

### continue_on_error = false (default)
- Loop stops immediately on first iteration error
- Termination reason: `"iteration_failed"`
- Useful for critical tasks that must succeed

### continue_on_error = true
- Loop continues executing even if iterations fail
- Tracks success/failure counts
- Useful for best-effort processing or retry scenarios

## Programmatic Usage

```python
from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.nodes.loop import LoopNode, LoopCondition

# Create a loop node definition
definition = NodeDefinition(
    id="loop.process",
    type=NodeType.LOOP,
    config={
        "body_node": "model.worker",
        "condition_type": "until_success",
        "max_iterations": 5,
        "timeout_ms": 30000,
        "continue_on_error": True,
    }
)

# Create the node
loop_node = LoopNode(definition)

# Execute (in context of graph execution)
result = await loop_node.execute(message, context)

# Access loop results
loop_result = result.metadata["loop_result"]
print(f"Iterations: {loop_result['iterations_executed']}")
print(f"Success rate: {loop_result['success_rate']}")
print(f"Termination: {loop_result['termination_reason']}")
```

## Output Messages

The LoopNode creates a single output message containing:

- `payload.content`: Final output from last successful iteration
- `payload.metadata.loop_result`: Complete LoopResult object
- `payload.metadata.iterations`: Number of iterations executed
- `payload.metadata.termination`: Termination reason

## Best Practices

1. **Always set appropriate max_iterations**: Even with condition-based loops, set a reasonable max_iterations to prevent infinite loops

2. **Use timeout_ms for time-bounded operations**: Prevent loops from running indefinitely

3. **Choose the right condition type**:
   - Use `FIXED_COUNT` for deterministic iteration counts
   - Use `UNTIL_SUCCESS` for retry logic
   - Use `UNTIL_CONDITION` for dynamic termination based on results
   - Use `WHILE_CONDITION` for traditional while-loop behavior

4. **Enable continue_on_error for resilient workflows**: When some iteration failures are acceptable

5. **Use collect_results wisely**: Disable if iterations produce large outputs and you only need the final result

6. **Pass iteration metadata**: Enable `pass_iteration_number` to help body nodes track their iteration context

7. **Test condition expressions**: Ensure your condition expressions are correct and handle edge cases

## Common Patterns

### Retry with Exponential Backoff (simulated)
```yaml
- id: loop.retry_backoff
  type: loop
  config:
    body_node: model.api_call
    condition_type: until_success
    max_iterations: 5
    continue_on_error: true
```

### Batch Processing
```yaml
- id: loop.batch
  type: loop
  config:
    body_node: model.process_item
    condition_type: fixed_count
    fixed_count: 100
    collect_results: true
```

### Convergence Loop
```yaml
- id: loop.converge
  type: loop
  config:
    body_node: model.optimize
    condition_type: until_condition
    condition_expression: "last_success and float(last_output) < 0.001"
    max_iterations: 100
```

### Validation Loop
```yaml
- id: loop.validate
  type: loop
  config:
    body_node: model.validate
    condition_type: until_condition
    condition_expression: "success_count >= 3 and failure_count == 0"
    max_iterations: 10
    continue_on_error: false
```
