# TinyLLM Workflow Examples - Overview

This document provides a visual overview of the example workflows and their architectural patterns.

## Workflow Complexity Matrix

| Workflow | Nodes | Complexity | Execution Time | Best For |
|----------|-------|------------|----------------|----------|
| **parallel_consensus.yaml** | 7 | Low | Fast (parallel) | Quick validation, consensus |
| **qa_pipeline.yaml** | 9 | Medium | Medium | Question routing, quality control |
| **data_processing.yaml** | 9 | Medium | Fast | ETL, data validation |
| **iterative_refinement.yaml** | 8 | Medium | Slow (iterative) | Content polishing |
| **multi_stage_analysis.yaml** | 19 | High | Slow (parallel + iterative) | Comprehensive analysis |

## Node Type Usage Across Workflows

| Node Type | qa_pipeline | parallel_consensus | iterative_refinement | data_processing | multi_stage_analysis |
|-----------|-------------|-------------------|---------------------|-----------------|---------------------|
| Entry | 1 | 1 | 1 | 1 | 1 |
| Exit | 2 | 1 | 2 | 2 | 2 |
| Router | 1 | - | - | - | 1 |
| Model | 4 | 3 | 2 | 1 | 11 |
| Gate | 1 | - | 1 | 1 | 1 |
| Transform | - | 1 | 1 | 4 | 1 |
| Loop | - | - | 1 | - | 1 |
| Fanout | - | 1 | - | - | 3 |

## Architectural Patterns

### 1. Linear Pipeline (Simplest)
```
Entry → Transform → Transform → Exit
```
Used in: Basic data processing stages

### 2. Router Pattern
```
Entry → Router → [Model A | Model B | Model C] → Exit
```
Used in: `qa_pipeline.yaml`, `multi_stage_analysis.yaml`

### 3. Fanout-Aggregate Pattern
```
Entry → Fanout → [Model 1 | Model 2 | Model 3] → Aggregate → Exit
```
Used in: `parallel_consensus.yaml`, `multi_stage_analysis.yaml`

### 4. Loop Pattern
```
Entry → Loop(Model → Evaluate) → Exit
```
Used in: `iterative_refinement.yaml`, `multi_stage_analysis.yaml`

### 5. Gate Pattern
```
Entry → Model → Gate → [Success Exit | Failure Exit]
```
Used in: All workflows except `parallel_consensus.yaml`

### 6. Complex Composite (Most Advanced)
```
Entry → Router → Fanout → Loop → Transform → Gate → Exit
```
Used in: `multi_stage_analysis.yaml`

## Feature Comparison

### Quality Control
- **qa_pipeline.yaml**: LLM-based quality gate
- **parallel_consensus.yaml**: Majority vote validation
- **iterative_refinement.yaml**: Self-evaluation loop
- **data_processing.yaml**: LLM validation + expression gate
- **multi_stage_analysis.yaml**: Final LLM quality gate

### Parallelization
- **parallel_consensus.yaml**: 3 models in parallel
- **multi_stage_analysis.yaml**: 2-3 models per domain in parallel
- Others: Sequential execution

### Iteration
- **iterative_refinement.yaml**: Up to 5 refinement passes
- **multi_stage_analysis.yaml**: 2 synthesis passes
- Others: Single-pass

### Routing Intelligence
- **qa_pipeline.yaml**: 4-way question type routing
- **multi_stage_analysis.yaml**: 3-way domain routing + 9 specialists
- Others: No routing

## Learning Path

### Beginner (Start Here)
1. **parallel_consensus.yaml** - Learn fanout and aggregation
2. **qa_pipeline.yaml** - Learn routing and gates

### Intermediate
3. **data_processing.yaml** - Learn transforms and data validation
4. **iterative_refinement.yaml** - Learn loops and conditions

### Advanced
5. **multi_stage_analysis.yaml** - Combine all patterns

## Use Case Decision Tree

```
Need parallel execution?
├─ Yes → parallel_consensus.yaml
└─ No → Need routing?
    ├─ Yes → Need analysis?
    │   ├─ Yes → multi_stage_analysis.yaml
    │   └─ No → qa_pipeline.yaml
    └─ No → Need iteration?
        ├─ Yes → iterative_refinement.yaml
        └─ No → data_processing.yaml
```

## Performance Characteristics

### Latency (Estimated)

**parallel_consensus.yaml**: ~10-15s
- 3 models run in parallel
- Limited by slowest model

**qa_pipeline.yaml**: ~5-10s  
- Single routing decision
- One specialist model
- One quality check

**data_processing.yaml**: ~3-7s
- Mostly transforms (fast)
- One validation step

**iterative_refinement.yaml**: ~30-60s
- 2-5 iterations
- Each iteration requires full model inference

**multi_stage_analysis.yaml**: ~45-90s
- Domain routing
- 2-3 parallel analysts
- 2 synthesis iterations  
- Quality validation

### Throughput

For batch processing:
- **Best**: `data_processing.yaml` (mostly transforms)
- **Good**: `qa_pipeline.yaml`, `parallel_consensus.yaml`
- **Moderate**: `iterative_refinement.yaml`
- **Slow**: `multi_stage_analysis.yaml`

## Customization Patterns

### Adding Specialists

**In Router Workflows** (qa_pipeline, multi_stage_analysis):
```yaml
# 1. Add route definition
routes:
  - name: new_type
    description: When to use this
    target: model.new_specialist
    
# 2. Add model node
- id: model.new_specialist
  type: model
  config:
    model: qwen2.5:3b
    system_prompt: "Your specialization..."

# 3. Add edge
- from_node: router.xxx
  to_node: model.new_specialist
  condition: "route == 'new_type'"
```

### Adjusting Parallel Execution

**In Fanout Workflows** (parallel_consensus, multi_stage_analysis):
```yaml
# More models = stronger consensus
target_nodes:
  - model.one
  - model.two
  - model.three
  - model.four  # Add more

# Require all to succeed
require_all_success: true

# Or allow some to fail
require_all_success: false
```

### Changing Loop Behavior

**In Loop Workflows** (iterative_refinement, multi_stage_analysis):
```yaml
# Fixed iterations
condition_type: fixed_count
fixed_count: 3

# Or condition-based
condition_type: until_condition
condition_expression: "quality > 0.9"

# Or success-based
condition_type: until_success
```

## Common Modification Scenarios

### Scenario 1: Need Faster Responses
1. Use smaller models (`qwen2.5:0.5b` instead of `3b`)
2. Reduce loop iterations
3. Remove quality gates
4. Use expression gates instead of LLM gates

### Scenario 2: Need Better Quality
1. Use larger models (`qwen2.5:7b` or `14b`)
2. Increase loop iterations
3. Add more quality checks
4. Use parallel consensus pattern

### Scenario 3: Need Lower Temperature (More Deterministic)
```yaml
config:
  temperature: 0.1  # Very deterministic
```

### Scenario 4: Need Higher Temperature (More Creative)
```yaml
config:
  temperature: 0.9  # Very creative
```

## Error Handling Strategies

### Graceful Degradation
```yaml
# In fanout nodes
require_all_success: false

# In loop nodes
continue_on_error: true

# In transform nodes
stop_on_error: false
```

### Fail Fast
```yaml
# In fanout nodes
fail_fast: true

# In transform nodes
stop_on_error: true
```

## Testing Workflows

### Unit Testing Individual Nodes
Test each node type in isolation before combining.

### Integration Testing
Test complete workflow end-to-end with sample inputs.

### Load Testing
For production use, test with concurrent requests:
```bash
# Simulate 10 concurrent requests
for i in {1..10}; do
  tinyllm run workflow.yaml "test input $i" &
done
wait
```

## Monitoring and Observability

All workflows include metadata tracking:
- **Latency**: Execution time per node
- **Token Usage**: Tokens consumed per model call
- **Route Decisions**: Which paths were taken
- **Quality Scores**: When applicable
- **Iteration Counts**: For loop-based workflows

Access via result metadata:
```python
result.metadata['latency_ms']
result.metadata['tokens']
result.metadata['route']
```

## Production Deployment Checklist

- [ ] Tested with production-like data
- [ ] Set appropriate timeouts
- [ ] Configured error handling
- [ ] Validated model availability
- [ ] Set resource limits (max_iterations, timeout_ms)
- [ ] Added logging/monitoring
- [ ] Documented expected inputs/outputs
- [ ] Load tested under expected volume
- [ ] Configured retry policies
- [ ] Set up alerting for failures

## Further Reading

- [Graph Configuration](../../src/tinyllm/config/graph.py) - Schema definitions
- [Node Implementations](../../src/tinyllm/nodes/) - How each node type works
- [Multi-Domain Example](../../graphs/multi_domain.yaml) - Another complex example

---

**Version**: 1.0  
**Last Updated**: 2024-01-15
