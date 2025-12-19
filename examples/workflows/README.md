# TinyLLM Workflow Examples

This directory contains real-world workflow examples demonstrating TinyLLM's capabilities for building complex LLM-powered applications.

## Available Workflows

### 1. QA Pipeline (`qa_pipeline.yaml`)

**Purpose**: Question-answering system with intelligent routing and quality validation

**Flow**: Entry → Router → Specialized Models → Quality Gate → Exit

**Key Features**:
- Routes questions by type (factual, analytical, creative, technical)
- Specialized models for each question type
- LLM-based quality gate to validate answers
- Multiple exit paths (success/low-quality)

**Use Cases**:
- Customer support systems
- Knowledge base assistants
- Educational Q&A platforms

**Example Input**:
```
"What is the time complexity of quicksort?"
→ Routes to technical_qa → Validates quality → Returns answer
```

---

### 2. Parallel Consensus (`parallel_consensus.yaml`)

**Purpose**: Multi-model consensus using parallel execution and majority voting

**Flow**: Entry → Fanout (3 models) → Aggregate (majority vote) → Exit

**Key Features**:
- Runs 3 models in parallel with different configurations
- Uses majority voting to find consensus answer
- Reduces single-model bias
- Formatted output with consensus indication

**Use Cases**:
- Critical decision validation
- High-stakes question answering
- Bias reduction in AI responses

**Example Input**:
```
"Is this contract clause enforceable?"
→ 3 models analyze → Majority vote determines answer → Returns consensus
```

---

### 3. Iterative Refinement (`iterative_refinement.yaml`)

**Purpose**: Iterative quality improvement through feedback loops

**Flow**: Entry → Loop (generate → evaluate → refine until quality > 0.8) → Exit

**Key Features**:
- Loop node with condition-based termination
- Self-evaluating quality scores
- Iterative improvement up to 5 attempts
- Extraction of final polished content

**Use Cases**:
- Content writing and editing
- Report generation
- Documentation creation
- Creative writing that benefits from revision

**Example Input**:
```
"Write a product description for an eco-friendly water bottle"
→ Generate → Score 0.6 → Refine → Score 0.85 → Exit with polished content
```

---

### 4. Data Processing (`data_processing.yaml`)

**Purpose**: ETL pipeline for structured data transformation and validation

**Flow**: Entry → Parse JSON → Extract Fields → Validate → Format → Exit

**Key Features**:
- Chained transform nodes for data processing
- JSON parsing and field extraction
- LLM-based data validation
- Conditional routing based on validation results
- Professional formatting of output

**Use Cases**:
- API response processing
- Data validation pipelines
- Format conversion (JSON → formatted output)
- Data quality assurance

**Example Input**:
```json
{
  "data": {
    "id": 123,
    "name": "Sample Record",
    "values": [1, 2, 3],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```
→ Parse → Extract → Validate → Format → Return processed data

---

### 5. Multi-Stage Analysis (`multi_stage_analysis.yaml`)

**Purpose**: Comprehensive analytical pipeline combining multiple workflow patterns

**Flow**: Entry → Router → Fanout → Loop → Transform → Gate → Exit

**Key Features**:
- Domain-based routing (technical/business/research)
- Parallel multi-perspective analysis (2-3 specialists per domain)
- Iterative synthesis and refinement
- Quality validation gate
- Professional report formatting

**Use Cases**:
- Comprehensive system reviews
- Multi-faceted business analysis
- Research synthesis
- Strategic planning documents

**Example Input**:
```
"Analyze our microservices architecture for scalability issues"
→ Routes to technical domain
→ 3 specialists analyze (architecture, security, performance)
→ Synthesize insights (2 refinement passes)
→ Format as report
→ Validate quality
→ Return comprehensive analysis
```

---

## Workflow Patterns Demonstrated

### Routing Patterns
- **Single-label routing** (`qa_pipeline.yaml`): One route per request
- **Multi-domain routing** (`multi_stage_analysis.yaml`): Routes based on domain classification

### Execution Patterns
- **Sequential execution**: Most workflows follow linear paths
- **Parallel execution** (`parallel_consensus.yaml`): Fanout to multiple nodes simultaneously
- **Iterative execution** (`iterative_refinement.yaml`): Loop until condition met

### Aggregation Patterns
- **Majority vote** (`parallel_consensus.yaml`): Consensus from multiple models
- **All results** (`multi_stage_analysis.yaml`): Combine all parallel outputs

### Quality Control
- **LLM-based gates** (`qa_pipeline.yaml`): Model evaluates quality
- **Expression gates** (`iterative_refinement.yaml`): Condition-based checks
- **Validation loops** (`iterative_refinement.yaml`): Repeat until quality threshold

### Data Transformation
- **JSON operations** (`data_processing.yaml`): Parse, extract, stringify
- **Text operations**: Template, format, extract
- **Regex operations**: Pattern matching and replacement

---

## Running Workflows

### Prerequisites
1. TinyLLM installed and configured
2. Ollama running with required models:
   - `qwen2.5:0.5b` (for routing and gates)
   - `qwen2.5:3b` (for main processing)

### Basic Usage

```bash
# Run a workflow
tinyllm run examples/workflows/qa_pipeline.yaml "What is recursion?"

# Specify custom input
tinyllm run examples/workflows/parallel_consensus.yaml "Is AI consciousness possible?"

# Process data file
cat data.json | tinyllm run examples/workflows/data_processing.yaml
```

### Testing Workflows

```bash
# Validate workflow structure
tinyllm validate examples/workflows/qa_pipeline.yaml

# Dry run (validate without execution)
tinyllm run --dry-run examples/workflows/multi_stage_analysis.yaml "test input"
```

---

## Customization Tips

### Adjusting Model Selection
Change the `model` field in node configs:
```yaml
config:
  model: qwen2.5:7b  # Use larger model for better quality
  temperature: 0.5
```

### Modifying Thresholds
Adjust quality gates and loop conditions:
```yaml
# Loop until higher quality
condition_expression: "last_result.get('metadata', {}).get('quality_score', 0) > 0.9"

# More iterations allowed
max_iterations: 10
```

### Adding New Routes
Extend router configurations:
```yaml
routes:
  - name: new_category
    description: Description of when to use this route
    target: model.new_specialist
    priority: 2
```

### Chaining Transforms
Add more transformation stages:
```yaml
transforms:
  - type: strip
    params: {}
  - type: lowercase
    params: {}
  - type: regex_replace
    params:
      pattern: "\s+"
      replacement: " "
```

---

## Performance Considerations

### Parallel Execution
- Fanout nodes execute targets in parallel by default
- Set `parallel: false` for sequential execution
- Use `timeout_ms` to prevent hanging on slow models

### Loop Optimization
- Set reasonable `max_iterations` to prevent infinite loops
- Use `timeout_ms` to bound execution time
- Set `collect_results: false` if you only need final output

### Model Selection
- Use smaller models (`0.5b`) for routing and gates
- Use larger models (`3b`, `7b`) for main tasks
- Balance quality vs. speed based on use case

---

## Advanced Patterns

### Conditional Branching
```yaml
- id: gate.check_complexity
  type: gate
  config:
    mode: expression
    conditions:
      - name: simple
        expression: "len(content) < 100"
        target: model.simple_handler
      - name: complex
        expression: "len(content) >= 100"
        target: model.complex_handler
```

### Multi-Level Routing
```yaml
# First router: Domain
router.domain → technical/business/creative

# Second router (per domain): Subdomain
router.technical_subdomain → code/infrastructure/security
```

### Error Handling
```yaml
config:
  continue_on_error: true  # For loops
  require_all_success: false  # For fanouts
  stop_on_error: false  # For transforms
```

---

## Workflow Design Best Practices

1. **Start Simple**: Begin with linear flows, add complexity as needed
2. **Use Routing**: Direct tasks to specialized models for better quality
3. **Validate Quality**: Add gates to ensure output meets standards
4. **Iterate Wisely**: Use loops for tasks that benefit from refinement
5. **Parallelize**: Use fanout for independent tasks to improve speed
6. **Transform Data**: Clean and format data at appropriate stages
7. **Handle Errors**: Set appropriate error handling for production use

---

## Troubleshooting

### Workflow Fails to Load
- Validate YAML syntax
- Check node IDs match pattern: `[a-z][a-z0-9_\.]*`
- Ensure all edges reference existing nodes

### Execution Errors
- Verify required models are available in Ollama
- Check loop conditions are valid Python expressions
- Ensure transform parameters are complete

### Performance Issues
- Reduce `max_iterations` in loops
- Use smaller models for non-critical tasks
- Enable `parallel: true` in fanout nodes
- Adjust `timeout_ms` settings

---

## Contributing

To add new workflow examples:

1. Follow existing naming conventions
2. Include comprehensive comments
3. Add metadata section with description
4. Document use cases and example inputs
5. Test with various inputs before submitting

---

## Additional Resources

- [TinyLLM Documentation](../../README.md)
- [Graph Configuration Schema](../../src/tinyllm/config/graph.py)
- [Node Types Reference](../../src/tinyllm/nodes/)
- [Example Graph](../../graphs/multi_domain.yaml)

---

**Last Updated**: 2024-01-15
