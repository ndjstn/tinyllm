# Integration Tests for TinyLLM Workflows

This directory contains comprehensive integration tests for the TinyLLM project, focusing on complete workflows that chain multiple nodes together.

## Overview

The integration tests verify end-to-end execution paths through the graph, testing real-world scenarios with multiple node types working together.

## Test File: test_workflows.py

Total: **21 test cases** across **8 test classes**

### Test Coverage

#### 1. Router → Model → Gate Workflow (2 tests)
- `test_successful_routing_to_model_and_gate_pass`: Verifies query routing through router, model execution, and quality gate passing
- `test_routing_to_wrong_model_gate_fails`: Tests gate rejection when output doesn't meet quality criteria

#### 2. Fanout → Aggregate Workflow (3 tests)
- `test_fanout_all_strategy`: Tests parallel execution with ALL aggregation strategy
- `test_fanout_first_success_strategy`: Tests FIRST_SUCCESS aggregation (early termination)
- `test_fanout_require_all_success`: Tests require_all_success configuration option

#### 3. Loop → Transform Workflow (2 tests)
- `test_loop_fixed_count_with_transform`: Tests fixed iteration loop with transformation
- `test_loop_until_success`: Tests UNTIL_SUCCESS loop condition

#### 4. Full Pipeline Test (1 test)
- `test_full_pipeline_entry_router_model_transform_gate_exit`: Complete end-to-end pipeline testing Entry → Router → Model → Transform → Gate → Exit

#### 5. Edge Cases and Error Handling (6 tests)
- `test_empty_input_handling`: Minimal/empty input handling
- `test_executor_timeout`: Executor timeout with long-running workflows
- `test_max_steps_exceeded`: Max steps limit enforcement
- `test_missing_required_fields`: Entry node validation with missing fields
- `test_transform_with_invalid_json`: Transform failure on invalid JSON
- `test_gate_with_no_matching_conditions`: Gate behavior when no conditions match

#### 6. Complex Multi-Path Workflows (2 tests)
- `test_router_with_multiple_models`: Router directing to different models based on classification
- `test_fanout_with_transforms_and_gate`: Fanout followed by transforms and quality gate

#### 7. Transform Pipelines (3 tests)
- `test_multi_transform_pipeline`: Multiple transforms in sequence (strip, lowercase, truncate)
- `test_regex_transform`: Regex extraction transform
- `test_template_transform`: Template-based transform

#### 8. Performance and Concurrency (2 tests)
- `test_parallel_fanout_performance`: Parallel fanout execution performance
- `test_multiple_concurrent_executions`: Multiple concurrent graph executions

## Key Features Tested

### Node Types
- ✓ Entry nodes (with validation)
- ✓ Exit nodes (success/fallback statuses)
- ✓ Router nodes (single-label routing)
- ✓ Model nodes (LLM invocation)
- ✓ Gate nodes (expression-based conditional branching)
- ✓ Fanout nodes (parallel execution with aggregation)
- ✓ Loop nodes (fixed count, until success)
- ✓ Transform nodes (text manipulation, JSON, regex, templates)

### Execution Patterns
- ✓ Sequential execution
- ✓ Parallel execution (fanout)
- ✓ Iterative execution (loops)
- ✓ Conditional branching (gates)
- ✓ Error handling and recovery
- ✓ Timeout management
- ✓ Max steps enforcement

### Mocking Strategy
- All tests use mocked Ollama client to avoid real LLM calls
- Mock responses are configured per test scenario
- Fixtures provide reusable mock configurations

## Running the Tests

### With pytest (if environment is configured)
```bash
pytest tests/integration/test_workflows.py -v
```

### Note on pytest-asyncio compatibility
The tests use `pytest.mark.asyncio` decorators and are designed to work with pytest-asyncio. If you encounter version compatibility issues between pytest and pytest-asyncio, the tests can still be imported and run programmatically using asyncio.run().

## Test Structure

Each test follows this pattern:
1. **Setup**: Create graph definition with nodes and edges
2. **Build**: Instantiate nodes and add to graph
3. **Execute**: Run task through executor
4. **Assert**: Verify expected outcomes

Example:
```python
@pytest.mark.asyncio
async def test_workflow(self, mock_ollama_client):
    # Setup graph
    graph_def = GraphDefinition(...)

    # Build graph
    graph = Graph(graph_def)
    for node_def in graph_def.nodes:
        # Add nodes...

    # Execute
    executor = Executor(graph)
    response = await executor.execute(task)

    # Assert
    assert response.success is True
```

## Coverage Statistics

- **Success paths**: 15 tests
- **Error/failure paths**: 6 tests
- **Edge cases**: 6 tests
- **Performance tests**: 2 tests

## Future Enhancements

Potential areas for expansion:
- Multi-label router workflows
- Compound route testing
- Tool-enabled model nodes
- Hybrid gate evaluation (LLM + expression)
- More complex loop conditions
- Advanced aggregation strategies
- Streaming responses
- State persistence and recovery
