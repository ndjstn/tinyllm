# TinyLLM Roadmap

> **Note**: This project was developed in December 2025. All completed phases were implemented in a single development sprint to prove the concept. Future phases represent genuine planned work.

This document outlines the development phases for TinyLLM. Each phase builds on the previous and has clear completion criteria.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DEVELOPMENT PHASES                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 0: Foundation          ████████████████████  COMPLETE ✓      │
│  Phase 1: Core Engine         ████████████████████  COMPLETE ✓      │
│  Phase 2: Tools               ████████████████████  COMPLETE ✓      │
│  Phase 3: Routing             ████████████████████  COMPLETE ✓      │
│  Phase 4: Grading             ████████████████████  COMPLETE ✓      │
│  Phase 5: Self-Improvement    ████████████████████  COMPLETE ✓      │
│  Phase 6: Memory              ████████████████████  COMPLETE ✓      │
│  Phase 7: Advanced Features   ████████████████████  COMPLETE ✓      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Foundation

**Goal**: Basic infrastructure that everything else builds on.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Config Loader | `src/tinyllm/config/loader.py` | ✅ |
| System Config Model | `src/tinyllm/config/system.py` | ✅ |
| Model Config Model | `src/tinyllm/config/models.py` | ✅ |
| Graph Config Model | `src/tinyllm/config/graph.py` | ✅ |
| Message Types | `src/tinyllm/core/message.py` | ✅ |
| Ollama Client | `src/tinyllm/models/client.py` | ✅ |
| CLI Skeleton | `src/tinyllm/cli.py` | ✅ |

### Acceptance Criteria

```bash
# Config loading works
python -c "from tinyllm.config import load_config; c = load_config('config/')"

# Ollama client works
python -c "from tinyllm.models import OllamaClient; import asyncio; asyncio.run(OllamaClient().generate('hello', 'qwen2.5:0.5b'))"

# CLI responds
tinyllm --version
tinyllm doctor
```

---

## Phase 1: Core Engine

**Goal**: Graph structure and basic execution.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Base Node | `src/tinyllm/core/node.py` | ✅ |
| Node Registry | `src/tinyllm/core/registry.py` | ✅ |
| Graph Structure | `src/tinyllm/core/graph.py` | ✅ |
| Graph Builder | `src/tinyllm/core/builder.py` | ✅ |
| Executor | `src/tinyllm/core/executor.py` | ✅ |
| Trace Recorder | `src/tinyllm/core/trace.py` | ✅ |

### Acceptance Criteria

```bash
# Can execute a simple graph
tinyllm run "hello world"

# Trace is recorded
tinyllm run --trace "test"
```

---

## Phase 2: Tools

**Goal**: Tool system with initial implementations.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Tool Base Class | `src/tinyllm/tools/base.py` | ✅ |
| Tool Registry | `src/tinyllm/tools/registry.py` | ✅ |
| Calculator | `src/tinyllm/tools/calculator.py` | ✅ |
| Code Executor | `src/tinyllm/tools/code_executor.py` | ✅ |
| Sandbox | `src/tinyllm/tools/sandbox.py` | ✅ |

### Acceptance Criteria

```bash
# Calculator works
tinyllm tool calculator "2 + 2"  # → 4

# Code executor works (sandboxed)
tinyllm tool code "print('hello')"  # → hello
```

---

## Phase 3: Routing & Specialists

**Goal**: Router nodes and specialist model nodes.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Router Node | `src/tinyllm/nodes/router.py` | ✅ |
| Model Node | `src/tinyllm/nodes/model.py` | ✅ |
| Gate Node | `src/tinyllm/nodes/gate.py` | ✅ |
| Prompt Loader | `src/tinyllm/prompts/loader.py` | ✅ |
| Task Classifier Prompt | `prompts/routing/task_classifier.yaml` | ✅ |
| Specialist Prompts | `prompts/specialists/*.yaml` | ✅ |

### Acceptance Criteria

```bash
# Router correctly classifies
tinyllm run "write a function"  # → routed to code specialist
tinyllm run "what is 2+2"  # → routed to math specialist

# Quality gate works
tinyllm run --trace "test"  # Shows gate pass/fail
```

---

## Phase 4: Grading System

**Goal**: LLM-as-judge evaluation and metrics.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Grade Models | `src/tinyllm/grading/models.py` | ✅ |
| Rule-Based Evaluator | `src/tinyllm/grading/rules.py` | ✅ |
| LLM Judge | `src/tinyllm/grading/llm_judge.py` | ✅ |
| Metrics Tracker | `src/tinyllm/grading/metrics.py` | ✅ |
| Failure Forensics | `src/tinyllm/grading/forensics.py` | ✅ |
| Reward Models | `src/tinyllm/grading/reward.py` | ✅ |

### Acceptance Criteria

```bash
# Can grade an output
tinyllm grade --input "2+2" --output "4" --expected "4"

# Metrics are tracked
tinyllm metrics show --node specialist.math

# Failures are categorized
tinyllm failures analyze --last 100
```

---

## Phase 5: Self-Improvement

**Goal**: Recursive expansion and pruning.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Expansion Models | `src/tinyllm/expansion/models.py` | ✅ |
| Pattern Analyzer | `src/tinyllm/expansion/analyzer.py` | ✅ |
| Strategy Generator | `src/tinyllm/expansion/strategies.py` | ✅ |
| Expansion Engine | `src/tinyllm/expansion/engine.py` | ✅ |
| Version Control | `src/tinyllm/expansion/versioning.py` | ✅ |
| Node Spawning | `src/tinyllm/expansion/spawning.py` | ✅ |
| Node Merging | `src/tinyllm/expansion/merging.py` | ✅ |
| Adaptive Pruning | `src/tinyllm/expansion/pruning.py` | ✅ |

### Acceptance Criteria

```bash
# Expansion is triggered
# (After running 100+ requests with >30% failure rate on a node)
tinyllm expansion status  # Shows pending expansions

# Graph versioning
tinyllm graph versions     # List versions
tinyllm graph save "msg"   # Save new version
tinyllm graph rollback v1  # Rollback
```

---

## Phase 6: Memory System

**Goal**: Short-term and long-term memory for conversational context.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Memory Models | `src/tinyllm/memory/models.py` | ✅ |
| Short-Term Memory | `src/tinyllm/memory/stm.py` | ✅ |
| Long-Term Memory | `src/tinyllm/memory/ltm.py` | ✅ |
| Memory Store | `src/tinyllm/memory/store.py` | ✅ |

### Acceptance Criteria

```bash
# Interactive chat with memory
tinyllm chat

# Memory persists across conversation
# Context is retrieved for prompts
```

---

## Phase 7: Advanced Features

**Goal**: Parallel execution, iterative workflows, and data transformation.

**Status**: ✅ Complete

### Deliverables

| Component | File | Status |
|-----------|------|--------|
| Fanout Node | `src/tinyllm/nodes/fanout.py` | ✅ |
| Loop Node | `src/tinyllm/nodes/loop.py` | ✅ |
| Transform Node | `src/tinyllm/nodes/transform.py` | ✅ |
| Fanout Tests | `tests/unit/test_fanout.py` | ✅ |
| Loop Tests | `tests/unit/test_loop.py` | ✅ |
| Transform Tests | `tests/unit/test_transform.py` | ✅ |

### Features

**FanoutNode** - Parallel execution with 4 aggregation strategies:
- FIRST_SUCCESS: Return first successful result
- ALL: Collect all results
- MAJORITY_VOTE: Return most common answer
- BEST_SCORE: Return highest scored result

**LoopNode** - Iterative workflows with 4 termination conditions:
- FIXED_COUNT: Run N iterations
- UNTIL_SUCCESS: Loop until success
- UNTIL_CONDITION: Loop until condition met
- WHILE_CONDITION: Loop while condition holds

**TransformNode** - Data transformation with 13 transform types:
- Text: uppercase, lowercase, strip, truncate
- JSON: extract, wrap, parse, stringify
- Regex: extract, replace
- Structural: template, split, join

---

## Future Phases

### Phase 8: Observability

- Web dashboard
- Real-time metrics
- Alert system

### Phase 9: Scaling

- Multi-GPU support
- Distributed execution
- Model caching

### Phase 10: Ecosystem

- Plugin system
- Community prompts
- Benchmark suite

---

## Milestone Targets

| Milestone | Description | Status |
|-----------|-------------|--------|
| **M1: First Response** | Execute a query end-to-end | ✅ Complete |
| **M2: Tool Use** | Model calls a tool successfully | ✅ Complete |
| **M3: Smart Routing** | Router picks correct specialist | ✅ Complete |
| **M4: Quality Gates** | Outputs are validated | ✅ Complete |
| **M5: Graded Outputs** | LLM judges outputs | ✅ Complete |
| **M6: First Expansion** | Node expands automatically | ✅ Complete |
| **M7: Memory System** | Context persists | ✅ Complete |
| **M8: Advanced Workflows** | Parallel execution and loops | ✅ Complete |

---

## Test Coverage

| Component | Tests |
|-----------|-------|
| Core (messages, graph, executor) | 40+ |
| Nodes (router, model, gate) | 30+ |
| Tools (calculator, code executor) | 35+ |
| Grading (rules, metrics, forensics) | 50+ |
| Expansion (analyzer, strategies) | 35+ |
| Memory (STM, LTM, store) | 40+ |
| Versioning | 25 |
| Spawning | 53+ |
| Merging | 35+ |
| Pruning | 76+ |
| Fanout Node | 45 |
| Loop Node | 44 |
| Transform Node | 41 |
| **Total** | **587 tests** |

---

## Release Schedule

| Version | Content | Status |
|---------|---------|--------|
| 0.1.0 | Phase 0-1 complete | ✅ Released |
| 0.2.0 | Phase 2-3 complete | ✅ Released |
| 0.3.0 | Phase 4 complete | ✅ Released |
| 0.4.0 | Phase 5-6 complete | ✅ Released |
| 0.5.0 | Phase 7 complete | ✅ Released |
| 1.0.0 | Stable release | 🚧 In Progress |

---

*Last updated: 2025-12-18*
