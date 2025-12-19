# TinyLLM Roadmap

This document outlines the development phases for TinyLLM. Each phase builds on the previous and has clear completion criteria.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DEVELOPMENT PHASES                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 0: Foundation          ████████░░░░░░░░░░░░  IN PROGRESS     │
│  Phase 1: Core Engine         ░░░░░░░░░░░░░░░░░░░░  Not Started     │
│  Phase 2: Tools               ░░░░░░░░░░░░░░░░░░░░  Not Started     │
│  Phase 3: Routing             ░░░░░░░░░░░░░░░░░░░░  Not Started     │
│  Phase 4: Grading             ░░░░░░░░░░░░░░░░░░░░  Not Started     │
│  Phase 5: Self-Improvement    ░░░░░░░░░░░░░░░░░░░░  Not Started     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Foundation

**Goal**: Basic infrastructure that everything else builds on.

**Status**: 🚧 In Progress

### Deliverables

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| Config Loader | `src/tinyllm/config/loader.py` | ⬜ | #TBD |
| System Config Model | `src/tinyllm/config/system.py` | ⬜ | #TBD |
| Model Config Model | `src/tinyllm/config/models.py` | ⬜ | #TBD |
| Graph Config Model | `src/tinyllm/config/graph.py` | ⬜ | #TBD |
| Message Types | `src/tinyllm/core/message.py` | ⬜ | #TBD |
| Ollama Client | `src/tinyllm/models/client.py` | ⬜ | #TBD |
| CLI Skeleton | `src/tinyllm/cli.py` | ⬜ | #TBD |

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

### Dependencies

- None (this is the foundation)

---

## Phase 1: Core Engine

**Goal**: Graph structure and basic execution.

**Status**: ⬜ Not Started

### Deliverables

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| Base Node | `src/tinyllm/core/node.py` | ⬜ | #TBD |
| Node Registry | `src/tinyllm/core/registry.py` | ⬜ | #TBD |
| Graph Structure | `src/tinyllm/core/graph.py` | ⬜ | #TBD |
| Graph Builder | `src/tinyllm/core/builder.py` | ⬜ | #TBD |
| Executor | `src/tinyllm/core/executor.py` | ⬜ | #TBD |
| Trace Recorder | `src/tinyllm/core/trace.py` | ⬜ | #TBD |

### Acceptance Criteria

```bash
# Can execute a simple graph
tinyllm run "hello world"

# Trace is recorded
tinyllm run --trace "test" | jq .trace
```

### Dependencies

- Phase 0 complete

---

## Phase 2: Tools

**Goal**: Tool system with initial implementations.

**Status**: ⬜ Not Started

### Deliverables

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| Tool Base Class | `src/tinyllm/tools/base.py` | ⬜ | #TBD |
| Tool Registry | `src/tinyllm/tools/registry.py` | ⬜ | #TBD |
| Calculator | `src/tinyllm/tools/calculator.py` | ⬜ | #TBD |
| Code Executor | `src/tinyllm/tools/code_executor.py` | ⬜ | #TBD |
| Sandbox | `src/tinyllm/tools/sandbox.py` | ⬜ | #TBD |

### Acceptance Criteria

```bash
# Calculator works
tinyllm tool calculator "2 + 2"  # → 4

# Code executor works (sandboxed)
tinyllm tool code "print('hello')"  # → hello
```

### Dependencies

- Phase 1 complete

---

## Phase 3: Routing & Specialists

**Goal**: Router nodes and specialist model nodes.

**Status**: ⬜ Not Started

### Deliverables

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| Router Node | `src/tinyllm/nodes/router.py` | ⬜ | #TBD |
| Model Node | `src/tinyllm/nodes/model.py` | ⬜ | #TBD |
| Gate Node | `src/tinyllm/nodes/gate.py` | ⬜ | #TBD |
| Prompt Loader | `src/tinyllm/prompts/loader.py` | ⬜ | #TBD |
| Task Classifier Prompt | `prompts/routing/task_classifier.yaml` | ⬜ | #TBD |
| Code Specialist Prompt | `prompts/specialists/code.yaml` | ⬜ | #TBD |
| Math Specialist Prompt | `prompts/specialists/math.yaml` | ⬜ | #TBD |
| General Specialist Prompt | `prompts/specialists/general.yaml` | ⬜ | #TBD |

### Acceptance Criteria

```bash
# Router correctly classifies
tinyllm run "write a function"  # → routed to code specialist
tinyllm run "what is 2+2"  # → routed to math specialist

# Quality gate works
tinyllm run --verbose "test"  # Shows gate pass/fail
```

### Dependencies

- Phase 2 complete

---

## Phase 4: Grading System

**Goal**: LLM-as-judge evaluation and metrics.

**Status**: ⬜ Not Started

### Deliverables

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| Judge Interface | `src/tinyllm/grading/judge.py` | ⬜ | #TBD |
| Rule-Based Evaluator | `src/tinyllm/grading/rules.py` | ⬜ | #TBD |
| LLM Judge | `src/tinyllm/grading/llm_judge.py` | ⬜ | #TBD |
| Metrics Tracker | `src/tinyllm/grading/metrics.py` | ⬜ | #TBD |
| Failure Forensics | `src/tinyllm/grading/forensics.py` | ⬜ | #TBD |
| Judge Prompts | `prompts/grading/*.yaml` | ⬜ | #TBD |

### Acceptance Criteria

```bash
# Can grade an output
tinyllm grade --input "2+2" --output "4" --expected "4"

# Metrics are tracked
tinyllm metrics show --node specialist.math

# Failures are categorized
tinyllm failures analyze --last 100
```

### Dependencies

- Phase 3 complete

---

## Phase 5: Self-Improvement

**Goal**: Recursive expansion and pruning.

**Status**: ⬜ Not Started

### Deliverables

| Component | File | Status | Issue |
|-----------|------|--------|-------|
| Expansion Triggers | `src/tinyllm/expansion/triggers.py` | ⬜ | #TBD |
| Graph Mutations | `src/tinyllm/expansion/mutations.py` | ⬜ | #TBD |
| Pruning System | `src/tinyllm/expansion/pruning.py` | ⬜ | #TBD |
| A/B Testing | `src/tinyllm/expansion/ab_test.py` | ⬜ | #TBD |
| Version Control | `src/tinyllm/expansion/versioning.py` | ⬜ | #TBD |
| Expansion Prompts | `prompts/meta/expansion_*.yaml` | ⬜ | #TBD |

### Acceptance Criteria

```bash
# Expansion is triggered
# (After running 100+ requests with >30% failure rate on a node)
tinyllm expansion status  # Shows pending expansions

# Manual expansion
tinyllm expansion trigger --node specialist.math

# Rollback works
tinyllm graph rollback --version 1.0.0
```

### Dependencies

- Phase 4 complete

---

## Future Phases

### Phase 6: Memory System

- Short-term memory (conversation)
- Long-term memory (vector store)
- Memory-augmented retrieval

### Phase 7: Advanced Features

- Parallel execution (fanout)
- Iterative workflows (loops)
- Dynamic graph modification

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

| Milestone | Description | Target |
|-----------|-------------|--------|
| **M1: First Response** | Execute a query end-to-end | Phase 1 |
| **M2: Tool Use** | Model calls a tool successfully | Phase 2 |
| **M3: Smart Routing** | Router picks correct specialist | Phase 3 |
| **M4: Quality Gates** | Outputs are validated | Phase 3 |
| **M5: Graded Outputs** | LLM judges outputs | Phase 4 |
| **M6: First Expansion** | Node expands automatically | Phase 5 |
| **M7: Stable System** | 24h+ without manual intervention | Phase 5 |

---

## How to Contribute

1. **Find an issue** for your skill level and interest
2. **Claim it** by commenting
3. **Implement** following our [Contributing Guide](CONTRIBUTING.md)
4. **Submit PR** with tests

```bash
# Find open issues
gh issue list --label "help-wanted"

# Filter by phase
gh issue list --label "phase-0"
```

---

## Release Schedule

| Version | Content | Status |
|---------|---------|--------|
| 0.1.0 | Phase 0-1 complete | 🚧 In Progress |
| 0.2.0 | Phase 2-3 complete | ⬜ Planned |
| 0.3.0 | Phase 4 complete | ⬜ Planned |
| 0.4.0 | Phase 5 complete | ⬜ Planned |
| 1.0.0 | Stable release | ⬜ Planned |

---

*Last updated: 2024-12-18*
