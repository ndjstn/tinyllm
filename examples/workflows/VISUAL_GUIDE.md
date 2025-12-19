# Visual Guide to TinyLLM Workflows

This guide provides ASCII diagrams of each workflow for quick understanding.

---

## 1. QA Pipeline (`qa_pipeline.yaml`)

**Purpose**: Route questions to specialized models and validate quality

```
┌─────────────────────────────────────────────────────────────────┐
│                    QA Pipeline Workflow                         │
└─────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │ entry.main  │
                         └──────┬──────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ router.question_type    │
                   │ (LLM-based routing)     │
                   └─────────┬───────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │factual_qa│       │analytical│       │technical │
   │  model   │       │_qa model │       │_qa model │
   └─────┬────┘       └─────┬────┘       └─────┬────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            ▼
                  ┌────────────────────┐
                  │ gate.quality_check │
                  │  (LLM evaluation)  │
                  └─────────┬──────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
            ┌──────────────┐ ┌───────────────┐
            │exit.success  │ │exit.low_quality│
            └──────────────┘ └───────────────┘
```

**Key Features**:
- 4 specialized models (factual, analytical, creative, technical)
- LLM-based quality gate
- 2 exit paths (success/low-quality)

---

## 2. Parallel Consensus (`parallel_consensus.yaml`)

**Purpose**: Get consensus from multiple models running in parallel

```
┌─────────────────────────────────────────────────────────────────┐
│                 Parallel Consensus Workflow                     │
└─────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │ entry.main  │
                         └──────┬──────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   fanout.models         │
                   │ (Parallel execution)    │
                   │ Strategy: majority_vote │
                   └─────────┬───────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ primary  │       │secondary │       │ tertiary │
   │  model   │       │  model   │       │  model   │
   │ (temp:.5)│       │ (temp:.3)│       │ (temp:.7)│
   └─────┬────┘       └─────┬────┘       └─────┬────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            │
                  (Automatic aggregation by fanout)
                            │
                            ▼
                ┌──────────────────────────┐
                │ transform.format_consensus│
                └───────────┬───────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │exit.success  │
                    └──────────────┘
```

**Key Features**:
- 3 models with different temperatures (diversity)
- Parallel execution (fastest completion)
- Majority vote aggregation
- Formatted consensus output

---

## 3. Iterative Refinement (`iterative_refinement.yaml`)

**Purpose**: Iteratively improve content until quality threshold met

```
┌─────────────────────────────────────────────────────────────────┐
│              Iterative Refinement Workflow                      │
└─────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │ entry.main  │
                         └──────┬──────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   loop.refine           │
                   │ Until: quality > 0.8    │
                   │ Max iterations: 5       │
                   └─────────┬───────────────┘
                             │
                   ╔═════════▼═══════════╗
                   ║  ┌──────────────┐   ║
                   ║  │model.refiner │   ║ (Loop Body)
                   ║  │ Generate or  │   ║
                   ║  │ improve      │   ║
                   ║  └──────┬───────┘   ║
                   ║         │           ║
                   ║         ▼           ║
                   ║    Evaluate         ║
                   ║   Quality Score     ║
                   ║         │           ║
                   ║    ┌────┴─────┐     ║
                   ║    │ Score    │     ║
                   ║    │ < 0.8?   │     ║
                   ║    └────┬─────┘     ║
                   ║         │           ║
                   ╚═════════╬═══════════╝
                             │ (Exit loop when quality > 0.8
                             │  or max iterations reached)
                             ▼
                   ┌──────────────────┐
                   │ model.evaluator  │
                   └────────┬─────────┘
                            │
                            ▼
                ┌──────────────────────────┐
                │transform.extract_final   │
                └───────────┬──────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │gate.check_success│
                   └────────┬─────────┘
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
            ┌──────────────┐  ┌─────────────────┐
            │exit.success  │  │exit.needs_work  │
            └──────────────┘  └─────────────────┘
```

**Key Features**:
- Loop with condition-based termination
- Self-evaluation (quality score)
- Up to 5 refinement iterations
- Post-loop evaluation and extraction

---

## 4. Data Processing (`data_processing.yaml`)

**Purpose**: ETL pipeline with validation

```
┌─────────────────────────────────────────────────────────────────┐
│                  Data Processing Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │ entry.main  │
                         │ (Raw JSON)  │
                         └──────┬──────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │transform.parse_json     │
                   │ (Validate structure)    │
                   └─────────┬───────────────┘
                             │
                             ▼
                   ┌─────────────────────────┐
                   │transform.extract_fields │
                   │ (Get 'data' field)      │
                   └─────────┬───────────────┘
                             │
                             ▼
                   ┌─────────────────────────┐
                   │   model.validator       │
                   │ (LLM checks validity)   │
                   └─────────┬───────────────┘
                             │
                             ▼
                   ┌─────────────────────────┐
                   │  gate.validation_check  │
                   └─────────┬───────────────┘
                             │
                     ┌───────┴────────┐
                     │                │
                     ▼                ▼
        ┌─────────────────────┐  ┌──────────────┐
        │transform.extract_   │  │exit.invalid_ │
        │validated            │  │data          │
        └──────┬──────────────┘  └──────────────┘
               │
               ▼
        ┌──────────────────┐
        │transform.format_ │
        │output            │
        └──────┬───────────┘
               │
               ▼
        ┌──────────────┐
        │exit.success  │
        └──────────────┘
```

**Key Features**:
- Chain of 4 transform nodes
- LLM-based validation
- Conditional routing (valid/invalid paths)
- Professional output formatting

---

## 5. Multi-Stage Analysis (`multi_stage_analysis.yaml`)

**Purpose**: Comprehensive analysis combining all patterns

```
┌─────────────────────────────────────────────────────────────────┐
│            Multi-Stage Analysis Pipeline (Complex)              │
└─────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │ entry.main  │
                         └──────┬──────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   router.domain         │
                   └─────────┬───────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌────────┐         ┌─────────┐        ┌─────────┐
    │technical│        │business │        │research │
    │ fanout │        │ fanout  │        │ fanout  │
    └────┬───┘        └────┬────┘        └────┬────┘
         │                 │                   │
    ┌────┼────┬────┐  ┌────┼────┬────┐   ┌────┼────┐
    │    │    │    │  │    │    │    │   │    │    │
    ▼    ▼    ▼    │  ▼    ▼    ▼    │   ▼    ▼    │
  arch  sec  perf  │ strat ops  mkt  │ quant qual │
  model model model│ model model model│ model model│
    │    │    │    │  │    │    │    │   │    │    │
    └────┼────┴────┘  └────┼────┴────┘   └────┼────┘
         │                 │                   │
         │    (Fanout aggregates results)      │
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ loop.refine_    │
                  │ analysis        │
                  │ (2 iterations)  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ transform.      │
                  │ format_report   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ gate.quality_   │
                  │ check           │
                  └────────┬────────┘
                           │
                   ┌───────┴────────┐
                   │                │
                   ▼                ▼
           ┌──────────────┐  ┌─────────────┐
           │exit.success  │  │exit.quality_│
           └──────────────┘  │failed       │
                             └─────────────┘
```

**Key Features**:
- Router → 3 domains
- Each domain fans out to 2-3 specialists (9 total models)
- All results aggregated
- Loop refines synthesis (2 passes)
- Transform formats as report
- Gate validates final quality

**Complexity**:
- 19 nodes total
- Combines routing, fanout, loops, transforms, gates
- Parallel + iterative execution
- Multiple quality checks

---

## Pattern Summary

### Sequential Pattern
```
A → B → C → D
```
Used in: data_processing (transform chain)

### Branching Pattern
```
     ┌→ B
A → Router
     └→ C
```
Used in: qa_pipeline, multi_stage_analysis

### Parallel Pattern
```
     ┌→ B ┐
A → Fanout→ Aggregate → D
     └→ C ┘
```
Used in: parallel_consensus, multi_stage_analysis

### Loop Pattern
```
A → ╔═══════╗ → B
    ║ Loop  ║
    ║ Body  ║
    ╚═══════╝
```
Used in: iterative_refinement, multi_stage_analysis

### Gate Pattern
```
       ┌→ Success
A → Gate
       └→ Failure
```
Used in: All workflows (except parallel_consensus)

---

## Decision Matrix

| Need | Use Workflow | Pattern |
|------|-------------|---------|
| Route by type | qa_pipeline | Router |
| Multiple opinions | parallel_consensus | Fanout |
| Improve quality | iterative_refinement | Loop |
| Process data | data_processing | Transform chain |
| Complete analysis | multi_stage_analysis | All patterns |

---

## Execution Flow Symbols

```
┌────┐  Single node
└────┘

┌────────┐
│        │  Node with description
│        │
└────────┘

╔════════╗
║        ║  Repeated section (loop body)
╚════════╝

→  Sequential flow
├  Branch point
└  Merge point
```

---

**Last Updated**: 2024-01-15
