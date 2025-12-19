# TinyLLM Benchmarks

Performance benchmarks for the TinyLLM system.

## Running Benchmarks

```bash
uv run python benchmarks/run_benchmarks.py
```

## Latest Results

**Run Date:** 2025-12-18

| Metric | Value |
|--------|-------|
| Total Queries | 20 |
| Success Rate | 100% |
| Avg Latency | 7,506ms |
| Min Latency | 830ms |
| Max Latency | 12,680ms |
| Avg Response | 2,567 chars |

### Latency by Category

| Category | Avg | Min | Max |
|----------|-----|-----|-----|
| Math | 4,845ms | 3,483ms | 9,819ms |
| Code | 8,646ms | 7,403ms | 12,187ms |
| General | 5,955ms | 830ms | 10,301ms |
| Compound | 10,578ms | 9,258ms | 12,680ms |

## Visualizations

### Benchmark Dashboard
![Dashboard](results/benchmark_dashboard.png)

### Latency by Category (Box Plot)
![Latency Box Plot](results/latency_by_category.png)

### Latency Timeline
![Timeline](results/latency_timeline.png)

### Response Length Distribution
![Response Length](results/response_length_dist.png)

## Query Categories

- **Math**: Simple arithmetic, algebra, percentages
- **Code**: Function writing, file I/O, data structures
- **General**: Science, history, explanations
- **Compound**: Code + Math combined queries

## Hardware

Benchmarks were run on:
- AMD Ryzen 7 3700X
- 128GB RAM
- 2x RTX 3060 (24GB VRAM total)
- Ollama with qwen2.5:3b model

## Output Files

- `results/benchmark_results.json` - Detailed per-query results
- `results/benchmark_summary.json` - Summary statistics
- `results/*.png` - Visualization charts
