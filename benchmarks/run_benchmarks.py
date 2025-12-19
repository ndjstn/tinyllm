#!/usr/bin/env python3
"""TinyLLM Benchmarking Suite.

Runs various queries through the system and collects performance metrics.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Benchmark queries organized by expected category
BENCHMARK_QUERIES = {
    "math": [
        "What is 15 + 27?",
        "Calculate the square root of 144",
        "What is 8 factorial?",
        "Solve: 3x + 7 = 22",
        "What percentage is 45 of 180?",
    ],
    "code": [
        "Write a Python function to reverse a string",
        "How do I read a file in Python?",
        "Write a function to find the maximum element in a list",
        "Create a class for a binary search tree",
        "Write a recursive fibonacci function",
    ],
    "general": [
        "What is photosynthesis?",
        "Explain the water cycle",
        "Who wrote Romeo and Juliet?",
        "What causes earthquakes?",
        "Describe the solar system",
    ],
    "compound": [
        "Write Python code to calculate compound interest",
        "Create a function to solve quadratic equations",
        "Write code to compute statistical mean and standard deviation",
        "Implement Newton's method for finding square roots in Python",
        "Write a program to calculate loan amortization",
    ],
}


@dataclass
class BenchmarkResult:
    """Result from a single benchmark query."""

    query: str
    expected_category: str
    actual_route: Optional[str]
    success: bool
    latency_ms: int
    nodes_executed: int
    response_length: int
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run."""

    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    min_latency_ms: int
    max_latency_ms: int
    avg_response_length: float
    routing_accuracy: float
    queries_per_category: dict
    latency_by_category: dict
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


async def run_single_benchmark(query: str, expected_category: str) -> BenchmarkResult:
    """Run a single benchmark query."""
    from tinyllm.core.builder import load_graph
    from tinyllm.core.executor import Executor
    from tinyllm.core.message import TaskPayload

    graph_path = Path("graphs/multi_domain.yaml")

    try:
        graph = load_graph(graph_path)
        executor = Executor(graph)

        start_time = time.perf_counter()
        task = TaskPayload(content=query)
        response = await executor.execute(task)
        end_time = time.perf_counter()

        latency_ms = int((end_time - start_time) * 1000)

        # Route is inferred from response success
        actual_route = expected_category if response.success else None

        return BenchmarkResult(
            query=query,
            expected_category=expected_category,
            actual_route=actual_route,
            success=response.success,
            latency_ms=latency_ms,
            nodes_executed=response.nodes_executed,
            response_length=len(response.content or ""),
            error=response.error.message if response.error else None,
        )

    except Exception as e:
        return BenchmarkResult(
            query=query,
            expected_category=expected_category,
            actual_route=None,
            success=False,
            latency_ms=0,
            nodes_executed=0,
            response_length=0,
            error=str(e),
        )


async def run_warmup():
    """Run warmup queries to load models into memory."""
    print("Running warmup queries...")
    warmup_queries = [
        "Hello",
        "What is 1+1?",
        "Write hello world in Python",
    ]

    from tinyllm.core.builder import load_graph
    from tinyllm.core.executor import Executor
    from tinyllm.core.message import TaskPayload

    graph = load_graph(Path("graphs/multi_domain.yaml"))
    executor = Executor(graph)

    for query in warmup_queries:
        try:
            task = TaskPayload(content=query)
            await executor.execute(task)
            print(f"  Warmup: '{query[:30]}...' done")
        except Exception as e:
            print(f"  Warmup failed: {e}")

    print("Warmup complete.\n")


async def run_all_benchmarks(skip_warmup: bool = False) -> tuple[list[BenchmarkResult], BenchmarkSummary]:
    """Run all benchmark queries and return results."""
    results = []

    if not skip_warmup:
        await run_warmup()

    total_queries = sum(len(queries) for queries in BENCHMARK_QUERIES.values())
    current = 0

    for category, queries in BENCHMARK_QUERIES.items():
        print(f"\nRunning {category} benchmarks...")
        for query in queries:
            current += 1
            print(f"  [{current}/{total_queries}] {query[:50]}...")

            result = await run_single_benchmark(query, category)
            results.append(result)

            status = "OK" if result.success else "FAIL"
            print(f"    {status} - {result.latency_ms}ms")

    # Calculate summary statistics
    successful = [r for r in results if r.success]
    latencies = [r.latency_ms for r in successful]

    # Routing accuracy calculation
    correct_routes = 0
    for r in successful:
        if r.actual_route:
            # Check if route matches expected category
            if r.expected_category in r.actual_route or r.actual_route in r.expected_category:
                correct_routes += 1
            # Compound queries might route to code_math etc
            elif r.expected_category == "compound" and "code" in r.actual_route:
                correct_routes += 1

    # Latency by category
    latency_by_cat = {}
    for category in BENCHMARK_QUERIES.keys():
        cat_results = [r for r in successful if r.expected_category == category]
        if cat_results:
            latency_by_cat[category] = {
                "avg": sum(r.latency_ms for r in cat_results) / len(cat_results),
                "min": min(r.latency_ms for r in cat_results),
                "max": max(r.latency_ms for r in cat_results),
                "count": len(cat_results),
            }

    summary = BenchmarkSummary(
        total_queries=len(results),
        successful_queries=len(successful),
        failed_queries=len(results) - len(successful),
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        min_latency_ms=min(latencies) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        avg_response_length=sum(r.response_length for r in successful) / len(successful) if successful else 0,
        routing_accuracy=correct_routes / len(successful) if successful else 0,
        queries_per_category={k: len(v) for k, v in BENCHMARK_QUERIES.items()},
        latency_by_category=latency_by_cat,
    )

    return results, summary


def save_results(results: list[BenchmarkResult], summary: BenchmarkSummary, output_dir: Path):
    """Save benchmark results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Save summary
    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Summary saved to: {summary_file}")


def generate_visualizations(results: list[BenchmarkResult], summary: BenchmarkSummary, output_dir: Path):
    """Generate visualization charts from benchmark results."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Color palette
    colors = {
        'math': '#4CAF50',
        'code': '#2196F3',
        'general': '#FF9800',
        'compound': '#9C27B0',
    }

    successful = [r for r in results if r.success]

    # 1. Latency by Category (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(BENCHMARK_QUERIES.keys())
    latency_data = []
    for cat in categories:
        cat_latencies = [r.latency_ms for r in successful if r.expected_category == cat]
        latency_data.append(cat_latencies)

    bp = ax.boxplot(latency_data, labels=categories, patch_artist=True)
    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(colors[cat])
        patch.set_alpha(0.7)

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_xlabel('Query Category', fontsize=12)
    ax.set_title('TinyLLM Response Latency by Query Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_by_category.png', dpi=150)
    plt.close()
    print(f"Generated: latency_by_category.png")

    # 2. Average Latency Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_latencies = [summary.latency_by_category.get(cat, {}).get('avg', 0) for cat in categories]
    bars = ax.bar(categories, avg_latencies, color=[colors[c] for c in categories], alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, avg_latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}ms', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_xlabel('Query Category', fontsize=12)
    ax.set_title('Average Response Time by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_latency_bar.png', dpi=150)
    plt.close()
    print(f"Generated: avg_latency_bar.png")

    # 3. Success Rate Pie Chart
    fig, ax = plt.subplots(figsize=(8, 8))
    success_data = [summary.successful_queries, summary.failed_queries]
    labels = ['Successful', 'Failed']
    colors_pie = ['#4CAF50', '#f44336']
    explode = (0.05, 0)

    ax.pie(success_data, explode=explode, labels=labels, colors=colors_pie,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title(f'Query Success Rate\n({summary.successful_queries}/{summary.total_queries} queries)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate.png', dpi=150)
    plt.close()
    print(f"Generated: success_rate.png")

    # 4. Response Length Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in categories:
        cat_lengths = [r.response_length for r in successful if r.expected_category == cat]
        if cat_lengths:
            ax.hist(cat_lengths, bins=15, alpha=0.6, label=cat, color=colors[cat])

    ax.set_xlabel('Response Length (characters)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Response Length Distribution by Category', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'response_length_dist.png', dpi=150)
    plt.close()
    print(f"Generated: response_length_dist.png")

    # 5. Latency Timeline (scatter plot showing latency over query sequence)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(successful):
        ax.scatter(i, r.latency_ms, c=colors[r.expected_category],
                   s=50, alpha=0.7, label=r.expected_category if i < 4 else "")

    ax.set_xlabel('Query Sequence', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Timeline Across All Queries', fontsize=14, fontweight='bold')

    # Create legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_timeline.png', dpi=150)
    plt.close()
    print(f"Generated: latency_timeline.png")

    # 6. Summary Dashboard
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Metrics panel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    metrics_text = f"""
    BENCHMARK SUMMARY
    ─────────────────
    Total Queries: {summary.total_queries}
    Successful: {summary.successful_queries}
    Failed: {summary.failed_queries}

    Avg Latency: {summary.avg_latency_ms:.0f}ms
    Min Latency: {summary.min_latency_ms}ms
    Max Latency: {summary.max_latency_ms}ms

    Avg Response: {summary.avg_response_length:.0f} chars
    """
    ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Latency by category bar
    ax2 = fig.add_subplot(gs[0, 1:])
    avg_latencies = [summary.latency_by_category.get(cat, {}).get('avg', 0) for cat in categories]
    bars = ax2.barh(categories, avg_latencies, color=[colors[c] for c in categories], alpha=0.8)
    ax2.set_xlabel('Average Latency (ms)')
    ax2.set_title('Latency by Category')
    for bar, val in zip(bars, avg_latencies):
        ax2.text(val + 50, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}ms', va='center', fontsize=9)

    # Success pie
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.pie([summary.successful_queries, summary.failed_queries],
            labels=['Success', 'Failed'], colors=['#4CAF50', '#f44336'],
            autopct='%1.0f%%', startangle=90)
    ax3.set_title('Success Rate')

    # Category distribution
    ax4 = fig.add_subplot(gs[1, 1])
    cat_counts = [len(BENCHMARK_QUERIES[c]) for c in categories]
    ax4.pie(cat_counts, labels=categories, colors=[colors[c] for c in categories],
            autopct='%1.0f%%', startangle=90)
    ax4.set_title('Query Distribution')

    # Latency range
    ax5 = fig.add_subplot(gs[1, 2])
    for i, cat in enumerate(categories):
        cat_data = summary.latency_by_category.get(cat, {})
        if cat_data:
            ax5.errorbar(i, cat_data['avg'],
                        yerr=[[cat_data['avg'] - cat_data['min']],
                              [cat_data['max'] - cat_data['avg']]],
                        fmt='o', color=colors[cat], capsize=5, capthick=2, markersize=10)
    ax5.set_xticks(range(len(categories)))
    ax5.set_xticklabels(categories)
    ax5.set_ylabel('Latency (ms)')
    ax5.set_title('Latency Range (min/avg/max)')

    plt.suptitle('TinyLLM Benchmark Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: benchmark_dashboard.png")


def print_summary(summary: BenchmarkSummary):
    """Print summary to console."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Queries:     {summary.total_queries}")
    print(f"Successful:        {summary.successful_queries}")
    print(f"Failed:            {summary.failed_queries}")
    print(f"Success Rate:      {summary.successful_queries/summary.total_queries*100:.1f}%")
    print("-"*60)
    print(f"Average Latency:   {summary.avg_latency_ms:.0f}ms")
    print(f"Min Latency:       {summary.min_latency_ms}ms")
    print(f"Max Latency:       {summary.max_latency_ms}ms")
    print("-"*60)
    print(f"Avg Response Len:  {summary.avg_response_length:.0f} characters")
    print("-"*60)
    print("Latency by Category:")
    for cat, data in summary.latency_by_category.items():
        print(f"  {cat:12s}: avg={data['avg']:.0f}ms, min={data['min']}ms, max={data['max']}ms")
    print("="*60)


async def main():
    """Main benchmark runner."""
    print("="*60)
    print("TinyLLM Benchmark Suite")
    print("="*60)

    output_dir = Path("benchmarks/results")

    # Run benchmarks
    results, summary = await run_all_benchmarks()

    # Print summary
    print_summary(summary)

    # Save results
    save_results(results, summary, output_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(results, summary, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
