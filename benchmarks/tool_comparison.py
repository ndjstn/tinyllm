#!/usr/bin/env python3
"""
Tool-Assisted Benchmark Comparison

Compares LLM performance WITH and WITHOUT tool assistance to answer:
"Do small tools help agents work faster or slow them down?"

Hypothesis: Tools should HELP because they:
1. Offload computation (calculator does math faster than LLM)
2. Provide accurate results (no hallucinated calculations)
3. Free up LLM context for reasoning
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tinyllm.core.builder import load_graph
from tinyllm.core.executor import Executor
from tinyllm.core.message import TaskPayload


@dataclass
class ComparisonResult:
    query: str
    category: str

    # With tools
    tool_success: bool
    tool_latency_ms: int
    tool_response_length: int
    tool_correct: bool | None  # Manual verification needed for some

    # Without tools (pure LLM)
    pure_success: bool
    pure_latency_ms: int
    pure_response_length: int
    pure_correct: bool | None

    # Comparison
    latency_diff_ms: int  # positive = tools are faster
    tool_advantage: str  # "faster", "slower", "similar"


# Test queries with known correct answers for verification
COMPARISON_QUERIES = [
    # Calculator-friendly math
    {
        "query": "What is 847 * 923?",
        "category": "arithmetic",
        "expected": "781481",
    },
    {
        "query": "Calculate 15% of 8,450",
        "category": "percentage",
        "expected": "1267.5",
    },
    {
        "query": "What is the square root of 2401?",
        "category": "arithmetic",
        "expected": "49",
    },
    {
        "query": "Calculate (125 + 375) / 10 - 23",
        "category": "arithmetic",
        "expected": "27",
    },
    {
        "query": "What is 2^10?",
        "category": "exponent",
        "expected": "1024",
    },

    # Questions where tools don't help (baseline)
    {
        "query": "What causes thunder?",
        "category": "general",
        "expected": None,  # No specific answer to verify
    },
    {
        "query": "Explain what recursion is in programming",
        "category": "general",
        "expected": None,
    },

    # Complex math where calculator helps
    {
        "query": "If I invest $5000 at 7% annual interest compounded monthly for 5 years, what is my final balance?",
        "category": "compound_math",
        "expected": "7088",  # Approximately
    },
    {
        "query": "A store has a 30% off sale. If something costs $89.99, what is the sale price?",
        "category": "percentage",
        "expected": "62.99",
    },
    {
        "query": "Calculate the area of a circle with radius 7.5",
        "category": "geometry",
        "expected": "176.7",  # Approximately
    },
]


def check_answer(response: str, expected: str | None) -> bool | None:
    """Check if response contains the expected answer."""
    if expected is None:
        return None  # Can't verify general questions

    # Look for the expected value in the response
    # Handle various number formats
    expected_clean = expected.replace(",", "")
    response_lower = response.lower().replace(",", "")

    # Check for exact match or close match
    if expected_clean in response_lower:
        return True

    # For approximate values, check if close
    try:
        exp_val = float(expected_clean)
        # Find numbers in response
        import re
        numbers = re.findall(r'[\d,]+\.?\d*', response)
        for num_str in numbers:
            try:
                num = float(num_str.replace(",", ""))
                if abs(num - exp_val) < exp_val * 0.05:  # Within 5%
                    return True
            except ValueError:
                continue
    except ValueError:
        pass

    return False


async def run_query(executor: Executor, query: str) -> tuple[bool, int, int, str]:
    """Run a single query and return (success, latency_ms, response_length, response)."""
    start = time.perf_counter()

    try:
        task = TaskPayload(content=query)
        response = await executor.execute(task)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if response.success:
            return True, latency_ms, len(response.content), response.content
        else:
            return False, latency_ms, 0, ""
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"  Error: {e}")
        return False, latency_ms, 0, ""


async def run_comparison():
    """Run comparison benchmarks."""
    print("=" * 60)
    print("Tool-Assisted Benchmark Comparison")
    print("=" * 60)
    print("\nComparing performance WITH and WITHOUT tool assistance")
    print("-" * 60)

    # Load the graph with tools enabled
    graph_path = Path("graphs/multi_domain.yaml")
    graph = load_graph(graph_path)
    executor = Executor(graph)

    results = []

    for i, test in enumerate(COMPARISON_QUERIES, 1):
        query = test["query"]
        category = test["category"]
        expected = test["expected"]

        print(f"\n[{i}/{len(COMPARISON_QUERIES)}] {category.upper()}")
        print(f"  Query: {query[:60]}...")

        # Run with tools (our normal graph has calculator tool)
        print("  Running with tools...", end=" ", flush=True)
        tool_success, tool_latency, tool_length, tool_response = await run_query(executor, query)
        tool_correct = check_answer(tool_response, expected) if tool_success else False
        print(f"{tool_latency}ms", "✓" if tool_correct else "○")

        # Run pure LLM query (request explicit reasoning without tools)
        # We add instruction to NOT use tools and solve directly
        pure_query = f"Solve this WITHOUT using any external tools or calculators. Show your work step by step: {query}"
        print("  Running pure LLM...", end=" ", flush=True)
        pure_success, pure_latency, pure_length, pure_response = await run_query(executor, pure_query)
        pure_correct = check_answer(pure_response, expected) if pure_success else False
        print(f"{pure_latency}ms", "✓" if pure_correct else "○")

        # Calculate comparison
        latency_diff = pure_latency - tool_latency  # Positive = tools faster

        if abs(latency_diff) < 500:  # Within 500ms
            advantage = "similar"
        elif latency_diff > 0:
            advantage = "tools_faster"
        else:
            advantage = "pure_faster"

        result = ComparisonResult(
            query=query,
            category=category,
            tool_success=tool_success,
            tool_latency_ms=tool_latency,
            tool_response_length=tool_length,
            tool_correct=tool_correct,
            pure_success=pure_success,
            pure_latency_ms=pure_latency,
            pure_response_length=pure_length,
            pure_correct=pure_correct,
            latency_diff_ms=latency_diff,
            tool_advantage=advantage,
        )
        results.append(result)

        print(f"  Diff: {latency_diff:+d}ms ({advantage})")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    math_results = [r for r in results if r.category != "general"]
    general_results = [r for r in results if r.category == "general"]

    # Math queries analysis
    if math_results:
        print("\n### Math/Calculation Queries ###")
        tool_faster_count = sum(1 for r in math_results if r.tool_advantage == "tools_faster")
        pure_faster_count = sum(1 for r in math_results if r.tool_advantage == "pure_faster")
        similar_count = sum(1 for r in math_results if r.tool_advantage == "similar")

        avg_tool_latency = sum(r.tool_latency_ms for r in math_results) / len(math_results)
        avg_pure_latency = sum(r.pure_latency_ms for r in math_results) / len(math_results)
        avg_diff = sum(r.latency_diff_ms for r in math_results) / len(math_results)

        tool_correct = sum(1 for r in math_results if r.tool_correct is True)
        pure_correct = sum(1 for r in math_results if r.pure_correct is True)

        print(f"  Tools faster: {tool_faster_count}/{len(math_results)}")
        print(f"  Pure faster:  {pure_faster_count}/{len(math_results)}")
        print(f"  Similar:      {similar_count}/{len(math_results)}")
        print(f"\n  Avg latency (tools):  {avg_tool_latency:.0f}ms")
        print(f"  Avg latency (pure):   {avg_pure_latency:.0f}ms")
        print(f"  Avg difference:       {avg_diff:+.0f}ms")
        print(f"\n  Correct answers (tools): {tool_correct}/{len(math_results)}")
        print(f"  Correct answers (pure):  {pure_correct}/{len(math_results)}")

    # General queries analysis
    if general_results:
        print("\n### General Knowledge Queries (Baseline) ###")
        avg_tool_latency = sum(r.tool_latency_ms for r in general_results) / len(general_results)
        avg_pure_latency = sum(r.pure_latency_ms for r in general_results) / len(general_results)

        print(f"  Avg latency (tools):  {avg_tool_latency:.0f}ms")
        print(f"  Avg latency (pure):   {avg_pure_latency:.0f}ms")
        print("  (General queries don't use tools, so difference is routing overhead)")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if math_results:
        speed_winner = "Tools" if avg_diff > 0 else "Pure LLM"
        accuracy_winner = "Tools" if tool_correct > pure_correct else ("Pure LLM" if pure_correct > tool_correct else "Tied")

        print(f"\n  Speed winner:    {speed_winner}")
        print(f"  Accuracy winner: {accuracy_winner}")

        if tool_correct > pure_correct:
            print("\n  🎯 Tools improve ACCURACY (calculators don't hallucinate)")
        if avg_diff > 0:
            print("  ⚡ Tools improve SPEED (offload computation)")
        else:
            print("  ⏱️ Pure LLM slightly faster (tool routing overhead)")

    # Save results
    results_data = {
        "results": [
            {
                "query": r.query,
                "category": r.category,
                "tool_latency_ms": r.tool_latency_ms,
                "pure_latency_ms": r.pure_latency_ms,
                "latency_diff_ms": r.latency_diff_ms,
                "tool_correct": r.tool_correct,
                "pure_correct": r.pure_correct,
                "tool_advantage": r.tool_advantage,
            }
            for r in results
        ],
        "summary": {
            "math_queries": {
                "total": len(math_results),
                "tools_faster": sum(1 for r in math_results if r.tool_advantage == "tools_faster"),
                "avg_tool_latency_ms": avg_tool_latency if math_results else 0,
                "avg_pure_latency_ms": avg_pure_latency if math_results else 0,
                "tool_correct": tool_correct if math_results else 0,
                "pure_correct": pure_correct if math_results else 0,
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_file = Path("benchmarks/results/tool_comparison.json")
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return results_data


if __name__ == "__main__":
    asyncio.run(run_comparison())
