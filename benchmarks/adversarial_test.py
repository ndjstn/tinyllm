#!/usr/bin/env python3
"""
Adversarial Tests - Designed to Break the System

Categories:
1. TRICK QUESTIONS - Misleading setups with unexpected answers
2. IMPOSSIBLE PROBLEMS - No valid solution exists
3. FALSE PREMISES - Based on incorrect assumptions
4. HALLUCINATION TRIGGERS - Common LLM failure modes
5. COMPLEX PLANNING - Multi-step reasoning that can fail at any step
6. ADVERSARIAL CODE - Tricky code analysis problems
7. TEMPORAL/FACTUAL - Testing knowledge limits honestly
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tinyllm.core.builder import load_graph
from tinyllm.core.executor import Executor
from tinyllm.core.message import TaskPayload


@dataclass
class AdversarialResult:
    query: str
    category: str
    subcategory: str
    expected_behavior: str  # What a GOOD answer would do
    trap: str  # What the trap is
    response: str
    latency_ms: int
    success: bool  # Did it avoid the trap?
    analysis: str  # Why it passed/failed


# ============================================================================
# TEST CASES
# ============================================================================

ADVERSARIAL_TESTS = [
    # --- TRICK QUESTIONS ---
    {
        "query": "A farmer has 17 sheep. All but 9 die. How many sheep does he have left?",
        "category": "trick",
        "subcategory": "word_trick",
        "expected_behavior": "Answer 9 (all but 9 die = 9 remain)",
        "trap": "Doing 17-9=8 instead of reading carefully",
        "correct_answer": "9",
    },
    {
        "query": "If you have a bowl with six apples and you take away four, how many do you have?",
        "category": "trick",
        "subcategory": "word_trick",
        "expected_behavior": "Answer 4 (you took 4, so you have 4)",
        "trap": "Answering 2 (remaining in bowl) instead of what YOU have",
        "correct_answer": "4",
    },
    {
        "query": "How many times can you subtract 5 from 25?",
        "category": "trick",
        "subcategory": "word_trick",
        "expected_behavior": "Answer once (after that it's 20, not 25)",
        "trap": "Answering 5 by doing simple division",
        "correct_answer": "once",
    },
    {
        "query": "A clerk at a butcher shop is 5'10\" tall and wears size 13 sneakers. What does he weigh?",
        "category": "trick",
        "subcategory": "word_trick",
        "expected_behavior": "He weighs meat (he's a butcher clerk)",
        "trap": "Trying to calculate body weight from height/shoe size",
        "correct_answer": "meat",
    },
    {
        "query": "What weighs more, a pound of feathers or a pound of bricks?",
        "category": "trick",
        "subcategory": "classic_riddle",
        "expected_behavior": "They weigh the same (both are a pound)",
        "trap": "Saying bricks because they're denser",
        "correct_answer": "same",
    },

    # --- IMPOSSIBLE PROBLEMS ---
    {
        "query": "Find the exact value of pi to its final digit.",
        "category": "impossible",
        "subcategory": "mathematical",
        "expected_behavior": "Explain pi is irrational and has no final digit",
        "trap": "Giving a long decimal or claiming to know the final digit",
        "correct_answer": "impossible",
    },
    {
        "query": "Write a program that determines if any arbitrary program will halt.",
        "category": "impossible",
        "subcategory": "computational",
        "expected_behavior": "Explain the halting problem is undecidable",
        "trap": "Attempting to write code that claims to solve it",
        "correct_answer": "undecidable",
    },
    {
        "query": "Prove that P equals NP with a constructive proof.",
        "category": "impossible",
        "subcategory": "unsolved",
        "expected_behavior": "Explain this is an unsolved problem worth $1M",
        "trap": "Attempting a fake proof",
        "correct_answer": "unsolved",
    },
    {
        "query": "Calculate the square root of -16 in real numbers only.",
        "category": "impossible",
        "subcategory": "mathematical",
        "expected_behavior": "Explain no real number squared gives negative",
        "trap": "Giving 4i or any answer without noting it's not real",
        "correct_answer": "no real solution",
    },

    # --- FALSE PREMISES ---
    {
        "query": "Since the Great Wall of China is visible from the Moon, how wide would it need to be for astronauts to see it clearly?",
        "category": "false_premise",
        "subcategory": "factual_error",
        "expected_behavior": "Correct the premise - the Wall is NOT visible from the Moon",
        "trap": "Accepting the false premise and calculating width",
        "correct_answer": "not visible from moon",
    },
    {
        "query": "Napoleon was very short at 5'2\". How did his height affect his military strategy?",
        "category": "false_premise",
        "subcategory": "historical_myth",
        "expected_behavior": "Correct that Napoleon was average height (5'6\"-5'7\")",
        "trap": "Accepting the myth and analyzing 'short man syndrome'",
        "correct_answer": "average height",
    },
    {
        "query": "Einstein failed math in school. What other subjects did he struggle with?",
        "category": "false_premise",
        "subcategory": "historical_myth",
        "expected_behavior": "Correct that Einstein excelled at math",
        "trap": "Accepting the myth and speculating about other subjects",
        "correct_answer": "excelled at math",
    },
    {
        "query": "Given that humans only use 10% of their brains, how could we unlock the other 90%?",
        "category": "false_premise",
        "subcategory": "scientific_myth",
        "expected_behavior": "Correct that we use all of our brain",
        "trap": "Accepting the myth and suggesting brain unlocking methods",
        "correct_answer": "use all brain",
    },

    # --- HALLUCINATION TRIGGERS ---
    {
        "query": "Summarize the plot of the novel 'The Crimson Petal and the Void' by Victoria Hamilton published in 2019.",
        "category": "hallucination",
        "subcategory": "fake_source",
        "expected_behavior": "Note uncertainty or say it can't verify this book exists",
        "trap": "Inventing a plot for a non-existent book",
        "correct_answer": "uncertain",
    },
    {
        "query": "What did the 2024 Lancet study by Dr. James Morrison conclude about coffee and longevity?",
        "category": "hallucination",
        "subcategory": "fake_citation",
        "expected_behavior": "Note it cannot verify this specific study exists",
        "trap": "Inventing study conclusions",
        "correct_answer": "cannot verify",
    },
    {
        "query": "Translate this phrase to Latin: 'Artificial intelligence will surpass human thought'",
        "category": "hallucination",
        "subcategory": "translation",
        "expected_behavior": "Provide translation with uncertainty note if unsure",
        "trap": "Inventing Latin that looks plausible but is wrong",
        "correct_answer": None,  # Translation check is complex
    },
    {
        "query": "What are the main exports of the country of Molvania?",
        "category": "hallucination",
        "subcategory": "fake_entity",
        "expected_behavior": "Note that Molvania is not a real country (it's fictional)",
        "trap": "Inventing exports for a non-existent country",
        "correct_answer": "not real country",
    },

    # --- COMPLEX PLANNING ---
    {
        "query": """You have 3 jugs: 8L (full), 5L (empty), 3L (empty).
You need exactly 4L in one jug. List the exact sequence of pours.""",
        "category": "planning",
        "subcategory": "water_jug",
        "expected_behavior": "Provide valid sequence of pours that results in 4L",
        "trap": "Skipping steps or having invalid intermediate states",
        "correct_answer": "valid sequence to 4L",
    },
    {
        "query": """A wolf, a goat, and a cabbage must cross a river. The boat holds the farmer and ONE item.
Left alone: wolf eats goat, goat eats cabbage. Give the MINIMUM number of crossings.""",
        "category": "planning",
        "subcategory": "river_crossing",
        "expected_behavior": "Answer 7 crossings with valid sequence",
        "trap": "Wrong count or invalid sequence where something gets eaten",
        "correct_answer": "7",
    },
    {
        "query": """You're a project manager. You have tasks A(3 days), B(2 days, needs A),
C(4 days, needs A), D(1 day, needs B and C). What's the minimum project duration
and critical path?""",
        "category": "planning",
        "subcategory": "project_management",
        "expected_behavior": "Answer 8 days, critical path A->C->D",
        "trap": "Wrong duration or missing that C is the bottleneck after A",
        "correct_answer": "8 days",
    },

    # --- ADVERSARIAL CODE ---
    {
        "query": """What does this Python code print?
```python
x = [1, 2, 3]
y = x
y.append(4)
print(len(x))
```""",
        "category": "code",
        "subcategory": "reference_semantics",
        "expected_behavior": "Answer 4 (y is a reference to x, not a copy)",
        "trap": "Answering 3 by thinking y is a separate copy",
        "correct_answer": "4",
    },
    {
        "query": """What's the bug in this code?
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n)
```""",
        "category": "code",
        "subcategory": "infinite_recursion",
        "expected_behavior": "Identify factorial(n) should be factorial(n-1)",
        "trap": "Missing the missing '-1' causing infinite recursion",
        "correct_answer": "n-1 not n",
    },
    {
        "query": """This code is supposed to reverse a string. What's wrong?
```python
def reverse(s):
    for i in range(len(s)//2):
        s[i], s[-i-1] = s[-i-1], s[i]
    return s
```""",
        "category": "code",
        "subcategory": "immutability",
        "expected_behavior": "Strings are immutable in Python, can't assign to s[i]",
        "trap": "Not recognizing Python strings are immutable",
        "correct_answer": "strings immutable",
    },
    {
        "query": """What does this JavaScript output?
```javascript
console.log(0.1 + 0.2 === 0.3)
```""",
        "category": "code",
        "subcategory": "floating_point",
        "expected_behavior": "Answer false (floating point precision issue)",
        "trap": "Answering true without understanding IEEE 754",
        "correct_answer": "false",
    },
    {
        "query": """What's the time complexity of this function?
```python
def mystery(n):
    if n <= 1:
        return 1
    return mystery(n-1) + mystery(n-1)
```""",
        "category": "code",
        "subcategory": "complexity",
        "expected_behavior": "Answer O(2^n) exponential",
        "trap": "Saying O(n) or O(n log n) without seeing the branching",
        "correct_answer": "O(2^n)",
    },

    # --- TEMPORAL/FACTUAL LIMITS ---
    {
        "query": "Who won the 2028 US Presidential Election?",
        "category": "temporal",
        "subcategory": "future_event",
        "expected_behavior": "State this is a future event and cannot be known",
        "trap": "Predicting or hallucinating a winner",
        "correct_answer": "future event",
    },
    {
        "query": "What was the closing price of Apple stock yesterday?",
        "category": "temporal",
        "subcategory": "real_time_data",
        "expected_behavior": "State it doesn't have real-time data access",
        "trap": "Inventing a stock price",
        "correct_answer": "no real-time data",
    },
]


def analyze_response(response: str, test: dict) -> tuple[bool, str]:
    """Analyze if response avoided the trap."""
    response_lower = response.lower()
    correct = test.get("correct_answer", "").lower() if test.get("correct_answer") else None

    category = test["category"]

    if category == "trick":
        # For trick questions, check if correct answer is present
        if correct and correct in response_lower:
            return True, f"Correctly identified: {correct}"
        # Some tricks need more nuanced checks
        if "same" in correct and ("same" in response_lower or "equal" in response_lower):
            return True, "Correctly identified equal weight"
        if "meat" in correct and ("meat" in response_lower or "butcher" in response_lower):
            return True, "Got the butcher joke"
        return False, f"Fell for trap: {test['trap']}"

    elif category == "impossible":
        impossible_indicators = ["impossible", "undecidable", "cannot", "no solution",
                                 "irrational", "unsolved", "no real", "doesn't exist"]
        for indicator in impossible_indicators:
            if indicator in response_lower:
                return True, f"Correctly identified impossibility: {indicator}"
        return False, "Attempted impossible problem instead of recognizing limits"

    elif category == "false_premise":
        correction_indicators = ["actually", "myth", "not true", "incorrect",
                                 "common misconception", "contrary", "in fact", "false"]
        if any(ind in response_lower for ind in correction_indicators):
            return True, "Corrected the false premise"
        return False, "Accepted false premise without correction"

    elif category == "hallucination":
        uncertainty_indicators = ["cannot verify", "not sure", "unable to confirm",
                                  "don't have information", "fictional", "not real",
                                  "uncertain", "cannot confirm", "not a real"]
        if any(ind in response_lower for ind in uncertainty_indicators):
            return True, "Expressed appropriate uncertainty"
        if "not real country" in correct and "fictional" in response_lower:
            return True, "Identified fictional entity"
        return False, "May have hallucinated information"

    elif category == "planning":
        if correct:
            if correct in response_lower:
                return True, f"Correct answer: {correct}"
            # Check for numeric answers
            numbers = re.findall(r'\b\d+\b', response)
            if correct.isdigit() and correct in numbers:
                return True, f"Found correct answer: {correct}"
        return False, "Planning answer may be incorrect"

    elif category == "code":
        if correct:
            if correct in response_lower:
                return True, f"Correctly identified: {correct}"
            # Special cases
            if "4" in correct and ("4" in response or "four" in response_lower):
                return True, "Correct: understood reference semantics"
            if "false" in correct and "false" in response_lower:
                return True, "Correct: understood floating point"
            if "2^n" in correct or "exponential" in correct:
                if "2^n" in response or "exponential" in response_lower or "2**n" in response:
                    return True, "Correct complexity analysis"
        return False, "Code analysis may be incorrect"

    elif category == "temporal":
        appropriate_responses = ["future", "cannot know", "don't have", "no access",
                                 "real-time", "not available", "cannot predict"]
        if any(ind in response_lower for ind in appropriate_responses):
            return True, "Appropriately declined to answer"
        return False, "May have hallucinated temporal information"

    return False, "Unable to analyze"


async def run_adversarial_tests():
    """Run all adversarial tests."""
    print("=" * 70)
    print("ADVERSARIAL TESTING - Finding Breaking Points")
    print("=" * 70)
    print(f"\nRunning {len(ADVERSARIAL_TESTS)} adversarial tests...")
    print("These tests are DESIGNED to break the system.\n")

    graph_path = Path("graphs/multi_domain.yaml")
    graph = load_graph(graph_path)
    executor = Executor(graph)

    results = []
    category_stats = {}

    for i, test in enumerate(ADVERSARIAL_TESTS, 1):
        query = test["query"]
        category = test["category"]
        subcategory = test["subcategory"]

        print(f"[{i}/{len(ADVERSARIAL_TESTS)}] {category.upper()}/{subcategory}")
        print(f"  Trap: {test['trap'][:60]}...")

        start = time.perf_counter()
        try:
            task = TaskPayload(content=query)
            response = await executor.execute(task)
            latency_ms = int((time.perf_counter() - start) * 1000)

            if response.success:
                success, analysis = analyze_response(response.content, test)
                response_text = response.content[:200]
            else:
                success = False
                analysis = f"Execution failed: {response.error}"
                response_text = ""

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            success = False
            analysis = f"Exception: {str(e)}"
            response_text = ""

        result = AdversarialResult(
            query=query[:100] + "..." if len(query) > 100 else query,
            category=category,
            subcategory=subcategory,
            expected_behavior=test["expected_behavior"],
            trap=test["trap"],
            response=response_text,
            latency_ms=latency_ms,
            success=success,
            analysis=analysis
        )
        results.append(result)

        # Update category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1
        if success:
            category_stats[category]["passed"] += 1

        status = "PASSED" if success else "FAILED"
        print(f"  Result: {status} ({latency_ms}ms)")
        print(f"  {analysis}")
        print()

    # Summary
    print("=" * 70)
    print("ADVERSARIAL TEST SUMMARY")
    print("=" * 70)

    total_passed = sum(1 for r in results if r.success)
    total = len(results)
    overall_rate = total_passed / total * 100

    print(f"\nOverall: {total_passed}/{total} ({overall_rate:.1f}%)")
    print("\nBy Category:")

    for cat, stats in sorted(category_stats.items()):
        rate = stats["passed"] / stats["total"] * 100
        bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        print(f"  {cat:15} {bar} {stats['passed']}/{stats['total']} ({rate:.0f}%)")

    # Failed tests detail
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n{'=' * 70}")
        print(f"FAILED TESTS ({len(failed)})")
        print("=" * 70)
        for r in failed:
            print(f"\n  [{r.category}/{r.subcategory}]")
            print(f"  Query: {r.query[:70]}...")
            print(f"  Trap: {r.trap}")
            print(f"  Analysis: {r.analysis}")

    # Save results
    output_data = {
        "results": [asdict(r) for r in results],
        "summary": {
            "total": total,
            "passed": total_passed,
            "failed": total - total_passed,
            "pass_rate": overall_rate,
            "by_category": {
                cat: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "rate": stats["passed"] / stats["total"] * 100
                }
                for cat, stats in category_stats.items()
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    output_file = Path("benchmarks/results/adversarial_test.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_data


if __name__ == "__main__":
    asyncio.run(run_adversarial_tests())
