#!/usr/bin/env python3
"""Test coverage gate checker.

This script runs tests with coverage and enforces minimum coverage thresholds.
Exits with non-zero status if coverage is below threshold.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_coverage() -> dict:
    """Run pytest with coverage and return coverage report."""
    print("Running tests with coverage...")

    # Run pytest with coverage
    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "--cov=src/tinyllm",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-q",
            "-m",
            "not integration and not slow",  # Skip slow/integration tests
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode not in [0, 5]:  # 0=pass, 5=no tests collected
        print(f"Tests failed with exit code {result.returncode}")
        # Continue to check coverage even if tests fail

    # Load coverage JSON report
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        print("ERROR: coverage.json not found")
        sys.exit(1)

    with open(coverage_file) as f:
        return json.load(f)


def check_coverage_threshold(coverage_data: dict, threshold: float = 80.0) -> bool:
    """Check if coverage meets threshold.

    Args:
        coverage_data: Coverage JSON data
        threshold: Minimum required coverage percentage

    Returns:
        True if coverage meets threshold, False otherwise
    """
    total_coverage = coverage_data["totals"]["percent_covered"]

    print(f"\n{'='*60}")
    print(f"COVERAGE GATE CHECK")
    print(f"{'='*60}")
    print(f"Total Coverage: {total_coverage:.2f}%")
    print(f"Required:       {threshold:.2f}%")
    print(f"{'='*60}")

    if total_coverage >= threshold:
        print(f"✓ PASS: Coverage {total_coverage:.2f}% >= {threshold:.2f}%")
        return True
    else:
        diff = threshold - total_coverage
        print(f"✗ FAIL: Coverage {total_coverage:.2f}% < {threshold:.2f}%")
        print(f"         Need {diff:.2f}% more coverage")
        return False


def print_uncovered_modules(coverage_data: dict, max_show: int = 10):
    """Print modules with lowest coverage.

    Args:
        coverage_data: Coverage JSON data
        max_show: Maximum number of modules to show
    """
    files = coverage_data.get("files", {})

    # Calculate per-file coverage
    file_coverage = []
    for filepath, file_data in files.items():
        summary = file_data.get("summary", {})
        covered = summary.get("covered_lines", 0)
        total = summary.get("num_statements", 0)
        if total > 0:
            pct = (covered / total) * 100
            file_coverage.append((filepath, pct, covered, total))

    # Sort by coverage percentage (lowest first)
    file_coverage.sort(key=lambda x: x[1])

    print(f"\n{'='*60}")
    print(f"MODULES WITH LOWEST COVERAGE (Top {max_show})")
    print(f"{'='*60}")

    for filepath, pct, covered, total in file_coverage[:max_show]:
        # Show relative path from src/
        if "src/tinyllm/" in filepath:
            filepath = filepath.split("src/tinyllm/")[1]
        print(f"{filepath:50s} {pct:6.2f}% ({covered}/{total})")


def main():
    """Main entry point."""
    # Run coverage
    coverage_data = run_coverage()

    # Print uncovered modules
    print_uncovered_modules(coverage_data)

    # Check threshold
    threshold = 80.0  # Match pyproject.toml
    passed = check_coverage_threshold(coverage_data, threshold)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
