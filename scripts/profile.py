#!/usr/bin/env python3
"""Profiling script for TinyLLM with flame graph generation.

This script provides comprehensive profiling capabilities including:
- CPU profiling with py-spy
- Memory profiling
- Flame graph generation
- Profile visualization
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Try to import py-spy programmatically
try:
    import py_spy  # type: ignore

    HAS_PY_SPY = True
except ImportError:
    HAS_PY_SPY = False


def run_cpu_profile(
    command: list[str],
    output_file: Path,
    duration: Optional[int] = None,
    rate: int = 100,
    format: str = "flamegraph",
) -> int:
    """Run CPU profiling with py-spy.

    Args:
        command: Command to profile.
        output_file: Output file path.
        duration: Optional duration in seconds.
        rate: Sampling rate in Hz.
        format: Output format (flamegraph, speedscope, raw).

    Returns:
        Exit code.
    """
    if not HAS_PY_SPY:
        print(
            "Error: py-spy is not installed. Install with: pip install py-spy",
            file=sys.stderr,
        )
        print(
            "Alternatively, run: sudo py-spy record -o profile.svg -- python your_script.py",
            file=sys.stderr,
        )
        return 1

    # Build py-spy command
    py_spy_cmd = [
        "py-spy",
        "record",
        "--rate",
        str(rate),
        "--format",
        format,
        "--output",
        str(output_file),
    ]

    if duration:
        py_spy_cmd.extend(["--duration", str(duration)])

    py_spy_cmd.append("--")
    py_spy_cmd.extend(command)

    print(f"Running py-spy profiler...")
    print(f"Command: {' '.join(py_spy_cmd)}")

    try:
        result = subprocess.run(py_spy_cmd, check=True)
        print(f"\nProfile saved to: {output_file}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: py-spy failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(
            "Error: py-spy not found. Install with: pip install py-spy",
            file=sys.stderr,
        )
        return 1


def run_memory_profile(
    command: list[str],
    output_file: Path,
    interval: float = 0.1,
) -> int:
    """Run memory profiling with tracemalloc.

    Args:
        command: Command to profile.
        output_file: Output file path.
        interval: Sampling interval in seconds.

    Returns:
        Exit code.
    """
    # Create a wrapper script that enables memory profiling
    wrapper_script = f"""
import sys
import tracemalloc
import json
import time

# Start memory tracking
tracemalloc.start()

# Import and run the target
try:
    import runpy
    sys.argv = {command}
    runpy.run_path(sys.argv[0], run_name="__main__")
except Exception as e:
    print(f"Error running script: {{e}}", file=sys.stderr)
    sys.exit(1)
finally:
    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()

    # Get top allocations
    top_stats = snapshot.statistics('lineno')[:20]

    # Save results
    results = {{
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
        "top_allocations": [
            {{
                "file": str(stat.traceback),
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count,
            }}
            for stat in top_stats
        ]
    }}

    with open("{output_file}", "w") as f:
        json.dump(results, f, indent=2)

    tracemalloc.stop()
    print(f"\\nMemory profile saved to: {output_file}")
    print(f"Peak memory usage: {{peak / 1024 / 1024:.2f}} MB")
"""

    # Write wrapper script
    wrapper_path = Path("/tmp/tinyllm_memory_profile.py")
    with open(wrapper_path, "w") as f:
        f.write(wrapper_script)

    print(f"Running memory profiler...")

    try:
        result = subprocess.run([sys.executable, str(wrapper_path)], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Memory profiling failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode


def generate_flamegraph_from_profile(
    profile_file: Path,
    output_file: Path,
) -> int:
    """Generate flame graph from profile data.

    Args:
        profile_file: Input profile file (JSON format).
        output_file: Output SVG file.

    Returns:
        Exit code.
    """
    # Check if flamegraph.pl is available
    try:
        subprocess.run(["which", "flamegraph.pl"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(
            "Warning: flamegraph.pl not found. Install from: https://github.com/brendangregg/FlameGraph",
            file=sys.stderr,
        )
        print("Or use py-spy's built-in flamegraph format", file=sys.stderr)
        return 1

    # Read profile data
    try:
        with open(profile_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading profile file: {e}", file=sys.stderr)
        return 1

    # Convert to folded format
    folded_data = []
    for entry in data.get("function_stats", []):
        stack = entry.get("stack", [])
        count = entry.get("count", 0)
        folded_data.append(f"{';'.join(stack)} {count}")

    # Write folded data
    folded_file = profile_file.with_suffix(".folded")
    with open(folded_file, "w") as f:
        f.write("\n".join(folded_data))

    # Generate flame graph
    try:
        with open(output_file, "w") as f:
            subprocess.run(
                ["flamegraph.pl", str(folded_file)],
                stdout=f,
                check=True,
            )
        print(f"Flame graph saved to: {output_file}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error generating flame graph: {e}", file=sys.stderr)
        return e.returncode


def profile_tinyllm_graph(
    graph_file: Path,
    query: str,
    output_dir: Path,
    profile_type: str = "cpu",
) -> int:
    """Profile a TinyLLM graph execution.

    Args:
        graph_file: Path to graph YAML file.
        query: Query to execute.
        output_dir: Output directory for profiles.
        profile_type: Type of profiling (cpu, memory, both).

    Returns:
        Exit code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tinyllm command
    tinyllm_cmd = [
        sys.executable,
        "-m",
        "tinyllm.cli",
        "run",
        "--graph",
        str(graph_file),
        query,
    ]

    exit_code = 0

    if profile_type in ("cpu", "both"):
        cpu_output = output_dir / "cpu_profile.svg"
        result = run_cpu_profile(
            tinyllm_cmd,
            cpu_output,
            format="flamegraph",
        )
        if result != 0:
            exit_code = result

    if profile_type in ("memory", "both"):
        memory_output = output_dir / "memory_profile.json"
        result = run_memory_profile(
            tinyllm_cmd,
            memory_output,
        )
        if result != 0:
            exit_code = result

    return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile TinyLLM with flame graph generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile CPU with flame graph
  %(prog)s cpu --command "tinyllm run 'What is 2+2?'" --output cpu_profile.svg

  # Profile memory
  %(prog)s memory --command "tinyllm run 'What is 2+2?'" --output memory.json

  # Profile a graph execution
  %(prog)s graph --graph graphs/multi_domain.yaml --query "Write Python to calculate pi" --output-dir profiles/

  # Quick CPU profile
  %(prog)s cpu -- tinyllm run "What is 2+2?"
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Profiling mode")

    # CPU profiling mode
    cpu_parser = subparsers.add_parser("cpu", help="CPU profiling with flame graph")
    cpu_parser.add_argument(
        "--command", "-c", help="Command to profile (quoted string)"
    )
    cpu_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("cpu_profile.svg"),
        help="Output file path",
    )
    cpu_parser.add_argument(
        "--duration", "-d", type=int, help="Profile duration in seconds"
    )
    cpu_parser.add_argument(
        "--rate", "-r", type=int, default=100, help="Sampling rate in Hz"
    )
    cpu_parser.add_argument(
        "--format",
        "-f",
        choices=["flamegraph", "speedscope", "raw"],
        default="flamegraph",
        help="Output format",
    )

    # Memory profiling mode
    mem_parser = subparsers.add_parser("memory", help="Memory profiling")
    mem_parser.add_argument(
        "--command", "-c", required=True, help="Command to profile (quoted string)"
    )
    mem_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("memory_profile.json"),
        help="Output file path",
    )
    mem_parser.add_argument(
        "--interval", "-i", type=float, default=0.1, help="Sampling interval in seconds"
    )

    # Graph profiling mode
    graph_parser = subparsers.add_parser("graph", help="Profile TinyLLM graph execution")
    graph_parser.add_argument(
        "--graph", "-g", type=Path, required=True, help="Graph YAML file"
    )
    graph_parser.add_argument("--query", "-q", required=True, help="Query to execute")
    graph_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("profiles"),
        help="Output directory",
    )
    graph_parser.add_argument(
        "--type",
        "-t",
        choices=["cpu", "memory", "both"],
        default="both",
        help="Profile type",
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return 1

    # Execute based on mode
    if args.mode == "cpu":
        if args.command:
            command = args.command.split()
        else:
            # Assume remaining args are the command
            command = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
            if not command:
                print("Error: No command specified", file=sys.stderr)
                return 1

        return run_cpu_profile(
            command,
            args.output,
            duration=args.duration,
            rate=args.rate,
            format=args.format,
        )

    elif args.mode == "memory":
        command = args.command.split()
        return run_memory_profile(command, args.output, interval=args.interval)

    elif args.mode == "graph":
        return profile_tinyllm_graph(
            args.graph,
            args.query,
            args.output_dir,
            profile_type=args.type,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
