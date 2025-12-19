"""Performance and memory profiling for TinyLLM.

This module provides comprehensive profiling capabilities including:
- Performance profiling with cProfile and optional py-spy
- Memory profiling with tracemalloc
- Function-level timing and memory tracking
- Profile data export for visualization
- Flame graph generation
- Slow query detection and alerting
"""

import asyncio
import cProfile
import functools
import io
import pstats
import time
import tracemalloc
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="profiling")

T = TypeVar("T")


class ProfileStats(BaseModel):
    """Statistics from a profiling session."""

    name: str = Field(..., description="Profile name or identifier")
    duration_ms: float = Field(..., description="Duration in milliseconds")
    cpu_time_ms: float = Field(..., description="CPU time in milliseconds")
    call_count: int = Field(default=0, description="Number of function calls")
    memory_delta_bytes: Optional[int] = Field(None, description="Memory delta in bytes")
    memory_peak_bytes: Optional[int] = Field(None, description="Peak memory usage in bytes")
    function_stats: Optional[Dict[str, Any]] = Field(None, description="Per-function statistics")


class MemorySnapshot(BaseModel):
    """Memory usage snapshot."""

    timestamp: float = Field(..., description="Timestamp of snapshot")
    current_bytes: int = Field(..., description="Current memory usage in bytes")
    peak_bytes: int = Field(..., description="Peak memory usage in bytes")
    top_allocations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top memory allocations"
    )


class PerformanceProfiler:
    """Performance profiler for tracking execution time and call counts."""

    def __init__(self, enabled: bool = True):
        """Initialize profiler.

        Args:
            enabled: Whether profiling is enabled.
        """
        self.enabled = enabled
        self._profiles: Dict[str, ProfileStats] = {}
        self._active_profile: Optional[cProfile.Profile] = None

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        """Context manager for profiling a code block.

        Args:
            name: Name for this profiling session.

        Yields:
            None

        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.profile("my_function"):
            ...     # Code to profile
            ...     pass
        """
        if not self.enabled:
            yield
            return

        pr = cProfile.Profile()
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        pr.enable()
        try:
            yield
        finally:
            pr.disable()
            end_time = time.perf_counter()
            end_cpu = time.process_time()

            # Collect statistics
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats(pstats.SortKey.CUMULATIVE)

            # Extract key metrics
            total_calls = ps.total_calls
            duration_ms = (end_time - start_time) * 1000
            cpu_time_ms = (end_cpu - start_cpu) * 1000

            # Store profile stats
            self._profiles[name] = ProfileStats(
                name=name,
                duration_ms=duration_ms,
                cpu_time_ms=cpu_time_ms,
                call_count=total_calls,
            )

            logger.debug(
                "profile_completed",
                name=name,
                duration_ms=duration_ms,
                cpu_time_ms=cpu_time_ms,
                call_count=total_calls,
            )

    @asynccontextmanager
    async def profile_async(self, name: str):
        """Async context manager for profiling async code.

        Args:
            name: Name for this profiling session.

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        start_cpu = time.process_time()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_cpu = time.process_time()

            duration_ms = (end_time - start_time) * 1000
            cpu_time_ms = (end_cpu - start_cpu) * 1000

            self._profiles[name] = ProfileStats(
                name=name,
                duration_ms=duration_ms,
                cpu_time_ms=cpu_time_ms,
                call_count=1,
            )

            logger.debug(
                "async_profile_completed",
                name=name,
                duration_ms=duration_ms,
                cpu_time_ms=cpu_time_ms,
            )

    def get_stats(self, name: Optional[str] = None) -> Union[ProfileStats, Dict[str, ProfileStats]]:
        """Get profiling statistics.

        Args:
            name: Optional profile name to get stats for.

        Returns:
            ProfileStats for named profile, or all profiles if name is None.
        """
        if name:
            return self._profiles.get(name, ProfileStats(name=name, duration_ms=0, cpu_time_ms=0))
        return self._profiles.copy()

    def clear_stats(self) -> None:
        """Clear all profiling statistics."""
        self._profiles.clear()
        logger.debug("profiling_stats_cleared")

    def export_stats(self, filepath: Path) -> None:
        """Export profiling statistics to a file.

        Args:
            filepath: Path to export file.
        """
        import json

        data = {name: stats.model_dump() for name, stats in self._profiles.items()}

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("profiling_stats_exported", filepath=str(filepath), count=len(self._profiles))


class MemoryProfiler:
    """Memory profiler using tracemalloc."""

    def __init__(self, enabled: bool = True, max_frames: int = 10):
        """Initialize memory profiler.

        Args:
            enabled: Whether profiling is enabled.
            max_frames: Maximum number of stack frames to capture.
        """
        self.enabled = enabled
        self.max_frames = max_frames
        self._snapshots: List[MemorySnapshot] = []
        self._tracking = False

    def start(self) -> None:
        """Start memory tracking."""
        if not self.enabled:
            return

        if not self._tracking:
            tracemalloc.start(self.max_frames)
            self._tracking = True
            logger.info("memory_tracking_started", max_frames=self.max_frames)

    def stop(self) -> None:
        """Stop memory tracking."""
        if self._tracking:
            tracemalloc.stop()
            self._tracking = False
            logger.info("memory_tracking_stopped")

    def take_snapshot(self, top_n: int = 10) -> MemorySnapshot:
        """Take a memory usage snapshot.

        Args:
            top_n: Number of top allocations to include.

        Returns:
            MemorySnapshot with current memory state.
        """
        if not self._tracking:
            self.start()

        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()

        # Get top allocations
        top_stats = snapshot.statistics("lineno")[:top_n]
        top_allocations = []
        for stat in top_stats:
            top_allocations.append(
                {
                    "filename": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_bytes": stat.size,
                    "count": stat.count,
                }
            )

        memory_snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_bytes=current,
            peak_bytes=peak,
            top_allocations=top_allocations,
        )

        self._snapshots.append(memory_snapshot)
        logger.debug(
            "memory_snapshot_taken",
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
        )

        return memory_snapshot

    @contextmanager
    def track_memory(self, name: str) -> Iterator[None]:
        """Context manager for tracking memory usage of a code block.

        Args:
            name: Name for this tracking session.

        Yields:
            None
        """
        if not self.enabled:
            yield
            return

        if not self._tracking:
            self.start()

        # Take snapshot before
        snapshot_before = tracemalloc.take_snapshot()
        current_before, peak_before = tracemalloc.get_traced_memory()

        try:
            yield
        finally:
            # Take snapshot after
            snapshot_after = tracemalloc.take_snapshot()
            current_after, peak_after = tracemalloc.get_traced_memory()

            # Compare snapshots
            top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

            delta = current_after - current_before
            logger.info(
                "memory_tracking_completed",
                name=name,
                delta_bytes=delta,
                delta_mb=delta / 1024 / 1024,
                peak_mb=peak_after / 1024 / 1024,
                top_increases=len([s for s in top_stats if s.size_diff > 0]),
            )

    def get_current_memory(self) -> tuple[int, int]:
        """Get current memory usage.

        Returns:
            Tuple of (current_bytes, peak_bytes).
        """
        if not self._tracking:
            return (0, 0)
        return tracemalloc.get_traced_memory()

    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all memory snapshots.

        Returns:
            List of memory snapshots.
        """
        return self._snapshots.copy()

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
        logger.debug("memory_snapshots_cleared")

    def export_snapshots(self, filepath: Path) -> None:
        """Export memory snapshots to a file.

        Args:
            filepath: Path to export file.
        """
        import json

        data = [snapshot.model_dump() for snapshot in self._snapshots]

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("memory_snapshots_exported", filepath=str(filepath), count=len(self._snapshots))


class ProfilingContext:
    """Combined profiling context for performance and memory."""

    def __init__(
        self,
        enable_performance: bool = True,
        enable_memory: bool = True,
    ):
        """Initialize profiling context.

        Args:
            enable_performance: Enable performance profiling.
            enable_memory: Enable memory profiling.
        """
        self.perf_profiler = PerformanceProfiler(enabled=enable_performance)
        self.mem_profiler = MemoryProfiler(enabled=enable_memory)

    @contextmanager
    def profile(self, name: str, track_memory: bool = True) -> Iterator[None]:
        """Profile both performance and memory.

        Args:
            name: Name for this profiling session.
            track_memory: Whether to track memory.

        Yields:
            None
        """
        with self.perf_profiler.profile(name):
            if track_memory:
                with self.mem_profiler.track_memory(name):
                    yield
            else:
                yield

    @asynccontextmanager
    async def profile_async(self, name: str):
        """Profile async code.

        Args:
            name: Name for this profiling session.

        Yields:
            None
        """
        async with self.perf_profiler.profile_async(name):
            yield

    def export_all(self, directory: Path) -> None:
        """Export all profiling data.

        Args:
            directory: Directory to export to.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.perf_profiler.export_stats(directory / "performance.json")
        self.mem_profiler.export_snapshots(directory / "memory.json")
        logger.info("profiling_data_exported", directory=str(directory))


# Global profiling context
_global_profiling: Optional[ProfilingContext] = None


def get_profiling_context(
    enable_performance: bool = True,
    enable_memory: bool = True,
) -> ProfilingContext:
    """Get or create global profiling context.

    Args:
        enable_performance: Enable performance profiling.
        enable_memory: Enable memory profiling.

    Returns:
        Global ProfilingContext instance.
    """
    global _global_profiling
    if _global_profiling is None:
        _global_profiling = ProfilingContext(
            enable_performance=enable_performance,
            enable_memory=enable_memory,
        )
    return _global_profiling


def profile(name: Optional[str] = None, track_memory: bool = False):
    """Decorator for profiling functions.

    Args:
        name: Optional name for the profile. Uses function name if None.
        track_memory: Whether to track memory usage.

    Returns:
        Decorated function.

    Example:
        >>> @profile()
        ... def my_function():
        ...     # Code to profile
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        profile_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                ctx = get_profiling_context()
                async with ctx.profile_async(profile_name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                ctx = get_profiling_context()
                with ctx.profile(profile_name, track_memory=track_memory):
                    return func(*args, **kwargs)

            return wrapper

    return decorator


class FlameGraphGenerator:
    """Generate flame graphs from profiling data.

    Flame graphs are a visualization of profiled software, allowing the most
    frequent code-paths to be identified quickly and accurately.
    """

    def __init__(self):
        """Initialize flame graph generator."""
        self._samples: List[tuple[float, List[str]]] = []

    def add_sample(self, timestamp: float, stack: List[str]) -> None:
        """Add a stack trace sample.

        Args:
            timestamp: Sample timestamp.
            stack: Call stack (from bottom to top).
        """
        self._samples.append((timestamp, stack))

    def generate_folded_stacks(self) -> str:
        """Generate folded stack format (input for flamegraph.pl).

        Returns:
            Folded stacks as string.
        """
        # Aggregate stack traces
        stack_counts: Dict[str, int] = defaultdict(int)

        for timestamp, stack in self._samples:
            # Create folded stack (semicolon-separated)
            folded = ";".join(stack)
            stack_counts[folded] += 1

        # Format as "stack count" lines
        lines = []
        for stack, count in sorted(stack_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{stack} {count}")

        return "\n".join(lines)

    def export_folded(self, filepath: Path) -> None:
        """Export folded stacks to file.

        Args:
            filepath: Path to output file.
        """
        folded = self.generate_folded_stacks()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(folded)

        logger.info("flame_graph_data_exported", filepath=str(filepath), samples=len(self._samples))

    def generate_from_profile(self, profile: cProfile.Profile) -> str:
        """Generate folded stacks from cProfile.Profile object.

        Args:
            profile: cProfile.Profile instance.

        Returns:
            Folded stacks as string.
        """
        # Extract stats from profile
        stats = pstats.Stats(profile)

        # Build call graph
        stack_counts: Dict[str, int] = defaultdict(int)

        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, func_name = func
            # Create simplified stack entry
            entry = f"{filename}:{func_name}:{line}"
            stack_counts[entry] += int(nc)

        # Format as folded stacks
        lines = []
        for stack, count in sorted(stack_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"{stack} {count}")

        return "\n".join(lines)

    def export_profile_folded(self, profile: cProfile.Profile, filepath: Path) -> None:
        """Export cProfile data as folded stacks.

        Args:
            profile: cProfile.Profile instance.
            filepath: Path to output file.
        """
        folded = self.generate_from_profile(profile)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(folded)

        logger.info("profile_flame_graph_exported", filepath=str(filepath))

    def clear_samples(self) -> None:
        """Clear all samples."""
        self._samples.clear()


class SlowQueryDetector:
    """Detect and log slow operations (queries) in the system.

    Helps identify performance bottlenecks by tracking operations that
    exceed configurable thresholds.
    """

    def __init__(
        self,
        threshold_ms: float = 1000.0,
        enable_alerts: bool = True,
        track_history: bool = True,
        max_history: int = 1000,
    ):
        """Initialize slow query detector.

        Args:
            threshold_ms: Threshold in milliseconds for slow operations.
            enable_alerts: Whether to log alerts for slow queries.
            track_history: Whether to track history of slow queries.
            max_history: Maximum number of slow queries to track.
        """
        self.threshold_ms = threshold_ms
        self.enable_alerts = enable_alerts
        self.track_history = track_history
        self.max_history = max_history
        self._slow_queries: List[Dict[str, Any]] = []
        self._query_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "min_ms": float("inf")}
        )

    @contextmanager
    def track_operation(
        self,
        operation_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """Track an operation and detect if it's slow.

        Args:
            operation_name: Name of the operation.
            context: Additional context for the operation.

        Yields:
            None
        """
        start_time = time.perf_counter()
        exception_raised = False

        try:
            yield
        except Exception as e:
            exception_raised = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update statistics
            stats = self._query_stats[operation_name]
            stats["count"] += 1
            stats["total_ms"] += duration_ms
            stats["max_ms"] = max(stats["max_ms"], duration_ms)
            stats["min_ms"] = min(stats["min_ms"], duration_ms)

            # Check if slow
            if duration_ms >= self.threshold_ms:
                slow_query = {
                    "operation": operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "timestamp": time.time(),
                    "context": context or {},
                    "exception": exception_raised,
                }

                if self.track_history:
                    self._slow_queries.append(slow_query)
                    # Trim history if needed
                    if len(self._slow_queries) > self.max_history:
                        self._slow_queries.pop(0)

                if self.enable_alerts:
                    logger.warning(
                        "slow_query_detected",
                        operation=operation_name,
                        duration_ms=round(duration_ms, 2),
                        threshold_ms=self.threshold_ms,
                        context=context,
                        exception=exception_raised,
                    )

    def get_slow_queries(
        self,
        operation_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get slow queries from history.

        Args:
            operation_name: Filter by operation name.
            limit: Maximum number of queries to return.

        Returns:
            List of slow query records.
        """
        queries = self._slow_queries

        if operation_name:
            queries = [q for q in queries if q["operation"] == operation_name]

        if limit:
            queries = queries[-limit:]

        return queries

    def get_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get operation statistics.

        Args:
            operation_name: Optional operation name to filter by.

        Returns:
            Statistics dictionary.
        """
        if operation_name:
            if operation_name not in self._query_stats:
                return {
                    "operation": operation_name,
                    "count": 0,
                    "total_ms": 0.0,
                    "avg_ms": 0.0,
                    "max_ms": 0.0,
                    "min_ms": 0.0,
                }

            stats = self._query_stats[operation_name]
            avg_ms = stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0.0

            return {
                "operation": operation_name,
                "count": stats["count"],
                "total_ms": round(stats["total_ms"], 2),
                "avg_ms": round(avg_ms, 2),
                "max_ms": round(stats["max_ms"], 2),
                "min_ms": round(stats["min_ms"], 2) if stats["min_ms"] != float("inf") else 0.0,
            }

        # Return all statistics
        all_stats = {}
        for op_name, stats in self._query_stats.items():
            avg_ms = stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0.0
            all_stats[op_name] = {
                "count": stats["count"],
                "total_ms": round(stats["total_ms"], 2),
                "avg_ms": round(avg_ms, 2),
                "max_ms": round(stats["max_ms"], 2),
                "min_ms": round(stats["min_ms"], 2) if stats["min_ms"] != float("inf") else 0.0,
            }

        return all_stats

    def set_threshold(self, threshold_ms: float) -> None:
        """Update the slow query threshold.

        Args:
            threshold_ms: New threshold in milliseconds.
        """
        self.threshold_ms = threshold_ms
        logger.info("slow_query_threshold_updated", threshold_ms=threshold_ms)

    def clear_history(self) -> None:
        """Clear slow query history."""
        self._slow_queries.clear()
        logger.debug("slow_query_history_cleared")

    def clear_statistics(self) -> None:
        """Clear all statistics."""
        self._query_stats.clear()
        logger.debug("slow_query_statistics_cleared")

    def export_report(self, filepath: Path) -> None:
        """Export slow query report.

        Args:
            filepath: Path to output file.
        """
        import json

        report = {
            "threshold_ms": self.threshold_ms,
            "slow_queries_count": len(self._slow_queries),
            "slow_queries": self._slow_queries,
            "statistics": self.get_statistics(),
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("slow_query_report_exported", filepath=str(filepath))


# Global instances
_global_flame_graph: Optional[FlameGraphGenerator] = None
_global_slow_query_detector: Optional[SlowQueryDetector] = None


def get_flame_graph_generator() -> FlameGraphGenerator:
    """Get or create global flame graph generator.

    Returns:
        Global FlameGraphGenerator instance.
    """
    global _global_flame_graph
    if _global_flame_graph is None:
        _global_flame_graph = FlameGraphGenerator()
    return _global_flame_graph


def get_slow_query_detector(threshold_ms: float = 1000.0) -> SlowQueryDetector:
    """Get or create global slow query detector.

    Args:
        threshold_ms: Threshold in milliseconds for slow operations.

    Returns:
        Global SlowQueryDetector instance.
    """
    global _global_slow_query_detector
    if _global_slow_query_detector is None:
        _global_slow_query_detector = SlowQueryDetector(threshold_ms=threshold_ms)
    return _global_slow_query_detector
