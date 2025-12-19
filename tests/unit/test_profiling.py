"""Tests for performance and memory profiling."""

import asyncio
import cProfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tinyllm.profiling import (
    FlameGraphGenerator,
    MemoryProfiler,
    MemorySnapshot,
    PerformanceProfiler,
    ProfileStats,
    ProfilingContext,
    SlowQueryDetector,
    get_flame_graph_generator,
    get_profiling_context,
    get_slow_query_detector,
    profile,
)


class TestProfileStats:
    """Test ProfileStats model."""

    def test_create_stats(self):
        """Test creating profile stats."""
        stats = ProfileStats(
            name="test_function",
            duration_ms=100.5,
            cpu_time_ms=95.2,
            call_count=10,
            memory_delta_bytes=1024,
        )

        assert stats.name == "test_function"
        assert stats.duration_ms == 100.5
        assert stats.cpu_time_ms == 95.2
        assert stats.call_count == 10
        assert stats.memory_delta_bytes == 1024


class TestMemorySnapshot:
    """Test MemorySnapshot model."""

    def test_create_snapshot(self):
        """Test creating memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_bytes=1024 * 1024,
            peak_bytes=2048 * 1024,
            top_allocations=[
                {
                    "filename": "test.py:10",
                    "size_bytes": 512,
                    "count": 5,
                }
            ],
        )

        assert snapshot.current_bytes == 1024 * 1024
        assert snapshot.peak_bytes == 2048 * 1024
        assert len(snapshot.top_allocations) == 1


class TestPerformanceProfiler:
    """Test PerformanceProfiler."""

    def test_init(self):
        """Test initializing profiler."""
        profiler = PerformanceProfiler()
        assert profiler.enabled is True

    def test_init_disabled(self):
        """Test initializing disabled profiler."""
        profiler = PerformanceProfiler(enabled=False)
        assert profiler.enabled is False

    def test_profile_sync(self):
        """Test profiling synchronous code."""
        profiler = PerformanceProfiler()

        with profiler.profile("test_function"):
            # Simulate some work
            total = sum(range(1000))

        stats = profiler.get_stats("test_function")
        assert stats.name == "test_function"
        assert stats.duration_ms > 0
        assert stats.cpu_time_ms > 0
        assert stats.call_count > 0

    def test_profile_disabled(self):
        """Test that disabled profiler doesn't collect stats."""
        profiler = PerformanceProfiler(enabled=False)

        with profiler.profile("test_function"):
            pass

        stats = profiler.get_stats("test_function")
        # Should return empty stats
        assert stats.duration_ms == 0

    @pytest.mark.asyncio
    async def test_profile_async(self):
        """Test profiling async code."""
        profiler = PerformanceProfiler()

        async with profiler.profile_async("async_function"):
            await asyncio.sleep(0.01)

        stats = profiler.get_stats("async_function")
        assert stats.name == "async_function"
        assert stats.duration_ms >= 10  # At least 10ms
        assert stats.cpu_time_ms >= 0

    def test_get_all_stats(self):
        """Test getting all stats."""
        profiler = PerformanceProfiler()

        with profiler.profile("func1"):
            pass

        with profiler.profile("func2"):
            pass

        all_stats = profiler.get_stats()
        assert "func1" in all_stats
        assert "func2" in all_stats
        assert len(all_stats) == 2

    def test_clear_stats(self):
        """Test clearing stats."""
        profiler = PerformanceProfiler()

        with profiler.profile("test_function"):
            pass

        assert len(profiler.get_stats()) == 1

        profiler.clear_stats()
        assert len(profiler.get_stats()) == 0

    def test_export_stats(self):
        """Test exporting stats to file."""
        profiler = PerformanceProfiler()

        with profiler.profile("func1"):
            pass

        with profiler.profile("func2"):
            pass

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stats.json"
            profiler.export_stats(filepath)

            assert filepath.exists()

            # Verify file content
            import json

            with open(filepath) as f:
                data = json.load(f)
                assert "func1" in data
                assert "func2" in data


class TestMemoryProfiler:
    """Test MemoryProfiler."""

    def test_init(self):
        """Test initializing memory profiler."""
        profiler = MemoryProfiler()
        assert profiler.enabled is True
        assert profiler.max_frames == 10

    def test_start_stop(self):
        """Test starting and stopping tracking."""
        profiler = MemoryProfiler()
        assert profiler._tracking is False

        profiler.start()
        assert profiler._tracking is True

        profiler.stop()
        assert profiler._tracking is False

    def test_take_snapshot(self):
        """Test taking memory snapshot."""
        profiler = MemoryProfiler()
        profiler.start()

        snapshot = profiler.take_snapshot(top_n=5)

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.current_bytes >= 0
        assert snapshot.peak_bytes >= 0
        assert len(snapshot.top_allocations) <= 5

        profiler.stop()

    def test_track_memory(self):
        """Test tracking memory of code block."""
        profiler = MemoryProfiler()

        with profiler.track_memory("allocate_list"):
            # Allocate some memory
            big_list = [i for i in range(10000)]

        profiler.stop()

    def test_get_current_memory(self):
        """Test getting current memory usage."""
        profiler = MemoryProfiler()
        profiler.start()

        current, peak = profiler.get_current_memory()
        assert current >= 0
        assert peak >= current

        profiler.stop()

    def test_get_snapshots(self):
        """Test getting all snapshots."""
        profiler = MemoryProfiler()
        profiler.start()

        profiler.take_snapshot()
        profiler.take_snapshot()

        snapshots = profiler.get_snapshots()
        assert len(snapshots) == 2

        profiler.stop()

    def test_clear_snapshots(self):
        """Test clearing snapshots."""
        profiler = MemoryProfiler()
        profiler.start()

        profiler.take_snapshot()
        assert len(profiler.get_snapshots()) == 1

        profiler.clear_snapshots()
        assert len(profiler.get_snapshots()) == 0

        profiler.stop()

    def test_export_snapshots(self):
        """Test exporting snapshots to file."""
        profiler = MemoryProfiler()
        profiler.start()

        profiler.take_snapshot()
        profiler.take_snapshot()

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "memory.json"
            profiler.export_snapshots(filepath)

            assert filepath.exists()

            # Verify file content
            import json

            with open(filepath) as f:
                data = json.load(f)
                assert len(data) == 2

        profiler.stop()


class TestProfilingContext:
    """Test ProfilingContext."""

    def test_init(self):
        """Test initializing profiling context."""
        ctx = ProfilingContext()
        assert ctx.perf_profiler.enabled is True
        assert ctx.mem_profiler.enabled is True

    def test_init_selective(self):
        """Test selective profiling."""
        ctx = ProfilingContext(enable_performance=True, enable_memory=False)
        assert ctx.perf_profiler.enabled is True
        assert ctx.mem_profiler.enabled is False

    def test_profile_combined(self):
        """Test combined profiling."""
        ctx = ProfilingContext()

        with ctx.profile("test_function", track_memory=True):
            # Do some work
            result = sum(range(1000))

        stats = ctx.perf_profiler.get_stats("test_function")
        assert stats.duration_ms > 0

    @pytest.mark.asyncio
    async def test_profile_async_combined(self):
        """Test combined async profiling."""
        ctx = ProfilingContext()

        async with ctx.profile_async("async_function"):
            await asyncio.sleep(0.01)

        stats = ctx.perf_profiler.get_stats("async_function")
        assert stats.duration_ms >= 10

    def test_export_all(self):
        """Test exporting all profiling data."""
        ctx = ProfilingContext()

        with ctx.profile("func1"):
            pass

        ctx.mem_profiler.start()
        ctx.mem_profiler.take_snapshot()

        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "profiles"
            ctx.export_all(directory)

            assert (directory / "performance.json").exists()
            assert (directory / "memory.json").exists()

        ctx.mem_profiler.stop()


class TestProfileDecorator:
    """Test profile decorator."""

    def test_profile_decorator_sync(self):
        """Test profile decorator on sync function."""

        @profile(name="decorated_sync")
        def my_function(x, y):
            return x + y

        result = my_function(2, 3)
        assert result == 5

        ctx = get_profiling_context()
        stats = ctx.perf_profiler.get_stats("decorated_sync")
        assert stats.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_profile_decorator_async(self):
        """Test profile decorator on async function."""

        @profile(name="decorated_async")
        async def my_async_function(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await my_async_function(5)
        assert result == 10

        ctx = get_profiling_context()
        stats = ctx.perf_profiler.get_stats("decorated_async")
        assert stats.duration_ms >= 10

    def test_profile_decorator_auto_name(self):
        """Test profile decorator with auto naming."""

        @profile()
        def auto_named_function():
            return 42

        result = auto_named_function()
        assert result == 42

        ctx = get_profiling_context()
        stats = ctx.perf_profiler.get_stats("auto_named_function")
        assert stats.duration_ms >= 0

    def test_profile_decorator_with_memory(self):
        """Test profile decorator with memory tracking."""

        @profile(name="memory_tracked", track_memory=True)
        def allocate_memory():
            return [i for i in range(10000)]

        result = allocate_memory()
        assert len(result) == 10000

        ctx = get_profiling_context()
        stats = ctx.perf_profiler.get_stats("memory_tracked")
        assert stats.duration_ms >= 0


class TestGlobalContext:
    """Test global profiling context."""

    def test_get_global_context(self):
        """Test getting global context."""
        ctx1 = get_profiling_context()
        ctx2 = get_profiling_context()

        # Should return same instance
        assert ctx1 is ctx2


class TestFlameGraphGenerator:
    """Test FlameGraphGenerator."""

    def test_init(self):
        """Test initializing flame graph generator."""
        generator = FlameGraphGenerator()
        assert generator._samples == []

    def test_add_sample(self):
        """Test adding stack samples."""
        generator = FlameGraphGenerator()

        stack1 = ["main", "function_a", "function_b"]
        stack2 = ["main", "function_a", "function_c"]

        generator.add_sample(1.0, stack1)
        generator.add_sample(2.0, stack2)

        assert len(generator._samples) == 2

    def test_generate_folded_stacks(self):
        """Test generating folded stack format."""
        generator = FlameGraphGenerator()

        # Add samples
        generator.add_sample(1.0, ["main", "func_a", "func_b"])
        generator.add_sample(2.0, ["main", "func_a", "func_b"])
        generator.add_sample(3.0, ["main", "func_c"])

        folded = generator.generate_folded_stacks()

        assert "main;func_a;func_b 2" in folded
        assert "main;func_c 1" in folded

    def test_export_folded(self):
        """Test exporting folded stacks to file."""
        generator = FlameGraphGenerator()

        generator.add_sample(1.0, ["main", "func_a"])
        generator.add_sample(2.0, ["main", "func_a"])

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "flamegraph.txt"
            generator.export_folded(filepath)

            assert filepath.exists()

            content = filepath.read_text()
            assert "main;func_a" in content

    def test_generate_from_profile(self):
        """Test generating flame graph from cProfile."""
        generator = FlameGraphGenerator()

        # Create a simple profile
        pr = cProfile.Profile()
        pr.enable()

        def test_func():
            return sum(range(100))

        test_func()
        pr.disable()

        folded = generator.generate_from_profile(pr)

        # Should contain some profiling data
        assert len(folded) > 0

    def test_export_profile_folded(self):
        """Test exporting cProfile as folded stacks."""
        generator = FlameGraphGenerator()

        # Create profile
        pr = cProfile.Profile()
        pr.enable()
        _ = sum(range(1000))
        pr.disable()

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "profile_flame.txt"
            generator.export_profile_folded(pr, filepath)

            assert filepath.exists()

    def test_clear_samples(self):
        """Test clearing samples."""
        generator = FlameGraphGenerator()

        generator.add_sample(1.0, ["main"])
        generator.add_sample(2.0, ["main"])

        assert len(generator._samples) == 2

        generator.clear_samples()
        assert len(generator._samples) == 0


class TestSlowQueryDetector:
    """Test SlowQueryDetector."""

    def test_init(self):
        """Test initializing detector."""
        detector = SlowQueryDetector(threshold_ms=500.0)

        assert detector.threshold_ms == 500.0
        assert detector.enable_alerts is True
        assert detector.track_history is True

    def test_track_operation_fast(self):
        """Test tracking fast operation."""
        detector = SlowQueryDetector(threshold_ms=1000.0)

        with detector.track_operation("fast_op"):
            time.sleep(0.001)  # 1ms

        # Should not be recorded as slow
        slow_queries = detector.get_slow_queries()
        assert len(slow_queries) == 0

    def test_track_operation_slow(self):
        """Test tracking slow operation."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        with detector.track_operation("slow_op"):
            time.sleep(0.02)  # 20ms

        # Should be recorded as slow
        slow_queries = detector.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0]["operation"] == "slow_op"
        assert slow_queries[0]["duration_ms"] >= 10.0

    def test_track_operation_with_context(self):
        """Test tracking with context."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        context = {"user_id": "123", "query": "SELECT *"}

        with detector.track_operation("database_query", context=context):
            time.sleep(0.02)

        slow_queries = detector.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0]["context"] == context

    def test_track_operation_with_exception(self):
        """Test tracking operation that raises exception."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        with pytest.raises(ValueError):
            with detector.track_operation("failing_op"):
                time.sleep(0.02)
                raise ValueError("Test error")

        slow_queries = detector.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0]["exception"] is True

    def test_get_slow_queries_filtered(self):
        """Test getting filtered slow queries."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        with detector.track_operation("op_a"):
            time.sleep(0.02)

        with detector.track_operation("op_b"):
            time.sleep(0.02)

        with detector.track_operation("op_a"):
            time.sleep(0.02)

        # Get all
        all_queries = detector.get_slow_queries()
        assert len(all_queries) == 3

        # Filter by operation
        op_a_queries = detector.get_slow_queries(operation_name="op_a")
        assert len(op_a_queries) == 2

    def test_get_slow_queries_limited(self):
        """Test getting limited slow queries."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        for i in range(5):
            with detector.track_operation(f"op_{i}"):
                time.sleep(0.02)

        queries = detector.get_slow_queries(limit=3)
        assert len(queries) == 3

    def test_get_statistics(self):
        """Test getting operation statistics."""
        detector = SlowQueryDetector(threshold_ms=1000.0)

        # Record multiple operations
        with detector.track_operation("test_op"):
            time.sleep(0.01)

        with detector.track_operation("test_op"):
            time.sleep(0.02)

        stats = detector.get_statistics("test_op")

        assert stats["operation"] == "test_op"
        assert stats["count"] == 2
        assert stats["avg_ms"] > 0
        assert stats["max_ms"] >= stats["min_ms"]

    def test_get_all_statistics(self):
        """Test getting all statistics."""
        detector = SlowQueryDetector(threshold_ms=1000.0)

        with detector.track_operation("op_a"):
            time.sleep(0.01)

        with detector.track_operation("op_b"):
            time.sleep(0.01)

        all_stats = detector.get_statistics()

        assert "op_a" in all_stats
        assert "op_b" in all_stats
        assert all_stats["op_a"]["count"] == 1
        assert all_stats["op_b"]["count"] == 1

    def test_set_threshold(self):
        """Test updating threshold."""
        detector = SlowQueryDetector(threshold_ms=100.0)

        assert detector.threshold_ms == 100.0

        detector.set_threshold(500.0)
        assert detector.threshold_ms == 500.0

    def test_clear_history(self):
        """Test clearing slow query history."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        with detector.track_operation("slow_op"):
            time.sleep(0.02)

        assert len(detector.get_slow_queries()) == 1

        detector.clear_history()
        assert len(detector.get_slow_queries()) == 0

    def test_clear_statistics(self):
        """Test clearing statistics."""
        detector = SlowQueryDetector(threshold_ms=1000.0)

        with detector.track_operation("test_op"):
            time.sleep(0.01)

        assert len(detector.get_statistics()) > 0

        detector.clear_statistics()
        assert len(detector.get_statistics()) == 0

    def test_export_report(self):
        """Test exporting slow query report."""
        detector = SlowQueryDetector(threshold_ms=10.0)

        with detector.track_operation("slow_op"):
            time.sleep(0.02)

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "slow_queries.json"
            detector.export_report(filepath)

            assert filepath.exists()

            import json
            with open(filepath) as f:
                report = json.load(f)

            assert report["threshold_ms"] == 10.0
            assert report["slow_queries_count"] == 1
            assert "statistics" in report

    def test_max_history(self):
        """Test max history limit."""
        detector = SlowQueryDetector(threshold_ms=10.0, max_history=3)

        # Add more than max_history slow queries
        for i in range(5):
            with detector.track_operation(f"op_{i}"):
                time.sleep(0.02)

        queries = detector.get_slow_queries()
        # Should only keep last 3
        assert len(queries) == 3

    def test_disabled_alerts(self):
        """Test detector with alerts disabled."""
        detector = SlowQueryDetector(threshold_ms=10.0, enable_alerts=False)

        # This should not log warning (hard to test, but shouldn't error)
        with detector.track_operation("slow_op"):
            time.sleep(0.02)

        queries = detector.get_slow_queries()
        assert len(queries) == 1

    def test_disabled_history(self):
        """Test detector with history disabled."""
        detector = SlowQueryDetector(threshold_ms=10.0, track_history=False)

        with detector.track_operation("slow_op"):
            time.sleep(0.02)

        # History should be empty
        queries = detector.get_slow_queries()
        assert len(queries) == 0

        # But statistics should still be tracked
        stats = detector.get_statistics("slow_op")
        assert stats["count"] == 1


class TestGlobalInstances:
    """Test global instance functions."""

    def test_get_flame_graph_generator(self):
        """Test getting global flame graph generator."""
        gen1 = get_flame_graph_generator()
        gen2 = get_flame_graph_generator()

        # Should return same instance
        assert gen1 is gen2

    def test_get_slow_query_detector(self):
        """Test getting global slow query detector."""
        det1 = get_slow_query_detector(threshold_ms=100.0)
        det2 = get_slow_query_detector(threshold_ms=200.0)

        # Should return same instance (threshold not updated)
        assert det1 is det2
