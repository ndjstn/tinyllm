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
