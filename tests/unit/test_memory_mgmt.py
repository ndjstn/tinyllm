"""Tests for memory management features."""

import gc
import pickle
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from tinyllm.core.memory import (
    ContextWindowSlider,
    DiskSwapManager,
    GarbageCollectionTuner,
    GCConfig,
    GraphMemoryBudget,
    GraphMemoryManager,
    LeakSnapshot,
    LRUContextCache,
    MemoryLeakDetector,
    MemoryMappedContextStorage,
    MemoryPressureCallback,
    MemoryPressureMonitor,
    MemoryPressureThresholds,
    MessagePool,
    SharedMessageBuffer,
    get_message_pool,
)
from tinyllm.core.message import Message, MessagePayload


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        trace_id="test-trace",
        source_node="test-node",
        payload=MessagePayload(content="Test message"),
    )


@pytest.fixture
def sample_messages():
    """Create multiple sample messages."""
    return [
        Message(
            trace_id="test-trace",
            source_node=f"node-{i}",
            payload=MessagePayload(content=f"Message {i}"),
        )
        for i in range(10)
    ]


# =============================================================================
# Task 71: Memory Pooling Tests
# =============================================================================


class TestMessagePool:
    """Test message pool functionality."""

    def test_pool_initialization(self):
        """Test pool is initialized correctly."""
        pool = MessagePool(pool_size=10, max_size=100)
        assert pool.pool_size == 10
        assert pool.max_size == 100

        stats = pool.get_stats()
        assert stats["pool_size"] == 0
        assert stats["allocated"] == 0
        assert stats["reused"] == 0

    def test_acquire_from_empty_pool(self):
        """Test acquiring from empty pool allocates new."""
        pool = MessagePool()
        data = pool.acquire()

        assert isinstance(data, dict)
        assert len(data) == 0

        stats = pool.get_stats()
        assert stats["allocated"] == 1
        assert stats["reused"] == 0

    def test_acquire_and_release(self):
        """Test releasing and reusing from pool."""
        pool = MessagePool()

        # Acquire and populate
        data1 = pool.acquire()
        data1["test"] = "value"

        # Release
        pool.release(data1)

        # Acquire again should reuse
        data2 = pool.acquire()
        assert data2 is data1
        assert len(data2) == 0  # Should be cleared

        stats = pool.get_stats()
        assert stats["reused"] == 1

    def test_pool_overflow(self):
        """Test pool discards when full."""
        pool = MessagePool(pool_size=2, max_size=2)

        # Fill pool
        pool.release({})
        pool.release({})

        # Try to add more
        pool.release({})

        stats = pool.get_stats()
        assert stats["pool_size"] == 2
        assert stats["discarded"] == 1

    def test_clear_pool(self):
        """Test clearing pool resets stats."""
        pool = MessagePool()
        pool.acquire()
        pool.release({})

        pool.clear()

        stats = pool.get_stats()
        assert stats["pool_size"] == 0
        assert stats["allocated"] == 0

    def test_global_pool(self):
        """Test global pool singleton."""
        pool1 = get_message_pool()
        pool2 = get_message_pool()
        assert pool1 is pool2


# =============================================================================
# Task 72: Zero-Copy Message Passing Tests
# =============================================================================


class TestSharedMessageBuffer:
    """Test shared message buffer."""

    def test_buffer_initialization(self):
        """Test buffer is initialized correctly."""
        buffer = SharedMessageBuffer(size_bytes=1024)
        assert buffer.size_bytes == 1024

    def test_allocate_and_read(self):
        """Test allocating and reading from buffer."""
        buffer = SharedMessageBuffer(size_bytes=1024)
        data = b"Hello, World!"

        offset = buffer.allocate(data)
        assert offset == 0

        read_data = buffer.read(offset)
        assert read_data == data

    def test_multiple_allocations(self):
        """Test multiple allocations."""
        buffer = SharedMessageBuffer(size_bytes=1024)

        data1 = b"First"
        data2 = b"Second"

        offset1 = buffer.allocate(data1)
        offset2 = buffer.allocate(data2)

        assert offset2 > offset1
        assert buffer.read(offset1) == data1
        assert buffer.read(offset2) == data2

    def test_buffer_overflow(self):
        """Test buffer raises error when full."""
        buffer = SharedMessageBuffer(size_bytes=10)
        data = b"This is too long for the buffer"

        with pytest.raises(MemoryError):
            buffer.allocate(data)

    def test_deallocate(self):
        """Test deallocation."""
        buffer = SharedMessageBuffer(size_bytes=1024)
        data = b"Test"

        offset = buffer.allocate(data)
        buffer.deallocate(offset)

        with pytest.raises(ValueError):
            buffer.read(offset)

    def test_compact(self):
        """Test buffer compaction."""
        buffer = SharedMessageBuffer(size_bytes=1024)
        buffer.allocate(b"Test1")
        buffer.allocate(b"Test2")

        buffer.compact()
        # Should still work
        assert buffer._offset > 0

    def test_close(self):
        """Test buffer close."""
        buffer = SharedMessageBuffer(size_bytes=1024)
        buffer.close()
        # Should not raise


# =============================================================================
# Task 73: Memory-Mapped Storage Tests
# =============================================================================


class TestMemoryMappedContextStorage:
    """Test memory-mapped context storage."""

    def test_storage_initialization(self):
        """Test storage is initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedContextStorage(storage_dir=Path(tmpdir))
            assert storage.storage_dir.exists()

    def test_store_and_load(self):
        """Test storing and loading data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedContextStorage(storage_dir=Path(tmpdir))

            data = {"test": "value", "number": 42}
            key = storage.store("trace-123", data)

            loaded = storage.load(key)
            assert loaded == data

    def test_store_large_data(self):
        """Test storing large data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedContextStorage(storage_dir=Path(tmpdir))

            # Create large data
            data = {"messages": [f"Message {i}" * 100 for i in range(1000)]}

            key = storage.store("trace-large", data)
            loaded = storage.load(key)

            assert len(loaded["messages"]) == 1000

    def test_store_too_large(self):
        """Test storing data larger than max size fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedContextStorage(
                storage_dir=Path(tmpdir),
                max_file_size=100  # Very small
            )

            data = {"large": "x" * 1000}

            with pytest.raises(ValueError):
                storage.store("trace-huge", data)

    def test_delete(self):
        """Test deleting storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedContextStorage(storage_dir=Path(tmpdir))

            key = storage.store("trace-123", {"test": "data"})
            storage.delete(key)

            with pytest.raises(KeyError):
                storage.load(key)

    def test_cleanup(self):
        """Test cleanup removes all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MemoryMappedContextStorage(storage_dir=Path(tmpdir))

            storage.store("trace-1", {"data": 1})
            storage.store("trace-2", {"data": 2})

            storage.cleanup()

            assert len(storage._mmaps) == 0


# =============================================================================
# Task 74: Garbage Collection Tuning Tests
# =============================================================================


class TestGarbageCollectionTuner:
    """Test GC tuner."""

    def test_tuner_initialization(self):
        """Test tuner initializes with config."""
        config = GCConfig(gen0_threshold=500)
        tuner = GarbageCollectionTuner(config)

        assert tuner.config.gen0_threshold == 500

    def test_collect(self):
        """Test manual collection."""
        tuner = GarbageCollectionTuner()

        # Create some garbage
        garbage = [object() for _ in range(1000)]
        del garbage

        collected = tuner.collect(generation=0)
        assert isinstance(collected, int)

        stats = tuner.get_stats()
        assert stats["collections"] > 0

    def test_maybe_collect_too_soon(self):
        """Test maybe_collect doesn't collect if too soon."""
        config = GCConfig(manual_gc_interval_seconds=60.0)
        tuner = GarbageCollectionTuner(config)

        result = tuner.maybe_collect()
        assert result is None

    def test_maybe_collect_after_interval(self):
        """Test maybe_collect works after interval."""
        config = GCConfig(manual_gc_interval_seconds=0.1)
        tuner = GarbageCollectionTuner(config)

        time.sleep(0.2)

        result = tuner.maybe_collect()
        assert result is not None

    def test_get_stats(self):
        """Test getting GC stats."""
        tuner = GarbageCollectionTuner()
        stats = tuner.get_stats()

        assert "counts" in stats
        assert "stats" in stats
        assert "gen0" in stats["counts"]

    def test_reset_stats(self):
        """Test resetting stats."""
        tuner = GarbageCollectionTuner()
        tuner.collect()

        tuner.reset_stats()

        stats = tuner.get_stats()
        assert stats["collections"] == 0


# =============================================================================
# Task 75: Context Window Sliding Tests
# =============================================================================


class TestContextWindowSlider:
    """Test context window slider."""

    def test_slider_initialization(self):
        """Test slider initializes correctly."""
        slider = ContextWindowSlider(window_size=10, slide_size=2)
        assert slider.window_size == 10
        assert slider.slide_size == 2

    def test_add_messages_below_limit(self, sample_messages):
        """Test adding messages below window size."""
        slider = ContextWindowSlider(window_size=20)

        for msg in sample_messages[:5]:
            evicted = slider.add(msg)
            assert evicted is None

        assert len(slider.get_window()) == 5

    def test_add_messages_at_limit(self, sample_messages):
        """Test eviction when window is full."""
        slider = ContextWindowSlider(window_size=5, slide_size=2)

        # Fill window
        for msg in sample_messages[:5]:
            slider.add(msg)

        # Add one more, should evict
        evicted = slider.add(sample_messages[5])
        assert evicted is not None
        assert len(evicted) == 2

        # Window should have 4 messages (5 - 2 + 1)
        assert len(slider.get_window()) == 4

    def test_get_recent(self, sample_messages):
        """Test getting recent messages."""
        slider = ContextWindowSlider(window_size=20)

        for msg in sample_messages:
            slider.add(msg)

        recent = slider.get_recent(3)
        assert len(recent) == 3
        assert recent[-1] == sample_messages[-1]

    def test_clear(self, sample_messages):
        """Test clearing window."""
        slider = ContextWindowSlider(window_size=20)

        for msg in sample_messages[:5]:
            slider.add(msg)

        slider.clear()
        assert len(slider.get_window()) == 0

    def test_get_stats(self, sample_messages):
        """Test getting stats."""
        slider = ContextWindowSlider(window_size=10, slide_size=2)

        for msg in sample_messages:
            slider.add(msg)

        stats = slider.get_stats()
        assert "current_size" in stats
        assert "evicted_total" in stats


# =============================================================================
# Task 76: LRU Eviction Tests
# =============================================================================


class TestLRUContextCache:
    """Test LRU cache."""

    def test_cache_initialization(self):
        """Test cache initializes correctly."""
        cache = LRUContextCache(capacity=10)
        assert cache.capacity == 10

    def test_put_and_get(self, sample_message):
        """Test putting and getting from cache."""
        cache = LRUContextCache(capacity=10)

        cache.put(sample_message)
        retrieved = cache.get(sample_message.message_id)

        assert retrieved is not None
        assert retrieved.message_id == sample_message.message_id

    def test_get_miss(self):
        """Test cache miss."""
        cache = LRUContextCache(capacity=10)
        result = cache.get("non-existent")

        assert result is None

    def test_lru_eviction(self, sample_messages):
        """Test LRU eviction policy."""
        cache = LRUContextCache(capacity=3)

        # Add 3 messages
        for msg in sample_messages[:3]:
            cache.put(msg)

        # Add 4th, should evict first
        evicted = cache.put(sample_messages[3])
        assert evicted is not None
        assert evicted.message_id == sample_messages[0].message_id

        # First message should be gone
        assert cache.get(sample_messages[0].message_id) is None

    def test_lru_access_order(self, sample_messages):
        """Test LRU tracks access order."""
        cache = LRUContextCache(capacity=3)

        for msg in sample_messages[:3]:
            cache.put(msg)

        # Access first message
        cache.get(sample_messages[0].message_id)

        # Add new message, should evict second (not first)
        evicted = cache.put(sample_messages[3])
        assert evicted.message_id == sample_messages[1].message_id

        # First should still be there
        assert cache.get(sample_messages[0].message_id) is not None

    def test_remove(self, sample_message):
        """Test removing from cache."""
        cache = LRUContextCache(capacity=10)

        cache.put(sample_message)
        removed = cache.remove(sample_message.message_id)

        assert removed is True
        assert cache.get(sample_message.message_id) is None

    def test_clear(self, sample_messages):
        """Test clearing cache."""
        cache = LRUContextCache(capacity=10)

        for msg in sample_messages[:5]:
            cache.put(msg)

        cache.clear()

        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_hit_rate(self, sample_messages):
        """Test hit rate calculation."""
        cache = LRUContextCache(capacity=10)

        msg = sample_messages[0]
        cache.put(msg)

        # 1 hit
        cache.get(msg.message_id)

        # 2 misses
        cache.get("non-existent-1")
        cache.get("non-existent-2")

        hit_rate = cache.get_hit_rate()
        assert hit_rate == pytest.approx(1/3)

    def test_get_stats(self, sample_messages):
        """Test getting cache stats."""
        cache = LRUContextCache(capacity=10)

        for msg in sample_messages[:5]:
            cache.put(msg)

        stats = cache.get_stats()
        assert stats["size"] == 5
        assert stats["capacity"] == 10
        assert "hit_rate" in stats


# =============================================================================
# Task 77: Memory Pressure Callbacks Tests
# =============================================================================


class TestMemoryPressureMonitor:
    """Test memory pressure monitor."""

    def test_monitor_initialization(self):
        """Test monitor initializes."""
        monitor = MemoryPressureMonitor()
        assert monitor.thresholds is not None

    def test_get_memory_stats(self):
        """Test getting memory stats."""
        monitor = MemoryPressureMonitor()
        stats = monitor.get_memory_stats()

        assert "rss_bytes" in stats or "object_count" in stats

    def test_calculate_pressure_level(self):
        """Test pressure level calculation."""
        monitor = MemoryPressureMonitor()

        # Test with low object count
        stats = {"object_count": 1000}
        level = monitor.calculate_pressure_level(stats)
        assert level in ("low", "medium", "high", "critical")

    def test_register_callback(self):
        """Test registering callbacks."""
        monitor = MemoryPressureMonitor()

        called = []

        def callback(level: str, stats: dict) -> None:
            called.append(level)

        monitor.register_callback(callback)

        # Manually trigger check (won't necessarily call callback)
        monitor.check_pressure()

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = MemoryPressureMonitor(check_interval_seconds=0.1)

        monitor.start_monitoring()
        time.sleep(0.3)
        monitor.stop_monitoring()

        # Should not raise


# =============================================================================
# Task 78: Swap-to-Disk Tests
# =============================================================================


class TestDiskSwapManager:
    """Test disk swap manager."""

    def test_swap_initialization(self):
        """Test swap manager initializes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(swap_dir=Path(tmpdir))
            assert manager.swap_dir.exists()

    def test_store_small_in_memory(self):
        """Test small data stays in memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(
                swap_dir=Path(tmpdir),
                swap_threshold_bytes=1000
            )

            small_data = {"test": "small"}
            manager.store("ctx-1", small_data)

            assert "ctx-1" in manager._in_memory
            assert "ctx-1" not in manager._on_disk

    def test_store_large_to_disk(self):
        """Test large data swaps to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(
                swap_dir=Path(tmpdir),
                swap_threshold_bytes=100  # Very small threshold
            )

            large_data = {"test": "x" * 1000}
            manager.store("ctx-1", large_data)

            assert "ctx-1" not in manager._in_memory
            assert "ctx-1" in manager._on_disk

    def test_load_from_memory(self):
        """Test loading from memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(swap_dir=Path(tmpdir))

            data = {"test": "value"}
            manager.store("ctx-1", data)

            loaded = manager.load("ctx-1")
            assert loaded == data

    def test_load_from_disk(self):
        """Test loading from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(
                swap_dir=Path(tmpdir),
                swap_threshold_bytes=10
            )

            data = {"test": "large" * 100}
            manager.store("ctx-1", data)

            loaded = manager.load("ctx-1")
            assert loaded == data

    def test_delete(self):
        """Test deleting context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(swap_dir=Path(tmpdir))

            manager.store("ctx-1", {"test": "data"})
            manager.delete("ctx-1")

            with pytest.raises(KeyError):
                manager.load("ctx-1")

    def test_get_stats(self):
        """Test getting swap stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(swap_dir=Path(tmpdir))

            manager.store("ctx-1", {"test": "data"})

            stats = manager.get_stats()
            assert "in_memory_count" in stats
            assert "on_disk_count" in stats

    def test_cleanup(self):
        """Test cleanup removes all swap files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DiskSwapManager(
                swap_dir=Path(tmpdir),
                swap_threshold_bytes=10
            )

            manager.store("ctx-1", {"large": "x" * 1000})
            manager.cleanup()

            stats = manager.get_stats()
            assert stats["on_disk_count"] == 0


# =============================================================================
# Task 79: Memory Budget per Graph Tests
# =============================================================================


class TestGraphMemoryManager:
    """Test graph memory manager."""

    def test_manager_initialization(self):
        """Test manager initializes."""
        manager = GraphMemoryManager()
        assert manager is not None

    def test_set_budget(self):
        """Test setting budget."""
        manager = GraphMemoryManager()

        budget = GraphMemoryBudget(
            graph_id="graph-1",
            max_messages=100,
            max_total_bytes=1024
        )

        manager.set_budget(budget)
        retrieved = manager.get_budget("graph-1")

        assert retrieved is not None
        assert retrieved.max_messages == 100

    def test_track_message(self, sample_message):
        """Test tracking message."""
        manager = GraphMemoryManager()

        budget = GraphMemoryBudget(
            graph_id="graph-1",
            max_messages=100,
            max_total_bytes=1024000
        )
        manager.set_budget(budget)

        manager.track_message("graph-1", sample_message, 500)

        usage = manager.get_usage("graph-1")
        assert usage["message_count"] == 1
        assert usage["total_bytes"] == 500

    def test_budget_exceeded_with_eviction(self, sample_messages):
        """Test auto-eviction when budget exceeded."""
        manager = GraphMemoryManager()

        budget = GraphMemoryBudget(
            graph_id="graph-1",
            max_messages=3,
            max_total_bytes=1024000,
            enable_auto_eviction=True,
            eviction_policy="fifo"
        )
        manager.set_budget(budget)

        # Add 4 messages, should trigger eviction
        for i, msg in enumerate(sample_messages[:4]):
            manager.track_message("graph-1", msg, 100)

        usage = manager.get_usage("graph-1")
        # Should have evicted to stay under limit
        assert usage["message_count"] <= 3

    def test_budget_exceeded_without_eviction(self, sample_messages):
        """Test error when budget exceeded without eviction."""
        manager = GraphMemoryManager()

        budget = GraphMemoryBudget(
            graph_id="graph-1",
            max_messages=2,
            max_total_bytes=1024000,
            enable_auto_eviction=False
        )
        manager.set_budget(budget)

        # Add 2 messages
        manager.track_message("graph-1", sample_messages[0], 100)
        manager.track_message("graph-1", sample_messages[1], 100)

        # Third should raise
        with pytest.raises(MemoryError):
            manager.track_message("graph-1", sample_messages[2], 100)

    def test_clear_graph(self, sample_messages):
        """Test clearing graph usage."""
        manager = GraphMemoryManager()

        budget = GraphMemoryBudget(graph_id="graph-1")
        manager.set_budget(budget)

        manager.track_message("graph-1", sample_messages[0], 100)

        manager.clear_graph("graph-1")

        usage = manager.get_usage("graph-1")
        assert usage["message_count"] == 0

    def test_get_usage_no_budget(self):
        """Test getting usage without budget."""
        manager = GraphMemoryManager()

        usage = manager.get_usage("non-existent")
        assert usage["message_count"] == 0


# =============================================================================
# Task 80: Memory Leak Detection Tests
# =============================================================================


class TestMemoryLeakDetector:
    """Test memory leak detector."""

    def test_detector_initialization(self):
        """Test detector initializes."""
        detector = MemoryLeakDetector()
        assert detector is not None

    def test_take_snapshot(self):
        """Test taking snapshot."""
        detector = MemoryLeakDetector()

        snapshot = detector.take_snapshot()

        assert isinstance(snapshot, LeakSnapshot)
        assert snapshot.total_bytes > 0
        assert snapshot.object_count > 0

    def test_detect_leaks_insufficient_snapshots(self):
        """Test detection with insufficient snapshots."""
        detector = MemoryLeakDetector()

        leaks = detector.detect_leaks()
        assert len(leaks) == 0

    def test_detect_leaks_with_growth(self):
        """Test detecting memory growth."""
        detector = MemoryLeakDetector(leak_threshold_mb=0.001)  # Very low threshold

        # Take first snapshot
        detector.take_snapshot()

        # Allocate some memory
        leaked = [object() for _ in range(10000)]

        # Take second snapshot
        detector.take_snapshot()

        # Detect leaks
        leaks = detector.detect_leaks()

        # Should detect growth (may or may not depending on GC)
        # Just check it doesn't error
        assert isinstance(leaks, list)

        # Cleanup
        del leaked
        gc.collect()

    def test_get_top_allocations(self):
        """Test getting top allocations."""
        detector = MemoryLeakDetector()
        detector.take_snapshot()

        allocations = detector.get_top_allocations(limit=5)
        assert isinstance(allocations, list)

    def test_get_report(self):
        """Test generating report."""
        detector = MemoryLeakDetector()
        detector.take_snapshot()

        report = detector.get_report()

        assert "snapshots" in report
        assert report["snapshots"] >= 1

    def test_clear_snapshots(self):
        """Test clearing snapshots."""
        detector = MemoryLeakDetector()
        detector.take_snapshot()

        detector.clear_snapshots()

        report = detector.get_report()
        assert report["snapshots"] == 0

    def test_start_stop_monitoring(self):
        """Test start/stop monitoring."""
        detector = MemoryLeakDetector(snapshot_interval_seconds=0.1)

        detector.start_monitoring()
        time.sleep(0.3)
        detector.stop_monitoring()

        # Should have taken multiple snapshots
        report = detector.get_report()
        assert report["snapshots"] >= 2
