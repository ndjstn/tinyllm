"""Advanced memory management for TinyLLM execution contexts.

This module provides comprehensive memory management features including:
- Object pooling for message allocation
- Zero-copy message passing with shared memory
- Memory-mapped storage for large contexts
- Garbage collection tuning and monitoring
- Context window sliding and LRU eviction
- Memory pressure callbacks and swap-to-disk
- Per-graph memory budgets
- Memory leak detection

These features enable efficient memory usage in long-running graph executions
and high-throughput scenarios.
"""

import gc
import mmap
import os
import pickle
import resource
import sys
import tempfile
import threading
import tracemalloc
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Set

from tinyllm.core.message import Message
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="memory")


# =============================================================================
# Task 71: Memory Pooling for Message Objects
# =============================================================================


class MessagePool:
    """Object pool for message instances to reduce allocation overhead.

    Maintains a pool of pre-allocated message objects that can be reused,
    reducing GC pressure and allocation latency in high-throughput scenarios.

    Thread-safe for concurrent access.
    """

    def __init__(self, pool_size: int = 1000, max_size: int = 10000):
        """Initialize message pool.

        Args:
            pool_size: Initial number of message slots to pre-allocate.
            max_size: Maximum pool size before overflow.
        """
        self.pool_size = pool_size
        self.max_size = max_size
        self._pool: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self._lock = threading.Lock()

        # Statistics
        self._allocated = 0
        self._reused = 0
        self._discarded = 0

        logger.info(
            "message_pool_initialized",
            pool_size=pool_size,
            max_size=max_size,
        )

    def acquire(self) -> Dict[str, Any]:
        """Acquire a message data dict from the pool.

        Returns:
            Reusable message data dictionary.
        """
        with self._lock:
            if self._pool:
                data = self._pool.pop()
                self._reused += 1
                logger.debug("message_pool_acquire_reused", reused_count=self._reused)
                return data
            else:
                # Pool exhausted, allocate new
                self._allocated += 1
                logger.debug("message_pool_acquire_new", allocated_count=self._allocated)
                return {}

    def release(self, data: Dict[str, Any]) -> None:
        """Return a message data dict to the pool.

        Args:
            data: Message data to return to pool.
        """
        with self._lock:
            if len(self._pool) < self.max_size:
                # Clear the data for reuse
                data.clear()
                self._pool.append(data)
                logger.debug("message_pool_released", pool_size=len(self._pool))
            else:
                # Pool full, discard
                self._discarded += 1
                logger.debug("message_pool_discarded", discarded_count=self._discarded)

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics.

        Returns:
            Dictionary with allocation stats.
        """
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "allocated": self._allocated,
                "reused": self._reused,
                "discarded": self._discarded,
                "reuse_rate": self._reused / max(self._allocated, 1),
            }

    def clear(self) -> None:
        """Clear the pool and reset statistics."""
        with self._lock:
            self._pool.clear()
            self._allocated = 0
            self._reused = 0
            self._discarded = 0
            logger.info("message_pool_cleared")


# Global message pool instance
_global_message_pool = MessagePool()


def get_message_pool() -> MessagePool:
    """Get the global message pool instance."""
    return _global_message_pool


# =============================================================================
# Task 72: Zero-Copy Message Passing
# =============================================================================


class SharedMessageBuffer:
    """Zero-copy message buffer using shared memory.

    Allows multiple nodes to access message data without copying,
    reducing memory overhead for large message payloads.

    Uses Python's mmap for shared memory regions.
    """

    def __init__(self, size_bytes: int = 1024 * 1024):  # 1MB default
        """Initialize shared buffer.

        Args:
            size_bytes: Size of shared memory region.
        """
        self.size_bytes = size_bytes
        self._buffer = mmap.mmap(-1, size_bytes)
        self._offset = 0
        self._lock = threading.Lock()
        self._allocations: Dict[int, int] = {}  # offset -> size

        logger.info("shared_buffer_initialized", size_bytes=size_bytes)

    def allocate(self, data: bytes) -> int:
        """Allocate space and write data to shared buffer.

        Args:
            data: Bytes to write.

        Returns:
            Offset of allocated region.

        Raises:
            MemoryError: If buffer is full.
        """
        size = len(data)

        with self._lock:
            if self._offset + size > self.size_bytes:
                raise MemoryError(
                    f"Shared buffer full: {self._offset + size} > {self.size_bytes}"
                )

            offset = self._offset
            self._buffer[offset:offset + size] = data
            self._allocations[offset] = size
            self._offset += size

            logger.debug(
                "shared_buffer_allocated",
                offset=offset,
                size=size,
                utilization_pct=int((self._offset / self.size_bytes) * 100),
            )

            return offset

    def read(self, offset: int) -> bytes:
        """Read data from shared buffer without copying.

        Args:
            offset: Offset to read from.

        Returns:
            Bytes at offset (view, not copy).
        """
        with self._lock:
            if offset not in self._allocations:
                raise ValueError(f"Invalid offset: {offset}")

            size = self._allocations[offset]
            # Return a view, not a copy
            return bytes(self._buffer[offset:offset + size])

    def deallocate(self, offset: int) -> None:
        """Mark region as deallocated.

        Args:
            offset: Offset to deallocate.
        """
        with self._lock:
            if offset in self._allocations:
                del self._allocations[offset]
                logger.debug("shared_buffer_deallocated", offset=offset)

    def compact(self) -> None:
        """Compact buffer by removing gaps (simple implementation)."""
        # For production, implement proper defragmentation
        with self._lock:
            self._offset = sum(self._allocations.values())
            logger.info("shared_buffer_compacted", new_offset=self._offset)

    def close(self) -> None:
        """Close and cleanup shared buffer."""
        self._buffer.close()
        logger.info("shared_buffer_closed")


# =============================================================================
# Task 73: Memory-Mapped Context Storage
# =============================================================================


class MemoryMappedContextStorage:
    """Memory-mapped storage for large execution contexts.

    Stores context data in memory-mapped files, allowing the OS to
    manage paging and reducing memory pressure for large contexts.
    """

    def __init__(self, storage_dir: Optional[Path] = None, max_file_size: int = 100 * 1024 * 1024):
        """Initialize memory-mapped storage.

        Args:
            storage_dir: Directory for storage files (temp dir if None).
            max_file_size: Maximum size per storage file.
        """
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "tinyllm_mmap"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size

        self._mmaps: Dict[str, mmap.mmap] = {}
        self._files: Dict[str, Any] = {}
        self._lock = threading.Lock()

        logger.info(
            "mmap_storage_initialized",
            storage_dir=str(self.storage_dir),
            max_file_size=max_file_size,
        )

    def store(self, trace_id: str, data: Any) -> str:
        """Store data in memory-mapped file.

        Args:
            trace_id: Trace ID for context.
            data: Data to store (will be pickled).

        Returns:
            Storage key for retrieval.
        """
        storage_key = f"{trace_id}_{datetime.utcnow().timestamp()}"
        file_path = self.storage_dir / f"{storage_key}.mmap"

        # Serialize data
        serialized = pickle.dumps(data)
        size = len(serialized)

        if size > self.max_file_size:
            raise ValueError(
                f"Data size {size} exceeds max file size {self.max_file_size}"
            )

        with self._lock:
            # Create file
            with open(file_path, "wb") as f:
                f.write(serialized)

            # Memory map it
            file_obj = open(file_path, "r+b")
            mmap_obj = mmap.mmap(file_obj.fileno(), 0)

            self._files[storage_key] = file_obj
            self._mmaps[storage_key] = mmap_obj

            logger.info(
                "mmap_storage_stored",
                storage_key=storage_key,
                size_bytes=size,
                file_path=str(file_path),
            )

            return storage_key

    def load(self, storage_key: str) -> Any:
        """Load data from memory-mapped file.

        Args:
            storage_key: Key returned from store().

        Returns:
            Deserialized data.
        """
        with self._lock:
            if storage_key not in self._mmaps:
                raise KeyError(f"Storage key not found: {storage_key}")

            mmap_obj = self._mmaps[storage_key]
            data = pickle.loads(mmap_obj[:])

            logger.debug("mmap_storage_loaded", storage_key=storage_key)
            return data

    def delete(self, storage_key: str) -> None:
        """Delete memory-mapped storage.

        Args:
            storage_key: Key to delete.
        """
        with self._lock:
            if storage_key in self._mmaps:
                # Close mmap and file
                self._mmaps[storage_key].close()
                self._files[storage_key].close()

                # Delete file
                file_path = self.storage_dir / f"{storage_key}.mmap"
                if file_path.exists():
                    file_path.unlink()

                del self._mmaps[storage_key]
                del self._files[storage_key]

                logger.info("mmap_storage_deleted", storage_key=storage_key)

    def cleanup(self) -> None:
        """Clean up all storage files."""
        with self._lock:
            for key in list(self._mmaps.keys()):
                self.delete(key)

            logger.info("mmap_storage_cleaned_up")


# =============================================================================
# Task 74: Garbage Collection Tuning
# =============================================================================


@dataclass
class GCConfig:
    """Configuration for garbage collection tuning."""

    # GC generation thresholds
    gen0_threshold: int = 700
    gen1_threshold: int = 10
    gen2_threshold: int = 10

    # GC control
    enable_auto_gc: bool = True
    manual_gc_interval_seconds: float = 60.0

    # Monitoring
    track_allocations: bool = False
    log_collections: bool = True


class GarbageCollectionTuner:
    """Tune and monitor garbage collection for optimal memory performance.

    Provides control over Python's garbage collector, including:
    - Custom generation thresholds
    - Manual collection triggers
    - Collection statistics and monitoring
    """

    def __init__(self, config: Optional[GCConfig] = None):
        """Initialize GC tuner.

        Args:
            config: GC configuration.
        """
        self.config = config or GCConfig()
        self._collection_stats: List[Dict[str, Any]] = []
        self._last_manual_gc = datetime.utcnow()

        self._apply_config()

        logger.info(
            "gc_tuner_initialized",
            gen0_threshold=self.config.gen0_threshold,
            gen1_threshold=self.config.gen1_threshold,
            gen2_threshold=self.config.gen2_threshold,
            enable_auto_gc=self.config.enable_auto_gc,
        )

    def _apply_config(self) -> None:
        """Apply GC configuration."""
        # Set generation thresholds
        gc.set_threshold(
            self.config.gen0_threshold,
            self.config.gen1_threshold,
            self.config.gen2_threshold,
        )

        # Enable/disable auto GC
        if self.config.enable_auto_gc:
            gc.enable()
        else:
            gc.disable()

        # Enable tracking if requested
        if self.config.track_allocations:
            tracemalloc.start()

    def collect(self, generation: int = 2) -> int:
        """Manually trigger garbage collection.

        Args:
            generation: GC generation to collect (0, 1, or 2).

        Returns:
            Number of objects collected.
        """
        before = len(gc.get_objects())
        collected = gc.collect(generation)
        after = len(gc.get_objects())

        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "generation": generation,
            "collected": collected,
            "objects_before": before,
            "objects_after": after,
            "freed": before - after,
        }

        self._collection_stats.append(stats)
        self._last_manual_gc = datetime.utcnow()

        if self.config.log_collections:
            logger.info("gc_collection_completed", **stats)

        return collected

    def maybe_collect(self) -> Optional[int]:
        """Collect if enough time has passed since last manual collection.

        Returns:
            Number of objects collected, or None if skipped.
        """
        elapsed = (datetime.utcnow() - self._last_manual_gc).total_seconds()

        if elapsed >= self.config.manual_gc_interval_seconds:
            return self.collect()

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get GC statistics.

        Returns:
            Dictionary with GC stats.
        """
        counts = gc.get_count()
        stats = gc.get_stats()

        return {
            "counts": {"gen0": counts[0], "gen1": counts[1], "gen2": counts[2]},
            "stats": stats,
            "collections": len(self._collection_stats),
            "last_collection": self._collection_stats[-1] if self._collection_stats else None,
        }

    def reset_stats(self) -> None:
        """Reset collection statistics."""
        self._collection_stats.clear()
        logger.info("gc_stats_reset")


# =============================================================================
# Task 75: Context Window Sliding
# =============================================================================


class ContextWindowSlider:
    """Implements sliding window over context messages.

    Maintains a fixed-size window of recent messages, automatically
    evicting old messages as new ones are added.
    """

    def __init__(self, window_size: int = 100, slide_size: int = 10):
        """Initialize context window.

        Args:
            window_size: Maximum number of messages in window.
            slide_size: Number of messages to remove when window is full.
        """
        self.window_size = window_size
        self.slide_size = slide_size
        self._messages: Deque[Message] = deque(maxlen=window_size)
        self._evicted_count = 0

        logger.info(
            "context_window_initialized",
            window_size=window_size,
            slide_size=slide_size,
        )

    def add(self, message: Message) -> Optional[List[Message]]:
        """Add message to window, evicting old messages if needed.

        Args:
            message: Message to add.

        Returns:
            List of evicted messages, or None if no eviction.
        """
        evicted = None

        # Check if we need to evict
        if len(self._messages) >= self.window_size:
            # Evict slide_size messages from the start
            evicted = [self._messages.popleft() for _ in range(min(self.slide_size, len(self._messages)))]
            self._evicted_count += len(evicted)

            logger.debug(
                "context_window_evicted",
                evicted_count=len(evicted),
                total_evicted=self._evicted_count,
                window_size=len(self._messages),
            )

        self._messages.append(message)

        return evicted

    def get_window(self) -> List[Message]:
        """Get current window of messages.

        Returns:
            List of messages in window (most recent last).
        """
        return list(self._messages)

    def get_recent(self, count: int) -> List[Message]:
        """Get most recent N messages.

        Args:
            count: Number of recent messages to get.

        Returns:
            List of recent messages.
        """
        count = min(count, len(self._messages))
        return list(self._messages)[-count:] if count > 0 else []

    def clear(self) -> None:
        """Clear the window."""
        self._messages.clear()
        logger.debug("context_window_cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get window statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "current_size": len(self._messages),
            "max_size": self.window_size,
            "evicted_total": self._evicted_count,
            "utilization_pct": int((len(self._messages) / self.window_size) * 100),
        }


# =============================================================================
# Task 76: LRU Eviction for Context
# =============================================================================


class LRUContextCache:
    """LRU cache for context messages with automatic eviction.

    Uses Least Recently Used eviction policy to manage message cache.
    """

    def __init__(self, capacity: int = 1000):
        """Initialize LRU cache.

        Args:
            capacity: Maximum number of messages to cache.
        """
        self.capacity = capacity
        self._cache: OrderedDict[str, Message] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info("lru_cache_initialized", capacity=capacity)

    def get(self, message_id: str) -> Optional[Message]:
        """Get message from cache.

        Args:
            message_id: Message ID to retrieve.

        Returns:
            Message if found, None otherwise.
        """
        with self._lock:
            if message_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(message_id)
                self._hits += 1
                logger.debug("lru_cache_hit", message_id=message_id, hit_rate=self.get_hit_rate())
                return self._cache[message_id]

            self._misses += 1
            logger.debug("lru_cache_miss", message_id=message_id, hit_rate=self.get_hit_rate())
            return None

    def put(self, message: Message) -> Optional[Message]:
        """Put message in cache, evicting LRU if needed.

        Args:
            message: Message to cache.

        Returns:
            Evicted message if cache was full, None otherwise.
        """
        with self._lock:
            message_id = message.message_id

            if message_id in self._cache:
                # Update and move to end
                self._cache.move_to_end(message_id)
                self._cache[message_id] = message
                return None

            # Add new message
            evicted = None
            if len(self._cache) >= self.capacity:
                # Evict LRU (first item)
                evicted_id, evicted = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(
                    "lru_cache_evicted",
                    evicted_id=evicted_id,
                    evictions_total=self._evictions,
                )

            self._cache[message_id] = message
            logger.debug("lru_cache_put", message_id=message_id, cache_size=len(self._cache))

            return evicted

    def remove(self, message_id: str) -> bool:
        """Remove message from cache.

        Args:
            message_id: Message ID to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if message_id in self._cache:
                del self._cache[message_id]
                logger.debug("lru_cache_removed", message_id=message_id)
                return True
            return False

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            logger.info("lru_cache_cleared")

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate (0.0 to 1.0).
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with stats.
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "capacity": self.capacity,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self.get_hit_rate(),
                "utilization_pct": int((len(self._cache) / self.capacity) * 100),
            }


# =============================================================================
# Task 77: Memory Pressure Callbacks
# =============================================================================


# Callback protocol
class MemoryPressureCallback(Protocol):
    """Protocol for memory pressure callbacks."""

    def __call__(self, pressure_level: str, stats: Dict[str, Any]) -> None:
        """Handle memory pressure event.

        Args:
            pressure_level: One of 'low', 'medium', 'high', 'critical'.
            stats: Memory statistics.
        """
        ...


@dataclass
class MemoryPressureThresholds:
    """Thresholds for memory pressure levels."""

    low_percent: float = 50.0
    medium_percent: float = 70.0
    high_percent: float = 85.0
    critical_percent: float = 95.0


class MemoryPressureMonitor:
    """Monitor memory usage and trigger callbacks on pressure events.

    Periodically checks memory usage and invokes registered callbacks
    when pressure thresholds are exceeded.
    """

    def __init__(
        self,
        thresholds: Optional[MemoryPressureThresholds] = None,
        check_interval_seconds: float = 5.0,
    ):
        """Initialize pressure monitor.

        Args:
            thresholds: Pressure level thresholds.
            check_interval_seconds: How often to check memory.
        """
        self.thresholds = thresholds or MemoryPressureThresholds()
        self.check_interval_seconds = check_interval_seconds
        self._callbacks: List[MemoryPressureCallback] = []
        self._last_level = "low"
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None

        logger.info(
            "memory_pressure_monitor_initialized",
            thresholds=self.thresholds.__dict__,
        )

    def register_callback(self, callback: MemoryPressureCallback) -> None:
        """Register a callback for pressure events.

        Args:
            callback: Callback function.
        """
        self._callbacks.append(callback)
        logger.debug("memory_pressure_callback_registered", callback_count=len(self._callbacks))

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics.

        Returns:
            Dictionary with memory stats.
        """
        # Get process memory info
        rusage = resource.getrusage(resource.RUSAGE_SELF)

        # Get GC stats
        gc_counts = gc.get_count()

        stats = {
            "rss_bytes": rusage.ru_maxrss * 1024,  # Convert KB to bytes on Linux
            "gc_gen0": gc_counts[0],
            "gc_gen1": gc_counts[1],
            "gc_gen2": gc_counts[2],
            "object_count": len(gc.get_objects()),
        }

        # Get tracemalloc stats if available
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            stats["traced_current_bytes"] = current
            stats["traced_peak_bytes"] = peak

        return stats

    def calculate_pressure_level(self, stats: Dict[str, Any]) -> str:
        """Calculate current pressure level.

        Args:
            stats: Memory statistics.

        Returns:
            Pressure level: 'low', 'medium', 'high', or 'critical'.
        """
        # Use RSS as primary metric
        rss = stats.get("rss_bytes", 0)

        # Get system memory limit (this is a simplified approach)
        # In production, use proper system memory detection
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                for line in meminfo.split("\n"):
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        total_bytes = total_kb * 1024
                        usage_percent = (rss / total_bytes) * 100

                        if usage_percent >= self.thresholds.critical_percent:
                            return "critical"
                        elif usage_percent >= self.thresholds.high_percent:
                            return "high"
                        elif usage_percent >= self.thresholds.medium_percent:
                            return "medium"
                        else:
                            return "low"
        except Exception as e:
            logger.warning("failed_to_read_meminfo", error=str(e))

        # Fallback: use object count heuristic
        obj_count = stats.get("object_count", 0)
        if obj_count > 1000000:
            return "critical"
        elif obj_count > 500000:
            return "high"
        elif obj_count > 100000:
            return "medium"
        else:
            return "low"

    def check_pressure(self) -> None:
        """Check memory pressure and invoke callbacks if needed."""
        stats = self.get_memory_stats()
        level = self.calculate_pressure_level(stats)

        # Only trigger callbacks if level changed or is high/critical
        if level != self._last_level or level in ("high", "critical"):
            logger.info(
                "memory_pressure_detected",
                level=level,
                previous_level=self._last_level,
                **stats,
            )

            # Invoke callbacks
            for callback in self._callbacks:
                try:
                    callback(level, stats)
                except Exception as e:
                    logger.error(
                        "memory_pressure_callback_failed",
                        error=str(e),
                        callback=str(callback),
                    )

            self._last_level = level

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitoring:
            logger.warning("memory_pressure_monitoring_already_started")
            return

        self._monitoring = True

        def monitor_loop():
            import time
            while self._monitoring:
                self.check_pressure()
                time.sleep(self.check_interval_seconds)

        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()

        logger.info("memory_pressure_monitoring_started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)

        logger.info("memory_pressure_monitoring_stopped")


# =============================================================================
# Task 78: Swap-to-Disk for Large Contexts
# =============================================================================


class DiskSwapManager:
    """Manage swapping of large contexts to disk to free memory.

    Automatically swaps out inactive contexts to disk and swaps them
    back in when accessed.
    """

    def __init__(
        self,
        swap_dir: Optional[Path] = None,
        swap_threshold_bytes: int = 10 * 1024 * 1024,  # 10MB
    ):
        """Initialize swap manager.

        Args:
            swap_dir: Directory for swap files.
            swap_threshold_bytes: Threshold for swapping contexts.
        """
        self.swap_dir = swap_dir or Path(tempfile.gettempdir()) / "tinyllm_swap"
        self.swap_dir.mkdir(parents=True, exist_ok=True)
        self.swap_threshold_bytes = swap_threshold_bytes

        self._in_memory: Dict[str, Any] = {}
        self._on_disk: Set[str] = set()
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        self._swapped_out = 0
        self._swapped_in = 0

        logger.info(
            "disk_swap_manager_initialized",
            swap_dir=str(self.swap_dir),
            swap_threshold_bytes=swap_threshold_bytes,
        )

    def _get_swap_path(self, context_id: str) -> Path:
        """Get swap file path for context."""
        return self.swap_dir / f"{context_id}.swap"

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        return sys.getsizeof(pickle.dumps(data))

    def store(self, context_id: str, data: Any) -> None:
        """Store context, swapping to disk if large.

        Args:
            context_id: Context identifier.
            data: Context data.
        """
        with self._lock:
            size = self._estimate_size(data)

            if size >= self.swap_threshold_bytes:
                # Swap to disk
                swap_path = self._get_swap_path(context_id)
                with open(swap_path, "wb") as f:
                    pickle.dump(data, f)

                self._on_disk.add(context_id)
                self._swapped_out += 1

                logger.info(
                    "context_swapped_to_disk",
                    context_id=context_id,
                    size_bytes=size,
                    swap_path=str(swap_path),
                )
            else:
                # Keep in memory
                self._in_memory[context_id] = data

                logger.debug(
                    "context_stored_in_memory",
                    context_id=context_id,
                    size_bytes=size,
                )

            self._access_times[context_id] = datetime.utcnow()

    def load(self, context_id: str) -> Any:
        """Load context, swapping from disk if needed.

        Args:
            context_id: Context identifier.

        Returns:
            Context data.
        """
        with self._lock:
            # Update access time
            self._access_times[context_id] = datetime.utcnow()

            # Check if in memory
            if context_id in self._in_memory:
                logger.debug("context_loaded_from_memory", context_id=context_id)
                return self._in_memory[context_id]

            # Check if on disk
            if context_id in self._on_disk:
                swap_path = self._get_swap_path(context_id)
                with open(swap_path, "rb") as f:
                    data = pickle.load(f)

                self._swapped_in += 1

                logger.info(
                    "context_swapped_from_disk",
                    context_id=context_id,
                    swap_path=str(swap_path),
                )

                return data

            raise KeyError(f"Context not found: {context_id}")

    def delete(self, context_id: str) -> None:
        """Delete context from memory and disk.

        Args:
            context_id: Context identifier.
        """
        with self._lock:
            # Remove from memory
            if context_id in self._in_memory:
                del self._in_memory[context_id]

            # Remove from disk
            if context_id in self._on_disk:
                swap_path = self._get_swap_path(context_id)
                if swap_path.exists():
                    swap_path.unlink()
                self._on_disk.remove(context_id)

            # Remove access time
            if context_id in self._access_times:
                del self._access_times[context_id]

            logger.debug("context_deleted", context_id=context_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get swap statistics.

        Returns:
            Dictionary with stats.
        """
        with self._lock:
            return {
                "in_memory_count": len(self._in_memory),
                "on_disk_count": len(self._on_disk),
                "swapped_out": self._swapped_out,
                "swapped_in": self._swapped_in,
            }

    def cleanup(self) -> None:
        """Clean up all swap files."""
        with self._lock:
            for context_id in list(self._on_disk):
                self.delete(context_id)

            logger.info("disk_swap_cleaned_up")


# =============================================================================
# Task 79: Memory Budget per Graph
# =============================================================================


@dataclass
class GraphMemoryBudget:
    """Memory budget configuration for a graph."""

    graph_id: str
    max_messages: int = 1000
    max_total_bytes: int = 100 * 1024 * 1024  # 100MB
    max_message_bytes: int = 1 * 1024 * 1024  # 1MB
    enable_auto_eviction: bool = True
    eviction_policy: str = "lru"  # 'lru', 'fifo', 'sliding'


class GraphMemoryManager:
    """Manage memory budgets for multiple graphs.

    Enforces per-graph memory limits and handles eviction when
    budgets are exceeded.
    """

    def __init__(self):
        """Initialize graph memory manager."""
        self._budgets: Dict[str, GraphMemoryBudget] = {}
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        logger.info("graph_memory_manager_initialized")

    def set_budget(self, budget: GraphMemoryBudget) -> None:
        """Set memory budget for a graph.

        Args:
            budget: Budget configuration.
        """
        with self._lock:
            self._budgets[budget.graph_id] = budget

            if budget.graph_id not in self._usage:
                self._usage[budget.graph_id] = {
                    "message_count": 0,
                    "total_bytes": 0,
                    "messages": [],
                }

            logger.info(
                "graph_budget_set",
                graph_id=budget.graph_id,
                max_messages=budget.max_messages,
                max_total_bytes=budget.max_total_bytes,
            )

    def get_budget(self, graph_id: str) -> Optional[GraphMemoryBudget]:
        """Get budget for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Budget if set, None otherwise.
        """
        with self._lock:
            return self._budgets.get(graph_id)

    def track_message(self, graph_id: str, message: Message, size_bytes: int) -> None:
        """Track message allocation for a graph.

        Args:
            graph_id: Graph identifier.
            message: Message being tracked.
            size_bytes: Size of message in bytes.

        Raises:
            MemoryError: If budget exceeded and auto-eviction disabled.
        """
        with self._lock:
            if graph_id not in self._budgets:
                logger.warning("graph_budget_not_set", graph_id=graph_id)
                return

            budget = self._budgets[graph_id]
            usage = self._usage[graph_id]

            # Check if we need to evict
            if usage["message_count"] >= budget.max_messages or \
               usage["total_bytes"] + size_bytes > budget.max_total_bytes:

                if budget.enable_auto_eviction:
                    self._evict_messages(graph_id, budget)
                else:
                    raise MemoryError(
                        f"Graph {graph_id} memory budget exceeded: "
                        f"{usage['message_count']}/{budget.max_messages} messages, "
                        f"{usage['total_bytes']}/{budget.max_total_bytes} bytes"
                    )

            # Track message
            usage["messages"].append({"message": message, "size": size_bytes})
            usage["message_count"] += 1
            usage["total_bytes"] += size_bytes

            logger.debug(
                "graph_message_tracked",
                graph_id=graph_id,
                message_count=usage["message_count"],
                total_bytes=usage["total_bytes"],
            )

    def _evict_messages(self, graph_id: str, budget: GraphMemoryBudget) -> None:
        """Evict messages according to budget policy.

        Args:
            graph_id: Graph identifier.
            budget: Budget configuration.
        """
        usage = self._usage[graph_id]

        if budget.eviction_policy == "fifo":
            # Remove oldest (first) message
            if usage["messages"]:
                evicted = usage["messages"].pop(0)
                usage["message_count"] -= 1
                usage["total_bytes"] -= evicted["size"]

                logger.info(
                    "graph_message_evicted_fifo",
                    graph_id=graph_id,
                    message_id=evicted["message"].message_id,
                )

        elif budget.eviction_policy == "lru":
            # For simplicity, remove oldest (FIFO for now)
            # In production, track access times
            if usage["messages"]:
                evicted = usage["messages"].pop(0)
                usage["message_count"] -= 1
                usage["total_bytes"] -= evicted["size"]

                logger.info(
                    "graph_message_evicted_lru",
                    graph_id=graph_id,
                    message_id=evicted["message"].message_id,
                )

        elif budget.eviction_policy == "sliding":
            # Remove multiple old messages (25% of max)
            evict_count = max(1, budget.max_messages // 4)
            for _ in range(min(evict_count, len(usage["messages"]))):
                evicted = usage["messages"].pop(0)
                usage["message_count"] -= 1
                usage["total_bytes"] -= evicted["size"]

            logger.info(
                "graph_messages_evicted_sliding",
                graph_id=graph_id,
                evicted_count=evict_count,
            )

    def get_usage(self, graph_id: str) -> Dict[str, Any]:
        """Get current memory usage for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Usage statistics.
        """
        with self._lock:
            if graph_id not in self._usage:
                return {"message_count": 0, "total_bytes": 0}

            usage = self._usage[graph_id]
            budget = self._budgets.get(graph_id)

            stats = {
                "message_count": usage["message_count"],
                "total_bytes": usage["total_bytes"],
            }

            if budget:
                stats["message_utilization_pct"] = int(
                    (usage["message_count"] / budget.max_messages) * 100
                )
                stats["bytes_utilization_pct"] = int(
                    (usage["total_bytes"] / budget.max_total_bytes) * 100
                )

            return stats

    def clear_graph(self, graph_id: str) -> None:
        """Clear all tracked messages for a graph.

        Args:
            graph_id: Graph identifier.
        """
        with self._lock:
            if graph_id in self._usage:
                self._usage[graph_id] = {
                    "message_count": 0,
                    "total_bytes": 0,
                    "messages": [],
                }

                logger.info("graph_memory_cleared", graph_id=graph_id)


# =============================================================================
# Task 80: Memory Leak Detection
# =============================================================================


@dataclass
class LeakSnapshot:
    """Snapshot of memory state for leak detection."""

    timestamp: datetime
    total_bytes: int
    object_count: int
    top_types: Dict[str, int]
    tracemalloc_snapshot: Any = None


class MemoryLeakDetector:
    """Detect memory leaks by tracking allocations over time.

    Uses tracemalloc and periodic snapshots to identify growing
    memory usage patterns that may indicate leaks.
    """

    def __init__(
        self,
        snapshot_interval_seconds: float = 60.0,
        leak_threshold_mb: float = 10.0,
    ):
        """Initialize leak detector.

        Args:
            snapshot_interval_seconds: How often to take snapshots.
            leak_threshold_mb: MB growth threshold to flag as potential leak.
        """
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.leak_threshold_bytes = int(leak_threshold_mb * 1024 * 1024)

        self._snapshots: List[LeakSnapshot] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None

        # Start tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)  # Track top 10 frames

        logger.info(
            "memory_leak_detector_initialized",
            snapshot_interval_seconds=snapshot_interval_seconds,
            leak_threshold_mb=leak_threshold_mb,
        )

    def take_snapshot(self) -> LeakSnapshot:
        """Take a memory snapshot.

        Returns:
            Snapshot of current memory state.
        """
        # Get object counts by type
        objects = gc.get_objects()
        type_counts: Dict[str, int] = {}

        for obj in objects:
            type_name = type(obj).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Get top 10 types
        top_types = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        # Get tracemalloc snapshot
        tm_snapshot = None
        if tracemalloc.is_tracing():
            tm_snapshot = tracemalloc.take_snapshot()

        # Get current memory
        current_bytes, _ = tracemalloc.get_traced_memory()

        snapshot = LeakSnapshot(
            timestamp=datetime.utcnow(),
            total_bytes=current_bytes,
            object_count=len(objects),
            top_types=top_types,
            tracemalloc_snapshot=tm_snapshot,
        )

        self._snapshots.append(snapshot)

        logger.debug(
            "memory_snapshot_taken",
            total_bytes=current_bytes,
            object_count=len(objects),
            snapshot_count=len(self._snapshots),
        )

        return snapshot

    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Analyze snapshots to detect potential leaks.

        Returns:
            List of detected leak patterns.
        """
        if len(self._snapshots) < 2:
            logger.debug("not_enough_snapshots_for_leak_detection")
            return []

        leaks = []

        # Compare most recent snapshot with first
        first = self._snapshots[0]
        latest = self._snapshots[-1]

        growth_bytes = latest.total_bytes - first.total_bytes
        growth_objects = latest.object_count - first.object_count

        if growth_bytes > self.leak_threshold_bytes:
            leak = {
                "type": "total_memory_growth",
                "growth_bytes": growth_bytes,
                "growth_mb": growth_bytes / (1024 * 1024),
                "start_bytes": first.total_bytes,
                "end_bytes": latest.total_bytes,
                "duration_seconds": (latest.timestamp - first.timestamp).total_seconds(),
            }

            leaks.append(leak)

            logger.warning(
                "potential_memory_leak_detected",
                **leak,
            )

        # Check for growing object types
        for type_name, latest_count in latest.top_types.items():
            first_count = first.top_types.get(type_name, 0)
            growth = latest_count - first_count

            if growth > 1000:  # Arbitrary threshold
                leak = {
                    "type": "object_type_growth",
                    "object_type": type_name,
                    "growth_count": growth,
                    "start_count": first_count,
                    "end_count": latest_count,
                }

                leaks.append(leak)

                logger.warning(
                    "potential_object_leak_detected",
                    **leak,
                )

        return leaks

    def get_top_allocations(self, limit: int = 10) -> List[str]:
        """Get top memory allocations from latest snapshot.

        Args:
            limit: Number of top allocations to return.

        Returns:
            List of allocation descriptions.
        """
        if not self._snapshots:
            return []

        latest = self._snapshots[-1]
        if not latest.tracemalloc_snapshot:
            return []

        top_stats = latest.tracemalloc_snapshot.statistics("lineno")

        return [
            f"{stat.traceback.format()[0]}: {stat.size / 1024:.1f} KB"
            for stat in top_stats[:limit]
        ]

    def start_monitoring(self) -> None:
        """Start background leak detection."""
        if self._monitoring:
            logger.warning("leak_detector_already_monitoring")
            return

        self._monitoring = True

        def monitor_loop():
            import time
            while self._monitoring:
                self.take_snapshot()
                self.detect_leaks()
                time.sleep(self.snapshot_interval_seconds)

        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()

        logger.info("leak_detector_monitoring_started")

    def stop_monitoring(self) -> None:
        """Stop background leak detection."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)

        logger.info("leak_detector_monitoring_stopped")

    def get_report(self) -> Dict[str, Any]:
        """Generate leak detection report.

        Returns:
            Dictionary with leak analysis.
        """
        if not self._snapshots:
            return {"snapshots": 0, "leaks": []}

        return {
            "snapshots": len(self._snapshots),
            "first_snapshot": {
                "timestamp": self._snapshots[0].timestamp.isoformat(),
                "total_bytes": self._snapshots[0].total_bytes,
                "object_count": self._snapshots[0].object_count,
            },
            "latest_snapshot": {
                "timestamp": self._snapshots[-1].timestamp.isoformat(),
                "total_bytes": self._snapshots[-1].total_bytes,
                "object_count": self._snapshots[-1].object_count,
            },
            "leaks": self.detect_leaks(),
            "top_allocations": self.get_top_allocations(),
        }

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
        logger.info("leak_detector_snapshots_cleared")
