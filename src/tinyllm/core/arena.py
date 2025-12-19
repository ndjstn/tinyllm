"""Arena allocators for efficient memory management.

Provides arena-based allocation for short-lived objects in graph execution.
Arenas allocate memory in chunks and can deallocate all at once, improving
performance and reducing fragmentation.
"""

import sys
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="arena")

T = TypeVar("T")


class ArenaStats(BaseModel):
    """Statistics about arena memory usage."""

    model_config = {"extra": "forbid"}

    arena_id: str = Field(description="Arena identifier")
    chunk_size: int = Field(ge=1, description="Size of each chunk in bytes")
    chunk_count: int = Field(ge=0, description="Number of allocated chunks")
    object_count: int = Field(ge=0, description="Number of objects in arena")
    total_allocated: int = Field(ge=0, description="Total bytes allocated")
    total_used: int = Field(ge=0, description="Total bytes used by objects")
    utilization: float = Field(ge=0.0, le=1.0, description="Memory utilization ratio")
    fragmentation: float = Field(ge=0.0, le=1.0, description="Fragmentation ratio")


class Arena(Generic[T]):
    """Arena allocator for objects of a specific type.

    Allocates memory in large chunks and provides objects from those chunks.
    Entire arena can be deallocated at once, making it efficient for
    short-lived object graphs.

    Example:
        >>> arena = Arena[Message](chunk_size=1024*1024)  # 1MB chunks
        >>> msg1 = arena.allocate(Message(...))
        >>> msg2 = arena.allocate(Message(...))
        >>> arena.reset()  # Deallocate all at once
    """

    def __init__(
        self,
        arena_id: str = "default",
        chunk_size: int = 1024 * 1024,  # 1MB default
        max_chunks: int = 100,
    ):
        """Initialize arena allocator.

        Args:
            arena_id: Unique identifier for this arena.
            chunk_size: Size of each chunk in bytes.
            max_chunks: Maximum number of chunks to allocate.
        """
        self.arena_id = arena_id
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks

        # Storage
        self._chunks: List[List[T]] = []
        self._current_chunk: List[T] = []
        self._current_size = 0
        self._object_count = 0
        self._total_size = 0

        # Metadata
        self._object_sizes: Dict[int, int] = {}  # id(obj) -> size

        logger.info(
            "arena_created",
            arena_id=arena_id,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
        )

    def allocate(self, obj: T) -> T:
        """Allocate object in arena.

        Args:
            obj: Object to allocate.

        Returns:
            The same object (for chaining).

        Raises:
            MemoryError: If max chunks exceeded.
        """
        # Calculate object size
        obj_size = self._get_object_size(obj)

        # Check if we need a new chunk
        if self._current_size + obj_size > self.chunk_size:
            self._new_chunk()

        # Add to current chunk
        self._current_chunk.append(obj)
        self._current_size += obj_size
        self._total_size += obj_size
        self._object_count += 1
        self._object_sizes[id(obj)] = obj_size

        logger.debug(
            "object_allocated",
            arena_id=self.arena_id,
            object_size=obj_size,
            chunk_count=len(self._chunks) + 1,
            object_count=self._object_count,
        )

        return obj

    def _new_chunk(self) -> None:
        """Allocate a new chunk.

        Raises:
            MemoryError: If max chunks exceeded.
        """
        if len(self._chunks) >= self.max_chunks:
            logger.error(
                "arena_max_chunks_exceeded",
                arena_id=self.arena_id,
                max_chunks=self.max_chunks,
            )
            raise MemoryError(
                f"Arena {self.arena_id} exceeded max chunks {self.max_chunks}"
            )

        if self._current_chunk:
            self._chunks.append(self._current_chunk)

        self._current_chunk = []
        self._current_size = 0

        logger.debug(
            "chunk_allocated",
            arena_id=self.arena_id,
            chunk_number=len(self._chunks) + 1,
        )

    def _get_object_size(self, obj: Any) -> int:
        """Get approximate size of object in bytes.

        Args:
            obj: Object to measure.

        Returns:
            Size in bytes.
        """
        try:
            # Try to use sys.getsizeof for rough estimate
            if hasattr(obj, "model_dump_json"):
                # Pydantic model
                return sys.getsizeof(obj.model_dump_json())
            else:
                return sys.getsizeof(obj)
        except Exception as e:
            logger.warning(
                "size_calculation_failed",
                arena_id=self.arena_id,
                error=str(e),
            )
            return 1024  # Default estimate

    def reset(self) -> int:
        """Reset arena, deallocating all objects.

        Returns:
            Number of objects deallocated.
        """
        count = self._object_count

        # Clear all chunks
        self._chunks.clear()
        self._current_chunk.clear()
        self._object_sizes.clear()

        # Reset counters
        self._current_size = 0
        self._object_count = 0
        self._total_size = 0

        logger.info(
            "arena_reset",
            arena_id=self.arena_id,
            objects_deallocated=count,
        )

        return count

    def get_stats(self) -> ArenaStats:
        """Get arena statistics.

        Returns:
            ArenaStats with current usage metrics.
        """
        # Count chunks (include current if it has any data)
        chunk_count = len(self._chunks)
        if self._current_chunk:
            chunk_count += 1

        # Calculate total allocated space
        # Only count allocated chunks
        if chunk_count == 0:
            total_allocated = self.chunk_size  # At least one chunk space
        else:
            total_allocated = chunk_count * self.chunk_size

        # Calculate utilization (cap at 1.0)
        utilization = min(1.0, self._total_size / total_allocated if total_allocated > 0 else 0.0)

        # Calculate fragmentation (wasted space in current chunk)
        if self._current_chunk or chunk_count > 0:
            wasted = self.chunk_size - self._current_size
            fragmentation = max(0.0, min(1.0, wasted / self.chunk_size if self.chunk_size > 0 else 0.0))
        else:
            fragmentation = 1.0  # Empty arena is fully fragmented

        return ArenaStats(
            arena_id=self.arena_id,
            chunk_size=self.chunk_size,
            chunk_count=chunk_count,
            object_count=self._object_count,
            total_allocated=total_allocated,
            total_used=self._total_size,
            utilization=utilization,
            fragmentation=fragmentation,
        )

    def __len__(self) -> int:
        """Get number of objects in arena."""
        return self._object_count

    def __contains__(self, obj: T) -> bool:
        """Check if object is in arena."""
        return id(obj) in self._object_sizes


class ArenaManager:
    """Manages multiple arenas for different object types."""

    def __init__(self):
        """Initialize arena manager."""
        self._arenas: Dict[str, Arena] = {}
        logger.info("arena_manager_created")

    def create_arena(
        self,
        arena_id: str,
        chunk_size: int = 1024 * 1024,
        max_chunks: int = 100,
    ) -> Arena:
        """Create a new arena.

        Args:
            arena_id: Unique identifier for arena.
            chunk_size: Size of each chunk in bytes.
            max_chunks: Maximum number of chunks.

        Returns:
            The created arena.

        Raises:
            ValueError: If arena_id already exists.
        """
        if arena_id in self._arenas:
            raise ValueError(f"Arena {arena_id} already exists")

        arena = Arena(arena_id=arena_id, chunk_size=chunk_size, max_chunks=max_chunks)
        self._arenas[arena_id] = arena

        logger.info("arena_registered", arena_id=arena_id)
        return arena

    def get_arena(self, arena_id: str) -> Optional[Arena]:
        """Get arena by ID.

        Args:
            arena_id: Arena identifier.

        Returns:
            Arena or None if not found.
        """
        return self._arenas.get(arena_id)

    def reset_arena(self, arena_id: str) -> int:
        """Reset specific arena.

        Args:
            arena_id: Arena to reset.

        Returns:
            Number of objects deallocated.

        Raises:
            KeyError: If arena not found.
        """
        if arena_id not in self._arenas:
            raise KeyError(f"Arena {arena_id} not found")

        return self._arenas[arena_id].reset()

    def reset_all(self) -> int:
        """Reset all arenas.

        Returns:
            Total number of objects deallocated.
        """
        total = 0
        for arena in self._arenas.values():
            total += arena.reset()

        logger.info("all_arenas_reset", total_objects=total)
        return total

    def get_all_stats(self) -> Dict[str, ArenaStats]:
        """Get statistics for all arenas.

        Returns:
            Dict mapping arena_id to ArenaStats.
        """
        return {
            arena_id: arena.get_stats()
            for arena_id, arena in self._arenas.items()
        }

    def remove_arena(self, arena_id: str) -> bool:
        """Remove and reset arena.

        Args:
            arena_id: Arena to remove.

        Returns:
            True if removed, False if not found.
        """
        if arena_id in self._arenas:
            self._arenas[arena_id].reset()
            del self._arenas[arena_id]
            logger.info("arena_removed", arena_id=arena_id)
            return True
        return False


# Global arena manager instance
_global_arena_manager: Optional[ArenaManager] = None


def get_arena_manager() -> ArenaManager:
    """Get global arena manager instance.

    Returns:
        The global ArenaManager.
    """
    global _global_arena_manager
    if _global_arena_manager is None:
        _global_arena_manager = ArenaManager()
    return _global_arena_manager
