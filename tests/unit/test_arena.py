"""Tests for arena allocators."""

import pytest

from tinyllm.core.arena import Arena, ArenaManager, get_arena_manager
from tinyllm.core.message import Message, MessagePayload


def _create_message(content: str, trace_id: str = "test-trace") -> Message:
    """Helper to create test messages."""
    return Message(
        trace_id=trace_id,
        source_node="test-node",
        payload=MessagePayload(content=content),
    )


def test_arena_basic_allocation():
    """Test basic arena allocation."""
    arena = Arena[Message](arena_id="test", chunk_size=1024)

    # Allocate some messages
    msg1 = arena.allocate(_create_message("test1"))
    msg2 = arena.allocate(_create_message("test2"))

    assert len(arena) == 2
    assert msg1 in arena
    assert msg2 in arena


def test_arena_chunk_overflow():
    """Test that arena creates new chunks when needed."""
    # Small chunk size to force multiple chunks
    arena = Arena[Message](arena_id="test", chunk_size=100)

    # Allocate messages that will exceed chunk size
    messages = []
    for i in range(10):
        msg = arena.allocate(_create_message(f"test message {i}" * 10))
        messages.append(msg)

    stats = arena.get_stats()
    assert stats.chunk_count > 1  # Should have created multiple chunks
    assert len(arena) == 10


def test_arena_max_chunks():
    """Test that arena enforces max chunks limit."""
    arena = Arena[Message](arena_id="test", chunk_size=100, max_chunks=2)

    # Try to allocate enough to exceed max chunks
    with pytest.raises(MemoryError, match="exceeded max chunks"):
        for i in range(1000):
            arena.allocate(_create_message("x" * 100))


def test_arena_reset():
    """Test arena reset functionality."""
    arena = Arena[Message](arena_id="test")

    # Allocate some objects
    for i in range(10):
        arena.allocate(_create_message(f"test {i}"))

    assert len(arena) == 10

    # Reset
    count = arena.reset()
    assert count == 10
    assert len(arena) == 0

    # Should be able to allocate again
    msg = arena.allocate(_create_message("after reset"))
    assert len(arena) == 1


def test_arena_stats():
    """Test arena statistics."""
    arena = Arena[Message](arena_id="test", chunk_size=1024)

    # Initially empty
    stats = arena.get_stats()
    assert stats.object_count == 0
    assert stats.chunk_count == 1  # One chunk always allocated

    # Add some objects
    for i in range(5):
        arena.allocate(_create_message(f"test {i}"))

    stats = arena.get_stats()
    assert stats.object_count == 5
    assert stats.total_used > 0
    assert stats.utilization > 0.0


def test_arena_manager_create():
    """Test arena manager creation."""
    manager = ArenaManager()

    arena1 = manager.create_arena("arena1")
    arena2 = manager.create_arena("arena2", chunk_size=2048)

    assert arena1 is not None
    assert arena2 is not None
    assert arena1 != arena2

    # Can't create duplicate
    with pytest.raises(ValueError, match="already exists"):
        manager.create_arena("arena1")


def test_arena_manager_get():
    """Test arena manager get functionality."""
    manager = ArenaManager()

    arena = manager.create_arena("test")
    retrieved = manager.get_arena("test")

    assert retrieved is arena

    # Non-existent arena
    assert manager.get_arena("nonexistent") is None


def test_arena_manager_reset():
    """Test arena manager reset."""
    manager = ArenaManager()

    arena1 = manager.create_arena("arena1")
    arena2 = manager.create_arena("arena2")

    # Add objects
    arena1.allocate(_create_message("test1"))
    arena2.allocate(_create_message("test2"))

    # Reset specific arena
    count = manager.reset_arena("arena1")
    assert count == 1
    assert len(arena1) == 0
    assert len(arena2) == 1

    # Reset all
    arena1.allocate(_create_message("test3"))
    total = manager.reset_all()
    assert total == 2
    assert len(arena1) == 0
    assert len(arena2) == 0


def test_arena_manager_remove():
    """Test arena manager remove."""
    manager = ArenaManager()

    manager.create_arena("test")
    assert manager.get_arena("test") is not None

    # Remove
    assert manager.remove_arena("test") is True
    assert manager.get_arena("test") is None

    # Remove non-existent
    assert manager.remove_arena("nonexistent") is False


def test_arena_manager_stats():
    """Test arena manager statistics."""
    manager = ArenaManager()

    arena1 = manager.create_arena("arena1")
    arena2 = manager.create_arena("arena2")

    arena1.allocate(_create_message("test1"))
    arena2.allocate(_create_message("test2"))

    stats = manager.get_all_stats()
    assert "arena1" in stats
    assert "arena2" in stats
    assert stats["arena1"].object_count == 1
    assert stats["arena2"].object_count == 1


def test_global_arena_manager():
    """Test global arena manager singleton."""
    manager1 = get_arena_manager()
    manager2 = get_arena_manager()

    assert manager1 is manager2  # Same instance


def test_arena_with_different_types():
    """Test arena with different object types."""
    # Test with simple strings
    arena = Arena[str](arena_id="strings")

    arena.allocate("test1")
    arena.allocate("test2")

    assert len(arena) == 2


def test_arena_utilization():
    """Test arena utilization calculation."""
    arena = Arena[Message](arena_id="test", chunk_size=10000)

    # Add messages
    for i in range(5):
        arena.allocate(_create_message("x" * 100))

    stats = arena.get_stats()
    assert 0.0 < stats.utilization <= 1.0
    assert 0.0 <= stats.fragmentation <= 1.0


def test_arena_contains():
    """Test arena __contains__ method."""
    arena = Arena[Message](arena_id="test")

    msg1 = arena.allocate(_create_message("test1"))
    msg2 = _create_message("test2")  # Not allocated

    assert msg1 in arena
    assert msg2 not in arena


def test_arena_large_objects():
    """Test arena with large objects."""
    arena = Arena[Message](arena_id="test", chunk_size=1024)

    # Create large message
    large_content = "x" * 10000
    msg = arena.allocate(_create_message(large_content))

    assert msg in arena
    stats = arena.get_stats()
    assert stats.object_count == 1
