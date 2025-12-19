"""Tests for copy-on-write contexts."""

import pytest

from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.cow import CoWContext, CoWContextManager, get_cow_manager
from tinyllm.core.message import Message, MessagePayload


def _create_context() -> ExecutionContext:
    """Helper to create test execution context."""
    return ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=Config(),
    )


def _create_message(content: str) -> Message:
    """Helper to create test messages."""
    return Message(
        trace_id="test-trace",
        source_node="test-node",
        payload=MessagePayload(content=content),
    )


def test_cow_context_basic():
    """Test basic CoW context creation."""
    ctx = _create_context()
    cow = CoWContext(ctx)

    assert cow.context is ctx
    assert cow.context_id.startswith("cow-")


def test_cow_context_clone():
    """Test CoW context cloning."""
    ctx = _create_context()
    ctx.add_message(_create_message("test"))
    ctx.set_variable("key", "value")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    assert cloned.context_id != cow.context_id
    assert cloned._parent is cow


def test_cow_shared_messages():
    """Test that messages are shared until modified."""
    ctx = _create_context()
    ctx.add_message(_create_message("original"))

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Initially shared
    stats = cloned.get_stats()
    assert stats.is_copy is True
    assert stats.shared_count > 0
    assert stats.copied_count == 0

    # Messages should be shared
    assert len(cloned.get_messages()) == 1
    assert cloned._shared_messages is not None


def test_cow_messages_copy_on_write():
    """Test copy-on-write for messages."""
    ctx = _create_context()
    ctx.add_message(_create_message("original"))

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Add message to clone - should trigger copy
    cloned.add_message(_create_message("cloned"))

    # Should have copied messages
    assert cloned._messages_copied is True
    assert cloned._shared_messages is None

    # Original should be unchanged
    assert len(cow.context.messages) == 1
    assert len(cloned.context.messages) == 2


def test_cow_shared_variables():
    """Test that variables are shared until modified."""
    ctx = _create_context()
    ctx.set_variable("key1", "value1")
    ctx.set_variable("key2", "value2")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Initially shared
    assert cloned._shared_variables is not None
    assert cloned.get_variable("key1") == "value1"


def test_cow_variables_copy_on_write():
    """Test copy-on-write for variables."""
    ctx = _create_context()
    ctx.set_variable("key1", "value1")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Set variable in clone - should trigger copy
    cloned.set_variable("key2", "value2")

    # Should have copied variables
    assert cloned._variables_copied is True
    assert cloned._shared_variables is None

    # Original should be unchanged
    assert cow.context.variables == {"key1": "value1"}
    assert cloned.context.variables == {"key1": "value1", "key2": "value2"}


def test_cow_shared_visited_nodes():
    """Test that visited nodes are shared until modified."""
    ctx = _create_context()
    ctx.visit_node("node1")
    ctx.visit_node("node2")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Initially shared
    assert cloned._shared_visited is not None
    assert len(cloned._shared_visited) == 2


def test_cow_visited_nodes_copy_on_write():
    """Test copy-on-write for visited nodes."""
    ctx = _create_context()
    ctx.visit_node("node1")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Visit node in clone - should trigger copy
    cloned.visit_node("node2")

    # Should have copied visited nodes
    assert cloned._visited_nodes_copied is True
    assert cloned._shared_visited is None

    # Original should be unchanged
    assert cow.context.visited_nodes == ["node1"]
    assert cloned.context.visited_nodes == ["node1", "node2"]


def test_cow_stats():
    """Test CoW statistics."""
    ctx = _create_context()
    ctx.add_message(_create_message("test"))
    ctx.set_variable("key", "value")
    ctx.visit_node("node1")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    stats = cloned.get_stats()
    assert stats.is_copy is True
    assert stats.shared_count == 3  # messages, variables, visited
    assert stats.copied_count == 0
    assert stats.memory_saved_bytes > 0
    assert stats.copy_depth == 1


def test_cow_multiple_clones():
    """Test multiple levels of cloning."""
    ctx = _create_context()
    ctx.add_message(_create_message("original"))

    cow1 = CoWContext(ctx)
    cow2 = cow1.clone()
    cow3 = cow2.clone()

    # Check depth
    assert cow1.get_stats().copy_depth == 0
    assert cow2.get_stats().copy_depth == 1
    assert cow3.get_stats().copy_depth == 2


def test_cow_memory_savings():
    """Test that CoW actually saves memory."""
    ctx = _create_context()
    # Add multiple messages
    for i in range(100):
        ctx.add_message(_create_message(f"message {i}"))

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Should show significant memory savings
    stats = cloned.get_stats()
    assert stats.memory_saved_bytes > 50000  # ~50KB saved


def test_cow_read_only_access():
    """Test that read-only access doesn't trigger copy."""
    ctx = _create_context()
    ctx.add_message(_create_message("test"))
    ctx.set_variable("key", "value")

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Read operations shouldn't trigger copy
    _ = cloned.get_variable("key")
    _ = cloned.has_variable("key")
    _ = cloned.get_messages()

    # Should still be sharing
    assert cloned._shared_messages is not None
    assert cloned._shared_variables is not None


def test_cow_manager_create():
    """Test CoW context manager creation."""
    manager = CoWContextManager()
    ctx = _create_context()

    cow = manager.create(ctx)
    assert cow is not None
    assert manager.get(cow.context_id) is cow


def test_cow_manager_get():
    """Test CoW context manager get."""
    manager = CoWContextManager()
    ctx = _create_context()

    cow = manager.create(ctx, context_id="test-cow")
    retrieved = manager.get("test-cow")

    assert retrieved is cow

    # Non-existent context
    assert manager.get("nonexistent") is None


def test_cow_manager_remove():
    """Test CoW context manager remove."""
    manager = CoWContextManager()
    ctx = _create_context()

    cow = manager.create(ctx, context_id="test-cow")
    assert manager.get("test-cow") is not None

    # Remove
    assert manager.remove("test-cow") is True
    assert manager.get("test-cow") is None

    # Remove non-existent
    assert manager.remove("nonexistent") is False


def test_cow_manager_stats():
    """Test CoW context manager statistics."""
    manager = CoWContextManager()

    ctx1 = _create_context()
    ctx2 = _create_context()

    cow1 = manager.create(ctx1, context_id="cow1")
    cow2 = manager.create(ctx2, context_id="cow2")

    stats = manager.get_all_stats()
    assert "cow1" in stats
    assert "cow2" in stats


def test_global_cow_manager():
    """Test global CoW manager singleton."""
    manager1 = get_cow_manager()
    manager2 = get_cow_manager()

    assert manager1 is manager2  # Same instance


def test_cow_deep_variable_copy():
    """Test that nested variables are deep copied."""
    ctx = _create_context()
    ctx.set_variable("nested", {"key": "value", "list": [1, 2, 3]})

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Modify nested structure in clone
    cloned.set_variable("nested2", {"new": "value"})

    # Get the nested variable from clone
    cloned._ensure_variables_copied()
    cloned.context.variables["nested"]["key"] = "modified"

    # Original should be unchanged
    assert cow.context.variables["nested"]["key"] == "value"


def test_cow_message_isolation():
    """Test that cloned messages are isolated."""
    ctx = _create_context()
    msg = _create_message("original")
    ctx.add_message(msg)

    cow = CoWContext(ctx)
    cloned = cow.clone()

    # Add message to clone
    cloned.add_message(_create_message("cloned"))

    # Messages should be isolated
    assert len(cow.context.messages) == 1
    assert len(cloned.context.messages) == 2
    assert cow.context.messages[0].payload.content == "original"
    assert cloned.context.messages[0].payload.content == "original"
    assert cloned.context.messages[1].payload.content == "cloned"


def test_cow_repr():
    """Test CoW context string representation."""
    ctx = _create_context()
    cow = CoWContext(ctx, context_id="test-cow")

    repr_str = repr(cow)
    assert "test-cow" in repr_str
    assert "test-trace" in repr_str
