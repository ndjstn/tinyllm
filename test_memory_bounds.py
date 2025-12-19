#!/usr/bin/env python
"""Test script for ExecutionContext memory bounds checking.

This script demonstrates and tests all the memory bounds features:
1. Configurable limits (max_messages, max_message_size_bytes, max_total_size_bytes)
2. BoundsExceededError raised when limits exceeded
3. Warning logs at 80% threshold
4. get_memory_usage() statistics
5. check_bounds() validation
6. prune_old_messages() cleanup

Run with: PYTHONPATH=src python test_memory_bounds.py
"""

from tinyllm.core.context import ExecutionContext, BoundsExceededError, MemoryUsageStats
from tinyllm.config.loader import Config
from tinyllm.core.message import Message, MessagePayload


def test_memory_usage_stats():
    """Test get_memory_usage() returns correct statistics."""
    print("\n=== Test 1: Memory Usage Statistics ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-stats',
        graph_id='test-graph',
        config=config,
        max_messages=10,
        max_message_size_bytes=10240,
        max_total_size_bytes=102400
    )

    # Add some messages
    for i in range(5):
        msg = Message(
            trace_id='test-stats',
            source_node='test-node',
            payload=MessagePayload(content=f'Test message {i}')
        )
        ctx.add_message(msg)

    stats = ctx.get_memory_usage()
    print(f"Message Count: {stats.message_count}/{stats.max_messages}")
    print(f"Total Size: {stats.total_size_bytes} bytes")
    print(f"Largest Message: {stats.largest_message_bytes} bytes")
    print(f"Message Utilization: {stats.message_count_utilization:.1%}")
    print(f"Size Utilization: {stats.total_size_utilization:.1%}")
    print(f"Near Message Limit: {stats.near_message_limit}")
    print(f"Near Size Limit: {stats.near_size_limit}")

    assert stats.message_count == 5
    assert stats.max_messages == 10
    assert stats.message_count_utilization == 0.5
    assert not stats.near_message_limit
    print("✓ Test 1 passed")


def test_bounds_checking():
    """Test check_bounds() validates current state."""
    print("\n=== Test 2: Bounds Checking ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-bounds',
        graph_id='test-graph',
        config=config,
        max_messages=5,
        max_message_size_bytes=10240,
        max_total_size_bytes=102400
    )

    # Add messages within bounds
    for i in range(3):
        msg = Message(
            trace_id='test-bounds',
            source_node='test-node',
            payload=MessagePayload(content=f'Message {i}')
        )
        ctx.add_message(msg)

    # Should pass
    ctx.check_bounds()
    print(f"Bounds check passed with {len(ctx.messages)} messages")
    print("✓ Test 2 passed")


def test_message_count_limit():
    """Test that message count limit is enforced."""
    print("\n=== Test 3: Message Count Limit ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-limit',
        graph_id='test-graph',
        config=config,
        max_messages=3,
        max_message_size_bytes=10240,
        max_total_size_bytes=102400
    )

    # Add up to limit
    for i in range(3):
        msg = Message(
            trace_id='test-limit',
            source_node='test-node',
            payload=MessagePayload(content=f'Message {i}')
        )
        ctx.add_message(msg)

    print(f"Successfully added {len(ctx.messages)} messages (at limit)")

    # Try to exceed limit
    try:
        msg = Message(
            trace_id='test-limit',
            source_node='test-node',
            payload=MessagePayload(content='This should fail')
        )
        ctx.add_message(msg)
        raise AssertionError("Should have raised BoundsExceededError")
    except BoundsExceededError as e:
        print(f"Caught expected error: {e.limit_type}")
        print(f"  Current: {e.current_value}, Limit: {e.limit_value}")
        assert e.limit_type == "message_count"
        assert e.current_value == 3
        assert e.limit_value == 3

    print("✓ Test 3 passed")


def test_message_size_limit():
    """Test that individual message size limit is enforced."""
    print("\n=== Test 4: Message Size Limit ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-size',
        graph_id='test-graph',
        config=config,
        max_messages=100,
        max_message_size_bytes=1024,  # 1 KB minimum
        max_total_size_bytes=102400
    )

    # Try to add a large message (larger than 1KB)
    try:
        large_content = "x" * 50000  # 50KB of content
        msg = Message(
            trace_id='test-size',
            source_node='test-node',
            payload=MessagePayload(content=large_content)
        )
        ctx.add_message(msg)
        raise AssertionError("Should have raised BoundsExceededError")
    except BoundsExceededError as e:
        print(f"Caught expected error: {e.limit_type}")
        print(f"  Message size: {e.current_value} bytes, Limit: {e.limit_value} bytes")
        assert e.limit_type == "message_size"
        assert e.current_value > e.limit_value

    print("✓ Test 4 passed")


def test_warning_threshold():
    """Test that warnings are logged at 80% threshold."""
    print("\n=== Test 5: Warning Threshold (80%) ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-warning',
        graph_id='test-graph',
        config=config,
        max_messages=10,
        max_message_size_bytes=10240,
        max_total_size_bytes=102400
    )

    # Add messages up to 70%
    for i in range(7):
        msg = Message(
            trace_id='test-warning',
            source_node='test-node',
            payload=MessagePayload(content=f'Message {i}')
        )
        ctx.add_message(msg)

    stats = ctx.get_memory_usage()
    print(f"At 70%: utilization={stats.message_count_utilization:.0%}, near_limit={stats.near_message_limit}")
    assert not stats.near_message_limit

    # Add one more to cross 80%
    msg = Message(
        trace_id='test-warning',
        source_node='test-node',
        payload=MessagePayload(content='Message 8')
    )
    ctx.add_message(msg)

    stats = ctx.get_memory_usage()
    print(f"At 80%: utilization={stats.message_count_utilization:.0%}, near_limit={stats.near_message_limit}")
    print("(Check logs above for warning message)")
    assert stats.near_message_limit

    print("✓ Test 5 passed")


def test_prune_messages():
    """Test pruning old messages."""
    print("\n=== Test 6: Prune Old Messages ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-prune',
        graph_id='test-graph',
        config=config,
        max_messages=10,
        max_message_size_bytes=10240,
        max_total_size_bytes=102400
    )

    # Add 8 messages
    for i in range(8):
        msg = Message(
            trace_id='test-prune',
            source_node='test-node',
            payload=MessagePayload(content=f'Message {i}')
        )
        ctx.add_message(msg)

    print(f"Before pruning: {len(ctx.messages)} messages")

    # Prune to keep only 3
    pruned_count = ctx.prune_old_messages(keep_count=3)

    print(f"After pruning: {len(ctx.messages)} messages")
    print(f"Pruned: {pruned_count} messages")

    assert len(ctx.messages) == 3
    assert pruned_count == 5

    # Verify we kept the most recent ones
    assert ctx.messages[0].payload.content == "Message 5"
    assert ctx.messages[1].payload.content == "Message 6"
    assert ctx.messages[2].payload.content == "Message 7"

    print("✓ Test 6 passed")


def test_default_prune():
    """Test default pruning (keeps half of max_messages)."""
    print("\n=== Test 7: Default Prune Strategy ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-default-prune',
        graph_id='test-graph',
        config=config,
        max_messages=10,
        max_message_size_bytes=10240,
        max_total_size_bytes=102400
    )

    # Add 9 messages
    for i in range(9):
        msg = Message(
            trace_id='test-default-prune',
            source_node='test-node',
            payload=MessagePayload(content=f'Message {i}')
        )
        ctx.add_message(msg)

    print(f"Before pruning: {len(ctx.messages)} messages")

    # Prune with default strategy (keep half of max_messages = 5)
    pruned_count = ctx.prune_old_messages()

    print(f"After pruning: {len(ctx.messages)} messages")
    print(f"Default strategy keeps: {ctx.max_messages // 2} messages")

    assert len(ctx.messages) == 5
    assert pruned_count == 4

    print("✓ Test 7 passed")


def test_configurable_limits():
    """Test that all three limits can be configured."""
    print("\n=== Test 8: Configurable Limits ===")

    config = Config()
    ctx = ExecutionContext(
        trace_id='test-config',
        graph_id='test-graph',
        config=config,
        max_messages=42,
        max_message_size_bytes=2_097_152,  # 2 MB
        max_total_size_bytes=52_428_800   # 50 MB
    )

    print(f"max_messages: {ctx.max_messages}")
    print(f"max_message_size_bytes: {ctx.max_message_size_bytes:,} bytes ({ctx.max_message_size_bytes / 1024 / 1024:.1f} MB)")
    print(f"max_total_size_bytes: {ctx.max_total_size_bytes:,} bytes ({ctx.max_total_size_bytes / 1024 / 1024:.1f} MB)")

    assert ctx.max_messages == 42
    assert ctx.max_message_size_bytes == 2_097_152
    assert ctx.max_total_size_bytes == 52_428_800

    print("✓ Test 8 passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing ExecutionContext Memory Bounds")
    print("=" * 60)

    test_memory_usage_stats()
    test_bounds_checking()
    test_message_count_limit()
    test_message_size_limit()
    test_warning_threshold()
    test_prune_messages()
    test_default_prune()
    test_configurable_limits()

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
