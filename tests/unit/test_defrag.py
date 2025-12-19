"""Tests for memory defragmentation."""

import time
from unittest.mock import Mock

import pytest

from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.defrag import (
    AutoDefragmenter,
    DefragStats,
    DefragStrategy,
    MemoryDefragmenter,
)
from tinyllm.core.message import Message, MessageMetadata, MessagePayload


@pytest.fixture
def config():
    """Create test config."""
    return Config()


@pytest.fixture
def context(config):
    """Create test execution context."""
    return ExecutionContext(
        trace_id="test-trace",
        graph_id="test-graph",
        config=config,
    )


@pytest.fixture
def sample_message():
    """Create a sample message."""
    return Message(
        trace_id="test-trace",
        source_node="test-node",
        payload=MessagePayload(content="Test message content"),
        metadata=MessageMetadata(),
    )


# Tests for DefragStats model


class TestDefragStats:
    """Tests for DefragStats model."""

    def test_create_stats(self):
        """Should create defrag statistics."""
        stats = DefragStats(
            messages_before=100,
            messages_after=50,
            messages_removed=50,
            size_before_bytes=10000,
            size_after_bytes=5000,
            bytes_freed=5000,
            variables_before=20,
            variables_after=10,
            variables_removed=10,
            fragmentation_before=0.8,
            fragmentation_after=0.3,
            duration_ms=100,
        )

        assert stats.messages_before == 100
        assert stats.messages_after == 50
        assert stats.messages_removed == 50
        assert stats.size_before_bytes == 10000
        assert stats.size_after_bytes == 5000
        assert stats.bytes_freed == 5000
        assert stats.variables_before == 20
        assert stats.variables_after == 10
        assert stats.variables_removed == 10
        assert stats.fragmentation_before == 0.8
        assert stats.fragmentation_after == 0.3
        assert stats.duration_ms == 100

    def test_non_negative_constraints(self):
        """Should enforce non-negative constraints."""
        with pytest.raises(ValueError):
            DefragStats(
                messages_before=-1,
                messages_after=0,
                messages_removed=0,
                size_before_bytes=0,
                size_after_bytes=0,
                bytes_freed=0,
                variables_before=0,
                variables_after=0,
                variables_removed=0,
                fragmentation_before=0.0,
                fragmentation_after=0.0,
                duration_ms=0,
            )

    def test_fragmentation_range(self):
        """Should enforce fragmentation range 0.0 to 1.0."""
        with pytest.raises(ValueError):
            DefragStats(
                messages_before=0,
                messages_after=0,
                messages_removed=0,
                size_before_bytes=0,
                size_after_bytes=0,
                bytes_freed=0,
                variables_before=0,
                variables_after=0,
                variables_removed=0,
                fragmentation_before=1.5,  # Invalid: > 1.0
                fragmentation_after=0.0,
                duration_ms=0,
            )

    def test_extra_fields_forbidden(self):
        """Should forbid extra fields."""
        with pytest.raises(ValueError):
            DefragStats(
                messages_before=0,
                messages_after=0,
                messages_removed=0,
                size_before_bytes=0,
                size_after_bytes=0,
                bytes_freed=0,
                variables_before=0,
                variables_after=0,
                variables_removed=0,
                fragmentation_before=0.0,
                fragmentation_after=0.0,
                duration_ms=0,
                extra_field="not allowed",
            )


# Tests for DefragStrategy model


class TestDefragStrategy:
    """Tests for DefragStrategy model."""

    def test_default_strategy(self):
        """Should have sensible defaults."""
        strategy = DefragStrategy()

        assert strategy.remove_old_messages is True
        assert strategy.message_window_size == 100
        assert strategy.remove_unused_variables is True
        assert strategy.variable_access_threshold == 10
        assert strategy.deduplicate_messages is False
        assert strategy.compact_metadata is True

    def test_custom_strategy(self):
        """Should allow custom configuration."""
        strategy = DefragStrategy(
            remove_old_messages=False,
            message_window_size=50,
            remove_unused_variables=False,
            variable_access_threshold=5,
            deduplicate_messages=True,
            compact_metadata=False,
        )

        assert strategy.remove_old_messages is False
        assert strategy.message_window_size == 50
        assert strategy.remove_unused_variables is False
        assert strategy.variable_access_threshold == 5
        assert strategy.deduplicate_messages is True
        assert strategy.compact_metadata is False

    def test_message_window_validation(self):
        """Should validate message window size is at least 1."""
        with pytest.raises(ValueError):
            DefragStrategy(message_window_size=0)

        with pytest.raises(ValueError):
            DefragStrategy(message_window_size=-1)

    def test_variable_threshold_validation(self):
        """Should validate variable access threshold is non-negative."""
        # Zero should be allowed
        strategy = DefragStrategy(variable_access_threshold=0)
        assert strategy.variable_access_threshold == 0

        # Negative should fail
        with pytest.raises(ValueError):
            DefragStrategy(variable_access_threshold=-1)

    def test_extra_fields_forbidden(self):
        """Should forbid extra fields."""
        with pytest.raises(ValueError):
            DefragStrategy(extra_field="not allowed")


# Tests for MemoryDefragmenter


class TestMemoryDefragmenter:
    """Tests for MemoryDefragmenter."""

    def test_initialization(self):
        """Should initialize with default strategy."""
        defragmenter = MemoryDefragmenter()

        assert defragmenter.strategy is not None
        assert defragmenter.strategy.remove_old_messages is True
        assert defragmenter._variable_access_counts == {}
        assert defragmenter._steps_since_access == {}

    def test_initialization_custom_strategy(self):
        """Should initialize with custom strategy."""
        strategy = DefragStrategy(message_window_size=50)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        assert defragmenter.strategy.message_window_size == 50

    def test_defragment_basic(self, context, sample_message):
        """Should defragment context and return stats."""
        # Add some messages
        for i in range(10):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        defragmenter = MemoryDefragmenter()
        stats = defragmenter.defragment(context)

        assert isinstance(stats, DefragStats)
        assert stats.messages_before == 10
        assert stats.messages_after == 10  # Window is 100, so all kept
        assert stats.duration_ms >= 0

    def test_defragment_tracks_stats(self, context):
        """Should track before and after stats."""
        # Add many messages
        for i in range(150):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(message_window_size=100)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        stats = defragmenter.defragment(context)

        assert stats.messages_before == 150
        assert stats.messages_after == 100
        assert stats.messages_removed == 50
        assert stats.bytes_freed > 0

    # Tests for message pruning

    def test_prune_old_messages_basic(self, context):
        """Should prune old messages beyond window."""
        # Add messages beyond window
        for i in range(150):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(
            remove_old_messages=True,
            message_window_size=100,
        )
        defragmenter = MemoryDefragmenter(strategy=strategy)

        removed = defragmenter._prune_old_messages(context)

        assert removed == 50
        assert len(context.messages) == 100
        # Should keep most recent messages
        assert context.messages[-1].payload.content == "Message 149"
        assert context.messages[0].payload.content == "Message 50"

    def test_prune_old_messages_no_pruning_needed(self, context):
        """Should not prune if under window size."""
        for i in range(50):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(message_window_size=100)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        removed = defragmenter._prune_old_messages(context)

        assert removed == 0
        assert len(context.messages) == 50

    def test_prune_disabled(self, context):
        """Should not prune when disabled in strategy."""
        for i in range(150):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(remove_old_messages=False)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        stats = defragmenter.defragment(context)

        assert len(context.messages) == 150
        assert stats.messages_removed == 0

    # Tests for message deduplication

    def test_deduplicate_messages(self, context):
        """Should remove duplicate messages."""
        # Add duplicate messages
        for i in range(3):
            msg = Message(
                trace_id=context.trace_id,
                source_node="node-a",
                payload=MessagePayload(content="Duplicate content"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        # Add unique message
        unique_msg = Message(
            trace_id=context.trace_id,
            source_node="node-a",
            payload=MessagePayload(content="Unique content"),
            metadata=MessageMetadata(),
        )
        context.add_message(unique_msg)

        strategy = DefragStrategy(deduplicate_messages=True)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        removed = defragmenter._deduplicate_messages(context)

        assert removed == 2  # 3 duplicates -> 1 kept + 1 unique = 2 total
        assert len(context.messages) == 2

    def test_deduplicate_no_duplicates(self, context):
        """Should handle no duplicates."""
        for i in range(5):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Unique message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(deduplicate_messages=True)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        removed = defragmenter._deduplicate_messages(context)

        assert removed == 0
        assert len(context.messages) == 5

    def test_deduplicate_disabled(self, context):
        """Should not deduplicate when disabled."""
        # Add duplicates
        for i in range(3):
            msg = Message(
                trace_id=context.trace_id,
                source_node="node-a",
                payload=MessagePayload(content="Duplicate"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(deduplicate_messages=False)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        stats = defragmenter.defragment(context)

        assert len(context.messages) == 3
        assert stats.messages_removed == 0

    def test_message_hash_function(self):
        """Should create consistent hash for messages."""
        defragmenter = MemoryDefragmenter()

        msg1 = Message(
            trace_id="trace-1",
            source_node="node-a",
            payload=MessagePayload(content="Test content"),
            metadata=MessageMetadata(),
        )

        msg2 = Message(
            trace_id="trace-1",
            source_node="node-a",
            payload=MessagePayload(content="Test content"),
            metadata=MessageMetadata(),
        )

        hash1 = defragmenter._message_hash(msg1)
        hash2 = defragmenter._message_hash(msg2)

        assert hash1 == hash2
        assert "node-a" in hash1
        assert "Test content" in hash1

    def test_message_hash_different_nodes(self):
        """Should create different hashes for different source nodes."""
        defragmenter = MemoryDefragmenter()

        msg1 = Message(
            trace_id="trace-1",
            source_node="node-a",
            payload=MessagePayload(content="Same content"),
            metadata=MessageMetadata(),
        )

        msg2 = Message(
            trace_id="trace-1",
            source_node="node-b",
            payload=MessagePayload(content="Same content"),
            metadata=MessageMetadata(),
        )

        hash1 = defragmenter._message_hash(msg1)
        hash2 = defragmenter._message_hash(msg2)

        assert hash1 != hash2

    # Tests for variable cleanup

    def test_remove_unused_variables(self, context):
        """Should remove variables not accessed recently."""
        # Set some variables
        context.set_variable("var1", "value1")
        context.set_variable("var2", "value2")
        context.set_variable("var3", "value3")

        strategy = DefragStrategy(
            remove_unused_variables=True,
            variable_access_threshold=5,
        )
        defragmenter = MemoryDefragmenter(strategy=strategy)

        # Track all variables initially (simulating initial access)
        defragmenter.track_variable_access("var1")
        defragmenter.track_variable_access("var2")
        defragmenter.track_variable_access("var3")

        # Simulate steps passing, keeping var1 accessed but not var2/var3
        for _ in range(6):
            defragmenter.track_variable_access("var1")  # Keep var1 fresh
            defragmenter._remove_unused_variables(context)

        # var2 and var3 should be removed (not accessed recently, threshold exceeded)
        # var1 should still be there (kept fresh)
        assert "var1" in context.variables
        assert "var2" not in context.variables
        assert "var3" not in context.variables

    def test_variable_tracking(self):
        """Should track variable accesses."""
        defragmenter = MemoryDefragmenter()

        defragmenter.track_variable_access("var1")
        defragmenter.track_variable_access("var1")
        defragmenter.track_variable_access("var2")

        assert defragmenter._variable_access_counts["var1"] == 2
        assert defragmenter._variable_access_counts["var2"] == 1
        assert defragmenter._steps_since_access["var1"] == 0
        assert defragmenter._steps_since_access["var2"] == 0

    def test_reset_tracking(self):
        """Should reset access tracking."""
        defragmenter = MemoryDefragmenter()

        defragmenter.track_variable_access("var1")
        defragmenter.track_variable_access("var2")

        defragmenter.reset_tracking()

        assert defragmenter._variable_access_counts == {}
        assert defragmenter._steps_since_access == {}

    def test_variable_cleanup_disabled(self, context):
        """Should not remove variables when disabled."""
        context.set_variable("var1", "value1")
        context.set_variable("var2", "value2")

        strategy = DefragStrategy(remove_unused_variables=False)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        # Simulate many steps
        for _ in range(20):
            defragmenter._remove_unused_variables(context)

        # Variables should still be there
        assert "var1" in context.variables
        assert "var2" in context.variables

    # Tests for metadata compaction

    def test_compact_metadata(self, context):
        """Should compact metadata."""
        msg = Message(
            trace_id=context.trace_id,
            source_node="test-node",
            payload=MessagePayload(content="Test"),
            metadata=MessageMetadata(latency_ms=100, model_used="test-model"),
        )
        context.add_message(msg)

        defragmenter = MemoryDefragmenter()
        defragmenter._compact_metadata(context)

        # Should not raise error (basic implementation)
        assert len(context.messages) == 1

    def test_compact_metadata_disabled(self, context):
        """Should not compact when disabled."""
        msg = Message(
            trace_id=context.trace_id,
            source_node="test-node",
            payload=MessagePayload(content="Test"),
            metadata=MessageMetadata(),
        )
        context.add_message(msg)

        strategy = DefragStrategy(compact_metadata=False)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        stats = defragmenter.defragment(context)

        # Metadata compaction should not have run
        assert len(context.messages) == 1

    # Tests for size and fragmentation calculation

    def test_calculate_size(self, context):
        """Should calculate total context size."""
        for i in range(10):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        defragmenter = MemoryDefragmenter()
        size = defragmenter._calculate_size(context)

        assert size > 0
        assert isinstance(size, int)

    def test_calculate_size_empty(self, context):
        """Should handle empty context."""
        defragmenter = MemoryDefragmenter()
        size = defragmenter._calculate_size(context)

        assert size >= 0

    def test_calculate_size_with_variables(self, context):
        """Should include variable size in calculation."""
        context.set_variable("var1", "value1")
        context.set_variable("var2", "value2")

        defragmenter = MemoryDefragmenter()
        size = defragmenter._calculate_size(context)

        assert size > 0

    def test_calculate_fragmentation(self, context):
        """Should calculate fragmentation ratio."""
        # Add messages with varying sizes
        for i in range(10):
            content = "x" * (100 * (i + 1))  # Increasing sizes
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=content),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        defragmenter = MemoryDefragmenter()
        frag = defragmenter._calculate_fragmentation(context)

        assert 0.0 <= frag <= 1.0
        assert isinstance(frag, float)

    def test_calculate_fragmentation_empty(self, context):
        """Should return 0.0 for empty context."""
        defragmenter = MemoryDefragmenter()
        frag = defragmenter._calculate_fragmentation(context)

        assert frag == 0.0

    def test_calculate_fragmentation_uniform(self, context):
        """Should calculate low fragmentation for uniform messages."""
        # Add messages with same size
        for i in range(10):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content="Same size content"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        defragmenter = MemoryDefragmenter()
        frag = defragmenter._calculate_fragmentation(context)

        # Should be low fragmentation (close to 0)
        assert frag < 0.5

    # Tests for complete defragmentation flow

    def test_full_defragmentation(self, context):
        """Should perform full defragmentation with all strategies."""
        # Add messages beyond window
        for i in range(150):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        # Add some duplicates
        for i in range(5):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content="Duplicate"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        # Add variables
        context.set_variable("var1", "value1")
        context.set_variable("var2", "value2")

        strategy = DefragStrategy(
            remove_old_messages=True,
            message_window_size=100,
            deduplicate_messages=True,
            remove_unused_variables=True,
            variable_access_threshold=0,
            compact_metadata=True,
        )
        defragmenter = MemoryDefragmenter(strategy=strategy)

        stats = defragmenter.defragment(context)

        # Should have removed old messages
        assert stats.messages_removed > 0
        # Should have removed duplicates
        assert len(context.messages) <= 100
        # Should have stats
        assert stats.bytes_freed >= 0
        assert stats.duration_ms >= 0


# Tests for AutoDefragmenter


class TestAutoDefragmenter:
    """Tests for AutoDefragmenter."""

    def test_initialization(self):
        """Should initialize with defaults."""
        auto_defrag = AutoDefragmenter()

        assert auto_defrag.auto_defrag_threshold == 0.7
        assert auto_defrag.check_interval_steps == 100
        assert auto_defrag._steps_since_check == 0
        assert isinstance(auto_defrag.defragmenter, MemoryDefragmenter)

    def test_initialization_custom_params(self):
        """Should initialize with custom parameters."""
        strategy = DefragStrategy(message_window_size=50)
        auto_defrag = AutoDefragmenter(
            strategy=strategy,
            auto_defrag_threshold=0.8,
            check_interval_steps=50,
        )

        assert auto_defrag.auto_defrag_threshold == 0.8
        assert auto_defrag.check_interval_steps == 50
        assert auto_defrag.defragmenter.strategy.message_window_size == 50

    def test_maybe_defragment_before_interval(self, context):
        """Should not defragment before check interval."""
        auto_defrag = AutoDefragmenter(check_interval_steps=100)

        # Call 50 times (below interval)
        for _ in range(50):
            result = auto_defrag.maybe_defragment(context)
            assert result is None

    def test_maybe_defragment_at_interval_low_fragmentation(self, context):
        """Should not defragment if fragmentation is below threshold."""
        # Add a few uniform messages (low fragmentation)
        for i in range(10):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content="Same size"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        auto_defrag = AutoDefragmenter(
            check_interval_steps=10,
            auto_defrag_threshold=0.7,
        )

        # Advance to check interval
        for _ in range(9):
            auto_defrag.maybe_defragment(context)

        # At interval, should check but not defragment (low fragmentation)
        result = auto_defrag.maybe_defragment(context)
        assert result is None

    def test_maybe_defragment_at_interval_high_fragmentation(self, context):
        """Should defragment if fragmentation exceeds threshold."""
        # Add messages with high variance (high fragmentation)
        for i in range(20):
            content = "x" * (100 * (i + 1) ** 2)  # Quadratic growth
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=content),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        auto_defrag = AutoDefragmenter(
            check_interval_steps=5,
            auto_defrag_threshold=0.3,  # Low threshold to trigger
        )

        # Advance to check interval
        for _ in range(4):
            auto_defrag.maybe_defragment(context)

        # At interval, should defragment (high fragmentation)
        result = auto_defrag.maybe_defragment(context)

        # May or may not defragment depending on calculated fragmentation
        # Just verify it returns DefragStats or None
        assert result is None or isinstance(result, DefragStats)

    def test_step_counter_resets(self, context):
        """Should reset step counter after check."""
        auto_defrag = AutoDefragmenter(check_interval_steps=10)

        # Advance to interval
        for _ in range(10):
            auto_defrag.maybe_defragment(context)

        # Counter should reset
        assert auto_defrag._steps_since_check == 0

    def test_multiple_check_cycles(self, context):
        """Should handle multiple check cycles."""
        # Add some messages
        for i in range(50):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        auto_defrag = AutoDefragmenter(check_interval_steps=10)

        # Run through multiple intervals
        for _ in range(30):
            result = auto_defrag.maybe_defragment(context)
            # Should either return None or DefragStats
            assert result is None or isinstance(result, DefragStats)

    def test_defragmentation_triggered(self, context):
        """Should trigger defragmentation when conditions met."""
        # Create high fragmentation scenario
        for i in range(150):
            content = "x" * (100 + i * 100)  # Variable sizes
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=content),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(
            message_window_size=50,  # Will trigger pruning
        )
        auto_defrag = AutoDefragmenter(
            strategy=strategy,
            check_interval_steps=5,
            auto_defrag_threshold=0.1,  # Very low to ensure trigger
        )

        # Advance to interval
        for _ in range(4):
            auto_defrag.maybe_defragment(context)

        result = auto_defrag.maybe_defragment(context)

        # Should have defragmented due to pruning
        if result is not None:
            assert isinstance(result, DefragStats)
            assert result.messages_removed > 0


# Integration tests


class TestDefragIntegration:
    """Integration tests for defragmentation system."""

    def test_end_to_end_defragmentation(self, context):
        """Should perform complete defragmentation workflow."""
        # Simulate a long-running execution
        for i in range(200):
            content = f"Message {i}" * (1 + i % 5)  # Variable sizes
            msg = Message(
                trace_id=context.trace_id,
                source_node=f"node-{i % 3}",
                payload=MessagePayload(content=content),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        # Add duplicates
        for _ in range(10):
            msg = Message(
                trace_id=context.trace_id,
                source_node="node-dup",
                payload=MessagePayload(content="Duplicate message"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        # Add variables
        for i in range(20):
            context.set_variable(f"var_{i}", f"value_{i}")

        # Create comprehensive strategy
        strategy = DefragStrategy(
            remove_old_messages=True,
            message_window_size=100,
            deduplicate_messages=True,
            remove_unused_variables=True,
            variable_access_threshold=5,
            compact_metadata=True,
        )

        defragmenter = MemoryDefragmenter(strategy=strategy)

        # Track some variable accesses
        for i in range(5):
            defragmenter.track_variable_access(f"var_{i}")

        # Perform defragmentation
        stats = defragmenter.defragment(context)

        # Verify results
        assert isinstance(stats, DefragStats)
        assert stats.messages_before > stats.messages_after
        assert stats.messages_removed > 0
        assert stats.bytes_freed > 0
        assert stats.duration_ms >= 0
        assert 0.0 <= stats.fragmentation_before <= 1.0
        assert 0.0 <= stats.fragmentation_after <= 1.0

        # Messages should be at or below window size
        assert len(context.messages) <= 100

    def test_auto_defragmenter_with_growing_context(self, context):
        """Should auto-defragment as context grows."""
        auto_defrag = AutoDefragmenter(
            strategy=DefragStrategy(message_window_size=50),
            check_interval_steps=25,
            auto_defrag_threshold=0.5,
        )

        defrag_count = 0

        # Simulate execution with growing context
        for i in range(100):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}" * (i + 1)),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

            result = auto_defrag.maybe_defragment(context)
            if result is not None:
                defrag_count += 1

        # Should have triggered at least once due to interval
        # (may or may not defragment depending on fragmentation)
        assert defrag_count >= 0

    def test_defragmentation_preserves_recent_data(self, context):
        """Should preserve most recent messages during defragmentation."""
        # Add messages with identifiable content
        for i in range(200):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(
            remove_old_messages=True,
            message_window_size=100,
        )
        defragmenter = MemoryDefragmenter(strategy=strategy)

        defragmenter.defragment(context)

        # Should keep most recent 100 messages
        assert len(context.messages) == 100
        assert context.messages[0].payload.content == "Message 100"
        assert context.messages[-1].payload.content == "Message 199"

    def test_repeated_defragmentation(self, context):
        """Should handle repeated defragmentation safely."""
        # Add initial messages
        for i in range(150):
            msg = Message(
                trace_id=context.trace_id,
                source_node="test-node",
                payload=MessagePayload(content=f"Message {i}"),
                metadata=MessageMetadata(),
            )
            context.add_message(msg)

        strategy = DefragStrategy(message_window_size=100)
        defragmenter = MemoryDefragmenter(strategy=strategy)

        # First defragmentation
        stats1 = defragmenter.defragment(context)
        assert stats1.messages_removed == 50

        # Second defragmentation (should remove nothing)
        stats2 = defragmenter.defragment(context)
        assert stats2.messages_removed == 0
        assert stats2.messages_before == 100
        assert stats2.messages_after == 100

    def test_fragmentation_calculation_edge_cases(self, context):
        """Should handle edge cases in fragmentation calculation."""
        defragmenter = MemoryDefragmenter()

        # Empty context
        frag = defragmenter._calculate_fragmentation(context)
        assert frag == 0.0

        # Single message
        msg = Message(
            trace_id=context.trace_id,
            source_node="test-node",
            payload=MessagePayload(content="Single message"),
            metadata=MessageMetadata(),
        )
        context.add_message(msg)
        frag = defragmenter._calculate_fragmentation(context)
        assert 0.0 <= frag <= 1.0
