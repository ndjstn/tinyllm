"""Tests for checkpoint functionality."""

import asyncio
import pytest

from tinyllm.core.checkpoint import CheckpointConfig, CheckpointManager
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.config.loader import Config
from tinyllm.persistence.interface import StorageConfig
from tinyllm.persistence.memory_backend import InMemoryCheckpointStorage
from tinyllm.persistence.sqlite_backend import SQLiteCheckpointStorage


@pytest.fixture
async def memory_checkpoint_manager():
    """Create a checkpoint manager with in-memory storage."""
    config = StorageConfig()
    storage = InMemoryCheckpointStorage(config)
    await storage.initialize()

    manager = CheckpointManager(storage)
    await manager.initialize()

    yield manager

    await manager.close()


@pytest.fixture
async def sqlite_checkpoint_manager(tmp_path):
    """Create a checkpoint manager with SQLite storage."""
    db_path = tmp_path / "checkpoints.db"
    config = StorageConfig(sqlite_path=str(db_path))
    storage = SQLiteCheckpointStorage(config)
    await storage.initialize()

    manager = CheckpointManager(storage)
    await manager.initialize()

    yield manager

    await manager.close()


@pytest.fixture
def execution_context():
    """Create a sample execution context."""
    context = ExecutionContext(
        trace_id="test-trace-123",
        graph_id="test-graph",
        config=Config(),
    )

    # Add some state
    context.set_variable("test_var", "test_value")
    context.visit_node("node1")

    # Add a message
    msg = Message(
        trace_id="test-trace-123",
        source_node="node1",
        target_node="node2",
        payload=MessagePayload(content="test message"),
    )
    context.add_message(msg)

    return context


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_creation_memory(self, memory_checkpoint_manager, execution_context):
        """Test creating a checkpoint with in-memory storage."""
        checkpoint = await memory_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node1",
            force=True,
        )

        assert checkpoint is not None
        assert checkpoint.graph_id == "test-graph"
        assert checkpoint.trace_id == "test-trace-123"
        assert checkpoint.node_id == "node1"
        assert checkpoint.step == execution_context.step_count

    @pytest.mark.asyncio
    async def test_checkpoint_creation_sqlite(self, sqlite_checkpoint_manager, execution_context):
        """Test creating a checkpoint with SQLite storage."""
        checkpoint = await sqlite_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node1",
            force=True,
        )

        assert checkpoint is not None
        assert checkpoint.graph_id == "test-graph"
        assert checkpoint.trace_id == "test-trace-123"
        assert checkpoint.node_id == "node1"

    @pytest.mark.asyncio
    async def test_checkpoint_restore_memory(self, memory_checkpoint_manager, execution_context):
        """Test restoring from a checkpoint with in-memory storage."""
        # Save checkpoint
        await memory_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node1",
            force=True,
        )

        # Load checkpoint
        checkpoint = await memory_checkpoint_manager.load_checkpoint(
            graph_id="test-graph",
            trace_id="test-trace-123",
        )

        assert checkpoint is not None

        # Restore to new context
        new_context = ExecutionContext(
            trace_id="test-trace-123",
            graph_id="test-graph",
            config=Config(),
        )

        await memory_checkpoint_manager.restore_context(checkpoint, new_context)

        # Verify state
        assert new_context.current_node == "node1"
        assert new_context.step_count == execution_context.step_count
        assert new_context.get_variable("test_var") == "test_value"
        assert len(new_context.messages) == len(execution_context.messages)

    @pytest.mark.asyncio
    async def test_checkpoint_restore_sqlite(self, sqlite_checkpoint_manager, execution_context):
        """Test restoring from a checkpoint with SQLite storage."""
        # Save checkpoint
        await sqlite_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node1",
            force=True,
        )

        # Load checkpoint
        checkpoint = await sqlite_checkpoint_manager.load_checkpoint(
            graph_id="test-graph",
            trace_id="test-trace-123",
        )

        assert checkpoint is not None

        # Restore to new context
        new_context = ExecutionContext(
            trace_id="test-trace-123",
            graph_id="test-graph",
            config=Config(),
        )

        await sqlite_checkpoint_manager.restore_context(checkpoint, new_context)

        # Verify state
        assert new_context.current_node == "node1"
        assert new_context.step_count == execution_context.step_count
        assert new_context.get_variable("test_var") == "test_value"
        assert len(new_context.messages) == len(execution_context.messages)

    @pytest.mark.asyncio
    async def test_checkpoint_interval(self):
        """Test checkpoint interval logic."""
        config = CheckpointConfig(
            checkpoint_interval_ms=1000,
            checkpoint_after_each_node=False,
        )
        storage = InMemoryCheckpointStorage(StorageConfig())
        await storage.initialize()

        manager = CheckpointManager(storage, config)
        await manager.initialize()

        # Should not checkpoint immediately
        assert not manager.should_checkpoint()

        # Force checkpoint should work
        assert manager.should_checkpoint(force=True)

        await manager.close()

    @pytest.mark.asyncio
    async def test_checkpoint_after_each_node(self):
        """Test checkpoint_after_each_node config."""
        config = CheckpointConfig(
            checkpoint_interval_ms=0,
            checkpoint_after_each_node=True,
        )
        storage = InMemoryCheckpointStorage(StorageConfig())
        await storage.initialize()

        manager = CheckpointManager(storage, config)
        await manager.initialize()

        # Should always checkpoint when after_each_node is true
        assert manager.should_checkpoint()

        await manager.close()

    @pytest.mark.asyncio
    async def test_multiple_checkpoints(self, memory_checkpoint_manager, execution_context):
        """Test creating multiple checkpoints for same trace."""
        # Create first checkpoint
        cp1 = await memory_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node1",
            force=True,
        )

        # Advance context
        execution_context.visit_node("node2")

        # Create second checkpoint
        cp2 = await memory_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node2",
            force=True,
        )

        assert cp1.step < cp2.step

        # Load latest should return second checkpoint
        latest = await memory_checkpoint_manager.load_checkpoint(
            graph_id="test-graph",
            trace_id="test-trace-123",
        )

        assert latest.id == cp2.id
        assert latest.step == cp2.step

    @pytest.mark.asyncio
    async def test_checkpoint_pruning(self, memory_checkpoint_manager, execution_context):
        """Test that old checkpoints are pruned."""
        # Set max checkpoints to 3
        memory_checkpoint_manager.config.max_checkpoints_per_trace = 3

        # Create 5 checkpoints
        for i in range(5):
            execution_context.visit_node(f"node{i}")
            await memory_checkpoint_manager.save_checkpoint(
                context=execution_context,
                current_node_id=f"node{i}",
                force=True,
            )

        # Should only have 3 checkpoints
        checkpoints = await memory_checkpoint_manager.list_checkpoints(
            graph_id="test-graph",
            trace_id="test-trace-123",
        )

        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, memory_checkpoint_manager):
        """Test listing checkpoints."""
        # Create contexts for different traces
        context1 = ExecutionContext(
            trace_id="trace1",
            graph_id="graph1",
            config=Config(),
        )
        context1.visit_node("node1")

        context2 = ExecutionContext(
            trace_id="trace2",
            graph_id="graph1",
            config=Config(),
        )
        context2.visit_node("node1")

        # Save checkpoints
        await memory_checkpoint_manager.save_checkpoint(context1, "node1", force=True)
        await memory_checkpoint_manager.save_checkpoint(context2, "node1", force=True)

        # List all for graph
        checkpoints = await memory_checkpoint_manager.list_checkpoints("graph1")
        assert len(checkpoints) == 2

        # List for specific trace
        checkpoints = await memory_checkpoint_manager.list_checkpoints("graph1", "trace1")
        assert len(checkpoints) == 1
        assert checkpoints[0].trace_id == "trace1"

    @pytest.mark.asyncio
    async def test_clear_checkpoints(self, memory_checkpoint_manager, execution_context):
        """Test clearing checkpoints."""
        # Create checkpoint
        await memory_checkpoint_manager.save_checkpoint(
            context=execution_context,
            current_node_id="node1",
            force=True,
        )

        # Clear checkpoints
        count = await memory_checkpoint_manager.clear_checkpoints(
            graph_id="test-graph",
            trace_id="test-trace-123",
        )

        assert count == 1

        # Verify cleared
        checkpoint = await memory_checkpoint_manager.load_checkpoint(
            graph_id="test-graph",
            trace_id="test-trace-123",
        )

        assert checkpoint is None

    @pytest.mark.asyncio
    async def test_message_serialization(self, memory_checkpoint_manager):
        """Test that messages are properly serialized and restored."""
        context = ExecutionContext(
            trace_id="test-trace",
            graph_id="test-graph",
            config=Config(),
        )

        # Add message with structured data
        msg = Message(
            trace_id="test-trace",
            source_node="node1",
            target_node="node2",
            payload=MessagePayload(
                content="test",
                structured={"key": "value", "nested": {"a": 1}},
            ),
        )
        context.add_message(msg)
        context.visit_node("node1")

        # Save and restore
        await memory_checkpoint_manager.save_checkpoint(context, "node1", force=True)

        checkpoint = await memory_checkpoint_manager.load_checkpoint(
            "test-graph", "test-trace"
        )

        new_context = ExecutionContext(
            trace_id="test-trace",
            graph_id="test-graph",
            config=Config(),
        )

        await memory_checkpoint_manager.restore_context(checkpoint, new_context)

        # Verify message restored
        assert len(new_context.messages) == 1
        restored_msg = new_context.messages[0]
        assert restored_msg.payload.content == "test"
        assert restored_msg.payload.structured["key"] == "value"
        assert restored_msg.payload.structured["nested"]["a"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
