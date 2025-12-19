"""Tests for node implementations."""

import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult, NodeStats
from tinyllm.core.registry import NodeRegistry
from tinyllm.nodes.entry_exit import EntryNode, ExitNode


@pytest.fixture
def execution_context():
    """Create a test execution context."""
    return ExecutionContext(
        trace_id="test-trace-123",
        graph_id="test-graph",
        config=Config(),
    )


@pytest.fixture
def sample_message():
    """Create a sample test message."""
    return Message(
        trace_id="test-trace-123",
        source_node="test",
        payload=MessagePayload(
            task="Test task",
            content="Test content",
        ),
    )


class TestNodeStats:
    """Tests for NodeStats model."""

    def test_initial_stats(self):
        """Test initial stats values."""
        stats = NodeStats()
        assert stats.total_executions == 0
        assert stats.success_rate == 0.0
        assert stats.failure_rate == 1.0
        assert stats.avg_latency_ms == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = NodeStats(
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
        )
        assert stats.success_rate == 0.8
        assert abs(stats.failure_rate - 0.2) < 0.01  # Float comparison

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        stats = NodeStats(
            total_executions=5,
            total_latency_ms=1000,
        )
        assert stats.avg_latency_ms == 200.0


class TestNodeResult:
    """Tests for NodeResult model."""

    def test_success_result(self):
        """Test creating a success result."""
        result = NodeResult.success_result(
            output_messages=[],
            next_nodes=["next.node"],
            latency_ms=100,
        )
        assert result.success is True
        assert result.next_nodes == ["next.node"]
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = NodeResult.failure_result(
            error="Something went wrong",
            latency_ms=50,
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestNodeRegistry:
    """Tests for NodeRegistry."""

    def test_entry_node_registered(self):
        """Test EntryNode is registered."""
        assert NodeRegistry.is_registered(NodeType.ENTRY)

    def test_exit_node_registered(self):
        """Test ExitNode is registered."""
        assert NodeRegistry.is_registered(NodeType.EXIT)

    def test_create_entry_node(self):
        """Test creating an entry node from definition."""
        definition = NodeDefinition(
            id="entry.test",
            type=NodeType.ENTRY,
            config={"timeout_ms": 3000},
        )
        node = NodeRegistry.create(definition)
        assert isinstance(node, EntryNode)
        assert node.id == "entry.test"

    def test_create_exit_node(self):
        """Test creating an exit node from definition."""
        definition = NodeDefinition(
            id="exit.test",
            type=NodeType.EXIT,
            config={"status": "success"},
        )
        node = NodeRegistry.create(definition)
        assert isinstance(node, ExitNode)
        assert node.id == "exit.test"


class TestEntryNode:
    """Tests for EntryNode."""

    @pytest.mark.asyncio
    async def test_basic_execution(self, execution_context, sample_message):
        """Test basic entry node execution."""
        definition = NodeDefinition(
            id="entry.main",
            type=NodeType.ENTRY,
        )
        node = EntryNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert len(result.output_messages) == 1
        assert result.metadata.get("validated") is True

    @pytest.mark.asyncio
    async def test_required_fields_present(self, execution_context, sample_message):
        """Test with required fields that are present."""
        definition = NodeDefinition(
            id="entry.main",
            type=NodeType.ENTRY,
            config={"required_fields": ["content"]},
        )
        node = EntryNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_required_fields_missing(self, execution_context):
        """Test with required fields that are missing."""
        definition = NodeDefinition(
            id="entry.main",
            type=NodeType.ENTRY,
            config={"required_fields": ["nonexistent_field"]},
        )
        node = EntryNode(definition)

        message = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )

        result = await node.execute(message, execution_context)

        assert result.success is False
        assert "Missing required fields" in result.error


class TestExitNode:
    """Tests for ExitNode."""

    @pytest.mark.asyncio
    async def test_success_exit(self, execution_context, sample_message):
        """Test success exit node."""
        definition = NodeDefinition(
            id="exit.success",
            type=NodeType.EXIT,
            config={"status": "success"},
        )
        node = ExitNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata.get("is_terminal") is True
        assert result.metadata.get("exit_status") == "success"
        assert result.next_nodes == []

    @pytest.mark.asyncio
    async def test_fallback_exit(self, execution_context, sample_message):
        """Test fallback exit node."""
        definition = NodeDefinition(
            id="exit.fallback",
            type=NodeType.EXIT,
            config={"status": "fallback"},
        )
        node = ExitNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is False
        assert result.metadata.get("exit_status") == "fallback"

    @pytest.mark.asyncio
    async def test_stats_update(self, execution_context, sample_message):
        """Test that stats are updated after execution."""
        definition = NodeDefinition(
            id="exit.test",
            type=NodeType.EXIT,
        )
        node = ExitNode(definition)

        # Initial stats
        assert node.stats.total_executions == 0

        # Execute
        result = await node.execute(sample_message, execution_context)
        node.update_stats(result.success, result.latency_ms)

        # Stats should be updated
        assert node.stats.total_executions == 1
        assert node.stats.successful_executions == 1
