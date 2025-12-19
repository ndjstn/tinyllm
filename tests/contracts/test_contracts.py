"""Contract tests for node interfaces.

These tests verify that all node implementations adhere to the BaseNode
interface contract, ensuring consistent behavior across the system.
"""

import asyncio
from abc import ABC
from typing import Type

import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult, NodeStats
from tinyllm.core.registry import NodeRegistry


class NodeContractTestBase(ABC):
    """Base class for node contract tests.

    Each node type should have tests that inherit from this class
    to ensure they meet the basic contract requirements.
    """

    @property
    def node_type(self) -> NodeType:
        """The node type being tested."""
        raise NotImplementedError

    @property
    def minimal_config(self) -> dict:
        """Minimal valid configuration for this node type."""
        return {}

    def create_node_definition(self, **config_overrides) -> NodeDefinition:
        """Create a node definition for testing.

        Args:
            **config_overrides: Additional config to merge with minimal_config.

        Returns:
            NodeDefinition instance.
        """
        config = {**self.minimal_config, **config_overrides}
        return NodeDefinition(
            id=f"test.{self.node_type.value}",
            type=self.node_type,
            config=config,
        )

    def create_node(self, **config_overrides) -> BaseNode:
        """Create a node instance for testing.

        Args:
            **config_overrides: Additional config to merge with minimal_config.

        Returns:
            BaseNode instance.
        """
        definition = self.create_node_definition(**config_overrides)
        return NodeRegistry.create(definition)

    @pytest.fixture
    def execution_context(self):
        """Create test execution context."""
        return ExecutionContext(
            trace_id="test-trace-contract",
            graph_id="test-graph",
            config=Config(),
        )

    @pytest.fixture
    def test_message(self):
        """Create test message."""
        return Message(
            trace_id="test-trace-contract",
            source_node="test",
            payload=MessagePayload(
                task="Test task for contract validation",
                content="Test content",
            ),
        )

    # Contract Tests

    def test_node_is_registered(self):
        """Contract: Node type must be registered in NodeRegistry."""
        assert NodeRegistry.is_registered(self.node_type), \
            f"{self.node_type} is not registered"

    def test_node_creation_from_definition(self):
        """Contract: Node must be creatable from NodeDefinition."""
        definition = self.create_node_definition()
        node = NodeRegistry.create(definition)

        assert node is not None
        assert isinstance(node, BaseNode)
        assert node.type == self.node_type
        assert node.id == definition.id

    def test_node_has_required_attributes(self):
        """Contract: Node must have all required attributes."""
        node = self.create_node()

        # Required attributes from BaseNode
        assert hasattr(node, "id")
        assert hasattr(node, "type")
        assert hasattr(node, "name")
        assert hasattr(node, "description")
        assert hasattr(node, "config")
        assert hasattr(node, "stats")
        assert hasattr(node, "execute")
        assert hasattr(node, "update_stats")

    def test_node_config_is_valid(self):
        """Contract: Node config must be NodeConfig instance."""
        node = self.create_node()
        assert isinstance(node.config, NodeConfig)

    def test_node_stats_is_valid(self):
        """Contract: Node stats must be NodeStats instance."""
        node = self.create_node()
        assert isinstance(node.stats, NodeStats)

    def test_node_stats_initialized_correctly(self):
        """Contract: Node stats must start at zero."""
        node = self.create_node()
        stats = node.stats

        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0
        assert stats.total_latency_ms == 0
        assert stats.last_execution is None

    @pytest.mark.asyncio
    async def test_execute_returns_node_result(self, execution_context, test_message):
        """Contract: execute() must return NodeResult."""
        node = self.create_node()
        result = await node.execute(test_message, execution_context)

        assert isinstance(result, NodeResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.output_messages, list)
        assert isinstance(result.next_nodes, list)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.latency_ms, int)

    @pytest.mark.asyncio
    async def test_execute_output_messages_are_valid(self, execution_context, test_message):
        """Contract: Output messages must be Message instances."""
        node = self.create_node()
        result = await node.execute(test_message, execution_context)

        for msg in result.output_messages:
            assert isinstance(msg, Message)
            assert msg.trace_id == test_message.trace_id
            assert msg.source_node == node.id

    @pytest.mark.asyncio
    async def test_execute_preserves_trace_id(self, execution_context, test_message):
        """Contract: Execution must preserve trace_id in output messages."""
        node = self.create_node()
        result = await node.execute(test_message, execution_context)

        if result.output_messages:
            for msg in result.output_messages:
                assert msg.trace_id == test_message.trace_id

    @pytest.mark.asyncio
    async def test_execute_sets_source_node(self, execution_context, test_message):
        """Contract: Output messages must have correct source_node."""
        node = self.create_node()
        result = await node.execute(test_message, execution_context)

        if result.output_messages:
            for msg in result.output_messages:
                assert msg.source_node == node.id

    @pytest.mark.asyncio
    async def test_execute_next_nodes_are_strings(self, execution_context, test_message):
        """Contract: next_nodes must be list of strings."""
        node = self.create_node()
        result = await node.execute(test_message, execution_context)

        assert isinstance(result.next_nodes, list)
        for next_node in result.next_nodes:
            assert isinstance(next_node, str)

    @pytest.mark.asyncio
    async def test_execute_error_on_failure(self, execution_context, test_message):
        """Contract: Failed execution must set error field."""
        node = self.create_node()
        result = await node.execute(test_message, execution_context)

        if not result.success:
            assert result.error is not None
            assert isinstance(result.error, str)
            assert len(result.error) > 0

    def test_update_stats_increments_total(self):
        """Contract: update_stats must increment total_executions."""
        node = self.create_node()
        initial_total = node.stats.total_executions

        node.update_stats(success=True, latency_ms=100)

        assert node.stats.total_executions == initial_total + 1

    def test_update_stats_increments_success(self):
        """Contract: update_stats with success=True increments successful_executions."""
        node = self.create_node()
        initial_success = node.stats.successful_executions

        node.update_stats(success=True, latency_ms=100)

        assert node.stats.successful_executions == initial_success + 1

    def test_update_stats_increments_failure(self):
        """Contract: update_stats with success=False increments failed_executions."""
        node = self.create_node()
        initial_failed = node.stats.failed_executions

        node.update_stats(success=False, latency_ms=100)

        assert node.stats.failed_executions == initial_failed + 1

    def test_update_stats_adds_latency(self):
        """Contract: update_stats must add latency to total."""
        node = self.create_node()
        initial_latency = node.stats.total_latency_ms

        node.update_stats(success=True, latency_ms=150)

        assert node.stats.total_latency_ms == initial_latency + 150

    def test_update_stats_sets_last_execution(self):
        """Contract: update_stats must set last_execution timestamp."""
        node = self.create_node()

        node.update_stats(success=True, latency_ms=100)

        assert node.stats.last_execution is not None

    def test_node_repr(self):
        """Contract: Node must have meaningful __repr__."""
        node = self.create_node()
        repr_str = repr(node)

        assert node.id in repr_str
        assert str(node.type.value) in repr_str or node.__class__.__name__ in repr_str

    def test_config_timeout_validation(self):
        """Contract: Config timeout_ms must be within valid range."""
        # Valid timeout
        node = self.create_node(timeout_ms=5000)
        assert node.config.timeout_ms == 5000

        # Too low - should fail
        with pytest.raises(Exception):
            self.create_node(timeout_ms=50)

        # Too high - should fail
        with pytest.raises(Exception):
            self.create_node(timeout_ms=200000)

    def test_config_retry_validation(self):
        """Contract: Config retry_count must be within valid range."""
        # Valid retry count
        node = self.create_node(retry_count=2)
        assert node.config.retry_count == 2

        # Too high - should fail
        with pytest.raises(Exception):
            self.create_node(retry_count=10)

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, execution_context, test_message):
        """Contract: Node must handle concurrent execution safely."""
        node = self.create_node()

        async def execute_node():
            return await node.execute(test_message, execution_context)

        # Execute concurrently
        results = await asyncio.gather(*[execute_node() for _ in range(5)])

        assert len(results) == 5
        for result in results:
            assert isinstance(result, NodeResult)


# Concrete contract test classes for each node type


class TestEntryNodeContract(NodeContractTestBase):
    """Contract tests for EntryNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.ENTRY


class TestExitNodeContract(NodeContractTestBase):
    """Contract tests for ExitNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.EXIT

    @property
    def minimal_config(self) -> dict:
        return {"status": "success"}


class TestModelNodeContract(NodeContractTestBase):
    """Contract tests for ModelNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.MODEL

    @property
    def minimal_config(self) -> dict:
        return {"model": "qwen2.5:3b"}

    @pytest.mark.asyncio
    async def test_model_config_has_model_field(self):
        """Model nodes must have model configuration."""
        node = self.create_node()
        assert hasattr(node, "model_config")
        assert hasattr(node.model_config, "model")


class TestRouterNodeContract(NodeContractTestBase):
    """Contract tests for RouterNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.ROUTER

    @property
    def minimal_config(self) -> dict:
        return {
            "model": "qwen2.5:0.5b",
            "routes": [
                {
                    "name": "code",
                    "description": "Programming and code-related tasks",
                    "target": "code.specialist",
                },
                {
                    "name": "math",
                    "description": "Mathematical and computational tasks",
                    "target": "math.specialist",
                },
                {
                    "name": "general",
                    "description": "General questions and tasks",
                    "target": "general.specialist",
                },
            ],
        }


class TestToolNodeContract(NodeContractTestBase):
    """Contract tests for ToolNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.TOOL

    @property
    def minimal_config(self) -> dict:
        return {"tool_id": "calculator"}


class TestGateNodeContract(NodeContractTestBase):
    """Contract tests for GateNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.GATE

    @property
    def minimal_config(self) -> dict:
        return {
            "model": "qwen2.5:3b",
            "pass_threshold": 0.7,
        }


class TestTransformNodeContract(NodeContractTestBase):
    """Contract tests for TransformNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.TRANSFORM

    @property
    def minimal_config(self) -> dict:
        return {
            "transforms": [
                {"type": "strip"},
            ]
        }


class TestLoopNodeContract(NodeContractTestBase):
    """Contract tests for LoopNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.LOOP

    @property
    def minimal_config(self) -> dict:
        return {
            "max_iterations": 3,
            "condition": "continue",
        }


class TestFanoutNodeContract(NodeContractTestBase):
    """Contract tests for FanoutNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.FANOUT

    @property
    def minimal_config(self) -> dict:
        return {
            "target_nodes": ["node1", "node2"],
            "mode": "parallel",
        }


class TestTimeoutNodeContract(NodeContractTestBase):
    """Contract tests for TimeoutNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.TIMEOUT

    @property
    def minimal_config(self) -> dict:
        return {
            "timeout_ms": 5000,
            "inner_node": "test.inner",
        }


class TestReasoningNodeContract(NodeContractTestBase):
    """Contract tests for ReasoningNode."""

    @property
    def node_type(self) -> NodeType:
        return NodeType.REASONING

    @property
    def minimal_config(self) -> dict:
        return {"model": "qwen2.5:3b"}


# Integration tests for node contract compliance


class TestAllNodesContractCompliance:
    """Tests to ensure all registered nodes comply with contracts."""

    def test_all_node_types_registered(self):
        """All NodeType enum values should be registered."""
        for node_type in NodeType:
            assert NodeRegistry.is_registered(node_type), \
                f"NodeType.{node_type.name} is not registered"

    def test_all_registered_nodes_inherit_basenode(self):
        """All registered nodes must inherit from BaseNode."""
        for node_type in NodeType:
            node_class = NodeRegistry._node_types.get(node_type)
            assert node_class is not None, f"NodeType.{node_type.name} not registered"
            assert issubclass(node_class, BaseNode)

    def test_all_nodes_have_unique_types(self):
        """Each node class should be registered to exactly one type."""
        type_to_class = {}
        for node_type in NodeType:
            node_class = NodeRegistry._node_types.get(node_type)
            if node_class in type_to_class.values():
                pytest.fail(f"Node class {node_class} registered to multiple types")
            type_to_class[node_type] = node_class
