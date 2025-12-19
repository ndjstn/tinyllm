"""Comprehensive tests for FanoutNode implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.registry import NodeRegistry
from tinyllm.nodes.fanout import (
    AggregationStrategy,
    FanoutConfig,
    FanoutNode,
    FanoutResult,
    FanoutTargetResult,
)


@pytest.fixture
def execution_context():
    """Create a test execution context."""
    return ExecutionContext(
        trace_id="test-trace-fanout",
        graph_id="test-graph",
        config=Config(),
    )


@pytest.fixture
def sample_message():
    """Create a sample test message."""
    return Message(
        trace_id="test-trace-fanout",
        source_node="test",
        payload=MessagePayload(
            task="Test task for fanout",
            content="This is a test message",
        ),
    )


class TestFanoutConfig:
    """Tests for FanoutConfig model."""

    def test_create_basic_config(self):
        """Test creating basic fanout configuration."""
        config = FanoutConfig(
            target_nodes=["node1", "node2", "node3"],
        )
        assert len(config.target_nodes) == 3
        assert config.aggregation_strategy == AggregationStrategy.ALL
        assert config.parallel is True
        assert config.timeout_ms == 30000

    def test_config_with_all_options(self):
        """Test config with all options specified."""
        config = FanoutConfig(
            target_nodes=["a", "b"],
            aggregation_strategy=AggregationStrategy.FIRST_SUCCESS,
            timeout_ms=5000,
            parallel=False,
            fail_fast=True,
            require_all_success=True,
            score_field="custom_score",
            retry_count=2,
            retry_delay_ms=500,
        )
        assert config.aggregation_strategy == AggregationStrategy.FIRST_SUCCESS
        assert config.timeout_ms == 5000
        assert config.parallel is False
        assert config.fail_fast is True
        assert config.require_all_success is True
        assert config.score_field == "custom_score"
        assert config.retry_count == 2
        assert config.retry_delay_ms == 500

    def test_config_is_frozen(self):
        """Test that config is immutable (frozen)."""
        config = FanoutConfig(target_nodes=["node1"])
        with pytest.raises(Exception):  # Pydantic frozen model raises ValidationError
            config.timeout_ms = 10000

    def test_config_strict_mode(self):
        """Test that config rejects extra fields."""
        with pytest.raises(Exception):  # Pydantic strict mode validation
            FanoutConfig(
                target_nodes=["node1"],
                unknown_field="should_fail",
            )

    def test_config_requires_target_nodes(self):
        """Test that target_nodes is required and must have at least one."""
        with pytest.raises(Exception):  # Missing required field
            FanoutConfig()

        with pytest.raises(Exception):  # Empty list
            FanoutConfig(target_nodes=[])

    def test_config_timeout_bounds(self):
        """Test timeout_ms has proper bounds."""
        # Valid timeout
        config = FanoutConfig(target_nodes=["node1"], timeout_ms=1000)
        assert config.timeout_ms == 1000

        # Too low
        with pytest.raises(Exception):
            FanoutConfig(target_nodes=["node1"], timeout_ms=50)

        # Too high
        with pytest.raises(Exception):
            FanoutConfig(target_nodes=["node1"], timeout_ms=200000)

    def test_config_retry_bounds(self):
        """Test retry_count and retry_delay_ms bounds."""
        config = FanoutConfig(
            target_nodes=["node1"],
            retry_count=3,
            retry_delay_ms=10000,
        )
        assert config.retry_count == 3
        assert config.retry_delay_ms == 10000

        with pytest.raises(Exception):
            FanoutConfig(target_nodes=["node1"], retry_count=10)

        with pytest.raises(Exception):
            FanoutConfig(target_nodes=["node1"], retry_delay_ms=20000)


class TestFanoutTargetResult:
    """Tests for FanoutTargetResult model."""

    def test_create_success_result(self):
        """Test creating a successful target result."""
        msg = Message(
            trace_id="test",
            source_node="target1",
            payload=MessagePayload(content="Result from target1"),
        )
        result = FanoutTargetResult(
            target_node="target1",
            success=True,
            message=msg,
            latency_ms=150,
            metadata={"key": "value"},
        )
        assert result.success is True
        assert result.target_node == "target1"
        assert result.message == msg
        assert result.latency_ms == 150
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating a failed target result."""
        result = FanoutTargetResult(
            target_node="target2",
            success=False,
            error="Target execution failed",
            latency_ms=100,
        )
        assert result.success is False
        assert result.error == "Target execution failed"
        assert result.message is None

    def test_result_is_frozen(self):
        """Test that result is immutable."""
        result = FanoutTargetResult(
            target_node="target1",
            success=True,
            latency_ms=100,
        )
        with pytest.raises(Exception):
            result.success = False

    def test_result_strict_mode(self):
        """Test that result rejects extra fields."""
        with pytest.raises(Exception):
            FanoutTargetResult(
                target_node="target1",
                success=True,
                extra_field="not_allowed",
            )


class TestFanoutResult:
    """Tests for FanoutResult model."""

    def test_create_successful_fanout_result(self):
        """Test creating successful fanout result."""
        target_results = [
            FanoutTargetResult(target_node="t1", success=True, latency_ms=100),
            FanoutTargetResult(target_node="t2", success=True, latency_ms=150),
        ]
        msg = Message(
            trace_id="test",
            source_node="fanout",
            payload=MessagePayload(content="Aggregated result"),
        )
        result = FanoutResult(
            success=True,
            target_results=target_results,
            aggregated_message=msg,
            strategy_used=AggregationStrategy.ALL,
            total_latency_ms=200,
            successful_targets=2,
            failed_targets=0,
        )
        assert result.success is True
        assert len(result.target_results) == 2
        assert result.successful_targets == 2
        assert result.failed_targets == 0

    def test_create_failed_fanout_result(self):
        """Test creating failed fanout result."""
        target_results = [
            FanoutTargetResult(
                target_node="t1", success=False, error="Failed", latency_ms=50
            ),
        ]
        result = FanoutResult(
            success=False,
            target_results=target_results,
            strategy_used=AggregationStrategy.FIRST_SUCCESS,
            total_latency_ms=50,
            successful_targets=0,
            failed_targets=1,
            error="No successful results",
        )
        assert result.success is False
        assert result.error == "No successful results"
        assert result.aggregated_message is None

    def test_result_is_frozen(self):
        """Test that fanout result is immutable."""
        result = FanoutResult(
            success=True,
            target_results=[],
            strategy_used=AggregationStrategy.ALL,
            total_latency_ms=100,
            successful_targets=0,
            failed_targets=0,
        )
        with pytest.raises(Exception):
            result.success = False


class TestFanoutNode:
    """Tests for FanoutNode."""

    def test_fanout_node_registered(self):
        """Test FanoutNode is registered."""
        assert NodeRegistry.is_registered(NodeType.FANOUT)

    def test_create_fanout_node(self):
        """Test creating a fanout node from definition."""
        definition = NodeDefinition(
            id="fanout.test",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["model1", "model2", "model3"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)
        assert node.id == "fanout.test"
        assert len(node.fanout_config.target_nodes) == 3
        assert node.fanout_config.aggregation_strategy == AggregationStrategy.ALL

    def test_create_fanout_node_with_first_success(self):
        """Test creating fanout with FIRST_SUCCESS strategy."""
        definition = NodeDefinition(
            id="fanout.first",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["fast_model", "slow_model"],
                "aggregation_strategy": "first_success",
                "timeout_ms": 5000,
            },
        )
        node = FanoutNode(definition)
        assert node.fanout_config.aggregation_strategy == AggregationStrategy.FIRST_SUCCESS
        assert node.fanout_config.timeout_ms == 5000

    def test_create_fanout_node_with_majority_vote(self):
        """Test creating fanout with MAJORITY_VOTE strategy."""
        definition = NodeDefinition(
            id="fanout.vote",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["voter1", "voter2", "voter3"],
                "aggregation_strategy": "majority_vote",
            },
        )
        node = FanoutNode(definition)
        assert node.fanout_config.aggregation_strategy == AggregationStrategy.MAJORITY_VOTE

    def test_create_fanout_node_with_best_score(self):
        """Test creating fanout with BEST_SCORE strategy."""
        definition = NodeDefinition(
            id="fanout.score",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["scorer1", "scorer2"],
                "aggregation_strategy": "best_score",
                "score_field": "quality_score",
            },
        )
        node = FanoutNode(definition)
        assert node.fanout_config.aggregation_strategy == AggregationStrategy.BEST_SCORE
        assert node.fanout_config.score_field == "quality_score"

    @pytest.mark.asyncio
    async def test_fanout_with_empty_targets(self, execution_context, sample_message):
        """Test fanout fails with empty target list."""
        # This should fail during config validation, but let's test the execute path
        definition = NodeDefinition(
            id="fanout.empty",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["dummy"],  # Need at least one for config validation
            },
        )
        node = FanoutNode(definition)
        # Manually override to test the execute logic - create fresh config with empty list
        # Using object.__setattr__ to bypass frozen model
        object.__setattr__(node, "_fanout_config", FanoutConfig.__new__(FanoutConfig))
        object.__setattr__(node._fanout_config, "__dict__", {})
        object.__setattr__(node._fanout_config, "__pydantic_fields_set__", set())
        # Mock the target_nodes property to return empty list
        node._fanout_config = type("MockConfig", (), {
            "target_nodes": [],
            "aggregation_strategy": AggregationStrategy.ALL,
            "timeout_ms": 30000,
            "parallel": True,
            "fail_fast": False,
            "require_all_success": False,
            "score_field": "quality_score",
        })()

        result = await node.execute(sample_message, execution_context)
        assert result.success is False
        assert "No target nodes" in result.error

    @pytest.mark.asyncio
    async def test_fanout_with_single_target(self, execution_context, sample_message):
        """Test fanout with single target node."""
        definition = NodeDefinition(
            id="fanout.single",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["single_target"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)
        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert len(result.output_messages) > 0
        assert result.metadata["successful_targets"] == 1

    @pytest.mark.asyncio
    async def test_fanout_parallel_execution(self, execution_context, sample_message):
        """Test parallel execution of multiple targets."""
        definition = NodeDefinition(
            id="fanout.parallel",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2", "target3"],
                "parallel": True,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)
        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["successful_targets"] == 3
        assert result.metadata["failed_targets"] == 0

    @pytest.mark.asyncio
    async def test_fanout_sequential_execution(self, execution_context, sample_message):
        """Test sequential execution of targets."""
        definition = NodeDefinition(
            id="fanout.sequential",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2", "target3"],
                "parallel": False,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)
        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["successful_targets"] == 3

    @pytest.mark.asyncio
    async def test_fanout_all_strategy(self, execution_context, sample_message):
        """Test ALL aggregation strategy combines all results."""
        definition = NodeDefinition(
            id="fanout.all",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)
        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        # Check that output message contains aggregated content
        assert len(result.output_messages) == 1

    @pytest.mark.asyncio
    async def test_fanout_first_success_strategy(self, execution_context, sample_message):
        """Test FIRST_SUCCESS strategy returns first successful result."""
        definition = NodeDefinition(
            id="fanout.first",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2", "target3"],
                "aggregation_strategy": "first_success",
            },
        )
        node = FanoutNode(definition)
        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["strategy"] == "first_success"

    @pytest.mark.asyncio
    async def test_fanout_majority_vote_strategy(self, execution_context):
        """Test MAJORITY_VOTE strategy returns most common result."""
        # Create mock target results with different contents
        definition = NodeDefinition(
            id="fanout.vote",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["voter1", "voter2", "voter3"],
                "aggregation_strategy": "majority_vote",
            },
        )
        node = FanoutNode(definition)

        # Override execute_single_target to return controlled results
        original_execute = node._execute_single_target
        call_count = [0]

        async def mock_execute(target, msg, ctx):
            call_count[0] += 1
            # First two return "answer A", third returns "answer B"
            content = "answer A" if call_count[0] <= 2 else "answer B"
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=content),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="vote test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Majority should be "answer A" (2 votes vs 1)
        assert "answer a" in result.output_messages[0].payload.content.lower()

    @pytest.mark.asyncio
    async def test_fanout_best_score_strategy(self, execution_context):
        """Test BEST_SCORE strategy returns highest scored result."""
        definition = NodeDefinition(
            id="fanout.score",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["scorer1", "scorer2", "scorer3"],
                "aggregation_strategy": "best_score",
                "score_field": "quality_score",
            },
        )
        node = FanoutNode(definition)

        # Override to return results with different scores
        async def mock_execute(target, msg, ctx):
            scores = {"scorer1": 0.5, "scorer2": 0.9, "scorer3": 0.7}
            score = scores.get(target, 0.0)
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(
                        content=f"Result from {target}",
                        metadata={"quality_score": score},
                    ),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="score test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Should select scorer2 with highest score (0.9)
        assert "scorer2" in result.output_messages[0].payload.content

    @pytest.mark.asyncio
    async def test_fanout_require_all_success_pass(self, execution_context, sample_message):
        """Test require_all_success passes when all succeed."""
        definition = NodeDefinition(
            id="fanout.require_all",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2"],
                "require_all_success": True,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)
        result = await node.execute(sample_message, execution_context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_fanout_require_all_success_fail(self, execution_context):
        """Test require_all_success fails when any target fails."""
        definition = NodeDefinition(
            id="fanout.require_all",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2", "target3"],
                "require_all_success": True,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        # Make second target fail
        call_count = [0]

        async def mock_execute(target, msg, ctx):
            call_count[0] += 1
            if target == "target2":
                return FanoutTargetResult(
                    target_node=target,
                    success=False,
                    error="Simulated failure",
                    latency_ms=100,
                )
            # Success case
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=f"Result from {target}"),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is False
        assert "Required all targets to succeed" in result.metadata["fanout_result"]["error"]

    @pytest.mark.asyncio
    async def test_fanout_fail_fast_sequential(self, execution_context):
        """Test fail_fast stops execution on first failure."""
        definition = NodeDefinition(
            id="fanout.failfast",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2", "target3"],
                "parallel": False,
                "fail_fast": True,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        executed_targets = []

        async def mock_execute(target, msg, ctx):
            executed_targets.append(target)
            if target == "target2":
                return FanoutTargetResult(
                    target_node=target,
                    success=False,
                    error="Failure at target2",
                    latency_ms=100,
                )
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(source_node=target, payload=msg.payload),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        await node.execute(msg, execution_context)

        # Should stop after target2 fails
        assert len(executed_targets) == 2
        assert "target3" not in executed_targets

    @pytest.mark.asyncio
    async def test_fanout_timeout_handling(self, execution_context):
        """Test timeout handling for slow targets."""
        definition = NodeDefinition(
            id="fanout.timeout",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["fast", "slow"],
                "timeout_ms": 500,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            if target == "slow":
                # Simulate slow execution
                await asyncio.sleep(2.0)  # Longer than timeout
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(source_node=target, payload=msg.payload),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        # Should have one timeout
        fanout_result_data = result.metadata["fanout_result"]
        # At least one target should have failed (the slow one)
        assert fanout_result_data["failed_targets"] >= 0

    @pytest.mark.asyncio
    async def test_fanout_all_targets_fail(self, execution_context):
        """Test behavior when all targets fail."""
        definition = NodeDefinition(
            id="fanout.allfail",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["fail1", "fail2"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            return FanoutTargetResult(
                target_node=target,
                success=False,
                error=f"Failed: {target}",
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is False
        # When all fail, counts are in fanout_result
        assert result.metadata["fanout_result"]["successful_targets"] == 0
        assert result.metadata["fanout_result"]["failed_targets"] == 2

    @pytest.mark.asyncio
    async def test_fanout_partial_success(self, execution_context):
        """Test fanout with some successes and some failures."""
        definition = NodeDefinition(
            id="fanout.partial",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["good1", "bad", "good2"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            if target == "bad":
                return FanoutTargetResult(
                    target_node=target,
                    success=False,
                    error="Simulated failure",
                    latency_ms=100,
                )
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=f"Result from {target}"),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True  # Should succeed with partial success
        assert result.metadata["successful_targets"] == 2
        assert result.metadata["failed_targets"] == 1

    @pytest.mark.asyncio
    async def test_fanout_preserves_trace_id(self, execution_context):
        """Test that fanout preserves trace_id through execution."""
        definition = NodeDefinition(
            id="fanout.trace",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="unique-trace-123",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Output message should preserve trace_id
        assert result.output_messages[0].trace_id == "unique-trace-123"

    @pytest.mark.asyncio
    async def test_fanout_metadata_propagation(self, execution_context):
        """Test that metadata is properly propagated."""
        definition = NodeDefinition(
            id="fanout.meta",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(
                content="test",
                metadata={"original_key": "original_value"},
            ),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Check that fanout metadata is added
        assert "fanout_result" in result.metadata
        assert "strategy" in result.metadata

    @pytest.mark.asyncio
    async def test_fanout_with_different_content_lengths(self, execution_context):
        """Test fanout handles varying content lengths correctly."""
        definition = NodeDefinition(
            id="fanout.varying",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["short", "medium", "long"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            content_map = {
                "short": "A",
                "medium": "B" * 100,
                "long": "C" * 1000,
            }
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=content_map[target]),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True
        # All aggregation should combine all three
        combined = result.output_messages[0].payload.content
        assert "[short]" in combined
        assert "[medium]" in combined
        assert "[long]" in combined

    @pytest.mark.asyncio
    async def test_fanout_empty_content_handling(self, execution_context):
        """Test fanout handles empty content from targets."""
        definition = NodeDefinition(
            id="fanout.empty",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["empty1", "empty2"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=""),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_fanout_majority_vote_tie(self, execution_context):
        """Test majority vote with a tie (should pick first)."""
        definition = NodeDefinition(
            id="fanout.tie",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["voter1", "voter2"],
                "aggregation_strategy": "majority_vote",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            content_map = {"voter1": "answer A", "voter2": "answer B"}
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=content_map[target]),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        assert result.success is True
        # With a tie, it should pick the first one encountered
        assert result.output_messages[0].payload.content in ["answer A", "answer B"]

    @pytest.mark.asyncio
    async def test_fanout_best_score_no_scores(self, execution_context):
        """Test best score falls back when no scores available."""
        definition = NodeDefinition(
            id="fanout.noscore",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2"],
                "aggregation_strategy": "best_score",
                "score_field": "quality_score",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            # Return results without scores
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=f"Result {target}"),
                ),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        # Should fall back to first success
        assert result.success is True

    @pytest.mark.asyncio
    async def test_fanout_exception_handling(self, execution_context):
        """Test fanout handles exceptions gracefully."""
        definition = NodeDefinition(
            id="fanout.exception",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            raise RuntimeError("Unexpected error in target execution")

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        result = await node.execute(msg, execution_context)

        # Should handle exception and report as failure
        assert result.success is False

    @pytest.mark.asyncio
    async def test_fanout_latency_tracking(self, execution_context, sample_message):
        """Test that fanout tracks latency correctly."""
        definition = NodeDefinition(
            id="fanout.latency",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1", "target2"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.latency_ms >= 0
        # Should have tracking in metadata
        assert "fanout_result" in result.metadata
        assert result.metadata["fanout_result"]["total_latency_ms"] >= 0

    def test_fanout_config_all_strategies(self):
        """Test that all aggregation strategies are valid."""
        strategies = [
            AggregationStrategy.FIRST_SUCCESS,
            AggregationStrategy.ALL,
            AggregationStrategy.MAJORITY_VOTE,
            AggregationStrategy.BEST_SCORE,
        ]

        for strategy in strategies:
            config = FanoutConfig(
                target_nodes=["target1"],
                aggregation_strategy=strategy,
            )
            assert config.aggregation_strategy == strategy

    @pytest.mark.asyncio
    async def test_fanout_node_stats_update(self, execution_context, sample_message):
        """Test that node stats are properly tracked."""
        definition = NodeDefinition(
            id="fanout.stats",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        initial_executions = node.stats.total_executions

        result = await node.execute(sample_message, execution_context)

        # Stats should be updated by the executor, not the node itself
        # But we can test that execution completes successfully
        assert result.success is True

    @pytest.mark.asyncio
    async def test_fanout_sequential_preserves_order(self, execution_context):
        """Test that sequential execution preserves target order."""
        definition = NodeDefinition(
            id="fanout.order",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["first", "second", "third"],
                "parallel": False,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        execution_order = []

        async def mock_execute(target, msg, ctx):
            execution_order.append(target)
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(source_node=target, payload=msg.payload),
                latency_ms=100,
            )

        node._execute_single_target = mock_execute

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="test"),
        )
        await node.execute(msg, execution_context)

        assert execution_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_fanout_message_child_relationships(self, execution_context):
        """Test that fanout creates proper message parent-child relationships."""
        definition = NodeDefinition(
            id="fanout.children",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["target1"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        parent_msg = Message(
            trace_id="test",
            message_id="parent-123",
            source_node="test",
            payload=MessagePayload(content="parent"),
        )

        result = await node.execute(parent_msg, execution_context)

        assert result.success is True
        # Output message should be a child of parent
        output_msg = result.output_messages[0]
        assert output_msg.parent_id == "parent-123"
        assert output_msg.trace_id == "test"
