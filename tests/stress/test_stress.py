"""Stress tests for TinyLLM system.

Tests system behavior under heavy load and edge conditions.
"""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock

import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.nodes.fanout import (
    AggregationStrategy,
    FanoutNode,
    FanoutTargetResult,
)
from tinyllm.nodes.loop import LoopCondition, LoopNode
from tinyllm.nodes.transform import TransformNode


@pytest.fixture
def execution_context():
    """Create test execution context."""
    return ExecutionContext(
        trace_id="stress-test",
        graph_id="stress-graph",
        config=Config(),
    )


@pytest.fixture
def sample_message():
    """Create sample message for tests."""
    return Message(
        trace_id="stress-test",
        source_node="test",
        payload=MessagePayload(
            task="Stress test task",
            content="Stress test content",
        ),
    )


class TestFanoutStress:
    """Stress tests for FanoutNode parallel execution."""

    @pytest.mark.asyncio
    async def test_many_parallel_targets(self, execution_context, sample_message):
        """Test fanout with many parallel targets (10+)."""
        targets = [f"target_{i}" for i in range(15)]
        definition = NodeDefinition(
            id="fanout.stress",
            type=NodeType.FANOUT,
            config={
                "target_nodes": targets,
                "aggregation_strategy": "all",
                "parallel": True,
                "timeout_ms": 30000,
            },
        )
        node = FanoutNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["successful_targets"] == 15
        assert result.metadata["failed_targets"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_fanout_executions(self, execution_context):
        """Test multiple concurrent fanout executions."""
        definition = NodeDefinition(
            id="fanout.concurrent",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["t1", "t2", "t3"],
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        # Run 10 concurrent executions
        messages = [
            Message(
                trace_id=f"concurrent-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Message {i}"),
            )
            for i in range(10)
        ]

        tasks = [node.execute(msg, execution_context) for msg in messages]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_fanout_with_slow_targets(self, execution_context, sample_message):
        """Test fanout with varying target latencies."""
        definition = NodeDefinition(
            id="fanout.slow",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["fast", "medium", "slow"],
                "aggregation_strategy": "all",
                "timeout_ms": 5000,
            },
        )
        node = FanoutNode(definition)

        # Mock with varying delays
        delays = {"fast": 0.01, "medium": 0.1, "slow": 0.3}

        async def mock_execute(target, msg, ctx):
            await asyncio.sleep(delays.get(target, 0))
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=f"Result from {target}"),
                ),
                latency_ms=int(delays.get(target, 0) * 1000),
            )

        node._execute_single_target = mock_execute

        start = time.time()
        result = await node.execute(sample_message, execution_context)
        elapsed = time.time() - start

        assert result.success is True
        # Parallel execution should take ~0.3s (slowest target), not 0.4s (sum)
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_fanout_first_success_early_exit(self, execution_context, sample_message):
        """Test first success strategy exits early with many targets."""
        targets = [f"target_{i}" for i in range(20)]
        definition = NodeDefinition(
            id="fanout.early",
            type=NodeType.FANOUT,
            config={
                "target_nodes": targets,
                "aggregation_strategy": "first_success",
                "parallel": True,
            },
        )
        node = FanoutNode(definition)

        execution_count = [0]

        async def mock_execute(target, msg, ctx):
            execution_count[0] += 1
            # First target succeeds immediately
            if target == "target_0":
                return FanoutTargetResult(
                    target_node=target,
                    success=True,
                    message=msg.create_child(
                        source_node=target,
                        payload=MessagePayload(content="Quick success"),
                    ),
                    latency_ms=1,
                )
            # Others take longer
            await asyncio.sleep(0.5)
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=f"Late {target}"),
                ),
                latency_ms=500,
            )

        node._execute_single_target = mock_execute

        start = time.time()
        result = await node.execute(sample_message, execution_context)
        elapsed = time.time() - start

        assert result.success is True
        # Should complete quickly, not waiting for all 20 targets
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_fanout_high_failure_rate(self, execution_context, sample_message):
        """Test fanout with high failure rate."""
        targets = [f"target_{i}" for i in range(10)]
        definition = NodeDefinition(
            id="fanout.failures",
            type=NodeType.FANOUT,
            config={
                "target_nodes": targets,
                "aggregation_strategy": "all",
            },
        )
        node = FanoutNode(definition)

        async def mock_execute(target, msg, ctx):
            # 70% failure rate
            fail = int(target.split("_")[1]) < 7
            if fail:
                return FanoutTargetResult(
                    target_node=target,
                    success=False,
                    error=f"Simulated failure for {target}",
                    latency_ms=10,
                )
            return FanoutTargetResult(
                target_node=target,
                success=True,
                message=msg.create_child(
                    source_node=target,
                    payload=MessagePayload(content=f"Success from {target}"),
                ),
                latency_ms=10,
            )

        node._execute_single_target = mock_execute

        result = await node.execute(sample_message, execution_context)

        # Should still succeed with partial results
        assert result.success is True
        assert result.metadata["failed_targets"] == 7
        assert result.metadata["successful_targets"] == 3


class TestLoopStress:
    """Stress tests for LoopNode iteration."""

    @pytest.mark.asyncio
    async def test_max_iterations_loop(self, execution_context, sample_message):
        """Test loop running to max iterations."""
        definition = NodeDefinition(
            id="loop.max",
            type=NodeType.LOOP,
            config={
                "body_node": "processor",
                "condition_type": "fixed_count",
                "fixed_count": 100,
                "max_iterations": 100,
                "timeout_ms": 60000,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["loop_result"]["iterations_executed"] == 100

    @pytest.mark.asyncio
    async def test_loop_with_accumulating_state(self, execution_context):
        """Test loop accumulating state over many iterations."""
        definition = NodeDefinition(
            id="loop.accumulate",
            type=NodeType.LOOP,
            config={
                "body_node": "counter",
                "condition_type": "fixed_count",
                "fixed_count": 50,
                "max_iterations": 50,
            },
        )
        node = LoopNode(definition)

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(
                content="0",
                metadata={"counter": 0},
            ),
        )

        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Should have accumulated 50 iterations worth of results
        loop_result = result.metadata["loop_result"]
        assert loop_result["iterations_executed"] == 50

    @pytest.mark.asyncio
    async def test_concurrent_loop_executions(self, execution_context):
        """Test multiple concurrent loop executions."""
        definition = NodeDefinition(
            id="loop.concurrent",
            type=NodeType.LOOP,
            config={
                "body_node": "processor",
                "condition_type": "fixed_count",
                "fixed_count": 10,
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        messages = [
            Message(
                trace_id=f"loop-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Loop input {i}"),
            )
            for i in range(5)
        ]

        tasks = [node.execute(msg, execution_context) for msg in messages]
        results = await asyncio.gather(*tasks)

        assert all(r.success for r in results)
        assert len(results) == 5


class TestTransformStress:
    """Stress tests for TransformNode processing."""

    @pytest.mark.asyncio
    async def test_transform_large_content(self, execution_context):
        """Test transform with very large content."""
        definition = NodeDefinition(
            id="transform.large",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "uppercase"},
                    {"type": "strip"},
                ],
            },
        )
        node = TransformNode(definition)

        # 100KB of content
        large_content = "a" * 100000

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content=large_content),
        )

        result = await node.execute(msg, execution_context)

        assert result.success is True
        assert len(result.output_messages[0].payload.content) == 100000

    @pytest.mark.asyncio
    async def test_transform_many_operations(self, execution_context):
        """Test transform with many chained operations."""
        # Create 20 transform operations
        transforms = [
            {"type": "strip"},
            {"type": "uppercase"},
            {"type": "lowercase"},
            {"type": "strip"},
        ] * 5  # 20 operations

        definition = NodeDefinition(
            id="transform.chain",
            type=NodeType.TRANSFORM,
            config={"transforms": transforms},
        )
        node = TransformNode(definition)

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content="  Test Content  "),
        )

        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Final result after all transforms
        assert result.output_messages[0].payload.content == "test content"

    @pytest.mark.asyncio
    async def test_concurrent_transforms(self, execution_context):
        """Test many concurrent transform executions."""
        definition = NodeDefinition(
            id="transform.concurrent",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "uppercase"},
                    {"type": "strip"},
                ],
            },
        )
        node = TransformNode(definition)

        messages = [
            Message(
                trace_id=f"transform-{i}",
                source_node="test",
                payload=MessagePayload(content=f"  content {i}  "),
            )
            for i in range(50)
        ]

        tasks = [node.execute(msg, execution_context) for msg in messages]
        results = await asyncio.gather(*tasks)

        assert all(r.success for r in results)
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_transform_unicode_content(self, execution_context):
        """Test transform with unicode and special characters."""
        definition = NodeDefinition(
            id="transform.unicode",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "strip"},
                    {"type": "uppercase"},
                ],
            },
        )
        node = TransformNode(definition)

        # Unicode content with emojis and special chars
        unicode_content = "  Hello 世界 🌍 مرحبا  "

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(content=unicode_content),
        )

        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Uppercase works on Unicode
        assert "HELLO" in result.output_messages[0].payload.content


class TestSystemStress:
    """System-wide stress tests."""

    @pytest.mark.asyncio
    async def test_rapid_message_creation(self):
        """Test creating many messages rapidly."""
        messages = []
        for i in range(1000):
            msg = Message(
                trace_id=f"rapid-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Content {i}"),
            )
            messages.append(msg)

        assert len(messages) == 1000
        # All should have unique trace IDs
        trace_ids = {m.trace_id for m in messages}
        assert len(trace_ids) == 1000

    @pytest.mark.asyncio
    async def test_deep_message_hierarchy(self):
        """Test deep parent-child message relationships."""
        root = Message(
            trace_id="root",
            source_node="test",
            payload=MessagePayload(content="Root"),
        )

        current = root
        for i in range(50):
            current = current.create_child(
                source_node=f"node_{i}",
                payload=MessagePayload(content=f"Child {i}"),
            )

        # Should have maintained parent chain
        assert current.parent_id is not None
        # All should share the same trace_id
        assert current.trace_id == "root"

    @pytest.mark.asyncio
    async def test_mixed_workload(self, execution_context):
        """Test mixed workload with different node types."""
        # Create multiple node types
        fanout_def = NodeDefinition(
            id="fanout.mixed",
            type=NodeType.FANOUT,
            config={
                "target_nodes": ["t1", "t2"],
                "aggregation_strategy": "all",
            },
        )
        loop_def = NodeDefinition(
            id="loop.mixed",
            type=NodeType.LOOP,
            config={
                "body_node": "body",
                "condition_type": "fixed_count",
                "fixed_count": 5,
                "max_iterations": 5,
            },
        )
        transform_def = NodeDefinition(
            id="transform.mixed",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )

        fanout = FanoutNode(fanout_def)
        loop = LoopNode(loop_def)
        transform = TransformNode(transform_def)

        # Run all concurrently
        msg = Message(
            trace_id="mixed",
            source_node="test",
            payload=MessagePayload(content="Mixed workload test"),
        )

        results = await asyncio.gather(
            fanout.execute(msg, execution_context),
            loop.execute(msg, execution_context),
            transform.execute(msg, execution_context),
        )

        # All should succeed
        assert all(r.success for r in results)


class TestMemoryPressure:
    """Tests for memory pressure scenarios."""

    @pytest.mark.asyncio
    async def test_large_metadata(self, execution_context):
        """Test handling large metadata objects."""
        # Create message with large metadata
        large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        msg = Message(
            trace_id="test",
            source_node="test",
            payload=MessagePayload(
                content="Test",
                metadata=large_metadata,
            ),
        )

        definition = NodeDefinition(
            id="transform.metadata",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "strip"}],
            },
        )
        node = TransformNode(definition)

        result = await node.execute(msg, execution_context)

        assert result.success is True
        # Metadata should be preserved
        output_metadata = result.output_messages[0].payload.metadata
        assert "transformed" in output_metadata

    @pytest.mark.asyncio
    async def test_many_node_results(self, execution_context):
        """Test collecting many node results."""
        from tinyllm.core.node import NodeResult

        results = []
        for i in range(100):
            msg = Message(
                trace_id=f"result-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Result {i}"),
            )
            result = NodeResult.success_result(
                output_messages=[msg],
                next_nodes=[],
                metadata={"index": i},
            )
            results.append(result)

        assert len(results) == 100
        assert all(r.success for r in results)
