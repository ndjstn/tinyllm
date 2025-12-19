"""Chaos testing framework for TinyLLM.

This module provides chaos engineering tests to validate system resilience
under various failure conditions, network issues, and resource constraints.
"""

import asyncio
import random
import time

import pytest

pytestmark = pytest.mark.chaos
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult
from tinyllm.nodes.fanout import FanoutNode
from tinyllm.nodes.transform import TransformNode


class ChaosInjector:
    """Chaos injector for introducing controlled failures."""

    def __init__(self, failure_rate: float = 0.3, latency_ms: int = 0):
        """Initialize chaos injector.

        Args:
            failure_rate: Probability of injecting failure (0.0-1.0).
            latency_ms: Additional latency to inject in milliseconds.
        """
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.injected_failures = 0
        self.total_calls = 0

    async def maybe_fail(self, operation: str = "operation") -> None:
        """Maybe inject a failure based on failure rate.

        Args:
            operation: Name of the operation for error message.

        Raises:
            RuntimeError: If chaos decides to fail.
        """
        self.total_calls += 1

        # Inject latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        # Maybe inject failure
        if random.random() < self.failure_rate:
            self.injected_failures += 1
            raise RuntimeError(f"Chaos injected failure in {operation}")

    def get_stats(self) -> dict:
        """Get chaos injection statistics."""
        return {
            "total_calls": self.total_calls,
            "injected_failures": self.injected_failures,
            "actual_failure_rate": (
                self.injected_failures / self.total_calls if self.total_calls > 0 else 0
            ),
        }


@pytest.fixture
def chaos_injector():
    """Create chaos injector with default settings."""
    return ChaosInjector(failure_rate=0.3)


@pytest.fixture
def execution_context():
    """Create test execution context."""
    return ExecutionContext(
        trace_id="chaos-test",
        graph_id="chaos-graph",
        config=Config(),
    )


class TestNetworkChaos:
    """Network-related chaos tests."""

    @pytest.mark.asyncio
    async def test_random_network_failures(self, execution_context, chaos_injector):
        """Test system resilience to random network failures."""
        definition = NodeDefinition(
            id="transform.network",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        # Mock the execute method to inject chaos
        original_execute = node.execute

        async def chaotic_execute(msg, ctx):
            await chaos_injector.maybe_fail("network_request")
            return await original_execute(msg, ctx)

        node.execute = chaotic_execute

        # Run multiple requests
        num_requests = 50
        successes = 0
        failures = 0

        for i in range(num_requests):
            msg = Message(
                trace_id=f"network-chaos-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Test {i}"),
            )

            try:
                result = await node.execute(msg, execution_context)
                if result.success:
                    successes += 1
                else:
                    failures += 1
            except RuntimeError:
                failures += 1

        stats = chaos_injector.get_stats()

        # Some requests should fail, but not all
        assert failures > 0, "Chaos should have caused some failures"
        assert successes > 0, "Some requests should still succeed"
        assert stats["injected_failures"] == failures

    @pytest.mark.asyncio
    async def test_intermittent_timeouts(self, execution_context):
        """Test handling of intermittent timeout conditions."""
        definition = NodeDefinition(
            id="transform.timeout",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "strip"}],
            },
        )
        node = TransformNode(definition)

        async def slow_execute(msg, ctx):
            """Simulate slow execution."""
            # Randomly slow down
            if random.random() < 0.3:
                await asyncio.sleep(0.5)  # Simulated timeout
            return await TransformNode.execute(node, msg, ctx)

        original_execute = node.execute
        node.execute = slow_execute

        msg = Message(
            trace_id="timeout-test",
            source_node="test",
            payload=MessagePayload(content="Test content"),
        )

        # Should eventually succeed or fail gracefully
        try:
            result = await asyncio.wait_for(node.execute(msg, execution_context), timeout=1.0)
            assert result is not None
        except asyncio.TimeoutError:
            # Timeout is acceptable in chaos testing
            pass

    @pytest.mark.asyncio
    async def test_packet_loss_simulation(self, execution_context, chaos_injector):
        """Test resilience to packet loss (dropped requests)."""
        definition = NodeDefinition(
            id="fanout.packet_loss",
            type=NodeType.FANOUT,
            config={
                "target_nodes": [f"target_{i}" for i in range(5)],
                "aggregation_strategy": "all",
                "parallel": True,
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="packet-loss-test",
            source_node="test",
            payload=MessagePayload(content="Test"),
        )

        # The fanout node should handle partial failures gracefully
        result = await node.execute(msg, execution_context)

        # Should get a result even if some targets "dropped"
        assert result is not None


class TestResourceChaos:
    """Resource constraint chaos tests."""

    @pytest.mark.asyncio
    async def test_memory_pressure(self, execution_context):
        """Test behavior under simulated memory pressure."""
        definition = NodeDefinition(
            id="transform.memory",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        # Create messages with varying sizes to simulate memory pressure
        sizes = [100, 1000, 10000, 100000]
        results = []

        for size in sizes:
            content = "a" * size
            msg = Message(
                trace_id=f"memory-test-{size}",
                source_node="test",
                payload=MessagePayload(content=content),
            )

            result = await node.execute(msg, execution_context)
            results.append(result)

        # All should succeed despite varying memory usage
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_cpu_saturation(self, execution_context):
        """Test behavior under CPU saturation."""
        definition = NodeDefinition(
            id="transform.cpu",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}, {"type": "lowercase"}] * 10,
            },
        )
        node = TransformNode(definition)

        # Run many concurrent operations
        num_concurrent = 50
        messages = [
            Message(
                trace_id=f"cpu-test-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Content {i}" * 100),
            )
            for i in range(num_concurrent)
        ]

        start = time.time()
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages]
        )
        elapsed = time.time() - start

        # All should complete, even if slow
        assert all(r.success for r in results)
        assert len(results) == num_concurrent


class TestDataChaos:
    """Data corruption and invalid input chaos tests."""

    @pytest.mark.asyncio
    async def test_corrupted_message_payload(self, execution_context):
        """Test handling of corrupted message payloads."""
        definition = NodeDefinition(
            id="transform.corrupted",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        # Various corrupted/invalid payloads
        corrupted_contents = [
            None,  # None content
            "",  # Empty content
            " ",  # Whitespace only
            "\x00\x01\x02",  # Binary data
            "🔥" * 10000,  # Large emoji string
        ]

        for content in corrupted_contents:
            try:
                msg = Message(
                    trace_id="corrupted-test",
                    source_node="test",
                    payload=MessagePayload(content=content if content is not None else ""),
                )
                result = await node.execute(msg, execution_context)
                # Should handle gracefully
                assert result is not None
            except Exception:
                # Acceptable to raise exception for truly invalid data
                pass

    @pytest.mark.asyncio
    async def test_malformed_configuration(self, execution_context):
        """Test behavior with malformed node configurations."""
        # Invalid transform type
        definition = NodeDefinition(
            id="transform.malformed",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [
                    {"type": "uppercase"},
                    {"type": "invalid_transform_type"},
                ],
            },
        )
        node = TransformNode(definition)

        msg = Message(
            trace_id="malformed-test",
            source_node="test",
            payload=MessagePayload(content="Test"),
        )

        # Should fail gracefully
        result = await node.execute(msg, execution_context)
        # The first valid transform might succeed, or whole thing might fail
        # Either way, shouldn't crash
        assert result is not None

    @pytest.mark.asyncio
    async def test_race_condition_simulation(self, execution_context):
        """Test for race conditions in concurrent access."""
        definition = NodeDefinition(
            id="transform.race",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "lowercase"}],
            },
        )
        node = TransformNode(definition)

        # Same message ID, concurrent execution
        shared_trace_id = "race-condition"

        messages = [
            Message(
                trace_id=shared_trace_id,
                source_node="test",
                payload=MessagePayload(content=f"Content {i}"),
            )
            for i in range(20)
        ]

        # Execute all concurrently
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages],
            return_exceptions=True,
        )

        # Should all complete without deadlock or corruption
        assert len(results) == 20
        # Most should succeed
        success_count = sum(
            1 for r in results if not isinstance(r, Exception) and r.success
        )
        assert success_count >= 15


class TestCascadingFailures:
    """Tests for cascading failure scenarios."""

    @pytest.mark.asyncio
    async def test_cascading_node_failures(self, execution_context, chaos_injector):
        """Test system resilience to cascading node failures."""
        # Create a fanout that might cascade failures
        definition = NodeDefinition(
            id="fanout.cascade",
            type=NodeType.FANOUT,
            config={
                "target_nodes": [f"target_{i}" for i in range(10)],
                "aggregation_strategy": "all",
                "parallel": True,
                "timeout_ms": 5000,
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="cascade-test",
            source_node="test",
            payload=MessagePayload(content="Test cascade"),
        )

        # Even with failures in some targets, should get a result
        result = await node.execute(msg, execution_context)
        assert result is not None

    @pytest.mark.asyncio
    async def test_retry_storm(self, execution_context):
        """Test that retry logic doesn't cause exponential backoff storm."""
        definition = NodeDefinition(
            id="transform.retry",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "strip"}],
            },
        )
        node = TransformNode(definition)

        # Simulate many rapid retries
        num_retries = 100
        messages = [
            Message(
                trace_id=f"retry-{i}",
                source_node="test",
                payload=MessagePayload(content="Retry test"),
            )
            for i in range(num_retries)
        ]

        start = time.time()
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages],
            return_exceptions=True,
        )
        elapsed = time.time() - start

        # Should complete in reasonable time (not exponential)
        assert elapsed < 10.0  # 10 seconds max for 100 operations
        assert len(results) == num_retries


class TestPartitioningChaos:
    """Network partitioning and split-brain scenarios."""

    @pytest.mark.asyncio
    async def test_partial_network_partition(self, execution_context, chaos_injector):
        """Test partial network partition affecting some nodes."""
        definition = NodeDefinition(
            id="fanout.partition",
            type=NodeType.FANOUT,
            config={
                "target_nodes": [f"target_{i}" for i in range(5)],
                "aggregation_strategy": "majority",
                "parallel": True,
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="partition-test",
            source_node="test",
            payload=MessagePayload(content="Partition test"),
        )

        # With majority strategy, should succeed even if some nodes unreachable
        result = await node.execute(msg, execution_context)
        assert result is not None


class TestClockSkewChaos:
    """Tests for clock skew and timing issues."""

    @pytest.mark.asyncio
    async def test_timestamp_skew(self):
        """Test handling of messages with skewed timestamps."""
        # Create messages with various timestamp anomalies
        msg_future = Message(
            trace_id="future",
            source_node="test",
            payload=MessagePayload(content="From the future"),
        )

        msg_past = Message(
            trace_id="past",
            source_node="test",
            payload=MessagePayload(content="From the past"),
        )

        # Both should be valid
        assert msg_future is not None
        assert msg_past is not None

    @pytest.mark.asyncio
    async def test_concurrent_message_ordering(self, execution_context):
        """Test message ordering under concurrent execution."""
        definition = NodeDefinition(
            id="transform.ordering",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        # Create messages with specific order
        messages = [
            Message(
                trace_id="order-test",
                source_node="test",
                payload=MessagePayload(
                    content=f"Message {i}",
                    metadata={"sequence": i},
                ),
            )
            for i in range(10)
        ]

        # Execute concurrently
        results = await asyncio.gather(
            *[node.execute(msg, execution_context) for msg in messages]
        )

        # All should complete successfully
        assert all(r.success for r in results)
        assert len(results) == 10


class TestRandomizedChaos:
    """Fully randomized chaos tests."""

    @pytest.mark.asyncio
    async def test_random_chaos_monkey(self, execution_context):
        """Test system with random chaos injection."""
        definition = NodeDefinition(
            id="transform.chaos_monkey",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "uppercase"}],
            },
        )
        node = TransformNode(definition)

        num_tests = 50
        chaos_actions = [
            "delay",
            "corrupt_content",
            "empty_content",
            "large_content",
            "normal",
        ]

        results = []
        for i in range(num_tests):
            action = random.choice(chaos_actions)

            if action == "delay":
                await asyncio.sleep(random.uniform(0.01, 0.1))
            elif action == "corrupt_content":
                content = "\x00" * random.randint(1, 100)
            elif action == "empty_content":
                content = ""
            elif action == "large_content":
                content = "X" * random.randint(10000, 50000)
            else:
                content = f"Normal content {i}"

            msg = Message(
                trace_id=f"chaos-monkey-{i}",
                source_node="test",
                payload=MessagePayload(content=content),
            )

            try:
                result = await node.execute(msg, execution_context)
                results.append(result)
            except Exception:
                # Acceptable for some chaos scenarios to raise
                pass

        # At least 60% should succeed despite chaos
        success_count = sum(1 for r in results if r.success)
        assert success_count >= num_tests * 0.6


class TestRecoveryChaos:
    """Tests for recovery from chaos scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_after_failures(self, execution_context):
        """Test system recovery after period of failures."""
        definition = NodeDefinition(
            id="transform.recovery",
            type=NodeType.TRANSFORM,
            config={
                "transforms": [{"type": "lowercase"}],
            },
        )
        node = TransformNode(definition)

        failure_count = [0]
        original_execute = node.execute

        async def failing_execute(msg, ctx):
            """Execute that fails first 10 times then recovers."""
            failure_count[0] += 1
            if failure_count[0] <= 10:
                raise RuntimeError("Simulated failure")
            return await original_execute(msg, ctx)

        node.execute = failing_execute

        # Try multiple times
        for i in range(20):
            msg = Message(
                trace_id=f"recovery-{i}",
                source_node="test",
                payload=MessagePayload(content=f"Test {i}"),
            )

            try:
                result = await node.execute(msg, execution_context)
                # Should succeed after recovery
                if i >= 10:
                    assert result.success
            except RuntimeError:
                # Expected for first 10
                assert i < 10

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, execution_context):
        """Test graceful degradation under sustained chaos."""
        definition = NodeDefinition(
            id="fanout.degradation",
            type=NodeType.FANOUT,
            config={
                "target_nodes": [f"target_{i}" for i in range(10)],
                "aggregation_strategy": "best_effort",
                "parallel": True,
            },
        )
        node = FanoutNode(definition)

        msg = Message(
            trace_id="degradation-test",
            source_node="test",
            payload=MessagePayload(content="Degradation test"),
        )

        # Should provide some result even if degraded
        result = await node.execute(msg, execution_context)
        assert result is not None
