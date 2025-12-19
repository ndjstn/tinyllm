"""Tests for TimeoutNode implementation."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.nodes.timeout import (
    TimeoutNode,
    TimeoutConfig,
    TimeoutAction,
    TimeoutMetrics,
)


@pytest.fixture
def execution_context():
    """Create a test execution context."""
    return ExecutionContext(
        trace_id="test-trace-timeout",
        graph_id="test-graph",
        config=Config(),
    )


@pytest.fixture
def sample_message():
    """Create a sample test message."""
    return Message(
        trace_id="test-trace-timeout",
        source_node="test",
        payload=MessagePayload(
            task="Process with timeout",
            content="test data",
        ),
    )


class TestTimeoutNodeRegistration:
    """Tests for TimeoutNode registration."""

    def test_timeout_node_registered(self):
        """Test TimeoutNode is registered in registry."""
        assert NodeRegistry.is_registered(NodeType.TIMEOUT)

    def test_create_timeout_node_from_registry(self):
        """Test creating TimeoutNode from registry."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
                "on_timeout": "error",
            },
        )
        node = NodeRegistry.create(definition)
        assert node is not None
        assert isinstance(node, TimeoutNode)
        assert node.id == "timeout.test"


class TestTimeoutConfig:
    """Tests for TimeoutConfig validation."""

    def test_timeout_config_defaults(self):
        """Test TimeoutConfig default values."""
        config = TimeoutConfig(inner_node="test.node")
        assert config.inner_node == "test.node"
        assert config.timeout_ms == 30000
        assert config.on_timeout == TimeoutAction.ERROR
        assert config.retry_count == 0
        assert config.retry_delay_ms == 1000
        assert config.propagate_metadata is True
        assert config.fallback_response is None

    def test_timeout_config_frozen(self):
        """Test TimeoutConfig is frozen."""
        config = TimeoutConfig(inner_node="test.node", timeout_ms=5000)
        with pytest.raises(Exception):  # Pydantic ValidationError
            config.timeout_ms = 10000

    def test_timeout_config_strict_validation(self):
        """Test TimeoutConfig strict validation."""
        # Invalid extra field should fail
        with pytest.raises(Exception):
            TimeoutConfig(inner_node="test.node", invalid_field="value")

    def test_timeout_config_timeout_bounds(self):
        """Test timeout_ms boundary validation."""
        # Valid bounds
        config = TimeoutConfig(inner_node="test.node", timeout_ms=100)
        assert config.timeout_ms == 100

        config = TimeoutConfig(inner_node="test.node", timeout_ms=300000)
        assert config.timeout_ms == 300000

        # Invalid bounds should fail
        with pytest.raises(Exception):
            TimeoutConfig(inner_node="test.node", timeout_ms=50)

        with pytest.raises(Exception):
            TimeoutConfig(inner_node="test.node", timeout_ms=400000)

    def test_timeout_config_retry_bounds(self):
        """Test retry_count boundary validation."""
        # Valid bounds
        config = TimeoutConfig(inner_node="test.node", retry_count=0)
        assert config.retry_count == 0

        config = TimeoutConfig(inner_node="test.node", retry_count=10)
        assert config.retry_count == 10

        # Invalid bounds should fail
        with pytest.raises(Exception):
            TimeoutConfig(inner_node="test.node", retry_count=-1)

        with pytest.raises(Exception):
            TimeoutConfig(inner_node="test.node", retry_count=11)

    def test_timeout_config_actions(self):
        """Test timeout action types."""
        # Error action
        config = TimeoutConfig(inner_node="test.node", on_timeout="error")
        assert config.on_timeout == TimeoutAction.ERROR

        # Fallback action
        config = TimeoutConfig(
            inner_node="test.node",
            on_timeout="fallback",
            fallback_response="Fallback response",
        )
        assert config.on_timeout == TimeoutAction.FALLBACK

        # Skip action
        config = TimeoutConfig(inner_node="test.node", on_timeout="skip")
        assert config.on_timeout == TimeoutAction.SKIP


class TestTimeoutNodeBasicExecution:
    """Tests for basic TimeoutNode execution."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_timeout(
        self, sample_message, execution_context
    ):
        """Test successful execution without timeout."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
                "on_timeout": "error",
            },
        )
        node = TimeoutNode(definition)

        # Mock successful fast execution
        async def fast_execution(msg, ctx):
            await asyncio.sleep(0.01)  # 10ms
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
                metadata={"processed": True},
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=fast_execution
        ):
            result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.error is None
        assert "timeout" in result.metadata
        assert result.metadata["timeout"]["timed_out"] is False
        assert node.metrics.successful_executions == 1
        assert node.metrics.timeouts_triggered == 0

    @pytest.mark.asyncio
    async def test_validation_error_no_inner_node(
        self, sample_message, execution_context
    ):
        """Test validation error when inner_node is missing."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "timeout_ms": 5000,
                "on_timeout": "error",
            },
        )

        with pytest.raises(Exception):  # Pydantic validation should fail
            TimeoutNode(definition)

    @pytest.mark.asyncio
    async def test_validation_error_fallback_without_response(
        self, sample_message, execution_context
    ):
        """Test validation error when fallback action has no response."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
                "on_timeout": "fallback",
                # Missing fallback_response
            },
        )
        node = TimeoutNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is False
        assert "fallback_response" in result.error


class TestTimeoutNodeTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_with_error_action(self, sample_message, execution_context):
        """Test timeout handling with error action."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 100,  # Very short timeout
                "on_timeout": "error",
            },
        )
        node = TimeoutNode(definition)

        # Mock slow execution
        async def slow_execution(msg, ctx):
            await asyncio.sleep(1.0)  # 1 second - will timeout
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=slow_execution
        ):
            result = await node.execute(sample_message, execution_context)

        assert result.success is False
        assert "Timeout" in result.error
        assert "100ms" in result.error
        assert node.metrics.timeouts_triggered == 1
        assert node.metrics.successful_executions == 0

    @pytest.mark.asyncio
    async def test_timeout_with_fallback_action(
        self, sample_message, execution_context
    ):
        """Test timeout handling with fallback action."""
        fallback_msg = "This is a fallback response"
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 100,
                "on_timeout": "fallback",
                "fallback_response": fallback_msg,
            },
        )
        node = TimeoutNode(definition)

        # Mock slow execution
        async def slow_execution(msg, ctx):
            await asyncio.sleep(1.0)
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=slow_execution
        ):
            result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.error is None
        assert len(result.output_messages) == 1
        assert result.output_messages[0].payload.content == fallback_msg
        assert result.output_messages[0].payload.metadata["fallback"] is True
        assert node.metrics.fallback_used == 1
        assert node.metrics.timeouts_triggered == 1

    @pytest.mark.asyncio
    async def test_timeout_with_skip_action(self, sample_message, execution_context):
        """Test timeout handling with skip action."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 100,
                "on_timeout": "skip",
            },
        )
        node = TimeoutNode(definition)

        # Mock slow execution
        async def slow_execution(msg, ctx):
            await asyncio.sleep(1.0)
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=slow_execution
        ):
            result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.error is None
        assert len(result.output_messages) == 0  # Empty result
        assert result.metadata["skipped"] is True
        assert node.metrics.skipped == 1
        assert node.metrics.timeouts_triggered == 1


class TestTimeoutNodeRetryBehavior:
    """Tests for retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, sample_message, execution_context):
        """Test retry behavior on timeout."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 100,
                "on_timeout": "error",
                "retry_count": 2,
                "retry_delay_ms": 10,
            },
        )
        node = TimeoutNode(definition)

        attempt_count = 0

        async def slow_execution(msg, ctx):
            nonlocal attempt_count
            attempt_count += 1
            await asyncio.sleep(1.0)  # Always timeout
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=slow_execution
        ):
            result = await node.execute(sample_message, execution_context)

        # Should have tried 1 initial + 2 retries = 3 times
        assert attempt_count == 3
        assert result.success is False
        assert node.metrics.total_attempts == 3
        assert node.metrics.timeouts_triggered == 3

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(
        self, sample_message, execution_context
    ):
        """Test successful retry on second attempt."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 200,
                "on_timeout": "error",
                "retry_count": 2,
                "retry_delay_ms": 10,
            },
        )
        node = TimeoutNode(definition)

        attempt_count = 0

        async def intermittent_execution(msg, ctx):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                await asyncio.sleep(1.0)  # First attempt times out
            else:
                await asyncio.sleep(0.01)  # Second attempt succeeds
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=intermittent_execution
        ):
            result = await node.execute(sample_message, execution_context)

        # Should succeed on second attempt
        assert attempt_count == 2
        assert result.success is True
        assert node.metrics.total_attempts == 2
        assert node.metrics.timeouts_triggered == 1
        assert node.metrics.successful_executions == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_non_timeout_error(
        self, sample_message, execution_context
    ):
        """Test no retry on non-timeout errors."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
                "on_timeout": "error",
                "retry_count": 3,
            },
        )
        node = TimeoutNode(definition)

        attempt_count = 0

        async def error_execution(msg, ctx):
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Something went wrong")

        with patch.object(
            node, "_simulate_inner_execution", side_effect=error_execution
        ):
            result = await node.execute(sample_message, execution_context)

        # Should only try once, no retries on non-timeout errors
        assert attempt_count == 1
        assert result.success is False
        assert "Something went wrong" in result.error
        assert node.metrics.total_attempts == 1


class TestTimeoutNodeMetrics:
    """Tests for timeout metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self, sample_message, execution_context):
        """Test metrics are updated on successful execution."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
            },
        )
        node = TimeoutNode(definition)

        async def fast_execution(msg, ctx):
            await asyncio.sleep(0.01)
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=fast_execution
        ):
            # Execute multiple times
            for _ in range(3):
                await node.execute(sample_message, execution_context)

        metrics = node.metrics
        assert metrics.total_attempts == 3
        assert metrics.successful_executions == 3
        assert metrics.timeouts_triggered == 0
        assert metrics.fallback_used == 0
        assert metrics.skipped == 0
        assert metrics.total_elapsed_ms > 0
        assert metrics.avg_execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_metrics_with_mixed_results(self, sample_message, execution_context):
        """Test metrics with mixed success/timeout results."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 100,
                "on_timeout": "fallback",
                "fallback_response": "Fallback",
            },
        )
        node = TimeoutNode(definition)

        call_count = 0

        async def mixed_execution(msg, ctx):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                await asyncio.sleep(1.0)  # Timeout
            else:
                await asyncio.sleep(0.01)  # Success
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=mixed_execution
        ):
            # Execute 4 times: 2 successes, 2 timeouts
            for _ in range(4):
                await node.execute(sample_message, execution_context)

        metrics = node.metrics
        assert metrics.total_attempts == 4
        assert metrics.successful_executions == 2
        assert metrics.timeouts_triggered == 2
        assert metrics.fallback_used == 2


class TestTimeoutNodeMetadataPropagation:
    """Tests for metadata propagation."""

    @pytest.mark.asyncio
    async def test_metadata_propagation_enabled(
        self, sample_message, execution_context
    ):
        """Test timeout metadata is propagated when enabled."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
                "propagate_metadata": True,
            },
        )
        node = TimeoutNode(definition)

        async def fast_execution(msg, ctx):
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=fast_execution
        ):
            result = await node.execute(sample_message, execution_context)

        assert "timeout" in result.metadata
        assert "timeout_ms" in result.metadata["timeout"]
        assert "elapsed_ms" in result.metadata["timeout"]
        assert "timed_out" in result.metadata["timeout"]

    @pytest.mark.asyncio
    async def test_metadata_propagation_disabled(
        self, sample_message, execution_context
    ):
        """Test timeout metadata is not propagated when disabled."""
        definition = NodeDefinition(
            id="timeout.test",
            type=NodeType.TIMEOUT,
            config={
                "inner_node": "model.process",
                "timeout_ms": 5000,
                "propagate_metadata": False,
            },
        )
        node = TimeoutNode(definition)

        async def fast_execution(msg, ctx):
            return NodeResult.success_result(
                output_messages=[sample_message],
                next_nodes=[],
            )

        with patch.object(
            node, "_simulate_inner_execution", side_effect=fast_execution
        ):
            result = await node.execute(sample_message, execution_context)

        assert "timeout" not in result.metadata


class TestTimeoutMetricsModel:
    """Tests for TimeoutMetrics model."""

    def test_timeout_metrics_defaults(self):
        """Test TimeoutMetrics default values."""
        metrics = TimeoutMetrics()
        assert metrics.total_attempts == 0
        assert metrics.timeouts_triggered == 0
        assert metrics.successful_executions == 0
        assert metrics.fallback_used == 0
        assert metrics.skipped == 0
        assert metrics.total_elapsed_ms == 0
        assert metrics.avg_execution_time_ms == 0.0

    def test_timeout_metrics_frozen(self):
        """Test TimeoutMetrics is frozen."""
        metrics = TimeoutMetrics(total_attempts=5)
        with pytest.raises(Exception):  # Pydantic ValidationError
            metrics.total_attempts = 10

    def test_timeout_metrics_strict_validation(self):
        """Test TimeoutMetrics strict validation."""
        # Invalid extra field should fail
        with pytest.raises(Exception):
            TimeoutMetrics(invalid_field="value")
