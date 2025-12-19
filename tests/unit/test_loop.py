"""Tests for LoopNode implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.registry import NodeRegistry
from tinyllm.nodes.loop import (
    LoopNode,
    LoopCondition,
    LoopConfig,
    LoopState,
    LoopResult,
)


@pytest.fixture
def execution_context():
    """Create a test execution context."""
    return ExecutionContext(
        trace_id="test-trace-loop",
        graph_id="test-graph",
        config=Config(),
    )


@pytest.fixture
def sample_message():
    """Create a sample test message."""
    return Message(
        trace_id="test-trace-loop",
        source_node="test",
        payload=MessagePayload(
            task="Process data iteratively",
            content="test data",
        ),
    )


class TestLoopNodeRegistration:
    """Tests for LoopNode registration."""

    def test_loop_node_registered(self):
        """Test LoopNode is registered in registry."""
        assert NodeRegistry.is_registered(NodeType.LOOP)

    def test_create_loop_node_from_registry(self):
        """Test creating LoopNode from registry."""
        definition = NodeDefinition(
            id="loop.test",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 5,
            },
        )
        node = NodeRegistry.create(definition)
        assert node is not None
        assert isinstance(node, LoopNode)
        assert node.id == "loop.test"


class TestLoopConfig:
    """Tests for LoopConfig validation."""

    def test_loop_config_defaults(self):
        """Test LoopConfig default values."""
        config = LoopConfig(body_node="test.node")
        assert config.body_node == "test.node"
        assert config.condition_type == LoopCondition.FIXED_COUNT
        assert config.max_iterations == 10
        assert config.timeout_ms == 60000
        assert config.collect_results is True
        assert config.continue_on_error is False

    def test_loop_config_frozen(self):
        """Test LoopConfig is frozen."""
        config = LoopConfig(body_node="test.node", max_iterations=5)
        with pytest.raises(Exception):  # Pydantic ValidationError
            config.max_iterations = 10

    def test_loop_config_strict_validation(self):
        """Test LoopConfig strict validation."""
        # Invalid extra field should fail
        with pytest.raises(Exception):
            LoopConfig(body_node="test.node", invalid_field="value")

    def test_loop_config_max_iterations_bounds(self):
        """Test max_iterations boundary validation."""
        # Valid bounds
        config = LoopConfig(body_node="test.node", max_iterations=1)
        assert config.max_iterations == 1

        config = LoopConfig(body_node="test.node", max_iterations=1000)
        assert config.max_iterations == 1000

        # Invalid bounds
        with pytest.raises(Exception):
            LoopConfig(body_node="test.node", max_iterations=0)

        with pytest.raises(Exception):
            LoopConfig(body_node="test.node", max_iterations=1001)

    def test_loop_config_timeout_bounds(self):
        """Test timeout_ms boundary validation."""
        # Valid bounds
        config = LoopConfig(body_node="test.node", timeout_ms=1000)
        assert config.timeout_ms == 1000

        config = LoopConfig(body_node="test.node", timeout_ms=300000)
        assert config.timeout_ms == 300000

        # Invalid bounds
        with pytest.raises(Exception):
            LoopConfig(body_node="test.node", timeout_ms=999)

        with pytest.raises(Exception):
            LoopConfig(body_node="test.node", timeout_ms=300001)


class TestLoopConditionEnum:
    """Tests for LoopCondition enum."""

    def test_loop_condition_values(self):
        """Test all LoopCondition enum values."""
        assert LoopCondition.FIXED_COUNT == "fixed_count"
        assert LoopCondition.UNTIL_SUCCESS == "until_success"
        assert LoopCondition.UNTIL_CONDITION == "until_condition"
        assert LoopCondition.WHILE_CONDITION == "while_condition"

    def test_loop_condition_in_config(self):
        """Test using LoopCondition enum in config."""
        for condition in LoopCondition:
            config = LoopConfig(body_node="test.node", condition_type=condition)
            assert config.condition_type == condition


class TestLoopState:
    """Tests for LoopState model."""

    def test_loop_state_defaults(self):
        """Test LoopState default values."""
        state = LoopState()
        assert state.iteration_count == 0
        assert state.accumulated_results == []
        assert state.elapsed_time_ms == 0
        assert state.success_count == 0
        assert state.failure_count == 0
        assert state.last_result is None
        assert state.terminated_by is None

    def test_loop_state_frozen(self):
        """Test LoopState is frozen."""
        state = LoopState(iteration_count=5)
        with pytest.raises(Exception):
            state.iteration_count = 10

    def test_loop_state_with_results(self):
        """Test LoopState with accumulated results."""
        results = [
            {"iteration": 1, "success": True, "output": "result1"},
            {"iteration": 2, "success": False, "error": "failed"},
            {"iteration": 3, "success": True, "output": "result3"},
        ]
        state = LoopState(
            iteration_count=3,
            accumulated_results=results,
            success_count=2,
            failure_count=1,
            elapsed_time_ms=1500,
            terminated_by="fixed_count_reached",
        )
        assert state.iteration_count == 3
        assert len(state.accumulated_results) == 3
        assert state.success_count == 2
        assert state.failure_count == 1


class TestLoopResult:
    """Tests for LoopResult model."""

    def test_loop_result_creation(self):
        """Test creating LoopResult."""
        result = LoopResult(
            success=True,
            iterations_executed=5,
            all_iterations=[],
            final_output="Final result",
            termination_reason="fixed_count_reached",
            total_elapsed_ms=2500,
            success_rate=1.0,
        )
        assert result.success is True
        assert result.iterations_executed == 5
        assert result.termination_reason == "fixed_count_reached"
        assert result.success_rate == 1.0

    def test_loop_result_frozen(self):
        """Test LoopResult is frozen."""
        result = LoopResult(
            success=True,
            iterations_executed=5,
            all_iterations=[],
            termination_reason="completed",
            total_elapsed_ms=1000,
            success_rate=1.0,
        )
        with pytest.raises(Exception):
            result.success = False


class TestLoopNodeCreation:
    """Tests for LoopNode instantiation."""

    def test_create_fixed_count_loop(self):
        """Test creating a fixed count loop node."""
        definition = NodeDefinition(
            id="loop.fixed",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 5,
            },
        )
        node = LoopNode(definition)
        assert node.id == "loop.fixed"
        assert node.loop_config.condition_type == LoopCondition.FIXED_COUNT
        assert node.loop_config.fixed_count == 5

    def test_create_until_success_loop(self):
        """Test creating an until-success loop node."""
        definition = NodeDefinition(
            id="loop.retry",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_success",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)
        assert node.loop_config.condition_type == LoopCondition.UNTIL_SUCCESS
        assert node.loop_config.max_iterations == 10

    def test_create_condition_loop(self):
        """Test creating a condition-based loop node."""
        definition = NodeDefinition(
            id="loop.condition",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "success_count >= 3",
                "max_iterations": 20,
            },
        )
        node = LoopNode(definition)
        assert node.loop_config.condition_type == LoopCondition.UNTIL_CONDITION
        assert node.loop_config.condition_expression == "success_count >= 3"


class TestFixedCountLoop:
    """Tests for FIXED_COUNT loop execution."""

    @pytest.mark.asyncio
    async def test_fixed_count_basic(self, sample_message, execution_context):
        """Test basic fixed count loop execution."""
        definition = NodeDefinition(
            id="loop.fixed",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 3,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["iterations"] == 3
        assert result.metadata["loop_result"]["termination_reason"] == "fixed_count_reached"

    @pytest.mark.asyncio
    async def test_fixed_count_with_max_iterations(self, sample_message, execution_context):
        """Test fixed count respects max_iterations limit."""
        definition = NodeDefinition(
            id="loop.fixed",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 100,
                "max_iterations": 50,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        # Should fail validation because fixed_count > max_iterations
        assert result.success is False
        assert "exceeds max_iterations" in result.error

    @pytest.mark.asyncio
    async def test_fixed_count_missing_parameter(self, sample_message, execution_context):
        """Test fixed count fails without fixed_count parameter."""
        definition = NodeDefinition(
            id="loop.fixed",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                # Missing fixed_count
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is False
        assert "requires 'fixed_count' parameter" in result.error


class TestUntilSuccessLoop:
    """Tests for UNTIL_SUCCESS loop execution."""

    @pytest.mark.asyncio
    async def test_until_success_immediate(self, sample_message, execution_context):
        """Test until success with immediate success."""
        definition = NodeDefinition(
            id="loop.retry",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_success",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        # Mock successful execution
        with patch.object(node, "_simulate_body_execution", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": True, "output": "Success!"}

            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 1
            assert result.metadata["loop_result"]["termination_reason"] == "success_achieved"

    @pytest.mark.asyncio
    async def test_until_success_with_retries(self, sample_message, execution_context):
        """Test until success with multiple retries."""
        definition = NodeDefinition(
            id="loop.retry",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_success",
                "max_iterations": 5,
                "continue_on_error": True,
            },
        )
        node = LoopNode(definition)

        # Mock failures then success
        call_count = 0

        async def mock_body_exec(message, context):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {"success": False, "error": "Failed"}
            return {"success": True, "output": "Success!"}

        with patch.object(node, "_simulate_body_execution", side_effect=mock_body_exec):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 3
            assert result.metadata["success_rate"] == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_until_success_max_iterations_reached(
        self, sample_message, execution_context
    ):
        """Test until success stops at max_iterations."""
        definition = NodeDefinition(
            id="loop.retry",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_success",
                "max_iterations": 5,
                "continue_on_error": True,
            },
        )
        node = LoopNode(definition)

        # Mock all failures
        with patch.object(node, "_simulate_body_execution", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": False, "error": "Always fails"}

            result = await node.execute(sample_message, execution_context)

            assert result.success is True  # Loop completes, even if no iteration succeeded
            assert result.metadata["iterations"] == 5
            assert result.metadata["loop_result"]["termination_reason"] == "max_iterations_reached"


class TestUntilConditionLoop:
    """Tests for UNTIL_CONDITION loop execution."""

    @pytest.mark.asyncio
    async def test_until_condition_success_count(self, sample_message, execution_context):
        """Test until condition based on success count."""
        definition = NodeDefinition(
            id="loop.condition",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "success_count >= 3",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        with patch.object(node, "_simulate_body_execution", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": True, "output": "Success"}

            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 3
            assert result.metadata["loop_result"]["termination_reason"] == "condition_met"

    @pytest.mark.asyncio
    async def test_until_condition_iteration_count(self, sample_message, execution_context):
        """Test until condition based on iteration count."""
        definition = NodeDefinition(
            id="loop.condition",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "iteration >= 5",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["iterations"] == 5

    @pytest.mark.asyncio
    async def test_until_condition_last_result(self, sample_message, execution_context):
        """Test until condition checking last result."""
        definition = NodeDefinition(
            id="loop.condition",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "last_success and 'done' in str(last_output)",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        call_count = 0

        async def mock_body_exec(message, context):
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                return {"success": True, "output": "Task is done"}
            return {"success": True, "output": "Processing..."}

        with patch.object(node, "_simulate_body_execution", side_effect=mock_body_exec):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 4

    @pytest.mark.asyncio
    async def test_until_condition_missing_parameter(
        self, sample_message, execution_context
    ):
        """Test until condition fails without condition_expression."""
        definition = NodeDefinition(
            id="loop.condition",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                # Missing condition_expression
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is False
        assert "requires 'condition_expression' parameter" in result.error


class TestWhileConditionLoop:
    """Tests for WHILE_CONDITION loop execution."""

    @pytest.mark.asyncio
    async def test_while_condition_basic(self, sample_message, execution_context):
        """Test while condition basic execution."""
        definition = NodeDefinition(
            id="loop.while",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "while_condition",
                "condition_expression": "iteration < 5",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        # Stops when iteration reaches 5 (condition becomes false before 6th iteration)
        assert result.metadata["iterations"] == 5
        assert result.metadata["loop_result"]["termination_reason"] == "condition_false"

    @pytest.mark.asyncio
    async def test_while_condition_with_success_count(
        self, sample_message, execution_context
    ):
        """Test while condition based on success count."""
        definition = NodeDefinition(
            id="loop.while",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "while_condition",
                "condition_expression": "success_count < 3",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        with patch.object(node, "_simulate_body_execution", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": True, "output": "Success"}

            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 3

    @pytest.mark.asyncio
    async def test_while_condition_missing_parameter(
        self, sample_message, execution_context
    ):
        """Test while condition fails without condition_expression."""
        definition = NodeDefinition(
            id="loop.while",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "while_condition",
                # Missing condition_expression
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is False
        assert "requires 'condition_expression' parameter" in result.error


class TestLoopSafetyLimits:
    """Tests for loop safety limits (max_iterations, timeout)."""

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, sample_message, execution_context):
        """Test loop stops at max_iterations limit."""
        definition = NodeDefinition(
            id="loop.limited",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_success",
                "max_iterations": 5,
                "continue_on_error": True,
            },
        )
        node = LoopNode(definition)

        # Mock always failing
        with patch.object(node, "_simulate_body_execution", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": False, "error": "Always fails"}

            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 5
            assert result.metadata["loop_result"]["termination_reason"] == "max_iterations_reached"

    @pytest.mark.asyncio
    async def test_timeout_limit(self, sample_message, execution_context):
        """Test loop stops at timeout limit."""
        definition = NodeDefinition(
            id="loop.timeout",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 100,
                "timeout_ms": 1000,  # Short timeout (minimum allowed)
                "max_iterations": 100,
            },
        )
        node = LoopNode(definition)

        # Mock slow execution
        async def slow_exec(message, context):
            import asyncio
            await asyncio.sleep(0.05)  # 50ms per iteration
            return {"success": True, "output": "Done"}

        with patch.object(node, "_simulate_body_execution", side_effect=slow_exec):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            # Should stop before completing 100 iterations due to timeout
            assert result.metadata["iterations"] < 100
            assert result.metadata["loop_result"]["termination_reason"] == "timeout_exceeded"


class TestLoopStateTracking:
    """Tests for loop state tracking."""

    @pytest.mark.asyncio
    async def test_collect_results_enabled(self, sample_message, execution_context):
        """Test loop collects all iteration results."""
        definition = NodeDefinition(
            id="loop.collect",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 3,
                "collect_results": True,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        loop_result = result.metadata["loop_result"]
        assert len(loop_result["all_iterations"]) == 3

    @pytest.mark.asyncio
    async def test_collect_results_disabled(self, sample_message, execution_context):
        """Test loop doesn't collect results when disabled."""
        definition = NodeDefinition(
            id="loop.no_collect",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 3,
                "collect_results": False,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        loop_result = result.metadata["loop_result"]
        assert len(loop_result["all_iterations"]) == 0

    @pytest.mark.asyncio
    async def test_iteration_metadata(self, sample_message, execution_context):
        """Test iteration metadata is passed correctly."""
        definition = NodeDefinition(
            id="loop.metadata",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 2,
                "pass_iteration_number": True,
            },
        )
        node = LoopNode(definition)

        # Track iteration messages
        iteration_messages = []

        async def capture_message(message, context):
            iteration_messages.append(message.payload.metadata.get("iteration"))
            return {"success": True, "output": "Done"}

        with patch.object(node, "_simulate_body_execution", side_effect=capture_message):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert iteration_messages == [1, 2]


class TestLoopErrorHandling:
    """Tests for loop error handling."""

    @pytest.mark.asyncio
    async def test_continue_on_error_enabled(self, sample_message, execution_context):
        """Test loop continues on error when enabled."""
        definition = NodeDefinition(
            id="loop.continue",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 5,
                "continue_on_error": True,
            },
        )
        node = LoopNode(definition)

        call_count = 0

        async def sometimes_fail(message, context):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return {"success": False, "error": "Failed"}
            return {"success": True, "output": "Success"}

        with patch.object(node, "_simulate_body_execution", side_effect=sometimes_fail):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["iterations"] == 5
            # Should have 3 successes and 2 failures
            assert result.metadata["loop_result"]["success_rate"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_continue_on_error_disabled(self, sample_message, execution_context):
        """Test loop stops on error when continue_on_error is disabled."""
        definition = NodeDefinition(
            id="loop.stop_on_error",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 5,
                "continue_on_error": False,
            },
        )
        node = LoopNode(definition)

        call_count = 0

        async def fail_on_third(message, context):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                return {"success": False, "error": "Failed"}
            return {"success": True, "output": "Success"}

        with patch.object(node, "_simulate_body_execution", side_effect=fail_on_third):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            # Should stop after 3 iterations
            assert result.metadata["iterations"] == 3
            assert result.metadata["loop_result"]["termination_reason"] == "iteration_failed"


class TestLoopConditionEvaluation:
    """Tests for condition expression evaluation."""

    @pytest.mark.asyncio
    async def test_condition_with_variables(self, sample_message, execution_context):
        """Test condition can access execution context variables."""
        execution_context.set_variable("target_count", 3)

        definition = NodeDefinition(
            id="loop.vars",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "iteration >= variables['target_count']",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["iterations"] == 3

    @pytest.mark.asyncio
    async def test_condition_with_utility_functions(self, sample_message, execution_context):
        """Test condition can use utility functions."""
        definition = NodeDefinition(
            id="loop.utils",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "len(results) >= 4",
                "max_iterations": 10,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert result.metadata["iterations"] == 4

    @pytest.mark.asyncio
    async def test_condition_evaluation_error(self, sample_message, execution_context):
        """Test loop handles condition evaluation errors gracefully."""
        definition = NodeDefinition(
            id="loop.bad_condition",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "until_condition",
                "condition_expression": "undefined_variable > 5",
                "max_iterations": 3,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        # Should complete all iterations since condition always evaluates to False on error
        assert result.success is True
        assert result.metadata["iterations"] == 3


class TestLoopOutputMessages:
    """Tests for loop output messages."""

    @pytest.mark.asyncio
    async def test_output_message_structure(self, sample_message, execution_context):
        """Test loop creates properly structured output message."""
        definition = NodeDefinition(
            id="loop.output",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 2,
            },
        )
        node = LoopNode(definition)

        result = await node.execute(sample_message, execution_context)

        assert result.success is True
        assert len(result.output_messages) == 1

        output_msg = result.output_messages[0]
        assert output_msg.source_node == "loop.output"
        assert output_msg.trace_id == sample_message.trace_id
        assert output_msg.parent_id == sample_message.message_id
        assert "loop_result" in output_msg.payload.metadata

    @pytest.mark.asyncio
    async def test_final_output_from_last_success(self, sample_message, execution_context):
        """Test final output is taken from last successful iteration."""
        definition = NodeDefinition(
            id="loop.final",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 3,
            },
        )
        node = LoopNode(definition)

        call_count = 0

        async def numbered_output(message, context):
            nonlocal call_count
            call_count += 1
            return {"success": True, "output": f"Result {call_count}"}

        with patch.object(node, "_simulate_body_execution", side_effect=numbered_output):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            loop_result = result.metadata["loop_result"]
            assert loop_result["final_output"] == "Result 3"


class TestLoopSuccessRate:
    """Tests for loop success rate calculation."""

    @pytest.mark.asyncio
    async def test_success_rate_all_success(self, sample_message, execution_context):
        """Test success rate with all successful iterations."""
        definition = NodeDefinition(
            id="loop.success",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 5,
                "continue_on_error": True,
            },
        )
        node = LoopNode(definition)

        with patch.object(node, "_simulate_body_execution", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": True, "output": "Success"}

            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_success_rate_mixed(self, sample_message, execution_context):
        """Test success rate with mixed success/failure."""
        definition = NodeDefinition(
            id="loop.mixed",
            type=NodeType.LOOP,
            config={
                "body_node": "model.process",
                "condition_type": "fixed_count",
                "fixed_count": 10,
                "continue_on_error": True,
            },
        )
        node = LoopNode(definition)

        call_count = 0

        async def alternating(message, context):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return {"success": False, "error": "Failed"}
            return {"success": True, "output": "Success"}

        with patch.object(node, "_simulate_body_execution", side_effect=alternating):
            result = await node.execute(sample_message, execution_context)

            assert result.success is True
            assert result.metadata["success_rate"] == 0.5
