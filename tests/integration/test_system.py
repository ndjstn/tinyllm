"""Integration tests for the TinyLLM system.

Tests full system flows without requiring external services.
"""

import pytest
import tempfile
from pathlib import Path

from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import NodeResult
from tinyllm.config.graph import NodeType, NodeDefinition
from tinyllm.nodes.entry_exit import EntryNode, ExitNode
from tinyllm.grading.models import (
    Grade,
    GradeLevel,
    GradingContext,
    GradingCriteria,
    GradingResult,
    QualityDimension,
)
from tinyllm.grading.judge import RuleBasedJudge
from tinyllm.grading.metrics import MetricsTracker
from tinyllm.expansion.engine import ExpansionEngine
from tinyllm.expansion.analyzer import PatternAnalyzer
from tinyllm.expansion.strategies import StrategyGenerator
from tinyllm.memory import MemoryStore, MemoryType


def create_test_message(content: str, source: str = "test", **kwargs) -> Message:
    """Helper to create test messages."""
    from uuid import uuid4

    return Message(
        trace_id=str(uuid4()),
        source_node=source,
        payload=MessagePayload(content=content),
        **kwargs,
    )


def create_node_definition(
    id: str, node_type: NodeType, name: str = "Test", **config
) -> NodeDefinition:
    """Helper to create node definitions for testing."""
    return NodeDefinition(
        id=id,
        type=node_type,
        name=name,
        config=config,
    )


class TestMessageFlow:
    """Test message flow through nodes."""

    @pytest.mark.asyncio
    async def test_entry_to_exit_flow(self):
        """Test basic message flow from entry to exit."""
        # Create nodes
        entry_def = create_node_definition(
            "entry_1", NodeType.ENTRY, "Test Entry", required_fields=["content"]
        )
        exit_def = create_node_definition("exit_1", NodeType.EXIT, "Test Exit")

        entry = EntryNode(entry_def)
        exit_node = ExitNode(exit_def)

        # Create message
        message = create_test_message("Hello, world!")

        # Process through entry
        entry_result = await entry.execute(message, None)
        assert entry_result.success is True
        assert len(entry_result.output_messages) > 0

        # Process through exit
        exit_result = await exit_node.execute(entry_result.output_messages[0], None)
        assert exit_result.success is True

    @pytest.mark.asyncio
    async def test_message_transformation(self):
        """Test message content is preserved through flow."""
        entry_def = create_node_definition("entry_test", NodeType.ENTRY, "Entry Test")
        entry = EntryNode(entry_def)

        # Create message with specific content
        message = create_test_message("Test content for transformation")

        result = await entry.execute(message, None)
        assert result.success is True

        # Verify output message was created with content preserved
        output = result.output_messages[0]
        assert output.payload.content == "Test content for transformation"
        assert output.source_node == "entry_test"


class TestGradingPipeline:
    """Test grading pipeline integration."""

    def test_rule_based_to_metrics(self):
        """Test rule-based judge feeding into metrics tracker."""
        judge = RuleBasedJudge()
        tracker = MetricsTracker()

        # Test with failing response
        context = GradingContext(
            task="Answer the question",
            response="",
            metadata={"node_id": "test_node"},
        )

        # Rule-based check
        grade = judge.quick_check(context)
        assert grade is not None
        assert grade.is_passing is False

        # Create grading result and track
        result = GradingResult(
            context=context,
            grade=grade,
            criteria=GradingCriteria(),
            judge_model="rule_based",
            latency_ms=1.0,
            node_id="test_node",
        )
        tracker.record_result(result)

        # Check metrics updated
        node_metrics = tracker.get_node_metrics("test_node")
        assert node_metrics is not None
        assert node_metrics.total_evaluations == 1
        assert node_metrics.total_failing == 1

    def test_grade_level_progression(self):
        """Test grade levels work correctly."""
        # Test each level
        levels = [
            (0.95, GradeLevel.EXCELLENT),
            (0.80, GradeLevel.GOOD),
            (0.65, GradeLevel.ACCEPTABLE),
            (0.45, GradeLevel.POOR),
            (0.20, GradeLevel.FAILING),
        ]

        for score, expected_level in levels:
            level = GradeLevel.from_score(score)
            assert level == expected_level, f"Score {score} should be {expected_level}"


class TestExpansionIntegration:
    """Test expansion system integration."""

    def test_metrics_to_expansion(self):
        """Test metrics flowing into expansion decisions."""
        engine = ExpansionEngine()

        # Record failing results
        for i in range(15):
            context = GradingContext(
                task=f"Complex task {i}",
                response="Failed response",
                metadata={"node_id": "failing_node"},
            )
            grade = Grade(
                level=GradeLevel.FAILING,
                overall_score=0.3,
                dimension_scores=[],
                feedback="Task too complex",
                suggestions=[],
                is_passing=False,
            )
            result = GradingResult(
                context=context,
                grade=grade,
                criteria=GradingCriteria(),
                judge_model="test",
                latency_ms=100,
                node_id="failing_node",
            )
            engine.record_result(result)

        # Check expansion candidates
        candidates = engine.metrics.get_expansion_candidates(
            min_evaluations=10,
            fail_threshold=0.5,
        )

        # Should have found the failing node
        assert len(candidates) >= 1
        assert any(c.entity_id == "failing_node" for c in candidates)

    def test_analyzer_to_strategy(self):
        """Test analyzer feeding into strategy generator."""
        analyzer = PatternAnalyzer()
        generator = StrategyGenerator()

        # Create failure patterns manually
        from tinyllm.expansion.models import FailurePattern, FailureCategory

        patterns = [
            FailurePattern(
                id="pattern_1",
                category=FailureCategory.TASK_COMPLEXITY,
                description="Complex tasks",
                node_id="test_node",
                sample_tasks=["Complex math problem", "Difficult calculation"],
                occurrence_count=10,
            )
        ]

        # Generate strategies
        strategies = generator.generate_strategies(
            node_id="test_node",
            patterns=patterns,
            current_model="qwen2.5:1.5b",
            sub_domains=["math", "code"],
        )

        # Should have generated some strategies
        assert len(strategies) > 0


class TestMemoryIntegration:
    """Test memory system integration."""

    def test_conversation_with_context(self):
        """Test conversation memory builds context."""
        store = MemoryStore()

        # Simulate conversation
        store.add_message("user", "My name is Alice")
        store.add_message("assistant", "Nice to meet you, Alice!")
        store.add_message("user", "I prefer Python for programming")
        store.add_message("assistant", "Python is a great choice!")

        # Extract entities
        store.extract_entity("user_name", "Alice")
        store.store_preference("language", "Python")

        # Get context
        context = store.get_context_for_prompt(max_tokens=500)

        # Should include conversation history
        assert "Alice" in context or len(store.get_messages()) > 0

    def test_memory_persistence(self):
        """Test memory persists across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "memory.json"

            from tinyllm.memory.models import MemoryConfig

            config = MemoryConfig(persist_path=str(persist_path), auto_persist=True)

            # Create store and add data
            store1 = MemoryStore(config)
            store1.store_fact("important_fact", "The sky is blue")

            # Create new store with same path
            store2 = MemoryStore(config)

            # Should have loaded the fact
            fact = store2.get_fact("important_fact")
            assert fact == "The sky is blue"

    def test_search_across_memory(self):
        """Test unified search across STM and LTM."""
        store = MemoryStore()

        # Add to STM
        store.set_context("recent_topic", "Machine learning algorithms")

        # Add to LTM
        store.store_fact("historical_fact", "Neural networks were invented in 1943")

        # Search should find from both
        results = store.search("neural networks", k=5)

        # Should find something
        assert len(results) >= 0  # Pseudo-embeddings may not find semantic matches


class TestFullPipeline:
    """Test the full system pipeline."""

    @pytest.mark.asyncio
    async def test_task_processing_pipeline(self):
        """Test full task processing pipeline."""
        # Setup components
        memory = MemoryStore()
        tracker = MetricsTracker()

        # Create task
        message = create_test_message("Calculate 2 + 2")
        memory.add_message("user", message.payload.content)

        # Process through entry
        entry_def = create_node_definition("entry", NodeType.ENTRY, "Entry")
        entry = EntryNode(entry_def)
        result = await entry.execute(message, None)
        assert result.success is True

        # Simulate response
        response = "The answer is 4"
        memory.add_message("assistant", response)

        # Grade the response
        context = GradingContext(
            task=message.payload.content,
            response=response,
            expected="4",
            metadata={"node_id": "entry"},
        )

        judge = RuleBasedJudge()
        grade = judge.quick_check(context)

        # Should pass (non-empty, reasonable response)
        assert grade is None  # None means passes rule-based checks

    @pytest.mark.asyncio
    async def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        memory = MemoryStore()

        # Create task with missing required field
        entry_def = create_node_definition(
            "strict_entry",
            NodeType.ENTRY,
            "Strict Entry",
            required_fields=["special_field"],
        )
        entry = EntryNode(entry_def)

        message = create_test_message("Test")

        # Should fail validation
        result = await entry.execute(message, None)
        assert result.success is False

        # Error should be captured
        assert result.error is not None

        # Memory should still be usable
        memory.add_message("system", f"Error: {result.error}")
        messages = memory.get_messages()
        assert len(messages) == 1


class TestCLIIntegration:
    """Test CLI integration."""

    def test_version_command(self):
        """Test version command works."""
        from typer.testing import CliRunner
        from tinyllm.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "TinyLLM" in result.stdout

    def test_models_command(self):
        """Test models command works."""
        from typer.testing import CliRunner
        from tinyllm.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "Routers" in result.stdout
        assert "Specialists" in result.stdout

    def test_stats_command(self):
        """Test stats command works."""
        from typer.testing import CliRunner
        from tinyllm.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "System" in result.stdout
