"""Tests for expansion system."""

import pytest
from datetime import datetime, timedelta

from tinyllm.expansion.models import (
    EdgeCreationSpec,
    ExpansionBenefit,
    ExpansionConfig,
    ExpansionCost,
    ExpansionProposal,
    ExpansionResult,
    ExpansionStrategy,
    FailureCategory,
    FailurePattern,
    NodeCreationSpec,
    StrategyType,
)
from tinyllm.expansion.analyzer import PatternAnalyzer, PatternAnalyzerConfig
from tinyllm.expansion.strategies import StrategyGenerator, StrategyGeneratorConfig
from tinyllm.expansion.engine import ExpansionEngine, ExpansionTrigger
from tinyllm.grading.models import (
    DimensionScore,
    Grade,
    GradeLevel,
    GradingContext,
    GradingCriteria,
    GradingResult,
    QualityDimension,
)


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_all_categories_exist(self):
        """All expected categories should be defined."""
        assert FailureCategory.TASK_COMPLEXITY.value == "task_complexity"
        assert FailureCategory.DOMAIN_MISMATCH.value == "domain_mismatch"
        assert FailureCategory.CONTEXT_OVERFLOW.value == "context_overflow"
        assert FailureCategory.TOOL_MISSING.value == "tool_missing"
        assert FailureCategory.INSTRUCTION_UNCLEAR.value == "instruction_unclear"
        assert FailureCategory.MODEL_LIMITATION.value == "model_limitation"
        assert FailureCategory.UNKNOWN.value == "unknown"


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_all_types_exist(self):
        """All expected strategy types should be defined."""
        assert StrategyType.PROMPT_REFINEMENT.value == "prompt_refinement"
        assert StrategyType.TOOL_AUGMENTATION.value == "tool_augmentation"
        assert StrategyType.MODEL_UPGRADE.value == "model_upgrade"
        assert StrategyType.SUB_ROUTING.value == "sub_routing"
        assert StrategyType.NONE.value == "none"


class TestFailurePattern:
    """Tests for FailurePattern model."""

    def test_create_pattern(self):
        """Should create a failure pattern."""
        pattern = FailurePattern(
            id="test_pattern",
            category=FailureCategory.TASK_COMPLEXITY,
            description="Complex math tasks",
            node_id="math_solver",
            occurrence_count=5,
        )

        assert pattern.id == "test_pattern"
        assert pattern.category == FailureCategory.TASK_COMPLEXITY
        assert pattern.occurrence_count == 5
        assert pattern.confidence == 0.5

    def test_merge_patterns(self):
        """Should merge two patterns."""
        p1 = FailurePattern(
            id="pattern_1",
            category=FailureCategory.TASK_COMPLEXITY,
            description="Complex tasks",
            node_id="node_1",
            sample_tasks=["task1", "task2"],
            occurrence_count=3,
        )
        p2 = FailurePattern(
            id="pattern_1",
            category=FailureCategory.TASK_COMPLEXITY,
            description="Complex tasks",
            node_id="node_1",
            sample_tasks=["task3", "task4"],
            occurrence_count=2,
        )

        merged = p1.merge(p2)
        assert merged.occurrence_count == 5
        assert len(merged.sample_tasks) == 4


class TestExpansionCost:
    """Tests for ExpansionCost model."""

    def test_total_score(self):
        """Should calculate total cost score."""
        cost = ExpansionCost(
            memory_mb=1000,  # 1 GB
            latency_ms=1000,  # 1 second
            complexity=0.5,
            maintenance=0.5,
        )

        # 1000/1000 + 1000/1000 + 0.5 + 0.5 = 3.0
        assert cost.total_score == 3.0


class TestExpansionBenefit:
    """Tests for ExpansionBenefit model."""

    def test_total_score(self):
        """Should calculate total benefit score."""
        benefit = ExpansionBenefit(
            expected_improvement=1.0,
            coverage_increase=1.0,
            reliability=1.0,
        )

        # 1.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2 = 1.0
        assert benefit.total_score == 1.0


class TestExpansionStrategy:
    """Tests for ExpansionStrategy model."""

    def test_strategy_score(self):
        """Should calculate strategy score."""
        strategy = ExpansionStrategy(
            id="test",
            type=StrategyType.PROMPT_REFINEMENT,
            description="Test strategy",
            target_node_id="node_1",
            cost=ExpansionCost(complexity=0.1),
            benefit=ExpansionBenefit(expected_improvement=0.5, reliability=0.8),
        )

        # benefit.total_score - cost.total_score * 0.5
        assert strategy.score > 0

    def test_create_sub_routing(self):
        """Should create sub-routing strategy."""
        patterns = [
            FailurePattern(
                id="p1",
                category=FailureCategory.DOMAIN_MISMATCH,
                description="Domain issues",
                node_id="node_1",
            )
        ]

        strategy = ExpansionStrategy.create_sub_routing(
            node_id="node_1",
            patterns=patterns,
            sub_domains=["math", "code"],
            expected_improvement=0.5,
        )

        assert strategy.type == StrategyType.SUB_ROUTING
        assert "math" in strategy.implementation["sub_domains"]
        assert "code" in strategy.implementation["sub_domains"]


class TestExpansionProposal:
    """Tests for ExpansionProposal model."""

    def test_is_safe(self):
        """Should determine if proposal is safe."""
        strategy = ExpansionStrategy(
            id="test",
            type=StrategyType.PROMPT_REFINEMENT,
            description="Test",
            target_node_id="node_1",
        )

        # Safe proposal (only creates nodes)
        safe_proposal = ExpansionProposal(
            id="safe",
            strategy=strategy,
            nodes_to_create=[
                NodeCreationSpec(id="new_node", type="model", name="New")
            ],
        )
        assert safe_proposal.is_safe is True

        # Unsafe proposal (modifies nodes)
        unsafe_proposal = ExpansionProposal(
            id="unsafe",
            strategy=strategy,
            nodes_to_modify={"node_1": {"model": "new_model"}},
        )
        assert unsafe_proposal.is_safe is False

    def test_approve(self):
        """Should approve proposal."""
        strategy = ExpansionStrategy(
            id="test",
            type=StrategyType.PROMPT_REFINEMENT,
            description="Test",
            target_node_id="node_1",
        )
        proposal = ExpansionProposal(id="test", strategy=strategy)

        assert proposal.approved is False
        proposal.approve("user")
        assert proposal.approved is True
        assert proposal.approved_by == "user"


class TestExpansionResult:
    """Tests for ExpansionResult model."""

    def test_success_result(self):
        """Should create success result."""
        result = ExpansionResult.success_result(
            proposal_id="prop_1",
            nodes_created=["node_1", "node_2"],
            edges_created=["edge_1"],
        )

        assert result.success is True
        assert len(result.nodes_created) == 2
        assert result.error is None

    def test_failure_result(self):
        """Should create failure result."""
        result = ExpansionResult.failure_result(
            proposal_id="prop_1",
            error="Something went wrong",
        )

        assert result.success is False
        assert "Something went wrong" in result.error


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer."""

    def _create_failing_result(
        self, task: str, feedback: str, node_id: str = "test_node"
    ) -> GradingResult:
        """Helper to create a failing grading result."""
        context = GradingContext(
            task=task,
            response="Some response",
            metadata={"node_id": node_id},
        )
        grade = Grade(
            level=GradeLevel.FAILING,
            overall_score=0.3,
            dimension_scores=[
                DimensionScore(
                    dimension=QualityDimension.CORRECTNESS,
                    score=0.3,
                    reasoning="Poor",
                )
            ],
            feedback=feedback,
            suggestions=[],
            is_passing=False,
        )
        return GradingResult(
            context=context,
            grade=grade,
            criteria=GradingCriteria(),
            judge_model="test",
            latency_ms=100,
        )

    def test_record_failure(self):
        """Should record failures."""
        analyzer = PatternAnalyzer()

        result = self._create_failing_result(
            task="Calculate 2+2",
            feedback="Wrong answer",
            node_id="math_node",
        )
        analyzer.record_failure(result)

        stats = analyzer.get_stats()
        assert stats["total_failures_in_buffer"] == 1

    def test_analyze_node_insufficient_samples(self):
        """Should return empty if insufficient samples."""
        analyzer = PatternAnalyzer(PatternAnalyzerConfig(min_samples=5))

        # Only add 2 failures
        for i in range(2):
            result = self._create_failing_result(
                task=f"Task {i}",
                feedback="Error",
                node_id="test_node",
            )
            analyzer.record_failure(result)

        patterns = analyzer.analyze_node("test_node")
        assert len(patterns) == 0

    def test_analyze_node_with_patterns(self):
        """Should identify patterns with sufficient samples."""
        analyzer = PatternAnalyzer(PatternAnalyzerConfig(min_samples=3))

        # Add failures with similar issues
        for i in range(5):
            result = self._create_failing_result(
                task=f"Complex task {i}",
                feedback="Task too complex for model",
                node_id="test_node",
            )
            analyzer.record_failure(result)

        patterns = analyzer.analyze_node("test_node")
        assert len(patterns) >= 1

    def test_identify_sub_domains(self):
        """Should identify sub-domains from patterns."""
        analyzer = PatternAnalyzer(PatternAnalyzerConfig(min_samples=2))

        patterns = [
            FailurePattern(
                id="p1",
                category=FailureCategory.TASK_COMPLEXITY,
                description="Math errors",
                node_id="solver",
                sample_tasks=["Calculate derivative of x^2", "Solve integral of sin(x)"],
            ),
            FailurePattern(
                id="p2",
                category=FailureCategory.DOMAIN_MISMATCH,
                description="Code errors",
                node_id="solver",
                sample_tasks=["Write Python function", "Debug JavaScript code"],
            ),
        ]

        domains = analyzer.identify_sub_domains("solver", patterns)
        assert len(domains) > 0


class TestStrategyGenerator:
    """Tests for StrategyGenerator."""

    def test_generate_strategies(self):
        """Should generate strategies for patterns."""
        generator = StrategyGenerator()

        patterns = [
            FailurePattern(
                id="p1",
                category=FailureCategory.INSTRUCTION_UNCLEAR,
                description="Unclear prompts",
                node_id="node_1",
                sample_errors=["Response was off-topic"],
            )
        ]

        strategies = generator.generate_strategies(
            node_id="node_1",
            patterns=patterns,
            current_model="qwen2.5:1.5b",
            sub_domains=["math", "code"],
        )

        assert len(strategies) > 0
        # Should be sorted by score
        scores = [s.score for s in strategies]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_strategy(self):
        """Should select best strategy above threshold."""
        generator = StrategyGenerator(
            StrategyGeneratorConfig(min_expected_improvement=0.1)
        )

        # Note: select_best_strategy assumes strategies are already sorted by score
        strategies = [
            ExpansionStrategy(
                id="s2",
                type=StrategyType.SUB_ROUTING,
                description="High improvement",
                target_node_id="node_1",
                benefit=ExpansionBenefit(expected_improvement=0.5, reliability=0.8),
            ),
            ExpansionStrategy(
                id="s1",
                type=StrategyType.PROMPT_REFINEMENT,
                description="Low improvement",
                target_node_id="node_1",
                benefit=ExpansionBenefit(expected_improvement=0.15, reliability=0.5),
            ),
        ]

        best = generator.select_best_strategy(strategies)
        assert best is not None
        assert best.id == "s2"

    def test_select_best_strategy_none_above_threshold(self):
        """Should return None if no strategy meets threshold."""
        generator = StrategyGenerator(
            StrategyGeneratorConfig(min_expected_improvement=0.9)
        )

        strategies = [
            ExpansionStrategy(
                id="s1",
                type=StrategyType.PROMPT_REFINEMENT,
                description="Low improvement",
                target_node_id="node_1",
                benefit=ExpansionBenefit(expected_improvement=0.2),
            ),
        ]

        best = generator.select_best_strategy(strategies)
        assert best is None

    def test_create_routing_proposal(self):
        """Should create routing proposal with nodes and edges."""
        generator = StrategyGenerator()

        strategy = ExpansionStrategy(
            id="route_1",
            type=StrategyType.SUB_ROUTING,
            description="Sub-routing",
            target_node_id="node_1",
            implementation={"sub_domains": ["math", "code"]},
        )

        proposal = generator.create_proposal(
            strategy, node_config={"model": "qwen2.5:1.5b"}
        )

        assert len(proposal.nodes_to_create) == 3  # 1 router + 2 specialists
        assert len(proposal.edges_to_create) == 2  # router -> each specialist


class TestExpansionEngine:
    """Tests for ExpansionEngine."""

    def _create_result(
        self, node_id: str, is_passing: bool, score: float
    ) -> GradingResult:
        """Helper to create grading result."""
        context = GradingContext(
            task="Test task",
            response="Test response",
            metadata={"node_id": node_id},
        )
        level = GradeLevel.GOOD if is_passing else GradeLevel.FAILING
        grade = Grade(
            level=level,
            overall_score=score,
            dimension_scores=[],
            feedback="Test",
            suggestions=[],
            is_passing=is_passing,
        )
        return GradingResult(
            context=context,
            grade=grade,
            criteria=GradingCriteria(),
            judge_model="test",
            latency_ms=100,
            node_id=node_id,  # Set node_id on result for metrics tracking
        )

    def test_record_result(self):
        """Should record results to metrics."""
        engine = ExpansionEngine()

        result = self._create_result("node_1", True, 0.8)
        engine.record_result(result)

        # Check that result was recorded in metrics tracker
        node_metrics = engine.metrics.get_node_metrics("node_1")
        assert node_metrics is not None
        assert node_metrics.total_evaluations == 1

    def test_check_for_expansion_no_failures(self):
        """Should return empty proposals when no failures."""
        engine = ExpansionEngine(ExpansionConfig(min_evaluations=5))

        # Add only passing results
        for i in range(10):
            result = self._create_result("node_1", True, 0.8)
            engine.record_result(result)

        proposals = engine.check_for_expansion()
        assert len(proposals) == 0

    def test_protect_node(self):
        """Should protect nodes from expansion."""
        engine = ExpansionEngine()
        engine.protect_node("protected_node")

        stats = engine.get_stats()
        assert "protected_node" in engine._protected_nodes

    def test_approve_proposal(self):
        """Should approve pending proposal."""
        engine = ExpansionEngine()

        strategy = ExpansionStrategy(
            id="s1",
            type=StrategyType.PROMPT_REFINEMENT,
            description="Test",
            target_node_id="node_1",
        )
        proposal = ExpansionProposal(id="prop_1", strategy=strategy)
        engine._pending_proposals["prop_1"] = proposal

        approved = engine.approve_proposal("prop_1", "test_user")
        assert approved is not None
        assert approved.approved is True
        assert approved.approved_by == "test_user"

    def test_reject_proposal(self):
        """Should reject and remove pending proposal."""
        engine = ExpansionEngine()

        strategy = ExpansionStrategy(
            id="s1",
            type=StrategyType.PROMPT_REFINEMENT,
            description="Test",
            target_node_id="node_1",
        )
        proposal = ExpansionProposal(id="prop_1", strategy=strategy)
        engine._pending_proposals["prop_1"] = proposal

        removed = engine.reject_proposal("prop_1")
        assert removed is True
        assert "prop_1" not in engine._pending_proposals

    def test_apply_approved_proposals(self):
        """Should apply approved proposals."""
        engine = ExpansionEngine()

        strategy = ExpansionStrategy(
            id="s1",
            type=StrategyType.PROMPT_REFINEMENT,
            description="Test",
            target_node_id="node_1",
        )
        proposal = ExpansionProposal(
            id="prop_1",
            strategy=strategy,
            nodes_to_create=[
                NodeCreationSpec(id="new_node", type="model", name="New")
            ],
        )
        proposal.approve("test")
        engine._pending_proposals["prop_1"] = proposal

        results = engine.apply_approved_proposals()
        assert len(results) == 1
        assert results[0].success is True
        assert "prop_1" not in engine._pending_proposals

    def test_reset(self):
        """Should reset engine state."""
        engine = ExpansionEngine()

        # Add some state
        result = self._create_result("node_1", False, 0.3)
        engine.record_result(result)
        engine.protect_node("protected")

        engine.reset()

        stats = engine.get_stats()
        assert stats["nodes_tracked"] == 0
        assert stats["protected_nodes"] == 0


class TestExpansionConfig:
    """Tests for ExpansionConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = ExpansionConfig()

        assert config.min_evaluations == 10
        assert config.fail_threshold == 0.4
        assert config.auto_approve is False
        assert config.cooldown_seconds == 3600

    def test_custom_config(self):
        """Should accept custom values."""
        config = ExpansionConfig(
            min_evaluations=20,
            fail_threshold=0.3,
            auto_approve=True,
        )

        assert config.min_evaluations == 20
        assert config.fail_threshold == 0.3
        assert config.auto_approve is True


class TestExpansionTrigger:
    """Tests for ExpansionTrigger."""

    def test_default_trigger(self):
        """Should have sensible defaults."""
        trigger = ExpansionTrigger()

        assert trigger.enabled is True
        assert trigger.check_interval_seconds == 300
        assert trigger.auto_apply is False
