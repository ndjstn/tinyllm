"""Tests for the grading system."""

import pytest
from datetime import datetime, timedelta

from tinyllm.grading.models import (
    Grade,
    GradeLevel,
    GradingCriteria,
    GradingContext,
    GradingResult,
    DimensionScore,
    QualityDimension,
)
from tinyllm.grading.judge import Judge, JudgeConfig, RuleBasedJudge
from tinyllm.grading.metrics import (
    MetricsTracker,
    NodeMetrics,
    QualityMetrics,
    DimensionMetrics,
)


class TestGradeLevel:
    """Tests for GradeLevel enum."""

    def test_from_score_excellent(self):
        """Test excellent grade from high score."""
        assert GradeLevel.from_score(0.95) == GradeLevel.EXCELLENT
        assert GradeLevel.from_score(0.90) == GradeLevel.EXCELLENT

    def test_from_score_good(self):
        """Test good grade from score."""
        assert GradeLevel.from_score(0.85) == GradeLevel.GOOD
        assert GradeLevel.from_score(0.75) == GradeLevel.GOOD

    def test_from_score_acceptable(self):
        """Test acceptable grade from score."""
        assert GradeLevel.from_score(0.70) == GradeLevel.ACCEPTABLE
        assert GradeLevel.from_score(0.60) == GradeLevel.ACCEPTABLE

    def test_from_score_poor(self):
        """Test poor grade from score."""
        assert GradeLevel.from_score(0.55) == GradeLevel.POOR
        assert GradeLevel.from_score(0.40) == GradeLevel.POOR

    def test_from_score_failing(self):
        """Test failing grade from score."""
        assert GradeLevel.from_score(0.30) == GradeLevel.FAILING
        assert GradeLevel.from_score(0.0) == GradeLevel.FAILING

    def test_is_passing(self):
        """Test is_passing property."""
        assert GradeLevel.EXCELLENT.is_passing is True
        assert GradeLevel.GOOD.is_passing is True
        assert GradeLevel.ACCEPTABLE.is_passing is True
        assert GradeLevel.POOR.is_passing is False
        assert GradeLevel.FAILING.is_passing is False


class TestGradingCriteria:
    """Tests for GradingCriteria model."""

    def test_default_criteria(self):
        """Test default grading criteria."""
        criteria = GradingCriteria()
        assert QualityDimension.CORRECTNESS in criteria.dimensions
        assert QualityDimension.COMPLETENESS in criteria.dimensions
        assert QualityDimension.RELEVANCE in criteria.dimensions
        assert criteria.passing_threshold == 0.6

    def test_custom_dimensions(self):
        """Test custom dimensions."""
        criteria = GradingCriteria(
            dimensions=[QualityDimension.CODE_QUALITY, QualityDimension.SAFETY]
        )
        assert len(criteria.dimensions) == 2
        assert QualityDimension.CODE_QUALITY in criteria.dimensions

    def test_get_weight_equal(self):
        """Test equal weight distribution."""
        criteria = GradingCriteria(
            dimensions=[
                QualityDimension.CORRECTNESS,
                QualityDimension.COMPLETENESS,
            ]
        )
        weight = criteria.get_weight(QualityDimension.CORRECTNESS)
        assert weight == 0.5

    def test_get_weight_custom(self):
        """Test custom weight."""
        criteria = GradingCriteria(
            dimensions=[
                QualityDimension.CORRECTNESS,
                QualityDimension.COMPLETENESS,
            ],
            weights={QualityDimension.CORRECTNESS: 0.7},
        )
        assert criteria.get_weight(QualityDimension.CORRECTNESS) == 0.7


class TestGrade:
    """Tests for Grade model."""

    def test_create_grade(self):
        """Test creating a grade from dimension scores."""
        dimension_scores = [
            DimensionScore(
                dimension=QualityDimension.CORRECTNESS,
                score=0.9,
                reasoning="Correct answer",
            ),
            DimensionScore(
                dimension=QualityDimension.COMPLETENESS,
                score=0.8,
                reasoning="Mostly complete",
            ),
        ]
        criteria = GradingCriteria(
            dimensions=[
                QualityDimension.CORRECTNESS,
                QualityDimension.COMPLETENESS,
            ]
        )

        grade = Grade.create(
            dimension_scores=dimension_scores,
            criteria=criteria,
            feedback="Good response",
        )

        assert abs(grade.overall_score - 0.85) < 0.001  # (0.9 + 0.8) / 2
        assert grade.level == GradeLevel.GOOD
        assert grade.is_passing is True

    def test_create_failing_grade(self):
        """Test creating a failing grade."""
        dimension_scores = [
            DimensionScore(
                dimension=QualityDimension.CORRECTNESS,
                score=0.3,
                reasoning="Incorrect",
            ),
        ]
        criteria = GradingCriteria(
            dimensions=[QualityDimension.CORRECTNESS]
        )

        grade = Grade.create(
            dimension_scores=dimension_scores,
            criteria=criteria,
            feedback="Poor response",
        )

        assert grade.overall_score == 0.3
        assert grade.level == GradeLevel.FAILING
        assert grade.is_passing is False

    def test_require_all_passing(self):
        """Test require_all_passing criteria."""
        dimension_scores = [
            DimensionScore(
                dimension=QualityDimension.CORRECTNESS,
                score=0.9,
                reasoning="Correct",
            ),
            DimensionScore(
                dimension=QualityDimension.SAFETY,
                score=0.4,  # Below threshold
                reasoning="Safety issue",
            ),
        ]
        criteria = GradingCriteria(
            dimensions=[QualityDimension.CORRECTNESS, QualityDimension.SAFETY],
            require_all_passing=True,
        )

        grade = Grade.create(
            dimension_scores=dimension_scores,
            criteria=criteria,
            feedback="Mixed quality",
        )

        # Overall score is 0.65, but one dimension fails
        assert grade.is_passing is False


class TestGradingContext:
    """Tests for GradingContext model."""

    def test_create_context(self):
        """Test creating grading context."""
        context = GradingContext(
            task="What is 2 + 2?",
            response="The answer is 4.",
            expected="4",
            task_type="math",
        )
        assert context.task == "What is 2 + 2?"
        assert context.response == "The answer is 4."
        assert context.expected == "4"

    def test_context_with_metadata(self):
        """Test context with metadata."""
        context = GradingContext(
            task="Test task",
            response="Test response",
            metadata={"route": "math", "model": "qwen2.5:3b"},
        )
        assert context.metadata["route"] == "math"


class TestGradingResult:
    """Tests for GradingResult model."""

    def test_to_metrics_dict(self):
        """Test conversion to metrics dictionary."""
        context = GradingContext(task="Test", response="Response")
        grade = Grade(
            level=GradeLevel.GOOD,
            overall_score=0.85,
            dimension_scores=[
                DimensionScore(
                    dimension=QualityDimension.CORRECTNESS,
                    score=0.85,
                    reasoning="Good",
                )
            ],
            feedback="Good job",
            is_passing=True,
        )
        criteria = GradingCriteria()

        result = GradingResult(
            context=context,
            grade=grade,
            criteria=criteria,
            judge_model="qwen3:14b",
            latency_ms=1500.0,
        )

        metrics = result.to_metrics_dict()
        assert metrics["overall_score"] == 0.85
        assert metrics["is_passing"] is True
        assert "correctness" in metrics["dimensions"]


class TestRuleBasedJudge:
    """Tests for RuleBasedJudge."""

    def test_empty_response(self):
        """Test detection of empty response."""
        judge = RuleBasedJudge()
        context = GradingContext(task="Test", response="")

        grade = judge.quick_check(context)
        assert grade is not None
        assert grade.is_passing is False
        assert "empty" in grade.feedback.lower()

    def test_very_short_response(self):
        """Test detection of very short response.

        Short responses alone don't trigger a grade (penalty < 0.5),
        they return None for LLM judging.
        """
        judge = RuleBasedJudge()
        context = GradingContext(task="Test", response="No")

        grade = judge.quick_check(context)
        # Short response penalty (0.3) is below threshold (0.5)
        # so it returns None for LLM judging
        assert grade is None

    def test_error_indicator(self):
        """Test detection of error indicators.

        Single error indicator alone doesn't reach penalty threshold.
        """
        judge = RuleBasedJudge()
        context = GradingContext(
            task="Test",
            response="I cannot help with this request. This is a detailed response explaining why.",
        )

        grade = judge.quick_check(context)
        # Error indicator penalty (0.2) alone is below threshold
        assert grade is None

    def test_multiple_issues(self):
        """Test that multiple issues trigger a grade."""
        judge = RuleBasedJudge()
        # Very short AND has error indicator = 0.3 + 0.2 = 0.5 penalty
        context = GradingContext(
            task="Test",
            response="Error:",
        )

        grade = judge.quick_check(context)
        assert grade is not None
        assert grade.is_passing is False

    def test_good_response_returns_none(self):
        """Test that good responses return None for LLM judging."""
        judge = RuleBasedJudge()
        context = GradingContext(
            task="What is the capital of France?",
            response="The capital of France is Paris. Paris is located in the north-central part of France along the Seine River.",
        )

        grade = judge.quick_check(context)
        assert grade is None  # Should proceed to LLM judging


class TestJudgeConfig:
    """Tests for JudgeConfig."""

    def test_default_config(self):
        """Test default judge configuration."""
        config = JudgeConfig()
        assert config.model == "qwen3:14b"
        assert config.temperature == 0.1
        assert config.max_retries == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = JudgeConfig(
            model="custom:model",
            temperature=0.2,
            max_retries=3,
        )
        assert config.model == "custom:model"
        assert config.temperature == 0.2


class TestDimensionMetrics:
    """Tests for DimensionMetrics."""

    def test_record_scores(self):
        """Test recording dimension scores."""
        metrics = DimensionMetrics(dimension=QualityDimension.CORRECTNESS)

        metrics.record(0.8)
        metrics.record(0.9)
        metrics.record(0.7)

        assert metrics.total_evaluations == 3
        assert abs(metrics.avg_score - 0.8) < 0.001
        assert metrics.min_score == 0.7
        assert metrics.max_score == 0.9


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_record_result(self):
        """Test recording a grading result."""
        metrics = QualityMetrics(entity_id="test_node", entity_type="node")

        context = GradingContext(task="Test", response="Response")
        grade = Grade(
            level=GradeLevel.GOOD,
            overall_score=0.85,
            dimension_scores=[
                DimensionScore(
                    dimension=QualityDimension.CORRECTNESS,
                    score=0.85,
                    reasoning="Good",
                )
            ],
            feedback="Good",
            is_passing=True,
        )

        result = GradingResult(
            context=context,
            grade=grade,
            criteria=GradingCriteria(),
            judge_model="test",
            latency_ms=100.0,
        )

        metrics.record(result)

        assert metrics.total_evaluations == 1
        assert metrics.total_passing == 1
        assert metrics.pass_rate == 1.0
        assert metrics.avg_score == 0.85

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        metrics = QualityMetrics(entity_id="test", entity_type="node")

        # Record 3 passing, 1 failing
        for score, passing in [(0.8, True), (0.9, True), (0.7, True), (0.3, False)]:
            context = GradingContext(task="Test", response="Response")
            grade = Grade(
                level=GradeLevel.from_score(score),
                overall_score=score,
                feedback="Test",
                is_passing=passing,
            )
            result = GradingResult(
                context=context,
                grade=grade,
                criteria=GradingCriteria(),
                judge_model="test",
                latency_ms=100.0,
            )
            metrics.record(result)

        assert metrics.pass_rate == 0.75
        assert metrics.fail_rate == 0.25


class TestNodeMetrics:
    """Tests for NodeMetrics."""

    def test_should_expand_insufficient_data(self):
        """Test that expansion isn't triggered with insufficient data."""
        metrics = NodeMetrics(entity_id="test_node", entity_type="node")

        # Only 5 evaluations (below threshold of 10)
        for _ in range(5):
            context = GradingContext(task="Test", response="Response")
            grade = Grade(
                level=GradeLevel.FAILING,
                overall_score=0.3,
                feedback="Failing",
                is_passing=False,
            )
            result = GradingResult(
                context=context,
                grade=grade,
                criteria=GradingCriteria(),
                judge_model="test",
                latency_ms=100.0,
            )
            metrics.record(result)

        assert metrics.should_expand(min_evaluations=10) is False

    def test_should_expand_high_fail_rate(self):
        """Test that expansion is triggered with high fail rate."""
        metrics = NodeMetrics(entity_id="test_node", entity_type="node")

        # 10 evaluations, all failing
        for _ in range(10):
            context = GradingContext(task="Test", response="Response")
            grade = Grade(
                level=GradeLevel.FAILING,
                overall_score=0.3,
                feedback="Failing",
                is_passing=False,
            )
            result = GradingResult(
                context=context,
                grade=grade,
                criteria=GradingCriteria(),
                judge_model="test",
                latency_ms=100.0,
            )
            metrics.record(result)

        assert metrics.should_expand(min_evaluations=10, fail_threshold=0.4) is True


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_record_and_retrieve(self):
        """Test recording and retrieving metrics."""
        tracker = MetricsTracker()

        context = GradingContext(
            task="Test",
            response="Response",
            metadata={"route": "math"},
        )
        grade = Grade(
            level=GradeLevel.GOOD,
            overall_score=0.85,
            feedback="Good",
            is_passing=True,
        )
        result = GradingResult(
            context=context,
            grade=grade,
            criteria=GradingCriteria(),
            judge_model="test",
            latency_ms=100.0,
            node_id="test_node",
        )

        tracker.record_result(result)

        node_metrics = tracker.get_node_metrics("test_node")
        assert node_metrics is not None
        assert node_metrics.total_evaluations == 1

        route_metrics = tracker.get_route_metrics("math")
        assert route_metrics is not None
        assert route_metrics.total_evaluations == 1

    def test_get_failing_nodes(self):
        """Test getting failing nodes."""
        tracker = MetricsTracker()

        # Record failing results for a node
        for _ in range(15):
            context = GradingContext(task="Test", response="Response")
            grade = Grade(
                level=GradeLevel.FAILING,
                overall_score=0.2,
                feedback="Failing",
                is_passing=False,
            )
            result = GradingResult(
                context=context,
                grade=grade,
                criteria=GradingCriteria(),
                judge_model="test",
                latency_ms=100.0,
                node_id="failing_node",
            )
            tracker.record_result(result)

        failing = tracker.get_failing_nodes(min_evaluations=10)
        assert len(failing) == 1
        assert failing[0].entity_id == "failing_node"

    def test_system_summary(self):
        """Test getting system summary."""
        tracker = MetricsTracker()

        # Record some results
        for i, (node_id, score) in enumerate([
            ("node_a", 0.9),
            ("node_b", 0.8),
            ("node_a", 0.85),
        ]):
            context = GradingContext(task=f"Test {i}", response="Response")
            grade = Grade(
                level=GradeLevel.from_score(score),
                overall_score=score,
                feedback="Test",
                is_passing=score >= 0.6,
            )
            result = GradingResult(
                context=context,
                grade=grade,
                criteria=GradingCriteria(),
                judge_model="test",
                latency_ms=100.0,
                node_id=node_id,
            )
            tracker.record_result(result)

        summary = tracker.get_system_summary()
        assert summary["total_nodes_tracked"] == 2
        assert summary["total_evaluations"] == 3
        assert summary["system_pass_rate"] == 1.0

    def test_reset(self):
        """Test resetting metrics."""
        tracker = MetricsTracker()

        context = GradingContext(task="Test", response="Response")
        grade = Grade(
            level=GradeLevel.GOOD,
            overall_score=0.8,
            feedback="Test",
            is_passing=True,
        )
        result = GradingResult(
            context=context,
            grade=grade,
            criteria=GradingCriteria(),
            judge_model="test",
            latency_ms=100.0,
            node_id="test_node",
        )
        tracker.record_result(result)

        tracker.reset()

        assert tracker.get_node_metrics("test_node") is None
        summary = tracker.get_system_summary()
        assert summary["total_nodes_tracked"] == 0
