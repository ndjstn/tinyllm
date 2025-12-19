"""Grading models and schemas.

Defines the data structures for quality evaluation, grades,
and grading criteria used by the LLM-as-judge system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GradeLevel(str, Enum):
    """Grade levels for quality assessment."""

    EXCELLENT = "A"  # 90-100%: Exceptional quality
    GOOD = "B"  # 75-89%: Good quality, minor issues
    ACCEPTABLE = "C"  # 60-74%: Acceptable, noticeable issues
    POOR = "D"  # 40-59%: Poor quality, significant issues
    FAILING = "F"  # 0-39%: Unacceptable, major failures

    @classmethod
    def from_score(cls, score: float) -> "GradeLevel":
        """Convert numeric score (0-1) to grade level."""
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.75:
            return cls.GOOD
        elif score >= 0.6:
            return cls.ACCEPTABLE
        elif score >= 0.4:
            return cls.POOR
        else:
            return cls.FAILING

    @property
    def is_passing(self) -> bool:
        """Check if this grade level is considered passing."""
        return self in (GradeLevel.EXCELLENT, GradeLevel.GOOD, GradeLevel.ACCEPTABLE)


class QualityDimension(str, Enum):
    """Dimensions of quality to evaluate."""

    CORRECTNESS = "correctness"  # Is the answer correct?
    COMPLETENESS = "completeness"  # Is the answer complete?
    RELEVANCE = "relevance"  # Is the answer relevant to the question?
    CLARITY = "clarity"  # Is the answer clear and well-structured?
    CONCISENESS = "conciseness"  # Is the answer appropriately concise?
    CODE_QUALITY = "code_quality"  # For code: Is it well-written?
    SAFETY = "safety"  # Does the answer avoid harmful content?


class GradingCriteria(BaseModel):
    """Criteria for grading a response."""

    dimensions: List[QualityDimension] = Field(
        default_factory=lambda: [
            QualityDimension.CORRECTNESS,
            QualityDimension.COMPLETENESS,
            QualityDimension.RELEVANCE,
        ],
        description="Quality dimensions to evaluate",
    )
    weights: Dict[QualityDimension, float] = Field(
        default_factory=dict,
        description="Weight for each dimension (default: equal)",
    )
    passing_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum score to pass",
    )
    require_all_passing: bool = Field(
        default=False,
        description="Require all dimensions to pass individually",
    )

    def get_weight(self, dimension: QualityDimension) -> float:
        """Get the weight for a dimension (default: equal weighting)."""
        if self.weights:
            return self.weights.get(dimension, 1.0 / len(self.dimensions))
        return 1.0 / len(self.dimensions)


class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    dimension: QualityDimension
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation for the score")
    evidence: Optional[str] = Field(
        default=None, description="Specific evidence from the response"
    )


class Grade(BaseModel):
    """A complete grade for a response."""

    level: GradeLevel
    overall_score: float = Field(ge=0.0, le=1.0)
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    feedback: str = Field(description="Summary feedback")
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )
    is_passing: bool = Field(description="Whether the response passed")
    graded_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        dimension_scores: List[DimensionScore],
        criteria: GradingCriteria,
        feedback: str,
        suggestions: Optional[List[str]] = None,
    ) -> "Grade":
        """Create a grade from dimension scores."""
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for ds in dimension_scores:
            weight = criteria.get_weight(ds.dimension)
            weighted_sum += ds.score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine if passing
        if criteria.require_all_passing:
            is_passing = all(
                ds.score >= criteria.passing_threshold for ds in dimension_scores
            )
        else:
            is_passing = overall_score >= criteria.passing_threshold

        return cls(
            level=GradeLevel.from_score(overall_score),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            feedback=feedback,
            suggestions=suggestions or [],
            is_passing=is_passing,
        )


class GradingContext(BaseModel):
    """Context provided to the judge for grading."""

    task: str = Field(description="Original task/question")
    response: str = Field(description="Response to grade")
    expected: Optional[str] = Field(
        default=None, description="Expected/reference answer if available"
    )
    task_type: Optional[str] = Field(
        default=None, description="Type of task (code, math, general)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class GradingResult(BaseModel):
    """Complete result of a grading operation."""

    context: GradingContext
    grade: Grade
    criteria: GradingCriteria
    judge_model: str = Field(description="Model used for judging")
    latency_ms: float = Field(description="Time taken to grade")
    trace_id: Optional[str] = Field(default=None, description="Trace ID if available")
    node_id: Optional[str] = Field(
        default=None, description="Node ID that produced the response"
    )

    @property
    def passed(self) -> bool:
        """Check if the response passed grading."""
        return self.grade.is_passing

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metrics tracking."""
        return {
            "overall_score": self.grade.overall_score,
            "level": self.grade.level.value,
            "is_passing": self.grade.is_passing,
            "latency_ms": self.latency_ms,
            "judge_model": self.judge_model,
            "dimensions": {
                ds.dimension.value: ds.score for ds in self.grade.dimension_scores
            },
        }
