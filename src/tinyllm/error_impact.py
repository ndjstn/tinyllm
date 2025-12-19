"""Error impact scoring for TinyLLM.

This module provides algorithms to score error impact based on
severity, frequency, affected components, and business criticality.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.error_aggregation import AggregatedError
from tinyllm.errors import EnrichedError, ErrorCategory, ErrorSeverity
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="error_impact")


class ImpactLevel(str, Enum):
    """Impact level classification."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class ImpactScore(BaseModel):
    """Comprehensive impact score for an error.

    Combines multiple factors to assess overall error impact.
    """

    model_config = {"extra": "forbid"}

    # Overall score
    total_score: float = Field(ge=0.0, le=100.0, description="Total impact score (0-100)")
    impact_level: ImpactLevel = Field(description="Impact level classification")

    # Component scores
    severity_score: float = Field(ge=0.0, le=100.0, description="Severity score")
    frequency_score: float = Field(ge=0.0, le=100.0, description="Frequency score")
    scope_score: float = Field(ge=0.0, le=100.0, description="Scope score")
    recency_score: float = Field(ge=0.0, le=100.0, description="Recency score")
    criticality_score: float = Field(ge=0.0, le=100.0, description="Component criticality score")

    # Weights used
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights used for scoring"
    )

    # Metadata
    error_id: str = Field(description="Error ID")
    signature_hash: Optional[str] = Field(
        default=None,
        description="Signature hash for aggregated errors"
    )
    computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When score was computed"
    )

    # Recommendations
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions based on impact"
    )


class ImpactScorer:
    """Calculate impact scores for errors.

    Uses configurable weights to assess error impact across
    multiple dimensions.
    """

    # Default weights for score components
    DEFAULT_WEIGHTS = {
        "severity": 0.30,
        "frequency": 0.25,
        "scope": 0.20,
        "recency": 0.15,
        "criticality": 0.10,
    }

    # Severity to score mapping
    SEVERITY_SCORES = {
        ErrorSeverity.DEBUG: 0.0,
        ErrorSeverity.INFO: 10.0,
        ErrorSeverity.WARNING: 30.0,
        ErrorSeverity.ERROR: 60.0,
        ErrorSeverity.CRITICAL: 85.0,
        ErrorSeverity.FATAL: 100.0,
    }

    # Category criticality scores
    CATEGORY_CRITICALITY = {
        ErrorCategory.VALIDATION: 20.0,
        ErrorCategory.EXECUTION: 70.0,
        ErrorCategory.TIMEOUT: 40.0,
        ErrorCategory.RESOURCE: 60.0,
        ErrorCategory.NETWORK: 50.0,
        ErrorCategory.MODEL: 80.0,
        ErrorCategory.CONFIGURATION: 90.0,
        ErrorCategory.AUTHENTICATION: 75.0,
        ErrorCategory.PERMISSION: 65.0,
        ErrorCategory.DATA: 55.0,
        ErrorCategory.INTERNAL: 70.0,
        ErrorCategory.UNKNOWN: 50.0,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        component_criticality: Optional[Dict[str, float]] = None,
    ):
        """Initialize impact scorer.

        Args:
            weights: Custom weights for score components.
            component_criticality: Custom criticality scores for components.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.component_criticality = component_criticality or {}

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "impact_scorer_weights_invalid",
                weight_sum=weight_sum,
                message="Weights should sum to 1.0, normalizing",
            )
            # Normalize weights
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

        logger.info(
            "impact_scorer_initialized",
            weights=self.weights,
        )

    def score_error(self, error: EnrichedError) -> ImpactScore:
        """Score a single error.

        Args:
            error: Enriched error to score.

        Returns:
            ImpactScore with detailed breakdown.
        """
        # Calculate component scores
        severity_score = self._score_severity(error)
        frequency_score = self._score_frequency_single(error)
        scope_score = self._score_scope(error)
        recency_score = self._score_recency(error.context.timestamp)
        criticality_score = self._score_criticality(error)

        # Calculate weighted total
        total_score = (
            severity_score * self.weights["severity"]
            + frequency_score * self.weights["frequency"]
            + scope_score * self.weights["scope"]
            + recency_score * self.weights["recency"]
            + criticality_score * self.weights["criticality"]
        )

        # Determine impact level
        impact_level = self._classify_impact(total_score)

        # Get recommendations
        recommendations = self._get_recommendations(
            total_score,
            impact_level,
            error
        )

        score = ImpactScore(
            total_score=round(total_score, 2),
            impact_level=impact_level,
            severity_score=round(severity_score, 2),
            frequency_score=round(frequency_score, 2),
            scope_score=round(scope_score, 2),
            recency_score=round(recency_score, 2),
            criticality_score=round(criticality_score, 2),
            weights=self.weights.copy(),
            error_id=error.error_id,
            recommended_actions=recommendations,
        )

        logger.debug(
            "error_scored",
            error_id=error.error_id,
            total_score=total_score,
            impact_level=impact_level.value,
        )

        return score

    def score_aggregated_error(self, agg_error: AggregatedError) -> ImpactScore:
        """Score an aggregated error.

        Takes into account frequency and recency of occurrences.

        Args:
            agg_error: Aggregated error to score.

        Returns:
            ImpactScore with detailed breakdown.
        """
        # Use first sample error for severity/scope/criticality
        sample = agg_error.sample_errors[0] if agg_error.sample_errors else None
        if not sample:
            logger.warning(
                "aggregated_error_no_samples",
                signature_hash=agg_error.signature.signature_hash,
            )
            # Return minimal score
            return ImpactScore(
                total_score=0.0,
                impact_level=ImpactLevel.NEGLIGIBLE,
                severity_score=0.0,
                frequency_score=0.0,
                scope_score=0.0,
                recency_score=0.0,
                criticality_score=0.0,
                weights=self.weights.copy(),
                error_id="unknown",
                signature_hash=agg_error.signature.signature_hash,
            )

        # Calculate component scores
        severity_score = self._score_severity_aggregated(agg_error)
        frequency_score = self._score_frequency_aggregated(agg_error)
        scope_score = self._score_scope_aggregated(agg_error)
        recency_score = self._score_recency(agg_error.last_seen)
        criticality_score = self._score_criticality(sample)

        # Calculate weighted total
        total_score = (
            severity_score * self.weights["severity"]
            + frequency_score * self.weights["frequency"]
            + scope_score * self.weights["scope"]
            + recency_score * self.weights["recency"]
            + criticality_score * self.weights["criticality"]
        )

        # Determine impact level
        impact_level = self._classify_impact(total_score)

        # Get recommendations
        recommendations = self._get_recommendations(
            total_score,
            impact_level,
            sample,
            agg_error
        )

        score = ImpactScore(
            total_score=round(total_score, 2),
            impact_level=impact_level,
            severity_score=round(severity_score, 2),
            frequency_score=round(frequency_score, 2),
            scope_score=round(scope_score, 2),
            recency_score=round(recency_score, 2),
            criticality_score=round(criticality_score, 2),
            weights=self.weights.copy(),
            error_id=sample.error_id,
            signature_hash=agg_error.signature.signature_hash,
            recommended_actions=recommendations,
        )

        logger.info(
            "aggregated_error_scored",
            signature_hash=agg_error.signature.signature_hash,
            total_score=total_score,
            impact_level=impact_level.value,
            occurrence_count=agg_error.count,
        )

        return score

    def _score_severity(self, error: EnrichedError) -> float:
        """Score based on error severity.

        Args:
            error: Error to score.

        Returns:
            Severity score (0-100).
        """
        return self.SEVERITY_SCORES.get(error.severity, 50.0)

    def _score_severity_aggregated(self, agg_error: AggregatedError) -> float:
        """Score based on highest severity in aggregation.

        Args:
            agg_error: Aggregated error to score.

        Returns:
            Severity score (0-100).
        """
        return self.SEVERITY_SCORES.get(agg_error.highest_severity, 50.0)

    def _score_frequency_single(self, error: EnrichedError) -> float:
        """Score based on frequency (single error always 0).

        Args:
            error: Error to score.

        Returns:
            Frequency score (0-100).
        """
        # Single errors have no frequency data
        return 0.0

    def _score_frequency_aggregated(self, agg_error: AggregatedError) -> float:
        """Score based on error frequency.

        Args:
            agg_error: Aggregated error to score.

        Returns:
            Frequency score (0-100).
        """
        # Score based on occurrence count and rate
        count_score = min(agg_error.count / 100.0 * 100, 100.0)

        # Calculate rate (errors per hour)
        rate = agg_error.get_occurrence_rate(window_minutes=60)
        rate_score = min(rate / 10.0 * 100, 100.0)  # 10+ per hour = 100

        # Combine count and rate
        return (count_score * 0.6 + rate_score * 0.4)

    def _score_scope(self, error: EnrichedError) -> float:
        """Score based on scope of impact.

        Args:
            error: Error to score.

        Returns:
            Scope score (0-100).
        """
        # Single error affects limited scope
        return 20.0

    def _score_scope_aggregated(self, agg_error: AggregatedError) -> float:
        """Score based on scope of impact.

        Args:
            agg_error: Aggregated error to score.

        Returns:
            Scope score (0-100).
        """
        # Score based on number of affected components
        node_score = min(len(agg_error.affected_nodes) / 10.0 * 100, 100.0)
        graph_score = min(len(agg_error.affected_graphs) / 5.0 * 100, 100.0)
        trace_score = min(len(agg_error.affected_traces) / 20.0 * 100, 100.0)

        # Weight: nodes most important, then graphs, then traces
        return (node_score * 0.5 + graph_score * 0.3 + trace_score * 0.2)

    def _score_recency(self, timestamp: datetime) -> float:
        """Score based on how recent the error is.

        Args:
            timestamp: Error timestamp.

        Returns:
            Recency score (0-100).
        """
        age_minutes = (datetime.utcnow() - timestamp).total_seconds() / 60

        # Exponential decay: fresh errors score higher
        if age_minutes < 5:
            return 100.0
        elif age_minutes < 15:
            return 90.0
        elif age_minutes < 60:
            return 70.0
        elif age_minutes < 240:  # 4 hours
            return 50.0
        elif age_minutes < 1440:  # 24 hours
            return 30.0
        else:
            return 10.0

    def _score_criticality(self, error: EnrichedError) -> float:
        """Score based on component criticality.

        Args:
            error: Error to score.

        Returns:
            Criticality score (0-100).
        """
        # Start with category criticality
        category_score = self.CATEGORY_CRITICALITY.get(
            error.category,
            50.0
        )

        # Check for custom component criticality
        component_scores = []
        if error.context.node_id:
            node_key = f"node:{error.context.node_id}"
            if node_key in self.component_criticality:
                component_scores.append(self.component_criticality[node_key])

        if error.context.graph_id:
            graph_key = f"graph:{error.context.graph_id}"
            if graph_key in self.component_criticality:
                component_scores.append(self.component_criticality[graph_key])

        # Use max of category and component scores
        if component_scores:
            return max(category_score, max(component_scores))

        return category_score

    def _classify_impact(self, score: float) -> ImpactLevel:
        """Classify impact level from score.

        Args:
            score: Total impact score.

        Returns:
            Impact level classification.
        """
        if score >= 90:
            return ImpactLevel.CATASTROPHIC
        elif score >= 75:
            return ImpactLevel.CRITICAL
        elif score >= 50:
            return ImpactLevel.HIGH
        elif score >= 30:
            return ImpactLevel.MEDIUM
        elif score >= 10:
            return ImpactLevel.LOW
        else:
            return ImpactLevel.NEGLIGIBLE

    def _get_recommendations(
        self,
        score: float,
        level: ImpactLevel,
        error: EnrichedError,
        agg_error: Optional[AggregatedError] = None,
    ) -> List[str]:
        """Get recommended actions based on impact.

        Args:
            score: Total impact score.
            level: Impact level.
            error: Sample error.
            agg_error: Aggregated error if available.

        Returns:
            List of recommended actions.
        """
        recommendations = []

        # Based on impact level
        if level in {ImpactLevel.CATASTROPHIC, ImpactLevel.CRITICAL}:
            recommendations.append("IMMEDIATE ACTION REQUIRED")
            recommendations.append("Page on-call engineer")
            recommendations.append("Consider emergency rollback")

        elif level == ImpactLevel.HIGH:
            recommendations.append("Investigate within 1 hour")
            recommendations.append("Notify team lead")
            recommendations.append("Monitor error rate closely")

        elif level == ImpactLevel.MEDIUM:
            recommendations.append("Investigate within 4 hours")
            recommendations.append("Add to sprint backlog")

        # Based on frequency
        if agg_error and agg_error.count > 50:
            recommendations.append("High frequency error - consider circuit breaker")

        # Based on category
        if error.category == ErrorCategory.CONFIGURATION:
            recommendations.append("Review configuration files")
        elif error.category == ErrorCategory.MODEL:
            recommendations.append("Check model availability and health")
        elif error.category == ErrorCategory.RESOURCE:
            recommendations.append("Review resource limits and usage")

        # Based on retryability
        if error.is_retryable:
            recommendations.append("Implement retry logic with backoff")

        return recommendations


# Global scorer instance
_scorer: Optional[ImpactScorer] = None


def get_scorer() -> ImpactScorer:
    """Get global impact scorer instance.

    Returns:
        Global ImpactScorer instance.
    """
    global _scorer
    if _scorer is None:
        _scorer = ImpactScorer()
    return _scorer
