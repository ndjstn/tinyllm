"""Quality metrics tracking.

Tracks and aggregates quality metrics for nodes, routes,
and the overall system.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tinyllm.grading.models import GradeLevel, GradingResult, QualityDimension


class DimensionMetrics(BaseModel):
    """Aggregated metrics for a single quality dimension."""

    dimension: QualityDimension
    total_evaluations: int = 0
    sum_scores: float = 0.0
    min_score: float = 1.0
    max_score: float = 0.0

    @property
    def avg_score(self) -> float:
        """Calculate average score."""
        if self.total_evaluations == 0:
            return 0.0
        return self.sum_scores / self.total_evaluations

    def record(self, score: float) -> None:
        """Record a new score."""
        self.total_evaluations += 1
        self.sum_scores += score
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)


class QualityMetrics(BaseModel):
    """Quality metrics for a single entity (node, route, etc.)."""

    entity_id: str
    entity_type: str = Field(description="node, route, graph, etc.")

    # Aggregate scores
    total_evaluations: int = 0
    total_passing: int = 0
    total_failing: int = 0
    sum_overall_scores: float = 0.0

    # Grade distribution
    grade_counts: Dict[str, int] = Field(default_factory=dict)

    # Dimension metrics
    dimension_metrics: Dict[str, DimensionMetrics] = Field(default_factory=dict)

    # Timing
    sum_latency_ms: float = 0.0
    first_evaluation: Optional[datetime] = None
    last_evaluation: Optional[datetime] = None

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_evaluations == 0:
            return 0.0
        return self.total_passing / self.total_evaluations

    @property
    def fail_rate(self) -> float:
        """Calculate fail rate."""
        return 1.0 - self.pass_rate

    @property
    def avg_score(self) -> float:
        """Calculate average overall score."""
        if self.total_evaluations == 0:
            return 0.0
        return self.sum_overall_scores / self.total_evaluations

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average grading latency."""
        if self.total_evaluations == 0:
            return 0.0
        return self.sum_latency_ms / self.total_evaluations

    def record(self, result: GradingResult) -> None:
        """Record a grading result."""
        now = datetime.utcnow()

        self.total_evaluations += 1
        if result.passed:
            self.total_passing += 1
        else:
            self.total_failing += 1

        self.sum_overall_scores += result.grade.overall_score
        self.sum_latency_ms += result.latency_ms

        # Update grade distribution
        grade_key = result.grade.level.value
        self.grade_counts[grade_key] = self.grade_counts.get(grade_key, 0) + 1

        # Update dimension metrics
        for ds in result.grade.dimension_scores:
            dim_key = ds.dimension.value
            if dim_key not in self.dimension_metrics:
                self.dimension_metrics[dim_key] = DimensionMetrics(
                    dimension=ds.dimension
                )
            self.dimension_metrics[dim_key].record(ds.score)

        # Update timing
        if self.first_evaluation is None:
            self.first_evaluation = now
        self.last_evaluation = now

    def get_weakest_dimension(self) -> Optional[Tuple[QualityDimension, float]]:
        """Get the dimension with lowest average score."""
        if not self.dimension_metrics:
            return None

        weakest = min(
            self.dimension_metrics.values(),
            key=lambda dm: dm.avg_score,
        )
        return (weakest.dimension, weakest.avg_score)

    def get_strongest_dimension(self) -> Optional[Tuple[QualityDimension, float]]:
        """Get the dimension with highest average score."""
        if not self.dimension_metrics:
            return None

        strongest = max(
            self.dimension_metrics.values(),
            key=lambda dm: dm.avg_score,
        )
        return (strongest.dimension, strongest.avg_score)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "total_evaluations": self.total_evaluations,
            "pass_rate": round(self.pass_rate, 3),
            "avg_score": round(self.avg_score, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "grade_distribution": self.grade_counts,
            "dimension_scores": {
                k: round(v.avg_score, 3) for k, v in self.dimension_metrics.items()
            },
        }


class NodeMetrics(QualityMetrics):
    """Extended metrics specific to nodes."""

    node_type: Optional[str] = None
    model: Optional[str] = None

    # Expansion tracking
    expansion_triggered: bool = False
    expansion_count: int = 0

    def should_expand(
        self,
        min_evaluations: int = 10,
        fail_threshold: float = 0.4,
    ) -> bool:
        """Determine if node should trigger expansion.

        Args:
            min_evaluations: Minimum evaluations before considering expansion.
            fail_threshold: Fail rate threshold to trigger expansion.

        Returns:
            True if expansion should be triggered.
        """
        if self.total_evaluations < min_evaluations:
            return False

        if self.expansion_triggered:
            return False

        return self.fail_rate >= fail_threshold


class MetricsTracker:
    """Central tracker for all quality metrics.

    Aggregates metrics across nodes, routes, and provides
    system-wide quality insights.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self._node_metrics: Dict[str, NodeMetrics] = {}
        self._route_metrics: Dict[str, QualityMetrics] = {}
        self._graph_metrics: Dict[str, QualityMetrics] = {}

        # Recent results for trend analysis
        self._recent_results: List[GradingResult] = []
        self._max_recent = 1000

    def record_result(self, result: GradingResult) -> None:
        """Record a grading result.

        Automatically updates relevant node, route, and graph metrics.
        """
        # Track by node
        if result.node_id:
            if result.node_id not in self._node_metrics:
                self._node_metrics[result.node_id] = NodeMetrics(
                    entity_id=result.node_id,
                    entity_type="node",
                )
            self._node_metrics[result.node_id].record(result)

        # Track by route (if available in metadata)
        route = result.context.metadata.get("route")
        if route:
            if route not in self._route_metrics:
                self._route_metrics[route] = QualityMetrics(
                    entity_id=route,
                    entity_type="route",
                )
            self._route_metrics[route].record(result)

        # Track by graph
        if result.trace_id:
            graph_id = result.trace_id.split("-")[0] if "-" in result.trace_id else "default"
            if graph_id not in self._graph_metrics:
                self._graph_metrics[graph_id] = QualityMetrics(
                    entity_id=graph_id,
                    entity_type="graph",
                )
            self._graph_metrics[graph_id].record(result)

        # Add to recent results
        self._recent_results.append(result)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent :]

    def get_node_metrics(self, node_id: str) -> Optional[NodeMetrics]:
        """Get metrics for a specific node."""
        return self._node_metrics.get(node_id)

    def get_route_metrics(self, route: str) -> Optional[QualityMetrics]:
        """Get metrics for a specific route."""
        return self._route_metrics.get(route)

    def get_all_node_metrics(self) -> Dict[str, NodeMetrics]:
        """Get all node metrics."""
        return self._node_metrics.copy()

    def get_failing_nodes(
        self,
        min_evaluations: int = 10,
        fail_threshold: float = 0.4,
    ) -> List[NodeMetrics]:
        """Get nodes that are failing above threshold.

        Args:
            min_evaluations: Minimum evaluations to consider.
            fail_threshold: Fail rate threshold.

        Returns:
            List of failing node metrics.
        """
        failing = []
        for metrics in self._node_metrics.values():
            if metrics.total_evaluations >= min_evaluations:
                if metrics.fail_rate >= fail_threshold:
                    failing.append(metrics)

        # Sort by fail rate (worst first)
        return sorted(failing, key=lambda m: m.fail_rate, reverse=True)

    def get_expansion_candidates(
        self,
        min_evaluations: int = 10,
        fail_threshold: float = 0.4,
    ) -> List[NodeMetrics]:
        """Get nodes that should be considered for expansion.

        Args:
            min_evaluations: Minimum evaluations to consider.
            fail_threshold: Fail rate threshold.

        Returns:
            List of expansion candidate node metrics.
        """
        candidates = []
        for metrics in self._node_metrics.values():
            if metrics.should_expand(min_evaluations, fail_threshold):
                candidates.append(metrics)

        return sorted(candidates, key=lambda m: m.fail_rate, reverse=True)

    def get_recent_trend(
        self,
        window_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Get quality trend for recent period.

        Args:
            window_minutes: Time window in minutes.

        Returns:
            Trend summary dictionary.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent = [
            r for r in self._recent_results if r.grade.graded_at >= cutoff
        ]

        if not recent:
            return {
                "period_minutes": window_minutes,
                "evaluations": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
            }

        passing = sum(1 for r in recent if r.passed)
        avg_score = sum(r.grade.overall_score for r in recent) / len(recent)

        return {
            "period_minutes": window_minutes,
            "evaluations": len(recent),
            "pass_rate": round(passing / len(recent), 3),
            "avg_score": round(avg_score, 3),
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide quality summary."""
        total_evals = sum(m.total_evaluations for m in self._node_metrics.values())
        total_passing = sum(m.total_passing for m in self._node_metrics.values())

        return {
            "total_nodes_tracked": len(self._node_metrics),
            "total_routes_tracked": len(self._route_metrics),
            "total_evaluations": total_evals,
            "system_pass_rate": round(
                total_passing / total_evals if total_evals > 0 else 0.0, 3
            ),
            "failing_nodes": len(self.get_failing_nodes()),
            "expansion_candidates": len(self.get_expansion_candidates()),
            "recent_trend": self.get_recent_trend(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._node_metrics.clear()
        self._route_metrics.clear()
        self._graph_metrics.clear()
        self._recent_results.clear()
