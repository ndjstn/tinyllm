"""Failure pattern analyzer.

Analyzes grading results to identify failure patterns
and cluster similar failures for expansion strategies.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from pydantic import BaseModel, Field

from tinyllm.grading.models import GradingResult, QualityDimension
from tinyllm.expansion.models import FailureCategory, FailurePattern


class PatternAnalyzerConfig(BaseModel):
    """Configuration for pattern analyzer."""

    min_samples: int = Field(default=3, ge=1, description="Min samples to form pattern")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold for clustering"
    )
    max_patterns: int = Field(
        default=10, ge=1, description="Max patterns to track per node"
    )
    window_hours: int = Field(
        default=24, ge=1, description="Time window for pattern analysis"
    )


class PatternAnalyzer:
    """Analyzes grading failures to identify patterns.

    Uses simple heuristics and keyword matching to categorize
    failures and cluster similar ones together.
    """

    # Keywords for failure category detection
    CATEGORY_KEYWORDS: Dict[FailureCategory, List[str]] = {
        FailureCategory.TASK_COMPLEXITY: [
            "complex",
            "difficult",
            "multi-step",
            "advanced",
            "complicated",
            "too hard",
        ],
        FailureCategory.DOMAIN_MISMATCH: [
            "outside",
            "not my area",
            "wrong domain",
            "specialized",
            "expertise",
            "unfamiliar",
        ],
        FailureCategory.CONTEXT_OVERFLOW: [
            "too long",
            "truncated",
            "context",
            "memory",
            "limit",
            "overflow",
        ],
        FailureCategory.TOOL_MISSING: [
            "tool",
            "cannot access",
            "no way to",
            "need to",
            "require",
            "calculator",
            "search",
        ],
        FailureCategory.INSTRUCTION_UNCLEAR: [
            "unclear",
            "ambiguous",
            "confusing",
            "vague",
            "interpret",
            "misunderstood",
        ],
        FailureCategory.MODEL_LIMITATION: [
            "cannot",
            "unable",
            "limitation",
            "beyond",
            "capability",
            "impossible",
        ],
    }

    # Domain keywords for sub-categorization
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "arithmetic": ["add", "subtract", "multiply", "divide", "number", "calculate"],
        "algebra": ["equation", "variable", "solve", "x", "polynomial", "factor"],
        "calculus": ["derivative", "integral", "limit", "calculus", "differential"],
        "geometry": ["angle", "triangle", "circle", "area", "volume", "shape"],
        "statistics": ["probability", "mean", "median", "distribution", "variance"],
        "code_python": ["python", "def ", "class ", "import ", ".py"],
        "code_js": ["javascript", "function", "const ", "let ", "=>", ".js"],
        "code_sql": ["select", "from", "where", "join", "sql", "query"],
        "writing": ["write", "essay", "article", "paragraph", "story", "content"],
        "reasoning": ["think", "reason", "logic", "why", "because", "explain"],
    }

    def __init__(self, config: Optional[PatternAnalyzerConfig] = None):
        """Initialize the pattern analyzer.

        Args:
            config: Configuration options.
        """
        self.config = config or PatternAnalyzerConfig()
        self._patterns: Dict[str, Dict[str, FailurePattern]] = defaultdict(dict)
        self._failure_buffer: Dict[str, List[Tuple[GradingResult, datetime]]] = (
            defaultdict(list)
        )

    def record_failure(self, result: GradingResult) -> None:
        """Record a grading failure for pattern analysis.

        Args:
            result: The failing grading result.
        """
        if result.grade.is_passing:
            return  # Only track failures

        node_id = result.context.metadata.get("node_id", "unknown")
        self._failure_buffer[node_id].append((result, datetime.utcnow()))

        # Prune old entries
        self._prune_buffer(node_id)

    def analyze_node(self, node_id: str) -> List[FailurePattern]:
        """Analyze failures for a specific node.

        Args:
            node_id: The node to analyze.

        Returns:
            List of identified failure patterns.
        """
        failures = [f for f, _ in self._failure_buffer.get(node_id, [])]
        if len(failures) < self.config.min_samples:
            return []

        # Categorize each failure
        categorized: Dict[FailureCategory, List[GradingResult]] = defaultdict(list)
        for failure in failures:
            category = self._categorize_failure(failure)
            categorized[category].append(failure)

        # Create patterns from categories with enough samples
        patterns = []
        for category, results in categorized.items():
            if len(results) >= self.config.min_samples:
                pattern = self._create_pattern(node_id, category, results)
                patterns.append(pattern)

        # Store patterns and limit count
        for pattern in patterns:
            self._patterns[node_id][pattern.id] = pattern

        # Prune to max patterns
        if len(self._patterns[node_id]) > self.config.max_patterns:
            sorted_patterns = sorted(
                self._patterns[node_id].values(),
                key=lambda p: (p.occurrence_count, p.last_seen),
                reverse=True,
            )
            self._patterns[node_id] = {p.id: p for p in sorted_patterns[: self.config.max_patterns]}

        return list(self._patterns[node_id].values())

    def get_patterns(self, node_id: str) -> List[FailurePattern]:
        """Get stored patterns for a node.

        Args:
            node_id: The node ID.

        Returns:
            List of failure patterns.
        """
        return list(self._patterns.get(node_id, {}).values())

    def identify_sub_domains(
        self, node_id: str, patterns: List[FailurePattern]
    ) -> List[str]:
        """Identify sub-domains that could benefit from specialization.

        Args:
            node_id: The node being analyzed.
            patterns: Failure patterns to analyze.

        Returns:
            List of recommended sub-domain names.
        """
        domain_counts: Dict[str, int] = defaultdict(int)

        for pattern in patterns:
            # Analyze sample tasks for domain keywords
            for task in pattern.sample_tasks:
                task_lower = task.lower()
                for domain, keywords in self.DOMAIN_KEYWORDS.items():
                    if any(kw in task_lower for kw in keywords):
                        domain_counts[domain] += 1

        # Return domains with significant presence
        threshold = max(1, len(patterns) // 2)
        return [
            domain
            for domain, count in sorted(
                domain_counts.items(), key=lambda x: x[1], reverse=True
            )
            if count >= threshold
        ][:5]  # Max 5 sub-domains

    def _categorize_failure(self, result: GradingResult) -> FailureCategory:
        """Categorize a failure based on feedback and scores.

        Args:
            result: The grading result.

        Returns:
            The failure category.
        """
        feedback = (result.grade.feedback or "").lower()
        suggestions = " ".join(result.grade.suggestions).lower()
        combined = f"{feedback} {suggestions}"

        # Score each category
        category_scores: Dict[FailureCategory, int] = defaultdict(int)

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined:
                    category_scores[category] += 1

        # Also check dimension scores for hints
        for dim_score in result.grade.dimension_scores:
            if dim_score.score < 0.4:
                if dim_score.dimension == QualityDimension.CORRECTNESS:
                    category_scores[FailureCategory.TASK_COMPLEXITY] += 1
                elif dim_score.dimension == QualityDimension.COMPLETENESS:
                    category_scores[FailureCategory.CONTEXT_OVERFLOW] += 1
                elif dim_score.dimension == QualityDimension.RELEVANCE:
                    category_scores[FailureCategory.DOMAIN_MISMATCH] += 1

        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])
        return FailureCategory.UNKNOWN

    def _create_pattern(
        self,
        node_id: str,
        category: FailureCategory,
        results: List[GradingResult],
    ) -> FailurePattern:
        """Create a failure pattern from categorized results.

        Args:
            node_id: The node ID.
            category: The failure category.
            results: Grading results in this category.

        Returns:
            A new failure pattern.
        """
        # Generate deterministic ID
        pattern_id = hashlib.md5(
            f"{node_id}_{category.value}".encode()
        ).hexdigest()[:8]

        # Extract samples
        sample_tasks = [r.context.task for r in results[:5]]
        sample_errors = [r.grade.feedback for r in results[:5] if r.grade.feedback]

        # Calculate confidence based on sample size
        confidence = min(1.0, len(results) / 10)

        return FailurePattern(
            id=f"pattern_{pattern_id}",
            category=category,
            description=self._generate_description(category, results),
            sample_tasks=sample_tasks,
            sample_errors=sample_errors,
            occurrence_count=len(results),
            node_id=node_id,
            confidence=confidence,
        )

    def _generate_description(
        self, category: FailureCategory, results: List[GradingResult]
    ) -> str:
        """Generate a human-readable description for a pattern.

        Args:
            category: The failure category.
            results: The results in this category.

        Returns:
            Description string.
        """
        descriptions = {
            FailureCategory.TASK_COMPLEXITY: "Tasks are too complex for current model capability",
            FailureCategory.DOMAIN_MISMATCH: "Tasks fall outside the node's domain expertise",
            FailureCategory.CONTEXT_OVERFLOW: "Context exceeds model's effective window",
            FailureCategory.TOOL_MISSING: "Required tools are not available to the node",
            FailureCategory.INSTRUCTION_UNCLEAR: "Prompts may need clarification",
            FailureCategory.MODEL_LIMITATION: "Model has fundamental capability limitations",
            FailureCategory.UNKNOWN: "Failure pattern could not be determined",
        }

        base = descriptions.get(category, "Unknown failure pattern")
        return f"{base} ({len(results)} occurrences)"

    def _prune_buffer(self, node_id: str) -> None:
        """Prune old entries from the failure buffer.

        Args:
            node_id: The node to prune.
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=self.config.window_hours)
        self._failure_buffer[node_id] = [
            (f, t) for f, t in self._failure_buffer[node_id] if t > cutoff
        ]

    def clear_node(self, node_id: str) -> None:
        """Clear all data for a node.

        Args:
            node_id: The node to clear.
        """
        self._patterns.pop(node_id, None)
        self._failure_buffer.pop(node_id, None)

    def clear_all(self) -> None:
        """Clear all analyzer data."""
        self._patterns.clear()
        self._failure_buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics.

        Returns:
            Statistics dictionary.
        """
        total_patterns = sum(len(p) for p in self._patterns.values())
        total_failures = sum(len(f) for f in self._failure_buffer.values())

        return {
            "nodes_tracked": len(self._patterns),
            "total_patterns": total_patterns,
            "total_failures_in_buffer": total_failures,
            "patterns_by_node": {
                node_id: len(patterns)
                for node_id, patterns in self._patterns.items()
            },
        }
