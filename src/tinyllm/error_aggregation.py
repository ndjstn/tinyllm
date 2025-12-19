"""Error aggregation and deduplication for TinyLLM.

This module provides functionality to aggregate similar errors,
deduplicate error reports, and track error patterns over time.
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from tinyllm.errors import EnrichedError, ErrorCategory, ErrorSeverity
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="error_aggregation")


class ErrorSignature(BaseModel):
    """Signature for error deduplication.

    Used to identify similar errors across different occurrences.
    """

    model_config = {"extra": "forbid"}

    signature_hash: str = Field(description="Hash of error signature")
    category: ErrorCategory = Field(description="Error category")
    exception_type: str = Field(description="Exception type")
    node_id: Optional[str] = Field(default=None, description="Node ID")
    graph_id: Optional[str] = Field(default=None, description="Graph ID")
    message_pattern: str = Field(description="Normalized error message")

    @classmethod
    def from_enriched_error(cls, error: EnrichedError) -> "ErrorSignature":
        """Create signature from enriched error.

        Args:
            error: Enriched error to create signature from.

        Returns:
            ErrorSignature for deduplication.
        """
        # Normalize error message (remove variable parts)
        message_pattern = cls._normalize_message(error.message)

        # Create signature components
        components = [
            error.category.value,
            error.context.exception_type,
            error.context.node_id or "unknown",
            error.context.graph_id or "unknown",
            message_pattern,
        ]

        # Hash signature
        signature_str = "|".join(components)
        signature_hash = hashlib.sha256(signature_str.encode()).hexdigest()[:16]

        return cls(
            signature_hash=signature_hash,
            category=error.category,
            exception_type=error.context.exception_type,
            node_id=error.context.node_id,
            graph_id=error.context.graph_id,
            message_pattern=message_pattern,
        )

    @staticmethod
    def _normalize_message(message: str) -> str:
        """Normalize error message for pattern matching.

        Removes variable parts like IDs, timestamps, and numbers
        to enable pattern matching.

        Args:
            message: Raw error message.

        Returns:
            Normalized message pattern.
        """
        import re

        # Remove UUIDs
        normalized = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "<UUID>",
            message,
            flags=re.IGNORECASE,
        )

        # Remove numeric IDs
        normalized = re.sub(r"\b\d{4,}\b", "<ID>", normalized)

        # Remove timestamps
        normalized = re.sub(
            r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}",
            "<TIMESTAMP>",
            normalized,
        )

        # Remove file paths
        normalized = re.sub(r"/[\w/.-]+\.\w+", "<PATH>", normalized)
        normalized = re.sub(r"[A-Z]:\\[\w\\.-]+\.\w+", "<PATH>", normalized)

        # Remove IP addresses
        normalized = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>", normalized)

        # Remove ports
        normalized = re.sub(r":\d{2,5}\b", ":<PORT>", normalized)

        return normalized.strip()


class AggregatedError(BaseModel):
    """Aggregated error with multiple occurrences.

    Represents a group of similar errors deduplicated by signature.
    """

    model_config = {"extra": "forbid"}

    signature: ErrorSignature = Field(description="Error signature")
    count: int = Field(default=1, ge=1, description="Number of occurrences")
    first_seen: datetime = Field(
        default_factory=datetime.utcnow, description="First occurrence"
    )
    last_seen: datetime = Field(
        default_factory=datetime.utcnow, description="Last occurrence"
    )
    error_ids: List[str] = Field(
        default_factory=list, description="Individual error IDs"
    )
    sample_errors: List[EnrichedError] = Field(
        default_factory=list, description="Sample errors (max 5)"
    )
    affected_nodes: Set[str] = Field(
        default_factory=set, description="Nodes affected"
    )
    affected_graphs: Set[str] = Field(
        default_factory=set, description="Graphs affected"
    )
    affected_traces: Set[str] = Field(
        default_factory=set, description="Traces affected"
    )

    # Severity tracking
    highest_severity: ErrorSeverity = Field(
        default=ErrorSeverity.ERROR, description="Highest severity seen"
    )
    severity_counts: Dict[str, int] = Field(
        default_factory=dict, description="Count per severity"
    )

    # Metadata
    metadata: Dict[str, any] = Field(
        default_factory=dict, description="Aggregated metadata"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def add_occurrence(self, error: EnrichedError) -> None:
        """Add another occurrence of this error.

        Args:
            error: Error to add to aggregation.
        """
        self.count += 1
        self.last_seen = datetime.utcnow()
        self.error_ids.append(error.error_id)

        # Update severity tracking
        if error.severity.value > self.highest_severity.value:
            self.highest_severity = error.severity
        severity_key = error.severity.value
        self.severity_counts[severity_key] = self.severity_counts.get(severity_key, 0) + 1

        # Keep sample errors (max 5)
        if len(self.sample_errors) < 5:
            self.sample_errors.append(error)

        # Track affected components
        if error.context.node_id:
            self.affected_nodes.add(error.context.node_id)
        if error.context.graph_id:
            self.affected_graphs.add(error.context.graph_id)
        if error.context.trace_id:
            self.affected_traces.add(error.context.trace_id)

        logger.debug(
            "error_occurrence_added",
            signature_hash=self.signature.signature_hash,
            count=self.count,
            error_id=error.error_id,
        )

    def get_occurrence_rate(self, window_minutes: int = 60) -> float:
        """Calculate occurrence rate within time window.

        Args:
            window_minutes: Time window in minutes.

        Returns:
            Occurrences per minute.
        """
        if self.count <= 1:
            return 0.0

        time_span = (self.last_seen - self.first_seen).total_seconds() / 60
        if time_span == 0:
            return float(self.count)

        return self.count / max(time_span, 1)


class ErrorAggregator:
    """Error aggregator and deduplicator.

    Aggregates similar errors, tracks patterns, and provides
    deduplication for error reporting.
    """

    def __init__(
        self,
        max_aggregations: int = 10000,
        cleanup_age_hours: int = 24,
    ):
        """Initialize error aggregator.

        Args:
            max_aggregations: Maximum number of aggregations to keep.
            cleanup_age_hours: Age in hours after which to clean up old errors.
        """
        self.max_aggregations = max_aggregations
        self.cleanup_age_hours = cleanup_age_hours

        # Storage
        self._aggregations: Dict[str, AggregatedError] = {}
        self._signature_index: Dict[str, str] = {}  # signature_hash -> error_id

        # Statistics
        self._total_errors_seen = 0
        self._total_aggregations = 0
        self._last_cleanup = datetime.utcnow()

        logger.info(
            "error_aggregator_initialized",
            max_aggregations=max_aggregations,
            cleanup_age_hours=cleanup_age_hours,
        )

    def add_error(self, error: EnrichedError) -> AggregatedError:
        """Add error to aggregation.

        Args:
            error: Enriched error to add.

        Returns:
            AggregatedError for this error (new or existing).
        """
        self._total_errors_seen += 1

        # Create signature
        signature = ErrorSignature.from_enriched_error(error)

        # Check if we've seen this error before
        if signature.signature_hash in self._signature_index:
            # Update existing aggregation
            agg_id = self._signature_index[signature.signature_hash]
            aggregation = self._aggregations[agg_id]
            aggregation.add_occurrence(error)

            logger.debug(
                "error_aggregated",
                signature_hash=signature.signature_hash,
                total_count=aggregation.count,
                error_id=error.error_id,
            )
        else:
            # Create new aggregation
            aggregation = AggregatedError(
                signature=signature,
                count=1,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                error_ids=[error.error_id],
                sample_errors=[error],
                affected_nodes={error.context.node_id} if error.context.node_id else set(),
                affected_graphs={error.context.graph_id} if error.context.graph_id else set(),
                affected_traces={error.context.trace_id} if error.context.trace_id else set(),
                highest_severity=error.severity,
                severity_counts={error.severity.value: 1},
            )

            self._aggregations[error.error_id] = aggregation
            self._signature_index[signature.signature_hash] = error.error_id
            self._total_aggregations += 1

            logger.info(
                "error_aggregation_created",
                signature_hash=signature.signature_hash,
                category=error.category.value,
                severity=error.severity.value,
                error_id=error.error_id,
            )

        # Periodic cleanup
        self._maybe_cleanup()

        return aggregation

    def get_aggregation(self, signature_hash: str) -> Optional[AggregatedError]:
        """Get aggregation by signature hash.

        Args:
            signature_hash: Signature hash to look up.

        Returns:
            AggregatedError if found, None otherwise.
        """
        agg_id = self._signature_index.get(signature_hash)
        if agg_id:
            return self._aggregations.get(agg_id)
        return None

    def get_top_errors(
        self,
        limit: int = 10,
        category: Optional[ErrorCategory] = None,
        min_severity: Optional[ErrorSeverity] = None,
        time_window_hours: Optional[int] = None,
    ) -> List[AggregatedError]:
        """Get top errors by occurrence count.

        Args:
            limit: Maximum number of errors to return.
            category: Filter by category.
            min_severity: Minimum severity to include.
            time_window_hours: Only include errors within this window.

        Returns:
            List of top aggregated errors.
        """
        errors = list(self._aggregations.values())

        # Apply filters
        if category:
            errors = [e for e in errors if e.signature.category == category]

        if min_severity:
            min_severity_value = min_severity.value
            errors = [
                e for e in errors
                if e.highest_severity.value >= min_severity_value
            ]

        if time_window_hours:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            errors = [e for e in errors if e.last_seen >= cutoff]

        # Sort by count descending
        errors.sort(key=lambda e: e.count, reverse=True)

        return errors[:limit]

    def get_recent_errors(
        self,
        limit: int = 10,
        hours: int = 1,
    ) -> List[AggregatedError]:
        """Get recent errors within time window.

        Args:
            limit: Maximum number of errors to return.
            hours: Time window in hours.

        Returns:
            List of recent aggregated errors.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        errors = [
            e for e in self._aggregations.values()
            if e.last_seen >= cutoff
        ]

        # Sort by last seen descending
        errors.sort(key=lambda e: e.last_seen, reverse=True)

        return errors[:limit]

    def get_statistics(self) -> Dict[str, any]:
        """Get aggregator statistics.

        Returns:
            Dictionary of statistics.
        """
        # Count by category
        by_category = defaultdict(int)
        by_severity = defaultdict(int)

        for agg in self._aggregations.values():
            by_category[agg.signature.category.value] += agg.count
            by_severity[agg.highest_severity.value] += agg.count

        return {
            "total_errors_seen": self._total_errors_seen,
            "total_aggregations": self._total_aggregations,
            "unique_signatures": len(self._signature_index),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "last_cleanup": self._last_cleanup.isoformat(),
        }

    def _maybe_cleanup(self) -> None:
        """Cleanup old aggregations if needed."""
        now = datetime.utcnow()

        # Check if cleanup is needed
        if (now - self._last_cleanup).total_seconds() < 3600:  # 1 hour
            return

        if len(self._aggregations) < self.max_aggregations:
            return

        logger.info("error_aggregation_cleanup_starting")

        # Remove old aggregations
        cutoff = now - timedelta(hours=self.cleanup_age_hours)
        removed = 0

        for error_id, agg in list(self._aggregations.items()):
            if agg.last_seen < cutoff:
                # Remove from index
                del self._signature_index[agg.signature.signature_hash]
                # Remove aggregation
                del self._aggregations[error_id]
                removed += 1

        self._last_cleanup = now

        logger.info(
            "error_aggregation_cleanup_completed",
            removed_count=removed,
            remaining_count=len(self._aggregations),
        )

    def clear(self) -> None:
        """Clear all aggregations."""
        self._aggregations.clear()
        self._signature_index.clear()
        self._total_errors_seen = 0
        self._total_aggregations = 0
        self._last_cleanup = datetime.utcnow()

        logger.info("error_aggregations_cleared")


# Global aggregator instance
_aggregator: Optional[ErrorAggregator] = None


def get_aggregator() -> ErrorAggregator:
    """Get global error aggregator instance.

    Returns:
        Global ErrorAggregator instance.
    """
    global _aggregator
    if _aggregator is None:
        _aggregator = ErrorAggregator()
    return _aggregator
