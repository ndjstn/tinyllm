"""Tool usage analytics for TinyLLM.

This module provides analytics and insights about tool usage patterns,
including usage statistics, trends, and performance metrics.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TimeWindow(str, Enum):
    """Time windows for analytics."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class UsageRecord:
    """A single usage record."""

    tool_id: str
    timestamp: datetime
    duration_ms: float
    success: bool
    user_id: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolStats:
    """Statistics for a single tool."""

    tool_id: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_used: Optional[datetime] = None
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float("inf") else 0,
            "max_duration_ms": self.max_duration_ms,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_counts": self.error_counts,
        }


@dataclass
class TimeBucket:
    """Statistics for a time bucket."""

    start_time: datetime
    end_time: datetime
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_ms: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls


@dataclass
class AnalyticsReport:
    """A complete analytics report."""

    start_time: datetime
    end_time: datetime
    total_calls: int
    total_tools: int
    top_tools: List[Tuple[str, int]]
    slowest_tools: List[Tuple[str, float]]
    error_prone_tools: List[Tuple[str, float]]
    hourly_distribution: Dict[int, int]
    tool_stats: Dict[str, ToolStats]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_calls": self.total_calls,
            "total_tools": self.total_tools,
            "top_tools": self.top_tools,
            "slowest_tools": self.slowest_tools,
            "error_prone_tools": self.error_prone_tools,
            "hourly_distribution": self.hourly_distribution,
            "tool_stats": {k: v.to_dict() for k, v in self.tool_stats.items()},
        }


class UsageAnalytics:
    """Main analytics engine for tool usage."""

    def __init__(
        self,
        max_records: int = 100000,
        retention_days: int = 30,
    ):
        """Initialize analytics engine.

        Args:
            max_records: Maximum records to keep.
            retention_days: Days to retain records.
        """
        self.max_records = max_records
        self.retention_days = retention_days
        self._records: List[UsageRecord] = []
        self._tool_stats: Dict[str, ToolStats] = {}

    def record(
        self,
        tool_id: str,
        duration_ms: float,
        success: bool,
        user_id: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a tool usage.

        Args:
            tool_id: Tool identifier.
            duration_ms: Execution duration.
            success: Whether execution succeeded.
            user_id: Optional user ID.
            error_type: Optional error type.
            metadata: Optional metadata.

        Returns:
            The created record.
        """
        record = UsageRecord(
            tool_id=tool_id,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            success=success,
            user_id=user_id,
            error_type=error_type,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._update_stats(record)
        self._cleanup()

        return record

    def _update_stats(self, record: UsageRecord) -> None:
        """Update tool statistics with new record."""
        if record.tool_id not in self._tool_stats:
            self._tool_stats[record.tool_id] = ToolStats(tool_id=record.tool_id)

        stats = self._tool_stats[record.tool_id]
        stats.total_calls += 1
        stats.total_duration_ms += record.duration_ms
        stats.min_duration_ms = min(stats.min_duration_ms, record.duration_ms)
        stats.max_duration_ms = max(stats.max_duration_ms, record.duration_ms)
        stats.last_used = record.timestamp

        if record.success:
            stats.successful_calls += 1
        else:
            stats.failed_calls += 1
            if record.error_type:
                stats.error_counts[record.error_type] = (
                    stats.error_counts.get(record.error_type, 0) + 1
                )

    def _cleanup(self) -> None:
        """Clean up old records."""
        # Remove excess records
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        # Remove old records
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        self._records = [r for r in self._records if r.timestamp > cutoff]

    def get_tool_stats(self, tool_id: str) -> Optional[ToolStats]:
        """Get statistics for a specific tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            ToolStats or None.
        """
        return self._tool_stats.get(tool_id)

    def get_all_stats(self) -> Dict[str, ToolStats]:
        """Get statistics for all tools.

        Returns:
            Dictionary of tool stats.
        """
        return dict(self._tool_stats)

    def get_top_tools(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get most used tools.

        Args:
            n: Number of tools to return.

        Returns:
            List of (tool_id, call_count) tuples.
        """
        sorted_stats = sorted(
            self._tool_stats.items(),
            key=lambda x: x[1].total_calls,
            reverse=True,
        )
        return [(s.tool_id, s.total_calls) for _, s in sorted_stats[:n]]

    def get_slowest_tools(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get slowest tools by average duration.

        Args:
            n: Number of tools to return.

        Returns:
            List of (tool_id, avg_duration_ms) tuples.
        """
        sorted_stats = sorted(
            self._tool_stats.items(),
            key=lambda x: x[1].avg_duration_ms,
            reverse=True,
        )
        return [(s.tool_id, s.avg_duration_ms) for _, s in sorted_stats[:n]]

    def get_error_prone_tools(
        self, n: int = 10, min_calls: int = 1
    ) -> List[Tuple[str, float]]:
        """Get tools with highest failure rates.

        Args:
            n: Number of tools to return.
            min_calls: Minimum calls required.

        Returns:
            List of (tool_id, failure_rate) tuples.
        """
        filtered_stats = [
            s for s in self._tool_stats.values() if s.total_calls >= min_calls
        ]
        sorted_stats = sorted(
            filtered_stats, key=lambda x: x.failure_rate, reverse=True
        )
        return [(s.tool_id, s.failure_rate) for s in sorted_stats[:n]]

    def get_time_series(
        self,
        window: TimeWindow = TimeWindow.HOUR,
        tool_id: Optional[str] = None,
        hours: int = 24,
    ) -> List[TimeBucket]:
        """Get time series data.

        Args:
            window: Time window for buckets.
            tool_id: Optional tool filter.
            hours: Hours of history.

        Returns:
            List of time buckets.
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)

        # Filter records
        records = [r for r in self._records if r.timestamp > cutoff]
        if tool_id:
            records = [r for r in records if r.tool_id == tool_id]

        # Determine bucket size
        if window == TimeWindow.MINUTE:
            bucket_size = timedelta(minutes=1)
        elif window == TimeWindow.HOUR:
            bucket_size = timedelta(hours=1)
        elif window == TimeWindow.DAY:
            bucket_size = timedelta(days=1)
        else:
            bucket_size = timedelta(hours=1)

        # Create buckets
        buckets: Dict[datetime, TimeBucket] = {}
        for record in records:
            # Round to bucket start
            bucket_start = record.timestamp.replace(
                minute=0 if window != TimeWindow.MINUTE else record.timestamp.minute,
                second=0,
                microsecond=0,
            )

            if window == TimeWindow.HOUR:
                bucket_start = bucket_start.replace(minute=0)
            elif window == TimeWindow.DAY:
                bucket_start = bucket_start.replace(hour=0, minute=0)

            if bucket_start not in buckets:
                buckets[bucket_start] = TimeBucket(
                    start_time=bucket_start,
                    end_time=bucket_start + bucket_size,
                )

            bucket = buckets[bucket_start]
            bucket.total_calls += 1
            bucket.total_duration_ms += record.duration_ms

            if record.success:
                bucket.successful_calls += 1
            else:
                bucket.failed_calls += 1

        return sorted(buckets.values(), key=lambda b: b.start_time)

    def get_hourly_distribution(
        self, tool_id: Optional[str] = None
    ) -> Dict[int, int]:
        """Get distribution of calls by hour of day.

        Args:
            tool_id: Optional tool filter.

        Returns:
            Dictionary of hour -> call count.
        """
        distribution = defaultdict(int)

        records = self._records
        if tool_id:
            records = [r for r in records if r.tool_id == tool_id]

        for record in records:
            distribution[record.timestamp.hour] += 1

        return dict(distribution)

    def get_user_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by user.

        Returns:
            Dictionary of user_id -> stats.
        """
        user_stats: Dict[str, Dict[str, Any]] = {}

        for record in self._records:
            if record.user_id is None:
                continue

            if record.user_id not in user_stats:
                user_stats[record.user_id] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "tools_used": set(),
                }

            stats = user_stats[record.user_id]
            stats["total_calls"] += 1
            stats["tools_used"].add(record.tool_id)

            if record.success:
                stats["successful_calls"] += 1
            else:
                stats["failed_calls"] += 1

        # Convert sets to lists for serialization
        for user_id in user_stats:
            user_stats[user_id]["tools_used"] = list(
                user_stats[user_id]["tools_used"]
            )

        return user_stats

    def generate_report(
        self, hours: int = 24
    ) -> AnalyticsReport:
        """Generate a comprehensive analytics report.

        Args:
            hours: Hours of data to include.

        Returns:
            AnalyticsReport.
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)

        records = [r for r in self._records if r.timestamp > cutoff]

        return AnalyticsReport(
            start_time=cutoff,
            end_time=now,
            total_calls=len(records),
            total_tools=len(self._tool_stats),
            top_tools=self.get_top_tools(10),
            slowest_tools=self.get_slowest_tools(10),
            error_prone_tools=self.get_error_prone_tools(10),
            hourly_distribution=self.get_hourly_distribution(),
            tool_stats=dict(self._tool_stats),
        )

    def reset(self) -> None:
        """Reset all analytics data."""
        self._records.clear()
        self._tool_stats.clear()


class AnalyticsToolWrapper:
    """Wrapper that tracks tool usage for analytics."""

    def __init__(
        self,
        tool: Any,
        analytics: Optional[UsageAnalytics] = None,
        user_id: Optional[str] = None,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            analytics: Analytics engine.
            user_id: Optional user ID.
        """
        self.tool = tool
        self.analytics = analytics or UsageAnalytics()
        self.user_id = user_id

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input_data: Any) -> Any:
        """Execute tool and record analytics.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        tool_id = self.tool.metadata.id
        start_time = time.time()
        success = True
        error_type = None

        try:
            result = await self.tool.execute(input_data)
            return result

        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.analytics.record(
                tool_id=tool_id,
                duration_ms=duration_ms,
                success=success,
                user_id=self.user_id,
                error_type=error_type,
            )


# Convenience functions


def with_analytics(
    tool: Any,
    analytics: Optional[UsageAnalytics] = None,
    user_id: Optional[str] = None,
) -> AnalyticsToolWrapper:
    """Add analytics tracking to a tool.

    Args:
        tool: Tool to wrap.
        analytics: Analytics engine.
        user_id: Optional user ID.

    Returns:
        AnalyticsToolWrapper.
    """
    return AnalyticsToolWrapper(tool, analytics=analytics, user_id=user_id)


def create_analytics(
    max_records: int = 100000,
    retention_days: int = 30,
) -> UsageAnalytics:
    """Create an analytics engine.

    Args:
        max_records: Maximum records.
        retention_days: Retention period.

    Returns:
        UsageAnalytics instance.
    """
    return UsageAnalytics(max_records=max_records, retention_days=retention_days)
