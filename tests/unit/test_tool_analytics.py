"""Tests for tool usage analytics."""

import pytest
from datetime import datetime, timedelta
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.analytics import (
    AnalyticsReport,
    AnalyticsToolWrapper,
    TimeBucket,
    TimeWindow,
    ToolStats,
    UsageAnalytics,
    UsageRecord,
    create_analytics,
    with_analytics,
)


class AnalyticsInput(BaseModel):
    """Input for analytics tests."""

    value: int = 0


class AnalyticsOutput(BaseModel):
    """Output for analytics tests."""

    result: int = 0


class SuccessTool(BaseTool[AnalyticsInput, AnalyticsOutput]):
    """Tool that succeeds."""

    metadata = ToolMetadata(
        id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        category="utility",
    )
    input_type = AnalyticsInput
    output_type = AnalyticsOutput

    async def execute(self, input: AnalyticsInput) -> AnalyticsOutput:
        return AnalyticsOutput(result=input.value * 2)


class FailTool(BaseTool[AnalyticsInput, AnalyticsOutput]):
    """Tool that fails."""

    metadata = ToolMetadata(
        id="fail_tool",
        name="Fail Tool",
        description="Always fails",
        category="utility",
    )
    input_type = AnalyticsInput
    output_type = AnalyticsOutput

    async def execute(self, input: AnalyticsInput) -> AnalyticsOutput:
        raise ValueError("Intentional failure")


class TestUsageRecord:
    """Tests for UsageRecord."""

    def test_creation(self):
        """Test record creation."""
        record = UsageRecord(
            tool_id="my_tool",
            timestamp=datetime.now(),
            duration_ms=100.5,
            success=True,
        )

        assert record.tool_id == "my_tool"
        assert record.duration_ms == 100.5
        assert record.success is True

    def test_with_user(self):
        """Test record with user."""
        record = UsageRecord(
            tool_id="tool",
            timestamp=datetime.now(),
            duration_ms=50.0,
            success=True,
            user_id="user123",
        )

        assert record.user_id == "user123"

    def test_with_error(self):
        """Test record with error."""
        record = UsageRecord(
            tool_id="tool",
            timestamp=datetime.now(),
            duration_ms=50.0,
            success=False,
            error_type="ValueError",
        )

        assert record.error_type == "ValueError"
        assert record.success is False

    def test_with_metadata(self):
        """Test record with metadata."""
        record = UsageRecord(
            tool_id="tool",
            timestamp=datetime.now(),
            duration_ms=50.0,
            success=True,
            metadata={"key": "value"},
        )

        assert record.metadata["key"] == "value"


class TestToolStats:
    """Tests for ToolStats."""

    def test_creation(self):
        """Test stats creation."""
        stats = ToolStats(tool_id="my_tool")

        assert stats.tool_id == "my_tool"
        assert stats.total_calls == 0

    def test_success_rate_empty(self):
        """Test success rate with no calls."""
        stats = ToolStats(tool_id="tool")

        assert stats.success_rate == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = ToolStats(
            tool_id="tool",
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
        )

        assert stats.success_rate == 0.8

    def test_failure_rate(self):
        """Test failure rate calculation."""
        stats = ToolStats(
            tool_id="tool",
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
        )

        assert stats.failure_rate == pytest.approx(0.2)

    def test_avg_duration_empty(self):
        """Test average duration with no calls."""
        stats = ToolStats(tool_id="tool")

        assert stats.avg_duration_ms == 0.0

    def test_avg_duration(self):
        """Test average duration calculation."""
        stats = ToolStats(
            tool_id="tool",
            total_calls=5,
            total_duration_ms=500.0,
        )

        assert stats.avg_duration_ms == 100.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = ToolStats(
            tool_id="tool",
            total_calls=10,
            successful_calls=9,
            failed_calls=1,
            total_duration_ms=1000.0,
        )

        d = stats.to_dict()

        assert d["tool_id"] == "tool"
        assert d["total_calls"] == 10
        assert d["success_rate"] == 0.9


class TestTimeBucket:
    """Tests for TimeBucket."""

    def test_creation(self):
        """Test bucket creation."""
        now = datetime.now()
        bucket = TimeBucket(
            start_time=now,
            end_time=now + timedelta(hours=1),
        )

        assert bucket.total_calls == 0

    def test_avg_duration_empty(self):
        """Test average duration with no calls."""
        now = datetime.now()
        bucket = TimeBucket(
            start_time=now,
            end_time=now + timedelta(hours=1),
        )

        assert bucket.avg_duration_ms == 0.0

    def test_avg_duration(self):
        """Test average duration calculation."""
        now = datetime.now()
        bucket = TimeBucket(
            start_time=now,
            end_time=now + timedelta(hours=1),
            total_calls=4,
            total_duration_ms=400.0,
        )

        assert bucket.avg_duration_ms == 100.0


class TestAnalyticsReport:
    """Tests for AnalyticsReport."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = datetime.now()
        report = AnalyticsReport(
            start_time=now - timedelta(hours=24),
            end_time=now,
            total_calls=100,
            total_tools=5,
            top_tools=[("tool1", 50), ("tool2", 30)],
            slowest_tools=[("tool3", 500.0)],
            error_prone_tools=[("tool4", 0.3)],
            hourly_distribution={10: 20, 11: 30},
            tool_stats={},
        )

        d = report.to_dict()

        assert d["total_calls"] == 100
        assert d["total_tools"] == 5
        assert len(d["top_tools"]) == 2


class TestUsageAnalytics:
    """Tests for UsageAnalytics."""

    def test_creation(self):
        """Test analytics creation."""
        analytics = UsageAnalytics()

        assert analytics.max_records == 100000
        assert analytics.retention_days == 30

    def test_record(self):
        """Test recording usage."""
        analytics = UsageAnalytics()

        record = analytics.record(
            tool_id="my_tool",
            duration_ms=100.0,
            success=True,
        )

        assert record.tool_id == "my_tool"

    def test_get_tool_stats(self):
        """Test getting tool stats."""
        analytics = UsageAnalytics()

        analytics.record("tool1", 100.0, True)
        analytics.record("tool1", 200.0, True)
        analytics.record("tool1", 150.0, False, error_type="ValueError")

        stats = analytics.get_tool_stats("tool1")

        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
        assert stats.error_counts.get("ValueError") == 1

    def test_get_tool_stats_not_found(self):
        """Test getting stats for non-existent tool."""
        analytics = UsageAnalytics()

        stats = analytics.get_tool_stats("nonexistent")

        assert stats is None

    def test_get_all_stats(self):
        """Test getting all stats."""
        analytics = UsageAnalytics()

        analytics.record("tool1", 100.0, True)
        analytics.record("tool2", 200.0, True)

        all_stats = analytics.get_all_stats()

        assert "tool1" in all_stats
        assert "tool2" in all_stats

    def test_get_top_tools(self):
        """Test getting top tools."""
        analytics = UsageAnalytics()

        for _ in range(10):
            analytics.record("popular", 100.0, True)
        for _ in range(5):
            analytics.record("medium", 100.0, True)
        for _ in range(2):
            analytics.record("rare", 100.0, True)

        top = analytics.get_top_tools(2)

        assert len(top) == 2
        assert top[0][0] == "popular"
        assert top[0][1] == 10

    def test_get_slowest_tools(self):
        """Test getting slowest tools."""
        analytics = UsageAnalytics()

        analytics.record("fast", 50.0, True)
        analytics.record("slow", 500.0, True)
        analytics.record("medium", 200.0, True)

        slowest = analytics.get_slowest_tools(2)

        assert len(slowest) == 2
        assert slowest[0][0] == "slow"

    def test_get_error_prone_tools(self):
        """Test getting error-prone tools."""
        analytics = UsageAnalytics()

        # Reliable tool: 9/10 success
        for _ in range(9):
            analytics.record("reliable", 100.0, True)
        analytics.record("reliable", 100.0, False)

        # Unreliable tool: 5/10 success
        for _ in range(5):
            analytics.record("unreliable", 100.0, True)
        for _ in range(5):
            analytics.record("unreliable", 100.0, False)

        error_prone = analytics.get_error_prone_tools(2, min_calls=5)

        assert len(error_prone) == 2
        assert error_prone[0][0] == "unreliable"
        assert error_prone[0][1] == 0.5

    def test_get_time_series(self):
        """Test getting time series data."""
        analytics = UsageAnalytics()

        # Record some data
        analytics.record("tool", 100.0, True)
        analytics.record("tool", 150.0, True)
        analytics.record("tool", 200.0, False)

        buckets = analytics.get_time_series(
            window=TimeWindow.HOUR,
            hours=1,
        )

        assert len(buckets) >= 1
        total_calls = sum(b.total_calls for b in buckets)
        assert total_calls == 3

    def test_get_time_series_filtered(self):
        """Test filtered time series."""
        analytics = UsageAnalytics()

        analytics.record("tool1", 100.0, True)
        analytics.record("tool2", 100.0, True)

        buckets = analytics.get_time_series(
            window=TimeWindow.HOUR,
            tool_id="tool1",
            hours=1,
        )

        total_calls = sum(b.total_calls for b in buckets)
        assert total_calls == 1

    def test_get_hourly_distribution(self):
        """Test hourly distribution."""
        analytics = UsageAnalytics()

        analytics.record("tool", 100.0, True)
        analytics.record("tool", 100.0, True)

        dist = analytics.get_hourly_distribution()

        current_hour = datetime.now().hour
        assert current_hour in dist
        assert dist[current_hour] >= 2

    def test_get_hourly_distribution_filtered(self):
        """Test filtered hourly distribution."""
        analytics = UsageAnalytics()

        analytics.record("tool1", 100.0, True)
        analytics.record("tool2", 100.0, True)
        analytics.record("tool2", 100.0, True)

        dist = analytics.get_hourly_distribution(tool_id="tool2")

        current_hour = datetime.now().hour
        assert dist[current_hour] == 2

    def test_get_user_stats(self):
        """Test user statistics."""
        analytics = UsageAnalytics()

        analytics.record("tool1", 100.0, True, user_id="user1")
        analytics.record("tool2", 100.0, True, user_id="user1")
        analytics.record("tool1", 100.0, False, user_id="user2")

        user_stats = analytics.get_user_stats()

        assert "user1" in user_stats
        assert "user2" in user_stats
        assert user_stats["user1"]["total_calls"] == 2
        assert user_stats["user1"]["successful_calls"] == 2
        assert len(user_stats["user1"]["tools_used"]) == 2

    def test_generate_report(self):
        """Test report generation."""
        analytics = UsageAnalytics()

        for i in range(10):
            analytics.record(f"tool{i % 3}", 100.0 * (i + 1), i % 2 == 0)

        report = analytics.generate_report(hours=24)

        assert report.total_calls == 10
        assert len(report.top_tools) > 0

    def test_reset(self):
        """Test resetting analytics."""
        analytics = UsageAnalytics()

        analytics.record("tool", 100.0, True)
        analytics.record("tool", 100.0, True)

        analytics.reset()

        assert analytics.get_tool_stats("tool") is None

    def test_max_records_cleanup(self):
        """Test max records cleanup."""
        analytics = UsageAnalytics(max_records=5)

        for i in range(10):
            analytics.record("tool", 100.0, True)

        # Internal records should be limited
        assert len(analytics._records) <= 5

    def test_min_max_duration(self):
        """Test min/max duration tracking."""
        analytics = UsageAnalytics()

        analytics.record("tool", 50.0, True)
        analytics.record("tool", 200.0, True)
        analytics.record("tool", 100.0, True)

        stats = analytics.get_tool_stats("tool")

        assert stats.min_duration_ms == 50.0
        assert stats.max_duration_ms == 200.0


class TestAnalyticsToolWrapper:
    """Tests for AnalyticsToolWrapper."""

    @pytest.mark.asyncio
    async def test_success_recorded(self):
        """Test successful execution is recorded."""
        analytics = UsageAnalytics()
        wrapper = AnalyticsToolWrapper(SuccessTool(), analytics=analytics)

        result = await wrapper.execute(AnalyticsInput(value=5))

        assert result.result == 10

        stats = analytics.get_tool_stats("success_tool")
        assert stats.total_calls == 1
        assert stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_failure_recorded(self):
        """Test failed execution is recorded."""
        analytics = UsageAnalytics()
        wrapper = AnalyticsToolWrapper(FailTool(), analytics=analytics)

        with pytest.raises(ValueError):
            await wrapper.execute(AnalyticsInput(value=5))

        stats = analytics.get_tool_stats("fail_tool")
        assert stats.total_calls == 1
        assert stats.failed_calls == 1
        assert "ValueError" in stats.error_counts

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        """Test duration is recorded."""
        analytics = UsageAnalytics()
        wrapper = AnalyticsToolWrapper(SuccessTool(), analytics=analytics)

        await wrapper.execute(AnalyticsInput(value=5))

        stats = analytics.get_tool_stats("success_tool")
        assert stats.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_with_user_id(self):
        """Test recording with user ID."""
        analytics = UsageAnalytics()
        wrapper = AnalyticsToolWrapper(
            SuccessTool(),
            analytics=analytics,
            user_id="test_user",
        )

        await wrapper.execute(AnalyticsInput(value=5))

        user_stats = analytics.get_user_stats()
        assert "test_user" in user_stats

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        wrapper = AnalyticsToolWrapper(SuccessTool())

        assert wrapper.metadata.id == "success_tool"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_analytics(self):
        """Test with_analytics function."""
        wrapper = with_analytics(SuccessTool())

        result = await wrapper.execute(AnalyticsInput(value=5))

        assert result.result == 10

    @pytest.mark.asyncio
    async def test_with_analytics_custom(self):
        """Test with_analytics with custom analytics."""
        analytics = UsageAnalytics()
        wrapper = with_analytics(
            SuccessTool(),
            analytics=analytics,
            user_id="custom_user",
        )

        await wrapper.execute(AnalyticsInput(value=5))

        stats = analytics.get_tool_stats("success_tool")
        assert stats.total_calls == 1

    def test_create_analytics(self):
        """Test create_analytics function."""
        analytics = create_analytics(
            max_records=5000,
            retention_days=7,
        )

        assert analytics.max_records == 5000
        assert analytics.retention_days == 7


class TestTimeWindows:
    """Tests for different time windows."""

    def test_minute_window(self):
        """Test minute time window."""
        analytics = UsageAnalytics()

        analytics.record("tool", 100.0, True)

        buckets = analytics.get_time_series(
            window=TimeWindow.MINUTE,
            hours=1,
        )

        assert len(buckets) >= 1

    def test_hour_window(self):
        """Test hour time window."""
        analytics = UsageAnalytics()

        analytics.record("tool", 100.0, True)

        buckets = analytics.get_time_series(
            window=TimeWindow.HOUR,
            hours=1,
        )

        assert len(buckets) >= 1

    def test_day_window(self):
        """Test day time window."""
        analytics = UsageAnalytics()

        analytics.record("tool", 100.0, True)

        buckets = analytics.get_time_series(
            window=TimeWindow.DAY,
            hours=24,
        )

        assert len(buckets) >= 1
