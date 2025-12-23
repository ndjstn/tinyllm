"""Tests for health check system."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinyllm.health import (
    DependencyType,
    HealthCheckConfig,
    HealthChecker,
    HealthMonitor,
    HealthStatus,
    NodeHealthTracker,
    NodeHealthTrackerConfig,
    OllamaHealthChecker,
    RedisHealthChecker,
    SQLiteHealthChecker,
    TelemetryHealthChecker,
    configure_default_health_checks,
    get_health_monitor,
    get_node_health_tracker,
    reset_node_health_tracker,
)


class TestHealthCheckConfig:
    """Test HealthCheckConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HealthCheckConfig()

        assert config.enabled is True
        assert config.interval_seconds == 30
        assert config.timeout_seconds == 5
        assert config.failure_threshold == 3
        assert config.success_threshold == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = HealthCheckConfig(
            enabled=False,
            interval_seconds=60,
            timeout_seconds=10,
            failure_threshold=5,
            success_threshold=3,
        )

        assert config.enabled is False
        assert config.interval_seconds == 60
        assert config.timeout_seconds == 10
        assert config.failure_threshold == 5
        assert config.success_threshold == 3


class MockHealthChecker(HealthChecker):
    """Mock health checker for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        super().__init__(name, DependencyType.EXTERNAL_API)
        self.should_fail = should_fail

    async def _check_impl(self):
        if self.should_fail:
            raise Exception("Mock failure")

        from tinyllm.health import HealthCheckResult

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Mock healthy",
        )


@pytest.mark.asyncio
class TestHealthChecker:
    """Test HealthChecker base class."""

    async def test_healthy_check(self):
        """Test successful health check."""
        checker = MockHealthChecker("test_service")
        result = await checker.check()

        assert result.status == HealthStatus.HEALTHY
        assert result.name == "test_service"
        assert result.response_time_ms > 0

    async def test_failing_check(self):
        """Test failing health check."""
        checker = MockHealthChecker("test_service", should_fail=True)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Mock failure" in result.message

    async def test_timeout_check(self):
        """Test health check timeout."""

        class SlowChecker(HealthChecker):
            async def _check_impl(self):
                await asyncio.sleep(10)
                from tinyllm.health import HealthCheckResult

                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Should not reach here",
                )

        config = HealthCheckConfig(timeout_seconds=1)
        checker = SlowChecker("slow_service", DependencyType.EXTERNAL_API, config)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "timeout" in result.message.lower()

    async def test_threshold_logic(self):
        """Test failure and success thresholds."""
        config = HealthCheckConfig(failure_threshold=2, success_threshold=2)
        checker = MockHealthChecker("test_service")
        checker.config = config

        # First success - should be UNKNOWN/DEGRADED
        result = await checker.check()
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

        # Second success - should be HEALTHY
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY

        # Make it fail
        checker.should_fail = True
        result = await checker.check()
        # Status can be DEGRADED or UNHEALTHY depending on threshold
        assert result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]

        # Second failure - should be UNHEALTHY
        result = await checker.check()
        assert result.status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
class TestOllamaHealthChecker:
    """Test OllamaHealthChecker."""

    async def test_healthy_ollama(self):
        """Test healthy Ollama service."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "qwen2.5:0.5b"},
                    {"name": "llama2:7b"},
                ]
            }
            mock_get.return_value = mock_response

            checker = OllamaHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.HEALTHY
            assert result.metadata["model_count"] == 2
            assert "qwen2.5:0.5b" in result.metadata["models"]

            await checker.close()

    async def test_unhealthy_ollama_bad_status(self):
        """Test Ollama service with bad status code."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            checker = OllamaHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.UNHEALTHY
            assert "500" in result.message

            await checker.close()

    async def test_unhealthy_ollama_connection_error(self):
        """Test Ollama service connection error."""
        import httpx

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            checker = OllamaHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.UNHEALTHY
            assert "connect" in result.message.lower()

            await checker.close()


@pytest.mark.asyncio
class TestRedisHealthChecker:
    """Test RedisHealthChecker."""

    async def test_healthy_redis(self):
        """Test healthy Redis."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            "redis_version": "7.0.0",
            "uptime_in_seconds": 1000,
        }

        with patch("redis.asyncio.Redis", return_value=mock_redis):
            checker = RedisHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.HEALTHY
            assert result.metadata["redis_version"] == "7.0.0"

            await checker.close()

    async def test_unhealthy_redis_ping_failed(self):
        """Test Redis ping failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = False

        with patch("redis.asyncio.Redis", return_value=mock_redis):
            checker = RedisHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.UNHEALTHY
            assert "ping failed" in result.message.lower()

            await checker.close()

    async def test_unhealthy_redis_exception(self):
        """Test Redis exception."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection error")

        with patch("redis.asyncio.Redis", return_value=mock_redis):
            checker = RedisHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.UNHEALTHY
            assert "Connection error" in result.message

            await checker.close()


@pytest.mark.asyncio
class TestSQLiteHealthChecker:
    """Test SQLiteHealthChecker."""

    async def test_healthy_sqlite(self, tmp_path):
        """Test healthy SQLite database."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")

        # Create a test database
        async with aiosqlite.connect(db_path) as db:
            await db.execute("CREATE TABLE test (id INTEGER)")
            await db.commit()

        checker = SQLiteHealthChecker(db_path)
        result = await checker.check()

        assert result.status == HealthStatus.HEALTHY
        assert result.metadata["size_bytes"] > 0

    async def test_unhealthy_sqlite_bad_path(self):
        """Test SQLite with non-existent path."""
        checker = SQLiteHealthChecker("/nonexistent/path/db.sqlite")
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
class TestTelemetryHealthChecker:
    """Test TelemetryHealthChecker."""

    async def test_telemetry_disabled(self):
        """Test when telemetry is disabled."""
        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=False):
            checker = TelemetryHealthChecker()
            result = await checker.check()

            assert result.status == HealthStatus.DEGRADED
            assert "disabled" in result.message.lower()

    async def test_telemetry_enabled(self):
        """Test when telemetry is enabled."""
        with patch("tinyllm.telemetry.is_telemetry_enabled", return_value=True):
            with patch("tinyllm.telemetry.get_current_trace_id", return_value=None):
                checker = TelemetryHealthChecker()
                result = await checker.check()

                assert result.status == HealthStatus.HEALTHY


@pytest.mark.asyncio
class TestHealthMonitor:
    """Test HealthMonitor."""

    async def test_register_unregister_checker(self):
        """Test registering and unregistering checkers."""
        monitor = HealthMonitor()
        checker = MockHealthChecker("test_service")

        monitor.register_checker(checker)
        assert "test_service" in monitor._checkers

        monitor.unregister_checker("test_service")
        assert "test_service" not in monitor._checkers

    async def test_check_all(self):
        """Test checking all dependencies."""
        monitor = HealthMonitor()

        checker1 = MockHealthChecker("service1")
        checker2 = MockHealthChecker("service2")

        monitor.register_checker(checker1)
        monitor.register_checker(checker2)

        results = await monitor.check_all()

        assert len(results) == 2
        assert "service1" in results
        assert "service2" in results
        assert results["service1"].status == HealthStatus.HEALTHY
        assert results["service2"].status == HealthStatus.HEALTHY

    async def test_overall_status_healthy(self):
        """Test overall status when all healthy."""
        monitor = HealthMonitor()

        checker1 = MockHealthChecker("service1")
        checker2 = MockHealthChecker("service2")

        monitor.register_checker(checker1)
        monitor.register_checker(checker2)

        status = await monitor.get_overall_status()
        assert status == HealthStatus.HEALTHY

    async def test_overall_status_unhealthy(self):
        """Test overall status when one unhealthy."""
        monitor = HealthMonitor()

        checker1 = MockHealthChecker("service1")
        checker2 = MockHealthChecker("service2", should_fail=True)

        monitor.register_checker(checker1)
        monitor.register_checker(checker2)

        # Run checks to trigger failure
        await monitor.check_all()

        status = await monitor.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    async def test_monitoring_lifecycle(self):
        """Test starting and stopping monitoring."""
        config = HealthCheckConfig(interval_seconds=1)
        monitor = HealthMonitor(config)

        checker = MockHealthChecker("service1")
        monitor.register_checker(checker)

        # Start monitoring
        await monitor.start_monitoring()
        assert monitor._monitoring is True

        # Let it run for a bit
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor._monitoring is False

    async def test_close(self):
        """Test closing monitor."""
        monitor = HealthMonitor()
        checker = MockHealthChecker("service1")
        monitor.register_checker(checker)

        await monitor.start_monitoring()
        await monitor.close()

        assert monitor._monitoring is False


@pytest.mark.asyncio
class TestConfigureDefaultHealthChecks:
    """Test configure_default_health_checks function."""

    async def test_configure_all_checks(self):
        """Test configuring all default health checks."""
        # Clear global monitor
        import tinyllm.health

        tinyllm.health._health_monitor = None

        await configure_default_health_checks(
            ollama_host="http://localhost:11434",
            redis_host="localhost",
            redis_port=6379,
            sqlite_path="/tmp/test.db",
            otlp_endpoint="http://localhost:4317",
        )

        monitor = get_health_monitor()

        assert "ollama" in monitor._checkers
        assert "redis" in monitor._checkers
        assert "sqlite" in monitor._checkers
        assert "telemetry" in monitor._checkers

    async def test_configure_selective_checks(self):
        """Test configuring only some health checks."""
        # Clear global monitor
        import tinyllm.health

        tinyllm.health._health_monitor = None

        await configure_default_health_checks(
            ollama_host="http://localhost:11434",
            redis_host=None,  # Skip Redis
            sqlite_path=None,  # Skip SQLite
        )

        monitor = get_health_monitor()

        assert "ollama" in monitor._checkers
        assert "redis" not in monitor._checkers
        assert "sqlite" not in monitor._checkers
        assert "telemetry" in monitor._checkers


# ============================================================================
# Node-level Circuit Breaker Tests
# ============================================================================


class TestNodeHealthTrackerConfig:
    """Test NodeHealthTrackerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = NodeHealthTrackerConfig()

        assert config.failure_threshold == 3
        assert config.cooldown_seconds == 60.0
        assert config.success_threshold == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = NodeHealthTrackerConfig(
            failure_threshold=5,
            cooldown_seconds=120.0,
            success_threshold=3,
        )

        assert config.failure_threshold == 5
        assert config.cooldown_seconds == 120.0
        assert config.success_threshold == 3


class TestNodeHealthTracker:
    """Test NodeHealthTracker circuit breaker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = NodeHealthTracker()

        assert tracker.config.failure_threshold == 3
        assert tracker.config.cooldown_seconds == 60.0
        assert tracker._total_failures == 0
        assert tracker._total_successes == 0

    def test_record_success(self):
        """Test recording node success."""
        tracker = NodeHealthTracker()

        tracker.record_success("node1")

        assert tracker._success_counts["node1"] == 1
        assert tracker._failure_counts["node1"] == 0
        assert tracker._total_successes == 1

    def test_record_failure(self):
        """Test recording node failure."""
        tracker = NodeHealthTracker()

        tracker.record_failure("node1", error="Test error")

        assert tracker._failure_counts["node1"] == 1
        assert tracker._success_counts["node1"] == 0
        assert tracker._total_failures == 1

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = NodeHealthTrackerConfig(failure_threshold=3)
        tracker = NodeHealthTracker(config)

        # Record failures
        tracker.record_failure("node1")
        tracker.record_failure("node1")
        assert tracker.is_healthy("node1")  # Still healthy

        tracker.record_failure("node1")  # Should open circuit
        assert not tracker.is_healthy("node1")  # Circuit open

    def test_circuit_closes_after_success(self):
        """Test circuit closes after success threshold."""
        import time

        config = NodeHealthTrackerConfig(
            failure_threshold=2,
            success_threshold=2,
            cooldown_seconds=0.1,  # Short cooldown for testing
        )
        tracker = NodeHealthTracker(config)

        # Open circuit
        tracker.record_failure("node1")
        tracker.record_failure("node1")
        assert not tracker.is_healthy("node1")

        # Wait for cooldown
        time.sleep(0.15)

        # Should allow retry
        assert tracker.is_healthy("node1")

        # Record successes
        tracker.record_success("node1")
        tracker.record_success("node1")

        # Circuit should be closed
        assert "node1" not in tracker._circuit_breakers

    def test_cooldown_period(self):
        """Test cooldown period prevents retry."""
        import time

        config = NodeHealthTrackerConfig(
            failure_threshold=2,
            cooldown_seconds=0.2,  # 200ms cooldown
        )
        tracker = NodeHealthTracker(config)

        # Open circuit
        tracker.record_failure("node1")
        tracker.record_failure("node1")

        # Circuit should be open
        assert not tracker.is_healthy("node1")

        # Still in cooldown
        time.sleep(0.1)
        assert not tracker.is_healthy("node1")

        # After cooldown
        time.sleep(0.15)
        assert tracker.is_healthy("node1")

    def test_success_resets_failure_count(self):
        """Test that success resets failure count."""
        tracker = NodeHealthTracker()

        tracker.record_failure("node1")
        tracker.record_failure("node1")
        assert tracker._failure_counts["node1"] == 2

        tracker.record_success("node1")
        assert tracker._failure_counts["node1"] == 0
        assert tracker._success_counts["node1"] == 1

    def test_failure_resets_success_count(self):
        """Test that failure resets success count."""
        tracker = NodeHealthTracker()

        tracker.record_success("node1")
        tracker.record_success("node1")
        assert tracker._success_counts["node1"] == 2

        tracker.record_failure("node1")
        assert tracker._success_counts["node1"] == 0
        assert tracker._failure_counts["node1"] == 1

    def test_get_node_status(self):
        """Test getting node status."""
        tracker = NodeHealthTracker()

        tracker.record_failure("node1")
        status = tracker.get_node_status("node1")

        assert status.node_id == "node1"
        assert status.is_healthy is True  # Only 1 failure
        assert status.failure_count == 1
        assert status.success_count == 0
        assert status.circuit_open is False

    def test_get_all_statuses(self):
        """Test getting all node statuses."""
        tracker = NodeHealthTracker()

        tracker.record_failure("node1")
        tracker.record_success("node2")

        statuses = tracker.get_all_statuses()

        assert len(statuses) == 2
        assert "node1" in statuses
        assert "node2" in statuses

    def test_get_unhealthy_nodes(self):
        """Test getting unhealthy nodes."""
        config = NodeHealthTrackerConfig(failure_threshold=2)
        tracker = NodeHealthTracker(config)

        tracker.record_failure("node1")
        tracker.record_failure("node1")  # Opens circuit
        tracker.record_success("node2")

        unhealthy = tracker.get_unhealthy_nodes()

        assert len(unhealthy) == 1
        assert "node1" in unhealthy
        assert "node2" not in unhealthy

    def test_reset_node(self):
        """Test resetting a specific node."""
        tracker = NodeHealthTracker()

        tracker.record_failure("node1")
        tracker.record_success("node2")

        tracker.reset_node("node1")

        assert tracker._failure_counts["node1"] == 0
        assert tracker._success_counts["node2"] == 1  # Node2 unchanged

    def test_reset_all(self):
        """Test resetting all nodes."""
        tracker = NodeHealthTracker()

        tracker.record_failure("node1")
        tracker.record_success("node2")

        tracker.reset_all()

        assert len(tracker._failure_counts) == 0
        assert len(tracker._success_counts) == 0
        assert tracker._total_failures == 0
        assert tracker._total_successes == 0

    def test_get_stats(self):
        """Test getting tracker statistics."""
        config = NodeHealthTrackerConfig(failure_threshold=2)
        tracker = NodeHealthTracker(config)

        tracker.record_failure("node1")
        tracker.record_failure("node1")  # Opens circuit
        tracker.record_success("node2")

        stats = tracker.get_stats()

        assert stats["total_failures"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_circuit_breaks"] == 1
        assert stats["currently_open_circuits"] == 1
        assert stats["tracked_nodes"] == 2
        assert stats["failure_threshold"] == 2

    def test_multiple_nodes_independent(self):
        """Test that different nodes are tracked independently."""
        config = NodeHealthTrackerConfig(failure_threshold=2)
        tracker = NodeHealthTracker(config)

        # Node1 fails
        tracker.record_failure("node1")
        tracker.record_failure("node1")

        # Node2 succeeds
        tracker.record_success("node2")

        assert not tracker.is_healthy("node1")  # Circuit open
        assert tracker.is_healthy("node2")  # Still healthy

    def test_global_tracker_singleton(self):
        """Test global tracker singleton."""
        reset_node_health_tracker()

        tracker1 = get_node_health_tracker()
        tracker2 = get_node_health_tracker()

        assert tracker1 is tracker2

    def test_global_tracker_reset(self):
        """Test resetting global tracker."""
        reset_node_health_tracker()

        tracker1 = get_node_health_tracker()
        tracker1.record_failure("node1")

        reset_node_health_tracker()

        tracker2 = get_node_health_tracker()
        assert tracker1 is not tracker2
        assert tracker2._total_failures == 0
