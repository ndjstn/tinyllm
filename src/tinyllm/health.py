"""Health check system for TinyLLM dependencies.

This module provides comprehensive health checks for all external dependencies
including Ollama, Redis, databases, and other services.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="health")


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DependencyType(str, Enum):
    """Type of dependency."""

    LLM_SERVICE = "llm_service"
    CACHE = "cache"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    TELEMETRY = "telemetry"
    EXTERNAL_API = "external_api"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class HealthCheckConfig(BaseModel):
    """Configuration for health checks."""

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable health checks")
    interval_seconds: int = Field(default=30, ge=1, description="Check interval")
    timeout_seconds: int = Field(default=5, ge=1, description="Check timeout")
    failure_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures before unhealthy",
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        description="Consecutive successes before healthy",
    )


class HealthChecker:
    """Base health checker class."""

    def __init__(
        self,
        name: str,
        dependency_type: DependencyType,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize health checker.

        Args:
            name: Name of the dependency.
            dependency_type: Type of dependency.
            config: Health check configuration.
        """
        self.name = name
        self.dependency_type = dependency_type
        self.config = config or HealthCheckConfig()
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_status = HealthStatus.UNKNOWN
        self._last_check: Optional[HealthCheckResult] = None

    async def check(self) -> HealthCheckResult:
        """Perform health check.

        Returns:
            Health check result.
        """
        start_time = time.perf_counter()

        try:
            # Run the actual check with timeout
            result = await asyncio.wait_for(
                self._check_impl(),
                timeout=self.config.timeout_seconds,
            )

            # Calculate response time
            result.response_time_ms = (time.perf_counter() - start_time) * 1000

            # Update consecutive counts
            if result.status == HealthStatus.HEALTHY:
                self._consecutive_successes += 1
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                self._consecutive_successes = 0

            # Update status based on thresholds
            if self._consecutive_failures >= self.config.failure_threshold:
                result.status = HealthStatus.UNHEALTHY
            elif self._consecutive_successes >= self.config.success_threshold:
                result.status = HealthStatus.HEALTHY
            elif self._last_status == HealthStatus.HEALTHY:
                result.status = HealthStatus.DEGRADED

            self._last_status = result.status
            self._last_check = result

            logger.info(
                "health_check_completed",
                name=self.name,
                status=result.status.value,
                response_time_ms=result.response_time_ms,
            )

            return result

        except asyncio.TimeoutError:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {self.config.timeout_seconds}s",
                response_time_ms=(time.perf_counter() - start_time) * 1000,
            )

            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_status = HealthStatus.UNHEALTHY
            self._last_check = result

            logger.warning(
                "health_check_timeout",
                name=self.name,
                timeout_seconds=self.config.timeout_seconds,
            )

            return result

        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                response_time_ms=(time.perf_counter() - start_time) * 1000,
            )

            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_status = HealthStatus.UNHEALTHY
            self._last_check = result

            logger.error(
                "health_check_error",
                name=self.name,
                error=str(e),
                exc_info=True,
            )

            return result

    async def _check_impl(self) -> HealthCheckResult:
        """Implementation of health check. Override in subclasses.

        Returns:
            Health check result.
        """
        raise NotImplementedError

    def get_last_result(self) -> Optional[HealthCheckResult]:
        """Get the last health check result.

        Returns:
            Last health check result or None.
        """
        return self._last_check


class OllamaHealthChecker(HealthChecker):
    """Health checker for Ollama service."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize Ollama health checker.

        Args:
            host: Ollama server URL.
            config: Health check configuration.
        """
        super().__init__("ollama", DependencyType.LLM_SERVICE, config)
        self.host = host
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._client

    async def _check_impl(self) -> HealthCheckResult:
        """Check Ollama service health."""
        client = await self._get_client()

        # Try to list models as a health check
        try:
            response = await client.get(f"{self.host}/api/tags")

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Ollama service is healthy, {len(models)} models available",
                    metadata={
                        "host": self.host,
                        "model_count": len(models),
                        "models": [m.get("name", "unknown") for m in models[:5]],
                    },
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Ollama service returned status {response.status_code}",
                    metadata={"host": self.host, "status_code": response.status_code},
                )

        except httpx.ConnectError as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Cannot connect to Ollama service: {str(e)}",
                metadata={"host": self.host, "error": "connection_refused"},
            )

    async def close(self):
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class RedisHealthChecker(HealthChecker):
    """Health checker for Redis cache."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize Redis health checker.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            config: Health check configuration.
        """
        super().__init__("redis", DependencyType.CACHE, config)
        self.host = host
        self.port = port
        self.db = db
        self._redis = None

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=True,
                    socket_connect_timeout=self.config.timeout_seconds,
                )
            except ImportError:
                raise ImportError(
                    "redis package is required for Redis health checks. "
                    "Install with: pip install redis"
                )

        return self._redis

    async def _check_impl(self) -> HealthCheckResult:
        """Check Redis health."""
        try:
            redis = await self._get_redis()

            # Ping Redis
            pong = await redis.ping()

            if not pong:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed",
                    metadata={"host": self.host, "port": self.port, "db": self.db},
                )

            # Get server info
            info = await redis.info("server")

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Redis is healthy",
                metadata={
                    "host": self.host,
                    "port": self.port,
                    "db": self.db,
                    "redis_version": info.get("redis_version", "unknown"),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis health check failed: {str(e)}",
                metadata={"host": self.host, "port": self.port, "db": self.db},
            )

    async def close(self):
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


class SQLiteHealthChecker(HealthChecker):
    """Health checker for SQLite database."""

    def __init__(
        self,
        db_path: str,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize SQLite health checker.

        Args:
            db_path: Path to SQLite database.
            config: Health check configuration.
        """
        super().__init__("sqlite", DependencyType.DATABASE, config)
        self.db_path = db_path

    async def _check_impl(self) -> HealthCheckResult:
        """Check SQLite database health."""
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                # Run a simple query
                cursor = await db.execute("SELECT 1")
                result = await cursor.fetchone()

                if result and result[0] == 1:
                    # Get database size
                    import os
                    size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="SQLite database is healthy",
                        metadata={
                            "db_path": self.db_path,
                            "size_bytes": size_bytes,
                            "size_mb": round(size_bytes / 1024 / 1024, 2),
                        },
                    )
                else:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.UNHEALTHY,
                        message="SQLite query returned unexpected result",
                        metadata={"db_path": self.db_path},
                    )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"SQLite health check failed: {str(e)}",
                metadata={"db_path": self.db_path},
            )


class TelemetryHealthChecker(HealthChecker):
    """Health checker for OpenTelemetry exporter."""

    def __init__(
        self,
        otlp_endpoint: Optional[str] = None,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize telemetry health checker.

        Args:
            otlp_endpoint: OTLP exporter endpoint.
            config: Health check configuration.
        """
        super().__init__("telemetry", DependencyType.TELEMETRY, config)
        self.otlp_endpoint = otlp_endpoint

    async def _check_impl(self) -> HealthCheckResult:
        """Check telemetry exporter health."""
        try:
            from tinyllm.telemetry import is_telemetry_enabled, get_current_trace_id

            if not is_telemetry_enabled():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Telemetry is disabled",
                    metadata={"enabled": False},
                )

            # Check if we can create spans
            trace_id = get_current_trace_id()

            # If OTLP endpoint is configured, check connectivity
            if self.otlp_endpoint:
                # Parse endpoint URL
                import httpx

                # Try to connect to OTLP endpoint
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    try:
                        # Most OTLP endpoints don't have a health endpoint,
                        # so we just check if we can connect
                        response = await client.get(self.otlp_endpoint)
                        # Any response is good (even 404)
                    except httpx.ConnectError:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message="Cannot connect to OTLP endpoint",
                            metadata={"endpoint": self.otlp_endpoint},
                        )

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Telemetry is healthy",
                metadata={
                    "enabled": True,
                    "otlp_endpoint": self.otlp_endpoint,
                    "in_trace_context": trace_id is not None,
                },
            )

        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="OpenTelemetry packages not installed",
                metadata={"enabled": False},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Telemetry health check failed: {str(e)}",
            )


class HealthMonitor:
    """Monitors health of all dependencies."""

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        """Initialize health monitor.

        Args:
            config: Default health check configuration.
        """
        self.config = config or HealthCheckConfig()
        self._checkers: dict[str, HealthChecker] = {}
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    def register_checker(self, checker: HealthChecker):
        """Register a health checker.

        Args:
            checker: Health checker to register.
        """
        self._checkers[checker.name] = checker
        logger.info("health_checker_registered", name=checker.name, type=checker.dependency_type.value)

    def unregister_checker(self, name: str):
        """Unregister a health checker.

        Args:
            name: Name of checker to unregister.
        """
        if name in self._checkers:
            del self._checkers[name]
            logger.info("health_checker_unregistered", name=name)

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Check health of all registered dependencies.

        Returns:
            Dictionary mapping dependency name to health check result.
        """
        if not self._checkers:
            logger.warning("health_check_no_checkers_registered")
            return {}

        results = {}

        # Run all checks concurrently
        tasks = {
            name: checker.check()
            for name, checker in self._checkers.items()
        }

        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for name, result in zip(tasks.keys(), completed):
            if isinstance(result, Exception):
                logger.error(
                    "health_check_failed",
                    name=name,
                    error=str(result),
                    exc_info=result,
                )
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                )
            else:
                results[name] = result

        return results

    async def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.

        Returns:
            Overall health status based on all dependencies.
        """
        results = await self.check_all()

        if not results:
            return HealthStatus.UNKNOWN

        # Overall status is the worst status of all dependencies
        statuses = [result.status for result in results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            logger.warning("health_monitoring_already_started")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "health_monitoring_started",
            interval_seconds=self.config.interval_seconds,
            checker_count=len(self._checkers),
        )

    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("health_monitoring_stopped")

    async def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                results = await self.check_all()

                # Log summary
                healthy = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
                degraded = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
                unhealthy = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)

                logger.info(
                    "health_monitoring_summary",
                    total=len(results),
                    healthy=healthy,
                    degraded=degraded,
                    unhealthy=unhealthy,
                )

                await asyncio.sleep(self.config.interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "health_monitoring_error",
                    error=str(e),
                    exc_info=True,
                )
                await asyncio.sleep(self.config.interval_seconds)

    async def close(self):
        """Close all health checkers."""
        await self.stop_monitoring()

        for checker in self._checkers.values():
            if hasattr(checker, "close"):
                await checker.close()


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance.

    Returns:
        Global health monitor.
    """
    global _health_monitor

    if _health_monitor is None:
        _health_monitor = HealthMonitor()

    return _health_monitor


async def configure_default_health_checks(
    ollama_host: str = "http://localhost:11434",
    redis_host: Optional[str] = None,
    redis_port: int = 6379,
    sqlite_path: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    config: Optional[HealthCheckConfig] = None,
):
    """Configure default health checks for common dependencies.

    Args:
        ollama_host: Ollama server URL.
        redis_host: Redis host (if None, Redis check is skipped).
        redis_port: Redis port.
        sqlite_path: SQLite database path (if None, SQLite check is skipped).
        otlp_endpoint: OTLP endpoint (if None, connectivity check is skipped).
        config: Health check configuration.
    """
    monitor = get_health_monitor()

    # Ollama health check
    ollama_checker = OllamaHealthChecker(host=ollama_host, config=config)
    monitor.register_checker(ollama_checker)

    # Redis health check (if configured)
    if redis_host:
        redis_checker = RedisHealthChecker(
            host=redis_host,
            port=redis_port,
            config=config,
        )
        monitor.register_checker(redis_checker)

    # SQLite health check (if configured)
    if sqlite_path:
        sqlite_checker = SQLiteHealthChecker(db_path=sqlite_path, config=config)
        monitor.register_checker(sqlite_checker)

    # Telemetry health check
    telemetry_checker = TelemetryHealthChecker(
        otlp_endpoint=otlp_endpoint,
        config=config,
    )
    monitor.register_checker(telemetry_checker)

    logger.info(
        "health_checks_configured",
        checker_count=len(monitor._checkers),
        checkers=list(monitor._checkers.keys()),
    )


# ============================================================================
# Node-level Circuit Breaker (for graph execution)
# ============================================================================


@dataclass
class NodeHealthStatus:
    """Health status for a graph execution node."""

    node_id: str
    is_healthy: bool
    failure_count: int
    success_count: int
    circuit_open: bool
    cooldown_until: Optional[float] = None
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


@dataclass
class NodeHealthTrackerConfig:
    """Configuration for NodeHealthTracker."""

    failure_threshold: int = 3
    """Number of consecutive failures before opening circuit."""

    cooldown_seconds: float = 60.0
    """Cooldown period before attempting to use node again."""

    success_threshold: int = 2
    """Number of consecutive successes to close circuit."""

    health_check_interval: float = 300.0
    """Interval for resetting health stats (seconds)."""


class NodeHealthTracker:
    """Track node health and implement circuit breaker pattern for graph execution.

    The circuit breaker pattern prevents cascading failures by:
    1. Tracking failure rates for each node in the graph
    2. Opening circuit (blocking requests) after threshold failures
    3. Implementing cooldown period before retry
    4. Closing circuit after successful recovery

    This is specifically for graph execution nodes, not external dependencies.
    For dependency health checks, use HealthMonitor instead.

    Usage:
        tracker = NodeHealthTracker()

        # Before executing node
        if not tracker.is_healthy(node_id):
            # Skip node or find alternative path
            pass

        # After execution
        if success:
            tracker.record_success(node_id)
        else:
            tracker.record_failure(node_id)
    """

    def __init__(self, config: Optional[NodeHealthTrackerConfig] = None):
        """Initialize node health tracker.

        Args:
            config: Configuration for health tracking behavior
        """
        self.config = config or NodeHealthTrackerConfig()

        # Tracking state
        from collections import defaultdict

        self._failure_counts: dict[str, int] = defaultdict(int)
        self._success_counts: dict[str, int] = defaultdict(int)
        self._circuit_breakers: dict[str, float] = {}  # node_id → cooldown_until
        self._last_failure_time: dict[str, float] = {}
        self._last_success_time: dict[str, float] = {}

        # Stats
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._circuit_breaks: int = 0
        self._last_reset_time: float = time.time()

    def record_failure(self, node_id: str, error: Optional[str] = None) -> None:
        """Record a failure for a node.

        Args:
            node_id: ID of the node that failed
            error: Optional error message for logging
        """
        current_time = time.time()

        # Increment failure count
        self._failure_counts[node_id] += 1
        self._total_failures += 1
        self._last_failure_time[node_id] = current_time

        # Reset success count on failure
        self._success_counts[node_id] = 0

        logger.warning(
            "node_failure_recorded",
            node_id=node_id,
            failure_count=self._failure_counts[node_id],
            threshold=self.config.failure_threshold,
            error=error,
        )

        # Check if we should open circuit
        if self._failure_counts[node_id] >= self.config.failure_threshold:
            self._open_circuit(node_id)

    def record_success(self, node_id: str) -> None:
        """Record a success for a node.

        Args:
            node_id: ID of the node that succeeded
        """
        current_time = time.time()

        # Increment success count
        self._success_counts[node_id] += 1
        self._total_successes += 1
        self._last_success_time[node_id] = current_time

        # Reset failure count on success
        self._failure_counts[node_id] = 0

        logger.debug(
            "node_success_recorded",
            node_id=node_id,
            success_count=self._success_counts[node_id],
        )

        # Check if we should close circuit
        if node_id in self._circuit_breakers:
            if self._success_counts[node_id] >= self.config.success_threshold:
                self._close_circuit(node_id)

    def is_healthy(self, node_id: str) -> bool:
        """Check if a node is healthy and can be used.

        Args:
            node_id: ID of the node to check

        Returns:
            True if node is healthy, False if circuit is open
        """
        # Check if circuit is open
        if node_id in self._circuit_breakers:
            cooldown_until = self._circuit_breakers[node_id]
            current_time = time.time()

            if current_time < cooldown_until:
                # Still in cooldown
                remaining = cooldown_until - current_time
                logger.debug(
                    "node_circuit_open",
                    node_id=node_id,
                    cooldown_remaining_seconds=remaining,
                )
                return False

            # Cooldown expired - allow retry
            logger.info(
                "node_cooldown_expired",
                node_id=node_id,
                attempting_retry=True,
            )
            # Don't remove from circuit breakers yet - wait for success

        return True

    def _open_circuit(self, node_id: str) -> None:
        """Open circuit breaker for a node.

        Args:
            node_id: ID of the node to block
        """
        cooldown_until = time.time() + self.config.cooldown_seconds
        self._circuit_breakers[node_id] = cooldown_until
        self._circuit_breaks += 1

        logger.error(
            "circuit_breaker_opened",
            node_id=node_id,
            failure_count=self._failure_counts[node_id],
            cooldown_seconds=self.config.cooldown_seconds,
            cooldown_until_timestamp=cooldown_until,
        )

    def _close_circuit(self, node_id: str) -> None:
        """Close circuit breaker for a node.

        Args:
            node_id: ID of the node to unblock
        """
        if node_id in self._circuit_breakers:
            del self._circuit_breakers[node_id]

            logger.info(
                "circuit_breaker_closed",
                node_id=node_id,
                success_count=self._success_counts[node_id],
            )

    def get_node_status(self, node_id: str) -> NodeHealthStatus:
        """Get detailed health status for a node.

        Args:
            node_id: ID of the node

        Returns:
            NodeHealthStatus with current state
        """
        circuit_open = node_id in self._circuit_breakers
        cooldown_until = self._circuit_breakers.get(node_id)

        return NodeHealthStatus(
            node_id=node_id,
            is_healthy=self.is_healthy(node_id),
            failure_count=self._failure_counts[node_id],
            success_count=self._success_counts[node_id],
            circuit_open=circuit_open,
            cooldown_until=cooldown_until,
            last_failure_time=self._last_failure_time.get(node_id),
            last_success_time=self._last_success_time.get(node_id),
        )

    def get_all_statuses(self) -> dict[str, NodeHealthStatus]:
        """Get health status for all tracked nodes.

        Returns:
            Dictionary mapping node_id to NodeHealthStatus
        """
        all_nodes = set(self._failure_counts.keys()) | set(self._success_counts.keys())
        return {node_id: self.get_node_status(node_id) for node_id in all_nodes}

    def get_unhealthy_nodes(self) -> list[str]:
        """Get list of currently unhealthy node IDs.

        Returns:
            List of node IDs with open circuits
        """
        return [
            node_id
            for node_id in self._circuit_breakers.keys()
            if not self.is_healthy(node_id)
        ]

    def reset_node(self, node_id: str) -> None:
        """Reset health tracking for a specific node.

        Args:
            node_id: ID of the node to reset
        """
        self._failure_counts.pop(node_id, None)
        self._success_counts.pop(node_id, None)
        self._circuit_breakers.pop(node_id, None)
        self._last_failure_time.pop(node_id, None)
        self._last_success_time.pop(node_id, None)

        logger.info("node_health_reset", node_id=node_id)

    def reset_all(self) -> None:
        """Reset all health tracking state."""
        self._failure_counts.clear()
        self._success_counts.clear()
        self._circuit_breakers.clear()
        self._last_failure_time.clear()
        self._last_success_time.clear()
        self._total_failures = 0
        self._total_successes = 0
        self._circuit_breaks = 0
        self._last_reset_time = time.time()

        logger.info("all_node_health_tracking_reset")

    def get_stats(self) -> dict[str, Any]:
        """Get overall node health tracking statistics.

        Returns:
            Dictionary with tracking stats
        """
        current_time = time.time()
        uptime = current_time - self._last_reset_time

        return {
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_circuit_breaks": self._circuit_breaks,
            "currently_open_circuits": len(self._circuit_breakers),
            "tracked_nodes": len(
                set(self._failure_counts.keys()) | set(self._success_counts.keys())
            ),
            "unhealthy_nodes": len(self.get_unhealthy_nodes()),
            "uptime_seconds": uptime,
            "failure_threshold": self.config.failure_threshold,
            "cooldown_seconds": self.config.cooldown_seconds,
            "success_threshold": self.config.success_threshold,
        }


# Global node health tracker instance
_global_node_health_tracker: Optional[NodeHealthTracker] = None


def get_node_health_tracker(
    config: Optional[NodeHealthTrackerConfig] = None,
) -> NodeHealthTracker:
    """Get or create global node health tracker instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        Global NodeHealthTracker instance
    """
    global _global_node_health_tracker

    if _global_node_health_tracker is None:
        _global_node_health_tracker = NodeHealthTracker(config)

    return _global_node_health_tracker


def reset_node_health_tracker() -> None:
    """Reset global node health tracker (mainly for testing)."""
    global _global_node_health_tracker
    _global_node_health_tracker = None


__all__ = [
    # Dependency health checks
    "HealthStatus",
    "DependencyType",
    "HealthCheckResult",
    "HealthCheckConfig",
    "HealthChecker",
    "OllamaHealthChecker",
    "RedisHealthChecker",
    "SQLiteHealthChecker",
    "TelemetryHealthChecker",
    "HealthMonitor",
    "get_health_monitor",
    "configure_default_health_checks",
    # Node-level circuit breaker
    "NodeHealthStatus",
    "NodeHealthTrackerConfig",
    "NodeHealthTracker",
    "get_node_health_tracker",
    "reset_node_health_tracker",
]
