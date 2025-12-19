"""Horizontal scaling for distributed TinyLLM workloads.

This module provides horizontal scaling capabilities including:
- Worker instance management and health tracking
- Dynamic scaling policies (CPU, memory, request-based)
- Instance lifecycle management (starting, draining, termination)
- Health monitoring and automatic failover
- Scaling metrics and history tracking

These features enable elastic scaling of TinyLLM deployments across
multiple worker instances.
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="horizontal")


class InstanceStatus(str, Enum):
    """Status of a worker instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    STARTING = "starting"


class ScalingPolicy(str, Enum):
    """Scaling policy type."""

    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    CUSTOM = "custom"


class ScalingDirection(str, Enum):
    """Direction of scaling operation."""

    UP = "up"
    DOWN = "down"


class WorkerInstance(BaseModel):
    """Represents a worker instance in the cluster.

    Tracks instance metadata, health status, and current load.
    """

    model_config = {"extra": "allow"}

    instance_id: str = Field(description="Unique instance identifier")
    host: str = Field(description="Instance host/IP address")
    port: int = Field(ge=1, le=65535, description="Instance port number")
    status: InstanceStatus = Field(
        default=InstanceStatus.STARTING, description="Current instance status"
    )
    active_requests: int = Field(default=0, ge=0, description="Number of active requests")
    total_requests: int = Field(default=0, ge=0, description="Total requests processed")
    last_health_check: Optional[datetime] = Field(
        default=None, description="Last health check timestamp"
    )
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Instance start time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def can_accept_requests(self) -> bool:
        """Check if instance can accept new requests.

        Returns:
            True if instance is healthy and can accept requests.
        """
        return self.status == InstanceStatus.HEALTHY

    def increment_requests(self) -> None:
        """Increment request counters."""
        self.active_requests += 1
        self.total_requests += 1

    def decrement_requests(self) -> None:
        """Decrement active request counter."""
        if self.active_requests > 0:
            self.active_requests -= 1

    def update_health(self, is_healthy: bool) -> None:
        """Update instance health status.

        Args:
            is_healthy: Whether the instance is healthy.
        """
        self.last_health_check = datetime.utcnow()
        if is_healthy and self.status != InstanceStatus.DRAINING:
            self.status = InstanceStatus.HEALTHY
        elif not is_healthy:
            self.status = InstanceStatus.UNHEALTHY


class ScalingConfig(BaseModel):
    """Configuration for horizontal scaling."""

    model_config = {"extra": "forbid"}

    min_instances: int = Field(default=1, ge=0, description="Minimum number of instances")
    max_instances: int = Field(default=10, ge=1, description="Maximum number of instances")
    target_cpu_utilization: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Target CPU utilization (0.0-1.0)"
    )
    target_memory_utilization: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Target memory utilization (0.0-1.0)"
    )
    target_requests_per_instance: int = Field(
        default=100, ge=1, description="Target requests per instance"
    )
    scale_up_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Threshold to trigger scale up"
    )
    scale_down_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Threshold to trigger scale down"
    )
    cooldown_seconds: int = Field(
        default=60, ge=0, description="Cooldown period between scaling operations"
    )
    health_check_interval: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )
    health_check_timeout: int = Field(
        default=5, ge=1, description="Health check timeout in seconds"
    )
    unhealthy_threshold: int = Field(
        default=3, ge=1, description="Consecutive failures before marking unhealthy"
    )
    drain_timeout_seconds: int = Field(
        default=300, ge=0, description="Timeout for draining connections"
    )


class ScalingMetrics(BaseModel):
    """Metrics for scaling decisions."""

    model_config = {"extra": "forbid"}

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_instances: int = Field(default=0, ge=0)
    healthy_instances: int = Field(default=0, ge=0)
    draining_instances: int = Field(default=0, ge=0)
    unhealthy_instances: int = Field(default=0, ge=0)
    total_active_requests: int = Field(default=0, ge=0)
    average_requests_per_instance: float = Field(default=0.0, ge=0.0)
    cpu_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_utilization: float = Field(default=0.0, ge=0.0, le=1.0)


class ScalingEvent(BaseModel):
    """Record of a scaling event."""

    model_config = {"extra": "forbid"}

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    direction: ScalingDirection
    previous_count: int = Field(ge=0)
    new_count: int = Field(ge=0)
    reason: str
    metrics: Optional[ScalingMetrics] = None


class HorizontalScaler:
    """Manages horizontal scaling of worker instances.

    Provides dynamic scaling based on configurable policies, health monitoring,
    and graceful instance lifecycle management.
    """

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        instance_factory: Optional[Callable[[str], WorkerInstance]] = None,
        health_checker: Optional[Callable[[WorkerInstance], bool]] = None,
    ):
        """Initialize horizontal scaler.

        Args:
            config: Scaling configuration.
            instance_factory: Factory function to create new instances.
            health_checker: Function to check instance health.
        """
        self.config = config or ScalingConfig()
        self.instance_factory = instance_factory
        self.health_checker = health_checker

        self.instances: Dict[str, WorkerInstance] = {}
        self.scaling_history: List[ScalingEvent] = []

        self._last_scale_time: Optional[datetime] = None
        self._shutdown = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._instance_counter = 0

        logger.info(
            "horizontal_scaler_initialized",
            min_instances=self.config.min_instances,
            max_instances=self.config.max_instances,
        )

    async def start(self) -> None:
        """Start the horizontal scaler."""
        self._shutdown = False

        # Start with minimum instances
        for _ in range(self.config.min_instances):
            await self.add_instance()

        # Start health checking
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("horizontal_scaler_started", instances=len(self.instances))

    async def stop(self) -> None:
        """Stop the horizontal scaler."""
        self._shutdown = True

        # Cancel health checking
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all instances
        for instance_id in list(self.instances.keys()):
            await self.remove_instance(instance_id, graceful=True)

        logger.info("horizontal_scaler_stopped")

    async def add_instance(
        self, instance: Optional[WorkerInstance] = None
    ) -> Optional[WorkerInstance]:
        """Add a new worker instance.

        Args:
            instance: Optional pre-configured instance. If not provided, uses factory.

        Returns:
            The added instance, or None if max instances reached.
        """
        if len(self.instances) >= self.config.max_instances:
            logger.warning(
                "max_instances_reached",
                current=len(self.instances),
                max=self.config.max_instances,
            )
            return None

        # Create or use provided instance
        if instance is None:
            if self.instance_factory:
                instance_id = f"instance-{self._instance_counter}"
                self._instance_counter += 1
                instance = self.instance_factory(instance_id)
            else:
                # Create default instance
                instance_id = f"instance-{self._instance_counter}"
                self._instance_counter += 1
                instance = WorkerInstance(
                    instance_id=instance_id,
                    host="localhost",
                    port=8000 + self._instance_counter,
                    status=InstanceStatus.STARTING,
                )

        self.instances[instance.instance_id] = instance

        logger.info(
            "instance_added",
            instance_id=instance.instance_id,
            host=instance.host,
            port=instance.port,
            total_instances=len(self.instances),
        )

        return instance

    async def remove_instance(
        self, instance_id: str, graceful: bool = True
    ) -> Optional[WorkerInstance]:
        """Remove a worker instance.

        Args:
            instance_id: ID of instance to remove.
            graceful: Whether to drain connections before removing.

        Returns:
            The removed instance, or None if not found.
        """
        instance = self.instances.get(instance_id)
        if not instance:
            logger.warning("instance_not_found", instance_id=instance_id)
            return None

        if graceful:
            # Mark as draining
            instance.status = InstanceStatus.DRAINING
            logger.info("instance_draining", instance_id=instance_id)

            # Wait for active requests to complete
            start_time = time.time()
            while (
                instance.active_requests > 0
                and time.time() - start_time < self.config.drain_timeout_seconds
            ):
                await asyncio.sleep(1)

            if instance.active_requests > 0:
                logger.warning(
                    "drain_timeout",
                    instance_id=instance_id,
                    remaining_requests=instance.active_requests,
                )

        # Remove instance
        del self.instances[instance_id]

        logger.info(
            "instance_removed",
            instance_id=instance_id,
            total_instances=len(self.instances),
        )

        return instance

    async def scale_to(self, target_count: int, reason: str = "manual") -> None:
        """Scale to a specific number of instances.

        Args:
            target_count: Desired number of instances.
            reason: Reason for scaling operation.
        """
        current_count = len(self.instances)

        # Enforce limits
        target_count = max(self.config.min_instances, min(target_count, self.config.max_instances))

        if target_count == current_count:
            logger.debug("already_at_target", current=current_count, target=target_count)
            return

        # Determine direction
        direction = ScalingDirection.UP if target_count > current_count else ScalingDirection.DOWN

        # Check cooldown
        if self._last_scale_time:
            elapsed = (datetime.utcnow() - self._last_scale_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                logger.info(
                    "scaling_in_cooldown",
                    elapsed=elapsed,
                    cooldown=self.config.cooldown_seconds,
                )
                return

        logger.info(
            "scaling",
            direction=direction.value,
            current=current_count,
            target=target_count,
            reason=reason,
        )

        # Perform scaling
        if direction == ScalingDirection.UP:
            # Scale up
            for _ in range(target_count - current_count):
                await self.add_instance()
        else:
            # Scale down - remove least loaded instances
            to_remove = current_count - target_count
            instances_by_load = sorted(
                self.instances.values(),
                key=lambda i: i.active_requests,
            )
            for instance in instances_by_load[:to_remove]:
                await self.remove_instance(instance.instance_id, graceful=True)

        # Record event
        event = ScalingEvent(
            direction=direction,
            previous_count=current_count,
            new_count=len(self.instances),
            reason=reason,
            metrics=self.get_metrics(),
        )
        self.scaling_history.append(event)
        self._last_scale_time = datetime.utcnow()

        logger.info(
            "scaling_complete",
            direction=direction.value,
            previous=current_count,
            current=len(self.instances),
        )

    async def scale_up(self, count: int = 1, reason: str = "manual") -> None:
        """Scale up by adding instances.

        Args:
            count: Number of instances to add.
            reason: Reason for scaling.
        """
        target = len(self.instances) + count
        await self.scale_to(target, reason)

    async def scale_down(self, count: int = 1, reason: str = "manual") -> None:
        """Scale down by removing instances.

        Args:
            count: Number of instances to remove.
            reason: Reason for scaling.
        """
        target = len(self.instances) - count
        await self.scale_to(target, reason)

    def get_instance(self, instance_id: str) -> Optional[WorkerInstance]:
        """Get an instance by ID.

        Args:
            instance_id: Instance identifier.

        Returns:
            The instance, or None if not found.
        """
        return self.instances.get(instance_id)

    def get_healthy_instances(self) -> List[WorkerInstance]:
        """Get all healthy instances.

        Returns:
            List of healthy instances.
        """
        return [i for i in self.instances.values() if i.can_accept_requests]

    def get_metrics(self) -> ScalingMetrics:
        """Get current scaling metrics.

        Returns:
            Current metrics snapshot.
        """
        instances = list(self.instances.values())

        healthy = sum(1 for i in instances if i.status == InstanceStatus.HEALTHY)
        draining = sum(1 for i in instances if i.status == InstanceStatus.DRAINING)
        unhealthy = sum(1 for i in instances if i.status == InstanceStatus.UNHEALTHY)
        total_requests = sum(i.active_requests for i in instances)
        avg_requests = total_requests / len(instances) if instances else 0.0

        return ScalingMetrics(
            total_instances=len(instances),
            healthy_instances=healthy,
            draining_instances=draining,
            unhealthy_instances=unhealthy,
            total_active_requests=total_requests,
            average_requests_per_instance=avg_requests,
            cpu_utilization=0.0,  # Would be populated by monitoring
            memory_utilization=0.0,  # Would be populated by monitoring
        )

    async def _health_check_loop(self) -> None:
        """Background loop for health checking instances."""
        consecutive_failures: Dict[str, int] = {}

        while not self._shutdown:
            try:
                for instance in list(self.instances.values()):
                    # Perform health check
                    is_healthy = await self._check_instance_health(instance)

                    # Track failures
                    if not is_healthy:
                        consecutive_failures[instance.instance_id] = (
                            consecutive_failures.get(instance.instance_id, 0) + 1
                        )
                    else:
                        consecutive_failures[instance.instance_id] = 0

                    # Update status
                    if consecutive_failures[instance.instance_id] >= self.config.unhealthy_threshold:
                        instance.update_health(False)
                        logger.warning(
                            "instance_unhealthy",
                            instance_id=instance.instance_id,
                            failures=consecutive_failures[instance.instance_id],
                        )
                    else:
                        instance.update_health(True)

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_instance_health(self, instance: WorkerInstance) -> bool:
        """Check health of a single instance.

        Args:
            instance: Instance to check.

        Returns:
            True if healthy, False otherwise.
        """
        if self.health_checker:
            try:
                return await asyncio.wait_for(
                    asyncio.coroutine(lambda: self.health_checker(instance))(),
                    timeout=self.config.health_check_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "health_check_timeout",
                    instance_id=instance.instance_id,
                )
                return False
            except Exception as e:
                logger.error(
                    "health_check_failed",
                    instance_id=instance.instance_id,
                    error=str(e),
                )
                return False

        # Default: consider healthy if not already unhealthy
        return instance.status != InstanceStatus.UNHEALTHY

    def get_scaling_history(self, limit: int = 10) -> List[ScalingEvent]:
        """Get recent scaling events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of recent scaling events.
        """
        return self.scaling_history[-limit:]
