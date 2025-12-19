"""Scale-to-zero support for idle workloads."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field

from tinyllm.scaling.horizontal import HorizontalScaler


class ZeroScaleState(str, Enum):
    """State of scale-to-zero system."""

    ACTIVE = "active"
    IDLE = "idle"
    SCALED_TO_ZERO = "scaled_to_zero"
    SCALING_UP = "scaling_up"
    SCALING_DOWN = "scaling_down"


class ZeroScalePolicy(BaseModel):
    """Policy for scale-to-zero behavior."""

    idle_timeout_seconds: int = Field(
        default=300, ge=0, description="Time before scaling to zero"
    )
    min_idle_instances: int = Field(
        default=0, ge=0, description="Minimum instances when idle"
    )
    warmup_instances: int = Field(default=1, ge=1, description="Instances to start on wakeup")
    activity_check_interval: int = Field(
        default=30, ge=1, description="Activity check interval in seconds"
    )
    enable_scale_to_zero: bool = Field(default=True, description="Enable scale-to-zero")


@dataclass
class ActivityMetrics:
    """Metrics for activity tracking."""

    total_requests: int = 0
    active_requests: int = 0
    last_request_time: Optional[datetime] = None
    requests_per_minute: float = 0.0


class ScaleToZero:
    """Manages scale-to-zero for idle workloads."""

    def __init__(
        self,
        scaler: HorizontalScaler,
        policy: Optional[ZeroScalePolicy] = None,
        on_scaled_to_zero: Optional[Callable[[], None]] = None,
        on_scaled_from_zero: Optional[Callable[[], None]] = None,
    ):
        """Initialize scale-to-zero manager.

        Args:
            scaler: Horizontal scaler to control
            policy: Scale-to-zero policy
            on_scaled_to_zero: Callback when scaled to zero
            on_scaled_from_zero: Callback when scaled from zero
        """
        self.scaler = scaler
        self.policy = policy or ZeroScalePolicy()
        self.on_scaled_to_zero = on_scaled_to_zero
        self.on_scaled_from_zero = on_scaled_from_zero

        self.state = ZeroScaleState.ACTIVE
        self.activity_metrics = ActivityMetrics()
        self._last_activity_time: Optional[datetime] = None
        self._request_history: list[datetime] = []

        self._shutdown = False
        self._monitoring_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start scale-to-zero monitoring."""
        self._shutdown = False
        self._last_activity_time = datetime.utcnow()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop(self) -> None:
        """Stop scale-to-zero monitoring."""
        self._shutdown = True

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def record_request(self) -> None:
        """Record a request to track activity."""
        now = datetime.utcnow()
        self._last_activity_time = now
        self.activity_metrics.last_request_time = now
        self.activity_metrics.total_requests += 1
        self.activity_metrics.active_requests += 1

        # Add to request history
        self._request_history.append(now)

        # Clean old history (keep last 60 seconds)
        cutoff = now - timedelta(seconds=60)
        self._request_history = [t for t in self._request_history if t > cutoff]

        # Update requests per minute
        self.activity_metrics.requests_per_minute = len(self._request_history)

        # Wake up from zero if needed
        if self.state == ZeroScaleState.SCALED_TO_ZERO:
            await self._scale_from_zero()

    async def complete_request(self) -> None:
        """Mark a request as complete."""
        self.activity_metrics.active_requests = max(
            0, self.activity_metrics.active_requests - 1
        )

    def is_idle(self) -> bool:
        """Check if system is idle.

        Returns:
            True if system is idle
        """
        if not self._last_activity_time:
            return False

        # Check if no activity for idle timeout
        idle_duration = (datetime.utcnow() - self._last_activity_time).total_seconds()
        is_timeout = idle_duration >= self.policy.idle_timeout_seconds

        # Also check if no active requests
        no_active = self.activity_metrics.active_requests == 0

        return is_timeout and no_active

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._shutdown:
            try:
                await self._check_activity()
                await asyncio.sleep(self.policy.activity_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                await asyncio.sleep(self.policy.activity_check_interval)

    async def _check_activity(self) -> None:
        """Check activity and scale if needed."""
        if not self.policy.enable_scale_to_zero:
            return

        if self.state == ZeroScaleState.ACTIVE:
            if self.is_idle():
                # Transition to idle state
                self.state = ZeroScaleState.IDLE

        elif self.state == ZeroScaleState.IDLE:
            if not self.is_idle():
                # Activity detected, back to active
                self.state = ZeroScaleState.ACTIVE
            else:
                # Still idle, check if should scale to zero
                idle_duration = (
                    datetime.utcnow() - self._last_activity_time
                ).total_seconds()
                if idle_duration >= self.policy.idle_timeout_seconds:
                    await self._scale_to_zero()

        elif self.state == ZeroScaleState.SCALED_TO_ZERO:
            # Already at zero, nothing to do
            # Wakeup happens in record_request()
            pass

    async def _scale_to_zero(self) -> None:
        """Scale down to zero instances."""
        if self.state == ZeroScaleState.SCALED_TO_ZERO:
            return

        self.state = ZeroScaleState.SCALING_DOWN

        # Get current instance count
        current_count = self.scaler.get_instance_count()
        target_count = self.policy.min_idle_instances

        # Scale down to minimum
        if current_count > target_count:
            # Remove instances
            instances_to_remove = current_count - target_count
            for _ in range(instances_to_remove):
                # Remove one instance at a time
                instances = list(self.scaler.instances.keys())
                if instances:
                    await self.scaler.remove_instance(instances[0], drain=True)

        self.state = ZeroScaleState.SCALED_TO_ZERO

        if self.on_scaled_to_zero:
            self.on_scaled_to_zero()

    async def _scale_from_zero(self) -> None:
        """Scale up from zero instances."""
        if self.state == ZeroScaleState.ACTIVE:
            return

        self.state = ZeroScaleState.SCALING_UP

        # Start warmup instances
        current_count = self.scaler.get_instance_count()
        target_count = max(
            self.policy.warmup_instances, self.scaler.policy.min_instances
        )

        if current_count < target_count:
            # Add instances
            instances_to_add = target_count - current_count
            for i in range(instances_to_add):
                instance_id = f"warmup-{i}-{datetime.utcnow().timestamp()}"
                await self.scaler.add_instance(instance_id)

        self.state = ZeroScaleState.ACTIVE

        if self.on_scaled_from_zero:
            self.on_scaled_from_zero()

    def force_scale_to_zero(self) -> None:
        """Force immediate scale to zero (for testing/maintenance)."""
        asyncio.create_task(self._scale_to_zero())

    def force_wake_up(self) -> None:
        """Force wake up from zero (for testing)."""
        asyncio.create_task(self._scale_from_zero())

    def get_idle_duration(self) -> float:
        """Get duration of idle time in seconds.

        Returns:
            Idle duration in seconds
        """
        if not self._last_activity_time:
            return 0.0

        return (datetime.utcnow() - self._last_activity_time).total_seconds()

    def get_metrics(self) -> Dict[str, Any]:
        """Get scale-to-zero metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "state": self.state.value,
            "is_idle": self.is_idle(),
            "idle_duration_seconds": self.get_idle_duration(),
            "total_requests": self.activity_metrics.total_requests,
            "active_requests": self.activity_metrics.active_requests,
            "requests_per_minute": self.activity_metrics.requests_per_minute,
            "current_instances": self.scaler.get_instance_count(),
            "is_scaled_to_zero": self.state == ZeroScaleState.SCALED_TO_ZERO,
        }
