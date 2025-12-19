"""Auto-scaling triggers and policies."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.scaling.horizontal import HorizontalScaler, ScalingDirection


class TriggerType(str, Enum):
    """Type of scaling trigger."""

    METRIC_BASED = "metric_based"
    SCHEDULE_BASED = "schedule_based"
    QUEUE_BASED = "queue_based"
    PREDICTIVE = "predictive"


class MetricComparison(str, Enum):
    """Metric comparison operator."""

    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    EQUAL = "eq"


@dataclass
class ScalingTrigger:
    """Trigger condition for auto-scaling."""

    trigger_id: str
    trigger_type: TriggerType
    enabled: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MetricTrigger(ScalingTrigger):
    """Metric-based scaling trigger."""

    metric_name: str = ""
    threshold: float = 0.0
    comparison: MetricComparison = MetricComparison.GREATER_THAN
    duration_seconds: int = 60  # Threshold must be exceeded for this duration
    scale_direction: ScalingDirection = ScalingDirection.UP

    def __post_init__(self) -> None:
        super().__post_init__()
        self.trigger_type = TriggerType.METRIC_BASED


@dataclass
class ScheduleTrigger(ScalingTrigger):
    """Schedule-based scaling trigger."""

    cron_expression: str = ""
    target_instances: int = 1
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.trigger_type = TriggerType.SCHEDULE_BASED


@dataclass
class QueueTrigger(ScalingTrigger):
    """Queue-based scaling trigger."""

    queue_name: str = ""
    queue_length_threshold: int = 100
    messages_per_instance: int = 10
    scale_direction: ScalingDirection = ScalingDirection.UP

    def __post_init__(self) -> None:
        super().__post_init__()
        self.trigger_type = TriggerType.QUEUE_BASED


class AutoScalerConfig(BaseModel):
    """Configuration for auto-scaler."""

    evaluation_interval: int = Field(default=30, ge=1, description="Evaluation interval in seconds")
    stabilization_window: int = Field(
        default=300, ge=0, description="Stabilization window in seconds"
    )
    enable_scale_up: bool = Field(default=True, description="Enable scale-up")
    enable_scale_down: bool = Field(default=True, description="Enable scale-down")


class AutoScaler:
    """Auto-scaler with multiple trigger types."""

    def __init__(
        self,
        scaler: HorizontalScaler,
        config: Optional[AutoScalerConfig] = None,
        metric_provider: Optional[Callable[[str], float]] = None,
    ):
        """Initialize auto-scaler.

        Args:
            scaler: Horizontal scaler to control
            config: Auto-scaler configuration
            metric_provider: Function to get metric values
        """
        self.scaler = scaler
        self.config = config or AutoScalerConfig()
        self.metric_provider = metric_provider

        self.triggers: Dict[str, ScalingTrigger] = {}
        self._trigger_states: Dict[str, Dict[str, Any]] = {}
        self._scaling_history: List[Dict[str, Any]] = []

        self._shutdown = False
        self._evaluation_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the auto-scaler."""
        self._shutdown = False
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

    async def stop(self) -> None:
        """Stop the auto-scaler."""
        self._shutdown = True

        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

    def add_trigger(self, trigger: ScalingTrigger) -> None:
        """Add a scaling trigger.

        Args:
            trigger: Trigger to add
        """
        self.triggers[trigger.trigger_id] = trigger
        self._trigger_states[trigger.trigger_id] = {
            "last_triggered": None,
            "trigger_count": 0,
            "threshold_exceeded_since": None,
        }

    def remove_trigger(self, trigger_id: str) -> None:
        """Remove a scaling trigger.

        Args:
            trigger_id: ID of trigger to remove
        """
        self.triggers.pop(trigger_id, None)
        self._trigger_states.pop(trigger_id, None)

    def enable_trigger(self, trigger_id: str) -> None:
        """Enable a trigger.

        Args:
            trigger_id: ID of trigger to enable
        """
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True

    def disable_trigger(self, trigger_id: str) -> None:
        """Disable a trigger.

        Args:
            trigger_id: ID of trigger to disable
        """
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False

    async def _evaluation_loop(self) -> None:
        """Background evaluation loop."""
        while not self._shutdown:
            try:
                await self._evaluate_triggers()
                await asyncio.sleep(self.config.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                await asyncio.sleep(self.config.evaluation_interval)

    async def _evaluate_triggers(self) -> None:
        """Evaluate all triggers."""
        for trigger_id, trigger in self.triggers.items():
            if not trigger.enabled:
                continue

            try:
                if isinstance(trigger, MetricTrigger):
                    await self._evaluate_metric_trigger(trigger)
                elif isinstance(trigger, ScheduleTrigger):
                    await self._evaluate_schedule_trigger(trigger)
                elif isinstance(trigger, QueueTrigger):
                    await self._evaluate_queue_trigger(trigger)
            except Exception:
                # Log error but continue with other triggers
                pass

    async def _evaluate_metric_trigger(self, trigger: MetricTrigger) -> None:
        """Evaluate a metric-based trigger.

        Args:
            trigger: Metric trigger to evaluate
        """
        if not self.metric_provider:
            return

        state = self._trigger_states[trigger.trigger_id]

        # Get current metric value
        try:
            metric_value = self.metric_provider(trigger.metric_name)
        except Exception:
            return

        # Check if threshold is exceeded
        threshold_exceeded = self._compare_metric(
            metric_value, trigger.threshold, trigger.comparison
        )

        now = datetime.utcnow()

        if threshold_exceeded:
            if state["threshold_exceeded_since"] is None:
                state["threshold_exceeded_since"] = now
            else:
                # Check if exceeded for required duration
                exceeded_duration = (now - state["threshold_exceeded_since"]).total_seconds()
                if exceeded_duration >= trigger.duration_seconds:
                    # Check stabilization window
                    if self._is_stabilized():
                        await self._execute_scaling(trigger.scale_direction)
                        state["last_triggered"] = now
                        state["trigger_count"] += 1
                        state["threshold_exceeded_since"] = None
        else:
            state["threshold_exceeded_since"] = None

    def _compare_metric(
        self, value: float, threshold: float, comparison: MetricComparison
    ) -> bool:
        """Compare metric value against threshold.

        Args:
            value: Metric value
            threshold: Threshold value
            comparison: Comparison operator

        Returns:
            True if comparison is satisfied
        """
        if comparison == MetricComparison.GREATER_THAN:
            return value > threshold
        elif comparison == MetricComparison.LESS_THAN:
            return value < threshold
        elif comparison == MetricComparison.GREATER_EQUAL:
            return value >= threshold
        elif comparison == MetricComparison.LESS_EQUAL:
            return value <= threshold
        elif comparison == MetricComparison.EQUAL:
            return value == threshold
        return False

    async def _evaluate_schedule_trigger(self, trigger: ScheduleTrigger) -> None:
        """Evaluate a schedule-based trigger.

        Args:
            trigger: Schedule trigger to evaluate
        """
        # In a real implementation, evaluate cron expression
        # For now, this is a placeholder
        pass

    async def _evaluate_queue_trigger(self, trigger: QueueTrigger) -> None:
        """Evaluate a queue-based trigger.

        Args:
            trigger: Queue trigger to evaluate
        """
        # In a real implementation, check queue length
        # For now, this is a placeholder
        pass

    def _is_stabilized(self) -> bool:
        """Check if system is stabilized (no recent scaling).

        Returns:
            True if stabilized
        """
        if not self._scaling_history:
            return True

        last_scaling = self._scaling_history[-1]
        elapsed = (datetime.utcnow() - last_scaling["timestamp"]).total_seconds()
        return elapsed >= self.config.stabilization_window

    async def _execute_scaling(self, direction: ScalingDirection) -> None:
        """Execute scaling action.

        Args:
            direction: Direction to scale
        """
        # Check if scaling is enabled
        if direction == ScalingDirection.UP and not self.config.enable_scale_up:
            return
        if direction == ScalingDirection.DOWN and not self.config.enable_scale_down:
            return

        # Perform scaling
        count = await self.scaler.scale(direction)

        if count > 0:
            # Record scaling action
            self._scaling_history.append(
                {
                    "timestamp": datetime.utcnow(),
                    "direction": direction.value,
                    "count": count,
                }
            )

            # Limit history size
            if len(self._scaling_history) > 100:
                self._scaling_history = self._scaling_history[-100:]

    def get_scaling_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get scaling history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of scaling events
        """
        if limit:
            return self._scaling_history[-limit:]
        return self._scaling_history.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Get auto-scaler metrics.

        Returns:
            Dictionary of metrics
        """
        total_triggers = len(self.triggers)
        enabled_triggers = sum(1 for t in self.triggers.values() if t.enabled)

        recent_scalings = [
            h
            for h in self._scaling_history
            if (datetime.utcnow() - h["timestamp"]).total_seconds() < 3600
        ]

        return {
            "total_triggers": total_triggers,
            "enabled_triggers": enabled_triggers,
            "disabled_triggers": total_triggers - enabled_triggers,
            "total_scaling_events": len(self._scaling_history),
            "recent_scaling_events": len(recent_scalings),
            "scale_up_events": sum(
                1 for h in self._scaling_history if h["direction"] == "up"
            ),
            "scale_down_events": sum(
                1 for h in self._scaling_history if h["direction"] == "down"
            ),
        }
