"""Preemptive scheduling for TinyLLM graph execution.

This module provides preemptive scheduling capabilities, allowing
high-priority tasks to interrupt lower-priority ones for responsive
execution.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="scheduling")


class Priority(int, Enum):
    """Task priority levels."""

    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


class SchedulingPolicy(str, Enum):
    """Scheduling policy types."""

    FIFO = "fifo"  # First In First Out
    PRIORITY = "priority"  # Priority-based
    ROUND_ROBIN = "round_robin"  # Round-robin
    FAIR_SHARE = "fair_share"  # Fair share
    PREEMPTIVE = "preemptive"  # Preemptive priority


class TaskState(str, Enum):
    """Task execution states."""

    PENDING = "pending"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledTask(BaseModel):
    """A task scheduled for execution."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    task_id: str = Field(description="Unique task identifier")
    priority: Priority = Field(default=Priority.NORMAL)
    state: TaskState = Field(default=TaskState.PENDING)
    submitted_at: float = Field(default_factory=time.perf_counter)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    preempt_count: int = Field(default=0, ge=0)
    execution_time: float = Field(default=0.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(BaseModel):
    """Configuration for task scheduler."""

    model_config = {"extra": "forbid"}

    policy: SchedulingPolicy = Field(
        default=SchedulingPolicy.PREEMPTIVE,
        description="Scheduling policy to use",
    )
    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum concurrent tasks",
    )
    preemption_enabled: bool = Field(
        default=True,
        description="Whether to allow task preemption",
    )
    time_slice_ms: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Time slice for round-robin in milliseconds",
    )
    starvation_threshold_ms: int = Field(
        default=5000,
        ge=100,
        description="Threshold to boost starved tasks",
    )


class PreemptiveScheduler:
    """Preemptive scheduler for graph execution tasks.

    Implements priority-based preemptive scheduling with support for
    multiple policies including FIFO, priority, round-robin, and fair share.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize preemptive scheduler.

        Args:
            config: Scheduler configuration.
        """
        self.config = config or SchedulerConfig()
        self._tasks: Dict[str, ScheduledTask] = {}
        self._ready_queue: List[str] = []  # Task IDs ready to run
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._round_robin_index = 0

    async def submit(
        self,
        task_id: str,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Submit a task for scheduling.

        Args:
            task_id: Unique task identifier.
            priority: Task priority.
            metadata: Optional task metadata.

        Returns:
            Scheduled task descriptor.
        """
        async with self._lock:
            scheduled_task = ScheduledTask(
                task_id=task_id,
                priority=priority,
                metadata=metadata or {},
            )

            self._tasks[task_id] = scheduled_task
            self._ready_queue.append(task_id)

            logger.info(
                "task_submitted",
                task_id=task_id,
                priority=priority.name,
            )

            # Sort ready queue by priority if using priority scheduling
            if self.config.policy in [
                SchedulingPolicy.PRIORITY,
                SchedulingPolicy.PREEMPTIVE,
            ]:
                self._sort_ready_queue()

            # Check if we should preempt a running task
            if self.config.preemption_enabled:
                await self._check_preemption()

            return scheduled_task

    async def execute(
        self,
        task_id: str,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a scheduled task.

        Args:
            task_id: Task identifier.
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of function execution.
        """
        # Wait for execution slot
        await self._semaphore.acquire()

        async with self._lock:
            if task_id not in self._tasks:
                self._semaphore.release()
                raise ValueError(f"Task {task_id} not found")

            task = self._tasks[task_id]
            task.state = TaskState.RUNNING
            task.started_at = time.perf_counter()

            logger.info(
                "task_started",
                task_id=task_id,
                priority=task.priority.name,
            )

        try:
            # Execute the function
            start_time = time.perf_counter()

            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            execution_time = time.perf_counter() - start_time

            async with self._lock:
                task.state = TaskState.COMPLETED
                task.completed_at = time.perf_counter()
                task.execution_time += execution_time

                logger.info(
                    "task_completed",
                    task_id=task_id,
                    execution_time_ms=int(execution_time * 1000),
                    total_time_ms=int((task.completed_at - task.submitted_at) * 1000),
                )

            return result

        except asyncio.CancelledError:
            async with self._lock:
                task.state = TaskState.PREEMPTED
                task.preempt_count += 1

                logger.warning(
                    "task_preempted",
                    task_id=task_id,
                    preempt_count=task.preempt_count,
                )

            raise

        except Exception as e:
            async with self._lock:
                task.state = TaskState.FAILED

                logger.error(
                    "task_failed",
                    task_id=task_id,
                    error=str(e),
                )

            raise

        finally:
            self._semaphore.release()

            async with self._lock:
                if task_id in self._running_tasks:
                    del self._running_tasks[task_id]
                if task_id in self._ready_queue:
                    self._ready_queue.remove(task_id)

    async def _check_preemption(self) -> None:
        """Check if any running task should be preempted."""
        if not self.config.preemption_enabled:
            return

        if not self._ready_queue or not self._running_tasks:
            return

        # Get highest priority waiting task
        waiting_task_id = self._ready_queue[0]
        waiting_task = self._tasks[waiting_task_id]

        # Find lowest priority running task
        lowest_priority_task_id = None
        lowest_priority = Priority.CRITICAL

        for running_id, running_asyncio_task in self._running_tasks.items():
            running_task = self._tasks[running_id]
            if running_task.priority.value > lowest_priority.value:
                lowest_priority = running_task.priority
                lowest_priority_task_id = running_id

        # Preempt if waiting task has higher priority
        if lowest_priority_task_id and waiting_task.priority.value < lowest_priority.value:
            logger.info(
                "preempting_task",
                preempted_task_id=lowest_priority_task_id,
                new_task_id=waiting_task_id,
                preempted_priority=lowest_priority.name,
                new_priority=waiting_task.priority.name,
            )

            # Cancel the lower priority task
            running_asyncio_task = self._running_tasks[lowest_priority_task_id]
            running_asyncio_task.cancel()

    def _sort_ready_queue(self) -> None:
        """Sort ready queue based on priority and submission time."""
        self._ready_queue.sort(
            key=lambda tid: (
                self._tasks[tid].priority.value,
                self._tasks[tid].submitted_at,
            )
        )

    async def get_next_task(self) -> Optional[str]:
        """Get next task to execute based on scheduling policy.

        Returns:
            Task ID to execute next, or None if no tasks ready.
        """
        async with self._lock:
            if not self._ready_queue:
                return None

            # Check for starvation and boost priority
            await self._handle_starvation()

            policy = self.config.policy

            if policy == SchedulingPolicy.FIFO:
                return self._ready_queue[0]

            elif policy in [SchedulingPolicy.PRIORITY, SchedulingPolicy.PREEMPTIVE]:
                # Already sorted by priority
                return self._ready_queue[0]

            elif policy == SchedulingPolicy.ROUND_ROBIN:
                # Simple round-robin
                if self._round_robin_index >= len(self._ready_queue):
                    self._round_robin_index = 0

                task_id = self._ready_queue[self._round_robin_index]
                self._round_robin_index += 1
                return task_id

            elif policy == SchedulingPolicy.FAIR_SHARE:
                # Fair share based on execution time
                # Favor tasks with less execution time
                task_id = min(
                    self._ready_queue,
                    key=lambda tid: self._tasks[tid].execution_time,
                )
                return task_id

            return self._ready_queue[0]

    async def _handle_starvation(self) -> None:
        """Handle task starvation by boosting priority."""
        current_time = time.perf_counter()

        for task_id in self._ready_queue:
            task = self._tasks[task_id]
            wait_time_ms = (current_time - task.submitted_at) * 1000

            if wait_time_ms > self.config.starvation_threshold_ms:
                # Boost priority if starving
                if task.priority.value < Priority.CRITICAL.value + 1:
                    continue  # Already highest priority

                old_priority = task.priority
                task.priority = Priority(task.priority.value - 1)

                logger.warning(
                    "task_priority_boosted",
                    task_id=task_id,
                    old_priority=old_priority.name,
                    new_priority=task.priority.name,
                    wait_time_ms=int(wait_time_ms),
                )

        # Re-sort after boosting
        if self.config.policy in [
            SchedulingPolicy.PRIORITY,
            SchedulingPolicy.PREEMPTIVE,
        ]:
            self._sort_ready_queue()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: Task to cancel.

        Returns:
            True if cancelled, False if not found or already completed.
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            if task.state in [TaskState.COMPLETED, TaskState.FAILED]:
                return False

            if task_id in self._running_tasks:
                self._running_tasks[task_id].cancel()

            task.state = TaskState.CANCELLED

            if task_id in self._ready_queue:
                self._ready_queue.remove(task_id)

            logger.info("task_cancelled", task_id=task_id)
            return True

    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """Get status of a task.

        Args:
            task_id: Task identifier.

        Returns:
            Task status or None if not found.
        """
        return self._tasks.get(task_id)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get scheduler queue statistics.

        Returns:
            Dictionary with queue statistics.
        """
        pending = sum(
            1 for t in self._tasks.values() if t.state == TaskState.PENDING
        )
        running = sum(
            1 for t in self._tasks.values() if t.state == TaskState.RUNNING
        )
        completed = sum(
            1 for t in self._tasks.values() if t.state == TaskState.COMPLETED
        )

        return {
            "total_tasks": len(self._tasks),
            "pending": pending,
            "running": running,
            "completed": completed,
            "ready_queue_size": len(self._ready_queue),
            "running_tasks": len(self._running_tasks),
            "max_concurrent": self.config.max_concurrent_tasks,
        }

    async def clear(self) -> None:
        """Clear all tasks from scheduler."""
        async with self._lock:
            # Cancel all running tasks
            for task in self._running_tasks.values():
                task.cancel()

            self._tasks.clear()
            self._ready_queue.clear()
            self._running_tasks.clear()
            self._round_robin_index = 0

            logger.info("scheduler_cleared")
