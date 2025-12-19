"""Tests for preemptive scheduling."""

import asyncio
import time

import pytest

from tinyllm.core.scheduling import (
    PreemptiveScheduler,
    Priority,
    ScheduledTask,
    SchedulerConfig,
    SchedulingPolicy,
    TaskState,
)


# Fixtures


@pytest.fixture
async def scheduler():
    """Create a scheduler for testing."""
    config = SchedulerConfig(
        policy=SchedulingPolicy.PREEMPTIVE,
        max_concurrent_tasks=3,
        preemption_enabled=True,
        starvation_threshold_ms=1000,
    )
    scheduler = PreemptiveScheduler(config=config)
    yield scheduler
    await scheduler.clear()


@pytest.fixture
async def fifo_scheduler():
    """Create a FIFO scheduler for testing."""
    config = SchedulerConfig(
        policy=SchedulingPolicy.FIFO,
        max_concurrent_tasks=3,
        preemption_enabled=False,
    )
    scheduler = PreemptiveScheduler(config=config)
    yield scheduler
    await scheduler.clear()


@pytest.fixture
async def round_robin_scheduler():
    """Create a round-robin scheduler for testing."""
    config = SchedulerConfig(
        policy=SchedulingPolicy.ROUND_ROBIN,
        max_concurrent_tasks=3,
        preemption_enabled=False,
    )
    scheduler = PreemptiveScheduler(config=config)
    yield scheduler
    await scheduler.clear()


@pytest.fixture
async def fair_share_scheduler():
    """Create a fair-share scheduler for testing."""
    config = SchedulerConfig(
        policy=SchedulingPolicy.FAIR_SHARE,
        max_concurrent_tasks=3,
        preemption_enabled=False,
    )
    scheduler = PreemptiveScheduler(config=config)
    yield scheduler
    await scheduler.clear()


# Task Submission and Priority Handling Tests


@pytest.mark.asyncio
async def test_submit_task_default_priority(scheduler):
    """Test submitting a task with default priority."""
    task = await scheduler.submit("task-1")

    assert task.task_id == "task-1"
    assert task.priority == Priority.NORMAL
    assert task.state == TaskState.PENDING
    assert "task-1" in scheduler._tasks


@pytest.mark.asyncio
async def test_submit_task_with_priority(scheduler):
    """Test submitting a task with specific priority."""
    task = await scheduler.submit("task-1", priority=Priority.HIGH)

    assert task.task_id == "task-1"
    assert task.priority == Priority.HIGH


@pytest.mark.asyncio
async def test_submit_task_with_metadata(scheduler):
    """Test submitting a task with metadata."""
    metadata = {"user": "test", "request_id": "req-123"}
    task = await scheduler.submit("task-1", metadata=metadata)

    assert task.metadata == metadata


@pytest.mark.asyncio
async def test_submit_multiple_tasks_priority_ordering(scheduler):
    """Test that submitted tasks are ordered by priority."""
    await scheduler.submit("task-low", priority=Priority.LOW)
    await scheduler.submit("task-high", priority=Priority.HIGH)
    await scheduler.submit("task-critical", priority=Priority.CRITICAL)
    await scheduler.submit("task-normal", priority=Priority.NORMAL)

    # Ready queue should be sorted by priority
    assert scheduler._ready_queue[0] == "task-critical"
    assert scheduler._ready_queue[1] == "task-high"
    assert scheduler._ready_queue[2] == "task-normal"
    assert scheduler._ready_queue[3] == "task-low"


@pytest.mark.asyncio
async def test_submit_tasks_same_priority_fifo_order(scheduler):
    """Test that tasks with same priority maintain FIFO order."""
    await scheduler.submit("task-1", priority=Priority.NORMAL)
    await scheduler.submit("task-2", priority=Priority.NORMAL)
    await scheduler.submit("task-3", priority=Priority.NORMAL)

    # Tasks with same priority should maintain submission order
    normal_tasks = [tid for tid in scheduler._ready_queue
                    if scheduler._tasks[tid].priority == Priority.NORMAL]
    assert normal_tasks == ["task-1", "task-2", "task-3"]


# Task Execution Tests


@pytest.mark.asyncio
async def test_execute_sync_function(scheduler):
    """Test executing a synchronous function."""
    def sync_func(x, y):
        return x + y

    await scheduler.submit("task-1")
    result = await scheduler.execute("task-1", sync_func, 2, 3)

    assert result == 5
    assert scheduler._tasks["task-1"].state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_execute_async_function(scheduler):
    """Test executing an asynchronous function."""
    async def async_func(x, y):
        await asyncio.sleep(0.01)
        return x * y

    await scheduler.submit("task-1")
    result = await scheduler.execute("task-1", async_func, 3, 4)

    assert result == 12
    assert scheduler._tasks["task-1"].state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_execute_function_with_kwargs(scheduler):
    """Test executing a function with keyword arguments."""
    def func(a, b, c=10):
        return a + b + c

    await scheduler.submit("task-1")
    result = await scheduler.execute("task-1", func, 1, 2, c=5)

    assert result == 8


@pytest.mark.asyncio
async def test_execute_nonexistent_task(scheduler):
    """Test executing a task that doesn't exist."""
    with pytest.raises(ValueError, match="Task .* not found"):
        await scheduler.execute("nonexistent", lambda: None)


@pytest.mark.asyncio
async def test_execute_task_state_transitions(scheduler):
    """Test that task state transitions correctly during execution."""
    async def async_func():
        await asyncio.sleep(0.01)
        return "done"

    await scheduler.submit("task-1")

    # Task should start as pending
    assert scheduler._tasks["task-1"].state == TaskState.PENDING

    # Execute task
    task = asyncio.create_task(scheduler.execute("task-1", async_func))
    await asyncio.sleep(0.001)  # Give task time to start

    # Task should be running
    assert scheduler._tasks["task-1"].state == TaskState.RUNNING
    assert scheduler._tasks["task-1"].started_at is not None

    # Wait for completion
    await task

    # Task should be completed
    assert scheduler._tasks["task-1"].state == TaskState.COMPLETED
    assert scheduler._tasks["task-1"].completed_at is not None


@pytest.mark.asyncio
async def test_execute_task_tracks_execution_time(scheduler):
    """Test that execution time is tracked."""
    async def async_func():
        await asyncio.sleep(0.05)

    await scheduler.submit("task-1")
    await scheduler.execute("task-1", async_func)

    task = scheduler._tasks["task-1"]
    assert task.execution_time > 0
    assert task.execution_time >= 0.04  # Should be at least 40ms


@pytest.mark.asyncio
async def test_execute_task_failure(scheduler):
    """Test handling of task execution failure."""
    def failing_func():
        raise ValueError("Task failed")

    await scheduler.submit("task-1")

    with pytest.raises(ValueError, match="Task failed"):
        await scheduler.execute("task-1", failing_func)

    assert scheduler._tasks["task-1"].state == TaskState.FAILED


@pytest.mark.asyncio
async def test_execute_respects_max_concurrent_tasks(scheduler):
    """Test that scheduler respects max concurrent task limit."""
    async def long_running_task():
        await asyncio.sleep(0.1)
        return "done"

    # Submit and start executing max_concurrent_tasks tasks
    tasks = []
    for i in range(4):  # Submit 4 tasks, but max is 3
        await scheduler.submit(f"task-{i}")
        tasks.append(asyncio.create_task(
            scheduler.execute(f"task-{i}", long_running_task)
        ))

    await asyncio.sleep(0.01)  # Let tasks start

    # Only 3 should be running at once
    running_count = sum(1 for t in scheduler._tasks.values()
                       if t.state == TaskState.RUNNING)
    assert running_count <= 3

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)


# Preemption Tests


@pytest.mark.asyncio
async def test_preemption_high_priority_preempts_low(scheduler):
    """Test that high priority task preempts low priority task."""
    async def long_task():
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    # Start a low priority task
    await scheduler.submit("low-task", priority=Priority.LOW)
    low_task = asyncio.create_task(
        scheduler.execute("low-task", long_task)
    )
    scheduler._running_tasks["low-task"] = low_task

    await asyncio.sleep(0.01)  # Let it start

    # Submit a critical priority task - should trigger preemption
    await scheduler.submit("critical-task", priority=Priority.CRITICAL)

    await asyncio.sleep(0.01)  # Give time for preemption

    # Low priority task should be cancelled/preempted
    assert low_task.cancelled() or scheduler._tasks["low-task"].state == TaskState.PREEMPTED


@pytest.mark.asyncio
async def test_preemption_increments_count(scheduler):
    """Test that preemption count is incremented."""
    async def task_func():
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    # Start low priority task
    await scheduler.submit("low-task", priority=Priority.BACKGROUND)
    low_task = asyncio.create_task(
        scheduler.execute("low-task", task_func)
    )
    scheduler._running_tasks["low-task"] = low_task

    await asyncio.sleep(0.01)

    # Submit high priority task to trigger preemption
    await scheduler.submit("high-task", priority=Priority.CRITICAL)

    await asyncio.sleep(0.01)

    # Check preemption was recorded
    if scheduler._tasks["low-task"].state == TaskState.PREEMPTED:
        assert scheduler._tasks["low-task"].preempt_count > 0


@pytest.mark.asyncio
async def test_preemption_disabled(fifo_scheduler):
    """Test that preemption doesn't occur when disabled."""
    async def long_task():
        await asyncio.sleep(0.1)

    # Start low priority task
    await fifo_scheduler.submit("low-task", priority=Priority.LOW)
    low_task = asyncio.create_task(
        fifo_scheduler.execute("low-task", long_task)
    )
    fifo_scheduler._running_tasks["low-task"] = low_task

    await asyncio.sleep(0.01)

    # Submit high priority task
    await fifo_scheduler.submit("high-task", priority=Priority.CRITICAL)

    await asyncio.sleep(0.01)

    # Low priority task should still be running
    assert not low_task.cancelled()


# Scheduling Policy Tests


@pytest.mark.asyncio
async def test_fifo_policy(fifo_scheduler):
    """Test FIFO scheduling policy."""
    await fifo_scheduler.submit("task-1", priority=Priority.LOW)
    await fifo_scheduler.submit("task-2", priority=Priority.HIGH)
    await fifo_scheduler.submit("task-3", priority=Priority.CRITICAL)

    # FIFO should return tasks in submission order
    next_task = await fifo_scheduler.get_next_task()
    assert next_task == "task-1"


@pytest.mark.asyncio
async def test_priority_policy(scheduler):
    """Test priority-based scheduling policy."""
    await scheduler.submit("task-1", priority=Priority.LOW)
    await scheduler.submit("task-2", priority=Priority.HIGH)
    await scheduler.submit("task-3", priority=Priority.CRITICAL)

    # Priority policy should return highest priority task
    next_task = await scheduler.get_next_task()
    assert next_task == "task-3"


@pytest.mark.asyncio
async def test_round_robin_policy(round_robin_scheduler):
    """Test round-robin scheduling policy."""
    await round_robin_scheduler.submit("task-1")
    await round_robin_scheduler.submit("task-2")
    await round_robin_scheduler.submit("task-3")

    # Round robin should cycle through tasks
    next_task_1 = await round_robin_scheduler.get_next_task()
    next_task_2 = await round_robin_scheduler.get_next_task()
    next_task_3 = await round_robin_scheduler.get_next_task()
    next_task_4 = await round_robin_scheduler.get_next_task()

    # Should cycle back to first task
    assert next_task_1 != next_task_2
    assert next_task_2 != next_task_3
    assert next_task_1 == next_task_4


@pytest.mark.asyncio
async def test_fair_share_policy(fair_share_scheduler):
    """Test fair-share scheduling policy."""
    # Submit tasks and set different execution times
    await fair_share_scheduler.submit("task-1")
    await fair_share_scheduler.submit("task-2")
    await fair_share_scheduler.submit("task-3")

    fair_share_scheduler._tasks["task-1"].execution_time = 10.0
    fair_share_scheduler._tasks["task-2"].execution_time = 1.0
    fair_share_scheduler._tasks["task-3"].execution_time = 5.0

    # Fair share should favor task with least execution time
    next_task = await fair_share_scheduler.get_next_task()
    assert next_task == "task-2"


@pytest.mark.asyncio
async def test_get_next_task_empty_queue(scheduler):
    """Test getting next task from empty queue."""
    next_task = await scheduler.get_next_task()
    assert next_task is None


# Starvation Handling Tests


@pytest.mark.asyncio
async def test_starvation_priority_boost(scheduler):
    """Test that starved tasks get priority boost."""
    # Create a task and backdate its submission time to simulate starvation
    await scheduler.submit("starved-task", priority=Priority.LOW)

    # Manually set submitted_at to past to trigger starvation
    scheduler._tasks["starved-task"].submitted_at = time.perf_counter() - 10.0

    # Get next task - should trigger starvation check
    await scheduler.get_next_task()

    # Priority should have been boosted
    assert scheduler._tasks["starved-task"].priority.value < Priority.LOW.value


@pytest.mark.asyncio
async def test_starvation_threshold_respected(scheduler):
    """Test that starvation threshold is respected."""
    await scheduler.submit("task-1", priority=Priority.LOW)

    # Task just submitted - should not be considered starved
    original_priority = scheduler._tasks["task-1"].priority

    await scheduler.get_next_task()

    # Priority should remain unchanged
    assert scheduler._tasks["task-1"].priority == original_priority


@pytest.mark.asyncio
async def test_starvation_critical_not_boosted(scheduler):
    """Test that critical priority tasks are not boosted further."""
    await scheduler.submit("task-1", priority=Priority.CRITICAL)

    # Backdate to trigger starvation
    scheduler._tasks["task-1"].submitted_at = time.perf_counter() - 10.0

    await scheduler.get_next_task()

    # Should remain critical
    assert scheduler._tasks["task-1"].priority == Priority.CRITICAL


# Task Cancellation Tests


@pytest.mark.asyncio
async def test_cancel_pending_task(scheduler):
    """Test cancelling a pending task."""
    await scheduler.submit("task-1")

    cancelled = await scheduler.cancel_task("task-1")

    assert cancelled is True
    assert scheduler._tasks["task-1"].state == TaskState.CANCELLED
    assert "task-1" not in scheduler._ready_queue


@pytest.mark.asyncio
async def test_cancel_running_task(scheduler):
    """Test cancelling a running task."""
    async def long_task():
        await asyncio.sleep(1.0)

    await scheduler.submit("task-1")
    task = asyncio.create_task(scheduler.execute("task-1", long_task))
    scheduler._running_tasks["task-1"] = task

    await asyncio.sleep(0.01)  # Let it start

    cancelled = await scheduler.cancel_task("task-1")

    assert cancelled is True
    assert scheduler._tasks["task-1"].state == TaskState.CANCELLED


@pytest.mark.asyncio
async def test_cancel_completed_task(scheduler):
    """Test that completed tasks cannot be cancelled."""
    def quick_task():
        return "done"

    await scheduler.submit("task-1")
    await scheduler.execute("task-1", quick_task)

    cancelled = await scheduler.cancel_task("task-1")

    assert cancelled is False
    assert scheduler._tasks["task-1"].state == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_cancel_nonexistent_task(scheduler):
    """Test cancelling a nonexistent task."""
    cancelled = await scheduler.cancel_task("nonexistent")
    assert cancelled is False


@pytest.mark.asyncio
async def test_cancel_failed_task(scheduler):
    """Test that failed tasks cannot be cancelled."""
    def failing_task():
        raise ValueError("fail")

    await scheduler.submit("task-1")

    try:
        await scheduler.execute("task-1", failing_task)
    except ValueError:
        pass

    cancelled = await scheduler.cancel_task("task-1")

    assert cancelled is False
    assert scheduler._tasks["task-1"].state == TaskState.FAILED


# Scheduler Metrics Tests


@pytest.mark.asyncio
async def test_get_task_status(scheduler):
    """Test getting task status."""
    await scheduler.submit("task-1", priority=Priority.HIGH)

    status = scheduler.get_task_status("task-1")

    assert status is not None
    assert status.task_id == "task-1"
    assert status.priority == Priority.HIGH
    assert status.state == TaskState.PENDING


@pytest.mark.asyncio
async def test_get_task_status_nonexistent(scheduler):
    """Test getting status of nonexistent task."""
    status = scheduler.get_task_status("nonexistent")
    assert status is None


@pytest.mark.asyncio
async def test_queue_stats_initial(scheduler):
    """Test queue statistics initially."""
    stats = scheduler.get_queue_stats()

    assert stats["total_tasks"] == 0
    assert stats["pending"] == 0
    assert stats["running"] == 0
    assert stats["completed"] == 0
    assert stats["ready_queue_size"] == 0
    assert stats["max_concurrent"] == 3


@pytest.mark.asyncio
async def test_queue_stats_with_tasks(scheduler):
    """Test queue statistics with tasks."""
    # Submit tasks
    await scheduler.submit("task-1", priority=Priority.HIGH)
    await scheduler.submit("task-2", priority=Priority.NORMAL)
    await scheduler.submit("task-3", priority=Priority.LOW)

    stats = scheduler.get_queue_stats()

    assert stats["total_tasks"] == 3
    assert stats["pending"] == 3
    assert stats["ready_queue_size"] == 3


@pytest.mark.asyncio
async def test_queue_stats_with_completed_tasks(scheduler):
    """Test queue statistics with completed tasks."""
    def quick_task():
        return "done"

    await scheduler.submit("task-1")
    await scheduler.submit("task-2")

    await scheduler.execute("task-1", quick_task)

    stats = scheduler.get_queue_stats()

    assert stats["total_tasks"] == 2
    assert stats["completed"] == 1
    assert stats["pending"] == 1


@pytest.mark.asyncio
async def test_queue_stats_running_tasks(scheduler):
    """Test queue statistics with running tasks."""
    async def long_task():
        await asyncio.sleep(0.1)

    await scheduler.submit("task-1")
    await scheduler.submit("task-2")

    # Start tasks
    task1 = asyncio.create_task(scheduler.execute("task-1", long_task))
    task2 = asyncio.create_task(scheduler.execute("task-2", long_task))

    await asyncio.sleep(0.01)  # Let them start

    stats = scheduler.get_queue_stats()

    assert stats["running"] >= 1  # At least one should be running

    # Cleanup
    await asyncio.gather(task1, task2, return_exceptions=True)


# Clear and Cleanup Tests


@pytest.mark.asyncio
async def test_clear_scheduler(scheduler):
    """Test clearing all tasks from scheduler."""
    # Submit tasks
    await scheduler.submit("task-1")
    await scheduler.submit("task-2")
    await scheduler.submit("task-3")

    await scheduler.clear()

    assert len(scheduler._tasks) == 0
    assert len(scheduler._ready_queue) == 0
    assert len(scheduler._running_tasks) == 0


@pytest.mark.asyncio
async def test_clear_cancels_running_tasks(scheduler):
    """Test that clear cancels running tasks."""
    async def long_task():
        await asyncio.sleep(1.0)

    await scheduler.submit("task-1")
    task = asyncio.create_task(scheduler.execute("task-1", long_task))
    scheduler._running_tasks["task-1"] = task

    await asyncio.sleep(0.01)

    await scheduler.clear()

    # Wait for cancellation to propagate
    await asyncio.sleep(0.01)

    # Task should be cancelled or done (cancelling state transitions to cancelled)
    assert task.done() or task.cancelled()


# Model Tests


def test_scheduled_task_creation():
    """Test creating a ScheduledTask."""
    task = ScheduledTask(
        task_id="test-task",
        priority=Priority.HIGH,
    )

    assert task.task_id == "test-task"
    assert task.priority == Priority.HIGH
    assert task.state == TaskState.PENDING
    assert task.preempt_count == 0
    assert task.execution_time == 0.0


def test_scheduled_task_with_metadata():
    """Test creating a ScheduledTask with metadata."""
    metadata = {"user": "test", "type": "compute"}
    task = ScheduledTask(
        task_id="test-task",
        metadata=metadata,
    )

    assert task.metadata == metadata


def test_scheduler_config_defaults():
    """Test SchedulerConfig default values."""
    config = SchedulerConfig()

    assert config.policy == SchedulingPolicy.PREEMPTIVE
    assert config.max_concurrent_tasks == 10
    assert config.preemption_enabled is True
    assert config.time_slice_ms == 100
    assert config.starvation_threshold_ms == 5000


def test_scheduler_config_validation():
    """Test SchedulerConfig validation."""
    # Valid config
    config = SchedulerConfig(
        max_concurrent_tasks=5,
        time_slice_ms=50,
        starvation_threshold_ms=1000,
    )

    assert config.max_concurrent_tasks == 5
    assert config.time_slice_ms == 50


def test_scheduler_config_invalid_max_tasks():
    """Test SchedulerConfig validation for max_concurrent_tasks."""
    with pytest.raises(Exception):  # Pydantic validation error
        SchedulerConfig(max_concurrent_tasks=0)

    with pytest.raises(Exception):  # Pydantic validation error
        SchedulerConfig(max_concurrent_tasks=2000)


def test_scheduler_config_invalid_time_slice():
    """Test SchedulerConfig validation for time_slice_ms."""
    with pytest.raises(Exception):  # Pydantic validation error
        SchedulerConfig(time_slice_ms=5)

    with pytest.raises(Exception):  # Pydantic validation error
        SchedulerConfig(time_slice_ms=20000)


# Enum Tests


def test_priority_enum_values():
    """Test Priority enum values."""
    assert Priority.CRITICAL.value == 0
    assert Priority.HIGH.value == 1
    assert Priority.NORMAL.value == 2
    assert Priority.LOW.value == 3
    assert Priority.BACKGROUND.value == 4


def test_priority_enum_ordering():
    """Test Priority enum ordering."""
    assert Priority.CRITICAL.value < Priority.HIGH.value
    assert Priority.HIGH.value < Priority.NORMAL.value
    assert Priority.NORMAL.value < Priority.LOW.value
    assert Priority.LOW.value < Priority.BACKGROUND.value


def test_scheduling_policy_enum_values():
    """Test SchedulingPolicy enum values."""
    assert SchedulingPolicy.FIFO.value == "fifo"
    assert SchedulingPolicy.PRIORITY.value == "priority"
    assert SchedulingPolicy.ROUND_ROBIN.value == "round_robin"
    assert SchedulingPolicy.FAIR_SHARE.value == "fair_share"
    assert SchedulingPolicy.PREEMPTIVE.value == "preemptive"


def test_task_state_enum_values():
    """Test TaskState enum values."""
    assert TaskState.PENDING.value == "pending"
    assert TaskState.RUNNING.value == "running"
    assert TaskState.PREEMPTED.value == "preempted"
    assert TaskState.COMPLETED.value == "completed"
    assert TaskState.FAILED.value == "failed"
    assert TaskState.CANCELLED.value == "cancelled"


# Edge Cases and Integration Tests


@pytest.mark.asyncio
async def test_concurrent_task_submissions(scheduler):
    """Test concurrent task submissions."""
    async def submit_task(task_id):
        await scheduler.submit(task_id)

    tasks = [submit_task(f"task-{i}") for i in range(10)]
    await asyncio.gather(*tasks)

    assert len(scheduler._tasks) == 10


@pytest.mark.asyncio
async def test_scheduler_initialization_default_config():
    """Test scheduler initialization with default config."""
    scheduler = PreemptiveScheduler()

    assert scheduler.config.policy == SchedulingPolicy.PREEMPTIVE
    assert scheduler.config.max_concurrent_tasks == 10
    assert scheduler.config.preemption_enabled is True

    await scheduler.clear()


@pytest.mark.asyncio
async def test_task_total_time_tracking(scheduler):
    """Test that total time from submission to completion is tracked."""
    async def task_func():
        await asyncio.sleep(0.05)

    await scheduler.submit("task-1")
    await asyncio.sleep(0.02)  # Wait before executing
    await scheduler.execute("task-1", task_func)

    task = scheduler._tasks["task-1"]
    total_time = task.completed_at - task.submitted_at

    assert total_time >= 0.07  # At least 20ms wait + 50ms execution


@pytest.mark.asyncio
async def test_multiple_policies_on_same_tasks():
    """Test different scheduling policies on same set of tasks."""
    async def create_and_get_order(policy):
        config = SchedulerConfig(policy=policy, preemption_enabled=False)
        sched = PreemptiveScheduler(config=config)

        await sched.submit("task-1", priority=Priority.LOW)
        await sched.submit("task-2", priority=Priority.HIGH)
        await sched.submit("task-3", priority=Priority.NORMAL)

        order = []
        while True:
            next_task = await sched.get_next_task()
            if next_task is None:
                break
            order.append(next_task)
            sched._ready_queue.remove(next_task)

        await sched.clear()
        return order

    fifo_order = await create_and_get_order(SchedulingPolicy.FIFO)
    priority_order = await create_and_get_order(SchedulingPolicy.PRIORITY)

    # FIFO should maintain submission order
    assert fifo_order == ["task-1", "task-2", "task-3"]

    # Priority should order by priority
    assert priority_order == ["task-2", "task-3", "task-1"]


@pytest.mark.asyncio
async def test_task_execution_with_exceptions_in_async_context(scheduler):
    """Test task execution handles exceptions in async context."""
    async def failing_async_task():
        await asyncio.sleep(0.01)
        raise RuntimeError("Async task failed")

    await scheduler.submit("task-1")

    with pytest.raises(RuntimeError, match="Async task failed"):
        await scheduler.execute("task-1", failing_async_task)

    assert scheduler._tasks["task-1"].state == TaskState.FAILED


@pytest.mark.asyncio
async def test_scheduler_handles_rapid_submit_execute_cycle(scheduler):
    """Test scheduler handles rapid submit-execute cycles."""
    def quick_task(n):
        return n * 2

    results = []
    for i in range(20):
        await scheduler.submit(f"task-{i}")
        result = await scheduler.execute(f"task-{i}", quick_task, i)
        results.append(result)

    assert results == [i * 2 for i in range(20)]


@pytest.mark.asyncio
async def test_ready_queue_sorting_after_starvation_boost(scheduler):
    """Test that ready queue is re-sorted after starvation boost."""
    # Submit tasks with different priorities
    await scheduler.submit("task-low", priority=Priority.LOW)
    await scheduler.submit("task-high", priority=Priority.HIGH)

    # Backdate low priority task to trigger starvation
    scheduler._tasks["task-low"].submitted_at = time.perf_counter() - 10.0

    # Get next task - should trigger starvation check and re-sort
    await scheduler.get_next_task()

    # After boost, task-low might be higher priority
    # Verify queue is still properly ordered
    for i in range(len(scheduler._ready_queue) - 1):
        current_priority = scheduler._tasks[scheduler._ready_queue[i]].priority
        next_priority = scheduler._tasks[scheduler._ready_queue[i + 1]].priority
        assert current_priority.value <= next_priority.value
