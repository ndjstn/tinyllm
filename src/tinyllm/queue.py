"""Request queuing system with backpressure for TinyLLM.

This module provides a priority-based request queue with configurable backpressure,
fair scheduling, worker pool management, and comprehensive metrics integration.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from tinyllm.core.executor import Executor
from tinyllm.core.message import ErrorInfo, TaskPayload, TaskResponse
from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="queue")
metrics = get_metrics_collector()


class Priority(str, Enum):
    """Request priority levels."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    @property
    def value_int(self) -> int:
        """Get integer value for priority comparison (higher is better)."""
        return {"high": 3, "normal": 2, "low": 1}[self.value]


class BackpressureMode(str, Enum):
    """Backpressure handling modes."""

    REJECT = "reject"  # Reject requests when queue is full
    BLOCK = "block"  # Block until space available


class QueueStatus(BaseModel):
    """Queue status information."""

    total_queued: int = Field(description="Total requests in queue")
    high_priority: int = Field(description="High priority requests")
    normal_priority: int = Field(description="Normal priority requests")
    low_priority: int = Field(description="Low priority requests")
    active_workers: int = Field(description="Currently active workers")
    max_workers: int = Field(description="Maximum concurrent workers")
    max_queue_size: int = Field(description="Maximum queue size")
    total_processed: int = Field(description="Total requests processed")
    total_rejected: int = Field(description="Total requests rejected")
    average_wait_time_ms: float = Field(description="Average time in queue")


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue with priority ordering."""

    # Order by priority (descending), then timestamp (ascending)
    priority_int: int = field(compare=True)
    timestamp: float = field(compare=True)

    # Request data (not used in comparison)
    request_id: str = field(compare=False)
    priority: Priority = field(compare=False)
    task: TaskPayload = field(compare=False)
    timeout_ms: Optional[int] = field(compare=False)
    future: asyncio.Future = field(compare=False)
    enqueued_at: float = field(compare=False, default_factory=time.monotonic)

    def __post_init__(self):
        """Invert priority for max-heap behavior (higher priority first)."""
        self.priority_int = -self.priority.value_int


class WorkerHealth(BaseModel):
    """Worker health status."""

    worker_id: int
    is_healthy: bool = True
    total_processed: int = 0
    total_errors: int = 0
    current_request_id: Optional[str] = None
    started_at: Optional[float] = None


class RequestQueue:
    """Priority-based request queue with backpressure control.

    Features:
    - Three priority levels (high, normal, low)
    - Configurable max queue size
    - Backpressure modes (reject or block)
    - Fair scheduling across priorities
    - Queue depth metrics
    - Position tracking for requests
    """

    def __init__(
        self,
        max_size: int = 1000,
        backpressure_mode: BackpressureMode = BackpressureMode.REJECT,
    ):
        """Initialize request queue.

        Args:
            max_size: Maximum queue size (0 for unlimited).
            backpressure_mode: How to handle full queue.
        """
        self.max_size = max_size
        self.backpressure_mode = backpressure_mode
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_size if max_size > 0 else 0
        )
        self._requests: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_rejected = 0
        self._wait_times_ms: list[float] = []

        # Priority counters
        self._priority_counts = {
            Priority.HIGH: 0,
            Priority.NORMAL: 0,
            Priority.LOW: 0,
        }

        logger.info(
            "queue_initialized",
            max_size=max_size,
            backpressure_mode=backpressure_mode.value,
        )

    async def enqueue(
        self,
        task: TaskPayload,
        priority: Priority = Priority.NORMAL,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Enqueue a request.

        Args:
            task: Task payload to execute.
            priority: Request priority level.
            timeout_ms: Optional timeout for waiting in queue.

        Returns:
            Request ID.

        Raises:
            asyncio.QueueFull: If queue is full and mode is REJECT.
            asyncio.TimeoutError: If timeout expires while waiting.
        """
        request_id = str(uuid4())
        future: asyncio.Future = asyncio.Future()

        queued_request = QueuedRequest(
            priority_int=0,  # Will be set in __post_init__
            timestamp=time.monotonic(),
            request_id=request_id,
            priority=priority,
            task=task,
            timeout_ms=timeout_ms,
            future=future,
        )

        async with self._lock:
            # Check if queue is full
            if self.max_size > 0 and self._queue.qsize() >= self.max_size:
                if self.backpressure_mode == BackpressureMode.REJECT:
                    self._total_rejected += 1
                    logger.warning(
                        "request_rejected_queue_full",
                        request_id=request_id,
                        queue_size=self._queue.qsize(),
                        max_size=self.max_size,
                    )
                    metrics.increment_queue_rejected()
                    raise asyncio.QueueFull("Request queue is full")

            # In BLOCK mode, queue.put() will wait for space
            try:
                if timeout_ms:
                    await asyncio.wait_for(
                        self._queue.put(queued_request),
                        timeout=timeout_ms / 1000,
                    )
                else:
                    await self._queue.put(queued_request)

                self._requests[request_id] = queued_request
                self._total_enqueued += 1
                self._priority_counts[priority] += 1

                logger.info(
                    "request_enqueued",
                    request_id=request_id,
                    priority=priority.value,
                    queue_size=self._queue.qsize(),
                    position=self._estimate_position(queued_request),
                )

                # Update metrics
                metrics.increment_queue_request(priority=priority.value)
                metrics.update_queue_size(self._queue.qsize(), priority="all")
                metrics.update_queue_size(
                    self._priority_counts[priority], priority=priority.value
                )

                return request_id

            except asyncio.TimeoutError:
                self._total_rejected += 1
                logger.warning(
                    "request_timeout_enqueue",
                    request_id=request_id,
                    timeout_ms=timeout_ms,
                )
                raise

    async def dequeue(self) -> QueuedRequest:
        """Dequeue the highest priority request.

        Returns:
            Queued request.
        """
        queued_request = await self._queue.get()

        async with self._lock:
            self._total_dequeued += 1
            self._priority_counts[queued_request.priority] -= 1

            # Track wait time
            wait_time_ms = (time.monotonic() - queued_request.enqueued_at) * 1000
            self._wait_times_ms.append(wait_time_ms)
            if len(self._wait_times_ms) > 1000:
                self._wait_times_ms.pop(0)

            logger.debug(
                "request_dequeued",
                request_id=queued_request.request_id,
                priority=queued_request.priority.value,
                wait_time_ms=wait_time_ms,
                queue_size=self._queue.qsize(),
            )

            # Update metrics
            metrics.record_queue_wait_time(
                wait_time_ms / 1000, priority=queued_request.priority.value
            )
            metrics.update_queue_size(self._queue.qsize(), priority="all")
            metrics.update_queue_size(
                self._priority_counts[queued_request.priority],
                priority=queued_request.priority.value,
            )

        return queued_request

    def _estimate_position(self, request: QueuedRequest) -> int:
        """Estimate queue position for a request.

        Args:
            request: Queued request.

        Returns:
            Estimated position (1-indexed).
        """
        # Count requests with higher or equal priority that were enqueued earlier
        position = 1
        for req in self._requests.values():
            if req.priority.value_int > request.priority.value_int:
                position += 1
            elif (
                req.priority.value_int == request.priority.value_int
                and req.timestamp < request.timestamp
            ):
                position += 1
        return position

    def get_position(self, request_id: str) -> Optional[int]:
        """Get current queue position for a request.

        Args:
            request_id: Request ID.

        Returns:
            Queue position (1-indexed) or None if not found.
        """
        request = self._requests.get(request_id)
        if not request:
            return None
        return self._estimate_position(request)

    def get_status(self) -> QueueStatus:
        """Get current queue status.

        Returns:
            Queue status information.
        """
        avg_wait = (
            sum(self._wait_times_ms) / len(self._wait_times_ms)
            if self._wait_times_ms
            else 0.0
        )

        return QueueStatus(
            total_queued=self._queue.qsize(),
            high_priority=self._priority_counts[Priority.HIGH],
            normal_priority=self._priority_counts[Priority.NORMAL],
            low_priority=self._priority_counts[Priority.LOW],
            active_workers=0,  # Set by QueuedExecutor
            max_workers=0,  # Set by QueuedExecutor
            max_queue_size=self.max_size,
            total_processed=self._total_dequeued,
            total_rejected=self._total_rejected,
            average_wait_time_ms=avg_wait,
        )

    def mark_complete(self, request_id: str) -> None:
        """Mark a request as complete and remove from tracking.

        Args:
            request_id: Request ID to mark complete.
        """
        self._requests.pop(request_id, None)

    @property
    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class QueuedExecutor:
    """Executor wrapper with request queuing and worker pool.

    This wraps the graph Executor to provide:
    - Request queuing with priorities
    - Concurrent worker pool
    - Backpressure handling
    - Queue position tracking
    - Graceful shutdown
    """

    def __init__(
        self,
        executor: Executor,
        max_workers: int = 5,
        max_queue_size: int = 1000,
        backpressure_mode: BackpressureMode = BackpressureMode.REJECT,
    ):
        """Initialize queued executor.

        Args:
            executor: Graph executor to wrap.
            max_workers: Maximum concurrent workers.
            max_queue_size: Maximum queue size.
            backpressure_mode: Backpressure handling mode.
        """
        self.executor = executor
        self.max_workers = max_workers
        self.queue = RequestQueue(
            max_size=max_queue_size,
            backpressure_mode=backpressure_mode,
        )

        # Worker management
        self._workers: list[asyncio.Task] = []
        self._worker_health: Dict[int, WorkerHealth] = {}
        self._worker_semaphore = asyncio.Semaphore(max_workers)
        self._shutdown_event = asyncio.Event()
        self._running = False

        logger.info(
            "queued_executor_initialized",
            max_workers=max_workers,
            max_queue_size=max_queue_size,
            graph_id=executor.graph.id,
        )

    async def start(self) -> None:
        """Start worker pool."""
        if self._running:
            logger.warning("worker_pool_already_running")
            return

        self._running = True
        self._shutdown_event.clear()

        # Start worker tasks
        for worker_id in range(self.max_workers):
            worker_task = asyncio.create_task(
                self._worker_loop(worker_id),
                name=f"worker-{worker_id}",
            )
            self._workers.append(worker_task)
            self._worker_health[worker_id] = WorkerHealth(worker_id=worker_id)

        logger.info(
            "worker_pool_started",
            worker_count=self.max_workers,
            graph_id=self.executor.graph.id,
        )

    async def shutdown(self, drain: bool = True, timeout: float = 30.0) -> None:
        """Shutdown worker pool.

        Args:
            drain: If True, process remaining queued requests before shutdown.
            timeout: Maximum time to wait for drain/shutdown.
        """
        if not self._running:
            logger.warning("worker_pool_not_running")
            return

        logger.info(
            "worker_pool_shutdown_initiated",
            drain=drain,
            timeout=timeout,
            queue_size=self.queue.qsize,
        )

        if drain:
            # Wait for queue to drain
            start = time.monotonic()
            while not self.queue.empty:
                if time.monotonic() - start > timeout:
                    logger.warning(
                        "drain_timeout_exceeded",
                        remaining=self.queue.qsize,
                    )
                    break
                await asyncio.sleep(0.1)

        # Signal shutdown
        self._running = False
        self._shutdown_event.set()

        # Wait for workers to complete
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("worker_shutdown_timeout")
                for worker in self._workers:
                    worker.cancel()

        self._workers.clear()
        logger.info("worker_pool_shutdown_complete")

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop that processes requests from queue.

        Args:
            worker_id: Worker identifier.
        """
        logger.info("worker_started", worker_id=worker_id)
        health = self._worker_health[worker_id]

        try:
            while self._running or not self.queue.empty:
                try:
                    # Get next request (with timeout to check shutdown)
                    queued_request = await asyncio.wait_for(
                        self.queue.dequeue(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Update health
                health.current_request_id = queued_request.request_id
                health.started_at = time.monotonic()

                # Update active workers metric
                active_count = sum(
                    1 for h in self._worker_health.values()
                    if h.current_request_id is not None
                )
                metrics.update_active_workers(active_count)

                # Process request
                try:
                    async with self._worker_semaphore:
                        response = await self._execute_request(
                            queued_request, worker_id
                        )

                        if not queued_request.future.done():
                            queued_request.future.set_result(response)

                        health.total_processed += 1

                except Exception as e:
                    logger.error(
                        "worker_execution_error",
                        worker_id=worker_id,
                        request_id=queued_request.request_id,
                        error=str(e),
                        exc_info=True,
                    )
                    health.total_errors += 1
                    health.is_healthy = health.total_errors < 10

                    if not queued_request.future.done():
                        queued_request.future.set_exception(e)

                finally:
                    # Clean up
                    self.queue.mark_complete(queued_request.request_id)
                    health.current_request_id = None
                    health.started_at = None

                    # Update active workers metric
                    active_count = sum(
                        1 for h in self._worker_health.values()
                        if h.current_request_id is not None
                    )
                    metrics.update_active_workers(active_count)

        except asyncio.CancelledError:
            logger.info("worker_cancelled", worker_id=worker_id)
        except Exception as e:
            logger.error(
                "worker_fatal_error",
                worker_id=worker_id,
                error=str(e),
                exc_info=True,
            )
            health.is_healthy = False
        finally:
            logger.info(
                "worker_stopped",
                worker_id=worker_id,
                total_processed=health.total_processed,
                total_errors=health.total_errors,
            )

    async def _execute_request(
        self,
        queued_request: QueuedRequest,
        worker_id: int,
    ) -> TaskResponse:
        """Execute a queued request.

        Args:
            queued_request: Request to execute.
            worker_id: Worker executing the request.

        Returns:
            Task response.
        """
        logger.info(
            "request_execution_started",
            request_id=queued_request.request_id,
            worker_id=worker_id,
            priority=queued_request.priority.value,
        )

        start_time = time.monotonic()

        try:
            # Execute with timeout if specified
            if queued_request.timeout_ms:
                response = await asyncio.wait_for(
                    self.executor.execute(queued_request.task),
                    timeout=queued_request.timeout_ms / 1000,
                )
            else:
                response = await self.executor.execute(queued_request.task)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "request_execution_completed",
                request_id=queued_request.request_id,
                worker_id=worker_id,
                success=response.success,
                elapsed_ms=elapsed_ms,
            )

            return response

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "request_execution_timeout",
                request_id=queued_request.request_id,
                worker_id=worker_id,
                timeout_ms=queued_request.timeout_ms,
                elapsed_ms=elapsed_ms,
            )

            return TaskResponse(
                trace_id=queued_request.request_id,
                success=False,
                error=ErrorInfo(
                    code=ErrorInfo.Codes.TIMEOUT,
                    message=f"Request timed out after {queued_request.timeout_ms}ms",
                    recoverable=False,
                ),
                total_latency_ms=int(elapsed_ms),
                nodes_executed=0,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "request_execution_error",
                request_id=queued_request.request_id,
                worker_id=worker_id,
                error=str(e),
                elapsed_ms=elapsed_ms,
                exc_info=True,
            )

            return TaskResponse(
                trace_id=queued_request.request_id,
                success=False,
                error=ErrorInfo(
                    code=ErrorInfo.Codes.UNKNOWN,
                    message=str(e),
                    recoverable=True,
                ),
                total_latency_ms=int(elapsed_ms),
                nodes_executed=0,
            )

    async def submit(
        self,
        task: TaskPayload,
        priority: Priority = Priority.NORMAL,
        timeout_ms: Optional[int] = None,
    ) -> asyncio.Future:
        """Submit a request to the queue.

        Args:
            task: Task to execute.
            priority: Request priority.
            timeout_ms: Optional timeout for execution.

        Returns:
            Future that will contain the TaskResponse.

        Raises:
            RuntimeError: If worker pool is not running.
        """
        if not self._running:
            raise RuntimeError("Worker pool is not running. Call start() first.")

        request_id = await self.queue.enqueue(task, priority, timeout_ms)
        request = self.queue._requests.get(request_id)

        if not request:
            raise RuntimeError(f"Request {request_id} not found after enqueue")

        logger.debug(
            "request_submitted",
            request_id=request_id,
            priority=priority.value,
            position=self.queue.get_position(request_id),
        )

        return request.future

    async def execute(
        self,
        task: TaskPayload,
        priority: Priority = Priority.NORMAL,
        timeout_ms: Optional[int] = None,
    ) -> TaskResponse:
        """Submit and wait for request completion.

        Args:
            task: Task to execute.
            priority: Request priority.
            timeout_ms: Optional timeout for execution.

        Returns:
            Task response.
        """
        future = await self.submit(task, priority, timeout_ms)
        return await future

    def get_status(self) -> QueueStatus:
        """Get queue and worker status.

        Returns:
            Queue status with worker information.
        """
        status = self.queue.get_status()

        # Add worker info
        active_workers = sum(
            1 for h in self._worker_health.values()
            if h.current_request_id is not None
        )
        status.active_workers = active_workers
        status.max_workers = self.max_workers

        return status

    def get_worker_health(self) -> Dict[int, WorkerHealth]:
        """Get health status for all workers.

        Returns:
            Dictionary mapping worker ID to health status.
        """
        return self._worker_health.copy()

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Context manager for executor lifespan.

        Example:
            async with queued_executor.lifespan():
                response = await queued_executor.execute(task)
        """
        await self.start()
        try:
            yield
        finally:
            await self.shutdown(drain=True)
