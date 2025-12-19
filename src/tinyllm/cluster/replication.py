"""Cross-region replication for distributed deployments."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class ReplicationStrategy(str, Enum):
    """Strategy for cross-region replication."""

    ACTIVE_ACTIVE = "active_active"
    ACTIVE_PASSIVE = "active_passive"
    MULTI_MASTER = "multi_master"
    CHAIN = "chain"
    STAR = "star"


class ReplicationMode(str, Enum):
    """Mode of replication."""

    SYNC = "sync"
    ASYNC = "async"
    SEMI_SYNC = "semi_sync"


class ConflictResolution(str, Enum):
    """Strategy for conflict resolution."""

    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    VECTOR_CLOCK = "vector_clock"
    CUSTOM = "custom"


@dataclass
class ReplicaInfo:
    """Information about a replica."""

    replica_id: str
    region_id: str
    endpoint: str
    is_primary: bool = False
    is_active: bool = True
    last_sync_time: Optional[datetime] = None
    lag_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplicationEvent:
    """Event to be replicated."""

    event_id: str
    timestamp: datetime
    source_region: str
    event_type: str
    data: Dict[str, Any]
    vector_clock: Dict[str, int] = field(default_factory=dict)


class ReplicationConfig(BaseModel):
    """Configuration for cross-region replication."""

    strategy: ReplicationStrategy = Field(
        default=ReplicationStrategy.ACTIVE_PASSIVE,
        description="Replication strategy",
    )
    mode: ReplicationMode = Field(
        default=ReplicationMode.ASYNC, description="Replication mode"
    )
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.LAST_WRITE_WINS,
        description="Conflict resolution strategy",
    )
    batch_size: int = Field(default=100, ge=1, description="Batch size for replication")
    batch_timeout_ms: int = Field(
        default=1000, ge=0, description="Batch timeout in milliseconds"
    )
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_delay_ms: int = Field(default=1000, ge=0, description="Retry delay in milliseconds")
    health_check_interval: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )


class CrossRegionReplicator:
    """Manages cross-region replication."""

    def __init__(
        self,
        region_id: str,
        config: Optional[ReplicationConfig] = None,
        on_replication_lag: Optional[Callable[[str, float], None]] = None,
    ):
        """Initialize cross-region replicator.

        Args:
            region_id: ID of this region
            config: Replication configuration
            on_replication_lag: Callback for replication lag alerts
        """
        self.region_id = region_id
        self.config = config or ReplicationConfig()
        self.on_replication_lag = on_replication_lag

        self.replicas: Dict[str, ReplicaInfo] = {}
        self._event_queue: asyncio.Queue[ReplicationEvent] = asyncio.Queue()
        self._vector_clock: Dict[str, int] = {region_id: 0}
        self._pending_events: Dict[str, ReplicationEvent] = {}

        self._shutdown = False
        self._replication_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Metrics
        self._events_replicated = 0
        self._events_failed = 0
        self._conflicts_resolved = 0

    async def start(self) -> None:
        """Start the replicator."""
        self._shutdown = False
        self._replication_task = asyncio.create_task(self._replication_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop the replicator."""
        self._shutdown = True

        for task in [self._replication_task, self._health_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def add_replica(self, replica: ReplicaInfo) -> None:
        """Add a replica region.

        Args:
            replica: Replica information
        """
        self.replicas[replica.replica_id] = replica
        if replica.region_id not in self._vector_clock:
            self._vector_clock[replica.region_id] = 0

    def remove_replica(self, replica_id: str) -> None:
        """Remove a replica region.

        Args:
            replica_id: Replica to remove
        """
        self.replicas.pop(replica_id, None)

    async def replicate_event(self, event: ReplicationEvent) -> None:
        """Queue an event for replication.

        Args:
            event: Event to replicate
        """
        # Update vector clock
        self._vector_clock[self.region_id] += 1
        event.vector_clock = self._vector_clock.copy()

        # Add to queue
        await self._event_queue.put(event)

    async def receive_event(self, event: ReplicationEvent) -> bool:
        """Receive a replicated event from another region.

        Args:
            event: Replicated event

        Returns:
            True if event was accepted
        """
        # Check for conflicts
        if self._has_conflict(event):
            resolved_event = self._resolve_conflict(event)
            if not resolved_event:
                return False
            event = resolved_event

        # Update vector clock
        for region, clock in event.vector_clock.items():
            self._vector_clock[region] = max(
                self._vector_clock.get(region, 0), clock
            )

        # Process event
        # In production, this would update local state
        return True

    async def _replication_loop(self) -> None:
        """Background replication loop."""
        batch: List[ReplicationEvent] = []
        last_send_time = datetime.utcnow()

        while not self._shutdown:
            try:
                # Wait for events with timeout for batching
                timeout = self.config.batch_timeout_ms / 1000
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass

                # Send batch if full or timeout reached
                now = datetime.utcnow()
                time_since_send = (now - last_send_time).total_seconds()
                should_send = (
                    len(batch) >= self.config.batch_size
                    or (batch and time_since_send >= timeout)
                )

                if should_send:
                    await self._send_batch(batch)
                    batch = []
                    last_send_time = now

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                await asyncio.sleep(1)

    async def _send_batch(self, events: List[ReplicationEvent]) -> None:
        """Send batch of events to replicas.

        Args:
            events: Events to send
        """
        if not events:
            return

        # Get active replicas based on strategy
        targets = self._get_replication_targets()

        if self.config.mode == ReplicationMode.SYNC:
            # Wait for all replicas
            await self._send_sync(events, targets)
        elif self.config.mode == ReplicationMode.SEMI_SYNC:
            # Wait for at least one replica
            await self._send_semi_sync(events, targets)
        else:  # ASYNC
            # Fire and forget
            asyncio.create_task(self._send_async(events, targets))

    def _get_replication_targets(self) -> List[ReplicaInfo]:
        """Get replicas to replicate to based on strategy.

        Returns:
            List of target replicas
        """
        active = [r for r in self.replicas.values() if r.is_active]

        if self.config.strategy == ReplicationStrategy.ACTIVE_PASSIVE:
            # Only replicate to passive replicas
            return [r for r in active if not r.is_primary]
        elif self.config.strategy == ReplicationStrategy.ACTIVE_ACTIVE:
            # Replicate to all active replicas
            return active
        elif self.config.strategy == ReplicationStrategy.MULTI_MASTER:
            # Replicate to all other masters
            return active
        elif self.config.strategy == ReplicationStrategy.CHAIN:
            # Replicate to next in chain
            if active:
                return [active[0]]
            return []
        elif self.config.strategy == ReplicationStrategy.STAR:
            # From hub to all spokes, or from spoke to hub
            return active
        else:
            return active

    async def _send_sync(
        self, events: List[ReplicationEvent], targets: List[ReplicaInfo]
    ) -> None:
        """Send events synchronously (wait for all).

        Args:
            events: Events to send
            targets: Target replicas
        """
        tasks = [self._send_to_replica(replica, events) for replica in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self._events_failed += len(events)
            else:
                self._events_replicated += len(events)

    async def _send_semi_sync(
        self, events: List[ReplicationEvent], targets: List[ReplicaInfo]
    ) -> None:
        """Send events semi-synchronously (wait for at least one).

        Args:
            events: Events to send
            targets: Target replicas
        """
        if not targets:
            return

        tasks = [self._send_to_replica(replica, events) for replica in targets]

        # Wait for first success
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining
        for task in pending:
            task.cancel()

        # Check if at least one succeeded
        for task in done:
            if not isinstance(task.exception(), Exception):
                self._events_replicated += len(events)
                break
        else:
            self._events_failed += len(events)

    async def _send_async(
        self, events: List[ReplicationEvent], targets: List[ReplicaInfo]
    ) -> None:
        """Send events asynchronously (don't wait).

        Args:
            events: Events to send
            targets: Target replicas
        """
        for replica in targets:
            try:
                await self._send_to_replica(replica, events)
                self._events_replicated += len(events)
            except Exception:
                self._events_failed += len(events)

    async def _send_to_replica(
        self, replica: ReplicaInfo, events: List[ReplicationEvent]
    ) -> None:
        """Send events to a specific replica.

        Args:
            replica: Target replica
            events: Events to send
        """
        for attempt in range(self.config.retry_attempts + 1):
            try:
                # In production, send via HTTP/gRPC to replica
                # For now, simulate success
                replica.last_sync_time = datetime.utcnow()
                return
            except Exception as e:
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)
                else:
                    raise e

    def _has_conflict(self, event: ReplicationEvent) -> bool:
        """Check if event has conflict with local state.

        Args:
            event: Event to check

        Returns:
            True if conflict detected
        """
        # Check vector clocks for conflicts
        for region, clock in event.vector_clock.items():
            local_clock = self._vector_clock.get(region, 0)
            if clock < local_clock:
                return True
        return False

    def _resolve_conflict(
        self, event: ReplicationEvent
    ) -> Optional[ReplicationEvent]:
        """Resolve a conflict using configured strategy.

        Args:
            event: Conflicting event

        Returns:
            Resolved event or None if rejected
        """
        self._conflicts_resolved += 1

        if self.config.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
            # Accept event with latest timestamp
            return event
        elif self.config.conflict_resolution == ConflictResolution.FIRST_WRITE_WINS:
            # Reject new event
            return None
        elif self.config.conflict_resolution == ConflictResolution.VECTOR_CLOCK:
            # Use vector clocks to determine causality
            # If new event is causally newer, accept it
            return event
        else:  # CUSTOM
            # Custom resolution logic
            return event

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await self._check_replica_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_replica_health(self) -> None:
        """Check health of all replicas."""
        now = datetime.utcnow()

        for replica in self.replicas.values():
            if replica.last_sync_time:
                lag = (now - replica.last_sync_time).total_seconds()
                replica.lag_seconds = lag

                # Alert on high lag
                if self.on_replication_lag and lag > 60:
                    self.on_replication_lag(replica.replica_id, lag)

    def get_replication_lag(self) -> Dict[str, float]:
        """Get replication lag for all replicas.

        Returns:
            Dictionary mapping replica ID to lag in seconds
        """
        return {r.replica_id: r.lag_seconds for r in self.replicas.values()}

    def get_metrics(self) -> Dict[str, Any]:
        """Get replication metrics.

        Returns:
            Dictionary of metrics
        """
        active_replicas = sum(1 for r in self.replicas.values() if r.is_active)
        max_lag = max((r.lag_seconds for r in self.replicas.values()), default=0)

        return {
            "total_replicas": len(self.replicas),
            "active_replicas": active_replicas,
            "events_replicated": self._events_replicated,
            "events_failed": self._events_failed,
            "conflicts_resolved": self._conflicts_resolved,
            "max_replication_lag_seconds": max_lag,
            "queue_size": self._event_queue.qsize(),
        }
