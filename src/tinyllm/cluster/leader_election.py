"""Leader election for distributed coordination."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field


class LeadershipState(str, Enum):
    """State of leadership."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LeaderElectionConfig(BaseModel):
    """Configuration for leader election."""

    election_timeout_min: int = Field(default=1500, ge=100, description="Min timeout in ms")
    election_timeout_max: int = Field(default=3000, ge=100, description="Max timeout in ms")
    heartbeat_interval: int = Field(default=500, ge=50, description="Heartbeat interval in ms")
    lease_duration: int = Field(default=10, ge=1, description="Leader lease duration in seconds")


@dataclass
class LeaderInfo:
    """Information about the current leader."""

    node_id: str
    term: int
    elected_at: datetime
    last_heartbeat: datetime
    metadata: Dict[str, Any]


class LeaderElector:
    """Implements leader election using a simplified Raft-like protocol."""

    def __init__(
        self,
        node_id: str,
        config: Optional[LeaderElectionConfig] = None,
        on_became_leader: Optional[Callable[[], None]] = None,
        on_lost_leadership: Optional[Callable[[], None]] = None,
    ):
        """Initialize leader elector.

        Args:
            node_id: Unique identifier for this node
            config: Election configuration
            on_became_leader: Callback when becoming leader
            on_lost_leadership: Callback when losing leadership
        """
        self.node_id = node_id
        self.config = config or LeaderElectionConfig()
        self.on_became_leader = on_became_leader
        self.on_lost_leadership = on_lost_leadership

        self.state = LeadershipState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.leader_info: Optional[LeaderInfo] = None

        self._shutdown = False
        self._election_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat_received: Optional[datetime] = None

    async def start(self) -> None:
        """Start the leader election process."""
        self._shutdown = False
        self._election_task = asyncio.create_task(self._election_loop())

    async def stop(self) -> None:
        """Stop the leader election process."""
        self._shutdown = True

        if self._election_task:
            self._election_task.cancel()
            try:
                await self._election_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    def is_leader(self) -> bool:
        """Check if this node is the current leader.

        Returns:
            True if this node is the leader
        """
        return self.state == LeadershipState.LEADER

    def get_leader(self) -> Optional[LeaderInfo]:
        """Get information about the current leader.

        Returns:
            Leader information if known, None otherwise
        """
        return self.leader_info

    async def _election_loop(self) -> None:
        """Main election loop."""
        while not self._shutdown:
            try:
                if self.state == LeadershipState.FOLLOWER:
                    await self._follower_loop()
                elif self.state == LeadershipState.CANDIDATE:
                    await self._candidate_loop()
                elif self.state == LeadershipState.LEADER:
                    await self._leader_loop()
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error and continue
                await asyncio.sleep(1)

    async def _follower_loop(self) -> None:
        """Follower state loop."""
        # Wait for heartbeat or timeout
        timeout = self._random_election_timeout()

        try:
            await asyncio.wait_for(self._wait_for_heartbeat(), timeout=timeout / 1000)
        except asyncio.TimeoutError:
            # No heartbeat received, become candidate
            await self._become_candidate()

    async def _candidate_loop(self) -> None:
        """Candidate state loop."""
        # Increment term and vote for self
        self.current_term += 1
        self.voted_for = self.node_id
        votes_received = 1

        # In a real implementation, request votes from other nodes
        # For simplicity, we simulate by becoming leader immediately
        # (assuming single-node cluster or testing)
        await self._become_leader()

    async def _leader_loop(self) -> None:
        """Leader state loop."""
        # Send heartbeats
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

        # Check if we still have the lease
        if self.leader_info:
            elapsed = (datetime.utcnow() - self.leader_info.last_heartbeat).total_seconds()
            if elapsed > self.config.lease_duration:
                # Lost leadership
                await self._become_follower()

    async def _become_follower(self) -> None:
        """Transition to follower state."""
        was_leader = self.state == LeadershipState.LEADER
        self.state = LeadershipState.FOLLOWER
        self.leader_info = None

        if was_leader and self.on_lost_leadership:
            self.on_lost_leadership()

    async def _become_candidate(self) -> None:
        """Transition to candidate state."""
        self.state = LeadershipState.CANDIDATE
        self.voted_for = None

    async def _become_leader(self) -> None:
        """Transition to leader state."""
        was_follower = self.state != LeadershipState.LEADER
        self.state = LeadershipState.LEADER
        self.leader_info = LeaderInfo(
            node_id=self.node_id,
            term=self.current_term,
            elected_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            metadata={},
        )

        if was_follower and self.on_became_leader:
            self.on_became_leader()

    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeats as leader."""
        while self.state == LeadershipState.LEADER and not self._shutdown:
            try:
                # Update heartbeat time
                if self.leader_info:
                    self.leader_info.last_heartbeat = datetime.utcnow()

                # In a real implementation, send heartbeats to all followers
                await asyncio.sleep(self.config.heartbeat_interval / 1000)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                await asyncio.sleep(self.config.heartbeat_interval / 1000)

    async def _wait_for_heartbeat(self) -> None:
        """Wait for a heartbeat from the leader."""
        # In a real implementation, this would wait for an actual heartbeat message
        # For testing, we just wait indefinitely
        await asyncio.sleep(float("inf"))

    def _random_election_timeout(self) -> int:
        """Generate a random election timeout.

        Returns:
            Timeout in milliseconds
        """
        import random

        return random.randint(
            self.config.election_timeout_min, self.config.election_timeout_max
        )

    async def receive_heartbeat(self, leader_id: str, term: int) -> None:
        """Receive a heartbeat from the leader.

        Args:
            leader_id: ID of the leader sending the heartbeat
            term: Current term of the leader
        """
        # Update term if higher
        if term > self.current_term:
            self.current_term = term
            await self._become_follower()

        # Update leader info
        if term >= self.current_term and self.state == LeadershipState.FOLLOWER:
            self.leader_info = LeaderInfo(
                node_id=leader_id,
                term=term,
                elected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                metadata={},
            )
            self._last_heartbeat_received = datetime.utcnow()

    def get_state(self) -> LeadershipState:
        """Get current leadership state.

        Returns:
            Current state
        """
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get election metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "state": self.state.value,
            "is_leader": self.is_leader(),
            "current_term": self.current_term,
            "leader_id": self.leader_info.node_id if self.leader_info else None,
            "voted_for": self.voted_for,
        }
