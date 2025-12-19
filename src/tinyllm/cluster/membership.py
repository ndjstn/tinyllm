"""Cluster membership management."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class MemberState(str, Enum):
    """State of a cluster member."""

    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"
    LEFT = "left"


@dataclass
class MemberInfo:
    """Information about a cluster member."""

    node_id: str
    host: str
    port: int
    state: MemberState = MemberState.ALIVE
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    incarnation: int = 0  # Used for conflict resolution

    @property
    def address(self) -> str:
        """Get the member's address."""
        return f"{self.host}:{self.port}"


class MembershipConfig(BaseModel):
    """Configuration for cluster membership."""

    gossip_interval: int = Field(default=1, ge=1, description="Gossip interval in seconds")
    probe_interval: int = Field(default=5, ge=1, description="Probe interval in seconds")
    probe_timeout: int = Field(default=3, ge=1, description="Probe timeout in seconds")
    suspect_timeout: int = Field(default=10, ge=1, description="Time before marking suspect")
    dead_timeout: int = Field(default=30, ge=1, description="Time before marking dead")
    gossip_fanout: int = Field(default=3, ge=1, description="Number of nodes to gossip to")


class ClusterMembership:
    """Manages cluster membership using a gossip protocol."""

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        config: Optional[MembershipConfig] = None,
        on_member_joined: Optional[Callable[[MemberInfo], None]] = None,
        on_member_left: Optional[Callable[[MemberInfo], None]] = None,
        on_member_failed: Optional[Callable[[MemberInfo], None]] = None,
    ):
        """Initialize cluster membership.

        Args:
            node_id: Unique identifier for this node
            host: Host address of this node
            port: Port of this node
            config: Membership configuration
            on_member_joined: Callback when a member joins
            on_member_left: Callback when a member leaves
            on_member_failed: Callback when a member fails
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.config = config or MembershipConfig()
        self.on_member_joined = on_member_joined
        self.on_member_left = on_member_left
        self.on_member_failed = on_member_failed

        # Self member info
        self.self_member = MemberInfo(
            node_id=node_id, host=host, port=port, state=MemberState.ALIVE
        )

        # Cluster members
        self.members: Dict[str, MemberInfo] = {node_id: self.self_member}

        self._shutdown = False
        self._gossip_task: Optional[asyncio.Task] = None
        self._probe_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the membership protocol."""
        self._shutdown = False
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        self._probe_task = asyncio.create_task(self._probe_loop())

    async def stop(self) -> None:
        """Stop the membership protocol."""
        self._shutdown = True

        # Mark self as left
        self.self_member.state = MemberState.LEFT

        # Cancel tasks
        for task in [self._gossip_task, self._probe_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def join(self, seed_nodes: List[tuple[str, int]]) -> None:
        """Join the cluster using seed nodes.

        Args:
            seed_nodes: List of (host, port) tuples for seed nodes
        """
        # In a real implementation, contact seed nodes to join
        # For now, just mark as joined
        pass

    async def leave(self) -> None:
        """Gracefully leave the cluster."""
        self.self_member.state = MemberState.LEFT

        # Broadcast leave message via gossip
        await self._broadcast_state_change(self.self_member)

        # Stop membership protocol
        await self.stop()

    def add_member(self, member: MemberInfo) -> bool:
        """Add or update a member.

        Args:
            member: Member information

        Returns:
            True if member was added/updated
        """
        if member.node_id == self.node_id:
            return False

        existing = self.members.get(member.node_id)

        # Update if newer incarnation or state change
        if not existing or member.incarnation > existing.incarnation:
            old_state = existing.state if existing else None
            self.members[member.node_id] = member

            # Trigger callbacks
            if not existing and self.on_member_joined:
                self.on_member_joined(member)
            elif existing and member.state == MemberState.LEFT and self.on_member_left:
                self.on_member_left(member)
            elif existing and member.state == MemberState.DEAD and self.on_member_failed:
                self.on_member_failed(member)

            return True

        return False

    def remove_member(self, node_id: str) -> Optional[MemberInfo]:
        """Remove a member from the cluster.

        Args:
            node_id: Node to remove

        Returns:
            Removed member info if found
        """
        return self.members.pop(node_id, None)

    def get_member(self, node_id: str) -> Optional[MemberInfo]:
        """Get member information.

        Args:
            node_id: Node identifier

        Returns:
            Member information if found
        """
        return self.members.get(node_id)

    def get_alive_members(self) -> List[MemberInfo]:
        """Get list of alive members.

        Returns:
            List of alive members
        """
        return [m for m in self.members.values() if m.state == MemberState.ALIVE]

    def get_all_members(self) -> List[MemberInfo]:
        """Get all members.

        Returns:
            List of all members
        """
        return list(self.members.values())

    def get_member_count(self) -> int:
        """Get total member count."""
        return len(self.members)

    async def _gossip_loop(self) -> None:
        """Background gossip loop."""
        while not self._shutdown:
            try:
                await self._perform_gossip()
                await asyncio.sleep(self.config.gossip_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                await asyncio.sleep(self.config.gossip_interval)

    async def _perform_gossip(self) -> None:
        """Perform one round of gossip."""
        # Select random nodes to gossip to
        alive_members = [m for m in self.members.values() if m.state == MemberState.ALIVE]

        if not alive_members:
            return

        import random

        targets = random.sample(
            alive_members, min(self.config.gossip_fanout, len(alive_members))
        )

        # In a real implementation, send membership list to targets
        # For now, just update last_seen
        self.self_member.last_seen = datetime.utcnow()

    async def _probe_loop(self) -> None:
        """Background probe loop to detect failures."""
        while not self._shutdown:
            try:
                await self._perform_probes()
                await asyncio.sleep(self.config.probe_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                await asyncio.sleep(self.config.probe_interval)

    async def _perform_probes(self) -> None:
        """Perform failure detection probes."""
        now = datetime.utcnow()

        for member in list(self.members.values()):
            if member.node_id == self.node_id:
                continue

            time_since_seen = (now - member.last_seen).total_seconds()

            # Update member state based on time since last seen
            if member.state == MemberState.ALIVE:
                if time_since_seen > self.config.suspect_timeout:
                    member.state = MemberState.SUSPECT
                    member.incarnation += 1
            elif member.state == MemberState.SUSPECT:
                if time_since_seen > self.config.dead_timeout:
                    member.state = MemberState.DEAD
                    member.incarnation += 1
                    if self.on_member_failed:
                        self.on_member_failed(member)

        # Remove dead members after a while
        dead_timeout = timedelta(seconds=self.config.dead_timeout * 2)
        for node_id, member in list(self.members.items()):
            if member.state == MemberState.DEAD:
                if now - member.last_seen > dead_timeout:
                    self.remove_member(node_id)

    async def _broadcast_state_change(self, member: MemberInfo) -> None:
        """Broadcast a state change to all members.

        Args:
            member: Member with state change
        """
        # In a real implementation, send state change to all alive members
        pass

    async def receive_gossip(self, members: List[MemberInfo]) -> None:
        """Receive gossip from another node.

        Args:
            members: Member information from another node
        """
        for member in members:
            self.add_member(member)

    def get_metrics(self) -> Dict[str, Any]:
        """Get membership metrics.

        Returns:
            Dictionary of metrics
        """
        states = {}
        for state in MemberState:
            states[state.value] = sum(1 for m in self.members.values() if m.state == state)

        return {
            "total_members": len(self.members),
            "alive_members": states.get(MemberState.ALIVE.value, 0),
            "suspect_members": states.get(MemberState.SUSPECT.value, 0),
            "dead_members": states.get(MemberState.DEAD.value, 0),
            "left_members": states.get(MemberState.LEFT.value, 0),
        }
