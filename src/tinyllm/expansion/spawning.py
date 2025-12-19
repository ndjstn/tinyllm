"""Dynamic node spawning system for TinyLLM.

This module provides dynamic node spawning capabilities for runtime
graph expansion, allowing nodes to be created, cloned, and specialized
based on performance metrics and workload requirements.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.node import BaseNode
from tinyllm.core.registry import NodeRegistry


class SpawnTrigger(str, Enum):
    """Trigger reasons for spawning new nodes."""

    PERFORMANCE_THRESHOLD = "performance_threshold"  # Performance below threshold
    WORKLOAD_SPIKE = "workload_spike"  # Sudden increase in workload
    SPECIALIZATION_NEEDED = "specialization_needed"  # Need specialized node
    MANUAL = "manual"  # Manual spawn request


class SpawnConfig(BaseModel):
    """Configuration for node spawning behavior."""

    model_config = {"extra": "forbid"}

    max_spawns: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of spawns allowed",
    )
    cooldown_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Cooldown period between spawns in milliseconds",
    )
    performance_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Performance threshold below which spawning is triggered",
    )
    workload_threshold: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Workload threshold that triggers spawning",
    )
    enable_auto_spawn: bool = Field(
        default=False,
        description="Enable automatic spawning based on metrics",
    )
    max_spawn_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum depth of spawn chains (clones of clones)",
    )

    @field_validator("cooldown_ms")
    @classmethod
    def validate_cooldown(cls, v: int) -> int:
        """Ensure cooldown is reasonable."""
        if v < 100:
            raise ValueError("Cooldown must be at least 100ms")
        return v


class SpawnRecord(BaseModel):
    """Record of a node spawn operation."""

    model_config = {"extra": "forbid"}

    id: str = Field(
        default_factory=lambda: f"spawn_{uuid4().hex[:8]}",
        description="Unique spawn record ID",
    )
    spawned_node_id: str = Field(description="ID of the spawned node")
    template_node_id: str = Field(description="ID of the template/parent node")
    trigger: SpawnTrigger = Field(description="What triggered the spawn")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the spawn occurred",
    )
    specialization: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specialization parameters applied",
    )
    spawn_depth: int = Field(
        default=0,
        ge=0,
        description="Depth in the spawn chain (0 = from original template)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional spawn metadata",
    )


class SpawnMetrics(BaseModel):
    """Metrics for spawn operations."""

    model_config = {"extra": "forbid"}

    total_spawns: int = Field(default=0, ge=0, description="Total spawns created")
    active_spawns: int = Field(
        default=0,
        ge=0,
        description="Currently active spawned nodes",
    )
    failed_spawns: int = Field(default=0, ge=0, description="Failed spawn attempts")
    avg_spawn_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average spawn operation latency",
    )
    spawns_by_trigger: Dict[str, int] = Field(
        default_factory=dict,
        description="Spawn counts by trigger type",
    )


class NodeFactory:
    """Factory for creating node instances dynamically."""

    @staticmethod
    def create_from_definition(definition: NodeDefinition) -> BaseNode:
        """Create a node from a NodeDefinition.

        Args:
            definition: Node definition to instantiate.

        Returns:
            Instantiated node.

        Raises:
            ValueError: If node type is not registered.
        """
        return NodeRegistry.create(definition)

    @staticmethod
    def create_from_template(
        template: NodeDefinition,
        new_id: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> BaseNode:
        """Create a node from a template with modifications.

        Args:
            template: Template node definition.
            new_id: ID for the new node.
            config_overrides: Configuration overrides to apply.

        Returns:
            New node instance.

        Raises:
            ValueError: If template is invalid or node type not registered.
        """
        # Validate new_id format
        if not new_id or not isinstance(new_id, str):
            raise ValueError("new_id must be a non-empty string")

        # Create new config by merging template config with overrides
        new_config = {**template.config}
        if config_overrides:
            new_config.update(config_overrides)

        # Create new definition
        new_definition = NodeDefinition(
            id=new_id,
            type=template.type,
            name=template.name,
            description=template.description,
            config=new_config,
        )

        return NodeFactory.create_from_definition(new_definition)

    @staticmethod
    def clone_node_definition(
        node: BaseNode,
        new_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> NodeDefinition:
        """Create a NodeDefinition by cloning an existing node.

        Args:
            node: Node to clone.
            new_id: ID for the cloned node.
            modifications: Modifications to apply to config.

        Returns:
            New node definition.

        Raises:
            ValueError: If node or new_id is invalid.
        """
        if not isinstance(node, BaseNode):
            raise ValueError("node must be a BaseNode instance")
        if not new_id or not isinstance(new_id, str):
            raise ValueError("new_id must be a non-empty string")

        # Build config from node's raw config
        new_config = {**node._raw_config}
        if modifications:
            new_config.update(modifications)

        return NodeDefinition(
            id=new_id,
            type=node.type,
            name=node.name,
            description=node.description,
            config=new_config,
        )


class NodeSpawner:
    """Manages dynamic node spawning with limits and cooldowns."""

    def __init__(self, config: Optional[SpawnConfig] = None):
        """Initialize the node spawner.

        Args:
            config: Spawn configuration. Uses defaults if not provided.
        """
        self.config = config or SpawnConfig()
        self._spawn_history: List[SpawnRecord] = []
        self._active_spawns: Dict[str, SpawnRecord] = {}
        self._last_spawn_time: Optional[datetime] = None
        self._factory = NodeFactory()
        self._metrics = SpawnMetrics()

    def can_spawn(self, check_cooldown: bool = True) -> bool:
        """Check if spawning is currently allowed.

        Args:
            check_cooldown: Whether to check cooldown period.

        Returns:
            True if spawning is allowed, False otherwise.
        """
        # Check max spawns limit
        if len(self._active_spawns) >= self.config.max_spawns:
            return False

        # Check cooldown if requested
        if check_cooldown and self._last_spawn_time:
            elapsed = datetime.utcnow() - self._last_spawn_time
            cooldown_delta = timedelta(milliseconds=self.config.cooldown_ms)
            if elapsed < cooldown_delta:
                return False

        return True

    def spawn_node(
        self,
        template: NodeDefinition,
        specialization: Optional[Dict[str, Any]] = None,
        trigger: SpawnTrigger = SpawnTrigger.MANUAL,
        spawn_depth: int = 0,
    ) -> BaseNode:
        """Spawn a new node from a template.

        Args:
            template: Template node definition.
            specialization: Specialization parameters to apply.
            trigger: What triggered this spawn.
            spawn_depth: Depth in spawn chain.

        Returns:
            Spawned node instance.

        Raises:
            RuntimeError: If spawning is not allowed.
            ValueError: If template is invalid or spawn depth exceeded.
        """
        if not self.can_spawn():
            raise RuntimeError(
                f"Cannot spawn: max_spawns={self.config.max_spawns}, "
                f"active={len(self._active_spawns)}, "
                f"cooldown_ms={self.config.cooldown_ms}"
            )

        if spawn_depth > self.config.max_spawn_depth:
            raise ValueError(
                f"Spawn depth {spawn_depth} exceeds max {self.config.max_spawn_depth}"
            )

        # Generate unique ID for spawned node
        spawn_id = f"{template.id}_spawn_{uuid4().hex[:8]}"

        # Create the node
        start_time = datetime.utcnow()
        try:
            node = self._factory.create_from_template(
                template=template,
                new_id=spawn_id,
                config_overrides=specialization,
            )

            # Record the spawn
            spawn_record = SpawnRecord(
                spawned_node_id=spawn_id,
                template_node_id=template.id,
                trigger=trigger,
                specialization=specialization or {},
                spawn_depth=spawn_depth,
                metadata={
                    "template_type": template.type.value,
                    "template_name": template.name,
                },
            )

            self._spawn_history.append(spawn_record)
            self._active_spawns[spawn_id] = spawn_record
            self._last_spawn_time = datetime.utcnow()

            # Update metrics
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_metrics(trigger, latency_ms, success=True)

            return node

        except Exception as e:
            self._update_metrics(trigger, 0, success=False)
            raise RuntimeError(f"Failed to spawn node from template {template.id}: {e}")

    def clone_node(
        self,
        node: BaseNode,
        modifications: Optional[Dict[str, Any]] = None,
        trigger: SpawnTrigger = SpawnTrigger.MANUAL,
    ) -> BaseNode:
        """Clone an existing node with optional modifications.

        Args:
            node: Node to clone.
            modifications: Configuration modifications to apply.
            trigger: What triggered this clone.

        Returns:
            Cloned node instance.

        Raises:
            RuntimeError: If cloning is not allowed.
            ValueError: If node is invalid.
        """
        if not self.can_spawn():
            raise RuntimeError(
                f"Cannot clone: max_spawns={self.config.max_spawns}, "
                f"active={len(self._active_spawns)}"
            )

        # Determine spawn depth
        spawn_depth = 0
        if node.id in self._active_spawns:
            spawn_depth = self._active_spawns[node.id].spawn_depth + 1

        # Create definition from existing node
        clone_id = f"{node.id}_clone_{uuid4().hex[:8]}"
        definition = self._factory.clone_node_definition(
            node=node,
            new_id=clone_id,
            modifications=modifications,
        )

        # Spawn from the definition
        return self.spawn_node(
            template=definition,
            specialization=modifications,
            trigger=trigger,
            spawn_depth=spawn_depth,
        )

    def despawn_node(self, node_id: str) -> bool:
        """Remove a spawned node from active tracking.

        Args:
            node_id: ID of node to despawn.

        Returns:
            True if node was despawned, False if not found.
        """
        if node_id in self._active_spawns:
            del self._active_spawns[node_id]
            self._metrics.active_spawns = len(self._active_spawns)
            return True
        return False

    def get_spawn_history(
        self,
        node_id: Optional[str] = None,
        trigger: Optional[SpawnTrigger] = None,
        limit: Optional[int] = None,
    ) -> List[SpawnRecord]:
        """Get spawn history with optional filtering.

        Args:
            node_id: Filter by spawned node ID.
            trigger: Filter by spawn trigger.
            limit: Maximum number of records to return.

        Returns:
            List of spawn records, most recent first.
        """
        records = self._spawn_history

        # Apply filters
        if node_id:
            records = [r for r in records if r.spawned_node_id == node_id]
        if trigger:
            records = [r for r in records if r.trigger == trigger]

        # Sort by timestamp descending (most recent first)
        records = sorted(records, key=lambda r: r.timestamp, reverse=True)

        # Apply limit
        if limit and limit > 0:
            records = records[:limit]

        return records

    def get_spawn_record(self, node_id: str) -> Optional[SpawnRecord]:
        """Get spawn record for a specific node.

        Args:
            node_id: ID of spawned node.

        Returns:
            Spawn record if found, None otherwise.
        """
        return self._active_spawns.get(node_id)

    def get_children(self, template_id: str) -> List[SpawnRecord]:
        """Get all spawns created from a template.

        Args:
            template_id: ID of template node.

        Returns:
            List of spawn records for nodes spawned from this template.
        """
        return [
            record
            for record in self._spawn_history
            if record.template_node_id == template_id
        ]

    def get_metrics(self) -> SpawnMetrics:
        """Get current spawn metrics.

        Returns:
            Current metrics.
        """
        self._metrics.active_spawns = len(self._active_spawns)
        return self._metrics.model_copy(deep=True)

    def check_performance_trigger(
        self,
        node_id: str,
        current_performance: float,
    ) -> bool:
        """Check if performance threshold triggers spawning.

        Args:
            node_id: Node to check.
            current_performance: Current performance score (0.0 to 1.0).

        Returns:
            True if spawn should be triggered.
        """
        if not self.config.enable_auto_spawn:
            return False

        if current_performance < self.config.performance_threshold:
            return self.can_spawn()

        return False

    def check_workload_trigger(
        self,
        node_id: str,
        current_workload: int,
    ) -> bool:
        """Check if workload spike triggers spawning.

        Args:
            node_id: Node to check.
            current_workload: Current workload (requests/tasks).

        Returns:
            True if spawn should be triggered.
        """
        if not self.config.enable_auto_spawn:
            return False

        if current_workload > self.config.workload_threshold:
            return self.can_spawn()

        return False

    def reset(self) -> None:
        """Reset spawner state (for testing/debugging)."""
        self._spawn_history.clear()
        self._active_spawns.clear()
        self._last_spawn_time = None
        self._metrics = SpawnMetrics()

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed spawner statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "config": self.config.model_dump(),
            "active_spawns": len(self._active_spawns),
            "total_spawns": len(self._spawn_history),
            "can_spawn": self.can_spawn(),
            "last_spawn_time": self._last_spawn_time.isoformat()
            if self._last_spawn_time
            else None,
            "metrics": self._metrics.model_dump(),
            "spawn_history_size": len(self._spawn_history),
        }

    def _update_metrics(
        self,
        trigger: SpawnTrigger,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Update internal metrics.

        Args:
            trigger: Spawn trigger type.
            latency_ms: Spawn operation latency.
            success: Whether spawn succeeded.
        """
        if success:
            self._metrics.total_spawns += 1
            self._metrics.active_spawns = len(self._active_spawns)

            # Update average latency
            n = self._metrics.total_spawns
            old_avg = self._metrics.avg_spawn_latency_ms
            self._metrics.avg_spawn_latency_ms = (
                old_avg * (n - 1) + latency_ms
            ) / n

            # Update trigger counts
            trigger_key = trigger.value
            self._metrics.spawns_by_trigger[trigger_key] = (
                self._metrics.spawns_by_trigger.get(trigger_key, 0) + 1
            )
        else:
            self._metrics.failed_spawns += 1
