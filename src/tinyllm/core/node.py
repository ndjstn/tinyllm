"""Base node class and types for TinyLLM.

This module defines the abstract base class for all nodes and related
models for node execution and statistics.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message
from tinyllm.logging import get_logger

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext

logger = get_logger(__name__, component="node")


class NodeStats(BaseModel):
    """Runtime statistics for a node."""

    model_config = {"extra": "forbid"}

    total_executions: int = Field(default=0, ge=0)
    successful_executions: int = Field(default=0, ge=0)
    failed_executions: int = Field(default=0, ge=0)
    total_latency_ms: int = Field(default=0, ge=0)
    last_execution: Optional[datetime] = None
    expansion_count: int = Field(default=0, ge=0)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 to 1.0)."""
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.total_executions == 0:
            return 0.0
        return self.total_latency_ms / self.total_executions


class NodeConfig(BaseModel):
    """Base configuration for all nodes."""

    model_config = {"extra": "allow"}

    timeout_ms: int = Field(default=30000, ge=100, le=120000)
    retry_count: int = Field(default=0, ge=0, le=3)
    retry_delay_ms: int = Field(default=1000, ge=0, le=10000)


class NodeResult(BaseModel):
    """Result of node execution."""

    model_config = {"extra": "forbid"}

    success: bool = Field(description="Whether execution succeeded")
    output_messages: List[Message] = Field(
        default_factory=list, description="Output messages from this node"
    )
    next_nodes: List[str] = Field(
        default_factory=list, description="Node IDs to execute next"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    latency_ms: int = Field(default=0, ge=0, description="Execution time in ms")

    @classmethod
    def success_result(
        cls,
        output_messages: List[Message],
        next_nodes: List[str],
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NodeResult":
        """Create a successful result."""
        return cls(
            success=True,
            output_messages=output_messages,
            next_nodes=next_nodes,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NodeResult":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )


class BaseNode(ABC):
    """Abstract base class for all nodes.

    Nodes are the fundamental units of computation in TinyLLM graphs.
    Each node receives messages, processes them, and produces output.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize node from definition.

        Args:
            definition: Node definition from graph config.
        """
        self.id = definition.id
        self.type = definition.type
        self.name = definition.name or definition.id
        self.description = definition.description
        self._config = NodeConfig(**definition.config)
        self._stats = NodeStats()
        self._raw_config = definition.config

    @property
    def config(self) -> NodeConfig:
        """Get node configuration."""
        return self._config

    @property
    def stats(self) -> NodeStats:
        """Get node statistics."""
        return self._stats

    @abstractmethod
    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute the node's logic.

        Args:
            message: Input message to process.
            context: Execution context with graph state, memory, etc.

        Returns:
            NodeResult containing output message(s) and execution metadata.
        """
        pass

    def update_stats(self, success: bool, latency_ms: int) -> None:
        """Update node statistics after execution.

        Args:
            success: Whether the execution succeeded.
            latency_ms: Execution time in milliseconds.
        """
        self._stats.total_executions += 1
        if success:
            self._stats.successful_executions += 1
        else:
            self._stats.failed_executions += 1
        self._stats.total_latency_ms += latency_ms
        self._stats.last_execution = datetime.utcnow()

        # Log stats periodically (every 10 executions)
        if self._stats.total_executions % 10 == 0:
            logger.debug(
                "node_stats_update",
                node_id=self.id,
                node_type=self.type.value,
                total_executions=self._stats.total_executions,
                success_rate=self._stats.success_rate,
                avg_latency_ms=self._stats.avg_latency_ms,
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} type={self.type}>"


# Type alias for node type enumeration (re-exported for convenience)
__all__ = [
    "NodeType",
    "NodeStats",
    "NodeConfig",
    "NodeResult",
    "BaseNode",
]
