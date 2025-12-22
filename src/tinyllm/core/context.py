"""Execution context for TinyLLM graph execution.

This module provides the ExecutionContext class that maintains state
during graph execution, including visited nodes, messages, and variables.
"""

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from tinyllm.config.loader import Config
from tinyllm.core.message import Message
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="context")


class BoundsExceededError(Exception):
    """Exception raised when execution context exceeds memory bounds.

    Attributes:
        limit_type: Type of limit exceeded (messages, message_size, total_size).
        current_value: Current value that exceeded the limit.
        limit_value: The limit that was exceeded.
        details: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        limit_type: str,
        current_value: int,
        limit_value: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.details = details or {}


class MemoryUsageStats(BaseModel):
    """Statistics about context memory usage.

    Provides detailed metrics about current memory consumption
    and utilization relative to configured limits.
    """

    model_config = {"extra": "forbid"}

    # Current usage
    message_count: int = Field(ge=0, description="Number of messages in context")
    total_size_bytes: int = Field(ge=0, description="Total memory size in bytes")
    largest_message_bytes: int = Field(
        ge=0, description="Size of largest message in bytes"
    )

    # Limits
    max_messages: int = Field(ge=1, description="Maximum allowed messages")
    max_message_size_bytes: int = Field(
        ge=1, description="Maximum allowed size per message"
    )
    max_total_size_bytes: int = Field(
        ge=1, description="Maximum allowed total size"
    )

    # Utilization percentages (0.0 to 1.0)
    message_count_utilization: float = Field(
        ge=0.0, le=1.0, description="Message count utilization ratio"
    )
    total_size_utilization: float = Field(
        ge=0.0, le=1.0, description="Total size utilization ratio"
    )

    # Flags
    near_message_limit: bool = Field(
        default=False, description="True if approaching message count limit (>80%)"
    )
    near_size_limit: bool = Field(
        default=False, description="True if approaching size limit (>80%)"
    )


class ExecutionContext(BaseModel):
    """Context maintained during graph execution.

    The ExecutionContext holds all state needed during the execution
    of a graph, including visited nodes, messages, and arbitrary variables.

    Memory bounds are enforced to prevent unbounded growth and OOM errors.
    Bounds are checked on message addition and can trigger pruning or errors.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Identifiers
    trace_id: str = Field(description="Unique trace identifier")
    graph_id: str = Field(description="Graph being executed")

    # Timing
    start_time: datetime = Field(
        default_factory=datetime.utcnow, description="Execution start time"
    )

    # State
    current_node: Optional[str] = Field(
        default=None, description="Currently executing node"
    )
    visited_nodes: List[str] = Field(
        default_factory=list, description="Nodes visited in order"
    )
    messages: List[Message] = Field(
        default_factory=list, description="All messages in this execution"
    )

    # Configuration
    config: Config = Field(description="System configuration")

    # Memory/Variables
    variables: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary execution variables"
    )

    # Counters
    step_count: int = Field(default=0, ge=0, description="Number of steps executed")
    total_tokens_in: int = Field(default=0, ge=0, description="Total input tokens")
    total_tokens_out: int = Field(default=0, ge=0, description="Total output tokens")

    # Memory Bounds Configuration
    max_messages: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of messages allowed in context"
    )
    max_message_size_bytes: int = Field(
        default=1_048_576,  # 1 MB
        ge=1024,
        description="Maximum size per message in bytes"
    )
    max_total_size_bytes: int = Field(
        default=104_857_600,  # 100 MB
        ge=1024,
        description="Maximum total context size in bytes"
    )

    # Memory tracking (private attributes)
    _warned_message_limit: bool = PrivateAttr(default=False)
    _warned_size_limit: bool = PrivateAttr(default=False)
    _total_size_bytes: int = PrivateAttr(default=0)
    _message_size_cache: Dict[str, int] = PrivateAttr(default_factory=dict)

    def _get_message_size(self, message: Message, use_cache: bool = True) -> int:
        """Calculate approximate size of a message in bytes.

        Uses caching to avoid recalculating sizes for the same messages.

        Args:
            message: Message to measure.
            use_cache: Whether to use cached size if available (default: True).

        Returns:
            Approximate size in bytes.
        """
        # Check cache first
        if use_cache and message.message_id in self._message_size_cache:
            return self._message_size_cache[message.message_id]

        # Calculate size
        try:
            # Get base size
            size = sys.getsizeof(message.model_dump_json())
        except Exception as e:
            logger.warning(
                "failed_to_calculate_message_size",
                trace_id=self.trace_id,
                message_id=message.message_id,
                error=str(e),
            )
            # Return a conservative estimate
            size = 1024

        # Cache the size
        self._message_size_cache[message.message_id] = size
        return size

    def _calculate_total_size(self) -> int:
        """Get total size of all messages in bytes.

        Returns cached running total for O(1) performance.
        Only recalculates if cache is stale.

        Returns:
            Total size in bytes.
        """
        return self._total_size_bytes

    def _prune_messages(self) -> int:
        """Prune old messages to make room for new ones.

        Automatically called when approaching capacity limits (80%).
        Removes oldest 20% of messages.

        Returns:
            Number of messages pruned.
        """
        # Calculate how many to remove (20% of current messages)
        remove_count = max(1, len(self.messages) // 5)

        if remove_count == 0:
            return 0

        # Store messages to remove for size calculation
        removed_messages = self.messages[:remove_count]
        removed_size = sum(
            self._message_size_cache.get(msg.message_id, self._get_message_size(msg))
            for msg in removed_messages
        )

        # Remove from cache
        for msg in removed_messages:
            self._message_size_cache.pop(msg.message_id, None)

        # Keep only the most recent messages
        self.messages = self.messages[remove_count:]

        # Update running total
        self._total_size_bytes -= removed_size

        logger.info(
            "messages_auto_pruned",
            trace_id=self.trace_id,
            removed_count=remove_count,
            remaining_count=len(self.messages),
            freed_bytes=removed_size,
            new_total_bytes=self._total_size_bytes,
        )

        # Reset warning flags since we've pruned
        self._warned_message_limit = False
        self._warned_size_limit = False

        return remove_count

    def add_message(self, message: Message) -> None:
        """Add a message to the execution with bounds checking.

        Uses O(1) incremental size tracking for performance.
        Automatically prunes old messages when approaching limits (80%).

        Args:
            message: Message to add.

        Raises:
            BoundsExceededError: If message exceeds size limits.
        """
        # Check individual message size
        message_size = self._get_message_size(message)
        if message_size > self.max_message_size_bytes:
            logger.error(
                "message_size_exceeded",
                trace_id=self.trace_id,
                message_id=message.message_id,
                size_bytes=message_size,
                limit_bytes=self.max_message_size_bytes,
            )
            raise BoundsExceededError(
                f"Message size {message_size} bytes exceeds limit of {self.max_message_size_bytes} bytes",
                limit_type="message_size",
                current_value=message_size,
                limit_value=self.max_message_size_bytes,
                details={
                    "message_id": message.message_id,
                    "source_node": message.source_node,
                },
            )

        # Proactive check BEFORE adding - prune if needed
        projected_total = self._total_size_bytes + message_size
        projected_count = len(self.messages) + 1

        # Auto-prune at 80% capacity
        if (projected_total > self.max_total_size_bytes * 0.8 or
            projected_count > self.max_messages * 0.8):
            logger.info(
                "auto_pruning_triggered",
                trace_id=self.trace_id,
                projected_total_bytes=projected_total,
                projected_count=projected_count,
                utilization_bytes=projected_total / self.max_total_size_bytes,
                utilization_count=projected_count / self.max_messages,
            )
            self._prune_messages()

        # Final bounds check after potential pruning
        if self._total_size_bytes + message_size > self.max_total_size_bytes:
            logger.error(
                "total_size_exceeded",
                trace_id=self.trace_id,
                current_size=self._total_size_bytes,
                message_size=message_size,
                projected_total=self._total_size_bytes + message_size,
                limit_bytes=self.max_total_size_bytes,
            )
            raise BoundsExceededError(
                f"Total context size would exceed limit ({self._total_size_bytes + message_size} > {self.max_total_size_bytes} bytes)",
                limit_type="total_size",
                current_value=self._total_size_bytes + message_size,
                limit_value=self.max_total_size_bytes,
                details={
                    "message_count": len(self.messages),
                    "trace_id": self.trace_id,
                },
            )

        if len(self.messages) >= self.max_messages:
            logger.error(
                "message_count_exceeded",
                trace_id=self.trace_id,
                message_count=len(self.messages),
                limit=self.max_messages,
            )
            raise BoundsExceededError(
                f"Message count {len(self.messages)} exceeds limit of {self.max_messages}",
                limit_type="message_count",
                current_value=len(self.messages),
                limit_value=self.max_messages,
                details={"trace_id": self.trace_id},
            )

        # Add message and update running total atomically
        self.messages.append(message)
        self._total_size_bytes += message_size

        # Log memory metrics (using cached total)
        message_utilization = len(self.messages) / self.max_messages
        size_utilization = self._total_size_bytes / self.max_total_size_bytes

        logger.debug(
            "message_added",
            trace_id=self.trace_id,
            message_id=message.message_id,
            message_count=len(self.messages),
            message_size_bytes=message_size,
            total_size_bytes=self._total_size_bytes,
            message_utilization_pct=int(message_utilization * 100),
            size_utilization_pct=int(size_utilization * 100),
        )

    def get_latest_message(self) -> Optional[Message]:
        """Get the most recent message.

        Returns:
            The last message or None if no messages.
        """
        return self.messages[-1] if self.messages else None

    def get_messages_from_node(self, node_id: str) -> List[Message]:
        """Get all messages from a specific node.

        Args:
            node_id: Node ID to filter by.

        Returns:
            List of messages from that node.
        """
        return [m for m in self.messages if m.source_node == node_id]

    def set_variable(self, key: str, value: Any) -> None:
        """Set an execution variable.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get an execution variable.

        Args:
            key: Variable name.
            default: Default value if not found.

        Returns:
            Variable value or default.
        """
        return self.variables.get(key, default)

    def has_variable(self, key: str) -> bool:
        """Check if a variable exists.

        Args:
            key: Variable name.

        Returns:
            True if variable exists.
        """
        return key in self.variables

    def visit_node(self, node_id: str) -> None:
        """Record visiting a node.

        Args:
            node_id: Node being visited.
        """
        self.current_node = node_id
        self.visited_nodes.append(node_id)
        self.step_count += 1

    def has_visited(self, node_id: str) -> bool:
        """Check if a node has been visited.

        Args:
            node_id: Node to check.

        Returns:
            True if node was visited.
        """
        return node_id in self.visited_nodes

    def visit_count(self, node_id: str) -> int:
        """Count how many times a node was visited.

        Args:
            node_id: Node to count.

        Returns:
            Number of visits.
        """
        return self.visited_nodes.count(node_id)

    def add_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Add to token counters.

        Args:
            tokens_in: Input tokens to add.
            tokens_out: Output tokens to add.
        """
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

    def check_bounds(self) -> None:
        """Verify current state is within configured memory bounds.

        Checks that current message count and total size are within limits.
        This is useful for periodic validation during long-running executions.

        Raises:
            BoundsExceededError: If any limit is exceeded.
        """
        message_count = len(self.messages)
        total_size = self._calculate_total_size()

        # Check message count
        if message_count > self.max_messages:
            logger.error(
                "bounds_check_failed_message_count",
                trace_id=self.trace_id,
                message_count=message_count,
                limit=self.max_messages,
            )
            raise BoundsExceededError(
                f"Message count {message_count} exceeds limit of {self.max_messages}",
                limit_type="message_count",
                current_value=message_count,
                limit_value=self.max_messages,
                details={"trace_id": self.trace_id},
            )

        # Check total size
        if total_size > self.max_total_size_bytes:
            logger.error(
                "bounds_check_failed_total_size",
                trace_id=self.trace_id,
                total_size_bytes=total_size,
                limit_bytes=self.max_total_size_bytes,
            )
            raise BoundsExceededError(
                f"Total context size {total_size} bytes exceeds limit of {self.max_total_size_bytes} bytes",
                limit_type="total_size",
                current_value=total_size,
                limit_value=self.max_total_size_bytes,
                details={
                    "message_count": message_count,
                    "trace_id": self.trace_id,
                },
            )

        # Check individual message sizes
        for msg in self.messages:
            msg_size = self._get_message_size(msg)
            if msg_size > self.max_message_size_bytes:
                logger.error(
                    "bounds_check_failed_message_size",
                    trace_id=self.trace_id,
                    message_id=msg.message_id,
                    size_bytes=msg_size,
                    limit_bytes=self.max_message_size_bytes,
                )
                raise BoundsExceededError(
                    f"Message {msg.message_id} size {msg_size} bytes exceeds limit of {self.max_message_size_bytes} bytes",
                    limit_type="message_size",
                    current_value=msg_size,
                    limit_value=self.max_message_size_bytes,
                    details={
                        "message_id": msg.message_id,
                        "source_node": msg.source_node,
                    },
                )

        logger.debug(
            "bounds_check_passed",
            trace_id=self.trace_id,
            message_count=message_count,
            total_size_bytes=total_size,
        )

    def get_memory_usage(self) -> MemoryUsageStats:
        """Get current memory usage statistics.

        Returns detailed metrics about context memory consumption
        and utilization relative to configured limits.

        Returns:
            MemoryUsageStats with current usage and utilization metrics.
        """
        message_count = len(self.messages)
        total_size = self._calculate_total_size()

        # Find largest message
        largest_size = 0
        if self.messages:
            largest_size = max(
                self._get_message_size(msg) for msg in self.messages
            )

        # Calculate utilization ratios
        message_utilization = message_count / self.max_messages
        size_utilization = total_size / self.max_total_size_bytes

        # Determine if near limits (80% threshold)
        near_message_limit = message_utilization >= 0.8
        near_size_limit = size_utilization >= 0.8

        stats = MemoryUsageStats(
            message_count=message_count,
            total_size_bytes=total_size,
            largest_message_bytes=largest_size,
            max_messages=self.max_messages,
            max_message_size_bytes=self.max_message_size_bytes,
            max_total_size_bytes=self.max_total_size_bytes,
            message_count_utilization=message_utilization,
            total_size_utilization=size_utilization,
            near_message_limit=near_message_limit,
            near_size_limit=near_size_limit,
        )

        logger.debug(
            "memory_usage_calculated",
            trace_id=self.trace_id,
            message_count=message_count,
            total_size_bytes=total_size,
            message_utilization_pct=int(message_utilization * 100),
            size_utilization_pct=int(size_utilization * 100),
            near_message_limit=near_message_limit,
            near_size_limit=near_size_limit,
        )

        return stats

    def prune_old_messages(self, keep_count: Optional[int] = None) -> int:
        """Remove oldest messages when at or approaching limit.

        Prunes messages from the beginning of the list (oldest first),
        keeping the most recent messages. This is useful for managing
        memory in long-running executions.

        Maintains O(1) performance by updating running total incrementally.

        Args:
            keep_count: Number of messages to keep. If None, keeps
                       half of max_messages (default pruning strategy).

        Returns:
            Number of messages pruned.

        Raises:
            ValueError: If keep_count is invalid.
        """
        if keep_count is None:
            # Default: keep half of max messages
            keep_count = self.max_messages // 2

        if keep_count < 0:
            raise ValueError(f"keep_count must be non-negative, got {keep_count}")

        if keep_count >= len(self.messages):
            logger.debug(
                "prune_skipped_nothing_to_remove",
                trace_id=self.trace_id,
                message_count=len(self.messages),
                keep_count=keep_count,
            )
            return 0

        # Calculate how many to remove
        current_count = len(self.messages)
        remove_count = current_count - keep_count

        # Store messages to remove for logging and size calculation
        removed_messages = self.messages[:remove_count]
        removed_ids = [msg.message_id for msg in removed_messages]

        # Calculate size of removed messages
        removed_size = sum(
            self._message_size_cache.get(msg.message_id, self._get_message_size(msg))
            for msg in removed_messages
        )

        # Remove from cache
        for msg in removed_messages:
            self._message_size_cache.pop(msg.message_id, None)

        # Keep only the most recent messages
        self.messages = self.messages[-keep_count:]

        # Update running total
        self._total_size_bytes -= removed_size

        logger.info(
            "messages_pruned",
            trace_id=self.trace_id,
            removed_count=remove_count,
            remaining_count=len(self.messages),
            kept_count=keep_count,
            removed_message_ids=removed_ids[:10],  # Log first 10 IDs
            freed_bytes=removed_size,
        )

        # Reset warning flags since we've pruned
        self._warned_message_limit = False
        self._warned_size_limit = False

        # Log new memory stats after pruning
        logger.info(
            "memory_after_pruning",
            trace_id=self.trace_id,
            message_count=len(self.messages),
            total_size_bytes=self._total_size_bytes,
            size_utilization_pct=int((self._total_size_bytes / self.max_total_size_bytes) * 100),
        )

        return remove_count

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        delta = datetime.utcnow() - self.start_time
        return int(delta.total_seconds() * 1000)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_tokens_in + self.total_tokens_out
