"""Memory defragmentation for execution contexts.

Provides defragmentation capabilities to compact memory and reduce fragmentation
in long-running graph executions by reorganizing messages and context data.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="defrag")


class DefragStats(BaseModel):
    """Statistics about defragmentation operation."""

    model_config = {"extra": "forbid"}

    messages_before: int = Field(ge=0, description="Message count before defrag")
    messages_after: int = Field(ge=0, description="Message count after defrag")
    messages_removed: int = Field(ge=0, description="Messages removed")

    size_before_bytes: int = Field(ge=0, description="Total size before defrag")
    size_after_bytes: int = Field(ge=0, description="Total size after defrag")
    bytes_freed: int = Field(ge=0, description="Bytes freed by defrag")

    variables_before: int = Field(ge=0, description="Variables before defrag")
    variables_after: int = Field(ge=0, description="Variables after defrag")
    variables_removed: int = Field(ge=0, description="Variables removed")

    fragmentation_before: float = Field(
        ge=0.0, le=1.0, description="Fragmentation ratio before"
    )
    fragmentation_after: float = Field(
        ge=0.0, le=1.0, description="Fragmentation ratio after"
    )

    duration_ms: int = Field(ge=0, description="Defrag duration in milliseconds")


class DefragStrategy(BaseModel):
    """Configuration for defragmentation strategy."""

    model_config = {"extra": "forbid"}

    # Message pruning
    remove_old_messages: bool = Field(
        default=True, description="Remove old messages beyond window"
    )
    message_window_size: int = Field(
        default=100, ge=1, description="Number of recent messages to keep"
    )

    # Variable cleanup
    remove_unused_variables: bool = Field(
        default=True, description="Remove variables not accessed recently"
    )
    variable_access_threshold: int = Field(
        default=10, ge=0, description="Steps since last access before removal"
    )

    # Message deduplication
    deduplicate_messages: bool = Field(
        default=False, description="Remove duplicate messages"
    )

    # Compact metadata
    compact_metadata: bool = Field(
        default=True, description="Remove unnecessary metadata fields"
    )


class MemoryDefragmenter:
    """Handles memory defragmentation for execution contexts.

    Reduces memory fragmentation by:
    - Removing old messages beyond a sliding window
    - Cleaning up unused variables
    - Deduplicating messages
    - Compacting metadata
    """

    def __init__(self, strategy: Optional[DefragStrategy] = None):
        """Initialize defragmenter.

        Args:
            strategy: Defragmentation strategy to use.
        """
        self.strategy = strategy or DefragStrategy()
        self._variable_access_counts: Dict[str, int] = {}
        self._steps_since_access: Dict[str, int] = {}

        logger.info(
            "defragmenter_created",
            message_window=self.strategy.message_window_size,
            remove_old=self.strategy.remove_old_messages,
            deduplicate=self.strategy.deduplicate_messages,
        )

    def defragment(self, context: ExecutionContext) -> DefragStats:
        """Defragment an execution context.

        Args:
            context: Context to defragment.

        Returns:
            DefragStats with operation results.
        """
        import time
        start = time.time()

        # Collect before stats
        messages_before = len(context.messages)
        variables_before = len(context.variables)
        size_before = self._calculate_size(context)
        frag_before = self._calculate_fragmentation(context)

        logger.info(
            "defragmentation_started",
            trace_id=context.trace_id,
            messages=messages_before,
            variables=variables_before,
            size_bytes=size_before,
        )

        # Perform defragmentation steps
        messages_removed = 0
        variables_removed = 0

        if self.strategy.remove_old_messages:
            messages_removed += self._prune_old_messages(context)

        if self.strategy.deduplicate_messages:
            messages_removed += self._deduplicate_messages(context)

        if self.strategy.remove_unused_variables:
            variables_removed += self._remove_unused_variables(context)

        if self.strategy.compact_metadata:
            self._compact_metadata(context)

        # Collect after stats
        messages_after = len(context.messages)
        variables_after = len(context.variables)
        size_after = self._calculate_size(context)
        frag_after = self._calculate_fragmentation(context)

        duration_ms = int((time.time() - start) * 1000)

        stats = DefragStats(
            messages_before=messages_before,
            messages_after=messages_after,
            messages_removed=messages_removed,
            size_before_bytes=size_before,
            size_after_bytes=size_after,
            bytes_freed=size_before - size_after,
            variables_before=variables_before,
            variables_after=variables_after,
            variables_removed=variables_removed,
            fragmentation_before=frag_before,
            fragmentation_after=frag_after,
            duration_ms=duration_ms,
        )

        logger.info(
            "defragmentation_completed",
            trace_id=context.trace_id,
            messages_removed=messages_removed,
            variables_removed=variables_removed,
            bytes_freed=size_before - size_after,
            duration_ms=duration_ms,
        )

        return stats

    def _prune_old_messages(self, context: ExecutionContext) -> int:
        """Remove old messages beyond window.

        Args:
            context: Context to prune.

        Returns:
            Number of messages removed.
        """
        if len(context.messages) <= self.strategy.message_window_size:
            return 0

        messages_to_keep = context.messages[-self.strategy.message_window_size:]
        removed_count = len(context.messages) - len(messages_to_keep)

        context.messages = messages_to_keep

        logger.debug(
            "old_messages_pruned",
            trace_id=context.trace_id,
            removed=removed_count,
            remaining=len(messages_to_keep),
        )

        return removed_count

    def _deduplicate_messages(self, context: ExecutionContext) -> int:
        """Remove duplicate messages.

        Args:
            context: Context to deduplicate.

        Returns:
            Number of duplicates removed.
        """
        seen: Dict[str, Message] = {}
        unique_messages: List[Message] = []

        for msg in context.messages:
            # Create hash key from content
            key = self._message_hash(msg)

            if key not in seen:
                seen[key] = msg
                unique_messages.append(msg)

        removed_count = len(context.messages) - len(unique_messages)
        if removed_count > 0:
            context.messages = unique_messages

            logger.debug(
                "messages_deduplicated",
                trace_id=context.trace_id,
                removed=removed_count,
                unique=len(unique_messages),
            )

        return removed_count

    def _remove_unused_variables(self, context: ExecutionContext) -> int:
        """Remove variables not accessed recently.

        Args:
            context: Context to clean.

        Returns:
            Number of variables removed.
        """
        # Increment access counters
        for key in self._steps_since_access:
            self._steps_since_access[key] += 1

        # Find variables to remove
        to_remove = []
        for key in list(context.variables.keys()):
            steps = self._steps_since_access.get(key, 0)
            if steps >= self.strategy.variable_access_threshold:
                to_remove.append(key)

        # Remove them
        for key in to_remove:
            del context.variables[key]
            if key in self._steps_since_access:
                del self._steps_since_access[key]

        if to_remove:
            logger.debug(
                "unused_variables_removed",
                trace_id=context.trace_id,
                removed=len(to_remove),
                removed_keys=to_remove[:10],  # Log first 10
            )

        return len(to_remove)

    def _compact_metadata(self, context: ExecutionContext) -> None:
        """Compact metadata in messages.

        Args:
            context: Context to compact.
        """
        # Remove empty metadata fields
        for msg in context.messages:
            # Remove None values from metadata
            if hasattr(msg.metadata, "model_dump"):
                metadata_dict = msg.metadata.model_dump(exclude_none=True)
                # Could rebuild metadata, but for now just note it
                pass

        logger.debug(
            "metadata_compacted",
            trace_id=context.trace_id,
            message_count=len(context.messages),
        )

    def _message_hash(self, msg: Message) -> str:
        """Create hash for message deduplication.

        Args:
            msg: Message to hash.

        Returns:
            Hash string.
        """
        # Simple hash based on content and source
        content = msg.payload.content or ""
        return f"{msg.source_node}:{content[:100]}"

    def _calculate_size(self, context: ExecutionContext) -> int:
        """Calculate total context size.

        Args:
            context: Context to measure.

        Returns:
            Size in bytes.
        """
        total = 0

        # Messages
        for msg in context.messages:
            try:
                total += sys.getsizeof(msg.model_dump_json())
            except Exception:
                total += 1024  # Estimate

        # Variables
        try:
            total += sys.getsizeof(str(context.variables))
        except Exception:
            total += len(context.variables) * 100

        return total

    def _calculate_fragmentation(self, context: ExecutionContext) -> float:
        """Calculate memory fragmentation ratio.

        Fragmentation is estimated based on the ratio of "wasted" space
        (gaps between objects, overhead, etc.) to total allocated space.

        Args:
            context: Context to analyze.

        Returns:
            Fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented).
        """
        if not context.messages:
            return 0.0

        # Estimate based on message size variance
        sizes = []
        for msg in context.messages:
            try:
                sizes.append(sys.getsizeof(msg.model_dump_json()))
            except Exception:
                sizes.append(1024)

        if not sizes:
            return 0.0

        # High variance in sizes suggests fragmentation
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_dev = variance ** 0.5

        # Normalize to 0-1 range (cap at 1.0)
        fragmentation = min(1.0, std_dev / (avg_size + 1))

        return fragmentation

    def track_variable_access(self, key: str) -> None:
        """Track that a variable was accessed.

        Args:
            key: Variable key that was accessed.
        """
        self._variable_access_counts[key] = \
            self._variable_access_counts.get(key, 0) + 1
        self._steps_since_access[key] = 0

    def reset_tracking(self) -> None:
        """Reset access tracking."""
        self._variable_access_counts.clear()
        self._steps_since_access.clear()


class AutoDefragmenter:
    """Automatically defragments contexts based on triggers.

    Monitors context health and automatically defragments when
    certain thresholds are exceeded.
    """

    def __init__(
        self,
        strategy: Optional[DefragStrategy] = None,
        auto_defrag_threshold: float = 0.7,
        check_interval_steps: int = 100,
    ):
        """Initialize auto-defragmenter.

        Args:
            strategy: Defragmentation strategy.
            auto_defrag_threshold: Fragmentation threshold to trigger defrag.
            check_interval_steps: Steps between fragmentation checks.
        """
        self.defragmenter = MemoryDefragmenter(strategy)
        self.auto_defrag_threshold = auto_defrag_threshold
        self.check_interval_steps = check_interval_steps
        self._steps_since_check = 0

        logger.info(
            "auto_defragmenter_created",
            threshold=auto_defrag_threshold,
            check_interval=check_interval_steps,
        )

    def maybe_defragment(self, context: ExecutionContext) -> Optional[DefragStats]:
        """Defragment context if needed.

        Args:
            context: Context to check.

        Returns:
            DefragStats if defragmentation occurred, None otherwise.
        """
        self._steps_since_check += 1

        if self._steps_since_check < self.check_interval_steps:
            return None

        self._steps_since_check = 0

        # Check fragmentation
        frag = self.defragmenter._calculate_fragmentation(context)

        if frag >= self.auto_defrag_threshold:
            logger.info(
                "auto_defrag_triggered",
                trace_id=context.trace_id,
                fragmentation=frag,
                threshold=self.auto_defrag_threshold,
            )
            return self.defragmenter.defragment(context)

        return None
