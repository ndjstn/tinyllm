"""Checkpoint manager for long-running graph executions.

This module provides checkpointing functionality for resumable execution
of graphs, with support for configurable checkpoint intervals and state
persistence.
"""

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message
from tinyllm.logging import get_logger
from tinyllm.persistence.interface import CheckpointRecord, CheckpointStorage

logger = get_logger(__name__, component="checkpoint")


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint manager."""

    model_config = {"extra": "forbid"}

    checkpoint_interval_ms: int = Field(
        default=5000,
        ge=0,
        description="Checkpoint interval in milliseconds (0 = disabled)",
    )
    checkpoint_after_each_node: bool = Field(
        default=False,
        description="Whether to checkpoint after each node execution",
    )
    max_checkpoints_per_trace: int = Field(
        default=100,
        ge=1,
        description="Maximum checkpoints to keep per trace",
    )


class CheckpointManager:
    """Manages checkpointing for graph executions.

    The CheckpointManager handles saving and restoring execution state
    at configurable intervals or after each node execution.
    """

    def __init__(
        self,
        storage: CheckpointStorage,
        config: Optional[CheckpointConfig] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            storage: Checkpoint storage backend.
            config: Checkpoint configuration.
        """
        self.storage = storage
        self.config = config or CheckpointConfig()
        self._last_checkpoint_time: float = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the checkpoint storage."""
        if not self._initialized:
            await self.storage.initialize()
            self._initialized = True
            self._last_checkpoint_time = time.perf_counter()
            logger.info("checkpoint_manager_initialized")

    async def close(self) -> None:
        """Close the checkpoint storage."""
        if self._initialized:
            await self.storage.close()
            self._initialized = False
            logger.info("checkpoint_manager_closed")

    def should_checkpoint(self, force: bool = False) -> bool:
        """Check if a checkpoint should be created.

        Args:
            force: Force checkpoint regardless of interval.

        Returns:
            True if checkpoint should be created.
        """
        if force or self.config.checkpoint_after_each_node:
            return True

        if self.config.checkpoint_interval_ms == 0:
            return False

        current_time = time.perf_counter()
        elapsed_ms = (current_time - self._last_checkpoint_time) * 1000

        return elapsed_ms >= self.config.checkpoint_interval_ms

    async def save_checkpoint(
        self,
        context: ExecutionContext,
        current_node_id: str,
        force: bool = False,
    ) -> Optional[CheckpointRecord]:
        """Save a checkpoint of the current execution state.

        Args:
            context: Execution context to checkpoint.
            current_node_id: Current node being executed.
            force: Force checkpoint regardless of interval.

        Returns:
            The created checkpoint record, or None if checkpoint was skipped.
        """
        if not self.should_checkpoint(force):
            return None

        logger.debug(
            "saving_checkpoint",
            trace_id=context.trace_id,
            graph_id=context.graph_id,
            step=context.step_count,
            node_id=current_node_id,
        )

        # Serialize context state
        state = {
            "current_node": context.current_node,
            "visited_nodes": context.visited_nodes,
            "variables": context.variables,
            "step_count": context.step_count,
            "total_tokens_in": context.total_tokens_in,
            "total_tokens_out": context.total_tokens_out,
            "start_time": context.start_time.isoformat(),
        }

        # Serialize messages
        messages = [self._serialize_message(msg) for msg in context.messages]

        # Save checkpoint
        record = await self.storage.save_checkpoint(
            graph_id=context.graph_id,
            trace_id=context.trace_id,
            step=context.step_count,
            state=state,
            messages=messages,
            node_id=current_node_id,
        )

        self._last_checkpoint_time = time.perf_counter()

        logger.info(
            "checkpoint_saved",
            trace_id=context.trace_id,
            checkpoint_id=record.id,
            step=context.step_count,
            node_id=current_node_id,
            message_count=len(messages),
        )

        # Clean up old checkpoints if needed
        await self._prune_old_checkpoints(context.graph_id, context.trace_id)

        return record

    async def load_checkpoint(
        self,
        graph_id: str,
        trace_id: str,
    ) -> Optional[CheckpointRecord]:
        """Load the latest checkpoint for a trace.

        Args:
            graph_id: Graph identifier.
            trace_id: Trace identifier.

        Returns:
            The latest checkpoint record, or None if not found.
        """
        logger.debug(
            "loading_checkpoint",
            graph_id=graph_id,
            trace_id=trace_id,
        )

        checkpoint = await self.storage.get_latest_checkpoint(graph_id, trace_id)

        if checkpoint:
            logger.info(
                "checkpoint_loaded",
                checkpoint_id=checkpoint.id,
                graph_id=graph_id,
                trace_id=trace_id,
                step=checkpoint.step,
                node_id=checkpoint.node_id,
            )
        else:
            logger.info(
                "no_checkpoint_found",
                graph_id=graph_id,
                trace_id=trace_id,
            )

        return checkpoint

    async def restore_context(
        self,
        checkpoint: CheckpointRecord,
        context: ExecutionContext,
    ) -> None:
        """Restore execution context from a checkpoint.

        Args:
            checkpoint: Checkpoint to restore from.
            context: Context to restore into.
        """
        from datetime import datetime

        logger.info(
            "restoring_checkpoint",
            checkpoint_id=checkpoint.id,
            trace_id=checkpoint.trace_id,
            step=checkpoint.step,
        )

        # Restore state
        state = checkpoint.state
        context.current_node = state.get("current_node")
        context.visited_nodes = state.get("visited_nodes", [])
        context.variables = state.get("variables", {})
        context.step_count = state.get("step_count", 0)
        context.total_tokens_in = state.get("total_tokens_in", 0)
        context.total_tokens_out = state.get("total_tokens_out", 0)

        if "start_time" in state:
            context.start_time = datetime.fromisoformat(state["start_time"])

        # Restore messages
        context.messages = [
            self._deserialize_message(msg) for msg in checkpoint.messages
        ]

        logger.info(
            "checkpoint_restored",
            checkpoint_id=checkpoint.id,
            step=context.step_count,
            message_count=len(context.messages),
            visited_nodes=len(context.visited_nodes),
        )

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        """Serialize a message for checkpoint storage.

        Args:
            message: Message to serialize.

        Returns:
            Serialized message dictionary.
        """
        return message.model_dump(mode="json")

    def _deserialize_message(self, data: Dict[str, Any]) -> Message:
        """Deserialize a message from checkpoint storage.

        Args:
            data: Serialized message data.

        Returns:
            Reconstructed message.
        """
        return Message(**data)

    async def _prune_old_checkpoints(self, graph_id: str, trace_id: str) -> None:
        """Remove old checkpoints beyond the configured limit.

        Args:
            graph_id: Graph identifier.
            trace_id: Trace identifier.
        """
        checkpoints = await self.storage.list_checkpoints(graph_id, trace_id)

        if len(checkpoints) > self.config.max_checkpoints_per_trace:
            # Sort by step descending (newest first)
            checkpoints.sort(key=lambda c: c.step, reverse=True)

            # Delete old checkpoints
            to_delete = checkpoints[self.config.max_checkpoints_per_trace :]
            for checkpoint in to_delete:
                await self.storage.delete(checkpoint.id)
                logger.debug(
                    "checkpoint_pruned",
                    checkpoint_id=checkpoint.id,
                    step=checkpoint.step,
                )

            logger.info(
                "checkpoints_pruned",
                graph_id=graph_id,
                trace_id=trace_id,
                count=len(to_delete),
            )

    async def list_checkpoints(
        self,
        graph_id: str,
        trace_id: Optional[str] = None,
    ) -> List[CheckpointRecord]:
        """List all checkpoints for a graph or trace.

        Args:
            graph_id: Graph identifier.
            trace_id: Optional trace identifier.

        Returns:
            List of checkpoint records.
        """
        return await self.storage.list_checkpoints(graph_id, trace_id)

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        """Get a specific checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier.

        Returns:
            Checkpoint record or None if not found.
        """
        return await self.storage.get(checkpoint_id)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier.

        Returns:
            True if deleted, False if not found.
        """
        deleted = await self.storage.delete(checkpoint_id)
        if deleted:
            logger.info("checkpoint_deleted", checkpoint_id=checkpoint_id)
        return deleted

    async def clear_checkpoints(
        self,
        graph_id: str,
        trace_id: Optional[str] = None,
    ) -> int:
        """Clear all checkpoints for a graph or trace.

        Args:
            graph_id: Graph identifier.
            trace_id: Optional trace identifier to filter by.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = await self.list_checkpoints(graph_id, trace_id)
        count = 0
        for checkpoint in checkpoints:
            if await self.delete_checkpoint(checkpoint.id):
                count += 1

        logger.info(
            "checkpoints_cleared",
            graph_id=graph_id,
            trace_id=trace_id,
            count=count,
        )
        return count
