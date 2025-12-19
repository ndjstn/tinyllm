"""Copy-on-write (CoW) contexts for efficient context cloning.

Provides copy-on-write semantics for execution contexts, allowing efficient
cloning by sharing data until modifications occur. This reduces memory usage
and improves performance for branching workflows.
"""

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from tinyllm.config.loader import Config
from tinyllm.core.context import ExecutionContext
from tinyllm.core.message import Message
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="cow")


class CoWStats(BaseModel):
    """Statistics about copy-on-write usage."""

    model_config = {"extra": "forbid"}

    context_id: str = Field(description="Context identifier")
    parent_id: Optional[str] = Field(default=None, description="Parent context ID")
    is_copy: bool = Field(default=False, description="Whether this is a CoW copy")
    shared_count: int = Field(ge=0, description="Number of shared objects")
    copied_count: int = Field(ge=0, description="Number of copied objects")
    memory_saved_bytes: int = Field(
        ge=0, description="Estimated memory saved by sharing"
    )
    copy_depth: int = Field(ge=0, description="Depth of copy chain")


class CoWContext:
    """Copy-on-write wrapper for ExecutionContext.

    Allows efficient context cloning by sharing data until modifications occur.
    When a modification is detected, the data is copied before being modified.

    Example:
        >>> original = ExecutionContext(...)
        >>> cow_ctx = CoWContext(original)
        >>> cloned = cow_ctx.clone()
        >>> # Messages are shared until modified
        >>> cloned.add_message(msg)  # Triggers copy-on-write
    """

    def __init__(
        self,
        context: ExecutionContext,
        parent: Optional["CoWContext"] = None,
        context_id: Optional[str] = None,
    ):
        """Initialize CoW context.

        Args:
            context: The ExecutionContext to wrap.
            parent: Parent CoW context (if this is a clone).
            context_id: Unique identifier for this context.
        """
        self._context = context
        self._parent = parent
        self._context_id = context_id or f"cow-{id(self)}"

        # Track what has been copied
        self._messages_copied = False
        self._variables_copied = False
        self._visited_nodes_copied = False

        # Track shared state
        self._shared_messages: Optional[List[Message]] = None
        self._shared_variables: Optional[Dict[str, Any]] = None
        self._shared_visited: Optional[List[str]] = None

        # If this is a clone, share the data from parent
        if parent is not None:
            self._shared_messages = parent._context.messages
            self._shared_variables = parent._context.variables
            self._shared_visited = parent._context.visited_nodes

        logger.debug(
            "cow_context_created",
            context_id=self._context_id,
            is_clone=parent is not None,
            trace_id=context.trace_id,
        )

    @property
    def context(self) -> ExecutionContext:
        """Get the underlying ExecutionContext."""
        return self._context

    @property
    def context_id(self) -> str:
        """Get CoW context ID."""
        return self._context_id

    def clone(self, context_id: Optional[str] = None) -> "CoWContext":
        """Create a copy-on-write clone of this context.

        The clone shares data with this context until modifications occur.

        Args:
            context_id: Optional ID for the cloned context.

        Returns:
            A new CoWContext that shares data with this one.
        """
        # Create a shallow copy of the ExecutionContext
        # Pydantic models support model_copy()
        new_context = self._context.model_copy(deep=False)

        # Create CoW wrapper
        cloned = CoWContext(
            context=new_context,
            parent=self,
            context_id=context_id,
        )

        logger.info(
            "cow_context_cloned",
            parent_id=self._context_id,
            clone_id=cloned._context_id,
            trace_id=self._context.trace_id,
        )

        return cloned

    def add_message(self, message: Message) -> None:
        """Add message with copy-on-write semantics.

        If messages are shared, copies them before adding.

        Args:
            message: Message to add.
        """
        self._ensure_messages_copied()
        self._context.add_message(message)

        logger.debug(
            "cow_message_added",
            context_id=self._context_id,
            message_id=message.message_id,
            messages_copied=self._messages_copied,
        )

    def set_variable(self, key: str, value: Any) -> None:
        """Set variable with copy-on-write semantics.

        If variables are shared, copies them before setting.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self._ensure_variables_copied()
        self._context.set_variable(key, value)

        logger.debug(
            "cow_variable_set",
            context_id=self._context_id,
            key=key,
            variables_copied=self._variables_copied,
        )

    def visit_node(self, node_id: str) -> None:
        """Record node visit with copy-on-write semantics.

        If visited nodes are shared, copies them before adding.

        Args:
            node_id: Node being visited.
        """
        self._ensure_visited_copied()
        self._context.visit_node(node_id)

        logger.debug(
            "cow_node_visited",
            context_id=self._context_id,
            node_id=node_id,
            visited_copied=self._visited_nodes_copied,
        )

    def _ensure_messages_copied(self) -> None:
        """Ensure messages are copied before modification."""
        if not self._messages_copied and self._shared_messages is not None:
            # Copy the shared messages
            self._context.messages = [
                msg.model_copy(deep=True) for msg in self._shared_messages
            ]
            self._messages_copied = True
            self._shared_messages = None

            logger.info(
                "cow_messages_copied",
                context_id=self._context_id,
                message_count=len(self._context.messages),
            )

    def _ensure_variables_copied(self) -> None:
        """Ensure variables are copied before modification."""
        if not self._variables_copied and self._shared_variables is not None:
            # Deep copy the shared variables
            self._context.variables = copy.deepcopy(self._shared_variables)
            self._variables_copied = True
            self._shared_variables = None

            logger.info(
                "cow_variables_copied",
                context_id=self._context_id,
                variable_count=len(self._context.variables),
            )

    def _ensure_visited_copied(self) -> None:
        """Ensure visited nodes list is copied before modification."""
        if not self._visited_nodes_copied and self._shared_visited is not None:
            # Copy the shared visited list
            self._context.visited_nodes = list(self._shared_visited)
            self._visited_nodes_copied = True
            self._shared_visited = None

            logger.info(
                "cow_visited_copied",
                context_id=self._context_id,
                visited_count=len(self._context.visited_nodes),
            )

    def get_stats(self) -> CoWStats:
        """Get copy-on-write statistics.

        Returns:
            CoWStats with usage metrics.
        """
        # Count shared objects
        shared_count = 0
        if self._shared_messages is not None:
            shared_count += len(self._shared_messages)
        if self._shared_variables is not None:
            shared_count += len(self._shared_variables)
        if self._shared_visited is not None:
            shared_count += len(self._shared_visited)

        # Count copied objects
        copied_count = 0
        if self._messages_copied:
            copied_count += len(self._context.messages)
        if self._variables_copied:
            copied_count += len(self._context.variables)
        if self._visited_nodes_copied:
            copied_count += len(self._context.visited_nodes)

        # Estimate memory saved (rough estimate)
        # Assume ~1KB per message, ~100 bytes per variable, ~50 bytes per node ID
        memory_saved = 0
        if self._shared_messages is not None:
            memory_saved += len(self._shared_messages) * 1024
        if self._shared_variables is not None:
            memory_saved += len(self._shared_variables) * 100
        if self._shared_visited is not None:
            memory_saved += len(self._shared_visited) * 50

        # Calculate copy depth
        depth = 0
        current = self._parent
        while current is not None:
            depth += 1
            current = current._parent

        return CoWStats(
            context_id=self._context_id,
            parent_id=self._parent._context_id if self._parent else None,
            is_copy=self._parent is not None,
            shared_count=shared_count,
            copied_count=copied_count,
            memory_saved_bytes=memory_saved,
            copy_depth=depth,
        )

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get variable without triggering copy.

        Args:
            key: Variable name.
            default: Default value if not found.

        Returns:
            Variable value or default.
        """
        return self._context.get_variable(key, default)

    def has_variable(self, key: str) -> bool:
        """Check if variable exists without triggering copy.

        Args:
            key: Variable name.

        Returns:
            True if variable exists.
        """
        return self._context.has_variable(key)

    def get_messages(self) -> List[Message]:
        """Get messages without triggering copy (read-only).

        Returns:
            List of messages.
        """
        if self._shared_messages is not None:
            return self._shared_messages
        return self._context.messages

    def __repr__(self) -> str:
        """String representation."""
        return f"CoWContext(id={self._context_id}, trace={self._context.trace_id})"


class CoWContextManager:
    """Manages multiple CoW contexts."""

    def __init__(self):
        """Initialize CoW context manager."""
        self._contexts: Dict[str, CoWContext] = {}
        logger.info("cow_manager_created")

    def create(self, context: ExecutionContext, context_id: Optional[str] = None) -> CoWContext:
        """Create a new CoW context.

        Args:
            context: ExecutionContext to wrap.
            context_id: Optional context ID.

        Returns:
            The created CoWContext.
        """
        cow_ctx = CoWContext(context=context, context_id=context_id)
        self._contexts[cow_ctx.context_id] = cow_ctx

        logger.info("cow_context_registered", context_id=cow_ctx.context_id)
        return cow_ctx

    def get(self, context_id: str) -> Optional[CoWContext]:
        """Get CoW context by ID.

        Args:
            context_id: Context identifier.

        Returns:
            CoWContext or None if not found.
        """
        return self._contexts.get(context_id)

    def remove(self, context_id: str) -> bool:
        """Remove CoW context.

        Args:
            context_id: Context to remove.

        Returns:
            True if removed, False if not found.
        """
        if context_id in self._contexts:
            del self._contexts[context_id]
            logger.info("cow_context_removed", context_id=context_id)
            return True
        return False

    def get_all_stats(self) -> Dict[str, CoWStats]:
        """Get statistics for all contexts.

        Returns:
            Dict mapping context_id to CoWStats.
        """
        return {
            ctx_id: ctx.get_stats()
            for ctx_id, ctx in self._contexts.items()
        }


# Global CoW context manager instance
_global_cow_manager: Optional[CoWContextManager] = None


def get_cow_manager() -> CoWContextManager:
    """Get global CoW context manager instance.

    Returns:
        The global CoWContextManager.
    """
    global _global_cow_manager
    if _global_cow_manager is None:
        _global_cow_manager = CoWContextManager()
    return _global_cow_manager
