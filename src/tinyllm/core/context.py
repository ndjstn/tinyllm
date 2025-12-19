"""Execution context for TinyLLM graph execution.

This module provides the ExecutionContext class that maintains state
during graph execution, including visited nodes, messages, and variables.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.config.loader import Config
from tinyllm.core.message import Message


class ExecutionContext(BaseModel):
    """Context maintained during graph execution.

    The ExecutionContext holds all state needed during the execution
    of a graph, including visited nodes, messages, and arbitrary variables.
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

    def add_message(self, message: Message) -> None:
        """Add a message to the execution.

        Args:
            message: Message to add.
        """
        self.messages.append(message)

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

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        delta = datetime.utcnow() - self.start_time
        return int(delta.total_seconds() * 1000)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_tokens_in + self.total_tokens_out
