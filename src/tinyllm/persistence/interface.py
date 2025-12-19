"""Abstract storage interface for TinyLLM persistence.

This module defines the abstract interface that all storage backends must implement,
enabling a pluggable persistence layer.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, TypeVar, Generic
from uuid import UUID

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Configuration for storage backends."""

    # Common settings
    namespace: str = "default"
    max_items: int = 10000
    ttl_seconds: Optional[int] = None

    # SQLite settings
    sqlite_path: Optional[str] = None
    sqlite_wal_mode: bool = True

    # Redis settings
    redis_url: Optional[str] = None
    redis_prefix: str = "tinyllm"

    # Serialization
    compression_enabled: bool = False
    encryption_key: Optional[str] = None


class StorageItem(BaseModel):
    """Base class for storable items."""

    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(StorageItem):
    """A memory record for agent storage."""

    agent_id: str
    key: str
    value: Any
    memory_type: str = "general"  # general, fact, context, episodic
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = Field(default_factory=list)


class CheckpointRecord(StorageItem):
    """A checkpoint for resumable execution."""

    graph_id: str
    trace_id: str
    step: int
    state: Dict[str, Any]
    messages: List[Dict[str, Any]]
    node_id: str
    status: str = "pending"  # pending, completed, failed


class MessageRecord(StorageItem):
    """A message for inter-agent communication."""

    source_agent: str
    target_agent: Optional[str] = None  # None = broadcast
    channel: str = "default"
    payload: Dict[str, Any]
    priority: int = 0
    acknowledged: bool = False
    expires_at: Optional[datetime] = None


T = TypeVar("T", bound=StorageItem)


class StorageBackend(ABC, Generic[T]):
    """Abstract base class for storage backends.

    All storage implementations must implement these methods to provide
    a consistent interface for the persistence layer.
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, connections, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        pass

    @abstractmethod
    async def put(self, item: T) -> None:
        """Store an item.

        Args:
            item: The item to store.
        """
        pass

    @abstractmethod
    async def get(self, item_id: str) -> Optional[T]:
        """Retrieve an item by ID.

        Args:
            item_id: The item's unique identifier.

        Returns:
            The item if found, None otherwise.
        """
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID.

        Args:
            item_id: The item's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """List items with optional filtering.

        Args:
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            filters: Key-value pairs to filter by.

        Returns:
            List of matching items.
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count items with optional filtering.

        Args:
            filters: Key-value pairs to filter by.

        Returns:
            Number of matching items.
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all items.

        Returns:
            Number of items deleted.
        """
        pass

    # Convenience methods with default implementations

    async def exists(self, item_id: str) -> bool:
        """Check if an item exists.

        Args:
            item_id: The item's unique identifier.

        Returns:
            True if exists, False otherwise.
        """
        return await self.get(item_id) is not None

    async def get_many(self, item_ids: List[str]) -> List[T]:
        """Retrieve multiple items by ID.

        Args:
            item_ids: List of item identifiers.

        Returns:
            List of found items (may be shorter than input if some not found).
        """
        results = []
        for item_id in item_ids:
            item = await self.get(item_id)
            if item:
                results.append(item)
        return results

    async def put_many(self, items: List[T]) -> None:
        """Store multiple items.

        Args:
            items: List of items to store.
        """
        for item in items:
            await self.put(item)

    async def delete_many(self, item_ids: List[str]) -> int:
        """Delete multiple items.

        Args:
            item_ids: List of item identifiers.

        Returns:
            Number of items deleted.
        """
        count = 0
        for item_id in item_ids:
            if await self.delete(item_id):
                count += 1
        return count


class MemoryStorage(StorageBackend[MemoryRecord]):
    """Storage backend specialized for agent memory."""

    @abstractmethod
    async def search_by_key(self, agent_id: str, key_pattern: str) -> List[MemoryRecord]:
        """Search memories by key pattern.

        Args:
            agent_id: The agent's identifier.
            key_pattern: Pattern to match keys (supports wildcards).

        Returns:
            List of matching memory records.
        """
        pass

    @abstractmethod
    async def search_by_tags(self, agent_id: str, tags: List[str]) -> List[MemoryRecord]:
        """Search memories by tags.

        Args:
            agent_id: The agent's identifier.
            tags: Tags to match (OR logic).

        Returns:
            List of matching memory records.
        """
        pass

    @abstractmethod
    async def search_by_embedding(
        self,
        agent_id: str,
        embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[tuple[MemoryRecord, float]]:
        """Search memories by embedding similarity.

        Args:
            agent_id: The agent's identifier.
            embedding: Query embedding vector.
            top_k: Number of results to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of (record, similarity) tuples.
        """
        pass

    @abstractmethod
    async def get_recent(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryRecord]:
        """Get most recently accessed memories.

        Args:
            agent_id: The agent's identifier.
            memory_type: Optional type filter.
            limit: Maximum number of records.

        Returns:
            List of memory records sorted by last access time.
        """
        pass

    @abstractmethod
    async def prune_old(
        self,
        agent_id: str,
        max_age_seconds: int,
        keep_important: bool = True,
    ) -> int:
        """Remove old memories.

        Args:
            agent_id: The agent's identifier.
            max_age_seconds: Maximum age in seconds.
            keep_important: Whether to keep high-importance records.

        Returns:
            Number of records deleted.
        """
        pass


class CheckpointStorage(StorageBackend[CheckpointRecord]):
    """Storage backend specialized for execution checkpoints."""

    @abstractmethod
    async def save_checkpoint(
        self,
        graph_id: str,
        trace_id: str,
        step: int,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        node_id: str,
    ) -> CheckpointRecord:
        """Save an execution checkpoint.

        Args:
            graph_id: Graph identifier.
            trace_id: Trace identifier.
            step: Execution step number.
            state: Current execution state.
            messages: Messages at this point.
            node_id: Current node identifier.

        Returns:
            The created checkpoint record.
        """
        pass

    @abstractmethod
    async def get_latest_checkpoint(
        self,
        graph_id: str,
        trace_id: str,
    ) -> Optional[CheckpointRecord]:
        """Get the most recent checkpoint for an execution.

        Args:
            graph_id: Graph identifier.
            trace_id: Trace identifier.

        Returns:
            The latest checkpoint if found.
        """
        pass

    @abstractmethod
    async def list_checkpoints(
        self,
        graph_id: str,
        trace_id: Optional[str] = None,
    ) -> List[CheckpointRecord]:
        """List checkpoints for a graph/trace.

        Args:
            graph_id: Graph identifier.
            trace_id: Optional trace filter.

        Returns:
            List of checkpoint records.
        """
        pass


class MessageQueue(StorageBackend[MessageRecord]):
    """Storage backend for inter-agent messaging."""

    @abstractmethod
    async def publish(
        self,
        channel: str,
        payload: Dict[str, Any],
        source_agent: str,
        target_agent: Optional[str] = None,
        priority: int = 0,
        ttl_seconds: Optional[int] = None,
    ) -> MessageRecord:
        """Publish a message to a channel.

        Args:
            channel: Channel name.
            payload: Message payload.
            source_agent: Sending agent ID.
            target_agent: Optional target (None = broadcast).
            priority: Message priority (higher = more urgent).
            ttl_seconds: Time-to-live in seconds.

        Returns:
            The created message record.
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        channel: str,
        agent_id: str,
    ) -> AsyncIterator[MessageRecord]:
        """Subscribe to messages on a channel.

        Args:
            channel: Channel name.
            agent_id: Subscribing agent ID.

        Yields:
            Message records as they arrive.
        """
        pass

    @abstractmethod
    async def acknowledge(self, message_id: str) -> bool:
        """Acknowledge receipt of a message.

        Args:
            message_id: Message identifier.

        Returns:
            True if acknowledged, False if not found.
        """
        pass

    @abstractmethod
    async def get_pending(
        self,
        agent_id: str,
        channel: Optional[str] = None,
        limit: int = 100,
    ) -> List[MessageRecord]:
        """Get pending messages for an agent.

        Args:
            agent_id: Agent identifier.
            channel: Optional channel filter.
            limit: Maximum messages to return.

        Returns:
            List of unacknowledged messages.
        """
        pass
