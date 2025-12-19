"""In-memory storage backend for TinyLLM.

Provides a simple in-memory storage implementation for testing and development.
"""

from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

from tinyllm.persistence.interface import (
    CheckpointRecord,
    CheckpointStorage,
    MemoryRecord,
    MemoryStorage,
    MessageQueue,
    MessageRecord,
    StorageConfig,
)


class InMemoryBackend(MemoryStorage):
    """In-memory storage for agent memory.

    Useful for testing and development. Data is not persisted.
    """

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._data: Dict[str, MemoryRecord] = {}
        self._key_index: Dict[Tuple[str, str], str] = {}  # (agent_id, key) -> id

    async def initialize(self) -> None:
        """Initialize the backend."""
        self._initialized = True

    async def close(self) -> None:
        """Close the backend."""
        self._data.clear()
        self._key_index.clear()
        self._initialized = False

    async def put(self, item: MemoryRecord) -> None:
        """Store or update a memory record."""
        # Update key index
        old_key = (item.agent_id, item.key)
        if old_key in self._key_index:
            old_id = self._key_index[old_key]
            if old_id != item.id:
                del self._data[old_id]

        item.updated_at = datetime.utcnow()
        self._data[item.id] = item
        self._key_index[(item.agent_id, item.key)] = item.id

    async def get(self, item_id: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID."""
        item = self._data.get(item_id)
        if item:
            item.access_count += 1
            item.last_accessed = datetime.utcnow()
        return item

    async def get_by_key(self, agent_id: str, key: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by agent ID and key."""
        item_id = self._key_index.get((agent_id, key))
        if item_id:
            return await self.get(item_id)
        return None

    async def delete(self, item_id: str) -> bool:
        """Delete a memory record by ID."""
        if item_id in self._data:
            item = self._data[item_id]
            del self._key_index[(item.agent_id, item.key)]
            del self._data[item_id]
            return True
        return False

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """List memory records with optional filtering."""
        items = list(self._data.values())

        if filters:
            if "agent_id" in filters:
                items = [i for i in items if i.agent_id == filters["agent_id"]]
            if "memory_type" in filters:
                items = [i for i in items if i.memory_type == filters["memory_type"]]
            if "min_importance" in filters:
                items = [i for i in items if i.importance >= filters["min_importance"]]

        items.sort(key=lambda x: x.updated_at, reverse=True)
        return items[offset : offset + limit]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count memory records with optional filtering."""
        return len(await self.list(limit=100000, filters=filters))

    async def clear(self) -> int:
        """Clear all memory records."""
        count = len(self._data)
        self._data.clear()
        self._key_index.clear()
        return count

    async def search_by_key(
        self, agent_id: str, key_pattern: str
    ) -> List[MemoryRecord]:
        """Search memories by key pattern."""
        import fnmatch

        results = []
        for item in self._data.values():
            if item.agent_id == agent_id and fnmatch.fnmatch(item.key, key_pattern):
                results.append(item)
        return results

    async def search_by_tags(
        self, agent_id: str, tags: List[str]
    ) -> List[MemoryRecord]:
        """Search memories containing any of the specified tags."""
        results = []
        tag_set = set(tags)
        for item in self._data.values():
            if item.agent_id == agent_id and tag_set & set(item.tags):
                results.append(item)
        return results

    async def search_by_embedding(
        self,
        agent_id: str,
        embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search memories by embedding similarity."""
        import math

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            if len(a) != len(b) or not a:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        results = []
        for item in self._data.values():
            if item.agent_id == agent_id and item.embedding:
                similarity = cosine_similarity(embedding, item.embedding)
                if similarity >= min_similarity:
                    results.append((item, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get_recent(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryRecord]:
        """Get most recently accessed memories."""
        items = [i for i in self._data.values() if i.agent_id == agent_id]
        if memory_type:
            items = [i for i in items if i.memory_type == memory_type]

        items.sort(
            key=lambda x: x.last_accessed or datetime.min,
            reverse=True,
        )
        return items[:limit]

    async def prune_old(
        self,
        agent_id: str,
        max_age_seconds: int,
        keep_important: bool = True,
    ) -> int:
        """Remove old memories, optionally keeping important ones."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        to_delete = []

        for item_id, item in self._data.items():
            if item.agent_id != agent_id:
                continue
            if item.updated_at < cutoff:
                if keep_important and item.importance >= 0.8:
                    continue
                to_delete.append(item_id)

        for item_id in to_delete:
            await self.delete(item_id)

        return len(to_delete)


class InMemoryCheckpointStorage(CheckpointStorage):
    """In-memory storage for execution checkpoints."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._data: Dict[str, CheckpointRecord] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._data.clear()
        self._initialized = False

    async def put(self, item: CheckpointRecord) -> None:
        item.updated_at = datetime.utcnow()
        self._data[item.id] = item

    async def get(self, item_id: str) -> Optional[CheckpointRecord]:
        return self._data.get(item_id)

    async def delete(self, item_id: str) -> bool:
        if item_id in self._data:
            del self._data[item_id]
            return True
        return False

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CheckpointRecord]:
        items = list(self._data.values())

        if filters:
            if "graph_id" in filters:
                items = [i for i in items if i.graph_id == filters["graph_id"]]
            if "trace_id" in filters:
                items = [i for i in items if i.trace_id == filters["trace_id"]]
            if "status" in filters:
                items = [i for i in items if i.status == filters["status"]]

        items.sort(key=lambda x: x.created_at, reverse=True)
        return items[offset : offset + limit]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        return len(await self.list(limit=100000, filters=filters))

    async def clear(self) -> int:
        count = len(self._data)
        self._data.clear()
        return count

    async def save_checkpoint(
        self,
        graph_id: str,
        trace_id: str,
        step: int,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        node_id: str,
    ) -> CheckpointRecord:
        record = CheckpointRecord(
            id=str(uuid4()),
            graph_id=graph_id,
            trace_id=trace_id,
            step=step,
            state=state,
            messages=messages,
            node_id=node_id,
        )
        await self.put(record)
        return record

    async def get_latest_checkpoint(
        self,
        graph_id: str,
        trace_id: str,
    ) -> Optional[CheckpointRecord]:
        matching = [
            c for c in self._data.values()
            if c.graph_id == graph_id and c.trace_id == trace_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda c: c.step)

    async def list_checkpoints(
        self,
        graph_id: str,
        trace_id: Optional[str] = None,
    ) -> List[CheckpointRecord]:
        filters: Dict[str, Any] = {"graph_id": graph_id}
        if trace_id:
            filters["trace_id"] = trace_id
        return await self.list(filters=filters)
