"""SQLite storage backend for TinyLLM persistence.

Provides per-agent SQLite databases for isolated memory storage with support for:
- Key-value storage with metadata
- Full-text search
- Semantic search via embeddings
- Automatic schema migration
"""

import asyncio
import json
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite

from tinyllm.persistence.interface import (
    CheckpointRecord,
    CheckpointStorage,
    MemoryRecord,
    MemoryStorage,
    MessageQueue,
    MessageRecord,
    StorageConfig,
)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteBackend(MemoryStorage):
    """SQLite-based storage for agent memory.

    Each agent gets its own SQLite database file for isolation.
    """

    def __init__(self, config: StorageConfig, agent_id: str = "default"):
        super().__init__(config)
        self.agent_id = agent_id
        self._db: Optional[aiosqlite.Connection] = None
        self._db_path = self._get_db_path()

    def _get_db_path(self) -> Path:
        """Get the database file path for this agent."""
        if self.config.sqlite_path:
            base_path = Path(self.config.sqlite_path)
        else:
            base_path = Path.home() / ".tinyllm" / "data"

        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / f"{self.agent_id}.db"

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self._initialized:
            return

        self._db = await aiosqlite.connect(str(self._db_path))

        if self.config.sqlite_wal_mode:
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA synchronous=NORMAL")

        # Create tables
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                memory_type TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                embedding TEXT,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(agent_id, key)
            )
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_agent_key
            ON memories(agent_id, key)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(agent_id, memory_type)
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(agent_id, importance DESC)
        """)

        await self._db.commit()
        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        self._initialized = False

    def _record_to_dict(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row to a MemoryRecord."""
        return MemoryRecord(
            id=row["id"],
            agent_id=row["agent_id"],
            key=row["key"],
            value=json.loads(row["value"]),
            memory_type=row["memory_type"],
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def put(self, item: MemoryRecord) -> None:
        """Store or update a memory record."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        now = datetime.utcnow().isoformat()

        await self._db.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, agent_id, key, value, memory_type, importance, access_count,
             last_accessed, embedding, tags, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.agent_id,
                item.key,
                json.dumps(item.value),
                item.memory_type,
                item.importance,
                item.access_count,
                item.last_accessed.isoformat() if item.last_accessed else None,
                json.dumps(item.embedding) if item.embedding else None,
                json.dumps(item.tags),
                json.dumps(item.metadata),
                item.created_at.isoformat(),
                now,
            ),
        )
        await self._db.commit()

    async def get(self, item_id: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM memories WHERE id = ?", (item_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Update access stats
                await self._db.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE id = ?
                    """,
                    (datetime.utcnow().isoformat(), item_id),
                )
                await self._db.commit()
                return self._record_to_dict(row)
        return None

    async def get_by_key(self, agent_id: str, key: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by agent ID and key."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND key = ?",
            (agent_id, key),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                # Update access stats
                await self._db.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE agent_id = ? AND key = ?
                    """,
                    (datetime.utcnow().isoformat(), agent_id, key),
                )
                await self._db.commit()
                return self._record_to_dict(row)
        return None

    async def delete(self, item_id: str) -> bool:
        """Delete a memory record by ID."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute(
            "DELETE FROM memories WHERE id = ?", (item_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        """List memory records with optional filtering."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        query = "SELECT * FROM memories WHERE 1=1"
        params: List[Any] = []

        if filters:
            if "agent_id" in filters:
                query += " AND agent_id = ?"
                params.append(filters["agent_id"])
            if "memory_type" in filters:
                query += " AND memory_type = ?"
                params.append(filters["memory_type"])
            if "min_importance" in filters:
                query += " AND importance >= ?"
                params.append(filters["min_importance"])

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._record_to_dict(row) for row in rows]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count memory records with optional filtering."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        query = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params: List[Any] = []

        if filters:
            if "agent_id" in filters:
                query += " AND agent_id = ?"
                params.append(filters["agent_id"])
            if "memory_type" in filters:
                query += " AND memory_type = ?"
                params.append(filters["memory_type"])

        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear(self) -> int:
        """Clear all memory records for this agent."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute(
            "DELETE FROM memories WHERE agent_id = ?", (self.agent_id,)
        )
        await self._db.commit()
        return cursor.rowcount

    async def search_by_key(
        self, agent_id: str, key_pattern: str
    ) -> List[MemoryRecord]:
        """Search memories by key pattern using LIKE."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        # Convert wildcards
        sql_pattern = key_pattern.replace("*", "%").replace("?", "_")

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND key LIKE ?",
            (agent_id, sql_pattern),
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._record_to_dict(row) for row in rows]

    async def search_by_tags(
        self, agent_id: str, tags: List[str]
    ) -> List[MemoryRecord]:
        """Search memories containing any of the specified tags."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        # Build OR query for tags
        conditions = " OR ".join(["tags LIKE ?" for _ in tags])
        params = [agent_id] + [f'%"{tag}"%' for tag in tags]

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            f"SELECT * FROM memories WHERE agent_id = ? AND ({conditions})",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._record_to_dict(row) for row in rows]

    async def search_by_embedding(
        self,
        agent_id: str,
        embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Search memories by embedding similarity (brute-force for now)."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND embedding IS NOT NULL",
            (agent_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        results: List[Tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._record_to_dict(row)
            if record.embedding:
                similarity = cosine_similarity(embedding, record.embedding)
                if similarity >= min_similarity:
                    results.append((record, similarity))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get_recent(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryRecord]:
        """Get most recently accessed memories."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        query = "SELECT * FROM memories WHERE agent_id = ?"
        params: List[Any] = [agent_id]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY last_accessed DESC NULLS LAST LIMIT ?"
        params.append(limit)

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [self._record_to_dict(row) for row in rows]

    async def prune_old(
        self,
        agent_id: str,
        max_age_seconds: int,
        keep_important: bool = True,
    ) -> int:
        """Remove old memories, optionally keeping important ones."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        cutoff = (datetime.utcnow() - timedelta(seconds=max_age_seconds)).isoformat()

        query = "DELETE FROM memories WHERE agent_id = ? AND updated_at < ?"
        params: List[Any] = [agent_id, cutoff]

        if keep_important:
            query += " AND importance < 0.8"

        cursor = await self._db.execute(query, params)
        await self._db.commit()
        return cursor.rowcount


class SQLiteCheckpointStorage(CheckpointStorage):
    """SQLite-based storage for execution checkpoints."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._db: Optional[aiosqlite.Connection] = None
        self._db_path = self._get_db_path()

    def _get_db_path(self) -> Path:
        """Get the database file path."""
        if self.config.sqlite_path:
            base_path = Path(self.config.sqlite_path)
        else:
            base_path = Path.home() / ".tinyllm" / "data"

        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / "checkpoints.db"

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self._initialized:
            return

        self._db = await aiosqlite.connect(str(self._db_path))

        if self.config.sqlite_wal_mode:
            await self._db.execute("PRAGMA journal_mode=WAL")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                state TEXT NOT NULL,
                messages TEXT NOT NULL,
                node_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_graph_trace
            ON checkpoints(graph_id, trace_id, step DESC)
        """)

        await self._db.commit()
        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        self._initialized = False

    async def put(self, item: CheckpointRecord) -> None:
        """Store a checkpoint record."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        now = datetime.utcnow().isoformat()

        await self._db.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (id, graph_id, trace_id, step, state, messages, node_id, status,
             metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.graph_id,
                item.trace_id,
                item.step,
                json.dumps(item.state),
                json.dumps(item.messages),
                item.node_id,
                item.status,
                json.dumps(item.metadata),
                item.created_at.isoformat(),
                now,
            ),
        )
        await self._db.commit()

    async def get(self, item_id: str) -> Optional[CheckpointRecord]:
        """Retrieve a checkpoint by ID."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM checkpoints WHERE id = ?", (item_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return CheckpointRecord(
                    id=row["id"],
                    graph_id=row["graph_id"],
                    trace_id=row["trace_id"],
                    step=row["step"],
                    state=json.loads(row["state"]),
                    messages=json.loads(row["messages"]),
                    node_id=row["node_id"],
                    status=row["status"],
                    metadata=json.loads(row["metadata"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None

    async def delete(self, item_id: str) -> bool:
        """Delete a checkpoint by ID."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute(
            "DELETE FROM checkpoints WHERE id = ?", (item_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CheckpointRecord]:
        """List checkpoints with optional filtering."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        query = "SELECT * FROM checkpoints WHERE 1=1"
        params: List[Any] = []

        if filters:
            if "graph_id" in filters:
                query += " AND graph_id = ?"
                params.append(filters["graph_id"])
            if "trace_id" in filters:
                query += " AND trace_id = ?"
                params.append(filters["trace_id"])
            if "status" in filters:
                query += " AND status = ?"
                params.append(filters["status"])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [
                CheckpointRecord(
                    id=row["id"],
                    graph_id=row["graph_id"],
                    trace_id=row["trace_id"],
                    step=row["step"],
                    state=json.loads(row["state"]),
                    messages=json.loads(row["messages"]),
                    node_id=row["node_id"],
                    status=row["status"],
                    metadata=json.loads(row["metadata"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count checkpoints."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        query = "SELECT COUNT(*) FROM checkpoints WHERE 1=1"
        params: List[Any] = []

        if filters:
            if "graph_id" in filters:
                query += " AND graph_id = ?"
                params.append(filters["graph_id"])
            if "trace_id" in filters:
                query += " AND trace_id = ?"
                params.append(filters["trace_id"])

        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear(self) -> int:
        """Clear all checkpoints."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute("DELETE FROM checkpoints")
        await self._db.commit()
        return cursor.rowcount

    async def save_checkpoint(
        self,
        graph_id: str,
        trace_id: str,
        step: int,
        state: Dict[str, Any],
        messages: List[Dict[str, Any]],
        node_id: str,
    ) -> CheckpointRecord:
        """Save an execution checkpoint."""
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
        """Get the most recent checkpoint for an execution."""
        if not self._db:
            raise RuntimeError("Database not initialized")

        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            """
            SELECT * FROM checkpoints
            WHERE graph_id = ? AND trace_id = ?
            ORDER BY step DESC
            LIMIT 1
            """,
            (graph_id, trace_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return CheckpointRecord(
                    id=row["id"],
                    graph_id=row["graph_id"],
                    trace_id=row["trace_id"],
                    step=row["step"],
                    state=json.loads(row["state"]),
                    messages=json.loads(row["messages"]),
                    node_id=row["node_id"],
                    status=row["status"],
                    metadata=json.loads(row["metadata"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
        return None

    async def list_checkpoints(
        self,
        graph_id: str,
        trace_id: Optional[str] = None,
    ) -> List[CheckpointRecord]:
        """List checkpoints for a graph/trace."""
        filters: Dict[str, Any] = {"graph_id": graph_id}
        if trace_id:
            filters["trace_id"] = trace_id
        return await self.list(filters=filters)
