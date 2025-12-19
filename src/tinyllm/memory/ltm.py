"""Long-Term Memory (LTM) implementation.

Provides persistent memory with vector-based semantic search
across sessions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import hashlib
import math

from pydantic import BaseModel

from tinyllm.memory.models import (
    MemoryConfig,
    MemoryEntry,
    MemoryType,
    SearchResult,
)


class VectorStore:
    """Simple in-memory vector store with cosine similarity search.

    For production use, this should be replaced with a proper
    vector database like ChromaDB, Pinecone, or Qdrant.
    """

    def __init__(self, dim: int = 768):
        """Initialize vector store.

        Args:
            dim: Embedding dimension.
        """
        self.dim = dim
        self._vectors: Dict[str, List[float]] = {}
        self._entries: Dict[str, MemoryEntry] = {}

    def add(self, entry_id: str, embedding: List[float], entry: MemoryEntry) -> None:
        """Add an entry with its embedding.

        Args:
            entry_id: Unique identifier.
            embedding: Vector embedding.
            entry: Memory entry.
        """
        if len(embedding) != self.dim:
            raise ValueError(f"Embedding dim {len(embedding)} != store dim {self.dim}")
        self._vectors[entry_id] = embedding
        self._entries[entry_id] = entry

    def remove(self, entry_id: str) -> bool:
        """Remove an entry.

        Args:
            entry_id: Entry to remove.

        Returns:
            True if entry was removed.
        """
        if entry_id in self._vectors:
            del self._vectors[entry_id]
            del self._entries[entry_id]
            return True
        return False

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[tuple]:
        """Search for similar entries.

        Args:
            query_embedding: Query vector.
            k: Number of results.
            threshold: Minimum similarity.

        Returns:
            List of (entry_id, score, entry) tuples.
        """
        if len(query_embedding) != self.dim:
            raise ValueError(
                f"Query embedding dim {len(query_embedding)} != store dim {self.dim}"
            )

        results = []
        for entry_id, vec in self._vectors.items():
            score = self._cosine_similarity(query_embedding, vec)
            if score >= threshold:
                results.append((entry_id, score, self._entries[entry_id]))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID.

        Args:
            entry_id: Entry ID.

        Returns:
            Memory entry or None.
        """
        return self._entries.get(entry_id)

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class LTM:
    """Long-Term Memory with vector-based semantic search.

    Provides persistent memory across sessions using:
    - Vector embeddings for semantic search
    - Optional Ollama integration for embeddings
    - JSON file persistence

    Features:
    - Semantic similarity search
    - Memory types and metadata
    - Automatic expiration handling
    - Persistence to disk
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """Initialize LTM.

        Args:
            config: Memory configuration.
            embed_fn: Optional embedding function. If not provided,
                     uses simple hash-based pseudo-embeddings.
        """
        self.config = config or MemoryConfig()
        self._embed_fn = embed_fn
        self._store = VectorStore(dim=self.config.embedding_dim)
        self._facts: Dict[str, MemoryEntry] = {}  # For direct fact lookup
        self._loaded = False

        # Load from persistence if configured
        if self.config.persist_path:
            self._load()

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store a memory entry.

        Args:
            content: Text content to store.
            memory_type: Type of memory.
            metadata: Additional metadata.
            entry_id: Optional ID (generated if not provided).

        Returns:
            The stored entry.
        """
        # Generate ID if not provided
        if not entry_id:
            entry_id = f"ltm_{hashlib.md5(content.encode()).hexdigest()[:12]}"

        # Get embedding
        embedding = self._get_embedding(content)

        # Create entry
        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            source="ltm",
        )

        # Store in vector store
        self._store.add(entry_id, embedding, entry)

        # Also store in facts dict for direct lookup
        if memory_type == MemoryType.FACT:
            self._facts[entry_id] = entry

        # Persist if configured
        if self.config.auto_persist:
            self._persist()

        return entry

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> List[SearchResult]:
        """Retrieve memories by semantic similarity.

        Args:
            query: Search query.
            k: Number of results (default from config).
            memory_type: Filter by type.

        Returns:
            List of search results.
        """
        k = k or self.config.ltm_search_k
        threshold = self.config.ltm_similarity_threshold

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Search vector store
        results = self._store.search(query_embedding, k=k * 2, threshold=threshold)

        # Filter by type and convert to SearchResult
        search_results = []
        for entry_id, score, entry in results:
            if memory_type and entry.type != memory_type:
                continue
            if entry.is_expired():
                continue

            entry.touch()
            search_results.append(SearchResult(entry=entry, score=score, source="ltm"))

        return search_results[:k]

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID.

        Args:
            entry_id: Entry ID.

        Returns:
            Memory entry or None.
        """
        entry = self._store.get(entry_id)
        if entry:
            entry.touch()
        return entry

    def get_fact(self, key: str) -> Optional[MemoryEntry]:
        """Get a fact by key/ID.

        Args:
            key: Fact key.

        Returns:
            Memory entry or None.
        """
        return self._facts.get(key)

    def delete(self, entry_id: str) -> bool:
        """Delete an entry.

        Args:
            entry_id: Entry to delete.

        Returns:
            True if deleted.
        """
        removed = self._store.remove(entry_id)
        self._facts.pop(entry_id, None)

        if removed and self.config.auto_persist:
            self._persist()

        return removed

    def get_all(
        self, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryEntry]:
        """Get all entries, optionally filtered by type.

        Args:
            memory_type: Filter by type.

        Returns:
            List of entries.
        """
        entries = []
        for entry in self._store._entries.values():
            if memory_type and entry.type != memory_type:
                continue
            if not entry.is_expired():
                entries.append(entry)
        return entries

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        expired = [
            entry_id
            for entry_id, entry in self._store._entries.items()
            if entry.is_expired()
        ]

        for entry_id in expired:
            self.delete(entry_id)

        return len(expired)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if self._embed_fn:
            return self._embed_fn(text)

        # Fallback: simple hash-based pseudo-embedding
        return self._pseudo_embedding(text)

    def _pseudo_embedding(self, text: str) -> List[float]:
        """Generate a pseudo-embedding using hashing.

        This is NOT suitable for real semantic search but works
        for testing and basic deduplication.
        """
        # Use multiple hashes to create a vector
        embedding = []
        for i in range(self.config.embedding_dim):
            h = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
            # Convert hex to float between -1 and 1
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1
            embedding.append(val)

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _persist(self) -> None:
        """Persist LTM to disk."""
        if not self.config.persist_path:
            return

        path = Path(self.config.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize entries
        data = {
            "entries": [
                {
                    "id": entry.id,
                    "type": entry.type.value,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "embedding": entry.embedding,
                    "source": entry.source,
                    "confidence": entry.confidence,
                    "created_at": entry.created_at.isoformat(),
                    "accessed_at": entry.accessed_at.isoformat(),
                    "access_count": entry.access_count,
                    "ttl_seconds": entry.ttl_seconds,
                }
                for entry in self._store._entries.values()
            ],
            "config": {
                "embedding_dim": self.config.embedding_dim,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load LTM from disk."""
        if not self.config.persist_path:
            return

        path = Path(self.config.persist_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = MemoryEntry(
                    id=entry_data["id"],
                    type=MemoryType(entry_data["type"]),
                    content=entry_data["content"],
                    metadata=entry_data.get("metadata", {}),
                    embedding=entry_data.get("embedding"),
                    source=entry_data.get("source", "ltm"),
                    confidence=entry_data.get("confidence", 1.0),
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    accessed_at=datetime.fromisoformat(entry_data["accessed_at"]),
                    access_count=entry_data.get("access_count", 0),
                    ttl_seconds=entry_data.get("ttl_seconds"),
                )

                if entry.embedding:
                    self._store.add(entry.id, entry.embedding, entry)
                    if entry.type == MemoryType.FACT:
                        self._facts[entry.id] = entry

            self._loaded = True

        except Exception:
            # Log error in real implementation
            pass

    def clear(self) -> None:
        """Clear all LTM data."""
        self._store = VectorStore(dim=self.config.embedding_dim)
        self._facts.clear()

        if self.config.auto_persist:
            self._persist()

    def get_stats(self) -> Dict[str, Any]:
        """Get LTM statistics."""
        type_counts: Dict[str, int] = {}
        for entry in self._store._entries.values():
            type_counts[entry.type.value] = type_counts.get(entry.type.value, 0) + 1

        return {
            "total_entries": len(self._store),
            "facts": len(self._facts),
            "type_counts": type_counts,
            "max_entries": self.config.ltm_max_entries,
            "loaded_from_disk": self._loaded,
        }
