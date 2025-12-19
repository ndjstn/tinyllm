"""Persistence layer for TinyLLM.

Provides abstract storage interfaces and implementations for:
- Per-agent SQLite memory storage
- Redis pub/sub for inter-agent messaging
- Checkpointing and state persistence
"""

from tinyllm.persistence.interface import StorageBackend, StorageConfig
from tinyllm.persistence.sqlite_backend import SQLiteBackend
from tinyllm.persistence.memory_backend import InMemoryBackend
from tinyllm.persistence.redis_backend import RedisMessageQueue

__all__ = [
    "StorageBackend",
    "StorageConfig",
    "SQLiteBackend",
    "InMemoryBackend",
    "RedisMessageQueue",
]
