"""Memory system data models.

Defines structures for short-term and long-term memory storage.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory entries."""

    FACT = "fact"  # Extracted facts
    PREFERENCE = "preference"  # User preferences
    CONVERSATION = "conversation"  # Conversation history
    ENTITY = "entity"  # Named entities
    CONTEXT = "context"  # Contextual information
    PROCEDURE = "procedure"  # How-to knowledge


class MemoryEntry(BaseModel):
    """A single memory entry."""

    id: str = Field(description="Unique identifier")
    type: MemoryType = Field(description="Type of memory")
    content: str = Field(description="Text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )
    source: Optional[str] = Field(default=None, description="Source of this memory")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this memory"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)
    ttl_seconds: Optional[int] = Field(
        default=None, description="Time-to-live in seconds"
    )

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


class ConversationMessage(BaseModel):
    """A message in conversation history."""

    role: str = Field(description="sender role (user, assistant, system)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    node_id: Optional[str] = Field(default=None, description="Node that generated this")


class ConversationSummary(BaseModel):
    """Summary of a conversation segment."""

    summary: str = Field(description="Text summary")
    key_points: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    message_count: int = Field(default=0)
    start_time: datetime
    end_time: datetime


class SearchResult(BaseModel):
    """Result from memory search."""

    entry: MemoryEntry
    score: float = Field(description="Relevance score 0-1")
    source: str = Field(description="Memory source (stm/ltm)")


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    # STM settings
    stm_max_messages: int = Field(default=20, ge=5)
    stm_summarize_threshold: int = Field(default=15, ge=3)

    # LTM settings
    ltm_max_entries: int = Field(default=10000, ge=100)
    ltm_search_k: int = Field(default=5, ge=1)
    ltm_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Embedding settings
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_dim: int = Field(default=768, ge=1)

    # Persistence
    persist_path: Optional[str] = Field(default=None)
    auto_persist: bool = Field(default=True)
