"""Short-Term Memory (STM) implementation.

Manages conversation history and extracted context
within a single session.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib

from pydantic import BaseModel, Field

from tinyllm.memory.models import (
    ConversationMessage,
    ConversationSummary,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
    SearchResult,
)


class STM:
    """Short-Term Memory for managing conversation state.

    Maintains recent messages and extracted context within
    a single conversation session.

    Features:
    - Sliding window of recent messages
    - Automatic summarization when threshold reached
    - Context extraction and storage
    - Fast keyword-based search
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize STM.

        Args:
            config: Memory configuration.
        """
        self.config = config or MemoryConfig()
        self._messages: List[ConversationMessage] = []
        self._context: Dict[str, MemoryEntry] = {}
        self._summaries: List[ConversationSummary] = []
        self._entities: Dict[str, str] = {}  # entity -> most recent value

    @property
    def messages(self) -> List[ConversationMessage]:
        """Get all messages."""
        return self._messages.copy()

    @property
    def message_count(self) -> int:
        """Get current message count."""
        return len(self._messages)

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> ConversationMessage:
        """Add a message to STM.

        Args:
            role: Message role (user, assistant, system).
            content: Message content.
            metadata: Additional metadata.
            node_id: Node that generated this message.

        Returns:
            The created message.
        """
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            node_id=node_id,
        )
        self._messages.append(message)

        # Check if we need to summarize
        if len(self._messages) >= self.config.stm_max_messages:
            self._summarize_and_prune()

        return message

    def add_context(
        self,
        key: str,
        content: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Add or update context entry.

        Args:
            key: Context key.
            content: Context content.
            memory_type: Type of memory entry.
            metadata: Additional metadata.

        Returns:
            The created/updated entry.
        """
        entry_id = f"stm_{hashlib.md5(key.encode()).hexdigest()[:8]}"

        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            metadata=metadata or {},
            source="stm",
        )
        self._context[key] = entry

        return entry

    def get_context(self, key: str) -> Optional[MemoryEntry]:
        """Get a context entry by key.

        Args:
            key: Context key.

        Returns:
            Memory entry or None.
        """
        entry = self._context.get(key)
        if entry:
            entry.touch()
        return entry

    def extract_entity(self, name: str, value: str) -> None:
        """Extract and store a named entity.

        Args:
            name: Entity name.
            value: Entity value.
        """
        self._entities[name] = value
        self.add_context(
            f"entity_{name}",
            f"{name}: {value}",
            memory_type=MemoryType.ENTITY,
            metadata={"entity_name": name, "entity_value": value},
        )

    def get_entity(self, name: str) -> Optional[str]:
        """Get an entity value.

        Args:
            name: Entity name.

        Returns:
            Entity value or None.
        """
        return self._entities.get(name)

    def get_recent_messages(
        self, n: Optional[int] = None, role: Optional[str] = None
    ) -> List[ConversationMessage]:
        """Get recent messages.

        Args:
            n: Number of messages (default: all).
            role: Filter by role.

        Returns:
            List of recent messages.
        """
        messages = self._messages
        if role:
            messages = [m for m in messages if m.role == role]

        if n is not None:
            messages = messages[-n:]

        return messages

    def search(
        self, query: str, limit: int = 5
    ) -> List[SearchResult]:
        """Search STM with keyword matching.

        Args:
            query: Search query.
            limit: Max results.

        Returns:
            Matching results.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Search context
        for key, entry in self._context.items():
            content_lower = entry.content.lower()
            key_lower = key.lower()

            # Score based on word overlap
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)

            if overlap > 0 or query_lower in content_lower or query_lower in key_lower:
                score = overlap / max(len(query_words), 1)
                if query_lower in content_lower:
                    score += 0.3
                if query_lower in key_lower:
                    score += 0.2

                results.append(SearchResult(entry=entry, score=min(1.0, score), source="stm"))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_conversation_context(self, max_tokens: int = 2000) -> str:
        """Get formatted conversation context for prompts.

        Args:
            max_tokens: Approximate max tokens (chars/4).

        Returns:
            Formatted context string.
        """
        max_chars = max_tokens * 4
        parts = []

        # Add summaries first
        for summary in self._summaries[-3:]:  # Last 3 summaries
            parts.append(f"[Summary] {summary.summary}")

        # Add recent messages
        for msg in self._messages[-10:]:
            prefix = f"[{msg.role}]"
            parts.append(f"{prefix} {msg.content}")

        # Add key context
        for key, entry in list(self._context.items())[:5]:
            if entry.type in (MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.ENTITY):
                parts.append(f"[Context: {key}] {entry.content}")

        # Combine and truncate
        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[-max_chars:]
            # Find first newline for clean break
            first_nl = result.find("\n")
            if first_nl > 0:
                result = result[first_nl + 1:]

        return result

    def _summarize_and_prune(self) -> None:
        """Summarize older messages and prune to maintain size."""
        if len(self._messages) <= self.config.stm_summarize_threshold:
            return

        # Take messages to summarize (keep last few)
        keep_count = 5
        to_summarize = self._messages[:-keep_count]
        self._messages = self._messages[-keep_count:]

        # Create simple summary
        summary_parts = []
        for msg in to_summarize:
            if msg.role == "user":
                summary_parts.append(f"User asked about: {msg.content[:100]}...")
            elif msg.role == "assistant":
                summary_parts.append(f"Assistant responded: {msg.content[:100]}...")

        summary = ConversationSummary(
            summary=" | ".join(summary_parts[:5]),
            key_points=[m.content[:50] for m in to_summarize if m.role == "user"][:3],
            entities=list(self._entities.keys())[:10],
            message_count=len(to_summarize),
            start_time=to_summarize[0].timestamp,
            end_time=to_summarize[-1].timestamp,
        )
        self._summaries.append(summary)

    def clear(self) -> None:
        """Clear all STM data."""
        self._messages.clear()
        self._context.clear()
        self._summaries.clear()
        self._entities.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get STM statistics."""
        return {
            "message_count": len(self._messages),
            "context_entries": len(self._context),
            "summaries": len(self._summaries),
            "entities": len(self._entities),
            "max_messages": self.config.stm_max_messages,
        }
