"""Unified memory store.

Combines STM and LTM into a single interface for
memory operations across the system.
"""

from typing import Any, Callable, Dict, List, Optional

from tinyllm.memory.models import (
    ConversationMessage,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
    SearchResult,
)
from tinyllm.memory.stm import STM
from tinyllm.memory.ltm import LTM


class MemoryStore:
    """Unified memory store combining STM and LTM.

    Provides a single interface for:
    - Conversation history (STM)
    - Context tracking (STM)
    - Persistent facts (LTM)
    - Semantic search (LTM)
    - Cross-memory search

    Usage:
        store = MemoryStore()

        # Conversation
        store.add_message("user", "Hello!")
        store.add_message("assistant", "Hi there!")

        # Facts
        store.store_fact("user_name", "The user's name is Alice")

        # Search
        results = store.search("What is the user's name?")
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """Initialize memory store.

        Args:
            config: Memory configuration.
            embed_fn: Optional embedding function for LTM.
        """
        self.config = config or MemoryConfig()
        self.stm = STM(self.config)
        self.ltm = LTM(self.config, embed_fn)

    # ==================== Message Operations ====================

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> ConversationMessage:
        """Add a conversation message.

        Args:
            role: Message role (user, assistant, system).
            content: Message content.
            metadata: Additional metadata.
            node_id: Node that generated this.

        Returns:
            The created message.
        """
        return self.stm.add_message(role, content, metadata, node_id)

    def get_messages(
        self, n: Optional[int] = None, role: Optional[str] = None
    ) -> List[ConversationMessage]:
        """Get recent messages.

        Args:
            n: Number of messages.
            role: Filter by role.

        Returns:
            List of messages.
        """
        return self.stm.get_recent_messages(n, role)

    # ==================== Context Operations ====================

    def set_context(
        self,
        key: str,
        value: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        persistent: bool = False,
    ) -> MemoryEntry:
        """Set a context value.

        Args:
            key: Context key.
            value: Context value.
            memory_type: Type of memory.
            persistent: Store in LTM if True.

        Returns:
            The created entry.
        """
        if persistent:
            return self.ltm.store(value, memory_type, {"key": key}, f"ctx_{key}")
        else:
            return self.stm.add_context(key, value, memory_type)

    def get_context(self, key: str) -> Optional[str]:
        """Get a context value.

        Args:
            key: Context key.

        Returns:
            Context value or None.
        """
        # Try STM first
        entry = self.stm.get_context(key)
        if entry:
            return entry.content

        # Try LTM
        entry = self.ltm.get(f"ctx_{key}")
        if entry:
            return entry.content

        return None

    # ==================== Entity Operations ====================

    def extract_entity(self, name: str, value: str, persistent: bool = False) -> None:
        """Extract and store a named entity.

        Args:
            name: Entity name.
            value: Entity value.
            persistent: Store in LTM if True.
        """
        self.stm.extract_entity(name, value)

        if persistent:
            self.ltm.store(
                f"{name}: {value}",
                MemoryType.ENTITY,
                {"entity_name": name, "entity_value": value},
                f"entity_{name}",
            )

    def get_entity(self, name: str) -> Optional[str]:
        """Get an entity value.

        Args:
            name: Entity name.

        Returns:
            Entity value or None.
        """
        # Try STM first
        value = self.stm.get_entity(name)
        if value:
            return value

        # Try LTM
        entry = self.ltm.get(f"entity_{name}")
        if entry:
            meta = entry.metadata
            return meta.get("entity_value", entry.content)

        return None

    # ==================== Fact Operations ====================

    def store_fact(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Store a persistent fact.

        Args:
            key: Fact key.
            content: Fact content.
            metadata: Additional metadata.

        Returns:
            The stored entry.
        """
        meta = metadata or {}
        meta["key"] = key
        return self.ltm.store(content, MemoryType.FACT, meta, f"fact_{key}")

    def get_fact(self, key: str) -> Optional[str]:
        """Get a fact by key.

        Args:
            key: Fact key.

        Returns:
            Fact content or None.
        """
        entry = self.ltm.get(f"fact_{key}")
        return entry.content if entry else None

    # ==================== Preference Operations ====================

    def store_preference(self, key: str, value: str) -> MemoryEntry:
        """Store a user preference.

        Args:
            key: Preference key.
            value: Preference value.

        Returns:
            The stored entry.
        """
        return self.ltm.store(
            f"{key}: {value}",
            MemoryType.PREFERENCE,
            {"key": key, "value": value},
            f"pref_{key}",
        )

    def get_preference(self, key: str) -> Optional[str]:
        """Get a preference.

        Args:
            key: Preference key.

        Returns:
            Preference value or None.
        """
        entry = self.ltm.get(f"pref_{key}")
        if entry:
            return entry.metadata.get("value", entry.content)
        return None

    # ==================== Search Operations ====================

    def search(
        self,
        query: str,
        k: int = 5,
        sources: Optional[List[str]] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> List[SearchResult]:
        """Search across all memory.

        Args:
            query: Search query.
            k: Number of results.
            sources: Filter by sources ("stm", "ltm").
            memory_type: Filter by memory type.

        Returns:
            Combined search results.
        """
        sources = sources or ["stm", "ltm"]
        results = []

        # Search STM
        if "stm" in sources:
            stm_results = self.stm.search(query, k)
            results.extend(stm_results)

        # Search LTM
        if "ltm" in sources:
            ltm_results = self.ltm.retrieve(query, k, memory_type)
            results.extend(ltm_results)

        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.entry.id not in seen:
                seen.add(r.entry.id)
                unique_results.append(r)

        return unique_results[:k]

    def get_context_for_prompt(self, max_tokens: int = 2000) -> str:
        """Get formatted context for prompts.

        Args:
            max_tokens: Approximate max tokens.

        Returns:
            Formatted context string.
        """
        return self.stm.get_conversation_context(max_tokens)

    # ==================== Memory Management ====================

    def promote_to_ltm(self, key: str) -> Optional[MemoryEntry]:
        """Promote a STM context entry to LTM.

        Args:
            key: Context key in STM.

        Returns:
            The LTM entry or None.
        """
        entry = self.stm.get_context(key)
        if not entry:
            return None

        return self.ltm.store(
            entry.content,
            entry.type,
            entry.metadata,
        )

    def clear_stm(self) -> None:
        """Clear short-term memory."""
        self.stm.clear()

    def clear_ltm(self) -> None:
        """Clear long-term memory."""
        self.ltm.clear()

    def clear_all(self) -> None:
        """Clear all memory."""
        self.stm.clear()
        self.ltm.clear()

    def cleanup(self) -> Dict[str, int]:
        """Cleanup expired entries.

        Returns:
            Cleanup statistics.
        """
        expired = self.ltm.cleanup_expired()
        return {"ltm_expired": expired}

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "stm": self.stm.get_stats(),
            "ltm": self.ltm.get_stats(),
        }
