"""Tests for memory system."""

import pytest
import tempfile
from pathlib import Path

from tinyllm.memory.models import (
    ConversationMessage,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
)
from tinyllm.memory.stm import STM
from tinyllm.memory.ltm import LTM, VectorStore
from tinyllm.memory.store import MemoryStore


class TestMemoryEntry:
    """Tests for MemoryEntry model."""

    def test_create_entry(self):
        """Should create a memory entry."""
        entry = MemoryEntry(
            id="test_1",
            type=MemoryType.FACT,
            content="The sky is blue",
        )

        assert entry.id == "test_1"
        assert entry.type == MemoryType.FACT
        assert entry.content == "The sky is blue"
        assert entry.confidence == 1.0
        assert entry.access_count == 0

    def test_touch_updates_access(self):
        """Should update access time and count."""
        entry = MemoryEntry(
            id="test_1",
            type=MemoryType.FACT,
            content="Test",
        )

        original_time = entry.accessed_at
        entry.touch()

        assert entry.access_count == 1
        assert entry.accessed_at >= original_time

    def test_expiration(self):
        """Should detect expired entries."""
        entry = MemoryEntry(
            id="test_1",
            type=MemoryType.FACT,
            content="Test",
            ttl_seconds=0,  # Immediately expired
        )

        assert entry.is_expired() is True

    def test_no_expiration(self):
        """Should not expire without TTL."""
        entry = MemoryEntry(
            id="test_1",
            type=MemoryType.FACT,
            content="Test",
        )

        assert entry.is_expired() is False


class TestConversationMessage:
    """Tests for ConversationMessage model."""

    def test_create_message(self):
        """Should create a conversation message."""
        msg = ConversationMessage(
            role="user",
            content="Hello!",
        )

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.node_id is None


class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = MemoryConfig()

        assert config.stm_max_messages == 20
        assert config.ltm_search_k == 5
        assert config.embedding_dim == 768


class TestSTM:
    """Tests for Short-Term Memory."""

    def test_add_message(self):
        """Should add messages."""
        stm = STM()

        msg = stm.add_message("user", "Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert stm.message_count == 1

    def test_add_multiple_messages(self):
        """Should maintain message history."""
        stm = STM()

        stm.add_message("user", "Hello!")
        stm.add_message("assistant", "Hi there!")
        stm.add_message("user", "How are you?")

        assert stm.message_count == 3

    def test_get_recent_messages(self):
        """Should get recent messages."""
        stm = STM()

        for i in range(5):
            stm.add_message("user", f"Message {i}")

        recent = stm.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"

    def test_get_messages_by_role(self):
        """Should filter by role."""
        stm = STM()

        stm.add_message("user", "User 1")
        stm.add_message("assistant", "Assistant 1")
        stm.add_message("user", "User 2")

        user_msgs = stm.get_recent_messages(role="user")
        assert len(user_msgs) == 2
        assert all(m.role == "user" for m in user_msgs)

    def test_add_context(self):
        """Should add context entries."""
        stm = STM()

        entry = stm.add_context("user_name", "Alice")
        assert entry.content == "Alice"

        retrieved = stm.get_context("user_name")
        assert retrieved is not None
        assert retrieved.content == "Alice"

    def test_extract_entity(self):
        """Should extract and store entities."""
        stm = STM()

        stm.extract_entity("user_name", "Bob")
        assert stm.get_entity("user_name") == "Bob"

    def test_search(self):
        """Should search context."""
        stm = STM()

        stm.add_context("topic_1", "Python programming language")
        stm.add_context("topic_2", "JavaScript web development")
        stm.add_context("topic_3", "Database management")

        results = stm.search("Python", limit=2)
        assert len(results) > 0
        assert "Python" in results[0].entry.content

    def test_summarize_and_prune(self):
        """Should summarize when threshold reached."""
        config = MemoryConfig(stm_max_messages=10, stm_summarize_threshold=5)
        stm = STM(config)

        # Add more than max messages
        for i in range(15):
            stm.add_message("user", f"Message {i}")

        # Should have pruned
        assert stm.message_count < 15

    def test_clear(self):
        """Should clear all data."""
        stm = STM()

        stm.add_message("user", "Test")
        stm.add_context("key", "value")
        stm.clear()

        assert stm.message_count == 0
        assert stm.get_context("key") is None


class TestVectorStore:
    """Tests for VectorStore."""

    def test_add_and_get(self):
        """Should add and retrieve entries."""
        store = VectorStore(dim=3)

        entry = MemoryEntry(id="test", type=MemoryType.FACT, content="Test")
        store.add("test", [0.1, 0.2, 0.3], entry)

        retrieved = store.get("test")
        assert retrieved is not None
        assert retrieved.content == "Test"

    def test_search(self):
        """Should search by similarity."""
        store = VectorStore(dim=3)

        e1 = MemoryEntry(id="e1", type=MemoryType.FACT, content="Test 1")
        e2 = MemoryEntry(id="e2", type=MemoryType.FACT, content="Test 2")

        store.add("e1", [1.0, 0.0, 0.0], e1)
        store.add("e2", [0.0, 1.0, 0.0], e2)

        results = store.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        assert results[0][0] == "e1"  # Most similar

    def test_remove(self):
        """Should remove entries."""
        store = VectorStore(dim=3)

        entry = MemoryEntry(id="test", type=MemoryType.FACT, content="Test")
        store.add("test", [0.1, 0.2, 0.3], entry)

        assert store.remove("test") is True
        assert store.get("test") is None


class TestLTM:
    """Tests for Long-Term Memory."""

    def test_store_and_retrieve(self):
        """Should store and retrieve memories."""
        ltm = LTM()

        entry = ltm.store("The capital of France is Paris", MemoryType.FACT)
        assert entry.id is not None

        # With pseudo-embeddings, semantic search won't work well
        # but retrieve should return entries (with low threshold)
        config = MemoryConfig(ltm_similarity_threshold=0.0)
        ltm_low_threshold = LTM(config)
        ltm_low_threshold.store("The capital of France is Paris", MemoryType.FACT)
        results = ltm_low_threshold.retrieve("capital France")
        assert len(results) > 0

    def test_get_by_id(self):
        """Should get entry by ID."""
        ltm = LTM()

        entry = ltm.store("Test content", entry_id="custom_id")
        retrieved = ltm.get("custom_id")

        assert retrieved is not None
        assert retrieved.content == "Test content"

    def test_delete(self):
        """Should delete entries."""
        ltm = LTM()

        ltm.store("Test", entry_id="to_delete")
        assert ltm.delete("to_delete") is True
        assert ltm.get("to_delete") is None

    def test_get_all(self):
        """Should get all entries."""
        ltm = LTM()

        ltm.store("Fact 1", MemoryType.FACT)
        ltm.store("Fact 2", MemoryType.FACT)
        ltm.store("Preference 1", MemoryType.PREFERENCE)

        all_entries = ltm.get_all()
        assert len(all_entries) == 3

        facts_only = ltm.get_all(MemoryType.FACT)
        assert len(facts_only) == 2

    def test_persistence(self):
        """Should persist to and load from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ltm.json"
            config = MemoryConfig(persist_path=str(path), auto_persist=True)

            # Create and populate LTM
            ltm1 = LTM(config)
            ltm1.store("Persistent fact", MemoryType.FACT, entry_id="fact_1")

            # Create new LTM with same path
            ltm2 = LTM(config)

            # Should have loaded the entry
            entry = ltm2.get("fact_1")
            assert entry is not None
            assert entry.content == "Persistent fact"

    def test_clear(self):
        """Should clear all data."""
        ltm = LTM()

        ltm.store("Test 1")
        ltm.store("Test 2")
        ltm.clear()

        assert len(ltm.get_all()) == 0


class TestMemoryStore:
    """Tests for unified MemoryStore."""

    def test_add_message(self):
        """Should add conversation messages."""
        store = MemoryStore()

        msg = store.add_message("user", "Hello!")
        assert msg.content == "Hello!"

        messages = store.get_messages()
        assert len(messages) == 1

    def test_set_and_get_context(self):
        """Should manage context."""
        store = MemoryStore()

        store.set_context("topic", "Python programming")
        assert store.get_context("topic") == "Python programming"

    def test_persistent_context(self):
        """Should store persistent context in LTM."""
        store = MemoryStore()

        store.set_context("persistent_key", "Persistent value", persistent=True)

        # Should be in LTM
        ltm_stats = store.ltm.get_stats()
        assert ltm_stats["total_entries"] > 0

    def test_store_and_get_fact(self):
        """Should store and retrieve facts."""
        store = MemoryStore()

        store.store_fact("capital_france", "Paris is the capital of France")
        assert store.get_fact("capital_france") == "Paris is the capital of France"

    def test_store_and_get_preference(self):
        """Should store and retrieve preferences."""
        store = MemoryStore()

        store.store_preference("language", "Python")
        assert store.get_preference("language") == "Python"

    def test_search_across_memory(self):
        """Should search both STM and LTM."""
        store = MemoryStore()

        # Add to STM
        store.set_context("stm_topic", "Machine learning algorithms")

        # Add to LTM
        store.store_fact("ltm_topic", "Deep learning neural networks")

        # Search should find both
        results = store.search("learning", k=5)
        assert len(results) >= 1

    def test_search_by_source(self):
        """Should filter search by source."""
        store = MemoryStore()

        store.set_context("stm_only", "STM content")
        store.store_fact("ltm_only", "LTM content")

        stm_results = store.search("content", sources=["stm"])
        ltm_results = store.search("content", sources=["ltm"])

        assert all(r.source == "stm" for r in stm_results)
        assert all(r.source == "ltm" for r in ltm_results)

    def test_entity_extraction(self):
        """Should extract and retrieve entities."""
        store = MemoryStore()

        store.extract_entity("user_name", "Alice")
        assert store.get_entity("user_name") == "Alice"

    def test_persistent_entity(self):
        """Should persist entities to LTM."""
        store = MemoryStore()

        store.extract_entity("persistent_entity", "Value", persistent=True)

        # Should be in LTM
        entry = store.ltm.get("entity_persistent_entity")
        assert entry is not None

    def test_promote_to_ltm(self):
        """Should promote STM context to LTM."""
        store = MemoryStore()

        store.set_context("to_promote", "Important content")
        promoted = store.promote_to_ltm("to_promote")

        assert promoted is not None
        assert store.ltm.get(promoted.id) is not None

    def test_clear_operations(self):
        """Should clear STM, LTM, or both."""
        store = MemoryStore()

        store.add_message("user", "Test")
        store.store_fact("test", "Test fact")

        store.clear_stm()
        assert store.stm.message_count == 0
        assert store.ltm.get_stats()["total_entries"] > 0

        store.clear_ltm()
        assert store.ltm.get_stats()["total_entries"] == 0

    def test_get_stats(self):
        """Should return combined stats."""
        store = MemoryStore()

        store.add_message("user", "Test")
        store.store_fact("test", "Test fact")

        stats = store.get_stats()
        assert "stm" in stats
        assert "ltm" in stats
        assert stats["stm"]["message_count"] == 1
        assert stats["ltm"]["total_entries"] == 1
