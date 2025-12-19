#!/usr/bin/env python3
"""Quick test script to verify RedisMessageQueue implementation."""

import asyncio
from tinyllm.persistence.interface import StorageConfig
from tinyllm.persistence.redis_backend import RedisMessageQueue


async def test_redis_backend():
    """Test basic RedisMessageQueue functionality."""

    # Create config
    config = StorageConfig(
        redis_url="redis://localhost:6379/0",
        redis_prefix="tinyllm_test",
        ttl_seconds=300,
    )

    # Create Redis message queue
    queue = RedisMessageQueue(config)

    try:
        # Initialize
        print("Initializing Redis backend...")
        await queue.initialize()
        print("✓ Redis backend initialized")

        # Test publish
        print("\nPublishing test message...")
        message = await queue.publish(
            channel="test_channel",
            payload={"text": "Hello, Redis!", "data": [1, 2, 3]},
            source_agent="test_agent_1",
            target_agent="test_agent_2",
            priority=10,
            ttl_seconds=300,
        )
        print(f"✓ Message published: {message.id}")

        # Test get
        print("\nRetrieving message...")
        retrieved = await queue.get(message.id)
        assert retrieved is not None
        assert retrieved.payload["text"] == "Hello, Redis!"
        print(f"✓ Message retrieved: {retrieved.payload}")

        # Test get_pending
        print("\nGetting pending messages...")
        pending = await queue.get_pending("test_agent_2", channel="test_channel")
        assert len(pending) > 0
        assert pending[0].id == message.id
        print(f"✓ Found {len(pending)} pending message(s)")

        # Test acknowledge
        print("\nAcknowledging message...")
        acked = await queue.acknowledge(message.id)
        assert acked is True
        print("✓ Message acknowledged")

        # Verify acknowledged
        pending_after = await queue.get_pending("test_agent_2", channel="test_channel")
        assert len(pending_after) == 0
        print("✓ No pending messages after acknowledgement")

        # Test broadcast
        print("\nPublishing broadcast message...")
        broadcast_msg = await queue.publish(
            channel="broadcast_channel",
            payload={"announcement": "System update"},
            source_agent="system",
            target_agent=None,  # Broadcast
            priority=5,
        )
        print(f"✓ Broadcast message published: {broadcast_msg.id}")

        # Test list
        print("\nListing messages...")
        all_messages = await queue.list(limit=10)
        print(f"✓ Found {len(all_messages)} total messages")

        # Test count
        count = await queue.count()
        print(f"✓ Total message count: {count}")

        # Test delete
        print("\nDeleting message...")
        deleted = await queue.delete(message.id)
        assert deleted is True
        print("✓ Message deleted")

        # Test clear
        print("\nClearing all messages...")
        cleared_count = await queue.clear()
        print(f"✓ Cleared {cleared_count} message(s)")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        await queue.close()
        print("\n✓ Redis backend closed")


if __name__ == "__main__":
    asyncio.run(test_redis_backend())
