"""Redis messaging backend for TinyLLM.

Provides a Redis-based implementation of the MessageQueue interface with support for:
- Publish/subscribe messaging patterns
- Connection pooling for high performance
- Support for both single Redis instance and Redis Cluster
- Message expiration via TTL
- Serialization/deserialization of message payloads
- Message acknowledgement and pending message tracking
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

import redis.asyncio as aioredis
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from tinyllm.logging import get_logger
from tinyllm.persistence.interface import (
    MessageQueue,
    MessageRecord,
    StorageConfig,
)

logger = get_logger(__name__, component="redis_backend")


class RedisMessageQueue(MessageQueue):
    """Redis-based message queue for inter-agent communication.

    This implementation uses Redis Streams for reliable message delivery,
    pub/sub for real-time notifications, and sorted sets for pending message tracking.

    Features:
    - Connection pooling for efficient resource usage
    - Support for single Redis instance and Redis Cluster
    - Automatic message expiration via TTL
    - JSON serialization of message payloads
    - Priority-based message delivery
    - Acknowledgement tracking for reliable delivery

    Architecture:
    - Redis Streams: Primary storage for messages (stream key: {prefix}:stream:{channel})
    - Redis Pub/Sub: Real-time notifications (channel: {prefix}:notify:{channel})
    - Redis Sorted Sets: Pending messages index (key: {prefix}:pending:{agent_id})
    - Redis Hashes: Message metadata (key: {prefix}:msg:{message_id})
    """

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._redis: Optional[Redis | RedisCluster] = None
        self._pool: Optional[ConnectionPool] = None
        self._pubsub_task: Optional[asyncio.Task[None]] = None
        self._subscribers: Dict[str, List[asyncio.Queue[MessageRecord]]] = {}
        self._is_cluster = False

    async def initialize(self) -> None:
        """Initialize Redis connection pool and client."""
        if self._initialized:
            return

        redis_url = self.config.redis_url or "redis://localhost:6379/0"

        try:
            # Determine if we're connecting to a cluster
            self._is_cluster = "cluster" in redis_url.lower()

            if self._is_cluster:
                # Redis Cluster setup
                logger.info("initializing_redis_cluster", url=redis_url)
                self._redis = RedisCluster.from_url(
                    redis_url,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
            else:
                # Single Redis instance with connection pooling
                logger.info("initializing_redis_single", url=redis_url)
                self._pool = ConnectionPool.from_url(
                    redis_url,
                    decode_responses=False,
                    max_connections=50,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30,
                )
                self._redis = Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()
            self._initialized = True
            logger.info("redis_initialized", is_cluster=self._is_cluster)

        except RedisError as e:
            logger.error("redis_initialization_failed", error=str(e), url=redis_url)
            raise

    async def close(self) -> None:
        """Close Redis connections and cleanup resources."""
        if not self._initialized:
            return

        logger.info("closing_redis_connection")

        # Cancel pubsub task if running
        if self._pubsub_task and not self._pubsub_task.done():
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass

        # Close Redis connection
        if self._redis:
            await self._redis.aclose()
            self._redis = None

        # Close connection pool
        if self._pool:
            await self._pool.aclose()
            self._pool = None

        self._subscribers.clear()
        self._initialized = False
        logger.info("redis_connection_closed")

    def _get_stream_key(self, channel: str) -> str:
        """Get Redis stream key for a channel."""
        return f"{self.config.redis_prefix}:stream:{channel}"

    def _get_notify_key(self, channel: str) -> str:
        """Get Redis pub/sub notification key for a channel."""
        return f"{self.config.redis_prefix}:notify:{channel}"

    def _get_pending_key(self, agent_id: str) -> str:
        """Get Redis sorted set key for pending messages."""
        return f"{self.config.redis_prefix}:pending:{agent_id}"

    def _get_message_key(self, message_id: str) -> str:
        """Get Redis hash key for message metadata."""
        return f"{self.config.redis_prefix}:msg:{message_id}"

    def _serialize_payload(self, payload: Dict[str, Any]) -> str:
        """Serialize message payload to JSON."""
        return json.dumps(payload, default=str)

    def _deserialize_payload(self, data: str) -> Dict[str, Any]:
        """Deserialize message payload from JSON."""
        return json.loads(data)

    def _message_to_dict(self, message: MessageRecord) -> Dict[str, str]:
        """Convert MessageRecord to Redis hash format."""
        return {
            "id": message.id,
            "source_agent": message.source_agent,
            "target_agent": message.target_agent or "",
            "channel": message.channel,
            "payload": self._serialize_payload(message.payload),
            "priority": str(message.priority),
            "acknowledged": str(message.acknowledged),
            "created_at": message.created_at.isoformat(),
            "expires_at": message.expires_at.isoformat() if message.expires_at else "",
        }

    def _dict_to_message(self, data: Dict[Any, Any]) -> MessageRecord:
        """Convert Redis hash to MessageRecord."""
        # Handle both bytes and str keys/values from Redis
        def get_str(key: str) -> str:
            value = data.get(key) or data.get(key.encode("utf-8"), b"")
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

        expires_at_str = get_str("expires_at")
        target_agent_str = get_str("target_agent")

        return MessageRecord(
            id=get_str("id"),
            source_agent=get_str("source_agent"),
            target_agent=target_agent_str if target_agent_str else None,
            channel=get_str("channel"),
            payload=self._deserialize_payload(get_str("payload")),
            priority=int(get_str("priority")),
            acknowledged=get_str("acknowledged").lower() == "true",
            created_at=datetime.fromisoformat(get_str("created_at")),
            expires_at=datetime.fromisoformat(expires_at_str) if expires_at_str else None,
        )

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
            payload: Message payload (must be JSON-serializable).
            source_agent: Sending agent ID.
            target_agent: Optional target agent (None = broadcast).
            priority: Message priority (higher = more urgent).
            ttl_seconds: Time-to-live in seconds (overrides config default).

        Returns:
            The created message record.

        Raises:
            RedisError: If Redis operation fails.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        # Create message record
        message_id = str(uuid4())
        ttl = ttl_seconds or self.config.ttl_seconds
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None

        message = MessageRecord(
            id=message_id,
            source_agent=source_agent,
            target_agent=target_agent,
            channel=channel,
            payload=payload,
            priority=priority,
            acknowledged=False,
            expires_at=expires_at,
        )

        try:
            # Store message metadata in hash
            msg_key = self._get_message_key(message_id)
            msg_data = self._message_to_dict(message)
            await self._redis.hset(msg_key, mapping=msg_data)

            # Set TTL on message hash if specified
            if ttl:
                await self._redis.expire(msg_key, ttl)

            # Add to stream for persistence and ordering
            stream_key = self._get_stream_key(channel)
            stream_data = {
                "message_id": message_id,
                "priority": str(priority),
            }
            await self._redis.xadd(stream_key, stream_data, maxlen=self.config.max_items)  # type: ignore[arg-type]

            # Add to pending set for target agent or all subscribers
            if target_agent:
                pending_key = self._get_pending_key(target_agent)
                # Use negative priority for sorting (higher priority first)
                await self._redis.zadd(pending_key, {message_id: -priority})  # type: ignore[arg-type]
                if ttl:
                    await self._redis.expire(pending_key, ttl)
            else:
                # Broadcast: notify all subscribers via pub/sub
                notify_key = self._get_notify_key(channel)
                await self._redis.publish(notify_key, message_id)  # type: ignore[arg-type]

            logger.debug(
                "message_published",
                message_id=message_id,
                channel=channel,
                source=source_agent,
                target=target_agent,
                priority=priority,
            )

            return message

        except RedisError as e:
            logger.error(
                "message_publish_failed",
                error=str(e),
                channel=channel,
                source=source_agent,
            )
            raise

    async def subscribe(
        self,
        channel: str,
        agent_id: str,
    ) -> AsyncIterator[MessageRecord]:
        """Subscribe to messages on a channel.

        This method yields messages as they arrive. It combines:
        1. Pending messages from the agent's queue
        2. Real-time messages via Redis pub/sub

        Args:
            channel: Channel name.
            agent_id: Subscribing agent ID.

        Yields:
            Message records as they arrive.

        Raises:
            RedisError: If Redis operation fails.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        logger.info("subscribing_to_channel", channel=channel, agent_id=agent_id)

        # Create a queue for this subscription
        queue: asyncio.Queue[MessageRecord] = asyncio.Queue()
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        self._subscribers[channel].append(queue)

        try:
            # First, yield any pending messages
            pending = await self.get_pending(agent_id, channel)
            for msg in pending:
                yield msg

            # Set up pub/sub for real-time messages
            pubsub = self._redis.pubsub()
            notify_key = self._get_notify_key(channel)
            await pubsub.subscribe(notify_key)

            try:
                # Listen for new messages
                async for redis_msg in pubsub.listen():
                    if redis_msg["type"] == "message":
                        message_id = redis_msg["data"].decode("utf-8")
                        msg_key = self._get_message_key(message_id)

                        # Fetch message data
                        msg_data = await self._redis.hgetall(msg_key)
                        if msg_data:
                            message = self._dict_to_message(msg_data)

                            # Check if message is for this agent (or broadcast)
                            if message.target_agent is None or message.target_agent == agent_id:
                                # Check expiration
                                if (
                                    message.expires_at is None
                                    or message.expires_at > datetime.utcnow()
                                ):
                                    yield message

            finally:
                await pubsub.unsubscribe(notify_key)
                await pubsub.aclose()

        finally:
            # Clean up subscription
            if channel in self._subscribers:
                self._subscribers[channel].remove(queue)
                if not self._subscribers[channel]:
                    del self._subscribers[channel]

            logger.info("unsubscribed_from_channel", channel=channel, agent_id=agent_id)

    async def acknowledge(self, message_id: str) -> bool:
        """Acknowledge receipt of a message.

        Args:
            message_id: Message identifier.

        Returns:
            True if acknowledged, False if not found.

        Raises:
            RedisError: If Redis operation fails.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            msg_key = self._get_message_key(message_id)

            # Check if message exists
            exists = await self._redis.exists(msg_key)
            if not exists:
                logger.warning("message_not_found", message_id=message_id)
                return False

            # Mark as acknowledged
            await self._redis.hset(msg_key, "acknowledged", "true")  # type: ignore[arg-type]

            # Get message data to remove from pending set
            msg_data = await self._redis.hgetall(msg_key)
            if msg_data:
                # Handle both bytes and str from Redis
                target_key = msg_data.get("target_agent") or msg_data.get(b"target_agent")  # type: ignore[call-overload]
                if target_key:
                    target_agent = target_key.decode("utf-8") if isinstance(target_key, bytes) else str(target_key)
                    pending_key = self._get_pending_key(target_agent)
                    await self._redis.zrem(pending_key, message_id)

            logger.debug("message_acknowledged", message_id=message_id)
            return True

        except RedisError as e:
            logger.error("message_acknowledge_failed", error=str(e), message_id=message_id)
            raise

    async def get_pending(
        self,
        agent_id: str,
        channel: Optional[str] = None,
        limit: int = 100,
    ) -> List[MessageRecord]:
        """Get pending (unacknowledged) messages for an agent.

        Messages are returned in priority order (highest priority first),
        then by creation time (oldest first).

        Args:
            agent_id: Agent identifier.
            channel: Optional channel filter.
            limit: Maximum messages to return.

        Returns:
            List of unacknowledged messages.

        Raises:
            RedisError: If Redis operation fails.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            pending_key = self._get_pending_key(agent_id)

            # Get message IDs from sorted set (by priority)
            # Using ZRANGE with negative scores (higher priority first)
            message_ids = await self._redis.zrange(pending_key, 0, limit - 1)

            messages: List[MessageRecord] = []
            for msg_id_item in message_ids:
                message_id = msg_id_item.decode("utf-8") if isinstance(msg_id_item, bytes) else str(msg_id_item)
                msg_key = self._get_message_key(message_id)

                # Fetch message data
                msg_data = await self._redis.hgetall(msg_key)
                if not msg_data:
                    # Message expired or deleted, remove from pending
                    await self._redis.zrem(pending_key, message_id)
                    continue

                message = self._dict_to_message(msg_data)

                # Check expiration
                if message.expires_at and message.expires_at <= datetime.utcnow():
                    # Message expired, remove it
                    await self._redis.delete(msg_key)
                    await self._redis.zrem(pending_key, message_id)
                    continue

                # Filter by channel if specified
                if channel and message.channel != channel:
                    continue

                # Skip already acknowledged messages
                if message.acknowledged:
                    await self._redis.zrem(pending_key, message_id)
                    continue

                messages.append(message)

            logger.debug(
                "pending_messages_retrieved",
                agent_id=agent_id,
                channel=channel,
                count=len(messages),
            )

            return messages

        except RedisError as e:
            logger.error("get_pending_failed", error=str(e), agent_id=agent_id)
            raise

    # StorageBackend interface implementation

    async def put(self, item: MessageRecord) -> None:
        """Store a message record.

        Args:
            item: The message record to store.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            msg_key = self._get_message_key(item.id)
            msg_data = self._message_to_dict(item)
            await self._redis.hset(msg_key, mapping=msg_data)

            # Set TTL if message has expiration
            if item.expires_at:
                ttl_seconds = int((item.expires_at - datetime.utcnow()).total_seconds())
                if ttl_seconds > 0:
                    await self._redis.expire(msg_key, ttl_seconds)

        except RedisError as e:
            logger.error("put_message_failed", error=str(e), message_id=item.id)
            raise

    async def get(self, item_id: str) -> Optional[MessageRecord]:
        """Retrieve a message by ID.

        Args:
            item_id: The message's unique identifier.

        Returns:
            The message if found, None otherwise.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            msg_key = self._get_message_key(item_id)
            msg_data = await self._redis.hgetall(msg_key)

            if not msg_data:
                return None

            message = self._dict_to_message(msg_data)

            # Check expiration
            if message.expires_at and message.expires_at <= datetime.utcnow():
                await self._redis.delete(msg_key)
                return None

            return message

        except RedisError as e:
            logger.error("get_message_failed", error=str(e), message_id=item_id)
            raise

    async def delete(self, item_id: str) -> bool:
        """Delete a message by ID.

        Args:
            item_id: The message's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            msg_key = self._get_message_key(item_id)

            # Get message data to remove from pending set
            msg_data = await self._redis.hgetall(msg_key)
            if msg_data:
                # Handle both bytes and str from Redis
                target_key = msg_data.get("target_agent") or msg_data.get(b"target_agent")  # type: ignore[call-overload]
                if target_key:
                    target_agent = target_key.decode("utf-8") if isinstance(target_key, bytes) else str(target_key)
                    pending_key = self._get_pending_key(target_agent)
                    await self._redis.zrem(pending_key, item_id)

            # Delete message hash
            result = await self._redis.delete(msg_key)
            return bool(result > 0)

        except RedisError as e:
            logger.error("delete_message_failed", error=str(e), message_id=item_id)
            raise

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MessageRecord]:
        """List messages with optional filtering.

        Note: This is a relatively expensive operation in Redis as it requires
        scanning keys. Use get_pending() for better performance when possible.

        Args:
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            filters: Key-value pairs to filter by (channel, source_agent, target_agent).

        Returns:
            List of matching messages.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            # Scan for message keys
            pattern = f"{self.config.redis_prefix}:msg:*"
            messages: List[MessageRecord] = []
            cursor = 0

            while True:
                cursor, keys = await self._redis.scan(
                    cursor, match=pattern, count=100
                )

                for key in keys:
                    msg_data = await self._redis.hgetall(key)
                    if not msg_data:
                        continue

                    message = self._dict_to_message(msg_data)

                    # Check expiration
                    if message.expires_at and message.expires_at <= datetime.utcnow():
                        continue

                    # Apply filters
                    if filters:
                        if "channel" in filters and message.channel != filters["channel"]:
                            continue
                        if (
                            "source_agent" in filters
                            and message.source_agent != filters["source_agent"]
                        ):
                            continue
                        if (
                            "target_agent" in filters
                            and message.target_agent != filters["target_agent"]
                        ):
                            continue
                        if "acknowledged" in filters and message.acknowledged != filters[
                            "acknowledged"
                        ]:
                            continue

                    messages.append(message)

                if cursor == 0:
                    break

            # Sort by creation time and apply offset/limit
            messages.sort(key=lambda x: x.created_at, reverse=True)
            return messages[offset : offset + limit]

        except RedisError as e:
            logger.error("list_messages_failed", error=str(e))
            raise

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count messages with optional filtering.

        Args:
            filters: Key-value pairs to filter by.

        Returns:
            Number of matching messages.
        """
        messages = await self.list(limit=100000, filters=filters)
        return len(messages)

    async def clear(self) -> int:
        """Clear all messages.

        Returns:
            Number of messages deleted.
        """
        if not self._initialized or not self._redis:
            raise RuntimeError("Redis backend not initialized")

        try:
            # Delete all message keys
            pattern = f"{self.config.redis_prefix}:msg:*"
            count = 0
            cursor = 0

            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break

            # Clear all pending sets
            pattern = f"{self.config.redis_prefix}:pending:*"
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

            # Clear all streams
            pattern = f"{self.config.redis_prefix}:stream:*"
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

            logger.info("messages_cleared", count=count)
            return count

        except RedisError as e:
            logger.error("clear_messages_failed", error=str(e))
            raise
