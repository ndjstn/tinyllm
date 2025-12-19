"""Messaging tools for TinyLLM (Slack/Discord integration).

This module provides tools for sending messages to Slack and Discord
via webhooks and APIs.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class MessagingPlatform(str, Enum):
    """Supported messaging platforms."""

    SLACK = "slack"
    DISCORD = "discord"


class MessageType(str, Enum):
    """Types of messages."""

    TEXT = "text"
    RICH = "rich"
    EMBED = "embed"


@dataclass
class MessageEmbed:
    """An embedded message (rich content)."""

    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    color: Optional[int] = None  # Hex color as integer
    footer: Optional[str] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    author_name: Optional[str] = None
    author_url: Optional[str] = None
    author_icon_url: Optional[str] = None
    fields: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: Optional[str] = None

    def to_slack_attachment(self) -> Dict[str, Any]:
        """Convert to Slack attachment format."""
        attachment = {}

        if self.title:
            attachment["title"] = self.title
        if self.description:
            attachment["text"] = self.description
        if self.url:
            attachment["title_link"] = self.url
        if self.color:
            attachment["color"] = f"#{self.color:06x}"
        if self.footer:
            attachment["footer"] = self.footer
        if self.image_url:
            attachment["image_url"] = self.image_url
        if self.thumbnail_url:
            attachment["thumb_url"] = self.thumbnail_url
        if self.author_name:
            attachment["author_name"] = self.author_name
        if self.author_url:
            attachment["author_link"] = self.author_url
        if self.author_icon_url:
            attachment["author_icon"] = self.author_icon_url
        if self.timestamp:
            attachment["ts"] = self.timestamp

        if self.fields:
            attachment["fields"] = [
                {"title": f.get("name", ""), "value": f.get("value", ""), "short": f.get("inline", False)}
                for f in self.fields
            ]

        return attachment

    def to_discord_embed(self) -> Dict[str, Any]:
        """Convert to Discord embed format."""
        embed = {}

        if self.title:
            embed["title"] = self.title
        if self.description:
            embed["description"] = self.description
        if self.url:
            embed["url"] = self.url
        if self.color:
            embed["color"] = self.color
        if self.footer:
            embed["footer"] = {"text": self.footer}
        if self.image_url:
            embed["image"] = {"url": self.image_url}
        if self.thumbnail_url:
            embed["thumbnail"] = {"url": self.thumbnail_url}
        if self.timestamp:
            embed["timestamp"] = self.timestamp

        if self.author_name:
            author = {"name": self.author_name}
            if self.author_url:
                author["url"] = self.author_url
            if self.author_icon_url:
                author["icon_url"] = self.author_icon_url
            embed["author"] = author

        if self.fields:
            embed["fields"] = [
                {"name": f.get("name", "Field"), "value": f.get("value", ""), "inline": f.get("inline", False)}
                for f in self.fields
            ]

        return embed


@dataclass
class WebhookConfig:
    """Webhook configuration."""

    url: str
    platform: MessagingPlatform
    name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    default_channel: Optional[str] = None


@dataclass
class SendResult:
    """Result of sending a message."""

    success: bool
    message_id: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None
    platform: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "error": self.error,
            "platform": self.platform,
        }


class WebhookSender:
    """Sends messages via webhooks."""

    def __init__(self, config: WebhookConfig):
        """Initialize sender.

        Args:
            config: Webhook configuration.
        """
        self.config = config

    def _send_request(self, payload: Dict[str, Any]) -> SendResult:
        """Send HTTP request to webhook.

        Args:
            payload: Request payload.

        Returns:
            Send result.
        """
        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}

            req = urllib.request.Request(
                self.config.url,
                data=data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                response_body = response.read().decode("utf-8")

                # Parse response for message ID if available
                message_id = None
                timestamp = None

                if response_body:
                    try:
                        resp_data = json.loads(response_body)
                        message_id = resp_data.get("id") or resp_data.get("ts")
                        timestamp = resp_data.get("ts")
                    except json.JSONDecodeError:
                        pass

                return SendResult(
                    success=True,
                    message_id=message_id,
                    timestamp=timestamp,
                    platform=self.config.platform.value,
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            return SendResult(
                success=False,
                error=f"HTTP {e.code}: {error_body or e.reason}",
                platform=self.config.platform.value,
            )
        except urllib.error.URLError as e:
            return SendResult(
                success=False,
                error=f"Connection error: {e.reason}",
                platform=self.config.platform.value,
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                platform=self.config.platform.value,
            )

    def send_text(
        self,
        text: str,
        channel: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Send a text message.

        Args:
            text: Message text.
            channel: Channel name or ID.
            thread_id: Thread ID for replies.

        Returns:
            Send result.
        """
        if self.config.platform == MessagingPlatform.SLACK:
            return self._send_slack_text(text, channel, thread_id)
        else:
            return self._send_discord_text(text)

    def _send_slack_text(
        self,
        text: str,
        channel: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Send Slack text message."""
        payload: Dict[str, Any] = {"text": text}

        if channel:
            payload["channel"] = channel
        elif self.config.default_channel:
            payload["channel"] = self.config.default_channel

        if self.config.username:
            payload["username"] = self.config.username

        if self.config.avatar_url:
            payload["icon_url"] = self.config.avatar_url

        if thread_id:
            payload["thread_ts"] = thread_id

        return self._send_request(payload)

    def _send_discord_text(self, text: str) -> SendResult:
        """Send Discord text message."""
        payload: Dict[str, Any] = {"content": text}

        if self.config.username:
            payload["username"] = self.config.username

        if self.config.avatar_url:
            payload["avatar_url"] = self.config.avatar_url

        return self._send_request(payload)

    def send_embed(
        self,
        embed: MessageEmbed,
        text: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> SendResult:
        """Send an embedded message.

        Args:
            embed: Message embed.
            text: Optional text with embed.
            channel: Channel (Slack only).

        Returns:
            Send result.
        """
        if self.config.platform == MessagingPlatform.SLACK:
            return self._send_slack_embed(embed, text, channel)
        else:
            return self._send_discord_embed(embed, text)

    def _send_slack_embed(
        self,
        embed: MessageEmbed,
        text: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> SendResult:
        """Send Slack embed as attachment."""
        payload: Dict[str, Any] = {
            "attachments": [embed.to_slack_attachment()],
        }

        if text:
            payload["text"] = text

        if channel:
            payload["channel"] = channel
        elif self.config.default_channel:
            payload["channel"] = self.config.default_channel

        if self.config.username:
            payload["username"] = self.config.username

        if self.config.avatar_url:
            payload["icon_url"] = self.config.avatar_url

        return self._send_request(payload)

    def _send_discord_embed(
        self,
        embed: MessageEmbed,
        text: Optional[str] = None,
    ) -> SendResult:
        """Send Discord embed."""
        payload: Dict[str, Any] = {
            "embeds": [embed.to_discord_embed()],
        }

        if text:
            payload["content"] = text

        if self.config.username:
            payload["username"] = self.config.username

        if self.config.avatar_url:
            payload["avatar_url"] = self.config.avatar_url

        return self._send_request(payload)


class MessagingManager:
    """Manager for messaging webhooks."""

    def __init__(self):
        """Initialize manager."""
        self._webhooks: Dict[str, WebhookSender] = {}

    def add_webhook(self, name: str, sender: WebhookSender) -> None:
        """Add a webhook sender.

        Args:
            name: Webhook name.
            sender: Webhook sender.
        """
        self._webhooks[name] = sender

    def get_webhook(self, name: str) -> Optional[WebhookSender]:
        """Get a webhook sender.

        Args:
            name: Webhook name.

        Returns:
            Webhook sender or None.
        """
        return self._webhooks.get(name)

    def remove_webhook(self, name: str) -> bool:
        """Remove a webhook.

        Args:
            name: Webhook name.

        Returns:
            True if removed.
        """
        if name in self._webhooks:
            del self._webhooks[name]
            return True
        return False

    def list_webhooks(self) -> List[str]:
        """List all webhook names."""
        return list(self._webhooks.keys())


# Pydantic models for tool inputs/outputs


class SendMessageInput(BaseModel):
    """Input for sending a message."""

    webhook: str = Field(
        default="default",
        description="Name of the webhook to use",
    )
    text: str = Field(
        ...,
        description="Message text to send",
    )
    channel: Optional[str] = Field(
        default=None,
        description="Channel name or ID (Slack only)",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for replies (Slack only)",
    )


class SendMessageOutput(BaseModel):
    """Output from sending a message."""

    success: bool = Field(description="Whether message was sent")
    message_id: Optional[str] = Field(
        default=None,
        description="Message ID if available",
    )
    platform: Optional[str] = Field(
        default=None,
        description="Platform the message was sent to",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )


class SendEmbedInput(BaseModel):
    """Input for sending an embedded message."""

    webhook: str = Field(
        default="default",
        description="Name of the webhook to use",
    )
    title: Optional[str] = Field(
        default=None,
        description="Embed title",
    )
    description: Optional[str] = Field(
        default=None,
        description="Embed description/body",
    )
    url: Optional[str] = Field(
        default=None,
        description="URL for the title link",
    )
    color: Optional[str] = Field(
        default=None,
        description="Hex color (e.g., '#FF0000')",
    )
    footer: Optional[str] = Field(
        default=None,
        description="Footer text",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Image URL",
    )
    text: Optional[str] = Field(
        default=None,
        description="Text to send with embed",
    )
    channel: Optional[str] = Field(
        default=None,
        description="Channel (Slack only)",
    )
    fields: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Embed fields [{name, value, inline}]",
    )


class CreateWebhookInput(BaseModel):
    """Input for creating a webhook."""

    name: str = Field(
        ...,
        description="Name for the webhook",
    )
    url: str = Field(
        ...,
        description="Webhook URL",
    )
    platform: str = Field(
        ...,
        description="Platform (slack or discord)",
    )
    username: Optional[str] = Field(
        default=None,
        description="Username override for messages",
    )
    avatar_url: Optional[str] = Field(
        default=None,
        description="Avatar URL override",
    )
    default_channel: Optional[str] = Field(
        default=None,
        description="Default channel (Slack only)",
    )


class CreateWebhookOutput(BaseModel):
    """Output from creating a webhook."""

    success: bool = Field(description="Whether webhook was created")
    name: str = Field(description="Webhook name")
    platform: str = Field(description="Platform")
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )


class ListWebhooksOutput(BaseModel):
    """Output from listing webhooks."""

    webhooks: List[str] = Field(
        default_factory=list,
        description="List of webhook names",
    )


# Tool implementations


class SendMessageTool(BaseTool[SendMessageInput, SendMessageOutput]):
    """Tool for sending text messages."""

    metadata = ToolMetadata(
        id="send_message",
        name="Send Message",
        description="Send a text message to Slack or Discord",
        category="utility",
    )
    input_type = SendMessageInput
    output_type = SendMessageOutput

    def __init__(self, manager: MessagingManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SendMessageInput) -> SendMessageOutput:
        """Send a message."""
        webhook = self.manager.get_webhook(input.webhook)

        if not webhook:
            return SendMessageOutput(
                success=False,
                error=f"Webhook '{input.webhook}' not found",
            )

        result = webhook.send_text(
            text=input.text,
            channel=input.channel,
            thread_id=input.thread_id,
        )

        return SendMessageOutput(
            success=result.success,
            message_id=result.message_id,
            platform=result.platform,
            error=result.error,
        )


class SendEmbedTool(BaseTool[SendEmbedInput, SendMessageOutput]):
    """Tool for sending embedded messages."""

    metadata = ToolMetadata(
        id="send_embed",
        name="Send Embed",
        description="Send a rich embedded message to Slack or Discord",
        category="utility",
    )
    input_type = SendEmbedInput
    output_type = SendMessageOutput

    def __init__(self, manager: MessagingManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SendEmbedInput) -> SendMessageOutput:
        """Send an embedded message."""
        webhook = self.manager.get_webhook(input.webhook)

        if not webhook:
            return SendMessageOutput(
                success=False,
                error=f"Webhook '{input.webhook}' not found",
            )

        # Parse color
        color = None
        if input.color:
            try:
                color_str = input.color.lstrip("#")
                color = int(color_str, 16)
            except ValueError:
                pass

        embed = MessageEmbed(
            title=input.title,
            description=input.description,
            url=input.url,
            color=color,
            footer=input.footer,
            image_url=input.image_url,
            fields=input.fields or [],
        )

        result = webhook.send_embed(
            embed=embed,
            text=input.text,
            channel=input.channel,
        )

        return SendMessageOutput(
            success=result.success,
            message_id=result.message_id,
            platform=result.platform,
            error=result.error,
        )


class CreateWebhookTool(BaseTool[CreateWebhookInput, CreateWebhookOutput]):
    """Tool for creating webhooks."""

    metadata = ToolMetadata(
        id="create_messaging_webhook",
        name="Create Messaging Webhook",
        description="Create a Slack or Discord webhook",
        category="utility",
    )
    input_type = CreateWebhookInput
    output_type = CreateWebhookOutput

    def __init__(self, manager: MessagingManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateWebhookInput) -> CreateWebhookOutput:
        """Create a webhook."""
        try:
            platform = MessagingPlatform(input.platform.lower())
        except ValueError:
            return CreateWebhookOutput(
                success=False,
                name=input.name,
                platform=input.platform,
                error=f"Invalid platform: {input.platform}. Use 'slack' or 'discord'",
            )

        config = WebhookConfig(
            url=input.url,
            platform=platform,
            name=input.name,
            username=input.username,
            avatar_url=input.avatar_url,
            default_channel=input.default_channel,
        )

        sender = WebhookSender(config)
        self.manager.add_webhook(input.name, sender)

        return CreateWebhookOutput(
            success=True,
            name=input.name,
            platform=platform.value,
        )


class ListWebhooksTool(BaseTool[BaseModel, ListWebhooksOutput]):
    """Tool for listing webhooks."""

    metadata = ToolMetadata(
        id="list_messaging_webhooks",
        name="List Messaging Webhooks",
        description="List all configured messaging webhooks",
        category="utility",
    )
    input_type = BaseModel
    output_type = ListWebhooksOutput

    def __init__(self, manager: MessagingManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: BaseModel) -> ListWebhooksOutput:
        """List webhooks."""
        return ListWebhooksOutput(webhooks=self.manager.list_webhooks())


# Convenience functions


def create_webhook_config(
    url: str,
    platform: MessagingPlatform,
    username: Optional[str] = None,
    avatar_url: Optional[str] = None,
    default_channel: Optional[str] = None,
) -> WebhookConfig:
    """Create a webhook configuration.

    Args:
        url: Webhook URL.
        platform: Messaging platform.
        username: Username override.
        avatar_url: Avatar URL.
        default_channel: Default channel.

    Returns:
        Webhook configuration.
    """
    return WebhookConfig(
        url=url,
        platform=platform,
        username=username,
        avatar_url=avatar_url,
        default_channel=default_channel,
    )


def create_webhook_sender(config: WebhookConfig) -> WebhookSender:
    """Create a webhook sender.

    Args:
        config: Webhook configuration.

    Returns:
        Webhook sender.
    """
    return WebhookSender(config)


def create_messaging_manager() -> MessagingManager:
    """Create a messaging manager.

    Returns:
        Messaging manager.
    """
    return MessagingManager()


def create_messaging_tools(manager: MessagingManager) -> Dict[str, BaseTool]:
    """Create all messaging tools.

    Args:
        manager: Messaging manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "send_message": SendMessageTool(manager),
        "send_embed": SendEmbedTool(manager),
        "create_messaging_webhook": CreateWebhookTool(manager),
        "list_messaging_webhooks": ListWebhooksTool(manager),
    }
