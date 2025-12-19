"""Tests for messaging tools (Slack/Discord integration)."""

import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from tinyllm.tools.messaging import (
    MessagingPlatform,
    MessageType,
    MessageEmbed,
    WebhookConfig,
    SendResult,
    WebhookSender,
    MessagingManager,
    SendMessageInput,
    SendMessageOutput,
    SendEmbedInput,
    CreateWebhookInput,
    CreateWebhookOutput,
    ListWebhooksOutput,
    SendMessageTool,
    SendEmbedTool,
    CreateWebhookTool,
    ListWebhooksTool,
    create_webhook_config,
    create_webhook_sender,
    create_messaging_manager,
    create_messaging_tools,
)


# ============================================================================
# MessageEmbed Tests
# ============================================================================


class TestMessageEmbed:
    """Tests for MessageEmbed dataclass."""

    def test_message_embed_default(self):
        """Test creating empty embed."""
        embed = MessageEmbed()

        assert embed.title is None
        assert embed.description is None
        assert embed.url is None
        assert embed.color is None
        assert embed.footer is None
        assert embed.image_url is None
        assert embed.thumbnail_url is None
        assert embed.author_name is None
        assert embed.author_url is None
        assert embed.author_icon_url is None
        assert embed.fields == []
        assert embed.timestamp is None

    def test_message_embed_with_values(self):
        """Test creating embed with values."""
        embed = MessageEmbed(
            title="Test Title",
            description="Test Description",
            url="https://example.com",
            color=0xFF0000,
            footer="Footer text",
            image_url="https://example.com/image.png",
            thumbnail_url="https://example.com/thumb.png",
            author_name="Author",
            author_url="https://author.com",
            author_icon_url="https://author.com/icon.png",
            fields=[{"name": "Field1", "value": "Value1", "inline": True}],
            timestamp="2024-01-01T00:00:00Z",
        )

        assert embed.title == "Test Title"
        assert embed.description == "Test Description"
        assert embed.color == 0xFF0000
        assert len(embed.fields) == 1

    def test_to_slack_attachment_basic(self):
        """Test converting basic embed to Slack attachment."""
        embed = MessageEmbed(
            title="Alert",
            description="System is healthy",
        )

        attachment = embed.to_slack_attachment()

        assert attachment["title"] == "Alert"
        assert attachment["text"] == "System is healthy"

    def test_to_slack_attachment_full(self):
        """Test converting full embed to Slack attachment."""
        embed = MessageEmbed(
            title="Deployment Update",
            description="v2.0 deployed successfully",
            url="https://deploy.example.com",
            color=0x00FF00,
            footer="CI/CD Pipeline",
            image_url="https://example.com/graph.png",
            thumbnail_url="https://example.com/thumb.png",
            author_name="Jenkins",
            author_url="https://jenkins.example.com",
            author_icon_url="https://jenkins.example.com/icon.png",
            fields=[
                {"name": "Environment", "value": "Production", "inline": True},
                {"name": "Version", "value": "2.0.0", "inline": True},
            ],
            timestamp="1704067200",
        )

        attachment = embed.to_slack_attachment()

        assert attachment["title"] == "Deployment Update"
        assert attachment["text"] == "v2.0 deployed successfully"
        assert attachment["title_link"] == "https://deploy.example.com"
        assert attachment["color"] == "#00ff00"
        assert attachment["footer"] == "CI/CD Pipeline"
        assert attachment["image_url"] == "https://example.com/graph.png"
        assert attachment["thumb_url"] == "https://example.com/thumb.png"
        assert attachment["author_name"] == "Jenkins"
        assert attachment["author_link"] == "https://jenkins.example.com"
        assert attachment["author_icon"] == "https://jenkins.example.com/icon.png"
        assert attachment["ts"] == "1704067200"
        assert len(attachment["fields"]) == 2
        assert attachment["fields"][0]["title"] == "Environment"
        assert attachment["fields"][0]["value"] == "Production"
        assert attachment["fields"][0]["short"] is True

    def test_to_discord_embed_basic(self):
        """Test converting basic embed to Discord format."""
        embed = MessageEmbed(
            title="Alert",
            description="System is healthy",
        )

        discord_embed = embed.to_discord_embed()

        assert discord_embed["title"] == "Alert"
        assert discord_embed["description"] == "System is healthy"

    def test_to_discord_embed_full(self):
        """Test converting full embed to Discord format."""
        embed = MessageEmbed(
            title="Deployment Update",
            description="v2.0 deployed successfully",
            url="https://deploy.example.com",
            color=0x00FF00,
            footer="CI/CD Pipeline",
            image_url="https://example.com/graph.png",
            thumbnail_url="https://example.com/thumb.png",
            author_name="Jenkins",
            author_url="https://jenkins.example.com",
            author_icon_url="https://jenkins.example.com/icon.png",
            fields=[
                {"name": "Environment", "value": "Production", "inline": True},
            ],
            timestamp="2024-01-01T00:00:00Z",
        )

        discord_embed = embed.to_discord_embed()

        assert discord_embed["title"] == "Deployment Update"
        assert discord_embed["description"] == "v2.0 deployed successfully"
        assert discord_embed["url"] == "https://deploy.example.com"
        assert discord_embed["color"] == 0x00FF00
        assert discord_embed["footer"]["text"] == "CI/CD Pipeline"
        assert discord_embed["image"]["url"] == "https://example.com/graph.png"
        assert discord_embed["thumbnail"]["url"] == "https://example.com/thumb.png"
        assert discord_embed["author"]["name"] == "Jenkins"
        assert discord_embed["author"]["url"] == "https://jenkins.example.com"
        assert discord_embed["author"]["icon_url"] == "https://jenkins.example.com/icon.png"
        assert discord_embed["timestamp"] == "2024-01-01T00:00:00Z"
        assert len(discord_embed["fields"]) == 1

    def test_to_discord_embed_author_partial(self):
        """Test Discord embed with partial author info."""
        embed = MessageEmbed(author_name="Bot")

        discord_embed = embed.to_discord_embed()

        assert discord_embed["author"]["name"] == "Bot"
        assert "url" not in discord_embed["author"]
        assert "icon_url" not in discord_embed["author"]


# ============================================================================
# WebhookConfig Tests
# ============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_webhook_config_minimal(self):
        """Test creating config with minimal options."""
        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
        )

        assert config.url == "https://hooks.slack.com/services/XXX"
        assert config.platform == MessagingPlatform.SLACK
        assert config.name is None
        assert config.username is None
        assert config.avatar_url is None
        assert config.default_channel is None

    def test_webhook_config_full(self):
        """Test creating config with all options."""
        config = WebhookConfig(
            url="https://discord.com/api/webhooks/XXX/YYY",
            platform=MessagingPlatform.DISCORD,
            name="alerts",
            username="AlertBot",
            avatar_url="https://example.com/bot.png",
            default_channel="#general",
        )

        assert config.platform == MessagingPlatform.DISCORD
        assert config.name == "alerts"
        assert config.username == "AlertBot"
        assert config.avatar_url == "https://example.com/bot.png"


# ============================================================================
# SendResult Tests
# ============================================================================


class TestSendResult:
    """Tests for SendResult dataclass."""

    def test_send_result_success(self):
        """Test successful send result."""
        result = SendResult(
            success=True,
            message_id="12345.67890",
            timestamp="12345.67890",
            platform="slack",
        )

        assert result.success is True
        assert result.message_id == "12345.67890"
        assert result.error is None

    def test_send_result_failure(self):
        """Test failed send result."""
        result = SendResult(
            success=False,
            error="HTTP 404: Not Found",
            platform="discord",
        )

        assert result.success is False
        assert result.error == "HTTP 404: Not Found"

    def test_send_result_to_dict(self):
        """Test converting to dictionary."""
        result = SendResult(
            success=True,
            message_id="123",
            timestamp="12345",
            error=None,
            platform="slack",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["message_id"] == "123"
        assert d["platform"] == "slack"


# ============================================================================
# WebhookSender Tests
# ============================================================================


class TestWebhookSender:
    """Tests for WebhookSender class."""

    def test_create_webhook_sender(self):
        """Test creating webhook sender."""
        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
        )
        sender = WebhookSender(config)

        assert sender.config == config

    @patch("urllib.request.urlopen")
    def test_send_slack_text_success(self, mock_urlopen):
        """Test sending Slack text message."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"ok": True, "ts": "123.456"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
            username="TestBot",
        )
        sender = WebhookSender(config)

        result = sender.send_text("Hello World")

        assert result.success is True
        assert result.message_id == "123.456"
        assert result.platform == "slack"
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_send_slack_text_with_channel(self, mock_urlopen):
        """Test sending Slack message to specific channel."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
            default_channel="#general",
        )
        sender = WebhookSender(config)

        result = sender.send_text("Hello", channel="#alerts")

        assert result.success is True
        # Verify the request payload included the channel
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())
        assert payload["channel"] == "#alerts"

    @patch("urllib.request.urlopen")
    def test_send_slack_text_with_thread(self, mock_urlopen):
        """Test sending Slack reply in thread."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
        )
        sender = WebhookSender(config)

        result = sender.send_text("Reply", thread_id="123.456")

        assert result.success is True
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())
        assert payload["thread_ts"] == "123.456"

    @patch("urllib.request.urlopen")
    def test_send_discord_text_success(self, mock_urlopen):
        """Test sending Discord text message."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"id": "123456789"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            url="https://discord.com/api/webhooks/XXX/YYY",
            platform=MessagingPlatform.DISCORD,
            username="TestBot",
            avatar_url="https://example.com/avatar.png",
        )
        sender = WebhookSender(config)

        result = sender.send_text("Hello Discord")

        assert result.success is True
        assert result.message_id == "123456789"
        assert result.platform == "discord"

        # Verify Discord payload format
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())
        assert payload["content"] == "Hello Discord"
        assert payload["username"] == "TestBot"
        assert payload["avatar_url"] == "https://example.com/avatar.png"

    @patch("urllib.request.urlopen")
    def test_send_slack_embed(self, mock_urlopen):
        """Test sending Slack embed (attachment)."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
        )
        sender = WebhookSender(config)

        embed = MessageEmbed(
            title="Alert",
            description="Something happened",
            color=0xFF0000,
        )

        result = sender.send_embed(embed, text="Check this out")

        assert result.success is True
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())
        assert "attachments" in payload
        assert payload["text"] == "Check this out"

    @patch("urllib.request.urlopen")
    def test_send_discord_embed(self, mock_urlopen):
        """Test sending Discord embed."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            url="https://discord.com/api/webhooks/XXX/YYY",
            platform=MessagingPlatform.DISCORD,
        )
        sender = WebhookSender(config)

        embed = MessageEmbed(
            title="Notification",
            description="Build complete",
            color=0x00FF00,
        )

        result = sender.send_embed(embed)

        assert result.success is True
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode())
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1

    @patch("urllib.request.urlopen")
    def test_send_http_error(self, mock_urlopen):
        """Test handling HTTP error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        config = WebhookConfig(
            url="https://hooks.slack.com/services/XXX",
            platform=MessagingPlatform.SLACK,
        )
        sender = WebhookSender(config)

        result = sender.send_text("Hello")

        assert result.success is False
        assert "403" in result.error
        assert result.platform == "slack"

    @patch("urllib.request.urlopen")
    def test_send_url_error(self, mock_urlopen):
        """Test handling URL error (connection error)."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        config = WebhookConfig(
            url="https://invalid.example.com/webhook",
            platform=MessagingPlatform.DISCORD,
        )
        sender = WebhookSender(config)

        result = sender.send_text("Hello")

        assert result.success is False
        assert "Connection" in result.error

    @patch("urllib.request.urlopen")
    def test_send_generic_error(self, mock_urlopen):
        """Test handling generic exception."""
        mock_urlopen.side_effect = Exception("Unexpected error")

        config = WebhookConfig(
            url="https://example.com/webhook",
            platform=MessagingPlatform.SLACK,
        )
        sender = WebhookSender(config)

        result = sender.send_text("Hello")

        assert result.success is False
        assert "Unexpected error" in result.error


# ============================================================================
# MessagingManager Tests
# ============================================================================


class TestMessagingManager:
    """Tests for MessagingManager class."""

    def test_create_manager(self):
        """Test creating messaging manager."""
        manager = MessagingManager()

        assert manager.list_webhooks() == []

    def test_add_webhook(self):
        """Test adding webhook to manager."""
        manager = MessagingManager()
        config = WebhookConfig(
            url="https://hooks.slack.com/XXX",
            platform=MessagingPlatform.SLACK,
        )
        sender = WebhookSender(config)

        manager.add_webhook("alerts", sender)

        assert "alerts" in manager.list_webhooks()
        assert manager.get_webhook("alerts") == sender

    def test_get_webhook_not_found(self):
        """Test getting non-existent webhook."""
        manager = MessagingManager()

        result = manager.get_webhook("nonexistent")

        assert result is None

    def test_remove_webhook(self):
        """Test removing webhook."""
        manager = MessagingManager()
        config = WebhookConfig(
            url="https://example.com/webhook",
            platform=MessagingPlatform.DISCORD,
        )
        manager.add_webhook("test", WebhookSender(config))

        result = manager.remove_webhook("test")

        assert result is True
        assert manager.get_webhook("test") is None

    def test_remove_webhook_not_found(self):
        """Test removing non-existent webhook."""
        manager = MessagingManager()

        result = manager.remove_webhook("nonexistent")

        assert result is False

    def test_list_webhooks(self):
        """Test listing webhooks."""
        manager = MessagingManager()

        for name in ["slack1", "discord1", "alerts"]:
            config = WebhookConfig(
                url=f"https://example.com/{name}",
                platform=MessagingPlatform.SLACK,
            )
            manager.add_webhook(name, WebhookSender(config))

        webhooks = manager.list_webhooks()

        assert len(webhooks) == 3
        assert "slack1" in webhooks
        assert "discord1" in webhooks
        assert "alerts" in webhooks


# ============================================================================
# Tool Input/Output Model Tests
# ============================================================================


class TestInputOutputModels:
    """Tests for Pydantic input/output models."""

    def test_send_message_input(self):
        """Test SendMessageInput model."""
        input_data = SendMessageInput(
            webhook="alerts",
            text="Hello World",
            channel="#general",
            thread_id="123.456",
        )

        assert input_data.webhook == "alerts"
        assert input_data.text == "Hello World"
        assert input_data.channel == "#general"
        assert input_data.thread_id == "123.456"

    def test_send_message_input_defaults(self):
        """Test SendMessageInput default values."""
        input_data = SendMessageInput(text="Hello")

        assert input_data.webhook == "default"
        assert input_data.channel is None
        assert input_data.thread_id is None

    def test_send_embed_input(self):
        """Test SendEmbedInput model."""
        input_data = SendEmbedInput(
            webhook="alerts",
            title="Alert",
            description="Something happened",
            color="#FF0000",
            fields=[{"name": "Status", "value": "Critical", "inline": True}],
        )

        assert input_data.title == "Alert"
        assert input_data.color == "#FF0000"
        assert len(input_data.fields) == 1

    def test_create_webhook_input(self):
        """Test CreateWebhookInput model."""
        input_data = CreateWebhookInput(
            name="production-alerts",
            url="https://hooks.slack.com/XXX",
            platform="slack",
            username="AlertBot",
        )

        assert input_data.name == "production-alerts"
        assert input_data.platform == "slack"


# ============================================================================
# Tool Tests
# ============================================================================


class TestSendMessageTool:
    """Tests for SendMessageTool."""

    @pytest.fixture
    def manager_with_webhook(self):
        """Create manager with a mock webhook."""
        manager = MessagingManager()
        config = WebhookConfig(
            url="https://hooks.slack.com/XXX",
            platform=MessagingPlatform.SLACK,
        )
        manager.add_webhook("default", WebhookSender(config))
        return manager

    @pytest.mark.asyncio
    async def test_send_message_webhook_not_found(self):
        """Test sending message with non-existent webhook."""
        manager = MessagingManager()
        tool = SendMessageTool(manager)

        input_data = SendMessageInput(
            webhook="nonexistent",
            text="Hello",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_send_message_success(self, mock_urlopen, manager_with_webhook):
        """Test successful message send."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"ts": "123.456"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = SendMessageTool(manager_with_webhook)

        input_data = SendMessageInput(
            webhook="default",
            text="Hello World",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.platform == "slack"

    def test_tool_metadata(self):
        """Test tool metadata."""
        manager = MessagingManager()
        tool = SendMessageTool(manager)

        assert tool.metadata.id == "send_message"
        assert tool.metadata.name == "Send Message"


class TestSendEmbedTool:
    """Tests for SendEmbedTool."""

    @pytest.fixture
    def manager_with_discord(self):
        """Create manager with Discord webhook."""
        manager = MessagingManager()
        config = WebhookConfig(
            url="https://discord.com/api/webhooks/XXX/YYY",
            platform=MessagingPlatform.DISCORD,
        )
        manager.add_webhook("discord", WebhookSender(config))
        return manager

    @pytest.mark.asyncio
    async def test_send_embed_webhook_not_found(self):
        """Test sending embed with non-existent webhook."""
        manager = MessagingManager()
        tool = SendEmbedTool(manager)

        input_data = SendEmbedInput(
            webhook="nonexistent",
            title="Test",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_send_embed_success(self, mock_urlopen, manager_with_discord):
        """Test successful embed send."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"id": "123"}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = SendEmbedTool(manager_with_discord)

        input_data = SendEmbedInput(
            webhook="discord",
            title="Alert",
            description="Something happened",
            color="#FF0000",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.platform == "discord"

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_send_embed_with_fields(self, mock_urlopen, manager_with_discord):
        """Test sending embed with fields."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = SendEmbedTool(manager_with_discord)

        input_data = SendEmbedInput(
            webhook="discord",
            title="Status Report",
            fields=[
                {"name": "Status", "value": "Online", "inline": True},
                {"name": "Uptime", "value": "99.9%", "inline": True},
            ],
        )

        result = await tool.execute(input_data)

        assert result.success is True

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_send_embed_invalid_color(self, mock_urlopen, manager_with_discord):
        """Test sending embed with invalid color (should still work)."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"{}"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = SendEmbedTool(manager_with_discord)

        input_data = SendEmbedInput(
            webhook="discord",
            title="Test",
            color="invalid-color",
        )

        result = await tool.execute(input_data)

        # Should still succeed, just without color
        assert result.success is True


class TestCreateWebhookTool:
    """Tests for CreateWebhookTool."""

    @pytest.mark.asyncio
    async def test_create_slack_webhook(self):
        """Test creating Slack webhook."""
        manager = MessagingManager()
        tool = CreateWebhookTool(manager)

        input_data = CreateWebhookInput(
            name="alerts",
            url="https://hooks.slack.com/services/XXX",
            platform="slack",
            username="AlertBot",
            default_channel="#alerts",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.name == "alerts"
        assert result.platform == "slack"
        assert manager.get_webhook("alerts") is not None

    @pytest.mark.asyncio
    async def test_create_discord_webhook(self):
        """Test creating Discord webhook."""
        manager = MessagingManager()
        tool = CreateWebhookTool(manager)

        input_data = CreateWebhookInput(
            name="discord-alerts",
            url="https://discord.com/api/webhooks/XXX/YYY",
            platform="discord",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.platform == "discord"

    @pytest.mark.asyncio
    async def test_create_webhook_invalid_platform(self):
        """Test creating webhook with invalid platform."""
        manager = MessagingManager()
        tool = CreateWebhookTool(manager)

        input_data = CreateWebhookInput(
            name="test",
            url="https://example.com/webhook",
            platform="teams",  # Not supported
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "Invalid platform" in result.error


class EmptyInput(BaseModel):
    """Empty input model for testing."""
    pass


class TestListWebhooksTool:
    """Tests for ListWebhooksTool."""

    @pytest.mark.asyncio
    async def test_list_webhooks_empty(self):
        """Test listing webhooks when none exist."""
        manager = MessagingManager()
        tool = ListWebhooksTool(manager)

        result = await tool.execute(EmptyInput())

        assert result.webhooks == []

    @pytest.mark.asyncio
    async def test_list_webhooks_with_entries(self):
        """Test listing webhooks."""
        manager = MessagingManager()

        for name in ["slack1", "discord1"]:
            config = WebhookConfig(
                url=f"https://example.com/{name}",
                platform=MessagingPlatform.SLACK,
            )
            manager.add_webhook(name, WebhookSender(config))

        tool = ListWebhooksTool(manager)

        result = await tool.execute(EmptyInput())

        assert len(result.webhooks) == 2
        assert "slack1" in result.webhooks
        assert "discord1" in result.webhooks


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_webhook_config(self):
        """Test create_webhook_config function."""
        config = create_webhook_config(
            url="https://hooks.slack.com/XXX",
            platform=MessagingPlatform.SLACK,
            username="Bot",
            avatar_url="https://example.com/avatar.png",
            default_channel="#general",
        )

        assert config.url == "https://hooks.slack.com/XXX"
        assert config.platform == MessagingPlatform.SLACK
        assert config.username == "Bot"
        assert config.default_channel == "#general"

    def test_create_webhook_sender(self):
        """Test create_webhook_sender function."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            platform=MessagingPlatform.DISCORD,
        )

        sender = create_webhook_sender(config)

        assert isinstance(sender, WebhookSender)
        assert sender.config == config

    def test_create_messaging_manager(self):
        """Test create_messaging_manager function."""
        manager = create_messaging_manager()

        assert isinstance(manager, MessagingManager)
        assert manager.list_webhooks() == []

    def test_create_messaging_tools(self):
        """Test create_messaging_tools function."""
        manager = MessagingManager()
        tools = create_messaging_tools(manager)

        assert "send_message" in tools
        assert "send_embed" in tools
        assert "create_messaging_webhook" in tools
        assert "list_messaging_webhooks" in tools

        assert isinstance(tools["send_message"], SendMessageTool)
        assert isinstance(tools["send_embed"], SendEmbedTool)
        assert isinstance(tools["create_messaging_webhook"], CreateWebhookTool)
        assert isinstance(tools["list_messaging_webhooks"], ListWebhooksTool)


# ============================================================================
# Platform Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_messaging_platform_values(self):
        """Test MessagingPlatform enum values."""
        assert MessagingPlatform.SLACK.value == "slack"
        assert MessagingPlatform.DISCORD.value == "discord"

    def test_message_type_values(self):
        """Test MessageType enum values."""
        assert MessageType.TEXT.value == "text"
        assert MessageType.RICH.value == "rich"
        assert MessageType.EMBED.value == "embed"
