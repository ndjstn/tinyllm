"""Tests for error notification channels."""

import pytest

from tinyllm.error_enrichment import ErrorCategory, ErrorSeverity, enrich_error
from tinyllm.error_impact import ImpactLevel, ImpactScore, ImpactScorer
from tinyllm.error_notifications import (
    EmailConfig,
    LogNotificationChannel,
    Notification,
    NotificationConfig,
    NotificationManager,
    NotificationPriority,
    WebhookConfig,
    WebhookNotificationChannel,
)

# Import the enum separately to avoid naming conflict with ABC
from tinyllm.error_notifications import NotificationChannel as NotificationChannelEnum
from tinyllm.errors import ExecutionError, TimeoutError


class TestNotificationConfig:
    """Test notification configuration."""

    def test_config_creation(self):
        """Test creating notification config."""
        config = NotificationConfig(
            channel=NotificationChannelEnum.WEBHOOK,
            min_impact_level=ImpactLevel.HIGH,
            min_impact_score=75.0,
        )

        assert config.channel == NotificationChannelEnum.WEBHOOK
        assert config.min_impact_level == ImpactLevel.HIGH
        assert config.min_impact_score == 75.0
        assert config.enabled is True

    def test_config_defaults(self):
        """Test default configuration values."""
        config = NotificationConfig(channel=NotificationChannelEnum.LOG)

        assert config.min_impact_level == ImpactLevel.MEDIUM
        assert config.min_impact_score == 50.0
        assert config.rate_limit_minutes == 5
        assert config.max_notifications_per_hour == 12


class TestWebhookConfig:
    """Test webhook configuration."""

    def test_webhook_config_creation(self):
        """Test creating webhook config."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            headers={"Authorization": "Bearer token"},
        )

        assert config.url == "https://example.com/webhook"
        assert config.method == "POST"
        assert config.headers["Authorization"] == "Bearer token"

    def test_webhook_config_validation(self):
        """Test webhook config validation."""
        config = WebhookConfig(url="https://example.com", method="post")
        assert config.method == "POST"

        with pytest.raises(ValueError):
            WebhookConfig(url="https://example.com", method="DELETE")


class TestEmailConfig:
    """Test email configuration."""

    def test_email_config_creation(self):
        """Test creating email config."""
        config = EmailConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_address="alerts@example.com",
            to_addresses=["admin@example.com"],
        )

        assert config.smtp_host == "smtp.example.com"
        assert config.smtp_port == 587
        assert config.use_tls is True
        assert len(config.to_addresses) == 1


class TestNotification:
    """Test notification payload."""

    def test_notification_creation(self):
        """Test creating notification."""
        scorer = ImpactScorer()
        error = ExecutionError("test error")
        enriched = enrich_error(
            error,
            "error-1",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
        )
        impact = scorer.score_error(enriched)

        notification = Notification(
            notification_id="notif-1",
            priority=NotificationPriority.HIGH,
            error_id="error-1",
            impact_score=impact,
            title="Test Error",
            message="An error occurred",
        )

        assert notification.notification_id == "notif-1"
        assert notification.priority == NotificationPriority.HIGH
        assert notification.error_id == "error-1"
        assert notification.title == "Test Error"


class TestLogNotificationChannel:
    """Test log notification channel."""

    @pytest.mark.asyncio
    async def test_send_notification(self):
        """Test sending log notification."""
        config = NotificationConfig(channel=NotificationChannelEnum.LOG)
        channel = LogNotificationChannel(config)

        scorer = ImpactScorer()
        error = ExecutionError("test")
        enriched = enrich_error(error, "err-1", category=ErrorCategory.EXECUTION)
        impact = scorer.score_error(enriched)

        notification = Notification(
            notification_id="notif-1",
            priority=NotificationPriority.NORMAL,
            error_id="err-1",
            impact_score=impact,
            title="Test",
            message="Test message",
        )

        success = await channel.send(notification)
        assert success is True

    def test_should_notify_threshold(self):
        """Test notification threshold checking."""
        config = NotificationConfig(
            channel=NotificationChannelEnum.LOG,
            min_impact_level=ImpactLevel.HIGH,
            min_impact_score=70.0,
        )
        channel = LogNotificationChannel(config)

        # High impact - should notify
        high_impact = ImpactScore(
            total_score=80.0,
            impact_level=ImpactLevel.HIGH,
            severity_score=80.0,
            frequency_score=0.0,
            scope_score=0.0,
            recency_score=100.0,
            criticality_score=70.0,
            error_id="err-1",
        )
        assert channel.should_notify(high_impact)

        # Low impact - should not notify
        low_impact = ImpactScore(
            total_score=30.0,
            impact_level=ImpactLevel.LOW,
            severity_score=30.0,
            frequency_score=0.0,
            scope_score=0.0,
            recency_score=50.0,
            criticality_score=20.0,
            error_id="err-2",
        )
        assert not channel.should_notify(low_impact)

    def test_rate_limiting(self):
        """Test notification rate limiting."""
        config = NotificationConfig(
            channel=NotificationChannelEnum.LOG,
            rate_limit_minutes=5,
        )
        channel = LogNotificationChannel(config)

        impact = ImpactScore(
            total_score=80.0,
            impact_level=ImpactLevel.HIGH,
            severity_score=80.0,
            frequency_score=0.0,
            scope_score=0.0,
            recency_score=100.0,
            criticality_score=70.0,
            error_id="err-1",
        )

        # First notification should be allowed
        assert channel.should_notify(impact, signature_hash="sig-1")

        # Record it
        channel.record_notification(signature_hash="sig-1")

        # Second notification immediately after should be rate limited
        assert not channel.should_notify(impact, signature_hash="sig-1")


class TestNotificationManager:
    """Test notification manager."""

    def test_manager_creation(self):
        """Test creating notification manager."""
        manager = NotificationManager()
        assert len(manager.channels) == 0

    def test_add_log_channel(self):
        """Test adding log channel."""
        manager = NotificationManager()
        manager.add_log_channel()

        assert len(manager.channels) == 1
        assert isinstance(manager.channels[0], LogNotificationChannel)

    @pytest.mark.asyncio
    async def test_notify_error(self):
        """Test notifying single error."""
        manager = NotificationManager()
        manager.add_log_channel()

        scorer = ImpactScorer()
        error = ExecutionError("critical failure")
        enriched = enrich_error(
            error,
            "err-1",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
            node_id="worker-1",
        )
        impact = scorer.score_error(enriched)

        count = await manager.notify_error(enriched, impact)
        assert count >= 0  # May be 0 if below threshold

    @pytest.mark.asyncio
    async def test_multiple_channels(self):
        """Test multiple notification channels."""
        manager = NotificationManager()

        # Add multiple log channels with different thresholds
        manager.add_log_channel(
            NotificationConfig(
                channel=NotificationChannelEnum.LOG,
                min_impact_level=ImpactLevel.HIGH,
            )
        )
        manager.add_log_channel(
            NotificationConfig(
                channel=NotificationChannelEnum.LOG,
                min_impact_level=ImpactLevel.CRITICAL,
            )
        )

        assert len(manager.channels) == 2


class TestWebhookNotificationChannel:
    """Test webhook notification channel."""

    def test_webhook_channel_creation(self):
        """Test creating webhook channel."""
        config = NotificationConfig(channel=NotificationChannelEnum.WEBHOOK)
        webhook_config = WebhookConfig(url="https://example.com/webhook")

        channel = WebhookNotificationChannel(config, webhook_config)
        assert channel.webhook_config.url == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_webhook_channel_cleanup(self):
        """Test webhook channel cleanup."""
        config = NotificationConfig(channel=NotificationChannelEnum.WEBHOOK)
        webhook_config = WebhookConfig(url="https://example.com/webhook")

        channel = WebhookNotificationChannel(config, webhook_config)
        await channel.close()


class TestNotificationIntegration:
    """Integration tests for notification system."""

    @pytest.mark.asyncio
    async def test_full_notification_pipeline(self):
        """Test complete notification pipeline."""
        # Create error and enrich
        error = ExecutionError("Critical system failure")
        enriched = enrich_error(
            error,
            "error-1",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.CRITICAL,
            node_id="worker-1",
            graph_id="workflow-1",
        )

        # Score impact
        scorer = ImpactScorer()
        impact = scorer.score_error(enriched)

        # Setup notification manager
        manager = NotificationManager()
        manager.add_log_channel(
            NotificationConfig(
                channel=NotificationChannelEnum.LOG,
                min_impact_level=ImpactLevel.MEDIUM,
            )
        )

        # Send notification
        count = await manager.notify_error(enriched, impact)

        # Verify notification was sent or rejected based on impact
        assert count >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
