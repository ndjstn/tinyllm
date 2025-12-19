"""Error notification channels for TinyLLM.

This module provides webhook and email notification capabilities
for error alerts based on impact scores and thresholds.
"""

import asyncio
import json
import smtplib
from abc import ABC, abstractmethod
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from tinyllm.error_aggregation import AggregatedError
from tinyllm.error_enrichment import EnrichedError
from tinyllm.error_impact import ImpactLevel, ImpactScore
from tinyllm.logging import get_logger

logger = get_logger(__name__, component="error_notifications")


class NotificationChannelType(str, Enum):
    """Notification channel types."""

    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    LOG = "log"


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationConfig(BaseModel):
    """Configuration for notification channels."""

    model_config = {"extra": "forbid"}

    # Channel settings
    channel: NotificationChannelType = Field(description="Channel type")
    enabled: bool = Field(default=True, description="Whether channel is enabled")

    # Filtering
    min_impact_level: ImpactLevel = Field(
        default=ImpactLevel.MEDIUM,
        description="Minimum impact level to notify"
    )
    min_impact_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum impact score to notify"
    )

    # Rate limiting
    rate_limit_minutes: int = Field(
        default=5,
        ge=0,
        description="Minimum minutes between notifications"
    )
    max_notifications_per_hour: int = Field(
        default=12,
        ge=1,
        description="Maximum notifications per hour"
    )

    # Channel-specific config
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Channel-specific configuration"
    )


class WebhookConfig(BaseModel):
    """Webhook-specific configuration."""

    model_config = {"extra": "forbid"}

    url: str = Field(description="Webhook URL")
    method: str = Field(default="POST", description="HTTP method")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers"
    )
    timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Request timeout"
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Number of retries"
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate HTTP method."""
        v = v.upper()
        if v not in {"GET", "POST", "PUT", "PATCH"}:
            raise ValueError(f"Invalid HTTP method: {v}")
        return v


class EmailConfig(BaseModel):
    """Email-specific configuration."""

    model_config = {"extra": "forbid"}

    smtp_host: str = Field(description="SMTP server host")
    smtp_port: int = Field(default=587, ge=1, le=65535, description="SMTP port")
    smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    use_tls: bool = Field(default=True, description="Use TLS")
    from_address: str = Field(description="From email address")
    to_addresses: List[str] = Field(description="To email addresses")
    subject_prefix: str = Field(
        default="[TinyLLM Error]",
        description="Email subject prefix"
    )


class Notification(BaseModel):
    """Notification payload."""

    model_config = {"extra": "forbid"}

    # Core information
    notification_id: str = Field(description="Unique notification ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When notification was created"
    )
    priority: NotificationPriority = Field(description="Notification priority")

    # Error information
    error_id: str = Field(description="Error ID")
    signature_hash: Optional[str] = Field(
        default=None,
        description="Signature hash for aggregated errors"
    )
    impact_score: ImpactScore = Field(description="Impact score")

    # Message
    title: str = Field(description="Notification title")
    message: str = Field(description="Notification message")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details"
    )

    # Actions
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Suggested actions"
    )


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    def __init__(self, config: NotificationConfig):
        """Initialize channel.

        Args:
            config: Channel configuration.
        """
        self.config = config
        self._last_notification: Dict[str, datetime] = {}
        self._hourly_counts: Dict[int, int] = {}

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification.

        Args:
            notification: Notification to send.

        Returns:
            True if sent successfully.
        """
        pass

    def should_notify(
        self,
        impact_score: ImpactScore,
        signature_hash: Optional[str] = None,
    ) -> bool:
        """Check if notification should be sent.

        Args:
            impact_score: Impact score to check.
            signature_hash: Optional signature hash for deduplication.

        Returns:
            True if notification should be sent.
        """
        # Check if channel is enabled
        if not self.config.enabled:
            return False

        # Check impact thresholds
        if impact_score.total_score < self.config.min_impact_score:
            return False

        impact_values = {
            ImpactLevel.NEGLIGIBLE: 0,
            ImpactLevel.LOW: 1,
            ImpactLevel.MEDIUM: 2,
            ImpactLevel.HIGH: 3,
            ImpactLevel.CRITICAL: 4,
            ImpactLevel.CATASTROPHIC: 5,
        }

        if impact_values[impact_score.impact_level] < impact_values[self.config.min_impact_level]:
            return False

        # Check rate limiting
        key = signature_hash or impact_score.error_id
        now = datetime.utcnow()

        # Check time since last notification
        if key in self._last_notification:
            last_time = self._last_notification[key]
            minutes_since = (now - last_time).total_seconds() / 60
            if minutes_since < self.config.rate_limit_minutes:
                logger.debug(
                    "notification_rate_limited",
                    key=key,
                    minutes_since=minutes_since,
                    limit_minutes=self.config.rate_limit_minutes,
                )
                return False

        # Check hourly limit
        current_hour = now.hour
        hourly_count = self._hourly_counts.get(current_hour, 0)
        if hourly_count >= self.config.max_notifications_per_hour:
            logger.warning(
                "notification_hourly_limit_reached",
                current_hour=current_hour,
                count=hourly_count,
                limit=self.config.max_notifications_per_hour,
            )
            return False

        return True

    def record_notification(
        self,
        signature_hash: Optional[str] = None,
        error_id: Optional[str] = None,
    ) -> None:
        """Record that a notification was sent.

        Args:
            signature_hash: Optional signature hash.
            error_id: Optional error ID.
        """
        key = signature_hash or error_id or "unknown"
        now = datetime.utcnow()

        self._last_notification[key] = now

        # Increment hourly count
        current_hour = now.hour
        self._hourly_counts[current_hour] = self._hourly_counts.get(current_hour, 0) + 1

        # Cleanup old hourly counts (keep last 2 hours)
        hours_to_keep = {current_hour, (current_hour - 1) % 24}
        self._hourly_counts = {
            h: c for h, c in self._hourly_counts.items()
            if h in hours_to_keep
        }


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""

    def __init__(self, config: NotificationConfig, webhook_config: WebhookConfig):
        """Initialize webhook channel.

        Args:
            config: Base configuration.
            webhook_config: Webhook-specific configuration.
        """
        super().__init__(config)
        self.webhook_config = webhook_config
        self._client = httpx.AsyncClient(timeout=webhook_config.timeout_seconds)

    async def send(self, notification: Notification) -> bool:
        """Send notification via webhook.

        Args:
            notification: Notification to send.

        Returns:
            True if sent successfully.
        """
        # Prepare payload
        payload = {
            "notification_id": notification.notification_id,
            "timestamp": notification.timestamp.isoformat(),
            "priority": notification.priority.value,
            "error_id": notification.error_id,
            "impact_level": notification.impact_score.impact_level.value,
            "impact_score": notification.impact_score.total_score,
            "title": notification.title,
            "message": notification.message,
            "details": notification.details,
            "suggested_actions": notification.suggested_actions,
        }

        # Send with retries
        for attempt in range(self.webhook_config.retry_count + 1):
            try:
                response = await self._client.request(
                    method=self.webhook_config.method,
                    url=self.webhook_config.url,
                    json=payload,
                    headers=self.webhook_config.headers,
                )

                if response.status_code < 400:
                    logger.info(
                        "webhook_notification_sent",
                        notification_id=notification.notification_id,
                        url=self.webhook_config.url,
                        status_code=response.status_code,
                    )
                    return True
                else:
                    logger.warning(
                        "webhook_notification_failed",
                        notification_id=notification.notification_id,
                        url=self.webhook_config.url,
                        status_code=response.status_code,
                        attempt=attempt + 1,
                    )

            except Exception as e:
                logger.error(
                    "webhook_notification_error",
                    notification_id=notification.notification_id,
                    url=self.webhook_config.url,
                    error=str(e),
                    attempt=attempt + 1,
                )

            # Wait before retry (exponential backoff)
            if attempt < self.webhook_config.retry_count:
                await asyncio.sleep(2 ** attempt)

        return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(self, config: NotificationConfig, email_config: EmailConfig):
        """Initialize email channel.

        Args:
            config: Base configuration.
            email_config: Email-specific configuration.
        """
        super().__init__(config)
        self.email_config = email_config

    async def send(self, notification: Notification) -> bool:
        """Send notification via email.

        Args:
            notification: Notification to send.

        Returns:
            True if sent successfully.
        """
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{self.email_config.subject_prefix} {notification.title}"
            msg["From"] = self.email_config.from_address
            msg["To"] = ", ".join(self.email_config.to_addresses)

            # Create body
            text_body = self._format_text_body(notification)
            html_body = self._format_html_body(notification)

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(
                self.email_config.smtp_host,
                self.email_config.smtp_port
            ) as server:
                if self.email_config.use_tls:
                    server.starttls()

                if self.email_config.smtp_username and self.email_config.smtp_password:
                    server.login(
                        self.email_config.smtp_username,
                        self.email_config.smtp_password,
                    )

                server.send_message(msg)

            logger.info(
                "email_notification_sent",
                notification_id=notification.notification_id,
                to_addresses=self.email_config.to_addresses,
            )
            return True

        except Exception as e:
            logger.error(
                "email_notification_error",
                notification_id=notification.notification_id,
                error=str(e),
            )
            return False

    def _format_text_body(self, notification: Notification) -> str:
        """Format plain text email body.

        Args:
            notification: Notification to format.

        Returns:
            Plain text body.
        """
        lines = [
            f"Error Notification: {notification.title}",
            "=" * 60,
            f"Priority: {notification.priority.value.upper()}",
            f"Impact Level: {notification.impact_score.impact_level.value.upper()}",
            f"Impact Score: {notification.impact_score.total_score}/100",
            f"Timestamp: {notification.timestamp.isoformat()}",
            "",
            "Message:",
            notification.message,
            "",
        ]

        if notification.suggested_actions:
            lines.append("Suggested Actions:")
            for action in notification.suggested_actions:
                lines.append(f"  - {action}")
            lines.append("")

        if notification.details:
            lines.append("Details:")
            for key, value in notification.details.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def _format_html_body(self, notification: Notification) -> str:
        """Format HTML email body.

        Args:
            notification: Notification to format.

        Returns:
            HTML body.
        """
        impact_color = {
            ImpactLevel.NEGLIGIBLE: "#28a745",
            ImpactLevel.LOW: "#6c757d",
            ImpactLevel.MEDIUM: "#ffc107",
            ImpactLevel.HIGH: "#fd7e14",
            ImpactLevel.CRITICAL: "#dc3545",
            ImpactLevel.CATASTROPHIC: "#721c24",
        }.get(notification.impact_score.impact_level, "#6c757d")

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {impact_color}; color: white; padding: 20px;">
                <h2 style="margin: 0;">{notification.title}</h2>
                <p style="margin: 5px 0 0 0;">Priority: {notification.priority.value.upper()}</p>
            </div>

            <div style="padding: 20px; background-color: #f8f9fa;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">Impact Level:</td>
                        <td style="padding: 5px;">{notification.impact_score.impact_level.value.upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">Impact Score:</td>
                        <td style="padding: 5px;">{notification.impact_score.total_score}/100</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">Timestamp:</td>
                        <td style="padding: 5px;">{notification.timestamp.isoformat()}</td>
                    </tr>
                </table>
            </div>

            <div style="padding: 20px;">
                <h3>Message</h3>
                <p>{notification.message}</p>
        """

        if notification.suggested_actions:
            html += """
                <h3>Suggested Actions</h3>
                <ul>
            """
            for action in notification.suggested_actions:
                html += f"<li>{action}</li>"
            html += "</ul>"

        if notification.details:
            html += """
                <h3>Details</h3>
                <table style="width: 100%; border-collapse: collapse;">
            """
            for key, value in notification.details.items():
                html += f"""
                <tr>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd; font-weight: bold;">{key}:</td>
                    <td style="padding: 5px; border-bottom: 1px solid #ddd;">{value}</td>
                </tr>
                """
            html += "</table>"

        html += """
            </div>
        </body>
        </html>
        """

        return html


class LogNotificationChannel(NotificationChannel):
    """Log-only notification channel (for testing)."""

    async def send(self, notification: Notification) -> bool:
        """Log notification.

        Args:
            notification: Notification to log.

        Returns:
            Always returns True.
        """
        logger.info(
            "log_notification",
            notification_id=notification.notification_id,
            title=notification.title,
            priority=notification.priority.value,
            impact_level=notification.impact_score.impact_level.value,
            impact_score=notification.impact_score.total_score,
        )
        return True


class NotificationManager:
    """Manage error notifications across multiple channels."""

    def __init__(self):
        """Initialize notification manager."""
        self.channels: List[NotificationChannel] = []
        logger.info("notification_manager_initialized")

    def add_webhook_channel(
        self,
        webhook_config: WebhookConfig,
        channel_config: Optional[NotificationConfig] = None,
    ) -> None:
        """Add webhook notification channel.

        Args:
            webhook_config: Webhook configuration.
            channel_config: Optional base channel configuration.
        """
        if channel_config is None:
            channel_config = NotificationConfig(channel=NotificationChannel.WEBHOOK)

        channel = WebhookNotificationChannel(channel_config, webhook_config)
        self.channels.append(channel)

        logger.info(
            "webhook_channel_added",
            url=webhook_config.url,
            min_impact_level=channel_config.min_impact_level.value,
        )

    def add_email_channel(
        self,
        email_config: EmailConfig,
        channel_config: Optional[NotificationConfig] = None,
    ) -> None:
        """Add email notification channel.

        Args:
            email_config: Email configuration.
            channel_config: Optional base channel configuration.
        """
        if channel_config is None:
            channel_config = NotificationConfig(channel=NotificationChannel.EMAIL)

        channel = EmailNotificationChannel(channel_config, email_config)
        self.channels.append(channel)

        logger.info(
            "email_channel_added",
            to_addresses=email_config.to_addresses,
            min_impact_level=channel_config.min_impact_level.value,
        )

    def add_log_channel(
        self,
        channel_config: Optional[NotificationConfig] = None,
    ) -> None:
        """Add log notification channel.

        Args:
            channel_config: Optional base channel configuration.
        """
        if channel_config is None:
            channel_config = NotificationConfig(channel=NotificationChannel.LOG)

        channel = LogNotificationChannel(channel_config)
        self.channels.append(channel)

        logger.info("log_channel_added")

    async def notify_error(
        self,
        error: EnrichedError,
        impact_score: ImpactScore,
    ) -> int:
        """Send notification for a single error.

        Args:
            error: Enriched error.
            impact_score: Impact score for error.

        Returns:
            Number of channels notified.
        """
        notification = self._create_notification(
            error_id=error.error_id,
            impact_score=impact_score,
            title=f"Error in {error.context.node_id or 'Unknown Node'}",
            message=error.message,
            details={
                "category": error.category.value,
                "severity": error.severity.value,
                "node_id": error.context.node_id or "unknown",
                "graph_id": error.context.graph_id or "unknown",
            },
        )

        return await self._send_notification(notification, None)

    async def notify_aggregated_error(
        self,
        agg_error: AggregatedError,
        impact_score: ImpactScore,
    ) -> int:
        """Send notification for an aggregated error.

        Args:
            agg_error: Aggregated error.
            impact_score: Impact score for error.

        Returns:
            Number of channels notified.
        """
        notification = self._create_notification(
            error_id=agg_error.signature.signature_hash,
            impact_score=impact_score,
            signature_hash=agg_error.signature.signature_hash,
            title=f"Recurring Error: {agg_error.signature.exception_type}",
            message=f"Error occurred {agg_error.count} times",
            details={
                "category": agg_error.signature.category.value,
                "occurrence_count": agg_error.count,
                "affected_nodes": len(agg_error.affected_nodes),
                "affected_graphs": len(agg_error.affected_graphs),
                "first_seen": agg_error.first_seen.isoformat(),
                "last_seen": agg_error.last_seen.isoformat(),
            },
        )

        return await self._send_notification(
            notification,
            agg_error.signature.signature_hash
        )

    def _create_notification(
        self,
        error_id: str,
        impact_score: ImpactScore,
        title: str,
        message: str,
        details: Dict[str, Any],
        signature_hash: Optional[str] = None,
    ) -> Notification:
        """Create notification from error and impact.

        Args:
            error_id: Error ID.
            impact_score: Impact score.
            title: Notification title.
            message: Notification message.
            details: Additional details.
            signature_hash: Optional signature hash.

        Returns:
            Notification object.
        """
        # Determine priority from impact level
        priority_map = {
            ImpactLevel.NEGLIGIBLE: NotificationPriority.LOW,
            ImpactLevel.LOW: NotificationPriority.LOW,
            ImpactLevel.MEDIUM: NotificationPriority.NORMAL,
            ImpactLevel.HIGH: NotificationPriority.HIGH,
            ImpactLevel.CRITICAL: NotificationPriority.URGENT,
            ImpactLevel.CATASTROPHIC: NotificationPriority.URGENT,
        }

        priority = priority_map.get(
            impact_score.impact_level,
            NotificationPriority.NORMAL
        )

        return Notification(
            notification_id=f"notif_{error_id}_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow(),
            priority=priority,
            error_id=error_id,
            signature_hash=signature_hash,
            impact_score=impact_score,
            title=title,
            message=message,
            details=details,
            suggested_actions=impact_score.recommended_actions,
        )

    async def _send_notification(
        self,
        notification: Notification,
        signature_hash: Optional[str],
    ) -> int:
        """Send notification to all applicable channels.

        Args:
            notification: Notification to send.
            signature_hash: Optional signature hash for deduplication.

        Returns:
            Number of channels notified.
        """
        notified_count = 0

        for channel in self.channels:
            if channel.should_notify(notification.impact_score, signature_hash):
                try:
                    success = await channel.send(notification)
                    if success:
                        channel.record_notification(
                            signature_hash=signature_hash,
                            error_id=notification.error_id,
                        )
                        notified_count += 1
                except Exception as e:
                    logger.error(
                        "notification_channel_error",
                        channel=channel.__class__.__name__,
                        error=str(e),
                    )

        logger.info(
            "notification_sent",
            notification_id=notification.notification_id,
            channels_notified=notified_count,
            total_channels=len(self.channels),
        )

        return notified_count
