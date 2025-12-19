"""Email tools for TinyLLM.

This module provides tools for sending and composing emails,
with support for SMTP, templates, and attachments.
"""

import base64
import logging
import mimetypes
import os
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, EmailStr, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class EmailFormat(str, Enum):
    """Email content formats."""

    PLAIN = "plain"
    HTML = "html"


class SmtpSecurity(str, Enum):
    """SMTP security options."""

    NONE = "none"
    STARTTLS = "starttls"
    SSL = "ssl"


@dataclass
class SmtpConfig:
    """SMTP server configuration."""

    host: str = "localhost"
    port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    security: SmtpSecurity = SmtpSecurity.STARTTLS
    timeout: float = 30.0
    from_address: Optional[str] = None
    from_name: Optional[str] = None

    @property
    def from_header(self) -> str:
        """Get formatted from header."""
        if self.from_name and self.from_address:
            return f"{self.from_name} <{self.from_address}>"
        return self.from_address or ""


@dataclass
class Attachment:
    """Email attachment."""

    filename: str
    content: bytes
    mime_type: Optional[str] = None
    content_id: Optional[str] = None  # For inline attachments

    @classmethod
    def from_file(cls, path: str) -> "Attachment":
        """Create attachment from file.

        Args:
            path: Path to file.

        Returns:
            Attachment instance.
        """
        file_path = Path(path)
        mime_type, _ = mimetypes.guess_type(file_path.name)

        with open(file_path, "rb") as f:
            content = f.read()

        return cls(
            filename=file_path.name,
            content=content,
            mime_type=mime_type or "application/octet-stream",
        )

    @classmethod
    def from_base64(
        cls,
        filename: str,
        base64_content: str,
        mime_type: Optional[str] = None,
    ) -> "Attachment":
        """Create attachment from base64 content.

        Args:
            filename: Attachment filename.
            base64_content: Base64-encoded content.
            mime_type: Optional MIME type.

        Returns:
            Attachment instance.
        """
        content = base64.b64decode(base64_content)
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)

        return cls(
            filename=filename,
            content=content,
            mime_type=mime_type or "application/octet-stream",
        )


@dataclass
class EmailMessage:
    """An email message."""

    to: List[str]
    subject: str
    body: str
    format: EmailFormat = EmailFormat.PLAIN
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None
    attachments: List[Attachment] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def all_recipients(self) -> List[str]:
        """Get all recipients (to + cc + bcc)."""
        return self.to + self.cc + self.bcc


@dataclass
class SendResult:
    """Result of sending an email."""

    success: bool
    message_id: Optional[str] = None
    recipients_accepted: List[str] = field(default_factory=list)
    recipients_rejected: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message_id": self.message_id,
            "recipients_accepted": self.recipients_accepted,
            "recipients_rejected": self.recipients_rejected,
            "error": self.error,
        }


class EmailTemplate:
    """Email template with variable substitution."""

    def __init__(
        self,
        subject_template: str,
        body_template: str,
        format: EmailFormat = EmailFormat.PLAIN,
    ):
        """Initialize template.

        Args:
            subject_template: Subject template string.
            body_template: Body template string.
            format: Email format.
        """
        self.subject_template = subject_template
        self.body_template = body_template
        self.format = format

    def render(self, **variables: Any) -> tuple:
        """Render template with variables.

        Args:
            **variables: Template variables.

        Returns:
            Tuple of (subject, body).
        """
        subject = self.subject_template.format(**variables)
        body = self.body_template.format(**variables)
        return subject, body

    def create_message(
        self,
        to: List[str],
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Attachment]] = None,
        **variables: Any,
    ) -> EmailMessage:
        """Create an email message from template.

        Args:
            to: Recipients.
            cc: CC recipients.
            bcc: BCC recipients.
            attachments: Attachments.
            **variables: Template variables.

        Returns:
            Email message.
        """
        subject, body = self.render(**variables)

        return EmailMessage(
            to=to,
            subject=subject,
            body=body,
            format=self.format,
            cc=cc or [],
            bcc=bcc or [],
            attachments=attachments or [],
        )


class SmtpSender:
    """SMTP email sender."""

    def __init__(self, config: SmtpConfig):
        """Initialize sender.

        Args:
            config: SMTP configuration.
        """
        self.config = config

    def _create_mime_message(
        self,
        message: EmailMessage,
        from_address: str,
    ) -> MIMEMultipart:
        """Create MIME message from email message.

        Args:
            message: Email message.
            from_address: Sender address.

        Returns:
            MIME message.
        """
        if message.attachments:
            mime = MIMEMultipart("mixed")
        elif message.format == EmailFormat.HTML:
            mime = MIMEMultipart("alternative")
        else:
            mime = MIMEMultipart()

        # Set headers
        mime["Subject"] = message.subject
        mime["From"] = from_address
        mime["To"] = ", ".join(message.to)

        if message.cc:
            mime["Cc"] = ", ".join(message.cc)

        if message.reply_to:
            mime["Reply-To"] = message.reply_to

        for name, value in message.headers.items():
            mime[name] = value

        # Add body
        body_part = MIMEText(
            message.body,
            "html" if message.format == EmailFormat.HTML else "plain",
            "utf-8",
        )
        mime.attach(body_part)

        # Add attachments
        for attachment in message.attachments:
            main_type, sub_type = (
                attachment.mime_type.split("/", 1)
                if attachment.mime_type
                else ("application", "octet-stream")
            )

            part = MIMEBase(main_type, sub_type)
            part.set_payload(attachment.content)

            from email import encoders

            encoders.encode_base64(part)

            if attachment.content_id:
                part.add_header("Content-ID", f"<{attachment.content_id}>")
                part.add_header(
                    "Content-Disposition",
                    "inline",
                    filename=attachment.filename,
                )
            else:
                part.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=attachment.filename,
                )

            mime.attach(part)

        return mime

    def _get_connection(self) -> smtplib.SMTP:
        """Get SMTP connection.

        Returns:
            SMTP connection.
        """
        if self.config.security == SmtpSecurity.SSL:
            smtp = smtplib.SMTP_SSL(
                self.config.host,
                self.config.port,
                timeout=self.config.timeout,
            )
        else:
            smtp = smtplib.SMTP(
                self.config.host,
                self.config.port,
                timeout=self.config.timeout,
            )

            if self.config.security == SmtpSecurity.STARTTLS:
                smtp.starttls()

        if self.config.username and self.config.password:
            smtp.login(self.config.username, self.config.password)

        return smtp

    def send(
        self,
        message: EmailMessage,
        from_address: Optional[str] = None,
    ) -> SendResult:
        """Send an email.

        Args:
            message: Email message to send.
            from_address: Override from address.

        Returns:
            Send result.
        """
        sender = from_address or self.config.from_header

        if not sender:
            return SendResult(
                success=False,
                error="No from address specified",
            )

        if not message.all_recipients:
            return SendResult(
                success=False,
                error="No recipients specified",
            )

        try:
            mime = self._create_mime_message(message, sender)
            message_id = mime["Message-ID"]

            with self._get_connection() as smtp:
                rejected = smtp.sendmail(
                    sender,
                    message.all_recipients,
                    mime.as_string(),
                )

                accepted = [
                    r for r in message.all_recipients if r not in rejected
                ]

                return SendResult(
                    success=len(rejected) == 0,
                    message_id=message_id,
                    recipients_accepted=accepted,
                    recipients_rejected=rejected,
                )

        except smtplib.SMTPAuthenticationError as e:
            return SendResult(
                success=False,
                error=f"Authentication failed: {e}",
            )
        except smtplib.SMTPRecipientsRefused as e:
            return SendResult(
                success=False,
                recipients_rejected={str(k): str(v) for k, v in e.recipients.items()},
                error="All recipients rejected",
            )
        except smtplib.SMTPException as e:
            return SendResult(
                success=False,
                error=f"SMTP error: {e}",
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
            )

    def send_template(
        self,
        template: EmailTemplate,
        to: List[str],
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Attachment]] = None,
        from_address: Optional[str] = None,
        **variables: Any,
    ) -> SendResult:
        """Send email using template.

        Args:
            template: Email template.
            to: Recipients.
            cc: CC recipients.
            bcc: BCC recipients.
            attachments: Attachments.
            from_address: Override from address.
            **variables: Template variables.

        Returns:
            Send result.
        """
        message = template.create_message(
            to=to,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            **variables,
        )
        return self.send(message, from_address)


class EmailManager:
    """Manager for email senders."""

    def __init__(self):
        """Initialize manager."""
        self._senders: Dict[str, SmtpSender] = {}
        self._templates: Dict[str, EmailTemplate] = {}

    def add_sender(self, name: str, sender: SmtpSender) -> None:
        """Add an email sender.

        Args:
            name: Sender name.
            sender: SMTP sender.
        """
        self._senders[name] = sender

    def get_sender(self, name: str) -> Optional[SmtpSender]:
        """Get an email sender.

        Args:
            name: Sender name.

        Returns:
            SMTP sender or None.
        """
        return self._senders.get(name)

    def remove_sender(self, name: str) -> bool:
        """Remove an email sender.

        Args:
            name: Sender name.

        Returns:
            True if removed.
        """
        if name in self._senders:
            del self._senders[name]
            return True
        return False

    def list_senders(self) -> List[str]:
        """List all sender names."""
        return list(self._senders.keys())

    def add_template(self, name: str, template: EmailTemplate) -> None:
        """Add an email template.

        Args:
            name: Template name.
            template: Email template.
        """
        self._templates[name] = template

    def get_template(self, name: str) -> Optional[EmailTemplate]:
        """Get an email template.

        Args:
            name: Template name.

        Returns:
            Email template or None.
        """
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self._templates.keys())


# Pydantic models for tool inputs/outputs


class SendEmailInput(BaseModel):
    """Input for sending email."""

    sender: str = Field(
        default="default",
        description="Name of the email sender to use",
    )
    to: List[str] = Field(
        ...,
        description="List of recipient email addresses",
    )
    subject: str = Field(
        ...,
        description="Email subject",
    )
    body: str = Field(
        ...,
        description="Email body content",
    )
    format: str = Field(
        default="plain",
        description="Email format (plain or html)",
    )
    cc: Optional[List[str]] = Field(
        default=None,
        description="CC recipients",
    )
    bcc: Optional[List[str]] = Field(
        default=None,
        description="BCC recipients",
    )
    reply_to: Optional[str] = Field(
        default=None,
        description="Reply-To address",
    )


class SendEmailOutput(BaseModel):
    """Output from sending email."""

    success: bool = Field(description="Whether email was sent successfully")
    message_id: Optional[str] = Field(
        default=None,
        description="Message ID if sent",
    )
    recipients_accepted: List[str] = Field(
        default_factory=list,
        description="Recipients that accepted the email",
    )
    recipients_rejected: Dict[str, str] = Field(
        default_factory=dict,
        description="Recipients that rejected the email",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )


class SendTemplateInput(BaseModel):
    """Input for sending templated email."""

    sender: str = Field(
        default="default",
        description="Name of the email sender to use",
    )
    template: str = Field(
        ...,
        description="Name of the template to use",
    )
    to: List[str] = Field(
        ...,
        description="List of recipient email addresses",
    )
    cc: Optional[List[str]] = Field(
        default=None,
        description="CC recipients",
    )
    bcc: Optional[List[str]] = Field(
        default=None,
        description="BCC recipients",
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template variables",
    )


class CreateSenderInput(BaseModel):
    """Input for creating email sender."""

    name: str = Field(
        ...,
        description="Name for the sender",
    )
    host: str = Field(
        ...,
        description="SMTP server hostname",
    )
    port: int = Field(
        default=587,
        description="SMTP server port",
    )
    username: Optional[str] = Field(
        default=None,
        description="SMTP username",
    )
    password: Optional[str] = Field(
        default=None,
        description="SMTP password",
    )
    security: str = Field(
        default="starttls",
        description="Security type (none, starttls, ssl)",
    )
    from_address: str = Field(
        ...,
        description="Default from email address",
    )
    from_name: Optional[str] = Field(
        default=None,
        description="Default from name",
    )


class CreateSenderOutput(BaseModel):
    """Output from creating email sender."""

    success: bool = Field(description="Whether sender was created")
    name: str = Field(description="Sender name")
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )


class CreateTemplateInput(BaseModel):
    """Input for creating email template."""

    name: str = Field(
        ...,
        description="Name for the template",
    )
    subject_template: str = Field(
        ...,
        description="Subject template with {variable} placeholders",
    )
    body_template: str = Field(
        ...,
        description="Body template with {variable} placeholders",
    )
    format: str = Field(
        default="plain",
        description="Email format (plain or html)",
    )


class CreateTemplateOutput(BaseModel):
    """Output from creating email template."""

    success: bool = Field(description="Whether template was created")
    name: str = Field(description="Template name")
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )


# Tool implementations


class SendEmailTool(BaseTool[SendEmailInput, SendEmailOutput]):
    """Tool for sending emails."""

    metadata = ToolMetadata(
        id="send_email",
        name="Send Email",
        description="Send an email message",
        category="utility",
    )
    input_type = SendEmailInput
    output_type = SendEmailOutput

    def __init__(self, manager: EmailManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SendEmailInput) -> SendEmailOutput:
        """Send an email."""
        sender = self.manager.get_sender(input.sender)

        if not sender:
            return SendEmailOutput(
                success=False,
                error=f"Sender '{input.sender}' not found",
            )

        try:
            email_format = EmailFormat(input.format.lower())
        except ValueError:
            email_format = EmailFormat.PLAIN

        message = EmailMessage(
            to=input.to,
            subject=input.subject,
            body=input.body,
            format=email_format,
            cc=input.cc or [],
            bcc=input.bcc or [],
            reply_to=input.reply_to,
        )

        result = sender.send(message)

        return SendEmailOutput(
            success=result.success,
            message_id=result.message_id,
            recipients_accepted=result.recipients_accepted,
            recipients_rejected=result.recipients_rejected,
            error=result.error,
        )


class SendTemplateTool(BaseTool[SendTemplateInput, SendEmailOutput]):
    """Tool for sending templated emails."""

    metadata = ToolMetadata(
        id="send_template_email",
        name="Send Template Email",
        description="Send an email using a template",
        category="utility",
    )
    input_type = SendTemplateInput
    output_type = SendEmailOutput

    def __init__(self, manager: EmailManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SendTemplateInput) -> SendEmailOutput:
        """Send a templated email."""
        sender = self.manager.get_sender(input.sender)

        if not sender:
            return SendEmailOutput(
                success=False,
                error=f"Sender '{input.sender}' not found",
            )

        template = self.manager.get_template(input.template)

        if not template:
            return SendEmailOutput(
                success=False,
                error=f"Template '{input.template}' not found",
            )

        try:
            result = sender.send_template(
                template=template,
                to=input.to,
                cc=input.cc,
                bcc=input.bcc,
                **input.variables,
            )

            return SendEmailOutput(
                success=result.success,
                message_id=result.message_id,
                recipients_accepted=result.recipients_accepted,
                recipients_rejected=result.recipients_rejected,
                error=result.error,
            )

        except KeyError as e:
            return SendEmailOutput(
                success=False,
                error=f"Missing template variable: {e}",
            )


class CreateEmailSenderTool(BaseTool[CreateSenderInput, CreateSenderOutput]):
    """Tool for creating email senders."""

    metadata = ToolMetadata(
        id="create_email_sender",
        name="Create Email Sender",
        description="Create a new email sender configuration",
        category="utility",
    )
    input_type = CreateSenderInput
    output_type = CreateSenderOutput

    def __init__(self, manager: EmailManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateSenderInput) -> CreateSenderOutput:
        """Create an email sender."""
        try:
            security = SmtpSecurity(input.security.lower())
        except ValueError:
            return CreateSenderOutput(
                success=False,
                name=input.name,
                error=f"Invalid security type: {input.security}",
            )

        config = SmtpConfig(
            host=input.host,
            port=input.port,
            username=input.username,
            password=input.password,
            security=security,
            from_address=input.from_address,
            from_name=input.from_name,
        )

        sender = SmtpSender(config)
        self.manager.add_sender(input.name, sender)

        return CreateSenderOutput(
            success=True,
            name=input.name,
        )


class CreateEmailTemplateTool(BaseTool[CreateTemplateInput, CreateTemplateOutput]):
    """Tool for creating email templates."""

    metadata = ToolMetadata(
        id="create_email_template",
        name="Create Email Template",
        description="Create a new email template",
        category="utility",
    )
    input_type = CreateTemplateInput
    output_type = CreateTemplateOutput

    def __init__(self, manager: EmailManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateTemplateInput) -> CreateTemplateOutput:
        """Create an email template."""
        try:
            email_format = EmailFormat(input.format.lower())
        except ValueError:
            email_format = EmailFormat.PLAIN

        template = EmailTemplate(
            subject_template=input.subject_template,
            body_template=input.body_template,
            format=email_format,
        )

        self.manager.add_template(input.name, template)

        return CreateTemplateOutput(
            success=True,
            name=input.name,
        )


# Convenience functions


def create_smtp_config(
    host: str = "localhost",
    port: int = 587,
    username: Optional[str] = None,
    password: Optional[str] = None,
    security: SmtpSecurity = SmtpSecurity.STARTTLS,
    from_address: Optional[str] = None,
    from_name: Optional[str] = None,
) -> SmtpConfig:
    """Create SMTP configuration.

    Args:
        host: SMTP server host.
        port: SMTP server port.
        username: Username.
        password: Password.
        security: Security type.
        from_address: From address.
        from_name: From name.

    Returns:
        SMTP configuration.
    """
    return SmtpConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        security=security,
        from_address=from_address,
        from_name=from_name,
    )


def create_smtp_sender(config: SmtpConfig) -> SmtpSender:
    """Create SMTP sender.

    Args:
        config: SMTP configuration.

    Returns:
        SMTP sender.
    """
    return SmtpSender(config)


def create_email_manager() -> EmailManager:
    """Create email manager.

    Returns:
        Email manager.
    """
    return EmailManager()


def create_email_tools(manager: EmailManager) -> Dict[str, BaseTool]:
    """Create all email tools.

    Args:
        manager: Email manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "send_email": SendEmailTool(manager),
        "send_template_email": SendTemplateTool(manager),
        "create_email_sender": CreateEmailSenderTool(manager),
        "create_email_template": CreateEmailTemplateTool(manager),
    }
