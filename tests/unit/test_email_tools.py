"""Tests for email tools."""

import base64
import pytest
from unittest.mock import MagicMock, patch
import smtplib

from tinyllm.tools.email import (
    Attachment,
    CreateEmailSenderTool,
    CreateEmailTemplateTool,
    CreateSenderInput,
    CreateSenderOutput,
    CreateTemplateInput,
    CreateTemplateOutput,
    EmailFormat,
    EmailManager,
    EmailMessage,
    EmailTemplate,
    SendEmailInput,
    SendEmailOutput,
    SendEmailTool,
    SendResult,
    SendTemplateInput,
    SendTemplateTool,
    SmtpConfig,
    SmtpSecurity,
    SmtpSender,
    create_email_manager,
    create_email_tools,
    create_smtp_config,
    create_smtp_sender,
)


class TestSmtpConfig:
    """Tests for SmtpConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SmtpConfig()

        assert config.host == "localhost"
        assert config.port == 587
        assert config.security == SmtpSecurity.STARTTLS

    def test_custom_config(self):
        """Test custom configuration."""
        config = SmtpConfig(
            host="smtp.example.com",
            port=465,
            username="user",
            password="pass",
            security=SmtpSecurity.SSL,
            from_address="sender@example.com",
            from_name="Test Sender",
        )

        assert config.host == "smtp.example.com"
        assert config.port == 465
        assert config.from_address == "sender@example.com"

    def test_from_header(self):
        """Test from header generation."""
        config = SmtpConfig(
            from_address="sender@example.com",
            from_name="Test Sender",
        )

        assert config.from_header == "Test Sender <sender@example.com>"

    def test_from_header_no_name(self):
        """Test from header without name."""
        config = SmtpConfig(from_address="sender@example.com")

        assert config.from_header == "sender@example.com"


class TestAttachment:
    """Tests for Attachment."""

    def test_creation(self):
        """Test attachment creation."""
        attachment = Attachment(
            filename="test.txt",
            content=b"Hello, World!",
            mime_type="text/plain",
        )

        assert attachment.filename == "test.txt"
        assert attachment.content == b"Hello, World!"

    def test_from_base64(self):
        """Test creating from base64."""
        content = base64.b64encode(b"Test content").decode()

        attachment = Attachment.from_base64(
            filename="test.txt",
            base64_content=content,
            mime_type="text/plain",
        )

        assert attachment.content == b"Test content"
        assert attachment.mime_type == "text/plain"

    def test_from_base64_infer_mime(self):
        """Test inferring MIME type from filename."""
        content = base64.b64encode(b"{}").decode()

        attachment = Attachment.from_base64(
            filename="data.json",
            base64_content=content,
        )

        assert "json" in attachment.mime_type


class TestEmailMessage:
    """Tests for EmailMessage."""

    def test_creation(self):
        """Test message creation."""
        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test Subject",
            body="Test body",
        )

        assert message.to == ["recipient@example.com"]
        assert message.subject == "Test Subject"
        assert message.format == EmailFormat.PLAIN

    def test_all_recipients(self):
        """Test getting all recipients."""
        message = EmailMessage(
            to=["to@example.com"],
            subject="Test",
            body="Test",
            cc=["cc@example.com"],
            bcc=["bcc@example.com"],
        )

        all_recipients = message.all_recipients

        assert "to@example.com" in all_recipients
        assert "cc@example.com" in all_recipients
        assert "bcc@example.com" in all_recipients

    def test_html_format(self):
        """Test HTML format."""
        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test",
            body="<h1>Hello</h1>",
            format=EmailFormat.HTML,
        )

        assert message.format == EmailFormat.HTML


class TestSendResult:
    """Tests for SendResult."""

    def test_success_result(self):
        """Test successful result."""
        result = SendResult(
            success=True,
            message_id="<123@example.com>",
            recipients_accepted=["recipient@example.com"],
        )

        assert result.success is True
        assert len(result.recipients_accepted) == 1

    def test_failure_result(self):
        """Test failure result."""
        result = SendResult(
            success=False,
            error="Connection refused",
        )

        assert result.success is False
        assert result.error == "Connection refused"

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = SendResult(
            success=True,
            message_id="<123@example.com>",
            recipients_accepted=["test@example.com"],
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["message_id"] == "<123@example.com>"


class TestEmailTemplate:
    """Tests for EmailTemplate."""

    def test_creation(self):
        """Test template creation."""
        template = EmailTemplate(
            subject_template="Hello {name}",
            body_template="Dear {name}, welcome to {service}!",
        )

        assert "{name}" in template.subject_template
        assert template.format == EmailFormat.PLAIN

    def test_render(self):
        """Test template rendering."""
        template = EmailTemplate(
            subject_template="Hello {name}",
            body_template="Welcome, {name}!",
        )

        subject, body = template.render(name="John")

        assert subject == "Hello John"
        assert body == "Welcome, John!"

    def test_create_message(self):
        """Test creating message from template."""
        template = EmailTemplate(
            subject_template="Order #{order_id}",
            body_template="Your order #{order_id} has shipped.",
        )

        message = template.create_message(
            to=["customer@example.com"],
            order_id="12345",
        )

        assert message.subject == "Order #12345"
        assert "12345" in message.body

    def test_html_template(self):
        """Test HTML template."""
        template = EmailTemplate(
            subject_template="Welcome",
            body_template="<h1>Hello {name}</h1>",
            format=EmailFormat.HTML,
        )

        message = template.create_message(
            to=["user@example.com"],
            name="John",
        )

        assert message.format == EmailFormat.HTML
        assert "<h1>Hello John</h1>" == message.body


class TestSmtpSender:
    """Tests for SmtpSender."""

    @pytest.fixture
    def sender(self):
        """Create test sender."""
        config = SmtpConfig(
            host="smtp.example.com",
            port=587,
            username="user",
            password="pass",
            from_address="sender@example.com",
        )
        return SmtpSender(config)

    def test_creation(self, sender):
        """Test sender creation."""
        assert sender.config.host == "smtp.example.com"

    def test_create_mime_message(self, sender):
        """Test MIME message creation."""
        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test Subject",
            body="Test body",
        )

        mime = sender._create_mime_message(message, "sender@example.com")

        assert mime["Subject"] == "Test Subject"
        assert mime["To"] == "recipient@example.com"
        assert mime["From"] == "sender@example.com"

    def test_create_mime_message_with_cc(self, sender):
        """Test MIME message with CC."""
        message = EmailMessage(
            to=["to@example.com"],
            subject="Test",
            body="Test",
            cc=["cc1@example.com", "cc2@example.com"],
        )

        mime = sender._create_mime_message(message, "sender@example.com")

        assert "cc1@example.com" in mime["Cc"]
        assert "cc2@example.com" in mime["Cc"]

    def test_create_mime_message_with_attachment(self, sender):
        """Test MIME message with attachment."""
        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test",
            body="Test",
            attachments=[
                Attachment(
                    filename="test.txt",
                    content=b"Test content",
                    mime_type="text/plain",
                )
            ],
        )

        mime = sender._create_mime_message(message, "sender@example.com")

        # Check that it has multiple parts (body + attachment)
        assert mime.is_multipart()

    @patch("smtplib.SMTP")
    def test_send_success(self, mock_smtp, sender):
        """Test successful send."""
        mock_instance = MagicMock()
        mock_instance.sendmail.return_value = {}
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test",
            body="Test body",
        )

        result = sender.send(message)

        assert result.success is True

    @patch("smtplib.SMTP")
    def test_send_auth_error(self, mock_smtp, sender):
        """Test authentication error."""
        mock_smtp.return_value.starttls.return_value = None
        mock_smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"Authentication failed"
        )

        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test",
            body="Test",
        )

        result = sender.send(message)

        assert result.success is False
        assert "Authentication failed" in result.error

    def test_send_no_from_address(self):
        """Test send without from address."""
        config = SmtpConfig(host="smtp.example.com")
        sender = SmtpSender(config)

        message = EmailMessage(
            to=["recipient@example.com"],
            subject="Test",
            body="Test",
        )

        result = sender.send(message)

        assert result.success is False
        assert "No from address" in result.error

    def test_send_no_recipients(self, sender):
        """Test send without recipients."""
        message = EmailMessage(
            to=[],
            subject="Test",
            body="Test",
        )

        result = sender.send(message)

        assert result.success is False
        assert "No recipients" in result.error


class TestEmailManager:
    """Tests for EmailManager."""

    def test_add_sender(self):
        """Test adding a sender."""
        manager = EmailManager()
        config = SmtpConfig(from_address="test@example.com")
        sender = SmtpSender(config)

        manager.add_sender("default", sender)

        assert manager.get_sender("default") == sender

    def test_get_sender_not_found(self):
        """Test getting non-existent sender."""
        manager = EmailManager()

        sender = manager.get_sender("nonexistent")

        assert sender is None

    def test_remove_sender(self):
        """Test removing a sender."""
        manager = EmailManager()
        config = SmtpConfig(from_address="test@example.com")
        sender = SmtpSender(config)

        manager.add_sender("default", sender)
        removed = manager.remove_sender("default")

        assert removed is True
        assert manager.get_sender("default") is None

    def test_list_senders(self):
        """Test listing senders."""
        manager = EmailManager()
        config = SmtpConfig(from_address="test@example.com")

        manager.add_sender("smtp1", SmtpSender(config))
        manager.add_sender("smtp2", SmtpSender(config))

        senders = manager.list_senders()

        assert "smtp1" in senders
        assert "smtp2" in senders

    def test_add_template(self):
        """Test adding a template."""
        manager = EmailManager()
        template = EmailTemplate(
            subject_template="Welcome",
            body_template="Hello!",
        )

        manager.add_template("welcome", template)

        assert manager.get_template("welcome") == template

    def test_list_templates(self):
        """Test listing templates."""
        manager = EmailManager()

        manager.add_template(
            "welcome",
            EmailTemplate("Welcome", "Hello!"),
        )
        manager.add_template(
            "goodbye",
            EmailTemplate("Goodbye", "See you!"),
        )

        templates = manager.list_templates()

        assert "welcome" in templates
        assert "goodbye" in templates


class TestSendEmailTool:
    """Tests for SendEmailTool."""

    @pytest.fixture
    def setup_manager(self):
        """Set up email manager."""
        manager = EmailManager()
        config = SmtpConfig(
            host="smtp.example.com",
            from_address="sender@example.com",
        )
        manager.add_sender("default", SmtpSender(config))
        return manager

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_send_success(self, mock_smtp, setup_manager):
        """Test successful send."""
        mock_instance = MagicMock()
        mock_instance.sendmail.return_value = {}
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        tool = SendEmailTool(setup_manager)

        result = await tool.execute(
            SendEmailInput(
                sender="default",
                to=["recipient@example.com"],
                subject="Test",
                body="Test body",
            )
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_sender_not_found(self, setup_manager):
        """Test with non-existent sender."""
        tool = SendEmailTool(setup_manager)

        result = await tool.execute(
            SendEmailInput(
                sender="nonexistent",
                to=["recipient@example.com"],
                subject="Test",
                body="Test",
            )
        )

        assert result.success is False
        assert "not found" in result.error


class TestSendTemplateTool:
    """Tests for SendTemplateTool."""

    @pytest.fixture
    def setup_manager(self):
        """Set up email manager with sender and template."""
        manager = EmailManager()
        config = SmtpConfig(
            host="smtp.example.com",
            from_address="sender@example.com",
        )
        manager.add_sender("default", SmtpSender(config))
        manager.add_template(
            "welcome",
            EmailTemplate(
                subject_template="Welcome, {name}!",
                body_template="Hello {name}, welcome to {service}!",
            ),
        )
        return manager

    @pytest.mark.asyncio
    @patch("smtplib.SMTP")
    async def test_send_template_success(self, mock_smtp, setup_manager):
        """Test successful template send."""
        mock_instance = MagicMock()
        mock_instance.sendmail.return_value = {}
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        tool = SendTemplateTool(setup_manager)

        result = await tool.execute(
            SendTemplateInput(
                sender="default",
                template="welcome",
                to=["user@example.com"],
                variables={"name": "John", "service": "TinyLLM"},
            )
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_template_not_found(self, setup_manager):
        """Test with non-existent template."""
        tool = SendTemplateTool(setup_manager)

        result = await tool.execute(
            SendTemplateInput(
                sender="default",
                template="nonexistent",
                to=["user@example.com"],
                variables={},
            )
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_missing_variable(self, setup_manager):
        """Test with missing template variable."""
        tool = SendTemplateTool(setup_manager)

        result = await tool.execute(
            SendTemplateInput(
                sender="default",
                template="welcome",
                to=["user@example.com"],
                variables={"name": "John"},  # Missing 'service'
            )
        )

        assert result.success is False
        assert "Missing template variable" in result.error


class TestCreateEmailSenderTool:
    """Tests for CreateEmailSenderTool."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        return EmailManager()

    @pytest.mark.asyncio
    async def test_create_sender(self, manager):
        """Test creating a sender."""
        tool = CreateEmailSenderTool(manager)

        result = await tool.execute(
            CreateSenderInput(
                name="test",
                host="smtp.example.com",
                port=587,
                from_address="sender@example.com",
            )
        )

        assert result.success is True
        assert manager.get_sender("test") is not None

    @pytest.mark.asyncio
    async def test_create_sender_with_auth(self, manager):
        """Test creating sender with authentication."""
        tool = CreateEmailSenderTool(manager)

        result = await tool.execute(
            CreateSenderInput(
                name="auth_sender",
                host="smtp.example.com",
                username="user",
                password="pass",
                from_address="sender@example.com",
            )
        )

        assert result.success is True

        sender = manager.get_sender("auth_sender")
        assert sender.config.username == "user"

    @pytest.mark.asyncio
    async def test_invalid_security(self, manager):
        """Test with invalid security type."""
        tool = CreateEmailSenderTool(manager)

        result = await tool.execute(
            CreateSenderInput(
                name="bad",
                host="smtp.example.com",
                from_address="sender@example.com",
                security="invalid",
            )
        )

        assert result.success is False
        assert "Invalid security type" in result.error


class TestCreateEmailTemplateTool:
    """Tests for CreateEmailTemplateTool."""

    @pytest.fixture
    def manager(self):
        """Create manager."""
        return EmailManager()

    @pytest.mark.asyncio
    async def test_create_template(self, manager):
        """Test creating a template."""
        tool = CreateEmailTemplateTool(manager)

        result = await tool.execute(
            CreateTemplateInput(
                name="welcome",
                subject_template="Welcome, {name}!",
                body_template="Hello {name}!",
            )
        )

        assert result.success is True
        assert manager.get_template("welcome") is not None

    @pytest.mark.asyncio
    async def test_create_html_template(self, manager):
        """Test creating HTML template."""
        tool = CreateEmailTemplateTool(manager)

        result = await tool.execute(
            CreateTemplateInput(
                name="html_welcome",
                subject_template="Welcome",
                body_template="<h1>Hello {name}</h1>",
                format="html",
            )
        )

        assert result.success is True

        template = manager.get_template("html_welcome")
        assert template.format == EmailFormat.HTML


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_smtp_config(self):
        """Test creating SMTP config."""
        config = create_smtp_config(
            host="smtp.example.com",
            port=465,
            security=SmtpSecurity.SSL,
            from_address="sender@example.com",
        )

        assert config.host == "smtp.example.com"
        assert config.port == 465
        assert config.security == SmtpSecurity.SSL

    def test_create_smtp_sender(self):
        """Test creating SMTP sender."""
        config = create_smtp_config(
            host="smtp.example.com",
            from_address="sender@example.com",
        )

        sender = create_smtp_sender(config)

        assert isinstance(sender, SmtpSender)

    def test_create_email_manager(self):
        """Test creating email manager."""
        manager = create_email_manager()

        assert isinstance(manager, EmailManager)

    def test_create_email_tools(self):
        """Test creating all email tools."""
        manager = create_email_manager()
        tools = create_email_tools(manager)

        assert "send_email" in tools
        assert "send_template_email" in tools
        assert "create_email_sender" in tools
        assert "create_email_template" in tools
