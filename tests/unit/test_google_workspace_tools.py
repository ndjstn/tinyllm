"""Tests for Google Workspace tools."""

import json
from unittest.mock import MagicMock, patch
import pytest

from tinyllm.tools.google_workspace import (
    GoogleService,
    MimeType,
    GoogleConfig,
    GmailMessage,
    DriveFile,
    CalendarEvent,
    GoogleResult,
    GoogleClient,
    GoogleManager,
    CreateGoogleClientInput,
    CreateGoogleClientOutput,
    ListMessagesInput,
    MessageOutput,
    GetMessageInput,
    SendMessageInput,
    ListFilesInput,
    FileOutput,
    GetFileInput,
    CreateFolderInput,
    DeleteFileInput,
    SimpleOutput,
    ListEventsInput,
    EventOutput,
    GetEventInput,
    CreateEventInput,
    DeleteEventInput,
    CreateGoogleClientTool,
    ListMessagesTool,
    GetMessageTool,
    SendMessageTool,
    ListFilesTool,
    GetFileTool,
    CreateFolderTool,
    DeleteFileTool,
    ListEventsTool,
    GetEventTool,
    CreateEventTool,
    DeleteEventTool,
    create_google_config,
    create_google_client,
    create_google_manager,
    create_google_tools,
)


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_google_service_values(self):
        """Test GoogleService enum values."""
        assert GoogleService.GMAIL.value == "gmail"
        assert GoogleService.DRIVE.value == "drive"
        assert GoogleService.CALENDAR.value == "calendar"
        assert GoogleService.DOCS.value == "docs"

    def test_mime_type_values(self):
        """Test MimeType enum values."""
        assert MimeType.FOLDER.value == "application/vnd.google-apps.folder"
        assert MimeType.DOCUMENT.value == "application/vnd.google-apps.document"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestGoogleConfig:
    """Tests for GoogleConfig dataclass."""

    def test_config_with_defaults(self):
        """Test config with default values."""
        config = GoogleConfig(access_token="test-token")

        assert config.access_token == "test-token"
        assert config.timeout == 30

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = GoogleConfig(
            access_token="my-token",
            timeout=60,
        )

        assert config.timeout == 60


class TestGmailMessage:
    """Tests for GmailMessage dataclass."""

    def test_message_from_api_response(self):
        """Test creating GmailMessage from API response."""
        api_data = {
            "id": "msg-123",
            "threadId": "thread-123",
            "snippet": "Hello there...",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Email"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"},
                ],
            },
        }

        message = GmailMessage.from_api_response(api_data)

        assert message.id == "msg-123"
        assert message.thread_id == "thread-123"
        assert message.subject == "Test Email"
        assert message.sender == "sender@example.com"
        assert message.is_read is False  # UNREAD label present
        assert "INBOX" in message.labels

    def test_message_to_dict(self):
        """Test converting GmailMessage to dict."""
        message = GmailMessage(
            id="msg-id",
            thread_id="thread-id",
            subject="Test",
        )

        d = message.to_dict()

        assert d["id"] == "msg-id"
        assert d["subject"] == "Test"


class TestDriveFile:
    """Tests for DriveFile dataclass."""

    def test_file_from_api_response(self):
        """Test creating DriveFile from API response."""
        api_data = {
            "id": "file-123",
            "name": "Document.docx",
            "mimeType": "application/vnd.google-apps.document",
            "parents": ["folder-123"],
            "webViewLink": "https://docs.google.com/document/d/file-123",
            "size": "1024",
            "createdTime": "2024-01-01T00:00:00.000Z",
            "modifiedTime": "2024-01-02T00:00:00.000Z",
            "trashed": False,
        }

        file = DriveFile.from_api_response(api_data)

        assert file.id == "file-123"
        assert file.name == "Document.docx"
        assert file.size == 1024
        assert file.trashed is False

    def test_file_to_dict(self):
        """Test converting DriveFile to dict."""
        file = DriveFile(
            id="file-id",
            name="Test.txt",
            mime_type="text/plain",
        )

        d = file.to_dict()

        assert d["id"] == "file-id"
        assert d["name"] == "Test.txt"


class TestCalendarEvent:
    """Tests for CalendarEvent dataclass."""

    def test_event_from_api_response(self):
        """Test creating CalendarEvent from API response."""
        api_data = {
            "id": "event-123",
            "summary": "Meeting",
            "description": "Team meeting",
            "location": "Conference Room",
            "start": {"dateTime": "2024-01-01T10:00:00Z"},
            "end": {"dateTime": "2024-01-01T11:00:00Z"},
            "attendees": [
                {"email": "person1@example.com"},
                {"email": "person2@example.com"},
            ],
            "organizer": {"email": "organizer@example.com"},
            "htmlLink": "https://calendar.google.com/event/123",
            "status": "confirmed",
        }

        event = CalendarEvent.from_api_response(api_data)

        assert event.id == "event-123"
        assert event.summary == "Meeting"
        assert event.location == "Conference Room"
        assert len(event.attendees) == 2
        assert event.organizer == "organizer@example.com"
        assert event.recurring is False

    def test_event_to_dict(self):
        """Test converting CalendarEvent to dict."""
        event = CalendarEvent(
            id="event-id",
            summary="Test Event",
        )

        d = event.to_dict()

        assert d["id"] == "event-id"
        assert d["summary"] == "Test Event"


class TestGoogleResult:
    """Tests for GoogleResult dataclass."""

    def test_result_success(self):
        """Test successful result."""
        result = GoogleResult(
            success=True,
            data={"id": "123"},
            status_code=200,
        )

        assert result.success is True
        assert result.data == {"id": "123"}

    def test_result_failure(self):
        """Test failed result."""
        result = GoogleResult(
            success=False,
            error="Not authorized",
            status_code=401,
        )

        assert result.success is False
        assert "Not authorized" in result.error


# ============================================================================
# GoogleClient Tests
# ============================================================================


class TestGoogleClient:
    """Tests for GoogleClient class."""

    def test_create_client(self):
        """Test creating a client."""
        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        assert client.config == config

    @patch("urllib.request.urlopen")
    def test_list_messages_success(self, mock_urlopen):
        """Test listing Gmail messages."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "messages": [
                {"id": "msg-1", "threadId": "thread-1"},
                {"id": "msg-2", "threadId": "thread-2"},
            ],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.list_messages(max_results=10)

        assert result.success is True
        assert len(result.data["messages"]) == 2

    @patch("urllib.request.urlopen")
    def test_get_message_success(self, mock_urlopen):
        """Test getting a Gmail message."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "msg-123",
            "threadId": "thread-123",
            "snippet": "Hello",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test"},
                    {"name": "From", "value": "test@example.com"},
                ],
            },
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.get_message("msg-123")

        assert result.success is True
        assert result.data["id"] == "msg-123"

    @patch("urllib.request.urlopen")
    def test_send_message_success(self, mock_urlopen):
        """Test sending a Gmail message."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "sent-msg-123",
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.send_message(
            to="recipient@example.com",
            subject="Test",
            body="Hello",
        )

        assert result.success is True

    @patch("urllib.request.urlopen")
    def test_list_files_success(self, mock_urlopen):
        """Test listing Drive files."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "files": [
                {
                    "id": "file-1",
                    "name": "Document.docx",
                    "mimeType": "application/vnd.google-apps.document",
                },
            ],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.list_files()

        assert result.success is True
        assert len(result.data["files"]) == 1

    @patch("urllib.request.urlopen")
    def test_create_folder_success(self, mock_urlopen):
        """Test creating a Drive folder."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "folder-123",
            "name": "New Folder",
            "mimeType": "application/vnd.google-apps.folder",
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.create_folder("New Folder")

        assert result.success is True
        assert result.data["name"] == "New Folder"

    @patch("urllib.request.urlopen")
    def test_list_events_success(self, mock_urlopen):
        """Test listing Calendar events."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "items": [
                {
                    "id": "event-1",
                    "summary": "Meeting",
                    "start": {"dateTime": "2024-01-01T10:00:00Z"},
                    "end": {"dateTime": "2024-01-01T11:00:00Z"},
                },
            ],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.list_events()

        assert result.success is True
        assert len(result.data["events"]) == 1

    @patch("urllib.request.urlopen")
    def test_create_event_success(self, mock_urlopen):
        """Test creating a Calendar event."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "new-event-123",
            "summary": "New Meeting",
            "start": {"dateTime": "2024-01-01T10:00:00Z"},
            "end": {"dateTime": "2024-01-01T11:00:00Z"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = GoogleConfig(access_token="test-token")
        client = GoogleClient(config)

        result = client.create_event(
            summary="New Meeting",
            start="2024-01-01T10:00:00Z",
            end="2024-01-01T11:00:00Z",
        )

        assert result.success is True
        assert result.data["summary"] == "New Meeting"

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Test handling HTTP errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://gmail.googleapis.com",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"error": {"message": "Invalid token"}}')),
        )

        config = GoogleConfig(access_token="bad-token")
        client = GoogleClient(config)

        result = client.list_messages()

        assert result.success is False
        assert "401" in result.error


# ============================================================================
# GoogleManager Tests
# ============================================================================


class TestGoogleManager:
    """Tests for GoogleManager class."""

    def test_create_manager(self):
        """Test creating manager."""
        manager = GoogleManager()

        assert manager.list_clients() == []

    def test_add_client(self):
        """Test adding a client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test")
        client = GoogleClient(config)

        manager.add_client("default", client)

        assert "default" in manager.list_clients()

    def test_get_client_not_found(self):
        """Test getting non-existent client."""
        manager = GoogleManager()

        result = manager.get_client("nonexistent")

        assert result is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test")
        manager.add_client("test", GoogleClient(config))

        result = manager.remove_client("test")

        assert result is True
        assert manager.get_client("test") is None


# ============================================================================
# Tool Tests
# ============================================================================


class TestCreateGoogleClientTool:
    """Tests for CreateGoogleClientTool."""

    @pytest.mark.asyncio
    async def test_create_client(self):
        """Test creating a Google client."""
        manager = GoogleManager()
        tool = CreateGoogleClientTool(manager)

        input_data = CreateGoogleClientInput(
            name="my-client",
            access_token="test-token",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.name == "my-client"
        assert manager.get_client("my-client") is not None


class TestListMessagesTool:
    """Tests for ListMessagesTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test-token")
        manager.add_client("default", GoogleClient(config))
        return manager

    @pytest.mark.asyncio
    async def test_client_not_found(self):
        """Test with non-existent client."""
        manager = GoogleManager()
        tool = ListMessagesTool(manager)

        input_data = ListMessagesInput(
            client="nonexistent",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_list_messages_success(self, mock_urlopen, manager_with_client):
        """Test listing messages."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "messages": [{"id": "msg-1", "threadId": "thread-1"}],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = ListMessagesTool(manager_with_client)

        input_data = ListMessagesInput()

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.messages) == 1


class TestListFilesTool:
    """Tests for ListFilesTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test-token")
        manager.add_client("default", GoogleClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_list_files_success(self, mock_urlopen, manager_with_client):
        """Test listing files."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "files": [
                {
                    "id": "file-1",
                    "name": "Doc.txt",
                    "mimeType": "text/plain",
                },
            ],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = ListFilesTool(manager_with_client)

        input_data = ListFilesInput()

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.files) == 1


class TestCreateFolderTool:
    """Tests for CreateFolderTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test-token")
        manager.add_client("default", GoogleClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_create_folder_success(self, mock_urlopen, manager_with_client):
        """Test creating a folder."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "folder-123",
            "name": "New Folder",
            "mimeType": "application/vnd.google-apps.folder",
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = CreateFolderTool(manager_with_client)

        input_data = CreateFolderInput(name="New Folder")

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.file["name"] == "New Folder"


class TestListEventsTool:
    """Tests for ListEventsTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test-token")
        manager.add_client("default", GoogleClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_list_events_success(self, mock_urlopen, manager_with_client):
        """Test listing events."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "items": [
                {
                    "id": "event-1",
                    "summary": "Meeting",
                    "start": {"dateTime": "2024-01-01T10:00:00Z"},
                    "end": {"dateTime": "2024-01-01T11:00:00Z"},
                },
            ],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = ListEventsTool(manager_with_client)

        input_data = ListEventsInput()

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.events) == 1


class TestCreateEventTool:
    """Tests for CreateEventTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = GoogleManager()
        config = GoogleConfig(access_token="test-token")
        manager.add_client("default", GoogleClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_create_event_success(self, mock_urlopen, manager_with_client):
        """Test creating an event."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "new-event",
            "summary": "Team Sync",
            "start": {"dateTime": "2024-01-01T10:00:00Z"},
            "end": {"dateTime": "2024-01-01T11:00:00Z"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = CreateEventTool(manager_with_client)

        input_data = CreateEventInput(
            summary="Team Sync",
            start="2024-01-01T10:00:00Z",
            end="2024-01-01T11:00:00Z",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.event["summary"] == "Team Sync"


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_google_config(self):
        """Test create_google_config function."""
        config = create_google_config(
            access_token="my-token",
            timeout=60,
        )

        assert config.access_token == "my-token"
        assert config.timeout == 60

    def test_create_google_client(self):
        """Test create_google_client function."""
        config = GoogleConfig(access_token="test")
        client = create_google_client(config)

        assert isinstance(client, GoogleClient)

    def test_create_google_manager(self):
        """Test create_google_manager function."""
        manager = create_google_manager()

        assert isinstance(manager, GoogleManager)

    def test_create_google_tools(self):
        """Test create_google_tools function."""
        manager = GoogleManager()
        tools = create_google_tools(manager)

        # Check client tool
        assert "create_google_client" in tools

        # Check Gmail tools
        assert "list_gmail_messages" in tools
        assert "get_gmail_message" in tools
        assert "send_gmail_message" in tools

        # Check Drive tools
        assert "list_drive_files" in tools
        assert "get_drive_file" in tools
        assert "create_drive_folder" in tools
        assert "delete_drive_file" in tools

        # Check Calendar tools
        assert "list_calendar_events" in tools
        assert "get_calendar_event" in tools
        assert "create_calendar_event" in tools
        assert "delete_calendar_event" in tools
