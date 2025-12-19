"""Google Workspace tools for TinyLLM.

This module provides tools for interacting with Google Workspace APIs
including Gmail, Drive, Calendar, and Docs.
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class GoogleService(str, Enum):
    """Google Workspace services."""

    GMAIL = "gmail"
    DRIVE = "drive"
    CALENDAR = "calendar"
    DOCS = "docs"
    SHEETS = "sheets"


class MimeType(str, Enum):
    """Common MIME types for Google Drive."""

    FOLDER = "application/vnd.google-apps.folder"
    DOCUMENT = "application/vnd.google-apps.document"
    SPREADSHEET = "application/vnd.google-apps.spreadsheet"
    PRESENTATION = "application/vnd.google-apps.presentation"
    FORM = "application/vnd.google-apps.form"
    PDF = "application/pdf"
    TEXT = "text/plain"


@dataclass
class GoogleConfig:
    """Google API configuration."""

    access_token: str
    timeout: int = 30


@dataclass
class GmailMessage:
    """Gmail message representation."""

    id: str
    thread_id: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    snippet: Optional[str] = None
    body: Optional[str] = None
    date: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    is_read: bool = True
    has_attachments: bool = False

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "GmailMessage":
        """Create GmailMessage from Gmail API response."""
        payload = data.get("payload", {})
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

        # Extract body
        body = None
        parts = payload.get("parts", [])
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    body_data = part.get("body", {}).get("data", "")
                    if body_data:
                        import base64
                        body = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                    break
        elif payload.get("body", {}).get("data"):
            import base64
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

        # Check for attachments
        has_attachments = any(p.get("filename") for p in parts) if parts else False

        label_ids = data.get("labelIds", [])

        return cls(
            id=data["id"],
            thread_id=data.get("threadId", ""),
            subject=headers.get("subject"),
            sender=headers.get("from"),
            recipients=headers.get("to", "").split(",") if headers.get("to") else [],
            snippet=data.get("snippet"),
            body=body,
            date=headers.get("date"),
            labels=label_ids,
            is_read="UNREAD" not in label_ids,
            has_attachments=has_attachments,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "recipients": self.recipients,
            "snippet": self.snippet,
            "body": self.body,
            "date": self.date,
            "labels": self.labels,
            "is_read": self.is_read,
            "has_attachments": self.has_attachments,
        }


@dataclass
class DriveFile:
    """Google Drive file representation."""

    id: str
    name: str
    mime_type: str
    parents: List[str] = field(default_factory=list)
    web_view_link: Optional[str] = None
    web_content_link: Optional[str] = None
    size: Optional[int] = None
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    trashed: bool = False

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DriveFile":
        """Create DriveFile from Drive API response."""
        size = data.get("size")
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            mime_type=data.get("mimeType", ""),
            parents=data.get("parents", []),
            web_view_link=data.get("webViewLink"),
            web_content_link=data.get("webContentLink"),
            size=int(size) if size else None,
            created_time=data.get("createdTime"),
            modified_time=data.get("modifiedTime"),
            trashed=data.get("trashed", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "parents": self.parents,
            "web_view_link": self.web_view_link,
            "web_content_link": self.web_content_link,
            "size": self.size,
            "created_time": self.created_time,
            "modified_time": self.modified_time,
            "trashed": self.trashed,
        }


@dataclass
class CalendarEvent:
    """Google Calendar event representation."""

    id: str
    summary: str
    description: Optional[str] = None
    location: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    organizer: Optional[str] = None
    html_link: Optional[str] = None
    status: str = "confirmed"
    recurring: bool = False

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "CalendarEvent":
        """Create CalendarEvent from Calendar API response."""
        start = data.get("start", {})
        end = data.get("end", {})

        # Handle both dateTime and date formats
        start_time = start.get("dateTime") or start.get("date")
        end_time = end.get("dateTime") or end.get("date")

        attendees = [a.get("email", "") for a in data.get("attendees", [])]
        organizer = data.get("organizer", {}).get("email")

        return cls(
            id=data["id"],
            summary=data.get("summary", ""),
            description=data.get("description"),
            location=data.get("location"),
            start=start_time,
            end=end_time,
            attendees=attendees,
            organizer=organizer,
            html_link=data.get("htmlLink"),
            status=data.get("status", "confirmed"),
            recurring="recurringEventId" in data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "summary": self.summary,
            "description": self.description,
            "location": self.location,
            "start": self.start,
            "end": self.end,
            "attendees": self.attendees,
            "organizer": self.organizer,
            "html_link": self.html_link,
            "status": self.status,
            "recurring": self.recurring,
        }


@dataclass
class GoogleResult:
    """Result from Google API operation."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
        }


class GoogleClient:
    """Client for Google APIs."""

    BASE_URLS = {
        GoogleService.GMAIL: "https://gmail.googleapis.com/gmail/v1",
        GoogleService.DRIVE: "https://www.googleapis.com/drive/v3",
        GoogleService.CALENDAR: "https://www.googleapis.com/calendar/v3",
        GoogleService.DOCS: "https://docs.googleapis.com/v1",
        GoogleService.SHEETS: "https://sheets.googleapis.com/v4",
    }

    def __init__(self, config: GoogleConfig):
        """Initialize client.

        Args:
            config: Google configuration.
        """
        self.config = config

    def _make_request(
        self,
        method: str,
        service: GoogleService,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> GoogleResult:
        """Make HTTP request to Google API.

        Args:
            method: HTTP method.
            service: Google service.
            endpoint: API endpoint.
            data: Request body data.
            params: Query parameters.

        Returns:
            Google result.
        """
        base_url = self.BASE_URLS[service]
        url = f"{base_url}{endpoint}"

        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query_string:
                url = f"{url}?{query_string}"

        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }

        request_data = json.dumps(data).encode("utf-8") if data else None

        try:
            req = urllib.request.Request(
                url,
                data=request_data,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                response_body = response.read().decode("utf-8")
                response_data = json.loads(response_body) if response_body else None

                return GoogleResult(
                    success=True,
                    data=response_data,
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
                error_data = json.loads(error_body)
                error_message = error_data.get("error", {}).get("message", error_body)
            except Exception:
                error_message = error_body or str(e)

            return GoogleResult(
                success=False,
                error=f"HTTP {e.code}: {error_message}",
                status_code=e.code,
            )
        except urllib.error.URLError as e:
            return GoogleResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )
        except Exception as e:
            return GoogleResult(
                success=False,
                error=str(e),
            )

    # Gmail operations

    def list_messages(
        self,
        query: Optional[str] = None,
        max_results: int = 10,
        label_ids: Optional[List[str]] = None,
    ) -> GoogleResult:
        """List Gmail messages.

        Args:
            query: Search query.
            max_results: Maximum results.
            label_ids: Filter by labels.

        Returns:
            Google result with message list.
        """
        params = {"maxResults": str(max_results)}

        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = ",".join(label_ids)

        result = self._make_request(
            "GET",
            GoogleService.GMAIL,
            "/users/me/messages",
            params=params,
        )

        if result.success and result.data:
            result.data = {
                "messages": [
                    {"id": m["id"], "thread_id": m.get("threadId", "")}
                    for m in result.data.get("messages", [])
                ],
                "next_page_token": result.data.get("nextPageToken"),
            }

        return result

    def get_message(self, message_id: str) -> GoogleResult:
        """Get a Gmail message.

        Args:
            message_id: Message ID.

        Returns:
            Google result with message data.
        """
        result = self._make_request(
            "GET",
            GoogleService.GMAIL,
            f"/users/me/messages/{message_id}",
        )

        if result.success and result.data:
            result.data = GmailMessage.from_api_response(result.data).to_dict()

        return result

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
    ) -> GoogleResult:
        """Send a Gmail message.

        Args:
            to: Recipient email.
            subject: Email subject.
            body: Email body.
            cc: CC recipients.
            bcc: BCC recipients.

        Returns:
            Google result.
        """
        import base64

        # Build email message
        message_lines = [
            f"To: {to}",
            f"Subject: {subject}",
        ]

        if cc:
            message_lines.append(f"Cc: {cc}")
        if bcc:
            message_lines.append(f"Bcc: {bcc}")

        message_lines.extend(["Content-Type: text/plain; charset=utf-8", "", body])

        raw_message = "\r\n".join(message_lines)
        encoded = base64.urlsafe_b64encode(raw_message.encode("utf-8")).decode("utf-8")

        return self._make_request(
            "POST",
            GoogleService.GMAIL,
            "/users/me/messages/send",
            data={"raw": encoded},
        )

    # Drive operations

    def list_files(
        self,
        query: Optional[str] = None,
        folder_id: Optional[str] = None,
        max_results: int = 100,
    ) -> GoogleResult:
        """List Drive files.

        Args:
            query: Search query.
            folder_id: Parent folder ID.
            max_results: Maximum results.

        Returns:
            Google result with file list.
        """
        params = {
            "pageSize": str(max_results),
            "fields": "files(id,name,mimeType,parents,webViewLink,size,createdTime,modifiedTime,trashed)",
        }

        q_parts = []
        if query:
            q_parts.append(query)
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")

        if q_parts:
            params["q"] = " and ".join(q_parts)

        result = self._make_request(
            "GET",
            GoogleService.DRIVE,
            "/files",
            params=params,
        )

        if result.success and result.data:
            result.data = {
                "files": [
                    DriveFile.from_api_response(f).to_dict()
                    for f in result.data.get("files", [])
                ],
            }

        return result

    def get_file(self, file_id: str) -> GoogleResult:
        """Get Drive file metadata.

        Args:
            file_id: File ID.

        Returns:
            Google result with file data.
        """
        params = {
            "fields": "id,name,mimeType,parents,webViewLink,webContentLink,size,createdTime,modifiedTime,trashed",
        }

        result = self._make_request(
            "GET",
            GoogleService.DRIVE,
            f"/files/{file_id}",
            params=params,
        )

        if result.success and result.data:
            result.data = DriveFile.from_api_response(result.data).to_dict()

        return result

    def create_folder(
        self,
        name: str,
        parent_id: Optional[str] = None,
    ) -> GoogleResult:
        """Create a Drive folder.

        Args:
            name: Folder name.
            parent_id: Parent folder ID.

        Returns:
            Google result with created folder.
        """
        metadata: Dict[str, Any] = {
            "name": name,
            "mimeType": MimeType.FOLDER.value,
        }

        if parent_id:
            metadata["parents"] = [parent_id]

        result = self._make_request(
            "POST",
            GoogleService.DRIVE,
            "/files",
            data=metadata,
        )

        if result.success and result.data:
            result.data = DriveFile.from_api_response(result.data).to_dict()

        return result

    def delete_file(self, file_id: str) -> GoogleResult:
        """Delete a Drive file.

        Args:
            file_id: File ID.

        Returns:
            Google result.
        """
        return self._make_request(
            "DELETE",
            GoogleService.DRIVE,
            f"/files/{file_id}",
        )

    # Calendar operations

    def list_events(
        self,
        calendar_id: str = "primary",
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 100,
    ) -> GoogleResult:
        """List Calendar events.

        Args:
            calendar_id: Calendar ID.
            time_min: Start time (RFC3339).
            time_max: End time (RFC3339).
            max_results: Maximum results.

        Returns:
            Google result with event list.
        """
        params = {"maxResults": str(max_results), "singleEvents": "true"}

        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max

        result = self._make_request(
            "GET",
            GoogleService.CALENDAR,
            f"/calendars/{calendar_id}/events",
            params=params,
        )

        if result.success and result.data:
            result.data = {
                "events": [
                    CalendarEvent.from_api_response(e).to_dict()
                    for e in result.data.get("items", [])
                ],
            }

        return result

    def get_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> GoogleResult:
        """Get a Calendar event.

        Args:
            event_id: Event ID.
            calendar_id: Calendar ID.

        Returns:
            Google result with event data.
        """
        result = self._make_request(
            "GET",
            GoogleService.CALENDAR,
            f"/calendars/{calendar_id}/events/{event_id}",
        )

        if result.success and result.data:
            result.data = CalendarEvent.from_api_response(result.data).to_dict()

        return result

    def create_event(
        self,
        summary: str,
        start: str,
        end: str,
        calendar_id: str = "primary",
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
    ) -> GoogleResult:
        """Create a Calendar event.

        Args:
            summary: Event title.
            start: Start time (RFC3339).
            end: End time (RFC3339).
            calendar_id: Calendar ID.
            description: Event description.
            location: Event location.
            attendees: Attendee emails.

        Returns:
            Google result with created event.
        """
        event: Dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }

        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if attendees:
            event["attendees"] = [{"email": a} for a in attendees]

        result = self._make_request(
            "POST",
            GoogleService.CALENDAR,
            f"/calendars/{calendar_id}/events",
            data=event,
        )

        if result.success and result.data:
            result.data = CalendarEvent.from_api_response(result.data).to_dict()

        return result

    def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> GoogleResult:
        """Delete a Calendar event.

        Args:
            event_id: Event ID.
            calendar_id: Calendar ID.

        Returns:
            Google result.
        """
        return self._make_request(
            "DELETE",
            GoogleService.CALENDAR,
            f"/calendars/{calendar_id}/events/{event_id}",
        )


class GoogleManager:
    """Manager for Google clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, GoogleClient] = {}

    def add_client(self, name: str, client: GoogleClient) -> None:
        """Add a Google client.

        Args:
            name: Client name.
            client: Google client.
        """
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[GoogleClient]:
        """Get a Google client.

        Args:
            name: Client name.

        Returns:
            Google client or None.
        """
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove a Google client.

        Args:
            name: Client name.

        Returns:
            True if removed.
        """
        if name in self._clients:
            del self._clients[name]
            return True
        return False

    def list_clients(self) -> List[str]:
        """List all client names."""
        return list(self._clients.keys())


# Pydantic models for tool inputs/outputs


class CreateGoogleClientInput(BaseModel):
    """Input for creating a Google client."""

    name: str = Field(..., description="Name for the client")
    access_token: str = Field(..., description="OAuth2 access token")


class CreateGoogleClientOutput(BaseModel):
    """Output from creating a Google client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListMessagesInput(BaseModel):
    """Input for listing Gmail messages."""

    client: str = Field(default="default", description="Google client name")
    query: Optional[str] = Field(default=None, description="Search query")
    max_results: int = Field(default=10, description="Maximum results")
    label_ids: Optional[List[str]] = Field(default=None, description="Filter by labels")


class MessageOutput(BaseModel):
    """Output containing message data."""

    success: bool = Field(description="Whether operation succeeded")
    message: Optional[Dict[str, Any]] = Field(default=None, description="Message data")
    messages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Message list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetMessageInput(BaseModel):
    """Input for getting a Gmail message."""

    client: str = Field(default="default", description="Google client name")
    message_id: str = Field(..., description="Message ID")


class SendMessageInput(BaseModel):
    """Input for sending a Gmail message."""

    client: str = Field(default="default", description="Google client name")
    to: str = Field(..., description="Recipient email")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    cc: Optional[str] = Field(default=None, description="CC recipients")
    bcc: Optional[str] = Field(default=None, description="BCC recipients")


class ListFilesInput(BaseModel):
    """Input for listing Drive files."""

    client: str = Field(default="default", description="Google client name")
    query: Optional[str] = Field(default=None, description="Search query")
    folder_id: Optional[str] = Field(default=None, description="Parent folder ID")
    max_results: int = Field(default=100, description="Maximum results")


class FileOutput(BaseModel):
    """Output containing file data."""

    success: bool = Field(description="Whether operation succeeded")
    file: Optional[Dict[str, Any]] = Field(default=None, description="File data")
    files: Optional[List[Dict[str, Any]]] = Field(default=None, description="File list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetFileInput(BaseModel):
    """Input for getting a Drive file."""

    client: str = Field(default="default", description="Google client name")
    file_id: str = Field(..., description="File ID")


class CreateFolderInput(BaseModel):
    """Input for creating a Drive folder."""

    client: str = Field(default="default", description="Google client name")
    name: str = Field(..., description="Folder name")
    parent_id: Optional[str] = Field(default=None, description="Parent folder ID")


class DeleteFileInput(BaseModel):
    """Input for deleting a Drive file."""

    client: str = Field(default="default", description="Google client name")
    file_id: str = Field(..., description="File ID")


class SimpleOutput(BaseModel):
    """Simple success/error output."""

    success: bool = Field(description="Whether operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ListEventsInput(BaseModel):
    """Input for listing Calendar events."""

    client: str = Field(default="default", description="Google client name")
    calendar_id: str = Field(default="primary", description="Calendar ID")
    time_min: Optional[str] = Field(default=None, description="Start time (RFC3339)")
    time_max: Optional[str] = Field(default=None, description="End time (RFC3339)")
    max_results: int = Field(default=100, description="Maximum results")


class EventOutput(BaseModel):
    """Output containing event data."""

    success: bool = Field(description="Whether operation succeeded")
    event: Optional[Dict[str, Any]] = Field(default=None, description="Event data")
    events: Optional[List[Dict[str, Any]]] = Field(default=None, description="Event list")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetEventInput(BaseModel):
    """Input for getting a Calendar event."""

    client: str = Field(default="default", description="Google client name")
    event_id: str = Field(..., description="Event ID")
    calendar_id: str = Field(default="primary", description="Calendar ID")


class CreateEventInput(BaseModel):
    """Input for creating a Calendar event."""

    client: str = Field(default="default", description="Google client name")
    summary: str = Field(..., description="Event title")
    start: str = Field(..., description="Start time (RFC3339)")
    end: str = Field(..., description="End time (RFC3339)")
    calendar_id: str = Field(default="primary", description="Calendar ID")
    description: Optional[str] = Field(default=None, description="Event description")
    location: Optional[str] = Field(default=None, description="Event location")
    attendees: Optional[List[str]] = Field(default=None, description="Attendee emails")


class DeleteEventInput(BaseModel):
    """Input for deleting a Calendar event."""

    client: str = Field(default="default", description="Google client name")
    event_id: str = Field(..., description="Event ID")
    calendar_id: str = Field(default="primary", description="Calendar ID")


# Tool implementations


class CreateGoogleClientTool(BaseTool[CreateGoogleClientInput, CreateGoogleClientOutput]):
    """Tool for creating a Google client."""

    metadata = ToolMetadata(
        id="create_google_client",
        name="Create Google Client",
        description="Create a Google API client with OAuth2 authentication",
        category="utility",
    )
    input_type = CreateGoogleClientInput
    output_type = CreateGoogleClientOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateGoogleClientInput) -> CreateGoogleClientOutput:
        """Create a Google client."""
        config = GoogleConfig(access_token=input.access_token)
        client = GoogleClient(config)
        self.manager.add_client(input.name, client)

        return CreateGoogleClientOutput(
            success=True,
            name=input.name,
        )


class ListMessagesTool(BaseTool[ListMessagesInput, MessageOutput]):
    """Tool for listing Gmail messages."""

    metadata = ToolMetadata(
        id="list_gmail_messages",
        name="List Gmail Messages",
        description="List messages from Gmail",
        category="utility",
    )
    input_type = ListMessagesInput
    output_type = MessageOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListMessagesInput) -> MessageOutput:
        """List messages."""
        client = self.manager.get_client(input.client)

        if not client:
            return MessageOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.list_messages(
            query=input.query,
            max_results=input.max_results,
            label_ids=input.label_ids,
        )

        if result.success:
            return MessageOutput(success=True, messages=result.data.get("messages"))
        return MessageOutput(success=False, error=result.error)


class GetMessageTool(BaseTool[GetMessageInput, MessageOutput]):
    """Tool for getting a Gmail message."""

    metadata = ToolMetadata(
        id="get_gmail_message",
        name="Get Gmail Message",
        description="Get a specific Gmail message",
        category="utility",
    )
    input_type = GetMessageInput
    output_type = MessageOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetMessageInput) -> MessageOutput:
        """Get message."""
        client = self.manager.get_client(input.client)

        if not client:
            return MessageOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_message(input.message_id)

        if result.success:
            return MessageOutput(success=True, message=result.data)
        return MessageOutput(success=False, error=result.error)


class SendMessageTool(BaseTool[SendMessageInput, SimpleOutput]):
    """Tool for sending a Gmail message."""

    metadata = ToolMetadata(
        id="send_gmail_message",
        name="Send Gmail Message",
        description="Send an email via Gmail",
        category="utility",
    )
    input_type = SendMessageInput
    output_type = SimpleOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SendMessageInput) -> SimpleOutput:
        """Send message."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.send_message(
            to=input.to,
            subject=input.subject,
            body=input.body,
            cc=input.cc,
            bcc=input.bcc,
        )

        if result.success:
            return SimpleOutput(success=True)
        return SimpleOutput(success=False, error=result.error)


class ListFilesTool(BaseTool[ListFilesInput, FileOutput]):
    """Tool for listing Drive files."""

    metadata = ToolMetadata(
        id="list_drive_files",
        name="List Drive Files",
        description="List files from Google Drive",
        category="utility",
    )
    input_type = ListFilesInput
    output_type = FileOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListFilesInput) -> FileOutput:
        """List files."""
        client = self.manager.get_client(input.client)

        if not client:
            return FileOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.list_files(
            query=input.query,
            folder_id=input.folder_id,
            max_results=input.max_results,
        )

        if result.success:
            return FileOutput(success=True, files=result.data.get("files"))
        return FileOutput(success=False, error=result.error)


class GetFileTool(BaseTool[GetFileInput, FileOutput]):
    """Tool for getting a Drive file."""

    metadata = ToolMetadata(
        id="get_drive_file",
        name="Get Drive File",
        description="Get metadata for a Google Drive file",
        category="utility",
    )
    input_type = GetFileInput
    output_type = FileOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetFileInput) -> FileOutput:
        """Get file."""
        client = self.manager.get_client(input.client)

        if not client:
            return FileOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_file(input.file_id)

        if result.success:
            return FileOutput(success=True, file=result.data)
        return FileOutput(success=False, error=result.error)


class CreateFolderTool(BaseTool[CreateFolderInput, FileOutput]):
    """Tool for creating a Drive folder."""

    metadata = ToolMetadata(
        id="create_drive_folder",
        name="Create Drive Folder",
        description="Create a folder in Google Drive",
        category="utility",
    )
    input_type = CreateFolderInput
    output_type = FileOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateFolderInput) -> FileOutput:
        """Create folder."""
        client = self.manager.get_client(input.client)

        if not client:
            return FileOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.create_folder(
            name=input.name,
            parent_id=input.parent_id,
        )

        if result.success:
            return FileOutput(success=True, file=result.data)
        return FileOutput(success=False, error=result.error)


class DeleteFileTool(BaseTool[DeleteFileInput, SimpleOutput]):
    """Tool for deleting a Drive file."""

    metadata = ToolMetadata(
        id="delete_drive_file",
        name="Delete Drive File",
        description="Delete a file from Google Drive",
        category="utility",
    )
    input_type = DeleteFileInput
    output_type = SimpleOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: DeleteFileInput) -> SimpleOutput:
        """Delete file."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.delete_file(input.file_id)

        if result.success:
            return SimpleOutput(success=True)
        return SimpleOutput(success=False, error=result.error)


class ListEventsTool(BaseTool[ListEventsInput, EventOutput]):
    """Tool for listing Calendar events."""

    metadata = ToolMetadata(
        id="list_calendar_events",
        name="List Calendar Events",
        description="List events from Google Calendar",
        category="utility",
    )
    input_type = ListEventsInput
    output_type = EventOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: ListEventsInput) -> EventOutput:
        """List events."""
        client = self.manager.get_client(input.client)

        if not client:
            return EventOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.list_events(
            calendar_id=input.calendar_id,
            time_min=input.time_min,
            time_max=input.time_max,
            max_results=input.max_results,
        )

        if result.success:
            return EventOutput(success=True, events=result.data.get("events"))
        return EventOutput(success=False, error=result.error)


class GetEventTool(BaseTool[GetEventInput, EventOutput]):
    """Tool for getting a Calendar event."""

    metadata = ToolMetadata(
        id="get_calendar_event",
        name="Get Calendar Event",
        description="Get a specific Google Calendar event",
        category="utility",
    )
    input_type = GetEventInput
    output_type = EventOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetEventInput) -> EventOutput:
        """Get event."""
        client = self.manager.get_client(input.client)

        if not client:
            return EventOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_event(
            event_id=input.event_id,
            calendar_id=input.calendar_id,
        )

        if result.success:
            return EventOutput(success=True, event=result.data)
        return EventOutput(success=False, error=result.error)


class CreateEventTool(BaseTool[CreateEventInput, EventOutput]):
    """Tool for creating a Calendar event."""

    metadata = ToolMetadata(
        id="create_calendar_event",
        name="Create Calendar Event",
        description="Create an event in Google Calendar",
        category="utility",
    )
    input_type = CreateEventInput
    output_type = EventOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateEventInput) -> EventOutput:
        """Create event."""
        client = self.manager.get_client(input.client)

        if not client:
            return EventOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.create_event(
            summary=input.summary,
            start=input.start,
            end=input.end,
            calendar_id=input.calendar_id,
            description=input.description,
            location=input.location,
            attendees=input.attendees,
        )

        if result.success:
            return EventOutput(success=True, event=result.data)
        return EventOutput(success=False, error=result.error)


class DeleteEventTool(BaseTool[DeleteEventInput, SimpleOutput]):
    """Tool for deleting a Calendar event."""

    metadata = ToolMetadata(
        id="delete_calendar_event",
        name="Delete Calendar Event",
        description="Delete an event from Google Calendar",
        category="utility",
    )
    input_type = DeleteEventInput
    output_type = SimpleOutput

    def __init__(self, manager: GoogleManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: DeleteEventInput) -> SimpleOutput:
        """Delete event."""
        client = self.manager.get_client(input.client)

        if not client:
            return SimpleOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.delete_event(
            event_id=input.event_id,
            calendar_id=input.calendar_id,
        )

        if result.success:
            return SimpleOutput(success=True)
        return SimpleOutput(success=False, error=result.error)


# Convenience functions


def create_google_config(
    access_token: str,
    timeout: int = 30,
) -> GoogleConfig:
    """Create a Google configuration.

    Args:
        access_token: OAuth2 access token.
        timeout: Request timeout.

    Returns:
        Google configuration.
    """
    return GoogleConfig(access_token=access_token, timeout=timeout)


def create_google_client(config: GoogleConfig) -> GoogleClient:
    """Create a Google client.

    Args:
        config: Google configuration.

    Returns:
        Google client.
    """
    return GoogleClient(config)


def create_google_manager() -> GoogleManager:
    """Create a Google manager.

    Returns:
        Google manager.
    """
    return GoogleManager()


def create_google_tools(manager: GoogleManager) -> Dict[str, BaseTool]:
    """Create all Google Workspace tools.

    Args:
        manager: Google manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "create_google_client": CreateGoogleClientTool(manager),
        # Gmail
        "list_gmail_messages": ListMessagesTool(manager),
        "get_gmail_message": GetMessageTool(manager),
        "send_gmail_message": SendMessageTool(manager),
        # Drive
        "list_drive_files": ListFilesTool(manager),
        "get_drive_file": GetFileTool(manager),
        "create_drive_folder": CreateFolderTool(manager),
        "delete_drive_file": DeleteFileTool(manager),
        # Calendar
        "list_calendar_events": ListEventsTool(manager),
        "get_calendar_event": GetEventTool(manager),
        "create_calendar_event": CreateEventTool(manager),
        "delete_calendar_event": DeleteEventTool(manager),
    }
