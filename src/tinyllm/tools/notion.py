"""Notion tools for TinyLLM.

This module provides tools for interacting with Notion API
including pages, databases, and blocks.
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class BlockType(str, Enum):
    """Notion block types."""

    PARAGRAPH = "paragraph"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    BULLETED_LIST_ITEM = "bulleted_list_item"
    NUMBERED_LIST_ITEM = "numbered_list_item"
    TODO = "to_do"
    TOGGLE = "toggle"
    CODE = "code"
    QUOTE = "quote"
    CALLOUT = "callout"
    DIVIDER = "divider"


class PropertyType(str, Enum):
    """Notion property types."""

    TITLE = "title"
    RICH_TEXT = "rich_text"
    NUMBER = "number"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    CHECKBOX = "checkbox"
    URL = "url"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"


@dataclass
class NotionConfig:
    """Notion API configuration."""

    api_key: str
    base_url: str = "https://api.notion.com/v1"
    notion_version: str = "2022-06-28"
    timeout: int = 30


@dataclass
class RichText:
    """Notion rich text object."""

    content: str
    link: Optional[str] = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False
    code: bool = False

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Notion API format."""
        text_obj: Dict[str, Any] = {"content": self.content}

        if self.link:
            text_obj["link"] = {"url": self.link}

        annotations = {
            "bold": self.bold,
            "italic": self.italic,
            "strikethrough": self.strikethrough,
            "underline": self.underline,
            "code": self.code,
        }

        return {"type": "text", "text": text_obj, "annotations": annotations}


@dataclass
class Page:
    """Notion page representation."""

    id: str
    title: str
    parent_type: str
    parent_id: str
    url: Optional[str] = None
    archived: bool = False
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Page":
        """Create Page from Notion API response."""
        # Extract title from properties
        title = ""
        properties = data.get("properties", {})

        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_list = prop_value.get("title", [])
                if title_list:
                    title = title_list[0].get("plain_text", "")
                break

        # Extract parent info
        parent = data.get("parent", {})
        parent_type = parent.get("type", "")
        # The key for the ID is the same as parent_type (e.g., "database_id", "page_id")
        parent_id = parent.get(parent_type, "") if parent_type else ""

        return cls(
            id=data["id"],
            title=title,
            parent_type=parent_type,
            parent_id=parent_id,
            url=data.get("url"),
            archived=data.get("archived", False),
            created_time=data.get("created_time"),
            last_edited_time=data.get("last_edited_time"),
            properties=properties,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "parent_type": self.parent_type,
            "parent_id": self.parent_id,
            "url": self.url,
            "archived": self.archived,
            "created_time": self.created_time,
            "last_edited_time": self.last_edited_time,
        }


@dataclass
class Database:
    """Notion database representation."""

    id: str
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    archived: bool = False
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Database":
        """Create Database from Notion API response."""
        # Extract title
        title_list = data.get("title", [])
        title = title_list[0].get("plain_text", "") if title_list else ""

        # Extract description
        description = None
        desc_list = data.get("description", [])
        if desc_list:
            description = desc_list[0].get("plain_text", "")

        return cls(
            id=data["id"],
            title=title,
            description=description,
            url=data.get("url"),
            archived=data.get("archived", False),
            created_time=data.get("created_time"),
            last_edited_time=data.get("last_edited_time"),
            properties=data.get("properties", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "archived": self.archived,
            "created_time": self.created_time,
            "last_edited_time": self.last_edited_time,
            "property_names": list(self.properties.keys()),
        }


@dataclass
class Block:
    """Notion block representation."""

    id: str
    type: str
    has_children: bool = False
    archived: bool = False
    content: Optional[str] = None
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "Block":
        """Create Block from Notion API response."""
        block_type = data.get("type", "")

        # Extract text content from block
        content = None
        if block_type in data:
            block_data = data[block_type]
            if isinstance(block_data, dict):
                rich_text = block_data.get("rich_text", [])
                if rich_text:
                    content = " ".join(rt.get("plain_text", "") for rt in rich_text)

        return cls(
            id=data["id"],
            type=block_type,
            has_children=data.get("has_children", False),
            archived=data.get("archived", False),
            content=content,
            created_time=data.get("created_time"),
            last_edited_time=data.get("last_edited_time"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "has_children": self.has_children,
            "archived": self.archived,
            "content": self.content,
            "created_time": self.created_time,
            "last_edited_time": self.last_edited_time,
        }


@dataclass
class NotionResult:
    """Result from Notion API operation."""

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


class NotionClient:
    """Client for Notion API."""

    def __init__(self, config: NotionConfig):
        """Initialize client.

        Args:
            config: Notion configuration.
        """
        self.config = config

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> NotionResult:
        """Make HTTP request to Notion API.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request body data.
            params: Query parameters.

        Returns:
            Notion result.
        """
        url = f"{self.config.base_url}{endpoint}"

        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query_string:
                url = f"{url}?{query_string}"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Notion-Version": self.config.notion_version,
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

                return NotionResult(
                    success=True,
                    data=response_data,
                    status_code=response.getcode(),
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
                error_data = json.loads(error_body)
                error_message = error_data.get("message", error_body)
            except Exception:
                error_message = error_body or str(e)

            return NotionResult(
                success=False,
                error=f"HTTP {e.code}: {error_message}",
                status_code=e.code,
            )
        except urllib.error.URLError as e:
            return NotionResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )
        except Exception as e:
            return NotionResult(
                success=False,
                error=str(e),
            )

    # Page operations

    def get_page(self, page_id: str) -> NotionResult:
        """Get a page by ID.

        Args:
            page_id: Page ID.

        Returns:
            Notion result with page data.
        """
        result = self._make_request("GET", f"/pages/{page_id}")

        if result.success and result.data:
            result.data = Page.from_api_response(result.data).to_dict()

        return result

    def create_page(
        self,
        parent_type: str,
        parent_id: str,
        title: str,
        properties: Optional[Dict[str, Any]] = None,
        children: Optional[List[Dict[str, Any]]] = None,
    ) -> NotionResult:
        """Create a new page.

        Args:
            parent_type: Parent type (database_id or page_id).
            parent_id: Parent ID.
            title: Page title.
            properties: Additional properties.
            children: Child blocks.

        Returns:
            Notion result with created page.
        """
        # Build parent object
        parent = {parent_type: parent_id}

        # Build properties with title
        page_properties: Dict[str, Any] = properties or {}

        if parent_type == "database_id":
            # For database pages, title is in "Name" or first title property
            page_properties["Name"] = {
                "title": [{"text": {"content": title}}]
            }
        else:
            # For regular pages
            page_properties["title"] = {
                "title": [{"text": {"content": title}}]
            }

        data: Dict[str, Any] = {
            "parent": parent,
            "properties": page_properties,
        }

        if children:
            data["children"] = children

        result = self._make_request("POST", "/pages", data=data)

        if result.success and result.data:
            result.data = Page.from_api_response(result.data).to_dict()

        return result

    def update_page(
        self,
        page_id: str,
        properties: Optional[Dict[str, Any]] = None,
        archived: Optional[bool] = None,
    ) -> NotionResult:
        """Update a page.

        Args:
            page_id: Page ID.
            properties: Properties to update.
            archived: Archive status.

        Returns:
            Notion result with updated page.
        """
        data: Dict[str, Any] = {}

        if properties is not None:
            data["properties"] = properties
        if archived is not None:
            data["archived"] = archived

        if not data:
            return NotionResult(success=False, error="No updates provided")

        result = self._make_request("PATCH", f"/pages/{page_id}", data=data)

        if result.success and result.data:
            result.data = Page.from_api_response(result.data).to_dict()

        return result

    # Database operations

    def get_database(self, database_id: str) -> NotionResult:
        """Get a database by ID.

        Args:
            database_id: Database ID.

        Returns:
            Notion result with database data.
        """
        result = self._make_request("GET", f"/databases/{database_id}")

        if result.success and result.data:
            result.data = Database.from_api_response(result.data).to_dict()

        return result

    def query_database(
        self,
        database_id: str,
        filter_obj: Optional[Dict[str, Any]] = None,
        sorts: Optional[List[Dict[str, Any]]] = None,
        page_size: int = 100,
        start_cursor: Optional[str] = None,
    ) -> NotionResult:
        """Query a database.

        Args:
            database_id: Database ID.
            filter_obj: Filter object.
            sorts: Sort objects.
            page_size: Results per page.
            start_cursor: Pagination cursor.

        Returns:
            Notion result with query results.
        """
        data: Dict[str, Any] = {"page_size": page_size}

        if filter_obj:
            data["filter"] = filter_obj
        if sorts:
            data["sorts"] = sorts
        if start_cursor:
            data["start_cursor"] = start_cursor

        result = self._make_request("POST", f"/databases/{database_id}/query", data=data)

        if result.success and result.data:
            pages = [
                Page.from_api_response(item).to_dict()
                for item in result.data.get("results", [])
            ]
            result.data = {
                "results": pages,
                "has_more": result.data.get("has_more", False),
                "next_cursor": result.data.get("next_cursor"),
            }

        return result

    # Block operations

    def get_block(self, block_id: str) -> NotionResult:
        """Get a block by ID.

        Args:
            block_id: Block ID.

        Returns:
            Notion result with block data.
        """
        result = self._make_request("GET", f"/blocks/{block_id}")

        if result.success and result.data:
            result.data = Block.from_api_response(result.data).to_dict()

        return result

    def get_block_children(
        self,
        block_id: str,
        page_size: int = 100,
        start_cursor: Optional[str] = None,
    ) -> NotionResult:
        """Get children of a block.

        Args:
            block_id: Block ID.
            page_size: Results per page.
            start_cursor: Pagination cursor.

        Returns:
            Notion result with child blocks.
        """
        params = {"page_size": str(page_size)}
        if start_cursor:
            params["start_cursor"] = start_cursor

        result = self._make_request("GET", f"/blocks/{block_id}/children", params=params)

        if result.success and result.data:
            blocks = [
                Block.from_api_response(item).to_dict()
                for item in result.data.get("results", [])
            ]
            result.data = {
                "results": blocks,
                "has_more": result.data.get("has_more", False),
                "next_cursor": result.data.get("next_cursor"),
            }

        return result

    def append_block_children(
        self,
        block_id: str,
        children: List[Dict[str, Any]],
    ) -> NotionResult:
        """Append children to a block.

        Args:
            block_id: Block ID.
            children: Child blocks to append.

        Returns:
            Notion result with appended blocks.
        """
        data = {"children": children}

        result = self._make_request("PATCH", f"/blocks/{block_id}/children", data=data)

        if result.success and result.data:
            blocks = [
                Block.from_api_response(item).to_dict()
                for item in result.data.get("results", [])
            ]
            result.data = {"results": blocks}

        return result

    def delete_block(self, block_id: str) -> NotionResult:
        """Delete (archive) a block.

        Args:
            block_id: Block ID.

        Returns:
            Notion result.
        """
        return self._make_request("DELETE", f"/blocks/{block_id}")

    # Search operation

    def search(
        self,
        query: Optional[str] = None,
        filter_type: Optional[str] = None,
        sort_direction: str = "descending",
        page_size: int = 100,
        start_cursor: Optional[str] = None,
    ) -> NotionResult:
        """Search for pages and databases.

        Args:
            query: Search query.
            filter_type: Filter by type (page or database).
            sort_direction: Sort direction.
            page_size: Results per page.
            start_cursor: Pagination cursor.

        Returns:
            Notion result with search results.
        """
        data: Dict[str, Any] = {"page_size": page_size}

        if query:
            data["query"] = query
        if filter_type:
            data["filter"] = {"property": "object", "value": filter_type}
        if sort_direction:
            data["sort"] = {"direction": sort_direction, "timestamp": "last_edited_time"}
        if start_cursor:
            data["start_cursor"] = start_cursor

        result = self._make_request("POST", "/search", data=data)

        if result.success and result.data:
            items = []
            for item in result.data.get("results", []):
                if item.get("object") == "page":
                    items.append({"type": "page", **Page.from_api_response(item).to_dict()})
                elif item.get("object") == "database":
                    items.append({"type": "database", **Database.from_api_response(item).to_dict()})
                else:
                    items.append(item)

            result.data = {
                "results": items,
                "has_more": result.data.get("has_more", False),
                "next_cursor": result.data.get("next_cursor"),
            }

        return result


class NotionManager:
    """Manager for Notion clients."""

    def __init__(self):
        """Initialize manager."""
        self._clients: Dict[str, NotionClient] = {}

    def add_client(self, name: str, client: NotionClient) -> None:
        """Add a Notion client.

        Args:
            name: Client name.
            client: Notion client.
        """
        self._clients[name] = client

    def get_client(self, name: str) -> Optional[NotionClient]:
        """Get a Notion client.

        Args:
            name: Client name.

        Returns:
            Notion client or None.
        """
        return self._clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove a Notion client.

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


class CreateNotionClientInput(BaseModel):
    """Input for creating a Notion client."""

    name: str = Field(..., description="Name for the client")
    api_key: str = Field(..., description="Notion integration token")


class CreateNotionClientOutput(BaseModel):
    """Output from creating a Notion client."""

    success: bool = Field(description="Whether client was created")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetPageInput(BaseModel):
    """Input for getting a page."""

    client: str = Field(default="default", description="Notion client name")
    page_id: str = Field(..., description="Page ID")


class PageOutput(BaseModel):
    """Output containing page data."""

    success: bool = Field(description="Whether operation succeeded")
    page: Optional[Dict[str, Any]] = Field(default=None, description="Page data")
    pages: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of pages")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CreatePageInput(BaseModel):
    """Input for creating a page."""

    client: str = Field(default="default", description="Notion client name")
    parent_type: str = Field(..., description="Parent type (database_id or page_id)")
    parent_id: str = Field(..., description="Parent ID")
    title: str = Field(..., description="Page title")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Additional properties")


class UpdatePageInput(BaseModel):
    """Input for updating a page."""

    client: str = Field(default="default", description="Notion client name")
    page_id: str = Field(..., description="Page ID")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Properties to update")
    archived: Optional[bool] = Field(default=None, description="Archive status")


class GetDatabaseInput(BaseModel):
    """Input for getting a database."""

    client: str = Field(default="default", description="Notion client name")
    database_id: str = Field(..., description="Database ID")


class DatabaseOutput(BaseModel):
    """Output containing database data."""

    success: bool = Field(description="Whether operation succeeded")
    database: Optional[Dict[str, Any]] = Field(default=None, description="Database data")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class QueryDatabaseInput(BaseModel):
    """Input for querying a database."""

    client: str = Field(default="default", description="Notion client name")
    database_id: str = Field(..., description="Database ID")
    filter_obj: Optional[Dict[str, Any]] = Field(default=None, description="Filter object")
    sorts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sort objects")
    page_size: int = Field(default=100, description="Results per page")


class QueryDatabaseOutput(BaseModel):
    """Output from querying a database."""

    success: bool = Field(description="Whether operation succeeded")
    results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Query results")
    has_more: bool = Field(default=False, description="Whether more results exist")
    next_cursor: Optional[str] = Field(default=None, description="Pagination cursor")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetBlockChildrenInput(BaseModel):
    """Input for getting block children."""

    client: str = Field(default="default", description="Notion client name")
    block_id: str = Field(..., description="Block or page ID")
    page_size: int = Field(default=100, description="Results per page")


class BlockOutput(BaseModel):
    """Output containing block data."""

    success: bool = Field(description="Whether operation succeeded")
    blocks: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of blocks")
    has_more: bool = Field(default=False, description="Whether more results exist")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AppendBlocksInput(BaseModel):
    """Input for appending blocks."""

    client: str = Field(default="default", description="Notion client name")
    block_id: str = Field(..., description="Block or page ID")
    blocks: List[Dict[str, Any]] = Field(..., description="Blocks to append")


class SearchInput(BaseModel):
    """Input for searching."""

    client: str = Field(default="default", description="Notion client name")
    query: Optional[str] = Field(default=None, description="Search query")
    filter_type: Optional[str] = Field(default=None, description="Filter by type (page/database)")
    page_size: int = Field(default=100, description="Results per page")


class SearchOutput(BaseModel):
    """Output from search."""

    success: bool = Field(description="Whether operation succeeded")
    results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Search results")
    has_more: bool = Field(default=False, description="Whether more results exist")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# Tool implementations


class CreateNotionClientTool(BaseTool[CreateNotionClientInput, CreateNotionClientOutput]):
    """Tool for creating a Notion client."""

    metadata = ToolMetadata(
        id="create_notion_client",
        name="Create Notion Client",
        description="Create a Notion API client with authentication",
        category="utility",
    )
    input_type = CreateNotionClientInput
    output_type = CreateNotionClientOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreateNotionClientInput) -> CreateNotionClientOutput:
        """Create a Notion client."""
        config = NotionConfig(api_key=input.api_key)
        client = NotionClient(config)
        self.manager.add_client(input.name, client)

        return CreateNotionClientOutput(
            success=True,
            name=input.name,
        )


class GetPageTool(BaseTool[GetPageInput, PageOutput]):
    """Tool for getting a Notion page."""

    metadata = ToolMetadata(
        id="get_notion_page",
        name="Get Notion Page",
        description="Get a page from Notion by ID",
        category="utility",
    )
    input_type = GetPageInput
    output_type = PageOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetPageInput) -> PageOutput:
        """Get a page."""
        client = self.manager.get_client(input.client)

        if not client:
            return PageOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_page(input.page_id)

        if result.success:
            return PageOutput(success=True, page=result.data)
        return PageOutput(success=False, error=result.error)


class CreatePageTool(BaseTool[CreatePageInput, PageOutput]):
    """Tool for creating a Notion page."""

    metadata = ToolMetadata(
        id="create_notion_page",
        name="Create Notion Page",
        description="Create a new page in Notion",
        category="utility",
    )
    input_type = CreatePageInput
    output_type = PageOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: CreatePageInput) -> PageOutput:
        """Create a page."""
        client = self.manager.get_client(input.client)

        if not client:
            return PageOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.create_page(
            parent_type=input.parent_type,
            parent_id=input.parent_id,
            title=input.title,
            properties=input.properties,
        )

        if result.success:
            return PageOutput(success=True, page=result.data)
        return PageOutput(success=False, error=result.error)


class UpdatePageTool(BaseTool[UpdatePageInput, PageOutput]):
    """Tool for updating a Notion page."""

    metadata = ToolMetadata(
        id="update_notion_page",
        name="Update Notion Page",
        description="Update an existing Notion page",
        category="utility",
    )
    input_type = UpdatePageInput
    output_type = PageOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: UpdatePageInput) -> PageOutput:
        """Update a page."""
        client = self.manager.get_client(input.client)

        if not client:
            return PageOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.update_page(
            page_id=input.page_id,
            properties=input.properties,
            archived=input.archived,
        )

        if result.success:
            return PageOutput(success=True, page=result.data)
        return PageOutput(success=False, error=result.error)


class GetDatabaseTool(BaseTool[GetDatabaseInput, DatabaseOutput]):
    """Tool for getting a Notion database."""

    metadata = ToolMetadata(
        id="get_notion_database",
        name="Get Notion Database",
        description="Get a database from Notion by ID",
        category="utility",
    )
    input_type = GetDatabaseInput
    output_type = DatabaseOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetDatabaseInput) -> DatabaseOutput:
        """Get a database."""
        client = self.manager.get_client(input.client)

        if not client:
            return DatabaseOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_database(input.database_id)

        if result.success:
            return DatabaseOutput(success=True, database=result.data)
        return DatabaseOutput(success=False, error=result.error)


class QueryDatabaseTool(BaseTool[QueryDatabaseInput, QueryDatabaseOutput]):
    """Tool for querying a Notion database."""

    metadata = ToolMetadata(
        id="query_notion_database",
        name="Query Notion Database",
        description="Query a Notion database with filters and sorts",
        category="utility",
    )
    input_type = QueryDatabaseInput
    output_type = QueryDatabaseOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: QueryDatabaseInput) -> QueryDatabaseOutput:
        """Query a database."""
        client = self.manager.get_client(input.client)

        if not client:
            return QueryDatabaseOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.query_database(
            database_id=input.database_id,
            filter_obj=input.filter_obj,
            sorts=input.sorts,
            page_size=input.page_size,
        )

        if result.success:
            return QueryDatabaseOutput(
                success=True,
                results=result.data.get("results"),
                has_more=result.data.get("has_more", False),
                next_cursor=result.data.get("next_cursor"),
            )
        return QueryDatabaseOutput(success=False, error=result.error)


class GetBlockChildrenTool(BaseTool[GetBlockChildrenInput, BlockOutput]):
    """Tool for getting block children."""

    metadata = ToolMetadata(
        id="get_notion_block_children",
        name="Get Notion Block Children",
        description="Get child blocks of a Notion block or page",
        category="utility",
    )
    input_type = GetBlockChildrenInput
    output_type = BlockOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: GetBlockChildrenInput) -> BlockOutput:
        """Get block children."""
        client = self.manager.get_client(input.client)

        if not client:
            return BlockOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.get_block_children(
            block_id=input.block_id,
            page_size=input.page_size,
        )

        if result.success:
            return BlockOutput(
                success=True,
                blocks=result.data.get("results"),
                has_more=result.data.get("has_more", False),
            )
        return BlockOutput(success=False, error=result.error)


class AppendBlocksTool(BaseTool[AppendBlocksInput, BlockOutput]):
    """Tool for appending blocks to a page."""

    metadata = ToolMetadata(
        id="append_notion_blocks",
        name="Append Notion Blocks",
        description="Append blocks to a Notion page or block",
        category="utility",
    )
    input_type = AppendBlocksInput
    output_type = BlockOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: AppendBlocksInput) -> BlockOutput:
        """Append blocks."""
        client = self.manager.get_client(input.client)

        if not client:
            return BlockOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.append_block_children(
            block_id=input.block_id,
            children=input.blocks,
        )

        if result.success:
            return BlockOutput(
                success=True,
                blocks=result.data.get("results"),
            )
        return BlockOutput(success=False, error=result.error)


class SearchNotionTool(BaseTool[SearchInput, SearchOutput]):
    """Tool for searching Notion."""

    metadata = ToolMetadata(
        id="search_notion",
        name="Search Notion",
        description="Search for pages and databases in Notion",
        category="search",
    )
    input_type = SearchInput
    output_type = SearchOutput

    def __init__(self, manager: NotionManager):
        """Initialize tool."""
        self.manager = manager

    async def execute(self, input: SearchInput) -> SearchOutput:
        """Search Notion."""
        client = self.manager.get_client(input.client)

        if not client:
            return SearchOutput(
                success=False,
                error=f"Client '{input.client}' not found",
            )

        result = client.search(
            query=input.query,
            filter_type=input.filter_type,
            page_size=input.page_size,
        )

        if result.success:
            return SearchOutput(
                success=True,
                results=result.data.get("results"),
                has_more=result.data.get("has_more", False),
            )
        return SearchOutput(success=False, error=result.error)


# Convenience functions


def create_notion_config(
    api_key: str,
    base_url: str = "https://api.notion.com/v1",
    notion_version: str = "2022-06-28",
    timeout: int = 30,
) -> NotionConfig:
    """Create a Notion configuration.

    Args:
        api_key: Notion integration token.
        base_url: Notion API base URL.
        notion_version: Notion API version.
        timeout: Request timeout.

    Returns:
        Notion configuration.
    """
    return NotionConfig(
        api_key=api_key,
        base_url=base_url,
        notion_version=notion_version,
        timeout=timeout,
    )


def create_notion_client(config: NotionConfig) -> NotionClient:
    """Create a Notion client.

    Args:
        config: Notion configuration.

    Returns:
        Notion client.
    """
    return NotionClient(config)


def create_notion_manager() -> NotionManager:
    """Create a Notion manager.

    Returns:
        Notion manager.
    """
    return NotionManager()


def create_notion_tools(manager: NotionManager) -> Dict[str, BaseTool]:
    """Create all Notion tools.

    Args:
        manager: Notion manager.

    Returns:
        Dictionary of tool name to tool instance.
    """
    return {
        "create_notion_client": CreateNotionClientTool(manager),
        "get_notion_page": GetPageTool(manager),
        "create_notion_page": CreatePageTool(manager),
        "update_notion_page": UpdatePageTool(manager),
        "get_notion_database": GetDatabaseTool(manager),
        "query_notion_database": QueryDatabaseTool(manager),
        "get_notion_block_children": GetBlockChildrenTool(manager),
        "append_notion_blocks": AppendBlocksTool(manager),
        "search_notion": SearchNotionTool(manager),
    }


def create_paragraph_block(text: str) -> Dict[str, Any]:
    """Create a paragraph block.

    Args:
        text: Paragraph text.

    Returns:
        Block object.
    """
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def create_heading_block(text: str, level: int = 1) -> Dict[str, Any]:
    """Create a heading block.

    Args:
        text: Heading text.
        level: Heading level (1, 2, or 3).

    Returns:
        Block object.
    """
    heading_type = f"heading_{level}"
    return {
        "object": "block",
        "type": heading_type,
        heading_type: {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def create_todo_block(text: str, checked: bool = False) -> Dict[str, Any]:
    """Create a to-do block.

    Args:
        text: Todo text.
        checked: Whether checked.

    Returns:
        Block object.
    """
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"type": "text", "text": {"content": text}}],
            "checked": checked,
        },
    }


def create_bulleted_list_block(text: str) -> Dict[str, Any]:
    """Create a bulleted list item block.

    Args:
        text: Item text.

    Returns:
        Block object.
    """
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        },
    }


def create_code_block(code: str, language: str = "plain text") -> Dict[str, Any]:
    """Create a code block.

    Args:
        code: Code content.
        language: Programming language.

    Returns:
        Block object.
    """
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": code}}],
            "language": language,
        },
    }
