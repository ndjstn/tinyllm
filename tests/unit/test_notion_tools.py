"""Tests for Notion tools."""

import json
from unittest.mock import MagicMock, patch
import pytest

from tinyllm.tools.notion import (
    BlockType,
    PropertyType,
    NotionConfig,
    RichText,
    Page,
    Database,
    Block,
    NotionResult,
    NotionClient,
    NotionManager,
    CreateNotionClientInput,
    CreateNotionClientOutput,
    GetPageInput,
    PageOutput,
    CreatePageInput,
    UpdatePageInput,
    GetDatabaseInput,
    DatabaseOutput,
    QueryDatabaseInput,
    QueryDatabaseOutput,
    GetBlockChildrenInput,
    BlockOutput,
    AppendBlocksInput,
    SearchInput,
    SearchOutput,
    CreateNotionClientTool,
    GetPageTool,
    CreatePageTool,
    UpdatePageTool,
    GetDatabaseTool,
    QueryDatabaseTool,
    GetBlockChildrenTool,
    AppendBlocksTool,
    SearchNotionTool,
    create_notion_config,
    create_notion_client,
    create_notion_manager,
    create_notion_tools,
    create_paragraph_block,
    create_heading_block,
    create_todo_block,
    create_bulleted_list_block,
    create_code_block,
)


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_block_type_values(self):
        """Test BlockType enum values."""
        assert BlockType.PARAGRAPH.value == "paragraph"
        assert BlockType.HEADING_1.value == "heading_1"
        assert BlockType.TODO.value == "to_do"
        assert BlockType.CODE.value == "code"

    def test_property_type_values(self):
        """Test PropertyType enum values."""
        assert PropertyType.TITLE.value == "title"
        assert PropertyType.RICH_TEXT.value == "rich_text"
        assert PropertyType.NUMBER.value == "number"
        assert PropertyType.CHECKBOX.value == "checkbox"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestNotionConfig:
    """Tests for NotionConfig dataclass."""

    def test_config_with_defaults(self):
        """Test config with default values."""
        config = NotionConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.base_url == "https://api.notion.com/v1"
        assert config.notion_version == "2022-06-28"
        assert config.timeout == 30

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = NotionConfig(
            api_key="my-key",
            base_url="https://custom.notion.com/v1",
            notion_version="2023-01-01",
            timeout=60,
        )

        assert config.base_url == "https://custom.notion.com/v1"
        assert config.notion_version == "2023-01-01"


class TestRichText:
    """Tests for RichText dataclass."""

    def test_rich_text_basic(self):
        """Test basic rich text."""
        rt = RichText(content="Hello world")

        api_format = rt.to_api_format()

        assert api_format["type"] == "text"
        assert api_format["text"]["content"] == "Hello world"
        assert api_format["annotations"]["bold"] is False

    def test_rich_text_with_formatting(self):
        """Test rich text with formatting."""
        rt = RichText(
            content="Important",
            bold=True,
            italic=True,
            link="https://example.com",
        )

        api_format = rt.to_api_format()

        assert api_format["annotations"]["bold"] is True
        assert api_format["annotations"]["italic"] is True
        assert api_format["text"]["link"]["url"] == "https://example.com"


class TestPage:
    """Tests for Page dataclass."""

    def test_page_from_api_response(self):
        """Test creating Page from API response."""
        api_data = {
            "id": "page-123",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "My Page"}],
                }
            },
            "parent": {
                "type": "database_id",
                "database_id": "db-456",
            },
            "url": "https://notion.so/page-123",
            "archived": False,
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
        }

        page = Page.from_api_response(api_data)

        assert page.id == "page-123"
        assert page.title == "My Page"
        assert page.parent_type == "database_id"
        assert page.parent_id == "db-456"
        assert page.archived is False

    def test_page_to_dict(self):
        """Test converting Page to dict."""
        page = Page(
            id="test-id",
            title="Test",
            parent_type="page_id",
            parent_id="parent-123",
        )

        d = page.to_dict()

        assert d["id"] == "test-id"
        assert d["title"] == "Test"


class TestDatabase:
    """Tests for Database dataclass."""

    def test_database_from_api_response(self):
        """Test creating Database from API response."""
        api_data = {
            "id": "db-123",
            "title": [{"plain_text": "My Database"}],
            "description": [{"plain_text": "A test database"}],
            "url": "https://notion.so/db-123",
            "archived": False,
            "properties": {
                "Name": {"type": "title"},
                "Status": {"type": "select"},
            },
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
        }

        db = Database.from_api_response(api_data)

        assert db.id == "db-123"
        assert db.title == "My Database"
        assert db.description == "A test database"
        assert "Name" in db.properties
        assert "Status" in db.properties

    def test_database_to_dict(self):
        """Test converting Database to dict."""
        db = Database(
            id="db-id",
            title="Test DB",
            properties={"Name": {}, "Status": {}},
        )

        d = db.to_dict()

        assert d["id"] == "db-id"
        assert d["title"] == "Test DB"
        assert "Name" in d["property_names"]


class TestBlock:
    """Tests for Block dataclass."""

    def test_block_from_api_response_paragraph(self):
        """Test creating Block from paragraph API response."""
        api_data = {
            "id": "block-123",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"plain_text": "Some text"}],
            },
            "has_children": False,
            "archived": False,
        }

        block = Block.from_api_response(api_data)

        assert block.id == "block-123"
        assert block.type == "paragraph"
        assert block.content == "Some text"
        assert block.has_children is False

    def test_block_to_dict(self):
        """Test converting Block to dict."""
        block = Block(
            id="block-id",
            type="heading_1",
            content="Header",
        )

        d = block.to_dict()

        assert d["id"] == "block-id"
        assert d["type"] == "heading_1"
        assert d["content"] == "Header"


class TestNotionResult:
    """Tests for NotionResult dataclass."""

    def test_result_success(self):
        """Test successful result."""
        result = NotionResult(
            success=True,
            data={"id": "123"},
            status_code=200,
        )

        assert result.success is True
        assert result.data == {"id": "123"}

    def test_result_failure(self):
        """Test failed result."""
        result = NotionResult(
            success=False,
            error="Not found",
            status_code=404,
        )

        assert result.success is False
        assert result.error == "Not found"


# ============================================================================
# NotionClient Tests
# ============================================================================


class TestNotionClient:
    """Tests for NotionClient class."""

    def test_create_client(self):
        """Test creating a client."""
        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        assert client.config == config

    @patch("urllib.request.urlopen")
    def test_get_page_success(self, mock_urlopen):
        """Test getting a page."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "page-123",
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"plain_text": "Test Page"}],
                }
            },
            "parent": {"type": "page_id", "page_id": "parent-123"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.get_page("page-123")

        assert result.success is True
        assert result.data["id"] == "page-123"

    @patch("urllib.request.urlopen")
    def test_create_page_success(self, mock_urlopen):
        """Test creating a page."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "new-page-123",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "New Page"}],
                }
            },
            "parent": {"type": "database_id", "database_id": "db-123"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.create_page(
            parent_type="database_id",
            parent_id="db-123",
            title="New Page",
        )

        assert result.success is True
        assert result.data["id"] == "new-page-123"

    @patch("urllib.request.urlopen")
    def test_update_page_success(self, mock_urlopen):
        """Test updating a page."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "page-123",
            "archived": True,
            "properties": {},
            "parent": {"type": "page_id", "page_id": "parent"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.update_page("page-123", archived=True)

        assert result.success is True

    def test_update_page_no_updates(self):
        """Test update with no changes."""
        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.update_page("page-123")

        assert result.success is False
        assert "No updates" in result.error

    @patch("urllib.request.urlopen")
    def test_get_database_success(self, mock_urlopen):
        """Test getting a database."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "db-123",
            "title": [{"plain_text": "My Database"}],
            "properties": {
                "Name": {"type": "title"},
            },
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.get_database("db-123")

        assert result.success is True
        assert result.data["id"] == "db-123"

    @patch("urllib.request.urlopen")
    def test_query_database_success(self, mock_urlopen):
        """Test querying a database."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "id": "page-1",
                    "properties": {
                        "Name": {"type": "title", "title": [{"plain_text": "Item 1"}]},
                    },
                    "parent": {"type": "database_id", "database_id": "db-123"},
                },
            ],
            "has_more": False,
            "next_cursor": None,
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.query_database("db-123")

        assert result.success is True
        assert len(result.data["results"]) == 1

    @patch("urllib.request.urlopen")
    def test_get_block_children_success(self, mock_urlopen):
        """Test getting block children."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "id": "block-1",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"plain_text": "Text"}]},
                    "has_children": False,
                },
            ],
            "has_more": False,
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.get_block_children("page-123")

        assert result.success is True
        assert len(result.data["results"]) == 1

    @patch("urllib.request.urlopen")
    def test_append_block_children_success(self, mock_urlopen):
        """Test appending block children."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "id": "new-block-1",
                    "type": "paragraph",
                    "paragraph": {"rich_text": []},
                    "has_children": False,
                },
            ],
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.append_block_children(
            "page-123",
            [create_paragraph_block("New text")],
        )

        assert result.success is True

    @patch("urllib.request.urlopen")
    def test_search_success(self, mock_urlopen):
        """Test searching Notion."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "object": "page",
                    "id": "page-1",
                    "properties": {
                        "title": {"type": "title", "title": [{"plain_text": "Found"}]},
                    },
                    "parent": {"type": "page_id", "page_id": "parent"},
                },
            ],
            "has_more": False,
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = NotionConfig(api_key="test-key")
        client = NotionClient(config)

        result = client.search(query="test")

        assert result.success is True
        assert len(result.data["results"]) == 1

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen):
        """Test handling HTTP errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.notion.com",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"message": "Invalid token"}')),
        )

        config = NotionConfig(api_key="bad-key")
        client = NotionClient(config)

        result = client.get_page("page-123")

        assert result.success is False
        assert "401" in result.error


# ============================================================================
# NotionManager Tests
# ============================================================================


class TestNotionManager:
    """Tests for NotionManager class."""

    def test_create_manager(self):
        """Test creating manager."""
        manager = NotionManager()

        assert manager.list_clients() == []

    def test_add_client(self):
        """Test adding a client."""
        manager = NotionManager()
        config = NotionConfig(api_key="test")
        client = NotionClient(config)

        manager.add_client("default", client)

        assert "default" in manager.list_clients()

    def test_get_client_not_found(self):
        """Test getting non-existent client."""
        manager = NotionManager()

        result = manager.get_client("nonexistent")

        assert result is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = NotionManager()
        config = NotionConfig(api_key="test")
        manager.add_client("test", NotionClient(config))

        result = manager.remove_client("test")

        assert result is True
        assert manager.get_client("test") is None


# ============================================================================
# Tool Tests
# ============================================================================


class TestCreateNotionClientTool:
    """Tests for CreateNotionClientTool."""

    @pytest.mark.asyncio
    async def test_create_client(self):
        """Test creating a Notion client."""
        manager = NotionManager()
        tool = CreateNotionClientTool(manager)

        input_data = CreateNotionClientInput(
            name="my-client",
            api_key="test-key",
        )

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.name == "my-client"
        assert manager.get_client("my-client") is not None


class TestGetPageTool:
    """Tests for GetPageTool."""

    @pytest.mark.asyncio
    async def test_client_not_found(self):
        """Test with non-existent client."""
        manager = NotionManager()
        tool = GetPageTool(manager)

        input_data = GetPageInput(
            client="nonexistent",
            page_id="page-123",
        )

        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.error

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = NotionManager()
        config = NotionConfig(api_key="test-key")
        manager.add_client("default", NotionClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_get_page_success(self, mock_urlopen, manager_with_client):
        """Test getting a page."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "page-123",
            "properties": {
                "title": {"type": "title", "title": [{"plain_text": "Test"}]},
            },
            "parent": {"type": "page_id", "page_id": "parent"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = GetPageTool(manager_with_client)

        input_data = GetPageInput(page_id="page-123")

        result = await tool.execute(input_data)

        assert result.success is True
        assert result.page["id"] == "page-123"


class TestCreatePageTool:
    """Tests for CreatePageTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = NotionManager()
        config = NotionConfig(api_key="test-key")
        manager.add_client("default", NotionClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_create_page_success(self, mock_urlopen, manager_with_client):
        """Test creating a page."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "id": "new-page",
            "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "New"}]},
            },
            "parent": {"type": "database_id", "database_id": "db-123"},
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = CreatePageTool(manager_with_client)

        input_data = CreatePageInput(
            parent_type="database_id",
            parent_id="db-123",
            title="New Page",
        )

        result = await tool.execute(input_data)

        assert result.success is True


class TestQueryDatabaseTool:
    """Tests for QueryDatabaseTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = NotionManager()
        config = NotionConfig(api_key="test-key")
        manager.add_client("default", NotionClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_query_database_success(self, mock_urlopen, manager_with_client):
        """Test querying a database."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "id": "page-1",
                    "properties": {
                        "Name": {"type": "title", "title": [{"plain_text": "Item"}]},
                    },
                    "parent": {"type": "database_id", "database_id": "db-123"},
                },
            ],
            "has_more": True,
            "next_cursor": "cursor-123",
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = QueryDatabaseTool(manager_with_client)

        input_data = QueryDatabaseInput(database_id="db-123")

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.results) == 1
        assert result.has_more is True


class TestSearchNotionTool:
    """Tests for SearchNotionTool."""

    @pytest.fixture
    def manager_with_client(self):
        """Create manager with client."""
        manager = NotionManager()
        config = NotionConfig(api_key="test-key")
        manager.add_client("default", NotionClient(config))
        return manager

    @pytest.mark.asyncio
    @patch("urllib.request.urlopen")
    async def test_search_success(self, mock_urlopen, manager_with_client):
        """Test searching Notion."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "object": "page",
                    "id": "page-1",
                    "properties": {
                        "title": {"type": "title", "title": [{"plain_text": "Found"}]},
                    },
                    "parent": {"type": "page_id", "page_id": "parent"},
                },
            ],
            "has_more": False,
        }).encode()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        tool = SearchNotionTool(manager_with_client)

        input_data = SearchInput(query="test")

        result = await tool.execute(input_data)

        assert result.success is True
        assert len(result.results) == 1


# ============================================================================
# Block Helper Function Tests
# ============================================================================


class TestBlockHelpers:
    """Tests for block helper functions."""

    def test_create_paragraph_block(self):
        """Test creating paragraph block."""
        block = create_paragraph_block("Hello world")

        assert block["type"] == "paragraph"
        assert block["paragraph"]["rich_text"][0]["text"]["content"] == "Hello world"

    def test_create_heading_block(self):
        """Test creating heading block."""
        block = create_heading_block("My Header", level=2)

        assert block["type"] == "heading_2"
        assert block["heading_2"]["rich_text"][0]["text"]["content"] == "My Header"

    def test_create_todo_block(self):
        """Test creating todo block."""
        block = create_todo_block("Do this", checked=True)

        assert block["type"] == "to_do"
        assert block["to_do"]["checked"] is True

    def test_create_bulleted_list_block(self):
        """Test creating bulleted list block."""
        block = create_bulleted_list_block("List item")

        assert block["type"] == "bulleted_list_item"

    def test_create_code_block(self):
        """Test creating code block."""
        block = create_code_block("print('hello')", language="python")

        assert block["type"] == "code"
        assert block["code"]["language"] == "python"


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_notion_config(self):
        """Test create_notion_config function."""
        config = create_notion_config(
            api_key="my-key",
            base_url="https://custom.notion.com/v1",
            notion_version="2023-01-01",
            timeout=60,
        )

        assert config.api_key == "my-key"
        assert config.base_url == "https://custom.notion.com/v1"
        assert config.notion_version == "2023-01-01"

    def test_create_notion_client(self):
        """Test create_notion_client function."""
        config = NotionConfig(api_key="test")
        client = create_notion_client(config)

        assert isinstance(client, NotionClient)

    def test_create_notion_manager(self):
        """Test create_notion_manager function."""
        manager = create_notion_manager()

        assert isinstance(manager, NotionManager)

    def test_create_notion_tools(self):
        """Test create_notion_tools function."""
        manager = NotionManager()
        tools = create_notion_tools(manager)

        assert "create_notion_client" in tools
        assert "get_notion_page" in tools
        assert "create_notion_page" in tools
        assert "update_notion_page" in tools
        assert "get_notion_database" in tools
        assert "query_notion_database" in tools
        assert "get_notion_block_children" in tools
        assert "append_notion_blocks" in tools
        assert "search_notion" in tools
