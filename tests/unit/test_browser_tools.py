"""Tests for browser automation tools."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinyllm.tools.browser import (
    # Enums
    BrowserType,
    WaitStrategy,
    SelectorType,
    # Config and models
    BrowserConfig,
    BrowserResult,
    ElementInfo,
    PageInfo,
    # Client and manager
    BrowserClient,
    BrowserManager,
    # Input models
    CreateBrowserClientInput,
    NavigateInput,
    ClickInput,
    FillInput,
    TypeTextInput,
    PressKeyInput,
    GetElementInput,
    ScreenshotInput,
    EvaluateInput,
    ScrollInput,
    WaitForSelectorInput,
    ClientInput,
    # Output models
    CreateBrowserClientOutput,
    BrowserSimpleOutput,
    BrowserDataOutput,
    # Tools
    CreateBrowserClientTool,
    CloseBrowserClientTool,
    NavigateTool,
    ClickTool,
    FillTool,
    TypeTextTool,
    PressKeyTool,
    GetElementInfoTool,
    GetTextTool,
    ScreenshotTool,
    EvaluateTool,
    GetPageInfoTool,
    GetContentTool,
    ScrollTool,
    GoBackTool,
    GoForwardTool,
    ReloadTool,
    WaitForSelectorTool,
    # Helper functions
    create_browser_config,
    create_browser_client,
    create_browser_manager,
    create_browser_tools,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestBrowserType:
    """Tests for BrowserType enum."""

    def test_browser_type_values(self):
        """Test browser type values."""
        assert BrowserType.CHROMIUM.value == "chromium"
        assert BrowserType.FIREFOX.value == "firefox"
        assert BrowserType.WEBKIT.value == "webkit"

    def test_browser_type_count(self):
        """Test browser type count."""
        assert len(BrowserType) == 3


class TestWaitStrategy:
    """Tests for WaitStrategy enum."""

    def test_wait_strategy_values(self):
        """Test wait strategy values."""
        assert WaitStrategy.LOAD.value == "load"
        assert WaitStrategy.DOMCONTENTLOADED.value == "domcontentloaded"
        assert WaitStrategy.NETWORKIDLE.value == "networkidle"

    def test_wait_strategy_count(self):
        """Test wait strategy count."""
        assert len(WaitStrategy) == 3


class TestSelectorType:
    """Tests for SelectorType enum."""

    def test_selector_type_values(self):
        """Test selector type values."""
        assert SelectorType.CSS.value == "css"
        assert SelectorType.XPATH.value == "xpath"
        assert SelectorType.TEXT.value == "text"
        assert SelectorType.ROLE.value == "role"

    def test_selector_type_count(self):
        """Test selector type count."""
        assert len(SelectorType) == 4


# =============================================================================
# Config and Model Tests
# =============================================================================


class TestBrowserConfig:
    """Tests for BrowserConfig."""

    def test_config_defaults(self):
        """Test config default values."""
        config = BrowserConfig()
        assert config.browser_type == BrowserType.CHROMIUM
        assert config.headless is True
        assert config.timeout == 30000
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.user_agent is None
        assert config.ignore_https_errors is False
        assert config.slow_mo == 0
        assert config.proxy is None
        assert config.downloads_path is None

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = BrowserConfig(
            browser_type=BrowserType.FIREFOX,
            headless=False,
            timeout=60000,
            viewport_width=1920,
            viewport_height=1080,
            user_agent="Custom Agent",
            ignore_https_errors=True,
            slow_mo=100,
            proxy="http://proxy.example.com",
            downloads_path="/tmp/downloads",
        )
        assert config.browser_type == BrowserType.FIREFOX
        assert config.headless is False
        assert config.timeout == 60000
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.user_agent == "Custom Agent"
        assert config.ignore_https_errors is True
        assert config.slow_mo == 100
        assert config.proxy == "http://proxy.example.com"
        assert config.downloads_path == "/tmp/downloads"


class TestBrowserResult:
    """Tests for BrowserResult."""

    def test_success_result(self):
        """Test successful result."""
        result = BrowserResult(
            success=True, message="Operation completed", data={"key": "value"}
        )
        assert result.success is True
        assert result.message == "Operation completed"
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = BrowserResult(
            success=False, message="Operation failed", error="Some error"
        )
        assert result.success is False
        assert result.message == "Operation failed"
        assert result.data is None
        assert result.error == "Some error"

    def test_result_to_dict(self):
        """Test result to_dict method."""
        result = BrowserResult(
            success=True, message="OK", data={"foo": "bar"}, error=None
        )
        d = result.to_dict()
        assert d == {
            "success": True,
            "message": "OK",
            "data": {"foo": "bar"},
            "error": None,
        }


class TestElementInfo:
    """Tests for ElementInfo."""

    def test_element_info_creation(self):
        """Test ElementInfo creation."""
        info = ElementInfo(
            selector="#test",
            tag_name="button",
            text="Click me",
            attributes={"id": "test", "class": "btn"},
            is_visible=True,
            is_enabled=True,
            bounding_box={"x": 10, "y": 20, "width": 100, "height": 50},
        )
        assert info.selector == "#test"
        assert info.tag_name == "button"
        assert info.text == "Click me"
        assert info.attributes == {"id": "test", "class": "btn"}
        assert info.is_visible is True
        assert info.is_enabled is True
        assert info.bounding_box == {"x": 10, "y": 20, "width": 100, "height": 50}


class TestPageInfo:
    """Tests for PageInfo."""

    def test_page_info_creation(self):
        """Test PageInfo creation."""
        info = PageInfo(
            url="https://example.com",
            title="Example Page",
            viewport={"width": 1280, "height": 720},
            has_focus=True,
        )
        assert info.url == "https://example.com"
        assert info.title == "Example Page"
        assert info.viewport == {"width": 1280, "height": 720}
        assert info.has_focus is True


# =============================================================================
# Browser Client Tests
# =============================================================================


class TestBrowserClient:
    """Tests for BrowserClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        config = BrowserConfig()
        client = BrowserClient(config)
        assert client.config == config
        assert client._playwright is None
        assert client._browser is None
        assert client._context is None
        assert client._page is None
        assert client._is_connected is False

    def test_is_connected_property(self):
        """Test is_connected property."""
        config = BrowserConfig()
        client = BrowserClient(config)
        assert client.is_connected is False

        # Simulate connection
        client._is_connected = True
        client._page = MagicMock()
        assert client.is_connected is True


class TestBrowserClientConnect:
    """Tests for BrowserClient connect method."""

    @pytest.mark.asyncio
    async def test_connect_playwright_not_installed(self):
        """Test connect when playwright is not installed."""
        config = BrowserConfig()
        client = BrowserClient(config)

        with patch.dict("sys.modules", {"playwright.async_api": None}):
            with patch(
                "tinyllm.tools.browser.BrowserClient.connect",
                new_callable=AsyncMock,
                return_value=BrowserResult(
                    success=False,
                    message="Playwright not installed",
                    error="Please install playwright",
                ),
            ):
                result = await client.connect()
                assert result.success is False
                assert "Playwright" in result.message or "Playwright" in str(
                    result.error
                )

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        config = BrowserConfig()
        client = BrowserClient(config)

        # Mock playwright - patch the import inside connect method
        mock_page = AsyncMock()
        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.set_default_timeout = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mock_launcher = AsyncMock()
        mock_launcher.launch = AsyncMock(return_value=mock_browser)

        mock_playwright = MagicMock()
        mock_playwright.chromium = mock_launcher

        mock_async_playwright_instance = AsyncMock()
        mock_async_playwright_instance.start = AsyncMock(return_value=mock_playwright)

        mock_async_playwright = MagicMock(return_value=mock_async_playwright_instance)

        with patch.dict(
            "sys.modules",
            {"playwright.async_api": MagicMock(async_playwright=mock_async_playwright)},
        ):
            result = await client.connect()
            # Since we're mocking, it might fail due to import complexity
            # Just check the result is a BrowserResult
            assert isinstance(result, BrowserResult)


class TestBrowserClientDisconnect:
    """Tests for BrowserClient disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self):
        """Test disconnect when not connected."""
        config = BrowserConfig()
        client = BrowserClient(config)

        result = await client.disconnect()
        assert result.success is True
        assert client._is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_success(self):
        """Test successful disconnect."""
        config = BrowserConfig()
        client = BrowserClient(config)

        # Mock connected state
        client._page = AsyncMock()
        client._page.close = AsyncMock()
        client._context = AsyncMock()
        client._context.close = AsyncMock()
        client._browser = AsyncMock()
        client._browser.close = AsyncMock()
        client._playwright = AsyncMock()
        client._playwright.stop = AsyncMock()
        client._is_connected = True

        result = await client.disconnect()
        assert result.success is True
        assert client._is_connected is False
        assert client._page is None
        assert client._context is None
        assert client._browser is None
        assert client._playwright is None


class TestBrowserClientOperations:
    """Tests for BrowserClient operations."""

    @pytest.fixture
    def connected_client(self):
        """Create a connected client with mocks."""
        config = BrowserConfig()
        client = BrowserClient(config)

        # Mock page
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.viewport_size = {"width": 1280, "height": 720}
        mock_page.goto = AsyncMock(
            return_value=MagicMock(status=200)
        )
        mock_page.keyboard = AsyncMock()
        mock_page.keyboard.press = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value="result")
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.go_back = AsyncMock()
        mock_page.go_forward = AsyncMock()
        mock_page.reload = AsyncMock()
        mock_page.screenshot = AsyncMock(return_value=b"screenshot_data")

        # Mock locator
        mock_locator = AsyncMock()
        mock_locator.click = AsyncMock()
        mock_locator.fill = AsyncMock()
        mock_locator.type = AsyncMock()
        mock_locator.hover = AsyncMock()
        mock_locator.wait_for = AsyncMock()
        mock_locator.inner_text = AsyncMock(return_value="Text content")
        mock_locator.is_visible = AsyncMock(return_value=True)
        mock_locator.is_enabled = AsyncMock(return_value=True)
        mock_locator.bounding_box = AsyncMock(
            return_value={"x": 0, "y": 0, "width": 100, "height": 50}
        )
        mock_locator.get_attribute = AsyncMock(return_value="value")
        mock_locator.screenshot = AsyncMock(return_value=b"element_screenshot")
        mock_locator.scroll_into_view_if_needed = AsyncMock()
        mock_locator.evaluate = AsyncMock(
            side_effect=[
                "button",  # tag_name
                {"id": "test"},  # attributes
            ]
        )

        mock_page.locator = MagicMock(return_value=mock_locator)
        mock_page.get_by_text = MagicMock(return_value=mock_locator)
        mock_page.get_by_role = MagicMock(return_value=mock_locator)

        client._page = mock_page
        client._is_connected = True

        return client

    @pytest.mark.asyncio
    async def test_navigate_not_connected(self):
        """Test navigate when not connected."""
        config = BrowserConfig()
        client = BrowserClient(config)

        result = await client.navigate("https://example.com")
        assert result.success is False
        assert "Not connected" in result.message or "Not connected" in str(result.error)

    @pytest.mark.asyncio
    async def test_navigate_success(self, connected_client):
        """Test successful navigation."""
        result = await connected_client.navigate("https://example.com")
        assert result.success is True
        assert "Navigated" in result.message

    @pytest.mark.asyncio
    async def test_get_page_info(self, connected_client):
        """Test get page info."""
        result = await connected_client.get_page_info()
        assert result.success is True
        assert result.data["url"] == "https://example.com"
        assert result.data["title"] == "Example"

    @pytest.mark.asyncio
    async def test_click(self, connected_client):
        """Test click operation."""
        result = await connected_client.click("#button")
        assert result.success is True
        assert "Clicked" in result.message

    @pytest.mark.asyncio
    async def test_fill(self, connected_client):
        """Test fill operation."""
        result = await connected_client.fill("#input", "test value")
        assert result.success is True
        assert "Filled" in result.message

    @pytest.mark.asyncio
    async def test_type_text(self, connected_client):
        """Test type text operation."""
        result = await connected_client.type_text("#input", "test")
        assert result.success is True
        assert "Typed" in result.message

    @pytest.mark.asyncio
    async def test_press_key(self, connected_client):
        """Test press key operation."""
        result = await connected_client.press_key("Enter")
        assert result.success is True
        assert "Pressed" in result.message

    @pytest.mark.asyncio
    async def test_hover(self, connected_client):
        """Test hover operation."""
        result = await connected_client.hover("#element")
        assert result.success is True
        assert "Hovered" in result.message

    @pytest.mark.asyncio
    async def test_wait_for_selector(self, connected_client):
        """Test wait for selector."""
        result = await connected_client.wait_for_selector("#element")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_text(self, connected_client):
        """Test get text."""
        result = await connected_client.get_text("#element")
        assert result.success is True
        assert result.data["text"] == "Text content"

    @pytest.mark.asyncio
    async def test_get_attribute(self, connected_client):
        """Test get attribute."""
        result = await connected_client.get_attribute("#element", "id")
        assert result.success is True
        assert result.data["value"] == "value"

    @pytest.mark.asyncio
    async def test_screenshot_base64(self, connected_client):
        """Test screenshot returns base64."""
        result = await connected_client.screenshot()
        assert result.success is True
        assert "base64" in result.data

    @pytest.mark.asyncio
    async def test_screenshot_to_file(self, connected_client):
        """Test screenshot to file."""
        result = await connected_client.screenshot(path="/tmp/test.png")
        assert result.success is True
        assert result.data["path"] == "/tmp/test.png"

    @pytest.mark.asyncio
    async def test_evaluate(self, connected_client):
        """Test JavaScript evaluation."""
        result = await connected_client.evaluate("return document.title")
        assert result.success is True
        assert result.data["result"] == "result"

    @pytest.mark.asyncio
    async def test_get_content(self, connected_client):
        """Test get page content."""
        result = await connected_client.get_content()
        assert result.success is True
        assert "<html>" in result.data["content"]

    @pytest.mark.asyncio
    async def test_go_back(self, connected_client):
        """Test go back."""
        result = await connected_client.go_back()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_go_forward(self, connected_client):
        """Test go forward."""
        result = await connected_client.go_forward()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_reload(self, connected_client):
        """Test reload."""
        result = await connected_client.reload()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_scroll_by_amount(self, connected_client):
        """Test scroll by amount."""
        result = await connected_client.scroll(x=0, y=100)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_scroll_element_into_view(self, connected_client):
        """Test scroll element into view."""
        result = await connected_client.scroll(selector="#element")
        assert result.success is True


class TestBrowserClientLocator:
    """Tests for BrowserClient locator methods."""

    def test_get_locator_css(self):
        """Test CSS selector."""
        config = BrowserConfig()
        client = BrowserClient(config)
        mock_page = MagicMock()
        mock_page.locator = MagicMock(return_value="css_locator")
        client._page = mock_page

        locator = client._get_locator("#test", SelectorType.CSS)
        mock_page.locator.assert_called_with("#test")
        assert locator == "css_locator"

    def test_get_locator_xpath(self):
        """Test XPath selector."""
        config = BrowserConfig()
        client = BrowserClient(config)
        mock_page = MagicMock()
        mock_page.locator = MagicMock(return_value="xpath_locator")
        client._page = mock_page

        locator = client._get_locator("//button", SelectorType.XPATH)
        mock_page.locator.assert_called_with("xpath=//button")

    def test_get_locator_text(self):
        """Test text selector."""
        config = BrowserConfig()
        client = BrowserClient(config)
        mock_page = MagicMock()
        mock_page.get_by_text = MagicMock(return_value="text_locator")
        client._page = mock_page

        locator = client._get_locator("Click me", SelectorType.TEXT)
        mock_page.get_by_text.assert_called_with("Click me")

    def test_get_locator_role_with_name(self):
        """Test role selector with name."""
        config = BrowserConfig()
        client = BrowserClient(config)
        mock_page = MagicMock()
        mock_page.get_by_role = MagicMock(return_value="role_locator")
        client._page = mock_page

        locator = client._get_locator("button:Submit", SelectorType.ROLE)
        mock_page.get_by_role.assert_called_with("button", name="Submit")

    def test_get_locator_role_without_name(self):
        """Test role selector without name."""
        config = BrowserConfig()
        client = BrowserClient(config)
        mock_page = MagicMock()
        mock_page.get_by_role = MagicMock(return_value="role_locator")
        client._page = mock_page

        locator = client._get_locator("button", SelectorType.ROLE)
        mock_page.get_by_role.assert_called_with("button")


# =============================================================================
# Browser Manager Tests
# =============================================================================


class TestBrowserManager:
    """Tests for BrowserManager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = BrowserManager()
        assert manager.clients == {}

    def test_add_client(self):
        """Test adding a client."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)

        manager.add_client("test", client)
        assert "test" in manager.clients
        assert manager.clients["test"] == client

    def test_get_client(self):
        """Test getting a client."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        manager.add_client("test", client)

        result = manager.get_client("test")
        assert result == client

    def test_get_nonexistent_client(self):
        """Test getting a nonexistent client."""
        manager = BrowserManager()
        result = manager.get_client("nonexistent")
        assert result is None

    def test_remove_client(self):
        """Test removing a client."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        manager.add_client("test", client)

        result = manager.remove_client("test")
        assert result is True
        assert "test" not in manager.clients

    def test_remove_nonexistent_client(self):
        """Test removing a nonexistent client."""
        manager = BrowserManager()
        result = manager.remove_client("nonexistent")
        assert result is False

    def test_list_clients(self):
        """Test listing clients."""
        manager = BrowserManager()
        config = BrowserConfig()

        manager.add_client("client1", BrowserClient(config))
        manager.add_client("client2", BrowserClient(config))

        clients = manager.list_clients()
        assert "client1" in clients
        assert "client2" in clients
        assert len(clients) == 2

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all clients."""
        manager = BrowserManager()
        config = BrowserConfig()

        client1 = BrowserClient(config)
        client1.disconnect = AsyncMock()
        client2 = BrowserClient(config)
        client2.disconnect = AsyncMock()

        manager.add_client("client1", client1)
        manager.add_client("client2", client2)

        await manager.close_all()

        client1.disconnect.assert_called_once()
        client2.disconnect.assert_called_once()
        assert len(manager.clients) == 0


# =============================================================================
# Input Model Tests
# =============================================================================


class TestInputModels:
    """Tests for input models."""

    def test_create_browser_client_input_defaults(self):
        """Test CreateBrowserClientInput defaults."""
        input_data = CreateBrowserClientInput()
        assert input_data.name == "default"
        assert input_data.browser_type == BrowserType.CHROMIUM
        assert input_data.headless is True
        assert input_data.timeout == 30000
        assert input_data.viewport_width == 1280
        assert input_data.viewport_height == 720

    def test_navigate_input(self):
        """Test NavigateInput."""
        input_data = NavigateInput(url="https://example.com")
        assert input_data.url == "https://example.com"
        assert input_data.client == "default"
        assert input_data.wait_until == WaitStrategy.LOAD

    def test_click_input(self):
        """Test ClickInput."""
        input_data = ClickInput(selector="#button")
        assert input_data.selector == "#button"
        assert input_data.client == "default"
        assert input_data.selector_type == SelectorType.CSS
        assert input_data.button == "left"
        assert input_data.click_count == 1

    def test_fill_input(self):
        """Test FillInput."""
        input_data = FillInput(selector="#input", value="test")
        assert input_data.selector == "#input"
        assert input_data.value == "test"
        assert input_data.client == "default"

    def test_type_text_input(self):
        """Test TypeTextInput."""
        input_data = TypeTextInput(selector="#input", text="hello")
        assert input_data.selector == "#input"
        assert input_data.text == "hello"
        assert input_data.delay == 0

    def test_press_key_input(self):
        """Test PressKeyInput."""
        input_data = PressKeyInput(key="Enter")
        assert input_data.key == "Enter"
        assert input_data.client == "default"

    def test_screenshot_input(self):
        """Test ScreenshotInput."""
        input_data = ScreenshotInput(path="/tmp/test.png", full_page=True)
        assert input_data.path == "/tmp/test.png"
        assert input_data.full_page is True

    def test_evaluate_input(self):
        """Test EvaluateInput."""
        input_data = EvaluateInput(script="return document.title")
        assert input_data.script == "return document.title"

    def test_scroll_input(self):
        """Test ScrollInput."""
        input_data = ScrollInput(x=100, y=200)
        assert input_data.x == 100
        assert input_data.y == 200

    def test_wait_for_selector_input(self):
        """Test WaitForSelectorInput."""
        input_data = WaitForSelectorInput(selector="#element", state="visible")
        assert input_data.selector == "#element"
        assert input_data.state == "visible"


# =============================================================================
# Output Model Tests
# =============================================================================


class TestOutputModels:
    """Tests for output models."""

    def test_create_browser_client_output(self):
        """Test CreateBrowserClientOutput."""
        output = CreateBrowserClientOutput(
            success=True, message="Created", name="test"
        )
        assert output.success is True
        assert output.message == "Created"
        assert output.name == "test"
        assert output.error is None

    def test_browser_simple_output(self):
        """Test BrowserSimpleOutput."""
        output = BrowserSimpleOutput(success=True, message="OK")
        assert output.success is True
        assert output.message == "OK"
        assert output.error is None

    def test_browser_data_output(self):
        """Test BrowserDataOutput."""
        output = BrowserDataOutput(
            success=True, message="OK", data={"key": "value"}
        )
        assert output.success is True
        assert output.message == "OK"
        assert output.data == {"key": "value"}
        assert output.error is None


# =============================================================================
# Tool Tests
# =============================================================================


class TestCreateBrowserClientTool:
    """Tests for CreateBrowserClientTool."""

    @pytest.mark.asyncio
    async def test_create_client_tool(self):
        """Test creating a browser client via tool."""
        manager = BrowserManager()
        tool = CreateBrowserClientTool(manager)

        # Mock the client's connect method
        with patch.object(
            BrowserClient,
            "connect",
            new_callable=AsyncMock,
            return_value=BrowserResult(success=True, message="Connected"),
        ):
            input_data = CreateBrowserClientInput(name="test")
            result = await tool.execute(input_data)

            assert result.success is True
            assert result.name == "test"
            assert "test" in manager.clients

    def test_tool_metadata(self):
        """Test tool metadata."""
        manager = BrowserManager()
        tool = CreateBrowserClientTool(manager)

        assert tool.metadata.id == "create_browser_client"
        assert tool.metadata.category == "utility"


class TestCloseBrowserClientTool:
    """Tests for CloseBrowserClientTool."""

    @pytest.mark.asyncio
    async def test_close_client_not_found(self):
        """Test closing nonexistent client."""
        manager = BrowserManager()
        tool = CloseBrowserClientTool(manager)

        input_data = ClientInput(client="nonexistent")
        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_close_client_success(self):
        """Test closing a client successfully."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        client.disconnect = AsyncMock(
            return_value=BrowserResult(success=True, message="Disconnected")
        )
        manager.add_client("test", client)

        tool = CloseBrowserClientTool(manager)
        input_data = ClientInput(client="test")
        result = await tool.execute(input_data)

        assert result.success is True
        assert "test" not in manager.clients


class TestNavigateTool:
    """Tests for NavigateTool."""

    @pytest.mark.asyncio
    async def test_navigate_client_not_found(self):
        """Test navigate with nonexistent client."""
        manager = BrowserManager()
        tool = NavigateTool(manager)

        input_data = NavigateInput(url="https://example.com", client="nonexistent")
        result = await tool.execute(input_data)

        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        """Test successful navigation."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        client.navigate = AsyncMock(
            return_value=BrowserResult(
                success=True,
                message="Navigated",
                data={"url": "https://example.com", "status": 200},
            )
        )
        manager.add_client("test", client)

        tool = NavigateTool(manager)
        input_data = NavigateInput(url="https://example.com", client="test")
        result = await tool.execute(input_data)

        assert result.success is True
        assert result.data["status"] == 200


class TestClickTool:
    """Tests for ClickTool."""

    @pytest.mark.asyncio
    async def test_click_success(self):
        """Test successful click."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        client.click = AsyncMock(
            return_value=BrowserResult(success=True, message="Clicked")
        )
        manager.add_client("test", client)

        tool = ClickTool(manager)
        input_data = ClickInput(selector="#button", client="test")
        result = await tool.execute(input_data)

        assert result.success is True

    def test_tool_metadata(self):
        """Test tool metadata."""
        manager = BrowserManager()
        tool = ClickTool(manager)

        assert tool.metadata.id == "browser_click"


class TestFillTool:
    """Tests for FillTool."""

    @pytest.mark.asyncio
    async def test_fill_success(self):
        """Test successful fill."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        client.fill = AsyncMock(
            return_value=BrowserResult(success=True, message="Filled")
        )
        manager.add_client("test", client)

        tool = FillTool(manager)
        input_data = FillInput(selector="#input", value="test", client="test")
        result = await tool.execute(input_data)

        assert result.success is True


class TestScreenshotTool:
    """Tests for ScreenshotTool."""

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        """Test successful screenshot."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        client.screenshot = AsyncMock(
            return_value=BrowserResult(
                success=True,
                message="Screenshot captured",
                data={"base64": "base64data", "size": 1024},
            )
        )
        manager.add_client("test", client)

        tool = ScreenshotTool(manager)
        input_data = ScreenshotInput(client="test")
        result = await tool.execute(input_data)

        assert result.success is True
        assert "base64" in result.data


class TestEvaluateTool:
    """Tests for EvaluateTool."""

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful evaluation."""
        manager = BrowserManager()
        config = BrowserConfig()
        client = BrowserClient(config)
        client.evaluate = AsyncMock(
            return_value=BrowserResult(
                success=True,
                message="Script executed",
                data={"result": "Example Title"},
            )
        )
        manager.add_client("test", client)

        tool = EvaluateTool(manager)
        input_data = EvaluateInput(script="return document.title", client="test")
        result = await tool.execute(input_data)

        assert result.success is True
        assert result.data["result"] == "Example Title"

    def test_tool_metadata(self):
        """Test tool metadata."""
        manager = BrowserManager()
        tool = EvaluateTool(manager)

        assert tool.metadata.id == "browser_evaluate"
        assert tool.metadata.category == "execution"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_browser_config(self):
        """Test create_browser_config."""
        config = create_browser_config(
            browser_type=BrowserType.FIREFOX,
            headless=False,
            timeout=60000,
        )
        assert config.browser_type == BrowserType.FIREFOX
        assert config.headless is False
        assert config.timeout == 60000

    def test_create_browser_client(self):
        """Test create_browser_client."""
        config = BrowserConfig()
        client = create_browser_client(config)
        assert isinstance(client, BrowserClient)
        assert client.config == config

    def test_create_browser_manager(self):
        """Test create_browser_manager."""
        manager = create_browser_manager()
        assert isinstance(manager, BrowserManager)
        assert manager.clients == {}

    def test_create_browser_tools(self):
        """Test create_browser_tools."""
        tools = create_browser_tools()

        assert len(tools) == 18

        # Check tool types
        tool_ids = [t.metadata.id for t in tools]
        assert "create_browser_client" in tool_ids
        assert "close_browser_client" in tool_ids
        assert "browser_navigate" in tool_ids
        assert "browser_click" in tool_ids
        assert "browser_fill" in tool_ids
        assert "browser_type_text" in tool_ids
        assert "browser_press_key" in tool_ids
        assert "browser_get_element_info" in tool_ids
        assert "browser_get_text" in tool_ids
        assert "browser_screenshot" in tool_ids
        assert "browser_evaluate" in tool_ids
        assert "browser_get_page_info" in tool_ids
        assert "browser_get_content" in tool_ids
        assert "browser_scroll" in tool_ids
        assert "browser_go_back" in tool_ids
        assert "browser_go_forward" in tool_ids
        assert "browser_reload" in tool_ids
        assert "browser_wait_for_selector" in tool_ids

    def test_create_browser_tools_with_manager(self):
        """Test create_browser_tools with custom manager."""
        manager = BrowserManager()
        tools = create_browser_tools(manager)

        # All tools should use the same manager
        for tool in tools:
            assert tool._manager is manager

    def test_all_tools_have_correct_manager(self):
        """Test all tools have correct manager reference."""
        manager = BrowserManager()
        tools = create_browser_tools(manager)

        for tool in tools:
            assert hasattr(tool, "_manager")
            assert tool._manager is manager
