"""Browser automation tools for TinyLLM.

This module provides tools for browser automation using Playwright,
including navigation, element interaction, and screenshot capture.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class BrowserType(str, Enum):
    """Browser types for automation."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class WaitStrategy(str, Enum):
    """Wait strategies for page loading."""

    LOAD = "load"
    DOMCONTENTLOADED = "domcontentloaded"
    NETWORKIDLE = "networkidle"


class SelectorType(str, Enum):
    """Element selector types."""

    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    ROLE = "role"


# =============================================================================
# Configuration and Models
# =============================================================================


class BrowserConfig(BaseModel):
    """Configuration for browser automation."""

    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    timeout: int = Field(
        default=30000, description="Default timeout in milliseconds", ge=1000, le=120000
    )
    viewport_width: int = Field(default=1280, description="Viewport width", ge=320)
    viewport_height: int = Field(default=720, description="Viewport height", ge=240)
    user_agent: Optional[str] = Field(default=None, description="Custom user agent")
    ignore_https_errors: bool = Field(
        default=False, description="Ignore HTTPS certificate errors"
    )
    slow_mo: int = Field(
        default=0, description="Slow down operations by milliseconds", ge=0
    )
    proxy: Optional[str] = Field(default=None, description="Proxy server URL")
    downloads_path: Optional[str] = Field(
        default=None, description="Path for downloaded files"
    )


@dataclass
class ElementInfo:
    """Information about a page element."""

    selector: str
    tag_name: str
    text: str
    attributes: dict[str, str]
    is_visible: bool
    is_enabled: bool
    bounding_box: Optional[dict[str, float]] = None


@dataclass
class PageInfo:
    """Information about the current page."""

    url: str
    title: str
    viewport: dict[str, int]
    has_focus: bool


@dataclass
class BrowserResult:
    """Result from a browser operation."""

    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }


# =============================================================================
# Browser Client
# =============================================================================


class BrowserClient:
    """Client for browser automation using Playwright."""

    def __init__(self, config: BrowserConfig):
        """Initialize browser client.

        Args:
            config: Browser configuration.
        """
        self.config = config
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._is_connected = False

    async def connect(self) -> BrowserResult:
        """Connect to the browser.

        Returns:
            Result of the connection attempt.
        """
        try:
            # Import playwright dynamically
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()

            # Get browser launcher
            browser_launcher = getattr(self._playwright, self.config.browser_type.value)

            # Launch browser
            launch_options: dict[str, Any] = {
                "headless": self.config.headless,
                "slow_mo": self.config.slow_mo,
            }

            if self.config.proxy:
                launch_options["proxy"] = {"server": self.config.proxy}

            if self.config.downloads_path:
                launch_options["downloads_path"] = self.config.downloads_path

            self._browser = await browser_launcher.launch(**launch_options)

            # Create context
            context_options: dict[str, Any] = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                "ignore_https_errors": self.config.ignore_https_errors,
            }

            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent

            self._context = await self._browser.new_context(**context_options)
            self._context.set_default_timeout(self.config.timeout)

            # Create page
            self._page = await self._context.new_page()
            self._is_connected = True

            logger.info(f"Connected to {self.config.browser_type.value} browser")
            return BrowserResult(
                success=True,
                message=f"Connected to {self.config.browser_type.value} browser",
            )

        except ImportError:
            return BrowserResult(
                success=False,
                message="Playwright not installed",
                error="Please install playwright: pip install playwright && playwright install",
            )
        except Exception as e:
            logger.error(f"Failed to connect to browser: {e}")
            return BrowserResult(
                success=False, message="Failed to connect to browser", error=str(e)
            )

    async def disconnect(self) -> BrowserResult:
        """Disconnect from the browser.

        Returns:
            Result of the disconnection.
        """
        try:
            if self._page:
                await self._page.close()
                self._page = None

            if self._context:
                await self._context.close()
                self._context = None

            if self._browser:
                await self._browser.close()
                self._browser = None

            if self._playwright:
                await self._playwright.stop()
                self._playwright = None

            self._is_connected = False
            logger.info("Disconnected from browser")
            return BrowserResult(success=True, message="Disconnected from browser")

        except Exception as e:
            logger.error(f"Failed to disconnect from browser: {e}")
            return BrowserResult(
                success=False, message="Failed to disconnect from browser", error=str(e)
            )

    @property
    def is_connected(self) -> bool:
        """Check if connected to browser."""
        return self._is_connected and self._page is not None

    async def navigate(
        self, url: str, wait_until: WaitStrategy = WaitStrategy.LOAD
    ) -> BrowserResult:
        """Navigate to a URL.

        Args:
            url: URL to navigate to.
            wait_until: Wait strategy for page loading.

        Returns:
            Result of the navigation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            response = await self._page.goto(url, wait_until=wait_until.value)
            status = response.status if response else 0

            return BrowserResult(
                success=True,
                message=f"Navigated to {url}",
                data={"url": url, "status": status},
            )

        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to navigate to {url}", error=str(e)
            )

    async def get_page_info(self) -> BrowserResult:
        """Get information about the current page.

        Returns:
            Result with page information.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            url = self._page.url
            title = await self._page.title()
            viewport = self._page.viewport_size

            info = PageInfo(
                url=url,
                title=title,
                viewport=viewport or {"width": 0, "height": 0},
                has_focus=True,
            )

            return BrowserResult(
                success=True,
                message="Retrieved page info",
                data={
                    "url": info.url,
                    "title": info.title,
                    "viewport": info.viewport,
                    "has_focus": info.has_focus,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get page info: {e}")
            return BrowserResult(
                success=False, message="Failed to get page info", error=str(e)
            )

    async def click(
        self,
        selector: str,
        selector_type: SelectorType = SelectorType.CSS,
        button: str = "left",
        click_count: int = 1,
        delay: int = 0,
    ) -> BrowserResult:
        """Click on an element.

        Args:
            selector: Element selector.
            selector_type: Type of selector.
            button: Mouse button (left, right, middle).
            click_count: Number of clicks.
            delay: Delay between clicks in milliseconds.

        Returns:
            Result of the click operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            await locator.click(button=button, click_count=click_count, delay=delay)

            return BrowserResult(
                success=True, message=f"Clicked on element: {selector}"
            )

        except Exception as e:
            logger.error(f"Failed to click on {selector}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to click on {selector}", error=str(e)
            )

    async def fill(
        self, selector: str, value: str, selector_type: SelectorType = SelectorType.CSS
    ) -> BrowserResult:
        """Fill a text input.

        Args:
            selector: Element selector.
            value: Value to fill.
            selector_type: Type of selector.

        Returns:
            Result of the fill operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            await locator.fill(value)

            return BrowserResult(
                success=True, message=f"Filled element {selector} with value"
            )

        except Exception as e:
            logger.error(f"Failed to fill {selector}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to fill {selector}", error=str(e)
            )

    async def type_text(
        self,
        selector: str,
        text: str,
        selector_type: SelectorType = SelectorType.CSS,
        delay: int = 0,
    ) -> BrowserResult:
        """Type text character by character.

        Args:
            selector: Element selector.
            text: Text to type.
            selector_type: Type of selector.
            delay: Delay between keystrokes in milliseconds.

        Returns:
            Result of the type operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            await locator.type(text, delay=delay)

            return BrowserResult(success=True, message=f"Typed text into {selector}")

        except Exception as e:
            logger.error(f"Failed to type into {selector}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to type into {selector}", error=str(e)
            )

    async def press_key(self, key: str) -> BrowserResult:
        """Press a keyboard key.

        Args:
            key: Key to press (e.g., 'Enter', 'Escape', 'Tab').

        Returns:
            Result of the key press.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            await self._page.keyboard.press(key)
            return BrowserResult(success=True, message=f"Pressed key: {key}")

        except Exception as e:
            logger.error(f"Failed to press key {key}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to press key {key}", error=str(e)
            )

    async def select_option(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
        selector_type: SelectorType = SelectorType.CSS,
    ) -> BrowserResult:
        """Select an option from a dropdown.

        Args:
            selector: Element selector.
            value: Option value to select.
            label: Option label to select.
            index: Option index to select.
            selector_type: Type of selector.

        Returns:
            Result of the select operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)

            if value is not None:
                await locator.select_option(value=value)
            elif label is not None:
                await locator.select_option(label=label)
            elif index is not None:
                await locator.select_option(index=index)
            else:
                return BrowserResult(
                    success=False,
                    message="Must specify value, label, or index",
                    error="Missing selection criteria",
                )

            return BrowserResult(
                success=True, message=f"Selected option from {selector}"
            )

        except Exception as e:
            logger.error(f"Failed to select option from {selector}: {e}")
            return BrowserResult(
                success=False,
                message=f"Failed to select option from {selector}",
                error=str(e),
            )

    async def hover(
        self, selector: str, selector_type: SelectorType = SelectorType.CSS
    ) -> BrowserResult:
        """Hover over an element.

        Args:
            selector: Element selector.
            selector_type: Type of selector.

        Returns:
            Result of the hover operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            await locator.hover()

            return BrowserResult(success=True, message=f"Hovered over {selector}")

        except Exception as e:
            logger.error(f"Failed to hover over {selector}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to hover over {selector}", error=str(e)
            )

    async def wait_for_selector(
        self,
        selector: str,
        selector_type: SelectorType = SelectorType.CSS,
        state: str = "visible",
        timeout: Optional[int] = None,
    ) -> BrowserResult:
        """Wait for an element to appear.

        Args:
            selector: Element selector.
            selector_type: Type of selector.
            state: Element state to wait for (visible, hidden, attached, detached).
            timeout: Timeout in milliseconds.

        Returns:
            Result of the wait operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            await locator.wait_for(state=state, timeout=timeout)

            return BrowserResult(
                success=True, message=f"Element {selector} is {state}"
            )

        except Exception as e:
            logger.error(f"Failed waiting for {selector}: {e}")
            return BrowserResult(
                success=False, message=f"Failed waiting for {selector}", error=str(e)
            )

    async def get_element_info(
        self, selector: str, selector_type: SelectorType = SelectorType.CSS
    ) -> BrowserResult:
        """Get information about an element.

        Args:
            selector: Element selector.
            selector_type: Type of selector.

        Returns:
            Result with element information.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)

            # Get element properties
            tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
            text = await locator.inner_text()
            is_visible = await locator.is_visible()
            is_enabled = await locator.is_enabled()
            bounding_box = await locator.bounding_box()

            # Get attributes
            attributes = await locator.evaluate(
                """el => {
                const attrs = {};
                for (const attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }"""
            )

            info = ElementInfo(
                selector=selector,
                tag_name=tag_name,
                text=text,
                attributes=attributes,
                is_visible=is_visible,
                is_enabled=is_enabled,
                bounding_box=bounding_box,
            )

            return BrowserResult(
                success=True,
                message="Retrieved element info",
                data={
                    "selector": info.selector,
                    "tag_name": info.tag_name,
                    "text": info.text,
                    "attributes": info.attributes,
                    "is_visible": info.is_visible,
                    "is_enabled": info.is_enabled,
                    "bounding_box": info.bounding_box,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get element info for {selector}: {e}")
            return BrowserResult(
                success=False,
                message=f"Failed to get element info for {selector}",
                error=str(e),
            )

    async def get_text(
        self, selector: str, selector_type: SelectorType = SelectorType.CSS
    ) -> BrowserResult:
        """Get text content of an element.

        Args:
            selector: Element selector.
            selector_type: Type of selector.

        Returns:
            Result with text content.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            text = await locator.inner_text()

            return BrowserResult(
                success=True, message="Retrieved text content", data={"text": text}
            )

        except Exception as e:
            logger.error(f"Failed to get text from {selector}: {e}")
            return BrowserResult(
                success=False, message=f"Failed to get text from {selector}", error=str(e)
            )

    async def get_attribute(
        self,
        selector: str,
        attribute: str,
        selector_type: SelectorType = SelectorType.CSS,
    ) -> BrowserResult:
        """Get an attribute of an element.

        Args:
            selector: Element selector.
            attribute: Attribute name.
            selector_type: Type of selector.

        Returns:
            Result with attribute value.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            locator = self._get_locator(selector, selector_type)
            value = await locator.get_attribute(attribute)

            return BrowserResult(
                success=True,
                message=f"Retrieved attribute {attribute}",
                data={"attribute": attribute, "value": value},
            )

        except Exception as e:
            logger.error(f"Failed to get attribute {attribute} from {selector}: {e}")
            return BrowserResult(
                success=False,
                message=f"Failed to get attribute {attribute} from {selector}",
                error=str(e),
            )

    async def screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
        selector_type: SelectorType = SelectorType.CSS,
    ) -> BrowserResult:
        """Take a screenshot.

        Args:
            path: Path to save the screenshot.
            full_page: Capture the full scrollable page.
            selector: Optional element selector for element screenshot.
            selector_type: Type of selector.

        Returns:
            Result with screenshot data.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            if selector:
                locator = self._get_locator(selector, selector_type)
                screenshot_bytes = await locator.screenshot(path=path)
            else:
                screenshot_bytes = await self._page.screenshot(
                    path=path, full_page=full_page
                )

            # Convert to base64 if no path specified
            if not path:
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                return BrowserResult(
                    success=True,
                    message="Screenshot captured",
                    data={"base64": screenshot_b64, "size": len(screenshot_bytes)},
                )
            else:
                return BrowserResult(
                    success=True,
                    message=f"Screenshot saved to {path}",
                    data={"path": path, "size": len(screenshot_bytes)},
                )

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return BrowserResult(
                success=False, message="Failed to take screenshot", error=str(e)
            )

    async def evaluate(self, script: str) -> BrowserResult:
        """Execute JavaScript in the browser.

        Args:
            script: JavaScript code to execute.

        Returns:
            Result with script return value.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            result = await self._page.evaluate(script)

            return BrowserResult(
                success=True, message="Script executed", data={"result": result}
            )

        except Exception as e:
            logger.error(f"Failed to execute script: {e}")
            return BrowserResult(
                success=False, message="Failed to execute script", error=str(e)
            )

    async def get_content(self) -> BrowserResult:
        """Get the page HTML content.

        Returns:
            Result with HTML content.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            content = await self._page.content()

            return BrowserResult(
                success=True,
                message="Retrieved page content",
                data={"content": content, "length": len(content)},
            )

        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            return BrowserResult(
                success=False, message="Failed to get page content", error=str(e)
            )

    async def go_back(self) -> BrowserResult:
        """Navigate back in browser history.

        Returns:
            Result of navigation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            await self._page.go_back()
            return BrowserResult(success=True, message="Navigated back")

        except Exception as e:
            logger.error(f"Failed to go back: {e}")
            return BrowserResult(
                success=False, message="Failed to go back", error=str(e)
            )

    async def go_forward(self) -> BrowserResult:
        """Navigate forward in browser history.

        Returns:
            Result of navigation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            await self._page.go_forward()
            return BrowserResult(success=True, message="Navigated forward")

        except Exception as e:
            logger.error(f"Failed to go forward: {e}")
            return BrowserResult(
                success=False, message="Failed to go forward", error=str(e)
            )

    async def reload(self) -> BrowserResult:
        """Reload the current page.

        Returns:
            Result of reload.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            await self._page.reload()
            return BrowserResult(success=True, message="Page reloaded")

        except Exception as e:
            logger.error(f"Failed to reload page: {e}")
            return BrowserResult(
                success=False, message="Failed to reload page", error=str(e)
            )

    async def scroll(
        self, x: int = 0, y: int = 0, selector: Optional[str] = None
    ) -> BrowserResult:
        """Scroll the page or an element.

        Args:
            x: Horizontal scroll amount.
            y: Vertical scroll amount.
            selector: Optional element selector to scroll into view.

        Returns:
            Result of scroll operation.
        """
        if not self.is_connected:
            return BrowserResult(
                success=False, message="Not connected to browser", error="Not connected"
            )

        try:
            if selector:
                locator = self._get_locator(selector, SelectorType.CSS)
                await locator.scroll_into_view_if_needed()
                message = f"Scrolled element {selector} into view"
            else:
                await self._page.evaluate(f"window.scrollBy({x}, {y})")
                message = f"Scrolled by ({x}, {y})"

            return BrowserResult(success=True, message=message)

        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            return BrowserResult(
                success=False, message="Failed to scroll", error=str(e)
            )

    def _get_locator(self, selector: str, selector_type: SelectorType):
        """Get a Playwright locator for the given selector.

        Args:
            selector: Element selector.
            selector_type: Type of selector.

        Returns:
            Playwright locator.
        """
        if selector_type == SelectorType.CSS:
            return self._page.locator(selector)
        elif selector_type == SelectorType.XPATH:
            return self._page.locator(f"xpath={selector}")
        elif selector_type == SelectorType.TEXT:
            return self._page.get_by_text(selector)
        elif selector_type == SelectorType.ROLE:
            # Parse role selector: "button:Submit" -> role=button, name=Submit
            parts = selector.split(":", 1)
            role = parts[0]
            name = parts[1] if len(parts) > 1 else None
            if name:
                return self._page.get_by_role(role, name=name)
            return self._page.get_by_role(role)
        else:
            return self._page.locator(selector)


# =============================================================================
# Browser Manager
# =============================================================================


@dataclass
class BrowserManager:
    """Manager for multiple browser clients."""

    clients: dict[str, BrowserClient] = field(default_factory=dict)

    def add_client(self, name: str, client: BrowserClient) -> None:
        """Add a browser client.

        Args:
            name: Client name.
            client: Browser client instance.
        """
        self.clients[name] = client

    def get_client(self, name: str) -> Optional[BrowserClient]:
        """Get a browser client by name.

        Args:
            name: Client name.

        Returns:
            Browser client or None.
        """
        return self.clients.get(name)

    def remove_client(self, name: str) -> bool:
        """Remove a browser client.

        Args:
            name: Client name.

        Returns:
            True if removed, False if not found.
        """
        if name in self.clients:
            del self.clients[name]
            return True
        return False

    def list_clients(self) -> list[str]:
        """List all client names.

        Returns:
            List of client names.
        """
        return list(self.clients.keys())

    async def close_all(self) -> None:
        """Close all browser clients."""
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()


# =============================================================================
# Tool Input/Output Models
# =============================================================================


class CreateBrowserClientInput(BaseModel):
    """Input for creating a browser client."""

    name: str = Field(default="default", description="Name for the browser client")
    browser_type: BrowserType = Field(
        default=BrowserType.CHROMIUM, description="Browser type"
    )
    headless: bool = Field(default=True, description="Run in headless mode")
    timeout: int = Field(default=30000, description="Default timeout in milliseconds")
    viewport_width: int = Field(default=1280, description="Viewport width")
    viewport_height: int = Field(default=720, description="Viewport height")


class CreateBrowserClientOutput(BaseModel):
    """Output for creating a browser client."""

    success: bool = Field(description="Whether creation was successful")
    message: str = Field(description="Result message")
    name: str = Field(description="Client name")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class NavigateInput(BaseModel):
    """Input for navigation."""

    url: str = Field(description="URL to navigate to")
    client: str = Field(default="default", description="Browser client name")
    wait_until: WaitStrategy = Field(
        default=WaitStrategy.LOAD, description="Wait strategy"
    )


class BrowserSimpleOutput(BaseModel):
    """Simple output for browser operations."""

    success: bool = Field(description="Whether operation was successful")
    message: str = Field(description="Result message")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BrowserDataOutput(BaseModel):
    """Output with data for browser operations."""

    success: bool = Field(description="Whether operation was successful")
    message: str = Field(description="Result message")
    data: Optional[dict[str, Any]] = Field(default=None, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ClickInput(BaseModel):
    """Input for click operation."""

    selector: str = Field(description="Element selector")
    client: str = Field(default="default", description="Browser client name")
    selector_type: SelectorType = Field(
        default=SelectorType.CSS, description="Selector type"
    )
    button: str = Field(default="left", description="Mouse button")
    click_count: int = Field(default=1, description="Number of clicks")


class FillInput(BaseModel):
    """Input for fill operation."""

    selector: str = Field(description="Element selector")
    value: str = Field(description="Value to fill")
    client: str = Field(default="default", description="Browser client name")
    selector_type: SelectorType = Field(
        default=SelectorType.CSS, description="Selector type"
    )


class TypeTextInput(BaseModel):
    """Input for type text operation."""

    selector: str = Field(description="Element selector")
    text: str = Field(description="Text to type")
    client: str = Field(default="default", description="Browser client name")
    selector_type: SelectorType = Field(
        default=SelectorType.CSS, description="Selector type"
    )
    delay: int = Field(default=0, description="Delay between keystrokes in ms")


class PressKeyInput(BaseModel):
    """Input for press key operation."""

    key: str = Field(description="Key to press (e.g., 'Enter', 'Escape')")
    client: str = Field(default="default", description="Browser client name")


class GetElementInput(BaseModel):
    """Input for getting element info."""

    selector: str = Field(description="Element selector")
    client: str = Field(default="default", description="Browser client name")
    selector_type: SelectorType = Field(
        default=SelectorType.CSS, description="Selector type"
    )


class ScreenshotInput(BaseModel):
    """Input for screenshot operation."""

    client: str = Field(default="default", description="Browser client name")
    path: Optional[str] = Field(default=None, description="Path to save screenshot")
    full_page: bool = Field(default=False, description="Capture full page")
    selector: Optional[str] = Field(
        default=None, description="Element selector for element screenshot"
    )


class EvaluateInput(BaseModel):
    """Input for JavaScript evaluation."""

    script: str = Field(description="JavaScript code to execute")
    client: str = Field(default="default", description="Browser client name")


class ScrollInput(BaseModel):
    """Input for scroll operation."""

    client: str = Field(default="default", description="Browser client name")
    x: int = Field(default=0, description="Horizontal scroll amount")
    y: int = Field(default=0, description="Vertical scroll amount")
    selector: Optional[str] = Field(
        default=None, description="Element to scroll into view"
    )


class WaitForSelectorInput(BaseModel):
    """Input for wait for selector operation."""

    selector: str = Field(description="Element selector")
    client: str = Field(default="default", description="Browser client name")
    selector_type: SelectorType = Field(
        default=SelectorType.CSS, description="Selector type"
    )
    state: str = Field(default="visible", description="State to wait for")
    timeout: Optional[int] = Field(default=None, description="Timeout in milliseconds")


class ClientInput(BaseModel):
    """Input for client-only operations."""

    client: str = Field(default="default", description="Browser client name")


# =============================================================================
# Tools
# =============================================================================


class CreateBrowserClientTool(BaseTool[CreateBrowserClientInput, CreateBrowserClientOutput]):
    """Tool for creating a browser client."""

    metadata = ToolMetadata(
        id="create_browser_client",
        name="Create Browser Client",
        description="Create a new browser automation client",
        category="utility",
    )
    input_type = CreateBrowserClientInput
    output_type = CreateBrowserClientOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(
        self, input_data: CreateBrowserClientInput
    ) -> CreateBrowserClientOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        config = BrowserConfig(
            browser_type=input_data.browser_type,
            headless=input_data.headless,
            timeout=input_data.timeout,
            viewport_width=input_data.viewport_width,
            viewport_height=input_data.viewport_height,
        )

        client = BrowserClient(config)
        result = await client.connect()

        if result.success:
            self._manager.add_client(input_data.name, client)

        return CreateBrowserClientOutput(
            success=result.success,
            message=result.message,
            name=input_data.name,
            error=result.error,
        )


class CloseBrowserClientTool(BaseTool[ClientInput, BrowserSimpleOutput]):
    """Tool for closing a browser client."""

    metadata = ToolMetadata(
        id="close_browser_client",
        name="Close Browser Client",
        description="Close a browser client and disconnect",
        category="utility",
    )
    input_type = ClientInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClientInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.disconnect()
        if result.success:
            self._manager.remove_client(input_data.client)

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class NavigateTool(BaseTool[NavigateInput, BrowserDataOutput]):
    """Tool for navigating to a URL."""

    metadata = ToolMetadata(
        id="browser_navigate",
        name="Navigate",
        description="Navigate to a URL in the browser",
        category="utility",
    )
    input_type = NavigateInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: NavigateInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.navigate(input_data.url, input_data.wait_until)

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class ClickTool(BaseTool[ClickInput, BrowserSimpleOutput]):
    """Tool for clicking an element."""

    metadata = ToolMetadata(
        id="browser_click",
        name="Click",
        description="Click on an element in the browser",
        category="utility",
    )
    input_type = ClickInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClickInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.click(
            input_data.selector,
            input_data.selector_type,
            input_data.button,
            input_data.click_count,
        )

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class FillTool(BaseTool[FillInput, BrowserSimpleOutput]):
    """Tool for filling a form field."""

    metadata = ToolMetadata(
        id="browser_fill",
        name="Fill",
        description="Fill a form field in the browser",
        category="utility",
    )
    input_type = FillInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: FillInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.fill(
            input_data.selector, input_data.value, input_data.selector_type
        )

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class TypeTextTool(BaseTool[TypeTextInput, BrowserSimpleOutput]):
    """Tool for typing text character by character."""

    metadata = ToolMetadata(
        id="browser_type_text",
        name="Type Text",
        description="Type text character by character in the browser",
        category="utility",
    )
    input_type = TypeTextInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: TypeTextInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.type_text(
            input_data.selector,
            input_data.text,
            input_data.selector_type,
            input_data.delay,
        )

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class PressKeyTool(BaseTool[PressKeyInput, BrowserSimpleOutput]):
    """Tool for pressing a keyboard key."""

    metadata = ToolMetadata(
        id="browser_press_key",
        name="Press Key",
        description="Press a keyboard key in the browser",
        category="utility",
    )
    input_type = PressKeyInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: PressKeyInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.press_key(input_data.key)

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class GetElementInfoTool(BaseTool[GetElementInput, BrowserDataOutput]):
    """Tool for getting element information."""

    metadata = ToolMetadata(
        id="browser_get_element_info",
        name="Get Element Info",
        description="Get information about an element in the browser",
        category="utility",
    )
    input_type = GetElementInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: GetElementInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.get_element_info(
            input_data.selector, input_data.selector_type
        )

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class GetTextTool(BaseTool[GetElementInput, BrowserDataOutput]):
    """Tool for getting element text."""

    metadata = ToolMetadata(
        id="browser_get_text",
        name="Get Text",
        description="Get text content of an element in the browser",
        category="utility",
    )
    input_type = GetElementInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: GetElementInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.get_text(input_data.selector, input_data.selector_type)

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class ScreenshotTool(BaseTool[ScreenshotInput, BrowserDataOutput]):
    """Tool for taking screenshots."""

    metadata = ToolMetadata(
        id="browser_screenshot",
        name="Screenshot",
        description="Take a screenshot of the browser page",
        category="utility",
    )
    input_type = ScreenshotInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ScreenshotInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.screenshot(
            input_data.path, input_data.full_page, input_data.selector
        )

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class EvaluateTool(BaseTool[EvaluateInput, BrowserDataOutput]):
    """Tool for executing JavaScript."""

    metadata = ToolMetadata(
        id="browser_evaluate",
        name="Evaluate",
        description="Execute JavaScript in the browser",
        category="execution",
    )
    input_type = EvaluateInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: EvaluateInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.evaluate(input_data.script)

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class GetPageInfoTool(BaseTool[ClientInput, BrowserDataOutput]):
    """Tool for getting page information."""

    metadata = ToolMetadata(
        id="browser_get_page_info",
        name="Get Page Info",
        description="Get information about the current browser page",
        category="utility",
    )
    input_type = ClientInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClientInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.get_page_info()

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class GetContentTool(BaseTool[ClientInput, BrowserDataOutput]):
    """Tool for getting page HTML content."""

    metadata = ToolMetadata(
        id="browser_get_content",
        name="Get Content",
        description="Get the HTML content of the current page",
        category="utility",
    )
    input_type = ClientInput
    output_type = BrowserDataOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClientInput) -> BrowserDataOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserDataOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.get_content()

        return BrowserDataOutput(
            success=result.success,
            message=result.message,
            data=result.data,
            error=result.error,
        )


class ScrollTool(BaseTool[ScrollInput, BrowserSimpleOutput]):
    """Tool for scrolling the page."""

    metadata = ToolMetadata(
        id="browser_scroll",
        name="Scroll",
        description="Scroll the browser page or element into view",
        category="utility",
    )
    input_type = ScrollInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ScrollInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.scroll(input_data.x, input_data.y, input_data.selector)

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class GoBackTool(BaseTool[ClientInput, BrowserSimpleOutput]):
    """Tool for navigating back."""

    metadata = ToolMetadata(
        id="browser_go_back",
        name="Go Back",
        description="Navigate back in browser history",
        category="utility",
    )
    input_type = ClientInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClientInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.go_back()

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class GoForwardTool(BaseTool[ClientInput, BrowserSimpleOutput]):
    """Tool for navigating forward."""

    metadata = ToolMetadata(
        id="browser_go_forward",
        name="Go Forward",
        description="Navigate forward in browser history",
        category="utility",
    )
    input_type = ClientInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClientInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.go_forward()

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class ReloadTool(BaseTool[ClientInput, BrowserSimpleOutput]):
    """Tool for reloading the page."""

    metadata = ToolMetadata(
        id="browser_reload",
        name="Reload",
        description="Reload the current browser page",
        category="utility",
    )
    input_type = ClientInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: ClientInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.reload()

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


class WaitForSelectorTool(BaseTool[WaitForSelectorInput, BrowserSimpleOutput]):
    """Tool for waiting for an element."""

    metadata = ToolMetadata(
        id="browser_wait_for_selector",
        name="Wait For Selector",
        description="Wait for an element to appear in the browser",
        category="utility",
    )
    input_type = WaitForSelectorInput
    output_type = BrowserSimpleOutput

    def __init__(self, manager: BrowserManager):
        """Initialize tool with manager.

        Args:
            manager: Browser manager instance.
        """
        self._manager = manager

    async def execute(self, input_data: WaitForSelectorInput) -> BrowserSimpleOutput:
        """Execute the tool.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.
        """
        client = self._manager.get_client(input_data.client)
        if not client:
            return BrowserSimpleOutput(
                success=False,
                message=f"Client '{input_data.client}' not found",
                error="Client not found",
            )

        result = await client.wait_for_selector(
            input_data.selector,
            input_data.selector_type,
            input_data.state,
            input_data.timeout,
        )

        return BrowserSimpleOutput(
            success=result.success, message=result.message, error=result.error
        )


# =============================================================================
# Helper Functions
# =============================================================================


def create_browser_config(
    browser_type: BrowserType = BrowserType.CHROMIUM,
    headless: bool = True,
    timeout: int = 30000,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    user_agent: Optional[str] = None,
    ignore_https_errors: bool = False,
    slow_mo: int = 0,
    proxy: Optional[str] = None,
    downloads_path: Optional[str] = None,
) -> BrowserConfig:
    """Create a browser configuration.

    Args:
        browser_type: Type of browser to use.
        headless: Run in headless mode.
        timeout: Default timeout in milliseconds.
        viewport_width: Viewport width.
        viewport_height: Viewport height.
        user_agent: Custom user agent.
        ignore_https_errors: Ignore HTTPS certificate errors.
        slow_mo: Slow down operations by milliseconds.
        proxy: Proxy server URL.
        downloads_path: Path for downloaded files.

    Returns:
        Browser configuration.
    """
    return BrowserConfig(
        browser_type=browser_type,
        headless=headless,
        timeout=timeout,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        user_agent=user_agent,
        ignore_https_errors=ignore_https_errors,
        slow_mo=slow_mo,
        proxy=proxy,
        downloads_path=downloads_path,
    )


def create_browser_client(config: BrowserConfig) -> BrowserClient:
    """Create a browser client.

    Args:
        config: Browser configuration.

    Returns:
        Browser client instance.
    """
    return BrowserClient(config)


def create_browser_manager() -> BrowserManager:
    """Create a browser manager.

    Returns:
        Browser manager instance.
    """
    return BrowserManager()


def create_browser_tools(
    manager: Optional[BrowserManager] = None,
) -> list[BaseTool]:
    """Create all browser tools.

    Args:
        manager: Optional browser manager. Creates one if not provided.

    Returns:
        List of browser tools.
    """
    if manager is None:
        manager = create_browser_manager()

    return [
        CreateBrowserClientTool(manager),
        CloseBrowserClientTool(manager),
        NavigateTool(manager),
        ClickTool(manager),
        FillTool(manager),
        TypeTextTool(manager),
        PressKeyTool(manager),
        GetElementInfoTool(manager),
        GetTextTool(manager),
        ScreenshotTool(manager),
        EvaluateTool(manager),
        GetPageInfoTool(manager),
        GetContentTool(manager),
        ScrollTool(manager),
        GoBackTool(manager),
        GoForwardTool(manager),
        ReloadTool(manager),
        WaitForSelectorTool(manager),
    ]
