"""Tool registry for TinyLLM.

This module provides a registry for tools that allows dynamic
registration, lookup, and discovery.
"""

import importlib
import importlib.util
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from tinyllm.tools.base import BaseTool, ToolMetadata

logger = logging.getLogger(__name__)


class ToolDiscoveryError(Exception):
    """Error during tool discovery."""

    pass


class ToolRegistry:
    """Registry for available tools.

    The ToolRegistry provides a central location for registering,
    looking up, and discovering tools. Supports:
    - Manual registration
    - Auto-discovery from modules/packages
    - Entry point based discovery
    - Dynamic loading/unloading
    - Capability querying
    """

    _tools: Dict[str, BaseTool] = {}
    _tool_classes: Dict[str, Type[BaseTool]] = {}
    _discovery_callbacks: List[Callable[[], List[BaseTool]]] = []

    @classmethod
    def register(cls, tool: BaseTool) -> BaseTool:
        """Register a tool instance.

        Args:
            tool: Tool instance to register.

        Returns:
            The registered tool.
        """
        cls._tools[tool.metadata.id] = tool
        cls._tool_classes[tool.metadata.id] = type(tool)
        logger.debug(f"Registered tool: {tool.metadata.id}")
        return tool

    @classmethod
    def register_class(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """Register a tool class (decorator).

        Args:
            tool_class: Tool class to register.

        Returns:
            The registered tool class.
        """
        instance = tool_class()
        cls.register(instance)
        return tool_class

    @classmethod
    def unregister(cls, tool_id: str) -> bool:
        """Unregister a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            True if tool was unregistered.
        """
        if tool_id in cls._tools:
            del cls._tools[tool_id]
            cls._tool_classes.pop(tool_id, None)
            logger.debug(f"Unregistered tool: {tool_id}")
            return True
        return False

    @classmethod
    def get(cls, tool_id: str) -> Optional[BaseTool]:
        """Get a tool by ID.

        Args:
            tool_id: Tool identifier.

        Returns:
            Tool instance or None if not found.
        """
        return cls._tools.get(tool_id)

    @classmethod
    def has(cls, tool_id: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_id: Tool identifier.

        Returns:
            True if tool is registered.
        """
        return tool_id in cls._tools

    @classmethod
    def list_tools(cls) -> List[ToolMetadata]:
        """List all available tools.

        Returns:
            List of tool metadata.
        """
        return [t.metadata for t in cls._tools.values()]

    @classmethod
    def list_enabled_tools(cls) -> List[ToolMetadata]:
        """List only enabled tools.

        Returns:
            List of enabled tool metadata.
        """
        return [t.metadata for t in cls._tools.values() if t.config.enabled]

    @classmethod
    def get_tool_ids(cls) -> List[str]:
        """Get all registered tool IDs.

        Returns:
            List of tool IDs.
        """
        return list(cls._tools.keys())

    @classmethod
    def get_tool_descriptions(cls) -> str:
        """Get descriptions for all enabled tools.

        This is useful for including in prompts to tell
        the LLM what tools are available.

        Returns:
            Formatted string of tool descriptions.
        """
        descriptions = []
        for tool in cls._tools.values():
            if tool.config.enabled:
                descriptions.append(tool.get_schema_description())
        return "\n\n".join(descriptions)

    @classmethod
    def get_tools_by_category(cls, category: str) -> List[BaseTool]:
        """Get all tools in a category.

        Args:
            category: Category to filter by.

        Returns:
            List of tools in that category.
        """
        return [t for t in cls._tools.values() if t.metadata.category == category]

    @classmethod
    def get_tools_by_version(cls, version: str) -> List[BaseTool]:
        """Get all tools matching a version pattern.

        Args:
            version: Version string or prefix (e.g., "1.0", "2").

        Returns:
            List of matching tools.
        """
        return [
            t for t in cls._tools.values() if t.metadata.version.startswith(version)
        ]

    @classmethod
    def search_tools(
        cls,
        query: str,
        categories: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[BaseTool]:
        """Search tools by name or description.

        Args:
            query: Search query (case-insensitive).
            categories: Optional category filter.
            enabled_only: Only return enabled tools.

        Returns:
            List of matching tools.
        """
        query = query.lower()
        results = []

        for tool in cls._tools.values():
            if enabled_only and not tool.config.enabled:
                continue
            if categories and tool.metadata.category not in categories:
                continue

            if (
                query in tool.metadata.name.lower()
                or query in tool.metadata.description.lower()
                or query in tool.metadata.id.lower()
            ):
                results.append(tool)

        return results

    @classmethod
    def get_categories(cls) -> Set[str]:
        """Get all unique tool categories.

        Returns:
            Set of category names.
        """
        return {t.metadata.category for t in cls._tools.values()}

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (for testing)."""
        cls._tools.clear()
        cls._tool_classes.clear()

    # Discovery methods

    @classmethod
    def discover_from_module(cls, module_name: str) -> List[BaseTool]:
        """Discover and register tools from a Python module.

        Args:
            module_name: Fully qualified module name.

        Returns:
            List of discovered tools.

        Raises:
            ToolDiscoveryError: If module cannot be loaded.
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ToolDiscoveryError(f"Failed to import module {module_name}: {e}")

        discovered = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTool)
                and obj is not BaseTool
                and hasattr(obj, "metadata")
            ):
                try:
                    instance = obj()
                    cls.register(instance)
                    discovered.append(instance)
                    logger.info(f"Discovered tool {instance.metadata.id} from {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to instantiate {name}: {e}")

        return discovered

    @classmethod
    def discover_from_package(
        cls, package_name: str, recursive: bool = True
    ) -> List[BaseTool]:
        """Discover and register tools from a Python package.

        Args:
            package_name: Fully qualified package name.
            recursive: Whether to scan subpackages.

        Returns:
            List of discovered tools.
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            raise ToolDiscoveryError(f"Failed to import package {package_name}: {e}")

        discovered = []

        # Get tools from the package itself
        discovered.extend(cls.discover_from_module(package_name))

        # Scan submodules
        if hasattr(package, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__, prefix=package_name + "."
            ):
                if not recursive and ispkg:
                    continue
                try:
                    discovered.extend(cls.discover_from_module(modname))
                except ToolDiscoveryError as e:
                    logger.warning(str(e))

        return discovered

    @classmethod
    def discover_from_directory(cls, directory: str | Path) -> List[BaseTool]:
        """Discover and register tools from a directory of Python files.

        Args:
            directory: Path to directory containing tool files.

        Returns:
            List of discovered tools.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ToolDiscoveryError(f"Not a directory: {directory}")

        discovered = []
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, BaseTool)
                            and obj is not BaseTool
                            and hasattr(obj, "metadata")
                        ):
                            try:
                                instance = obj()
                                cls.register(instance)
                                discovered.append(instance)
                                logger.info(
                                    f"Discovered tool {instance.metadata.id} from {py_file}"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to instantiate {name}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load {py_file}: {e}")

        return discovered

    @classmethod
    def discover_from_entry_points(cls, group: str = "tinyllm.tools") -> List[BaseTool]:
        """Discover tools from package entry points.

        Args:
            group: Entry point group name.

        Returns:
            List of discovered tools.
        """
        discovered = []

        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            # Handle both Python 3.9 and 3.10+ API
            if hasattr(eps, "select"):
                tool_eps = eps.select(group=group)
            else:
                tool_eps = eps.get(group, [])

            for ep in tool_eps:
                try:
                    tool_class = ep.load()
                    if issubclass(tool_class, BaseTool):
                        instance = tool_class()
                        cls.register(instance)
                        discovered.append(instance)
                        logger.info(f"Discovered tool {instance.metadata.id} from entry point {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load entry point {ep.name}: {e}")
        except ImportError:
            logger.debug("importlib.metadata not available")

        return discovered

    @classmethod
    def add_discovery_callback(cls, callback: Callable[[], List[BaseTool]]) -> None:
        """Add a custom discovery callback.

        Args:
            callback: Function that returns a list of tools.
        """
        cls._discovery_callbacks.append(callback)

    @classmethod
    def run_discovery(cls) -> Dict[str, List[BaseTool]]:
        """Run all discovery mechanisms.

        Returns:
            Dict mapping discovery source to discovered tools.
        """
        results: Dict[str, List[BaseTool]] = {}

        # Run entry point discovery
        results["entry_points"] = cls.discover_from_entry_points()

        # Run custom callbacks
        for i, callback in enumerate(cls._discovery_callbacks):
            try:
                results[f"callback_{i}"] = callback()
            except Exception as e:
                logger.warning(f"Discovery callback {i} failed: {e}")
                results[f"callback_{i}"] = []

        return results

    @classmethod
    def get_tool_class(cls, tool_id: str) -> Optional[Type[BaseTool]]:
        """Get the class for a registered tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            Tool class or None.
        """
        return cls._tool_classes.get(tool_id)

    @classmethod
    def reload_tool(cls, tool_id: str) -> Optional[BaseTool]:
        """Reload a tool by recreating its instance.

        Args:
            tool_id: Tool identifier.

        Returns:
            New tool instance or None.
        """
        tool_class = cls._tool_classes.get(tool_id)
        if tool_class:
            try:
                instance = tool_class()
                cls._tools[tool_id] = instance
                logger.debug(f"Reloaded tool: {tool_id}")
                return instance
            except Exception as e:
                logger.error(f"Failed to reload tool {tool_id}: {e}")
        return None


def register_default_tools() -> None:
    """Register the default built-in tools.

    Call this at startup to register all built-in tools.
    """
    from tinyllm.tools.calculator import CalculatorTool
    from tinyllm.tools.code_executor import CodeExecutorTool
    from tinyllm.tools.web_search import WebSearchTool

    ToolRegistry.register(CalculatorTool())
    ToolRegistry.register(CodeExecutorTool())
    ToolRegistry.register(WebSearchTool())
