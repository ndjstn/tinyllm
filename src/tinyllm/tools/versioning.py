"""Tool versioning for TinyLLM.

This module provides version management for tools including
semantic versioning support, compatibility checking, and
version-based tool selection.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class VersionConstraint(str, Enum):
    """Version constraint operators."""

    EXACT = "=="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    COMPATIBLE = "~="  # Compatible release (major.minor.*)
    ANY = "*"


@dataclass
class SemanticVersion:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_string: str) -> "SemanticVersion":
        """Parse a version string into a SemanticVersion.

        Args:
            version_string: Version string like "1.2.3", "1.2.3-beta", "1.2.3+build123"

        Returns:
            SemanticVersion instance.

        Raises:
            ValueError: If version string is invalid.
        """
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_string.strip())

        if not match:
            raise ValueError(f"Invalid version string: {version_string}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch):
            return True
        if (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch):
            return False
        # Same version, check prerelease
        if self.prerelease and not other.prerelease:
            return True  # Prerelease is less than release
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __le__(self, other: "SemanticVersion") -> bool:
        return self < other or self == other

    def __gt__(self, other: "SemanticVersion") -> bool:
        return not self <= other

    def __ge__(self, other: "SemanticVersion") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is compatible with another.

        Compatible means same major version (for major > 0) or
        same major.minor (for major == 0).
        """
        if self.major == 0 or other.major == 0:
            return self.major == other.major and self.minor == other.minor
        return self.major == other.major


@dataclass
class VersionRequirement:
    """A version requirement with constraint."""

    constraint: VersionConstraint
    version: Optional[SemanticVersion] = None

    @classmethod
    def parse(cls, requirement: str) -> "VersionRequirement":
        """Parse a requirement string.

        Args:
            requirement: Requirement like ">=1.0.0", "==2.3.4", "~=1.2.0", "*"

        Returns:
            VersionRequirement instance.
        """
        requirement = requirement.strip()

        if requirement == "*":
            return cls(constraint=VersionConstraint.ANY)

        for op in [">=", "<=", "~=", "==", ">", "<"]:
            if requirement.startswith(op):
                version_str = requirement[len(op):].strip()
                return cls(
                    constraint=VersionConstraint(op),
                    version=SemanticVersion.parse(version_str),
                )

        # No operator, assume exact match
        return cls(
            constraint=VersionConstraint.EXACT,
            version=SemanticVersion.parse(requirement),
        )

    def matches(self, version: SemanticVersion) -> bool:
        """Check if a version matches this requirement.

        Args:
            version: Version to check.

        Returns:
            True if version matches.
        """
        if self.constraint == VersionConstraint.ANY:
            return True

        if self.version is None:
            return True

        if self.constraint == VersionConstraint.EXACT:
            return version == self.version
        elif self.constraint == VersionConstraint.GREATER_THAN:
            return version > self.version
        elif self.constraint == VersionConstraint.GREATER_EQUAL:
            return version >= self.version
        elif self.constraint == VersionConstraint.LESS_THAN:
            return version < self.version
        elif self.constraint == VersionConstraint.LESS_EQUAL:
            return version <= self.version
        elif self.constraint == VersionConstraint.COMPATIBLE:
            return version.is_compatible_with(self.version) and version >= self.version

        return False

    def __str__(self) -> str:
        if self.constraint == VersionConstraint.ANY:
            return "*"
        return f"{self.constraint.value}{self.version}"


class ToolVersionInfo(BaseModel):
    """Version information for a tool."""

    version: str
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    min_compatible_version: Optional[str] = None
    changelog: Optional[str] = None
    release_date: Optional[str] = None


class VersionedToolRegistry:
    """Registry that supports multiple versions of tools."""

    def __init__(self):
        """Initialize the registry."""
        # tool_id -> version -> tool instance
        self._tools: Dict[str, Dict[str, Any]] = {}
        # tool_id -> default version
        self._defaults: Dict[str, str] = {}
        # tool_id -> version info
        self._version_info: Dict[str, Dict[str, ToolVersionInfo]] = {}

    def register(
        self,
        tool: Any,
        version_info: Optional[ToolVersionInfo] = None,
        set_default: bool = True,
    ) -> None:
        """Register a tool version.

        Args:
            tool: Tool instance to register.
            version_info: Optional version metadata.
            set_default: Whether to set as default version.
        """
        from tinyllm.tools.base import BaseTool

        if not isinstance(tool, BaseTool):
            raise ValueError(f"Expected BaseTool, got {type(tool).__name__}")

        tool_id = tool.metadata.id
        version = tool.metadata.version

        if tool_id not in self._tools:
            self._tools[tool_id] = {}
            self._version_info[tool_id] = {}

        self._tools[tool_id][version] = tool

        if version_info:
            self._version_info[tool_id][version] = version_info
        else:
            self._version_info[tool_id][version] = ToolVersionInfo(version=version)

        if set_default or tool_id not in self._defaults:
            self._defaults[tool_id] = version

    def get(
        self,
        tool_id: str,
        version: Optional[str] = None,
        requirement: Optional[str] = None,
    ) -> Optional[Any]:
        """Get a tool by ID and optional version.

        Args:
            tool_id: Tool identifier.
            version: Exact version (mutually exclusive with requirement).
            requirement: Version requirement string (mutually exclusive with version).

        Returns:
            Tool instance or None.
        """
        if tool_id not in self._tools:
            return None

        if version:
            return self._tools[tool_id].get(version)

        if requirement:
            return self._find_matching_version(tool_id, requirement)

        # Return default version
        default = self._defaults.get(tool_id)
        if default:
            return self._tools[tool_id].get(default)

        return None

    def _find_matching_version(self, tool_id: str, requirement: str) -> Optional[Any]:
        """Find the best matching version for a requirement.

        Args:
            tool_id: Tool identifier.
            requirement: Version requirement string.

        Returns:
            Best matching tool or None.
        """
        req = VersionRequirement.parse(requirement)
        versions = self._tools.get(tool_id, {})

        matching = []
        for version_str, tool in versions.items():
            try:
                version = SemanticVersion.parse(version_str)
                if req.matches(version):
                    matching.append((version, tool))
            except ValueError:
                continue

        if not matching:
            return None

        # Return the highest matching version
        matching.sort(key=lambda x: x[0], reverse=True)
        return matching[0][1]

    def list_versions(self, tool_id: str) -> List[str]:
        """List all versions of a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            List of version strings.
        """
        return list(self._tools.get(tool_id, {}).keys())

    def get_version_info(
        self, tool_id: str, version: Optional[str] = None
    ) -> Optional[ToolVersionInfo]:
        """Get version info for a tool.

        Args:
            tool_id: Tool identifier.
            version: Specific version or None for default.

        Returns:
            ToolVersionInfo or None.
        """
        if tool_id not in self._version_info:
            return None

        if version:
            return self._version_info[tool_id].get(version)

        default = self._defaults.get(tool_id)
        if default:
            return self._version_info[tool_id].get(default)

        return None

    def set_default_version(self, tool_id: str, version: str) -> bool:
        """Set the default version for a tool.

        Args:
            tool_id: Tool identifier.
            version: Version to set as default.

        Returns:
            True if successful.
        """
        if tool_id in self._tools and version in self._tools[tool_id]:
            self._defaults[tool_id] = version
            return True
        return False

    def deprecate_version(
        self,
        tool_id: str,
        version: str,
        message: Optional[str] = None,
    ) -> bool:
        """Mark a version as deprecated.

        Args:
            tool_id: Tool identifier.
            version: Version to deprecate.
            message: Optional deprecation message.

        Returns:
            True if successful.
        """
        if tool_id in self._version_info and version in self._version_info[tool_id]:
            info = self._version_info[tool_id][version]
            self._version_info[tool_id][version] = ToolVersionInfo(
                version=info.version,
                deprecated=True,
                deprecation_message=message or "This version is deprecated",
                min_compatible_version=info.min_compatible_version,
                changelog=info.changelog,
                release_date=info.release_date,
            )
            return True
        return False

    def is_deprecated(self, tool_id: str, version: Optional[str] = None) -> bool:
        """Check if a tool version is deprecated.

        Args:
            tool_id: Tool identifier.
            version: Specific version or None for default.

        Returns:
            True if deprecated.
        """
        info = self.get_version_info(tool_id, version)
        return info.deprecated if info else False

    def get_latest_version(self, tool_id: str) -> Optional[str]:
        """Get the latest version of a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            Latest version string or None.
        """
        versions = self.list_versions(tool_id)
        if not versions:
            return None

        parsed = []
        for v in versions:
            try:
                parsed.append((SemanticVersion.parse(v), v))
            except ValueError:
                continue

        if not parsed:
            return versions[0]

        parsed.sort(key=lambda x: x[0], reverse=True)
        return parsed[0][1]

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._defaults.clear()
        self._version_info.clear()


def check_version_compatibility(
    version1: str,
    version2: str,
) -> bool:
    """Check if two versions are compatible.

    Args:
        version1: First version string.
        version2: Second version string.

    Returns:
        True if versions are compatible.
    """
    v1 = SemanticVersion.parse(version1)
    v2 = SemanticVersion.parse(version2)
    return v1.is_compatible_with(v2)


def compare_versions(version1: str, version2: str) -> int:
    """Compare two version strings.

    Args:
        version1: First version.
        version2: Second version.

    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2.
    """
    v1 = SemanticVersion.parse(version1)
    v2 = SemanticVersion.parse(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    return 0
