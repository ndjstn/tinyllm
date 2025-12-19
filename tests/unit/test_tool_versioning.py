"""Tests for tool versioning."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.versioning import (
    SemanticVersion,
    ToolVersionInfo,
    VersionConstraint,
    VersionedToolRegistry,
    VersionRequirement,
    check_version_compatibility,
    compare_versions,
)


class VersionInput(BaseModel):
    """Input for version test tools."""

    data: str = ""


class VersionOutput(BaseModel):
    """Output for version test tools."""

    success: bool = True
    error: str | None = None


def create_tool(version: str, tool_id: str = "version_tool") -> BaseTool:
    """Create a tool with a specific version."""

    class VersionTool(BaseTool[VersionInput, VersionOutput]):
        metadata = ToolMetadata(
            id=tool_id,
            name=f"Version Tool {version}",
            description=f"Tool version {version}",
            version=version,
            category="utility",
        )
        input_type = VersionInput
        output_type = VersionOutput

        async def execute(self, input: VersionInput) -> VersionOutput:
            return VersionOutput()

    return VersionTool()


class TestSemanticVersion:
    """Tests for SemanticVersion."""

    def test_parse_basic(self):
        """Test parsing basic version."""
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None
        assert v.build is None

    def test_parse_with_prerelease(self):
        """Test parsing version with prerelease."""
        v = SemanticVersion.parse("1.2.3-beta.1")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "beta.1"

    def test_parse_with_build(self):
        """Test parsing version with build metadata."""
        v = SemanticVersion.parse("1.2.3+build.123")
        assert v.build == "build.123"

    def test_parse_full(self):
        """Test parsing version with all parts."""
        v = SemanticVersion.parse("1.2.3-alpha+build")
        assert v.major == 1
        assert v.prerelease == "alpha"
        assert v.build == "build"

    def test_parse_invalid(self):
        """Test parsing invalid version."""
        with pytest.raises(ValueError):
            SemanticVersion.parse("invalid")

        with pytest.raises(ValueError):
            SemanticVersion.parse("1.2")

    def test_str(self):
        """Test string representation."""
        v = SemanticVersion(1, 2, 3)
        assert str(v) == "1.2.3"

        v = SemanticVersion(1, 2, 3, prerelease="beta")
        assert str(v) == "1.2.3-beta"

        v = SemanticVersion(1, 2, 3, prerelease="beta", build="123")
        assert str(v) == "1.2.3-beta+123"

    def test_comparison_basic(self):
        """Test basic version comparison."""
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v3 = SemanticVersion.parse("1.1.0")
        v4 = SemanticVersion.parse("1.0.1")

        assert v1 < v2
        assert v1 < v3
        assert v1 < v4
        assert v3 < v2
        assert v4 < v3

    def test_comparison_equal(self):
        """Test version equality."""
        v1 = SemanticVersion.parse("1.2.3")
        v2 = SemanticVersion.parse("1.2.3")

        assert v1 == v2
        assert v1 >= v2
        assert v1 <= v2

    def test_comparison_prerelease(self):
        """Test prerelease comparison."""
        release = SemanticVersion.parse("1.0.0")
        prerelease = SemanticVersion.parse("1.0.0-beta")

        assert prerelease < release  # Prerelease is less than release

    def test_is_compatible_with(self):
        """Test compatibility check."""
        v1 = SemanticVersion.parse("1.2.3")
        v2 = SemanticVersion.parse("1.3.0")
        v3 = SemanticVersion.parse("2.0.0")

        assert v1.is_compatible_with(v2)  # Same major
        assert not v1.is_compatible_with(v3)  # Different major


class TestVersionRequirement:
    """Tests for VersionRequirement."""

    def test_parse_exact(self):
        """Test parsing exact requirement."""
        req = VersionRequirement.parse("==1.2.3")
        assert req.constraint == VersionConstraint.EXACT
        assert req.version == SemanticVersion.parse("1.2.3")

    def test_parse_greater_equal(self):
        """Test parsing >= requirement."""
        req = VersionRequirement.parse(">=1.0.0")
        assert req.constraint == VersionConstraint.GREATER_EQUAL

    def test_parse_compatible(self):
        """Test parsing ~= requirement."""
        req = VersionRequirement.parse("~=1.2.0")
        assert req.constraint == VersionConstraint.COMPATIBLE

    def test_parse_any(self):
        """Test parsing * requirement."""
        req = VersionRequirement.parse("*")
        assert req.constraint == VersionConstraint.ANY

    def test_parse_no_operator(self):
        """Test parsing version without operator (exact)."""
        req = VersionRequirement.parse("1.2.3")
        assert req.constraint == VersionConstraint.EXACT

    def test_matches_exact(self):
        """Test exact matching."""
        req = VersionRequirement.parse("==1.2.3")

        assert req.matches(SemanticVersion.parse("1.2.3"))
        assert not req.matches(SemanticVersion.parse("1.2.4"))

    def test_matches_greater_equal(self):
        """Test >= matching."""
        req = VersionRequirement.parse(">=1.0.0")

        assert req.matches(SemanticVersion.parse("1.0.0"))
        assert req.matches(SemanticVersion.parse("2.0.0"))
        assert not req.matches(SemanticVersion.parse("0.9.0"))

    def test_matches_less_than(self):
        """Test < matching."""
        req = VersionRequirement.parse("<2.0.0")

        assert req.matches(SemanticVersion.parse("1.9.9"))
        assert not req.matches(SemanticVersion.parse("2.0.0"))

    def test_matches_compatible(self):
        """Test ~= matching."""
        req = VersionRequirement.parse("~=1.2.0")

        assert req.matches(SemanticVersion.parse("1.2.0"))
        assert req.matches(SemanticVersion.parse("1.3.0"))
        assert not req.matches(SemanticVersion.parse("1.1.0"))  # Less than
        assert not req.matches(SemanticVersion.parse("2.0.0"))  # Different major

    def test_matches_any(self):
        """Test * matching."""
        req = VersionRequirement.parse("*")

        assert req.matches(SemanticVersion.parse("0.0.1"))
        assert req.matches(SemanticVersion.parse("99.99.99"))


class TestVersionedToolRegistry:
    """Tests for VersionedToolRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return VersionedToolRegistry()

    def test_register_tool(self, registry):
        """Test registering a tool."""
        tool = create_tool("1.0.0")
        registry.register(tool)

        assert "1.0.0" in registry.list_versions("version_tool")

    def test_register_multiple_versions(self, registry):
        """Test registering multiple versions."""
        registry.register(create_tool("1.0.0"))
        registry.register(create_tool("1.1.0"))
        registry.register(create_tool("2.0.0"))

        versions = registry.list_versions("version_tool")
        assert len(versions) == 3

    def test_get_default_version(self, registry):
        """Test getting default version."""
        registry.register(create_tool("1.0.0"))
        registry.register(create_tool("2.0.0"))

        tool = registry.get("version_tool")
        assert tool is not None
        assert tool.metadata.version == "2.0.0"  # Last registered is default

    def test_get_specific_version(self, registry):
        """Test getting specific version."""
        registry.register(create_tool("1.0.0"))
        registry.register(create_tool("2.0.0"))

        tool = registry.get("version_tool", version="1.0.0")
        assert tool is not None
        assert tool.metadata.version == "1.0.0"

    def test_get_with_requirement(self, registry):
        """Test getting with version requirement."""
        registry.register(create_tool("1.0.0"))
        registry.register(create_tool("1.5.0"))
        registry.register(create_tool("2.0.0"))

        # Test single requirement - gets highest matching
        tool = registry.get("version_tool", requirement="<2.0.0")
        assert tool is not None
        assert tool.metadata.version == "1.5.0"

        tool = registry.get("version_tool", requirement=">=1.5.0")
        assert tool is not None
        assert tool.metadata.version == "2.0.0"

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent tool."""
        assert registry.get("nonexistent") is None

    def test_set_default_version(self, registry):
        """Test setting default version."""
        registry.register(create_tool("1.0.0"))
        registry.register(create_tool("2.0.0"))

        registry.set_default_version("version_tool", "1.0.0")
        tool = registry.get("version_tool")
        assert tool.metadata.version == "1.0.0"

    def test_get_version_info(self, registry):
        """Test getting version info."""
        info = ToolVersionInfo(
            version="1.0.0",
            changelog="Initial release",
            release_date="2024-01-01",
        )
        registry.register(create_tool("1.0.0"), version_info=info)

        retrieved = registry.get_version_info("version_tool", "1.0.0")
        assert retrieved is not None
        assert retrieved.changelog == "Initial release"

    def test_deprecate_version(self, registry):
        """Test deprecating a version."""
        registry.register(create_tool("1.0.0"))

        assert not registry.is_deprecated("version_tool", "1.0.0")

        registry.deprecate_version("version_tool", "1.0.0", "Use 2.0.0 instead")

        assert registry.is_deprecated("version_tool", "1.0.0")
        info = registry.get_version_info("version_tool", "1.0.0")
        assert info.deprecation_message == "Use 2.0.0 instead"

    def test_get_latest_version(self, registry):
        """Test getting latest version."""
        registry.register(create_tool("1.0.0"))
        registry.register(create_tool("1.5.0"))
        registry.register(create_tool("2.0.0"))

        latest = registry.get_latest_version("version_tool")
        assert latest == "2.0.0"

    def test_get_latest_version_with_prerelease(self, registry):
        """Test that prerelease is less than release of same version."""
        registry.register(create_tool("2.0.0-beta"))
        registry.register(create_tool("2.0.0"))

        latest = registry.get_latest_version("version_tool")
        assert latest == "2.0.0"  # Release > prerelease

    def test_clear(self, registry):
        """Test clearing registry."""
        registry.register(create_tool("1.0.0"))
        registry.clear()

        assert registry.list_versions("version_tool") == []


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_version_compatibility_compatible(self):
        """Test compatibility check for compatible versions."""
        assert check_version_compatibility("1.2.3", "1.5.0")
        assert check_version_compatibility("2.0.0", "2.9.9")

    def test_check_version_compatibility_incompatible(self):
        """Test compatibility check for incompatible versions."""
        assert not check_version_compatibility("1.0.0", "2.0.0")

    def test_compare_versions(self):
        """Test version comparison."""
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "1.0.0") == 0
