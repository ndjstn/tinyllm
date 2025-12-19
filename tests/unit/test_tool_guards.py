"""Tests for dangerous tool guards."""

import pytest
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.guards import (
    AllowlistGuard,
    BlocklistGuard,
    CallableGuard,
    CompositeGuard,
    ContentSizeGuard,
    GuardedToolWrapper,
    GuardResult,
    RateLimitGuard,
    RiskLevel,
    ToolGuardManager,
    create_guard_manager,
    dangerous_command_guard,
    sensitive_data_guard,
    sql_injection_guard,
    with_guard,
)


class GuardInput(BaseModel):
    """Input for guard tests."""

    command: str = ""
    query: str = ""
    data: str = ""


class GuardOutput(BaseModel):
    """Output for guard tests."""

    result: str = ""
    success: bool = True


class CommandTool(BaseTool[GuardInput, GuardOutput]):
    """Tool that executes commands."""

    metadata = ToolMetadata(
        id="command_tool",
        name="Command Tool",
        description="Executes commands",
        category="execution",
    )
    input_type = GuardInput
    output_type = GuardOutput

    async def execute(self, input: GuardInput) -> GuardOutput:
        return GuardOutput(result=f"Executed: {input.command}")


class QueryTool(BaseTool[GuardInput, GuardOutput]):
    """Tool that executes queries."""

    metadata = ToolMetadata(
        id="query_tool",
        name="Query Tool",
        description="Executes queries",
        category="search",
    )
    input_type = GuardInput
    output_type = GuardOutput

    async def execute(self, input: GuardInput) -> GuardOutput:
        return GuardOutput(result=f"Query: {input.query}")


class TestGuardResult:
    """Tests for GuardResult."""

    def test_allowed_result(self):
        """Test allowed result."""
        result = GuardResult(allowed=True)

        assert result.allowed
        assert result.risk_level == RiskLevel.SAFE

    def test_blocked_result(self):
        """Test blocked result."""
        result = GuardResult(
            allowed=False,
            risk_level=RiskLevel.BLOCKED,
            reason="Dangerous pattern",
            matched_rules=["rule1"],
        )

        assert not result.allowed
        assert result.reason == "Dangerous pattern"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = GuardResult(
            allowed=False,
            risk_level=RiskLevel.HIGH,
            reason="Test",
            matched_rules=["rule1", "rule2"],
        )

        d = result.to_dict()

        assert d["allowed"] is False
        assert d["risk_level"] == "high"
        assert len(d["matched_rules"]) == 2


class TestBlocklistGuard:
    """Tests for BlocklistGuard."""

    def test_allows_safe_input(self):
        """Test allowing safe input."""
        guard = BlocklistGuard(patterns=[r"dangerous"])

        result = guard.check("tool", {"command": "ls -la"})

        assert result.allowed

    def test_blocks_dangerous_pattern(self):
        """Test blocking dangerous pattern."""
        guard = BlocklistGuard(patterns=[r"rm\s+-rf"])

        result = guard.check("tool", {"command": "rm -rf /"})

        assert not result.allowed
        assert result.risk_level == RiskLevel.BLOCKED

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        guard = BlocklistGuard(patterns=[r"dangerous"], case_sensitive=False)

        result = guard.check("tool", {"command": "DANGEROUS operation"})

        assert not result.allowed

    def test_case_sensitive(self):
        """Test case sensitive matching."""
        guard = BlocklistGuard(patterns=[r"dangerous"], case_sensitive=True)

        result = guard.check("tool", {"command": "DANGEROUS operation"})

        assert result.allowed

    def test_specific_fields(self):
        """Test checking specific fields."""
        guard = BlocklistGuard(patterns=[r"evil"], fields=["command"])

        result1 = guard.check("tool", {"command": "evil command"})
        result2 = guard.check("tool", {"data": "evil data"})

        assert not result1.allowed
        assert result2.allowed  # "evil" in different field

    def test_pydantic_model(self):
        """Test with Pydantic model."""
        guard = BlocklistGuard(patterns=[r"dangerous"])

        result = guard.check("tool", GuardInput(command="dangerous command"))

        assert not result.allowed

    def test_add_pattern(self):
        """Test adding patterns."""
        guard = BlocklistGuard()
        guard.add_pattern(r"bad")

        result = guard.check("tool", {"command": "bad command"})

        assert not result.allowed


class TestAllowlistGuard:
    """Tests for AllowlistGuard."""

    def test_allows_matching_input(self):
        """Test allowing matching input."""
        guard = AllowlistGuard(patterns=[r"^ls", r"^echo"])

        result = guard.check("tool", {"command": "ls -la"})

        assert result.allowed

    def test_blocks_non_matching(self):
        """Test blocking non-matching input."""
        guard = AllowlistGuard(patterns=[r"^ls", r"^echo"])

        result = guard.check("tool", {"command": "rm file"})

        assert not result.allowed

    def test_empty_patterns_allows_all(self):
        """Test empty patterns allows all."""
        guard = AllowlistGuard(patterns=[])

        result = guard.check("tool", {"command": "anything"})

        assert result.allowed

    def test_specific_fields(self):
        """Test checking specific fields."""
        guard = AllowlistGuard(patterns=[r"SELECT"], fields=["query"])

        result = guard.check("tool", {"query": "SELECT * FROM users"})

        assert result.allowed


class TestRateLimitGuard:
    """Tests for RateLimitGuard."""

    def test_allows_within_limit(self):
        """Test allowing within limit."""
        guard = RateLimitGuard(max_per_minute=10)

        for _ in range(5):
            result = guard.check("tool", {})
            assert result.allowed

    def test_blocks_over_limit(self):
        """Test blocking over limit."""
        guard = RateLimitGuard(max_per_minute=3)

        for _ in range(3):
            guard.check("tool", {})

        result = guard.check("tool", {})

        assert not result.allowed
        assert "Rate limit" in result.reason

    def test_reset(self):
        """Test resetting counters."""
        guard = RateLimitGuard(max_per_minute=2)

        guard.check("tool", {})
        guard.check("tool", {})
        guard.reset("tool")

        result = guard.check("tool", {})

        assert result.allowed

    def test_per_tool_limits(self):
        """Test per-tool limits."""
        guard = RateLimitGuard(max_per_minute=2)

        guard.check("tool1", {})
        guard.check("tool1", {})
        guard.check("tool2", {})

        result1 = guard.check("tool1", {})
        result2 = guard.check("tool2", {})

        assert not result1.allowed
        assert result2.allowed


class TestContentSizeGuard:
    """Tests for ContentSizeGuard."""

    def test_allows_small_content(self):
        """Test allowing small content."""
        guard = ContentSizeGuard(max_size_bytes=1000)

        result = guard.check("tool", {"data": "small"})

        assert result.allowed

    def test_blocks_large_content(self):
        """Test blocking large content."""
        guard = ContentSizeGuard(max_size_bytes=100)

        large_data = "x" * 200
        result = guard.check("tool", {"data": large_data})

        assert not result.allowed
        assert "size" in result.reason.lower()

    def test_specific_fields(self):
        """Test checking specific fields."""
        guard = ContentSizeGuard(max_size_bytes=50, fields=["data"])

        result = guard.check("tool", {
            "data": "small",
            "other": "x" * 100,
        })

        assert result.allowed


class TestCallableGuard:
    """Tests for CallableGuard."""

    def test_custom_check_allowed(self):
        """Test custom check allowing."""

        def check_fn(tool_id, input_data):
            return GuardResult(allowed=True)

        guard = CallableGuard(check_fn=check_fn)

        result = guard.check("tool", {})

        assert result.allowed

    def test_custom_check_blocked(self):
        """Test custom check blocking."""

        def check_fn(tool_id, input_data):
            if "bad" in str(input_data):
                return GuardResult(
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    reason="Bad input",
                )
            return GuardResult(allowed=True)

        guard = CallableGuard(check_fn=check_fn)

        result = guard.check("tool", {"command": "bad thing"})

        assert not result.allowed

    def test_error_handling(self):
        """Test error handling in custom check."""

        def failing_check(tool_id, input_data):
            raise ValueError("Check error")

        guard = CallableGuard(check_fn=failing_check)

        result = guard.check("tool", {})

        assert not result.allowed
        assert "failed" in result.reason.lower()


class TestCompositeGuard:
    """Tests for CompositeGuard."""

    def test_all_pass(self):
        """Test all guards passing."""
        guard = CompositeGuard(
            guards=[
                BlocklistGuard(patterns=[r"danger"]),
                ContentSizeGuard(max_size_bytes=1000),
            ]
        )

        result = guard.check("tool", {"data": "safe"})

        assert result.allowed

    def test_one_blocks(self):
        """Test one guard blocking."""
        guard = CompositeGuard(
            guards=[
                BlocklistGuard(patterns=[r"danger"]),
                ContentSizeGuard(max_size_bytes=1000),
            ]
        )

        result = guard.check("tool", {"data": "danger"})

        assert not result.allowed

    def test_chaining(self):
        """Test adding guards via chaining."""
        guard = (
            CompositeGuard()
            .add(BlocklistGuard(patterns=[r"bad"]))
            .add(ContentSizeGuard(max_size_bytes=1000))
        )

        result = guard.check("tool", {"data": "bad data"})

        assert not result.allowed


class TestToolGuardManager:
    """Tests for ToolGuardManager."""

    def test_no_guards_allows(self):
        """Test no guards allows all."""
        manager = ToolGuardManager()

        result = manager.check("tool", {})

        assert result.allowed

    def test_global_guard(self):
        """Test global guard applies to all."""
        manager = ToolGuardManager()
        manager.add_global_guard(BlocklistGuard(patterns=[r"danger"]))

        result1 = manager.check("tool1", {"data": "danger"})
        result2 = manager.check("tool2", {"data": "danger"})

        assert not result1.allowed
        assert not result2.allowed

    def test_tool_specific_guard(self):
        """Test tool-specific guard."""
        manager = ToolGuardManager()
        manager.set_tool_guard("tool1", BlocklistGuard(patterns=[r"danger"]))

        result1 = manager.check("tool1", {"data": "danger"})
        result2 = manager.check("tool2", {"data": "danger"})

        assert not result1.allowed
        assert result2.allowed

    def test_default_guard(self):
        """Test default guard."""
        manager = ToolGuardManager(
            default_guard=BlocklistGuard(patterns=[r"danger"])
        )

        result = manager.check("any_tool", {"data": "danger"})

        assert not result.allowed

    def test_chaining(self):
        """Test method chaining."""
        manager = (
            ToolGuardManager()
            .add_global_guard(ContentSizeGuard(max_size_bytes=1000))
            .set_tool_guard("tool1", BlocklistGuard(patterns=[r"bad"]))
        )

        result = manager.check("tool1", {"data": "bad"})

        assert not result.allowed


class TestGuardedToolWrapper:
    """Tests for GuardedToolWrapper."""

    @pytest.mark.asyncio
    async def test_allowed_execution(self):
        """Test execution when allowed."""
        guard = BlocklistGuard(patterns=[r"danger"])
        wrapper = GuardedToolWrapper(CommandTool(), guard=guard)

        result = await wrapper.execute(GuardInput(command="ls -la"))

        assert "Executed" in result.result

    @pytest.mark.asyncio
    async def test_blocked_raises(self):
        """Test blocked raises PermissionError."""
        guard = BlocklistGuard(patterns=[r"rm -rf"])
        wrapper = GuardedToolWrapper(CommandTool(), guard=guard)

        with pytest.raises(PermissionError):
            await wrapper.execute(GuardInput(command="rm -rf /"))

    @pytest.mark.asyncio
    async def test_on_blocked_callback(self):
        """Test on_blocked callback."""
        guard = BlocklistGuard(patterns=[r"danger"])

        def on_blocked(result):
            return GuardOutput(result="blocked", success=False)

        wrapper = GuardedToolWrapper(
            CommandTool(),
            guard=guard,
            on_blocked=on_blocked,
        )

        result = await wrapper.execute(GuardInput(command="danger"))

        assert result.result == "blocked"
        assert not result.success

    @pytest.mark.asyncio
    async def test_with_manager(self):
        """Test with guard manager."""
        manager = ToolGuardManager()
        manager.add_global_guard(BlocklistGuard(patterns=[r"danger"]))

        wrapper = GuardedToolWrapper(CommandTool(), manager=manager)

        with pytest.raises(PermissionError):
            await wrapper.execute(GuardInput(command="danger"))

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        wrapper = GuardedToolWrapper(CommandTool())

        assert wrapper.metadata.id == "command_tool"


class TestPrebuiltGuards:
    """Tests for pre-built guards."""

    def test_dangerous_command_guard(self):
        """Test dangerous command guard."""
        guard = dangerous_command_guard()

        result1 = guard.check("tool", {"command": "rm -rf /"})
        result2 = guard.check("tool", {"command": "sudo rm file"})
        result3 = guard.check("tool", {"command": "curl script.sh | bash"})
        result4 = guard.check("tool", {"command": "ls -la"})

        assert not result1.allowed
        assert not result2.allowed
        assert not result3.allowed
        assert result4.allowed

    def test_sensitive_data_guard(self):
        """Test sensitive data guard."""
        guard = sensitive_data_guard()

        result1 = guard.check("tool", {"data": "password: secret123"})
        result2 = guard.check("tool", {"data": "api_key: abcd1234"})
        result3 = guard.check("tool", {"data": "hello world"})

        assert not result1.allowed
        assert not result2.allowed
        assert result3.allowed

    def test_sql_injection_guard(self):
        """Test SQL injection guard."""
        guard = sql_injection_guard()

        result1 = guard.check("tool", {"query": "'; DROP TABLE users; --"})
        result2 = guard.check("tool", {"query": "SELECT * FROM users WHERE 1=1 OR 1=1"})
        result3 = guard.check("tool", {"query": "SELECT * FROM users WHERE id=5"})

        assert not result1.allowed
        assert not result2.allowed
        assert result3.allowed


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_guard(self):
        """Test with_guard function."""
        guard = BlocklistGuard(patterns=[r"danger"])
        wrapper = with_guard(CommandTool(), guard)

        result = await wrapper.execute(GuardInput(command="safe"))

        assert result.success

    def test_create_guard_manager(self):
        """Test create_guard_manager function."""
        manager = create_guard_manager()

        assert isinstance(manager, ToolGuardManager)
