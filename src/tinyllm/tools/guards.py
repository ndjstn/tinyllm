"""Dangerous tool guards for TinyLLM.

This module provides safety guards for dangerous tool operations,
including pattern detection, blocklists, and safety checks.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Pattern

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for operations."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    BLOCKED = "blocked"


@dataclass
class GuardResult:
    """Result of a guard check."""

    allowed: bool
    risk_level: RiskLevel = RiskLevel.SAFE
    reason: Optional[str] = None
    matched_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.value,
            "reason": self.reason,
            "matched_rules": self.matched_rules,
        }


class Guard(ABC):
    """Abstract base class for guards."""

    @abstractmethod
    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check if operation should be allowed.

        Args:
            tool_id: Tool identifier.
            input_data: Tool input data.

        Returns:
            GuardResult indicating if allowed.
        """
        pass


class BlocklistGuard(Guard):
    """Guard that blocks specific patterns."""

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """Initialize blocklist guard.

        Args:
            patterns: Regex patterns to block.
            fields: Input fields to check (None checks all).
            case_sensitive: Whether matching is case-sensitive.
        """
        self.fields = fields
        flags = 0 if case_sensitive else re.IGNORECASE
        self.patterns: List[tuple[str, Pattern]] = [
            (p, re.compile(p, flags)) for p in (patterns or [])
        ]

    def add_pattern(self, pattern: str) -> "BlocklistGuard":
        """Add a pattern to the blocklist.

        Args:
            pattern: Regex pattern.

        Returns:
            Self for chaining.
        """
        self.patterns.append((pattern, re.compile(pattern, re.IGNORECASE)))
        return self

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check input against blocklist."""
        values_to_check = self._extract_values(input_data)

        for value in values_to_check:
            for pattern_str, pattern in self.patterns:
                if pattern.search(str(value)):
                    return GuardResult(
                        allowed=False,
                        risk_level=RiskLevel.BLOCKED,
                        reason=f"Blocked pattern detected: {pattern_str}",
                        matched_rules=[pattern_str],
                    )

        return GuardResult(allowed=True, risk_level=RiskLevel.SAFE)

    def _extract_values(self, input_data: Any) -> List[Any]:
        """Extract values to check from input."""
        if self.fields is None:
            # Check all values
            if hasattr(input_data, "model_dump"):
                return list(input_data.model_dump().values())
            elif isinstance(input_data, dict):
                return list(input_data.values())
            else:
                return [input_data]

        # Check specific fields
        values = []
        if hasattr(input_data, "model_dump"):
            data = input_data.model_dump()
        elif isinstance(input_data, dict):
            data = input_data
        else:
            return [input_data]

        for field in self.fields:
            if field in data:
                values.append(data[field])

        return values


class AllowlistGuard(Guard):
    """Guard that only allows specific patterns."""

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """Initialize allowlist guard.

        Args:
            patterns: Regex patterns to allow.
            fields: Input fields to check.
            case_sensitive: Whether matching is case-sensitive.
        """
        self.fields = fields
        flags = 0 if case_sensitive else re.IGNORECASE
        self.patterns: List[tuple[str, Pattern]] = [
            (p, re.compile(p, flags)) for p in (patterns or [])
        ]

    def add_pattern(self, pattern: str) -> "AllowlistGuard":
        """Add a pattern to the allowlist.

        Args:
            pattern: Regex pattern.

        Returns:
            Self for chaining.
        """
        self.patterns.append((pattern, re.compile(pattern, re.IGNORECASE)))
        return self

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check input against allowlist."""
        if not self.patterns:
            return GuardResult(allowed=True, risk_level=RiskLevel.SAFE)

        values_to_check = self._extract_values(input_data)

        for value in values_to_check:
            str_value = str(value)
            allowed = any(p.search(str_value) for _, p in self.patterns)

            if not allowed:
                return GuardResult(
                    allowed=False,
                    risk_level=RiskLevel.BLOCKED,
                    reason=f"Value not in allowlist: {str_value[:50]}...",
                )

        return GuardResult(allowed=True, risk_level=RiskLevel.SAFE)

    def _extract_values(self, input_data: Any) -> List[Any]:
        """Extract values to check from input."""
        if self.fields is None:
            if hasattr(input_data, "model_dump"):
                return list(input_data.model_dump().values())
            elif isinstance(input_data, dict):
                return list(input_data.values())
            else:
                return [input_data]

        values = []
        if hasattr(input_data, "model_dump"):
            data = input_data.model_dump()
        elif isinstance(input_data, dict):
            data = input_data
        else:
            return [input_data]

        for field in self.fields:
            if field in data:
                values.append(data[field])

        return values


class RateLimitGuard(Guard):
    """Guard that rate limits operations."""

    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_hour: int = 1000,
    ):
        """Initialize rate limit guard.

        Args:
            max_per_minute: Max operations per minute.
            max_per_hour: Max operations per hour.
        """
        import time

        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._minute_counts: Dict[str, List[float]] = {}
        self._hour_counts: Dict[str, List[float]] = {}

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check rate limits."""
        import time

        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Clean old entries and count
        if tool_id not in self._minute_counts:
            self._minute_counts[tool_id] = []
        if tool_id not in self._hour_counts:
            self._hour_counts[tool_id] = []

        # Filter old entries
        self._minute_counts[tool_id] = [
            t for t in self._minute_counts[tool_id] if t > minute_ago
        ]
        self._hour_counts[tool_id] = [
            t for t in self._hour_counts[tool_id] if t > hour_ago
        ]

        # Check limits
        if len(self._minute_counts[tool_id]) >= self.max_per_minute:
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"Rate limit exceeded: {self.max_per_minute}/minute",
                matched_rules=["rate_limit_minute"],
            )

        if len(self._hour_counts[tool_id]) >= self.max_per_hour:
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"Rate limit exceeded: {self.max_per_hour}/hour",
                matched_rules=["rate_limit_hour"],
            )

        # Record this operation
        self._minute_counts[tool_id].append(now)
        self._hour_counts[tool_id].append(now)

        return GuardResult(allowed=True, risk_level=RiskLevel.SAFE)

    def reset(self, tool_id: Optional[str] = None) -> None:
        """Reset rate limit counters.

        Args:
            tool_id: Tool to reset (None resets all).
        """
        if tool_id:
            self._minute_counts.pop(tool_id, None)
            self._hour_counts.pop(tool_id, None)
        else:
            self._minute_counts.clear()
            self._hour_counts.clear()


class ContentSizeGuard(Guard):
    """Guard that limits content size."""

    def __init__(
        self,
        max_size_bytes: int = 1_000_000,  # 1MB
        fields: Optional[List[str]] = None,
    ):
        """Initialize content size guard.

        Args:
            max_size_bytes: Maximum content size in bytes.
            fields: Fields to check (None checks all).
        """
        self.max_size_bytes = max_size_bytes
        self.fields = fields

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check content size."""
        import json

        try:
            if self.fields:
                data = {}
                source = (
                    input_data.model_dump()
                    if hasattr(input_data, "model_dump")
                    else input_data
                )
                for field in self.fields:
                    if isinstance(source, dict) and field in source:
                        data[field] = source[field]
            else:
                if hasattr(input_data, "model_dump"):
                    data = input_data.model_dump()
                elif isinstance(input_data, dict):
                    data = input_data
                else:
                    data = input_data

            size = len(json.dumps(data, default=str).encode("utf-8"))

            if size > self.max_size_bytes:
                return GuardResult(
                    allowed=False,
                    risk_level=RiskLevel.HIGH,
                    reason=f"Content size {size} exceeds limit {self.max_size_bytes}",
                    matched_rules=["content_size"],
                    metadata={"size": size, "limit": self.max_size_bytes},
                )

            return GuardResult(allowed=True, risk_level=RiskLevel.SAFE)

        except Exception as e:
            logger.warning(f"Content size check error: {e}")
            return GuardResult(allowed=True, risk_level=RiskLevel.LOW)


class CallableGuard(Guard):
    """Guard that uses a callable for checking."""

    def __init__(
        self,
        check_fn: Callable[[str, Any], GuardResult],
        name: str = "custom",
    ):
        """Initialize callable guard.

        Args:
            check_fn: Function that performs the check.
            name: Name for this guard.
        """
        self.check_fn = check_fn
        self.name = name

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Execute the custom check."""
        try:
            return self.check_fn(tool_id, input_data)
        except Exception as e:
            logger.error(f"Custom guard {self.name} error: {e}")
            return GuardResult(
                allowed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"Guard check failed: {e}",
            )


class CompositeGuard(Guard):
    """Combines multiple guards."""

    def __init__(
        self,
        guards: Optional[List[Guard]] = None,
        require_all: bool = True,
    ):
        """Initialize composite guard.

        Args:
            guards: List of guards to combine.
            require_all: If True, all guards must allow. If False, any guard can block.
        """
        self.guards: List[Guard] = guards or []
        self.require_all = require_all

    def add(self, guard: Guard) -> "CompositeGuard":
        """Add a guard.

        Args:
            guard: Guard to add.

        Returns:
            Self for chaining.
        """
        self.guards.append(guard)
        return self

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check all guards."""
        all_matched_rules = []
        highest_risk = RiskLevel.SAFE

        for guard in self.guards:
            result = guard.check(tool_id, input_data)
            all_matched_rules.extend(result.matched_rules)

            # Update highest risk level
            if list(RiskLevel).index(result.risk_level) > list(RiskLevel).index(
                highest_risk
            ):
                highest_risk = result.risk_level

            if not result.allowed:
                return GuardResult(
                    allowed=False,
                    risk_level=result.risk_level,
                    reason=result.reason,
                    matched_rules=all_matched_rules,
                )

        return GuardResult(
            allowed=True,
            risk_level=highest_risk,
            matched_rules=all_matched_rules,
        )


class ToolGuardManager:
    """Manages guards for tools."""

    def __init__(self, default_guard: Optional[Guard] = None):
        """Initialize guard manager.

        Args:
            default_guard: Default guard for all tools.
        """
        self.default_guard = default_guard
        self._tool_guards: Dict[str, Guard] = {}
        self._global_guards: List[Guard] = []

    def add_global_guard(self, guard: Guard) -> "ToolGuardManager":
        """Add a global guard for all tools.

        Args:
            guard: Guard to add.

        Returns:
            Self for chaining.
        """
        self._global_guards.append(guard)
        return self

    def set_tool_guard(self, tool_id: str, guard: Guard) -> "ToolGuardManager":
        """Set guard for a specific tool.

        Args:
            tool_id: Tool identifier.
            guard: Guard to use.

        Returns:
            Self for chaining.
        """
        self._tool_guards[tool_id] = guard
        return self

    def check(self, tool_id: str, input_data: Any) -> GuardResult:
        """Check all applicable guards.

        Args:
            tool_id: Tool identifier.
            input_data: Tool input.

        Returns:
            Combined guard result.
        """
        all_matched_rules = []
        highest_risk = RiskLevel.SAFE

        # Check global guards first
        for guard in self._global_guards:
            result = guard.check(tool_id, input_data)
            all_matched_rules.extend(result.matched_rules)

            if not result.allowed:
                return GuardResult(
                    allowed=False,
                    risk_level=result.risk_level,
                    reason=result.reason,
                    matched_rules=all_matched_rules,
                )

            if list(RiskLevel).index(result.risk_level) > list(RiskLevel).index(
                highest_risk
            ):
                highest_risk = result.risk_level

        # Check tool-specific guard
        tool_guard = self._tool_guards.get(tool_id, self.default_guard)
        if tool_guard:
            result = tool_guard.check(tool_id, input_data)
            all_matched_rules.extend(result.matched_rules)

            if not result.allowed:
                return GuardResult(
                    allowed=False,
                    risk_level=result.risk_level,
                    reason=result.reason,
                    matched_rules=all_matched_rules,
                )

            if list(RiskLevel).index(result.risk_level) > list(RiskLevel).index(
                highest_risk
            ):
                highest_risk = result.risk_level

        return GuardResult(
            allowed=True,
            risk_level=highest_risk,
            matched_rules=all_matched_rules,
        )


class GuardedToolWrapper:
    """Wrapper that applies guards to tool execution."""

    def __init__(
        self,
        tool: Any,
        guard: Optional[Guard] = None,
        manager: Optional[ToolGuardManager] = None,
        on_blocked: Optional[Callable[[GuardResult], Any]] = None,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            guard: Guard to apply.
            manager: Guard manager.
            on_blocked: Callback when blocked.
        """
        self.tool = tool
        self.guard = guard
        self.manager = manager
        self.on_blocked = on_blocked

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(self, input_data: Any) -> Any:
        """Execute tool with guard checks.

        Args:
            input_data: Tool input.

        Returns:
            Tool output.

        Raises:
            PermissionError: If blocked by guard.
        """
        tool_id = self.tool.metadata.id

        # Check guard
        if self.manager:
            result = self.manager.check(tool_id, input_data)
        elif self.guard:
            result = self.guard.check(tool_id, input_data)
        else:
            result = GuardResult(allowed=True)

        if not result.allowed:
            logger.warning(
                f"Tool {tool_id} blocked: {result.reason}",
                extra=result.to_dict(),
            )

            if self.on_blocked:
                return self.on_blocked(result)

            raise PermissionError(
                f"Tool execution blocked: {result.reason}"
            )

        return await self.tool.execute(input_data)


# Pre-built guards for common patterns

DANGEROUS_COMMAND_PATTERNS = [
    r"rm\s+-rf",
    r"sudo\s+",
    r"chmod\s+777",
    r">\s*/dev/",
    r"mkfs\.",
    r"dd\s+if=",
    r":(){ :|:& };:",  # Fork bomb
    r"wget.*\|.*sh",
    r"curl.*\|.*bash",
]

SENSITIVE_DATA_PATTERNS = [
    r"password\s*[:=]",
    r"api[_-]?key\s*[:=]",
    r"secret\s*[:=]",
    r"token\s*[:=]",
    r"credential",
    r"-----BEGIN.*PRIVATE KEY-----",
    r"ssh-rsa\s+",
]

SQL_INJECTION_PATTERNS = [
    r";\s*DROP\s+",
    r";\s*DELETE\s+",
    r"UNION\s+SELECT",
    r"OR\s+1\s*=\s*1",
    r"--\s*$",
    r"'\s*OR\s*'",
]


def dangerous_command_guard() -> BlocklistGuard:
    """Create a guard for dangerous shell commands.

    Returns:
        BlocklistGuard configured for dangerous commands.
    """
    return BlocklistGuard(patterns=DANGEROUS_COMMAND_PATTERNS)


def sensitive_data_guard() -> BlocklistGuard:
    """Create a guard for sensitive data patterns.

    Returns:
        BlocklistGuard configured for sensitive data.
    """
    return BlocklistGuard(patterns=SENSITIVE_DATA_PATTERNS)


def sql_injection_guard() -> BlocklistGuard:
    """Create a guard for SQL injection patterns.

    Returns:
        BlocklistGuard configured for SQL injection.
    """
    return BlocklistGuard(patterns=SQL_INJECTION_PATTERNS)


# Convenience functions


def with_guard(
    tool: Any,
    guard: Guard,
    on_blocked: Optional[Callable[[GuardResult], Any]] = None,
) -> GuardedToolWrapper:
    """Add a guard to a tool.

    Args:
        tool: Tool to wrap.
        guard: Guard to apply.
        on_blocked: Callback when blocked.

    Returns:
        GuardedToolWrapper.
    """
    return GuardedToolWrapper(tool, guard=guard, on_blocked=on_blocked)


def create_guard_manager() -> ToolGuardManager:
    """Create a new guard manager.

    Returns:
        ToolGuardManager instance.
    """
    return ToolGuardManager()
