"""Tool approval workflows for TinyLLM.

This module provides approval workflows for tools,
requiring explicit approval before execution of sensitive operations.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalLevel(str, Enum):
    """Levels of approval required."""

    NONE = "none"  # No approval needed
    LOW = "low"  # Basic confirmation
    MEDIUM = "medium"  # Requires review
    HIGH = "high"  # Requires explicit approval
    CRITICAL = "critical"  # Requires multi-party approval


@dataclass
class ApprovalRequest:
    """Request for tool execution approval."""

    id: str
    tool_id: str
    input_data: Any
    level: ApprovalLevel
    requester: Optional[str] = None
    reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        if self.is_expired:
            return False
        return self.status == ApprovalStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "tool_id": self.tool_id,
            "level": self.level.value,
            "status": self.status.value,
            "requester": self.requester,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "approver": self.approver,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class ApprovalPolicy:
    """Policy for requiring approvals."""

    tool_ids: Set[str] = field(default_factory=set)
    level: ApprovalLevel = ApprovalLevel.MEDIUM
    timeout_seconds: float = 300.0  # 5 minutes
    require_reason: bool = False
    allowed_approvers: Optional[Set[str]] = None
    auto_approve_conditions: List[Callable[[Any], bool]] = field(default_factory=list)

    def matches(self, tool_id: str) -> bool:
        """Check if policy applies to a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            True if policy applies.
        """
        if not self.tool_ids:
            return True  # Applies to all
        return tool_id in self.tool_ids


class ApprovalHandler(ABC):
    """Abstract base for approval handlers."""

    @abstractmethod
    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval for a tool execution.

        Args:
            request: Approval request.

        Returns:
            Final approval status.
        """
        pass


class AutoApproveHandler(ApprovalHandler):
    """Handler that automatically approves all requests."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Automatically approve."""
        request.status = ApprovalStatus.APPROVED
        request.approver = "auto"
        request.approved_at = datetime.now()
        return ApprovalStatus.APPROVED


class AutoRejectHandler(ApprovalHandler):
    """Handler that automatically rejects all requests."""

    def __init__(self, reason: str = "Auto-rejected by policy"):
        """Initialize handler.

        Args:
            reason: Rejection reason.
        """
        self.reason = reason

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Automatically reject."""
        request.status = ApprovalStatus.REJECTED
        request.rejection_reason = self.reason
        return ApprovalStatus.REJECTED


class CallbackApprovalHandler(ApprovalHandler):
    """Handler that uses a callback for approval decisions."""

    def __init__(
        self,
        callback: Callable[[ApprovalRequest], bool],
        approver_name: str = "callback",
    ):
        """Initialize handler.

        Args:
            callback: Function that returns True to approve.
            approver_name: Name of the approver.
        """
        self.callback = callback
        self.approver_name = approver_name

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval via callback."""
        try:
            approved = self.callback(request)

            if approved:
                request.status = ApprovalStatus.APPROVED
                request.approver = self.approver_name
                request.approved_at = datetime.now()
                return ApprovalStatus.APPROVED
            else:
                request.status = ApprovalStatus.REJECTED
                return ApprovalStatus.REJECTED

        except Exception as e:
            logger.error(f"Approval callback error: {e}")
            request.status = ApprovalStatus.REJECTED
            request.rejection_reason = str(e)
            return ApprovalStatus.REJECTED


class AsyncCallbackApprovalHandler(ApprovalHandler):
    """Handler that uses an async callback for approval decisions."""

    def __init__(
        self,
        callback: Callable[[ApprovalRequest], Any],
        approver_name: str = "async_callback",
    ):
        """Initialize handler.

        Args:
            callback: Async function that returns True to approve.
            approver_name: Name of the approver.
        """
        self.callback = callback
        self.approver_name = approver_name

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Request approval via async callback."""
        try:
            approved = await self.callback(request)

            if approved:
                request.status = ApprovalStatus.APPROVED
                request.approver = self.approver_name
                request.approved_at = datetime.now()
                return ApprovalStatus.APPROVED
            else:
                request.status = ApprovalStatus.REJECTED
                return ApprovalStatus.REJECTED

        except Exception as e:
            logger.error(f"Async approval callback error: {e}")
            request.status = ApprovalStatus.REJECTED
            request.rejection_reason = str(e)
            return ApprovalStatus.REJECTED


class QueuedApprovalHandler(ApprovalHandler):
    """Handler that queues requests for later approval."""

    def __init__(self, timeout: float = 300.0):
        """Initialize handler.

        Args:
            timeout: Timeout in seconds.
        """
        self.timeout = timeout
        self._pending: Dict[str, ApprovalRequest] = {}
        self._results: Dict[str, asyncio.Event] = {}

    async def request_approval(self, request: ApprovalRequest) -> ApprovalStatus:
        """Queue request and wait for approval."""
        self._pending[request.id] = request
        self._results[request.id] = asyncio.Event()

        try:
            await asyncio.wait_for(
                self._results[request.id].wait(),
                timeout=self.timeout,
            )
            return request.status

        except asyncio.TimeoutError:
            request.status = ApprovalStatus.EXPIRED
            return ApprovalStatus.EXPIRED

        finally:
            del self._pending[request.id]
            del self._results[request.id]

    def approve(self, request_id: str, approver: str = "unknown") -> bool:
        """Approve a pending request.

        Args:
            request_id: Request ID.
            approver: Approver name.

        Returns:
            True if approved.
        """
        if request_id not in self._pending:
            return False

        request = self._pending[request_id]
        request.status = ApprovalStatus.APPROVED
        request.approver = approver
        request.approved_at = datetime.now()
        self._results[request_id].set()
        return True

    def reject(self, request_id: str, reason: Optional[str] = None) -> bool:
        """Reject a pending request.

        Args:
            request_id: Request ID.
            reason: Rejection reason.

        Returns:
            True if rejected.
        """
        if request_id not in self._pending:
            return False

        request = self._pending[request_id]
        request.status = ApprovalStatus.REJECTED
        request.rejection_reason = reason
        self._results[request_id].set()
        return True

    def get_pending(self) -> List[ApprovalRequest]:
        """Get pending requests.

        Returns:
            List of pending requests.
        """
        return list(self._pending.values())


class ApprovalManager:
    """Manages approval workflows for tools."""

    def __init__(
        self,
        default_handler: Optional[ApprovalHandler] = None,
        default_policy: Optional[ApprovalPolicy] = None,
    ):
        """Initialize approval manager.

        Args:
            default_handler: Default approval handler.
            default_policy: Default approval policy.
        """
        self.default_handler = default_handler or AutoApproveHandler()
        self.default_policy = default_policy
        self._policies: List[ApprovalPolicy] = []
        self._handlers: Dict[ApprovalLevel, ApprovalHandler] = {}
        self._history: List[ApprovalRequest] = []
        self._max_history = 1000

    def add_policy(self, policy: ApprovalPolicy) -> "ApprovalManager":
        """Add an approval policy.

        Args:
            policy: Policy to add.

        Returns:
            Self for chaining.
        """
        self._policies.append(policy)
        return self

    def set_handler(
        self, level: ApprovalLevel, handler: ApprovalHandler
    ) -> "ApprovalManager":
        """Set handler for an approval level.

        Args:
            level: Approval level.
            handler: Handler to use.

        Returns:
            Self for chaining.
        """
        self._handlers[level] = handler
        return self

    def get_policy(self, tool_id: str) -> Optional[ApprovalPolicy]:
        """Get applicable policy for a tool.

        Args:
            tool_id: Tool identifier.

        Returns:
            Matching policy or None.
        """
        for policy in self._policies:
            if policy.matches(tool_id):
                return policy
        return self.default_policy

    def get_handler(self, level: ApprovalLevel) -> ApprovalHandler:
        """Get handler for approval level.

        Args:
            level: Approval level.

        Returns:
            Handler for level.
        """
        return self._handlers.get(level, self.default_handler)

    async def request_approval(
        self,
        tool_id: str,
        input_data: Any,
        requester: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> ApprovalRequest:
        """Request approval for tool execution.

        Args:
            tool_id: Tool identifier.
            input_data: Tool input.
            requester: Who is requesting.
            reason: Reason for execution.

        Returns:
            ApprovalRequest with status.
        """
        policy = self.get_policy(tool_id)

        if policy is None:
            # No policy, create auto-approved request
            request = ApprovalRequest(
                id=str(uuid.uuid4()),
                tool_id=tool_id,
                input_data=input_data,
                level=ApprovalLevel.NONE,
                requester=requester,
                reason=reason,
                status=ApprovalStatus.APPROVED,
                approver="policy",
                approved_at=datetime.now(),
            )
            self._add_to_history(request)
            return request

        # Check auto-approve conditions
        for condition in policy.auto_approve_conditions:
            try:
                if condition(input_data):
                    request = ApprovalRequest(
                        id=str(uuid.uuid4()),
                        tool_id=tool_id,
                        input_data=input_data,
                        level=policy.level,
                        requester=requester,
                        reason=reason,
                        status=ApprovalStatus.APPROVED,
                        approver="auto_condition",
                        approved_at=datetime.now(),
                    )
                    self._add_to_history(request)
                    return request
            except Exception:
                pass

        # Create request
        expires_at = None
        if policy.timeout_seconds > 0:
            from datetime import timedelta

            expires_at = datetime.now() + timedelta(seconds=policy.timeout_seconds)

        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            tool_id=tool_id,
            input_data=input_data,
            level=policy.level,
            requester=requester,
            reason=reason,
            expires_at=expires_at,
        )

        # Get handler and request approval
        handler = self.get_handler(policy.level)
        await handler.request_approval(request)

        self._add_to_history(request)
        return request

    def _add_to_history(self, request: ApprovalRequest) -> None:
        """Add request to history."""
        self._history.append(request)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def get_history(
        self,
        tool_id: Optional[str] = None,
        status: Optional[ApprovalStatus] = None,
    ) -> List[ApprovalRequest]:
        """Get approval history.

        Args:
            tool_id: Filter by tool ID.
            status: Filter by status.

        Returns:
            Matching requests.
        """
        result = self._history

        if tool_id:
            result = [r for r in result if r.tool_id == tool_id]

        if status:
            result = [r for r in result if r.status == status]

        return result


class ApprovalRequiredWrapper:
    """Wrapper that requires approval before tool execution."""

    def __init__(
        self,
        tool: Any,
        manager: Optional[ApprovalManager] = None,
        requester: Optional[str] = None,
        on_rejection: Optional[Callable[[ApprovalRequest], Any]] = None,
    ):
        """Initialize wrapper.

        Args:
            tool: Tool to wrap.
            manager: Approval manager.
            requester: Default requester name.
            on_rejection: Callback on rejection.
        """
        self.tool = tool
        self.manager = manager or ApprovalManager()
        self.requester = requester
        self.on_rejection = on_rejection

    @property
    def metadata(self):
        """Proxy metadata access."""
        return self.tool.metadata

    async def execute(
        self,
        input_data: Any,
        reason: Optional[str] = None,
    ) -> Any:
        """Execute tool with approval.

        Args:
            input_data: Tool input.
            reason: Reason for execution.

        Returns:
            Tool output.

        Raises:
            PermissionError: If not approved.
        """
        tool_id = self.tool.metadata.id

        request = await self.manager.request_approval(
            tool_id=tool_id,
            input_data=input_data,
            requester=self.requester,
            reason=reason,
        )

        if request.status == ApprovalStatus.APPROVED:
            logger.info(f"Tool {tool_id} approved by {request.approver}")
            return await self.tool.execute(input_data)

        logger.warning(f"Tool {tool_id} not approved: {request.status.value}")

        if self.on_rejection:
            return self.on_rejection(request)

        raise PermissionError(
            f"Tool execution not approved: {request.status.value}. "
            f"Reason: {request.rejection_reason or 'No reason provided'}"
        )


# Convenience functions


def with_approval(
    tool: Any,
    manager: Optional[ApprovalManager] = None,
    policy: Optional[ApprovalPolicy] = None,
) -> ApprovalRequiredWrapper:
    """Add approval requirement to a tool.

    Args:
        tool: Tool to wrap.
        manager: Approval manager.
        policy: Approval policy.

    Returns:
        ApprovalRequiredWrapper.
    """
    if manager is None:
        manager = ApprovalManager()

    if policy is not None:
        manager.add_policy(policy)

    return ApprovalRequiredWrapper(tool, manager=manager)


def require_approval(
    tool_ids: Optional[Set[str]] = None,
    level: ApprovalLevel = ApprovalLevel.MEDIUM,
    timeout_seconds: float = 300.0,
) -> ApprovalPolicy:
    """Create an approval policy.

    Args:
        tool_ids: Tools requiring approval.
        level: Approval level.
        timeout_seconds: Timeout.

    Returns:
        ApprovalPolicy.
    """
    return ApprovalPolicy(
        tool_ids=tool_ids or set(),
        level=level,
        timeout_seconds=timeout_seconds,
    )
