"""Tests for tool approval workflows."""

import pytest
from datetime import datetime, timedelta
from pydantic import BaseModel

from tinyllm.tools.base import BaseTool, ToolMetadata
from tinyllm.tools.approval import (
    ApprovalLevel,
    ApprovalManager,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRequiredWrapper,
    ApprovalStatus,
    AsyncCallbackApprovalHandler,
    AutoApproveHandler,
    AutoRejectHandler,
    CallbackApprovalHandler,
    QueuedApprovalHandler,
    require_approval,
    with_approval,
)


class ApprovalInput(BaseModel):
    """Input for approval tests."""

    value: int = 0
    is_safe: bool = True


class ApprovalOutput(BaseModel):
    """Output for approval tests."""

    value: int = 0
    success: bool = True


class NormalTool(BaseTool[ApprovalInput, ApprovalOutput]):
    """Normal tool for testing."""

    metadata = ToolMetadata(
        id="normal_tool",
        name="Normal Tool",
        description="Normal operation",
        category="utility",
    )
    input_type = ApprovalInput
    output_type = ApprovalOutput

    async def execute(self, input: ApprovalInput) -> ApprovalOutput:
        return ApprovalOutput(value=input.value * 2)


class DangerousTool(BaseTool[ApprovalInput, ApprovalOutput]):
    """Dangerous tool requiring approval."""

    metadata = ToolMetadata(
        id="dangerous_tool",
        name="Dangerous Tool",
        description="Dangerous operation",
        category="execution",
    )
    input_type = ApprovalInput
    output_type = ApprovalOutput

    async def execute(self, input: ApprovalInput) -> ApprovalOutput:
        return ApprovalOutput(value=input.value * 10)


class TestApprovalRequest:
    """Tests for ApprovalRequest."""

    def test_creation(self):
        """Test request creation."""
        request = ApprovalRequest(
            id="test-123",
            tool_id="my_tool",
            input_data={"value": 5},
            level=ApprovalLevel.MEDIUM,
        )

        assert request.id == "test-123"
        assert request.tool_id == "my_tool"
        assert request.status == ApprovalStatus.PENDING

    def test_is_pending(self):
        """Test is_pending property."""
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.LOW,
        )

        assert request.is_pending

        request.status = ApprovalStatus.APPROVED
        assert not request.is_pending

    def test_is_expired(self):
        """Test is_expired property."""
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.LOW,
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert request.is_expired

    def test_is_not_expired(self):
        """Test not expired."""
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.LOW,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert not request.is_expired

    def test_to_dict(self):
        """Test converting to dictionary."""
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.HIGH,
            requester="user1",
        )

        d = request.to_dict()

        assert d["id"] == "test"
        assert d["tool_id"] == "tool"
        assert d["level"] == "high"
        assert d["requester"] == "user1"


class TestApprovalPolicy:
    """Tests for ApprovalPolicy."""

    def test_matches_specific_tool(self):
        """Test matching specific tools."""
        policy = ApprovalPolicy(tool_ids={"tool1", "tool2"})

        assert policy.matches("tool1")
        assert policy.matches("tool2")
        assert not policy.matches("tool3")

    def test_matches_all_when_empty(self):
        """Test matching all when no tools specified."""
        policy = ApprovalPolicy()

        assert policy.matches("any_tool")
        assert policy.matches("another_tool")


class TestApprovalHandlers:
    """Tests for approval handlers."""

    @pytest.mark.asyncio
    async def test_auto_approve(self):
        """Test auto-approve handler."""
        handler = AutoApproveHandler()
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.LOW,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.APPROVED
        assert request.approver == "auto"

    @pytest.mark.asyncio
    async def test_auto_reject(self):
        """Test auto-reject handler."""
        handler = AutoRejectHandler(reason="Not allowed")
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.HIGH,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.REJECTED
        assert request.rejection_reason == "Not allowed"

    @pytest.mark.asyncio
    async def test_callback_approve(self):
        """Test callback handler approving."""
        handler = CallbackApprovalHandler(
            callback=lambda r: True,
            approver_name="test_approver",
        )
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.MEDIUM,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.APPROVED
        assert request.approver == "test_approver"

    @pytest.mark.asyncio
    async def test_callback_reject(self):
        """Test callback handler rejecting."""
        handler = CallbackApprovalHandler(callback=lambda r: False)
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.MEDIUM,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_callback_error(self):
        """Test callback handler with error."""

        def failing_callback(r):
            raise ValueError("Callback error")

        handler = CallbackApprovalHandler(callback=failing_callback)
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.MEDIUM,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.REJECTED
        assert "Callback error" in request.rejection_reason

    @pytest.mark.asyncio
    async def test_async_callback_approve(self):
        """Test async callback handler."""

        async def approve_callback(r):
            return True

        handler = AsyncCallbackApprovalHandler(callback=approve_callback)
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.MEDIUM,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.APPROVED


class TestQueuedApprovalHandler:
    """Tests for QueuedApprovalHandler."""

    @pytest.mark.asyncio
    async def test_approve_pending(self):
        """Test approving a pending request."""
        import asyncio

        handler = QueuedApprovalHandler(timeout=5.0)
        request = ApprovalRequest(
            id="test-123",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.HIGH,
        )

        async def approve_later():
            await asyncio.sleep(0.1)
            handler.approve("test-123", "admin")

        asyncio.create_task(approve_later())

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.APPROVED
        assert request.approver == "admin"

    @pytest.mark.asyncio
    async def test_reject_pending(self):
        """Test rejecting a pending request."""
        import asyncio

        handler = QueuedApprovalHandler(timeout=5.0)
        request = ApprovalRequest(
            id="test-456",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.HIGH,
        )

        async def reject_later():
            await asyncio.sleep(0.1)
            handler.reject("test-456", "Not allowed")

        asyncio.create_task(reject_later())

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.REJECTED
        assert request.rejection_reason == "Not allowed"

    @pytest.mark.asyncio
    async def test_timeout_expires(self):
        """Test timeout expiration."""
        handler = QueuedApprovalHandler(timeout=0.1)
        request = ApprovalRequest(
            id="test",
            tool_id="tool",
            input_data={},
            level=ApprovalLevel.HIGH,
        )

        status = await handler.request_approval(request)

        assert status == ApprovalStatus.EXPIRED

    def test_get_pending(self):
        """Test getting pending requests."""
        import asyncio

        handler = QueuedApprovalHandler(timeout=10.0)

        async def add_request():
            request = ApprovalRequest(
                id="pending-1",
                tool_id="tool",
                input_data={},
                level=ApprovalLevel.HIGH,
            )
            # Start request but don't await
            task = asyncio.create_task(handler.request_approval(request))
            await asyncio.sleep(0.05)
            pending = handler.get_pending()
            handler.approve("pending-1")
            await task
            return pending

        pending = asyncio.get_event_loop().run_until_complete(add_request())
        assert len(pending) == 1


class TestApprovalManager:
    """Tests for ApprovalManager."""

    @pytest.mark.asyncio
    async def test_no_policy_auto_approves(self):
        """Test auto-approval when no policy."""
        manager = ApprovalManager()

        request = await manager.request_approval(
            tool_id="any_tool",
            input_data={"value": 5},
        )

        assert request.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_policy_matching(self):
        """Test policy matching."""
        manager = ApprovalManager()
        manager.add_policy(
            ApprovalPolicy(
                tool_ids={"dangerous_tool"},
                level=ApprovalLevel.HIGH,
            )
        )
        manager.set_handler(ApprovalLevel.HIGH, AutoRejectHandler())

        request = await manager.request_approval(
            tool_id="dangerous_tool",
            input_data={},
        )

        assert request.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_auto_approve_condition(self):
        """Test auto-approve condition."""
        manager = ApprovalManager()
        manager.add_policy(
            ApprovalPolicy(
                tool_ids={"conditional_tool"},
                level=ApprovalLevel.HIGH,
                auto_approve_conditions=[
                    lambda x: x.get("is_safe", False),
                ],
            )
        )
        manager.set_handler(ApprovalLevel.HIGH, AutoRejectHandler())

        # Should be auto-approved due to condition
        request = await manager.request_approval(
            tool_id="conditional_tool",
            input_data={"is_safe": True},
        )

        assert request.status == ApprovalStatus.APPROVED
        assert request.approver == "auto_condition"

    @pytest.mark.asyncio
    async def test_history(self):
        """Test approval history."""
        manager = ApprovalManager()

        await manager.request_approval("tool1", {})
        await manager.request_approval("tool2", {})
        await manager.request_approval("tool1", {})

        all_history = manager.get_history()
        assert len(all_history) == 3

        tool1_history = manager.get_history(tool_id="tool1")
        assert len(tool1_history) == 2

    @pytest.mark.asyncio
    async def test_handler_by_level(self):
        """Test different handlers for levels."""
        manager = ApprovalManager()
        manager.set_handler(ApprovalLevel.LOW, AutoApproveHandler())
        manager.set_handler(ApprovalLevel.HIGH, AutoRejectHandler())

        manager.add_policy(ApprovalPolicy(tool_ids={"low_tool"}, level=ApprovalLevel.LOW))
        manager.add_policy(ApprovalPolicy(tool_ids={"high_tool"}, level=ApprovalLevel.HIGH))

        low_request = await manager.request_approval("low_tool", {})
        high_request = await manager.request_approval("high_tool", {})

        assert low_request.status == ApprovalStatus.APPROVED
        assert high_request.status == ApprovalStatus.REJECTED


class TestApprovalRequiredWrapper:
    """Tests for ApprovalRequiredWrapper."""

    @pytest.mark.asyncio
    async def test_approved_execution(self):
        """Test execution when approved."""
        manager = ApprovalManager()  # Default auto-approves
        wrapper = ApprovalRequiredWrapper(NormalTool(), manager=manager)

        result = await wrapper.execute(ApprovalInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_rejected_raises(self):
        """Test rejection raises PermissionError."""
        manager = ApprovalManager(default_handler=AutoRejectHandler())
        manager.add_policy(ApprovalPolicy(level=ApprovalLevel.HIGH))

        wrapper = ApprovalRequiredWrapper(DangerousTool(), manager=manager)

        with pytest.raises(PermissionError) as exc_info:
            await wrapper.execute(ApprovalInput(value=5))

        assert "not approved" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_on_rejection_callback(self):
        """Test on_rejection callback."""
        manager = ApprovalManager(default_handler=AutoRejectHandler())
        manager.add_policy(ApprovalPolicy(level=ApprovalLevel.HIGH))

        def on_rejection(request):
            return ApprovalOutput(value=0, success=False)

        wrapper = ApprovalRequiredWrapper(
            DangerousTool(),
            manager=manager,
            on_rejection=on_rejection,
        )

        result = await wrapper.execute(ApprovalInput(value=5))

        assert result.success is False

    @pytest.mark.asyncio
    async def test_metadata_proxy(self):
        """Test metadata proxy."""
        wrapper = ApprovalRequiredWrapper(NormalTool())

        assert wrapper.metadata.id == "normal_tool"

    @pytest.mark.asyncio
    async def test_with_reason(self):
        """Test execution with reason."""
        manager = ApprovalManager()
        wrapper = ApprovalRequiredWrapper(NormalTool(), manager=manager)

        await wrapper.execute(ApprovalInput(value=5), reason="Testing")

        history = manager.get_history()
        assert history[0].reason == "Testing"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_with_approval(self):
        """Test with_approval function."""
        wrapper = with_approval(NormalTool())

        result = await wrapper.execute(ApprovalInput(value=5))

        assert result.value == 10

    @pytest.mark.asyncio
    async def test_with_approval_and_policy(self):
        """Test with_approval with policy."""
        policy = require_approval(
            tool_ids={"normal_tool"},
            level=ApprovalLevel.LOW,
        )

        wrapper = with_approval(NormalTool(), policy=policy)

        result = await wrapper.execute(ApprovalInput(value=5))

        assert result.value == 10

    def test_require_approval(self):
        """Test require_approval function."""
        policy = require_approval(
            tool_ids={"tool1", "tool2"},
            level=ApprovalLevel.HIGH,
            timeout_seconds=600.0,
        )

        assert policy.tool_ids == {"tool1", "tool2"}
        assert policy.level == ApprovalLevel.HIGH
        assert policy.timeout_seconds == 600.0
