"""Entry and Exit node implementations.

These are the fundamental nodes for starting and ending graph execution.
"""

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class EntryNodeConfig(NodeConfig):
    """Configuration for entry nodes."""

    input_schema: Optional[str] = Field(
        default=None, description="Pydantic model name for input validation"
    )
    required_fields: List[str] = Field(
        default_factory=list, description="Required fields in payload"
    )


@NodeRegistry.register(NodeType.ENTRY)
class EntryNode(BaseNode):
    """Entry point node for the graph.

    The EntryNode is the first node executed in a graph. It validates
    input and sets up the initial execution context.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize entry node."""
        super().__init__(definition)
        self._entry_config = EntryNodeConfig(**definition.config)

    @property
    def entry_config(self) -> EntryNodeConfig:
        """Get entry-specific configuration."""
        return self._entry_config

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute entry node logic.

        Validates the input message and prepares it for routing.

        Args:
            message: Input message.
            context: Execution context.

        Returns:
            NodeResult with validated message.
        """
        # Validate required fields
        if self._entry_config.required_fields:
            payload_dict = message.payload.model_dump()
            missing = []
            for field in self._entry_config.required_fields:
                if field not in payload_dict or payload_dict[field] is None:
                    missing.append(field)

            if missing:
                return NodeResult.failure_result(
                    error=f"Missing required fields: {', '.join(missing)}"
                )

        # Create output message
        output_message = message.create_child(
            source_node=self.id,
            payload=message.payload.model_copy(),
        )

        return NodeResult.success_result(
            output_messages=[output_message],
            next_nodes=[],  # Executor will determine from edges
            metadata={"validated": True},
        )


class ExitNodeConfig(NodeConfig):
    """Configuration for exit nodes."""

    output_schema: Optional[str] = Field(
        default=None, description="Pydantic model name for output"
    )
    status: str = Field(
        default="success",
        pattern=r"^(success|fallback|error)$",
        description="Exit status type",
    )


@NodeRegistry.register(NodeType.EXIT)
class ExitNode(BaseNode):
    """Exit point node for the graph.

    The ExitNode is a terminal node that packages the final response.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize exit node."""
        super().__init__(definition)
        self._exit_config = ExitNodeConfig(**definition.config)

    @property
    def exit_config(self) -> ExitNodeConfig:
        """Get exit-specific configuration."""
        return self._exit_config

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute exit node logic.

        Packages the final response for return to the caller.

        Args:
            message: Input message.
            context: Execution context.

        Returns:
            NodeResult with final message and empty next_nodes.
        """
        # Create final output message
        output_message = message.create_child(
            source_node=self.id,
            payload=MessagePayload(
                content=message.payload.content,
                structured=message.payload.structured,
            ),
        )

        # Determine success based on exit status
        success = self._exit_config.status == "success"

        return NodeResult(
            success=success,
            output_messages=[output_message],
            next_nodes=[],  # Exit nodes have no next nodes
            metadata={
                "exit_status": self._exit_config.status,
                "is_terminal": True,
            },
        )
