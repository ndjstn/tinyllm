"""Tool node implementation.

Executes tools and returns results.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class ToolNodeConfig(NodeConfig):
    """Configuration for tool nodes."""

    tool_id: str = Field(description="ID of the tool to execute")
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Map message fields to tool input parameters",
    )
    output_field: str = Field(
        default="tool_result",
        description="Field name for tool output in message",
    )
    continue_on_error: bool = Field(
        default=False,
        description="Continue execution even if tool fails",
    )


@NodeRegistry.register(NodeType.TOOL)
class ToolNode(BaseNode):
    """Executes a tool and returns results.

    The ToolNode wraps tool execution with proper error handling
    and result formatting.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize tool node."""
        super().__init__(definition)
        self._tool_config = ToolNodeConfig(**definition.config)

    @property
    def tool_config(self) -> ToolNodeConfig:
        """Get tool-specific configuration."""
        return self._tool_config

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute the configured tool.

        Args:
            message: Input message with tool parameters.
            context: Execution context.

        Returns:
            NodeResult with tool output.
        """
        # Get the tool
        tool = ToolRegistry.get(self._tool_config.tool_id)
        if tool is None:
            return NodeResult.failure_result(
                error=f"Tool not found: {self._tool_config.tool_id}"
            )

        # Check if tool is enabled
        if not tool.config.enabled:
            return NodeResult.failure_result(
                error=f"Tool is disabled: {self._tool_config.tool_id}"
            )

        # Extract tool input from message
        tool_input = self._extract_tool_input(message)

        try:
            # Execute the tool
            result = await tool.execute(tool_input)

            # Create output message with tool result
            output_payload = MessagePayload(
                task=message.payload.task,
                content=message.payload.content,
                tool_calls=[
                    {
                        "tool_id": self._tool_config.tool_id,
                        "input": tool_input,
                        "output": result.output,
                        "success": result.success,
                    }
                ],
                metadata={
                    **message.payload.metadata,
                    self._tool_config.output_field: result.output,
                    "tool_success": result.success,
                },
            )

            output_message = message.create_child(
                source_node=self.id,
                payload=output_payload,
            )

            if result.success:
                return NodeResult.success_result(
                    output_messages=[output_message],
                    next_nodes=[],
                    metadata={
                        "tool_id": self._tool_config.tool_id,
                        "tool_output": result.output,
                    },
                )
            else:
                if self._tool_config.continue_on_error:
                    return NodeResult.success_result(
                        output_messages=[output_message],
                        next_nodes=[],
                        metadata={
                            "tool_id": self._tool_config.tool_id,
                            "tool_error": result.error,
                        },
                    )
                return NodeResult.failure_result(
                    error=f"Tool execution failed: {result.error}"
                )

        except Exception as e:
            if self._tool_config.continue_on_error:
                # Create message with error info
                output_payload = MessagePayload(
                    task=message.payload.task,
                    content=message.payload.content,
                    metadata={
                        **message.payload.metadata,
                        "tool_error": str(e),
                    },
                )
                output_message = message.create_child(
                    source_node=self.id,
                    payload=output_payload,
                )
                return NodeResult.success_result(
                    output_messages=[output_message],
                    next_nodes=[],
                    metadata={"tool_error": str(e)},
                )
            return NodeResult.failure_result(
                error=f"Tool execution error: {str(e)}"
            )

    def _extract_tool_input(self, message: Message) -> Dict[str, Any]:
        """Extract tool input from message based on input mapping."""
        if not self._tool_config.input_mapping:
            # Default: use content as the tool input
            return {"input": message.payload.content}

        payload_dict = message.payload.model_dump()
        tool_input = {}

        for tool_param, message_field in self._tool_config.input_mapping.items():
            # Support nested field access with dot notation
            value = self._get_nested_field(payload_dict, message_field)
            if value is not None:
                tool_input[tool_param] = value

        return tool_input

    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get a nested field value using dot notation."""
        parts = field_path.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
