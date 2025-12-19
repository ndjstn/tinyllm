"""Model node implementation.

Invokes LLMs for task completion and response generation.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.models.client import OllamaClient
from tinyllm.prompts.loader import PromptLoader

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class ModelNodeConfig(NodeConfig):
    """Configuration for model nodes."""

    model: str = Field(
        default="qwen2.5:3b", description="Model to use for generation"
    )
    prompt_id: Optional[str] = Field(
        default=None, description="Prompt ID for task execution"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Direct system prompt (overrides prompt_id)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, le=32000, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Enable streaming output")
    tools_enabled: bool = Field(
        default=False, description="Enable tool use for this model"
    )
    tool_ids: List[str] = Field(
        default_factory=list, description="Specific tools to enable"
    )


@NodeRegistry.register(NodeType.MODEL)
class ModelNode(BaseNode):
    """Invokes an LLM for task completion.

    The ModelNode is a specialist that processes tasks using an LLM.
    It can be configured with specific prompts, models, and tool access.
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize model node."""
        super().__init__(definition)
        self._model_config = ModelNodeConfig(**definition.config)
        self._client: Optional[OllamaClient] = None
        self._prompt_loader = PromptLoader()

    @property
    def model_config(self) -> ModelNodeConfig:
        """Get model-specific configuration."""
        return self._model_config

    def _get_client(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Execute model inference.

        Processes the task using the configured LLM.

        Args:
            message: Input message with task.
            context: Execution context.

        Returns:
            NodeResult with generated response.
        """
        # Build prompt from message
        prompt = self._build_prompt(message)
        system = self._get_system_prompt(message)

        try:
            # Generate response
            client = self._get_client()
            response = await client.generate(
                model=self._model_config.model,
                prompt=prompt,
                system=system,
                options={
                    "temperature": self._model_config.temperature,
                    **(
                        {"num_predict": self._model_config.max_tokens}
                        if self._model_config.max_tokens
                        else {}
                    ),
                },
            )

            # Create output message with response
            output_payload = MessagePayload(
                task=message.payload.task,
                content=response.response,
                metadata={
                    **message.payload.metadata,
                    "model": self._model_config.model,
                    "tokens_generated": response.eval_count,
                    "tokens_prompt": response.prompt_eval_count,
                },
            )

            output_message = message.create_child(
                source_node=self.id,
                payload=output_payload,
            )

            return NodeResult.success_result(
                output_messages=[output_message],
                next_nodes=[],  # Determined by executor from edges
                metadata={
                    "model": self._model_config.model,
                    "tokens": response.eval_count + response.prompt_eval_count,
                    "eval_duration_ms": response.eval_duration / 1_000_000,
                },
            )

        except Exception as e:
            return NodeResult.failure_result(error=f"Model execution failed: {str(e)}")

    def _build_prompt(self, message: Message) -> str:
        """Build the user prompt from message content."""
        # Use task if available, otherwise content
        return message.payload.task or message.payload.content or ""

    def _get_system_prompt(self, message: Message) -> Optional[str]:
        """Get the system prompt for this model node."""
        # Direct system prompt takes precedence
        if self._model_config.system_prompt:
            return self._model_config.system_prompt

        # Try to load from prompt_id
        if self._model_config.prompt_id:
            try:
                prompt_def = self._prompt_loader.load(self._model_config.prompt_id)
                return prompt_def.system_prompt
            except FileNotFoundError:
                pass

        # Default system prompt
        return "You are a helpful assistant. Provide clear, accurate, and concise responses."

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for tool-enabled models."""
        if not self._model_config.tools_enabled:
            return []

        from tinyllm.tools.registry import ToolRegistry

        tools = []
        tool_ids = self._model_config.tool_ids or ToolRegistry.get_tool_ids()

        for tool_id in tool_ids:
            tool = ToolRegistry.get(tool_id)
            if tool:
                tools.append(tool.get_definition())

        return tools
