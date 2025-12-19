"""Gate node implementation.

Conditional branching based on message content or LLM evaluation.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.models.client import OllamaClient

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class GateCondition(BaseModel):
    """A condition for gate evaluation."""

    name: str = Field(description="Condition identifier")
    expression: str = Field(description="Python expression to evaluate")
    target: str = Field(description="Target node if condition is true")


class GateNodeConfig(NodeConfig):
    """Configuration for gate nodes."""

    mode: str = Field(
        default="expression",
        pattern=r"^(expression|llm|hybrid)$",
        description="Gate evaluation mode",
    )
    conditions: List[GateCondition] = Field(
        default_factory=list,
        description="Conditions to evaluate",
    )
    default_target: Optional[str] = Field(
        default=None,
        description="Default target if no conditions match",
    )
    # LLM mode settings
    model: str = Field(
        default="qwen2.5:0.5b",
        description="Model for LLM-based evaluation",
    )
    evaluation_prompt: Optional[str] = Field(
        default=None,
        description="Prompt for LLM evaluation",
    )


@NodeRegistry.register(NodeType.GATE)
class GateNode(BaseNode):
    """Conditional gate for branching logic.

    Gates can evaluate conditions using:
    - expression: Python expressions on message content
    - llm: LLM-based evaluation
    - hybrid: Expression first, LLM fallback
    """

    def __init__(self, definition: NodeDefinition):
        """Initialize gate node."""
        super().__init__(definition)
        self._gate_config = GateNodeConfig(**definition.config)
        self._client: Optional[OllamaClient] = None

    @property
    def gate_config(self) -> GateNodeConfig:
        """Get gate-specific configuration."""
        return self._gate_config

    def _get_client(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = OllamaClient()
        return self._client

    async def execute(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Evaluate gate conditions.

        Args:
            message: Input message.
            context: Execution context.

        Returns:
            NodeResult with next node based on conditions.
        """
        mode = self._gate_config.mode

        if mode == "expression":
            return await self._evaluate_expressions(message, context)
        elif mode == "llm":
            return await self._evaluate_llm(message, context)
        elif mode == "hybrid":
            result = await self._evaluate_expressions(message, context)
            if result.success and result.next_nodes:
                return result
            return await self._evaluate_llm(message, context)
        else:
            return NodeResult.failure_result(error=f"Unknown gate mode: {mode}")

    async def _evaluate_expressions(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Evaluate Python expressions against message content."""
        # Build evaluation context
        eval_context = self._build_eval_context(message, context)

        for condition in self._gate_config.conditions:
            try:
                # Safely evaluate expression
                result = eval(condition.expression, {"__builtins__": {}}, eval_context)
                if result:
                    output_message = message.create_child(
                        source_node=self.id,
                        payload=message.payload.model_copy(),
                    )
                    return NodeResult.success_result(
                        output_messages=[output_message],
                        next_nodes=[condition.target],
                        metadata={
                            "matched_condition": condition.name,
                            "expression": condition.expression,
                        },
                    )
            except Exception as e:
                # Log but continue checking other conditions
                continue

        # No condition matched, use default
        if self._gate_config.default_target:
            output_message = message.create_child(
                source_node=self.id,
                payload=message.payload.model_copy(),
            )
            return NodeResult.success_result(
                output_messages=[output_message],
                next_nodes=[self._gate_config.default_target],
                metadata={"matched_condition": "default"},
            )

        return NodeResult.failure_result(error="No gate condition matched")

    async def _evaluate_llm(
        self, message: Message, context: "ExecutionContext"
    ) -> NodeResult:
        """Use LLM to evaluate which path to take."""
        if not self._gate_config.conditions:
            return NodeResult.failure_result(error="No conditions defined for LLM gate")

        # Build prompt for LLM
        options = "\n".join(
            f"- {c.name}: {c.expression}" for c in self._gate_config.conditions
        )
        prompt = self._gate_config.evaluation_prompt or f"""Given the following content, which option best applies?

Content: {message.payload.content}

Options:
{options}

Respond with ONLY the option name, nothing else."""

        try:
            client = self._get_client()
            response = await client.generate(
                model=self._gate_config.model,
                prompt=prompt,
            )

            # Parse response to find matching condition
            choice = response.response.strip().lower()

            for condition in self._gate_config.conditions:
                if condition.name.lower() in choice:
                    output_message = message.create_child(
                        source_node=self.id,
                        payload=message.payload.model_copy(),
                    )
                    return NodeResult.success_result(
                        output_messages=[output_message],
                        next_nodes=[condition.target],
                        metadata={
                            "matched_condition": condition.name,
                            "llm_response": response.response,
                        },
                    )

            # No match found
            if self._gate_config.default_target:
                output_message = message.create_child(
                    source_node=self.id,
                    payload=message.payload.model_copy(),
                )
                return NodeResult.success_result(
                    output_messages=[output_message],
                    next_nodes=[self._gate_config.default_target],
                    metadata={
                        "matched_condition": "default",
                        "llm_response": response.response,
                    },
                )

            return NodeResult.failure_result(
                error=f"LLM response didn't match any condition: {response.response}"
            )

        except Exception as e:
            return NodeResult.failure_result(error=f"LLM gate evaluation failed: {str(e)}")

    def _build_eval_context(
        self, message: Message, context: "ExecutionContext"
    ) -> Dict[str, Any]:
        """Build safe evaluation context from message and execution context."""
        return {
            # Message content
            "content": message.payload.content or "",
            "task": message.payload.task or "",
            "structured": message.payload.structured or {},
            "metadata": message.payload.metadata,
            # Routing info from payload
            "route": message.payload.route,
            "routes": message.payload.metadata.get("routes", []),
            "compound_route": message.payload.metadata.get("compound_route", False),
            "confidence": message.payload.confidence,
            # Execution context
            "variables": context.variables,
            # Utility functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "any": any,
            "all": all,
        }
