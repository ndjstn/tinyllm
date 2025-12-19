"""Reasoning node implementation.

Provides chain-of-thought reasoning with adversarial defense.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import Field

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.core.message import Message, MessagePayload
from tinyllm.core.node import BaseNode, NodeConfig, NodeResult
from tinyllm.core.registry import NodeRegistry
from tinyllm.models.client import OllamaClient
from tinyllm.reasoning import (
    ChainStatus,
    EngineConfig,
    ReasoningEngine,
    TrapDetector,
    TrapType,
)

if TYPE_CHECKING:
    from tinyllm.core.context import ExecutionContext


class ReasoningNodeConfig(NodeConfig):
    """Configuration for reasoning nodes."""

    model: str = Field(
        default="qwen2.5:3b",
        description="Model to use for reasoning",
    )
    max_steps: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum reasoning steps",
    )
    max_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum reasoning depth",
    )
    require_verification: bool = Field(
        default=True,
        description="Require verification before conclusion",
    )
    min_verifications: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Minimum verification steps required",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept answer",
    )
    trap_detection: bool = Field(
        default=True,
        description="Enable adversarial trap detection",
    )
    allow_uncertain: bool = Field(
        default=True,
        description="Allow 'I don't know' responses",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    reasoning_timeout_ms: int = Field(
        default=60000,
        ge=5000,
        le=300000,
        description="Timeout for reasoning process",
    )


class OllamaLLMAdapter:
    """Adapter to use Ollama with the reasoning engine."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = OllamaClient(base_url=base_url)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate response from Ollama."""
        response = await self._client.generate(
            model=self._model,
            prompt=prompt,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.response


@NodeRegistry.register(NodeType.REASONING)
class ReasoningNode(BaseNode):
    """Node that performs chain-of-thought reasoning.

    This node wraps the reasoning engine to provide:
    - Step-by-step reasoning
    - Adversarial trap detection
    - Verification of claims
    - Appropriate uncertainty handling
    """

    def __init__(self, definition: NodeDefinition) -> None:
        """Initialize reasoning node."""
        super().__init__(definition)
        self._reasoning_config = ReasoningNodeConfig(**definition.config)
        self._trap_detector = TrapDetector()
        self._engine: ReasoningEngine | None = None

    @property
    def reasoning_config(self) -> ReasoningNodeConfig:
        """Get reasoning-specific configuration."""
        return self._reasoning_config

    def _get_engine(self) -> ReasoningEngine:
        """Get or create reasoning engine."""
        if self._engine is None:
            llm = OllamaLLMAdapter(model=self._reasoning_config.model)
            engine_config = EngineConfig(
                max_steps=self._reasoning_config.max_steps,
                max_depth=self._reasoning_config.max_depth,
                timeout_ms=self._reasoning_config.reasoning_timeout_ms,
                require_verification=self._reasoning_config.require_verification,
                min_verifications=self._reasoning_config.min_verifications,
                min_confidence=self._reasoning_config.min_confidence,
                allow_uncertain=self._reasoning_config.allow_uncertain,
                trap_detection=self._reasoning_config.trap_detection,
                temperature=self._reasoning_config.temperature,
            )
            self._engine = ReasoningEngine(llm, engine_config)
        return self._engine

    async def execute(
        self,
        message: Message,
        _context: ExecutionContext,  # noqa: ARG002
    ) -> NodeResult:
        """Execute reasoning on the input message.

        Args:
            message: Input message with query
            context: Execution context

        Returns:
            NodeResult with reasoned response
        """
        start_time = time.time()

        # Extract query from message
        query = message.payload.task or message.payload.content or ""
        if not query:
            return NodeResult.failure_result(
                error="No query provided for reasoning",
                latency_ms=0,
            )

        try:
            # Quick trap detection for early warning
            detected_trap = self._trap_detector.detect(query)
            trap_warning = ""
            if detected_trap != TrapType.NONE:
                trap_warning = f"[Detected: {detected_trap.value}] "

            # Execute reasoning
            engine = self._get_engine()
            chain = await engine.reason(query)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract result
            if chain.status == ChainStatus.COMPLETED and chain.conclusion:
                conclusion = chain.conclusion
                response_content = conclusion.answer

                # Add caveats if present
                if conclusion.caveats:
                    response_content += "\n\nCaveats:\n" + "\n".join(
                        f"- {c}" for c in conclusion.caveats
                    )

                # Add uncertainty info if applicable
                if conclusion.is_uncertain:
                    response_content = (
                        f"[Uncertain: {conclusion.uncertainty_reason}]\n\n"
                        + response_content
                    )

                # Add trap warning if detected
                if trap_warning:
                    response_content = trap_warning + response_content

                # Build output payload
                output_payload = MessagePayload(
                    task=message.payload.task,
                    content=response_content,
                    metadata={
                        **message.payload.metadata,
                        "reasoning_chain_id": chain.id,
                        "reasoning_steps": len(chain.steps),
                        "reasoning_confidence": chain.total_confidence,
                        "reasoning_traps_detected": [
                            t.value for t in chain.detected_trap_types
                        ],
                        "reasoning_verified": chain.verification_count > 0,
                    },
                )

                output_message = message.create_child(
                    source_node=self.id,
                    payload=output_payload,
                )

                return NodeResult.success_result(
                    output_messages=[output_message],
                    next_nodes=[],
                    latency_ms=latency_ms,
                    metadata={
                        "chain_id": chain.id,
                        "steps": len(chain.steps),
                        "confidence": chain.total_confidence,
                        "traps": [t.value for t in chain.detected_trap_types],
                    },
                )

            else:
                # Reasoning failed or incomplete
                error_msg = "Reasoning did not complete successfully"
                if chain.steps:
                    last_step = chain.steps[-1]
                    error_msg = f"Reasoning stopped at step {len(chain.steps)}: {last_step.content[:100]}"

                return NodeResult.failure_result(
                    error=error_msg,
                    latency_ms=latency_ms,
                    metadata={
                        "chain_id": chain.id,
                        "status": chain.status.value,
                        "steps": len(chain.steps),
                    },
                )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return NodeResult.failure_result(
                error=f"Reasoning execution failed: {e!s}",
                latency_ms=latency_ms,
            )


__all__ = [
    "ReasoningNodeConfig",
    "ReasoningNode",
]
