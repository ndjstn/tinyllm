"""
Reasoning chain manager with state machine logic.

Handles step-by-step reasoning execution, validation, and state transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from tinyllm.reasoning.models import (
    ActionStep,
    ChainStatus,
    ConclusionStep,
    ObservationStep,
    ReasoningChain,
    ReasoningStepUnion,
    ReasoningType,
    ThoughtStep,
    TrapType,
    VerificationStep,
    VerificationVerdict,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# STATE MACHINE
# =============================================================================


class ReasoningState(str, Enum):
    """States in the reasoning state machine."""

    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    DECOMPOSING = "decomposing"
    EXECUTING_ACTION = "executing_action"
    OBSERVING = "observing"
    VERIFYING = "verifying"
    SYNTHESIZING = "synthesizing"
    CONCLUDING = "concluding"
    FAILED = "failed"
    COMPLETED = "completed"


class StateTransition(BaseModel):
    """A state transition record."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    from_state: ReasoningState
    to_state: ReasoningState
    trigger: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Valid state transitions
VALID_TRANSITIONS: dict[ReasoningState, set[ReasoningState]] = {
    ReasoningState.INITIALIZING: {
        ReasoningState.ANALYZING,
        ReasoningState.FAILED,
    },
    ReasoningState.ANALYZING: {
        ReasoningState.DECOMPOSING,
        ReasoningState.EXECUTING_ACTION,
        ReasoningState.VERIFYING,
        ReasoningState.SYNTHESIZING,
        ReasoningState.CONCLUDING,
        ReasoningState.FAILED,
    },
    ReasoningState.DECOMPOSING: {
        ReasoningState.ANALYZING,
        ReasoningState.EXECUTING_ACTION,
        ReasoningState.SYNTHESIZING,
        ReasoningState.VERIFYING,
        ReasoningState.FAILED,
    },
    ReasoningState.EXECUTING_ACTION: {
        ReasoningState.OBSERVING,
        ReasoningState.FAILED,
    },
    ReasoningState.OBSERVING: {
        ReasoningState.ANALYZING,
        ReasoningState.VERIFYING,
        ReasoningState.SYNTHESIZING,
        ReasoningState.FAILED,
    },
    ReasoningState.VERIFYING: {
        ReasoningState.ANALYZING,
        ReasoningState.SYNTHESIZING,
        ReasoningState.VERIFYING,  # Can do multiple verifications
        ReasoningState.CONCLUDING,
        ReasoningState.FAILED,
    },
    ReasoningState.SYNTHESIZING: {
        ReasoningState.VERIFYING,
        ReasoningState.CONCLUDING,
        ReasoningState.FAILED,
    },
    ReasoningState.CONCLUDING: {
        ReasoningState.COMPLETED,
        ReasoningState.VERIFYING,  # Can go back for more verification
        ReasoningState.FAILED,
    },
    ReasoningState.FAILED: set(),  # Terminal state
    ReasoningState.COMPLETED: set(),  # Terminal state
}


# =============================================================================
# CHAIN MANAGER
# =============================================================================


class ChainManagerConfig(BaseModel):
    """Configuration for the chain manager."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    max_steps: Annotated[int, Field(ge=1, le=100)] = 20
    max_depth: Annotated[int, Field(ge=1, le=10)] = 5
    timeout_ms: Annotated[int, Field(ge=1000, le=600000)] = 120000
    require_verification: bool = True
    min_verifications: Annotated[int, Field(ge=0, le=10)] = 1
    min_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6
    allow_uncertain: bool = True
    trap_detection: bool = True


class ChainManager:
    """
    Manages reasoning chain execution with state machine logic.

    Responsibilities:
    - Track state transitions
    - Validate step sequences
    - Enforce constraints (depth, steps, timeout)
    - Coordinate trap detection
    """

    def __init__(
        self,
        config: ChainManagerConfig | None = None,
    ) -> None:
        self._config = config or ChainManagerConfig()
        self._state = ReasoningState.INITIALIZING
        self._chain: ReasoningChain | None = None
        self._transitions: list[StateTransition] = []
        self._start_time: datetime | None = None
        self._current_depth = 0

    @property
    def config(self) -> ChainManagerConfig:
        """Get configuration."""
        return self._config

    @property
    def state(self) -> ReasoningState:
        """Get current state."""
        return self._state

    @property
    def chain(self) -> ReasoningChain | None:
        """Get current chain."""
        return self._chain

    @property
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self._state in {ReasoningState.COMPLETED, ReasoningState.FAILED}

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self._start_time is None:
            return 0
        elapsed = datetime.now(UTC) - self._start_time
        return int(elapsed.total_seconds() * 1000)

    @property
    def remaining_steps(self) -> int:
        """Get remaining step budget."""
        if self._chain is None:
            return self._config.max_steps
        return max(0, self._config.max_steps - len(self._chain.steps))

    def initialize(self, query: str) -> ReasoningChain:
        """
        Initialize a new reasoning chain.

        Args:
            query: The query to reason about

        Returns:
            New ReasoningChain instance

        Raises:
            ValueError: If already initialized
        """
        if self._chain is not None:
            raise ValueError("Chain already initialized - create new manager")

        self._start_time = datetime.now(UTC)

        # Create initial thought step
        initial_thought = ThoughtStep(
            content=f"Analyzing query: {query}",
            confidence=0.5,
            reasoning_type=ReasoningType.ANALYSIS,
            detected_trap=TrapType.NONE,
        )

        self._chain = ReasoningChain(
            query=query,
            steps=[initial_thought],
            status=ChainStatus.IN_PROGRESS,
        )

        self._transition_to(ReasoningState.ANALYZING, "initialized")
        return self._chain

    def _transition_to(self, new_state: ReasoningState, trigger: str) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            trigger: What caused the transition

        Raises:
            ValueError: If transition is invalid
        """
        if new_state not in VALID_TRANSITIONS.get(self._state, set()):
            raise ValueError(
                f"Invalid transition: {self._state} -> {new_state} (trigger: {trigger})"
            )

        transition = StateTransition(
            from_state=self._state,
            to_state=new_state,
            trigger=trigger,
        )
        self._transitions.append(transition)
        self._state = new_state

    def add_thought(
        self,
        content: str,
        reasoning_type: ReasoningType,
        confidence: float = 0.5,
        detected_trap: TrapType = TrapType.NONE,
        key_insights: list[str] | None = None,
        parent_step_id: str | None = None,
    ) -> ReasoningChain:
        """
        Add a thought step to the chain.

        Returns:
            Updated chain
        """
        self._check_can_add_step()

        step = ThoughtStep(
            content=content,
            confidence=confidence,
            reasoning_type=reasoning_type,
            detected_trap=detected_trap,
            key_insights=key_insights or [],
            parent_step_id=parent_step_id,
        )

        self._chain = self._chain.add_step(step)

        # Update depth tracking
        if parent_step_id is not None:
            self._current_depth = min(self._current_depth + 1, self._config.max_depth)

        # State transition based on reasoning type
        state_map = {
            ReasoningType.ANALYSIS: ReasoningState.ANALYZING,
            ReasoningType.DECOMPOSITION: ReasoningState.DECOMPOSING,
            ReasoningType.CRITIQUE: ReasoningState.VERIFYING,
            ReasoningType.HYPOTHESIS: ReasoningState.ANALYZING,
            ReasoningType.COMPARISON: ReasoningState.ANALYZING,
            ReasoningType.SYNTHESIS: ReasoningState.SYNTHESIZING,
        }

        target_state = state_map.get(reasoning_type, ReasoningState.ANALYZING)
        if self._state != target_state:
            self._transition_to(target_state, f"thought:{reasoning_type.value}")

        return self._chain

    def add_action(
        self,
        content: str,
        tool_name: str,
        tool_input: dict[str, Any],
        expected_outcome: str,
        confidence: float = 0.5,
        timeout_ms: int = 30000,
    ) -> ReasoningChain:
        """
        Add an action step to the chain.

        Returns:
            Updated chain
        """
        self._check_can_add_step()

        step = ActionStep(
            content=content,
            confidence=confidence,
            tool_name=tool_name,
            tool_input=tool_input,
            expected_outcome=expected_outcome,
            timeout_ms=timeout_ms,
        )

        self._chain = self._chain.add_step(step)
        self._transition_to(ReasoningState.EXECUTING_ACTION, f"action:{tool_name}")
        return self._chain

    def add_observation(
        self,
        content: str,
        source_action_id: str,
        success: bool,
        confidence: float = 0.5,
        raw_output: str = "",
        parsed_output: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> ReasoningChain:
        """
        Add an observation step to the chain.

        Returns:
            Updated chain
        """
        self._check_can_add_step()

        step = ObservationStep(
            content=content,
            confidence=confidence,
            source_action_id=source_action_id,
            success=success,
            raw_output=raw_output,
            parsed_output=parsed_output,
            error_message=error_message,
        )

        self._chain = self._chain.add_step(step)
        self._transition_to(ReasoningState.OBSERVING, "observation")
        return self._chain

    def add_verification(
        self,
        content: str,
        claim: str,
        evidence: list[str],
        verdict: VerificationVerdict,
        reasoning: str,
        confidence: float = 0.5,
        source_step_ids: list[str] | None = None,
    ) -> ReasoningChain:
        """
        Add a verification step to the chain.

        Returns:
            Updated chain
        """
        self._check_can_add_step()

        step = VerificationStep(
            content=content,
            confidence=confidence,
            claim=claim,
            evidence=evidence,
            verdict=verdict,
            reasoning=reasoning,
            source_step_ids=source_step_ids or [],
        )

        self._chain = self._chain.add_step(step)
        self._transition_to(ReasoningState.VERIFYING, f"verification:{verdict.value}")
        return self._chain

    def conclude(
        self,
        content: str,
        answer: str,
        confidence: float = 0.5,
        caveats: list[str] | None = None,
        is_uncertain: bool = False,
        uncertainty_reason: str | None = None,
        alternative_interpretations: list[str] | None = None,
    ) -> ReasoningChain:
        """
        Add a conclusion step and complete the chain.

        Returns:
            Completed chain

        Raises:
            ValueError: If verification requirements not met
        """
        self._check_can_add_step()

        # Check verification requirements
        if self._config.require_verification:
            verification_count = self._chain.verification_count
            if verification_count < self._config.min_verifications:
                raise ValueError(
                    f"Insufficient verifications: {verification_count} < {self._config.min_verifications}"
                )

        # Check confidence threshold
        if confidence < self._config.min_confidence and not is_uncertain:
            if not self._config.allow_uncertain:
                raise ValueError(
                    f"Confidence {confidence} below threshold {self._config.min_confidence}"
                )
            is_uncertain = True
            uncertainty_reason = uncertainty_reason or "Low confidence score"

        step = ConclusionStep(
            content=content,
            confidence=confidence,
            answer=answer,
            caveats=caveats or [],
            is_uncertain=is_uncertain,
            uncertainty_reason=uncertainty_reason,
            alternative_interpretations=alternative_interpretations or [],
        )

        self._chain = self._chain.add_step(step)
        self._transition_to(ReasoningState.CONCLUDING, "conclusion")
        self._transition_to(ReasoningState.COMPLETED, "finalized")
        return self._chain

    def fail(self, reason: str) -> ReasoningChain:
        """
        Mark the chain as failed.

        Returns:
            Failed chain
        """
        if self._chain is None:
            raise ValueError("No chain to fail")

        # Add a thought explaining the failure
        failure_thought = ThoughtStep(
            content=f"Reasoning failed: {reason}",
            confidence=0.0,
            reasoning_type=ReasoningType.CRITIQUE,
            detected_trap=TrapType.NONE,
        )

        self._chain = self._chain.add_step(failure_thought)

        # Update chain status
        self._chain = ReasoningChain(
            id=self._chain.id,
            query=self._chain.query,
            steps=list(self._chain.steps),
            status=ChainStatus.FAILED,
            detected_traps=list(self._chain.detected_traps),
            started_at=self._chain.started_at,
            completed_at=datetime.now(UTC),
            total_confidence=0.0,
        )

        self._transition_to(ReasoningState.FAILED, f"failure:{reason}")
        return self._chain

    def _check_can_add_step(self) -> None:
        """
        Check if we can add another step.

        Raises:
            ValueError: If constraints violated
        """
        if self._chain is None:
            raise ValueError("Chain not initialized - call initialize() first")

        if self.is_terminal:
            raise ValueError(f"Cannot add step - chain is {self._state}")

        if self.remaining_steps <= 0:
            raise ValueError(f"Step limit reached: {self._config.max_steps}")

        if self.elapsed_ms > self._config.timeout_ms:
            raise ValueError(f"Timeout exceeded: {self.elapsed_ms}ms > {self._config.timeout_ms}ms")

        if self._current_depth > self._config.max_depth:
            raise ValueError(f"Max depth exceeded: {self._current_depth} > {self._config.max_depth}")

    def get_step_by_id(self, step_id: str) -> ReasoningStepUnion | None:
        """Get a step by its ID."""
        if self._chain is None:
            return None
        for step in self._chain.steps:
            if step.id == step_id:
                return step
        return None

    def get_last_action(self) -> ActionStep | None:
        """Get the most recent action step."""
        if self._chain is None:
            return None
        for step in reversed(self._chain.steps):
            if isinstance(step, ActionStep):
                return step
        return None

    def get_transitions_log(self) -> list[dict[str, Any]]:
        """Get state transition history."""
        return [
            {
                "from": t.from_state.value,
                "to": t.to_state.value,
                "trigger": t.trigger,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in self._transitions
        ]


# =============================================================================
# CHAIN BUILDER (Fluent API)
# =============================================================================


class ChainBuilder:
    """
    Fluent API for building reasoning chains.

    Example:
        chain = (
            ChainBuilder(query="What is 2+2?")
            .think("Breaking down the problem", ReasoningType.ANALYSIS)
            .verify("2+2=4", ["Basic arithmetic"], VerificationVerdict.VERIFIED, "Correct")
            .conclude("4", answer="4")
            .build()
        )
    """

    def __init__(self, query: str, config: ChainManagerConfig | None = None) -> None:
        self._manager = ChainManager(config)
        self._manager.initialize(query)

    def think(
        self,
        content: str,
        reasoning_type: ReasoningType = ReasoningType.ANALYSIS,
        confidence: float = 0.5,
        detected_trap: TrapType = TrapType.NONE,
        key_insights: list[str] | None = None,
    ) -> ChainBuilder:
        """Add a thought step."""
        self._manager.add_thought(
            content=content,
            reasoning_type=reasoning_type,
            confidence=confidence,
            detected_trap=detected_trap,
            key_insights=key_insights,
        )
        return self

    def act(
        self,
        content: str,
        tool_name: str,
        tool_input: dict[str, Any],
        expected_outcome: str,
        confidence: float = 0.5,
    ) -> ChainBuilder:
        """Add an action step."""
        self._manager.add_action(
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            expected_outcome=expected_outcome,
            confidence=confidence,
        )
        return self

    def observe(
        self,
        content: str,
        success: bool,
        raw_output: str = "",
        error_message: str | None = None,
        confidence: float = 0.5,
    ) -> ChainBuilder:
        """Add an observation step."""
        action = self._manager.get_last_action()
        if action is None:
            raise ValueError("Cannot observe without a preceding action")

        self._manager.add_observation(
            content=content,
            source_action_id=action.id,
            success=success,
            raw_output=raw_output,
            error_message=error_message,
            confidence=confidence,
        )
        return self

    def verify(
        self,
        claim: str,
        evidence: list[str],
        verdict: VerificationVerdict,
        reasoning: str,
        content: str | None = None,
        confidence: float = 0.5,
    ) -> ChainBuilder:
        """Add a verification step."""
        self._manager.add_verification(
            content=content or f"Verifying: {claim}",
            claim=claim,
            evidence=evidence,
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
        )
        return self

    def conclude(
        self,
        content: str,
        answer: str,
        confidence: float = 0.5,
        caveats: list[str] | None = None,
        is_uncertain: bool = False,
        uncertainty_reason: str | None = None,
    ) -> ChainBuilder:
        """Add conclusion and complete chain."""
        self._manager.conclude(
            content=content,
            answer=answer,
            confidence=confidence,
            caveats=caveats,
            is_uncertain=is_uncertain,
            uncertainty_reason=uncertainty_reason,
        )
        return self

    def build(self) -> ReasoningChain:
        """Get the built chain."""
        if self._manager.chain is None:
            raise ValueError("Chain not initialized")
        return self._manager.chain

    @property
    def manager(self) -> ChainManager:
        """Get the underlying manager."""
        return self._manager
