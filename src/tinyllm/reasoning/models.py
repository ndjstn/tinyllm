"""
Strict Pydantic models for Chain-of-Thought reasoning.

All models use:
- strict=True for type coercion prevention
- frozen=True where immutability is needed
- Comprehensive field validators
- Cross-field model validators
- Discriminated unions for polymorphism
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# =============================================================================
# CONSTANTS AND PATTERNS
# =============================================================================

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
STEP_ID_PATTERN = re.compile(r"^step_[a-f0-9]{8}$")


def generate_step_id() -> str:
    """Generate a valid step ID."""
    return f"step_{uuid.uuid4().hex[:8]}"


# =============================================================================
# ENUMS - Strict string enums
# =============================================================================


class StepType(str, Enum):
    """Types of reasoning steps."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    VERIFICATION = "verification"
    CONCLUSION = "conclusion"


class ReasoningType(str, Enum):
    """Types of reasoning within a thought step."""

    ANALYSIS = "analysis"  # Breaking down the problem
    DECOMPOSITION = "decomposition"  # Splitting into sub-problems
    CRITIQUE = "critique"  # Self-criticism
    HYPOTHESIS = "hypothesis"  # Proposing solutions
    COMPARISON = "comparison"  # Comparing alternatives
    SYNTHESIS = "synthesis"  # Combining insights


class VerificationVerdict(str, Enum):
    """Outcome of a verification step."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    NEEDS_MORE_INFO = "needs_more_info"


class ChainStatus(str, Enum):
    """Status of a reasoning chain."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_VERIFICATION = "needs_verification"
    TIMEOUT = "timeout"


class TrapType(str, Enum):
    """Types of adversarial traps detected."""

    FALSE_PREMISE = "false_premise"
    TRICK_QUESTION = "trick_question"
    HALLUCINATION_BAIT = "hallucination_bait"
    IMPOSSIBLE_TASK = "impossible_task"
    AMBIGUOUS_QUERY = "ambiguous_query"
    NONE = "none"


# =============================================================================
# BASE STEP MODEL
# =============================================================================


class BaseStep(BaseModel):
    """
    Base class for all reasoning steps.

    Immutable after creation. All steps have:
    - Unique ID following step_XXXXXXXX pattern
    - Type discriminator
    - Content (non-empty)
    - Confidence score [0, 1]
    - Timestamp in UTC
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        ser_json_timedelta="iso8601",
    )

    id: Annotated[
        str,
        Field(
            default_factory=generate_step_id,
            pattern=r"^step_[a-f0-9]{8}$",
            description="Unique step identifier",
        ),
    ]

    type: StepType = Field(..., description="Step type discriminator")

    content: Annotated[
        str,
        Field(
            min_length=1,
            max_length=10000,
            description="Step content (non-empty)",
        ),
    ]

    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence score [0, 1]",
        ),
    ] = 0.5

    timestamp: Annotated[
        datetime,
        Field(
            default_factory=lambda: datetime.now(UTC),
            description="UTC timestamp",
        ),
    ]

    @field_validator("content", mode="before")
    @classmethod
    def strip_and_validate_content(cls, v: Any) -> str:
        """Strip whitespace and ensure non-empty."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Content cannot be empty or whitespace-only")
        return v

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamp is UTC."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v.astimezone(UTC)
        return v


# =============================================================================
# SPECIALIZED STEP TYPES
# =============================================================================


class ThoughtStep(BaseStep):
    """
    Internal reasoning step.

    Used for analysis, decomposition, critique, etc.
    """

    type: Literal[StepType.THOUGHT] = StepType.THOUGHT

    reasoning_type: ReasoningType = Field(
        ...,
        description="Type of reasoning being performed",
    )

    parent_step_id: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^step_[a-f0-9]{8}$",
            description="Parent step for hierarchical reasoning",
        ),
    ] = None

    detected_trap: TrapType = Field(
        default=TrapType.NONE,
        description="Adversarial trap detected in this thought",
    )

    key_insights: Annotated[
        list[str],
        Field(
            max_length=10,
            description="Key insights from this thought",
        ),
    ] = Field(default_factory=list)

    @field_validator("key_insights", mode="before")
    @classmethod
    def validate_insights(cls, v: Any) -> list[str]:
        """Ensure insights are non-empty strings."""
        if not isinstance(v, list):
            return []
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]


class ActionStep(BaseStep):
    """
    Action step - calling a tool or making a sub-query.
    """

    type: Literal[StepType.ACTION] = StepType.ACTION

    tool_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            pattern=r"^[a-z][a-z0-9_]*$",
            description="Tool to invoke",
        ),
    ]

    tool_input: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Input parameters for the tool",
        ),
    ]

    expected_outcome: Annotated[
        str,
        Field(
            min_length=1,
            max_length=500,
            description="What we expect from this action",
        ),
    ]

    timeout_ms: Annotated[
        int,
        Field(
            ge=100,
            le=300000,
            description="Timeout in milliseconds",
        ),
    ] = 30000

    @model_validator(mode="after")
    def validate_tool_input_keys(self) -> ActionStep:
        """Ensure tool input keys are valid identifiers."""
        for key in self.tool_input:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid tool input key: {key}")
        return self


class ObservationStep(BaseStep):
    """
    Observation step - result from an action.
    """

    type: Literal[StepType.OBSERVATION] = StepType.OBSERVATION

    source_action_id: Annotated[
        str,
        Field(
            pattern=r"^step_[a-f0-9]{8}$",
            description="ID of the action that produced this observation",
        ),
    ]

    raw_output: Annotated[
        str,
        Field(
            max_length=50000,
            description="Raw output from the action",
        ),
    ] = ""

    parsed_output: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Structured parsed output",
        ),
    ] = None

    success: bool = Field(
        ...,
        description="Whether the action succeeded",
    )

    error_message: Annotated[
        str | None,
        Field(
            default=None,
            max_length=1000,
            description="Error message if action failed",
        ),
    ] = None

    @model_validator(mode="after")
    def validate_error_on_failure(self) -> ObservationStep:
        """Ensure error_message is present if success is False."""
        if not self.success and not self.error_message:
            raise ValueError("error_message required when success is False")
        return self


class VerificationStep(BaseStep):
    """
    Verification step - checking a claim or result.

    Critical for adversarial defense.
    """

    type: Literal[StepType.VERIFICATION] = StepType.VERIFICATION

    claim: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description="The claim being verified",
        ),
    ]

    evidence: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=10,
            description="Evidence supporting/refuting the claim",
        ),
    ]

    verdict: VerificationVerdict = Field(
        ...,
        description="Verification outcome",
    )

    reasoning: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description="Reasoning behind the verdict",
        ),
    ]

    source_step_ids: Annotated[
        list[str],
        Field(
            max_length=10,
            description="IDs of steps that led to this verification",
        ),
    ] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def validate_evidence(cls, v: Any) -> list[str]:
        """Ensure evidence items are non-empty."""
        if not isinstance(v, list):
            raise ValueError("Evidence must be a list")
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("At least one evidence item required")
        return cleaned

    @field_validator("source_step_ids", mode="before")
    @classmethod
    def validate_source_step_ids(cls, v: Any) -> list[str]:
        """Validate source step ID patterns."""
        if not isinstance(v, list):
            return []
        for step_id in v:
            if not STEP_ID_PATTERN.match(step_id):
                raise ValueError(f"Invalid step ID pattern: {step_id}")
        return v


class ConclusionStep(BaseStep):
    """
    Conclusion step - final answer with caveats.
    """

    type: Literal[StepType.CONCLUSION] = StepType.CONCLUSION

    answer: Annotated[
        str,
        Field(
            min_length=1,
            max_length=10000,
            description="The final answer",
        ),
    ]

    caveats: Annotated[
        list[str],
        Field(
            max_length=10,
            description="Important caveats or limitations",
        ),
    ] = Field(default_factory=list)

    is_uncertain: bool = Field(
        default=False,
        description="Whether the answer is uncertain",
    )

    uncertainty_reason: Annotated[
        str | None,
        Field(
            default=None,
            max_length=500,
            description="Reason for uncertainty",
        ),
    ] = None

    alternative_interpretations: Annotated[
        list[str],
        Field(
            max_length=5,
            description="Alternative valid interpretations",
        ),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_uncertainty(self) -> ConclusionStep:
        """Ensure uncertainty_reason is present if is_uncertain is True."""
        if self.is_uncertain and not self.uncertainty_reason:
            raise ValueError("uncertainty_reason required when is_uncertain is True")
        return self

    @field_validator("caveats", "alternative_interpretations", mode="before")
    @classmethod
    def clean_string_lists(cls, v: Any) -> list[str]:
        """Clean and validate string lists."""
        if not isinstance(v, list):
            return []
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]


# =============================================================================
# DISCRIMINATED UNION
# =============================================================================

# Discriminated union for all step types
ReasoningStepUnion = Annotated[
    ThoughtStep | ActionStep | ObservationStep | VerificationStep | ConclusionStep,
    Field(discriminator="type"),
]


# =============================================================================
# REASONING CHAIN
# =============================================================================


class ReasoningChain(BaseModel):
    """
    Container for a sequence of reasoning steps.

    Validates:
    - Non-empty chain
    - Valid step sequence
    - Unique step IDs
    - Proper action-observation pairing
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: Annotated[
        str,
        Field(
            default_factory=lambda: f"chain_{uuid.uuid4().hex[:12]}",
            pattern=r"^chain_[a-f0-9]{12}$",
            description="Unique chain identifier",
        ),
    ]

    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=5000,
            description="Original query being reasoned about",
        ),
    ]

    steps: Annotated[
        list[ReasoningStepUnion],
        Field(
            min_length=1,
            description="Ordered sequence of reasoning steps",
        ),
    ]

    status: ChainStatus = Field(
        default=ChainStatus.IN_PROGRESS,
        description="Current chain status",
    )

    detected_traps: Annotated[
        list[TrapType],
        Field(
            description="All traps detected during reasoning",
        ),
    ] = Field(default_factory=list)

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When reasoning started",
    )

    completed_at: datetime | None = Field(
        default=None,
        description="When reasoning completed",
    )

    total_confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Overall confidence in the chain",
        ),
    ] = 0.5

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, v: Any) -> str:
        """Strip and validate query."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Query cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_chain_integrity(self) -> ReasoningChain:
        """Validate chain integrity."""
        # Check for unique step IDs
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Duplicate step IDs in chain")

        # Check action-observation pairing
        action_ids = {
            step.id for step in self.steps if step.type == StepType.ACTION
        }
        observation_sources = {
            step.source_action_id
            for step in self.steps
            if step.type == StepType.OBSERVATION
        }

        # Every observation must reference a valid action
        invalid_refs = observation_sources - action_ids
        if invalid_refs:
            raise ValueError(f"Observations reference non-existent actions: {invalid_refs}")

        # Validate completion state
        if self.status == ChainStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed_at required when status is COMPLETED")

        # Must have conclusion if completed
        if self.status == ChainStatus.COMPLETED:
            has_conclusion = any(
                step.type == StepType.CONCLUSION for step in self.steps
            )
            if not has_conclusion:
                raise ValueError("Completed chain must have a conclusion step")

        return self

    def add_step(self, step: ReasoningStepUnion) -> ReasoningChain:
        """Add a step to the chain (returns new chain - immutable pattern)."""
        new_steps = list(self.steps) + [step]

        # Collect traps from thought steps
        new_traps = list(self.detected_traps)
        if (
            isinstance(step, ThoughtStep)
            and step.detected_trap != TrapType.NONE
            and step.detected_trap not in new_traps
        ):
            new_traps.append(step.detected_trap)

        # Update status if conclusion
        new_status = self.status
        completed_at = self.completed_at
        if isinstance(step, ConclusionStep):
            new_status = ChainStatus.COMPLETED
            completed_at = datetime.now(UTC)

        return ReasoningChain(
            id=self.id,
            query=self.query,
            steps=new_steps,
            status=new_status,
            detected_traps=new_traps,
            started_at=self.started_at,
            completed_at=completed_at,
            total_confidence=self._calculate_confidence(new_steps),
        )

    def _calculate_confidence(self, steps: list[ReasoningStepUnion]) -> float:
        """Calculate overall chain confidence."""
        if not steps:
            return 0.0

        # Weighted average with emphasis on verification and conclusion
        weights = {
            StepType.THOUGHT: 1.0,
            StepType.ACTION: 0.5,
            StepType.OBSERVATION: 0.5,
            StepType.VERIFICATION: 2.0,
            StepType.CONCLUSION: 3.0,
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for step in steps:
            w = weights.get(step.type, 1.0)
            weighted_confidence += step.confidence * w
            total_weight += w

        return weighted_confidence / total_weight if total_weight > 0 else 0.5

    @property
    def has_conclusion(self) -> bool:
        """Check if chain has a conclusion."""
        return any(step.type == StepType.CONCLUSION for step in self.steps)

    @property
    def conclusion(self) -> ConclusionStep | None:
        """Get the conclusion step if present."""
        for step in reversed(self.steps):
            if isinstance(step, ConclusionStep):
                return step
        return None

    @property
    def verification_count(self) -> int:
        """Count verification steps."""
        return sum(1 for step in self.steps if step.type == StepType.VERIFICATION)

    @property
    def detected_trap_types(self) -> set[TrapType]:
        """Get all unique trap types detected."""
        return set(self.detected_traps) - {TrapType.NONE}


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class ReasoningConfig(BaseModel):
    """Configuration for reasoning behavior."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    enabled: bool = Field(
        default=True,
        description="Whether reasoning is enabled",
    )

    max_steps: Annotated[
        int,
        Field(
            ge=1,
            le=50,
            description="Maximum reasoning steps",
        ),
    ] = 10

    max_depth: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            description="Maximum reasoning depth",
        ),
    ] = 3

    require_verification: bool = Field(
        default=True,
        description="Require verification before conclusion",
    )

    min_confidence_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum confidence to accept answer",
        ),
    ] = 0.6

    timeout_ms: Annotated[
        int,
        Field(
            ge=1000,
            le=300000,
            description="Total reasoning timeout",
        ),
    ] = 60000

    trap_detection_enabled: bool = Field(
        default=True,
        description="Enable adversarial trap detection",
    )

    allow_uncertain_answers: bool = Field(
        default=True,
        description="Allow 'I don't know' responses",
    )


# =============================================================================
# SOLUTION PATTERN (for memory)
# =============================================================================


class SolutionPattern(BaseModel):
    """
    A learned pattern for solving similar problems.

    Used for solution memory.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    id: Annotated[
        str,
        Field(
            default_factory=lambda: f"pattern_{uuid.uuid4().hex[:12]}",
            pattern=r"^pattern_[a-f0-9]{12}$",
            description="Unique pattern identifier",
        ),
    ]

    query_pattern: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description="Pattern/template for matching queries",
        ),
    ]

    category: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="Problem category",
        ),
    ]

    successful_chain_id: Annotated[
        str,
        Field(
            pattern=r"^chain_[a-f0-9]{12}$",
            description="ID of the successful reasoning chain",
        ),
    ]

    key_steps: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=20,
            description="Key steps that led to success",
        ),
    ]

    traps_avoided: Annotated[
        list[TrapType],
        Field(
            description="Traps that were successfully avoided",
        ),
    ] = Field(default_factory=list)

    success_count: Annotated[
        int,
        Field(
            ge=1,
            description="Number of times this pattern succeeded",
        ),
    ] = 1

    failure_count: Annotated[
        int,
        Field(
            ge=0,
            description="Number of times this pattern failed",
        ),
    ] = 0

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    last_used_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def confidence(self) -> float:
        """Calculate pattern confidence based on usage."""
        # More usage = more confidence, capped at 0.95
        total = self.success_count + self.failure_count
        usage_factor = min(total / 100, 1.0)
        return min(0.5 + (self.success_rate * 0.45 * usage_factor), 0.95)
