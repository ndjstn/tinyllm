"""
Reasoning engine - orchestrates chain-of-thought reasoning.

The engine coordinates:
- LLM calls for generating reasoning steps
- Tool execution for actions
- Trap detection for adversarial defense
- Pattern matching for solution memory
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from tinyllm.reasoning.chain import ChainManager, ChainManagerConfig
from tinyllm.reasoning.models import (
    ChainStatus,
    ReasoningChain,
    ReasoningType,
    SolutionPattern,
    TrapType,
    VerificationVerdict,
)
from tinyllm.reasoning.prompts import (
    ADVERSARIAL_DEFENSE_PROMPT,
    REASONING_SYSTEM_PROMPT,
    TrapDetector,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# PROTOCOLS
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the LLM."""
        ...


class ToolExecutor(Protocol):
    """Protocol for tool executors."""

    async def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        timeout_ms: int = 30000,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """
        Execute a tool.

        Returns:
            Tuple of (success, raw_output, parsed_output)
        """
        ...


class PatternStore(Protocol):
    """Protocol for solution pattern storage."""

    async def find_matching_patterns(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SolutionPattern]:
        """Find patterns matching the query."""
        ...

    async def save_pattern(self, pattern: SolutionPattern) -> None:
        """Save a new pattern."""
        ...

    async def update_pattern_stats(
        self,
        pattern_id: str,
        success: bool,
    ) -> None:
        """Update pattern success/failure stats."""
        ...


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================


class EngineConfig(BaseModel):
    """Configuration for the reasoning engine."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    # Chain configuration
    max_steps: Annotated[int, Field(ge=1, le=100)] = 20
    max_depth: Annotated[int, Field(ge=1, le=10)] = 5
    timeout_ms: Annotated[int, Field(ge=1000, le=600000)] = 120000

    # Verification settings
    require_verification: bool = True
    min_verifications: Annotated[int, Field(ge=0, le=10)] = 1
    min_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6

    # Behavior settings
    allow_uncertain: bool = True
    trap_detection: bool = True
    use_patterns: bool = True

    # LLM settings
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    max_tokens: Annotated[int, Field(ge=100, le=8192)] = 2048

    def to_chain_config(self) -> ChainManagerConfig:
        """Convert to ChainManagerConfig."""
        return ChainManagerConfig(
            max_steps=self.max_steps,
            max_depth=self.max_depth,
            timeout_ms=self.timeout_ms,
            require_verification=self.require_verification,
            min_verifications=self.min_verifications,
            min_confidence=self.min_confidence,
            allow_uncertain=self.allow_uncertain,
            trap_detection=self.trap_detection,
        )


# =============================================================================
# REASONING ENGINE
# =============================================================================


class ReasoningEngine:
    """
    Orchestrates chain-of-thought reasoning.

    The engine:
    1. Detects adversarial traps in queries
    2. Generates reasoning steps via LLM
    3. Executes tool actions
    4. Verifies results
    5. Learns from successful patterns
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: EngineConfig | None = None,
        tool_executor: ToolExecutor | None = None,
        pattern_store: PatternStore | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or EngineConfig()
        self._tool_executor = tool_executor
        self._pattern_store = pattern_store
        self._trap_detector = TrapDetector()

    @property
    def config(self) -> EngineConfig:
        """Get engine configuration."""
        return self._config

    async def reason(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> ReasoningChain:
        """
        Execute chain-of-thought reasoning on a query.

        Args:
            query: The query to reason about
            context: Optional context information

        Returns:
            Completed ReasoningChain
        """
        # Initialize chain manager
        manager = ChainManager(self._config.to_chain_config())
        manager.initialize(query)

        try:
            # Phase 1: Trap detection
            if self._config.trap_detection:
                detected_trap = await self._detect_trap(query)
                if detected_trap != TrapType.NONE:
                    manager.add_thought(
                        content=f"Detected potential trap: {detected_trap.value}",
                        reasoning_type=ReasoningType.CRITIQUE,
                        confidence=0.8,
                        detected_trap=detected_trap,
                        key_insights=[f"Query contains {detected_trap.value} pattern"],
                    )

            # Phase 2: Pattern matching
            if self._config.use_patterns and self._pattern_store:
                patterns = await self._pattern_store.find_matching_patterns(query)
                if patterns:
                    best_pattern = max(patterns, key=lambda p: p.confidence)
                    manager.add_thought(
                        content=f"Found similar problem pattern: {best_pattern.category}",
                        reasoning_type=ReasoningType.ANALYSIS,
                        confidence=best_pattern.confidence,
                        key_insights=best_pattern.key_steps[:3],
                    )

            # Phase 3: Main reasoning loop
            chain = await self._reasoning_loop(manager, query, context or {})

            # Phase 4: Save successful pattern
            if (
                chain.status == ChainStatus.COMPLETED
                and self._config.use_patterns
                and self._pattern_store
            ):
                await self._save_successful_pattern(chain)

            return chain

        except Exception as e:
            return manager.fail(str(e))

    async def _detect_trap(self, query: str) -> TrapType:
        """Detect adversarial traps in the query."""
        return self._trap_detector.detect(query)

    async def _reasoning_loop(
        self,
        manager: ChainManager,
        _query: str,  # noqa: ARG002
        context: dict[str, Any],
    ) -> ReasoningChain:
        """Execute the main reasoning loop."""
        iteration = 0
        max_iterations = self._config.max_steps

        while not manager.is_terminal and iteration < max_iterations:
            iteration += 1

            # Generate next step
            next_step = await self._generate_next_step(manager, context)

            if next_step is None:
                # LLM couldn't determine next step - try to conclude
                if manager.chain and manager.chain.verification_count >= self._config.min_verifications:
                    # Can conclude
                    await self._generate_conclusion(manager, context)
                else:
                    # Need more verification first
                    await self._generate_verification(manager, context)
                continue

            step_type, step_content = next_step

            if step_type == "thought":
                await self._add_thought_from_llm(manager, step_content)
            elif step_type == "action":
                await self._execute_action(manager, step_content)
            elif step_type == "verify":
                await self._generate_verification(manager, context)
            elif step_type == "conclude":
                await self._generate_conclusion(manager, context)

        if not manager.is_terminal:
            return manager.fail("Max iterations reached")

        return manager.chain

    async def _generate_next_step(
        self,
        manager: ChainManager,
        _context: dict[str, Any],  # noqa: ARG002
    ) -> tuple[str, str] | None:
        """Generate the next reasoning step via LLM."""
        # Build prompt with current chain state
        chain = manager.chain
        steps_summary = self._summarize_steps(chain.steps) if chain else ""

        prompt = f"""Query: {chain.query if chain else 'Unknown'}

Current reasoning steps:
{steps_summary}

State: {manager.state.value}
Steps remaining: {manager.remaining_steps}
Time remaining: {(self._config.timeout_ms - manager.elapsed_ms) / 1000:.1f}s

What should be the next step? Choose one:
- "thought: <your analysis>" - to reason about the problem
- "action: <tool_name>(<args>)" - to use a tool
- "verify: <claim>" - to verify a claim
- "conclude: <answer>" - to provide final answer

Next step:"""

        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=self._config.temperature,
            max_tokens=512,
        )

        # Parse response
        response = response.strip()
        if response.startswith("thought:"):
            return ("thought", response[8:].strip())
        elif response.startswith("action:"):
            return ("action", response[7:].strip())
        elif response.startswith("verify:"):
            return ("verify", response[7:].strip())
        elif response.startswith("conclude:"):
            return ("conclude", response[9:].strip())

        return None

    async def _add_thought_from_llm(
        self,
        manager: ChainManager,
        content: str,
    ) -> None:
        """Add a thought step from LLM output."""
        # Determine reasoning type from content
        reasoning_type = self._infer_reasoning_type(content)

        # Check for trap indicators
        detected_trap = self._trap_detector.detect(content)

        manager.add_thought(
            content=content,
            reasoning_type=reasoning_type,
            confidence=0.6,
            detected_trap=detected_trap,
        )

    async def _execute_action(
        self,
        manager: ChainManager,
        action_str: str,
    ) -> None:
        """Parse and execute an action."""
        # Parse action string: "tool_name(arg1=val1, arg2=val2)"
        match = re.match(r"(\w+)\((.*)\)", action_str)
        if not match:
            manager.add_thought(
                content=f"Failed to parse action: {action_str}",
                reasoning_type=ReasoningType.CRITIQUE,
                confidence=0.3,
            )
            return

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (simplified)
        tool_input = {}
        if args_str:
            for arg in args_str.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    tool_input[key.strip()] = value.strip().strip("\"'")

        # Add action step
        manager.add_action(
            content=f"Executing {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            expected_outcome=f"Result from {tool_name}",
            confidence=0.5,
        )

        # Execute if executor available
        if self._tool_executor:
            try:
                success, raw_output, parsed = await self._tool_executor.execute(
                    tool_name=tool_name,
                    tool_input=tool_input,
                )

                action = manager.get_last_action()
                manager.add_observation(
                    content=f"Received output from {tool_name}",
                    source_action_id=action.id,
                    success=success,
                    raw_output=raw_output,
                    parsed_output=parsed,
                    error_message=None if success else raw_output,
                    confidence=0.7 if success else 0.3,
                )
            except Exception as e:
                action = manager.get_last_action()
                manager.add_observation(
                    content=f"Tool execution failed: {e}",
                    source_action_id=action.id,
                    success=False,
                    error_message=str(e),
                    confidence=0.1,
                )
        else:
            # No executor - add placeholder observation
            action = manager.get_last_action()
            manager.add_observation(
                content="Tool execution not available",
                source_action_id=action.id,
                success=False,
                error_message="No tool executor configured",
                confidence=0.1,
            )

    async def _generate_verification(
        self,
        manager: ChainManager,
        _context: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Generate a verification step via LLM."""
        chain = manager.chain
        steps_summary = self._summarize_steps(chain.steps) if chain else ""

        prompt = f"""Based on the reasoning so far:
{steps_summary}

Verify the key claims. Provide:
1. The main claim to verify
2. Evidence supporting or refuting it
3. Your verdict (verified/refuted/uncertain)
4. Reasoning

Format:
CLAIM: <claim>
EVIDENCE: <evidence1>; <evidence2>
VERDICT: <verified|refuted|uncertain>
REASONING: <explanation>"""

        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=ADVERSARIAL_DEFENSE_PROMPT,
            temperature=0.3,  # Lower temperature for verification
            max_tokens=512,
        )

        # Parse verification response
        claim = ""
        evidence = []
        verdict = VerificationVerdict.UNCERTAIN
        reasoning = ""

        for line in response.strip().split("\n"):
            if line.startswith("CLAIM:"):
                claim = line[6:].strip()
            elif line.startswith("EVIDENCE:"):
                evidence = [e.strip() for e in line[9:].split(";") if e.strip()]
            elif line.startswith("VERDICT:"):
                verdict_str = line[8:].strip().lower()
                if verdict_str == "verified":
                    verdict = VerificationVerdict.VERIFIED
                elif verdict_str == "refuted":
                    verdict = VerificationVerdict.REFUTED
                else:
                    verdict = VerificationVerdict.UNCERTAIN
            elif line.startswith("REASONING:"):
                reasoning = line[10:].strip()

        if claim and evidence:
            manager.add_verification(
                content=f"Verifying: {claim}",
                claim=claim,
                evidence=evidence,
                verdict=verdict,
                reasoning=reasoning or "No reasoning provided",
                confidence=0.7 if verdict == VerificationVerdict.VERIFIED else 0.5,
            )

    async def _generate_conclusion(
        self,
        manager: ChainManager,
        _context: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Generate final conclusion via LLM."""
        chain = manager.chain
        steps_summary = self._summarize_steps(chain.steps) if chain else ""
        traps = chain.detected_trap_types if chain else set()

        trap_warning = ""
        if traps:
            trap_warning = f"\n\nWARNING: Detected traps: {', '.join(t.value for t in traps)}"

        prompt = f"""Based on the reasoning:
{steps_summary}{trap_warning}

Provide a final answer. Include:
1. The answer
2. Confidence level (0.0-1.0)
3. Any caveats or limitations
4. Whether you're uncertain (if applicable)

Format:
ANSWER: <your answer>
CONFIDENCE: <0.0-1.0>
CAVEATS: <caveat1>; <caveat2> (or "none")
UNCERTAIN: <yes|no>
REASON: <if uncertain, why>"""

        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=1024,
        )

        # Parse conclusion response
        answer = ""
        confidence = 0.5
        caveats = []
        is_uncertain = False
        uncertainty_reason = None

        for line in response.strip().split("\n"):
            if line.startswith("ANSWER:"):
                answer = line[7:].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("CAVEATS:"):
                caveats_str = line[8:].strip()
                if caveats_str.lower() != "none":
                    caveats = [c.strip() for c in caveats_str.split(";") if c.strip()]
            elif line.startswith("UNCERTAIN:"):
                is_uncertain = line[10:].strip().lower() in ("yes", "true", "1")
            elif line.startswith("REASON:"):
                uncertainty_reason = line[7:].strip()

        # Add trap caveats
        if traps:
            caveats.append(f"Query may contain adversarial elements: {', '.join(t.value for t in traps)}")

        manager.conclude(
            content=f"Conclusion: {answer}",
            answer=answer or "Unable to determine answer",
            confidence=confidence,
            caveats=caveats,
            is_uncertain=is_uncertain,
            uncertainty_reason=uncertainty_reason,
        )

    def _summarize_steps(self, steps: list) -> str:
        """Summarize reasoning steps for prompts."""
        if not steps:
            return "(no steps yet)"

        lines = []
        for i, step in enumerate(steps, 1):
            step_type = step.type.value
            content = step.content[:200] + "..." if len(step.content) > 200 else step.content
            confidence = f"[conf: {step.confidence:.2f}]"
            lines.append(f"{i}. [{step_type}] {content} {confidence}")

        return "\n".join(lines)

    def _infer_reasoning_type(self, content: str) -> ReasoningType:
        """Infer reasoning type from content."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["break down", "analyze", "examine"]):
            return ReasoningType.ANALYSIS
        elif any(word in content_lower for word in ["split", "decompose", "parts"]):
            return ReasoningType.DECOMPOSITION
        elif any(word in content_lower for word in ["but", "however", "wrong", "issue"]):
            return ReasoningType.CRITIQUE
        elif any(word in content_lower for word in ["perhaps", "maybe", "could be"]):
            return ReasoningType.HYPOTHESIS
        elif any(word in content_lower for word in ["compare", "versus", "vs", "difference"]):
            return ReasoningType.COMPARISON
        elif any(word in content_lower for word in ["combine", "together", "overall"]):
            return ReasoningType.SYNTHESIS

        return ReasoningType.ANALYSIS

    async def _save_successful_pattern(self, chain: ReasoningChain) -> None:
        """Save a successful reasoning chain as a pattern."""
        if not self._pattern_store or not chain.conclusion:
            return

        # Extract key steps
        key_steps = []
        for step in chain.steps:
            if step.type.value in ("thought", "verification", "conclusion"):
                key_steps.append(f"{step.type.value}: {step.content[:100]}")

        # Determine category from query
        category = self._categorize_query(chain.query)

        pattern = SolutionPattern(
            query_pattern=chain.query,
            category=category,
            successful_chain_id=chain.id,
            key_steps=key_steps[:20],
            traps_avoided=list(chain.detected_trap_types),
        )

        await self._pattern_store.save_pattern(pattern)

    def _categorize_query(self, query: str) -> str:
        """Categorize a query for pattern matching."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["calculate", "compute", "math", "+", "-", "*", "/"]):
            return "math"
        elif any(word in query_lower for word in ["code", "function", "class", "program"]):
            return "coding"
        elif any(word in query_lower for word in ["what is", "define", "explain"]):
            return "knowledge"
        elif any(word in query_lower for word in ["how to", "steps", "process"]):
            return "procedural"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "causal"

        return "general"


# =============================================================================
# SIMPLE LLM ADAPTER (for testing/integration)
# =============================================================================


class SimpleLLMAdapter:
    """
    Simple adapter for Ollama-style LLM providers.

    Adapts the raw HTTP interface to the LLMProvider protocol.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
    ) -> None:
        self._base_url = base_url
        self._model = model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate response from Ollama."""
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
