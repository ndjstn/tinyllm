"""
Comprehensive tests for the reasoning module.

Tests cover:
- Pydantic model validation (strict mode)
- Chain state machine transitions
- Trap detection accuracy
- Builder fluent API
- Engine orchestration
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tinyllm.reasoning import (
    # Enums
    StepType,
    ReasoningType,
    VerificationVerdict,
    ChainStatus,
    TrapType,
    # Models
    ThoughtStep,
    ActionStep,
    ObservationStep,
    VerificationStep,
    ConclusionStep,
    ReasoningChain,
    ReasoningConfig,
    SolutionPattern,
    generate_step_id,
    # Chain
    ReasoningState,
    ChainManager,
    ChainManagerConfig,
    ChainBuilder,
    VALID_TRANSITIONS,
    # Prompts
    TrapDetector,
    run_trap_detection_tests,
    # Engine
    EngineConfig,
)


# =============================================================================
# MODEL VALIDATION TESTS
# =============================================================================


class TestStepModels:
    """Tests for step model validation."""

    def test_thought_step_valid(self):
        """Test valid thought step creation."""
        step = ThoughtStep(
            content="Analyzing the problem",
            reasoning_type=ReasoningType.ANALYSIS,
            confidence=0.8,
        )
        assert step.type == StepType.THOUGHT
        assert step.content == "Analyzing the problem"
        assert step.reasoning_type == ReasoningType.ANALYSIS
        assert step.confidence == 0.8
        assert step.detected_trap == TrapType.NONE

    def test_thought_step_with_trap(self):
        """Test thought step with detected trap."""
        step = ThoughtStep(
            content="This query contains a false premise",
            reasoning_type=ReasoningType.CRITIQUE,
            detected_trap=TrapType.FALSE_PREMISE,
            key_insights=["The assumption is incorrect"],
        )
        assert step.detected_trap == TrapType.FALSE_PREMISE
        assert "The assumption is incorrect" in step.key_insights

    def test_thought_step_empty_content_fails(self):
        """Test that empty content fails validation."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            ThoughtStep(
                content="   ",  # whitespace only
                reasoning_type=ReasoningType.ANALYSIS,
            )

    def test_thought_step_invalid_confidence_fails(self):
        """Test that invalid confidence fails validation."""
        with pytest.raises(ValueError):
            ThoughtStep(
                content="Test",
                reasoning_type=ReasoningType.ANALYSIS,
                confidence=1.5,  # > 1.0
            )

    def test_action_step_valid(self):
        """Test valid action step creation."""
        step = ActionStep(
            content="Calling calculator",
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            expected_outcome="Numeric result",
        )
        assert step.type == StepType.ACTION
        assert step.tool_name == "calculator"
        assert step.tool_input == {"expression": "2+2"}

    def test_action_step_invalid_tool_name_fails(self):
        """Test that invalid tool name fails validation."""
        with pytest.raises(ValueError):
            ActionStep(
                content="Test",
                tool_name="Invalid Tool!",  # has space and special char
                tool_input={},
                expected_outcome="Test",
            )

    def test_action_step_invalid_input_key_fails(self):
        """Test that invalid input key fails validation."""
        with pytest.raises(ValueError, match="Invalid tool input key"):
            ActionStep(
                content="Test",
                tool_name="test_tool",
                tool_input={"invalid-key": "value"},  # hyphens not allowed
                expected_outcome="Test",
            )

    def test_observation_step_valid(self):
        """Test valid observation step creation."""
        action_id = generate_step_id()
        step = ObservationStep(
            content="Received result",
            source_action_id=action_id,
            success=True,
            raw_output="4",
        )
        assert step.type == StepType.OBSERVATION
        assert step.success is True
        assert step.source_action_id == action_id

    def test_observation_step_failure_requires_error(self):
        """Test that failed observation requires error message."""
        with pytest.raises(ValueError, match="error_message required"):
            ObservationStep(
                content="Failed",
                source_action_id=generate_step_id(),
                success=False,
                # missing error_message
            )

    def test_verification_step_valid(self):
        """Test valid verification step creation."""
        step = VerificationStep(
            content="Verifying claim",
            claim="2+2=4",
            evidence=["Basic arithmetic rules"],
            verdict=VerificationVerdict.VERIFIED,
            reasoning="This is correct by definition",
        )
        assert step.type == StepType.VERIFICATION
        assert step.verdict == VerificationVerdict.VERIFIED

    def test_verification_step_empty_evidence_fails(self):
        """Test that empty evidence fails validation."""
        with pytest.raises(ValueError, match="At least one evidence"):
            VerificationStep(
                content="Test",
                claim="Test claim",
                evidence=[],  # empty
                verdict=VerificationVerdict.VERIFIED,
                reasoning="Test",
            )

    def test_conclusion_step_valid(self):
        """Test valid conclusion step creation."""
        step = ConclusionStep(
            content="Final answer",
            answer="4",
            confidence=0.95,
            caveats=["Assuming base 10"],
        )
        assert step.type == StepType.CONCLUSION
        assert step.answer == "4"
        assert step.is_uncertain is False

    def test_conclusion_step_uncertain_requires_reason(self):
        """Test that uncertain conclusion requires reason."""
        with pytest.raises(ValueError, match="uncertainty_reason required"):
            ConclusionStep(
                content="Uncertain answer",
                answer="Maybe 4?",
                is_uncertain=True,
                # missing uncertainty_reason
            )

    def test_conclusion_step_uncertain_with_reason(self):
        """Test uncertain conclusion with reason."""
        step = ConclusionStep(
            content="Uncertain answer",
            answer="Maybe 4?",
            is_uncertain=True,
            uncertainty_reason="Ambiguous query",
        )
        assert step.is_uncertain is True
        assert step.uncertainty_reason == "Ambiguous query"


class TestReasoningChain:
    """Tests for reasoning chain validation."""

    def test_chain_creation(self):
        """Test basic chain creation."""
        thought = ThoughtStep(
            content="Initial analysis",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        chain = ReasoningChain(
            query="What is 2+2?",
            steps=[thought],
        )
        assert chain.query == "What is 2+2?"
        assert len(chain.steps) == 1
        assert chain.status == ChainStatus.IN_PROGRESS

    def test_chain_duplicate_step_ids_fails(self):
        """Test that duplicate step IDs fail validation."""
        step_id = generate_step_id()
        step1 = ThoughtStep(
            id=step_id,
            content="First",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        step2 = ThoughtStep(
            id=step_id,  # same ID!
            content="Second",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        with pytest.raises(ValueError, match="Duplicate step IDs"):
            ReasoningChain(query="Test", steps=[step1, step2])

    def test_chain_invalid_observation_reference_fails(self):
        """Test that observation referencing non-existent action fails."""
        thought = ThoughtStep(
            content="Test",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        observation = ObservationStep(
            content="Observation",
            source_action_id=generate_step_id(),  # doesn't exist
            success=True,
        )
        with pytest.raises(ValueError, match="non-existent actions"):
            ReasoningChain(query="Test", steps=[thought, observation])

    def test_chain_completed_without_conclusion_fails(self):
        """Test that completed chain without conclusion fails."""
        thought = ThoughtStep(
            content="Test",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        with pytest.raises(ValueError, match="must have a conclusion"):
            ReasoningChain(
                query="Test",
                steps=[thought],
                status=ChainStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc),
            )

    def test_chain_add_step(self):
        """Test adding step to chain (immutable pattern)."""
        thought1 = ThoughtStep(
            content="First",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        chain1 = ReasoningChain(query="Test", steps=[thought1])

        thought2 = ThoughtStep(
            content="Second",
            reasoning_type=ReasoningType.SYNTHESIS,
        )
        chain2 = chain1.add_step(thought2)

        # Original unchanged
        assert len(chain1.steps) == 1
        # New chain has both
        assert len(chain2.steps) == 2

    def test_chain_conclusion_completes(self):
        """Test that adding conclusion completes chain."""
        thought = ThoughtStep(
            content="Analysis",
            reasoning_type=ReasoningType.ANALYSIS,
        )
        chain = ReasoningChain(query="Test", steps=[thought])

        conclusion = ConclusionStep(
            content="Final",
            answer="Done",
            confidence=0.9,
        )
        completed_chain = chain.add_step(conclusion)

        assert completed_chain.status == ChainStatus.COMPLETED
        assert completed_chain.completed_at is not None

    def test_chain_properties(self):
        """Test chain properties."""
        thought = ThoughtStep(
            content="Analysis",
            reasoning_type=ReasoningType.ANALYSIS,
            detected_trap=TrapType.FALSE_PREMISE,
        )
        verification = VerificationStep(
            content="Verify",
            claim="Test claim",
            evidence=["Evidence"],
            verdict=VerificationVerdict.VERIFIED,
            reasoning="Correct",
        )
        conclusion = ConclusionStep(
            content="Final",
            answer="Done",
        )

        chain = ReasoningChain(
            query="Test",
            steps=[thought, verification, conclusion],
            status=ChainStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            detected_traps=[TrapType.FALSE_PREMISE],
        )

        assert chain.has_conclusion is True
        assert chain.conclusion == conclusion
        assert chain.verification_count == 1
        assert TrapType.FALSE_PREMISE in chain.detected_trap_types


# =============================================================================
# CHAIN MANAGER TESTS
# =============================================================================


class TestChainManager:
    """Tests for chain manager state machine."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = ChainManager()
        chain = manager.initialize("What is 2+2?")

        assert manager.state == ReasoningState.ANALYZING
        assert chain.query == "What is 2+2?"
        assert len(chain.steps) == 1

    def test_manager_double_init_fails(self):
        """Test that double initialization fails."""
        manager = ChainManager()
        manager.initialize("Test")

        with pytest.raises(ValueError, match="already initialized"):
            manager.initialize("Test again")

    def test_manager_add_thought(self):
        """Test adding thought step."""
        manager = ChainManager()
        manager.initialize("Test")

        chain = manager.add_thought(
            content="Deeper analysis",
            reasoning_type=ReasoningType.DECOMPOSITION,
            confidence=0.7,
        )

        assert len(chain.steps) == 2
        assert manager.state == ReasoningState.DECOMPOSING

    def test_manager_add_action_observation(self):
        """Test action-observation flow."""
        manager = ChainManager()
        manager.initialize("Test")

        # Add action
        chain = manager.add_action(
            content="Calling tool",
            tool_name="calculator",
            tool_input={"x": "2+2"},
            expected_outcome="4",
        )
        assert manager.state == ReasoningState.EXECUTING_ACTION

        # Get action ID
        action = manager.get_last_action()
        assert action is not None

        # Add observation
        chain = manager.add_observation(
            content="Got result",
            source_action_id=action.id,
            success=True,
            raw_output="4",
        )
        assert manager.state == ReasoningState.OBSERVING

    def test_manager_verification(self):
        """Test verification step."""
        manager = ChainManager()
        manager.initialize("Test")

        chain = manager.add_verification(
            content="Verifying",
            claim="2+2=4",
            evidence=["Math"],
            verdict=VerificationVerdict.VERIFIED,
            reasoning="Correct",
        )

        assert manager.state == ReasoningState.VERIFYING

    def test_manager_conclude_without_verification_fails(self):
        """Test that conclusion without verification fails when required."""
        config = ChainManagerConfig(require_verification=True, min_verifications=1)
        manager = ChainManager(config)
        manager.initialize("Test")

        with pytest.raises(ValueError, match="Insufficient verifications"):
            manager.conclude(
                content="Final",
                answer="4",
            )

    def test_manager_conclude_success(self):
        """Test successful conclusion."""
        config = ChainManagerConfig(require_verification=True, min_verifications=1)
        manager = ChainManager(config)
        manager.initialize("Test")

        # Add verification first
        manager.add_verification(
            content="Verify",
            claim="Test",
            evidence=["Evidence"],
            verdict=VerificationVerdict.VERIFIED,
            reasoning="OK",
        )

        # Now conclude
        chain = manager.conclude(
            content="Final",
            answer="4",
            confidence=0.9,
        )

        assert manager.state == ReasoningState.COMPLETED
        assert manager.is_terminal is True
        assert chain.status == ChainStatus.COMPLETED

    def test_manager_fail(self):
        """Test failure handling."""
        manager = ChainManager()
        manager.initialize("Test")

        chain = manager.fail("Something went wrong")

        assert manager.state == ReasoningState.FAILED
        assert manager.is_terminal is True
        assert chain.status == ChainStatus.FAILED

    def test_manager_step_limit(self):
        """Test step limit enforcement."""
        config = ChainManagerConfig(max_steps=3)
        manager = ChainManager(config)
        manager.initialize("Test")  # 1 step

        manager.add_thought("Second", ReasoningType.ANALYSIS)  # 2 steps
        manager.add_thought("Third", ReasoningType.ANALYSIS)  # 3 steps

        with pytest.raises(ValueError, match="Step limit reached"):
            manager.add_thought("Fourth", ReasoningType.ANALYSIS)

    def test_manager_transitions_log(self):
        """Test transition history logging."""
        manager = ChainManager()
        manager.initialize("Test")
        manager.add_thought("Analysis", ReasoningType.DECOMPOSITION)

        log = manager.get_transitions_log()
        assert len(log) >= 2
        assert log[0]["from"] == "initializing"
        assert log[0]["to"] == "analyzing"


class TestChainBuilder:
    """Tests for fluent chain builder API."""

    def test_builder_simple_chain(self):
        """Test building simple chain."""
        chain = (
            ChainBuilder(query="What is 2+2?")
            .think("Breaking down arithmetic", ReasoningType.ANALYSIS)
            .verify(
                claim="2+2=4",
                evidence=["Basic arithmetic"],
                verdict=VerificationVerdict.VERIFIED,
                reasoning="Correct by definition",
            )
            .conclude("The answer is 4", answer="4", confidence=0.99)
            .build()
        )

        assert chain.query == "What is 2+2?"
        assert chain.status == ChainStatus.COMPLETED
        assert chain.conclusion.answer == "4"
        assert chain.verification_count == 1

    def test_builder_with_action(self):
        """Test building chain with action."""
        builder = ChainBuilder(query="Calculate something")
        builder.think("Need to use calculator", ReasoningType.ANALYSIS)
        builder.act(
            content="Running calculation",
            tool_name="calculator",
            tool_input={"expr": "10*10"},
            expected_outcome="100",
        )
        builder.observe(
            content="Got result",
            success=True,
            raw_output="100",
        )
        builder.verify(
            claim="10*10=100",
            evidence=["Calculator output"],
            verdict=VerificationVerdict.VERIFIED,
            reasoning="Matches",
        )
        builder.conclude("Result is 100", answer="100")

        chain = builder.build()
        assert chain.status == ChainStatus.COMPLETED

    def test_builder_observe_without_action_fails(self):
        """Test that observe without action fails."""
        builder = ChainBuilder(query="Test")

        with pytest.raises(ValueError, match="without a preceding action"):
            builder.observe("Got result", success=True)


# =============================================================================
# TRAP DETECTION TESTS
# =============================================================================


class TestTrapDetector:
    """Tests for adversarial trap detection."""

    @pytest.fixture
    def detector(self):
        """Create detector fixture."""
        return TrapDetector()

    def test_false_premise_detection(self, detector):
        """Test false premise detection."""
        queries = [
            "Why is the Great Wall of China visible from space?",
            "Explain why goldfish have a 3 second memory.",
            "Since humans only use 10% of their brain, what's in the rest?",
        ]
        for query in queries:
            result = detector.detect(query)
            assert result == TrapType.FALSE_PREMISE, f"Failed for: {query}"

    def test_trick_question_detection(self, detector):
        """Test trick question detection."""
        queries = [
            "What's heavier, a pound of feathers or a pound of gold?",
        ]
        for query in queries:
            result = detector.detect(query)
            assert result == TrapType.TRICK_QUESTION, f"Failed for: {query}"

    def test_hallucination_bait_detection(self, detector):
        """Test hallucination bait detection."""
        queries = [
            "Tell me about Shakespeare's famous novel 'The Azure Gardens'.",
            "Describe the Battle of Vermillion in 1847.",
        ]
        for query in queries:
            result = detector.detect(query)
            assert result == TrapType.HALLUCINATION_BAIT, f"Failed for: {query}"

    def test_impossible_task_detection(self, detector):
        """Test impossible task detection."""
        queries = [
            "Predict next week's lottery numbers.",
            "Write a function that solves the halting problem.",
        ]
        for query in queries:
            result = detector.detect(query)
            assert result == TrapType.IMPOSSIBLE_TASK, f"Failed for: {query}"

    def test_ambiguous_query_detection(self, detector):
        """Test ambiguous query detection."""
        queries = [
            "What is the best?",
            "When did it happen?",
        ]
        for query in queries:
            result = detector.detect(query)
            assert result == TrapType.AMBIGUOUS_QUERY, f"Failed for: {query}"

    def test_safe_query_no_trap(self, detector):
        """Test safe queries don't trigger traps."""
        queries = [
            "What is 2 + 2?",
            "Explain photosynthesis.",
            "How do computers work?",
            "What is the capital of France?",
        ]
        for query in queries:
            result = detector.detect(query)
            assert result == TrapType.NONE, f"False positive for: {query}"

    def test_defense_strategy(self, detector):
        """Test defense strategy recommendations."""
        for trap_type in TrapType:
            strategy = detector.get_defense_strategy(trap_type)
            assert isinstance(strategy, str)
            assert len(strategy) > 0

    def test_run_trap_detection_tests(self):
        """Test the built-in trap detection test suite."""
        results = run_trap_detection_tests()
        assert "passed" in results
        assert "failed" in results
        # Should have high pass rate
        total = len(results["passed"]) + len(results["failed"])
        pass_rate = len(results["passed"]) / total if total > 0 else 0
        assert pass_rate >= 0.7, f"Pass rate too low: {pass_rate:.1%}"


# =============================================================================
# ENGINE CONFIG TESTS
# =============================================================================


class TestEngineConfig:
    """Tests for engine configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = EngineConfig()
        assert config.max_steps == 20
        assert config.require_verification is True
        assert config.trap_detection is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = EngineConfig(
            max_steps=50,
            timeout_ms=300000,
            min_confidence=0.8,
        )
        assert config.max_steps == 50

        # Invalid config
        with pytest.raises(ValueError):
            EngineConfig(max_steps=0)  # must be >= 1

        with pytest.raises(ValueError):
            EngineConfig(min_confidence=2.0)  # must be <= 1.0

    def test_config_to_chain_config(self):
        """Test conversion to chain config."""
        engine_config = EngineConfig(
            max_steps=30,
            max_depth=7,
            require_verification=False,
        )
        chain_config = engine_config.to_chain_config()

        assert chain_config.max_steps == 30
        assert chain_config.max_depth == 7
        assert chain_config.require_verification is False


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================


class TestStateTransitions:
    """Tests for valid state transitions."""

    def test_all_states_have_transitions(self):
        """Test that all states are defined in transitions."""
        for state in ReasoningState:
            assert state in VALID_TRANSITIONS

    def test_terminal_states_no_transitions(self):
        """Test that terminal states have no outgoing transitions."""
        assert len(VALID_TRANSITIONS[ReasoningState.FAILED]) == 0
        assert len(VALID_TRANSITIONS[ReasoningState.COMPLETED]) == 0

    def test_all_transitions_lead_to_valid_states(self):
        """Test that all transitions lead to valid states."""
        for from_state, to_states in VALID_TRANSITIONS.items():
            for to_state in to_states:
                assert isinstance(to_state, ReasoningState)

    def test_can_reach_completion(self):
        """Test that completion is reachable from initial state."""
        # BFS to check reachability
        visited = set()
        queue = [ReasoningState.INITIALIZING]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for next_state in VALID_TRANSITIONS.get(current, set()):
                if next_state not in visited:
                    queue.append(next_state)

        assert ReasoningState.COMPLETED in visited


# =============================================================================
# SOLUTION PATTERN TESTS
# =============================================================================


class TestSolutionPattern:
    """Tests for solution pattern model."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = SolutionPattern(
            query_pattern="What is X + Y?",
            category="math",
            successful_chain_id="chain_" + "a" * 12,
            key_steps=["analyze", "calculate", "verify"],
        )
        assert pattern.category == "math"
        assert pattern.success_count == 1
        assert pattern.failure_count == 0

    def test_pattern_success_rate(self):
        """Test success rate calculation."""
        pattern = SolutionPattern(
            query_pattern="Test",
            category="test",
            successful_chain_id="chain_" + "a" * 12,
            key_steps=["step1"],
            success_count=7,
            failure_count=3,
        )
        assert pattern.success_rate == 0.7

    def test_pattern_confidence(self):
        """Test confidence calculation."""
        # Low usage = lower confidence
        low_usage = SolutionPattern(
            query_pattern="Test",
            category="test",
            successful_chain_id="chain_" + "a" * 12,
            key_steps=["step1"],
            success_count=1,
            failure_count=0,
        )

        # High usage = higher confidence
        high_usage = SolutionPattern(
            query_pattern="Test",
            category="test",
            successful_chain_id="chain_" + "a" * 12,
            key_steps=["step1"],
            success_count=90,
            failure_count=10,
        )

        assert low_usage.confidence < high_usage.confidence
        assert high_usage.confidence <= 0.95  # capped


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestReasoningIntegration:
    """Integration tests combining multiple components."""

    def test_full_reasoning_flow(self):
        """Test complete reasoning flow without LLM."""
        config = ChainManagerConfig(
            require_verification=True,
            min_verifications=1,
        )

        chain = (
            ChainBuilder(query="What is 15 + 27?", config=config)
            .think(
                "This is an addition problem with two numbers",
                ReasoningType.ANALYSIS,
                confidence=0.9,
            )
            .think(
                "I'll add the ones place first: 5 + 7 = 12",
                ReasoningType.DECOMPOSITION,
                confidence=0.95,
            )
            .think(
                "Now the tens place: 10 + 20 + 10 (carry) = 40",
                ReasoningType.DECOMPOSITION,
                confidence=0.95,
            )
            .think(
                "Final sum: 42",
                ReasoningType.SYNTHESIS,
                confidence=0.95,
            )
            .verify(
                claim="15 + 27 = 42",
                evidence=[
                    "5 + 7 = 12 (carry 1)",
                    "1 + 2 + 1 = 4",
                    "42 verified",
                ],
                verdict=VerificationVerdict.VERIFIED,
                reasoning="Step-by-step addition confirmed",
                confidence=0.98,
            )
            .conclude(
                "The sum of 15 and 27 is 42",
                answer="42",
                confidence=0.98,
                caveats=["Assuming base-10 arithmetic"],
            )
            .build()
        )

        assert chain.status == ChainStatus.COMPLETED
        assert chain.conclusion.answer == "42"
        assert chain.total_confidence > 0.9

    def test_adversarial_query_handling(self):
        """Test handling adversarial query."""
        detector = TrapDetector()

        # Detect the trap
        query = "Why is the Great Wall of China visible from space?"
        trap = detector.detect(query)
        assert trap == TrapType.FALSE_PREMISE

        # Build a chain that handles it
        chain = (
            ChainBuilder(query=query)
            .think(
                "This query contains a false premise - the Great Wall is NOT visible from space",
                ReasoningType.CRITIQUE,
                confidence=0.9,
                detected_trap=TrapType.FALSE_PREMISE,
                key_insights=["Common misconception", "Wall is too narrow"],
            )
            .verify(
                claim="Great Wall visible from space",
                evidence=[
                    "NASA has confirmed it's not visible",
                    "Wall is ~30 feet wide, roads are wider and also not visible",
                ],
                verdict=VerificationVerdict.REFUTED,
                reasoning="Multiple authoritative sources refute this claim",
                confidence=0.95,
            )
            .conclude(
                "The premise is false - the Great Wall is NOT visible from space",
                answer="The Great Wall of China is NOT visible from space with the naked eye. "
                       "This is a common misconception. While the wall is very long, it's only "
                       "about 30 feet wide, which is too narrow to see from orbit.",
                confidence=0.95,
                caveats=["May be visible with telescopic equipment"],
                is_uncertain=False,
            )
            .build()
        )

        assert TrapType.FALSE_PREMISE in chain.detected_trap_types
        assert "NOT visible" in chain.conclusion.answer
