"""
TinyLLM Reasoning Module.

Provides chain-of-thought reasoning with adversarial defense.

Key Components:
- models: Strict Pydantic models for reasoning steps and chains
- chain: Chain manager with state machine logic
- engine: Reasoning engine orchestrator
- prompts: Adversarial defense prompts and trap detection

Usage:
    from tinyllm.reasoning import ReasoningEngine, EngineConfig, SimpleLLMAdapter

    # Create engine with Ollama
    llm = SimpleLLMAdapter(model="qwen2.5:3b")
    engine = ReasoningEngine(llm)

    # Execute reasoning
    chain = await engine.reason("What is 2 + 2?")
    print(chain.conclusion.answer)

For more control:
    from tinyllm.reasoning import ChainBuilder, ReasoningType, VerificationVerdict

    chain = (
        ChainBuilder(query="What is 2+2?")
        .think("This is basic arithmetic", ReasoningType.ANALYSIS)
        .verify("2+2=4", ["Basic math"], VerificationVerdict.VERIFIED, "Correct")
        .conclude("The answer is 4", answer="4", confidence=0.99)
        .build()
    )
"""

from tinyllm.reasoning.chain import (
    VALID_TRANSITIONS,
    # Builder
    ChainBuilder,
    ChainManager,
    # Manager
    ChainManagerConfig,
    # State machine
    ReasoningState,
    StateTransition,
)
from tinyllm.reasoning.engine import (
    # Engine
    EngineConfig,
    # Protocols
    LLMProvider,
    PatternStore,
    ReasoningEngine,
    SimpleLLMAdapter,
    ToolExecutor,
)
from tinyllm.reasoning.models import (
    ActionStep,
    # Step models
    BaseStep,
    ChainStatus,
    ConclusionStep,
    ObservationStep,
    # Chain and config
    ReasoningChain,
    ReasoningConfig,
    ReasoningStepUnion,
    ReasoningType,
    SolutionPattern,
    # Enums
    StepType,
    ThoughtStep,
    TrapType,
    VerificationStep,
    VerificationVerdict,
    # Utilities
    generate_step_id,
)
from tinyllm.reasoning.prompts import (
    ADVERSARIAL_DEFENSE_PROMPT,
    # Testing
    ADVERSARIAL_TEST_QUERIES,
    ANALYSIS_TEMPLATE,
    CONCLUSION_PROMPT,
    DECOMPOSITION_TEMPLATE,
    # Prompts
    REASONING_SYSTEM_PROMPT,
    SYNTHESIS_TEMPLATE,
    TRAP_DETECTION_PROMPT,
    VERIFICATION_TEMPLATE,
    # Templates
    PromptTemplate,
    # Detector
    TrapDetector,
    run_trap_detection_tests,
)

__all__ = [
    # Enums
    "StepType",
    "ReasoningType",
    "VerificationVerdict",
    "ChainStatus",
    "TrapType",
    # Step models
    "BaseStep",
    "ThoughtStep",
    "ActionStep",
    "ObservationStep",
    "VerificationStep",
    "ConclusionStep",
    "ReasoningStepUnion",
    # Chain and config
    "ReasoningChain",
    "ReasoningConfig",
    "SolutionPattern",
    # Utilities
    "generate_step_id",
    # State machine
    "ReasoningState",
    "StateTransition",
    "VALID_TRANSITIONS",
    # Manager
    "ChainManagerConfig",
    "ChainManager",
    # Builder
    "ChainBuilder",
    # Protocols
    "LLMProvider",
    "ToolExecutor",
    "PatternStore",
    # Engine
    "EngineConfig",
    "ReasoningEngine",
    "SimpleLLMAdapter",
    # Prompts
    "REASONING_SYSTEM_PROMPT",
    "ADVERSARIAL_DEFENSE_PROMPT",
    "TRAP_DETECTION_PROMPT",
    "CONCLUSION_PROMPT",
    # Detector
    "TrapDetector",
    # Templates
    "PromptTemplate",
    "ANALYSIS_TEMPLATE",
    "DECOMPOSITION_TEMPLATE",
    "VERIFICATION_TEMPLATE",
    "SYNTHESIS_TEMPLATE",
    # Testing
    "ADVERSARIAL_TEST_QUERIES",
    "run_trap_detection_tests",
]
