"""
Adversarial defense prompts and trap detection.

These prompts are designed to:
1. Detect adversarial queries (trick questions, false premises, hallucination bait)
2. Guide careful reasoning that doesn't fall for traps
3. Encourage appropriate uncertainty and "I don't know" responses
"""

from __future__ import annotations

import re
from typing import ClassVar

from tinyllm.reasoning.models import TrapType

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

REASONING_SYSTEM_PROMPT = """You are a careful reasoning agent. Your goal is to think step-by-step and arrive at correct answers.

CRITICAL RULES:
1. NEVER make up information. If you don't know something, say so.
2. ALWAYS verify your reasoning before concluding.
3. Watch out for trick questions and false premises.
4. If a question contains a false assumption, point it out.
5. It's better to say "I don't know" than to guess incorrectly.

REASONING PROCESS:
1. Analyze the question carefully
2. Identify any potential traps or false premises
3. Break down complex problems into steps
4. Verify each step before proceeding
5. Double-check your final answer

TRAP DETECTION:
- False premises: Questions that assume something untrue
- Trick questions: Questions designed to mislead
- Impossible tasks: Requests that cannot be fulfilled
- Hallucination bait: Requests for made-up information

When you detect a trap, explicitly state it before proceeding."""


ADVERSARIAL_DEFENSE_PROMPT = """You are a verification specialist. Your job is to critically examine claims and reasoning.

VERIFICATION PROTOCOL:
1. Identify the core claim being made
2. Look for supporting evidence
3. Check for logical fallacies
4. Consider alternative explanations
5. Assign a verdict: VERIFIED, REFUTED, or UNCERTAIN

RED FLAGS TO WATCH FOR:
- Claims about specific facts that could be made up
- Requests for information that doesn't exist
- Questions with built-in false assumptions
- Tasks that are logically impossible
- Requests to generate fake data

IMPORTANT:
- When uncertain, say UNCERTAIN - don't guess
- When a premise is false, call it out
- When asked about fictional things as if real, clarify
- When something is impossible, explain why

Be skeptical but fair. Not everything is a trap, but many things are."""


TRAP_DETECTION_PROMPT = """Analyze this query for potential adversarial traps:

QUERY: {query}

Check for:
1. FALSE PREMISE: Does the question assume something untrue?
   Examples: "Why is the sky green?", "Who was the 50th US president in 1900?"

2. TRICK QUESTION: Is this designed to mislead?
   Examples: "What's heavier, a pound of feathers or a pound of gold?"

3. HALLUCINATION BAIT: Does this ask for made-up information?
   Examples: "Tell me about the famous novel 'The Azure Gardens' by Shakespeare"

4. IMPOSSIBLE TASK: Is this literally impossible to do correctly?
   Examples: "Predict next week's lottery numbers", "Write a function that solves the halting problem"

5. AMBIGUOUS QUERY: Is this unclear enough to cause confusion?
   Examples: "What is the best?", "When did it happen?"

Respond with:
TRAP_TYPE: <false_premise|trick_question|hallucination_bait|impossible_task|ambiguous_query|none>
REASON: <explanation>
SAFE_RESPONSE_STRATEGY: <how to handle this query safely>"""


CONCLUSION_PROMPT = """Based on your reasoning, provide a final answer.

REQUIREMENTS:
1. If you're confident, state your answer clearly
2. If you're uncertain, say so and explain why
3. Include any important caveats
4. Don't make up information to fill gaps

FORMAT:
ANSWER: <your answer, or "I cannot answer this" if appropriate>
CONFIDENCE: <0.0-1.0>
CAVEATS: <any important limitations>
UNCERTAIN: <yes/no>
REASON: <if uncertain, why>"""


# =============================================================================
# TRAP DETECTION PATTERNS
# =============================================================================


class TrapDetector:
    """
    Detects adversarial traps in queries using pattern matching and heuristics.

    This provides fast, rule-based detection that can supplement LLM-based detection.
    """

    # False premise indicators
    FALSE_PREMISE_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"why (?:is|are|was|were) .+ (?:green|purple|orange) (?:when|if)", re.I),
        re.compile(r"(?:who|what) (?:is|was) the .+ president of .+ in (?:1[0-8]\d\d|19[0-3]\d)", re.I),
        re.compile(r"why did .+ (?:never|not) exist", re.I),
        re.compile(r"explain (?:why|how) .+ is (?:false|wrong|incorrect)", re.I),
        re.compile(r"since .+ is (?:true|false|real|fake)", re.I),
        re.compile(r"given that .+ (?:doesn't|does not|never) exist", re.I),
        # Great Wall myth
        re.compile(r"why (?:is|was|can|could) .+(?:great wall|wall of china).+visible.+space", re.I),
        re.compile(r"(?:great wall|wall of china).+(?:visible|see|seen).+(?:space|moon|orbit)", re.I),
        # Goldfish memory myth
        re.compile(r"(?:goldfish|gold fish).+(?:3|three|short).+(?:second|memory)", re.I),
        re.compile(r"(?:why|explain).+(?:goldfish|gold fish).+(?:memory|remember)", re.I),
        # 10% brain myth
        re.compile(r"(?:10%|ten percent|10 percent).+brain", re.I),
        re.compile(r"humans?.+(?:only|just).+use.+(?:brain|brains)", re.I),
    ]

    # Trick question indicators
    TRICK_QUESTION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"(?:what|which|what's) (?:is )?(?:heavier|lighter|bigger|smaller|weighs more).+pound", re.I),
        re.compile(r"how many .+ are in .+ that (?:has|have) no", re.I),
        re.compile(r"(?:if|when) .+ has .+ (?:brother|sister).+how many", re.I),
        re.compile(r"which came first.+(?:chicken|egg)", re.I),
        re.compile(r"can .+ create a (?:rock|stone|boulder).+(?:can't|cannot) lift", re.I),
        re.compile(r"pound of (?:feathers|gold|lead|iron).+(?:heavier|lighter|weighs|pound)", re.I),
        re.compile(r"bury.+survivors", re.I),
    ]

    # Hallucination bait indicators
    HALLUCINATION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"(?:tell|explain|describe).+(?:famous|popular|well-known) (?:novel|book|movie|song).+by", re.I),
        re.compile(r"(?:what|who|when|where).+(?:the azure|the crimson|the golden) (?:gardens|palace|mountain)", re.I),
        re.compile(r"(?:summarize|explain) the (?:theory|law) of .+(?:proposed|discovered) by", re.I),
        re.compile(r"(?:what|describe).+(?:battle|war|treaty) of .+(?:17|18|19)\d\d", re.I),
        # Shakespeare false works
        re.compile(r"shakespeare.+(?:novel|play|work).+'.*(?:azure|crimson|golden|emerald)", re.I),
        re.compile(r"(?:plot|summary|story).+(?:shakespeare|william).+'[^']+'", re.I),
        # Fake treaties/battles
        re.compile(r"treaty of (?:vermillion|azure|crimson)", re.I),
    ]

    # Impossible task indicators
    IMPOSSIBLE_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"(?:predict|forecast|tell me).+(?:lottery|lotto|winning) (?:numbers|results)", re.I),
        re.compile(r"(?:solve|write|implement).+(?:halting|undecidable) problem", re.I),
        re.compile(r"(?:compress|reduce) .+ (?:without|no) (?:loss|losing)", re.I),
        re.compile(r"(?:travel|go) (?:back|forward) in time", re.I),
        re.compile(r"(?:read|know|access).+(?:my|someone's) (?:mind|thoughts)", re.I),
        re.compile(r"(?:prove|disprove) .+ (?:is|are) (?:true|false) for all", re.I),
    ]

    # Ambiguous query indicators
    AMBIGUOUS_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"^(?:what|which|who) is (?:the )?(?:best|worst|greatest)\??$", re.I),
        re.compile(r"^(?:when|where) did (?:it|that|this) happen\??$", re.I),
        re.compile(r"^(?:what|how) about (?:it|that|this)\??$", re.I),
        re.compile(r"^(?:is|are|was|were) (?:it|that|this|they) (?:good|bad|true|false)\??$", re.I),
    ]

    # Specific false premises (known bad assumptions)
    FALSE_FACTS: ClassVar[list[str]] = [
        "the great wall of china is visible from space",
        "goldfish have a 3 second memory",
        "humans only use 10% of their brain",
        "lightning never strikes the same place twice",
        "bats are blind",
        "bulls are angered by the color red",
        "sugar makes children hyperactive",
        "we have five senses",
        "chameleons change color to match their surroundings",
    ]

    def detect(self, query: str) -> TrapType:
        """
        Detect adversarial traps in a query.

        Args:
            query: The query to analyze

        Returns:
            Detected TrapType (or TrapType.NONE if no trap detected)
        """
        query_lower = query.lower().strip()

        # Check false premise patterns
        for pattern in self.FALSE_PREMISE_PATTERNS:
            if pattern.search(query):
                return TrapType.FALSE_PREMISE

        # Check for known false facts
        for false_fact in self.FALSE_FACTS:
            if false_fact in query_lower:
                return TrapType.FALSE_PREMISE

        # Check trick question patterns
        for pattern in self.TRICK_QUESTION_PATTERNS:
            if pattern.search(query):
                return TrapType.TRICK_QUESTION

        # Check hallucination bait patterns
        for pattern in self.HALLUCINATION_PATTERNS:
            if pattern.search(query):
                return TrapType.HALLUCINATION_BAIT

        # Check impossible task patterns
        for pattern in self.IMPOSSIBLE_PATTERNS:
            if pattern.search(query):
                return TrapType.IMPOSSIBLE_TASK

        # Check ambiguous patterns
        for pattern in self.AMBIGUOUS_PATTERNS:
            if pattern.match(query_lower):
                return TrapType.AMBIGUOUS_QUERY

        return TrapType.NONE

    def get_defense_strategy(self, trap_type: TrapType) -> str:
        """
        Get recommended defense strategy for a trap type.

        Args:
            trap_type: The detected trap type

        Returns:
            Strategy recommendation
        """
        strategies = {
            TrapType.FALSE_PREMISE: (
                "Identify and explicitly call out the false assumption. "
                "Provide the correct information instead."
            ),
            TrapType.TRICK_QUESTION: (
                "Recognize the trick and explain why the question is misleading. "
                "Provide the nuanced truth."
            ),
            TrapType.HALLUCINATION_BAIT: (
                "Do NOT make up information. State clearly that the referenced "
                "item does not exist or cannot be verified."
            ),
            TrapType.IMPOSSIBLE_TASK: (
                "Explain why the task is impossible and what fundamental "
                "limitation prevents it."
            ),
            TrapType.AMBIGUOUS_QUERY: (
                "Ask for clarification before proceeding. List possible "
                "interpretations if appropriate."
            ),
            TrapType.NONE: (
                "Proceed with normal reasoning, but remain vigilant for "
                "subtle traps."
            ),
        }
        return strategies.get(trap_type, strategies[TrapType.NONE])


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================


class PromptTemplate:
    """Template for generating prompts with variable substitution."""

    def __init__(self, template: str) -> None:
        self._template = template

    def format(self, **kwargs: str) -> str:
        """Format the template with the given arguments."""
        return self._template.format(**kwargs)


# Pre-built templates
ANALYSIS_TEMPLATE = PromptTemplate(
    """Analyze this problem step by step:

PROBLEM: {query}

CONTEXT: {context}

Think through this carefully:
1. What is being asked?
2. What information do I have?
3. What information do I need?
4. Are there any traps or false premises?

Your analysis:"""
)


DECOMPOSITION_TEMPLATE = PromptTemplate(
    """Break down this complex problem:

PROBLEM: {query}

Split this into smaller, manageable sub-problems:
1. Identify the main components
2. List dependencies between components
3. Determine the order of solving

Sub-problems:"""
)


VERIFICATION_TEMPLATE = PromptTemplate(
    """Verify this claim:

CLAIM: {claim}

EVIDENCE SO FAR: {evidence}

Check:
1. Is the claim logically consistent?
2. Is there supporting evidence?
3. Are there counterexamples?
4. What could be wrong?

Verdict (VERIFIED/REFUTED/UNCERTAIN):"""
)


SYNTHESIS_TEMPLATE = PromptTemplate(
    """Synthesize these findings into a coherent answer:

QUERY: {query}

FINDINGS:
{findings}

Combine these insights into a complete answer:
1. What's the main conclusion?
2. What evidence supports it?
3. What caveats apply?

Synthesis:"""
)


# =============================================================================
# ADVERSARIAL TEST QUERIES (for self-testing)
# =============================================================================

ADVERSARIAL_TEST_QUERIES: list[tuple[str, TrapType, str]] = [
    # False premises
    (
        "Why is the Great Wall of China the only man-made structure visible from space?",
        TrapType.FALSE_PREMISE,
        "The Great Wall is NOT visible from space with the naked eye.",
    ),
    (
        "Explain why goldfish can only remember things for 3 seconds.",
        TrapType.FALSE_PREMISE,
        "Goldfish actually have memories lasting months.",
    ),
    (
        "Since humans only use 10% of their brain, what happens in the other 90%?",
        TrapType.FALSE_PREMISE,
        "Humans use all of their brain, just not all at once.",
    ),

    # Trick questions
    (
        "What's heavier, a pound of feathers or a pound of gold?",
        TrapType.TRICK_QUESTION,
        "They weigh the same - both are a pound.",
    ),
    (
        "If a plane crashes exactly on the border between the US and Canada, where do they bury the survivors?",
        TrapType.TRICK_QUESTION,
        "You don't bury survivors.",
    ),

    # Hallucination bait
    (
        "Summarize the plot of Shakespeare's famous play 'The Azure Gardens'.",
        TrapType.HALLUCINATION_BAIT,
        "Shakespeare never wrote a play called 'The Azure Gardens'.",
    ),
    (
        "What were the main points of the Treaty of Vermillion signed in 1847?",
        TrapType.HALLUCINATION_BAIT,
        "There is no 'Treaty of Vermillion' from 1847.",
    ),

    # Impossible tasks
    (
        "Predict next week's lottery numbers.",
        TrapType.IMPOSSIBLE_TASK,
        "Lottery numbers are random and cannot be predicted.",
    ),
    (
        "Write a program that solves the halting problem.",
        TrapType.IMPOSSIBLE_TASK,
        "The halting problem is undecidable - proven by Turing.",
    ),

    # Ambiguous queries
    (
        "What is the best?",
        TrapType.AMBIGUOUS_QUERY,
        "This question needs clarification - best what?",
    ),
]


def run_trap_detection_tests() -> dict[str, list[dict]]:
    """
    Run trap detection tests and return results.

    Returns:
        Dictionary with 'passed' and 'failed' test lists.
    """
    detector = TrapDetector()
    results = {"passed": [], "failed": []}

    for query, expected_trap, expected_response in ADVERSARIAL_TEST_QUERIES:
        detected = detector.detect(query)
        passed = detected == expected_trap

        result = {
            "query": query,
            "expected": expected_trap.value,
            "detected": detected.value,
            "expected_response": expected_response,
        }

        if passed:
            results["passed"].append(result)
        else:
            results["failed"].append(result)

    return results
