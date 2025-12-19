"""Task completion detection for TinyLLM (CaRT - Chain-of-thought Reasoning Task completion).

This module provides the TaskCompletionDetector class that analyzes responses
to determine if a task is complete at every level of execution. This is critical
for long-running tasks where users need feedback about progress and completion.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="completion")


class CompletionStatus(str, Enum):
    """Status of task completion.

    Represents the state of a task at any point in its execution lifecycle.
    """

    COMPLETE = "complete"  # Task fully addressed, no more work needed
    IN_PROGRESS = "in_progress"  # Task partially done, more work needed
    BLOCKED = "blocked"  # Cannot proceed, waiting on external dependency
    NEEDS_MORE_INFO = "needs_more_info"  # Need clarification from user
    FAILED = "failed"  # Task cannot be completed


class CompletionSignals(BaseModel):
    """Detected signals that indicate completion status.

    Tracks various linguistic and structural indicators found in responses.
    """

    model_config = {"extra": "forbid"}

    # Completion indicators
    has_completion_phrases: bool = Field(
        default=False,
        description="Response contains phrases indicating completion"
    )
    has_definitive_answer: bool = Field(
        default=False,
        description="Response provides a clear, definitive answer"
    )
    has_code_output: bool = Field(
        default=False,
        description="Response includes code or structured output"
    )

    # Incompletion indicators
    has_followup_questions: bool = Field(
        default=False,
        description="Response ends with questions for the user"
    )
    has_continuation_phrases: bool = Field(
        default=False,
        description="Response suggests more work is needed"
    )
    has_uncertainty_markers: bool = Field(
        default=False,
        description="Response expresses uncertainty or caveats"
    )
    has_blocking_errors: bool = Field(
        default=False,
        description="Response indicates a blocking error occurred"
    )

    # Progress indicators
    mentions_steps_remaining: bool = Field(
        default=False,
        description="Response mentions remaining steps or work"
    )
    mentions_waiting: bool = Field(
        default=False,
        description="Response mentions waiting for something"
    )

    # Detected phrases
    detected_phrases: List[str] = Field(
        default_factory=list,
        description="Specific phrases detected in the response"
    )


class CompletionAnalysis(BaseModel):
    """Result of completion detection analysis."""

    model_config = {"extra": "forbid"}

    status: CompletionStatus = Field(
        description="Determined completion status"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the status determination (0.0-1.0)"
    )
    signals: CompletionSignals = Field(
        description="Detected signals that led to this determination"
    )
    reasoning: str = Field(
        description="Human-readable explanation of the determination"
    )
    suggested_action: Optional[str] = Field(
        default=None,
        description="Suggested action for the user or system"
    )


class TaskCompletionDetector:
    """Detects task completion status from LLM responses.

    This detector uses multiple heuristics to determine if a task is complete:
    - Keyword analysis: Look for completion/continuation indicators
    - Question detection: Check if response ends with questions
    - Code detection: Identify if complete code is provided
    - Error analysis: Detect blocking vs recoverable errors
    - Sentiment analysis: Gauge certainty and definitiveness

    The detector is designed to work with both single responses and
    multi-turn conversations.
    """

    # Phrase patterns indicating completion
    COMPLETION_PHRASES: Set[str] = {
        "task is complete",
        "task completed",
        "finished",
        "done",
        "completed successfully",
        "here is the result",
        "here's the result",
        "here is your",
        "here's your",
        "successfully created",
        "successfully implemented",
        "all set",
        "ready to use",
        "implementation complete",
    }

    # Phrase patterns indicating continuation needed
    CONTINUATION_PHRASES: Set[str] = {
        "let me know if",
        "would you like me to",
        "should i",
        "do you want",
        "is there anything else",
        "what would you like",
        "please provide",
        "i need",
        "waiting for",
        "pending",
        "next step",
        "continue with",
        "shall i proceed",
        "would you prefer",
    }

    # Phrase patterns indicating need for more information
    INFO_REQUEST_PHRASES: Set[str] = {
        "please clarify",
        "could you specify",
        "need more details",
        "need more information",
        "can you provide",
        "please provide",
        "which one",
        "what do you mean",
        "unclear",
        "ambiguous",
        "need to know",
        "help me understand",
        "i need",
        "what language",
        "what error",
    }

    # Phrase patterns indicating blocking issues
    BLOCKING_PHRASES: Set[str] = {
        "cannot proceed",
        "unable to continue",
        "blocked by",
        "missing dependency",
        "not available",
        "access denied",
        "permission denied",
        "fatal error",
        "critical error",
        "cannot access",
    }

    # Phrase patterns indicating failure
    FAILURE_PHRASES: Set[str] = {
        "impossible",
        "cannot be done",
        "not possible",
        "will not work",
        "failed permanently",
        "cannot complete",
        "unable to complete",
        "task cannot",
        "this won't work",
    }

    # Uncertainty markers
    UNCERTAINTY_MARKERS: Set[str] = {
        "might",
        "maybe",
        "possibly",
        "perhaps",
        "i think",
        "probably",
        "not sure",
        "uncertain",
        "unclear",
        "could be",
        "may be",
    }

    def __init__(self):
        """Initialize the completion detector."""
        logger.debug("task_completion_detector_initialized")

    def detect_completion(
        self,
        response: str,
        task: str,
        conversation_history: Optional[List[str]] = None,
    ) -> CompletionAnalysis:
        """Detect if a task is complete based on the response.

        Args:
            response: The LLM response to analyze.
            task: The original task/query that was given.
            conversation_history: Optional list of previous messages in conversation.

        Returns:
            CompletionAnalysis with status and supporting evidence.
        """
        logger.debug(
            "detecting_task_completion",
            task_length=len(task),
            response_length=len(response),
            has_history=conversation_history is not None,
        )

        # Normalize for analysis
        response_lower = response.lower().strip()
        task_lower = task.lower().strip()

        # Detect various signals
        signals = self._detect_signals(response_lower, task_lower)

        # Determine status based on signals
        status, confidence, reasoning = self._determine_status(signals, response, task)

        # Generate suggested action
        suggested_action = self._suggest_action(status, signals)

        analysis = CompletionAnalysis(
            status=status,
            confidence=confidence,
            signals=signals,
            reasoning=reasoning,
            suggested_action=suggested_action,
        )

        logger.info(
            "completion_detected",
            status=status.value,
            confidence=confidence,
            has_followup_questions=signals.has_followup_questions,
            has_completion_phrases=signals.has_completion_phrases,
        )

        return analysis

    def _detect_signals(self, response_lower: str, task_lower: str) -> CompletionSignals:
        """Detect various completion signals in the response.

        Args:
            response_lower: Normalized lowercase response.
            task_lower: Normalized lowercase task.

        Returns:
            CompletionSignals with detected indicators.
        """
        signals = CompletionSignals()
        detected_phrases = []

        # Check for completion phrases
        for phrase in self.COMPLETION_PHRASES:
            if phrase in response_lower:
                signals.has_completion_phrases = True
                detected_phrases.append(phrase)

        # Check for continuation phrases
        for phrase in self.CONTINUATION_PHRASES:
            if phrase in response_lower:
                signals.has_continuation_phrases = True
                detected_phrases.append(phrase)

        # Check for info request phrases
        info_request_found = False
        for phrase in self.INFO_REQUEST_PHRASES:
            if phrase in response_lower:
                info_request_found = True
                detected_phrases.append(phrase)

        # Check for blocking phrases
        for phrase in self.BLOCKING_PHRASES:
            if phrase in response_lower:
                signals.has_blocking_errors = True
                detected_phrases.append(phrase)

        # Check for failure phrases
        failure_detected = False
        for phrase in self.FAILURE_PHRASES:
            if phrase in response_lower:
                failure_detected = True
                detected_phrases.append(phrase)

        # Check for uncertainty markers
        uncertainty_count = 0
        for marker in self.UNCERTAINTY_MARKERS:
            if marker in response_lower:
                uncertainty_count += 1

        # More than 2 uncertainty markers suggests uncertainty
        if uncertainty_count >= 2:
            signals.has_uncertainty_markers = True

        # Check for questions at the end
        signals.has_followup_questions = self._has_ending_questions(response_lower)

        # Check for code blocks
        signals.has_code_output = self._has_code_block(response_lower)

        # Check for definitive answer
        signals.has_definitive_answer = self._has_definitive_answer(
            response_lower,
            task_lower,
            signals
        )

        # Check for mentions of remaining steps or waiting
        signals.mentions_steps_remaining = any(
            phrase in response_lower
            for phrase in ["next step", "remaining", "still need to", "will need to"]
        )

        signals.mentions_waiting = any(
            phrase in response_lower
            for phrase in ["waiting", "pending", "once you", "after you"]
        )

        signals.detected_phrases = detected_phrases

        return signals

    def _has_ending_questions(self, response_lower: str) -> bool:
        """Check if response ends with questions.

        Args:
            response_lower: Normalized lowercase response.

        Returns:
            True if response appears to end with questions.
        """
        # Get last few sentences
        sentences = [s.strip() for s in response_lower.split('.') if s.strip()]
        if not sentences:
            return False

        # Check if last 1-2 sentences are questions
        last_sentences = sentences[-2:] if len(sentences) >= 2 else sentences[-1:]

        for sentence in last_sentences:
            # Explicit question mark
            if '?' in sentence:
                return True

            # Question words at start
            question_starters = ['what', 'when', 'where', 'why', 'how', 'which', 'who', 'should']
            if any(sentence.strip().startswith(word) for word in question_starters):
                return True

        return False

    def _has_code_block(self, response_lower: str) -> bool:
        """Check if response contains code blocks.

        Args:
            response_lower: Normalized lowercase response.

        Returns:
            True if code blocks are present.
        """
        # Look for markdown code blocks
        if '```' in response_lower:
            return True

        # Look for indented code patterns
        lines = response_lower.split('\n')
        code_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))

        # If >30% of lines are indented, likely has code
        if len(lines) > 0 and code_lines / len(lines) > 0.3:
            return True

        return False

    def _has_definitive_answer(
        self,
        response_lower: str,
        task_lower: str,
        signals: CompletionSignals
    ) -> bool:
        """Determine if response provides a definitive answer.

        Args:
            response_lower: Normalized lowercase response.
            task_lower: Normalized lowercase task.
            signals: Already detected signals.

        Returns:
            True if answer appears definitive.
        """
        # If it has code and completion phrases, likely definitive
        if signals.has_code_output and signals.has_completion_phrases:
            return True

        # If response is substantive and doesn't have questions/uncertainty
        is_substantive = len(response_lower) > 100
        no_questions = not signals.has_followup_questions
        no_uncertainty = not signals.has_uncertainty_markers

        if is_substantive and no_questions and no_uncertainty:
            return True

        # Check for direct answer patterns
        answer_patterns = [
            r'^(the answer is|the result is|the solution is)',
            r'^(yes|no),?\s',
            r'(here is|here\'s|here are)',
        ]

        for pattern in answer_patterns:
            if re.search(pattern, response_lower):
                return True

        return False

    def _determine_status(
        self,
        signals: CompletionSignals,
        response: str,
        task: str
    ) -> tuple[CompletionStatus, float, str]:
        """Determine completion status based on signals.

        Args:
            signals: Detected completion signals.
            response: Original response text.
            task: Original task text.

        Returns:
            Tuple of (status, confidence, reasoning).
        """
        # FAILED: Explicit failure indicators
        if any(phrase in response.lower() for phrase in self.FAILURE_PHRASES):
            return (
                CompletionStatus.FAILED,
                0.9,
                "Response contains explicit failure indicators suggesting the task cannot be completed."
            )

        # BLOCKED: Blocking errors or missing dependencies
        if signals.has_blocking_errors:
            return (
                CompletionStatus.BLOCKED,
                0.85,
                "Response indicates a blocking issue that prevents task completion."
            )

        # NEEDS_MORE_INFO: Explicit requests for clarification
        # Count info request phrases
        info_request_count = sum(
            1 for phrase in self.INFO_REQUEST_PHRASES
            if phrase in response.lower()
        )

        # Check if response is primarily asking for information
        response_lower = response.lower()
        has_provide = "provide" in response_lower or "clarify" in response_lower
        has_need = "need" in response_lower or "require" in response_lower

        # Strong signal: Multiple info requests OR info request + questions + need/provide keywords
        if info_request_count >= 1 and signals.has_followup_questions and (has_provide or has_need):
            return (
                CompletionStatus.NEEDS_MORE_INFO,
                0.8,
                "Response contains questions or requests for more information from the user."
            )

        # Moderate signal: Multiple info request phrases
        if info_request_count >= 2:
            return (
                CompletionStatus.NEEDS_MORE_INFO,
                0.75,
                "Response requests clarification and additional details."
            )

        # COMPLETE: Strong completion indicators
        if signals.has_completion_phrases and signals.has_definitive_answer:
            confidence = 0.9 if not signals.has_followup_questions else 0.75
            return (
                CompletionStatus.COMPLETE,
                confidence,
                "Response contains completion phrases and provides a definitive answer."
            )

        if signals.has_definitive_answer and not signals.has_continuation_phrases:
            return (
                CompletionStatus.COMPLETE,
                0.85,
                "Response provides a definitive answer without suggesting further work."
            )

        # IN_PROGRESS: Continuation indicators
        if signals.has_continuation_phrases or signals.mentions_steps_remaining:
            return (
                CompletionStatus.IN_PROGRESS,
                0.8,
                "Response suggests more work is needed or asks about next steps."
            )

        if signals.mentions_waiting:
            return (
                CompletionStatus.IN_PROGRESS,
                0.75,
                "Response indicates waiting for user input or external action."
            )

        # Default: Ambiguous case - check response substance
        if len(response) > 200 and not signals.has_followup_questions:
            return (
                CompletionStatus.COMPLETE,
                0.6,
                "Response is substantive and doesn't explicitly request further input."
            )

        if signals.has_followup_questions:
            return (
                CompletionStatus.IN_PROGRESS,
                0.65,
                "Response ends with questions suggesting user input is needed."
            )

        # Very short responses are likely incomplete
        if len(response) < 50:
            return (
                CompletionStatus.IN_PROGRESS,
                0.7,
                "Response is brief and may be incomplete."
            )

        # When in doubt, assume in progress with low confidence
        return (
            CompletionStatus.IN_PROGRESS,
            0.5,
            "Unable to definitively determine completion status."
        )

    def _suggest_action(
        self,
        status: CompletionStatus,
        signals: CompletionSignals
    ) -> Optional[str]:
        """Suggest next action based on completion status.

        Args:
            status: Determined completion status.
            signals: Detected signals.

        Returns:
            Suggested action string or None.
        """
        if status == CompletionStatus.COMPLETE:
            return "Task appears complete. Review the output and confirm it meets your needs."

        if status == CompletionStatus.NEEDS_MORE_INFO:
            return "Provide the requested information to allow the task to proceed."

        if status == CompletionStatus.BLOCKED:
            return "Resolve the blocking issue before the task can continue."

        if status == CompletionStatus.FAILED:
            return "Task cannot be completed as requested. Consider revising the task or approach."

        if status == CompletionStatus.IN_PROGRESS:
            if signals.has_followup_questions:
                return "Respond to the questions to continue progress on the task."
            if signals.mentions_waiting:
                return "Provide the required input when ready."
            return "Wait for the next update or provide additional guidance if needed."

        return None


# Convenience function for quick completion checks
def is_task_complete(response: str, task: str) -> bool:
    """Quick check if a task appears complete.

    Args:
        response: The LLM response to analyze.
        task: The original task/query.

    Returns:
        True if task appears complete with reasonable confidence.
    """
    detector = TaskCompletionDetector()
    analysis = detector.detect_completion(response, task)
    return analysis.status == CompletionStatus.COMPLETE and analysis.confidence >= 0.7
