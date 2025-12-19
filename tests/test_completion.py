"""Tests for task completion detection."""

import pytest

from tinyllm.core.completion import (
    CompletionStatus,
    TaskCompletionDetector,
    is_task_complete,
)


class TestTaskCompletionDetector:
    """Test suite for TaskCompletionDetector."""

    def test_complete_with_code(self):
        """Test detection of complete task with code output."""
        detector = TaskCompletionDetector()
        task = "Create a function to calculate fibonacci numbers"
        response = """Here is the implementation:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

The task is complete. You can now use this function to calculate fibonacci numbers."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.COMPLETE
        assert analysis.confidence >= 0.8
        assert analysis.signals.has_code_output
        assert analysis.signals.has_completion_phrases
        assert analysis.signals.has_definitive_answer

    def test_needs_more_info(self):
        """Test detection of need for more information."""
        detector = TaskCompletionDetector()
        task = "Fix the bug in my code"
        response = """I'd be happy to help fix the bug, but I need more information:

1. What programming language are you using?
2. What is the expected behavior?
3. What error are you seeing?

Please provide these details so I can assist you better."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.NEEDS_MORE_INFO
        assert analysis.confidence >= 0.7
        assert analysis.signals.has_followup_questions
        assert "provide" in analysis.suggested_action.lower()

    def test_blocked_status(self):
        """Test detection of blocked status."""
        detector = TaskCompletionDetector()
        task = "Access the production database"
        response = """I cannot proceed with this task because:

1. Access to the production database is denied
2. Missing required credentials
3. Network connection is blocked by firewall

These issues must be resolved before I can continue."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.BLOCKED
        assert analysis.confidence >= 0.8
        assert analysis.signals.has_blocking_errors
        assert "blocking" in analysis.reasoning.lower()

    def test_failed_status(self):
        """Test detection of failed status."""
        detector = TaskCompletionDetector()
        task = "Travel back in time"
        response = """This task is impossible and cannot be completed.
Time travel is not possible with current technology and physics.
This task will not work and cannot be done."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.FAILED
        assert analysis.confidence >= 0.8
        assert "cannot be completed" in analysis.reasoning.lower()

    def test_in_progress_with_questions(self):
        """Test detection of in-progress status with questions."""
        detector = TaskCompletionDetector()
        task = "Create a web application"
        response = """I've started working on the web application.
I've set up the basic structure and routing.

What frontend framework would you like me to use?
Should I include user authentication?"""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.IN_PROGRESS
        assert analysis.signals.has_followup_questions

    def test_in_progress_with_continuation(self):
        """Test detection of in-progress with continuation phrases."""
        detector = TaskCompletionDetector()
        task = "Write documentation"
        response = """I've written the introduction and overview sections.

Let me know if you'd like me to continue with the API reference
and examples, or if you want to review what I've done so far."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.IN_PROGRESS
        assert analysis.signals.has_continuation_phrases

    def test_complete_definitive_answer(self):
        """Test detection of complete with definitive answer."""
        detector = TaskCompletionDetector()
        task = "What is 2 + 2?"
        response = "The answer is 4."

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.COMPLETE
        assert analysis.signals.has_definitive_answer

    def test_uncertainty_markers(self):
        """Test detection of uncertainty markers."""
        detector = TaskCompletionDetector()
        task = "Predict the weather"
        response = """I think it might rain tomorrow, but I'm not sure.
It could be sunny, or perhaps cloudy. Maybe there will be storms,
but it's unclear at this point."""

        analysis = detector.detect_completion(response, task)

        assert analysis.signals.has_uncertainty_markers

    def test_complete_with_code_and_explanation(self):
        """Test complete status with code and comprehensive explanation."""
        detector = TaskCompletionDetector()
        task = "Write a function to validate email addresses"
        response = """Here's your email validation function:

```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

This function uses a regular expression to validate email format.
It checks for:
- Valid characters before the @
- Valid domain name
- Valid top-level domain

The implementation is complete and ready to use."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.COMPLETE
        assert analysis.confidence >= 0.85
        assert analysis.signals.has_code_output
        assert analysis.signals.has_completion_phrases

    def test_waiting_for_input(self):
        """Test detection of waiting for user input."""
        detector = TaskCompletionDetector()
        task = "Install the package"
        response = """I need you to confirm a few things before proceeding:

Waiting for your approval to install version 2.0 instead of 1.0.
Once you confirm, I'll continue with the installation."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.IN_PROGRESS
        assert analysis.signals.mentions_waiting

    def test_steps_remaining(self):
        """Test detection of remaining steps."""
        detector = TaskCompletionDetector()
        task = "Deploy the application"
        response = """I've completed the build step successfully.

Next steps:
1. Run tests
2. Deploy to staging
3. Deploy to production

Still need to complete the remaining deployment stages."""

        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.IN_PROGRESS
        assert analysis.signals.mentions_steps_remaining

    def test_ambiguous_short_response(self):
        """Test handling of ambiguous short response."""
        detector = TaskCompletionDetector()
        task = "Help me"
        response = "Sure, I can help."

        analysis = detector.detect_completion(response, task)

        # Short responses are typically considered in-progress
        assert analysis.status == CompletionStatus.IN_PROGRESS
        assert analysis.confidence <= 0.7

    def test_substantive_response_without_questions(self):
        """Test substantive response without follow-up questions."""
        detector = TaskCompletionDetector()
        task = "Explain how neural networks work"
        response = """Neural networks are computational models inspired by biological neural networks.
They consist of interconnected nodes (neurons) organized in layers:

1. Input Layer: Receives the initial data
2. Hidden Layers: Process the data through weighted connections
3. Output Layer: Produces the final result

The network learns by adjusting weights through backpropagation,
minimizing the difference between predicted and actual outputs.
This allows the network to recognize patterns and make predictions
based on training data.

The process involves forward propagation (making predictions) and
backward propagation (learning from errors). Over time, the network
becomes better at its task through this iterative learning process."""

        analysis = detector.detect_completion(response, task)

        # Substantive explanation without questions should be complete
        assert analysis.status == CompletionStatus.COMPLETE
        assert analysis.confidence >= 0.6

    def test_mixed_signals(self):
        """Test response with mixed completion signals."""
        detector = TaskCompletionDetector()
        task = "Create a database schema"
        response = """Here's the database schema:

```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE
);
```

The basic schema is complete. Would you like me to add
indexes or additional tables?"""

        analysis = detector.detect_completion(response, task)

        # Has completion phrases and code, but also follow-up question
        # Should lean toward complete but with lower confidence
        assert analysis.status in [CompletionStatus.COMPLETE, CompletionStatus.IN_PROGRESS]
        if analysis.status == CompletionStatus.COMPLETE:
            assert analysis.confidence < 0.9


class TestQuickCompletionCheck:
    """Test the convenience function for quick completion checks."""

    def test_is_task_complete_true(self):
        """Test quick check returns True for complete task."""
        task = "Calculate 5 + 3"
        response = "The answer is 8."

        assert is_task_complete(response, task) is True

    def test_is_task_complete_false_questions(self):
        """Test quick check returns False when questions present."""
        task = "Help me"
        response = "Sure! What do you need help with?"

        assert is_task_complete(response, task) is False

    def test_is_task_complete_false_blocked(self):
        """Test quick check returns False when blocked."""
        task = "Access file"
        response = "Cannot proceed. Access denied."

        assert is_task_complete(response, task) is False


class TestSignalDetection:
    """Test individual signal detection methods."""

    def test_ending_questions_explicit(self):
        """Test detection of explicit question marks."""
        detector = TaskCompletionDetector()
        response = "I've done some work. What should I do next?"

        signals = detector._detect_signals(response.lower(), "task")

        assert signals.has_followup_questions

    def test_ending_questions_implicit(self):
        """Test detection of implicit questions."""
        detector = TaskCompletionDetector()
        response = "I've done some work. Should I continue with the next phase"

        signals = detector._detect_signals(response.lower(), "task")

        # Should detect question starter word
        assert signals.has_followup_questions

    def test_code_block_detection_markdown(self):
        """Test detection of markdown code blocks."""
        detector = TaskCompletionDetector()
        response = "Here's the code:\n```python\nprint('hello')\n```"

        signals = detector._detect_signals(response.lower(), "task")

        assert signals.has_code_output

    def test_code_block_detection_indented(self):
        """Test detection of indented code blocks."""
        detector = TaskCompletionDetector()
        response = """Here's the code:
    def hello():
        print('hello')
        return True

    def world():
        print('world')"""

        signals = detector._detect_signals(response.lower(), "task")

        assert signals.has_code_output

    def test_completion_phrases_detected(self):
        """Test detection of completion phrases."""
        detector = TaskCompletionDetector()
        response = "The task is complete and ready to use."

        signals = detector._detect_signals(response.lower(), "task")

        assert signals.has_completion_phrases
        assert len(signals.detected_phrases) > 0

    def test_continuation_phrases_detected(self):
        """Test detection of continuation phrases."""
        detector = TaskCompletionDetector()
        response = "Let me know if you'd like me to continue."

        signals = detector._detect_signals(response.lower(), "task")

        assert signals.has_continuation_phrases
        assert len(signals.detected_phrases) > 0


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_response(self):
        """Test handling of empty response."""
        detector = TaskCompletionDetector()
        task = "Do something"
        response = ""

        # Should not crash, should handle gracefully
        analysis = detector.detect_completion(response, task)

        assert analysis.status == CompletionStatus.IN_PROGRESS
        assert analysis.confidence <= 0.7

    def test_very_long_response(self):
        """Test handling of very long response."""
        detector = TaskCompletionDetector()
        task = "Write an essay"
        response = "This is a detailed essay. " * 500  # Very long

        analysis = detector.detect_completion(response, task)

        # Should not crash and should work correctly
        assert analysis.status is not None
        assert 0.0 <= analysis.confidence <= 1.0

    def test_special_characters(self):
        """Test handling of special characters."""
        detector = TaskCompletionDetector()
        task = "Format text"
        response = "Done! ✓ Task completed successfully. 🎉"

        analysis = detector.detect_completion(response, task)

        # Should handle special chars and detect completion
        assert analysis.status == CompletionStatus.COMPLETE

    def test_multiple_code_blocks(self):
        """Test response with multiple code blocks."""
        detector = TaskCompletionDetector()
        task = "Create helper functions"
        response = """Here are the helper functions:

```python
def helper1():
    return 1
```

```python
def helper2():
    return 2
```

Both functions are complete."""

        analysis = detector.detect_completion(response, task)

        assert analysis.signals.has_code_output
        assert analysis.status == CompletionStatus.COMPLETE
