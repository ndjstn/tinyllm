# Task Completion Detection (CaRT)

Chain-of-thought Reasoning Task (CaRT) completion detection helps TinyLLM determine when tasks are complete at every level of execution. This is critical for long-running tasks where users need feedback about progress and completion status.

## Overview

The `TaskCompletionDetector` analyzes LLM responses to automatically determine completion status using multiple heuristics:

- **Keyword analysis**: Detects completion and continuation phrases
- **Question detection**: Identifies when clarification is needed
- **Code detection**: Recognizes complete code implementations
- **Error analysis**: Distinguishes blocking vs recoverable errors
- **Sentiment analysis**: Gauges certainty and definitiveness

## Completion States

### CompletionStatus Enum

```python
class CompletionStatus(str, Enum):
    COMPLETE = "complete"          # Task fully addressed, no more work needed
    IN_PROGRESS = "in_progress"    # Task partially done, more work needed
    BLOCKED = "blocked"             # Cannot proceed, waiting on dependency
    NEEDS_MORE_INFO = "needs_more_info"  # Need clarification from user
    FAILED = "failed"               # Task cannot be completed
```

### State Transitions

```
┌─────────────┐
│   START     │
└─────┬───────┘
      │
      ├──> COMPLETE ──────> [END]
      │
      ├──> IN_PROGRESS ──┬──> COMPLETE
      │                  ├──> BLOCKED
      │                  └──> NEEDS_MORE_INFO
      │
      ├──> NEEDS_MORE_INFO ──> IN_PROGRESS
      │
      ├──> BLOCKED ──────> IN_PROGRESS
      │
      └──> FAILED ────────> [END]
```

## Usage

### Basic Detection

```python
from tinyllm.core.completion import TaskCompletionDetector

detector = TaskCompletionDetector()
task = "Create a function to calculate fibonacci numbers"
response = """Here's the implementation:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

The task is complete."""

analysis = detector.detect_completion(response, task)
print(f"Status: {analysis.status.value}")
print(f"Confidence: {analysis.confidence:.0%}")
print(f"Reasoning: {analysis.reasoning}")
```

Output:
```
Status: complete
Confidence: 90%
Reasoning: Response contains completion phrases and provides a definitive answer.
```

### Quick Check

For simple yes/no completion checks:

```python
from tinyllm.core.completion import is_task_complete

if is_task_complete(response, task):
    print("Task is complete!")
else:
    print("Task needs more work...")
```

### Multi-Turn Conversations

Track completion across conversation turns:

```python
detector = TaskCompletionDetector()
conversation_history = []

for turn in conversation:
    analysis = detector.detect_completion(
        response=turn.response,
        task=original_task,
        conversation_history=conversation_history
    )

    if analysis.status == CompletionStatus.COMPLETE:
        break
    elif analysis.status == CompletionStatus.NEEDS_MORE_INFO:
        print(f"User input needed: {analysis.suggested_action}")

    conversation_history.append(turn.response)
```

## Integration with Executor

The completion detector can be integrated into the executor to track task completion:

```python
from tinyllm.core.executor import Executor
from tinyllm.core.completion import TaskCompletionDetector

executor = Executor(graph)
detector = TaskCompletionDetector()

response = await executor.execute(task)

if response.success:
    analysis = detector.detect_completion(
        response.content,
        task.content
    )

    print(f"Task status: {analysis.status.value}")

    if analysis.status != CompletionStatus.COMPLETE:
        print(f"Next action: {analysis.suggested_action}")
```

## Detection Signals

The detector analyzes multiple signals to determine completion status:

### CompletionSignals

```python
class CompletionSignals(BaseModel):
    # Completion indicators
    has_completion_phrases: bool      # "task is complete", "done", etc.
    has_definitive_answer: bool       # Clear, unambiguous answer
    has_code_output: bool             # Includes code blocks

    # Incompletion indicators
    has_followup_questions: bool      # Ends with questions
    has_continuation_phrases: bool    # "let me know if", etc.
    has_uncertainty_markers: bool     # "might", "maybe", etc.
    has_blocking_errors: bool         # "cannot proceed", etc.

    # Progress indicators
    mentions_steps_remaining: bool    # "next step", "remaining"
    mentions_waiting: bool            # "waiting for", "pending"
```

## Phrase Detection

The detector recognizes various phrase patterns:

### Completion Phrases

- "task is complete"
- "finished"
- "done"
- "successfully implemented"
- "here is the result"
- "ready to use"

### Continuation Phrases

- "let me know if"
- "would you like me to"
- "is there anything else"
- "next step"
- "shall i proceed"

### Info Request Phrases

- "please clarify"
- "need more information"
- "can you provide"
- "what do you mean"
- "i need"

### Blocking Phrases

- "cannot proceed"
- "blocked by"
- "missing dependency"
- "access denied"
- "fatal error"

### Failure Phrases

- "impossible"
- "cannot be done"
- "not possible"
- "will not work"

## Examples

### Example 1: Complete with Code

```python
task = "Create a function to calculate factorial"
response = """
Here's the factorial function:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

Implementation complete.
"""

# Result:
# Status: COMPLETE (confidence: 90%)
# Signals: has_code_output=True, has_completion_phrases=True
```

### Example 2: Needs More Information

```python
task = "Optimize the database"
response = """
I can help optimize the database, but I need more information:
1. What database system are you using?
2. What are the performance issues?

Please provide these details.
"""

# Result:
# Status: NEEDS_MORE_INFO (confidence: 80%)
# Signals: has_followup_questions=True
# Suggestion: Provide the requested information
```

### Example 3: Blocked

```python
task = "Deploy to production"
response = """
Cannot proceed with deployment because:
- Access denied to production server
- Missing SSH credentials

These blocking issues must be resolved first.
"""

# Result:
# Status: BLOCKED (confidence: 85%)
# Signals: has_blocking_errors=True
# Suggestion: Resolve the blocking issue
```

### Example 4: In Progress

```python
task = "Create a web application"
response = """
I've started the web application. So far:
- Set up routing
- Created basic structure

Next steps:
- Add authentication
- Implement database layer

Would you like me to continue?
"""

# Result:
# Status: IN_PROGRESS (confidence: 80%)
# Signals: mentions_steps_remaining=True, has_followup_questions=True
# Suggestion: Respond to continue progress
```

### Example 5: Failed

```python
task = "Time travel to yesterday"
response = """
This task is impossible and cannot be completed.
Time travel violates the laws of physics and is not possible.
"""

# Result:
# Status: FAILED (confidence: 90%)
# Suggestion: Consider revising the task
```

## Best Practices

### 1. Use Confidence Thresholds

```python
analysis = detector.detect_completion(response, task)

if analysis.status == CompletionStatus.COMPLETE and analysis.confidence >= 0.8:
    # High confidence completion
    finalize_task()
elif analysis.confidence < 0.6:
    # Low confidence, may need human review
    flag_for_review()
```

### 2. Provide Context

```python
# Better: Include conversation history
analysis = detector.detect_completion(
    response,
    task,
    conversation_history=previous_messages
)
```

### 3. Act on Suggested Actions

```python
analysis = detector.detect_completion(response, task)

if analysis.suggested_action:
    notify_user(analysis.suggested_action)
```

### 4. Log Completion Status

```python
from tinyllm.logging import get_logger

logger = get_logger(__name__)

analysis = detector.detect_completion(response, task)
logger.info(
    "task_completion_detected",
    status=analysis.status.value,
    confidence=analysis.confidence,
    has_code=analysis.signals.has_code_output
)
```

## UI Feedback

### Terminal Output

Use color coding to display status:

```python
from rich.console import Console

console = Console()

status_colors = {
    CompletionStatus.COMPLETE: "green",
    CompletionStatus.IN_PROGRESS: "yellow",
    CompletionStatus.BLOCKED: "red",
    CompletionStatus.NEEDS_MORE_INFO: "blue",
    CompletionStatus.FAILED: "red bold",
}

status_icons = {
    CompletionStatus.COMPLETE: "✓",
    CompletionStatus.IN_PROGRESS: "⋯",
    CompletionStatus.BLOCKED: "⊗",
    CompletionStatus.NEEDS_MORE_INFO: "?",
    CompletionStatus.FAILED: "✗",
}

color = status_colors[analysis.status]
icon = status_icons[analysis.status]

console.print(f"[{color}]{icon} {analysis.status.value.upper()}[/{color}]")
```

### Progress Indicators

```python
# Show progress bar based on completion signals
if analysis.status == CompletionStatus.IN_PROGRESS:
    if analysis.signals.mentions_steps_remaining:
        show_progress_bar()
    elif analysis.signals.mentions_waiting:
        show_waiting_spinner()
```

## Performance Considerations

### Efficient Detection

The detector is designed for fast analysis:

- Phrase matching: O(n) where n = response length
- Signal detection: ~1-2ms for typical responses
- Memory usage: Minimal, stateless detector

### Caching

For repeated checks on the same response:

```python
# Cache analysis results
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_detect(response_hash: str, task_hash: str):
    return detector.detect_completion(response, task)
```

## Limitations

1. **Language-dependent**: Optimized for English responses
2. **Heuristic-based**: Not 100% accurate, uses confidence scores
3. **Context-limited**: Works best with clear, structured responses
4. **No semantic understanding**: Relies on patterns, not deep comprehension

## Advanced Features

### Custom Phrase Sets

Extend the detector with domain-specific phrases:

```python
detector = TaskCompletionDetector()

# Add custom completion phrases
detector.COMPLETION_PHRASES.update({
    "deployment successful",
    "migration complete",
    "tests passing"
})

# Add custom blocking phrases
detector.BLOCKING_PHRASES.update({
    "database locked",
    "rate limit exceeded"
})
```

### Confidence Tuning

Adjust confidence thresholds based on use case:

```python
# Critical tasks: require higher confidence
CRITICAL_CONFIDENCE_THRESHOLD = 0.9

# Routine tasks: accept lower confidence
ROUTINE_CONFIDENCE_THRESHOLD = 0.6

if is_critical_task:
    threshold = CRITICAL_CONFIDENCE_THRESHOLD
else:
    threshold = ROUTINE_CONFIDENCE_THRESHOLD

if analysis.confidence >= threshold:
    proceed_with_task()
```

## See Also

- [Executor Documentation](./executor.md)
- [Message System](./messages.md)
- [Tracing](./tracing.md)
- [Examples](../examples/completion_detection_example.py)
