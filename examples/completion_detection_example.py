"""Example demonstrating task completion detection in TinyLLM.

This example shows how to use the TaskCompletionDetector to determine
when tasks are complete at various levels of execution.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tinyllm.core.completion import (
    CompletionStatus,
    TaskCompletionDetector,
    is_task_complete,
)

console = Console()


def display_analysis(task: str, response: str, analysis):
    """Display completion analysis in a formatted way."""
    # Status color coding
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

    color = status_colors.get(analysis.status, "white")
    icon = status_icons.get(analysis.status, "•")

    # Create panel title with status
    title = f"[{color}]{icon} {analysis.status.value.upper()}[/{color}] (confidence: {analysis.confidence:.0%})"

    # Build content
    content = f"[bold]Task:[/bold] {task}\n\n"
    content += f"[bold]Response:[/bold]\n{response[:200]}{'...' if len(response) > 200 else ''}\n\n"
    content += f"[bold]Reasoning:[/bold] {analysis.reasoning}\n\n"

    if analysis.suggested_action:
        content += f"[bold]Suggested Action:[/bold] {analysis.suggested_action}\n\n"

    # Signal indicators
    signals_table = Table(show_header=False, box=None, padding=(0, 1))
    signals_table.add_column(style="dim")
    signals_table.add_column()

    if analysis.signals.has_completion_phrases:
        signals_table.add_row("✓", "Completion phrases detected")
    if analysis.signals.has_definitive_answer:
        signals_table.add_row("✓", "Definitive answer provided")
    if analysis.signals.has_code_output:
        signals_table.add_row("✓", "Code output included")
    if analysis.signals.has_followup_questions:
        signals_table.add_row("!", "Follow-up questions present")
    if analysis.signals.has_continuation_phrases:
        signals_table.add_row("!", "Continuation phrases detected")
    if analysis.signals.has_blocking_errors:
        signals_table.add_row("✗", "Blocking errors detected")
    if analysis.signals.mentions_steps_remaining:
        signals_table.add_row("⋯", "Mentions remaining steps")
    if analysis.signals.mentions_waiting:
        signals_table.add_row("⋯", "Waiting for input")

    content += "[bold]Detected Signals:[/bold]\n"

    console.print(Panel(content, title=title, border_style=color))
    console.print(signals_table)
    console.print()


def example_complete_with_code():
    """Example: Complete task with code output."""
    console.print("\n[bold cyan]Example 1: Complete Task with Code[/bold cyan]\n")

    task = "Create a Python function to calculate the factorial of a number"
    response = """Here's the implementation:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

The task is complete. You can now use this function to calculate factorials."""

    detector = TaskCompletionDetector()
    analysis = detector.detect_completion(response, task)
    display_analysis(task, response, analysis)


def example_needs_info():
    """Example: Task needs more information."""
    console.print("\n[bold cyan]Example 2: Needs More Information[/bold cyan]\n")

    task = "Optimize the database query"
    response = """I can help optimize your database query, but I need some additional information:

1. What database system are you using (PostgreSQL, MySQL, etc.)?
2. Can you share the current query?
3. What performance issues are you experiencing?

Please provide these details so I can give you specific optimization advice."""

    detector = TaskCompletionDetector()
    analysis = detector.detect_completion(response, task)
    display_analysis(task, response, analysis)


def example_blocked():
    """Example: Task is blocked."""
    console.print("\n[bold cyan]Example 3: Blocked Task[/bold cyan]\n")

    task = "Deploy to production server"
    response = """I cannot proceed with the deployment because:

1. Access to the production server is denied
2. Missing required SSH credentials
3. Production firewall is blocking the connection

These blocking issues must be resolved before deployment can continue."""

    detector = TaskCompletionDetector()
    analysis = detector.detect_completion(response, task)
    display_analysis(task, response, analysis)


def example_in_progress():
    """Example: Task in progress."""
    console.print("\n[bold cyan]Example 4: Task In Progress[/bold cyan]\n")

    task = "Create a REST API for user management"
    response = """I've started implementing the REST API. So far I've completed:

- User registration endpoint (POST /api/users)
- User login endpoint (POST /api/auth/login)

Next steps:
- User profile endpoint (GET /api/users/:id)
- Password reset functionality
- JWT token refresh

Would you like me to continue with the remaining endpoints, or should I add
authentication middleware first?"""

    detector = TaskCompletionDetector()
    analysis = detector.detect_completion(response, task)
    display_analysis(task, response, analysis)


def example_failed():
    """Example: Task failed."""
    console.print("\n[bold cyan]Example 5: Failed Task[/bold cyan]\n")

    task = "Recover deleted files from 10 years ago without any backup"
    response = """Unfortunately, this task is impossible and cannot be completed.

Files deleted 10 years ago without any backup cannot be recovered because:
1. The data has been permanently erased
2. No backup or snapshot exists
3. Modern file systems overwrite deleted data
4. The time frame is too long for any recovery method

This task will not work. You may want to focus on implementing a proper
backup strategy for the future instead."""

    detector = TaskCompletionDetector()
    analysis = detector.detect_completion(response, task)
    display_analysis(task, response, analysis)


def example_quick_check():
    """Example: Using the quick completion check."""
    console.print("\n[bold cyan]Example 6: Quick Completion Checks[/bold cyan]\n")

    test_cases = [
        ("What is 2+2?", "The answer is 4."),
        ("Help me debug", "What error are you seeing?"),
        ("Write a function", "```python\ndef func(): pass\n```\nDone!"),
    ]

    for task, response in test_cases:
        is_complete = is_task_complete(response, task)
        color = "green" if is_complete else "yellow"
        icon = "✓" if is_complete else "⋯"
        console.print(f"[{color}]{icon}[/{color}] Task: '{task}' -> {is_complete}")


def example_conversation_tracking():
    """Example: Tracking completion across conversation turns."""
    console.print("\n[bold cyan]Example 7: Multi-Turn Conversation Tracking[/bold cyan]\n")

    task = "Help me build a web scraper"
    conversation = [
        ("What website do you want to scrape?", CompletionStatus.NEEDS_MORE_INFO),
        ("I want to scrape news articles from example.com", CompletionStatus.IN_PROGRESS),
        ("Here's a basic scraper:\n```python\nimport requests\n...\n```\nShould I add error handling?", CompletionStatus.IN_PROGRESS),
        ("Here's the complete scraper with error handling:\n```python\n...\n```\nAll done!", CompletionStatus.COMPLETE),
    ]

    detector = TaskCompletionDetector()

    for i, (response, expected_status) in enumerate(conversation, 1):
        console.print(f"\n[bold]Turn {i}:[/bold]")
        analysis = detector.detect_completion(response, task)

        status_color = "green" if analysis.status == expected_status else "red"
        console.print(f"Response: {response[:80]}...")
        console.print(f"Status: [{status_color}]{analysis.status.value}[/{status_color}] (expected: {expected_status.value})")
        console.print(f"Confidence: {analysis.confidence:.0%}")


async def main():
    """Run all examples."""
    console.print(Panel.fit(
        "[bold cyan]TinyLLM Task Completion Detection Examples[/bold cyan]\n"
        "Demonstrating CaRT (Chain-of-thought Reasoning Task completion)",
        border_style="cyan"
    ))

    # Run all examples
    example_complete_with_code()
    example_needs_info()
    example_blocked()
    example_in_progress()
    example_failed()
    example_quick_check()
    example_conversation_tracking()

    console.print("\n[bold green]All examples completed![/bold green]")
    console.print("\n[dim]The TaskCompletionDetector helps TinyLLM know when tasks are")
    console.print("complete at every level, providing feedback to users and enabling")
    console.print("the system to make intelligent decisions about task progress.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
