# TinyLLM Chat: Before and After Comparison

## Key Improvements Summary

| Feature | Before | After |
|---------|--------|-------|
| **Response Display** | Wait for complete response | Stream tokens in real-time |
| **Loading Feedback** | None | Animated spinner or streaming |
| **User Prompt** | `[cyan]You:[/cyan]` | `[bold cyan]You:[/bold cyan]` |
| **Assistant Label** | `[green]Assistant:[/green]` | `[bold green]Assistant:[/bold green]` |
| **Error Display** | `[red]Error: {e}[/red]` | `[bold red]Error:[/bold red] [red]{e}[/red]` |
| **Statistics** | None | Token count & elapsed time (>2s) |
| **Markdown Support** | Plain text | Rich markdown rendering |
| **Header** | Simple text | Bordered panel with info |
| **Exit Commands** | `quit` only | `quit` or `exit` |
| **History Clear** | Yellow text | Yellow with emoji-free design |

## Code Changes

### Function Signature
**Before:**
```python
def chat(
    model: str = typer.Option("qwen2.5:1.5b", "--model", "-m", help="Model to use"),
):
```

**After:**
```python
def chat(
    model: str = typer.Option("qwen2.5:1.5b", "--model", "-m", help="Model to use"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="Custom system prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream responses"),
):
```

### Imports
**Added:**
```python
import time
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
```

## Visual Comparison

### Before
```
TinyLLM Chat
Model: qwen2.5:1.5b
Type 'quit' to exit, 'clear' to clear history

You: What is Python?
Assistant: Python is a high-level programming language...
```

### After (Streaming Mode)
```
┌─────────────────────────────────────────────┐
│ TinyLLM Interactive Chat                    │
│                                             │
│ Model: qwen2.5:1.5b                        │
│ Identity: TinyLLM Assistant                │
│ Streaming: enabled                          │
│                                             │
│ Commands:                                   │
│   • Type 'quit' or 'exit' to leave         │
│   • Type 'clear' to clear history          │
└─────────────────────────────────────────────┘

You: What is Python?