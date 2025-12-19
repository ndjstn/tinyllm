# TinyLLM Dynamic Model Switching Guide

This guide explains the new dynamic model switching capabilities added to TinyLLM.

## Overview

TinyLLM now supports seamless model switching during chat sessions and programmatic use. This feature allows you to:

- Switch between different LLM models mid-conversation
- Track model health and performance statistics
- Use model aliases for quick switching
- Manage models via CLI commands

## Features

### 1. Model Registry (`ModelRegistry`)

Located at: `/home/uri/Desktop/tinyllm/src/tinyllm/models/registry.py`

The ModelRegistry tracks:
- Available models from Ollama
- Model capabilities (context size, vision support, parameter count)
- Health and latency statistics per model
- Model aliases for quick access

**Key methods:**
```python
from tinyllm.models.registry import get_model_registry

registry = get_model_registry()

# Register a model
registry.register_model(capabilities)

# Get model info
info = registry.get_model_info("qwen2.5:3b")

# Track requests for health monitoring
registry.record_request("qwen2.5:3b", latency_ms=150, success=True)

# Use aliases
registry.resolve_name("fast")  # Returns "qwen2.5:0.5b"
```

**Built-in aliases:**
- `fast` -> `qwen2.5:0.5b`
- `tiny` -> `tinyllama`
- `code` -> `granite-code:3b`
- `medium` -> `qwen2.5:3b`
- `large` -> `qwen3:8b`
- `judge` -> `qwen3:14b`

### 2. Enhanced OllamaClient

The OllamaClient now supports:

**Setting a default model:**
```python
from tinyllm.models import OllamaClient

client = OllamaClient(default_model="qwen2.5:3b")
```

**Switching models:**
```python
# Simple switch
client.set_model("qwen2.5:7b")

# Switch with compatibility check
success, warning = await client.switch_model("qwen2.5:7b")
if not success:
    print(f"Warning: {warning}")
```

**Generate without specifying model (uses default):**
```python
response = await client.generate(
    prompt="Hello, world!"
    # model parameter is now optional if default_model is set
)
```

### 3. Interactive Chat Commands

When using `tinyllm chat`, you can now use these commands:

**`/model <name>`** - Switch to a different model
```
You (qwen2.5:1.5b): /model fast
Switching from qwen2.5:1.5b to qwen2.5:0.5b...
Switched to model: qwen2.5:0.5b
Conversation context preserved
```

**`/models`** - List available models with statistics
```
You (qwen2.5:1.5b): /models

Available Models:
  • qwen2.5:0.5b (100% success, 150ms avg)
  • qwen2.5:1.5b (current) (95% success, 200ms avg)
  • qwen2.5:3b

Aliases:
  fast -> qwen2.5:0.5b
  medium -> qwen2.5:3b
```

**`/help`** - Show all available commands

**`/clear`** - Clear conversation history

**`/quit`** - Exit chat

### 4. CLI Commands

#### List Models
```bash
tinyllm models list
```

Shows a table of all available models with:
- Model name
- Parameter count
- Context size
- Request statistics
- Success rate
- Average latency

#### Pull a Model
```bash
tinyllm models pull qwen2.5:7b
```

Downloads a new model from Ollama registry.

#### Set Default Model
```bash
tinyllm models set-default qwen2.5:3b
```

Sets the default model for TinyLLM operations.

#### Get Model Info
```bash
tinyllm models info qwen2.5:3b
```

Shows detailed information about a specific model:
- Capabilities (family, parameters, context size, vision support)
- Statistics (requests, success rate, latency, health status)

## Usage Examples

### Example 1: Chat with Model Switching

```bash
$ tinyllm chat -m qwen2.5:1.5b

TinyLLM Interactive Chat
Model: qwen2.5:1.5b
Identity: TinyLLM Assistant
Commands:
  • /model <name> - Switch models
  • /models - List available models
  • /help - Show all commands
  • Type 'quit' or 'exit' to leave
  • Type 'clear' to clear history

You (qwen2.5:1.5b): Hello! How are you?
Assistant: I'm doing well, thank you! How can I help you today?

You (qwen2.5:1.5b): /model fast
Switching from qwen2.5:1.5b to qwen2.5:0.5b...
Switched to model: qwen2.5:0.5b
Conversation context preserved

You (qwen2.5:0.5b): Continue our conversation
Assistant: Of course! What would you like to discuss?
```

The conversation history is preserved across model switches, so the new model has context of previous messages.

### Example 2: Programmatic Model Switching

```python
import asyncio
from tinyllm.models import OllamaClient
from tinyllm.memory import MemoryStore

async def chat_with_model_switching():
    client = OllamaClient(default_model="qwen2.5:1.5b")
    memory = MemoryStore()

    # First interaction with default model
    memory.add_message("user", "Explain quantum computing")
    response = await client.generate(
        prompt="Explain quantum computing",
        system="You are a helpful assistant"
    )
    memory.add_message("assistant", response.response)

    # Switch to a larger model for more detailed answer
    await client.switch_model("qwen2.5:7b")

    # Continue conversation with context
    context = memory.get_context_for_prompt()
    response = await client.generate(
        prompt="Can you provide a more detailed explanation?",
        system=f"You are a helpful assistant.\n\nContext:\n{context}"
    )

    await client.close()

asyncio.run(chat_with_model_switching())
```

### Example 3: Using Model Registry for Health Tracking

```python
from tinyllm.models.registry import get_model_registry

registry = get_model_registry()

# Record successful request
registry.record_request("qwen2.5:3b", latency_ms=175, success=True)

# Record failed request
registry.record_request("qwen2.5:3b", latency_ms=0, success=False, error="Timeout")

# Check model health
health = registry.get_health("qwen2.5:3b")
print(f"Success rate: {health.success_rate:.1f}%")
print(f"Average latency: {health.average_latency_ms:.1f}ms")
print(f"Is healthy: {health.is_healthy}")

# Get only healthy models
healthy_models = registry.get_healthy_models()
```

## Architecture

### Components

1. **ModelRegistry** (`src/tinyllm/models/registry.py`)
   - Singleton instance via `get_model_registry()`
   - Tracks model capabilities and health
   - Manages aliases
   - Auto-syncs with Ollama

2. **OllamaClient Updates** (`src/tinyllm/models/client.py`)
   - `_current_model`: Stores default/current model
   - `set_model()`: Switch models
   - `get_model()`: Get current model
   - `switch_model()`: Switch with compatibility check
   - `check_model_compatibility()`: Verify model availability

3. **CLI Enhancements** (`src/tinyllm/cli.py`)
   - Chat command: Added `/model`, `/models`, `/help` commands
   - Model subcommands: `list`, `pull`, `set-default`, `info`
   - Real-time model indicator in prompt

### Model Health Tracking

The registry tracks these metrics per model:
- **Total requests**: Count of all requests
- **Failed requests**: Count of failed requests
- **Total latency**: Cumulative latency in ms
- **Last used**: Timestamp of last request
- **Last error**: Most recent error message
- **Last error time**: When the last error occurred

A model is considered unhealthy if:
- Success rate is below 50%
- Last error occurred within 60 seconds

### Conversation Context Preservation

When switching models during a chat:
1. Conversation history remains in MemoryStore
2. System prompt is updated for the new model
3. Context is passed to the new model on next request
4. Seamless transition without losing conversation state

## Best Practices

1. **Use aliases for quick switching**
   ```
   /model fast   # Instead of /model qwen2.5:0.5b
   ```

2. **Monitor model health**
   - Use `/models` command to see success rates
   - Check average latency to identify slow models

3. **Switch models based on task complexity**
   - Use smaller models (fast, tiny) for simple queries
   - Switch to larger models (medium, large) for complex tasks
   - Use specialized models (code) for specific domains

4. **Leverage conversation context**
   - Context is preserved across switches
   - New model sees previous conversation
   - No need to repeat information

5. **Check compatibility before switching**
   ```python
   success, warning = await client.switch_model("new-model")
   if not success:
       print(f"Model not available: {warning}")
   ```

## Configuration

Model aliases can be customized:

```python
from tinyllm.models.registry import get_model_registry

registry = get_model_registry()

# Add custom alias
registry.add_alias("my-fast-model", "qwen2.5:0.5b")

# Remove alias
registry.remove_alias("my-fast-model")
```

## Files Modified

1. `/home/uri/Desktop/tinyllm/src/tinyllm/models/registry.py` (NEW)
   - ModelRegistry class
   - ModelCapabilities dataclass
   - ModelHealth dataclass
   - get_model_registry() function

2. `/home/uri/Desktop/tinyllm/src/tinyllm/models/client.py` (MODIFIED)
   - Added `default_model` parameter to `__init__`
   - Added `_current_model` attribute
   - Added `set_model()` method
   - Added `get_model()` method
   - Added `switch_model()` method
   - Added `check_model_compatibility()` method
   - Made `model` parameter optional in `generate()` and `generate_stream()`

3. `/home/uri/Desktop/tinyllm/src/tinyllm/cli.py` (MODIFIED)
   - Updated `chat()` command with /model, /models, /help commands
   - Added `models` subcommand group
   - Added `models list` command
   - Added `models pull` command
   - Added `models set-default` command
   - Added `models info` command
   - Renamed old `models` command to `tiers`

4. `/home/uri/Desktop/tinyllm/src/tinyllm/models/__init__.py` (MODIFIED)
   - Exported ModelRegistry, ModelCapabilities, RegistryModelHealth
   - Exported get_model_registry()

## Testing

Run the test script:
```bash
python test_model_switching.py
```

This validates:
- OllamaClient default model support
- Model switching functionality
- ModelRegistry operations
- Model aliases
- Health tracking
- Model info retrieval
- Compatibility checks

## Future Enhancements

Potential improvements:
1. Persistent storage for model health stats
2. Model performance benchmarking
3. Automatic model selection based on query complexity
4. Model warmup and preloading
5. Multi-model consensus voting
6. Cost tracking per model
7. Model-specific prompt templates

## Troubleshooting

**Model not found error:**
```bash
# Pull the model first
tinyllm models pull qwen2.5:3b
```

**Compatibility check fails:**
```bash
# Verify Ollama is running
tinyllm health

# List available models
tinyllm models list
```

**Context not preserved:**
- Context preservation requires MemoryStore
- Make sure not to call `memory.clear_stm()` before switching

## Summary

The dynamic model switching feature provides:
- Seamless model switching in chat sessions
- Health and performance tracking
- Convenient CLI commands for model management
- Conversation context preservation
- Model aliases for quick access

This makes TinyLLM more flexible and allows you to optimize for speed, cost, or quality on a per-query basis.