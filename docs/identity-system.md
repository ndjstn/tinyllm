# TinyLLM Identity System

## Overview

The TinyLLM Identity System ensures that the assistant properly identifies itself as "TinyLLM Assistant" rather than claiming to be Claude, ChatGPT, or other commercial assistants. This system provides:

1. **Default system prompts** with proper identity information
2. **Configurable prompts** for customization
3. **Automatic identity correction** when users ask "what is your name"
4. **Model awareness** - the assistant knows which model is running

## Quick Start

### Using the Chat Command

The chat command now automatically uses the proper identity:

```bash
# Default chat with proper identity
tinyllm chat

# With custom model
tinyllm chat --model qwen2.5:3b

# With custom system prompt
tinyllm chat --system "You are a helpful coding assistant."
```

### Sample Interaction

```
You: what is your name
Assistant: I'm TinyLLM Assistant, a local LLM orchestration system. I'm currently
powered by qwen2.5:1.5b running through Ollama. I coordinate multiple small language
models to handle different tasks efficiently.

You: are you claude?
Assistant: I'm TinyLLM Assistant, not Claude. I'm a local LLM orchestration system.

You: hello!
Assistant: Hello! How can I help you today?
```

## Architecture

### Default Prompts

The system provides several default prompts in `/src/tinyllm/prompts/defaults.py`:

#### 1. ASSISTANT_IDENTITY
Core identity information used across all modes:
- Name: TinyLLM Assistant
- Purpose: Local LLM orchestration
- Architecture: Neural network of specialized models
- Runtime: Local via Ollama

#### 2. CHAT_SYSTEM_PROMPT
Default prompt for interactive chat sessions. Includes:
- Identity declaration
- Core principles (helpful, honest, accurate)
- Behavior guidelines
- Conversation context handling

#### 3. TASK_SYSTEM_PROMPT
Prompt for task execution nodes in graphs:
- Focused on task completion
- Minimal preamble
- Structured output support
- Part of larger workflow

#### 4. Specialized Prompts
- `ROUTER_SYSTEM_PROMPT` - For routing/classification nodes
- `SPECIALIST_SYSTEM_PROMPT` - For domain-specific nodes
- `JUDGE_SYSTEM_PROMPT` - For evaluation nodes

### Identity Correction

The system automatically detects and corrects identity confusion:

```python
from tinyllm.prompts import get_identity_correction

# Returns correction message if identity question detected
correction = get_identity_correction("what is your name")
# Returns: "I'm TinyLLM Assistant, a local LLM orchestration system..."

correction = get_identity_correction("are you chatgpt")
# Returns: "I'm TinyLLM Assistant, not ChatGPT..."

correction = get_identity_correction("hello")
# Returns: None (no correction needed)
```

Detected patterns:
- "what is your name"
- "who are you"
- "are you [assistant_name]"
- "is this [assistant_name]"

Recognized assistant names: claude, chatgpt, gpt, openai, anthropic, gemini, bard

## Configuration

### Using PromptConfig

Customize the identity system programmatically:

```python
from tinyllm.prompts import PromptConfig, set_default_config

# Create custom configuration
config = PromptConfig(
    assistant_name="MyBot",
    custom_identity="A specialized bot for my use case.",
    custom_chat_prompt="You are MyBot, a helpful assistant.",
    enable_identity_correction=True
)

# Set as default
set_default_config(config)

# Get configured prompt
prompt = config.get_chat_prompt(model_name="qwen2.5:1.5b")
```

### PromptConfig Options

```python
PromptConfig(
    assistant_name: str = "TinyLLM Assistant",
    custom_identity: Optional[str] = None,
    custom_chat_prompt: Optional[str] = None,
    custom_task_prompt: Optional[str] = None,
    enable_identity_correction: bool = True,
)
```

- `assistant_name`: Name the assistant uses for itself
- `custom_identity`: Override default identity description
- `custom_chat_prompt`: Override entire chat system prompt
- `custom_task_prompt`: Override task execution prompt
- `enable_identity_correction`: Enable/disable auto-correction

### Model-Specific Prompts

Generate prompts that include model information:

```python
from tinyllm.prompts import get_chat_prompt, get_task_prompt

# Chat prompt with model info
chat_prompt = get_chat_prompt("qwen2.5:1.5b")
# Includes: "Current Model: qwen2.5:1.5b (running locally via Ollama)"

# Task prompt with model and type
task_prompt = get_task_prompt(task_type="code", model_name="granite-code:3b")
# Includes code-specific instructions
```

### Task-Specific Prompts

Supported task types:
- `"code"` - Code generation with documentation
- `"analysis"` - Evidence-based analysis
- `"summary"` - Key point extraction
- `"classification"` - Accurate categorization
- `"extraction"` - Information extraction

## Integration with Nodes

The identity system integrates with TinyLLM's node system:

### Model Nodes

Model nodes can use the default prompts:

```yaml
nodes:
  - id: chat_node
    type: model
    config:
      model: qwen2.5:1.5b
      # No system_prompt specified - uses default with proper identity
```

Or override with custom prompts:

```yaml
nodes:
  - id: custom_node
    type: model
    config:
      model: qwen2.5:3b
      system_prompt: "You are a code review specialist..."
```

### Router Nodes

Routers use `ROUTER_SYSTEM_PROMPT`:

```python
from tinyllm.prompts import ROUTER_SYSTEM_PROMPT

# Router gets classification-focused prompt
# Doesn't claim to be any commercial assistant
```

### Specialist Nodes

Specialists use `SPECIALIST_SYSTEM_PROMPT`:

```python
from tinyllm.prompts import SPECIALIST_SYSTEM_PROMPT

# Domain-specific expertise
# Stays within assigned specialty
# Still identifies as TinyLLM Assistant
```

## Best Practices

### 1. Use Default Prompts When Possible

The defaults are carefully crafted to:
- Establish proper identity
- Prevent confusion with commercial assistants
- Include model awareness
- Provide clear behavioral guidelines

```python
# Good - uses defaults
from tinyllm.prompts import get_chat_prompt
prompt = get_chat_prompt(model_name)

# Also good - explicit defaults
from tinyllm.prompts import CHAT_SYSTEM_PROMPT
```

### 2. Customize Only When Needed

Override prompts for specific use cases:

```python
# Custom for specialized chatbot
config = PromptConfig(
    custom_chat_prompt="""You are CodeBot, a TinyLLM-powered coding assistant.

    You specialize in:
    - Code review
    - Bug fixing
    - Best practices

    Powered by TinyLLM's local model orchestration."""
)
```

### 3. Always Include Model Information

When creating custom prompts, include model awareness:

```python
custom_prompt = f"""You are MyAssistant.

Current Model: {model_name} (local via Ollama)
Part of the TinyLLM system.

..."""
```

### 4. Maintain Identity Clarity

If customizing, keep these principles:
- ✅ State what you ARE
- ✅ Clarify what you're NOT (if relevant)
- ✅ Mention local execution
- ✅ Include TinyLLM attribution
- ❌ Don't claim to be commercial assistants
- ❌ Don't omit model information

## Testing

Test the identity system:

```bash
# Run the test script
PYTHONPATH=src:$PYTHONPATH python test_identity.py
```

The test verifies:
1. Identity strings are correctly formatted
2. Model-specific prompts include model info
3. Identity correction works for common queries
4. No unwanted assistant names (except in negation)
5. TinyLLM branding is present
6. Custom configuration works

## API Reference

### Functions

#### `get_chat_prompt(model_name: str, include_context: bool = True) -> str`
Get chat system prompt with model information.

**Parameters:**
- `model_name`: Name of the underlying model (e.g., "qwen2.5:1.5b")
- `include_context`: Whether to mention conversation context

**Returns:**
- Complete chat system prompt string

#### `get_task_prompt(task_type: Optional[str] = None, model_name: Optional[str] = None) -> str`
Get task-specific system prompt.

**Parameters:**
- `task_type`: Type of task ("code", "analysis", "summary", etc.)
- `model_name`: Name of the underlying model

**Returns:**
- Task-specific system prompt string

#### `get_identity_correction(query: str) -> Optional[str]`
Check if query asks about identity and provide correction.

**Parameters:**
- `query`: User query to check

**Returns:**
- Correction string if identity confusion detected, None otherwise

#### `get_default_config() -> PromptConfig`
Get the current default prompt configuration.

#### `set_default_config(config: PromptConfig) -> None`
Set the default prompt configuration.

### Constants

- `ASSISTANT_IDENTITY` - Core identity description
- `CHAT_SYSTEM_PROMPT` - Default chat prompt
- `TASK_SYSTEM_PROMPT` - Default task prompt
- `ROUTER_SYSTEM_PROMPT` - Router classification prompt
- `SPECIALIST_SYSTEM_PROMPT` - Specialist domain prompt
- `JUDGE_SYSTEM_PROMPT` - Evaluation judge prompt

## Examples

### Example 1: Basic Chat

```python
from tinyllm.memory import MemoryStore
from tinyllm.models import OllamaClient
from tinyllm.prompts import get_chat_prompt

model = "qwen2.5:1.5b"
client = OllamaClient()
memory = MemoryStore()

# Get proper identity prompt
system_prompt = get_chat_prompt(model)

# User asks about identity
user_input = "what is your name"
context = memory.get_context_for_prompt()

response = await client.generate(
    model=model,
    prompt=user_input,
    system=system_prompt + f"\n\nContext:\n{context}"
)

# Response will correctly identify as TinyLLM Assistant
```

### Example 2: Custom Specialist Bot

```python
from tinyllm.prompts import PromptConfig

# Create specialized configuration
config = PromptConfig(
    assistant_name="TinyLLM Code Assistant",
    custom_chat_prompt="""You are TinyLLM Code Assistant, specialized for coding tasks.

**Identity:**
- Name: TinyLLM Code Assistant
- Specialization: Code review, debugging, best practices
- Runtime: Local via TinyLLM orchestration system

**Capabilities:**
- Code analysis and review
- Bug detection and fixes
- Best practice recommendations
- Documentation generation

You are NOT Claude, ChatGPT, or other commercial assistants.
You are part of the TinyLLM local system."""
)

# Use in chat
prompt = config.get_chat_prompt(model_name="granite-code:3b")
```

### Example 3: Identity Correction in CLI

```python
# In cli.py chat function
from tinyllm.prompts import get_identity_correction

user_input = console.input("You: ")

# Check for identity questions
correction = get_identity_correction(user_input)
if correction:
    console.print(f"Assistant: {correction}")
    memory.add_message("user", user_input)
    memory.add_message("assistant", correction)
    continue  # Skip normal LLM call
```

## Troubleshooting

### Issue: Assistant still says "I'm Claude"

**Cause:** Custom system prompt overriding defaults

**Solution:** Check that you're using the default prompts or including proper identity in custom prompts:

```python
# Wrong
system = "You are a helpful assistant."

# Right
from tinyllm.prompts import get_chat_prompt
system = get_chat_prompt(model_name)

# Or if custom
system = "You are TinyLLM Assistant. I'm NOT Claude..."
```

### Issue: Model doesn't mention which model it's using

**Cause:** Using basic CHAT_SYSTEM_PROMPT instead of get_chat_prompt()

**Solution:**
```python
# Wrong
from tinyllm.prompts import CHAT_SYSTEM_PROMPT
system = CHAT_SYSTEM_PROMPT

# Right
from tinyllm.prompts import get_chat_prompt
system = get_chat_prompt(model_name="qwen2.5:1.5b")
```

### Issue: Identity correction not working

**Cause:** Using custom configuration with `enable_identity_correction=False`

**Solution:**
```python
config = PromptConfig(
    enable_identity_correction=True  # Ensure this is True
)
```

## Migration Guide

If you have existing code using hardcoded prompts:

### Before
```python
system = "You are a helpful assistant."
```

### After
```python
from tinyllm.prompts import get_chat_prompt
system = get_chat_prompt(model_name="qwen2.5:1.5b")
```

### Before (CLI)
```python
system=f"You are a helpful assistant.\n\nConversation context:\n{context}"
```

### After (CLI)
```python
from tinyllm.prompts import get_chat_prompt

base_prompt = get_chat_prompt(model)
system = base_prompt
if context:
    system += f"\n\nConversation context:\n{context}"
```

## Future Enhancements

Planned improvements:
- YAML-based prompt configuration
- Per-model prompt templates
- Prompt versioning and tracking
- A/B testing for prompt effectiveness
- Community prompt library
- Multi-language identity support
