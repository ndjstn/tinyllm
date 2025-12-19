# Model Identity Fix - TinyLLM

## Summary

Fixed the model identity issue where TinyLLM chat would respond "My name is Claude" when asked "what is your name". The assistant now properly identifies as "TinyLLM Assistant" and includes awareness of which model is running.

## What Changed

### 1. New Prompts Module (`src/tinyllm/prompts/defaults.py`)

Created a comprehensive prompts system with:

- **ASSISTANT_IDENTITY**: Core identity information
- **CHAT_SYSTEM_PROMPT**: Default chat mode prompt
- **TASK_SYSTEM_PROMPT**: Task execution prompt
- **ROUTER_SYSTEM_PROMPT**: Routing/classification prompt
- **SPECIALIST_SYSTEM_PROMPT**: Domain specialist prompt
- **JUDGE_SYSTEM_PROMPT**: Evaluation prompt

All prompts establish clear identity:
- Name: TinyLLM Assistant
- Purpose: Local LLM orchestration system
- Architecture: Neural network of small models
- Runtime: Local via Ollama
- **Clear statement: NOT Claude, ChatGPT, or other commercial assistants**

### 2. Updated CLI (`src/tinyllm/cli.py`)

Modified the `chat` command to:

1. Use proper identity prompts by default
2. Support custom system prompts via `--system` flag
3. Auto-detect and correct identity confusion
4. Display model and identity information on startup

**New features:**
```bash
# Default with proper identity
tinyllm chat

# Custom system prompt
tinyllm chat --system "You are a helpful coding assistant."
```

### 3. Identity Correction System

Automatic detection and correction of identity questions:

- "what is your name" → Explains TinyLLM Assistant identity
- "are you claude" → "I'm TinyLLM Assistant, not Claude..."
- "who are you" → Provides full identity description

This happens **before** calling the LLM, ensuring consistent responses.

### 4. Configurable Identity

New `PromptConfig` class for customization:

```python
from tinyllm.prompts import PromptConfig, set_default_config

config = PromptConfig(
    assistant_name="MyBot",
    custom_identity="Custom description",
    custom_chat_prompt="Custom prompt...",
    enable_identity_correction=True
)

set_default_config(config)
```

### 5. Model Awareness

Prompts include which model is running:

```
Current Model: qwen2.5:1.5b (running locally via Ollama)
```

The assistant now knows and can reference its underlying model.

## Files Changed

1. **Created:**
   - `/src/tinyllm/prompts/defaults.py` - Default system prompts
   - `/docs/identity-system.md` - Comprehensive documentation
   - `/test_identity.py` - Test script for verification

2. **Modified:**
   - `/src/tinyllm/cli.py` - Updated chat command
   - `/src/tinyllm/prompts/__init__.py` - Export new defaults
   - `/src/tinyllm/__init__.py` - Package-level exports

## Testing

Run the test script to verify:

```bash
PYTHONPATH=src:$PYTHONPATH python test_identity.py
```

**Test results:**
```
✓ ASSISTANT_IDENTITY defined correctly
✓ CHAT_SYSTEM_PROMPT includes proper identity
✓ Model-specific prompts include model info
✓ Identity correction works for common queries
✓ No unwanted assistant names found
✓ TinyLLM branding present in all prompts
```

## Before vs After

### Before

**User:** what is your name
**Assistant:** My name is Claude.

**Issues:**
- Generic "helpful assistant" prompt
- No TinyLLM branding
- No model awareness
- Identity confusion with commercial assistants

### After

**User:** what is your name
**Assistant:** I'm TinyLLM Assistant, a local LLM orchestration system. I'm currently powered by qwen2.5:1.5b running through Ollama. I coordinate multiple small language models to handle different tasks efficiently.

**User:** are you claude?
**Assistant:** I'm TinyLLM Assistant, not Claude. I'm a local LLM orchestration system.

**Benefits:**
- ✅ Proper TinyLLM identity
- ✅ Model awareness
- ✅ Clear about local execution
- ✅ Explains architecture
- ✅ Auto-corrects identity confusion

## Example Usage

### Basic Chat

```bash
$ tinyllm chat
TinyLLM Chat
Model: qwen2.5:1.5b
Identity: TinyLLM Assistant
Type 'quit' to exit, 'clear' to clear history

You: what is your name
Assistant: I'm TinyLLM Assistant, a local LLM orchestration system...

You: what can you do?
Assistant: I coordinate multiple small language models to handle different tasks...
```

### Custom System Prompt

```bash
$ tinyllm chat --system "You are a Python coding expert."
TinyLLM Chat
Model: qwen2.5:1.5b
Identity: TinyLLM Assistant
...
```

### Programmatic Usage

```python
from tinyllm.prompts import get_chat_prompt, get_identity_correction
from tinyllm.models import OllamaClient

model = "qwen2.5:1.5b"
client = OllamaClient()

# Get proper identity prompt
system_prompt = get_chat_prompt(model)

# Check for identity questions
user_input = "what is your name"
correction = get_identity_correction(user_input)

if correction:
    print(f"Assistant: {correction}")
else:
    # Normal LLM call with proper prompt
    response = await client.generate(
        model=model,
        prompt=user_input,
        system=system_prompt
    )
```

## Key Features

### 1. Default Prompts
- Carefully crafted identity statements
- Clear about NOT being commercial assistants
- Include model and architecture information
- Consistent across all modes (chat, task, router, etc.)

### 2. Identity Correction
- Detects identity questions automatically
- Provides immediate, consistent responses
- No LLM call needed for basic identity queries
- Saves tokens and ensures accuracy

### 3. Model Awareness
- Prompts include specific model name
- Assistant knows which model is running
- Can discuss model capabilities appropriately

### 4. Configurability
- Override defaults when needed
- Custom assistant names
- Custom identity descriptions
- Custom full prompts
- Per-task-type prompts

### 5. Integration
- Works with existing chat command
- Compatible with graph nodes
- Exports at package level
- Backward compatible

## API Reference

### Functions

```python
# Get chat prompt with model info
get_chat_prompt(model_name: str, include_context: bool = True) -> str

# Get task-specific prompt
get_task_prompt(task_type: Optional[str] = None, model_name: Optional[str] = None) -> str

# Check for identity confusion
get_identity_correction(query: str) -> Optional[str]

# Configuration
get_default_config() -> PromptConfig
set_default_config(config: PromptConfig) -> None
```

### Constants

```python
ASSISTANT_IDENTITY        # Core identity description
CHAT_SYSTEM_PROMPT        # Default chat prompt
TASK_SYSTEM_PROMPT        # Task execution prompt
ROUTER_SYSTEM_PROMPT      # Router classification prompt
SPECIALIST_SYSTEM_PROMPT  # Domain specialist prompt
JUDGE_SYSTEM_PROMPT       # Evaluation prompt
```

### Classes

```python
class PromptConfig:
    def __init__(
        self,
        assistant_name: str = "TinyLLM Assistant",
        custom_identity: Optional[str] = None,
        custom_chat_prompt: Optional[str] = None,
        custom_task_prompt: Optional[str] = None,
        enable_identity_correction: bool = True,
    )

    def get_chat_prompt(self, model_name: str, context: Optional[str] = None) -> str
    def get_task_prompt(self, task_type: Optional[str] = None, model_name: Optional[str] = None) -> str
```

## Documentation

Full documentation available at:
- `/docs/identity-system.md` - Complete guide with examples
- This file - Quick reference for the fix

## Backward Compatibility

The fix is **backward compatible**:

1. **Existing graph YAML files work unchanged**
   - Nodes without system_prompt use new defaults
   - Nodes with system_prompt continue to use custom prompts

2. **Existing Python code works unchanged**
   - Old `client.generate()` calls work as before
   - New imports are optional

3. **CLI changes are additive**
   - Chat command has same defaults
   - New `--system` flag is optional

## Verification

To verify the fix works:

1. **Run the test:**
   ```bash
   PYTHONPATH=src:$PYTHONPATH python test_identity.py
   ```

2. **Try the chat:**
   ```bash
   tinyllm chat
   # Ask: "what is your name"
   # Should respond: "I'm TinyLLM Assistant..."
   ```

3. **Check identity correction:**
   ```python
   from tinyllm.prompts import get_identity_correction
   print(get_identity_correction("are you claude"))
   # Should print correction message
   ```

## Next Steps

### Recommended Actions

1. **Update any hardcoded prompts** in your code:
   ```python
   # Old
   system = "You are a helpful assistant."

   # New
   from tinyllm.prompts import get_chat_prompt
   system = get_chat_prompt(model_name)
   ```

2. **Test with your models** to ensure identity works correctly

3. **Customize if needed** using PromptConfig for specialized use cases

### Future Enhancements

Potential improvements:
- YAML-based prompt configuration files
- Per-model prompt templates
- Prompt versioning and A/B testing
- Community prompt library
- Multi-language support

## Support

For issues or questions:
1. Check `/docs/identity-system.md` for detailed documentation
2. Run `test_identity.py` to verify installation
3. Review examples in the documentation
4. File an issue with test results if problems persist

## License

This fix is part of TinyLLM and follows the same license.
