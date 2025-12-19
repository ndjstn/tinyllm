"""Prompt management for TinyLLM."""

from tinyllm.prompts.loader import PromptLoader, load_prompt
from tinyllm.prompts.schema import (
    OutputFormat,
    PromptCategory,
    PromptDefinition,
    PromptExample,
)
from tinyllm.prompts.defaults import (
    ASSISTANT_IDENTITY,
    CHAT_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PromptConfig,
    get_chat_prompt,
    get_task_prompt,
    get_identity_correction,
    get_default_config,
    set_default_config,
)

__all__ = [
    "PromptLoader",
    "load_prompt",
    "OutputFormat",
    "PromptCategory",
    "PromptDefinition",
    "PromptExample",
    "ASSISTANT_IDENTITY",
    "CHAT_SYSTEM_PROMPT",
    "TASK_SYSTEM_PROMPT",
    "ROUTER_SYSTEM_PROMPT",
    "SPECIALIST_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "PromptConfig",
    "get_chat_prompt",
    "get_task_prompt",
    "get_identity_correction",
    "get_default_config",
    "set_default_config",
]
