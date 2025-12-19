---
name: Prompt Template
about: Create or improve a prompt template
title: '[PROMPT] '
labels: prompt-engineering
assignees: ''
---

## Overview

Brief description of the prompt's purpose.

## Prompt Details

### Category

- [ ] Routing
- [ ] Specialist
- [ ] Thinking
- [ ] Tools
- [ ] Grading
- [ ] Meta
- [ ] Memory

### File Location

`prompts/<category>/<name>.yaml`

### Target Models

- Primary: `model-name`
- Fallback: `model-name`

## Requirements

### Inputs

| Variable | Type | Description |
|----------|------|-------------|
| `{{task}}` | string | The user's task |

### Expected Outputs

```yaml
output_format: JSON_SCHEMA  # or TEXT, STRUCTURED
schema:
  type: object
  properties:
    field: { type: string }
```

### Behavior

1. What the prompt should do
2. What to avoid
3. Edge cases to handle

## Test Cases

| Input | Expected Behavior |
|-------|-------------------|
| Example input 1 | Should output X |
| Example input 2 | Should output Y |

## Acceptance Criteria

- [ ] Prompt follows YAML schema in [prompt spec](../docs/specs/prompts.md)
- [ ] Tested with target model
- [ ] Works with low temperature (0.1-0.3)
- [ ] Handles edge cases gracefully

## Context

- **Phase**: `phase-X`
- **Related prompts**: `prompt-id-1`, `prompt-id-2`
