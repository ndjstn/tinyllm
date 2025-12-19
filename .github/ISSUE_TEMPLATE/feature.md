---
name: Feature Implementation
about: Implement a new feature or component
title: '[FEAT] '
labels: enhancement
assignees: ''
---

## Overview

Brief description of what needs to be implemented.

## Specification

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `path/to/file.py` | Create | Description |

### Interface

```python
# Input/Output types (Pydantic models)
class Input(BaseModel):
    field: str

class Output(BaseModel):
    result: str
```

### Behavior

1. Step one
2. Step two
3. Step three

### Error Cases

- Error case 1 → Handle by...
- Error case 2 → Handle by...

## Test Cases

| Input | Expected Output |
|-------|-----------------|
| `example1` | `result1` |
| `example2` | `result2` |

## Acceptance Criteria

- [ ] All tests pass
- [ ] Type checking passes (`mypy`)
- [ ] Linting passes (`ruff`)
- [ ] Documentation updated if needed

## Context

- **Phase**: `phase-X`
- **Dependencies**: #issue1, #issue2
- **Blocked by**: None

## Resources

- [Relevant spec](../docs/specs/relevant.md)
- [Architecture overview](../docs/ARCHITECTURE.md)
