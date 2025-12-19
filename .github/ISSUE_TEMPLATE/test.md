---
name: Test Cases
about: Add test cases for a component
title: '[TEST] '
labels: testing
assignees: ''
---

## Overview

What component/feature these tests cover.

## Test Type

- [ ] Unit test
- [ ] Integration test
- [ ] Seed test (gold standard)
- [ ] Regression test

## Test Location

`tests/<category>/test_<component>.py` or `tests/seed/<category>/<name>.yaml`

## Test Cases to Add

### Case 1: [Name]

```python
def test_case_name():
    # Setup
    input = ...
    expected = ...

    # Execute
    result = component.method(input)

    # Assert
    assert result == expected
```

### Case 2: [Name]

```python
def test_another_case():
    ...
```

## For Seed Tests (YAML format)

```yaml
tests:
  - id: test-001
    name: "Descriptive name"
    input:
      content: "User input"
    expected:
      route: "expected_route"  # or other expected fields
    tags: [category, difficulty]
```

## Coverage Requirements

- [ ] Happy path covered
- [ ] Error cases covered
- [ ] Edge cases covered
- [ ] Boundary conditions covered

## Acceptance Criteria

- [ ] All tests pass
- [ ] Tests are deterministic (no flakiness)
- [ ] Tests run in < 5 seconds (unit) or marked appropriately
- [ ] Clear test names describing what's tested
