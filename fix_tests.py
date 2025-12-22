#!/usr/bin/env python3
"""Fix test files to match actual ErrorContext/TinyLLMError structures."""

import re

# Read test_error_aggregation.py
with open("tests/unit/test_error_aggregation.py", "r") as f:
    agg_content = f.read()

# Fix ErrorContext creation - add required fields
agg_content = re.sub(
    r'ErrorContext\(([\s\S]*?)\)',
    lambda m: f'''ErrorContext(
                stack_trace="test stack trace",
                exception_type="{re.search(r'exception_type="([^"]+)"', m.group(1)).group(1) if 'exception_type' in m.group(1) else 'RuntimeError'}",
                exception_message="test exception message",
                {m.group(1).strip()}
            )''',
    agg_content
)

# Write back
with open("tests/unit/test_error_aggregation.py", "w") as f:
    f.write(agg_content)

print("Fixed test_error_aggregation.py")

# Fix test_error_impact.py
with open("tests/unit/test_error_impact.py", "r") as f:
    impact_content = f.read()

# Add required fields to all ErrorContext creations
impact_content = re.sub(
    r'context=ErrorContext\(([\s\S]*?)\)',
    lambda m: f'''context=ErrorContext(
                stack_trace="test stack trace",
                exception_type="{re.search(r'exception_type="([^"]+)"', m.group(1)).group(1) if 'exception_type' in m.group(1) else 'RuntimeError'}",
                exception_message="test exception message",
                {m.group(1).strip()}
            )''',
    impact_content
)

with open("tests/unit/test_error_impact.py", "w") as f:
    f.write(impact_content)

print("Fixed test_error_impact.py")

# Fix test_error_branching.py - use TinyLLMError without category/severity
with open("tests/unit/test_error_branching.py", "r") as f:
    branch_content = f.read()

# Remove TinyLLMError usages with category/severity - just use code and recoverable
branch_content = re.sub(
    r'TinyLLMError\(([^)]*?)category=ErrorCategory\.[A-Z_]+,([^)]*?)\)',
    lambda m: f'TinyLLMError({m.group(1).strip()}{m.group(2).strip()})',
    branch_content
)

branch_content = re.sub(
    r'TinyLLMError\(([^)]*?)severity=ErrorSeverity\.[A-Z_]+,([^)]*?)\)',
    lambda m: f'TinyLLMError({m.group(1).strip()}{m.group(2).strip()})',
    branch_content
)

# Clean up any remaining double commas
branch_content = re.sub(r',\s*,', ',', branch_content)

with open("tests/unit/test_error_branching.py", "w") as f:
    f.write(branch_content)

print("Fixed test_error_branching.py")
print("Done!")
