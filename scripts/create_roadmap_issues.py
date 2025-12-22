#!/usr/bin/env python3
"""
Create GitHub issues from ROADMAP_500.md

Excludes cloud provider tasks (121-132, 136-138) per local-only philosophy.
"""

import re
import subprocess
import sys
from pathlib import Path


# Tasks to exclude (cloud providers - violates local-only philosophy)
EXCLUDED_TASKS = set(range(121, 133)) | {136, 137, 138}  # 121-132, 136-138

# Priority mapping
PRIORITIES = {
    range(1, 16): "p0",      # Error handling
    range(16, 31): "p0",     # Testing
    range(121, 141): "p0",   # Providers (most excluded)
    range(31, 51): "p1",     # Observability
    range(51, 71): "p1",     # Execution
    range(261, 281): "p1",   # Agent capabilities
}

# Phase mapping
PHASES = {
    range(1, 51): "phase-1",
    range(51, 121): "phase-2",
    range(121, 181): "phase-3",
    range(181, 261): "phase-4",
    range(261, 341): "phase-5",
    range(341, 421): "phase-6",
    range(421, 501): "phase-7",
}

# Category mapping
CATEGORIES = {
    range(1, 16): "error-handling",
    range(16, 31): "testing",
    range(31, 51): "observability",
    range(51, 91): "performance",
    range(121, 181): "model-management",
    range(181, 261): "graph",
    range(261, 341): "agent",
    range(301, 321): "tooling",
    range(321, 341): "memory",
    range(341, 401): "ui-ux",
    range(421, 441): "security",
}


def get_priority(task_num):
    """Get priority label for task number."""
    for range_obj, priority in PRIORITIES.items():
        if task_num in range_obj:
            return f"priority:{priority}"
    return "priority:p2"  # default


def get_phase(task_num):
    """Get phase label for task number."""
    for range_obj, phase in PHASES.items():
        if task_num in range_obj:
            return phase
    return "phase-1"  # default


def get_category(task_num):
    """Get category label for task number."""
    for range_obj, category in CATEGORIES.items():
        if task_num in range_obj:
            return category
    return "enhancement"  # default


def parse_roadmap():
    """Parse ROADMAP_500.md and extract tasks."""
    roadmap_path = Path(__file__).parent.parent / "ROADMAP_500.md"
    content = roadmap_path.read_text()

    tasks = []
    current_section = ""
    current_phase = ""

    # Pattern to match task lines: - [ ] 123. Task description
    task_pattern = re.compile(r'^- \[ \] (\d+)\.\s+(.+)$')

    # Pattern to match section headers
    section_pattern = re.compile(r'^###\s+\d+\.\d+\s+(.+)\s+\((\d+)-(\d+)\)$')
    phase_pattern = re.compile(r'^##\s+Phase\s+\d+:\s+(.+)\s+\(Tasks\s+\d+-\d+\)')

    for line in content.split('\n'):
        # Track current phase
        phase_match = phase_pattern.match(line)
        if phase_match:
            current_phase = phase_match.group(1)
            continue

        # Track current section
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        # Parse task
        task_match = task_pattern.match(line)
        if task_match:
            task_num = int(task_match.group(1))
            task_desc = task_match.group(2)

            # Skip excluded tasks
            if task_num in EXCLUDED_TASKS:
                print(f"⏭️  Skipping task {task_num}: {task_desc} (cloud provider)")
                continue

            tasks.append({
                'number': task_num,
                'description': task_desc,
                'section': current_section,
                'phase': current_phase,
            })

    return tasks


def create_issue(task):
    """Create a GitHub issue for a task."""
    task_num = task['number']
    title = f"[Task {task_num}] {task['description']}"

    # Build labels
    labels = [
        "enhancement",
        get_priority(task_num),
        get_phase(task_num),
        get_category(task_num),
    ]
    labels_str = ",".join(labels)

    # Build body
    body = f"""## Task {task_num}: {task['description']}

**Phase**: {task['phase']}
**Section**: {task['section']}
**Priority**: {get_priority(task_num).replace('priority:', '').upper()}

### Description
{task['description']}

### Acceptance Criteria
- [ ] Implementation complete
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Code reviewed

### Related
See ROADMAP_500.md for full context and dependencies.

---
*Auto-generated from ROADMAP_500.md*
"""

    # Create issue using gh CLI
    cmd = [
        "gh", "issue", "create",
        "--title", title,
        "--body", body,
        "--label", labels_str,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        issue_url = result.stdout.strip()
        print(f"✅ Created: {issue_url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create issue {task_num}: {e.stderr}")
        return False


def main():
    """Main entry point."""
    # Check for --yes flag
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

    # Check for --limit flag
    limit = None
    for arg in sys.argv:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    print("📋 Parsing ROADMAP_500.md...")
    tasks = parse_roadmap()

    if limit:
        tasks = tasks[:limit]
        print(f"\n📊 Creating {len(tasks)} tasks (limited, excluded {len(EXCLUDED_TASKS)} cloud provider tasks)")
    else:
        print(f"\n📊 Found {len(tasks)} tasks to create (excluded {len(EXCLUDED_TASKS)} cloud provider tasks)")
    print(f"   Excluded tasks: {sorted(EXCLUDED_TASKS)}\n")

    # Confirm before creating
    if not auto_confirm:
        response = input("Create all issues? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1

    print("\n🚀 Creating issues...\n")

    created = 0
    failed = 0

    for task in tasks:
        if create_issue(task):
            created += 1
        else:
            failed += 1

    print(f"\n✨ Complete!")
    print(f"   Created: {created}")
    print(f"   Failed: {failed}")
    print(f"   Total: {created + failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
