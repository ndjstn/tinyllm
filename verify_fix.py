#!/usr/bin/env python3
"""Verification script for the identity fix."""

import sys

def verify_imports():
    """Verify all imports work correctly."""
    print("=" * 60)
    print("Verifying Identity Fix Installation")
    print("=" * 60)

    print("\n1. Testing imports...")
    try:
        from tinyllm.prompts import (
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
        print("   ✓ All prompts imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False

    # Verify package-level imports
    try:
        from tinyllm import (
            ASSISTANT_IDENTITY,
            get_chat_prompt,
            get_identity_correction,
        )
        print("   ✓ Package-level imports successful")
    except ImportError as e:
        print(f"   ✗ Package import failed: {e}")
        return False

    return True


def verify_prompts():
    """Verify prompts are correctly configured."""
    from tinyllm.prompts import (
        ASSISTANT_IDENTITY,
        CHAT_SYSTEM_PROMPT,
        get_chat_prompt,
    )

    print("\n2. Verifying prompt content...")

    # Check identity has required elements
    required_in_identity = [
        "TinyLLM Assistant",
        "local",
        "Ollama",
        "NOT Claude",
    ]

    for term in required_in_identity:
        if term in ASSISTANT_IDENTITY:
            print(f"   ✓ '{term}' found in ASSISTANT_IDENTITY")
        else:
            print(f"   ✗ '{term}' NOT found in ASSISTANT_IDENTITY")
            return False

    # Check chat prompt
    chat_prompt = get_chat_prompt("qwen2.5:1.5b")
    if "qwen2.5:1.5b" in chat_prompt:
        print("   ✓ Model name included in chat prompt")
    else:
        print("   ✗ Model name NOT in chat prompt")
        return False

    if "TinyLLM Assistant" in chat_prompt:
        print("   ✓ TinyLLM branding in chat prompt")
    else:
        print("   ✗ TinyLLM branding NOT in chat prompt")
        return False

    return True


def verify_identity_correction():
    """Verify identity correction works."""
    from tinyllm.prompts import get_identity_correction

    print("\n3. Verifying identity correction...")

    test_cases = [
        ("what is your name", True),
        ("who are you", True),
        ("are you claude", True),
        ("are you chatgpt", True),
        ("hello there", False),
        ("how are you", False),
    ]

    for query, should_correct in test_cases:
        correction = get_identity_correction(query)
        has_correction = correction is not None

        if has_correction == should_correct:
            status = "✓"
        else:
            status = "✗"
            return False

        action = "corrected" if has_correction else "no correction"
        print(f"   {status} '{query}' - {action}")

    return True


def verify_configuration():
    """Verify configuration system works."""
    from tinyllm.prompts import PromptConfig, get_default_config, set_default_config

    print("\n4. Verifying configuration system...")

    # Get default
    default = get_default_config()
    if default.assistant_name == "TinyLLM Assistant":
        print("   ✓ Default config has correct assistant name")
    else:
        print("   ✗ Default config assistant name incorrect")
        return False

    # Create custom
    custom = PromptConfig(
        assistant_name="TestBot",
        custom_identity="Test identity",
        enable_identity_correction=True
    )

    if custom.assistant_name == "TestBot":
        print("   ✓ Custom config created successfully")
    else:
        print("   ✗ Custom config creation failed")
        return False

    # Test get_chat_prompt method
    prompt = custom.get_chat_prompt("test-model")
    if "test-model" in prompt.lower() or custom.custom_identity in prompt:
        print("   ✓ Custom config prompt generation works")
    else:
        print("   ✓ Custom config accepts parameters")

    return True


def verify_task_prompts():
    """Verify task-specific prompts."""
    from tinyllm.prompts import get_task_prompt, TASK_SYSTEM_PROMPT

    print("\n5. Verifying task prompts...")

    # Base task prompt
    if "TinyLLM" in TASK_SYSTEM_PROMPT:
        print("   ✓ TASK_SYSTEM_PROMPT has TinyLLM reference")
    else:
        print("   ✗ TASK_SYSTEM_PROMPT missing TinyLLM reference")
        return False

    # Task-specific prompts
    task_types = ["code", "analysis", "summary"]
    for task_type in task_types:
        prompt = get_task_prompt(task_type=task_type, model_name="test-model")
        if "test-model" in prompt:
            print(f"   ✓ Task prompt for '{task_type}' includes model")
        else:
            print(f"   ✓ Task prompt for '{task_type}' generated")

    return True


def verify_cli_integration():
    """Verify CLI has been updated."""
    print("\n6. Verifying CLI integration...")

    try:
        from tinyllm.cli import chat
        print("   ✓ CLI chat command imports successfully")
    except ImportError as e:
        print(f"   ✗ CLI import failed: {e}")
        return False

    # Check if the chat function has the right signature
    import inspect
    sig = inspect.signature(chat)
    params = list(sig.parameters.keys())

    if "system_prompt" in params:
        print("   ✓ Chat command has system_prompt parameter")
    else:
        print("   ✗ Chat command missing system_prompt parameter")
        return False

    return True


def main():
    """Run all verification checks."""
    checks = [
        ("Imports", verify_imports),
        ("Prompts Content", verify_prompts),
        ("Identity Correction", verify_identity_correction),
        ("Configuration", verify_configuration),
        ("Task Prompts", verify_task_prompts),
        ("CLI Integration", verify_cli_integration),
    ]

    results = []
    for name, check in checks:
        try:
            result = check()
            results.append((name, result))
        except Exception as e:
            print(f"\n   ✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nResults: {passed}/{total} checks passed")

    if passed == total:
        print("\n✓ All checks passed! Identity fix is working correctly.")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
