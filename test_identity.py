#!/usr/bin/env python3
"""Quick test script to verify identity system works correctly."""

from tinyllm.prompts import (
    get_chat_prompt,
    get_identity_correction,
    ASSISTANT_IDENTITY,
    CHAT_SYSTEM_PROMPT,
    PromptConfig,
)


def test_identity_prompts():
    """Test that identity prompts are configured correctly."""
    print("=" * 60)
    print("Testing TinyLLM Identity System")
    print("=" * 60)

    # Test 1: Basic identity string
    print("\n1. ASSISTANT_IDENTITY:")
    print("-" * 60)
    print(ASSISTANT_IDENTITY)

    # Test 2: Chat system prompt
    print("\n2. CHAT_SYSTEM_PROMPT:")
    print("-" * 60)
    print(CHAT_SYSTEM_PROMPT[:200] + "...")

    # Test 3: Model-specific chat prompt
    print("\n3. Model-specific chat prompt (qwen2.5:1.5b):")
    print("-" * 60)
    prompt = get_chat_prompt("qwen2.5:1.5b")
    print(prompt[:300] + "...")

    # Test 4: Identity correction
    print("\n4. Identity correction tests:")
    print("-" * 60)

    test_queries = [
        "what is your name",
        "who are you",
        "are you claude",
        "are you chatgpt",
        "is this gpt",
        "hello there",  # Should return None
    ]

    for query in test_queries:
        correction = get_identity_correction(query)
        if correction:
            print(f"✓ Query: '{query}'")
            print(f"  Response: {correction[:80]}...")
        else:
            print(f"○ Query: '{query}' - No correction needed")

    # Test 5: Custom configuration
    print("\n5. Custom PromptConfig:")
    print("-" * 60)
    config = PromptConfig(
        assistant_name="MyCustomBot",
        custom_identity="This is a custom bot identity.",
    )
    print(f"Assistant name: {config.assistant_name}")
    print(f"Custom identity: {config.custom_identity}")

    # Test 6: Verify NO mentions of Claude, ChatGPT, etc.
    print("\n6. Verification - checking for unwanted identities:")
    print("-" * 60)

    unwanted = ["claude", "chatgpt", "openai", "anthropic", "gemini"]
    found_issues = False

    for term in unwanted:
        if term.lower() in prompt.lower():
            # Check if it's in a correction/negation context (which is okay)
            # Look for "NOT Term" or "not Term," etc.
            import re
            negation_pattern = r'\b(not|nor|isn\'t|aren\'t|wasn\'t|weren\'t)\s+([\w\s,]+\s+)?' + re.escape(term.lower())
            if re.search(negation_pattern, prompt.lower(), re.IGNORECASE):
                print(f"✓ '{term}' mentioned correctly in negation context")
            else:
                print(f"✗ WARNING: '{term}' found in prompt!")
                found_issues = True

    if not found_issues:
        print("✓ No unwanted identity references found")

    # Test 7: Verify TinyLLM branding is present
    print("\n7. Verification - checking for TinyLLM branding:")
    print("-" * 60)

    required = ["tinyllm", "local", "ollama"]
    for term in required:
        if term.lower() in prompt.lower():
            print(f"✓ '{term}' found in prompt")
        else:
            print(f"✗ WARNING: '{term}' NOT found in prompt!")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_identity_prompts()
