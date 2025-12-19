#!/usr/bin/env python
"""Example usage of the Mistral AI client.

This example demonstrates how to use the MistralClient for:
1. Chat completions
2. Streaming responses
3. Tool calling (function calling)
4. Vision with Pixtral models
5. Embeddings
"""

import asyncio
import os

from tinyllm.providers import (
    MistralClient,
    MistralChatMessage,
    MistralMessageRole,
    MistralToolDefinition,
    MistralFunctionDefinition,
    MistralToolChoice,
    SafeMode,
    get_shared_mistral_client,
)


async def example_chat_completion():
    """Example: Basic chat completion."""
    print("\n=== Example 1: Chat Completion ===")

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"))

    messages = [
        MistralChatMessage(
            role=MistralMessageRole.SYSTEM,
            content="You are a helpful assistant.",
        ),
        MistralChatMessage(
            role=MistralMessageRole.USER,
            content="What is the capital of France?"
        ),
    ]

    try:
        response = await client.chat_completion(
            messages=messages,
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=100,
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


async def example_streaming():
    """Example: Streaming chat completion."""
    print("\n=== Example 2: Streaming Response ===")

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"))

    messages = [
        MistralChatMessage(
            role=MistralMessageRole.USER,
            content="Write a short poem about coding.",
        ),
    ]

    try:
        print("Streaming response: ", end="", flush=True)
        async for chunk in client.chat_completion_stream(
            messages=messages,
            model="mistral-large-latest",
        ):
            print(chunk, end="", flush=True)
        print()  # New line at the end
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await client.close()


async def example_tool_calling():
    """Example: Function calling with tools."""
    print("\n=== Example 3: Tool Calling ===")

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"))

    # Define a weather tool
    tools = [
        MistralToolDefinition(
            type="function",
            function=MistralFunctionDefinition(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., 'Paris'",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
    ]

    messages = [
        MistralChatMessage(
            role=MistralMessageRole.USER,
            content="What's the weather like in Paris?",
        ),
    ]

    try:
        response = await client.chat_completion(
            messages=messages,
            model="mistral-large-latest",
            tools=tools,
            tool_choice=MistralToolChoice.AUTO,
        )

        # Check if the model wants to call a tool
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                print(f"Tool called: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
        else:
            print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


async def example_vision():
    """Example: Vision with Pixtral models."""
    print("\n=== Example 4: Vision (Pixtral) ===")

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"))

    # Create a message with an image
    message = client.create_image_message(
        text="What do you see in this image?",
        image_urls=["https://example.com/image.jpg"],
    )

    try:
        response = await client.chat_completion(
            messages=[message],
            model="pixtral-12b-2409",  # Pixtral vision model
            max_tokens=300,
        )

        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


async def example_embeddings():
    """Example: Creating embeddings."""
    print("\n=== Example 5: Embeddings ===")

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"))

    try:
        # Single text embedding
        response = await client.create_embedding(
            input_text="Hello, world!",
            model="mistral-embed",
        )

        print(f"Embedding dimensions: {len(response.data[0].embedding)}")
        print(f"First 5 values: {response.data[0].embedding[:5]}")

        # Batch embeddings
        batch_response = await client.create_embedding(
            input_text=["First text", "Second text", "Third text"],
            model="mistral-embed",
        )

        print(f"\nBatch embeddings created: {len(batch_response.data)}")
        print(f"Tokens used: {batch_response.usage.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


async def example_safe_mode():
    """Example: Using safe mode for content filtering."""
    print("\n=== Example 6: Safe Mode ===")

    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"))

    messages = [
        MistralChatMessage(
            role=MistralMessageRole.USER,
            content="Tell me about safety in AI systems.",
        ),
    ]

    try:
        response = await client.chat_completion(
            messages=messages,
            model="mistral-large-latest",
            safe_mode=SafeMode.HARD,  # Enable strict content filtering
            safe_prompt=True,
        )

        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


async def example_shared_client():
    """Example: Using shared client for connection pooling."""
    print("\n=== Example 7: Shared Client ===")

    # Get shared client (reused across the application)
    client = await get_shared_mistral_client(
        api_key=os.getenv("MISTRAL_API_KEY", "your-api-key"),
        rate_limit_rps=10.0,  # 10 requests per second
    )

    messages = [
        MistralChatMessage(
            role=MistralMessageRole.USER,
            content="Hello!",
        ),
    ]

    try:
        response = await client.chat_completion(messages=messages)
        print(f"Response: {response.choices[0].message.content}")

        # Get client statistics
        stats = client.get_stats()
        print(f"\nClient stats:")
        print(f"  Requests: {stats['request_count']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Circuit breaker state: {stats['circuit_breaker_state']}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples."""
    print("Mistral AI Client Examples")
    print("=" * 50)

    # Note: These examples require a valid MISTRAL_API_KEY
    if not os.getenv("MISTRAL_API_KEY"):
        print("\n⚠️  Set MISTRAL_API_KEY environment variable to run these examples")
        print("Example: export MISTRAL_API_KEY='your-api-key'")
        return

    await example_chat_completion()
    await example_streaming()
    await example_tool_calling()
    await example_vision()
    await example_embeddings()
    await example_safe_mode()
    await example_shared_client()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
