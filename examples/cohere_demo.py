"""Demonstration of Cohere API client features.

This script shows how to use the CohereClient for:
- Chat completions
- Streaming responses
- Tool use (function calling)
- Embeddings
- Document reranking

Note: You need to set COHERE_API_KEY environment variable to run this.
"""

import asyncio
import os
from tinyllm.providers import (
    CohereClient,
    CohereChatMessage,
    CohereChatRole,
    CohereTool,
    CohereToolParameterDefinition,
    CohereToolCall,
    CohereToolResult,
    get_shared_cohere_client,
)


async def demo_chat_completion():
    """Demo basic chat completion."""
    print("\n" + "=" * 60)
    print("DEMO: Chat Completion")
    print("=" * 60)

    client = CohereClient(api_key=os.getenv("COHERE_API_KEY", "demo-key"))

    # Note: This is a demo - actual API call would require valid key
    print("""
Example usage:
    response = await client.chat(
        message="What is the capital of France?",
        model="command-r-plus",
        temperature=0.3,
    )

    print(f"Response: {response.text}")
    print(f"Tokens used: {response.token_count.total_tokens}")
""")


async def demo_chat_with_history():
    """Demo chat with conversation history."""
    print("\n" + "=" * 60)
    print("DEMO: Chat with History")
    print("=" * 60)

    print("""
Example usage:
    chat_history = [
        CohereChatMessage(role=CohereChatRole.USER, message="Hi!"),
        CohereChatMessage(role=CohereChatRole.CHATBOT, message="Hello! How can I help?"),
    ]

    response = await client.chat(
        message="Tell me about Python",
        chat_history=chat_history,
    )
""")


async def demo_streaming():
    """Demo streaming responses."""
    print("\n" + "=" * 60)
    print("DEMO: Streaming Responses")
    print("=" * 60)

    print("""
Example usage:
    async for event in client.chat_stream(
        message="Write a short poem about AI",
        temperature=0.7,
    ):
        if event.text:
            print(event.text, end="", flush=True)

        if event.is_finished:
            print(f"\\n\\nFinish reason: {event.finish_reason}")
""")


async def demo_tool_use():
    """Demo tool use (function calling)."""
    print("\n" + "=" * 60)
    print("DEMO: Tool Use (Function Calling)")
    print("=" * 60)

    print("""
Example usage:
    # Define tools
    tools = [
        CohereTool(
            name="get_weather",
            description="Get current weather for a location",
            parameter_definitions={
                "location": CohereToolParameterDefinition(
                    description="City name",
                    type="string",
                    required=True,
                ),
                "unit": CohereToolParameterDefinition(
                    description="Temperature unit (celsius or fahrenheit)",
                    type="string",
                    required=False,
                ),
            },
        ),
    ]

    # First request with tools
    response = await client.chat(
        message="What's the weather in San Francisco?",
        tools=tools,
    )

    if response.tool_calls:
        print(f"Model wants to call: {response.tool_calls[0].name}")
        print(f"With parameters: {response.tool_calls[0].parameters}")

        # Simulate tool execution
        tool_results = [
            CohereToolResult(
                call=response.tool_calls[0],
                outputs=[{"temperature": "68°F", "condition": "Sunny"}],
            )
        ]

        # Send results back
        final_response = await client.chat(
            message="What's the weather in San Francisco?",
            tools=tools,
            tool_results=tool_results,
        )

        print(f"Final response: {final_response.text}")
""")


async def demo_embeddings():
    """Demo embedding generation."""
    print("\n" + "=" * 60)
    print("DEMO: Embeddings")
    print("=" * 60)

    print("""
Example usage:
    # Generate embeddings for search documents
    response = await client.embed(
        texts=[
            "Python is a programming language",
            "Machine learning uses data to train models",
            "The weather is sunny today",
        ],
        model="embed-english-v3.0",
        input_type="search_document",
    )

    for i, embedding in enumerate(response.embeddings):
        print(f"Text {i}: {len(embedding.values)} dimensions")

    # Generate query embedding
    query_response = await client.embed(
        texts=["Tell me about Python programming"],
        input_type="search_query",
    )
""")


async def demo_reranking():
    """Demo document reranking."""
    print("\n" + "=" * 60)
    print("DEMO: Document Reranking")
    print("=" * 60)

    print("""
Example usage:
    documents = [
        "Python is a high-level programming language",
        "The weather forecast shows rain tomorrow",
        "JavaScript is used for web development",
        "Python is popular for data science and machine learning",
    ]

    response = await client.rerank(
        query="Python programming language",
        documents=documents,
        model="rerank-english-v3.0",
        top_n=2,
    )

    print("\\nReranked results:")
    for result in response.results:
        print(f"Score: {result.relevance_score:.3f}")
        print(f"Document: {result.document['text']}")
        print()
""")


async def demo_advanced_features():
    """Demo advanced features."""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Features")
    print("=" * 60)

    print("""
Advanced features included:

1. Rate Limiting (Token Bucket Algorithm):
   - Automatically throttles requests to respect API limits
   - Configurable requests per second and burst size

2. Circuit Breaker Pattern:
   - Opens after configurable failure threshold
   - Automatic recovery after timeout
   - Prevents cascade failures

3. Retry Logic with Exponential Backoff:
   - Automatic retries on transient failures
   - Respects Retry-After headers
   - Jitter to prevent thundering herd

4. Connection Pooling:
   - Shared client instances across application
   - Efficient connection reuse

5. Comprehensive Statistics:
   stats = client.get_stats()
   print(stats)
   # {
   #     'request_count': 42,
   #     'total_tokens': 15000,
   #     'circuit_breaker_state': 'closed',
   #     'circuit_breaker_failures': 0,
   # }

6. Metrics Integration:
   - Automatic tracking of requests, tokens, errors
   - Graph context support for multi-model workflows
   - Compatible with Prometheus/OpenTelemetry
""")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("COHERE API CLIENT DEMONSTRATION")
    print("=" * 60)

    await demo_chat_completion()
    await demo_chat_with_history()
    await demo_streaming()
    await demo_tool_use()
    await demo_embeddings()
    await demo_reranking()
    await demo_advanced_features()

    print("\n" + "=" * 60)
    print("KEY FEATURES SUMMARY")
    print("=" * 60)
    print("""
The Cohere client provides:

✓ Chat API with conversation history
✓ Streaming responses
✓ Tool use (connectors/function calling)
✓ Embeddings with specialized input types
✓ Document reranking
✓ Rate limiting with token bucket algorithm
✓ Circuit breaker for fault tolerance
✓ Exponential backoff retry logic
✓ Connection pooling and reuse
✓ Comprehensive statistics tracking
✓ Full Pydantic validation
✓ Type hints throughout
✓ Environment variable support (COHERE_API_KEY)
✓ Async/await support
✓ Metrics integration
""")


if __name__ == "__main__":
    asyncio.run(main())
