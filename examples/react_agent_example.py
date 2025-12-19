#!/usr/bin/env python3
"""Example demonstrating the ReAct agent framework.

This example shows how to use the ReAct (Reasoning + Acting) agent
to solve problems using tools and step-by-step reasoning.

The ReAct pattern follows:
1. Thought: The LLM reasons about what to do
2. Action: The LLM decides on a tool and input
3. Observation: The tool executes and returns results
4. Repeat until a final answer is reached

Run this with: python examples/react_agent_example.py
"""

import asyncio
from tinyllm.agents import ReActAgent, ReActConfig
from tinyllm.models.client import OllamaClient
from tinyllm.tools.calculator import CalculatorTool


async def basic_example():
    """Basic example using calculator tool."""
    print("=" * 60)
    print("Example 1: Basic Calculator Usage")
    print("=" * 60)

    # Create LLM client
    client = OllamaClient(
        default_model="qwen2.5:0.5b",  # or any model you have
        host="http://localhost:11434",
    )

    # Create agent with configuration
    config = ReActConfig(
        max_iterations=5,
        temperature=0.0,  # Use deterministic responses
        verbose=True,  # Log detailed execution
    )
    agent = ReActAgent(llm_client=client, config=config)

    # Register tools
    agent.register_tool("calculator", CalculatorTool())

    # Run the agent
    try:
        question = "What is the square root of 144 plus 5 times 3?"
        print(f"\nQuestion: {question}\n")

        result = await agent.run(question)

        print(f"\n{'=' * 60}")
        print(f"Final Answer: {result}")
        print(f"{'=' * 60}")

        # Show the reasoning trace
        print("\nReasoning Trace:")
        print(agent.get_trace_string())

    except Exception as e:
        print(f"Error: {e}")

    # Clean up
    await client.close()


async def custom_tool_example():
    """Example with custom function as tool."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Tool Registration")
    print("=" * 60)

    client = OllamaClient(default_model="qwen2.5:0.5b")

    # Create agent
    agent = ReActAgent(
        llm_client=client,
        config=ReActConfig(max_iterations=5, verbose=False),
    )

    # Register calculator
    agent.register_tool("calculator", CalculatorTool())

    # Register a custom function as a tool
    def word_reverser(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    agent.register_function(
        "reverse",
        word_reverser,
        "Reverses the characters in a string",
    )

    # Run the agent
    try:
        question = 'Reverse the word "hello" and tell me the result'
        print(f"\nQuestion: {question}\n")

        result = await agent.run(question)

        print(f"\nFinal Answer: {result}")

    except Exception as e:
        print(f"Error: {e}")

    await client.close()


async def multi_step_example():
    """Example with multiple reasoning steps."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Step Reasoning")
    print("=" * 60)

    client = OllamaClient(default_model="qwen2.5:0.5b")

    # Create agent
    agent = ReActAgent(
        llm_client=client,
        config=ReActConfig(max_iterations=10, verbose=True),
    )

    # Register tools
    agent.register_tool("calculator", CalculatorTool())

    # Custom tool for simple lookups
    def lookup(query: str) -> str:
        """Simple lookup tool."""
        lookups = {
            "days in week": "7",
            "hours in day": "24",
            "minutes in hour": "60",
        }
        return lookups.get(query.lower(), "Information not found")

    agent.register_function("lookup", lookup, "Look up basic facts")

    # Run with complex question requiring multiple steps
    try:
        question = "How many minutes are in a week? Calculate step by step."
        print(f"\nQuestion: {question}\n")

        result = await agent.run(question)

        print(f"\n{'=' * 60}")
        print(f"Final Answer: {result}")
        print(f"{'=' * 60}")

        # Show trace
        trace = agent.get_trace()
        print(f"\nCompleted in {len(trace)} steps")

    except Exception as e:
        print(f"Error: {e}")

    await client.close()


async def error_handling_example():
    """Example showing error handling."""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    client = OllamaClient(default_model="qwen2.5:0.5b")

    # Create agent with strict error handling
    config = ReActConfig(
        max_iterations=5,
        stop_on_error=False,  # Continue even on errors
        verbose=True,
    )
    agent = ReActAgent(llm_client=client, config=config)

    # Register calculator
    agent.register_tool("calculator", CalculatorTool())

    try:
        # Ask a question that might cause errors
        question = "Calculate 10 divided by 0, then multiply by 5"
        print(f"\nQuestion: {question}\n")

        result = await agent.run(question)

        print(f"\nFinal Answer: {result}")

    except Exception as e:
        print(f"Agent failed with error: {e}")

    await client.close()


async def token_budget_example():
    """Example showing token budget limits."""
    print("\n" + "=" * 60)
    print("Example 5: Token Budget Management")
    print("=" * 60)

    client = OllamaClient(default_model="qwen2.5:0.5b")

    # Create agent with token budget
    config = ReActConfig(
        max_iterations=20,
        max_tokens=5000,  # Limit total tokens
        verbose=True,
    )
    agent = ReActAgent(llm_client=client, config=config)

    agent.register_tool("calculator", CalculatorTool())

    try:
        question = "What is 2 + 2?"
        result = await agent.run(question)

        print(f"\nFinal Answer: {result}")
        print(f"Total tokens used: {agent.total_tokens}")

    except ValueError as e:
        if "Token budget exceeded" in str(e):
            print(f"Agent stopped due to token budget: {e}")
        else:
            raise

    await client.close()


async def main():
    """Run all examples."""
    print("\nReAct Agent Examples")
    print("Make sure Ollama is running with qwen2.5:0.5b model")
    print("(or change the model in the examples)\n")

    try:
        # Run examples
        await basic_example()
        await custom_tool_example()
        await multi_step_example()
        await error_handling_example()
        await token_budget_example()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nFailed to run examples: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. You have the model installed (ollama pull qwen2.5:0.5b)")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
