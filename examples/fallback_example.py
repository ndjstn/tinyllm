"""Example demonstrating fallback model strategies.

This example shows how to use FallbackClient for resilient model execution:
1. Sequential fallback - try models in order
2. Fastest strategy - race all models
3. Load balanced - use health metrics for routing
4. Health tracking and metrics reporting
"""

import asyncio
from tinyllm.models.fallback import (
    FallbackClient,
    FallbackConfig,
    FallbackStrategy,
)
from tinyllm.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def example_sequential_fallback():
    """Demonstrate sequential fallback strategy."""
    logger.info("=== Sequential Fallback Example ===")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.SEQUENTIAL,
        timeout_ms=10000,
    )

    client = FallbackClient(config=config)

    try:
        result = await client.generate(
            prompt="What is the capital of France?",
            temperature=0.3,
        )

        logger.info(
            "sequential_result",
            model_used=result.model_used,
            fallback_occurred=result.fallback_occurred,
            attempts=result.attempts,
            response_length=len(result.response.response),
        )

        print(f"\nModel used: {result.model_used}")
        print(f"Fallback occurred: {result.fallback_occurred}")
        print(f"Response: {result.response.response[:200]}...")

    except Exception as e:
        logger.error("sequential_failed", error=str(e))


async def example_fastest_strategy():
    """Demonstrate fastest (racing) strategy."""
    logger.info("=== Fastest Strategy Example ===")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.FASTEST,
        timeout_ms=10000,
    )

    client = FallbackClient(config=config)

    try:
        result = await client.generate(
            prompt="Write a haiku about programming.",
            temperature=0.7,
        )

        logger.info(
            "fastest_result",
            model_used=result.model_used,
            total_latency_ms=result.total_latency_ms,
        )

        print(f"\nFastest model: {result.model_used}")
        print(f"Latency: {result.total_latency_ms:.2f}ms")
        print(f"Response:\n{result.response.response}")

    except Exception as e:
        logger.error("fastest_failed", error=str(e))


async def example_load_balanced():
    """Demonstrate load-balanced strategy with health tracking."""
    logger.info("=== Load Balanced Strategy Example ===")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.LOAD_BALANCED,
        enable_health_tracking=True,
        timeout_ms=10000,
    )

    client = FallbackClient(config=config)

    # Make several requests to build up health metrics
    prompts = [
        "What is Python?",
        "Explain async/await.",
        "What are design patterns?",
        "Describe REST APIs.",
        "What is machine learning?",
    ]

    for i, prompt in enumerate(prompts, 1):
        try:
            logger.info("load_balanced_request", request=i, total=len(prompts))

            result = await client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )

            logger.info(
                "request_complete",
                request=i,
                model_used=result.model_used,
                latency_ms=result.total_latency_ms,
            )

            print(f"\nRequest {i}/{len(prompts)}")
            print(f"  Model: {result.model_used}")
            print(f"  Latency: {result.total_latency_ms:.2f}ms")

        except Exception as e:
            logger.error("request_failed", request=i, error=str(e))

    # Show final health metrics
    metrics = client.get_health_metrics()
    print("\n=== Health Metrics ===")
    print(f"Overall success rate: {metrics['overall']['overall_success_rate']:.2%}")
    print(f"Total requests: {metrics['overall']['total_requests']}")

    print("\nPer-model metrics:")
    for model, stats in metrics["per_model"].items():
        print(f"\n{model}:")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Avg latency: {stats['average_latency_ms']:.2f}ms")
        print(f"  Requests: {stats['success_count'] + stats['failure_count']}")
        print(f"  Healthy: {stats['is_healthy']}")


async def example_fallback_statistics():
    """Demonstrate fallback statistics reporting."""
    logger.info("=== Fallback Statistics Example ===")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.SEQUENTIAL,
        timeout_ms=10000,
    )

    client = FallbackClient(config=config)

    # Make several requests
    prompts = [
        "Hello, how are you?",
        "What is AI?",
        "Explain neural networks.",
    ]

    for prompt in prompts:
        try:
            result = await client.generate(prompt=prompt, max_tokens=100)
            logger.info("request_succeeded", model=result.model_used)
        except Exception as e:
            logger.warning("request_failed", error=str(e))

    # Get fallback statistics
    stats = client.get_fallback_statistics()

    print("\n=== Fallback Statistics ===")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Primary model requests: {stats['primary_requests']}")
    print(f"Fallback requests: {stats['fallback_requests']}")
    print(f"Fallback rate: {stats['fallback_rate']:.2%}")

    print("\nModels used (by success count):")
    for model, count in stats["models_used"].items():
        print(f"  {model}: {count} successes")


async def example_health_recovery():
    """Demonstrate model health recovery after failures."""
    logger.info("=== Health Recovery Example ===")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:0.5b"],
        strategy=FallbackStrategy.SEQUENTIAL,
        enable_health_tracking=True,
    )

    client = FallbackClient(config=config)

    # Simulate some failures by using a bad model temporarily
    bad_config = FallbackConfig(
        primary_model="nonexistent-model",
        fallback_models=["qwen2.5:0.5b"],
        strategy=FallbackStrategy.SEQUENTIAL,
    )
    bad_client = FallbackClient(config=bad_config)

    print("\nSimulating failures with bad model...")
    for i in range(3):
        try:
            await bad_client.generate(prompt="test", max_tokens=10)
        except Exception:
            logger.info("expected_failure", attempt=i + 1)

    # Check health
    health = bad_client._health_tracker.get_health("nonexistent-model")
    print(f"After failures - Healthy: {health.is_healthy}")
    print(f"Consecutive failures: {health.consecutive_failures}")

    # Now make successful requests with good model
    print("\nMaking successful requests...")
    for i in range(3):
        try:
            result = await client.generate(
                prompt="What is Python?",
                max_tokens=50,
            )
            logger.info("success", attempt=i + 1, model=result.model_used)
        except Exception as e:
            logger.error("unexpected_failure", attempt=i + 1, error=str(e))

    # Check recovery
    health = client._health_tracker.get_health("qwen2.5:3b")
    print(f"\nAfter successes - Healthy: {health.is_healthy}")
    print(f"Success rate: {health.success_rate:.2%}")
    print(f"Consecutive failures: {health.consecutive_failures}")


async def main():
    """Run all examples."""
    configure_logging(log_level="INFO", log_format="console")

    print("=" * 60)
    print("Fallback Model Strategy Examples")
    print("=" * 60)

    try:
        # Example 1: Sequential fallback
        await example_sequential_fallback()
        await asyncio.sleep(1)

        # Example 2: Fastest strategy
        await example_fastest_strategy()
        await asyncio.sleep(1)

        # Example 3: Load balanced
        await example_load_balanced()
        await asyncio.sleep(1)

        # Example 4: Statistics
        await example_fallback_statistics()
        await asyncio.sleep(1)

        # Example 5: Health recovery
        await example_health_recovery()

    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("example_failed", error=str(e), exc_info=True)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
