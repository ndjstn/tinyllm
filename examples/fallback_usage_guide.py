"""Fallback Model Strategy - Usage Guide

This guide demonstrates all features of the fallback model system:
- Configuration options
- Strategy selection
- Health tracking
- Metrics and monitoring
- Integration with workflows
"""

import asyncio
from tinyllm.models.fallback import (
    FallbackClient,
    FallbackConfig,
    FallbackStrategy,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


# ==============================================================================
# BASIC CONFIGURATION
# ==============================================================================

def basic_config_example():
    """Example 1: Basic fallback configuration."""
    print_section("Example 1: Basic Configuration")

    # Minimal configuration - just specify primary and fallback models
    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
    )

    print("\nMinimal config created:")
    print(f"  Primary: {config.primary_model}")
    print(f"  Fallbacks: {config.fallback_models}")
    print(f"  Strategy: {config.strategy}")
    print(f"  Timeout: {config.timeout_ms}ms")


def advanced_config_example():
    """Example 2: Advanced configuration with all options."""
    print_section("Example 2: Advanced Configuration")

    config = FallbackConfig(
        # Model chain
        primary_model="qwen2.5:7b",
        fallback_models=["qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b"],

        # Error handling
        retry_on_errors=["timeout", "connection", "rate_limit", "server_error"],
        timeout_ms=60000,  # 60 seconds per model
        max_retries_per_model=3,

        # Strategy
        strategy=FallbackStrategy.LOAD_BALANCED,

        # Health tracking
        enable_health_tracking=True,
        health_check_interval_s=30.0,
    )

    print("\nAdvanced config created:")
    print(f"  Model chain: {[config.primary_model] + config.fallback_models}")
    print(f"  Retry errors: {config.retry_on_errors}")
    print(f"  Timeout: {config.timeout_ms}ms")
    print(f"  Max retries: {config.max_retries_per_model}")
    print(f"  Strategy: {config.strategy.value}")
    print(f"  Health tracking: {config.enable_health_tracking}")


# ==============================================================================
# FALLBACK STRATEGIES
# ==============================================================================

async def sequential_strategy_example():
    """Example 3: Sequential strategy - try models in order."""
    print_section("Example 3: Sequential Strategy")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.SEQUENTIAL,
    )

    client = FallbackClient(config=config)

    print("\nSequential strategy tries models in this order:")
    print("  1. qwen2.5:3b (primary)")
    print("  2. qwen2.5:1.5b (fallback 1)")
    print("  3. qwen2.5:0.5b (fallback 2)")
    print("\nUse case: When you have a preferred model hierarchy")
    print("          and want to fall back only on failure.")


async def fastest_strategy_example():
    """Example 4: Fastest strategy - race all models."""
    print_section("Example 4: Fastest Strategy")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.FASTEST,
    )

    client = FallbackClient(config=config)

    print("\nFastest strategy races ALL models simultaneously:")
    print("  - qwen2.5:3b")
    print("  - qwen2.5:1.5b")
    print("  - qwen2.5:0.5b")
    print("\nFirst model to respond wins!")
    print("\nUse case: Latency-critical applications where")
    print("          you need the fastest response available.")


async def load_balanced_strategy_example():
    """Example 5: Load balanced strategy - use health metrics."""
    print_section("Example 5: Load Balanced Strategy")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b", "qwen2.5:0.5b"],
        strategy=FallbackStrategy.LOAD_BALANCED,
        enable_health_tracking=True,
    )

    client = FallbackClient(config=config)

    print("\nLoad balanced strategy selects models based on:")
    print("  1. Success rate (higher is better)")
    print("  2. Average latency (lower is better)")
    print("  3. Health status (skip unhealthy models)")
    print("\nDynamically reorders models as health metrics change.")
    print("\nUse case: Production systems with varying model")
    print("          availability and performance.")


# ==============================================================================
# HEALTH TRACKING
# ==============================================================================

async def health_tracking_example():
    """Example 6: Understanding health tracking."""
    print_section("Example 6: Health Tracking")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b"],
        enable_health_tracking=True,
    )

    client = FallbackClient(config=config)

    print("\nHealth tracking monitors each model:")
    print("\nMetrics tracked:")
    print("  - Success count")
    print("  - Failure count")
    print("  - Success rate")
    print("  - Average latency")
    print("  - Consecutive failures")
    print("  - Health status")
    print("\nA model becomes UNHEALTHY after 3 consecutive failures.")
    print("Unhealthy models are skipped during fallback.")
    print("\nHealth resets after a successful request.")


async def metrics_example():
    """Example 7: Accessing metrics and statistics."""
    print_section("Example 7: Metrics and Statistics")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b"],
    )

    client = FallbackClient(config=config)

    print("\nTwo types of metrics available:")
    print("\n1. Health Metrics (per-model performance):")
    print("   metrics = client.get_health_metrics()")
    print("   - metrics['per_model'][model_name]")
    print("   - metrics['overall']")

    print("\n2. Fallback Statistics (usage patterns):")
    print("   stats = client.get_fallback_statistics()")
    print("   - stats['total_requests']")
    print("   - stats['fallback_requests']")
    print("   - stats['fallback_rate']")
    print("   - stats['models_used']")


# ==============================================================================
# WORKFLOW INTEGRATION
# ==============================================================================

def workflow_integration_example():
    """Example 8: Using fallback in workflow YAML configs."""
    print_section("Example 8: Workflow Integration")

    yaml_example = """
# In your workflow YAML:
nodes:
  - id: "analyzer"
    type: "model"
    config:
      # Primary model
      model: "qwen2.5:3b"

      # Enable fallback
      enable_fallback: true
      fallback_models:
        - "qwen2.5:1.5b"
        - "qwen2.5:0.5b"

      # Strategy
      fallback_strategy: "sequential"  # or "fastest" or "load_balanced"
      fallback_timeout_ms: 30000

      # Standard model config
      temperature: 0.7
      max_tokens: 2000
      system_prompt: "You are a helpful assistant."
"""

    print("\nYAML Configuration Example:")
    print(yaml_example)
    print("\nThe node will automatically:")
    print("  - Try the primary model first")
    print("  - Fall back on failure")
    print("  - Track health metrics")
    print("  - Report which model was used in metadata")


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

async def error_handling_example():
    """Example 9: Error handling and retry logic."""
    print_section("Example 9: Error Handling")

    config = FallbackConfig(
        primary_model="qwen2.5:3b",
        fallback_models=["qwen2.5:1.5b"],
        retry_on_errors=["timeout", "connection", "rate_limit"],
        max_retries_per_model=2,
    )

    print("\nError handling behavior:")
    print("\n1. Retryable errors (configured in retry_on_errors):")
    print("   - timeout")
    print("   - connection")
    print("   - rate_limit")
    print("   → Will retry up to max_retries_per_model times")

    print("\n2. Non-retryable errors:")
    print("   - Invalid model name")
    print("   - API authentication failures")
    print("   → Immediately try next model in chain")

    print("\n3. All models fail:")
    print("   → Raises RuntimeError with details")


# ==============================================================================
# BEST PRACTICES
# ==============================================================================

def best_practices_example():
    """Example 10: Best practices and recommendations."""
    print_section("Example 10: Best Practices")

    print("\n1. STRATEGY SELECTION:")
    print("   - Sequential: Most common, reliable, predictable costs")
    print("   - Fastest: Low latency requirements, higher costs")
    print("   - Load Balanced: Production systems, adaptive performance")

    print("\n2. MODEL CHAIN ORDERING:")
    print("   - Order by preference (quality, cost, speed)")
    print("   - Include at least one fast, reliable fallback")
    print("   - Example: [large-accurate, medium-balanced, small-fast]")

    print("\n3. TIMEOUT CONFIGURATION:")
    print("   - Set per-model timeout based on expected latency")
    print("   - Faster models can use shorter timeouts")
    print("   - Consider total latency: timeout × num_models")

    print("\n4. HEALTH TRACKING:")
    print("   - Enable in production for adaptive routing")
    print("   - Monitor metrics to identify model issues")
    print("   - Adjust health_check_interval_s based on traffic")

    print("\n5. ERROR CONFIGURATION:")
    print("   - Include transient errors in retry_on_errors")
    print("   - Don't retry on permanent failures (auth, invalid model)")
    print("   - Balance retries with total latency requirements")

    print("\n6. MONITORING:")
    print("   - Regularly check get_health_metrics()")
    print("   - Alert on high fallback_rate")
    print("   - Track per-model success rates")
    print("   - Monitor average latencies")


# ==============================================================================
# EXAMPLE USE CASES
# ==============================================================================

def use_cases_example():
    """Example 11: Real-world use cases."""
    print_section("Example 11: Real-World Use Cases")

    print("\nUSE CASE 1: High Availability Service")
    print("-" * 70)
    print("Goal: 99.9% uptime for production API")
    print("\nConfiguration:")
    print("  - Strategy: SEQUENTIAL")
    print("  - Models: [primary-large, backup-medium, backup-small]")
    print("  - Health tracking: Enabled")
    print("  - Timeout: 30s per model")
    print("\nBenefit: Automatic failover if primary model is down")

    print("\n\nUSE CASE 2: Low Latency Application")
    print("-" * 70)
    print("Goal: Sub-second response times")
    print("\nConfiguration:")
    print("  - Strategy: FASTEST")
    print("  - Models: [fast-model-1, fast-model-2, fast-model-3]")
    print("  - Timeout: 5s")
    print("\nBenefit: Always get fastest available response")

    print("\n\nUSE CASE 3: Cost Optimization")
    print("-" * 70)
    print("Goal: Minimize API costs while maintaining quality")
    print("\nConfiguration:")
    print("  - Strategy: LOAD_BALANCED")
    print("  - Models: [expensive-accurate, cheap-fast]")
    print("  - Health tracking: Enabled")
    print("\nBenefit: Use cheap model when healthy, fall back when needed")

    print("\n\nUSE CASE 4: Development & Testing")
    print("-" * 70)
    print("Goal: Test with multiple model sizes")
    print("\nConfiguration:")
    print("  - Strategy: SEQUENTIAL")
    print("  - Models: [local-3b, local-1.5b, local-0.5b]")
    print("\nBenefit: Graceful degradation during local development")


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  FALLBACK MODEL STRATEGY - COMPREHENSIVE USAGE GUIDE")
    print("=" * 70)

    # Configuration examples
    basic_config_example()
    advanced_config_example()

    # Strategy examples
    await sequential_strategy_example()
    await fastest_strategy_example()
    await load_balanced_strategy_example()

    # Monitoring examples
    await health_tracking_example()
    await metrics_example()

    # Integration examples
    workflow_integration_example()
    await error_handling_example()

    # Best practices
    best_practices_example()
    use_cases_example()

    print("\n" + "=" * 70)
    print("  END OF GUIDE")
    print("=" * 70)
    print("\nFor working code examples, see: fallback_example.py")
    print("For workflow integration, see: workflows/fallback_workflow.yaml")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
