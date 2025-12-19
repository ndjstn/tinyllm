"""Production-Ready Fallback Example

Demonstrates a real-world production setup with:
- Multi-tier model fallback
- Health monitoring
- Metrics collection
- Alert simulation
- Graceful degradation
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from tinyllm.models.fallback import (
    FallbackClient,
    FallbackConfig,
    FallbackStrategy,
)
from tinyllm.logging import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration for fallback system."""

    # Model tiers
    primary_model: str = "qwen2.5:7b"  # High quality
    secondary_model: str = "qwen2.5:3b"  # Balanced
    tertiary_model: str = "qwen2.5:1.5b"  # Fast
    emergency_model: str = "qwen2.5:0.5b"  # Ultra-fast fallback

    # Timeouts (in ms)
    primary_timeout: int = 60000  # 60s for large model
    secondary_timeout: int = 30000  # 30s for medium
    tertiary_timeout: int = 15000  # 15s for small
    emergency_timeout: int = 5000  # 5s for emergency

    # Health tracking
    enable_health: bool = True
    health_check_interval: float = 30.0  # 30 seconds

    # Alerting thresholds
    alert_fallback_rate: float = 0.3  # Alert if >30% fallback
    alert_failure_rate: float = 0.1  # Alert if >10% failures


class ProductionFallbackSystem:
    """Production-ready fallback system with monitoring."""

    def __init__(self, config: ProductionConfig):
        """Initialize production system."""
        self.config = config
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Create clients for different use cases."""

        # High-quality client (sequential, prefer quality)
        self.clients["quality"] = FallbackClient(
            config=FallbackConfig(
                primary_model=self.config.primary_model,
                fallback_models=[
                    self.config.secondary_model,
                    self.config.tertiary_model,
                    self.config.emergency_model,
                ],
                strategy=FallbackStrategy.SEQUENTIAL,
                timeout_ms=self.config.primary_timeout,
                enable_health_tracking=self.config.enable_health,
                health_check_interval_s=self.config.health_check_interval,
            )
        )

        # Low-latency client (fastest, prefer speed)
        self.clients["latency"] = FallbackClient(
            config=FallbackConfig(
                primary_model=self.config.tertiary_model,
                fallback_models=[
                    self.config.emergency_model,
                    self.config.secondary_model,
                ],
                strategy=FallbackStrategy.FASTEST,
                timeout_ms=self.config.tertiary_timeout,
                enable_health_tracking=self.config.enable_health,
            )
        )

        # Balanced client (load balanced, adaptive)
        self.clients["balanced"] = FallbackClient(
            config=FallbackConfig(
                primary_model=self.config.secondary_model,
                fallback_models=[
                    self.config.tertiary_model,
                    self.config.emergency_model,
                ],
                strategy=FallbackStrategy.LOAD_BALANCED,
                timeout_ms=self.config.secondary_timeout,
                enable_health_tracking=True,
                health_check_interval_s=self.config.health_check_interval,
            )
        )

        logger.info(
            "production_clients_initialized",
            client_types=list(self.clients.keys()),
        )

    async def generate_quality(self, prompt: str, **kwargs) -> dict:
        """Generate with quality-first strategy."""
        logger.info("quality_request", prompt_length=len(prompt))

        try:
            result = await self.clients["quality"].generate(prompt, **kwargs)

            logger.info(
                "quality_success",
                model_used=result.model_used,
                fallback_occurred=result.fallback_occurred,
                latency_ms=result.total_latency_ms,
            )

            return {
                "response": result.response.response,
                "model": result.model_used,
                "fallback": result.fallback_occurred,
                "latency_ms": result.total_latency_ms,
            }

        except Exception as e:
            logger.error("quality_failed", error=str(e))
            raise

    async def generate_fast(self, prompt: str, **kwargs) -> dict:
        """Generate with low-latency strategy."""
        logger.info("latency_request", prompt_length=len(prompt))

        try:
            result = await self.clients["latency"].generate(prompt, **kwargs)

            logger.info(
                "latency_success",
                model_used=result.model_used,
                latency_ms=result.total_latency_ms,
            )

            return {
                "response": result.response.response,
                "model": result.model_used,
                "latency_ms": result.total_latency_ms,
            }

        except Exception as e:
            logger.error("latency_failed", error=str(e))
            raise

    async def generate_balanced(self, prompt: str, **kwargs) -> dict:
        """Generate with balanced strategy."""
        logger.info("balanced_request", prompt_length=len(prompt))

        try:
            result = await self.clients["balanced"].generate(prompt, **kwargs)

            logger.info(
                "balanced_success",
                model_used=result.model_used,
                latency_ms=result.total_latency_ms,
            )

            return {
                "response": result.response.response,
                "model": result.model_used,
                "latency_ms": result.total_latency_ms,
            }

        except Exception as e:
            logger.error("balanced_failed", error=str(e))
            raise

    def get_system_health(self) -> dict:
        """Get health metrics for all clients."""
        health = {}

        for name, client in self.clients.items():
            metrics = client.get_health_metrics()
            stats = client.get_fallback_statistics()

            health[name] = {
                "overall_success_rate": metrics["overall"]["overall_success_rate"],
                "total_requests": metrics["overall"]["total_requests"],
                "fallback_rate": stats.get("fallback_rate", 0.0),
                "models": metrics["per_model"],
            }

        return health

    def check_alerts(self) -> list:
        """Check if any alerts should be triggered."""
        alerts = []

        for name, client in self.clients.items():
            metrics = client.get_health_metrics()
            stats = client.get_fallback_statistics()

            # Check fallback rate
            fallback_rate = stats.get("fallback_rate", 0.0)
            if fallback_rate > self.config.alert_fallback_rate:
                alerts.append({
                    "severity": "warning",
                    "client": name,
                    "type": "high_fallback_rate",
                    "value": fallback_rate,
                    "threshold": self.config.alert_fallback_rate,
                    "message": f"Client {name} has high fallback rate: {fallback_rate:.2%}",
                })

            # Check failure rate
            overall = metrics["overall"]
            if overall["total_requests"] > 0:
                failure_rate = 1 - overall["overall_success_rate"]
                if failure_rate > self.config.alert_failure_rate:
                    alerts.append({
                        "severity": "critical",
                        "client": name,
                        "type": "high_failure_rate",
                        "value": failure_rate,
                        "threshold": self.config.alert_failure_rate,
                        "message": f"Client {name} has high failure rate: {failure_rate:.2%}",
                    })

            # Check for unhealthy models
            for model, health in metrics["per_model"].items():
                if not health["is_healthy"]:
                    alerts.append({
                        "severity": "warning",
                        "client": name,
                        "type": "unhealthy_model",
                        "model": model,
                        "message": f"Model {model} is unhealthy in client {name}",
                    })

        return alerts


async def run_production_simulation():
    """Simulate production usage."""
    logger.info("=== Production Fallback System Simulation ===")

    # Initialize system
    config = ProductionConfig()
    system = ProductionFallbackSystem(config)

    # Simulate different workload types
    tasks = [
        # Quality-focused tasks
        ("quality", "Analyze the following code for security vulnerabilities..."),
        ("quality", "Write a comprehensive design document for..."),

        # Latency-focused tasks
        ("latency", "What is Python?"),
        ("latency", "Translate: Hello"),
        ("latency", "Summarize in one sentence:"),

        # Balanced tasks
        ("balanced", "Explain async/await in Python."),
        ("balanced", "What are the benefits of microservices?"),
        ("balanced", "How does caching improve performance?"),
    ]

    print("\n" + "=" * 70)
    print("Simulating Production Workload")
    print("=" * 70)

    for i, (client_type, prompt) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {client_type.upper()} request")
        print(f"Prompt: {prompt[:50]}...")

        try:
            if client_type == "quality":
                result = await system.generate_quality(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.3,
                )
            elif client_type == "latency":
                result = await system.generate_fast(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.5,
                )
            else:  # balanced
                result = await system.generate_balanced(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.4,
                )

            print(f"✓ Model: {result['model']}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            if result.get("fallback"):
                print(f"  ⚠ Fallback occurred")

        except Exception as e:
            print(f"✗ Failed: {e}")

        # Small delay between requests
        await asyncio.sleep(0.5)

    # Print health summary
    print("\n" + "=" * 70)
    print("System Health Summary")
    print("=" * 70)

    health = system.get_system_health()
    for name, metrics in health.items():
        print(f"\n{name.upper()} Client:")
        print(f"  Success Rate: {metrics['overall_success_rate']:.2%}")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Fallback Rate: {metrics['fallback_rate']:.2%}")

        print(f"  Models:")
        for model, stats in metrics['models'].items():
            if stats['success_count'] > 0 or stats['failure_count'] > 0:
                print(f"    {model}:")
                print(f"      Success Rate: {stats['success_rate']:.2%}")
                print(f"      Avg Latency: {stats['average_latency_ms']:.2f}ms")
                print(f"      Healthy: {'✓' if stats['is_healthy'] else '✗'}")

    # Check for alerts
    print("\n" + "=" * 70)
    print("Alert Status")
    print("=" * 70)

    alerts = system.check_alerts()
    if alerts:
        print(f"\n⚠ {len(alerts)} alert(s) detected:\n")
        for alert in alerts:
            severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
            print(f"{severity_icon} {alert['message']}")
    else:
        print("\n✓ No alerts - system healthy")


async def demonstrate_health_recovery():
    """Demonstrate health recovery after failures."""
    logger.info("\n=== Health Recovery Demonstration ===")

    print("\n" + "=" * 70)
    print("Health Recovery Demonstration")
    print("=" * 70)

    # Create client
    client = FallbackClient(
        config=FallbackConfig(
            primary_model="qwen2.5:3b",
            fallback_models=["qwen2.5:0.5b"],
            enable_health_tracking=True,
        )
    )

    # Make some successful requests
    print("\nPhase 1: Successful requests")
    for i in range(3):
        try:
            result = await client.generate(
                prompt=f"Request {i+1}",
                max_tokens=20,
            )
            print(f"  ✓ Request {i+1}: {result.model_used}")
        except Exception:
            print(f"  ✗ Request {i+1}: Failed")

    # Show health
    metrics = client.get_health_metrics()
    print("\nHealth after successful requests:")
    for model, health in metrics["per_model"].items():
        print(f"  {model}:")
        print(f"    Success Rate: {health['success_rate']:.2%}")
        print(f"    Healthy: {health['is_healthy']}")

    # Simulate recovery period
    print("\nPhase 2: Making more requests to verify stability")
    for i in range(2):
        try:
            result = await client.generate(
                prompt=f"Recovery request {i+1}",
                max_tokens=20,
            )
            print(f"  ✓ Request {i+1}: {result.model_used}")
        except Exception:
            print(f"  ✗ Request {i+1}: Failed")

    # Final health check
    metrics = client.get_health_metrics()
    print("\nFinal health status:")
    for model, health in metrics["per_model"].items():
        print(f"  {model}:")
        print(f"    Success Rate: {health['success_rate']:.2%}")
        print(f"    Total Requests: {health['success_count'] + health['failure_count']}")
        print(f"    Healthy: {'✓' if health['is_healthy'] else '✗'}")


async def main():
    """Run production examples."""
    configure_logging(log_level="INFO", log_format="console")

    try:
        # Run production simulation
        await run_production_simulation()

        # Demonstrate health recovery
        await demonstrate_health_recovery()

        print("\n" + "=" * 70)
        print("Production Simulation Complete!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  ✓ Multiple client types for different use cases")
        print("  ✓ Health tracking across all clients")
        print("  ✓ Automatic alerting on issues")
        print("  ✓ Graceful degradation with fallback")
        print("  ✓ Recovery after failures")
        print()

    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("simulation_failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
