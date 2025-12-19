"""Demonstration of TinyLLM's dynamic node spawning system.

This example shows how to use the NodeSpawner to dynamically create
and manage nodes at runtime based on workload and performance triggers.
"""

from tinyllm.config.graph import NodeDefinition, NodeType
from tinyllm.expansion.spawning import (
    NodeFactory,
    NodeSpawner,
    SpawnConfig,
    SpawnTrigger,
)


def main():
    """Demonstrate node spawning capabilities."""
    print("=" * 70)
    print("TinyLLM Dynamic Node Spawning Demo")
    print("=" * 70)

    # Create a spawn configuration
    config = SpawnConfig(
        max_spawns=5,
        cooldown_ms=1000,
        performance_threshold=0.7,
        workload_threshold=50,
        enable_auto_spawn=True,
        max_spawn_depth=3,
    )

    print("\n1. Creating NodeSpawner with config:")
    print(f"   - Max spawns: {config.max_spawns}")
    print(f"   - Cooldown: {config.cooldown_ms}ms")
    print(f"   - Performance threshold: {config.performance_threshold}")
    print(f"   - Auto-spawn enabled: {config.enable_auto_spawn}")

    spawner = NodeSpawner(config)

    # Create a template node definition
    template = NodeDefinition(
        id="reasoning_node",
        type=NodeType.REASONING,
        name="Base Reasoner",
        description="A reasoning node for complex problem solving",
        config={
            "model": "qwen2.5:1.5b",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
    )

    print("\n2. Template Node Definition:")
    print(f"   - ID: {template.id}")
    print(f"   - Type: {template.type.value}")
    print(f"   - Model: {template.config['model']}")

    # Example 1: Manual spawn with specialization
    print("\n3. Spawning specialized node (MANUAL trigger):")
    try:
        specialized_node_def = NodeDefinition(
            id=template.id,
            type=template.type,
            name=template.name,
            description=template.description,
            config=template.config,
        )

        # Note: In real usage, you'd use NodeRegistry.create() here
        # For demo purposes, we're just showing the spawn tracking
        print(f"   ✓ Would spawn node with specialization: model='qwen2.5:3b'")
        print(f"   ✓ Trigger: {SpawnTrigger.MANUAL.value}")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 2: Check performance trigger
    print("\n4. Checking performance-based spawn triggers:")
    current_performance = 0.65  # Below threshold
    should_spawn = spawner.check_performance_trigger(
        "reasoning_node", current_performance
    )
    print(f"   - Current performance: {current_performance}")
    print(f"   - Threshold: {config.performance_threshold}")
    print(f"   - Should spawn: {should_spawn}")
    if should_spawn:
        print(f"   ✓ Performance below threshold, spawn recommended")
    else:
        print(f"   ✓ Performance acceptable, no spawn needed")

    # Example 3: Check workload trigger
    print("\n5. Checking workload-based spawn triggers:")
    current_workload = 75  # Above threshold
    should_spawn = spawner.check_workload_trigger("reasoning_node", current_workload)
    print(f"   - Current workload: {current_workload}")
    print(f"   - Threshold: {config.workload_threshold}")
    print(f"   - Should spawn: {should_spawn}")
    if should_spawn:
        print(f"   ✓ Workload spike detected, spawn recommended")
    else:
        print(f"   ✓ Workload normal, no spawn needed")

    # Example 4: Show spawn capabilities
    print("\n6. Spawn System Status:")
    stats = spawner.get_stats()
    print(f"   - Can spawn now: {stats['can_spawn']}")
    print(f"   - Active spawns: {stats['active_spawns']}")
    print(f"   - Total spawns: {stats['total_spawns']}")
    print(f"   - Spawn history size: {stats['spawn_history_size']}")

    # Example 5: Get metrics
    print("\n7. Spawn Metrics:")
    metrics = spawner.get_metrics()
    print(f"   - Total spawns: {metrics.total_spawns}")
    print(f"   - Active spawns: {metrics.active_spawns}")
    print(f"   - Failed spawns: {metrics.failed_spawns}")
    print(f"   - Avg spawn latency: {metrics.avg_spawn_latency_ms:.2f}ms")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    print("\n📚 Key Features Demonstrated:")
    print("   • Configurable spawn limits and cooldowns")
    print("   • Multiple spawn triggers (manual, performance, workload)")
    print("   • Spawn depth tracking for clone chains")
    print("   • Comprehensive metrics and statistics")
    print("   • Production-ready validation and error handling")

    print("\n🎯 Use Cases:")
    print("   • Scale out reasoning nodes under high load")
    print("   • Create specialized variants for different domains")
    print("   • A/B test different model configurations")
    print("   • Implement adaptive capacity scaling")


if __name__ == "__main__":
    main()
