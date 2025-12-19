"""Tests for horizontal scaling."""

import asyncio
from datetime import datetime

import pytest

from tinyllm.scaling.horizontal import (
    HorizontalScaler,
    InstanceStatus,
    ScalingDirection,
    ScalingPolicy,
    WorkerInstance,
)


@pytest.fixture
def scaling_policy():
    """Create a scaling policy for testing."""
    return ScalingPolicy(
        min_instances=2,
        max_instances=10,
        target_cpu_percent=70.0,
        scale_up_cooldown_seconds=1,
        scale_down_cooldown_seconds=1,
    )


@pytest.fixture
async def instance_factory():
    """Create an instance factory for testing."""

    async def factory(instance_id: str) -> WorkerInstance:
        return WorkerInstance(
            instance_id=instance_id,
            host="localhost",
            port=8000,
            status=InstanceStatus.STARTING,
        )

    return factory


@pytest.fixture
async def scaler(scaling_policy, instance_factory):
    """Create a horizontal scaler for testing."""
    scaler = HorizontalScaler(
        policy=scaling_policy,
        instance_factory=instance_factory,
        health_check_interval=1,
    )
    await scaler.start()
    yield scaler
    await scaler.stop()


@pytest.mark.asyncio
async def test_scaler_starts_with_min_instances(scaler):
    """Test scaler starts with minimum instances."""
    assert scaler.get_instance_count() >= 2


@pytest.mark.asyncio
async def test_add_instance(scaler):
    """Test adding an instance."""
    initial_count = scaler.get_instance_count()
    instance = await scaler.add_instance("test-instance")

    assert instance.instance_id == "test-instance"
    assert scaler.get_instance_count() == initial_count + 1


@pytest.mark.asyncio
async def test_remove_instance(scaler):
    """Test removing an instance."""
    instance = await scaler.add_instance("test-instance")
    initial_count = scaler.get_instance_count()

    await scaler.remove_instance("test-instance", drain=False)

    assert scaler.get_instance_count() == initial_count - 1
    assert "test-instance" not in scaler.instances


@pytest.mark.asyncio
async def test_scale_up(scaler):
    """Test scaling up."""
    initial_count = scaler.get_instance_count()
    scaled = await scaler.scale(ScalingDirection.UP, count=2)

    assert scaled == 2
    assert scaler.get_instance_count() == initial_count + 2


@pytest.mark.asyncio
async def test_scale_down(scaler):
    """Test scaling down."""
    # Add some instances first
    await scaler.scale(ScalingDirection.UP, count=3)
    await asyncio.sleep(1.1)  # Wait for cooldown

    initial_count = scaler.get_instance_count()
    scaled = await scaler.scale(ScalingDirection.DOWN, count=2)

    assert scaled == 2
    assert scaler.get_instance_count() == initial_count - 2


@pytest.mark.asyncio
async def test_scale_respects_min_instances(scaler):
    """Test scaling down respects minimum instances."""
    # Try to scale below minimum
    await asyncio.sleep(1.1)  # Wait for cooldown
    current = scaler.get_instance_count()
    await scaler.scale(ScalingDirection.DOWN, count=100)

    assert scaler.get_instance_count() >= scaler.policy.min_instances


@pytest.mark.asyncio
async def test_scale_respects_max_instances(scaler):
    """Test scaling up respects maximum instances."""
    # Try to scale above maximum
    await scaler.scale(ScalingDirection.UP, count=100)

    assert scaler.get_instance_count() <= scaler.policy.max_instances


@pytest.mark.asyncio
async def test_scale_cooldown(scaler):
    """Test scale cooldown prevents rapid scaling."""
    await scaler.scale(ScalingDirection.UP, count=1)

    # Immediate second scale should be blocked by cooldown
    scaled = await scaler.scale(ScalingDirection.UP, count=1)
    assert scaled == 0


@pytest.mark.asyncio
async def test_evaluate_scaling_up(scaler):
    """Test evaluation triggers scale up."""
    # Simulate high CPU usage
    for instance in scaler.instances.values():
        instance.cpu_usage = 90.0

    direction = await scaler.evaluate_scaling()
    assert direction == ScalingDirection.UP


@pytest.mark.asyncio
async def test_evaluate_scaling_down(scaler):
    """Test evaluation triggers scale down."""
    # Add extra instances
    await scaler.scale(ScalingDirection.UP, count=3)

    # Simulate low CPU usage
    for instance in scaler.instances.values():
        instance.cpu_usage = 20.0

    direction = await scaler.evaluate_scaling()
    assert direction == ScalingDirection.DOWN


@pytest.mark.asyncio
async def test_get_healthy_instances(scaler):
    """Test getting healthy instances."""
    # Mark some instances as unhealthy
    instances = list(scaler.instances.values())
    if len(instances) > 1:
        instances[0].status = InstanceStatus.UNHEALTHY

    healthy = scaler.get_healthy_instances()
    assert all(i.is_healthy for i in healthy)
    assert len(healthy) < len(instances)


@pytest.mark.asyncio
async def test_get_metrics(scaler):
    """Test getting metrics."""
    metrics = scaler.get_metrics()

    assert "total_instances" in metrics
    assert "healthy_instances" in metrics
    assert "avg_cpu_usage" in metrics
    assert metrics["total_instances"] >= 0


@pytest.mark.asyncio
async def test_instance_health_check(scaler):
    """Test instance health checks."""
    # Add an instance
    instance = await scaler.add_instance("test-health")

    # Wait for health check to run
    await asyncio.sleep(2)

    # Instance should be marked healthy after startup
    assert instance.status in (InstanceStatus.HEALTHY, InstanceStatus.STARTING)


@pytest.mark.asyncio
async def test_select_instances_to_remove(scaler):
    """Test selecting instances to remove prioritizes unhealthy ones."""
    # Add instances with different states
    healthy = await scaler.add_instance("healthy-1")
    unhealthy = await scaler.add_instance("unhealthy-1")
    unhealthy.status = InstanceStatus.UNHEALTHY

    selected = scaler._select_instances_to_remove(1)

    assert "unhealthy-1" in selected
