"""Tests for load balancing."""

import pytest

from tinyllm.scaling.horizontal import InstanceStatus, WorkerInstance
from tinyllm.scaling.load_balancer import (
    IPHashBalancer,
    LeastConnectionsBalancer,
    LeastResponseTimeBalancer,
    LoadBalancer,
    LoadBalancingStrategy,
    PowerOfTwoBalancer,
    RandomBalancer,
    RequestContext,
    RoundRobinBalancer,
    WeightedRoundRobinBalancer,
)


@pytest.fixture
def instances():
    """Create test instances."""
    return [
        WorkerInstance(
            instance_id="instance-1",
            host="localhost",
            port=8001,
            status=InstanceStatus.HEALTHY,
            active_requests=5,
        ),
        WorkerInstance(
            instance_id="instance-2",
            host="localhost",
            port=8002,
            status=InstanceStatus.HEALTHY,
            active_requests=3,
        ),
        WorkerInstance(
            instance_id="instance-3",
            host="localhost",
            port=8003,
            status=InstanceStatus.HEALTHY,
            active_requests=8,
        ),
    ]


@pytest.mark.asyncio
async def test_round_robin_balancer(instances):
    """Test round-robin load balancing."""
    balancer = RoundRobinBalancer()

    # Should cycle through instances
    selected1 = await balancer.select_instance(instances)
    selected2 = await balancer.select_instance(instances)
    selected3 = await balancer.select_instance(instances)
    selected4 = await balancer.select_instance(instances)

    assert selected1 != selected2
    assert selected2 != selected3
    assert selected1 == selected4  # Wraps around


@pytest.mark.asyncio
async def test_least_connections_balancer(instances):
    """Test least connections load balancing."""
    balancer = LeastConnectionsBalancer()

    selected = await balancer.select_instance(instances)

    # Should select instance-2 with 3 active requests
    assert selected.instance_id == "instance-2"
    assert selected.active_requests == 3


@pytest.mark.asyncio
async def test_least_response_time_balancer(instances):
    """Test least response time load balancing."""
    balancer = LeastResponseTimeBalancer()

    # Record response times
    balancer.record_response_time("instance-1", 0.5)
    balancer.record_response_time("instance-2", 0.2)
    balancer.record_response_time("instance-3", 0.8)

    selected = await balancer.select_instance(instances)

    # Should select instance-2 with lowest response time
    assert selected.instance_id == "instance-2"


@pytest.mark.asyncio
async def test_weighted_round_robin_balancer(instances):
    """Test weighted round-robin load balancing."""
    weights = {"instance-1": 3, "instance-2": 1, "instance-3": 1}
    balancer = WeightedRoundRobinBalancer(weights=weights)

    # Select multiple times
    selections = []
    for _ in range(10):
        selected = await balancer.select_instance(instances)
        selections.append(selected.instance_id)

    # instance-1 should appear more frequently due to higher weight
    assert selections.count("instance-1") > selections.count("instance-2")


@pytest.mark.asyncio
async def test_ip_hash_balancer(instances):
    """Test IP hash load balancing."""
    balancer = IPHashBalancer()

    context1 = RequestContext(client_ip="192.168.1.1")
    context2 = RequestContext(client_ip="192.168.1.2")

    selected1 = await balancer.select_instance(instances, context1)
    selected2 = await balancer.select_instance(instances, context1)
    selected3 = await balancer.select_instance(instances, context2)

    # Same IP should route to same instance
    assert selected1 == selected2

    # Different IP might route to different instance
    # (not guaranteed due to hash collisions)


@pytest.mark.asyncio
async def test_random_balancer(instances):
    """Test random load balancing."""
    balancer = RandomBalancer()

    selected = await balancer.select_instance(instances)

    assert selected in instances


@pytest.mark.asyncio
async def test_power_of_two_balancer(instances):
    """Test power of two choices load balancing."""
    balancer = PowerOfTwoBalancer()

    selected = await balancer.select_instance(instances)

    # Should select one of the instances
    assert selected in instances


@pytest.mark.asyncio
async def test_load_balancer_select_instance(instances):
    """Test load balancer selects instance."""
    lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)

    selected = await lb.select_instance(instances)

    assert selected is not None
    assert selected in instances


@pytest.mark.asyncio
async def test_load_balancer_filters_unhealthy(instances):
    """Test load balancer filters out unhealthy instances."""
    # Mark one instance as unhealthy
    instances[0].status = InstanceStatus.UNHEALTHY

    lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    selected = await lb.select_instance(instances)

    # Should not select unhealthy instance
    assert selected.instance_id != "instance-1"


@pytest.mark.asyncio
async def test_load_balancer_change_strategy(instances):
    """Test changing load balancing strategy."""
    lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)

    # Change to least connections
    lb.set_strategy(LoadBalancingStrategy.LEAST_CONNECTIONS)

    selected = await lb.select_instance(instances)

    # Should use least connections strategy
    assert selected.instance_id == "instance-2"


@pytest.mark.asyncio
async def test_load_balancer_metrics(instances):
    """Test load balancer metrics."""
    lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)

    # Select some instances
    for _ in range(5):
        await lb.select_instance(instances)

    metrics = lb.get_metrics()

    assert metrics["total_requests"] == 5
    assert metrics["strategy"] == LoadBalancingStrategy.ROUND_ROBIN.value


@pytest.mark.asyncio
async def test_load_balancer_no_instances():
    """Test load balancer with no instances."""
    lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)

    selected = await lb.select_instance([])

    assert selected is None


@pytest.mark.asyncio
async def test_load_balancer_record_response_time(instances):
    """Test recording response times."""
    lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)

    lb.record_response_time("instance-1", 0.5)

    # Should have recorded the response time
    # (internal state check via algorithm)


@pytest.mark.asyncio
async def test_request_context():
    """Test request context creation."""
    context = RequestContext(
        client_ip="192.168.1.1", request_id="req-123", metadata={"user": "test"}
    )

    assert context.client_ip == "192.168.1.1"
    assert context.request_id == "req-123"
    assert context.metadata["user"] == "test"
