"""Load balancing strategies for horizontal scaling."""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from tinyllm.scaling.horizontal import WorkerInstance


class LoadBalancingStrategy(str, Enum):
    """Available load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    POWER_OF_TWO = "power_of_two"


@dataclass
class RequestContext:
    """Context for a load balancing request."""

    client_ip: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class LoadBalancingAlgorithm(ABC):
    """Base class for load balancing algorithms."""

    @abstractmethod
    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select an instance for the request.

        Args:
            instances: Available worker instances
            context: Request context

        Returns:
            Selected instance or None if no instances available
        """
        pass


class RoundRobinBalancer(LoadBalancingAlgorithm):
    """Round-robin load balancing."""

    def __init__(self) -> None:
        self._index = 0

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select next instance in round-robin order."""
        if not instances:
            return None

        instance = instances[self._index % len(instances)]
        self._index += 1
        return instance


class LeastConnectionsBalancer(LoadBalancingAlgorithm):
    """Least connections load balancing."""

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select instance with fewest active connections."""
        if not instances:
            return None

        return min(instances, key=lambda x: x.active_requests)


class LeastResponseTimeBalancer(LoadBalancingAlgorithm):
    """Least response time load balancing."""

    def __init__(self) -> None:
        self._response_times: Dict[str, List[float]] = {}
        self._max_samples = 100

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select instance with lowest average response time."""
        if not instances:
            return None

        # Calculate average response times
        avg_times = {}
        for instance in instances:
            times = self._response_times.get(instance.instance_id, [])
            avg_times[instance.instance_id] = sum(times) / len(times) if times else 0

        # Select instance with lowest response time
        # Fallback to least connections if no response time data
        if not avg_times or all(t == 0 for t in avg_times.values()):
            return min(instances, key=lambda x: x.active_requests)

        return min(instances, key=lambda x: avg_times.get(x.instance_id, float("inf")))

    def record_response_time(self, instance_id: str, response_time: float) -> None:
        """Record response time for an instance."""
        if instance_id not in self._response_times:
            self._response_times[instance_id] = []

        times = self._response_times[instance_id]
        times.append(response_time)

        # Keep only recent samples
        if len(times) > self._max_samples:
            self._response_times[instance_id] = times[-self._max_samples :]


class WeightedRoundRobinBalancer(LoadBalancingAlgorithm):
    """Weighted round-robin load balancing."""

    def __init__(self, weights: Optional[Dict[str, int]] = None) -> None:
        self.weights = weights or {}
        self._index = 0

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select instance based on weights."""
        if not instances:
            return None

        # Build weighted list
        weighted_instances = []
        for instance in instances:
            weight = self.weights.get(instance.instance_id, 1)
            weighted_instances.extend([instance] * weight)

        if not weighted_instances:
            return instances[0]

        instance = weighted_instances[self._index % len(weighted_instances)]
        self._index += 1
        return instance


class IPHashBalancer(LoadBalancingAlgorithm):
    """IP hash load balancing for sticky sessions."""

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select instance based on client IP hash."""
        if not instances:
            return None

        if not context or not context.client_ip:
            # Fallback to random if no IP
            return random.choice(instances)

        # Hash IP to select instance
        ip_hash = hash(context.client_ip)
        index = ip_hash % len(instances)
        return instances[index]


class RandomBalancer(LoadBalancingAlgorithm):
    """Random load balancing."""

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select random instance."""
        if not instances:
            return None

        return random.choice(instances)


class PowerOfTwoBalancer(LoadBalancingAlgorithm):
    """Power of two choices load balancing."""

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select best of two random instances."""
        if not instances:
            return None

        if len(instances) == 1:
            return instances[0]

        # Pick two random instances
        choices = random.sample(instances, min(2, len(instances)))

        # Return the one with fewer active requests
        return min(choices, key=lambda x: x.active_requests)


class LoadBalancer:
    """Load balancer with multiple strategies."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """Initialize load balancer.

        Args:
            strategy: Load balancing strategy to use
        """
        self.strategy = strategy
        self._algorithm = self._create_algorithm(strategy)
        self._request_count = 0
        self._error_count = 0

    def _create_algorithm(self, strategy: LoadBalancingStrategy) -> LoadBalancingAlgorithm:
        """Create algorithm instance for strategy."""
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return RoundRobinBalancer()
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return LeastConnectionsBalancer()
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return LeastResponseTimeBalancer()
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return WeightedRoundRobinBalancer()
        elif strategy == LoadBalancingStrategy.IP_HASH:
            return IPHashBalancer()
        elif strategy == LoadBalancingStrategy.RANDOM:
            return RandomBalancer()
        elif strategy == LoadBalancingStrategy.POWER_OF_TWO:
            return PowerOfTwoBalancer()
        else:
            return RoundRobinBalancer()

    async def select_instance(
        self, instances: List[WorkerInstance], context: Optional[RequestContext] = None
    ) -> Optional[WorkerInstance]:
        """Select an instance using the configured strategy.

        Args:
            instances: Available instances
            context: Request context

        Returns:
            Selected instance
        """
        # Filter to only healthy instances
        healthy = [i for i in instances if i.can_accept_requests]

        if not healthy:
            return None

        instance = await self._algorithm.select_instance(healthy, context)
        if instance:
            self._request_count += 1

        return instance

    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Change load balancing strategy.

        Args:
            strategy: New strategy to use
        """
        self.strategy = strategy
        self._algorithm = self._create_algorithm(strategy)

    def record_error(self) -> None:
        """Record a load balancing error."""
        self._error_count += 1

    def record_response_time(self, instance_id: str, response_time: float) -> None:
        """Record response time for an instance.

        Args:
            instance_id: Instance identifier
            response_time: Response time in seconds
        """
        if isinstance(self._algorithm, LeastResponseTimeBalancer):
            self._algorithm.record_response_time(instance_id, response_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get load balancer metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "strategy": self.strategy.value,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / self._request_count
            if self._request_count > 0
            else 0,
        }
