"""Global load balancing across multiple regions."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from tinyllm.cluster.geo_routing import ClientLocation, GeoRouter, RegionInfo


class GlobalRoutingPolicy(str, Enum):
    """Policy for global load balancing."""

    GEO_PROXIMITY = "geo_proximity"
    LEAST_LOADED = "least_loaded"
    LOWEST_LATENCY = "lowest_latency"
    COST_OPTIMIZED = "cost_optimized"
    CAPACITY_WEIGHTED = "capacity_weighted"
    MULTI_CRITERIA = "multi_criteria"


class HealthStatus(str, Enum):
    """Health status of a region."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class RegionMetrics:
    """Metrics for a region."""

    region_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    capacity_used: float = 0.0
    cost_per_request: float = 0.0
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_updated: datetime = field(default_factory=datetime.utcnow)


class GlobalLBConfig(BaseModel):
    """Configuration for global load balancer."""

    default_policy: GlobalRoutingPolicy = Field(
        default=GlobalRoutingPolicy.GEO_PROXIMITY,
        description="Default routing policy",
    )
    health_check_interval: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )
    metric_update_interval: int = Field(
        default=10, ge=1, description="Metric update interval in seconds"
    )
    failover_threshold: float = Field(
        default=0.9, ge=0, le=1, description="Capacity threshold for failover"
    )
    enable_auto_failover: bool = Field(
        default=True, description="Enable automatic failover"
    )
    enable_traffic_splitting: bool = Field(
        default=False, description="Enable traffic splitting across regions"
    )
    traffic_split_ratio: Dict[str, float] = Field(
        default_factory=dict, description="Traffic split ratios by region"
    )


class GlobalLoadBalancer:
    """Global load balancer across multiple regions."""

    def __init__(
        self,
        config: Optional[GlobalLBConfig] = None,
        geo_router: Optional[GeoRouter] = None,
        on_region_failover: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize global load balancer.

        Args:
            config: Global LB configuration
            geo_router: Geo-router instance
            on_region_failover: Callback for region failover events
        """
        self.config = config or GlobalLBConfig()
        self.geo_router = geo_router or GeoRouter()
        self.on_region_failover = on_region_failover

        self.region_metrics: Dict[str, RegionMetrics] = {}
        self._routing_decisions: List[Dict[str, Any]] = []
        self._failover_history: List[Dict[str, Any]] = []

        self._shutdown = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._metric_update_task: Optional[asyncio.Task] = None

        # Metrics
        self._total_requests = 0
        self._failed_requests = 0

    async def start(self) -> None:
        """Start the global load balancer."""
        self._shutdown = False
        await self.geo_router.start()

        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metric_update_task = asyncio.create_task(self._metric_update_loop())

    async def stop(self) -> None:
        """Stop the global load balancer."""
        self._shutdown = True

        await self.geo_router.stop()

        for task in [self._health_check_task, self._metric_update_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def add_region(self, region: RegionInfo) -> None:
        """Add a region to the load balancer.

        Args:
            region: Region information
        """
        self.geo_router.add_region(region)
        self.region_metrics[region.region_id] = RegionMetrics(region_id=region.region_id)

    def remove_region(self, region_id: str) -> None:
        """Remove a region from the load balancer.

        Args:
            region_id: Region to remove
        """
        self.geo_router.remove_region(region_id)
        self.region_metrics.pop(region_id, None)

    async def route_request(
        self,
        client_location: ClientLocation,
        policy: Optional[GlobalRoutingPolicy] = None,
    ) -> Optional[RegionInfo]:
        """Route a request to the best region.

        Args:
            client_location: Client location
            policy: Routing policy (uses default if not specified)

        Returns:
            Selected region
        """
        self._total_requests += 1
        policy = policy or self.config.default_policy

        # Get available regions
        available_regions = self._get_available_regions()

        if not available_regions:
            self._failed_requests += 1
            return None

        # Route based on policy
        if policy == GlobalRoutingPolicy.GEO_PROXIMITY:
            region = await self._route_geo_proximity(client_location, available_regions)
        elif policy == GlobalRoutingPolicy.LEAST_LOADED:
            region = self._route_least_loaded(available_regions)
        elif policy == GlobalRoutingPolicy.LOWEST_LATENCY:
            region = await self._route_lowest_latency(client_location, available_regions)
        elif policy == GlobalRoutingPolicy.COST_OPTIMIZED:
            region = self._route_cost_optimized(available_regions)
        elif policy == GlobalRoutingPolicy.CAPACITY_WEIGHTED:
            region = self._route_capacity_weighted(available_regions)
        elif policy == GlobalRoutingPolicy.MULTI_CRITERIA:
            region = await self._route_multi_criteria(client_location, available_regions)
        else:
            region = await self._route_geo_proximity(client_location, available_regions)

        # Record decision
        if region:
            self._record_routing_decision(client_location, region, policy)

        return region

    def _get_available_regions(self) -> List[RegionInfo]:
        """Get list of available regions.

        Returns:
            List of healthy, available regions
        """
        available = []
        for region_id, region in self.geo_router.regions.items():
            if not region.enabled:
                continue

            metrics = self.region_metrics.get(region_id)
            if not metrics:
                continue

            # Check health and capacity
            if metrics.health_status == HealthStatus.HEALTHY:
                if metrics.capacity_used < self.config.failover_threshold:
                    available.append(region)
            elif (
                metrics.health_status == HealthStatus.DEGRADED
                and metrics.capacity_used < 0.7
            ):
                available.append(region)

        return available

    async def _route_geo_proximity(
        self, client_location: ClientLocation, regions: List[RegionInfo]
    ) -> Optional[RegionInfo]:
        """Route based on geographic proximity.

        Args:
            client_location: Client location
            regions: Available regions

        Returns:
            Selected region
        """
        from tinyllm.cluster.geo_routing import GeoRoutingStrategy

        return await self.geo_router.route(client_location, GeoRoutingStrategy.NEAREST)

    def _route_least_loaded(self, regions: List[RegionInfo]) -> Optional[RegionInfo]:
        """Route to least loaded region.

        Args:
            regions: Available regions

        Returns:
            Least loaded region
        """
        if not regions:
            return None

        # Find region with lowest capacity usage
        region_loads = {}
        for region in regions:
            metrics = self.region_metrics.get(region.region_id)
            if metrics:
                region_loads[region.region_id] = metrics.capacity_used

        if not region_loads:
            return regions[0]

        best_region_id = min(region_loads.keys(), key=lambda k: region_loads[k])
        return next(r for r in regions if r.region_id == best_region_id)

    async def _route_lowest_latency(
        self, client_location: ClientLocation, regions: List[RegionInfo]
    ) -> Optional[RegionInfo]:
        """Route to region with lowest latency.

        Args:
            client_location: Client location
            regions: Available regions

        Returns:
            Region with lowest latency
        """
        from tinyllm.cluster.geo_routing import GeoRoutingStrategy

        return await self.geo_router.route(
            client_location, GeoRoutingStrategy.LOWEST_LATENCY
        )

    def _route_cost_optimized(self, regions: List[RegionInfo]) -> Optional[RegionInfo]:
        """Route to most cost-effective region.

        Args:
            regions: Available regions

        Returns:
            Most cost-effective region
        """
        if not regions:
            return None

        # Find region with lowest cost per request
        region_costs = {}
        for region in regions:
            metrics = self.region_metrics.get(region.region_id)
            if metrics:
                region_costs[region.region_id] = metrics.cost_per_request

        if not region_costs:
            return regions[0]

        best_region_id = min(region_costs.keys(), key=lambda k: region_costs[k])
        return next(r for r in regions if r.region_id == best_region_id)

    def _route_capacity_weighted(self, regions: List[RegionInfo]) -> Optional[RegionInfo]:
        """Route using capacity-weighted selection.

        Args:
            regions: Available regions

        Returns:
            Selected region
        """
        if not regions:
            return None

        import random

        # Calculate weights based on available capacity
        weights = {}
        for region in regions:
            metrics = self.region_metrics.get(region.region_id)
            if metrics:
                # Higher weight for more available capacity
                available = 1.0 - metrics.capacity_used
                weights[region.region_id] = max(0.1, available)

        if not weights:
            return regions[0]

        total_weight = sum(weights.values())
        rand = random.uniform(0, total_weight)

        cumulative = 0
        for region in regions:
            weight = weights.get(region.region_id, 0)
            cumulative += weight
            if rand <= cumulative:
                return region

        return regions[-1]

    async def _route_multi_criteria(
        self, client_location: ClientLocation, regions: List[RegionInfo]
    ) -> Optional[RegionInfo]:
        """Route using multiple criteria.

        Balances proximity, load, latency, and cost.

        Args:
            client_location: Client location
            regions: Available regions

        Returns:
            Best region based on multiple criteria
        """
        if not regions:
            return None

        scores = {}
        for region in regions:
            metrics = self.region_metrics.get(region.region_id)
            if not metrics:
                continue

            # Calculate distance score (0-1, lower is better)
            distance = region.distance_to(
                client_location.latitude, client_location.longitude
            )
            distance_score = min(1.0, distance / 10000)  # Normalize to 10k km

            # Calculate load score (0-1, lower is better)
            load_score = metrics.capacity_used

            # Calculate latency score (0-1, lower is better)
            latency_score = min(1.0, metrics.average_latency_ms / 1000)

            # Calculate cost score (0-1, lower is better)
            cost_score = min(1.0, metrics.cost_per_request / 0.001)

            # Weighted composite score (lower is better)
            composite_score = (
                0.4 * distance_score
                + 0.3 * load_score
                + 0.2 * latency_score
                + 0.1 * cost_score
            )

            scores[region.region_id] = composite_score

        if not scores:
            return regions[0]

        best_region_id = min(scores.keys(), key=lambda k: scores[k])
        return next(r for r in regions if r.region_id == best_region_id)

    def _record_routing_decision(
        self, client_location: ClientLocation, region: RegionInfo, policy: GlobalRoutingPolicy
    ) -> None:
        """Record a routing decision for analysis.

        Args:
            client_location: Client location
            region: Selected region
            policy: Routing policy used
        """
        self._routing_decisions.append(
            {
                "timestamp": datetime.utcnow(),
                "client_location": {
                    "lat": client_location.latitude,
                    "lon": client_location.longitude,
                },
                "region_id": region.region_id,
                "policy": policy.value,
            }
        )

        # Limit history size
        if len(self._routing_decisions) > 1000:
            self._routing_decisions = self._routing_decisions[-1000:]

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await self._check_region_health()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_region_health(self) -> None:
        """Check health of all regions."""
        for region_id, metrics in self.region_metrics.items():
            # Determine health based on metrics
            if metrics.error_rate > 0.1:
                new_status = HealthStatus.UNHEALTHY
            elif metrics.error_rate > 0.05 or metrics.capacity_used > 0.9:
                new_status = HealthStatus.DEGRADED
            else:
                new_status = HealthStatus.HEALTHY

            # Handle failover if needed
            if (
                new_status == HealthStatus.UNHEALTHY
                and metrics.health_status == HealthStatus.HEALTHY
            ):
                if self.config.enable_auto_failover:
                    await self._perform_failover(region_id)

            metrics.health_status = new_status

    async def _metric_update_loop(self) -> None:
        """Background metric update loop."""
        while not self._shutdown:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.config.metric_update_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.config.metric_update_interval)

    async def _update_metrics(self) -> None:
        """Update metrics for all regions."""
        # In production, fetch real metrics from each region
        # For now, just update timestamps
        for metrics in self.region_metrics.values():
            metrics.last_updated = datetime.utcnow()

    async def _perform_failover(self, failed_region_id: str) -> None:
        """Perform failover from failed region.

        Args:
            failed_region_id: ID of failed region
        """
        # Disable failed region
        self.geo_router.disable_region(failed_region_id)

        # Record failover
        self._failover_history.append(
            {
                "timestamp": datetime.utcnow(),
                "failed_region": failed_region_id,
                "reason": "health_check_failed",
            }
        )

        # Notify callback
        if self.on_region_failover:
            self.on_region_failover(failed_region_id, "health_check_failed")

    def update_region_metrics(self, region_id: str, metrics: RegionMetrics) -> None:
        """Update metrics for a region.

        Args:
            region_id: Region ID
            metrics: Updated metrics
        """
        self.region_metrics[region_id] = metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Get global load balancer metrics.

        Returns:
            Dictionary of metrics
        """
        total_regions = len(self.region_metrics)
        healthy_regions = sum(
            1
            for m in self.region_metrics.values()
            if m.health_status == HealthStatus.HEALTHY
        )

        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (self._total_requests - self._failed_requests)
            / self._total_requests
            if self._total_requests > 0
            else 0,
            "total_regions": total_regions,
            "healthy_regions": healthy_regions,
            "degraded_regions": sum(
                1
                for m in self.region_metrics.values()
                if m.health_status == HealthStatus.DEGRADED
            ),
            "unhealthy_regions": sum(
                1
                for m in self.region_metrics.values()
                if m.health_status == HealthStatus.UNHEALTHY
            ),
            "total_failovers": len(self._failover_history),
        }
