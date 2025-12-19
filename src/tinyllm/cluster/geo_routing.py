"""Geo-routing for multi-region deployments."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class GeoRoutingStrategy(str, Enum):
    """Strategy for geo-routing."""

    NEAREST = "nearest"
    LOWEST_LATENCY = "lowest_latency"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    FALLBACK = "fallback"


@dataclass
class RegionInfo:
    """Information about a deployment region."""

    region_id: str
    name: str
    latitude: float
    longitude: float
    endpoint: str
    weight: int = 1
    enabled: bool = True
    health_status: str = "healthy"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate distance to a coordinate using Haversine formula.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Distance in kilometers
        """
        import math

        # Convert to radians
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(lat), math.radians(lon)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in kilometers
        r = 6371
        return c * r


class GeoRoutingConfig(BaseModel):
    """Configuration for geo-routing."""

    default_strategy: GeoRoutingStrategy = Field(
        default=GeoRoutingStrategy.NEAREST, description="Default routing strategy"
    )
    latency_threshold_ms: float = Field(
        default=100, ge=0, description="Latency threshold for routing"
    )
    health_check_interval: int = Field(
        default=30, ge=1, description="Health check interval in seconds"
    )
    enable_failover: bool = Field(default=True, description="Enable failover to other regions")
    cache_ttl: int = Field(default=60, ge=0, description="Routing decision cache TTL in seconds")


@dataclass
class ClientLocation:
    """Client location information."""

    latitude: float
    longitude: float
    country: Optional[str] = None
    city: Optional[str] = None
    ip_address: Optional[str] = None


class GeoRouter:
    """Routes requests based on geographic location."""

    def __init__(self, config: Optional[GeoRoutingConfig] = None):
        """Initialize geo-router.

        Args:
            config: Geo-routing configuration
        """
        self.config = config or GeoRoutingConfig()
        self.regions: Dict[str, RegionInfo] = {}
        self._latency_cache: Dict[Tuple[str, str], float] = {}
        self._routing_cache: Dict[str, Tuple[RegionInfo, datetime]] = {}

        self._shutdown = False
        self._health_check_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the geo-router."""
        self._shutdown = False
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop the geo-router."""
        self._shutdown = True

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    def add_region(self, region: RegionInfo) -> None:
        """Add a region to the router.

        Args:
            region: Region information
        """
        self.regions[region.region_id] = region

    def remove_region(self, region_id: str) -> None:
        """Remove a region from the router.

        Args:
            region_id: Region to remove
        """
        self.regions.pop(region_id, None)

    def enable_region(self, region_id: str) -> None:
        """Enable a region.

        Args:
            region_id: Region to enable
        """
        if region_id in self.regions:
            self.regions[region_id].enabled = True

    def disable_region(self, region_id: str) -> None:
        """Disable a region.

        Args:
            region_id: Region to disable
        """
        if region_id in self.regions:
            self.regions[region_id].enabled = False

    async def route(
        self,
        client_location: ClientLocation,
        strategy: Optional[GeoRoutingStrategy] = None,
    ) -> Optional[RegionInfo]:
        """Route request to best region.

        Args:
            client_location: Client location
            strategy: Routing strategy (uses default if not specified)

        Returns:
            Selected region or None if no regions available
        """
        strategy = strategy or self.config.default_strategy

        # Get healthy, enabled regions
        available_regions = [
            r
            for r in self.regions.values()
            if r.enabled and r.health_status == "healthy"
        ]

        if not available_regions:
            return None

        # Check cache
        cache_key = f"{client_location.latitude},{client_location.longitude}"
        if cache_key in self._routing_cache:
            cached_region, cached_time = self._routing_cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.config.cache_ttl:
                if cached_region.region_id in self.regions:
                    return cached_region

        # Route based on strategy
        if strategy == GeoRoutingStrategy.NEAREST:
            region = self._route_nearest(client_location, available_regions)
        elif strategy == GeoRoutingStrategy.LOWEST_LATENCY:
            region = await self._route_lowest_latency(client_location, available_regions)
        elif strategy == GeoRoutingStrategy.ROUND_ROBIN:
            region = self._route_round_robin(available_regions)
        elif strategy == GeoRoutingStrategy.WEIGHTED:
            region = self._route_weighted(available_regions)
        elif strategy == GeoRoutingStrategy.FALLBACK:
            region = self._route_fallback(client_location, available_regions)
        else:
            region = self._route_nearest(client_location, available_regions)

        # Cache result
        if region:
            self._routing_cache[cache_key] = (region, datetime.utcnow())

        return region

    def _route_nearest(
        self, client_location: ClientLocation, regions: List[RegionInfo]
    ) -> Optional[RegionInfo]:
        """Route to nearest region by geographic distance.

        Args:
            client_location: Client location
            regions: Available regions

        Returns:
            Nearest region
        """
        if not regions:
            return None

        return min(
            regions,
            key=lambda r: r.distance_to(client_location.latitude, client_location.longitude),
        )

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
        if not regions:
            return None

        # Get or measure latency for each region
        latencies = {}
        for region in regions:
            cache_key = (
                f"{client_location.ip_address or 'unknown'}",
                region.region_id,
            )

            if cache_key in self._latency_cache:
                latencies[region.region_id] = self._latency_cache[cache_key]
            else:
                # Measure latency (in production, this would be actual network latency)
                # For now, estimate based on distance
                distance = region.distance_to(
                    client_location.latitude, client_location.longitude
                )
                estimated_latency = distance / 10  # Rough estimate: 10 km/ms
                latencies[region.region_id] = estimated_latency
                self._latency_cache[cache_key] = estimated_latency

        # Return region with lowest latency
        best_region_id = min(latencies.keys(), key=lambda k: latencies[k])
        return self.regions[best_region_id]

    def _route_round_robin(self, regions: List[RegionInfo]) -> Optional[RegionInfo]:
        """Route using round-robin.

        Args:
            regions: Available regions

        Returns:
            Next region in rotation
        """
        if not regions:
            return None

        if not hasattr(self, "_rr_index"):
            self._rr_index = 0

        region = regions[self._rr_index % len(regions)]
        self._rr_index += 1
        return region

    def _route_weighted(self, regions: List[RegionInfo]) -> Optional[RegionInfo]:
        """Route using weighted selection.

        Args:
            regions: Available regions

        Returns:
            Weighted random region
        """
        if not regions:
            return None

        import random

        total_weight = sum(r.weight for r in regions)
        rand = random.uniform(0, total_weight)

        cumulative = 0
        for region in regions:
            cumulative += region.weight
            if rand <= cumulative:
                return region

        return regions[-1]

    def _route_fallback(
        self, client_location: ClientLocation, regions: List[RegionInfo]
    ) -> Optional[RegionInfo]:
        """Route with fallback strategy.

        Tries nearest first, falls back to others if unavailable.

        Args:
            client_location: Client location
            regions: Available regions

        Returns:
            Selected region
        """
        # Try nearest first
        nearest = self._route_nearest(client_location, regions)
        if nearest and nearest.health_status == "healthy":
            return nearest

        # Fallback to any healthy region
        healthy = [r for r in regions if r.health_status == "healthy"]
        if healthy:
            return healthy[0]

        return None

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all regions."""
        for region in self.regions.values():
            try:
                # In production, perform actual health check
                # For now, assume healthy
                region.health_status = "healthy"
            except Exception:
                region.health_status = "unhealthy"

    def get_region_stats(self) -> Dict[str, Any]:
        """Get statistics for all regions.

        Returns:
            Dictionary of region statistics
        """
        return {
            "total_regions": len(self.regions),
            "enabled_regions": sum(1 for r in self.regions.values() if r.enabled),
            "healthy_regions": sum(
                1 for r in self.regions.values() if r.health_status == "healthy"
            ),
            "unhealthy_regions": sum(
                1 for r in self.regions.values() if r.health_status == "unhealthy"
            ),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get geo-routing metrics.

        Returns:
            Dictionary of metrics
        """
        stats = self.get_region_stats()
        stats["cache_size"] = len(self._routing_cache)
        stats["latency_cache_size"] = len(self._latency_cache)
        return stats
