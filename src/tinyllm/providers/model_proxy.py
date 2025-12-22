"""Model proxy/gateway for TinyLLM.

Provides load balancing, failover, and routing across multiple model servers.
The proxy implements the same interface as individual clients, making it
composable and transparent to users.

Supports multiple routing strategies:
- Round-robin: Distribute requests evenly across backends
- Random: Pick random backend for each request
- Least-loaded: Route to backend with fewest active requests
- Health-based: Only use healthy backends
- Priority/fallback: Try backends in order until one succeeds
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.metrics import get_metrics_collector

logger = get_logger(__name__, component="model_proxy")
metrics = get_metrics_collector()


# ============================================================================
# Routing Strategies
# ============================================================================


class RoutingStrategyType(str, Enum):
    """Type of routing strategy."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"
    HEALTH_BASED = "health_based"
    PRIORITY = "priority"


class BackendState(BaseModel):
    """State tracking for a backend server."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    client: Any = Field(description="Model client instance")
    name: str = Field(description="Backend name/identifier")
    is_healthy: bool = Field(default=True, description="Health status")
    active_requests: int = Field(default=0, description="Number of active requests")
    total_requests: int = Field(default=0, description="Total requests handled")
    total_failures: int = Field(default=0, description="Total failures")
    last_health_check: float = Field(default=0.0, description="Last health check timestamp")
    last_failure: Optional[float] = Field(default=None, description="Last failure timestamp")


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def select_backend(self, backends: list[BackendState]) -> Optional[BackendState]:
        """Select a backend from the available backends.

        Args:
            backends: List of backend states.

        Returns:
            Selected backend or None if no backends available.
        """
        pass


class RoundRobinStrategy(RoutingStrategy):
    """Round-robin routing strategy."""

    def __init__(self):
        """Initialize round-robin strategy."""
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def select_backend(self, backends: list[BackendState]) -> Optional[BackendState]:
        """Select next backend in round-robin order."""
        if not backends:
            return None

        async with self._lock:
            # Get current backend
            backend = backends[self._current_index % len(backends)]

            # Move to next
            self._current_index = (self._current_index + 1) % len(backends)

            return backend


class RandomStrategy(RoutingStrategy):
    """Random routing strategy."""

    async def select_backend(self, backends: list[BackendState]) -> Optional[BackendState]:
        """Select random backend."""
        if not backends:
            return None

        return random.choice(backends)


class LeastLoadedStrategy(RoutingStrategy):
    """Least-loaded routing strategy."""

    async def select_backend(self, backends: list[BackendState]) -> Optional[BackendState]:
        """Select backend with fewest active requests."""
        if not backends:
            return None

        # Find backend with minimum active requests
        return min(backends, key=lambda b: b.active_requests)


class HealthBasedStrategy(RoutingStrategy):
    """Health-based routing strategy."""

    def __init__(self, fallback_strategy: Optional[RoutingStrategy] = None):
        """Initialize health-based strategy.

        Args:
            fallback_strategy: Strategy to use among healthy backends.
                             Defaults to round-robin.
        """
        self.fallback_strategy = fallback_strategy or RoundRobinStrategy()

    async def select_backend(self, backends: list[BackendState]) -> Optional[BackendState]:
        """Select from healthy backends only."""
        if not backends:
            return None

        # Filter to healthy backends
        healthy_backends = [b for b in backends if b.is_healthy]

        if not healthy_backends:
            logger.warning("no_healthy_backends", total_backends=len(backends))
            return None

        # Use fallback strategy among healthy backends
        return await self.fallback_strategy.select_backend(healthy_backends)


class PriorityStrategy(RoutingStrategy):
    """Priority/fallback routing strategy.

    Always tries backends in order (priority), falling back to next if one fails.
    This is handled at the proxy level, not in selection.
    """

    async def select_backend(self, backends: list[BackendState]) -> Optional[BackendState]:
        """Select first backend (highest priority)."""
        if not backends:
            return None

        return backends[0]


# ============================================================================
# Configuration
# ============================================================================


class ModelProxyConfig(BaseModel):
    """Configuration for model proxy."""

    model_config = {"extra": "forbid"}

    name: str = Field(default="model-proxy", description="Proxy name")
    routing_strategy: RoutingStrategyType = Field(
        default=RoutingStrategyType.ROUND_ROBIN,
        description="Routing strategy to use",
    )
    health_check_interval_s: float = Field(
        default=60.0, ge=0, description="Health check interval in seconds"
    )
    failover_enabled: bool = Field(
        default=True, description="Try other backends if one fails"
    )
    max_retries_per_backend: int = Field(
        default=1, ge=1, description="Max retries per backend"
    )
    backend_timeout_s: float = Field(
        default=5.0, ge=0, description="Seconds to wait before marking backend unhealthy"
    )


# ============================================================================
# Model Proxy
# ============================================================================


class ModelProxy:
    """Load-balancing proxy for multiple model servers.

    Provides transparent load balancing and failover across multiple
    model server backends. Implements the standard client interface,
    so it can be used anywhere a regular client is used.

    Features:
    - Multiple routing strategies (round-robin, random, least-loaded, etc)
    - Automatic failover to healthy backends
    - Health checking and automatic recovery
    - Per-backend statistics tracking
    - Transparent to callers (implements standard client interface)
    """

    def __init__(
        self,
        backends: list[Any],
        config: Optional[ModelProxyConfig] = None,
    ):
        """Initialize model proxy.

        Args:
            backends: List of model clients to proxy to.
            config: Proxy configuration. Uses defaults if not provided.
        """
        if not backends:
            raise ValueError("At least one backend is required")

        self.config = config or ModelProxyConfig()
        self.name = self.config.name

        # Initialize backend states
        self.backends: list[BackendState] = []
        for i, client in enumerate(backends):
            backend_name = getattr(client, "name", f"backend-{i}")
            self.backends.append(
                BackendState(
                    client=client,
                    name=backend_name,
                    is_healthy=True,
                    active_requests=0,
                    total_requests=0,
                    total_failures=0,
                    last_health_check=time.time(),
                )
            )

        # Initialize routing strategy
        self.strategy = self._create_strategy(self.config.routing_strategy)

        # Proxy state
        self._current_graph = "default"
        self._current_model: Optional[str] = None
        self._total_requests = 0
        self._total_failures = 0
        self._health_check_task: Optional[asyncio.Task] = None

    def _create_strategy(self, strategy_type: RoutingStrategyType) -> RoutingStrategy:
        """Create routing strategy instance."""
        if strategy_type == RoutingStrategyType.ROUND_ROBIN:
            return RoundRobinStrategy()
        elif strategy_type == RoutingStrategyType.RANDOM:
            return RandomStrategy()
        elif strategy_type == RoutingStrategyType.LEAST_LOADED:
            return LeastLoadedStrategy()
        elif strategy_type == RoutingStrategyType.HEALTH_BASED:
            return HealthBasedStrategy()
        elif strategy_type == RoutingStrategyType.PRIORITY:
            return PriorityStrategy()
        else:
            raise ValueError(f"Unknown routing strategy: {strategy_type}")

    async def start_health_checks(self) -> None:
        """Start periodic health checking of backends."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("started_health_checks", name=self.name)

    async def stop_health_checks(self) -> None:
        """Stop health checking."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("stopped_health_checks", name=self.name)

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval_s)

                # Check each backend
                for backend in self.backends:
                    try:
                        # Only check if it has check_health method
                        if hasattr(backend.client, "check_health"):
                            is_healthy = await backend.client.check_health()
                            was_healthy = backend.is_healthy
                            backend.is_healthy = is_healthy
                            backend.last_health_check = time.time()

                            # Log health transitions
                            if was_healthy and not is_healthy:
                                logger.warning(
                                    "backend_unhealthy",
                                    proxy=self.name,
                                    backend=backend.name,
                                )
                            elif not was_healthy and is_healthy:
                                logger.info(
                                    "backend_recovered",
                                    proxy=self.name,
                                    backend=backend.name,
                                )
                    except Exception as e:
                        logger.warning(
                            "health_check_failed",
                            proxy=self.name,
                            backend=backend.name,
                            error=str(e),
                        )
                        backend.is_healthy = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", proxy=self.name, error=str(e))

    async def close(self) -> None:
        """Close all backend clients and stop health checks."""
        await self.stop_health_checks()

        for backend in self.backends:
            if hasattr(backend.client, "close"):
                try:
                    await backend.client.close()
                except Exception as e:
                    logger.warning(
                        "backend_close_failed",
                        backend=backend.name,
                        error=str(e),
                    )

        logger.info("proxy_closed", name=self.name)

    def set_graph_context(self, graph: str) -> None:
        """Set current graph context for metrics tracking."""
        self._current_graph = graph

        # Propagate to backends that support it
        for backend in self.backends:
            if hasattr(backend.client, "set_graph_context"):
                backend.client.set_graph_context(graph)

    def set_model(self, model: str) -> None:
        """Set the default model."""
        self._current_model = model

        # Propagate to backends that support it
        for backend in self.backends:
            if hasattr(backend.client, "set_model"):
                backend.client.set_model(model)

    def get_model(self) -> Optional[str]:
        """Get the current default model."""
        return self._current_model

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> Any:
        """Generate a completion via proxy.

        Selects a backend using the routing strategy and forwards the request.
        If failover is enabled and the backend fails, tries other backends.

        Args:
            prompt: Input prompt.
            model: Model name to use.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Returns:
            Response from backend client.

        Raises:
            RuntimeError: If all backends fail.
        """
        self._total_requests += 1

        # Track request with metrics
        model_name = model or self._current_model or "unknown"
        with metrics.track_request_latency(model=model_name, graph=self._current_graph):
            metrics.increment_request_count(
                model=model_name, graph=self._current_graph, request_type="generate"
            )

            # Try backends with routing strategy
            attempted_backends: set[str] = set()
            last_error: Optional[Exception] = None

            while len(attempted_backends) < len(self.backends):
                # Select backend
                available_backends = [
                    b for b in self.backends if b.name not in attempted_backends
                ]

                if not available_backends:
                    break

                backend = await self.strategy.select_backend(available_backends)
                if backend is None:
                    break

                attempted_backends.add(backend.name)

                # Try this backend
                backend.active_requests += 1
                backend.total_requests += 1

                try:
                    logger.debug(
                        "proxy_routing",
                        proxy=self.name,
                        backend=backend.name,
                        prompt=prompt[:50],
                    )

                    result = await backend.client.generate(
                        prompt=prompt,
                        model=model,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    logger.debug(
                        "proxy_success",
                        proxy=self.name,
                        backend=backend.name,
                    )

                    backend.active_requests -= 1
                    return result

                except Exception as e:
                    backend.active_requests -= 1
                    backend.total_failures += 1
                    backend.last_failure = time.time()
                    last_error = e

                    logger.warning(
                        "backend_request_failed",
                        proxy=self.name,
                        backend=backend.name,
                        error=str(e),
                    )

                    # If failover is disabled, fail immediately
                    if not self.config.failover_enabled:
                        break

                    # Otherwise, continue to try next backend

            # All backends failed
            self._total_failures += 1
            logger.error(
                "proxy_all_backends_failed",
                proxy=self.name,
                attempted=len(attempted_backends),
                total_backends=len(self.backends),
            )

            raise RuntimeError(
                f"All backends failed for proxy {self.name}"
            ) from last_error

    async def check_health(self) -> bool:
        """Check if any backend is healthy.

        Returns:
            True if at least one backend is healthy.
        """
        # Check each backend
        for backend in self.backends:
            try:
                if hasattr(backend.client, "check_health"):
                    is_healthy = await backend.client.check_health()
                    if is_healthy:
                        return True
            except Exception:
                pass

        return False

    def get_stats(self) -> dict[str, Any]:
        """Get proxy and backend statistics.

        Returns:
            Dictionary with proxy stats and per-backend stats.
        """
        backend_stats = []
        for backend in self.backends:
            stats = {
                "name": backend.name,
                "is_healthy": backend.is_healthy,
                "active_requests": backend.active_requests,
                "total_requests": backend.total_requests,
                "total_failures": backend.total_failures,
                "last_health_check": backend.last_health_check,
            }

            # Add backend's own stats if available
            if hasattr(backend.client, "get_stats"):
                try:
                    stats["client_stats"] = backend.client.get_stats()
                except Exception:
                    pass

            backend_stats.append(stats)

        return {
            "proxy_name": self.name,
            "routing_strategy": self.config.routing_strategy,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "backends": backend_stats,
        }

    def get_backend_states(self) -> list[BackendState]:
        """Get current state of all backends.

        Returns:
            List of backend states.
        """
        return self.backends.copy()
