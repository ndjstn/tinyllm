"""Fallback model strategies for TinyLLM.

Provides resilient model execution with automatic fallback to alternate models,
health tracking, and multiple routing strategies.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from tinyllm.logging import get_logger
from tinyllm.models.client import GenerateResponse, OllamaClient, get_shared_client

logger = get_logger(__name__, component="fallback")


class FallbackStrategy(str, Enum):
    """Strategy for fallback model selection."""

    SEQUENTIAL = "sequential"  # Try models in order
    FASTEST = "fastest"  # Race all models, use first response
    LOAD_BALANCED = "load_balanced"  # Distribute based on model health


class FallbackConfig(BaseModel):
    """Configuration for fallback model chains.

    Attributes:
        primary_model: Primary model to try first.
        fallback_models: Ordered list of fallback models by preference.
        retry_on_errors: Error types/patterns to retry on (e.g., ["timeout", "connection", "rate_limit"]).
        timeout_ms: Per-model timeout in milliseconds.
        strategy: Fallback strategy to use.
        max_retries_per_model: Maximum retries per individual model.
        enable_health_tracking: Whether to track model health for smart routing.
        health_check_interval_s: Seconds between health checks.
    """

    model_config = {"extra": "forbid"}

    primary_model: str = Field(description="Primary model name")
    fallback_models: List[str] = Field(
        default_factory=list, description="Ordered fallback models"
    )
    retry_on_errors: List[str] = Field(
        default_factory=lambda: ["timeout", "connection", "rate_limit"],
        description="Error patterns to retry on",
    )
    timeout_ms: int = Field(default=30000, ge=1000, description="Per-model timeout")
    strategy: FallbackStrategy = Field(
        default=FallbackStrategy.SEQUENTIAL, description="Fallback strategy"
    )
    max_retries_per_model: int = Field(
        default=2, ge=0, description="Max retries per model"
    )
    enable_health_tracking: bool = Field(
        default=True, description="Enable model health tracking"
    )
    health_check_interval_s: float = Field(
        default=60.0, ge=1.0, description="Health check interval"
    )


@dataclass
class ModelHealth:
    """Tracks health metrics for a model."""

    model_name: str
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    is_healthy: bool = True

    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)."""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()
        self.consecutive_failures = 0
        self.is_healthy = True

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.consecutive_failures += 1

        # Mark as unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False

    def reset_health(self) -> None:
        """Reset health status (used after recovery period)."""
        self.consecutive_failures = 0
        self.is_healthy = True


@dataclass
class FallbackResult:
    """Result from fallback execution.

    Attributes:
        response: The generated response.
        model_used: Which model actually generated the response.
        attempts: List of models attempted before success.
        total_latency_ms: Total time including retries.
        fallback_occurred: Whether fallback to alternate model happened.
    """

    response: GenerateResponse
    model_used: str
    attempts: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    fallback_occurred: bool = False


class HealthTracker:
    """Tracks health metrics across all models."""

    def __init__(self):
        """Initialize health tracker."""
        self._health: Dict[str, ModelHealth] = {}
        self._lock = asyncio.Lock()

    def get_health(self, model: str) -> ModelHealth:
        """Get or create health metrics for a model."""
        if model not in self._health:
            self._health[model] = ModelHealth(model_name=model)
        return self._health[model]

    async def record_success(self, model: str, latency_ms: float) -> None:
        """Record successful model execution."""
        async with self._lock:
            health = self.get_health(model)
            health.record_success(latency_ms)
            logger.info(
                "model_success_recorded",
                model=model,
                success_rate=health.success_rate,
                avg_latency_ms=health.average_latency_ms,
            )

    async def record_failure(self, model: str, error: str) -> None:
        """Record failed model execution."""
        async with self._lock:
            health = self.get_health(model)
            health.record_failure()
            logger.warning(
                "model_failure_recorded",
                model=model,
                consecutive_failures=health.consecutive_failures,
                is_healthy=health.is_healthy,
                error=error,
            )

    def is_healthy(self, model: str) -> bool:
        """Check if model is healthy."""
        health = self.get_health(model)
        return health.is_healthy

    def get_models_by_health(self, models: List[str]) -> List[str]:
        """Sort models by health score (success rate, then latency)."""
        scored = []
        for model in models:
            health = self.get_health(model)
            # Score combines success rate and inverse latency
            # Unhealthy models get lowest priority
            if not health.is_healthy:
                score = -1000.0
            else:
                # Higher success rate and lower latency = higher score
                latency_score = (
                    1.0 / (health.average_latency_ms + 1.0)
                    if health.average_latency_ms > 0
                    else 1.0
                )
                score = health.success_rate * 100 + latency_score
            scored.append((model, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [model for model, _ in scored]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all health metrics for reporting."""
        return {
            model: {
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "success_rate": health.success_rate,
                "average_latency_ms": health.average_latency_ms,
                "consecutive_failures": health.consecutive_failures,
                "is_healthy": health.is_healthy,
                "last_success": health.last_success_time,
                "last_failure": health.last_failure_time,
            }
            for model, health in self._health.items()
        }


class FallbackClient:
    """Client that implements fallback strategies for model execution.

    Provides automatic fallback to alternate models on failure, with
    health tracking and multiple routing strategies.
    """

    def __init__(
        self,
        config: FallbackConfig,
        host: str = "http://localhost:11434",
    ):
        """Initialize fallback client.

        Args:
            config: Fallback configuration.
            host: Ollama server URL.
        """
        self.config = config
        self.host = host
        self._health_tracker = HealthTracker()
        self._clients: Dict[str, OllamaClient] = {}
        self._model_order_cache: Optional[List[str]] = None
        self._last_health_check = 0.0

    async def _get_client_for_model(self, model: str) -> OllamaClient:
        """Get or create client for a specific model."""
        if model not in self._clients:
            # Use shared client pool for efficiency
            client = await get_shared_client(
                host=self.host,
                timeout_ms=self.config.timeout_ms,
                max_retries=self.config.max_retries_per_model,
            )
            self._clients[model] = client
        return self._clients[model]

    def _get_all_models(self) -> List[str]:
        """Get all models (primary + fallbacks)."""
        return [self.config.primary_model] + self.config.fallback_models

    def _should_retry_error(self, error: Exception) -> bool:
        """Check if error type should trigger retry."""
        error_str = str(error).lower()
        return any(
            pattern.lower() in error_str for pattern in self.config.retry_on_errors
        )

    async def _try_model(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> GenerateResponse:
        """Try to generate with a specific model.

        Args:
            model: Model name.
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            json_mode: Enable JSON mode.

        Returns:
            GenerateResponse on success.

        Raises:
            Exception on failure.
        """
        client = await self._get_client_for_model(model)
        start_time = time.time()

        try:
            response = await client.generate(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            latency_ms = (time.time() - start_time) * 1000

            # Record success
            if self.config.enable_health_tracking:
                await self._health_tracker.record_success(model, latency_ms)

            return response

        except Exception as e:
            # Record failure
            if self.config.enable_health_tracking:
                await self._health_tracker.record_failure(model, str(e))
            raise

    async def _sequential_fallback(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> FallbackResult:
        """Try models sequentially until one succeeds.

        Args:
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            json_mode: Enable JSON mode.

        Returns:
            FallbackResult with response and metadata.

        Raises:
            RuntimeError if all models fail.
        """
        models = self._get_model_order()
        attempts = []
        start_time = time.time()

        for model in models:
            if not self._health_tracker.is_healthy(model):
                logger.info("skipping_unhealthy_model", model=model)
                continue

            attempts.append(model)
            logger.info("trying_model", model=model, attempt=len(attempts))

            try:
                response = await self._try_model(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

                total_latency_ms = (time.time() - start_time) * 1000
                fallback_occurred = model != self.config.primary_model

                logger.info(
                    "model_succeeded",
                    model=model,
                    fallback_occurred=fallback_occurred,
                    total_latency_ms=total_latency_ms,
                )

                return FallbackResult(
                    response=response,
                    model_used=model,
                    attempts=attempts,
                    total_latency_ms=total_latency_ms,
                    fallback_occurred=fallback_occurred,
                )

            except Exception as e:
                logger.warning("model_failed", model=model, error=str(e))

                if not self._should_retry_error(e):
                    logger.error("non_retryable_error", model=model, error=str(e))
                    # Still continue to next model

        # All models failed
        raise RuntimeError(
            f"All models failed. Attempted: {', '.join(attempts)}"
        )

    async def _fastest_fallback(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> FallbackResult:
        """Race all models and return first successful response.

        Args:
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            json_mode: Enable JSON mode.

        Returns:
            FallbackResult with response and metadata.

        Raises:
            RuntimeError if all models fail.
        """
        models = [m for m in self._get_all_models() if self._health_tracker.is_healthy(m)]

        if not models:
            raise RuntimeError("No healthy models available")

        logger.info("racing_models", models=models)
        start_time = time.time()

        # Create tasks for all models
        tasks = [
            asyncio.create_task(
                self._try_model(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            )
            for model in models
        ]

        # Wait for first success
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()

        # Check if any succeeded
        for task in done:
            try:
                response = task.result()
                total_latency_ms = (time.time() - start_time) * 1000

                # Figure out which model succeeded
                model_used = models[tasks.index(task)]
                fallback_occurred = model_used != self.config.primary_model

                logger.info(
                    "fastest_model_won",
                    model=model_used,
                    total_latency_ms=total_latency_ms,
                )

                return FallbackResult(
                    response=response,
                    model_used=model_used,
                    attempts=models,
                    total_latency_ms=total_latency_ms,
                    fallback_occurred=fallback_occurred,
                )
            except Exception:
                continue

        # All failed
        raise RuntimeError("All models failed in race")

    async def _load_balanced_fallback(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> FallbackResult:
        """Select model based on health metrics, then try sequentially.

        Args:
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            json_mode: Enable JSON mode.

        Returns:
            FallbackResult with response and metadata.
        """
        # Get models ordered by health score
        models = self._health_tracker.get_models_by_health(self._get_all_models())
        logger.info("load_balanced_order", models=models)

        # Use sequential strategy with health-ordered models
        attempts = []
        start_time = time.time()

        for model in models:
            if not self._health_tracker.is_healthy(model):
                continue

            attempts.append(model)
            logger.info("trying_model", model=model, attempt=len(attempts))

            try:
                response = await self._try_model(
                    model=model,
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

                total_latency_ms = (time.time() - start_time) * 1000
                fallback_occurred = model != self.config.primary_model

                return FallbackResult(
                    response=response,
                    model_used=model,
                    attempts=attempts,
                    total_latency_ms=total_latency_ms,
                    fallback_occurred=fallback_occurred,
                )

            except Exception as e:
                logger.warning("model_failed", model=model, error=str(e))

        raise RuntimeError(f"All models failed. Attempted: {', '.join(attempts)}")

    def _get_model_order(self) -> List[str]:
        """Get ordered list of models to try based on strategy."""
        # Check if we need to refresh health-based ordering
        if self.config.enable_health_tracking:
            now = time.time()
            if now - self._last_health_check > self.config.health_check_interval_s:
                self._model_order_cache = None
                self._last_health_check = now

        # Use cached order if available
        if self._model_order_cache is not None:
            return self._model_order_cache

        # Build fresh order
        if self.config.strategy == FallbackStrategy.LOAD_BALANCED:
            models = self._health_tracker.get_models_by_health(self._get_all_models())
        else:
            models = self._get_all_models()

        self._model_order_cache = models
        return models

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> FallbackResult:
        """Generate response with fallback support.

        Tries models according to the configured strategy, falling back
        on failure. Tracks health metrics and routes intelligently.

        Args:
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request JSON output.

        Returns:
            FallbackResult with response and metadata about which model was used.

        Raises:
            RuntimeError if all models fail.
        """
        logger.info(
            "starting_fallback_generation",
            strategy=self.config.strategy,
            primary_model=self.config.primary_model,
        )

        if self.config.strategy == FallbackStrategy.SEQUENTIAL:
            return await self._sequential_fallback(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        elif self.config.strategy == FallbackStrategy.FASTEST:
            return await self._fastest_fallback(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        elif self.config.strategy == FallbackStrategy.LOAD_BALANCED:
            return await self._load_balanced_fallback(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for all models.

        Returns:
            Dictionary with per-model metrics including success rate,
            latency, and health status.
        """
        metrics = self._health_tracker.get_all_metrics()

        # Add overall statistics
        total_requests = sum(
            m["success_count"] + m["failure_count"] for m in metrics.values()
        )
        total_successes = sum(m["success_count"] for m in metrics.values())

        return {
            "per_model": metrics,
            "overall": {
                "total_requests": total_requests,
                "total_successes": total_successes,
                "overall_success_rate": (
                    total_successes / total_requests if total_requests > 0 else 0.0
                ),
                "models_tracked": len(metrics),
            },
        }

    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get statistics about fallback behavior.

        Returns:
            Statistics about how often fallback occurred and which models
            are being used most.
        """
        metrics = self._health_tracker.get_all_metrics()

        # Calculate fallback frequency (requests to non-primary models)
        primary_requests = metrics.get(self.config.primary_model, {}).get(
            "success_count", 0
        ) + metrics.get(self.config.primary_model, {}).get("failure_count", 0)

        total_requests = sum(
            m["success_count"] + m["failure_count"] for m in metrics.values()
        )

        fallback_requests = total_requests - primary_requests

        return {
            "total_requests": total_requests,
            "primary_requests": primary_requests,
            "fallback_requests": fallback_requests,
            "fallback_rate": (
                fallback_requests / total_requests if total_requests > 0 else 0.0
            ),
            "models_used": {
                model: m["success_count"] for model, m in metrics.items()
            },
        }
