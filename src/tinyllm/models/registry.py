"""Model Registry for tracking available models and their capabilities."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tinyllm.logging import get_logger

logger = get_logger(__name__, component="model_registry")


@dataclass
class ModelCapabilities:
    """Capabilities and metadata for a model."""

    name: str
    context_size: Optional[int] = None
    vision_support: bool = False
    embedding_support: bool = False
    size_gb: Optional[float] = None
    family: Optional[str] = None
    parameter_count: Optional[str] = None


@dataclass
class ModelHealth:
    """Health and performance statistics for a model."""

    model_name: str
    total_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_used: Optional[float] = None
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 100.0
        return ((self.total_requests - self.failed_requests) / self.total_requests) * 100

    @property
    def is_healthy(self) -> bool:
        """Determine if model is healthy based on recent performance."""
        # If never used, assume healthy
        if self.total_requests == 0:
            return True

        # If success rate is below 50%, unhealthy
        if self.success_rate < 50.0:
            return False

        # If last error was recent (within 60 seconds), unhealthy
        if self.last_error_time and (time.time() - self.last_error_time) < 60:
            return False

        return True


class ModelRegistry:
    """Registry for tracking available models and their capabilities."""

    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, ModelCapabilities] = {}
        self._health: Dict[str, ModelHealth] = {}
        self._aliases: Dict[str, str] = {
            "fast": "qwen2.5:0.5b",
            "tiny": "tinyllama",
            "code": "granite-code:3b",
            "medium": "qwen2.5:3b",
            "large": "qwen3:8b",
            "judge": "qwen3:14b",
        }
        self._default_model: Optional[str] = None
        logger.info("model_registry_initialized")

    def register_model(self, capabilities: ModelCapabilities) -> None:
        """Register a model with its capabilities.

        Args:
            capabilities: Model capabilities and metadata.
        """
        self._models[capabilities.name] = capabilities
        if capabilities.name not in self._health:
            self._health[capabilities.name] = ModelHealth(model_name=capabilities.name)
        logger.info("model_registered", model=capabilities.name)

    def get_model(self, name: str) -> Optional[ModelCapabilities]:
        """Get model capabilities by name or alias.

        Args:
            name: Model name or alias.

        Returns:
            ModelCapabilities if found, None otherwise.
        """
        # Check if it's an alias
        actual_name = self._aliases.get(name, name)
        return self._models.get(actual_name)

    def list_models(self) -> List[str]:
        """List all registered models.

        Returns:
            List of model names.
        """
        return list(self._models.keys())

    def list_aliases(self) -> Dict[str, str]:
        """List all model aliases.

        Returns:
            Dictionary mapping alias to model name.
        """
        return self._aliases.copy()

    def add_alias(self, alias: str, model_name: str) -> None:
        """Add or update a model alias.

        Args:
            alias: Alias name.
            model_name: Actual model name.
        """
        self._aliases[alias] = model_name
        logger.info("alias_added", alias=alias, model=model_name)

    def remove_alias(self, alias: str) -> None:
        """Remove a model alias.

        Args:
            alias: Alias to remove.
        """
        if alias in self._aliases:
            del self._aliases[alias]
            logger.info("alias_removed", alias=alias)

    def resolve_name(self, name: str) -> str:
        """Resolve a model name or alias to the actual model name.

        Args:
            name: Model name or alias.

        Returns:
            Actual model name.
        """
        return self._aliases.get(name, name)

    def get_health(self, name: str) -> Optional[ModelHealth]:
        """Get health statistics for a model.

        Args:
            name: Model name or alias.

        Returns:
            ModelHealth if found, None otherwise.
        """
        actual_name = self.resolve_name(name)
        return self._health.get(actual_name)

    def record_request(
        self,
        name: str,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a request to a model for health tracking.

        Args:
            name: Model name or alias.
            latency_ms: Request latency in milliseconds.
            success: Whether the request succeeded.
            error: Error message if request failed.
        """
        actual_name = self.resolve_name(name)

        if actual_name not in self._health:
            self._health[actual_name] = ModelHealth(model_name=actual_name)

        health = self._health[actual_name]
        health.total_requests += 1
        health.total_latency_ms += latency_ms
        health.last_used = time.time()

        if not success:
            health.failed_requests += 1
            health.last_error = error
            health.last_error_time = time.time()
            logger.warning(
                "model_request_failed",
                model=actual_name,
                error=error,
                failure_rate=100 - health.success_rate,
            )

    def get_healthy_models(self) -> List[str]:
        """Get list of healthy models.

        Returns:
            List of model names that are currently healthy.
        """
        return [
            name
            for name, health in self._health.items()
            if health.is_healthy
        ]

    def set_default_model(self, name: str) -> None:
        """Set the default model.

        Args:
            name: Model name or alias.
        """
        actual_name = self.resolve_name(name)
        self._default_model = actual_name
        logger.info("default_model_set", model=actual_name)

    def get_default_model(self) -> Optional[str]:
        """Get the default model name.

        Returns:
            Default model name if set, None otherwise.
        """
        return self._default_model

    def sync_from_ollama(self, model_list: List[str]) -> None:
        """Sync registry with models available in Ollama.

        Args:
            model_list: List of model names from Ollama.
        """
        for model_name in model_list:
            if model_name not in self._models:
                # Create basic capabilities for unknown models
                capabilities = self._parse_model_capabilities(model_name)
                self.register_model(capabilities)

        logger.info("registry_synced", model_count=len(model_list))

    def _parse_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Parse model name to infer capabilities.

        Args:
            model_name: Model name (e.g., "qwen2.5:3b").

        Returns:
            ModelCapabilities with inferred values.
        """
        # Extract family and size from name
        family = model_name.split(":")[0] if ":" in model_name else model_name

        # Infer parameter count
        param_count = None
        if "0.5b" in model_name.lower():
            param_count = "0.5B"
        elif "1.5b" in model_name.lower():
            param_count = "1.5B"
        elif "3b" in model_name.lower():
            param_count = "3B"
        elif "7b" in model_name.lower():
            param_count = "7B"
        elif "8b" in model_name.lower():
            param_count = "8B"
        elif "14b" in model_name.lower():
            param_count = "14B"

        # Infer context size (common defaults)
        context_size = 4096
        if "qwen" in family.lower():
            context_size = 32768
        elif "llama" in family.lower():
            context_size = 8192

        # Check for vision support
        vision_support = "vision" in model_name.lower() or "llava" in model_name.lower()

        return ModelCapabilities(
            name=model_name,
            context_size=context_size,
            vision_support=vision_support,
            embedding_support=False,
            family=family,
            parameter_count=param_count,
        )

    def get_model_info(self, name: str) -> Dict[str, any]:
        """Get comprehensive information about a model.

        Args:
            name: Model name or alias.

        Returns:
            Dictionary with model capabilities and health stats.
        """
        actual_name = self.resolve_name(name)
        capabilities = self._models.get(actual_name)
        health = self._health.get(actual_name)

        info = {
            "name": actual_name,
            "exists": capabilities is not None,
        }

        if capabilities:
            info.update({
                "family": capabilities.family,
                "parameters": capabilities.parameter_count,
                "context_size": capabilities.context_size,
                "vision_support": capabilities.vision_support,
            })

        if health:
            info.update({
                "total_requests": health.total_requests,
                "success_rate": f"{health.success_rate:.1f}%",
                "avg_latency_ms": f"{health.average_latency_ms:.1f}",
                "is_healthy": health.is_healthy,
            })

        return info


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        Global ModelRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
