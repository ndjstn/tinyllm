"""Ollama model integration."""

from tinyllm.models.client import OllamaClient, get_shared_client, close_all_clients
from tinyllm.models.fallback import (
    FallbackClient,
    FallbackConfig,
    FallbackResult,
    FallbackStrategy,
    HealthTracker,
    ModelHealth,
)
from tinyllm.models.registry import (
    ModelRegistry,
    ModelCapabilities,
    ModelHealth as RegistryModelHealth,
    get_model_registry,
)

__all__ = [
    "OllamaClient",
    "get_shared_client",
    "close_all_clients",
    "FallbackClient",
    "FallbackConfig",
    "FallbackResult",
    "FallbackStrategy",
    "HealthTracker",
    "ModelHealth",
    "ModelRegistry",
    "ModelCapabilities",
    "RegistryModelHealth",
    "get_model_registry",
]
