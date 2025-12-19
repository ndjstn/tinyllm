"""Configuration loader for TinyLLM.

Loads YAML configuration files and validates them against Pydantic models.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """Global system configuration."""

    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR)$")
    log_format: str = Field(default="json", pattern=r"^(json|text)$")
    data_dir: Path = Field(default=Path("data"))
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)


class OllamaConfig(BaseModel):
    """Ollama connection configuration."""

    host: str = Field(default="http://localhost:11434")
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)
    max_retries: int = Field(default=3, ge=0, le=10)


class ModelTier(BaseModel):
    """Configuration for a model tier."""

    name: str
    models: list[str]
    purpose: str
    vram_estimate_mb: int = Field(ge=0)


class ModelsConfig(BaseModel):
    """Model tier configuration."""

    tiers: Dict[str, ModelTier] = Field(default_factory=dict)
    default_router: str = Field(default="qwen2.5:0.5b")
    default_specialist: str = Field(default="qwen2.5:3b")
    default_judge: str = Field(default="qwen3:14b")


class GradingConfig(BaseModel):
    """Grading system configuration."""

    enabled: bool = Field(default=True)
    sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    judge_model: str = Field(default="qwen3:14b")
    threshold_pass: float = Field(default=0.6, ge=0.0, le=1.0)


class ExpansionConfig(BaseModel):
    """Expansion system configuration."""

    enabled: bool = Field(default=True)
    min_executions: int = Field(default=50, ge=10)
    failure_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_nodes: int = Field(default=200, ge=10)
    max_depth: int = Field(default=5, ge=1)
    cooldown_hours: int = Field(default=24, ge=1)


class Config(BaseModel):
    """Complete TinyLLM configuration."""

    version: str = Field(default="1.0")
    environment: str = Field(default="development")

    system: SystemConfig = Field(default_factory=SystemConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    grading: GradingConfig = Field(default_factory=GradingConfig)
    expansion: ExpansionConfig = Field(default_factory=ExpansionConfig)

    graph_file: Optional[Path] = Field(default=None)


def load_config(config_dir: Path | str) -> Config:
    """Load configuration from a directory.

    Args:
        config_dir: Path to configuration directory containing YAML files.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If config directory doesn't exist.
        ValidationError: If configuration is invalid.
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # Load main config
    main_file = config_dir / "tinyllm.yaml"
    if main_file.exists():
        with open(main_file) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Load includes
    includes = data.pop("includes", [])
    for include in includes:
        include_path = config_dir / include
        if include_path.exists():
            with open(include_path) as f:
                include_data = yaml.safe_load(f) or {}
                # Merge with main data
                _deep_merge(data, include_data)

    # Load environment override
    env = data.get("environment", "development")
    env_file = config_dir / "environments" / f"{env}.yaml"
    if env_file.exists():
        with open(env_file) as f:
            env_data = yaml.safe_load(f) or {}
            _deep_merge(data, env_data)

    return Config(**data)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override into base dictionary."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
