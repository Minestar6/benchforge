"""model_registry.yaml → ModelConfig 列表加载。"""

from pathlib import Path
from typing import Any

import yaml

from benchforge.config.config import ModelConfig, expand_env_recursive


def load_model_registry(path: str | Path) -> dict[str, ModelConfig]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f))

    registry: dict[str, ModelConfig] = {}
    for name, cfg in raw.get("models", {}).items():
        registry[name] = ModelConfig(
            model_name=cfg.get("model_name", name),
            provider=cfg.get("provider", "openai"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
        )
    return registry


def resolve_model_config(
    name: str,
    registry: dict[str, ModelConfig],
    generation_defaults: dict[str, Any],
) -> ModelConfig:
    if name not in registry:
        raise KeyError(f"Model '{name}' not found in registry")
    cfg = registry[name]
    cfg.temperature = generation_defaults.get("temperature", cfg.temperature)
    cfg.max_tokens = generation_defaults.get("max_tokens", cfg.max_tokens)
    return cfg
