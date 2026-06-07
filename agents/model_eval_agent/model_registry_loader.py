"""model_registry.yaml → ModelConfig 列表加载。"""

from pathlib import Path
from typing import Any

import yaml

from benchforge.config.config import ModelConfig, expand_env_recursive


def load_model_registry(path: str | Path) -> dict[str, ModelConfig]:
    """从 model_registry.yaml 加载全部模型注册项。

    Returns:
        {逻辑名: ModelConfig} 字典。ModelConfig 仅包含身份和连接信息，
        不包含 temperature / max_tokens（这些是调用参数，在各自 agent YAML 中）。
    """
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
            max_concurrent_requests=cfg.get("max_concurrent_requests", 4),
            max_retries=cfg.get("max_retries", 3),
            extra_parameters=cfg.get("extra_parameters", {}),
        )
    return registry


def resolve_model_config(
    name: str,
    registry: dict[str, ModelConfig],
) -> ModelConfig:
    """从 registry 中查找模型配置。

    不修改 ModelConfig 中的任何字段（调用参数在 agent YAML 中独立维护）。

    Args:
        name: model_registry.yaml 中的逻辑名
        registry: load_model_registry() 的返回值

    Returns:
        ModelConfig（纯身份+连接信息）

    Raises:
        KeyError: 逻辑名不在 registry 中
    """
    if name not in registry:
        raise KeyError(
            f"Model '{name}' not found in registry. Available: {list(registry.keys())}"
        )
    return registry[name]
