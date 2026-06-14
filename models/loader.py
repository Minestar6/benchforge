"""统一模型加载器——适配 pydantic ModelConfig（config.py）。"""
from typing import Any
import os

from benchforge.models.base import BaseModelClient
from benchforge.models.openai_client import OpenAIClient
from benchforge.models.fake import FakeModelClient


class ModelLoader:
    """统一模型加载器。

    接收 pydantic ModelConfig（config.py 中的，来自 model_registry_loader），
    返回 BaseModelClient 实例。
    默认温度 / max_tokens 从 agent YAML 的 model 段获取，
    Loader 不规定——使用 getattr 取默认值。
    """

    _provider_map = {
        "openai": OpenAIClient,
        "fake": FakeModelClient,
    }

    @classmethod
    def load_model(cls, config: Any) -> BaseModelClient:
        """加载单个模型。

        Args:
            config: pydantic ModelConfig 或兼容对象
                .provider, .api_key, .base_url, .model_name,
                .max_concurrent_requests, .max_retries
        """
        provider = getattr(config, "provider", "openai") or "openai"
        client_cls = cls._provider_map.get(provider.lower())
        if client_cls is None:
            raise ValueError(f"Unsupported provider: {provider}")

        if provider == "fake":
            delay = getattr(config, "extra_parameters", {}).get("delay", 0.1)
            client = FakeModelClient(delay=delay)
        else:
            client = client_cls(
                api_key=getattr(config, "api_key", "") or "",
                base_url=getattr(config, "base_url", "") or "",
                model_name=getattr(config, "model_name", "gpt-4o-mini"),
                temperature=getattr(config, "temperature", 0.7),
                max_tokens=getattr(config, "max_tokens", 2000),
                max_retries=getattr(config, "max_retries", 3),
            )

        client.model_name = getattr(config, "model_name", "unknown")
        client.provider = provider
        client.max_concurrent = getattr(config, "max_concurrent_requests", 4)
        client.extra_parameters = getattr(config, "extra_parameters", {})

        return client

    @classmethod
    def load_models(cls, configs: list, step_name: str | None = None) -> list:
        clients = []
        for cfg in configs:
            try:
                clients.append(cls.load_model(cfg))
            except Exception as e:
                print(f"Failed to load model {getattr(cfg, 'model_name', cfg)}: {e}")
        if step_name:
            names = [getattr(c, 'model_name', '?') for c in clients]
            print(f"Step '{step_name}' loaded {len(clients)} models: {names}")
        return clients
