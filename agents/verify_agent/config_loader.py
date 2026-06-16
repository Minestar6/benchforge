"""verify_agent.yaml → dataclass 配置加载。"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from benchforge.config.config import load_dotenv, expand_env_recursive


@dataclass
class CitationCfg:
    enabled: bool = True
    min_citation_score: float = 0.65
    min_chunk_citation_score: float = 0.85
    min_answer_citation_score: float = 0.75
    alpha: float = 0.7
    beta: float = 0.3
    citation_match_threshold: float = 0.8


@dataclass
class LLMValidationCfg:
    enabled: bool = True
    model: str = "gpt-4o-mini"      # model_registry.yaml 中的逻辑名
    temperature: float = 0.0          # 调用参数（不在 registry 中）
    max_tokens: int = 800
    min_overall_score: float = 0.75
    max_concurrency: int = 8
    max_retries: int = 2
    prompt_path: str = "benchforge/prompts/verify_agent/quality_prompt.md"
    prompt_user: str = ""


@dataclass
class SelectionCfg:
    mode: str = "light"
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_similarity_threshold: float = 0.9
    """sentence-transformers 模型。支持三种形式：
    - model_registry.yaml embeddings 段中的逻辑名（优先解析）
    - HuggingFace Hub ID（如 "all-MiniLM-L6-v2"，自动下载）
    - 本地文件系统路径（如 "D:/models/all-MiniLM-L6-v2"）
    """


@dataclass
class VerifyAgentConfig:
    """verify_agent 行为配置。

    task_id / run_id / input_paths 由流水线 shared_state.json 提供，
    不在此配置中。仅在独立运行时由调用方通过 run_verify_agent() 函数参数传入。
    """
    citation: CitationCfg = field(default_factory=CitationCfg)
    llm_validation: LLMValidationCfg = field(default_factory=LLMValidationCfg)
    selection: SelectionCfg = field(default_factory=SelectionCfg)


def _resolve_selection_mode(selection_raw: dict) -> str:
    mode = selection_raw.get("mode")
    if mode is None:
        mode = "light" if selection_raw.get("enabled", True) else "off"
    mode = str(mode).strip().lower()
    if mode not in {"off", "light", "strict"}:
        mode = "light"
    return mode


def load_verify_agent_config(path: str | Path) -> VerifyAgentConfig:
    """加载 verify_agent.yaml，返回 VerifyAgentConfig。

    llm_validation.model 是 model_registry.yaml 中的逻辑名，
    由调用方通过 ModelRegistryLoader + ModelLoader 解析为客户端。
    """
    from benchforge.utils.paths import get_project_root
    path = Path(path)
    load_dotenv(get_project_root() / ".env")

    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f))

    citation_raw = raw.get("citation_validation", {})
    llm_raw = raw.get("llm_validation", {})
    selection_raw = raw.get("selection", {})

    citation_cfg = CitationCfg(
        enabled=citation_raw.get("enabled", True),
        min_citation_score=citation_raw.get("min_citation_score", 0.65),
        min_chunk_citation_score=citation_raw.get("min_chunk_citation_score", 0.85),
        min_answer_citation_score=citation_raw.get("min_answer_citation_score", 0.75),
        alpha=citation_raw.get("alpha", 0.7),
        beta=citation_raw.get("beta", 0.3),
        citation_match_threshold=citation_raw.get("citation_match_threshold", 0.8),
    )

    llm_cfg = LLMValidationCfg(
        enabled=llm_raw.get("enabled", True),
        model=llm_raw.get("model", "gpt-4o-mini"),
        temperature=llm_raw.get("temperature", 0.0),
        max_tokens=llm_raw.get("max_tokens", 800),
        min_overall_score=llm_raw.get("min_overall_score", 0.75),
        max_concurrency=llm_raw.get("max_concurrency", 8),
        max_retries=llm_raw.get("max_retries", 2),
        prompt_path=llm_raw.get("prompt_path", "benchforge/prompts/verify_agent/quality_prompt.md"),
        prompt_user=llm_raw.get("prompt_user", ""),
    )

    selection_cfg = SelectionCfg(
        mode=_resolve_selection_mode(selection_raw),
        embedding_model=selection_raw.get("embedding_model", "all-MiniLM-L6-v2"),
        semantic_similarity_threshold=float(selection_raw.get("semantic_similarity_threshold", 0.9)),
    )

    return VerifyAgentConfig(
        citation=citation_cfg,
        llm_validation=llm_cfg,
        selection=selection_cfg,
    )
