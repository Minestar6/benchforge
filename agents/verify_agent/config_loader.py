"""verify_agent.yaml → dataclass 配置加载。"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from benchforge.config.config import load_dotenv, expand_env_recursive


@dataclass
class CitationCfg:
    enabled: bool = True
    min_citation_score: float = 0.65
    alpha: float = 0.7
    beta: float = 0.3
    citation_match_threshold: float = 0.8


@dataclass
class LLMValidationCfg:
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 800
    min_overall_score: float = 0.75
    max_concurrency: int = 8
    max_retries: int = 2
    hard_floor: dict = field(default_factory=lambda: {
        "clarity": 0.6,
        "answerability": 0.7,
        "faithfulness": 0.7,
        "mode_alignment": 0.7,
    })
    prompt_system: str = "benchforge/prompts/verify_agent/quality_system_prompt.md"
    prompt_user: str = "benchforge/prompts/verify_agent/quality_user_prompt.md"


@dataclass
class SelectionCfg:
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class VerifyAgentConfig:
    task_id: str = ""
    run_id: str = ""
    input_paths: list[str] = field(default_factory=list)
    chunk_index_path: str = ""
    citation: CitationCfg = field(default_factory=CitationCfg)
    llm_validation: LLMValidationCfg = field(default_factory=LLMValidationCfg)
    selection: SelectionCfg = field(default_factory=SelectionCfg)


def load_verify_agent_config(path: str | Path) -> VerifyAgentConfig:
    """加载 verify_agent.yaml，返回 VerifyAgentConfig。"""
    from benchforge.utils.paths import get_project_root
    path = Path(path)
    load_dotenv(get_project_root() / ".env")

    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f))

    run = raw.get("run", {})

    citation_raw = raw.get("citation_validation", {})
    llm_raw = raw.get("llm_validation", {})
    selection_raw = raw.get("selection", {})

    citation_cfg = CitationCfg(
        enabled=citation_raw.get("enabled", True),
        min_citation_score=citation_raw.get("min_citation_score", 0.65),
        alpha=citation_raw.get("alpha", 0.7),
        beta=citation_raw.get("beta", 0.3),
        citation_match_threshold=citation_raw.get("citation_match_threshold", 0.8),
    )

    hard_floor_default = {"clarity": 0.6, "answerability": 0.7, "faithfulness": 0.7, "mode_alignment": 0.7}
    llm_cfg = LLMValidationCfg(
        enabled=llm_raw.get("enabled", True),
        model=llm_raw.get("model", "gpt-4o-mini"),
        temperature=llm_raw.get("temperature", 0.0),
        max_tokens=llm_raw.get("max_tokens", 800),
        min_overall_score=llm_raw.get("min_overall_score", 0.75),
        max_concurrency=llm_raw.get("max_concurrency", 8),
        max_retries=llm_raw.get("max_retries", 2),
        hard_floor=llm_raw.get("hard_floor", hard_floor_default),
        prompt_system=llm_raw.get("prompt_system", "benchforge/prompts/verify_agent/quality_system_prompt.md"),
        prompt_user=llm_raw.get("prompt_user", "benchforge/prompts/verify_agent/quality_user_prompt.md"),
    )

    selection_cfg = SelectionCfg(
        enabled=selection_raw.get("enabled", True),
        embedding_model=selection_raw.get("embedding_model", "all-MiniLM-L6-v2"),
    )

    return VerifyAgentConfig(
        task_id=run.get("task_id", ""),
        run_id=run.get("run_id", ""),
        input_paths=run.get("input_paths", []),
        chunk_index_path=run.get("chunk_index_path", ""),
        citation=citation_cfg,
        llm_validation=llm_cfg,
        selection=selection_cfg,
    )
