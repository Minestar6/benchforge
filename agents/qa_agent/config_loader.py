"""Load qa_agent.yaml into AgentConfig and related dataclasses.

Blueprint 不在此加载——由调用方（编排器/规划智能体/run script）
程序化构造后传入 run_generation_agent(blueprint=...)。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchforge.agents.qa_agent.schema import (
    AgentConfig,
    CandidatePoolConfig, InitialBreadthConfig, PlannerConfig,
    ChunkMixConfig, ChunkMixDifficulty, ModeAdjustment,
    GenerationYield, ChunkLimitsForMode, ChunkKLimit, RuntimeConfig,
)
from benchforge.config.config import (
    RetrievalConfig, ChunkingConfig, SummarizationChunkingConfig,
    load_dotenv, expand_env_recursive,
)


@dataclass
class ModelRef:
    """模型引用：逻辑名（指向 model_registry.yaml）+ 调用参数。"""
    name: str                        # model_registry.yaml 中的 key
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3


def load_qa_agent_config(
    path: str | Path,
) -> tuple[AgentConfig, ModelRef, RetrievalConfig, ChunkingConfig, SummarizationChunkingConfig]:
    """加载 qa_agent.yaml，返回纯 agent 行为配置。

    不包含 Blueprint——Blueprint 由调用方程序化构造，
    传入 run_generation_agent(blueprint=blueprint, ...)。
    """
    from benchforge.utils.paths import get_project_root
    path = Path(path)
    load_dotenv(get_project_root() / ".env")

    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f))

    cm = raw["chunk_mix"]
    agent_config = AgentConfig(
        candidate_pool=CandidatePoolConfig(**raw["candidate_pool"]),
        initial_breadth=InitialBreadthConfig(**raw["initial_breadth"]),
        planner=PlannerConfig(**raw["planner"]),
        chunk_mix=ChunkMixConfig(
            by_difficulty={
                d: ChunkMixDifficulty(**v)
                for d, v in cm["by_difficulty"].items()
            },
            mode_adjustment={
                m: ModeAdjustment(**v)
                for m, v in cm.get("mode_adjustment", {}).items()
            },
        ),
        generation_yield={
            m: GenerationYield(**v)
            for m, v in raw["generation_yield"].items()
        },
        chunk_limits={
            m: ChunkLimitsForMode(
                single_k=ChunkKLimit(**v["single_k"]),
                multi_k=ChunkKLimit(**v["multi_k"]),
            )
            for m, v in raw["chunk_limits"].items()
        },
        runtime=RuntimeConfig(**raw["runtime"]),
    )

    model_raw = raw.get("model", {})
    model_ref = ModelRef(
        name=model_raw.get("name", "gpt-4o-mini"),
        temperature=float(model_raw.get("temperature", 0.7)),
        max_tokens=int(model_raw.get("max_tokens", 2000)),
        max_retries=int(model_raw.get("max_retries", 3)),
    )

    retrieval_cfg = RetrievalConfig(**raw["retrieval"]) if "retrieval" in raw else RetrievalConfig()
    chunking_cfg = ChunkingConfig(**raw["chunking"]) if "chunking" in raw else ChunkingConfig()
    sum_chunking_cfg = SummarizationChunkingConfig(**raw["summarization_chunking"]) if "summarization_chunking" in raw else SummarizationChunkingConfig()

    return agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg
