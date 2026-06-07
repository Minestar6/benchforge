"""Load qa_agent.yaml into Blueprint and AgentConfig dataclasses."""

from pathlib import Path

import yaml

from benchforge.agents.qa_agent.schema import (
    Blueprint, ModeCfg, AgentConfig,
    CandidatePoolConfig, InitialBreadthConfig, PlannerConfig,
    ChunkMixConfig, ChunkMixDifficulty, ModeAdjustment,
    GenerationYield, ChunkLimitsForMode, ChunkKLimit, RuntimeConfig,
)
from benchforge.config.config import (
    RetrievalConfig, ChunkingConfig, SummarizationChunkingConfig,
    load_dotenv, expand_env_recursive,
)


def load_qa_agent_config(
    path: str | Path,
) -> tuple[Blueprint, AgentConfig, dict, RetrievalConfig, ChunkingConfig, SummarizationChunkingConfig]:
    """Returns (blueprint, agent_config, model_cfg, retrieval_cfg, chunking_cfg, summarization_chunking_cfg)."""
    from benchforge.utils.paths import get_project_root
    path = Path(path)
    load_dotenv(get_project_root() / ".env")

    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f))

    run = raw["run"]
    bp_raw = raw["blueprint"]

    # run_id 的 "auto"/空值由 RunContext 统一处理
    blueprint = Blueprint(
        task_id=run["task_id"],
        run_id=run.get("run_id", ""),
        language=run["language"],
        topics=bp_raw["topics"],
        modes={
            mode: ModeCfg(
                count=cfg["count"],
                max_rounds=cfg["max_rounds"],
                difficulty_distribution=cfg["difficulty_distribution"],
            )
            for mode, cfg in bp_raw["modes"].items()
        },
    )

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

    model_cfg = {k: str(v) for k, v in raw.get("model", {}).items()}

    retrieval_cfg = RetrievalConfig(**raw["retrieval"]) if "retrieval" in raw else RetrievalConfig()
    chunking_cfg = ChunkingConfig(**raw["chunking"]) if "chunking" in raw else ChunkingConfig()
    sum_chunking_cfg = SummarizationChunkingConfig(**raw["summarization_chunking"]) if "summarization_chunking" in raw else SummarizationChunkingConfig()

    return blueprint, agent_config, model_cfg, retrieval_cfg, chunking_cfg, sum_chunking_cfg
