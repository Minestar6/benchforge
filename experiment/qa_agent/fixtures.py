"""共用 fixtures：blueprint、config 构造。

EvidenceManager 和 Generator 已由 run_generation_agent 内部自动创建，
fixtures 中不再需要 make_evidence_manager()。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.qa_agent.schema import (
    AgentConfig, CandidatePoolConfig, InitialBreadthConfig, PlannerConfig,
    ChunkMixConfig, ChunkMixDifficulty, ModeAdjustment,
    GenerationYield, ChunkLimitsForMode, ChunkKLimit, RuntimeConfig, DecisionConfig,
    Blueprint, ModeCfg,
)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "qa_agent.yaml"


def make_blueprint(
    count: int = 5,
    max_rounds: int = 5,
    run_id: str = "exp_run",
    task_id: str = "exp_task",
    topics: list[str] | None = None,
    language: str = "en",
    difficulty_distribution: dict[str, float] | None = None,
) -> Blueprint:
    if topics is None:
        topics = ["Climate Change", "Artificial Intelligence"]
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    mode_cfg = ModeCfg(count=count, max_rounds=max_rounds, difficulty_distribution=difficulty_distribution)
    return Blueprint(task_id=task_id, run_id=run_id, language=language, topics=topics,
                     modes={"qa": mode_cfg, "mcq": mode_cfg})


def make_config(hard_gap_threshold: float = 0.2, too_easy_ratio: float = 0.4) -> AgentConfig:
    return AgentConfig(
        candidate_pool=CandidatePoolConfig(
            min_candidate_multiplier=1.5,
            max_candidate_multiplier=2.0,
        ),
        initial_breadth=InitialBreadthConfig(enabled=True, max_topics_per_round=10, difficulty="medium"),
        planner=PlannerConfig(topics_per_round=3),
        chunk_mix=ChunkMixConfig(
            by_difficulty={
                "easy": ChunkMixDifficulty(single_ratio=0.7, multi_ratio=0.3),
                "medium": ChunkMixDifficulty(single_ratio=0.5, multi_ratio=0.5),
                "hard": ChunkMixDifficulty(single_ratio=0.1, multi_ratio=0.9),
            },
            mode_adjustment={
                "qa": ModeAdjustment(single_delta=0.1),
                "mcq": ModeAdjustment(single_delta=-0.1),
            },
        ),
        generation_yield={
            "qa": GenerationYield(single_chunk_avg_questions=2.0, multi_chunk_avg_questions=3.0),
            "mcq": GenerationYield(single_chunk_avg_questions=1.5, multi_chunk_avg_questions=2.0),
        },
        chunk_limits={
            "qa": ChunkLimitsForMode(single_k=ChunkKLimit(min=1, max=4), multi_k=ChunkKLimit(min=0, max=3)),
            "mcq": ChunkLimitsForMode(single_k=ChunkKLimit(min=0, max=3), multi_k=ChunkKLimit(min=1, max=4)),
        },
        runtime=RuntimeConfig(max_consecutive_empty_rounds_per_mode=3, max_failures_per_mode=8),
        decision=DecisionConfig(hard_gap_threshold=hard_gap_threshold, too_easy_ratio=too_easy_ratio),
    )
