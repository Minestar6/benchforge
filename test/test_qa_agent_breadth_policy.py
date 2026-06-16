import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.qa_agent.planner import build_mode_round_plan
from benchforge.agents.qa_agent.schema import (
    AgentConfig,
    Blueprint,
    CandidatePoolConfig,
    ChunkKLimit,
    ChunkLimitsForMode,
    ChunkMixConfig,
    ChunkMixDifficulty,
    GenerationYield,
    InitialBreadthConfig,
    ModeAdjustment,
    ModeCfg,
    PlannerConfig,
    RuntimeConfig,
)
from benchforge.agents.qa_agent.state import ModeState


def _blueprint() -> Blueprint:
    return Blueprint(
        task_id="task",
        run_id="run",
        language="en",
        topics=["Topic A", "Topic B", "Topic C"],
        modes={
            "qa": ModeCfg(
                count=10,
                max_rounds=5,
                difficulty_distribution={"easy": 0.2, "medium": 0.3, "hard": 0.5},
            ),
        },
    )


def _config() -> AgentConfig:
    return AgentConfig(
        candidate_pool=CandidatePoolConfig(min_candidate_multiplier=1.5, max_candidate_multiplier=2.0),
        initial_breadth=InitialBreadthConfig(
            enabled=True,
            max_topics_per_round=3,
            difficulty="medium",
            difficulty_policy="hard_aware",
            hard_ratio_threshold=0.3,
        ),
        planner=PlannerConfig(topics_per_round=2),
        chunk_mix=ChunkMixConfig(
            by_difficulty={
                "easy": ChunkMixDifficulty(single_ratio=0.8, multi_ratio=0.2),
                "medium": ChunkMixDifficulty(single_ratio=0.5, multi_ratio=0.5),
                "hard": ChunkMixDifficulty(single_ratio=0.2, multi_ratio=0.8),
            },
            mode_adjustment={"qa": ModeAdjustment(single_delta=0.1)},
        ),
        generation_yield={
            "qa": GenerationYield(single_chunk_avg_questions=3.0, multi_chunk_avg_questions=4.0),
        },
        chunk_limits={
            "qa": ChunkLimitsForMode(
                single_k=ChunkKLimit(min=1, max=4),
                multi_k=ChunkKLimit(min=0, max=4),
            ),
        },
        runtime=RuntimeConfig(),
    )


def test_initial_breadth_uses_hard_when_hard_target_is_high():
    blueprint = _blueprint()
    config = _config()
    mode_state = ModeState(mode="qa")

    plan = build_mode_round_plan("qa", blueprint.modes["qa"], blueprint, config, mode_state)

    assert plan.strategy == "initial_breadth"
    assert plan.difficulty == "hard"
