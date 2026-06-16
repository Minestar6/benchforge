import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.qa_agent.executor import update_generation_yield_estimates
from benchforge.agents.qa_agent.planner import compute_dynamic_chunk_k
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
        task_id="test",
        run_id="run",
        language="en",
        topics=["Topic A", "Topic B"],
        modes={
            "qa": ModeCfg(
                count=10,
                max_rounds=5,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            )
        },
    )


def _config() -> AgentConfig:
    return AgentConfig(
        candidate_pool=CandidatePoolConfig(min_candidate_multiplier=1.5, max_candidate_multiplier=2.0),
        initial_breadth=InitialBreadthConfig(enabled=True, max_topics_per_round=4, difficulty="medium"),
        planner=PlannerConfig(topics_per_round=2),
        chunk_mix=ChunkMixConfig(
            by_difficulty={
                "easy": ChunkMixDifficulty(single_ratio=0.7, multi_ratio=0.3),
                "medium": ChunkMixDifficulty(single_ratio=0.5, multi_ratio=0.5),
                "hard": ChunkMixDifficulty(single_ratio=0.1, multi_ratio=0.9),
            },
            mode_adjustment={"qa": ModeAdjustment(single_delta=0.1)},
        ),
        generation_yield={"qa": GenerationYield(single_chunk_avg_questions=3.0, multi_chunk_avg_questions=4.0)},
        chunk_limits={
            "qa": ChunkLimitsForMode(
                single_k=ChunkKLimit(min=1, max=6),
                multi_k=ChunkKLimit(min=0, max=6),
            )
        },
        runtime=RuntimeConfig(),
    )


def test_update_generation_yield_estimates_tracks_observed_output():
    mode_state = ModeState(mode="qa")
    round_results = [
        {
            "success": True,
            "single_unit_count": 2,
            "multi_unit_count": 1,
            "single_generated_count": 6,
            "multi_generated_count": 5,
        }
    ]

    update_generation_yield_estimates(mode_state, round_results)

    assert mode_state.single_yield_estimate == 3.0
    assert mode_state.multi_yield_estimate == 5.0
    assert mode_state.single_yield_samples == 2
    assert mode_state.multi_yield_samples == 1


def test_compute_dynamic_chunk_k_uses_dynamic_yield_estimates():
    blueprint = _blueprint()
    config = _config()

    baseline_state = ModeState(mode="qa")
    baseline_single_k, baseline_multi_k, _ = compute_dynamic_chunk_k(
        mode="qa",
        mode_cfg=blueprint.modes["qa"],
        difficulty="medium",
        selected_topic_count=2,
        mode_state=baseline_state,
        blueprint=blueprint,
        config=config,
    )

    corrected_state = ModeState(
        mode="qa",
        single_yield_estimate=6.0,
        multi_yield_estimate=8.0,
        single_yield_samples=4,
        multi_yield_samples=4,
    )
    corrected_single_k, corrected_multi_k, _ = compute_dynamic_chunk_k(
        mode="qa",
        mode_cfg=blueprint.modes["qa"],
        difficulty="medium",
        selected_topic_count=2,
        mode_state=corrected_state,
        blueprint=blueprint,
        config=config,
    )

    assert corrected_single_k <= baseline_single_k
    assert corrected_multi_k <= baseline_multi_k
