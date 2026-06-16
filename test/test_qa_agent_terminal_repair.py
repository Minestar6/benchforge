import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.qa_agent.agent import (
    _build_terminal_hard_repair_plan,
    run_mode_generation,
)
from benchforge.agents.qa_agent.storage import save_generation_report, save_mode_metrics
from benchforge.agents.qa_agent.planner import ModeRoundPlan, RoundStrategy
from benchforge.agents.qa_agent.schema import (
    AgentConfig,
    CandidatePoolConfig,
    ChunkKLimit,
    ChunkLimitsForMode,
    ChunkMixConfig,
    ChunkMixDifficulty,
    ExperimentConfig,
    GenerationYield,
    InitialBreadthConfig,
    ModeAdjustment,
    ModeCfg,
    PlannerConfig,
    RuntimeConfig,
)
from benchforge.agents.qa_agent.state import CandidateRecord, CandidateStatus, GlobalState, ModeState


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
            mode_adjustment={
                "qa": ModeAdjustment(single_delta=0.1),
                "multiple_choice": ModeAdjustment(single_delta=-0.1),
            },
        ),
        generation_yield={
            "qa": GenerationYield(single_chunk_avg_questions=3.0, multi_chunk_avg_questions=4.0),
        },
        chunk_limits={
            "qa": ChunkLimitsForMode(
                single_k=ChunkKLimit(min=1, max=4),
                multi_k=ChunkKLimit(min=0, max=3),
            ),
        },
        runtime=RuntimeConfig(),
        experiment=ExperimentConfig(
            enable_terminal_hard_repair=True,
            terminal_hard_repair_min_missing=2,
            terminal_hard_repair_gap_threshold=0.15,
            terminal_hard_repair_topics=2,
            terminal_hard_repair_multi_k_scale=1.0,
        ),
    )


def _mode_cfg() -> ModeCfg:
    return ModeCfg(
        count=10,
        max_rounds=3,
        difficulty_distribution={"easy": 0.3, "medium": 0.3, "hard": 0.4},
    )


def _record(question_id: str, topic: str, difficulty: str) -> CandidateRecord:
    return CandidateRecord(
        question_id=question_id,
        question=f"Question {question_id}",
        answer="Answer",
        topic=topic,
        difficulty=difficulty,
        status=CandidateStatus.ACCEPTED,
        source_round=1,
        source_strategy="normal_generate",
        chunk_ids=["doc::chunk_0"],
    )


def test_build_terminal_hard_repair_plan_uses_hard_multi_chunk_only():
    blueprint = SimpleNamespace(topics=["Topic A", "Topic B", "Topic C"])
    mode_cfg = _mode_cfg()
    config = _config()
    mode_state = ModeState(mode="qa")
    mode_state.round_in_mode = 3
    mode_state.candidate_questions = [
        _record("q1", "Topic A", "easy"),
        _record("q2", "Topic A", "medium"),
        _record("q3", "Topic B", "hard"),
        _record("q4", "Topic C", "easy"),
    ]

    plan = _build_terminal_hard_repair_plan("qa", mode_cfg, blueprint, config, mode_state)

    assert plan is not None
    assert plan.strategy == RoundStrategy.TERMINAL_HARD_REPAIR
    assert plan.difficulty == "hard"
    assert plan.single_k == 0
    assert plan.multi_k >= 1
    assert len(plan.topics) <= 2


@pytest.mark.asyncio
async def test_run_mode_generation_executes_terminal_hard_repair(monkeypatch):
    blueprint = SimpleNamespace(task_id="task", run_id="run", language="en", topics=["Topic A"], modes={})
    mode_cfg = ModeCfg(count=10, max_rounds=3, difficulty_distribution={"easy": 0.3, "medium": 0.3, "hard": 0.4})
    config = _config()
    global_state = GlobalState()
    mode_state = ModeState(mode="qa")
    mode_state.candidate_questions = [
        _record("q1", "Topic A", "easy"),
        _record("q2", "Topic A", "medium"),
        _record("q3", "Topic A", "medium"),
        _record("q4", "Topic A", "easy"),
    ]

    stop_calls = {"count": 0}
    executed_plans: list[ModeRoundPlan] = []

    def fake_mode_should_stop(**kwargs):
        stop_calls["count"] += 1
        return True, "max_candidate_pool_reached"

    async def fake_execute_mode_round_plan(**kwargs):
        round_plan = kwargs["round_plan"]
        executed_plans.append(round_plan)
        return [], SimpleNamespace(
            empty_round=True,
            generated_count=0,
            accepted_count=0,
            strategy=round_plan.strategy.value,
        )

    monkeypatch.setattr("benchforge.agents.qa_agent.agent.mode_should_stop", fake_mode_should_stop)
    monkeypatch.setattr("benchforge.agents.qa_agent.agent.execute_mode_round_plan", fake_execute_mode_round_plan)
    monkeypatch.setattr("benchforge.agents.qa_agent.agent.append_round_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr("benchforge.agents.qa_agent.agent.append_round_feedback", lambda *args, **kwargs: None)
    monkeypatch.setattr("benchforge.agents.qa_agent.agent.update_mode_trace", lambda *args, **kwargs: None)

    await run_mode_generation(
        mode="qa",
        mode_cfg=mode_cfg,
        blueprint=blueprint,
        config=config,
        global_state=global_state,
        mode_state=mode_state,
        evidence_manager=object(),
        generator=object(),
        tracer=None,
    )

    assert stop_calls["count"] >= 2
    assert len(executed_plans) == 1
    assert executed_plans[0].strategy == RoundStrategy.TERMINAL_HARD_REPAIR
    assert executed_plans[0].difficulty == "hard"
    assert executed_plans[0].single_k == 0
    assert mode_state.terminal_hard_repair_executed is True
    assert mode_state.terminal_hard_repair_rounds == 1


def test_terminal_hard_repair_is_written_to_metrics_and_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = _config()
    mode_cfg = _mode_cfg()
    mode_state = ModeState(mode="qa")
    mode_state.candidate_questions = [
        _record("q1", "Topic A", "hard"),
        _record("q2", "Topic A", "medium"),
    ]
    mode_state.stopped_reason = "max_candidate_pool_reached"
    mode_state.terminal_hard_repair_executed = True
    mode_state.terminal_hard_repair_rounds = 1
    mode_state.terminal_hard_repair_generated_count = 2
    mode_state.terminal_hard_repair_accepted_count = 1

    save_mode_metrics("task", "run", "qa", mode_state, mode_cfg=mode_cfg, config=config)
    report = save_generation_report(
        task_id="task",
        run_id="run",
        global_state=GlobalState(),
        mode_states={"qa": mode_state},
        mode_cfgs={"qa": mode_cfg},
        config=config,
    )

    metrics = __import__("json").loads((tmp_path / "runs" / "task" / "run" / "qa" / "mode_metrics.json").read_text())
    assert metrics["terminal_hard_repair"]["executed"] is True
    assert metrics["terminal_hard_repair"]["rounds"] == 1
    assert metrics["terminal_hard_repair"]["accepted_count"] == 1

    assert report["modes"]["qa"]["terminal_hard_repair"]["executed"] is True
    assert report["modes"]["qa"]["terminal_hard_repair"]["generated_count"] == 2
