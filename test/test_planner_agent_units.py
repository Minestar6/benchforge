"""planner_agent 单元测试。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.planner_agent.feedback import (
    build_evaluator_feedback,
    build_generator_feedback,
    build_validator_feedback,
)
from benchforge.agents.planner_agent.orchestrator import _load_verify_model_client
from benchforge.agents.planner_agent.planner import (
    _build_next_round_spec,
    _propose_topics_with_llm,
    _update_planner_state,
)
from benchforge.agents.planner_agent.schema import (
    GlobalBlueprint,
    FinalTargets,
    QuestionModeDefaults,
    EvaluatorDefaults,
    EvaluationRequirements,
    StopConditions,
    RoundSpec,
    PlannerState,
    TopicBacklog,
    ModelPool,
    GeneratorFeedback,
    GeneratorFeedbackArtifacts,
    GeneratorFeedbackSummary,
    ModeGeneratorFeedback,
    ValidatorFeedback,
    ValidatorFeedbackSummary,
    ValidatorQualitySignals,
)
from benchforge.agents.verify_agent.config_loader import VerifyAgentConfig, LLMValidationCfg
from benchforge.schemas import SharedState, AgentStatus
from benchforge.utils.shared_state import save_shared_state


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _make_round_spec() -> RoundSpec:
    return RoundSpec(
        task_id="task_x",
        blueprint_id="bp_x",
        round_id=3,
        run_id="run_y",
        objective="test",
        blueprint={
            "task_id": "task_x",
            "run_id": "run_y",
            "language": "zh",
            "topics": ["AI"],
            "modes": {
                "qa": {
                    "count": 2,
                    "max_rounds": 3,
                    "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
                },
                "multiple_choice": {
                    "count": 1,
                    "max_rounds": 3,
                    "difficulty_distribution": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
                }
            },
        },
    )


def _make_shared_state(run_dir: Path) -> Path:
    state = SharedState(
        task_id="task_x",
        run_id="run_y",
        blueprint={"topics": ["AI"], "modes": {"qa": {"count": 2, "difficulty_distribution": {"easy": 1.0}}}},
        artifacts={
            "generation_report": str(Path("runs") / "task_x" / "run_y" / "generation_report.json"),
            "qa_mode_state": str(Path("runs") / "task_x" / "run_y" / "qa" / "mode_state.json"),
            "validated_questions": str(Path("runs") / "task_x" / "run_y" / "validation" / "validated_questions.jsonl"),
            "validation_report": str(Path("runs") / "task_x" / "run_y" / "validation" / "validation_report.json"),
            "weighted_selection": str(Path("runs") / "task_x" / "run_y" / "validation" / "weighted_selection.json"),
            "evaluation_report": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "evaluation_report.json"),
            "dataset_quality_summary": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "dataset_report" / "dataset_quality_summary.json"),
            "model_overall_report": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "model_report" / "model_overall_report.csv"),
            "model_aggregate_report": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "model_report" / "model_aggregate_report.json"),
            "model_by_topic": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "model_report" / "model_by_topic.csv"),
            "model_by_difficulty": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "model_report" / "model_by_difficulty.csv"),
            "model_by_question_mode": str(Path("runs") / "task_x" / "run_y" / "evaluation" / "model_report" / "model_by_question_mode.csv"),
            "llm_calls": str(Path("runs") / "task_x" / "run_y" / "llm_calls.jsonl"),
        },
        agent_status={
            "generation": AgentStatus.COMPLETED,
            "verification": AgentStatus.COMPLETED,
            "evaluation": AgentStatus.COMPLETED,
        },
    )
    return save_shared_state(state, run_dir)


def test_build_generator_feedback_aggregates_trace_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "task_x" / "run_y"
    shared_state_path = _make_shared_state(run_dir)

    _write_json(run_dir / "generation_report.json", {
        "total_candidates": 4,
        "global_used_chunk_combinations": 2,
        "global_failures": 0,
        "modes": {"qa": {"candidate_count": 4, "target_candidate_count": 5, "stopped_reason": None}},
    })
    _write_json(run_dir / "qa" / "mode_state.json", {
        "difficulty_counts": {"easy": 1},
        "topic_counts": {"AI": 4},
    })
    _write_jsonl(run_dir / "llm_calls.jsonl", [
        {"agent": "qa_agent", "response": {"input_tokens": 111, "output_tokens": 22}},
        {"agent": "model_eval_agent", "response": {"input_tokens": 999, "output_tokens": 999}},
    ])

    fb = build_generator_feedback(shared_state_path, _make_round_spec())
    assert fb.round_id == 3
    assert fb.summary.llm_input_tokens == 111
    assert fb.summary.llm_output_tokens == 22
    assert fb.by_mode["qa"].topic_counts["AI"] == 4


def test_build_validator_feedback_uses_trace_tokens_and_round_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "task_x" / "run_y"
    shared_state_path = _make_shared_state(run_dir)

    _write_json(run_dir / "validation" / "validation_report.json", {
        "total_candidates": 4,
        "citation_passed": 3,
        "llm_passed": 2,
        "final_selected": 1,
        "failed_by_stage": {},
        "llm_usage": {"calls": 99, "input_tokens": 999, "output_tokens": 999},
    })
    _write_jsonl(run_dir / "validation" / "validated_questions.jsonl", [
        {
            "final_status": "selected",
            "candidate": {"question_mode": "qa", "estimated_difficulty": "easy", "topic": "AI"},
            "citation_validation": {"citation_score": 0.8},
            "llm_validation": {"overall_score": 0.9},
        }
    ])
    _write_json(run_dir / "validation" / "weighted_selection.json", {
        "dropped_as_duplicate": [],
        "dropped_as_overquota": [],
    })
    _write_jsonl(run_dir / "llm_calls.jsonl", [
        {"agent": "verify_agent", "response": {"input_tokens": 33, "output_tokens": 7}},
    ])

    fb = build_validator_feedback(shared_state_path, round_id=3)
    assert fb is not None
    assert fb.round_id == 3
    assert fb.summary.llm_input_tokens == 33
    assert fb.summary.llm_output_tokens == 7


def test_build_evaluator_feedback_uses_artifact_paths_and_trace_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "task_x" / "run_y"
    shared_state_path = _make_shared_state(run_dir)

    _write_json(run_dir / "evaluation" / "evaluation_report.json", {"num_questions": 2, "num_models": 1})
    _write_json(run_dir / "evaluation" / "dataset_report" / "dataset_quality_summary.json", {
        "citation_score": {"mean": 0.8},
        "citation_pass_rate": {"mean": 1.0},
        "diversity_score": 0.6,
    })
    _write_jsonl(run_dir / "llm_calls.jsonl", [
        {"agent": "model_eval_agent", "response": {"input_tokens": 44, "output_tokens": 11}},
    ])
    for rel in [
        "evaluation/model_report/model_overall_report.csv",
        "evaluation/model_report/model_aggregate_report.json",
        "evaluation/model_report/model_by_topic.csv",
        "evaluation/model_report/model_by_difficulty.csv",
        "evaluation/model_report/model_by_question_mode.csv",
    ]:
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    fb = build_evaluator_feedback(shared_state_path, round_id=3)
    assert fb is not None
    assert fb.round_id == 3
    assert fb.summary.llm_input_tokens == 44
    assert fb.summary.llm_output_tokens == 11
    assert fb.artifacts["dataset_quality_summary"].endswith("dataset_quality_summary.json")


def test_load_verify_model_client_uses_verify_config_model(monkeypatch):
    loaded = {}

    class DummyLoader:
        @staticmethod
        def load_model(cfg):
            loaded["model_name"] = cfg.model_name
            return "verify-client"

    monkeypatch.setattr("benchforge.agents.planner_agent.orchestrator.ModelLoader", DummyLoader)

    registry = {
        "verify-model": SimpleNamespace(model_name="verify-model"),
    }
    verify_config = VerifyAgentConfig(llm_validation=LLMValidationCfg(enabled=True, model="verify-model"))

    client = _load_verify_model_client(registry, verify_config)
    assert client == "verify-client"
    assert loaded["model_name"] == "verify-model"


def _make_global_blueprint() -> GlobalBlueprint:
    return GlobalBlueprint(
        task_id="task_x",
        blueprint_id="bp_x",
        user_goal="build benchmark",
        language="zh",
        seed_topics=["AI", "ML", "Systems"],
        final_targets=FinalTargets(qa=10, multiple_choice=8),
        default_modes={
            "qa": QuestionModeDefaults(
                max_rounds=3,
                difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
            ),
            "multiple_choice": QuestionModeDefaults(
                max_rounds=3,
                difficulty_distribution={"easy": 0.4, "medium": 0.4, "hard": 0.2},
            ),
        },
        evaluator_defaults=EvaluatorDefaults(candidate_model_names=["model-a"], judge_model_name="judge"),
        evaluation_requirements=EvaluationRequirements(),
        stop_conditions=StopConditions(max_rounds=4, min_selected_per_round=3, max_total_tokens=10000),
    )


def test_build_next_round_spec_clamps_targets_to_remaining_demand():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        completed_targets={"qa": 9, "multiple_choice": 7},
        topic_backlog=TopicBacklog(active=["AI", "ML"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )

    round_spec = _build_next_round_spec(
        state=state,
        blueprint=blueprint,
        proposed_topics=["AI"],
        run_id="run_002",
    )

    assert round_spec.blueprint["modes"]["qa"]["count"] == 1
    assert round_spec.blueprint["modes"]["multiple_choice"]["count"] == 1


@pytest.mark.asyncio
async def test_propose_topics_rehydrates_deferred_backlog():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=1,
        topic_backlog=TopicBacklog(active=[], deferred=["AI", "ML"]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )

    topics = await _propose_topics_with_llm(
        state=state,
        blueprint=blueprint,
        model_client=SimpleNamespace(),
    )

    assert topics
    assert set(topics).issubset({"AI", "ML"})
    assert state.topic_backlog.active


def test_update_planner_state_consumes_active_topics_into_deferred():
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=1,
        topic_backlog=TopicBacklog(active=["AI", "ML", "Systems"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    round_spec = _make_round_spec()
    round_spec.blueprint["topics"] = ["AI", "ML"]

    gen_fb = GeneratorFeedback(
        task_id="task_x",
        run_id="run_y",
        round_id=3,
        status="success",
        artifacts=GeneratorFeedbackArtifacts(
            shared_state_path="shared_state.json",
            generation_report="generation_report.json",
        ),
        summary=GeneratorFeedbackSummary(
            total_candidates=4,
            global_used_chunk_combinations=2,
            global_failures=0,
            llm_input_tokens=10,
            llm_output_tokens=5,
        ),
        by_mode={
            "qa": ModeGeneratorFeedback(
                candidate_count=4,
                target_candidate_count=6,
                fulfillment_rate=0.66,
            )
        },
    )
    val_fb = ValidatorFeedback(
        task_id="task_x",
        run_id="run_y",
        round_id=3,
        status="success",
        summary=ValidatorFeedbackSummary(
            total_candidates=4,
            citation_passed=3,
            llm_passed=2,
            final_selected=1,
            citation_pass_rate=0.75,
            llm_pass_rate_after_citation=0.66,
            final_selection_rate=0.25,
            llm_calls=2,
            llm_input_tokens=8,
            llm_output_tokens=4,
        ),
        quality_signals=ValidatorQualitySignals(),
        by_mode={"qa": {"selected": 1}},
    )

    _update_planner_state(state, round_spec, gen_fb, val_fb, eval_fb=None)

    assert state.topic_backlog.active == ["Systems"]
    assert state.topic_backlog.deferred == ["AI", "ML"]
