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
    diagnose_last_round,
    adjust_next_round_plan,
    question_difficulty_evolver,
    control_parameter_tuner,
    round_spec_builder,
    topic_adaptive_retriever,
    _build_next_round_spec,
    _propose_topics_with_llm,
    _update_planner_state,
)
from benchforge.agents.planner_agent.topic_search import summarize_topic_feedback
from benchforge.agents.planner_agent.utils import translate_eval_profile
from benchforge.agents.planner_agent.schema import (
    GlobalBlueprint,
    FinalTargets,
    QuestionModeDefaults,
    EvaluatorDefaults,
    EvaluationRequirements,
    JudgeMetricDef,
    StopConditions,
    RoundSpec,
    PlannerState,
    RunHistoryEntry,
    QuestionPlan,
    TopicBacklog,
    ModelPool,
    GeneratorFeedback,
    GeneratorFeedbackArtifacts,
    GeneratorFeedbackSummary,
    ModeGeneratorFeedback,
    EvaluatorFeedback,
    EvaluatorFeedbackSummary,
    DatasetSignals,
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

    _write_json(run_dir / "evaluation" / "evaluation_report.json", {
        "num_questions": 2,
        "num_models": 2,
        "discriminative_signals": {
            "overall_model_gap": 0.4,
            "best_vs_second_gap": 0.1,
            "top_vs_bottom_gap": 0.5,
            "per_question_variance_mean": 0.06,
            "discriminative_question_ratio": 0.5,
            "easy_questions_too_easy_ratio": 0.5,
            "all_models_fail_ratio": 0.0,
        },
        "by_topic": {
            "AI": {
                "question_count": 2,
                "model_gap": 0.4,
                "too_easy_ratio": 0.5,
                "all_models_fail_ratio": 0.0,
            }
        },
        "by_difficulty": {
            "easy": {"question_count": 1, "model_gap": 0.0},
            "hard": {"question_count": 1, "model_gap": 0.8},
        },
        "by_mode": {
            "qa": {"question_count": 2, "model_gap": 0.4},
        },
    })
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
    assert fb.derived_performance_signals["overall_model_gap"] == 0.4
    assert fb.derived_performance_signals["discriminative_question_ratio"] == 0.5
    assert fb.derived_performance_signals["easy_questions_too_easy_ratio"] == 0.5
    assert fb.derived_performance_signals["all_models_fail_ratio"] == 0.0
    assert fb.derived_performance_signals["by_topic"]["AI"]["model_gap"] == 0.4
    assert fb.derived_performance_signals["by_difficulty"]["hard"]["model_gap"] == 0.8


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
        control_plan=adjust_next_round_plan(
            state,
            blueprint,
            SimpleNamespace(label="cold_start", evidence={}, problems=[]),
        ),
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
        topics_budget=2,
    )

    assert topics
    assert set(topics).issubset({"AI", "ML"})
    assert state.topic_backlog.active


@pytest.mark.asyncio
async def test_topic_adaptive_retriever_uses_prompt_json_list():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI", "ML", "Systems"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )

    class TopicClient:
        model_name = "planner"

        async def complete(self, model: str, messages: list[dict[str, str]], **kwargs):
            return {
                "text": '["ML", "AI"]',
                "input_tokens": 1,
                "output_tokens": 1,
                "latency": 0.0,
                "raw": {},
            }

    topics = await topic_adaptive_retriever(
        state=state,
        blueprint=blueprint,
        model_client=TopicClient(),
        topic_budget=2,
    )

    assert topics == ["ML", "AI"]


@pytest.mark.asyncio
async def test_topic_adaptive_retriever_falls_back_when_prompt_invalid():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI", "ML"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )

    class TopicClient:
        model_name = "planner"

        async def complete(self, model: str, messages: list[dict[str, str]], **kwargs):
            return {
                "text": "not json",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency": 0.0,
                "raw": {},
            }

    topics = await topic_adaptive_retriever(
        state=state,
        blueprint=blueprint,
        model_client=TopicClient(),
        topic_budget=1,
    )

    assert topics
    assert set(topics).issubset({"AI", "ML"})


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

    question_plan = QuestionPlan(
        qa_count=2,
        mc_count=1,
        qa_difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        mc_difficulty_distribution={"easy": 0.4, "medium": 0.4, "hard": 0.2},
        min_candidate_multiplier=1.5,
        max_candidate_multiplier=2.0,
        topic_budget=2,
        min_selected_per_round=3,
    )

    _update_planner_state(state, round_spec, question_plan, gen_fb, val_fb, eval_fb=None)

    assert state.topic_backlog.active == ["Systems"]
    assert state.topic_backlog.deferred == ["AI", "ML"]
    assert state.last_question_plan is not None
    assert state.last_question_plan.topic_budget == 2
    assert state.run_history[-1].question_plan is not None


def test_diagnose_last_round_identifies_high_quality_low_separation():
    gen_fb = GeneratorFeedback(
        task_id="task_x",
        run_id="run_001",
        round_id=1,
        status="success",
        artifacts=GeneratorFeedbackArtifacts(
            shared_state_path="shared_state.json",
            generation_report="generation_report.json",
        ),
        summary=GeneratorFeedbackSummary(
            total_candidates=20,
            global_used_chunk_combinations=5,
            global_failures=0,
            llm_input_tokens=100,
            llm_output_tokens=50,
        ),
    )
    val_fb = ValidatorFeedback(
        task_id="task_x",
        run_id="run_001",
        round_id=1,
        status="success",
        summary=ValidatorFeedbackSummary(
            total_candidates=20,
            citation_passed=18,
            llm_passed=16,
            final_selected=12,
            citation_pass_rate=0.9,
            llm_pass_rate_after_citation=0.88,
            final_selection_rate=0.6,
            llm_calls=10,
            llm_input_tokens=50,
            llm_output_tokens=20,
        ),
        quality_signals=ValidatorQualitySignals(
            duplicate_rate=0.05,
            overquota_rate=0.05,
        ),
        by_mode={"qa": {"selected": 8}, "multiple_choice": {"selected": 4}},
    )
    eval_fb = EvaluatorFeedback(
        task_id="task_x",
        run_id="run_001",
        round_id=1,
        status="success",
        summary=EvaluatorFeedbackSummary(num_questions=12, num_models=3),
        dataset_signals=DatasetSignals(),
        derived_performance_signals={
            "overall_model_gap": 0.08,
            "discriminative_question_ratio": 0.2,
            "easy_questions_too_easy_ratio": 0.45,
            "all_models_fail_ratio": 0.05,
            "by_topic": {"AI": {"model_gap": 0.05}, "ML": {"model_gap": 0.12}},
        },
    )

    diagnosis = diagnose_last_round(gen_fb, val_fb, eval_fb)

    assert diagnosis.label == "high_quality_low_separation"
    assert "overall_model_gap" in diagnosis.evidence


def test_adjust_next_round_plan_increases_hard_ratio_for_low_separation():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=1,
        completed_targets={"qa": 0, "multiple_choice": 0},
        topic_backlog=TopicBacklog(active=["AI", "ML", "Systems"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    diagnosis = SimpleNamespace(
        label="high_quality_low_separation",
        evidence={
            "overall_model_gap": 0.08,
            "easy_questions_too_easy_ratio": 0.45,
            "strong_topics": ["ML"],
            "weak_topics": ["AI"],
        },
        problems=["low separation"],
    )

    plan = adjust_next_round_plan(state, blueprint, diagnosis)

    assert plan.qa_difficulty_distribution["hard"] > blueprint.default_modes["qa"].difficulty_distribution["hard"]
    assert plan.qa_difficulty_distribution["easy"] < blueprint.default_modes["qa"].difficulty_distribution["easy"]
    assert plan.eval_profile == "full"


def test_question_difficulty_evolver_only_returns_question_plan_fields():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=1,
        completed_targets={"qa": 0, "multiple_choice": 0},
        topic_backlog=TopicBacklog(active=["AI", "ML"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )

    plan = question_difficulty_evolver(
        state,
        blueprint,
        SimpleNamespace(label="high_quality_low_separation", evidence={}, problems=[]),
    )

    assert plan.topic_budget >= 1
    assert plan.max_candidate_multiplier >= plan.min_candidate_multiplier
    assert plan.qa_difficulty_distribution["hard"] > blueprint.default_modes["qa"].difficulty_distribution["hard"]
    assert not hasattr(plan, "selection_mode")
    assert not hasattr(plan, "eval_profile")


def test_question_difficulty_evolver_accepts_previous_question_plan_schema():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        completed_targets={"qa": 0, "multiple_choice": 0},
        topic_backlog=TopicBacklog(active=["AI"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    previous_plan = QuestionPlan(
        qa_count=5,
        mc_count=4,
        qa_difficulty_distribution={"easy": 0.1, "medium": 0.6, "hard": 0.3},
        mc_difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
        min_candidate_multiplier=1.7,
        max_candidate_multiplier=2.3,
        topic_budget=3,
        min_selected_per_round=4,
    )

    next_plan = question_difficulty_evolver(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
        previous_question_plan=previous_plan,
    )

    assert isinstance(next_plan, QuestionPlan)
    assert next_plan.topic_budget == 3
    assert next_plan.min_candidate_multiplier >= 1.7
    assert next_plan.max_candidate_multiplier >= next_plan.min_candidate_multiplier


def test_question_difficulty_evolver_uses_generation_and_validation_difficulty_feedback():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        completed_targets={"qa": 0, "multiple_choice": 0},
        topic_backlog=TopicBacklog(active=["AI", "ML"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    gen_fb = GeneratorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        artifacts=GeneratorFeedbackArtifacts(shared_state_path="", generation_report=""),
        summary=GeneratorFeedbackSummary(total_candidates=12, global_used_chunk_combinations=3, global_failures=0),
        by_mode={
            "qa": ModeGeneratorFeedback(
                candidate_count=12,
                target_candidate_count=12,
                fulfillment_rate=1.0,
                difficulty_counts={"easy": 2, "medium": 2, "hard": 8},
            )
        },
    )
    val_fb = ValidatorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        summary=ValidatorFeedbackSummary(
            total_candidates=12,
            citation_passed=6,
            llm_passed=5,
            final_selected=4,
            citation_pass_rate=0.5,
            llm_pass_rate_after_citation=0.83,
            final_selection_rate=0.33,
            llm_calls=1,
            llm_input_tokens=1,
            llm_output_tokens=1,
        ),
        quality_signals=ValidatorQualitySignals(),
        by_mode={"qa": {"selected": 4}},
        by_difficulty={
            "easy": {"selected": 1, "reserve": 0, "rejected": 1},
            "medium": {"selected": 3, "reserve": 0, "rejected": 1},
            "hard": {"selected": 0, "reserve": 0, "rejected": 6},
        },
    )

    plan = question_difficulty_evolver(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
        latest_gen_fb=gen_fb,
        latest_val_fb=val_fb,
    )

    assert plan.qa_difficulty_distribution["hard"] < blueprint.default_modes["qa"].difficulty_distribution["hard"]
    assert plan.qa_difficulty_distribution["medium"] > blueprint.default_modes["qa"].difficulty_distribution["medium"]


def test_control_parameter_tuner_only_returns_runtime_controls():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=1,
        topic_backlog=TopicBacklog(active=["AI"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )

    plan = control_parameter_tuner(
        state,
        blueprint,
        SimpleNamespace(label="high_separation_low_quality", evidence={}, problems=[]),
    )

    assert plan.selection_mode == "strict"
    assert plan.min_citation_score == 0.7
    assert not hasattr(plan, "qa_count")
    assert not hasattr(plan, "topic_budget")


def test_control_parameter_tuner_uses_feedback_signals_not_only_label():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    val_fb = ValidatorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        summary=ValidatorFeedbackSummary(
            total_candidates=10,
            citation_passed=9,
            llm_passed=8,
            final_selected=6,
            citation_pass_rate=0.9,
            llm_pass_rate_after_citation=0.88,
            final_selection_rate=0.6,
            llm_calls=1,
            llm_input_tokens=1,
            llm_output_tokens=1,
        ),
        quality_signals=ValidatorQualitySignals(duplicate_rate=0.35),
        by_mode={"qa": {"selected": 6}},
    )
    eval_fb = EvaluatorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        summary=EvaluatorFeedbackSummary(num_questions=6, num_models=2),
        dataset_signals=DatasetSignals(),
        derived_performance_signals={
            "overall_model_gap": 0.05,
            "easy_questions_too_easy_ratio": 0.5,
            "all_models_fail_ratio": 0.0,
        },
    )

    plan = control_parameter_tuner(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
        latest_val_fb=val_fb,
        latest_eval_fb=eval_fb,
    )

    assert plan.selection_mode == "strict"
    assert plan.semantic_similarity_threshold >= 0.92
    assert plan.eval_profile == "full"


def test_round_spec_builder_uses_tool_outputs_without_recomputing():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI", "ML", "Systems"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    question_plan = question_difficulty_evolver(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
    )
    control_plan = control_parameter_tuner(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
    )

    round_spec = round_spec_builder(
        state=state,
        blueprint=blueprint,
        target_topics=["AI", "ML", "Systems"],
        run_id="run_002",
        question_plan=question_plan,
        control_plan=control_plan,
    )

    assert round_spec.blueprint["topics"] == ["AI", "ML", "Systems"]
    assert round_spec.qa_agent_patch["planner"]["topics_per_round"] == question_plan.topic_budget
    assert round_spec.qa_agent_patch["candidate_pool"] == {
        "min_candidate_multiplier": question_plan.min_candidate_multiplier,
        "max_candidate_multiplier": question_plan.max_candidate_multiplier,
    }
    assert round_spec.model_eval_agent_patch["judge"]["enabled"] == control_plan.judge_enabled


def test_round_spec_builder_uses_control_plan_for_initial_breadth_without_state_inference():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    question_plan = question_difficulty_evolver(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
    )
    control_plan = control_parameter_tuner(
        state,
        blueprint,
        SimpleNamespace(label="cold_start", evidence={}, problems=[]),
    )
    control_plan.initial_breadth_enabled = False

    round_spec = round_spec_builder(
        state=state,
        blueprint=blueprint,
        target_topics=["AI"],
        run_id="run_002",
        question_plan=question_plan,
        control_plan=control_plan,
    )

    assert round_spec.qa_agent_patch["initial_breadth"]["enabled"] is False


def test_translate_eval_profile_uses_frozen_metrics_for_light_profile():
    blueprint = _make_global_blueprint()
    blueprint.evaluation_requirements = EvaluationRequirements(
        automatic_metrics={
            "qa": ["exact_match"],
            "multiple_choice": ["accuracy"],
        },
        llm_judge_metrics={
            "qa": [
                JudgeMetricDef(name="correctness", description="Judge whether answer is correct.")
            ],
            "multiple_choice": [],
        },
    )

    patch = translate_eval_profile("light", blueprint)

    assert patch["metrics"]["qa"]["llm_judge_metrics"][0]["description"] == "Judge whether answer is correct."
    assert patch["metrics"]["multiple_choice"]["llm_judge_metrics"] == []
    assert patch["metrics"]["multiple_choice"]["automatic_metrics"][0]["name"] == "accuracy"
    assert "judge" not in patch


def test_build_next_round_spec_preserves_topic_budget_as_single_source_of_truth():
    blueprint = _make_global_blueprint()
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI", "ML"], deferred=[]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
    )
    control_plan = adjust_next_round_plan(
        state,
        blueprint,
        SimpleNamespace(label="high_quality_low_separation", evidence={}, problems=[]),
    )

    round_spec = _build_next_round_spec(
        state=state,
        blueprint=blueprint,
        proposed_topics=["AI"],
        run_id="run_002",
        control_plan=control_plan,
    )

    assert round_spec.qa_agent_patch["planner"]["topics_per_round"] == control_plan.topics_per_round
    assert "metrics" not in round_spec.model_eval_agent_patch


def test_summarize_topic_feedback_uses_generator_coverage_and_fallback_fail_ratio():
    state = PlannerState(
        task_id="task_x",
        blueprint_id="bp_x",
        current_round=2,
        topic_backlog=TopicBacklog(active=["AI"], deferred=["ML"]),
        model_pool=ModelPool(active=["model-a"], removed=[]),
        run_history=[RunHistoryEntry(round_id=1, run_id="run_001", topics=["AI"], qa_target=2, multiple_choice_target=1, selected_count=1, evaluated=True)],
    )
    gen_fb = GeneratorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        artifacts=GeneratorFeedbackArtifacts(shared_state_path="", generation_report=""),
        summary=GeneratorFeedbackSummary(total_candidates=2, global_used_chunk_combinations=1, global_failures=0),
        topic_coverage={"ML": {"candidate_count": 1}},
    )
    val_fb = ValidatorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        summary=ValidatorFeedbackSummary(
            total_candidates=2,
            citation_passed=1,
            llm_passed=1,
            final_selected=1,
            citation_pass_rate=0.5,
            llm_pass_rate_after_citation=1.0,
            final_selection_rate=0.5,
            llm_calls=1,
            llm_input_tokens=1,
            llm_output_tokens=1,
        ),
        quality_signals=ValidatorQualitySignals(),
        by_topic={"AI": {"selected": 1, "reserve": 0, "rejected": 0, "avg_citation_score": 0.9}},
    )
    eval_fb = EvaluatorFeedback(
        task_id="task_x",
        run_id="run_002",
        round_id=2,
        status="success",
        summary=EvaluatorFeedbackSummary(num_questions=2, num_models=2),
        dataset_signals=DatasetSignals(),
        derived_performance_signals={
            "overall_model_gap": 0.2,
            "by_topic": {
                "AI": {"question_count": 2, "model_gap": 0.25, "too_easy_ratio": 0.0, "all_fail_ratio": 0.0},
                "ML": {"question_count": 1, "model_gap": 0.05, "too_easy_ratio": 0.0, "all_fail_ratio": 1.0},
            },
        },
    )

    summary = summarize_topic_feedback(state, gen_fb, val_fb, eval_fb)

    assert "AI" in summary.strong_topics
    assert "ML" in summary.low_yield_topics
    assert "ML" in summary.weak_topics
    assert summary.topic_stats["ML"]["all_models_fail_ratio"] == 1.0
