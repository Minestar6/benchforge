"""planner_agent 单元测试。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.planner_agent.feedback import (
    build_evaluator_feedback,
    build_generator_feedback,
    build_validator_feedback,
)
from benchforge.agents.planner_agent.orchestrator import _load_verify_model_client
from benchforge.agents.planner_agent.schema import (
    GlobalBlueprint,
    FinalTargets,
    QuestionModeDefaults,
    EvaluatorDefaults,
    EvaluationRequirements,
    StopConditions,
    RoundSpec,
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
