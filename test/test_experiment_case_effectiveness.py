"""Tests for experiment/case effectiveness analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.case.effectiveness import (
    compute_benchmark_metrics,
    compute_round_improvement_summary,
    latest_effectiveness_run,
    next_round_index,
    load_selected_questions,
    should_run_stage,
    task_round_run_dir,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_load_selected_questions_filters_by_round(tmp_path: Path) -> None:
    validated_path = tmp_path / "validated_questions.jsonl"
    _write_jsonl(
        validated_path,
        [
            {
                "question_id": "q1",
                "final_status": "selected",
                "candidate": {
                    "question_id": "q1",
                    "question_mode": "qa",
                    "topic": "t1",
                    "estimated_difficulty": "hard",
                    "generation_metadata": {"source_round": 1},
                },
            },
            {
                "question_id": "q2",
                "final_status": "selected",
                "candidate": {
                    "question_id": "q2",
                    "question_mode": "qa",
                    "topic": "t1",
                    "estimated_difficulty": "medium",
                    "generation_metadata": {"generation_round": 2},
                },
            },
            {
                "question_id": "q3",
                "final_status": "rejected_llm",
                "candidate": {
                    "question_id": "q3",
                    "question_mode": "qa",
                    "topic": "t2",
                    "estimated_difficulty": "easy",
                    "generation_metadata": {"source_round": 1},
                },
            },
        ],
    )

    all_selected = load_selected_questions(validated_path)
    round_one = load_selected_questions(validated_path, round_filter={1})

    assert [q["question_id"] for q in all_selected] == ["q1", "q2"]
    assert [q["question_id"] for q in round_one] == ["q1"]


def test_compute_round_improvement_summary_aggregates_round1_vs_later(tmp_path: Path) -> None:
    round_feedback_path = tmp_path / "qa" / "round_feedback.jsonl"
    validated_path = tmp_path / "validation" / "validated_questions.jsonl"
    _write_jsonl(
        round_feedback_path,
        [
            {
                "mode": "qa",
                "round_id": 1,
                "generated_count": 4,
                "accepted_count": 1,
                "too_easy_ratio": 0.75,
                "failure_reason_counts": {"too_easy": 2},
            },
            {
                "mode": "qa",
                "round_id": 2,
                "generated_count": 3,
                "accepted_count": 2,
                "too_easy_ratio": 0.0,
                "failure_reason_counts": {"too_easy": 0},
            },
            {
                "mode": "qa",
                "round_id": 3,
                "generated_count": 2,
                "accepted_count": 2,
                "too_easy_ratio": 0.0,
                "failure_reason_counts": {"too_easy": 0},
            },
        ],
    )
    _write_jsonl(
        validated_path,
        [
            {
                "question_id": "q1",
                "final_status": "selected",
                "candidate": {
                    "question_id": "q1",
                    "question_mode": "qa",
                    "estimated_difficulty": "medium",
                    "generation_metadata": {"source_round": 1},
                },
            },
            {
                "question_id": "q2",
                "final_status": "selected",
                "candidate": {
                    "question_id": "q2",
                    "question_mode": "qa",
                    "estimated_difficulty": "hard",
                    "generation_metadata": {"source_round": 2},
                },
            },
            {
                "question_id": "q3",
                "final_status": "selected",
                "candidate": {
                    "question_id": "q3",
                    "question_mode": "qa",
                    "estimated_difficulty": "hard",
                    "generation_metadata": {"source_round": 3},
                },
            },
        ],
    )

    summary = compute_round_improvement_summary(
        round_feedback_path=round_feedback_path,
        validated_path=validated_path,
    )

    assert summary["round_1"]["generated_count"] == 4
    assert summary["round_1"]["selected_count"] == 1
    assert summary["round_1"]["selected_hard_ratio"] == 0.0
    assert summary["round_2_plus"]["generated_count"] == 5
    assert summary["round_2_plus"]["selected_count"] == 2
    assert summary["round_2_plus"]["selected_hard_ratio"] == 1.0
    assert summary["delta"]["selected_rate"] == 0.15
    assert summary["delta"]["too_easy_ratio"] == -0.75


def test_compute_benchmark_metrics_reports_model_gaps() -> None:
    automatic_scores = [
        {
            "question_ids": ["q1", "q2"],
            "models": {
                "deepseek-v4": {"scores": [0.9, 0.8]},
                "deepseek-v4-pro": {"scores": [0.6, 0.7]},
                "minimax": {"scores": [0.2, 0.1]},
            },
        },
        {
            "question_ids": ["q1", "q2"],
            "models": {
                "deepseek-v4": {"scores": [0.8, 0.9]},
                "deepseek-v4-pro": {"scores": [0.5, 0.6]},
                "minimax": {"scores": [0.2, 0.2]},
            },
        },
    ]

    summary = compute_benchmark_metrics(automatic_scores)

    assert summary["num_questions"] == 2
    assert summary["num_models"] == 3
    assert summary["model_scores"]["deepseek-v4"] == 0.85
    assert summary["model_scores"]["deepseek-v4-pro"] == 0.6
    assert summary["model_scores"]["minimax"] == 0.175
    assert summary["top_bottom_gap"] == 0.675
    assert summary["adjacent_gap_mean"] == 0.3375
    assert summary["score_variance"] == 0.077639


def test_should_run_stage_uses_force_flag_and_required_paths(tmp_path: Path) -> None:
    marker = tmp_path / "done.json"

    assert should_run_stage([marker], force=False) is True

    marker.write_text("{}", encoding="utf-8")

    assert should_run_stage([marker], force=False) is False
    assert should_run_stage([marker], force=True) is True


def test_latest_effectiveness_run_picks_newest_matching_directory(tmp_path: Path) -> None:
    older = tmp_path / "effectiveness_case_20260618_120000"
    newer = tmp_path / "effectiveness_case_20260618_130000"
    ignore = tmp_path / "other_case_20260618_140000"
    older.mkdir()
    newer.mkdir()
    ignore.mkdir()

    assert latest_effectiveness_run(tmp_path) == newer


def test_task_round_run_dir_uses_three_digit_round_index(tmp_path: Path) -> None:
    run_dir = task_round_run_dir(tmp_path, "task_alpha", 7)

    assert run_dir == tmp_path / "task_alpha" / "round_007"


def test_next_round_index_counts_completed_round_directories(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_alpha"
    (task_dir / "round_001").mkdir(parents=True)
    (task_dir / "round_002").mkdir()
    (task_dir / "round_004").mkdir()

    assert next_round_index(task_dir) == 5


def test_initialize_task_upgrades_existing_total_rounds(tmp_path: Path, monkeypatch) -> None:
    import experiment.case.run_effectiveness_experiment as module

    task_id = "effectiveness_task_demo"
    monkeypatch.setattr(module, "RUNS_BASE", tmp_path / "runs")

    task_dir = module.RUNS_BASE / task_id
    inputs_dir = task_dir / "experiment_inputs"
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "case_blueprint_effectiveness.yaml").write_text(
        "language: en\ntopics: [A, B]\nmodes:\n  qa:\n    count: 4\n    max_rounds: 1\n    difficulty_distribution: {medium: 0.5, hard: 0.5}\n  multiple_choice:\n    count: 2\n    max_rounds: 1\n    difficulty_distribution: {medium: 0.5, hard: 0.5}\n",
        encoding="utf-8",
    )
    (inputs_dir / "qa_agent_effectiveness.yaml").write_text("model:\n  name: deepseek-v4\n", encoding="utf-8")
    (inputs_dir / "verify_agent_effectiveness.yaml").write_text("llm_validation:\n  model: deepseek-v4\n", encoding="utf-8")
    (inputs_dir / "model_eval_effectiveness.yaml").write_text("run:\n  shared_state_path: ''\n", encoding="utf-8")
    (task_dir / "task_metadata.json").write_text(
        json.dumps({"task_id": task_id, "total_rounds": 2}, ensure_ascii=False),
        encoding="utf-8",
    )

    _, _, _, _, _ = module._initialize_task(task_id, total_rounds=5)

    metadata = json.loads((task_dir / "task_metadata.json").read_text(encoding="utf-8"))
    assert metadata["total_rounds"] == 5


@pytest.mark.asyncio
async def test_run_eval_subset_uses_config_object_runner(tmp_path: Path, monkeypatch) -> None:
    import experiment.case.run_effectiveness_experiment as module

    eval_config_path = tmp_path / "model_eval.yaml"
    eval_config_path.write_text("run:\n  shared_state_path: ''\n", encoding="utf-8")

    async def fake_run_model_eval_from_config(config, registry_path):
        run_id = config.run.run_id
        out_dir = tmp_path / "runs" / "task_alpha" / run_id / "evaluation" / "model_report"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir.parent / "evaluation_report.json").write_text(
            json.dumps({"task_id": "task_alpha", "run_id": run_id, "num_questions": 1, "num_models": 3}),
            encoding="utf-8",
        )
        (out_dir / "automatic_scores.jsonl").write_text(
            json.dumps(
                {
                    "question_ids": ["q1"],
                    "models": {
                        "deepseek-v4": {"scores": [0.9]},
                        "deepseek-v4-pro": {"scores": [0.5]},
                        "minimax": {"scores": [0.2]},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"task_id": "task_alpha", "run_id": run_id, "num_questions": 1, "num_models": 3}

    class FakeRun:
        def __init__(self):
            self.shared_state_path = ""
            self.task_id = ""
            self.run_id = ""
            self.input_paths = []

    class FakeConfig:
        def __init__(self):
            self.run = FakeRun()

    monkeypatch.setattr(module, "RUNS_BASE", tmp_path / "runs")
    monkeypatch.setattr(module, "load_model_eval_config", lambda _: FakeConfig())
    monkeypatch.setattr(module, "_run_model_eval_from_config", fake_run_model_eval_from_config)

    result = await module._run_eval_subset(
        task_id="task_alpha",
        subset_name="b1_eval",
        subset_records=[{"question_id": "q1", "final_status": "selected"}],
        eval_config_path=eval_config_path,
        force=True,
    )

    assert result["evaluation_result"]["run_id"] == "b1_eval"
    assert result["benchmark_metrics"]["top_bottom_gap"] == 0.7
