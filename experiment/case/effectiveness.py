"""Helpers for the case effectiveness experiment."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_effectiveness_run(case_base: Path) -> Path | None:
    matches = sorted(
        [
            path for path in case_base.iterdir()
            if path.is_dir() and path.name.startswith("effectiveness_case_")
        ],
        reverse=True,
    ) if case_base.exists() else []
    return matches[0] if matches else None


def latest_effectiveness_task(runs_base: Path) -> Path | None:
    matches = sorted(
        [
            path for path in runs_base.iterdir()
            if path.is_dir() and path.name.startswith("effectiveness_task_")
        ],
        reverse=True,
    ) if runs_base.exists() else []
    return matches[0] if matches else None


def task_round_run_dir(task_base: Path, task_id: str, round_index: int) -> Path:
    return task_base / task_id / f"round_{round_index:03d}"


def next_round_index(task_dir: Path) -> int:
    round_indices = []
    if task_dir.exists():
        for child in task_dir.iterdir():
            if child.is_dir() and child.name.startswith("round_"):
                suffix = child.name.removeprefix("round_")
                if suffix.isdigit():
                    round_indices.append(int(suffix))
    return (max(round_indices) + 1) if round_indices else 1


def should_run_stage(required_paths: list[Path], *, force: bool = False) -> bool:
    if force:
        return True
    return not all(path.exists() for path in required_paths)


def _candidate_round(candidate: dict[str, Any]) -> int | None:
    metadata = candidate.get("generation_metadata") or {}
    round_id = metadata.get("source_round", metadata.get("generation_round"))
    if round_id is None:
        return None
    try:
        return int(round_id)
    except (TypeError, ValueError):
        return None


def load_selected_questions(
    validated_path: Path,
    round_filter: set[int] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in _read_jsonl(validated_path):
        if record.get("final_status") != "selected":
            continue
        candidate = dict(record.get("candidate") or {})
        round_id = _candidate_round(candidate)
        if round_filter is not None and round_id not in round_filter:
            continue
        selected.append(candidate)
    return selected


def compute_round_improvement_summary(
    round_feedback_path: Path,
    validated_path: Path,
) -> dict[str, Any]:
    feedback_rows = _read_jsonl(round_feedback_path)
    selected_questions = load_selected_questions(validated_path)

    selected_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for question in selected_questions:
        round_id = _candidate_round(question)
        if round_id is not None:
            selected_by_round[round_id].append(question)

    round_1_rows = [row for row in feedback_rows if int(row.get("round_id", 0)) == 1]
    later_rows = [row for row in feedback_rows if int(row.get("round_id", 0)) >= 2]

    round_1 = _aggregate_round_bucket(round_1_rows, selected_by_round, {1})
    round_2_plus = _aggregate_round_bucket(
        later_rows,
        selected_by_round,
        {int(row.get("round_id", 0)) for row in later_rows},
    )
    return {
        "round_1": round_1,
        "round_2_plus": round_2_plus,
        "delta": {
            "selected_rate": round(round_2_plus["selected_rate"] - round_1["selected_rate"], 6),
            "too_easy_ratio": round(round_2_plus["too_easy_ratio"] - round_1["too_easy_ratio"], 6),
            "selected_hard_ratio": round(
                round_2_plus["selected_hard_ratio"] - round_1["selected_hard_ratio"], 6
            ),
        },
    }


def _aggregate_round_bucket(
    feedback_rows: list[dict[str, Any]],
    selected_by_round: dict[int, list[dict[str, Any]]],
    rounds: set[int],
) -> dict[str, Any]:
    generated_count = sum(int(row.get("generated_count", 0)) for row in feedback_rows)
    too_easy_weighted = sum(
        float(row.get("too_easy_ratio", 0.0)) * int(row.get("generated_count", 0))
        for row in feedback_rows
    )
    selected = [q for round_id in rounds for q in selected_by_round.get(round_id, [])]
    hard_selected = [
        q for q in selected
        if str(q.get("estimated_difficulty", "")).lower() == "hard"
    ]
    selected_count = len(selected)
    return {
        "round_count": len(feedback_rows),
        "generated_count": generated_count,
        "selected_count": selected_count,
        "selected_rate": round(selected_count / max(1, generated_count), 6),
        "too_easy_ratio": round(too_easy_weighted / max(1, generated_count), 6),
        "selected_hard_ratio": round(len(hard_selected) / max(1, selected_count), 6),
    }


def compute_benchmark_metrics(automatic_scores: list[dict[str, Any]]) -> dict[str, Any]:
    per_question_model_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in automatic_scores:
        for model_name, info in row.get("models", {}).items():
            scores = info.get("scores", [])
            for question_id, score in zip(row.get("question_ids", []), scores):
                if score is None:
                    continue
                per_question_model_scores[question_id][model_name].append(float(score))

    overall_scores: dict[str, list[float]] = defaultdict(list)
    per_question_gap: list[float] = []
    for question_scores in per_question_model_scores.values():
        question_means: dict[str, float] = {}
        for model_name, scores in question_scores.items():
            if scores:
                model_mean = mean(scores)
                question_means[model_name] = model_mean
                overall_scores[model_name].append(model_mean)
        if len(question_means) >= 2:
            values = sorted(question_means.values(), reverse=True)
            per_question_gap.append(values[0] - values[-1])

    model_scores = {
        model_name: round(mean(scores), 6)
        for model_name, scores in sorted(overall_scores.items())
        if scores
    }
    sorted_scores = sorted(model_scores.values(), reverse=True)
    score_mean = mean(sorted_scores) if sorted_scores else 0.0
    variance = (
        sum((score - score_mean) ** 2 for score in sorted_scores) / len(sorted_scores)
        if len(sorted_scores) >= 2 else 0.0
    )
    adjacent_gap_mean = (
        mean(
            sorted_scores[idx] - sorted_scores[idx + 1]
            for idx in range(len(sorted_scores) - 1)
        )
        if len(sorted_scores) >= 2 else 0.0
    )
    return {
        "num_questions": len(per_question_model_scores),
        "num_models": len(model_scores),
        "model_scores": model_scores,
        "score_variance": round(variance, 6),
        "top_bottom_gap": round(sorted_scores[0] - sorted_scores[-1], 6) if len(sorted_scores) >= 2 else 0.0,
        "adjacent_gap_mean": round(adjacent_gap_mean, 6),
        "per_question_gap_mean": round(mean(per_question_gap), 6) if per_question_gap else 0.0,
    }
