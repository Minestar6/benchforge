"""Helpers for supplemental evaluation on effectiveness tasks."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchforge.agents.model_eval_agent.report_writer import build_comprehensive_eval_report

from experiment.case.effectiveness import compute_benchmark_metrics


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_eval_tag(tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "supplemental"


def make_eval_tag(models: list[str], explicit_tag: str | None = None) -> str:
    if explicit_tag:
        return sanitize_eval_tag(explicit_tag)
    return sanitize_eval_tag("__".join(models))


def resolve_supplemental_models(
    *,
    requested_models: list[str] | None,
    registry_models: set[str],
    baseline_models: list[str],
) -> list[str]:
    requested = requested_models or sorted(
        model_name
        for model_name in registry_models
        if model_name not in set(baseline_models) and model_name != "fake"
    )

    missing = [name for name in requested if name not in registry_models]
    if missing:
        raise KeyError(f"Models not found in registry: {missing}")

    filtered: list[str] = []
    seen: set[str] = set()
    baseline_set = set(baseline_models)
    for model_name in requested:
        if model_name in baseline_set or model_name == "fake" or model_name in seen:
            continue
        filtered.append(model_name)
        seen.add(model_name)

    if not filtered:
        raise ValueError("No supplemental models to evaluate after filtering baseline and duplicate entries")
    return filtered


def metric_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("type"),
        row.get("question_mode"),
        row.get("metric_name"),
        row.get("threshold"),
        tuple(row.get("question_ids", [])),
    )


def merge_metric_rows(
    baseline_rows: list[dict[str, Any]],
    supplemental_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in baseline_rows + supplemental_rows:
        key = metric_row_key(row)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                **row,
                "question_ids": list(row.get("question_ids", [])),
                "models": dict(row.get("models", {})),
            }
            continue
        existing["models"].update(row.get("models", {}))
    return list(merged.values())


def summarize_llm_judge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        question_mode = row.get("question_mode", "unknown")
        metric_name = row.get("metric_name", "unknown")
        for model_name, model_info in sorted((row.get("models") or {}).items()):
            summary.append(
                {
                    "question_mode": question_mode,
                    "metric_name": metric_name,
                    "model_name": model_name,
                    "mean": model_info.get("mean"),
                }
            )
    return summary


def summarize_model_score_rows(
    *,
    subset_name: str,
    benchmark_metrics: dict[str, Any],
    evaluation_result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, overall_score in sorted((benchmark_metrics.get("model_scores") or {}).items()):
        rows.append(
            {
                "subset": subset_name,
                "scope": "overall",
                "model_name": model_name,
                "score": overall_score,
            }
        )
    for mode in ("qa", "multiple_choice"):
        mode_scores = (((evaluation_result.get(mode) or {}).get("model")) or {})
        for model_name, score in sorted(mode_scores.items()):
            rows.append(
                {
                    "subset": subset_name,
                    "scope": mode,
                    "model_name": model_name,
                    "score": score,
                }
            )
    return rows


def build_combined_subset_report(
    *,
    subset_name: str,
    questions: list[dict[str, Any]],
    dataset_scores: list[dict[str, Any]],
    baseline_automatic_rows: list[dict[str, Any]],
    supplemental_automatic_rows: list[dict[str, Any]],
    baseline_judge_rows: list[dict[str, Any]],
    supplemental_judge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    combined_automatic = merge_metric_rows(baseline_automatic_rows, supplemental_automatic_rows)
    combined_judge = merge_metric_rows(baseline_judge_rows, supplemental_judge_rows)
    combined_eval = build_comprehensive_eval_report(
        questions=questions,
        automatic_scores=combined_automatic,
        mode_list=sorted({str(question.get("question_mode", "unknown")) for question in questions}),
        dataset_scores=dataset_scores,
    )
    benchmark_metrics = compute_benchmark_metrics(combined_automatic)
    combined_eval.update(
        {
            "num_questions": len(questions),
            "num_models": benchmark_metrics["num_models"],
        }
    )
    return {
        "subset_name": subset_name,
        "question_count": len(questions),
        "benchmark_metrics": benchmark_metrics,
        "evaluation_result": combined_eval,
        "merged_automatic_scores": combined_automatic,
        "merged_judge_scores": combined_judge,
        "judge_summary": summarize_llm_judge_rows(combined_judge),
    }


def build_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Supplemental Evaluation Summary",
        "",
        f"- task_id: `{report['task_id']}`",
        f"- baseline_models: `{', '.join(report['baseline_models'])}`",
        f"- supplemental_models: `{', '.join(report['supplemental_models'])}`",
        "",
    ]
    for subset in report["subsets"]:
        benchmark = subset["combined"]["benchmark_metrics"]
        lines.extend(
            [
                f"## {subset['subset_name']}",
                "",
                f"- questions: `{subset['combined']['question_count']}`",
                f"- models_after_merge: `{benchmark['num_models']}`",
                f"- top_bottom_gap: `{benchmark['top_bottom_gap']}`",
                f"- score_variance: `{benchmark['score_variance']}`",
                "",
                "| Model | Overall | QA | MC |",
                "|---|---:|---:|---:|",
            ]
        )
        combined_scores = summarize_model_score_rows(
            subset_name=subset["subset_name"],
            benchmark_metrics=benchmark,
            evaluation_result=subset["combined"]["evaluation_result"],
        )
        by_model: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in combined_scores:
            by_model[row["model_name"]][row["scope"]] = row["score"]
        for model_name in sorted(by_model):
            lines.append(
                "| {model} | {overall} | {qa} | {mc} |".format(
                    model=model_name,
                    overall=by_model[model_name].get("overall", ""),
                    qa=by_model[model_name].get("qa", ""),
                    mc=by_model[model_name].get("multiple_choice", ""),
                )
            )
        lines.append("")
    return "\n".join(lines)
