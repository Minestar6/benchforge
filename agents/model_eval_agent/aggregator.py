"""聚合报告生成器：将原始分数聚合为 CSV / JSON 报告。"""

from collections import defaultdict
from typing import Any


def _collect_metric_names(score_rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in score_rows:
        metric_name = row.get("metric_name")
        if metric_name and metric_name not in seen:
            names.append(metric_name)
            seen.add(metric_name)
    return names


def _subset_mean(
    score_rows: list[dict[str, Any]],
    metric_name: str,
    model_name: str,
    question_ids: set[str],
) -> float | None:
    subset_scores: list[float] = []
    for row in score_rows:
        if row.get("metric_name") != metric_name:
            continue
        model_data = row.get("models", {}).get(model_name)
        if not model_data:
            continue
        row_qids = row.get("question_ids", [])
        row_scores = model_data.get("scores", [])
        for qid, score in zip(row_qids, row_scores):
            if qid in question_ids and score is not None:
                subset_scores.append(score)
    if not subset_scores:
        return None
    return sum(subset_scores) / len(subset_scores)


def _auto_mean(scores: list[dict], metric_name: str, model_name: str) -> float | None:
    for row in scores:
        if row.get("metric_name") != metric_name:
            continue
        model_data = row.get("models", {}).get(model_name)
        if model_data:
            return model_data.get("mean")
    return None


def _judge_mean(scores: list[dict], metric_name: str, model_name: str) -> float | None:
    for row in scores:
        if row.get("metric_name") != metric_name:
            continue
        model_data = row.get("models", {}).get(model_name)
        if model_data:
            return model_data.get("mean")
    return None


def build_dataset_quality_summary(
    questions: list[dict[str, Any]],
    dataset_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_questions": len(questions)}

    for row in dataset_scores:
        mn = row.get("metric_name")
        if row.get("scope", "all") != "all":
            continue
        if mn == "citation_score":
            summary["citation_score"] = {
                "mean": row.get("mean"),
                "threshold": row.get("threshold"),
                "pass_rate": row.get("pass_rate"),
                "num_scored": row.get("num_scored"),
                "num_missing": row.get("num_missing"),
            }
        elif mn == "diversity_score":
            comp = row.get("components") or {}
            summary["diversity_score"] = {
                "score": row.get("score"),
                "embedding_dispersion": comp.get("embedding_dispersion"),
                "cluster_entropy": comp.get("cluster_entropy"),
            }
    return summary


def build_dataset_quality_by_group(
    dataset_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Group by (scope, group_value)
    groups: dict[tuple[str, str], dict] = {}

    for row in dataset_scores:
        scope = row.get("scope", "all")
        if scope == "all":
            continue
        gv = str(row.get("group_value", ""))
        key = (scope, gv)
        if key not in groups:
            groups[key] = {
                "group_type": scope,
                "group_value": gv,
                "num_questions": row.get("num_questions"),
            }
        mn = row.get("metric_name")
        if mn == "citation_score":
            groups[key].update({
                "avg_citation_score": row.get("mean"),
                "citation_pass_rate": row.get("pass_rate"),
            })
        elif mn == "diversity_score":
            comp = row.get("components") or {}
            groups[key].update({
                "diversity_score": row.get("score"),
                "embedding_dispersion": comp.get("embedding_dispersion"),
                "cluster_entropy": comp.get("cluster_entropy"),
            })

    return list(groups.values())


def build_model_overall_report(
    questions: list[dict[str, Any]],
    automatic_scores: list[dict[str, Any]],
    judge_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 (model_name, question_mode) 聚合，不参与的列填 None。"""
    all_models: set[str] = set()
    for row in automatic_scores + judge_scores:
        all_models.update(row.get("models", {}).keys())

    mode_questions: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        mode_questions[q.get("question_mode", "unknown")].append(q["question_id"])

    auto_metrics_by_mode: dict[str, list[str]] = defaultdict(list)
    for row in automatic_scores:
        mode = row.get("question_mode", "")
        metric_name = row.get("metric_name")
        if metric_name and metric_name not in auto_metrics_by_mode[mode]:
            auto_metrics_by_mode[mode].append(metric_name)

    judge_metrics_by_mode: dict[str, list[str]] = defaultdict(list)
    for row in judge_scores:
        mode = row.get("question_mode", "")
        metric_name = row.get("metric_name")
        if metric_name and metric_name not in judge_metrics_by_mode[mode]:
            judge_metrics_by_mode[mode].append(metric_name)

    rows = []
    for model_name in sorted(all_models):
        for mode, qids in mode_questions.items():
            row: dict[str, Any] = {
                "model_name": model_name,
                "question_mode": mode,
                "num_questions": len(qids),
            }
            mode_auto_rows = [r for r in automatic_scores if r.get("question_mode") == mode]
            for metric_name in auto_metrics_by_mode.get(mode, []):
                row[metric_name] = _auto_mean(mode_auto_rows, metric_name, model_name)
                pass_rate = None
                for r in mode_auto_rows:
                    if r.get("metric_name") == metric_name:
                        pass_rate = (r.get("models", {}).get(model_name) or {}).get("pass_rate")
                        break
                if pass_rate is not None:
                    row[f"{metric_name}_pass_rate"] = pass_rate

            mode_judge_rows = [r for r in judge_scores if r.get("question_mode") == mode]
            for jm in judge_metrics_by_mode.get(mode, []):
                row[f"judge_{jm}"] = _judge_mean(
                    mode_judge_rows,
                    jm, model_name,
                )
            rows.append(row)
    return rows


def build_model_by_dimension(
    questions: list[dict[str, Any]],
    automatic_scores: list[dict[str, Any]],
    judge_scores: list[dict[str, Any]],
    dimension_key: str,
) -> list[dict[str, Any]]:
    """按 (model_name, dimension_key) 聚合基础指标。"""
    all_models: set[str] = set()
    for row in automatic_scores + judge_scores:
        all_models.update(row.get("models", {}).keys())

    dim_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        if dimension_key == "question_mode_and_difficulty":
            key = (str(q.get("question_mode", "unknown")), str(q.get("estimated_difficulty", "unknown")))
        elif dimension_key == "topic_and_question_mode":
            key = (str(q.get("topic", "unknown")), str(q.get("question_mode", "unknown")))
        else:
            key = str(q.get(dimension_key, "unknown"))
        dim_groups[key].append(q)

    rows = []
    for model_name in sorted(all_models):
        for group_key, group_questions in dim_groups.items():
            qids = {q["question_id"] for q in group_questions}
            row: dict[str, Any] = {
                "model_name": model_name,
                "num_questions": len(group_questions),
            }

            if dimension_key == "question_mode_and_difficulty":
                row["question_mode"], row["estimated_difficulty"] = group_key
            elif dimension_key == "topic_and_question_mode":
                row["topic"], row["question_mode"] = group_key
            else:
                row[dimension_key] = group_key

            for metric_name in _collect_metric_names(automatic_scores):
                row[metric_name] = _subset_mean(automatic_scores, metric_name, model_name, qids)
            for metric_name in _collect_metric_names(judge_scores):
                row[f"judge_{metric_name}"] = _subset_mean(judge_scores, metric_name, model_name, qids)
            rows.append(row)
    return rows
