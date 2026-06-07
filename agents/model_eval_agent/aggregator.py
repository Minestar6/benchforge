"""聚合报告生成器：将原始分数聚合为 CSV / JSON 报告。"""

from collections import defaultdict
from typing import Any


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

    _AUTO_BY_MODE = {
        "multiple_choice": ["accuracy", "accuracy_pass_rate", "invalid_rate"],
        "qa": ["exact_match", "f1", "f1_pass_rate", "precision", "recall", "rouge_l", "bleu", "bertscore", "semantic_similarity"],
    }
    _JUDGE_METRICS = ["correctness", "completeness", "faithfulness", "hallucination"]

    rows = []
    for model_name in sorted(all_models):
        for mode, qids in mode_questions.items():
            row: dict[str, Any] = {
                "model_name": model_name,
                "question_mode": mode,
                "num_questions": len(qids),
            }
            # auto metrics
            relevant_auto = _AUTO_BY_MODE.get(mode, [])
            for mn in ["accuracy", "accuracy_pass_rate", "invalid_rate", "exact_match", "f1", "f1_pass_rate",
                       "precision", "recall", "rouge_l", "bleu", "bertscore", "semantic_similarity"]:
                if mn.endswith("_pass_rate"):
                    base = mn.replace("_pass_rate", "")
                    val = None
                    for r in automatic_scores:
                        if r.get("metric_name") == base and r.get("question_mode") == mode:
                            val = (r.get("models", {}).get(model_name) or {}).get("pass_rate")
                            break
                else:
                    val = _auto_mean(
                        [r for r in automatic_scores if r.get("question_mode") == mode],
                        mn, model_name,
                    )
                row[mn] = val if mn in relevant_auto or mn.replace("_pass_rate", "") in relevant_auto else None
            # judge metrics
            for jm in _JUDGE_METRICS:
                row[f"judge_{jm}"] = _judge_mean(
                    [r for r in judge_scores if r.get("question_mode") == mode],
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
                "accuracy": None, "exact_match": None, "f1": None, "semantic_similarity": None,
                "judge_correctness": None, "judge_faithfulness": None,
            }

            if dimension_key == "question_mode_and_difficulty":
                row["question_mode"], row["estimated_difficulty"] = group_key
            elif dimension_key == "topic_and_question_mode":
                row["topic"], row["question_mode"] = group_key
            else:
                row[dimension_key] = group_key

            for mn, field in [
                ("accuracy", "accuracy"),
                ("exact_match", "exact_match"),
                ("f1", "f1"),
                ("semantic_similarity", "semantic_similarity"),
            ]:
                row[field] = _subset_mean(automatic_scores, mn, model_name, qids)
            row["judge_correctness"] = _subset_mean(judge_scores, "correctness", model_name, qids)
            row["judge_faithfulness"] = _subset_mean(judge_scores, "faithfulness", model_name, qids)
            rows.append(row)
    return rows
