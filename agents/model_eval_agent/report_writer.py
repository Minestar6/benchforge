"""报告写入器：生成综合评估报告 JSON。"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchforge.schemas.core import difficulty_label


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def build_comprehensive_eval_report(
    questions: list[dict[str, Any]],
    automatic_scores: list[dict[str, Any]],
    mode_list: list[str] | None = None,
    dataset_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按 question_mode 分组，综合 Mode/Topic/model 维度的所有自动指标平均值。

    mode_list 来自蓝图 blueprint.mode_targets 的 keys，若为 None 则从题目数据中推导。
    输出结构:
    {
      "qa": {
        "Mode": {"easy": 0.85, "medium": 0.72, "hard": 0.61},
        "Topic": {"数学": 0.78, "编程": 0.82},
        "model": {"gpt-4": 0.88, "qwen": 0.75}
      },
      "multi_choice": { ... },
      "dataset_metrics": {
        "citation_score": {
          "overall": {"mean": 0.77, "num_scored": 37},
          "by_question_mode": {"qa": 0.77, "multiple_choice": 0.77},
          "by_topic": {"Quantum Computing": 0.77, ...},
          "by_difficulty": {"easy": 0.78, ...}
        },
        "diversity_score": { ... }
      }
    }"""

    report: dict[str, Any] = {}
    per_question_model_scores = _build_per_question_model_scores(automatic_scores)
    report["discriminative_signals"] = _build_discriminative_signals(per_question_model_scores)
    report["by_topic"] = _build_group_discriminative_summary(
        questions,
        per_question_model_scores,
        group_key="topic",
    )
    report["by_difficulty"] = _build_group_discriminative_summary(
        questions,
        per_question_model_scores,
        group_key="estimated_difficulty",
        normalize_group=True,
    )
    report["by_mode"] = _build_group_discriminative_summary(
        questions,
        per_question_model_scores,
        group_key="question_mode",
    )

    # ── 自动指标报告（Mode/Topic/model）──
    mode_questions: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        mode = q.get("question_mode", "unknown")
        mode_questions[mode].append(q)

    if mode_list is None:
        mode_list = sorted(mode_questions.keys())

    for mode in mode_list:
        mode_qs = mode_questions.get(mode, [])
        if not mode_qs:
            continue
        mode_report: dict[str, Any] = {}

        # --- Mode (难度维度) ---
        difficulty_groups: dict[str, set[str]] = defaultdict(set)
        for q in mode_qs:
            diff = difficulty_label(q.get("estimated_difficulty", "unknown"))
            difficulty_groups[diff].add(q["question_id"])

        difficulty_scores: dict[str, float] = {}
        for diff, qids in difficulty_groups.items():
            scores = _collect_subset_scores(automatic_scores, mode, qids)
            if scores:
                difficulty_scores[diff] = round(sum(scores) / len(scores), 6)
        if difficulty_scores:
            mode_report["Mode"] = difficulty_scores

        # --- Topic 维度 ---
        topic_groups: dict[str, set[str]] = defaultdict(set)
        for q in mode_qs:
            topic = str(q.get("topic", "unknown"))
            topic_groups[topic].add(q["question_id"])

        topic_scores: dict[str, float] = {}
        for topic, qids in topic_groups.items():
            scores = _collect_subset_scores(automatic_scores, mode, qids)
            if scores:
                topic_scores[topic] = round(sum(scores) / len(scores), 6)
        if topic_scores:
            mode_report["Topic"] = topic_scores

        # --- model 维度 ---
        all_mode_qids = {q["question_id"] for q in mode_qs}
        all_models: set[str] = set()
        for row in automatic_scores:
            if row.get("question_mode") == mode:
                all_models.update(row.get("models", {}).keys())

        model_scores: dict[str, float] = {}
        for model_name in sorted(all_models):
            scores = _collect_model_subset_scores(automatic_scores, mode, model_name, all_mode_qids)
            if scores:
                model_scores[model_name] = round(sum(scores) / len(scores), 6)
        if model_scores:
            mode_report["model"] = model_scores

        report[mode] = mode_report

    # ── 数据集指标报告（citation_score / diversity_score）──
    if dataset_scores:
        report["dataset_metrics"] = _build_dataset_metrics_summary(dataset_scores)

    return report


def _build_per_question_model_scores(
    automatic_scores: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    per_question: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in automatic_scores:
        question_ids = row.get("question_ids", [])
        for model_name, model_info in row.get("models", {}).items():
            scores = model_info.get("scores", [])
            for qid, score in zip(question_ids, scores):
                if score is not None:
                    per_question[qid][model_name].append(float(score))
    return per_question


def _question_gap(scores_by_model: dict[str, list[float]]) -> float | None:
    means = []
    for model_scores in scores_by_model.values():
        if model_scores:
            means.append(sum(model_scores) / len(model_scores))
    if len(means) < 2:
        return None
    return max(means) - min(means)


def _question_variance(scores_by_model: dict[str, list[float]]) -> float | None:
    means = []
    for model_scores in scores_by_model.values():
        if model_scores:
            means.append(sum(model_scores) / len(model_scores))
    if len(means) < 2:
        return None
    mean = sum(means) / len(means)
    return sum((score - mean) ** 2 for score in means) / len(means)


def _build_discriminative_signals(
    per_question_model_scores: dict[str, dict[str, list[float]]],
) -> dict[str, float | None]:
    question_gaps: list[float] = []
    variances: list[float] = []
    best_vs_second: list[float] = []
    easy_count = 0
    too_easy_count = 0
    all_fail_count = 0
    discriminative_count = 0

    for model_scores in per_question_model_scores.values():
        means = sorted(
            (
                sum(scores) / len(scores)
                for scores in model_scores.values()
                if scores
            ),
            reverse=True,
        )
        if len(means) < 2:
            continue
        gap = means[0] - means[-1]
        question_gaps.append(gap)
        variance = _question_variance(model_scores)
        if variance is not None:
            variances.append(variance)
        best_vs_second.append(means[0] - means[1])
        if means[0] >= 0.95:
            easy_count += 1
            if means[-1] >= 0.95:
                too_easy_count += 1
        if means[0] <= 0.05:
            all_fail_count += 1
        if gap >= 0.2:
            discriminative_count += 1

    total = len(question_gaps)
    if total == 0:
        return {
            "overall_model_gap": None,
            "best_vs_second_gap": None,
            "top_vs_bottom_gap": None,
            "per_question_variance_mean": None,
            "discriminative_question_ratio": None,
            "easy_questions_too_easy_ratio": None,
            "all_models_fail_ratio": None,
        }

    return {
        "overall_model_gap": round(sum(question_gaps) / total, 6),
        "best_vs_second_gap": round(sum(best_vs_second) / len(best_vs_second), 6) if best_vs_second else None,
        "top_vs_bottom_gap": round(sum(question_gaps) / total, 6),
        "per_question_variance_mean": round(sum(variances) / len(variances), 6) if variances else None,
        "discriminative_question_ratio": round(discriminative_count / total, 6),
        "easy_questions_too_easy_ratio": round(too_easy_count / max(easy_count, 1), 6) if easy_count else 0.0,
        "all_models_fail_ratio": round(all_fail_count / total, 6),
    }


def _build_group_discriminative_summary(
    questions: list[dict[str, Any]],
    per_question_model_scores: dict[str, dict[str, list[float]]],
    group_key: str,
    normalize_group: bool = False,
) -> dict[str, dict[str, float | int | None]]:
    grouped_qids: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        raw_group = question.get(group_key, "unknown")
        group = difficulty_label(raw_group) if normalize_group else str(raw_group)
        grouped_qids[group].append(question["question_id"])

    result: dict[str, dict[str, float | int | None]] = {}
    for group, qids in grouped_qids.items():
        gaps = []
        too_easy = 0
        all_fail = 0
        for qid in qids:
            model_scores = per_question_model_scores.get(qid, {})
            gap = _question_gap(model_scores)
            if gap is not None:
                gaps.append(gap)
            means = [
                sum(scores) / len(scores)
                for scores in model_scores.values()
                if scores
            ]
            if len(means) >= 2:
                if min(means) >= 0.95:
                    too_easy += 1
                if max(means) <= 0.05:
                    all_fail += 1
        result[group] = {
            "question_count": len(qids),
            "model_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
            "too_easy_ratio": round(too_easy / len(qids), 6) if qids else None,
            "all_models_fail_ratio": round(all_fail / len(qids), 6) if qids else None,
        }
    return result


def _collect_subset_scores(
    automatic_scores: list[dict[str, Any]],
    mode: str,
    qids: set[str],
) -> list[float]:
    """收集某个 mode 下、指定 qids 子集上所有模型×所有指标的有效分数。"""
    collected: list[float] = []
    for row in automatic_scores:
        if row.get("question_mode") != mode:
            continue
        row_qids = row.get("question_ids", [])
        for model_info in row.get("models", {}).values():
            model_scores = model_info.get("scores", [])
            for qid, score in zip(row_qids, model_scores):
                if qid in qids and score is not None:
                    collected.append(score)
    return collected


def _build_dataset_metrics_summary(
    dataset_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """提取 dataset_scores.jsonl 的整体指标为扁平格式。

    输出:
    {
      "citation_score": {"mean": 0.77, "num_scored": 37, "num_missing": 0},
      "diversity_score": {"score": 0.87, "embedding_dispersion": 0.78, "cluster_entropy": 0.96}
    }
    """
    summary: dict[str, Any] = {}

    for record in dataset_scores:
        metric_name = record.get("metric_name", "")
        scope = record.get("scope")

        if metric_name == "citation_score" and scope is None:
            mean_val = record.get("mean")
            summary["citation_score"] = {
                "mean": round(mean_val, 6) if mean_val is not None else None,
                "num_scored": record.get("num_scored", 0),
                "num_missing": record.get("num_missing", 0),
            }

        elif metric_name == "diversity_score" and scope == "all":
            score_val = record.get("score")
            summary["diversity_score"] = {
                "score": round(score_val, 6) if score_val is not None else None,
                "embedding_dispersion": round(record.get("embedding_dispersion", 0), 6),
                "cluster_entropy": round(record.get("cluster_entropy", 0), 6),
            }

    return summary


def _collect_model_subset_scores(
    automatic_scores: list[dict[str, Any]],
    mode: str,
    model_name: str,
    qids: set[str],
) -> list[float]:
    """收集某个 mode 下、指定模型和 qids 子集上所有指标的有效分数。"""
    collected: list[float] = []
    for row in automatic_scores:
        if row.get("question_mode") != mode:
            continue
        model_info = row.get("models", {}).get(model_name)
        if not model_info:
            continue
        row_qids = row.get("question_ids", [])
        model_scores = model_info.get("scores", [])
        for qid, score in zip(row_qids, model_scores):
            if qid in qids and score is not None:
                collected.append(score)
    return collected
