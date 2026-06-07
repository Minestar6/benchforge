"""自动指标执行器：逐题逐模型计算，落盘 automatic_scores.jsonl。"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from .metrics_registry import AUTO_METRIC_REGISTRY
from benchforge.utils.artifact_store import ArtifactStore

from .schema import AutoMetricSpec, QuestionModeMetricPlan


def run_automatic_metrics(
    questions: list[dict[str, Any]],
    model_responses: list[dict[str, Any]],
    metric_plans: dict[str, QuestionModeMetricPlan],
    output_dir: Path,
) -> list[dict[str, Any]]:
    # question_id → question dict
    q_index = {q["question_id"]: q for q in questions}
    mode_question_ids: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        mode_question_ids[q.get("question_mode", "")].append(q["question_id"])

    # (question_mode, metric_name) → {model_name: {question_id: score}}
    Bucket = dict[str, dict[str, float | None]]
    buckets: dict[tuple[str, str], Bucket] = defaultdict(lambda: defaultdict(dict))

    for resp in model_responses:
        qid = resp["question_id"]
        model_name = resp["model_name"]
        mode = resp.get("question_mode", "")
        prediction = resp.get("normalized_prediction", resp.get("prediction", ""))
        question = q_index.get(qid)
        plan = metric_plans.get(mode)
        if question is None or plan is None:
            continue

        for spec in plan.automatic_metrics:
            if resp.get("error"):
                score = None
            else:
                metric = AUTO_METRIC_REGISTRY.get(spec.name)
                if metric is None:
                    continue
                score = metric.compute(question, prediction)
            buckets[(mode, spec.name)][model_name][qid] = score

    results: list[dict[str, Any]] = []

    for (mode, metric_name), model_data in buckets.items():
        plan = metric_plans.get(mode)
        spec = next((s for s in plan.automatic_metrics if s.name == metric_name), None) if plan else None
        threshold = spec.threshold if spec else None

        question_ids = mode_question_ids.get(mode, [])

        models_payload: dict[str, Any] = {}
        for model_name, qid_scores in model_data.items():
            scores = [qid_scores.get(qid) for qid in question_ids]
            valid_scores = [s for s in scores if s is not None]
            mean = sum(valid_scores) / len(valid_scores) if valid_scores else None
            if threshold is not None and valid_scores:
                passed = [s >= threshold if s is not None else None for s in scores]
                pass_rate = sum(1 for s in valid_scores if s >= threshold) / len(valid_scores)
            else:
                passed = [None] * len(scores)
                pass_rate = None

            models_payload[model_name] = {
                "scores": scores,
                "passed": passed,
                "mean": mean,
                "pass_rate": pass_rate,
            }

        results.append({
            "type": "model_auto_metric",
            "question_mode": mode,
            "metric_name": metric_name,
            "threshold": threshold,
            "question_ids": question_ids,
            "models": models_payload,
        })

    store = ArtifactStore(str(output_dir))
    store.append_jsonl("automatic_scores.jsonl", results)
    logger.info(f"[AutoMetricExecutor] computed {len(results)} metric×mode rows")
    return results
