"""数据集级指标执行器：计算 citation_score / diversity_score，落盘 dataset_scores.jsonl。"""

from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.utils.metrics_dataset import DiversityScoreMetric
from .metrics_registry import DATASET_METRIC_REGISTRY
from benchforge.utils.artifact_store import ArtifactStore

from .schema import DatasetMetricSpec

_DIVERSITY_GROUP_KEYS = ["question_mode", "topic", "estimated_difficulty"]
_CITATION_GROUP_KEYS = ["question_mode", "topic", "estimated_difficulty"]


def run_dataset_metrics(
    questions: list[dict[str, Any]],
    metric_specs: list[DatasetMetricSpec],
    output_dir: Path,
) -> list[dict[str, Any]]:
    store = ArtifactStore(str(output_dir))
    results: list[dict[str, Any]] = []

    for spec in metric_specs:
        metric = DATASET_METRIC_REGISTRY.get(spec.name)
        if metric is None:
            logger.warning(f"[DatasetMetricExecutor] unknown metric: {spec.name}")
            continue

        context = {"threshold": spec.threshold}

        if spec.name == "diversity_score":
            dm = DiversityScoreMetric()
            # overall
            overall = dm.compute(questions, context)
            results.append(overall)
            # by group
            for key in _DIVERSITY_GROUP_KEYS:
                for rec in dm.compute_by_group(questions, key):
                    results.append(rec)
        elif spec.name == "citation_score" and hasattr(metric, "compute_by_group"):
            result = metric.compute(questions, context)
            results.append(result)
            for key in _CITATION_GROUP_KEYS:
                for rec in metric.compute_by_group(questions, key, context):
                    results.append(rec)
        else:
            result = metric.compute(questions, context)
            results.append(result)

        logger.info(f"[DatasetMetricExecutor] computed {spec.name}")

    store.append_jsonl("dataset_scores.jsonl", results)
    return results
