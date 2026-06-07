"""数据集级指标：跨题目的整体质量统计。

合并自 evaluation/dataset_metrics/，所有指标实现统一的 DatasetMetric 接口。
"""

import math
from abc import ABC, abstractmethod
from typing import Any


class DatasetMetric(ABC):
    """数据集级指标抽象基类。"""
    name: str

    @abstractmethod
    def compute(
        self,
        questions: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


# ─── citation_score ─────────────────────────────────────────────

def resolve_citation_score(question: dict[str, Any]) -> float | None:
    """从题目记录中解析 citation_score。"""
    if "citation_score" in question:
        val = question["citation_score"]
        if val is not None:
            return float(val)

    for key in ("citation_validation", "validation_result"):
        obj = question.get(key)
        if isinstance(obj, dict):
            for field in ("score", "citation_score"):
                if field in obj and obj[field] is not None:
                    return float(obj[field])

    return None


class CitationScoreMetric(DatasetMetric):
    name = "citation_score"

    def compute(
        self,
        questions: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        threshold = (context or {}).get("threshold")
        scored: list[tuple[str, float]] = []
        missing_ids: list[str] = []

        for q in questions:
            qid = q.get("question_id", "")
            score = resolve_citation_score(q)
            if score is not None:
                scored.append((qid, score))
            else:
                missing_ids.append(qid)

        if not scored:
            return {
                "type": "dataset_metric",
                "metric_name": self.name,
                "threshold": threshold,
                "question_ids": [],
                "scores": [],
                "passed": [],
                "mean": None,
                "pass_rate": None,
                "num_scored": 0,
                "num_missing": len(missing_ids),
            }

        ids = [s[0] for s in scored]
        scores = [s[1] for s in scored]
        mean = sum(scores) / len(scores)
        passed = [s >= threshold for s in scores] if threshold is not None else [None] * len(scores)
        pass_rate = sum(passed) / len(passed) if threshold is not None else None

        return {
            "type": "dataset_metric",
            "metric_name": self.name,
            "threshold": threshold,
            "question_ids": ids,
            "scores": scores,
            "passed": passed,
            "mean": mean,
            "pass_rate": pass_rate,
            "num_scored": len(scored),
            "num_missing": len(missing_ids),
        }

    def compute_by_group(
        self,
        questions: list[dict[str, Any]],
        group_key: str,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for q in questions:
            group_value = str(q.get(group_key, "unknown"))
            groups.setdefault(group_value, []).append(q)

        results: list[dict[str, Any]] = []
        for group_value, group_questions in groups.items():
            record = self.compute(group_questions, context)
            record["scope"] = group_key
            record["group_value"] = group_value
            record["num_questions"] = len(group_questions)
            results.append(record)
        return results


# ─── diversity_score ────────────────────────────────────────────

_MIN_GROUP_SIZE = 3


def _compute_diversity(texts: list[str]) -> dict[str, Any] | None:
    if len(texts) < _MIN_GROUP_SIZE:
        return None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        n = len(embs)
        if n > 500:
            idx = np.random.choice(n, 500, replace=False)
            sample = embs[idx]
        else:
            sample = embs
        sim_matrix = np.dot(sample, sample.T)
        np.fill_diagonal(sim_matrix, 1.0)
        upper = sim_matrix[np.triu_indices(len(sample), k=1)]
        embedding_dispersion = float(np.mean(1 - upper))

        k = min(max(2, len(texts) // 10), 20)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embs)
        counts = np.bincount(labels, minlength=k)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        cluster_entropy = float(-sum(p * math.log(p) for p in probs) / math.log(k))

        diversity_score = 0.5 * embedding_dispersion + 0.5 * cluster_entropy
        return {
            "score": round(diversity_score, 4),
            "embedding_dispersion": round(embedding_dispersion, 4),
            "cluster_entropy": round(cluster_entropy, 4),
        }
    except ImportError:
        return None


class DiversityScoreMetric(DatasetMetric):
    name = "diversity_score"

    def compute(
        self,
        questions: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        texts = [q.get("question", "") for q in questions]
        result = _compute_diversity(texts)
        base = {
            "type": "dataset_metric",
            "metric_name": self.name,
            "scope": "all",
            "num_questions": len(questions),
        }
        if result is None:
            return {**base, "score": None, "components": None}
        return {**base, **result, "components": {
            "embedding_dispersion": result["embedding_dispersion"],
            "cluster_entropy": result["cluster_entropy"],
        }}

    def compute_by_group(
        self,
        questions: list[dict[str, Any]],
        group_key: str,
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict]] = {}
        for q in questions:
            gv = str(q.get(group_key, "unknown"))
            groups.setdefault(gv, []).append(q)

        results = []
        for group_value, group_qs in groups.items():
            texts = [q.get("question", "") for q in group_qs]
            result = _compute_diversity(texts)
            record = {
                "type": "dataset_metric",
                "metric_name": self.name,
                "scope": group_key,
                "group_value": group_value,
                "num_questions": len(group_qs),
            }
            if result is None:
                record.update({"score": None, "components": None})
            else:
                record.update({**result, "components": {
                    "embedding_dispersion": result["embedding_dispersion"],
                    "cluster_entropy": result["cluster_entropy"],
                }})
            results.append(record)
        return results
