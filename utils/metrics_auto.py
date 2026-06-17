"""自动指标：逐题逐模型的公式计算。

合并自 evaluation/automatic_metrics/，所有指标实现统一的 AutoMetric 接口。
"""

import re
from abc import ABC, abstractmethod
from typing import Any


class AutoMetric(ABC):
    """自动指标抽象基类。"""
    name: str

    @abstractmethod
    def compute(
        self,
        question: dict[str, Any],
        prediction: str,
        context: dict[str, Any] | None = None,
    ) -> float | None:
        """计算单题单模型分数，无法计算时返回 None。"""
        raise NotImplementedError


# ─── 选择题工具函数 ─────────────────────────────────────────────

def _parse_choice(question: dict[str, Any], prediction: str) -> str | None:
    """将模型原始输出解析为规范化选项标签（A/B/C/D），无法唯一定位时返回 None。"""
    choices = question.get("choices") or question.get("options") or {}
    if isinstance(choices, list):
        choices = {chr(65 + i): v for i, v in enumerate(choices)}

    labels = list(choices.keys()) if choices else ["A", "B", "C", "D"]
    pred = prediction.strip()

    for label in labels:
        if re.fullmatch(rf"(?:Answer\s*:\s*)?{re.escape(label)}\.?", pred, re.IGNORECASE):
            return label.upper()

    json_match = re.search(r'"(?:answer|choice)"\s*:\s*"([^"]+)"', pred, re.IGNORECASE)
    if json_match:
        val = json_match.group(1).strip().upper()
        if val in [l.upper() for l in labels]:
            return val

    if choices:
        matched = [
            label for label, text in choices.items()
            if isinstance(text, str) and text.strip() and text.strip() in pred
        ]
        if len(matched) == 1:
            return matched[0].upper()

    return None


# ─── QA 文本工具函数 ────────────────────────────────────────────

def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9一-鿿\s]", " ", text)
    return text.split()


def _token_stats(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = _normalize(prediction)
    ref_tokens = _normalize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    pred_set = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1
    ref_set = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1
    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


# ─── 选择题指标 ─────────────────────────────────────────────────

class AccuracyMetric(AutoMetric):
    name = "accuracy"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        gold = str(question.get("answer", "")).strip().upper()
        parsed = _parse_choice(question, prediction)
        if parsed is None:
            return None
        return 1.0 if parsed == gold else 0.0


class InvalidRateMetric(AutoMetric):
    name = "invalid_rate"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        parsed = _parse_choice(question, prediction)
        return 1.0 if parsed is None else 0.0


# ─── QA 文本匹配指标 ────────────────────────────────────────────

class ExactMatchMetric(AutoMetric):
    name = "exact_match"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        gold = str(question.get("answer", "")).strip().lower()
        pred = prediction.strip().lower()
        return 1.0 if pred == gold else 0.0


class TokenF1Metric(AutoMetric):
    name = "f1"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        f1, _, _ = _token_stats(prediction, str(question.get("answer", "")))
        return f1


class PrecisionMetric(AutoMetric):
    name = "precision"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        _, precision, _ = _token_stats(prediction, str(question.get("answer", "")))
        return precision


class TokenRecallMetric(AutoMetric):
    name = "recall"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        _, _, recall = _token_stats(prediction, str(question.get("answer", "")))
        return recall


# ─── ROUGE-L ────────────────────────────────────────────────────

class RougeLMetric(AutoMetric):
    name = "rouge_l"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        ref = str(question.get("answer", "")).lower().split()
        pred = prediction.lower().split()
        if not ref or not pred:
            return 0.0
        lcs = _lcs_length(pred, ref)
        precision = lcs / len(pred)
        recall = lcs / len(ref)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# ─── 语义指标 ───────────────────────────────────────────────────

class BleuMetric(AutoMetric):
    name = "bleu"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            ref = str(question.get("answer", "")).lower().split()
            pred = prediction.lower().split()
            if not ref or not pred:
                return 0.0
            return sentence_bleu(
                [ref], pred,
                smoothing_function=SmoothingFunction().method1,
            )
        except ImportError:
            return None


_bertscorer_cache = None


def _get_bertscorer():
    global _bertscorer_cache
    if _bertscorer_cache is None:
        from bert_score import BERTScorer
        _bertscorer_cache = BERTScorer(lang="en", model_type="roberta-large")
    return _bertscorer_cache


class BertScoreMetric(AutoMetric):
    name = "bertscore"

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        try:
            scorer = _get_bertscorer()
            ref = str(question.get("answer", ""))
            P, R, F1 = scorer.score([prediction], [ref])
            return float(F1[0])
        except ImportError:
            return None


class SemanticSimilarityMetric(AutoMetric):
    name = "semantic_similarity"

    def _get_model(self):
        try:
            from benchforge.utils.metrics_dataset import _get_or_load_embedding_model
            return _get_or_load_embedding_model()
        except ImportError:
            return None

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        model = self._get_model()
        if model is None:
            return None
        import numpy as np
        ref = str(question.get("answer", ""))
        embs = model.encode([prediction, ref], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))


class SemanticAccuracyMetric(AutoMetric):
    name = "semantic_accuracy"
    default_threshold = 0.85

    def __init__(self, similarity_metric: SemanticSimilarityMetric | None = None):
        self.similarity_metric = similarity_metric or SemanticSimilarityMetric()

    def compute(self, question: dict, prediction: str, context: dict | None = None) -> float | None:
        raw_threshold = self.default_threshold if context is None else context.get("threshold", self.default_threshold)
        threshold = raw_threshold if raw_threshold is not None else self.default_threshold
        similarity = self.similarity_metric.compute(question, prediction, context)
        if similarity is None:
            return None
        return 1.0 if similarity >= threshold else 0.0
