"""指标注册表：自动指标 + 数据集指标。

model_eval_agent 通过此注册表查找和执行所有评测指标。
"""

from benchforge.utils.metrics_auto import (
    AutoMetric,
    AccuracyMetric,
    InvalidRateMetric,
    ExactMatchMetric,
    TokenF1Metric,
    PrecisionMetric,
    TokenRecallMetric,
    RougeLMetric,
    BleuMetric,
    BertScoreMetric,
    SemanticSimilarityMetric,
    SemanticAccuracyMetric,
)

from benchforge.utils.metrics_dataset import (
    DatasetMetric,
    CitationScoreMetric,
    DiversityScoreMetric,
)

# ─── 自动指标注册表 ─────────────────────────────────────────────

AUTO_METRIC_REGISTRY: dict[str, AutoMetric] = {
    "accuracy": AccuracyMetric(),
    "invalid_rate": InvalidRateMetric(),
    "exact_match": ExactMatchMetric(),
    "f1": TokenF1Metric(),
    "precision": PrecisionMetric(),
    "recall": TokenRecallMetric(),
    "rouge_l": RougeLMetric(),
    "bleu": BleuMetric(),
    "bertscore": BertScoreMetric(),
    "semantic_similarity": SemanticSimilarityMetric(),
    "semantic_accuracy": SemanticAccuracyMetric(),
}

# ─── 数据集指标注册表 ───────────────────────────────────────────

DATASET_METRIC_REGISTRY: dict[str, DatasetMetric] = {
    "citation_score": CitationScoreMetric(),
    "diversity_score": DiversityScoreMetric(),
}
