"""评分聚合工具：提取 auto_metric_executor 和 llm_judge_executor 的公共逻辑。"""

from collections import defaultdict
from typing import Any


def build_mode_question_ids(questions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """按 question_mode 分组 question_id 列表，去重并保持顺序。"""
    result: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for q in questions:
        qid = q["question_id"]
        if qid not in seen:
            seen.add(qid)
            result[q.get("question_mode", "")].append(qid)
    return result


def compute_scores_and_mean(
    qid_scores: dict[str, float | None],
    question_ids: list[str],
) -> tuple[list[float | None], float | None]:
    """从 {qid: score} 映射计算对齐的 scores 列表和均值。"""
    scores = [qid_scores.get(qid) for qid in question_ids]
    valid = [s for s in scores if s is not None]
    mean = sum(valid) / len(valid) if valid else None
    return scores, mean
