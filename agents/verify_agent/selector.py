"""Stage 3：分组 + exact 去重 + 聚类选题（k = target_count）。"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .config_loader import SelectionCfg
from .schema import (
    CitationValidationResult,
    FinalSelectionResult,
    FinalStatus,
    LLMValidationResult,
    QuestionCandidate,
    ValidationBlueprintView,
    ValidatedQuestionRecord,
)

if TYPE_CHECKING:
    pass

# ─── 懒加载 sentence-transformers ────────────────────────────────────────────

_embedding_model = None
_embedding_model_name: str | None = None


def _resolve_embedding_path(name_or_path: str) -> str:
    """解析嵌入模型路径。

    优先从 model_registry.yaml 的 embeddings 段查找逻辑名，
    找不到则直接返回原值（可能是 HF Hub ID 或本地路径）。
    """
    try:
        import yaml
        from pathlib import Path as _Path
        registry_path = _Path("config/model_registry.yaml")
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                registry = yaml.safe_load(f)
            embeddings = registry.get("embeddings", {})
            if name_or_path in embeddings:
                resolved = embeddings[name_or_path].get("path", name_or_path)
                logger.info(f"Resolved embedding '{name_or_path}' -> '{resolved}'")
                return resolved
    except Exception:
        pass
    return name_or_path


def _get_embedding_model(model_name: str):
    global _embedding_model, _embedding_model_name
    resolved = _resolve_embedding_path(model_name)
    if _embedding_model is None or _embedding_model_name != resolved:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading embedding model: {resolved}")
        _embedding_model = SentenceTransformer(resolved)
        _embedding_model_name = resolved
    return _embedding_model


def _compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    model = _get_embedding_model(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ─── difficulty 标准化 ────────────────────────────────────────────────────────

def _int_difficulty_to_label(val: int) -> str:
    if val <= 3:
        return "easy"
    if val <= 6:
        return "medium"
    if val <= 10:
        return "hard"
    return "unknown"


def _normalize_difficulty(difficulty) -> str:
    if difficulty is None:
        return "unknown"
    if isinstance(difficulty, int):
        return _int_difficulty_to_label(difficulty)
    if isinstance(difficulty, str):
        d = difficulty.strip().lower()
        if d in ("easy", "medium", "hard"):
            return d
        try:
            return _int_difficulty_to_label(int(d))
        except ValueError:
            return "unknown"
    return "unknown"


# ─── 质量分 ───────────────────────────────────────────────────────────────────

_SYNTHETIC_SUMMARIES = {"no_model_client", "llm_validation disabled", "llm_validation_disabled"}


def _quality_score(
    question_id: str,
    citation_results: dict[str, CitationValidationResult],
    llm_results: dict[str, LLMValidationResult],
) -> float:
    """
    质量分：LLM 验证有效时用 (citation + llm) 综合分，否则仅用 citation_score。
    """
    citation_score = citation_results[question_id].citation_score if question_id in citation_results else 0.0

    llm_result = llm_results.get(question_id)
    if llm_result is None or llm_result.judge_summary in _SYNTHETIC_SUMMARIES:
        return citation_score

    # LLM 验证有效：0.45 * citation + 0.55 * llm
    return 0.45 * citation_score + 0.55 * llm_result.overall_score


# ─── Exact 去重 ───────────────────────────────────────────────────────────────

def _normalize_question_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _exact_dedup(
    questions: list[QuestionCandidate],
    quality_scores: dict[str, float],
) -> tuple[list[QuestionCandidate], dict[str, str]]:
    """规范化文本 exact 去重，保留 quality_score 更高的那道。"""
    best: dict[str, tuple[QuestionCandidate, float]] = {}  # norm_text -> (q, score)

    for q in questions:
        norm = _normalize_question_text(q.question)
        score = quality_scores[q.question_id]
        if norm not in best or score > best[norm][1]:
            best[norm] = (q, score)

    kept = [v[0] for v in best.values()]
    kept_ids = {q.question_id for q in kept}
    duplicate_of: dict[str, str] = {}

    for q in questions:
        if q.question_id not in kept_ids:
            norm = _normalize_question_text(q.question)
            duplicate_of[q.question_id] = best[norm][0].question_id

    return kept, duplicate_of


# ─── 聚类选题（k = target_count） ────────────────────────────────────────────

def _cluster_select(
    questions: list[QuestionCandidate],
    quality_scores: dict[str, float],
    target_k: int,
    embeddings: np.ndarray,
) -> tuple[list[str], list[str]]:
    """
    K-means 聚类，k = target_k（等于桶内目标数量）。
    每个 cluster 选 quality_score 最高的代表题，其余标记为 overquota。
    """
    from sklearn.cluster import KMeans

    n = len(questions)
    if n <= target_k:
        return [q.question_id for q in questions], []

    kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    selected_ids: list[str] = []
    overquota_ids: list[str] = []

    for cluster_id in range(target_k):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not indices:
            continue
        best = max(indices, key=lambda i: quality_scores[questions[i].question_id])
        selected_ids.append(questions[best].question_id)
        for i in indices:
            if i != best:
                overquota_ids.append(questions[i].question_id)

    # 极端情况：某 cluster 为空，把未分配题放 overquota
    assigned = set(selected_ids) | set(overquota_ids)
    for q in questions:
        if q.question_id not in assigned:
            overquota_ids.append(q.question_id)

    return selected_ids, overquota_ids


# ─── 配额查找 ─────────────────────────────────────────────────────────────────

def _compute_bucket_targets(blueprint: ValidationBlueprintView) -> dict[str, dict[str, int]]:
    """根据 blueprint 动态生成 {mode: {difficulty: target_count}}。"""
    result: dict[str, dict[str, int]] = {}

    for mode, mode_cfg in blueprint.modes.items():
        dist = mode_cfg.difficulty_distribution or {}
        if not dist:
            result[mode] = {}
            continue

        values = list(dist.values())
        if math.isclose(sum(values), float(mode_cfg.count), rel_tol=0.0, abs_tol=1e-6):
            result[mode] = {difficulty: int(round(value)) for difficulty, value in dist.items()}
            continue

        raw = {difficulty: mode_cfg.count * float(ratio) for difficulty, ratio in dist.items()}
        base = {difficulty: int(math.floor(value)) for difficulty, value in raw.items()}
        remainder = mode_cfg.count - sum(base.values())
        fractions = sorted(
            ((raw[difficulty] - base[difficulty], difficulty) for difficulty in raw),
            key=lambda item: item[0],
            reverse=True,
        )
        for _, difficulty in fractions[: max(0, remainder)]:
            base[difficulty] += 1
        result[mode] = base

    return result


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def run_weighted_selection(
    questions: list[QuestionCandidate],
    blueprint: ValidationBlueprintView,
    citation_results: dict[str, CitationValidationResult],
    llm_results: dict[str, LLMValidationResult],
    cfg: SelectionCfg,
) -> tuple[FinalSelectionResult, list[ValidatedQuestionRecord]]:
    """
    Stage 3 主流程：
    1. Exact 去重
    2. 分桶（mode + difficulty）
    3. 桶内聚类选题，k = target_count（不超配额时保留全部）
    """
    bucket_targets = _compute_bucket_targets(blueprint)

    # ── 计算全局质量分 ─────────────────────────────────────────────────────────
    quality_scores = {
        q.question_id: _quality_score(q.question_id, citation_results, llm_results)
        for q in questions
    }

    # ── Step 1: Exact 去重 ────────────────────────────────────────────────────
    all_duplicate_of: dict[str, str] = {}
    unique_questions, dup_map = _exact_dedup(questions, quality_scores)
    all_duplicate_of.update(dup_map)

    # ── Step 2: 分桶 ──────────────────────────────────────────────────────────
    buckets: dict[str, list[QuestionCandidate]] = {}
    group_assignments: dict[str, str] = {}

    for q in unique_questions:
        diff = _normalize_difficulty(q.estimated_difficulty)
        key = f"{q.question_mode}::{diff}"
        buckets.setdefault(key, []).append(q)
        group_assignments[q.question_id] = key

    # ── Step 3: 桶内聚类选题 ──────────────────────────────────────────────────
    selected_ids: list[str] = []
    dropped_as_overquota: list[str] = []

    for bucket_key, bucket_qs in buckets.items():
        mode, diff = bucket_key.split("::", 1)
        target = bucket_targets.get(mode, {}).get(diff, 0)
        current = list(bucket_qs)

        if target > 0 and len(current) > target:
            try:
                embs = _compute_embeddings([q.question for q in current], cfg.embedding_model)
                sel, over = _cluster_select(current, quality_scores, target, embs)
                selected_ids.extend(sel)
                dropped_as_overquota.extend(over)
            except Exception as e:
                logger.warning(f"[{bucket_key}] clustering failed ({e}), fallback to score sort")
                _score_sort_select(current, quality_scores, target, selected_ids, dropped_as_overquota)
        elif target > 0:
            selected_ids.extend(q.question_id for q in current)
        # target <= 0：无配额，全部标记为 reserve（由外层逻辑处理）

    # ── 构建最终记录 ──────────────────────────────────────────────────────────
    selected_set = set(selected_ids)

    all_records: list[ValidatedQuestionRecord] = []
    for q in questions:
        qid = q.question_id
        if qid in all_duplicate_of:
            status = FinalStatus.duplicate.value
            dup_of = all_duplicate_of[qid]
        else:
            status = (
                FinalStatus.selected.value if qid in selected_set
                else FinalStatus.reserve.value
            )
            dup_of = None

        all_records.append(ValidatedQuestionRecord(
            question_id=qid,
            candidate=q,
            citation_validation=citation_results.get(qid),
            llm_validation=llm_results.get(qid),
            group_id=group_assignments.get(qid),
            duplicate_of=dup_of,
            final_weight=quality_scores.get(qid),
            final_status=status,
        ))

    logger.info(
        f"Selection: selected={len(selected_ids)} "
        f"duplicate={len(all_duplicate_of)} "
        f"overquota={len(dropped_as_overquota)}"
    )

    result = FinalSelectionResult(
        selected_question_ids=selected_ids,
        dropped_as_duplicate=list(all_duplicate_of.keys()),
        dropped_as_overquota=dropped_as_overquota,
        group_assignments=group_assignments,
        sampling_weights=quality_scores,
    )
    return result, all_records


def _score_sort_select(
    questions: list[QuestionCandidate],
    quality_scores: dict[str, float],
    target: int,
    selected_ids: list[str],
    overquota_ids: list[str],
) -> None:
    """按质量分降序截取，target <= 0 时全部放 reserve（由调用方处理）。"""
    if target <= 0:
        return
    sorted_qs = sorted(questions, key=lambda q: quality_scores[q.question_id], reverse=True)
    for i, q in enumerate(sorted_qs):
        if i < target:
            selected_ids.append(q.question_id)
        else:
            overquota_ids.append(q.question_id)
