"""Planner topic 自适应搜索工具链。"""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass, field
from typing import Any

from .schema import PlannerState


@dataclass
class TopicFeedbackSummary:
    strong_topics: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    noisy_topics: list[str] = field(default_factory=list)
    oversaturated_topics: list[str] = field(default_factory=list)
    low_yield_topics: list[str] = field(default_factory=list)
    under_observed_topics: list[str] = field(default_factory=list)
    topic_stats: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    reuse_counts: dict[str, int] = field(default_factory=dict)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        topic = str(item).strip()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        ordered.append(topic)
    return ordered


def _reuse_counts(state: PlannerState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in state.run_history:
        for topic in entry.topics:
            counts[topic] = counts.get(topic, 0) + 1
    return counts


def _read_group_all_fail_ratio(group_stats: dict[str, Any]) -> float:
    if not isinstance(group_stats, dict):
        return 0.0
    value = group_stats.get("all_models_fail_ratio")
    if value is None:
        value = group_stats.get("all_fail_ratio")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_topic_feedback(
    state: PlannerState,
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> TopicFeedbackSummary:
    """汇总生成/验证/评估三侧 topic 信号。"""
    topic_sources = (
        list(state.topic_backlog.active)
        + list(state.topic_backlog.deferred)
        + [topic for entry in state.run_history for topic in entry.topics]
    )
    if latest_gen_fb:
        topic_sources += list((latest_gen_fb.topic_coverage or {}).keys())
    if latest_val_fb:
        topic_sources += list((latest_val_fb.by_topic or {}).keys())
    if latest_eval_fb:
        topic_sources += list((latest_eval_fb.derived_performance_signals.get("by_topic", {}) or {}).keys())

    all_topics = _ordered_unique(topic_sources)
    reuse_counts = _reuse_counts(state)
    overall_gap = 0.0
    if latest_eval_fb:
        try:
            overall_gap = float(latest_eval_fb.derived_performance_signals.get("overall_model_gap") or 0.0)
        except (TypeError, ValueError):
            overall_gap = 0.0

    summary = TopicFeedbackSummary(reuse_counts=reuse_counts)
    strong_threshold = max(overall_gap, 0.15)
    weak_threshold = strong_threshold * 0.5 if strong_threshold > 0 else 0.08

    for topic in all_topics:
        gen_stats = (latest_gen_fb.topic_coverage or {}).get(topic, {}) if latest_gen_fb else {}
        val_stats = (latest_val_fb.by_topic or {}).get(topic, {}) if latest_val_fb else {}
        eval_stats = (latest_eval_fb.derived_performance_signals.get("by_topic", {}) or {}).get(topic, {}) if latest_eval_fb else {}

        candidate_count = int(gen_stats.get("candidate_count", 0) or 0)
        selected = int(val_stats.get("selected", 0) or 0)
        reserve = int(val_stats.get("reserve", 0) or 0)
        rejected = int(val_stats.get("rejected", 0) or 0)
        question_count = int(eval_stats.get("question_count", 0) or 0)
        observed_count = max(candidate_count, selected + reserve + rejected, question_count)

        model_gap = float(eval_stats.get("model_gap") or 0.0)
        too_easy_ratio = float(eval_stats.get("too_easy_ratio") or 0.0)
        all_fail_ratio = _read_group_all_fail_ratio(eval_stats)

        avg_citation_score = val_stats.get("avg_citation_score")
        avg_llm_overall_score = val_stats.get("avg_llm_overall_score")
        avg_citation_score = float(avg_citation_score) if avg_citation_score is not None else None
        avg_llm_overall_score = float(avg_llm_overall_score) if avg_llm_overall_score is not None else None

        topic_stats = {
            "candidate_count": candidate_count,
            "selected": selected,
            "reserve": reserve,
            "rejected": rejected,
            "observed_count": observed_count,
            "question_count": question_count,
            "model_gap": model_gap,
            "too_easy_ratio": too_easy_ratio,
            "all_models_fail_ratio": all_fail_ratio,
            "avg_citation_score": avg_citation_score,
            "avg_llm_overall_score": avg_llm_overall_score,
            "reuse_count": reuse_counts.get(topic, 0),
        }
        summary.topic_stats[topic] = topic_stats

        if observed_count < 2:
            summary.under_observed_topics.append(topic)
        if candidate_count and candidate_count < 2:
            summary.low_yield_topics.append(topic)
        if too_easy_ratio >= 0.4:
            summary.oversaturated_topics.append(topic)
        if (
            rejected > max(selected + reserve, 0)
            or (avg_citation_score is not None and avg_citation_score < 0.7)
            or (avg_llm_overall_score is not None and avg_llm_overall_score < 0.75)
        ):
            summary.noisy_topics.append(topic)
        if all_fail_ratio >= 0.5 or (observed_count >= 2 and model_gap < weak_threshold):
            summary.weak_topics.append(topic)
        if selected > 0 and model_gap >= strong_threshold and all_fail_ratio < 0.5:
            summary.strong_topics.append(topic)

    summary.strong_topics = _ordered_unique(summary.strong_topics)
    summary.weak_topics = _ordered_unique(summary.weak_topics)
    summary.noisy_topics = _ordered_unique(summary.noisy_topics)
    summary.oversaturated_topics = _ordered_unique(summary.oversaturated_topics)
    summary.low_yield_topics = _ordered_unique(summary.low_yield_topics)
    summary.under_observed_topics = _ordered_unique(summary.under_observed_topics)
    return summary


async def expand_topic_candidates(
    user_goal: str,
    state: PlannerState,
    model_client,
    summary: TopicFeedbackSummary,
    budget: int,
) -> list[str]:
    """生成 topic 候选池：基础 backlog + LLM 扩展候选。"""
    base_candidates = _ordered_unique(list(state.topic_backlog.active) or list(state.topic_backlog.deferred))
    if not base_candidates:
        base_candidates = _ordered_unique([topic for entry in state.run_history for topic in entry.topics])

    expanded_candidates: list[str] = []
    complete = getattr(model_client, "complete", None)
    if complete and inspect.iscoroutinefunction(complete):
        prompt = (
            "You are expanding benchmark topics for the next planning round.\n"
            f"User goal: {user_goal}\n"
            f"Current candidates: {base_candidates}\n"
            f"Strong topics: {summary.strong_topics}\n"
            f"Weak topics: {summary.weak_topics}\n"
            f"Oversaturated topics: {summary.oversaturated_topics}\n"
            f"Low-yield topics: {summary.low_yield_topics}\n"
            f"Under-observed topics: {summary.under_observed_topics}\n"
            f"Return up to {max(budget * 2, 3)} topic candidates as strict JSON only: "
            '{"topics": ["topic1", "topic2"]}'
        )
        try:
            response = await complete(
                model=getattr(model_client, "model_name", "planner"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=384,
            )
            text = response.get("text", "").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                payload = json.loads(text[start:end + 1])
                expanded_candidates = [
                    str(topic).strip()
                    for topic in payload.get("topics", [])
                    if str(topic).strip()
                ]
        except Exception:
            expanded_candidates = []

    return _ordered_unique(base_candidates + expanded_candidates)


def rank_topics(
    candidate_topics: list[str],
    budget: int,
    state: PlannerState,
    summary: TopicFeedbackSummary,
) -> list[str]:
    """在给定预算下对 topic 候选排序并裁剪。"""
    budget = max(1, int(budget or 1))
    candidate_topics = _ordered_unique(candidate_topics)
    if not candidate_topics:
        return []

    def score(topic: str) -> tuple[float, str]:
        stats = summary.topic_stats.get(topic, {})
        value = 0.0
        value += float(stats.get("model_gap") or 0.0) * 4.0
        value += float(stats.get("selected") or 0.0) * 0.6
        value += float(stats.get("candidate_count") or 0.0) * 0.1
        value += float(stats.get("avg_citation_score") or 0.0) * 0.5
        value += float(stats.get("avg_llm_overall_score") or 0.0) * 0.5
        value -= float(stats.get("too_easy_ratio") or 0.0) * 2.0
        value -= float(stats.get("all_models_fail_ratio") or 0.0) * 3.0
        value -= float(summary.reuse_counts.get(topic, 0)) * 1.25

        if topic in summary.strong_topics:
            value += 3.0
        if topic in summary.weak_topics:
            value -= 2.5
        if topic in summary.noisy_topics:
            value -= 2.0
        if topic in summary.oversaturated_topics:
            value -= 1.5
        if topic in summary.low_yield_topics:
            value -= 1.5
        if topic in summary.under_observed_topics:
            value += 0.25

        if topic in state.topic_backlog.active:
            value += 0.4
        return (value, topic)

    ranked = sorted(candidate_topics, key=score, reverse=True)
    return ranked[:budget]
