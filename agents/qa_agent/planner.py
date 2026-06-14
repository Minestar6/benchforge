"""Planning logic: round plan construction and difficulty/topic selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .state import ModeState, CandidateStatus


class RoundStrategy(str, Enum):
    INITIAL_BREADTH = "initial_breadth"
    NORMAL_GENERATE = "normal_generate"
    FOCUS_TOPIC = "focus_topic"
    FOCUS_DIFFICULTY = "focus_difficulty"
    EXPAND_EVIDENCE = "expand_evidence"
    EVOLVE_TO_HARDER = "evolve_to_harder"
    HARD_GENERATE = "hard_generate"

@dataclass(frozen=True)
class ModeRoundPlan:
    mode: str
    round_in_mode: int
    strategy: RoundStrategy
    difficulty: str
    topics: tuple[str, ...]
    single_k: int
    multi_k: int
    target_candidates_per_topic: int
    reason: str
    expand_topics: tuple[str, ...] = ()



def mode_initial_breadth_not_done(mode_state: ModeState, blueprint: Any) -> bool:
    return any(t not in mode_state.initial_coverage for t in blueprint.topics)


def mode_candidate_target(mode_cfg: Any, config: Any) -> int:
    return math.ceil(mode_cfg.count * config.candidate_pool.target_multiplier)


def remaining_mode_rounds(mode_cfg: Any, mode_state: ModeState) -> int:
    return max(1, mode_cfg.max_rounds - mode_state.round_in_mode + 1)


def resolve_chunk_mix(mode: str, difficulty: str, config: Any) -> tuple[float, float]:
    base = config.chunk_mix.by_difficulty[difficulty]
    adj = config.chunk_mix.mode_adjustment.get(mode)
    mode_delta = adj.single_delta if adj is not None else 0.0
    single_ratio = max(0.0, min(1.0, base.single_ratio + mode_delta))
    return single_ratio, 1.0 - single_ratio


def compute_dynamic_chunk_k(
    mode: str,
    mode_cfg: Any,
    difficulty: str,
    selected_topic_count: int,
    mode_state: ModeState,
    blueprint: Any,
    config: Any,
) -> tuple[int, int, int]:
    target = mode_candidate_target(mode_cfg, config)
    current = mode_state.accepted_count
    gap = max(0, target - current)

    rounds_left = remaining_mode_rounds(mode_cfg, mode_state)
    target_this_round = max(1, math.ceil(gap / rounds_left))
    target_per_topic = max(1, math.ceil(target_this_round / selected_topic_count))

    single_ratio, multi_ratio = resolve_chunk_mix(mode, difficulty, config)
    yield_cfg = config.generation_yield[mode]

    single_k = math.ceil(
        (target_per_topic * single_ratio) / max(0.1, yield_cfg.single_chunk_avg_questions)
    )
    multi_k = math.ceil(
        (target_per_topic * multi_ratio) / max(0.1, yield_cfg.multi_chunk_avg_questions)
    )

    limits = config.chunk_limits[mode]
    single_k = max(limits.single_k.min, min(limits.single_k.max, single_k))
    multi_k = max(limits.multi_k.min, min(limits.multi_k.max, multi_k))

    return single_k, multi_k, target_per_topic


def choose_difficulty_for_mode(mode_cfg: Any, mode_state: ModeState) -> str:
    total = max(1, mode_state.accepted_count)
    diff_counts = mode_state.get_difficulty_counts(CandidateStatus.ACCEPTED)
    current_ratio = {d: diff_counts.get(d, 0) / total for d in mode_cfg.difficulty_distribution}
    gaps = {d: mode_cfg.difficulty_distribution[d] - current_ratio.get(d, 0.0)
            for d in mode_cfg.difficulty_distribution}
    max_gap = max(gaps.values())
    candidates = [d for d, g in gaps.items() if abs(g - max_gap) < 1e-9]
    return random.choice(candidates)


def choose_low_coverage_topics(blueprint: Any, mode_state: ModeState, k: int) -> list[str]:
    expected = mode_state.accepted_count / max(1, len(blueprint.topics))
    topic_counts = mode_state.get_topic_counts(CandidateStatus.ACCEPTED)
    scored = sorted(
        blueprint.topics,
        key=lambda t: expected - topic_counts.get(t, 0),
        reverse=True,
    )
    return scored[:k]


def _missing_topics(blueprint: Any, mode_state: ModeState, min_per_topic: int = 1) -> list[str]:
    topic_counts = mode_state.get_topic_counts(CandidateStatus.ACCEPTED)
    return [t for t in blueprint.topics if topic_counts.get(t, 0) < min_per_topic]


# 鈹€鈹€ Rule-based strategy selection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def select_strategy(
    mode_cfg: Any,
    mode_state: ModeState,
    blueprint: Any,
    feedback: Any | None,
    config: Any,
) -> tuple[RoundStrategy, str]:
    hard_ratio = mode_cfg.difficulty_distribution.get("hard", 0.3)
    hard_gap_val = mode_state.hard_gap(hard_ratio)
    too_easy = mode_state.too_easy_ratio()
    acc_rate = mode_state.accept_rate()
    missing = _missing_topics(blueprint, mode_state)

    decision_cfg = getattr(config, "decision", None)
    hard_gap_threshold = getattr(decision_cfg, "hard_gap_threshold", 0.2)
    too_easy_threshold = getattr(decision_cfg, "too_easy_ratio", 0.4)
    accept_rate_threshold = getattr(decision_cfg, "accept_rate_threshold", 0.3)
    runtime_cfg = getattr(config, "runtime", None)
    empty_rounds_threshold = getattr(runtime_cfg, "max_consecutive_empty_rounds_per_mode", 3) - 1

    dup_ratio = feedback.duplicate_chunk_ratio if feedback else 0.0

    if mode_state.consecutive_empty_rounds >= empty_rounds_threshold and dup_ratio > 0.5:
        return RoundStrategy.EXPAND_EVIDENCE, (
            f"consecutive_empty={mode_state.consecutive_empty_rounds}, dup_ratio={dup_ratio:.2f}"
        )


    med_ratio = mode_cfg.difficulty_distribution.get("medium", 0.3)
    # EVOLVE_TO_HARDER is disabled; hard gap now triggers direct HARD_GENERATE.
    if hard_gap_val > 0 or too_easy > too_easy_threshold:
        return RoundStrategy.HARD_GENERATE, (
            f"hard_gap={hard_gap_val:.2f}, too_easy={too_easy:.2f}; "
            "use direct hard generation"
        )

    if acc_rate < accept_rate_threshold and len(mode_state.candidate_questions) > 0:
        if missing:
            return RoundStrategy.FOCUS_TOPIC, (
                f"low_accept_rate={acc_rate:.2f}, missing_topics={len(missing)}"
            )
        return RoundStrategy.FOCUS_DIFFICULTY, f"low_accept_rate={acc_rate:.2f}"

    if missing:
        return RoundStrategy.FOCUS_TOPIC, f"missing_topics={missing}"

    acc_diff = mode_state.get_difficulty_counts(CandidateStatus.ACCEPTED)
    acc_total = mode_state.accepted_count
    if acc_total > 0:
        diff_gaps = {
            d: mode_cfg.difficulty_distribution.get(d, 0) - acc_diff.get(d, 0) / acc_total
            for d in mode_cfg.difficulty_distribution
        }
        if max(diff_gaps.values()) > 0.1:
            return RoundStrategy.FOCUS_DIFFICULTY, f"difficulty_gap={diff_gaps}"

    return RoundStrategy.NORMAL_GENERATE, "normal"


# 鈹€鈹€ Plan builders 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def build_initial_breadth_plan(
    mode: str, mode_cfg: Any, blueprint: Any, config: Any, mode_state: ModeState,
) -> ModeRoundPlan:
    topics = tuple(
        t for t in blueprint.topics if t not in mode_state.initial_coverage
    )[: config.initial_breadth.max_topics_per_round]
    difficulty = config.initial_breadth.difficulty
    single_k, multi_k, tgt = compute_dynamic_chunk_k(
        mode=mode, mode_cfg=mode_cfg, difficulty=difficulty,
        selected_topic_count=max(1, len(topics)),
        mode_state=mode_state, blueprint=blueprint, config=config,
    )
    return ModeRoundPlan(
        mode=mode, round_in_mode=mode_state.round_in_mode,
        strategy=RoundStrategy.INITIAL_BREADTH, difficulty=difficulty,
        topics=topics, single_k=single_k, multi_k=multi_k,
        target_candidates_per_topic=tgt,
        reason=f"Initial breadth for mode={mode}.",
    )


def build_adaptive_plan(
    mode: str, mode_cfg: Any, blueprint: Any, config: Any,
    mode_state: ModeState, feedback: Any | None,
) -> ModeRoundPlan:
    strategy, reason = select_strategy(mode_cfg, mode_state, blueprint, feedback, config)

    if strategy == RoundStrategy.HARD_GENERATE:
        topics = tuple(
            choose_low_coverage_topics(blueprint, mode_state, config.planner.topics_per_round)
        )
        single_k, multi_k, tgt = compute_dynamic_chunk_k(
            mode=mode, mode_cfg=mode_cfg, difficulty="hard",
            selected_topic_count=max(1, len(topics)),
            mode_state=mode_state, blueprint=blueprint, config=config,
        )
        return ModeRoundPlan(
            mode=mode, round_in_mode=mode_state.round_in_mode,
            strategy=RoundStrategy.HARD_GENERATE, difficulty="hard",
            topics=topics, single_k=single_k, multi_k=multi_k,
            target_candidates_per_topic=tgt,
            reason=reason,
        )

    if strategy == RoundStrategy.EXPAND_EVIDENCE:
        missing = _missing_topics(blueprint, mode_state)
        expand_topics = tuple(missing[:3]) if missing else tuple(blueprint.topics[:1])
        return ModeRoundPlan(
            mode=mode, round_in_mode=mode_state.round_in_mode,
            strategy=strategy, difficulty="medium",
            topics=expand_topics, single_k=0, multi_k=0,
            target_candidates_per_topic=0,
            expand_topics=expand_topics,
            reason=reason,
        )

    difficulty = (
        choose_difficulty_for_mode(mode_cfg, mode_state)
    )
    topics = (
        tuple(_missing_topics(blueprint, mode_state)[:config.planner.topics_per_round])
        if strategy == RoundStrategy.FOCUS_TOPIC
        else tuple(choose_low_coverage_topics(blueprint, mode_state, config.planner.topics_per_round))
    )
    if not topics:
        topics = tuple(choose_low_coverage_topics(blueprint, mode_state, config.planner.topics_per_round))

    single_k, multi_k, tgt = compute_dynamic_chunk_k(
        mode=mode, mode_cfg=mode_cfg, difficulty=difficulty,
        selected_topic_count=max(1, len(topics)),
        mode_state=mode_state, blueprint=blueprint, config=config,
    )
    if difficulty == "hard":
        multi_k = max(multi_k, single_k)

    return ModeRoundPlan(
        mode=mode, round_in_mode=mode_state.round_in_mode,
        strategy=strategy, difficulty=difficulty,
        topics=topics, single_k=single_k, multi_k=multi_k,
        target_candidates_per_topic=tgt,
        reason=reason,
    )


def build_mode_round_plan(
    mode: str, mode_cfg: Any, blueprint: Any, config: Any,
    mode_state: ModeState, feedback: Any | None = None,
) -> ModeRoundPlan:
    if config.initial_breadth.enabled and mode_initial_breadth_not_done(mode_state, blueprint):
        return build_initial_breadth_plan(mode, mode_cfg, blueprint, config, mode_state)
    return build_adaptive_plan(mode, mode_cfg, blueprint, config, mode_state, feedback)


# Keep old name for backward compat
choose_low_coverage_topics_for_mode = choose_low_coverage_topics
