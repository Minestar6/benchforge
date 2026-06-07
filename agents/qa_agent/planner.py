"""Planning logic: round plan construction and difficulty/topic selection."""

import math
import random
from typing import Any

from .state import ModeState, ModeRoundPlan


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
    current = len(mode_state.candidate_questions)
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
    total = max(1, len(mode_state.candidate_questions))
    current_ratio = {
        d: mode_state.difficulty_counts.get(d, 0) / total
        for d in mode_cfg.difficulty_distribution
    }
    gaps = {d: mode_cfg.difficulty_distribution[d] - current_ratio.get(d, 0.0)
            for d in mode_cfg.difficulty_distribution}
    max_gap = max(gaps.values())
    # break ties randomly to avoid always picking the same difficulty
    candidates = [d for d, g in gaps.items() if abs(g - max_gap) < 1e-9]
    return random.choice(candidates)


def choose_low_coverage_topics_for_mode(blueprint: Any, mode_state: ModeState, k: int) -> list[str]:
    expected = len(mode_state.candidate_questions) / max(1, len(blueprint.topics))
    scored = sorted(
        blueprint.topics,
        key=lambda t: expected - mode_state.topic_counts.get(t, 0),
        reverse=True,
    )
    return scored[:k]


def build_initial_breadth_plan(
    mode: str,
    mode_cfg: Any,
    blueprint: Any,
    config: Any,
    mode_state: ModeState,
) -> ModeRoundPlan:
    topics = tuple(
        t for t in blueprint.topics if t not in mode_state.initial_coverage
    )[: config.initial_breadth.max_topics_per_round]

    difficulty = config.initial_breadth.difficulty
    single_k, multi_k, target_candidates_per_topic = compute_dynamic_chunk_k(
        mode=mode,
        mode_cfg=mode_cfg,
        difficulty=difficulty,
        selected_topic_count=max(1, len(topics)),
        mode_state=mode_state,
        blueprint=blueprint,
        config=config,
    )

    return ModeRoundPlan(
        mode=mode,
        round_in_mode=mode_state.round_in_mode,
        strategy="initial_breadth",
        difficulty=difficulty,
        topics=topics,
        single_k=single_k,
        multi_k=multi_k,
        target_candidates_per_topic=target_candidates_per_topic,
        reason=f"Initial breadth for mode={mode}.",
    )


def build_adaptive_plan(
    mode: str,
    mode_cfg: Any,
    blueprint: Any,
    config: Any,
    mode_state: ModeState,
) -> ModeRoundPlan:
    difficulty = choose_difficulty_for_mode(mode_cfg, mode_state)
    topics = tuple(choose_low_coverage_topics_for_mode(
        blueprint=blueprint,
        mode_state=mode_state,
        k=config.planner.topics_per_round,
    ))

    single_k, multi_k, target_candidates_per_topic = compute_dynamic_chunk_k(
        mode=mode,
        mode_cfg=mode_cfg,
        difficulty=difficulty,
        selected_topic_count=max(1, len(topics)),
        mode_state=mode_state,
        blueprint=blueprint,
        config=config,
    )

    return ModeRoundPlan(
        mode=mode,
        round_in_mode=mode_state.round_in_mode,
        strategy="adaptive",
        difficulty=difficulty,
        topics=topics,
        single_k=single_k,
        multi_k=multi_k,
        target_candidates_per_topic=target_candidates_per_topic,
        reason=f"Adaptive supplement for mode={mode}, difficulty={difficulty}.",
    )


def build_mode_round_plan(
    mode: str,
    mode_cfg: Any,
    blueprint: Any,
    config: Any,
    mode_state: ModeState,
) -> ModeRoundPlan:
    if config.initial_breadth.enabled and mode_initial_breadth_not_done(mode_state, blueprint):
        return build_initial_breadth_plan(mode, mode_cfg, blueprint, config, mode_state)
    return build_adaptive_plan(mode, mode_cfg, blueprint, config, mode_state)
