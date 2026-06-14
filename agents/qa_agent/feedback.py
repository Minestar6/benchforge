"""Round feedback aggregation for the Rule Planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import CandidateStatus, ModeState


@dataclass
class RoundFeedback:
    mode: str
    round_id: int
    strategy: str

    # Incremental counts for this round only
    generated_count: int       # accepted 有效题数
    raw_candidate_count: int   # LLM 产出总数（accepted + rejected）
    accepted_count: int
    rejected_count: int
    failure_reason_counts: dict[str, int]

    generated_topic_counts: dict[str, int]
    accepted_topic_counts: dict[str, int]
    rejected_topic_counts: dict[str, int]

    generated_difficulty_counts: dict[str, int]
    accepted_difficulty_counts: dict[str, int]
    rejected_difficulty_counts: dict[str, int]

    accept_rate: float
    duplicate_chunk_ratio: float
    empty_round: bool
    too_easy_ratio: float

    # Planner feedback
    target_candidate_count: int = 0
    over_target_count: int = 0


def build_round_feedback(
    mode: str,
    round_id: int,
    strategy: str,
    round_results: list[dict],
    new_records: list,
    mode_state_before: ModeState | None = None,
    target_candidate_count: int = 0,
) -> RoundFeedback:
    """Aggregate one round's execution results into a RoundFeedback.

    Args:
        round_results: raw topic result dicts from execute_mode_round_plan.
        new_records: CandidateRecord objects added to mode_state this round.
        mode_state_before: unused currently, reserved for future diff computation.
    """
    total_topics = len(round_results)
    dup_topics = sum(1 for r in round_results if r.get("duplicate_combination"))
    dup_ratio = dup_topics / total_topics if total_topics > 0 else 0.0

    gen_topic: dict[str, int] = {}
    for r in round_results:
        t = r.get("topic", "")
        gen_topic[t] = gen_topic.get(t, 0) + r.get("generated_count", 0)

    gen_diff: dict[str, int] = {}
    acc_diff: dict[str, int] = {}
    rej_diff: dict[str, int] = {}
    acc_topic: dict[str, int] = {}
    rej_topic: dict[str, int] = {}
    fail_reasons: dict[str, int] = {}

    acc_count = 0
    rej_count = 0

    for rec in new_records:
        d = rec.difficulty
        t = rec.topic
        if rec.status == CandidateStatus.ACCEPTED:
            acc_count += 1
            acc_diff[d] = acc_diff.get(d, 0) + 1
            acc_topic[t] = acc_topic.get(t, 0) + 1
        elif rec.status == CandidateStatus.REJECTED:
            rej_count += 1
            rej_diff[d] = rej_diff.get(d, 0) + 1
            rej_topic[t] = rej_topic.get(t, 0) + 1
            if rec.reject_reason:
                fail_reasons[rec.reject_reason] = fail_reasons.get(rec.reject_reason, 0) + 1
        gen_diff[d] = gen_diff.get(d, 0) + 1

    raw_candidate_count = len(new_records)
    generated_count = acc_count
    accept_rate = acc_count / raw_candidate_count if raw_candidate_count > 0 else 0.0
    easy_med = acc_diff.get("easy", 0) + acc_diff.get("medium", 0)
    too_easy = easy_med / acc_count if acc_count > 0 else 0.0

    over_target = max(0, acc_count - target_candidate_count) if target_candidate_count > 0 else 0

    return RoundFeedback(
        mode=mode,
        round_id=round_id,
        strategy=strategy,
        target_candidate_count=target_candidate_count,
        over_target_count=over_target,
        generated_count=generated_count,
        raw_candidate_count=raw_candidate_count,
        accepted_count=acc_count,
        rejected_count=rej_count,
        failure_reason_counts=fail_reasons,
        generated_topic_counts=gen_topic,
        accepted_topic_counts=acc_topic,
        rejected_topic_counts=rej_topic,
        generated_difficulty_counts=gen_diff,
        accepted_difficulty_counts=acc_diff,
        rejected_difficulty_counts=rej_diff,
        accept_rate=accept_rate,
        duplicate_chunk_ratio=dup_ratio,
        empty_round=(acc_count == 0),
        too_easy_ratio=too_easy,
    )
