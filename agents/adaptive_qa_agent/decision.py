from dataclasses import dataclass, field


@dataclass
class Action:
    action_type: str
    topics: list[str] = field(default_factory=list)
    difficulty: str | None = None
    evidence_strategy: str | None = None
    reason: str = ""


def decide(state, feedback, adaptive_config) -> Action:
    cfg = adaptive_config.decision
    max_repeat = cfg.max_consecutive_same_action

    def _should_suppress(action_type: str) -> bool:
        if action_type == "generate":
            return False
        return (state.last_action_type == action_type
                and state.consecutive_same_action >= max_repeat)

    if (feedback.ratio("answer_not_grounded") > cfg.answer_not_grounded_ratio
            and not _should_suppress("retrieve_more")):
        return Action("retrieve_more", topics=state.missing_topics,
                      evidence_strategy="new_document", reason="grounding failure high")

    if (feedback.ratio("evidence_insufficient") > cfg.evidence_insufficient_ratio
            and not _should_suppress("expand_evidence")):
        return Action("expand_evidence", topics=state.missing_topics,
                      evidence_strategy="neighbor", reason="evidence insufficient")

    if state.hard_gap > cfg.hard_gap_threshold and feedback.ratio("not_multihop") > cfg.not_multihop_ratio:
        return Action("generate", difficulty="hard",
                      evidence_strategy="multi_group", reason="hard+multihop gap")

    if state.hard_gap > cfg.hard_gap_threshold:
        return Action("generate", difficulty="hard",
                      evidence_strategy="high_hard_score", reason="hard gap")

    topic_gap_ratio = len(state.missing_topics) / max(1, len(state.all_topics))
    if topic_gap_ratio > cfg.topic_gap_ratio:
        return Action("generate", topics=state.missing_topics, reason="topic gap")

    if (feedback.ratio("too_easy") > cfg.too_easy_ratio
            and feedback.ratio("evolution_failed") < cfg.evolution_fail_cap
            and not _should_suppress("evolve")):
        return Action("evolve", reason="too many easy, evolve up")

    return Action("generate", reason="default")
