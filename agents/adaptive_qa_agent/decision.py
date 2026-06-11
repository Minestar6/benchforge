from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Action:
    action_type: str
    topics: list[str] = field(default_factory=list)
    difficulty: str | None = None
    evidence_strategy: str | None = None
    reason: str = ""


class DecisionEngine:
    def __init__(self, adaptive_config):
        self.cfg = adaptive_config.decision
        self.rules: list[Callable] = [
            self._rule_grounding_failure,
            self._rule_evidence_insufficient,
            self._rule_hard_multihop_gap,
            self._rule_hard_gap,
            self._rule_topic_gap,
            self._rule_evolve_easy,
        ]

    def decide(self, state, feedback) -> Action:
        for rule in self.rules:
            action = rule(state, feedback)
            if action is not None:
                return action
        return Action("generate", reason="default")

    def _should_suppress(self, state, action_type: str) -> bool:
        if action_type == "generate":
            return False
        return (state.last_action_type == action_type
                and state.consecutive_same_action >= self.cfg.max_consecutive_same_action)

    def _rule_grounding_failure(self, state, feedback) -> Action | None:
        if (feedback.ratio("answer_not_grounded") > self.cfg.answer_not_grounded_ratio
                and not self._should_suppress(state, "retrieve_more")):
            return Action("retrieve_more", topics=state.missing_topics,
                          evidence_strategy="new_document", reason="grounding failure high")
        return None

    def _rule_evidence_insufficient(self, state, feedback) -> Action | None:
        if (feedback.ratio("evidence_insufficient") > self.cfg.evidence_insufficient_ratio
                and not self._should_suppress(state, "expand_evidence")):
            return Action("expand_evidence", topics=state.missing_topics,
                          evidence_strategy="neighbor", reason="evidence insufficient")
        return None

    def _rule_hard_multihop_gap(self, state, feedback) -> Action | None:
        if (state.hard_gap > self.cfg.hard_gap_threshold
                and feedback.ratio("not_multihop") > self.cfg.not_multihop_ratio):
            return Action("generate", difficulty="hard",
                          evidence_strategy="multi_group", reason="hard+multihop gap")
        return None

    def _rule_hard_gap(self, state, feedback) -> Action | None:
        if state.hard_gap > self.cfg.hard_gap_threshold:
            return Action("generate", difficulty="hard",
                          evidence_strategy="high_hard_score", reason="hard gap")
        return None

    def _rule_topic_gap(self, state, feedback) -> Action | None:
        topic_gap_ratio = len(state.missing_topics) / max(1, len(state.all_topics))
        if topic_gap_ratio > self.cfg.topic_gap_ratio:
            return Action("generate", topics=state.missing_topics, reason="topic gap")
        return None

    def _rule_evolve_easy(self, state, feedback) -> Action | None:
        if (feedback.ratio("too_easy") > self.cfg.too_easy_ratio
                and feedback.ratio("evolution_failed") < self.cfg.evolution_fail_cap
                and not self._should_suppress(state, "evolve")):
            return Action("evolve", reason="too many easy, evolve up")
        return None


def decide(state, feedback, adaptive_config) -> Action:
    return DecisionEngine(adaptive_config).decide(state, feedback)
