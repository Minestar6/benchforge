from collections import deque
from dataclasses import dataclass, field


@dataclass
class CandidateRef:
    question_id: str
    topic: str
    difficulty: str
    status: str  # accepted | rejected
    reject_reason: str | None = None
    question: str = ""
    answer: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    source_round: int | None = None
    source_action_type: str = ""
    llm_call_id: str | None = None


@dataclass
class ModeState:
    mode: str
    all_topics: list[str] = field(default_factory=list)
    target_hard_ratio: float = 0.3
    target_candidates: int = 100
    max_rounds: int = 20
    round_in_mode: int = 0
    candidates: list[CandidateRef] = field(default_factory=list)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    consecutive_empty: int = 0
    failures: int = 0
    used_chunk_combinations: set[frozenset[str]] = field(default_factory=set)
    last_action_type: str = ""
    consecutive_same_action: int = 0
    trace: list[dict] = field(default_factory=list)
    failure_records: list[dict] = field(default_factory=list)
    stopped_reason: str | None = None

    @property
    def accepted_count(self) -> int:
        return sum(1 for c in self.candidates if c.status == "accepted")

    @property
    def hard_gap(self) -> float:
        total = max(1, self.accepted_count)
        hard = self.difficulty_counts.get("hard", 0)
        return max(0.0, self.target_hard_ratio - hard / total)

    @property
    def missing_topics(self) -> list[str]:
        if not self.all_topics:
            return []
        avg = self.accepted_count / max(1, len(self.all_topics))
        threshold = max(1.0, avg * 0.5)
        return [t for t in self.all_topics if self.topic_counts.get(t, 0) < threshold]

    @property
    def active_topics(self) -> list[str]:
        return self.missing_topics or self.all_topics

    def record_action(self, action_type: str):
        if action_type == self.last_action_type:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
            self.last_action_type = action_type

    def update(self, mapped_results: list):
        self.consecutive_empty = 0
        for item in mapped_results:
            ref = CandidateRef(
                question_id=item.question_id,
                topic=item.topic,
                difficulty=item.difficulty,
                status="accepted" if item.accepted else "rejected",
                reject_reason=item.reject_reason,
                question=item.question,
                answer=item.answer,
                chunk_ids=list(item.chunk_ids or []),
                source_round=item.source_round,
                source_action_type=item.source_action_type,
                llm_call_id=item.llm_call_id,
            )
            self.candidates.append(ref)
            if item.accepted:
                self.difficulty_counts[ref.difficulty] = self.difficulty_counts.get(ref.difficulty, 0) + 1
                self.topic_counts[ref.topic] = self.topic_counts.get(ref.topic, 0) + 1

    def record_chunk_combination(self, chunk_ids: list[str]):
        self.used_chunk_combinations.add(frozenset(chunk_ids))

    def is_chunk_combination_used(self, chunk_ids: list[str]) -> bool:
        return frozenset(chunk_ids) in self.used_chunk_combinations

    def export_metrics(self) -> dict:
        accepted = self.accepted_count
        return {
            "mode_name": self.mode,
            "mode_round_in_progress": self.round_in_mode,
            "mode_target_candidates": self.target_candidates,
            "mode_accepted_count": accepted,
            "mode_accept_progress": accepted / max(1, self.target_candidates),
            "mode_hard_gap": self.hard_gap,
            "mode_missing_topic_count": len(self.missing_topics),
            "mode_missing_topic_ratio": len(self.missing_topics) / max(1, len(self.all_topics)),
            "mode_consecutive_empty": self.consecutive_empty,
            "mode_failures": self.failures,
            "mode_last_action_type": self.last_action_type,
            "mode_consecutive_same_action": self.consecutive_same_action,
            "mode_difficulty_counts": dict(self.difficulty_counts),
            "mode_topic_counts": dict(self.topic_counts),
            "mode_stopped_reason": self.stopped_reason,
        }


class FeedbackState:
    def __init__(self, window_size: int = 5):
        self._window: deque[dict] = deque(maxlen=window_size)
        self.total_generated: int = 0
        self.total_accepted: int = 0

    def push(self, round_stats: dict):
        self._window.append(round_stats)
        self.total_generated += round_stats["total"]
        self.total_accepted += round_stats["accepted"]

    def ratio(self, key: str) -> float:
        if not self._window:
            return 0.0
        total = sum(r["total"] for r in self._window)
        bad = sum(r.get(key, 0) for r in self._window)
        return bad / max(1, total)

    def export_metrics(self) -> dict:
        return {
            "lifetime_total_generated": self.total_generated,
            "lifetime_total_accepted": self.total_accepted,
            "lifetime_accept_rate": self.total_accepted / max(1, self.total_generated),
            "recent_answer_not_grounded_ratio": self.ratio("answer_not_grounded"),
            "recent_evidence_insufficient_ratio": self.ratio("evidence_insufficient"),
            "recent_not_multihop_ratio": self.ratio("not_multihop"),
            "recent_too_easy_ratio": self.ratio("too_easy"),
            "recent_evolution_failed_ratio": self.ratio("evolution_failed"),
        }


def export_adaptive_metrics(state: ModeState, feedback: FeedbackState) -> dict:
    mode_metrics = state.export_metrics()
    feedback_metrics = feedback.export_metrics()
    conflict_keys = mode_metrics.keys() & feedback_metrics.keys()
    assert not conflict_keys, f"metrics key conflict: {sorted(conflict_keys)}"
    return {**mode_metrics, **feedback_metrics}
