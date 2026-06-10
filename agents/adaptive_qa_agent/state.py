from collections import deque
from dataclasses import dataclass, field


@dataclass
class CandidateRef:
    question_id: str
    topic: str
    difficulty: str
    status: str  # accepted | rejected
    reject_reason: str | None = None


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
            )
            self.candidates.append(ref)
            if item.accepted:
                self.difficulty_counts[ref.difficulty] = self.difficulty_counts.get(ref.difficulty, 0) + 1
                self.topic_counts[ref.topic] = self.topic_counts.get(ref.topic, 0) + 1

    def record_chunk_combination(self, chunk_ids: list[str]):
        self.used_chunk_combinations.add(frozenset(chunk_ids))

    def is_chunk_combination_used(self, chunk_ids: list[str]) -> bool:
        return frozenset(chunk_ids) in self.used_chunk_combinations


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
