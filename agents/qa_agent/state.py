"""State models for Mode-Staged Generation Agent."""

from __future__ import annotations
import math
from typing import Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

class CandidateStatus(str, Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EVOLVED = "evolved"


@dataclass
class CandidateRecord:
    question_id: str
    question: str
    answer: str
    topic: str
    difficulty: str
    status: CandidateStatus
    source_round: int
    source_strategy: str  # RoundStrategy value; str to avoid circular import
    chunk_ids: list[str]
    reject_reason: str | None = None
    parent_question_id: str | None = None
    choices: list[str] | None = None  # MCQ 选项列表，QA 模式为 None

    # Prompt output metadata
    question_mode: str | None = None
    question_type: str | None = None
    required_capability: str | None = None
    estimated_difficulty: int | float | str | None = None
    citations: list[Any] = field(default_factory=list)
    thought_process: str | None = None

    # Execution metadata
    llm_call_id: str | None = None
    chunks: list[str] = field(default_factory=list)
    generation_round: int | None = None

    # Preserve original model output
    raw_item: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalState:
    used_chunk_combinations: set[tuple[str, ...]] = field(default_factory=set)
    used_chunk_combination_order: deque[tuple[str, ...]] = field(default_factory=deque)
    chunk_usage_counts: dict[str, int] = field(default_factory=dict)
    global_failures: int = 0


@dataclass
class ModeState:
    mode: str
    target_count: int = 0

    candidate_questions: list[CandidateRecord] = field(default_factory=list)

    round_in_mode: int = 1
    consecutive_empty_rounds: int = 0
    initial_coverage: set[str] = field(default_factory=set)
    failures: list[dict] = field(default_factory=list)
    failures_count: int = 0
    stopped_reason: str | None = None
    trace: list[dict] = field(default_factory=list)

    # 鈹€鈹€ Derived statistics 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _by_status(self, status: CandidateStatus) -> list[CandidateRecord]:
        return [q for q in self.candidate_questions if q.status == status]

    @property
    def accepted(self) -> list[CandidateRecord]:
        return self._by_status(CandidateStatus.ACCEPTED)

    @property
    def accepted_count(self) -> int:
        return sum(1 for q in self.candidate_questions if q.status == CandidateStatus.ACCEPTED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for q in self.candidate_questions if q.status == CandidateStatus.REJECTED)

    @property
    def evolved_count(self) -> int:
        return sum(1 for q in self.candidate_questions if q.status == CandidateStatus.EVOLVED)

    def get_difficulty_counts(self, status: CandidateStatus | None = None) -> dict[str, int]:
        pool = self.candidate_questions if status is None else self._by_status(status)
        counts: dict[str, int] = {}
        for q in pool:
            counts[q.difficulty] = counts.get(q.difficulty, 0) + 1
        return counts

    def get_topic_counts(self, status: CandidateStatus | None = None) -> dict[str, int]:
        pool = self.candidate_questions if status is None else self._by_status(status)
        counts: dict[str, int] = {}
        for q in pool:
            counts[q.topic] = counts.get(q.topic, 0) + 1
        return counts

    def get_failure_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.candidate_questions:
            if q.status == CandidateStatus.REJECTED and q.reject_reason:
                counts[q.reject_reason] = counts.get(q.reject_reason, 0) + 1
        return counts

    def accept_rate(self) -> float:
        attempted = sum(
            1 for q in self.candidate_questions
            if q.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED)
        )
        return self.accepted_count / attempted if attempted > 0 else 0.0

    def too_easy_ratio(self) -> float:
        acc = self.accepted_count
        if acc == 0:
            return 0.0
        d = self.get_difficulty_counts(CandidateStatus.ACCEPTED)
        return (d.get("easy", 0) + d.get("medium", 0)) / acc

    def hard_gap(self, target_hard_ratio: float) -> float:
        acc = self.accepted_count
        if acc == 0:
            return target_hard_ratio
        current = self.get_difficulty_counts(CandidateStatus.ACCEPTED).get("hard", 0) / acc
        return max(0.0, target_hard_ratio - current)

    def evolvable_surplus(self, difficulty: str, target_ratio: float) -> int:
        """Accepted questions of given difficulty beyond their target quota."""
        acc = self.accepted_count
        if acc == 0:
            return 0
        quota = math.ceil(acc * target_ratio)
        actual = self.get_difficulty_counts(CandidateStatus.ACCEPTED).get(difficulty, 0)
        return max(0, actual - quota)
