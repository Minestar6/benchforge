"""State models for Mode-Staged Generation Agent."""

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GlobalState:
    used_chunk_combinations: set[tuple[str, ...]] = field(default_factory=set)
    used_chunk_combination_order: deque[tuple[str, ...]] = field(default_factory=deque)
    chunk_usage_counts: dict[str, int] = field(default_factory=dict)
    # Counts failures across all modes for reporting only. Does NOT stop any mode.
    global_failures: int = 0
    # Phase 2: topic_expansion_counts


@dataclass
class ModeState:
    mode: str
    round_in_mode: int = 1
    candidate_questions: list[dict] = field(default_factory=list)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    initial_coverage: set[str] = field(default_factory=set)
    consecutive_empty_rounds: int = 0
    failures: list[dict] = field(default_factory=list)
    failures_count: int = 0
    stopped_reason: str | None = None
    trace: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ModeRoundPlan:
    mode: str
    round_in_mode: int
    strategy: str
    difficulty: str
    topics: tuple[str, ...]
    single_k: int
    multi_k: int
    target_candidates_per_topic: int
    reason: str
