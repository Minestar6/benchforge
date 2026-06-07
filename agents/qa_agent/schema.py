"""Blueprint and AgentConfig dataclasses for qa_agent."""

from dataclasses import dataclass, field


@dataclass
class ChunkMixDifficulty:
    single_ratio: float
    multi_ratio: float


@dataclass
class ModeAdjustment:
    single_delta: float = 0.0


@dataclass
class ChunkMixConfig:
    by_difficulty: dict[str, ChunkMixDifficulty] = field(default_factory=dict)
    mode_adjustment: dict[str, ModeAdjustment] = field(default_factory=dict)


@dataclass
class GenerationYield:
    single_chunk_avg_questions: float
    multi_chunk_avg_questions: float


@dataclass
class ChunkKLimit:
    min: int
    max: int


@dataclass
class ChunkLimitsForMode:
    single_k: ChunkKLimit
    multi_k: ChunkKLimit


@dataclass
class RuntimeConfig:
    max_consecutive_empty_rounds_per_mode: int = 3
    max_failures_per_mode: int = 8
    max_used_chunk_combinations: int = 10000
    llm_timeout_seconds: int = 60
    retrieval_timeout_seconds: int = 30


@dataclass
class CandidatePoolConfig:
    target_multiplier: float = 2.5


@dataclass
class InitialBreadthConfig:
    enabled: bool = True
    max_topics_per_round: int = 10
    difficulty: str = "medium"


@dataclass
class PlannerConfig:
    topics_per_round: int = 3


@dataclass
class ModeCfg:
    count: int
    max_rounds: int
    difficulty_distribution: dict[str, float]


@dataclass
class Blueprint:
    task_id: str
    run_id: str
    language: str
    topics: list[str]
    modes: dict[str, ModeCfg]


@dataclass
class AgentConfig:
    candidate_pool: CandidatePoolConfig
    initial_breadth: InitialBreadthConfig
    planner: PlannerConfig
    chunk_mix: ChunkMixConfig
    generation_yield: dict[str, GenerationYield]
    chunk_limits: dict[str, ChunkLimitsForMode]
    runtime: RuntimeConfig
