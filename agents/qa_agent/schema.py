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
    min_candidate_multiplier: float = 1.5
    max_candidate_multiplier: float = 2.0


@dataclass
class InitialBreadthConfig:
    enabled: bool = True
    max_topics_per_round: int = 10
    difficulty: str = "medium"
    difficulty_policy: str = "fixed"
    hard_ratio_threshold: float = 0.3


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
class DecisionConfig:
    hard_gap_threshold: float = 0.2
    too_easy_ratio: float = 0.4
    accept_rate_threshold: float = 0.3


@dataclass
class ExperimentConfig:
    """消融实验控制开关。"""
    name: str = "full"

    # 是否允许根据上一轮反馈改变策略
    enable_feedback: bool = True

    # 是否允许 HARD_GENERATE 策略
    enable_hard_generate: bool = True

    # 是否允许根据难度缺口主动调节 difficulty
    enable_difficulty_adaptation: bool = True

    # 用于 B 组固定策略
    fixed_strategy: str | None = None

    # 用于 B 组固定难度
    fixed_difficulty: str | None = None

    # 是否禁用 initial breadth
    disable_initial_breadth: bool = False

    # 是否在 max_target 停止后执行一次终局 hard 修复轮
    enable_terminal_hard_repair: bool = True
    terminal_hard_repair_min_missing: int = 2
    terminal_hard_repair_gap_threshold: float = 0.15
    terminal_hard_repair_topics: int = 2
    terminal_hard_repair_multi_k_scale: float = 1.0


@dataclass
class AgentConfig:
    candidate_pool: CandidatePoolConfig
    initial_breadth: InitialBreadthConfig
    planner: PlannerConfig
    chunk_mix: ChunkMixConfig
    generation_yield: dict[str, GenerationYield]
    chunk_limits: dict[str, ChunkLimitsForMode]
    runtime: RuntimeConfig
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


@dataclass
class MultiChunkConfig:
    h_min: int = 2
    h_max: int = 5
    multi_ratio: float = 1.0          # multi_units 目标数量 = ceil(single_units × 此值)
    max_units_per_topic: int = 80
    max_units_per_expansion: int = 40
