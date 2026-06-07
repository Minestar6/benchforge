"""核心数据模式定义（基于设计文档 2026-05-21）。"""

from datetime import datetime
from typing import Any
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class QuestionMode(str, Enum):
    """问题模式枚举。"""
    MULTIPLE_CHOICE = "multiple_choice"
    QA = "qa"


class Difficulty(str, Enum):
    """难度枚举。"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DocumentStatus(str, Enum):
    """文档状态枚举。"""
    FETCHED = "fetched"
    PROCESSED = "processed"
    FAILED = "failed"


class QuestionStatus(str, Enum):
    """问题状态枚举（旧版兼容）。"""
    GENERATED = "generated"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    EVALUATED = "evaluated"


class QuestionType(str, Enum):
    """问题类型枚举（旧版兼容）。"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONCEPTUAL = "conceptual"
    APPLICATION_BASED = "application-based"
    CLARIFICATION = "clarification"
    COUNTERFACTUAL = "counterfactual"
    TRUE_FALSE = "true-false"
    OPEN_ENDED = "open-ended"
    FALSE_PREMISE = "false-premise"
    EDGE_CASE = "edge-case"


class TopicStatus(str, Enum):
    """主题状态枚举。"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    DEFERRED = "deferred"


class NextStepAction(str, Enum):
    """下一步动作枚举。"""
    CONTINUE_GENERATION = "continue_generation"
    EXPAND_RETRIEVAL = "expand_retrieval"
    INCREASE_MULTI_CHUNK_RATIO = "increase_multi_chunk_ratio"
    ENABLE_HARDENING = "enable_hardening"
    ENABLE_COMPLEXITY_EVOLUTION = "enable_complexity_evolution"
    SWITCH_TOPIC = "switch_topic"
    DEFER_TOPIC = "defer_topic"
    GLOBAL_BACKFILL = "global_backfill"


class SourceDocument(BaseModel):
    """源文档模式。"""

    document_id: str
    run_id: str
    topic: str
    language: str = "en"
    title: str
    url: str
    summary: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.FETCHED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SourceChunk(BaseModel):
    """文档分块模式。"""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    """引用模式。"""
    chunk_id: str
    text: str
    start_index: int = 0
    end_index: int = 0


class QuestionRecord(BaseModel):
    """问题记录模式（兼容 Yourbench 字段 + required_capability）。"""

    question_id: str = Field(default_factory=lambda: f"q_{uuid4().hex[:12]}")
    run_id: str
    question: str
    answer: str
    question_mode: QuestionMode
    required_capability: str = ""
    thought_process: str = ""  # Yourbench 字段
    question_type: str = "factual"  # Yourbench 字段
    citations: list[Citation] = Field(default_factory=list)
    estimated_difficulty: Difficulty = Difficulty.MEDIUM
    status: str = "generated"

    # 上下文信息
    document_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    language: str = "en"
    domain: str | None = None

    # 元数据
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("estimated_difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v: Any) -> Difficulty:
        """支持 1-10 数字难度到 easy/medium/hard 的转换。"""
        if isinstance(v, int):
            if v <= 3:
                return Difficulty.EASY
            elif v <= 7:
                return Difficulty.MEDIUM
            else:
                return Difficulty.HARD
        return v


class QuestionModeTarget(BaseModel):
    """问题模式目标。"""
    count: int
    difficulty_distribution: dict[str, float]

    @field_validator("difficulty_distribution")
    @classmethod
    def validate_difficulty_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        """验证难度分布。"""
        allowed_keys = {"easy", "medium", "hard"}
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"难度只能是 {allowed_keys} 之一")
        if not v:
            raise ValueError("难度分布不能为空")
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"难度分布总和必须为 1.0，当前为 {total}")
        return v


class GenerationPlan(BaseModel):
    """生成计划（PlannerAgent 输出）。"""

    task_id: str
    run_id: str
    goal: str
    topics: list[str]
    mode_targets: dict[str, QuestionModeTarget]
    max_rounds_per_topic: int
    max_total_rounds: int
    language: str = "en"
    retrieval_policy: str = "wikipedia_first"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("mode_targets")
    @classmethod
    def validate_mode_targets(cls, v: dict[str, QuestionModeTarget]) -> dict[str, QuestionModeTarget]:
        """验证模式目标。"""
        allowed_modes = {"multiple_choice", "qa"}
        for mode in v.keys():
            if mode not in allowed_modes:
                raise ValueError(f"模式只能是 {allowed_modes} 之一")
        return v


class TopicState(BaseModel):
    """主题状态。"""

    topic: str
    status: TopicStatus = TopicStatus.PENDING
    current_round: int = 0
    prefer_multi_chunk: bool = False
    target_counts: dict[str, int] = Field(default_factory=dict)
    completed_counts: dict[str, int] = Field(default_factory=dict)
    remaining_counts: dict[str, int] = Field(default_factory=dict)
    retrieved_documents: list[str] = Field(default_factory=list)
    available_single_chunk_ids: list[str] = Field(default_factory=list)
    available_multi_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("target_counts", "completed_counts", "remaining_counts")
    @classmethod
    def validate_counts(cls, v: dict[str, int]) -> dict[str, int]:
        """验证计数字段格式。"""
        for key in v.keys():
            if ":" not in key:
                raise ValueError(f"计数字段键必须包含 ':' 分隔符，如 'multiple_choice:easy'")
        return v


class SingleChunkUnit(BaseModel):
    """单证据单元。"""

    chunk_id: str
    document_id: str
    topic: str
    text: str
    tags: list[str] = Field(default_factory=list)
    mcq_score: float = 0.0
    qa_score: float = 0.0
    hard_score: float = 0.0
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MultiChunkUnit(BaseModel):
    """多证据单元。"""

    unit_id: str = Field(default_factory=lambda: f"multi_{uuid4().hex[:12]}")
    document_id: str
    topic: str
    chunk_ids: list[str]
    texts: list[str]
    tags: list[str] = Field(default_factory=list)
    mcq_score: float = 0.0
    qa_score: float = 0.0
    hard_score: float = 0.0
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceTypeStats(BaseModel):
    """证据类型统计。"""
    avg_candidate_count: float = 0.0
    avg_valid_count: float = 0.0
    mode_distribution: dict[str, float] = Field(default_factory=dict)
    difficulty_distribution: dict[str, float] = Field(default_factory=dict)


class EvidenceStats(BaseModel):
    """证据池统计。"""
    single_chunk_stats: EvidenceTypeStats = Field(default_factory=EvidenceTypeStats)
    multi_chunk_stats: EvidenceTypeStats = Field(default_factory=EvidenceTypeStats)


class EvidencePool(BaseModel):
    """证据池。"""
    topic: str
    single_chunks: list[SingleChunkUnit] = Field(default_factory=list)
    multi_chunks: list[MultiChunkUnit] = Field(default_factory=list)
    stats: EvidenceStats = Field(default_factory=EvidenceStats)


class GenerationBatch(BaseModel):
    """生成批次。"""
    topic: str
    target_mode: str
    target_difficulty: str
    remaining_count: int
    single_chunk_ids: list[str] = Field(default_factory=list)
    multi_chunk_ids: list[str] = Field(default_factory=list)
    prompt_template_id: str
    additional_instructions: str = ""
    requested_min_questions: int
    requested_target_questions: int


class NextStepPlan(BaseModel):
    """下一步计划。"""
    topic: str
    action: NextStepAction
    target_gap: str | None = None
    retrieval_expansion_queries: list[str] = Field(default_factory=list)
    sampling_mode: str | None = None
    prefer_multi_chunk: bool = False
    hardening_enabled: bool = False
    complexity_evolution_enabled: bool = False
    additional_instructions: str = ""
    reason: str


# ═══════════════════════════════════════════════════════════════
# 流水线共享状态
# ═══════════════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    """流水线 Agent 状态。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SharedState(BaseModel):
    """流水线共享状态模型。

    各 agent 通过 runs/{task_id}/{run_id}/shared_state.json 传递上下文。
    qa_agent 创建初始版本，verify_agent / model_eval_agent 依次加载并回写。
    """

    task_id: str
    run_id: str
    blueprint: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="产物名 → 文件路径，如 {'qa_candidate_pool': 'runs/.../candidate_pool.json'}",
    )
    agent_status: dict[str, AgentStatus] = Field(
        default_factory=dict,
        description="每个 agent 的当前状态，如 {'generation': 'completed', 'verification': 'pending'}",
    )

    def artifact(self, key: str) -> str | None:
        """获取产物路径，不存在返回 None。"""
        return self.artifacts.get(key)

    def is_completed(self, agent: str) -> bool:
        """检查某个 agent 是否已完成。"""
        return self.agent_status.get(agent) == AgentStatus.COMPLETED

    def set_agent_status(self, agent: str, status: AgentStatus) -> None:
        """更新 agent 状态。"""
        self.agent_status[agent] = status

    def add_artifact(self, key: str, path: str) -> None:
        """添加产物路径。"""
        self.artifacts[key] = str(path)
