"""PlannerAgent 核心数据结构（基于 docs/plan-runtime-aligned.md §4）。"""

import re
from typing import Any
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# § 4.0 UserIntent
# ═══════════════════════════════════════════════════════════════

_TASK_ID_PATTERN = re.compile(r"[^a-z0-9]+")


class UserIntent(BaseModel):
    user_goal: str
    task_id: str | None = None
    language: str = "zh"
    seed_topics: list[str] = Field(default_factory=list)
    qa_target: int = 20
    multiple_choice_target: int = 10
    candidate_model_names: list[str] = Field(default_factory=list)
    judge_model_name: str | None = None
    planner_model_name: str | None = None
    max_rounds: int = 3
    min_selected_per_round: int = 5
    max_total_tokens: int | None = None

    def resolved_task_id(self) -> str:
        if self.task_id:
            return self.task_id
        slug = _TASK_ID_PATTERN.sub("-", self.user_goal.strip().lower()).strip("-")
        return slug[:48] or "benchforge-task"


# ═══════════════════════════════════════════════════════════════
# § 4.1 GlobalBlueprint
# ═══════════════════════════════════════════════════════════════

class QuestionModeDefaults(BaseModel):
    max_rounds: int
    difficulty_distribution: dict[str, float]


class FinalTargets(BaseModel):
    qa: int
    multiple_choice: int


class EvaluatorDefaults(BaseModel):
    candidate_model_names: list[str]
    judge_model_name: str | None = None


class JudgeMetricDef(BaseModel):
    name: str
    description: str


class EvaluationRequirements(BaseModel):
    automatic_metrics: dict[str, list[str]] = Field(default_factory=dict)
    llm_judge_metrics: dict[str, list[JudgeMetricDef]] = Field(default_factory=dict)


class StopConditions(BaseModel):
    max_rounds: int
    min_selected_per_round: int
    max_total_tokens: int | None = None


class GlobalBlueprint(BaseModel):
    task_id: str
    blueprint_id: str
    user_goal: str
    language: str
    seed_topics: list[str]
    final_targets: FinalTargets
    default_modes: dict[str, QuestionModeDefaults]
    evaluator_defaults: EvaluatorDefaults
    evaluation_requirements: EvaluationRequirements
    stop_conditions: StopConditions


# ═══════════════════════════════════════════════════════════════
# § 4.2 PlannerState
# ═══════════════════════════════════════════════════════════════

class ResourceUsage(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    by_agent: dict[str, dict[str, int]] = Field(default_factory=dict)


class TopicBacklog(BaseModel):
    active: list[str] = Field(default_factory=list)
    deferred: list[str] = Field(default_factory=list)


class ModelPool(BaseModel):
    active: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


class RunHistoryEntry(BaseModel):
    round_id: int
    run_id: str
    topics: list[str]
    qa_target: int
    multiple_choice_target: int
    selected_count: int
    evaluated: bool


class PlannerState(BaseModel):
    task_id: str
    blueprint_id: str
    current_round: int = 0
    completed_targets: dict[str, int] = Field(default_factory=lambda: {"qa": 0, "multiple_choice": 0})
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    topic_backlog: TopicBacklog = Field(default_factory=TopicBacklog)
    model_pool: ModelPool = Field(default_factory=ModelPool)
    run_history: list[RunHistoryEntry] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# § 4.3 RoundSpec
# ═══════════════════════════════════════════════════════════════

class RoundPlannerHints(BaseModel):
    eval_profile: str | None = None  # "light" | "standard" | "full" | None


class RoundSpec(BaseModel):
    task_id: str
    blueprint_id: str
    round_id: int
    run_id: str
    objective: str
    blueprint: dict[str, Any]  # 直接序列化为 qa_agent.Blueprint 兼容格式
    qa_agent_patch: dict[str, Any] = Field(default_factory=dict)
    verify_agent_patch: dict[str, Any] = Field(default_factory=dict)
    model_eval_agent_patch: dict[str, Any] = Field(default_factory=dict)
    planner_hints: RoundPlannerHints = Field(default_factory=RoundPlannerHints)


# ═══════════════════════════════════════════════════════════════
# § 7 Feedback
# ═══════════════════════════════════════════════════════════════

class GeneratorFeedbackArtifacts(BaseModel):
    shared_state_path: str
    generation_report: str
    qa_candidate_pool: str | None = None
    multiple_choice_candidate_pool: str | None = None
    qa_mode_state: str | None = None
    multiple_choice_mode_state: str | None = None
    chunked_evidence: str | None = None


class GeneratorFeedbackSummary(BaseModel):
    total_candidates: int
    global_used_chunk_combinations: int
    global_failures: int
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


class ModeGeneratorFeedback(BaseModel):
    candidate_count: int
    target_candidate_count: int
    fulfillment_rate: float
    stopped_reason: str | None = None
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    topic_counts: dict[str, int] = Field(default_factory=dict)


class GeneratorFeedback(BaseModel):
    task_id: str
    run_id: str
    round_id: int
    status: str  # "success" | "partial_success" | "failed"
    artifacts: GeneratorFeedbackArtifacts
    summary: GeneratorFeedbackSummary
    by_mode: dict[str, ModeGeneratorFeedback] = Field(default_factory=dict)
    topic_coverage: dict[str, dict[str, int]] = Field(default_factory=dict)


class ValidatorFeedbackSummary(BaseModel):
    total_candidates: int
    citation_passed: int
    llm_passed: int
    final_selected: int
    citation_pass_rate: float
    llm_pass_rate_after_citation: float
    final_selection_rate: float
    llm_calls: int
    llm_input_tokens: int
    llm_output_tokens: int


class ValidatorQualitySignals(BaseModel):
    avg_citation_score_all: float | None = None
    avg_citation_score_selected: float | None = None
    avg_llm_overall_score_all: float | None = None
    avg_llm_overall_score_selected: float | None = None
    duplicate_rate: float = 0.0
    overquota_rate: float = 0.0


class ValidatorFeedback(BaseModel):
    task_id: str
    run_id: str
    round_id: int
    status: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: ValidatorFeedbackSummary
    quality_signals: ValidatorQualitySignals
    failed_by_stage: dict[str, int] = Field(default_factory=dict)
    by_final_status: dict[str, int] = Field(default_factory=dict)
    by_mode: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_difficulty: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_topic: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selection_summary: dict[str, int] = Field(default_factory=dict)


class EvaluatorFeedbackSummary(BaseModel):
    num_questions: int
    num_models: int
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


class DatasetSignals(BaseModel):
    citation_score_mean: float | None = None
    citation_pass_rate: float | None = None
    diversity_score: float | None = None
    embedding_dispersion: float | None = None
    cluster_entropy: float | None = None


class EvaluatorFeedback(BaseModel):
    task_id: str
    run_id: str
    round_id: int
    status: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: EvaluatorFeedbackSummary
    dataset_signals: DatasetSignals = Field(default_factory=DatasetSignals)
    derived_performance_signals: dict[str, Any] = Field(default_factory=dict)
    dataset_metrics_ref: dict[str, str] = Field(default_factory=dict)
    model_metrics_ref: dict[str, str] = Field(default_factory=dict)
