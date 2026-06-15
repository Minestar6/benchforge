"""verify_agent 所有核心数据类型。"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FinalStatus(str, Enum):
    rejected_citation = "rejected_citation"
    rejected_llm = "rejected_llm"
    validator_error = "validator_error"
    duplicate = "duplicate"
    overquota = "overquota"
    selected = "selected"
    reserve = "reserve"


class QuestionCandidate(BaseModel):
    question_id: str
    task_id: str
    run_id: str
    topic: str
    question: str
    answer: str
    choices: list[str] | None = None
    question_mode: str
    question_type: str = ""
    required_capability: str = ""
    estimated_difficulty: str | int | None = None
    citations: list[Any] = Field(default_factory=list)
    document_id: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


class CitationValidationResult(BaseModel):
    question_id: str
    passed: bool
    answer_citation_score: float
    chunk_citation_score: float
    citation_score: float
    citation_count: int
    matched_citation_count: int
    failed_reasons: list[str]
    matched_spans: list[dict[str, Any]] = Field(default_factory=list)


class LLMValidationResult(BaseModel):
    question_id: str
    passed: bool
    overall_score: float
    dimensions: dict[str, float]
    failed_reasons: list[str]
    judge_summary: str
    llm_call_id: str | None = None
    attempt_count: int = 1
    latency_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class FinalSelectionResult(BaseModel):
    selected_question_ids: list[str]
    dropped_as_duplicate: list[str]
    dropped_as_overquota: list[str]
    group_assignments: dict[str, str]  # question_id -> "mode::difficulty"
    sampling_weights: dict[str, float]  # question_id -> final_weight


class ModeCfg(BaseModel):
    count: int
    difficulty_distribution: dict[str, float]


class ValidationBlueprintView(BaseModel):
    topics: list[str]
    modes: dict[str, ModeCfg]


class ValidationRunState(BaseModel):
    task_id: str
    run_id: str
    total_candidates: int
    citation_passed: int = 0
    llm_passed: int = 0
    final_selected: int = 0
    failed_by_stage: dict[str, int] = Field(default_factory=dict)


class ValidationTaskResult(BaseModel):
    task_id: str
    run_id: str
    report_path: str
    selected_question_ids: list[str]
    failed_by_stage: dict[str, int]


class ValidatedQuestionRecord(BaseModel):
    question_id: str
    candidate: QuestionCandidate
    citation_validation: CitationValidationResult | None = None
    llm_validation: LLMValidationResult | None = None
    group_id: str | None = None
    duplicate_of: str | None = None
    final_weight: float | None = None
    final_status: str  # FinalStatus value
