from dataclasses import dataclass
from pathlib import Path

from agents.verify_agent.schema import (
    FinalStatus, ValidatedQuestionRecord, ValidationTaskResult, QuestionCandidate,
)


@dataclass
class MappedResult:
    question_id: str
    topic: str
    difficulty: str
    accepted: bool
    reject_reason: str | None = None
    question: str = ""
    answer: str = ""
    chunk_ids: list[str] | None = None
    source_round: int | None = None
    source_action_type: str = ""
    llm_call_id: str | None = None


def map_from_task_result(
    result: ValidationTaskResult,
    candidates: list[QuestionCandidate],
    validation_dir: Path | None = None,
) -> list[MappedResult]:
    selected_ids = set(result.selected_question_ids)
    candidate_map = {c.question_id: c for c in candidates}

    detailed_reasons: dict[str, str] = {}
    if validation_dir and (validation_dir / "validated_questions.jsonl").exists():
        records = _load_round_records(validation_dir, set(candidate_map.keys()))
        for rec in records:
            if rec.question_id not in selected_ids:
                detailed_reasons[rec.question_id] = _classify_rejection(rec)

    results = []
    for qid, cand in candidate_map.items():
        accepted = qid in selected_ids
        reject_reason = None if accepted else detailed_reasons.get(qid, "unknown")
        results.append(MappedResult(
            question_id=qid,
            topic=cand.topic,
            difficulty=str(cand.estimated_difficulty or "medium"),
            accepted=accepted,
            reject_reason=reject_reason,
            question=cand.question,
            answer=cand.answer,
            chunk_ids=list(cand.chunk_ids),
            source_round=cand.generation_metadata.get("round_in_mode"),
            source_action_type=str(cand.generation_metadata.get("action_type", "")),
            llm_call_id=cand.generation_metadata.get("llm_call_id"),
        ))
    return results


def _load_round_records(validation_dir: Path, question_ids: set[str]) -> list[ValidatedQuestionRecord]:
    jsonl_path = validation_dir / "validated_questions.jsonl"
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = ValidatedQuestionRecord.model_validate_json(line)
            if rec.question_id in question_ids:
                records.append(rec)
    return records


def _classify_rejection(rec: ValidatedQuestionRecord) -> str:
    status = rec.final_status

    if status == FinalStatus.rejected_citation.value:
        if rec.citation_validation and rec.citation_validation.answer_citation_score < 0.3:
            return "answer_not_grounded"
        return "evidence_insufficient"

    if status == FinalStatus.rejected_llm.value:
        if rec.llm_validation:
            dims = rec.llm_validation.dimensions
            reasons = rec.llm_validation.failed_reasons
            if dims.get("difficulty", 1.0) < 0.4:
                return "too_easy"
            if "not_multihop" in reasons or "single_source" in reasons:
                return "not_multihop"
        return "format_error"

    if status == FinalStatus.duplicate.value:
        return "duplicate"

    if status == FinalStatus.overquota.value:
        return "overquota"

    if status == FinalStatus.reserve.value:
        return "reserve"

    return "format_error"
