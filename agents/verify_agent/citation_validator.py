"""Stage 1：引用验证（基于 thefuzz 模糊匹配，复用 yourbench citation_score_filtering 思路）。"""

import re
from statistics import mean

from loguru import logger
from thefuzz import fuzz

from .config_loader import CitationCfg
from .schema import CitationValidationResult, QuestionCandidate


def _clean(text: str) -> str:
    """轻量清洗：strip + collapse whitespace。"""
    return re.sub(r'\s+', ' ', text.strip())


def _resolve_answer_text(candidate: QuestionCandidate) -> str:
    answer = candidate.answer or ""
    if candidate.question_mode != "multiple_choice":
        return answer

    choices = candidate.choices or candidate.generation_metadata.get("choices") or candidate.generation_metadata.get("options")
    if not choices:
        return answer

    normalized_answer = _clean(str(answer)).upper().rstrip(".")
    if isinstance(choices, dict):
        choice_text = choices.get(normalized_answer) or choices.get(normalized_answer.upper())
        return str(choice_text) if choice_text else answer

    label_index = ord(normalized_answer[:1]) - ord("A") if normalized_answer[:1].isalpha() else -1
    if 0 <= label_index < len(choices):
        return str(choices[label_index])
    return answer


def _score_single_citation(
    citation: str,
    chunks: list[str],
    answer: str,
) -> tuple[float, float]:
    """计算单条 citation 与 chunks / answer 的模糊匹配分数。"""
    c = _clean(citation)
    chunk_score = max(
        (fuzz.partial_ratio(c, _clean(ch)) / 100.0 for ch in chunks),
        default=0.0,
    )
    answer_score = fuzz.partial_ratio(c, _clean(answer)) / 100.0 if answer else 0.0
    return chunk_score, answer_score


def validate_citation(
    candidate: QuestionCandidate,
    cfg: CitationCfg,
) -> CitationValidationResult:
    """对单题执行引用验证，返回 CitationValidationResult。"""
    qid = candidate.question_id
    citations = candidate.citations
    chunks = candidate.chunks
    answer = _resolve_answer_text(candidate)

    # 无 citations → 直接 fail
    if not citations:
        return CitationValidationResult(
            question_id=qid,
            passed=False,
            answer_citation_score=0.0,
            chunk_citation_score=0.0,
            citation_score=0.0,
            citation_count=0,
            matched_citation_count=0,
            failed_reasons=["missing_citations"],
        )

    chunk_scores: list[float] = []
    answer_scores: list[float] = []
    matched_spans: list[dict] = []

    for i, citation in enumerate(citations):
        cs, ans = _score_single_citation(str(citation), chunks, answer)
        chunk_scores.append(cs)
        answer_scores.append(ans)
        if cs >= cfg.citation_match_threshold:
            matched_spans.append({"citation_idx": i, "citation": str(citation), "chunk_score": cs})

    chunk_citation_score = mean(chunk_scores) if chunk_scores else 0.0
    answer_citation_score = mean(answer_scores) if answer_scores else 0.0
    citation_score = cfg.alpha * chunk_citation_score + cfg.beta * answer_citation_score
    matched_citation_count = len(matched_spans)

    # 限制在 [0, 1]
    chunk_citation_score = min(1.0, max(0.0, chunk_citation_score))
    answer_citation_score = min(1.0, max(0.0, answer_citation_score))
    citation_score = min(1.0, max(0.0, citation_score))

    failed_reasons: list[str] = []
    if citation_score < cfg.min_citation_score:
        failed_reasons.append("citation_score_too_low")

    passed = len(failed_reasons) == 0

    logger.debug(
        f"[citation] {qid}: score={citation_score:.3f} "
        f"chunk={chunk_citation_score:.3f} ans={answer_citation_score:.3f} "
        f"matched={matched_citation_count}/{len(citations)} passed={passed}"
    )

    return CitationValidationResult(
        question_id=qid,
        passed=passed,
        answer_citation_score=answer_citation_score,
        chunk_citation_score=chunk_citation_score,
        citation_score=citation_score,
        citation_count=len(citations),
        matched_citation_count=matched_citation_count,
        failed_reasons=failed_reasons,
        matched_spans=matched_spans,
    )


def run_citation_validation(
    candidates: list[QuestionCandidate],
    cfg: CitationCfg,
) -> dict[str, CitationValidationResult]:
    """对候选题列表批量执行引用验证，返回 {question_id: CitationValidationResult}。"""
    if not cfg.enabled:
        # 若禁用，全部 pass
        return {
            c.question_id: CitationValidationResult(
                question_id=c.question_id,
                passed=True,
                answer_citation_score=1.0,
                chunk_citation_score=1.0,
                citation_score=1.0,
                citation_count=len(c.citations),
                matched_citation_count=len(c.citations),
                failed_reasons=[],
            )
            for c in candidates
        }

    results: dict[str, CitationValidationResult] = {}
    for c in candidates:
        results[c.question_id] = validate_citation(c, cfg)

    passed = sum(1 for r in results.values() if r.passed)
    logger.info(f"Citation validation: {passed}/{len(candidates)} passed")
    return results
