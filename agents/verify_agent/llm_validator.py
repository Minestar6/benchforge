"""Stage 2：LLM 验证（asyncio 并发，复用 model_client.complete()）。"""

import asyncio
import time
from pathlib import Path

from loguru import logger

from benchforge.config.config import load_prompt
from benchforge.models.base import BaseModelClient
from benchforge.utils.filter import parse_json_response
from benchforge.utils.llm_tracer import LLMTracer
from .config_loader import LLMValidationCfg
from .schema import LLMValidationResult, QuestionCandidate, ValidationBlueprintView


def _build_messages(
    candidate: QuestionCandidate,
    blueprint: ValidationBlueprintView,
    system_prompt: str,
    user_template: str,
) -> list[dict[str, str]]:
    mode_cfg = blueprint.modes.get(candidate.question_mode)
    blueprint_constraints = ""
    if mode_cfg:
        dist = mode_cfg.difficulty_distribution
        blueprint_constraints = (
            f"Mode: {candidate.question_mode}, "
            f"target count: {mode_cfg.count}, "
            f"difficulty distribution: {dist}"
        )

    citations_text = "\n".join(f"- {c}" for c in candidate.citations) if candidate.citations else "(none)"
    chunks_text = "\n\n".join(candidate.chunks[:3]) if candidate.chunks else "(none)"

    user_content = user_template.format(
        topic=candidate.topic,
        question_mode=candidate.question_mode,
        question_type=candidate.question_type,
        question=candidate.question,
        answer=candidate.answer,
        required_capability=candidate.required_capability,
        estimated_difficulty=candidate.estimated_difficulty,
        citations=citations_text,
        chunks=chunks_text,
        blueprint_constraints=blueprint_constraints,
    ) if user_template else (
        f"Topic: {candidate.topic}\n"
        f"Question: {candidate.question}\n"
        f"Answer: {candidate.answer}\n"
        f"Citations:\n{citations_text}\n"
        f"Source Chunks:\n{chunks_text}\n"
        f"Blueprint Constraints: {blueprint_constraints}\n\n"
        "Please evaluate this question and return JSON with fields: "
        "passed (bool), overall_score (0-1), dimensions (dict of clarity/answerability/"
        "faithfulness/difficulty_alignment/mode_alignment scores 0-1), "
        "failed_reasons (list[str]), judge_summary (str)."
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def _check_hard_floor(dimensions: dict[str, float], hard_floor: dict[str, float]) -> list[str]:
    failures = []
    for dim, floor in hard_floor.items():
        score = dimensions.get(dim, 1.0)
        if score < floor:
            failures.append(f"{dim}_too_low")
    return failures


async def _validate_one(
    candidate: QuestionCandidate,
    blueprint: ValidationBlueprintView,
    cfg: LLMValidationCfg,
    model_client: BaseModelClient,
    llm_trace_path: str,
    system_prompt: str,
    user_template: str,
    tracer: LLMTracer | None,
    sem: asyncio.Semaphore,
) -> LLMValidationResult:
    qid = candidate.question_id
    messages = _build_messages(candidate, blueprint, system_prompt, user_template)

    attempt_count = 0
    last_error: str | None = None

    async with sem:
        for attempt in range(1, cfg.max_retries + 2):
            attempt_count = attempt
            t0 = time.monotonic()
            try:
                # 使用 Span 自动记录（优先），向后兼容 llm_trace_path
                if tracer is not None:
                    span_ctx = tracer.span(
                        model=cfg.model,
                        provider=getattr(model_client, "provider", ""),
                        tags={
                            "question_id": qid,
                            "question_mode": candidate.question_mode or "",
                            "attempt": attempt,
                        },
                    )
                    span_ctx.__enter__()
                else:
                    span_ctx = None

                try:
                    resp = await model_client.complete(
                        model=cfg.model,
                        messages=messages,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                        llm_trace_path=llm_trace_path,
                    )
                finally:
                    if span_ctx is not None:
                        span_ctx.__exit__(None, None, None)

                latency_ms = int((time.monotonic() - t0) * 1000)
                llm_call_id = resp.get("llm_call_id")
                input_tokens = resp.get("input_tokens", 0)
                output_tokens = resp.get("output_tokens", 0)
                parsed = parse_json_response(resp.get("text", ""))

                passed_raw = parsed.get("passed", False)
                overall_score = float(parsed.get("overall_score", 0.0))
                dimensions = {k: float(v) for k, v in parsed.get("dimensions", {}).items()}
                failed_reasons = list(parsed.get("failed_reasons", []))
                judge_summary = str(parsed.get("judge_summary", ""))

                # 应用 hard_floor
                floor_failures = _check_hard_floor(dimensions, cfg.hard_floor)
                failed_reasons.extend(floor_failures)

                if overall_score < cfg.min_overall_score and "overall_score_too_low" not in failed_reasons:
                    failed_reasons.append("overall_score_too_low")

                passed = bool(passed_raw) and len(floor_failures) == 0 and overall_score >= cfg.min_overall_score

                return LLMValidationResult(
                    question_id=qid,
                    passed=passed,
                    overall_score=overall_score,
                    dimensions=dimensions,
                    failed_reasons=failed_reasons,
                    judge_summary=judge_summary,
                    llm_call_id=llm_call_id,
                    attempt_count=attempt_count,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=None,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[llm_validation] {qid} attempt {attempt} failed: {e}")
                if attempt <= cfg.max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))  # 指数退避

    # 全部重试耗尽
    logger.error(f"[llm_validation] {qid} exhausted retries, marking validator_error")
    return LLMValidationResult(
        question_id=qid,
        passed=False,
        overall_score=0.0,
        dimensions={},
        failed_reasons=["validator_error"],
        judge_summary="",
        llm_call_id=None,
        attempt_count=attempt_count,
        latency_ms=None,
        input_tokens=0,
        output_tokens=0,
        error=last_error,
    )


async def run_llm_validation(
    candidates: list[QuestionCandidate],
    blueprint: ValidationBlueprintView,
    cfg: LLMValidationCfg,
    model_client: BaseModelClient,
    llm_trace_path: str,
    tracer: LLMTracer | None = None,
) -> dict[str, LLMValidationResult]:
    """对候选题并发执行 LLM 验证，返回 {question_id: LLMValidationResult}。"""
    if not cfg.enabled:
        return {
            c.question_id: LLMValidationResult(
                question_id=c.question_id,
                passed=True,
                overall_score=1.0,
                dimensions={},
                failed_reasons=[],
                judge_summary="llm_validation disabled",
                input_tokens=0,
                output_tokens=0,
            )
            for c in candidates
        }

    system_prompt = load_prompt(cfg.prompt_system)
    user_template = load_prompt(cfg.prompt_user)

    sem = asyncio.Semaphore(cfg.max_concurrency)
    tasks = [
        _validate_one(c, blueprint, cfg, model_client, llm_trace_path, system_prompt, user_template, tracer, sem)
        for c in candidates
    ]
    results_list = await asyncio.gather(*tasks)
    results = {r.question_id: r for r in results_list}

    passed = sum(1 for r in results.values() if r.passed)
    errors = sum(1 for r in results.values() if r.error)
    total_input = sum(r.input_tokens for r in results.values())
    total_output = sum(r.output_tokens for r in results.values())
    logger.info(f"LLM validation: {passed}/{len(candidates)} passed, {errors} errors, "
                 f"tokens: {total_input} in / {total_output} out")
    return results
