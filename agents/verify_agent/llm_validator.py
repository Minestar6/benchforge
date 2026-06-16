"""Stage 2：LLM 验证（asyncio 并发，复用 model_client.complete()）。"""

import asyncio
import time

from loguru import logger

from benchforge.config.config import load_prompt
from benchforge.models.base import BaseModelClient
from benchforge.utils.filter import parse_json_response
from benchforge.utils.llm_tracer import LLMTracer
from .config_loader import LLMValidationCfg
from .schema import LLMValidationResult, QuestionCandidate, ValidationBlueprintView


_PROMPT_KEYS = {
    "topic",
    "question_mode",
    "question_type",
    "question",
    "answer",
    "required_capability",
    "estimated_difficulty",
    "difficulty",
    "citations",
    "chunks",
    "blueprint_constraints",
}


def _normalize_difficulty_label(difficulty: str | int | None) -> str:
    if difficulty is None:
        return "unknown"
    if isinstance(difficulty, int):
        if difficulty <= 3:
            return "easy"
        if difficulty <= 6:
            return "medium"
        if difficulty <= 10:
            return "hard"
        return "unknown"
    value = str(difficulty).strip().lower()
    if value in ("easy", "medium", "hard"):
        return value
    try:
        return _normalize_difficulty_label(int(value))
    except ValueError:
        return "unknown"


def _render_prompt_template(template: str, values: dict[str, object]) -> str:
    if not template:
        return ""

    rendered = template
    for key in _PROMPT_KEYS:
        rendered = rendered.replace(f"{{{key}}}", str(values.get(key, "")))
    return rendered


def _format_question_for_judge(candidate: QuestionCandidate) -> str:
    question_text = candidate.question
    if candidate.question_mode != "multiple_choice":
        return question_text

    choices = candidate.choices or candidate.generation_metadata.get("choices") or candidate.generation_metadata.get("options")
    if not choices:
        return question_text

    if isinstance(choices, dict):
        options_text = "\n".join(f"{key}. {value}" for key, value in choices.items())
    else:
        options_text = "\n".join(str(choice) for choice in choices)
    return f"{question_text}\n{options_text}"


def _score_to_unit_interval(raw_score: float) -> float:
    bounded = max(1.0, min(5.0, float(raw_score)))
    return (bounded - 1.0) / 4.0


def _normalize_judge_response(parsed: dict) -> tuple[bool, float, dict[str, float], list[str], str, str | None]:
    if "dimensions" in parsed or "overall_score" in parsed or "passed" in parsed:
        passed_raw = bool(parsed.get("passed", False))
        overall_score = float(parsed.get("overall_score", 0.0))
        dimensions = {k: float(v) for k, v in parsed.get("dimensions", {}).items()}
        failed_reasons = list(parsed.get("failed_reasons", []))
        judge_summary = str(parsed.get("judge_summary", ""))
        suggested_difficulty = parsed.get("suggested_difficulty")
        if suggested_difficulty is not None:
            suggested_difficulty = str(suggested_difficulty).strip().lower()
            if suggested_difficulty not in ("easy", "medium", "hard"):
                suggested_difficulty = None
        return passed_raw, overall_score, dimensions, failed_reasons, judge_summary, suggested_difficulty

    required_dims = ("correctness", "answerability", "clarity", "difficulty_consistency")
    dimensions = {
        key: _score_to_unit_interval(parsed[key])
        for key in required_dims
        if key in parsed
    }
    missing = [key for key in required_dims if key not in dimensions]
    if missing:
        raise ValueError(f"missing_dimensions: {', '.join(missing)}")

    overall_score = sum(dimensions.values()) / len(dimensions)
    failed_reasons: list[str] = []
    judge_summary = str(parsed.get("reason", ""))
    suggested_difficulty = parsed.get("suggested_difficulty")
    if suggested_difficulty is not None:
        suggested_difficulty = str(suggested_difficulty).strip().lower()
        if suggested_difficulty not in ("easy", "medium", "hard"):
            suggested_difficulty = None
    return True, overall_score, dimensions, failed_reasons, judge_summary, suggested_difficulty


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
    question_text = _format_question_for_judge(candidate)

    template_values = {
        "topic": candidate.topic,
        "question_mode": candidate.question_mode,
        "question_type": candidate.question_type,
        "question": question_text,
        "answer": candidate.answer,
        "required_capability": candidate.required_capability,
        "estimated_difficulty": candidate.estimated_difficulty,
        "difficulty": _normalize_difficulty_label(candidate.estimated_difficulty),
        "citations": citations_text,
        "chunks": chunks_text,
        "blueprint_constraints": blueprint_constraints,
    }

    if user_template:
        user_content = _render_prompt_template(user_template, template_values)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": _render_prompt_template(system_prompt, template_values)})
        messages.append({"role": "user", "content": user_content})
        return messages

    if system_prompt:
        return [{"role": "user", "content": _render_prompt_template(system_prompt, template_values)}]

    user_content = (
        f"Topic: {candidate.topic}\n"
        f"Question: {question_text}\n"
        f"Answer: {candidate.answer}\n"
        f"Citations:\n{citations_text}\n"
        f"Source Chunks:\n{chunks_text}\n"
        f"Blueprint Constraints: {blueprint_constraints}\n\n"
        "Please evaluate this question and return JSON with fields: "
        "correctness (1-5), answerability (1-5), clarity (1-5), "
        "difficulty_consistency (1-5), suggested_difficulty (easy/medium/hard), reason (str)."
    )
    return [{"role": "user", "content": user_content}]


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

    # 使用 client 已解析的 model_name，避免将 registry 逻辑名直接发给 API
    resolved_model = getattr(model_client, "model_name", cfg.model)

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
                        model=resolved_model,
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
                        model=resolved_model,
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

                passed_raw, overall_score, dimensions, failed_reasons, judge_summary, suggested_difficulty = _normalize_judge_response(parsed)

                if overall_score < cfg.min_overall_score and "overall_score_too_low" not in failed_reasons:
                    failed_reasons.append("overall_score_too_low")

                passed = bool(passed_raw) and overall_score >= cfg.min_overall_score

                return LLMValidationResult(
                    question_id=qid,
                    passed=passed,
                    overall_score=overall_score,
                    dimensions=dimensions,
                    failed_reasons=failed_reasons,
                    judge_summary=judge_summary,
                    suggested_difficulty=suggested_difficulty,
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

    system_prompt = load_prompt(cfg.prompt_path)
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
