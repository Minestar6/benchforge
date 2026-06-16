"""Execution logic: round execution, state update, stop conditions."""

import asyncio
import math
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from .state import GlobalState, ModeState, CandidateRecord, CandidateStatus
from .planner import (
    ModeRoundPlan,
    RoundStrategy,
    mode_candidate_target,
    mode_initial_breadth_not_done,
    mode_max_candidate_target,
)
from .feedback import build_round_feedback, RoundFeedback
from .sampling import sample_chunks, raw_chunk_ids, _unit_combos, record_global_chunk_usage, pre_sample_all_units
from benchforge.utils.filter import LightweightFilter
from benchforge.utils.llm_tracer import LLMTracer

_question_filter = LightweightFilter()


def get_question_mode(q: dict, fallback_mode: str) -> str:
    return q.get("question_mode") or fallback_mode


def get_answer(q: dict) -> str:
    return q.get("answer") or q.get("correct_answer") or ""


def get_choices(q: dict) -> list | None:
    return q.get("choices") or q.get("options")


def normalize_item_for_record(q: dict, fallback_mode: str) -> dict:
    """Normalize common aliases without dropping original fields."""
    item = dict(q)

    if not item.get("question_mode"):
        item["question_mode"] = fallback_mode

    if "answer" not in item and "correct_answer" in item:
        item["answer"] = item.get("correct_answer")

    if "choices" not in item and "options" in item:
        item["choices"] = item.get("options")

    citations = item.get("citations", [])
    if citations is None:
        citations = []
    elif isinstance(citations, str):
        citations = [citations]
    item["citations"] = citations

    return item


def resolve_prompt_template_id(mode: str, difficulty: str) -> str:
    mode = str(mode or "").lower()
    difficulty = str(difficulty or "").lower()

    if mode == "qa":
        return "qa_hard_generation_v1" if difficulty == "hard" else "qa_generation_v1"

    if mode == "multiple_choice":
        return (
            "mcq_hard_generation_v1"
            if difficulty == "hard"
            else "mcq_generation_v1"
        )

    return "qa_generation_v1"


def normalize_difficulty(value: Any) -> str:
    if not value:
        return "medium"
    v = str(value).lower().strip()
    if v in ("easy", "simple", "low"):
        return "easy"
    if v in ("hard", "difficult", "advanced", "high"):
        return "hard"
    # numeric: 1-3 easy, 4-7 medium, 8-10 hard
    try:
        n = float(v)
        if n <= 3:
            return "easy"
        if n <= 7:
            return "medium"
        return "hard"
    except ValueError:
        return "medium"


def parse_questions(raw_output: Any) -> list[dict]:
    if isinstance(raw_output, list):
        return raw_output
    from benchforge.utils.filter import parse_llm_response
    try:
        return parse_llm_response(raw_output) or []
    except Exception:
        return []


def update_mode_state(
    mode_state: ModeState,
    topic: str,
    round_plan: ModeRoundPlan,
    parsed_questions: list[dict],
    filter_failures: list[dict] | None = None,
    parent_question_id: str | None = None,
) -> list[CandidateRecord]:
    """Append new CandidateRecords; return the list of newly added records."""
    new_records: list[CandidateRecord] = []
    for q in parsed_questions:
        raw_item = dict(q)
        q_norm = normalize_item_for_record(q, round_plan.mode)

        raw_estimated = (
            q_norm.get("estimated_difficulty")
            or q_norm.get("difficulty")
            or round_plan.difficulty
        )
        difficulty = normalize_difficulty(raw_estimated)
        question_mode = get_question_mode(q_norm, round_plan.mode)

        record = CandidateRecord(
            question_id=q_norm.get("question_id") or str(uuid.uuid4()),
            question=q_norm.get("question", ""),
            answer=get_answer(q_norm),
            topic=q_norm.get("topic") or topic,
            difficulty=difficulty,
            status=CandidateStatus.ACCEPTED,
            source_round=round_plan.round_in_mode,
            source_strategy=round_plan.strategy.value,
            chunk_ids=q_norm.get("chunk_ids", []),
            parent_question_id=parent_question_id,
            choices=get_choices(q_norm) if question_mode == "multiple_choice" else None,

            question_mode=question_mode,
            question_type=q_norm.get("question_type"),
            required_capability=q_norm.get("required_capability"),
            estimated_difficulty=q_norm.get("estimated_difficulty"),
            citations=q_norm.get("citations", []),
            thought_process=q_norm.get("thought_process"),

            llm_call_id=q_norm.get("llm_call_id"),
            trace_call_id=q_norm.get("trace_call_id"),
            chunks=q_norm.get("chunks", []),
            generation_round=q_norm.get("generation_round"),
            raw_item=raw_item,
       )
        mode_state.candidate_questions.append(record)
        new_records.append(record)
    for item in (filter_failures or []):
        raw = dict(item.get("raw_item") or item)
        q_norm = normalize_item_for_record(raw, round_plan.mode)

        raw_estimated = (
            q_norm.get("estimated_difficulty")
            or q_norm.get("difficulty")
            or round_plan.difficulty
        )
        question_mode = get_question_mode(q_norm, round_plan.mode)

        record = CandidateRecord(
            question_id=q_norm.get("question_id") or str(uuid.uuid4()),
            question=q_norm.get("question", item.get("question", "")),
            answer=get_answer(q_norm),
            topic=item.get("topic") or q_norm.get("topic") or topic,
            difficulty=normalize_difficulty(raw_estimated),
            status=CandidateStatus.REJECTED,
            source_round=round_plan.round_in_mode,
            source_strategy=round_plan.strategy.value,
            chunk_ids=item.get("chunk_ids") or q_norm.get("chunk_ids", []),
            reject_reason=item.get("reason", "filter_rejected"),
            parent_question_id=None,

            choices=get_choices(q_norm) if question_mode == "multiple_choice" else None,

            question_mode=question_mode,
            question_type=q_norm.get("question_type"),
            required_capability=q_norm.get("required_capability"),
            estimated_difficulty=q_norm.get("estimated_difficulty"),
            citations=q_norm.get("citations", []),
            thought_process=q_norm.get("thought_process"),

            llm_call_id=item.get("llm_call_id") or q_norm.get("llm_call_id"),
            trace_call_id=item.get("trace_call_id") or q_norm.get("trace_call_id"),
            chunks=item.get("chunks") or q_norm.get("chunks", []),
            generation_round=item.get("generation_round") or q_norm.get("generation_round"),
            raw_item=raw,
       )
        mode_state.candidate_questions.append(record)
        new_records.append(record)
    return new_records


def update_mode_trace(
    mode_state: ModeState,
    round_plan: ModeRoundPlan,
    round_results: list[dict],
) -> None:
    mode_state.trace.append({
        "mode": round_plan.mode,
        "round_in_mode": round_plan.round_in_mode,
        "strategy": round_plan.strategy,
        "difficulty": round_plan.difficulty,
        "topics": list(round_plan.topics),
        "single_k": round_plan.single_k,
        "multi_k": round_plan.multi_k,
        "target_candidates_per_topic": round_plan.target_candidates_per_topic,
        "results": round_results,
    })


def update_generation_yield_estimates(mode_state: ModeState, round_results: list[dict]) -> None:
    """用本轮真实产出更新后续轮次的 single/multi 产题率估计。"""
    total_single_units = sum(int(r.get("single_unit_count", 0)) for r in round_results if r.get("success"))
    total_multi_units = sum(int(r.get("multi_unit_count", 0)) for r in round_results if r.get("success"))
    total_single_generated = sum(int(r.get("single_generated_count", 0)) for r in round_results if r.get("success"))
    total_multi_generated = sum(int(r.get("multi_generated_count", 0)) for r in round_results if r.get("success"))

    if total_single_units > 0:
        observed_single = total_single_generated / total_single_units
        prev_total = mode_state.single_yield_samples
        prev_est = mode_state.single_yield_estimate or 0.0
        mode_state.single_yield_estimate = (
            observed_single
            if prev_total <= 0
            else ((prev_est * prev_total) + total_single_generated) / (prev_total + total_single_units)
        )
        mode_state.single_yield_samples = prev_total + total_single_units

    if total_multi_units > 0:
        observed_multi = total_multi_generated / total_multi_units
        prev_total = mode_state.multi_yield_samples
        prev_est = mode_state.multi_yield_estimate or 0.0
        mode_state.multi_yield_estimate = (
            observed_multi
            if prev_total <= 0
            else ((prev_est * prev_total) + total_multi_generated) / (prev_total + total_multi_units)
        )
        mode_state.multi_yield_samples = prev_total + total_multi_units


def _per_difficulty_targets(mode_cfg: Any, config: Any) -> dict[str, int]:
    """计算每个难度级别的候选池目标数量。"""
    total = mode_candidate_target(mode_cfg, config)
    targets: dict[str, int] = {}
    for diff, ratio in (mode_cfg.difficulty_distribution or {}).items():
        targets[diff] = math.ceil(total * ratio)
    return targets


def _check_per_difficulty_sufficient(mode_state: ModeState, targets: dict[str, int]) -> bool:
    """检查每个难度级别的已接受数量是否达标。"""
    accepted_by_diff = mode_state.get_difficulty_counts(CandidateStatus.ACCEPTED)
    for diff, target in targets.items():
        if accepted_by_diff.get(diff, 0) < target:
            return False
    return True


def mode_should_stop(
    mode_cfg: Any,
    mode_state: ModeState,
    global_state: GlobalState,
    blueprint: Any,
    config: Any,
) -> tuple[bool, str | None]:
    """Pure predicate — no side effects. Caller sets mode_state.stopped_reason."""
    exp_cfg = getattr(config, "experiment", None)
    breadth_disabled = (
        getattr(exp_cfg, "disable_initial_breadth", False)
        if exp_cfg is not None
        else False
    )
    initial_done = (
        not mode_initial_breadth_not_done(mode_state, blueprint)
        or breadth_disabled
        or not config.initial_breadth.enabled
    )
    min_target = mode_candidate_target(mode_cfg, config)
    max_target = mode_max_candidate_target(mode_cfg, config)

    if initial_done and mode_state.accepted_count >= max_target:
        return True, "max_candidate_pool_reached"

    if initial_done and mode_state.accepted_count >= min_target:
        diff_targets = _per_difficulty_targets(mode_cfg, config)
        if _check_per_difficulty_sufficient(mode_state, diff_targets):
            return True, "min_candidate_pool_sufficient"

    if mode_state.round_in_mode > mode_cfg.max_rounds:
        return True, "max_rounds_reached"

    if mode_state.consecutive_empty_rounds >= config.runtime.max_consecutive_empty_rounds_per_mode:
        return True, "consecutive_empty_rounds_reached"

    if mode_state.failures_count >= config.runtime.max_failures_per_mode:
        return True, "mode_failure_limit_reached"

    return False, None


async def _generate_one_batch(
    topic: str,
    round_plan: ModeRoundPlan,
    blueprint: Any,
    global_state: GlobalState,
    mode_state: ModeState,
    evidence_manager: Any,
    generator: Any,
    unit_type: str,
    pre_sampled_chunks: list[Any] | None = None,
    tracer: LLMTracer | None = None,
) -> dict:
    """Generate questions from exactly 1 evidence unit.

    Args:
        pre_sampled_chunks: 预采样好的 evidence unit 列表（跳过采样）。
                            None 时内部自行采样。
    """
    llm_call_id = None
    try:
        if pre_sampled_chunks is not None:
            chunks = pre_sampled_chunks
            duplicate_combination = False
        else:
            chunks, duplicate_combination = sample_chunks(
                evidence_manager=evidence_manager,
                topic=topic,
                mode=round_plan.mode,
                difficulty=round_plan.difficulty,
                unit_type=unit_type,
                global_used_combinations=global_state.used_chunk_combinations,
                global_chunk_usage_counts=global_state.chunk_usage_counts,
                round_num=mode_state.round_in_mode,
            )

        if duplicate_combination:
            return {
                "success": True,
                "parsed_questions": [],
                "raw_count": 0,
                "filtered_count": 0,
                "filter_failures": [],
                "llm_call_id": None,
                "trace_call_id": None,
                "chunks": chunks,
                "duplicate_combination": True,
                "error": None,
            }

        evidence_pool = evidence_manager.evidence_pools.get(topic)
        batch = _make_batch(topic, round_plan, chunks, evidence_pool)
        document_summary = evidence_manager.get_document_summary(batch, evidence_pool) if evidence_pool else ""

        model_client = evidence_manager.model_client
        model_name = getattr(model_client, "model_name", "unknown")

        if tracer is not None:
            span = tracer.span(
                model=model_name,
                provider=getattr(model_client, "provider", ""),
                tags={
                    "topic": topic,
                    "mode": round_plan.mode,
                    "round": round_plan.round_in_mode,
                    "difficulty": round_plan.difficulty,
                    "strategy": round_plan.strategy,
                },
            )
            async with span:
                raw_items, _, llm_call_id, _ = await generator.generate(
                    batch=batch,
                    model_client=model_client,
                    evidence_pool=evidence_pool,
                    document_summary=document_summary,
                    language=blueprint.language,
                    llm_trace_path=str(Path("runs") / blueprint.task_id / blueprint.run_id / "llm_calls.jsonl"),
                )
                trace_call_id = span.call_id
        else:
            trace_call_id = None
            llm_trace_path = str(Path("runs") / blueprint.task_id / blueprint.run_id / "llm_calls.jsonl")
            raw_items, _, llm_call_id, _ = await generator.generate(
                batch=batch,
                model_client=model_client,
                evidence_pool=evidence_pool,
                document_summary=document_summary,
                language=blueprint.language,
                llm_trace_path=llm_trace_path,
            )

        raw_questions = parse_questions(raw_items)
        accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)

        chunk_id_list = raw_chunk_ids(chunks)
        chunk_text_map: dict[str, str] = {}
        for u in chunks:
            if hasattr(u, "chunk_ids") and hasattr(u, "texts"):
                for cid, txt in zip(u.chunk_ids, u.texts):
                    chunk_text_map[cid] = txt
            elif hasattr(u, "chunk_id"):
                chunk_text_map[u.chunk_id] = getattr(u, "text", "")
        chunk_texts = [chunk_text_map.get(cid, "") for cid in chunk_id_list]

        filter_failures = [
            {
                "question": item.get("question", ""),
                "reason": reason,
                "raw_item": dict(item),
                "topic": topic,
                "chunk_ids": chunk_id_list,
                "chunks": chunk_texts,
                "llm_call_id": llm_call_id,
                "trace_call_id": trace_call_id,
                "generation_round": round_plan.round_in_mode,
            }
            for item, reason in rejected_questions
        ]

        for q in accepted_questions:
            q["chunk_ids"] = chunk_id_list
            q["chunks"] = chunk_texts
            q["topic"] = topic
            q["llm_call_id"] = llm_call_id
            q["trace_call_id"] = trace_call_id
            q["generation_round"] = round_plan.round_in_mode

        return {
            "success": True,
            "parsed_questions": accepted_questions,
            "raw_count": len(raw_questions),
            "filtered_count": len(accepted_questions),
            "filter_failures": filter_failures,
            "llm_call_id": llm_call_id,
            "trace_call_id": trace_call_id,
            "chunks": chunks,
            "duplicate_combination": duplicate_combination,
            "error": None,
        }

    except Exception as exc:
        logger.warning(f"Topic {topic} [{unit_type}] failed in round {round_plan.round_in_mode}: {exc}")
        return {
            "success": False,
            "parsed_questions": [],
            "chunks": [],
            "duplicate_combination": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "llm_call_id": llm_call_id,
            "trace_call_id": None,
        }


async def _generate_for_topic(
    topic: str,
    round_plan: ModeRoundPlan,
    blueprint: Any,
    global_state: GlobalState,
    mode_state: ModeState,
    evidence_manager: Any,
    generator: Any,
    reservation_lock: asyncio.Lock | None = None,
    reserved_combinations: set[tuple[str, ...]] | None = None,
    tracer: LLMTracer | None = None,
) -> dict:
    """Run one topic's generation: pre-sample all units, then make one LLM call per unit."""
    # ── Phase 1: pre-sample all evidence units under lock ──
    if reservation_lock is not None and reserved_combinations is not None:
        async with reservation_lock:
            blocked = global_state.used_chunk_combinations | reserved_combinations
            pre_sampled = pre_sample_all_units(
                evidence_manager=evidence_manager,
                topic=topic,
                mode=round_plan.mode,
                difficulty=round_plan.difficulty,
                single_k=round_plan.single_k,
                multi_k=round_plan.multi_k,
                global_used_combinations=global_state.used_chunk_combinations,
                global_chunk_usage_counts=global_state.chunk_usage_counts,
                blocked_combinations=blocked,
                round_num=mode_state.round_in_mode,
            )
            # Reserve all valid combos
            for chunks, _ut, dup in pre_sampled:
                if not dup:
                    for combo in _unit_combos(chunks):
                        reserved_combinations.add(combo)
    else:
        pre_sampled = pre_sample_all_units(
            evidence_manager=evidence_manager,
            topic=topic,
            mode=round_plan.mode,
            difficulty=round_plan.difficulty,
            single_k=round_plan.single_k,
            multi_k=round_plan.multi_k,
            global_used_combinations=global_state.used_chunk_combinations,
            global_chunk_usage_counts=global_state.chunk_usage_counts,
            round_num=mode_state.round_in_mode,
        )

    # ── Phase 2: one LLM call per pre-sampled unit ──
    all_parsed: list[dict] = []
    all_failures: list[dict] = []
    all_chunks: list[Any] = []
    total_raw = 0
    total_filtered = 0
    single_unit_count = 0
    multi_unit_count = 0
    single_generated_count = 0
    multi_generated_count = 0
    any_dup = False
    last_llm_id = None
    last_trace_id = None

    for chunks, _unit_type, dup in pre_sampled:
        if dup or not chunks:
            any_dup = True
            continue

        result = await _generate_one_batch(
            topic=topic,
            round_plan=round_plan,
            blueprint=blueprint,
            global_state=global_state,
            mode_state=mode_state,
            evidence_manager=evidence_manager,
            generator=generator,
            unit_type=_unit_type,
            pre_sampled_chunks=chunks,
            tracer=tracer,
        )

        if not result["success"]:
            return {
                "topic": topic,
                "success": False,
                "parsed_questions": all_parsed,
                "raw_count": total_raw,
                "filtered_count": total_filtered,
                "filter_failures": all_failures,
                "llm_call_id": result.get("llm_call_id") or last_llm_id,
                "trace_call_id": result.get("trace_call_id") or last_trace_id,
                "chunks": all_chunks,
                "duplicate_combination": any_dup,
                "error": result.get("error"),
                "error_type": result.get("error_type"),
            }

        all_parsed.extend(result.get("parsed_questions", []))
        all_failures.extend(result.get("filter_failures", []))
        all_chunks.extend(result.get("chunks", []))
        total_raw += result.get("raw_count", 0)
        total_filtered += result.get("filtered_count", 0)
        generated_count = len(result.get("parsed_questions", []))
        if _unit_type == "single":
            single_unit_count += 1
            single_generated_count += generated_count
        else:
            multi_unit_count += 1
            multi_generated_count += generated_count
        if result.get("duplicate_combination"):
            any_dup = True
        if result.get("llm_call_id"):
            last_llm_id = result["llm_call_id"]
        if result.get("trace_call_id"):
            last_trace_id = result["trace_call_id"]

    return {
        "topic": topic,
        "success": True,
        "parsed_questions": all_parsed,
        "raw_count": total_raw,
        "filtered_count": total_filtered,
        "single_unit_count": single_unit_count,
        "multi_unit_count": multi_unit_count,
        "single_generated_count": single_generated_count,
        "multi_generated_count": multi_generated_count,
        "filter_failures": all_failures,
        "llm_call_id": last_llm_id,
        "trace_call_id": last_trace_id,
        "chunks": all_chunks,
        "duplicate_combination": any_dup,
        "error": None,
    }


async def execute_mode_round_plan(
    round_plan: ModeRoundPlan,
    blueprint: Any,
    config: Any,
    global_state: GlobalState,
    mode_state: ModeState,
    evidence_manager: Any,
    generator: Any,
    mode_cfg: Any = None,
    tracer: LLMTracer | None = None,
) -> tuple[list[dict], RoundFeedback]:
    # EXPAND_EVIDENCE: just mark topics as needing more evidence (no LLM call)
    if round_plan.strategy == RoundStrategy.EXPAND_EVIDENCE:
        for topic in round_plan.expand_topics:
            try:
                evidence_pool = evidence_manager.evidence_pools.get(topic)
                await evidence_manager.expand_retrieval(
                    topic=topic,
                    queries=[topic],
                    language=blueprint.language,
                    run_id=blueprint.run_id,
                    evidence_pool=evidence_pool,
                )
            except Exception as exc:
                logger.warning(f"expand_retrieval failed for {topic}: {exc}")
        empty_results = [{"topic": t, "success": True, "generated_count": 0,
                          "chunks": [], "duplicate_combination": False, "error": None}
                         for t in round_plan.expand_topics]
        feedback = build_round_feedback(
            mode=round_plan.mode, round_id=round_plan.round_in_mode,
            strategy=round_plan.strategy.value, round_results=empty_results, new_records=[],
        )
        return empty_results, feedback

    reservation_lock = asyncio.Lock()
    reserved_combinations: set[tuple[str, ...]] = set()
    topic_results = await asyncio.gather(*[
        _generate_for_topic(
            topic=topic,
            round_plan=round_plan,
            blueprint=blueprint,
            global_state=global_state,
            mode_state=mode_state,
            evidence_manager=evidence_manager,
            generator=generator,
            reservation_lock=reservation_lock,
            reserved_combinations=reserved_combinations,
            tracer=tracer,
        )
        for topic in round_plan.topics
    ])

    round_results = []
    all_new_records: list[CandidateRecord] = []

    for res in topic_results:
        topic = res["topic"]
        if res["success"]:
            parsed_questions = res["parsed_questions"]
            chunks = res["chunks"]

            new_records = update_mode_state(
                mode_state=mode_state,
                topic=topic,
                round_plan=round_plan,
                parsed_questions=parsed_questions,
                filter_failures=res.get("filter_failures", []),
            )
            all_new_records.extend(new_records)
            record_global_chunk_usage(
                global_state=global_state,
                chunks=chunks,
                max_size=config.runtime.max_used_chunk_combinations,
            )
            round_results.append({
                "topic": topic,
                "success": True,
                "generated_count": len(parsed_questions),
                "raw_count": res.get("raw_count", len(parsed_questions)),
                "filtered_count": res.get("filtered_count", len(parsed_questions)),
                "single_unit_count": res.get("single_unit_count", 0),
                "multi_unit_count": res.get("multi_unit_count", 0),
                "single_generated_count": res.get("single_generated_count", 0),
                "multi_generated_count": res.get("multi_generated_count", 0),
                "filter_rejected": (
                    res.get("raw_count", len(parsed_questions)) > 0
                    and res.get("filtered_count", len(parsed_questions)) == 0
                ),
                "filter_failures": res.get("filter_failures", []),
                "llm_call_id": res.get("llm_call_id"),
                "trace_call_id": res.get("trace_call_id"),
                "chunks": raw_chunk_ids(chunks),
                "duplicate_combination": res["duplicate_combination"],
                "error": None,
            })
        else:
            mode_state.failures_count += 1
            global_state.global_failures += 1
            mode_state.failures.append({
                "mode": round_plan.mode,
                "round_in_mode": round_plan.round_in_mode,
                "topic": topic,
                "difficulty": round_plan.difficulty,
                "error": res["error"],
                "error_type": res.get("error_type", ""),
                "llm_call_id": res.get("llm_call_id"),
                "trace_call_id": res.get("trace_call_id"),
            })
            round_results.append({
                "topic": topic,
                "success": False,
                "generated_count": 0,
                "chunks": [],
                "duplicate_combination": False,
                "error": res["error"],
                "error_type": res.get("error_type", ""),
                "llm_call_id": res.get("llm_call_id"),
                "trace_call_id": res.get("trace_call_id"),
            })

        if round_plan.strategy == RoundStrategy.INITIAL_BREADTH:
            mode_state.initial_coverage.add(topic)

    update_generation_yield_estimates(mode_state, round_results)

    feedback = build_round_feedback(
        mode=round_plan.mode,
        round_id=round_plan.round_in_mode,
        strategy=round_plan.strategy.value,
        round_results=round_results,
        new_records=all_new_records,
    )
    return round_results, feedback
def _make_batch(topic: str, round_plan: ModeRoundPlan, chunks: list, evidence_pool: Any) -> Any:
    """Build a GenerationBatch from sampled chunks."""
    from benchforge.schemas import GenerationBatch

    single_ids = [u.chunk_id for u in chunks if hasattr(u, "chunk_id") and not hasattr(u, "unit_id")]
    multi_ids = [u.unit_id for u in chunks if hasattr(u, "unit_id")]

    return GenerationBatch(
        topic=topic,
        target_mode=round_plan.mode,
        target_difficulty=round_plan.difficulty,
        single_chunk_ids=single_ids,
        multi_chunk_ids=multi_ids,
        prompt_template_id=resolve_prompt_template_id(
            round_plan.mode,
            round_plan.difficulty,
        ),
    )
