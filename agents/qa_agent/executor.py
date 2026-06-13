"""Execution logic: round execution, state update, stop conditions."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from .state import GlobalState, ModeState, CandidateRecord, CandidateStatus
from .planner import ModeRoundPlan, RoundStrategy, mode_candidate_target, mode_initial_breadth_not_done
from .feedback import build_round_feedback, RoundFeedback
from .sampling import sample_chunks, raw_chunk_ids, _unit_combos, record_global_chunk_usage
from benchforge.utils.filter import LightweightFilter
from benchforge.utils.llm_tracer import LLMTracer

_question_filter = LightweightFilter()


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
        difficulty = normalize_difficulty(q.get("estimated_difficulty") or q.get("difficulty") or round_plan.difficulty)
        record = CandidateRecord(
            question_id=q.get("question_id") or str(uuid.uuid4()),
            question=q.get("question", ""),
            answer=q.get("answer", ""),
            topic=topic,
            difficulty=difficulty,
            status=CandidateStatus.ACCEPTED,
            source_round=round_plan.round_in_mode,
            source_strategy=round_plan.strategy.value,
            chunk_ids=q.get("chunk_ids", []),
            parent_question_id=parent_question_id,
            choices=q.get("choices") if q.get("question_mode") == "multiple_choice" else None,
       )
        mode_state.candidate_questions.append(record)
        new_records.append(record)
    for item in (filter_failures or []):
        record = CandidateRecord(
            question_id=str(uuid.uuid4()),
            question=item.get("question", ""),
            answer="",
            topic=topic,
            difficulty=round_plan.difficulty,
            status=CandidateStatus.REJECTED,
            source_round=round_plan.round_in_mode,
            source_strategy=round_plan.strategy.value,
            chunk_ids=[],
            reject_reason=item.get("reason", "filter_rejected"),
            parent_question_id=None,  # REJECTED 记录不继承父题
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


def mode_should_stop(
    mode_cfg: Any,
    mode_state: ModeState,
    global_state: GlobalState,
    blueprint: Any,
    config: Any,
) -> tuple[bool, str | None]:
    """Pure predicate — no side effects. Caller sets mode_state.stopped_reason."""
    initial_done = not mode_initial_breadth_not_done(mode_state, blueprint)
    target = mode_candidate_target(mode_cfg, config)

    if initial_done and mode_state.accepted_count >= target:
        return True, "candidate_pool_sufficient"

    if mode_state.round_in_mode > mode_cfg.max_rounds:
        return True, "max_rounds_reached"

    if mode_state.consecutive_empty_rounds >= config.runtime.max_consecutive_empty_rounds_per_mode:
        return True, "consecutive_empty_rounds_reached"

    if mode_state.failures_count >= config.runtime.max_failures_per_mode:
        return True, "mode_failure_limit_reached"

    return False, None


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
    """Run one topic's LLM generation; returns raw result without state mutation."""
    llm_call_id = None
    reserved_unit_combos: list[tuple[str, ...]] | None = None
    try:
        if reservation_lock is not None and reserved_combinations is not None:
            async with reservation_lock:
                blocked_combinations = global_state.used_chunk_combinations | reserved_combinations
                chunks, duplicate_combination = sample_chunks(
                    evidence_manager=evidence_manager,
                    topic=topic,
                    mode=round_plan.mode,
                    difficulty=round_plan.difficulty,
                    single_k=round_plan.single_k,
                    multi_k=round_plan.multi_k,
                    global_used_combinations=global_state.used_chunk_combinations,
                    global_chunk_usage_counts=global_state.chunk_usage_counts,
                    blocked_combinations=blocked_combinations,
                    round_num=mode_state.round_in_mode,
                )
                if not duplicate_combination:
                    reserved_unit_combos = _unit_combos(chunks)
                    for combo in reserved_unit_combos:
                        reserved_combinations.add(combo)
        else:
            chunks, duplicate_combination = sample_chunks(
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

        if duplicate_combination:
            return {
                "topic": topic,
                "success": True,
                "parsed_questions": [],
                "raw_count": 0,
                "filtered_count": 0,
                "filter_failures": [],
                "llm_call_id": None,
                "chunks": chunks,
                "duplicate_combination": True,
                "error": None,
            }

        evidence_pool = evidence_manager.evidence_pools.get(topic)
        batch = _make_batch(topic, round_plan, chunks, evidence_pool)
        document_summary = evidence_manager.get_document_summary(batch, evidence_pool) if evidence_pool else ""

        model_client = evidence_manager.model_client
        model_name = getattr(model_client, "model_name", "unknown")

        # 使用 Span 自动记录（优先），向后兼容 llm_trace_path
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
                raw_items, _, llm_call_id = await generator.generate(
                    batch=batch,
                    model_client=model_client,
                    evidence_pool=evidence_pool,
                    document_summary=document_summary,
                    language=blueprint.language,
                    llm_trace_path=str(Path("runs") / blueprint.task_id / blueprint.run_id / "llm_calls.jsonl"),
                )
        else:
            llm_trace_path = str(Path("runs") / blueprint.task_id / blueprint.run_id / "llm_calls.jsonl")
            raw_items, _, llm_call_id = await generator.generate(
                batch=batch,
                model_client=model_client,
                evidence_pool=evidence_pool,
                document_summary=document_summary,
                language=blueprint.language,
                llm_trace_path=llm_trace_path,
            )

        raw_questions = parse_questions(raw_items)
        accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)
        filter_failures = [
            {
                "question": item.get("question", ""),
                "reason": reason,
            }
            for item, reason in rejected_questions
        ]

        chunk_id_list = raw_chunk_ids(chunks)
        chunk_text_map: dict[str, str] = {}
        for u in chunks:
            if hasattr(u, "chunk_ids") and hasattr(u, "texts"):
                for cid, txt in zip(u.chunk_ids, u.texts):
                    chunk_text_map[cid] = txt
            elif hasattr(u, "chunk_id"):
                chunk_text_map[u.chunk_id] = getattr(u, "text", "")
        for q in accepted_questions:
            q["chunk_ids"] = chunk_id_list
            q["chunks"] = [chunk_text_map.get(cid, "") for cid in chunk_id_list]
            q["topic"] = topic
            q["llm_call_id"] = llm_call_id
            q["generation_round"] = round_plan.round_in_mode

        return {
            "topic": topic,
            "success": True,
            "parsed_questions": accepted_questions,
            "raw_count": len(raw_questions),
            "filtered_count": len(accepted_questions),
            "filter_failures": filter_failures,
            "llm_call_id": llm_call_id,
            "chunks": chunks,
            "duplicate_combination": duplicate_combination,
            "error": None,
        }

    except Exception as exc:
        if reserved_unit_combos is not None and reserved_combinations is not None:
            for combo in reserved_unit_combos:
                reserved_combinations.discard(combo)
        logger.warning(f"Topic {topic} failed in round {round_plan.round_in_mode}: {exc}")
        return {
            "topic": topic,
            "success": False,
            "parsed_questions": [],
            "chunks": [],
            "duplicate_combination": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "llm_call_id": llm_call_id,
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
    # EVOLVE_TO_HARDER: re-generate from accepted easy/medium as seed context
    if round_plan.strategy == RoundStrategy.EVOLVE_TO_HARDER:
        return await _execute_evolve_round(
            round_plan=round_plan, blueprint=blueprint, config=config,
            global_state=global_state, mode_state=mode_state,
            evidence_manager=evidence_manager, generator=generator,
            mode_cfg=mode_cfg, tracer=tracer,
        )

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

            if mode_cfg is not None:
                target = mode_candidate_target(mode_cfg, config)
                remaining_quota = target - mode_state.accepted_count
                if remaining_quota <= 0:
                    parsed_questions = []
                elif len(parsed_questions) > remaining_quota:
                    parsed_questions = parsed_questions[:remaining_quota]

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
                "filter_rejected": (
                    res.get("raw_count", len(parsed_questions)) > 0
                    and res.get("filtered_count", len(parsed_questions)) == 0
                ),
                "filter_failures": res.get("filter_failures", []),
                "llm_call_id": res.get("llm_call_id"),
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
            })

        if round_plan.strategy == RoundStrategy.INITIAL_BREADTH:
            mode_state.initial_coverage.add(topic)

    feedback = build_round_feedback(
        mode=round_plan.mode,
        round_id=round_plan.round_in_mode,
        strategy=round_plan.strategy.value,
        round_results=round_results,
        new_records=all_new_records,
    )
    return round_results, feedback


async def _execute_evolve_round(
    round_plan: ModeRoundPlan,
    blueprint: Any,
    config: Any,
    global_state: GlobalState,
    mode_state: ModeState,
    evidence_manager: Any,
    generator: Any,
    mode_cfg: Any,
    tracer: LLMTracer | None,
) -> tuple[list[dict], "RoundFeedback"]:
    """Re-generate harder questions from accepted easy/medium seed questions."""
    from .state import CandidateStatus as CS

    seeds = [
        q for q in mode_state.candidate_questions
        if q.status == CS.ACCEPTED and q.difficulty == "medium"
    ][: min(round_plan.evolve_source_count, mode_state.evolvable_surplus(
        "medium", mode_cfg.difficulty_distribution.get("medium", 0.3) if mode_cfg else 0.3
    ))]

    round_results = []
    all_new_records: list[CandidateRecord] = []

    for seed in seeds:
        topic = seed.topic
        evidence_pool = evidence_manager.evidence_pools.get(topic)
        if evidence_pool is None:
            continue
        try:
            chunks, duplicate_combination = sample_chunks(
                evidence_manager=evidence_manager,
                topic=topic,
                mode=round_plan.mode,
                difficulty="hard",
                single_k=max(1, round_plan.single_k),
                multi_k=max(1, round_plan.multi_k),
                global_used_combinations=global_state.used_chunk_combinations,
                global_chunk_usage_counts=global_state.chunk_usage_counts,
                blocked_combinations=global_state.used_chunk_combinations,
                round_num=mode_state.round_in_mode,
            )
            if duplicate_combination:
                round_results.append({
                    "topic": topic, "success": True, "generated_count": 0,
                    "chunks": [], "duplicate_combination": True, "error": None,
                })
                continue
            batch = _make_batch(topic, round_plan, chunks, evidence_pool)
            document_summary = (
                evidence_manager.get_document_summary(batch, evidence_pool)
                if evidence_pool else ""
            )
            model_client = evidence_manager.model_client
            model_name = getattr(model_client, "model_name", "unknown")
            llm_trace_path = str(Path("runs") / blueprint.task_id / blueprint.run_id / "llm_calls.jsonl")
            if tracer is not None:
                span = tracer.span(
                    model=model_name,
                    provider=getattr(model_client, "provider", ""),
                    tags={"topic": topic, "mode": round_plan.mode, "round": round_plan.round_in_mode,
                          "difficulty": "hard", "strategy": "evolve_to_harder"},
                )
                async with span:
                    raw_items, _, llm_call_id = await generator.generate(
                        batch=batch, model_client=model_client, evidence_pool=evidence_pool,
                        document_summary=document_summary, language=blueprint.language,
                        llm_trace_path=llm_trace_path,
                    )
            else:
                raw_items, _, llm_call_id = await generator.generate(
                    batch=batch, model_client=model_client, evidence_pool=evidence_pool,
                    document_summary=document_summary, language=blueprint.language,
                    llm_trace_path=llm_trace_path,
                )
            raw_questions = parse_questions(raw_items)
            accepted_questions, rejected_questions = _question_filter.filter_questions(raw_questions)
            filter_failures = [{"question": item.get("question", ""), "reason": r}
                               for item, r in rejected_questions]
            chunk_id_list = raw_chunk_ids(chunks)
            for q in accepted_questions:
                q["chunk_ids"] = chunk_id_list
                q["topic"] = topic
                q["llm_call_id"] = llm_call_id
                q["generation_round"] = round_plan.round_in_mode

            new_records = update_mode_state(
                mode_state=mode_state,
                topic=topic,
                round_plan=round_plan,
                parsed_questions=accepted_questions,
                filter_failures=filter_failures,
                parent_question_id=seed.question_id,
            )
            if any(r.status == CS.ACCEPTED and r.difficulty == "hard" for r in new_records):
                seed.status = CS.EVOLVED
            all_new_records.extend(new_records)
            record_global_chunk_usage(
                global_state=global_state, chunks=chunks,
                max_size=config.runtime.max_used_chunk_combinations,
            )
            round_results.append({
                "topic": topic, "success": True,
                "generated_count": len(accepted_questions),
                "raw_count": len(raw_questions),
                "filtered_count": len(accepted_questions),
                "filter_failures": filter_failures,
                "llm_call_id": llm_call_id,
                "chunks": chunk_id_list,
                "duplicate_combination": False, "error": None,
            })
        except Exception as exc:
            mode_state.failures_count += 1
            global_state.global_failures += 1
            logger.warning(f"Evolve failed for seed {seed.question_id}: {exc}")
            round_results.append({
                "topic": topic, "success": False, "generated_count": 0,
                "chunks": [], "duplicate_combination": False, "error": str(exc),
            })

    evolved_seed_count = sum(1 for s in seeds if s.status == CS.EVOLVED)
    feedback = build_round_feedback(
        mode=round_plan.mode, round_id=round_plan.round_in_mode,
        strategy=round_plan.strategy.value, round_results=round_results,
        new_records=all_new_records,
        evolved_seed_count=evolved_seed_count,
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
        remaining_count=round_plan.target_candidates_per_topic,
        single_chunk_ids=single_ids,
        multi_chunk_ids=multi_ids,
        prompt_template_id=(
            "mcq_generation_v1" if round_plan.mode == "multiple_choice" else "qa_generation_v1"
        ),
        requested_min_questions=round_plan.target_candidates_per_topic,
        requested_target_questions=round_plan.target_candidates_per_topic,
    )
