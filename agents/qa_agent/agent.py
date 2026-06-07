"""Mode-Staged Generation Agent — main entry point."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.llm_tracer import LLMTracer
from .state import GlobalState, ModeState
from .planner import build_mode_round_plan
from .executor import (
    execute_mode_round_plan,
    mode_should_stop,
    update_mode_trace,
)
from .storage import save_mode_outputs, save_global_outputs, save_generation_report, save_shared_state


async def run_mode_generation(
    mode: str,
    mode_cfg: Any,
    blueprint: Any,
    config: Any,
    global_state: GlobalState,
    mode_state: ModeState,
    evidence_manager: Any,
    generator: Any,
    tracer: LLMTracer | None = None,
) -> None:
    while True:
        should_stop, reason = mode_should_stop(
            mode_cfg=mode_cfg,
            mode_state=mode_state,
            global_state=global_state,
            blueprint=blueprint,
            config=config,
        )
        if should_stop:
            mode_state.stopped_reason = reason
            logger.info(f"Mode {mode} stopped: {reason}")
            break

        round_plan = build_mode_round_plan(
            mode=mode,
            mode_cfg=mode_cfg,
            blueprint=blueprint,
            config=config,
            mode_state=mode_state,
        )

        if not round_plan.topics:
            mode_state.consecutive_empty_rounds += 1
            mode_state.round_in_mode += 1
            continue

        round_results = await execute_mode_round_plan(
            round_plan=round_plan,
            blueprint=blueprint,
            config=config,
            global_state=global_state,
            mode_state=mode_state,
            evidence_manager=evidence_manager,
            generator=generator,
            mode_cfg=mode_cfg,
            tracer=tracer,
        )

        total_generated = sum(r.get("generated_count", 0) for r in round_results)
        if total_generated == 0:
            mode_state.consecutive_empty_rounds += 1
        else:
            mode_state.consecutive_empty_rounds = 0

        update_mode_trace(mode_state, round_plan, round_results)
        mode_state.round_in_mode += 1

        logger.info(
            f"Mode={mode} round={round_plan.round_in_mode} "
            f"strategy={round_plan.strategy} generated={total_generated} "
            f"total_candidates={len(mode_state.candidate_questions)}"
        )


async def run_generation_agent(
    blueprint: Any,
    config: Any,
    evidence_manager: Any,
    generator: Any,
    tracer: LLMTracer | None = None,
) -> dict:
    """Main entry point for the mode-staged generation agent.

    Args:
        blueprint: Blueprint object with task_id, run_id, topics, modes, language.
        config: Config object matching the blueprint YAML schema.
        evidence_manager: EvidenceManager instance (must have evidence_pools populated).
        generator: Generator instance.
        tracer: Optional LLMTracer for automatic call recording.

    Returns:
        generation_report dict.
    """
    global_state = GlobalState()
    mode_states: dict[str, ModeState] = {}

    # 若未传入 tracer，创建默认 tracer（向后兼容）
    if tracer is None:
        from benchforge.utils.run_context import RunContext
        run_ctx = RunContext(task_id=blueprint.task_id, run_id=blueprint.run_id)
        tracer = run_ctx.create_tracer(agent="qa_agent", stage="generation")

    evidence_store = ArtifactStore(str(Path("runs") / blueprint.task_id / blueprint.run_id / "evidence"))

    all_single_units: dict[str, list] = {}
    all_multi_units: dict[str, list] = {}

    logger.info(f"Preparing evidence for {len(blueprint.topics)} topics (parallel)")

    async def _prepare_topic(topic: str):
        return topic, await evidence_manager.prepare_evidence(topic, blueprint)

    topic_evidence = await asyncio.gather(*[_prepare_topic(t) for t in blueprint.topics])

    chunked_rows_all: list[dict] = []
    for topic, (chunks, evidence_pool) in topic_evidence:
        evidence_manager.evidence_pools[topic] = evidence_pool

        chunks_by_doc: dict[str, list] = {}
        for chunk in chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_chunks_sorted = sorted(doc_chunks, key=lambda c: c.chunk_index)
            source_doc = evidence_manager.documents.get(doc_id)
            chunked_rows_all.append({
                "document_id": doc_id,
                "topic": topic,
                "document_title": source_doc.title if source_doc else "",
                "document_url": source_doc.url if source_doc else "",
                "document_text": source_doc.content if source_doc else "",
                "document_summary": evidence_manager.document_summaries.get(doc_id, ""),
                "chunks": [
                    {"chunk_id": c.chunk_id, "chunk_text": c.text}
                    for c in doc_chunks_sorted
                ],
            })

        if evidence_pool:
            all_single_units[topic] = [
                {
                    "chunk_id": u.chunk_id,
                    "document_id": u.document_id,
                    "text": u.text if hasattr(u, "text") else "",
                    "qa_score": u.qa_score,
                    "mcq_score": u.mcq_score,
                    "hard_score": u.hard_score,
                }
                for u in evidence_pool.single_chunks
            ]
            all_multi_units[topic] = [
                {
                    "unit_id": u.unit_id,
                    "chunk_ids": list(u.chunk_ids) if hasattr(u, "chunk_ids") else [],
                    "qa_score": u.qa_score,
                    "mcq_score": u.mcq_score,
                    "hard_score": u.hard_score,
                }
                for u in evidence_pool.multi_chunks
            ]

    if chunked_rows_all:
        evidence_store.append_jsonl("chunked.jsonl", chunked_rows_all)
    evidence_store.save_json("single_units.json", all_single_units)
    evidence_store.save_json("multi_units.json", all_multi_units)

    for mode, mode_cfg in blueprint.modes.items():
        logger.info(f"Starting mode: {mode}")
        mode_state = ModeState(mode=mode)
        mode_states[mode] = mode_state

        await run_mode_generation(
            mode=mode,
            mode_cfg=mode_cfg,
            blueprint=blueprint,
            config=config,
            global_state=global_state,
            mode_state=mode_state,
            evidence_manager=evidence_manager,
            generator=generator,
            tracer=tracer,
        )

        save_mode_outputs(blueprint.task_id, blueprint.run_id, mode, mode_state)
        logger.info(
            f"Mode {mode} complete: {len(mode_state.candidate_questions)} candidates, "
            f"stopped_reason={mode_state.stopped_reason}"
        )

    save_global_outputs(blueprint.task_id, blueprint.run_id, global_state)
    report = save_generation_report(
        task_id=blueprint.task_id,
        run_id=blueprint.run_id,
        global_state=global_state,
        mode_states=mode_states,
        mode_cfgs=dict(blueprint.modes),
        config=config,
    )
    save_shared_state(blueprint)

    logger.info(f"Generation complete. Output: runs/{blueprint.task_id}/{blueprint.run_id}/")
    return report
