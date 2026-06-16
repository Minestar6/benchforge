"""Mode-Staged Generation Agent — main entry point."""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.llm_tracer import LLMTracer
from .state import GlobalState, ModeState
from .planner import build_mode_round_plan, mode_candidate_target, ModeRoundPlan, RoundStrategy
from .executor import execute_mode_round_plan, mode_should_stop, update_mode_trace
from .storage import (
    save_mode_outputs, save_global_outputs, save_generation_report, save_shared_state,
    append_round_plan, append_round_feedback, save_mode_metrics,
)


# ============================================================================
# Model resolution — pluggable for testing
# ============================================================================

def _resolve_model_client(model_name: str, registry_path: str | Path) -> Any:
    """Resolve model client from registry by name."""
    from benchforge.models.loader import ModelLoader
    from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry
    registry = load_model_registry(registry_path)
    model_cfg = registry[model_name]
    return ModelLoader.load_model(model_cfg)


# Module-level hook: tests import agent.py and replace this with a lambda.
_resolve_model_client_fn = _resolve_model_client


# ============================================================================
# Public API
# ============================================================================

async def run_generation_agent(
    blueprint: Any,
    config_path: str | Path,
    *,
    registry_path: str | Path | None = None,
    tracer: LLMTracer | None = None,
) -> dict:
    """End-to-end entry point for the mode-staged generation agent.

    EvidenceManager, Generator, LLMTracer, and model resolution are handled
    automatically from qa_agent.yaml.  The only required inputs are a Blueprint
    and the path to qa_agent.yaml.

    Args:
        blueprint: Blueprint with task_id, run_id, topics, modes, language.
        config_path: Path to qa_agent.yaml.
        registry_path: Path to model_registry.yaml.
            Defaults to <project>/benchforge/config/model_registry.yaml.
        tracer: Optional LLMTracer.  Created automatically when None.

    Returns:
        generation_report dict.
    """
    from .config_loader import load_qa_agent_config
    from .evidence_manager import EvidenceManager
    from .generator import Generator

    config_path = Path(config_path)
    # multi_chunk_cfg 由 config_loader 返回，用于 YourBench 风格 multi_units 生成
    agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg = (
        load_qa_agent_config(config_path)
    )

    if registry_path is None:
        from benchforge.utils.paths import get_project_root
        registry_path = get_project_root() / "config" / "model_registry.yaml"

    model_client = _resolve_model_client_fn(model_ref.name, registry_path)

    # 将 model_ref 参数注入 model_client，确保后续调用（generator / evidence_manager）一致读取
    # 注意：model_name 由 ModelLoader 从 model_registry 设置（API 实际模型名），
    # 不要用 model_ref.name（逻辑名）覆盖，否则会导致 API 404
    setattr(model_client, "model_registry_name", model_ref.name)  # 逻辑名（仅用于日志/追踪）
    setattr(model_client, "temperature", model_ref.temperature)
    setattr(model_client, "max_tokens", max(model_ref.max_tokens, 1))  # 确保 > 0
    setattr(model_client, "max_retries", model_ref.max_retries)

    # Thin config wrapper matching the interface EvidenceManager expects
    class _EvidenceConfig:
        __slots__ = ("retrieval", "chunking", "summarization_chunking", "multi_chunk")
        def __init__(self, retrieval, chunking, summarization_chunking, multi_chunk):
            self.retrieval = retrieval
            self.chunking = chunking
            self.summarization_chunking = summarization_chunking
            self.multi_chunk = multi_chunk
        def get_resolved_output_path(self) -> Path:
            return Path("runs") / blueprint.task_id / blueprint.run_id

    evidence_config = _EvidenceConfig(retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg)

    return await _run_generation_agent_impl(
        blueprint=blueprint,
        config=agent_config,
        evidence_manager=EvidenceManager(evidence_config, model_client),
        generator=Generator(),
        tracer=tracer,
    )


# ============================================================================
# Internal implementation
# ============================================================================

async def _run_generation_agent_impl(
    blueprint: Any,
    config: Any,
    evidence_manager: Any,
    generator: Any,
    tracer: LLMTracer | None = None,
) -> dict:
    """Internal implementation used by both the public API and advanced callers
    (e.g. planner orchestrator with custom components).
    """
    global_state = GlobalState()
    mode_states: dict[str, ModeState] = {}

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
                "source": "initial_retrieval",
                "chunks": [
                    {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "chunk_text": c.text}
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
        # ? topic ??? {topic: [doc_records]} ??
        chunked_by_topic: dict[str, list] = {}
        for row in chunked_rows_all:
            topic = row.get("topic", "unknown")
            chunked_by_topic.setdefault(topic, []).append(row)
        evidence_store.save_json("chunked.json", chunked_by_topic)
    evidence_store.save_json("single_units.json", all_single_units)
    evidence_store.save_json("multi_units.json", all_multi_units)

    for mode, mode_cfg in blueprint.modes.items():
        logger.info(f"Starting mode: {mode}")
        target = mode_candidate_target(mode_cfg, config)
        mode_state = ModeState(mode=mode, target_count=target)
        mode_states[mode] = mode_state

        await run_mode_generation(
            mode=mode, mode_cfg=mode_cfg, blueprint=blueprint, config=config,
            global_state=global_state, mode_state=mode_state,
            evidence_manager=evidence_manager, generator=generator, tracer=tracer,
        )

        save_mode_outputs(blueprint.task_id, blueprint.run_id, mode, mode_state)
        save_mode_metrics(
            blueprint.task_id,
            blueprint.run_id,
            mode,
            mode_state,
            mode_cfg=mode_cfg,
            config=config,
        )
        logger.info(
            f"Mode {mode} complete: {mode_state.accepted_count} accepted, "
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


def _build_terminal_hard_repair_plan(
    mode: str,
    mode_cfg: Any,
    blueprint: Any,
    config: Any,
    mode_state: ModeState,
) -> ModeRoundPlan | None:
    exp_cfg = getattr(config, "experiment", None)
    if exp_cfg is None or not getattr(exp_cfg, "enable_terminal_hard_repair", True):
        return None

    target_hard = mode_cfg.difficulty_distribution.get("hard", 0.0)
    if target_hard <= 0:
        return None

    if target_hard <= 1.0:
        target_hard_count = int(round(mode_cfg.count * float(target_hard)))
    else:
        target_hard_count = int(round(float(target_hard)))

    accepted_hard = sum(1 for record in mode_state.accepted if record.difficulty == "hard")
    missing_hard = max(0, target_hard_count - accepted_hard)
    hard_gap_ratio = mode_state.hard_gap(float(target_hard) if target_hard <= 1.0 else target_hard_count / max(1, mode_cfg.count))

    if missing_hard < int(getattr(exp_cfg, "terminal_hard_repair_min_missing", 2)):
        return None
    if hard_gap_ratio < float(getattr(exp_cfg, "terminal_hard_repair_gap_threshold", 0.15)):
        return None

    hard_topic_counts: dict[str, int] = {topic: 0 for topic in blueprint.topics}
    for record in mode_state.accepted:
        if record.difficulty == "hard":
            hard_topic_counts[record.topic] = hard_topic_counts.get(record.topic, 0) + 1

    topic_limit = max(1, int(getattr(exp_cfg, "terminal_hard_repair_topics", 2)))
    topics = tuple(sorted(blueprint.topics, key=lambda topic: hard_topic_counts.get(topic, 0))[:topic_limit])
    if not topics:
        return None

    limits = config.chunk_limits[mode]
    multi_k = max(1, limits.multi_k.max)
    scale = float(getattr(exp_cfg, "terminal_hard_repair_multi_k_scale", 1.0))
    multi_k = max(1, min(limits.multi_k.max, int(round(multi_k * scale))))

    return ModeRoundPlan(
        mode=mode,
        round_in_mode=mode_state.round_in_mode,
        strategy=RoundStrategy.TERMINAL_HARD_REPAIR,
        difficulty="hard",
        topics=topics,
        single_k=0,
        multi_k=multi_k,
        target_candidates_per_topic=max(1, missing_hard),
        reason=f"terminal hard repair: missing_hard={missing_hard}, hard_gap={hard_gap_ratio:.2f}",
    )


# ============================================================================
# Per-mode generation loop
# ============================================================================

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
    last_feedback = None
    terminal_repair_executed = False

    while True:
        should_stop, reason = mode_should_stop(
            mode_cfg=mode_cfg, mode_state=mode_state,
            global_state=global_state, blueprint=blueprint, config=config,
        )
        if should_stop:
            if (
                reason == "max_candidate_pool_reached"
                and not terminal_repair_executed
            ):
                repair_plan = _build_terminal_hard_repair_plan(
                    mode=mode,
                    mode_cfg=mode_cfg,
                    blueprint=blueprint,
                    config=config,
                    mode_state=mode_state,
                )
                if repair_plan is not None:
                    append_round_plan(blueprint.task_id, blueprint.run_id, mode, repair_plan)
                    round_results, feedback = await execute_mode_round_plan(
                        round_plan=repair_plan, blueprint=blueprint, config=config,
                        global_state=global_state, mode_state=mode_state,
                        evidence_manager=evidence_manager, generator=generator,
                        mode_cfg=mode_cfg, tracer=tracer,
                    )
                    last_feedback = feedback
                    append_round_feedback(blueprint.task_id, blueprint.run_id, mode, feedback)
                    if feedback.empty_round:
                        mode_state.consecutive_empty_rounds += 1
                    else:
                        mode_state.consecutive_empty_rounds = 0
                    mode_state.terminal_hard_repair_executed = True
                    mode_state.terminal_hard_repair_rounds += 1
                    mode_state.terminal_hard_repair_generated_count += int(feedback.generated_count)
                    mode_state.terminal_hard_repair_accepted_count += int(feedback.accepted_count)
                    update_mode_trace(mode_state, repair_plan, round_results)
                    mode_state.round_in_mode += 1
                    terminal_repair_executed = True
                    logger.info(
                        f"Mode={mode} terminal repair executed "
                        f"generated={feedback.generated_count} accepted={feedback.accepted_count} "
                        f"total_accepted={mode_state.accepted_count}"
                    )
                    continue
            mode_state.stopped_reason = reason
            logger.info(f"Mode {mode} stopped: {reason}")
            break

        round_plan = build_mode_round_plan(
            mode=mode, mode_cfg=mode_cfg, blueprint=blueprint,
            config=config, mode_state=mode_state, feedback=last_feedback,
        )

        append_round_plan(blueprint.task_id, blueprint.run_id, mode, round_plan)

        if not round_plan.topics:
            mode_state.consecutive_empty_rounds += 1
            mode_state.round_in_mode += 1
            continue

        round_results, feedback = await execute_mode_round_plan(
            round_plan=round_plan, blueprint=blueprint, config=config,
            global_state=global_state, mode_state=mode_state,
            evidence_manager=evidence_manager, generator=generator,
            mode_cfg=mode_cfg, tracer=tracer,
        )
        last_feedback = feedback
        append_round_feedback(blueprint.task_id, blueprint.run_id, mode, feedback)

        if feedback.empty_round:
            if round_plan.strategy == RoundStrategy.EXPAND_EVIDENCE:
                mode_state.consecutive_empty_rounds = 0
            else:
                mode_state.consecutive_empty_rounds += 1
        else:
            mode_state.consecutive_empty_rounds = 0

        update_mode_trace(mode_state, round_plan, round_results)
        mode_state.round_in_mode += 1

        logger.info(
            f"Mode={mode} round={round_plan.round_in_mode} "
            f"strategy={round_plan.strategy.value} "
            f"generated={feedback.generated_count} accepted={feedback.accepted_count} "
            f"total_accepted={mode_state.accepted_count}"
        )
