"""VerifyAgent：三阶段题目质量验证智能体入口。"""

import time
from dataclasses import replace
from pathlib import Path

from loguru import logger

from benchforge.models.base import BaseModelClient
from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.shared_state import load_shared_state, update_and_save
from benchforge.schemas import SharedState, AgentStatus

from .citation_validator import run_citation_validation
from .config_loader import VerifyAgentConfig
from .llm_validator import run_llm_validation
from .normalizer import load_and_normalize
from .schema import (
    FinalStatus,
    ModeCfg,
    QuestionCandidate,
    ValidationBlueprintView,
    ValidationRunState,
    ValidationTaskResult,
    ValidatedQuestionRecord,
)
from .selector import (
    _compute_embeddings,
    _label_to_int_difficulty,
    _normalize_difficulty,
    _normalize_question_text,
    _question_text_for_dedup,
    run_weighted_selection,
)


def _blueprint_from_shared_state(state: SharedState) -> ValidationBlueprintView:
    raw = state.blueprint
    modes = {
        mode: ModeCfg(
            count=cfg["count"],
            difficulty_distribution=cfg["difficulty_distribution"],
        )
        for mode, cfg in raw.get("modes", {}).items()
    }
    return ValidationBlueprintView(
        topics=raw.get("topics", []),
        modes=modes,
    )


def _input_paths_from_shared_state(state: SharedState, run_dir: Path) -> list[str]:
    candidates: list[str] = []

    for key in ("qa_candidate_pool", "multiple_choice_candidate_pool"):
        if value := state.artifact(key):
            candidates.append(value)

    if candidates:
        return candidates

    inferred: list[str] = []
    for mode in ("qa", "multiple_choice"):
        candidate_path = run_dir / mode / "candidate_pool.json"
        if candidate_path.exists():
            inferred.append(str(candidate_path))
    return inferred


def _chunk_index_path_from_shared_state(state: SharedState, run_dir: Path) -> str:
    if chunked := state.artifact("chunked_evidence"):
        return chunked
    chunked_json = run_dir / "evidence" / "chunked.json"
    chunked_jsonl = run_dir / "evidence" / "chunked.jsonl"
    if chunked_json.exists():
        return str(chunked_json)
    if chunked_jsonl.exists():
        return str(chunked_jsonl)
    return ""


def _early_exact_dedup_candidates(
    candidates: list[QuestionCandidate],
) -> tuple[list[QuestionCandidate], dict[str, str]]:
    def _difficulty_rank(candidate: QuestionCandidate) -> tuple[int, float]:
        normalized = _normalize_difficulty(candidate.estimated_difficulty)
        label_rank = {"unknown": 0, "easy": 1, "medium": 2, "hard": 3}.get(normalized, 0)
        raw_score = candidate.estimated_difficulty
        if isinstance(raw_score, (int, float)):
            numeric = float(raw_score)
        else:
            try:
                numeric = float(str(raw_score).strip())
            except (TypeError, ValueError):
                numeric = float(label_rank)
        return label_rank, numeric

    best_by_question: dict[str, QuestionCandidate] = {}
    for candidate in candidates:
        normalized = _normalize_question_text(_question_text_for_dedup(candidate))
        existing = best_by_question.get(normalized)
        if existing is None or _difficulty_rank(candidate) > _difficulty_rank(existing):
            best_by_question[normalized] = candidate

    kept = list(best_by_question.values())
    kept_ids = {candidate.question_id for candidate in kept}
    duplicate_of: dict[str, str] = {}
    for candidate in candidates:
        if candidate.question_id in kept_ids:
            continue
        normalized = _normalize_question_text(_question_text_for_dedup(candidate))
        duplicate_of[candidate.question_id] = best_by_question[normalized].question_id
    return kept, duplicate_of


def _semantic_near_dedup_candidates(
    candidates: list[QuestionCandidate],
    citation_results: dict[str, object],
    embedding_model: str,
    similarity_threshold: float = 0.9,
) -> tuple[list[QuestionCandidate], dict[str, str]]:
    if len(candidates) <= 1:
        return candidates, {}

    buckets: dict[str, list[QuestionCandidate]] = {}
    for candidate in candidates:
        difficulty = _normalize_difficulty(candidate.estimated_difficulty)
        key = f"{candidate.question_mode}::{difficulty}"
        buckets.setdefault(key, []).append(candidate)

    kept: list[QuestionCandidate] = []
    duplicate_of: dict[str, str] = {}
    threshold = max(-1.0, min(1.0, similarity_threshold))

    for bucket_candidates in buckets.values():
        if len(bucket_candidates) <= 1:
            kept.extend(bucket_candidates)
            continue

        texts = [_question_text_for_dedup(candidate) for candidate in bucket_candidates]
        embeddings = _compute_embeddings(texts, embedding_model)
        sim = embeddings @ embeddings.T
        ordered = sorted(
            bucket_candidates,
            key=lambda candidate: citation_results[candidate.question_id].citation_score,
            reverse=True,
        )
        index_by_id = {candidate.question_id: idx for idx, candidate in enumerate(bucket_candidates)}
        kept_in_bucket: list[QuestionCandidate] = []

        for candidate in ordered:
            idx = index_by_id[candidate.question_id]
            parent = None
            for chosen in kept_in_bucket:
                chosen_idx = index_by_id[chosen.question_id]
                if float(sim[idx, chosen_idx]) >= threshold:
                    parent = chosen.question_id
                    break
            if parent is None:
                kept_in_bucket.append(candidate)
            else:
                duplicate_of[candidate.question_id] = parent

        kept.extend(kept_in_bucket)

    return kept, duplicate_of


class VerifyAgent:
    """三阶段题目质量验证智能体。"""

    def __init__(
        self,
        config: VerifyAgentConfig,
        model_client: BaseModelClient | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client

    async def run(
        self,
        task_id: str,
        run_id: str,
        blueprint: ValidationBlueprintView,
        candidates: list[QuestionCandidate],
    ) -> ValidationTaskResult:
        """驱动三阶段验证，落盘结果，返回 ValidationTaskResult。"""
        logger.info(
            f"[VerifyAgent] start task={task_id} run={run_id} "
            f"candidates={len(candidates)}"
        )

        run_dir = Path("runs") / task_id / run_id
        validation_dir = run_dir / "validation"
        store = ArtifactStore(str(validation_dir))
        llm_trace_path = str(run_dir / "llm_calls.jsonl")

        # 创建 tracer（用于 LLM 验证调用的自动记录）
        from benchforge.utils.run_context import RunContext
        run_ctx = RunContext(task_id=task_id, run_id=run_id)
        validation_tracer = run_ctx.create_tracer(agent="verify_agent", stage="validation")

        state = ValidationRunState(
            task_id=task_id,
            run_id=run_id,
            total_candidates=len(candidates),
        )

        # 落盘归一化候选题
        store.save_jsonl("normalized_candidates.jsonl", [c.model_dump() for c in candidates])
        deduped_candidates, early_duplicate_of = _early_exact_dedup_candidates(candidates)
        early_duplicate_records: list[ValidatedQuestionRecord] = [
            ValidatedQuestionRecord(
                question_id=c.question_id,
                candidate=c,
                duplicate_of=early_duplicate_of[c.question_id],
                final_status=FinalStatus.duplicate.value,
            )
            for c in candidates if c.question_id in early_duplicate_of
        ]
        if early_duplicate_of:
            logger.info(
                f"[VerifyAgent] early exact dedup removed {len(early_duplicate_of)} duplicates before validation"
            )
        candidates = deduped_candidates

        # Stage 1: 引用验证
        t1 = time.monotonic()
        citation_results = run_citation_validation(candidates, self.config.citation)
        citation_latency_ms = int((time.monotonic() - t1) * 1000)

        store.save_jsonl("citation_validation.jsonl", list(citation_results.values()))
        citation_passed = [c for c in candidates if citation_results[c.question_id].passed]
        state.citation_passed = len(citation_passed)

        # 被引用验证拒绝的题 → 记录 rejected_citation 状态
        citation_rejected_records: list[ValidatedQuestionRecord] = [
            ValidatedQuestionRecord(
                question_id=c.question_id,
                candidate=c,
                citation_validation=citation_results[c.question_id],
                final_status=FinalStatus.rejected_citation.value,
            )
            for c in candidates if not citation_results[c.question_id].passed
        ]
        llm_candidates, semantic_duplicate_of = _semantic_near_dedup_candidates(
            citation_passed,
            citation_results,
            embedding_model=self.config.selection.embedding_model,
            similarity_threshold=self.config.selection.semantic_similarity_threshold,
        )
        semantic_duplicate_records: list[ValidatedQuestionRecord] = [
            ValidatedQuestionRecord(
                question_id=c.question_id,
                candidate=c,
                citation_validation=citation_results[c.question_id],
                duplicate_of=semantic_duplicate_of[c.question_id],
                final_status=FinalStatus.duplicate.value,
            )
            for c in citation_passed if c.question_id in semantic_duplicate_of
        ]
        if semantic_duplicate_of:
            logger.info(
                f"[VerifyAgent] semantic near-dedup removed {len(semantic_duplicate_of)} candidates before llm validation"
            )

        # Stage 2: LLM 验证
        t2 = time.monotonic()
        should_run_llm = (
            self.model_client is not None
            and self.config.llm_validation.enabled
            and bool(llm_candidates)
        )
        if should_run_llm:
            llm_results = await run_llm_validation(
                candidates=llm_candidates,
                blueprint=blueprint,
                cfg=self.config.llm_validation,
                model_client=self.model_client,
                llm_trace_path=llm_trace_path,
                tracer=validation_tracer,
            )
        else:
            if llm_candidates and self.model_client is not None and not self.config.llm_validation.enabled:
                logger.info("LLM validation disabled by config, skipping")
            elif llm_candidates and self.model_client is None and self.config.llm_validation.enabled:
                logger.info("LLM validation enabled but no model_client provided, skipping")
            from .schema import LLMValidationResult
            llm_results = {
                c.question_id: LLMValidationResult(
                    question_id=c.question_id,
                    passed=True,
                    overall_score=0.0,   # 0.0 而非 1.0：不参与综合分，质量分退化为纯 citation_score
                    dimensions={},
                    failed_reasons=[],
                    judge_summary="llm_validation_disabled",
                )
                for c in llm_candidates
            }
        llm_latency_ms = int((time.monotonic() - t2) * 1000)

        store.save_jsonl("llm_validation.jsonl", list(llm_results.values()))
        llm_passed = [c for c in llm_candidates if llm_results[c.question_id].passed]
        state.llm_passed = len(llm_passed)

        # 被 LLM 拒绝或 validator_error 的题
        llm_rejected_records: list[ValidatedQuestionRecord] = []
        for c in llm_candidates:
            r = llm_results[c.question_id]
            if not r.passed:
                fs = FinalStatus.validator_error.value if r.error else FinalStatus.rejected_llm.value
                llm_rejected_records.append(ValidatedQuestionRecord(
                    question_id=c.question_id,
                    candidate=c,
                    citation_validation=citation_results.get(c.question_id),
                    llm_validation=r,
                    final_status=fs,
                ))

        # Stage 3: 分组 + 去重 + 加权选题
        t3 = time.monotonic()
        if llm_passed:
            selection_result, validated_records = run_weighted_selection(
                questions=llm_passed,
                blueprint=blueprint,
                citation_results=citation_results,
                llm_results=llm_results,
                cfg=self.config.selection,
            )
        else:
            from .schema import FinalSelectionResult
            selection_result = FinalSelectionResult(
                selected_question_ids=[],
                dropped_as_duplicate=[],
                dropped_as_overquota=[],
                group_assignments={},
                sampling_weights={},
            )
            validated_records = []
        selection_latency_ms = int((time.monotonic() - t3) * 1000)

        state.final_selected = len(selection_result.selected_question_ids)

        # 合并所有逐题记录
        all_records: list[ValidatedQuestionRecord] = (
            early_duplicate_records + citation_rejected_records + semantic_duplicate_records + llm_rejected_records + validated_records
        )

        # 回写 suggested_difficulty → candidate.estimated_difficulty，仅在难度区间改变时
        for rec in all_records:
            llm_val = rec.llm_validation
            if llm_val and llm_val.suggested_difficulty in ("easy", "medium", "hard"):
                original_label = _normalize_difficulty(rec.candidate.estimated_difficulty)
                if llm_val.suggested_difficulty != original_label:
                    rec.candidate.estimated_difficulty = _label_to_int_difficulty(llm_val.suggested_difficulty)

        store.save_jsonl("validated_questions.jsonl", all_records)
        store.save_json("weighted_selection.json", selection_result.model_dump())

        # 统计 failed_by_stage
        failed_by_stage = {
            "citation": len(citation_rejected_records),
            "llm": sum(1 for r in llm_rejected_records if r.final_status == FinalStatus.rejected_llm.value),
            "validator_error": sum(1 for r in llm_rejected_records if r.final_status == FinalStatus.validator_error.value),
            "duplicate": len(early_duplicate_records) + len(semantic_duplicate_records) + len(selection_result.dropped_as_duplicate),
            "overquota": len(selection_result.dropped_as_overquota),
        }
        state.failed_by_stage = failed_by_stage

        # llm 用量统计（从 LLMValidationResult 汇总真实数据）
        llm_calls = sum(1 for r in llm_results.values())
        llm_input_tokens = sum(r.input_tokens for r in llm_results.values())
        llm_output_tokens = sum(r.output_tokens for r in llm_results.values())

        report = {
            "task_id": task_id,
            "run_id": run_id,
            "total_candidates": state.total_candidates,
            "citation_passed": state.citation_passed,
            "llm_passed": state.llm_passed,
            "final_selected": state.final_selected,
            "stage_latencies_ms": {
                "citation_validation": citation_latency_ms,
                "llm_validation": llm_latency_ms,
                "selection": selection_latency_ms,
            },
            "llm_usage": {
                "calls": llm_calls,
                "input_tokens": llm_input_tokens,
                "output_tokens": llm_output_tokens,
            },
            "failed_by_stage": failed_by_stage,
        }
        store.save_json("validation_report.json", report)

        report_path = str(validation_dir / "validation_report.json")
        logger.info(
            f"[VerifyAgent] done: selected={state.final_selected} "
            f"report={report_path}"
        )

        return ValidationTaskResult(
            task_id=task_id,
            run_id=run_id,
            report_path=report_path,
            selected_question_ids=selection_result.selected_question_ids,
            failed_by_stage=failed_by_stage,
        )


async def run_verify_agent(
    input_paths: list[str],
    blueprint: ValidationBlueprintView,
    config: VerifyAgentConfig,
    task_id: str = "",
    run_id: str = "",
    chunk_index_path: str = "",
    model_client: BaseModelClient | None = None,
) -> ValidationTaskResult:
    """独立模式入口：加载 + 归一化 + 运行 VerifyAgent。

    task_id / run_id / chunk_index_path 仅在独立运行时需要；
    流水线模式下使用 run_verify_agent_from_shared_state()。
    """
    candidates = load_and_normalize(
        input_paths=input_paths,
        task_id=task_id,
        run_id=run_id,
        chunk_index_path=chunk_index_path,
    )

    if not candidates:
        raise ValueError(f"No candidates loaded from {input_paths}")

    agent = VerifyAgent(config=config, model_client=model_client)
    return await agent.run(
        task_id=task_id,
        run_id=run_id,
        blueprint=blueprint,
        candidates=candidates,
    )


async def run_verify_agent_from_shared_state(
    shared_state_path: str | Path,
    config: VerifyAgentConfig,
    model_client: BaseModelClient | None = None,
) -> ValidationTaskResult:
    """流水线模式入口：从 shared_state.json 恢复全部运行上下文，完成后回写。

    task_id / run_id / blueprint / input_paths 均从 shared_state 解析，
    不需要在 VerifyAgentConfig 中配置。
    """
    state = load_shared_state(shared_state_path)
    task_id = state.task_id
    run_id = state.run_id
    run_dir = Path("runs") / task_id / run_id
    blueprint = _blueprint_from_shared_state(state)
    input_paths = _input_paths_from_shared_state(state, run_dir)
    chunk_index_path = _chunk_index_path_from_shared_state(state, run_dir)

    result = await run_verify_agent(
        input_paths=input_paths,
        blueprint=blueprint,
        config=config,
        task_id=task_id,
        run_id=run_id,
        chunk_index_path=chunk_index_path,
        model_client=model_client,
    )

    # 回写 shared_state：告知下游 validated_questions 位置
    update_and_save(
        shared_state_path,
        agent="verification",
        artifacts={
            "validation_report": str(run_dir / "validation" / "validation_report.json"),
            "validated_questions": str(run_dir / "validation" / "validated_questions.jsonl"),
            "weighted_selection": str(run_dir / "validation" / "weighted_selection.json"),
        },
    )

    return result
