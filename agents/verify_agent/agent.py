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
from .selector import run_weighted_selection


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
    inferred = run_dir / "evidence" / "chunked.jsonl"
    return str(inferred) if inferred.exists() else ""


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
        store.append_jsonl("normalized_candidates.jsonl", [c.model_dump() for c in candidates])

        # Stage 1: 引用验证
        t1 = time.monotonic()
        citation_results = run_citation_validation(candidates, self.config.citation)
        citation_latency_ms = int((time.monotonic() - t1) * 1000)

        store.append_jsonl("citation_validation.jsonl", list(citation_results.values()))
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

        # Stage 2: LLM 验证
        t2 = time.monotonic()
        should_run_llm = (
            self.model_client is not None
            and self.config.llm_validation.enabled
            and bool(citation_passed)
        )
        if should_run_llm:
            llm_results = await run_llm_validation(
                candidates=citation_passed,
                blueprint=blueprint,
                cfg=self.config.llm_validation,
                model_client=self.model_client,
                llm_trace_path=llm_trace_path,
                tracer=validation_tracer,
            )
        else:
            if citation_passed and self.model_client is not None and not self.config.llm_validation.enabled:
                logger.info("LLM validation disabled by config, skipping")
            elif citation_passed and self.model_client is None and self.config.llm_validation.enabled:
                logger.warning("LLM validation enabled but no model_client provided, skipping")
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
                for c in citation_passed
            }
        llm_latency_ms = int((time.monotonic() - t2) * 1000)

        store.append_jsonl("llm_validation.jsonl", list(llm_results.values()))
        llm_passed = [c for c in citation_passed if llm_results[c.question_id].passed]
        state.llm_passed = len(llm_passed)

        # 被 LLM 拒绝或 validator_error 的题
        llm_rejected_records: list[ValidatedQuestionRecord] = []
        for c in citation_passed:
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
            citation_rejected_records + llm_rejected_records + validated_records
        )
        store.append_jsonl("validated_questions.jsonl", all_records)
        store.save_json("weighted_selection.json", selection_result.model_dump())

        # 统计 failed_by_stage
        failed_by_stage = {
            "citation": len(citation_rejected_records),
            "llm": sum(1 for r in llm_rejected_records if r.final_status == FinalStatus.rejected_llm.value),
            "validator_error": sum(1 for r in llm_rejected_records if r.final_status == FinalStatus.validator_error.value),
            "duplicate": len(selection_result.dropped_as_duplicate),
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
            "validated_questions": str(run_dir / "validation" / "validated_questions.jsonl"),
        },
    )

    return result
