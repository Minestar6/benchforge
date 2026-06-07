"""ModelEvaluationAgent 顶层入口。"""

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.models.base import BaseModelClient
from benchforge.models.loader import ModelLoader
from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.shared_state import load_shared_state, update_and_save
from benchforge.schemas import SharedState

from .aggregator import (
    build_dataset_quality_by_group,
    build_dataset_quality_summary,
    build_model_by_dimension,
    build_model_overall_report,
)
from .auto_metric_executor import run_automatic_metrics
from .config_loader import load_model_eval_config
from .dataset_metric_executor import run_dataset_metrics
from .llm_judge_executor import run_llm_judge
from .model_registry_loader import load_model_registry, resolve_model_config
from .model_runner import run_candidate_models
from .report_writer import write_reports
from .schema import ModelEvalAgentConfig


def _resolve_input_paths_from_shared_state(
    state: SharedState, run_dir: Path
) -> list[str]:
    paths: list[str] = []

    if val := state.artifact("validated_questions"):
        paths.append(val)
    if paths:
        return paths

    validated_path = run_dir / "validation" / "validated_questions.jsonl"
    if validated_path.exists():
        return [str(validated_path)]

    for key in ("accepted_questions", "accepted_questions_quality"):
        if val := state.artifact(key):
            paths.append(val)
    if paths:
        return paths

    fallback = run_dir / "accepted_questions.jsonl"
    if fallback.exists():
        paths.append(str(fallback))
    return paths


def _load_eval_questions(input_paths: list[str]) -> list[dict[str, Any]]:
    """加载题目，过滤掉非 selected 的记录。"""
    records: list[dict] = []
    for path in input_paths:
        p = Path(path)
        if not p.exists():
            logger.warning(f"[ModelEvalAgent] input not found: {path}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    normalized: list[dict] = []
    for record in records:
        if "final_status" in record:
            if record["final_status"] != "selected":
                continue
            candidate = dict(record.get("candidate", record))
            candidate["citation_validation"] = record.get("citation_validation")
            candidate["llm_validation"] = record.get("llm_validation")
            candidate["final_status"] = record["final_status"]
            normalized.append(candidate)
        else:
            normalized.append(record)

    if not normalized:
        raise ValueError("No evaluation questions found (no selected records in input)")
    return normalized


async def _run(
    config: ModelEvalAgentConfig,
    registry_path: str | None = None,
) -> dict[str, Any]:
    run = config.run
    task_id = run.task_id
    run_id = run.run_id
    input_paths = run.input_paths
    output_root = Path("runs") / task_id / run_id / "evaluation"
    llm_trace_path = str(output_root / "llm_calls.jsonl")

    # 创建 tracer（用于模型推理和 LLM Judge 调用的自动记录）
    from benchforge.utils.run_context import RunContext
    run_ctx = RunContext(task_id=task_id, run_id=run_id)
    inference_tracer = run_ctx.create_tracer(agent="model_eval_agent", stage="inference")
    judge_tracer = run_ctx.create_tracer(agent="model_eval_agent", stage="judge")

    logger.info(
        f"[ModelEvalAgent] start task={task_id} run={run_id}"
    )

    questions = _load_eval_questions(input_paths)
    logger.info(f"[ModelEvalAgent] loaded {len(questions)} evaluation questions")

    # 模型注册表
    registry = {}
    if registry_path:
        registry = load_model_registry(registry_path)

    # 候选模型（调用参数来自 config.models.generation_defaults）
    candidate_clients: list[tuple[str, BaseModelClient]] = []
    for name in config.models.candidate_model_names:
        if name in registry:
            cfg = resolve_model_config(name, registry)
            client = ModelLoader.load_model(cfg)
        else:
            logger.warning(f"[ModelEvalAgent] model '{name}' not in registry, skipping")
            continue
        candidate_clients.append((name, client))

    # Judge 模型（调用参数来自 config.models.judge_defaults）
    judge_client: BaseModelClient | None = None
    judge_model_name = config.models.judge_model_name or ""
    if config.judge.enabled and judge_model_name:
        if judge_model_name in registry:
            cfg = resolve_model_config(judge_model_name, registry)
            judge_client = ModelLoader.load_model(cfg)
        else:
            logger.warning(f"[ModelEvalAgent] judge model '{judge_model_name}' not in registry, judge disabled")

    # Step 1: 数据集指标
    dataset_scores = run_dataset_metrics(
        questions=questions,
        metric_specs=config.dataset_evaluation.metrics if config.dataset_evaluation.enabled else [],
        output_dir=output_root / "dataset_report",
    )

    # Step 2: 模型推理
    model_responses: list[dict] = []
    if candidate_clients:
        model_responses = await run_candidate_models(
            questions=questions,
            models=candidate_clients,
            generation_defaults=config.models.generation_defaults,
            output_dir=output_root / "model_report",
            llm_trace_path=llm_trace_path,
            tracer=inference_tracer,
        )

    # Step 3: 自动指标
    automatic_scores: list[dict] = []
    if model_responses:
        automatic_scores = run_automatic_metrics(
            questions=questions,
            model_responses=model_responses,
            metric_plans=config.metrics,
            output_dir=output_root / "model_report",
        )

    # Step 4: LLM Judge
    judge_scores: list[dict] = []
    if judge_client is not None and model_responses:
        judge_scores = await run_llm_judge(
            questions=questions,
            model_responses=model_responses,
            metric_plans=config.metrics,
            judge_client=judge_client,
            judge_model_name=judge_model_name,
            judge_config=config.judge,
            judge_defaults=config.models.judge_defaults,
            output_dir=output_root / "model_report",
            llm_trace_path=llm_trace_path,
            tracer=judge_tracer,
        )

    # Step 5: 聚合报告
    summary = build_dataset_quality_summary(questions, dataset_scores)
    by_group = build_dataset_quality_by_group(dataset_scores)
    overall = build_model_overall_report(questions, automatic_scores, judge_scores)
    by_dims = {
        "question_mode": build_model_by_dimension(questions, automatic_scores, judge_scores, "question_mode"),
        "topic": build_model_by_dimension(questions, automatic_scores, judge_scores, "topic"),
        "difficulty": build_model_by_dimension(questions, automatic_scores, judge_scores, "estimated_difficulty"),
        "question_mode_and_difficulty": build_model_by_dimension(
            questions, automatic_scores, judge_scores, "question_mode_and_difficulty"
        ),
        "topic_and_question_mode": build_model_by_dimension(
            questions, automatic_scores, judge_scores, "topic_and_question_mode"
        ),
    }

    write_reports(
        questions=questions,
        dataset_scores=dataset_scores,
        automatic_scores=automatic_scores,
        judge_scores=judge_scores,
        output_root=output_root,
        dataset_quality_summary=summary,
        dataset_quality_by_group=by_group,
        model_overall_report=overall,
        model_by_dimension=by_dims,
    )

    result = {
        "task_id": task_id,
        "run_id": run_id,
        "num_questions": len(questions),
        "num_models": len(candidate_clients),
        "output_root": str(output_root),
    }
    ArtifactStore(str(output_root)).save_json("evaluation_report.json", result)
    logger.info(f"[ModelEvalAgent] done: {output_root}")
    return result


async def run_model_eval_agent_from_shared_state(
    shared_state_path: str | Path,
    config: ModelEvalAgentConfig,
    registry_path: str | None = None,
) -> dict[str, Any]:
    """流水线模式：task_id / run_id / input_paths 从 shared_state.json 读取，完成后回写。"""
    state = load_shared_state(shared_state_path)
    task_id = state.task_id
    run_id = state.run_id
    run_dir = Path("runs") / task_id / run_id
    input_paths = _resolve_input_paths_from_shared_state(state, run_dir)

    if not input_paths:
        raise ValueError(f"No input paths resolved from shared_state: {shared_state_path}")

    effective_run = replace(config.run, task_id=task_id, run_id=run_id, input_paths=input_paths)
    effective_config = replace(config, run=effective_run)
    result = await _run(effective_config, registry_path=registry_path)

    # 回写 shared_state：标记 evaluation 完成
    update_and_save(
        shared_state_path,
        agent="evaluation",
        artifacts={
            "evaluation_report": str(run_dir / "evaluation" / "evaluation_report.json"),
        },
    )

    return result


async def run_model_eval_agent(
    config_path: str | Path,
    registry_path: str | None = None,
) -> dict[str, Any]:
    """独立运行模式：从 YAML 读取配置。"""
    config = load_model_eval_config(config_path)
    if config.run.shared_state_path:
        return await run_model_eval_agent_from_shared_state(
            config.run.shared_state_path, config, registry_path=registry_path
        )
    if not config.run.task_id or not config.run.run_id:
        raise ValueError("run.task_id and run.run_id must be set when not using shared_state_path")
    return await _run(config, registry_path=registry_path)
