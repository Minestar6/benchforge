"""断点续跑：跳过模型推理，从已有 model_responses.jsonl 继续执行评估。
Usage:
  cd /Users/zhaoziqing/Desktop/benchforge
  python experiment/case/resume_eval.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from loguru import logger

from benchforge.agents.model_eval_agent.agent import _load_eval_questions, _write_jsonl_overwrite
from benchforge.agents.model_eval_agent.aggregator import (
    build_dataset_quality_by_group,
    build_dataset_quality_summary,
    build_model_by_dimension,
    build_model_overall_report,
)
from benchforge.agents.model_eval_agent.auto_metric_executor import run_automatic_metrics
from benchforge.agents.model_eval_agent.config_loader import load_model_eval_config
from benchforge.agents.model_eval_agent.dataset_metric_executor import run_dataset_metrics
from benchforge.agents.model_eval_agent.llm_judge_executor import run_llm_judge
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry, resolve_model_config
from benchforge.agents.model_eval_agent.report_writer import write_reports
from benchforge.models.loader import ModelLoader
from benchforge.utils.artifact_store import ArtifactStore
from benchforge.utils.run_context import RunContext

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "configs" / "model_eval_case.yaml"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"
CASE_BASE = PROJECT_ROOT / "runs" / "case"
CASE_PREFIX = "d_full_case_"


def _latest_case_run(case_base: Path) -> Path:
    dirs = sorted(
        [d for d in case_base.iterdir() if d.is_dir() and d.name.startswith(CASE_PREFIX)],
        reverse=True,
    )
    if not dirs:
        raise FileNotFoundError(f"No {CASE_PREFIX}* run dirs found")
    run_dir = dirs[0]
    shared = run_dir / "shared_state.json"
    if not shared.exists():
        raise FileNotFoundError(f"shared_state.json not found: {shared}")
    return run_dir


async def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    run_dir = _latest_case_run(CASE_BASE)
    logger.info("Resuming from: {}", run_dir.name)

    config = load_model_eval_config(CONFIG_PATH)

    # 从 shared_state 解析 run 上下文
    from benchforge.utils.shared_state import load_shared_state
    state = load_shared_state(str(run_dir / "shared_state.json"))
    task_id = state.task_id
    run_id = state.run_id
    output_root = Path("runs") / task_id / run_id / "evaluation"

    # 加载已有的 evaluation_questions 和 model_responses
    questions_path = output_root / "intermediate" / "evaluation_questions.jsonl"
    responses_path = output_root / "model_report" / "model_responses.jsonl"

    if not questions_path.exists():
        logger.error("evaluation_questions.jsonl not found, run full eval first")
        return
    if not responses_path.exists():
        logger.error("model_responses.jsonl not found, run full eval first")
        return

    with open(questions_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    with open(responses_path, encoding="utf-8") as f:
        raw_responses = [json.loads(line) for line in f if line.strip()]

    # 去重：旧 run 的 model_responses 可能含重复 (question_id, model_name)，
    # 保留最后出现的记录（重试成功的条目会覆盖失败的条目）
    model_responses: list[dict] = []
    seen: dict[tuple[str, str], int] = {}
    for r in raw_responses:
        key = (r["question_id"], r["model_name"])
        if key in seen:
            model_responses[seen[key]] = r  # 原地替换为最新记录
        else:
            seen[key] = len(model_responses)
            model_responses.append(r)

    if len(model_responses) < len(raw_responses):
        logger.info("Deduped model responses: {} -> {}", len(raw_responses), len(model_responses))

    logger.info("Loaded {} questions, {} model responses", len(questions), len(model_responses))

    # tracer
    run_ctx = RunContext(task_id=task_id, run_id=run_id)
    llm_trace_path = run_ctx.llm_trace_path
    judge_tracer = run_ctx.create_tracer(agent="model_eval_agent", stage="judge")

    # judge client
    registry = load_model_registry(str(REGISTRY_PATH))
    judge_client = None
    judge_model_name = config.models.judge_model_name or ""
    if config.judge.enabled and judge_model_name:
        if judge_model_name in registry:
            cfg = resolve_model_config(judge_model_name, registry)
            judge_client = ModelLoader.load_model(cfg)
        else:
            logger.warning("Judge model '{}' not in registry", judge_model_name)

    # 清理旧的追加型中间产物，避免新旧数据混杂
    for cleanup_path in [
        output_root / "dataset_report" / "dataset_scores.jsonl",
        output_root / "model_report" / "automatic_scores.jsonl",
        output_root / "model_report" / "llm_judge_scores.jsonl",
        output_root / "model_report" / "traces" / "llm_judge_traces.jsonl",
        output_root / "model_report" / "traces" / "llm_judge_prompts.jsonl",
    ]:
        if cleanup_path.exists():
            cleanup_path.unlink()
            logger.info("Cleaned: {}", cleanup_path)

    # Step 1: 数据集指标（轻量，重跑）
    dataset_scores = run_dataset_metrics(
        questions=questions,
        metric_specs=config.dataset_evaluation.metrics if config.dataset_evaluation.enabled else [],
        output_dir=output_root / "dataset_report",
    )

    # Step 3: 自动指标
    automatic_scores = run_automatic_metrics(
        questions=questions,
        model_responses=model_responses,
        metric_plans=config.metrics,
        output_dir=output_root / "model_report",
    )
    logger.info("Auto metrics: {} rows", len(automatic_scores))

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
    logger.info("LLM Judge: {} rows", len(judge_scores))

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
        "num_responses": len(model_responses),
        "output_root": str(output_root),
    }
    ArtifactStore(str(output_root)).save_json("evaluation_report.json", result)

    # 回写 shared_state
    from benchforge.utils.shared_state import update_and_save
    update_and_save(
        str(run_dir / "shared_state.json"),
        agent="evaluation",
        artifacts={
            "llm_calls": str(run_dir / "llm_calls.jsonl"),
            "evaluation_report": str(output_root / "evaluation_report.json"),
            "dataset_quality_summary": str(output_root / "dataset_report" / "dataset_quality_summary.json"),
            "model_overall_report": str(output_root / "model_report" / "model_overall_report.csv"),
            "model_aggregate_report": str(output_root / "model_report" / "model_aggregate_report.json"),
            "model_by_topic": str(output_root / "model_report" / "model_by_topic.csv"),
            "model_by_difficulty": str(output_root / "model_report" / "model_by_difficulty.csv"),
            "model_by_question_mode": str(output_root / "model_report" / "model_by_question_mode.csv"),
        },
    )

    print("\n=== Resume Evaluation Done ===")
    print(f"run_dir         : {run_dir}")
    print(f"questions       : {len(questions)}")
    print(f"responses       : {len(model_responses)}")
    print(f"auto_metrics    : {len(automatic_scores)} rows")
    print(f"judge_metrics   : {len(judge_scores)} rows")
    print(f"output          : {output_root}")


if __name__ == "__main__":
    asyncio.run(main())
