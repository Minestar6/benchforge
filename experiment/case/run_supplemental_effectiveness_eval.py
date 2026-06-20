"""Run supplemental B1/Bfull evaluation for additional models on an effectiveness task.

Usage:
  python experiment/case/run_supplemental_effectiveness_eval.py --task-id effectiveness_task_... --models glm-4.7 kimi-k2
  python experiment/case/run_supplemental_effectiveness_eval.py --resume-latest --models qwen2.5-7b
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from copy import deepcopy
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from benchforge.agents.model_eval_agent.agent import _run as run_model_eval_internal
from benchforge.agents.model_eval_agent.config_loader import load_model_eval_config
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry
from benchforge.config.config import load_dotenv

from experiment.case.effectiveness import latest_effectiveness_task, should_run_stage
from experiment.case.supplemental_eval import (
    build_combined_subset_report,
    build_summary_markdown,
    make_eval_tag,
    read_json,
    read_jsonl,
    resolve_supplemental_models,
    summarize_model_score_rows,
    write_csv,
    write_json,
    write_jsonl,
)

RUNS_BASE = PROJECT_ROOT / "runs"
CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_MODEL_EVAL_CONFIG_PATH = CONFIG_DIR / "model_eval_case.yaml"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"


def _resolve_portable_path(path_str: str) -> Path:
    """将 task_report 中的路径转换为当前机器可用的路径。

    处理三种情况：
    1. 路径直接存在 → 原样返回
    2. 绝对路径（另一台机器生成）→ 提取 runs/ 之后部分，基于 RUNS_BASE 重定位
    3. PROJECT_ROOT 相对路径（如 runs/xxx/...）→ 基于 PROJECT_ROOT 解析
    """
    p = Path(path_str)
    if p.exists():
        return p

    # 尝试作为 PROJECT_ROOT 相对路径解析
    resolved = PROJECT_ROOT / path_str
    if resolved.exists():
        logger.info(f"Resolved relative path: {path_str} -> {resolved}")
        return resolved

    # 绝对路径：提取 runs/ 之后的相对部分，基于 RUNS_BASE 重定位
    parts = p.parts
    for i, part in enumerate(parts):
        if part.lower() == "runs":
            relative = Path(*parts[i + 1:])
            resolved = RUNS_BASE / relative
            if resolved.exists():
                logger.info(f"Rebased portable path: {path_str} -> {resolved}")
                return resolved
            break

    logger.warning(f"Path not found and cannot resolve: {path_str}")
    return p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run supplemental evaluation for an effectiveness task.")
    parser.add_argument("--task-id", help="Existing effectiveness task id.")
    parser.add_argument("--resume-latest", action="store_true", help="Use the latest effectiveness task.")
    parser.add_argument("--models", nargs="+", help="Supplemental candidate models to evaluate.")
    parser.add_argument("--judge-model", help="Override judge model.")
    parser.add_argument("--tag", help="Custom suffix for this supplemental eval.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun supplemental model evaluation.")
    parser.add_argument("--force-report", action="store_true", help="Rewrite merged report outputs.")
    return parser


def _resolve_task_id(args: argparse.Namespace) -> str:
    if args.task_id:
        return args.task_id
    latest = latest_effectiveness_task(RUNS_BASE)
    if latest is None:
        raise FileNotFoundError("No effectiveness_task_* directory found under runs/")
    return latest.name


def _task_dir(task_id: str) -> Path:
    return RUNS_BASE / task_id


def _task_metadata_path(task_id: str) -> Path:
    return _task_dir(task_id) / "task_metadata.json"


def _task_report_path(task_id: str) -> Path:
    return _task_dir(task_id) / "effectiveness" / "task_effectiveness_report.json"


def _task_inputs_dir(task_id: str) -> Path:
    return _task_dir(task_id) / "experiment_inputs"


def _apply_eval_run_overrides(config, *, task_id: str, run_id: str, input_path: Path):
    if is_dataclass(config) and is_dataclass(config.run):
        return replace(
            config,
            run=replace(
                config.run,
                shared_state_path="",
                task_id=task_id,
                run_id=run_id,
                input_paths=[str(input_path)],
            ),
        )
    config.run.shared_state_path = ""
    config.run.task_id = task_id
    config.run.run_id = run_id
    config.run.input_paths = [str(input_path)]
    return config


async def _run_model_eval_from_config(config, registry_path: str) -> dict[str, Any]:
    return await run_model_eval_internal(config, registry_path=registry_path)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _load_base_eval_config(task_id: str) -> Path:
    task_override = _task_inputs_dir(task_id) / "model_eval_effectiveness.yaml"
    if task_override.exists():
        return task_override
    return DEFAULT_MODEL_EVAL_CONFIG_PATH


def _load_dataset_inputs(eval_run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_dir = eval_run_dir / "evaluation"
    questions = read_jsonl(evaluation_dir / "intermediate" / "evaluation_questions.jsonl")
    dataset_scores = read_jsonl(evaluation_dir / "dataset_report" / "dataset_scores.jsonl")
    automatic_scores = read_jsonl(evaluation_dir / "model_report" / "automatic_scores.jsonl")
    judge_scores_path = evaluation_dir / "model_report" / "llm_judge_scores.jsonl"
    judge_scores = read_jsonl(judge_scores_path) if judge_scores_path.exists() else []
    return questions, dataset_scores, automatic_scores, judge_scores


async def _run_subset_eval(
    *,
    task_id: str,
    subset_name: str,
    input_path: Path,
    base_config_path: Path,
    supplemental_models: list[str],
    judge_model: str,
    eval_tag: str,
    force: bool,
) -> dict[str, Any]:
    run_id = f"{subset_name}__supp__{eval_tag}"
    run_root = RUNS_BASE / task_id / run_id
    report_path = run_root / "evaluation" / "evaluation_report.json"
    automatic_scores_path = run_root / "evaluation" / "model_report" / "automatic_scores.jsonl"

    config = load_model_eval_config(base_config_path)
    if is_dataclass(config):
        config = replace(
            config,
            models=replace(
                config.models,
                candidate_model_names=list(supplemental_models),
                judge_model_name=judge_model,
            ),
        )
    else:
        config.models.candidate_model_names = list(supplemental_models)
        config.models.judge_model_name = judge_model
    effective_config = _apply_eval_run_overrides(config, task_id=task_id, run_id=run_id, input_path=input_path)

    if should_run_stage([report_path, automatic_scores_path], force=force):
        logger.info("Supplemental eval start: subset={} models={}", subset_name, supplemental_models)
        await _run_model_eval_from_config(effective_config, registry_path=str(REGISTRY_PATH))
    else:
        logger.info("Supplemental eval skipped: subset={}", subset_name)

    return {
        "run_id": run_id,
        "run_dir": str(run_root),
        "evaluation_report_path": str(report_path),
        "automatic_scores_path": str(automatic_scores_path),
    }


def _write_analysis_tables(output_dir: Path, subset_reports: list[dict[str, Any]]) -> None:
    score_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    discrim_rows: list[dict[str, Any]] = []
    for subset in subset_reports:
        subset_name = subset["subset_name"]
        combined = subset["combined"]
        score_rows.extend(
            summarize_model_score_rows(
                subset_name=subset_name,
                benchmark_metrics=combined["benchmark_metrics"],
                evaluation_result=combined["evaluation_result"],
            )
        )
        for row in combined["judge_summary"]:
            judge_rows.append({"subset": subset_name, **row})
        discrim = combined["evaluation_result"].get("discriminative_signals", {})
        discrim_rows.append(
            {
                "subset": subset_name,
                "overall_model_gap": discrim.get("overall_model_gap"),
                "best_vs_second_gap": discrim.get("best_vs_second_gap"),
                "discriminative_question_ratio": discrim.get("discriminative_question_ratio"),
                "all_models_fail_ratio": discrim.get("all_models_fail_ratio"),
                "top_bottom_gap": combined["benchmark_metrics"].get("top_bottom_gap"),
                "score_variance": combined["benchmark_metrics"].get("score_variance"),
            }
        )

    write_csv(output_dir / "combined_model_scores.csv", score_rows, ["subset", "scope", "model_name", "score"])
    write_csv(
        output_dir / "combined_judge_scores.csv",
        judge_rows,
        ["subset", "question_mode", "metric_name", "model_name", "mean"],
    )
    write_csv(
        output_dir / "combined_discriminative_signals.csv",
        discrim_rows,
        [
            "subset",
            "overall_model_gap",
            "best_vs_second_gap",
            "discriminative_question_ratio",
            "all_models_fail_ratio",
            "top_bottom_gap",
            "score_variance",
        ],
    )


async def main(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("CUSTOM_API_KEY"):
        raise RuntimeError(f"CUSTOM_API_KEY not set. Expected .env at {PROJECT_ROOT / '.env'}")
    if not os.getenv("CUSTOM_API_BASE_URL"):
        raise RuntimeError(f"CUSTOM_API_BASE_URL not set. Expected .env at {PROJECT_ROOT / '.env'}")

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    task_id = _resolve_task_id(args)
    task_report = read_json(_task_report_path(task_id))
    task_metadata = read_json(_task_metadata_path(task_id))
    registry = load_model_registry(REGISTRY_PATH)

    baseline_models = list(task_metadata.get("eval_candidate_models", []))
    supplemental_models = resolve_supplemental_models(
        requested_models=args.models,
        registry_models=set(registry.keys()),
        baseline_models=baseline_models,
    )
    judge_model = args.judge_model or str(task_metadata.get("eval_judge_model", ""))
    eval_tag = make_eval_tag(supplemental_models, args.tag)
    base_config_path = _load_base_eval_config(task_id)

    output_dir = _task_dir(task_id) / "effectiveness" / "supplemental_eval" / eval_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    config_copy = deepcopy(read_json(base_config_path)) if base_config_path.suffix == ".json" else None
    if config_copy is None:
        with open(base_config_path, encoding="utf-8") as f:
            config_copy = yaml.safe_load(f)
    config_copy["models"]["candidate_model_names"] = supplemental_models
    config_copy["models"]["judge_model_name"] = judge_model
    _write_yaml(output_dir / "model_eval_effectiveness_supplemental.yaml", config_copy)

    subset_specs = [
        ("b1_eval", _resolve_portable_path(task_report["b1"]["input_path"]), _resolve_portable_path(task_report["b1"]["evaluation_run_dir"])),
        ("bfull_eval", _resolve_portable_path(task_report["bfull"]["input_path"]), _resolve_portable_path(task_report["bfull"]["evaluation_run_dir"])),
    ]

    subset_reports: list[dict[str, Any]] = []
    for subset_name, input_path, baseline_run_dir in subset_specs:
        supplemental_run = await _run_subset_eval(
            task_id=task_id,
            subset_name=subset_name,
            input_path=input_path,
            base_config_path=base_config_path,
            supplemental_models=supplemental_models,
            judge_model=judge_model,
            eval_tag=eval_tag,
            force=args.force_eval,
        )
        questions, dataset_scores, baseline_automatic, baseline_judge = _load_dataset_inputs(baseline_run_dir)
        _, _, supplemental_automatic, supplemental_judge = _load_dataset_inputs(Path(supplemental_run["run_dir"]))
        combined = build_combined_subset_report(
            subset_name=subset_name,
            questions=questions,
            dataset_scores=dataset_scores,
            baseline_automatic_rows=baseline_automatic,
            supplemental_automatic_rows=supplemental_automatic,
            baseline_judge_rows=baseline_judge,
            supplemental_judge_rows=supplemental_judge,
        )
        write_jsonl(output_dir / f"{subset_name}_combined_automatic_scores.jsonl", combined["merged_automatic_scores"])
        write_jsonl(output_dir / f"{subset_name}_combined_judge_scores.jsonl", combined["merged_judge_scores"])
        subset_reports.append(
            {
                "subset_name": subset_name,
                "baseline_evaluation_run_dir": str(baseline_run_dir),
                "supplemental_evaluation_run_dir": supplemental_run["run_dir"],
                "combined": combined,
            }
        )

    final_report = {
        "task_id": task_id,
        "baseline_models": baseline_models,
        "supplemental_models": supplemental_models,
        "judge_model": judge_model,
        "eval_tag": eval_tag,
        "subsets": subset_reports,
    }
    report_path = output_dir / "supplemental_eval_report.json"
    summary_path = output_dir / "supplemental_eval_summary.md"
    if should_run_stage([report_path, summary_path], force=args.force_report):
        write_json(report_path, final_report)
        _write_analysis_tables(output_dir, subset_reports)
        summary_path.write_text(
            build_summary_markdown(final_report),
            encoding="utf-8",
        )
    else:
        logger.info("Supplemental merged report skipped: {}", output_dir)

    print("\n=== Supplemental Effectiveness Evaluation ===")
    print(f"task_id                  : {task_id}")
    print(f"supplemental_models      : {', '.join(supplemental_models)}")
    print(f"judge_model              : {judge_model}")
    print(f"output_dir               : {output_dir}")


if __name__ == "__main__":
    parser = _build_parser()
    asyncio.run(main(parser.parse_args()))
