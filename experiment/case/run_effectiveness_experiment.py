"""Run or resume the effectiveness experiment with one run_id per outer round.

Layout:
  runs/<task_id>/round_001/
  runs/<task_id>/round_002/
  ...
  runs/<task_id>/effectiveness/task_effectiveness_report.json

Usage:
  python experiment/case/run_effectiveness_experiment.py
  python experiment/case/run_effectiveness_experiment.py --resume-latest
  python experiment/case/run_effectiveness_experiment.py --task-id effectiveness_task_...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from copy import deepcopy
from dataclasses import is_dataclass, replace
from datetime import datetime
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
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
from benchforge.agents.verify_agent.config_loader import load_verify_agent_config
from benchforge.config.config import load_dotenv
from benchforge.models.loader import ModelLoader

from experiment.case.effectiveness import (
    compute_benchmark_metrics,
    compute_round_improvement_summary,
    latest_effectiveness_task,
    next_round_index,
    should_run_stage,
    task_round_run_dir,
)

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
BLUEPRINT_PATH = CONFIG_DIR / "case_blueprint.yaml"
QA_CONFIG_PATH = CONFIG_DIR / "qa_agent_case_d_full.yaml"
VERIFY_CONFIG_PATH = CONFIG_DIR / "verify_agent_case.yaml"
MODEL_EVAL_CONFIG_PATH = CONFIG_DIR / "model_eval_case.yaml"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"
RUNS_BASE = PROJECT_ROOT / "runs"

TASK_PREFIX = "effectiveness_task_"
DEFAULT_TOTAL_ROUNDS = 2
GENERATOR_MODEL = "deepseek-v4"
VERIFY_MODEL = "deepseek-v4"
EVAL_CANDIDATE_MODELS = ["deepseek-v4", "deepseek-v4-pro", "minimax"]
EVAL_JUDGE_MODEL = "deepseek-v4"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume the effectiveness experiment.")
    parser.add_argument("--task-id", help="Task id to resume or create.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume the latest effectiveness task.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_TOTAL_ROUNDS, help="Total outer rounds for the task.")
    parser.add_argument("--force-generation", action="store_true", help="Rerun generation for unfinished rounds.")
    parser.add_argument("--force-verification", action="store_true", help="Rerun verification for unfinished rounds.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun B1/Bfull evaluations.")
    parser.add_argument("--force-report", action="store_true", help="Rewrite the aggregated task report.")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_blueprint_override() -> dict[str, Any]:
    raw = deepcopy(_load_yaml(BLUEPRINT_PATH))
    raw["language"] = "en"
    raw["topics"] = raw.get("topics") or ["Artificial Intelligence", "Quantum Computing"]
    raw["modes"]["qa"]["count"] = 4
    raw["modes"]["qa"]["max_rounds"] = 1
    raw["modes"]["multiple_choice"]["count"] = 2
    raw["modes"]["multiple_choice"]["max_rounds"] = 1
    return raw


def _prepare_qa_config_override() -> dict[str, Any]:
    raw = deepcopy(_load_yaml(QA_CONFIG_PATH))
    raw["model"]["name"] = GENERATOR_MODEL
    raw["candidate_pool"]["min_candidate_multiplier"] = 1.8
    raw["candidate_pool"]["max_candidate_multiplier"] = 2.2
    raw["runtime"]["max_consecutive_empty_rounds_per_mode"] = 1
    return raw


def _prepare_verify_config_override() -> dict[str, Any]:
    raw = deepcopy(_load_yaml(VERIFY_CONFIG_PATH))
    raw["llm_validation"]["model"] = VERIFY_MODEL
    raw["llm_validation"]["max_concurrency"] = 2
    return raw


def _prepare_model_eval_config_override() -> dict[str, Any]:
    raw = deepcopy(_load_yaml(MODEL_EVAL_CONFIG_PATH))
    raw["models"]["candidate_model_names"] = list(EVAL_CANDIDATE_MODELS)
    raw["models"]["judge_model_name"] = EVAL_JUDGE_MODEL
    return raw


def _build_blueprint(raw: dict[str, Any], *, task_id: str, run_id: str) -> Blueprint:
    modes = {
        mode: ModeCfg(
            count=int(cfg["count"]),
            max_rounds=int(cfg["max_rounds"]),
            difficulty_distribution=dict(cfg["difficulty_distribution"]),
        )
        for mode, cfg in raw["modes"].items()
    }
    return Blueprint(
        task_id=task_id,
        run_id=run_id,
        language=str(raw.get("language", "en")),
        topics=list(raw.get("topics", [])),
        modes=modes,
    )


def _task_dir(task_id: str) -> Path:
    return RUNS_BASE / task_id


def _task_metadata_path(task_id: str) -> Path:
    return _task_dir(task_id) / "task_metadata.json"


def _task_report_path(task_id: str) -> Path:
    return _task_dir(task_id) / "effectiveness" / "task_effectiveness_report.json"


def _task_inputs_dir(task_id: str) -> Path:
    return _task_dir(task_id) / "experiment_inputs"


def _round_run_id(round_index: int) -> str:
    return f"round_{round_index:03d}"


def _resolve_task_id(args: argparse.Namespace) -> str:
    if args.task_id:
        return args.task_id
    latest = latest_effectiveness_task(RUNS_BASE)
    if args.resume_latest and latest is not None:
        return latest.name
    if latest is not None and not _task_report_path(latest.name).exists():
        return latest.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{TASK_PREFIX}{timestamp}"


def _initialize_task(task_id: str, total_rounds: int) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    task_dir = _task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = _task_inputs_dir(task_id)
    blueprint_path = inputs_dir / "case_blueprint_effectiveness.yaml"
    qa_config_path = inputs_dir / "qa_agent_effectiveness.yaml"
    verify_config_path = inputs_dir / "verify_agent_effectiveness.yaml"
    eval_config_path = inputs_dir / "model_eval_effectiveness.yaml"

    if not blueprint_path.exists():
        _write_yaml(blueprint_path, _prepare_blueprint_override())
    if not qa_config_path.exists():
        _write_yaml(qa_config_path, _prepare_qa_config_override())
    if not verify_config_path.exists():
        _write_yaml(verify_config_path, _prepare_verify_config_override())
    if not eval_config_path.exists():
        _write_yaml(eval_config_path, _prepare_model_eval_config_override())

    metadata_path = _task_metadata_path(task_id)
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        metadata["total_rounds"] = int(metadata.get("total_rounds", total_rounds))
    else:
        metadata = {
            "task_id": task_id,
            "total_rounds": total_rounds,
            "generator_model": GENERATOR_MODEL,
            "verify_model": VERIFY_MODEL,
            "eval_candidate_models": EVAL_CANDIDATE_MODELS,
            "eval_judge_model": EVAL_JUDGE_MODEL,
            "registry_path": str(REGISTRY_PATH),
        }
        _write_json(metadata_path, metadata)

    return _load_yaml(blueprint_path), qa_config_path, verify_config_path, eval_config_path, task_dir


def _load_verify_model_client(config_path: Path):
    verify_cfg = load_verify_agent_config(config_path)
    registry = load_model_registry(REGISTRY_PATH)
    model_key = verify_cfg.llm_validation.model
    if model_key not in registry:
        raise KeyError(f"verify model '{model_key}' not found in {REGISTRY_PATH}")
    return verify_cfg, ModelLoader.load_model(registry[model_key])


def _load_automatic_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def _run_model_eval_from_config(config, registry_path: str) -> dict[str, Any]:
    return await run_model_eval_internal(config, registry_path=registry_path)


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


def _load_selected_validation_records(validated_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not validated_path.exists():
        return records
    with open(validated_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("final_status") == "selected":
                records.append(record)
    return records


async def _run_round(
    *,
    task_id: str,
    round_index: int,
    blueprint_raw: dict[str, Any],
    qa_config_path: Path,
    verify_config_path: Path,
    force_generation: bool,
    force_verification: bool,
) -> dict[str, Any]:
    run_id = _round_run_id(round_index)
    run_dir = task_round_run_dir(RUNS_BASE, task_id, round_index)
    blueprint = _build_blueprint(blueprint_raw, task_id=task_id, run_id=run_id)

    generation_report_path = run_dir / "generation_report.json"
    shared_state_path = run_dir / "shared_state.json"
    validated_path = run_dir / "validation" / "validated_questions.jsonl"

    if should_run_stage([generation_report_path, shared_state_path], force=force_generation):
        logger.info("Round {} generation start", run_id)
        generation_report = await run_generation_agent(
            blueprint=blueprint,
            config_path=qa_config_path,
            registry_path=REGISTRY_PATH,
        )
        logger.info("Round {} generation complete: {} candidates", run_id, generation_report["total_candidates"])
    else:
        logger.info("Round {} generation skipped", run_id)

    if should_run_stage([validated_path], force=force_verification):
        verify_cfg, verify_model_client = _load_verify_model_client(verify_config_path)
        logger.info("Round {} verification start", run_id)
        verify_result = await run_verify_agent_from_shared_state(
            shared_state_path=shared_state_path,
            config=verify_cfg,
            model_client=verify_model_client,
        )
        logger.info("Round {} verification complete: {} selected", run_id, len(verify_result.selected_question_ids))
    else:
        logger.info("Round {} verification skipped", run_id)

    round_summary: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "validated_path": str(validated_path),
        "selected_count": len(_load_selected_validation_records(validated_path)),
        "mode_round_summaries": {},
    }
    for mode in ("qa", "multiple_choice"):
        feedback_path = run_dir / mode / "round_feedback.jsonl"
        if feedback_path.exists() and validated_path.exists():
            round_summary["mode_round_summaries"][mode] = compute_round_improvement_summary(
                feedback_path, validated_path
            )
    return round_summary


async def _run_eval_subset(
    *,
    task_id: str,
    subset_name: str,
    subset_records: list[dict[str, Any]],
    eval_config_path: Path,
    force: bool,
) -> dict[str, Any]:
    task_dir = _task_dir(task_id)
    subset_dir = task_dir / "effectiveness" / subset_name
    input_path = subset_dir / "selected_questions.jsonl"
    _write_jsonl(input_path, subset_records)

    eval_run_id = subset_name
    eval_root = RUNS_BASE / task_id / eval_run_id / "evaluation"
    evaluation_report_path = eval_root / "evaluation_report.json"
    automatic_scores_path = eval_root / "model_report" / "automatic_scores.jsonl"

    if should_run_stage([evaluation_report_path, automatic_scores_path], force=force):
        config = load_model_eval_config(eval_config_path)
        effective = _apply_eval_run_overrides(
            config,
            task_id=task_id,
            run_id=eval_run_id,
            input_path=input_path,
        )
        result = await _run_model_eval_from_config(effective, registry_path=str(REGISTRY_PATH))
    else:
        result = _read_json(evaluation_report_path)

    benchmark_metrics = compute_benchmark_metrics(_load_automatic_scores(automatic_scores_path))
    return {
        "subset_name": subset_name,
        "question_count": len(subset_records),
        "input_path": str(input_path),
        "evaluation_run_dir": str(RUNS_BASE / task_id / eval_run_id),
        "evaluation_result": result,
        "benchmark_metrics": benchmark_metrics,
    }


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
    blueprint_raw, qa_config_path, verify_config_path, eval_config_path, task_dir = _initialize_task(
        task_id, args.rounds
    )
    logger.add(
        task_dir / "effectiveness.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        encoding="utf-8",
    )
    logger.info("Using task {}", task_id)

    total_rounds = int(_read_json(_task_metadata_path(task_id)).get("total_rounds", args.rounds))
    round_summaries: list[dict[str, Any]] = []
    for round_index in range(1, total_rounds + 1):
        round_summary = await _run_round(
            task_id=task_id,
            round_index=round_index,
            blueprint_raw=blueprint_raw,
            qa_config_path=qa_config_path,
            verify_config_path=verify_config_path,
            force_generation=args.force_generation,
            force_verification=args.force_verification,
        )
        round_summaries.append(round_summary)

    round_dirs = [task_round_run_dir(RUNS_BASE, task_id, i) for i in range(1, total_rounds + 1)]
    b1_records = _load_selected_validation_records(round_dirs[0] / "validation" / "validated_questions.jsonl")
    bfull_records = []
    for round_dir in round_dirs:
        bfull_records.extend(_load_selected_validation_records(round_dir / "validation" / "validated_questions.jsonl"))

    if not b1_records:
        raise ValueError("No round-1 selected questions found; cannot build B1 benchmark")

    b1_eval = await _run_eval_subset(
        task_id=task_id,
        subset_name="b1_eval",
        subset_records=b1_records,
        eval_config_path=eval_config_path,
        force=args.force_eval,
    )
    bfull_eval = await _run_eval_subset(
        task_id=task_id,
        subset_name="bfull_eval",
        subset_records=bfull_records,
        eval_config_path=eval_config_path,
        force=args.force_eval,
    )

    report = {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "total_rounds": total_rounds,
        "next_round_index": next_round_index(task_dir),
        "round_summaries": round_summaries,
        "round_1_selected_count": len(b1_records),
        "all_rounds_selected_count": len(bfull_records),
        "later_round_selected_count": len(bfull_records) - len(b1_records),
        "b1": b1_eval,
        "bfull": bfull_eval,
        "benchmark_delta": {
            "score_variance": round(
                bfull_eval["benchmark_metrics"]["score_variance"] - b1_eval["benchmark_metrics"]["score_variance"],
                6,
            ),
            "top_bottom_gap": round(
                bfull_eval["benchmark_metrics"]["top_bottom_gap"] - b1_eval["benchmark_metrics"]["top_bottom_gap"],
                6,
            ),
            "adjacent_gap_mean": round(
                bfull_eval["benchmark_metrics"]["adjacent_gap_mean"] - b1_eval["benchmark_metrics"]["adjacent_gap_mean"],
                6,
            ),
            "per_question_gap_mean": round(
                bfull_eval["benchmark_metrics"]["per_question_gap_mean"]
                - b1_eval["benchmark_metrics"]["per_question_gap_mean"],
                6,
            ),
        },
    }
    report_path = _task_report_path(task_id)
    if should_run_stage([report_path], force=args.force_report):
        _write_json(report_path, report)
    else:
        _write_json(report_path, report)

    print("\n=== Effectiveness Task Report ===")
    print(f"task_id                 : {task_id}")
    print(f"task_dir                : {task_dir}")
    print(f"rounds                  : {total_rounds}")
    print(f"round1_selected         : {len(b1_records)}")
    print(f"all_rounds_selected     : {len(bfull_records)}")
    print(f"B1 top_bottom_gap       : {b1_eval['benchmark_metrics']['top_bottom_gap']}")
    print(f"Bfull top_bottom_gap    : {bfull_eval['benchmark_metrics']['top_bottom_gap']}")
    print(f"delta top_bottom_gap    : {report['benchmark_delta']['top_bottom_gap']}")
    print(f"report_path             : {report_path}")


if __name__ == "__main__":
    parser = _build_parser()
    asyncio.run(main(parser.parse_args()))
