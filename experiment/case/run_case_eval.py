"""Evaluate the latest experiment/case run with model_eval_agent.

Usage:
  cd /Users/zhaoziqing/Desktop/benchforge
  python experiment/case/run_case_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from loguru import logger

from benchforge.agents.model_eval_agent.agent import run_model_eval_agent_from_shared_state
from benchforge.agents.model_eval_agent.config_loader import load_model_eval_config


BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
CONFIG_PATH = CONFIG_DIR / "model_eval_case.yaml"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"
CASE_BASE = PROJECT_ROOT / "runs" / "case"
CASE_PREFIX = "d_full_case_"


def _latest_case_run(case_base: Path) -> Path:
    if not case_base.exists():
        raise FileNotFoundError(f"No case runs found: {case_base}")

    dirs = sorted(
        [d for d in case_base.iterdir() if d.is_dir() and d.name.startswith(CASE_PREFIX)],
        reverse=True,
    )
    if not dirs:
        raise FileNotFoundError(f"No {CASE_PREFIX}* run dirs found under {case_base}")

    run_dir = dirs[0]
    shared_state_path = run_dir / "shared_state.json"
    if not shared_state_path.exists():
        raise FileNotFoundError(f"shared_state.json not found: {shared_state_path}")
    return run_dir


def _collect_selected_question_count(run_dir: Path) -> int:
    validated_path = run_dir / "validation" / "validated_questions.jsonl"
    if not validated_path.exists():
        return 0

    count = 0
    with open(validated_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("final_status") == "selected":
                count += 1
    return count


async def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    run_dir = _latest_case_run(CASE_BASE)
    shared_state_path = run_dir / "shared_state.json"

    logger.info("Using latest case run: {}", run_dir.name)

    config = load_model_eval_config(CONFIG_PATH)
    if is_dataclass(config) and is_dataclass(config.run):
        config = replace(config, run=replace(config.run, shared_state_path=str(shared_state_path)))
    else:
        config.run.shared_state_path = str(shared_state_path)

    result = await run_model_eval_agent_from_shared_state(
        shared_state_path=str(shared_state_path),
        config=config,
        registry_path=str(REGISTRY_PATH),
    )

    report = {
        "config_path": str(CONFIG_PATH),
        "registry_path": str(REGISTRY_PATH),
        "run_dir": str(run_dir),
        "shared_state_path": str(shared_state_path),
        "selected_questions": _collect_selected_question_count(run_dir),
        "evaluation": result,
    }
    report_path = CASE_BASE / "evaluate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Case Evaluation Report ===")
    print(f"run_dir             : {run_dir}")
    print(f"selected_questions  : {report['selected_questions']}")
    print(f"num_questions       : {result.get('num_questions', 0)}")
    print(f"num_models          : {result.get('num_models', 0)}")
    print(f"evaluation_output   : {result.get('output_root', '')}")
    print(f"report_path         : {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
