"""evaluate.py：对 verify_all.py 产出的所有实验结果调用 model_eval_agent 做模型评估。

用法:
  cd /Users/zhaoziqing/Desktop/benchforge
  python experiment/qa_agent/evaluate.py

前置条件：
- 已运行 experiment/qa_agent/verify_all.py
- 各实验目录下存在 shared_state.json，且验证阶段已写回 validated_questions artifact
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

from benchforge.agents.model_eval_agent.agent import run_model_eval_agent_from_shared_state
from benchforge.agents.model_eval_agent.config_loader import load_model_eval_config


EXPERIMENT_BASE = PROJECT_ROOT / "runs" / "qa_agent_exp" / "seed_42"
CONFIG_PATH = str(PROJECT_ROOT / "config" / "model_eval_agent.yaml")
REGISTRY_PATH = str(PROJECT_ROOT / "config" / "model_registry.yaml")
TARGET_RUN_PREFIXES = [
    "A_direct_multiple_choice_",
    "A_direct_qa_",
]


def _latest_run_for_prefix(prefix: str) -> Path | None:
    candidates = [
        d for d in EXPERIMENT_BASE.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def resolve_experiment_dirs() -> list[Path]:
    latest_dirs = []
    for prefix in TARGET_RUN_PREFIXES:
        latest = _latest_run_for_prefix(prefix)
        if latest is not None:
            latest_dirs.append(latest)
    if latest_dirs:
        return latest_dirs
    return sorted([d for d in EXPERIMENT_BASE.iterdir() if d.is_dir()], key=lambda p: p.name)


def collect_selected_question_count(run_dir: Path) -> int:
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


def collect_evaluation_report(run_dir: Path) -> dict:
    report_path = run_dir / "evaluation" / "evaluation_report.json"
    if not report_path.exists():
        return {}
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("  EVALUATE ALL — Summary Report")
    print("=" * 90)

    total_questions = sum(r["num_questions"] for r in results)
    total_groups = len(results)

    print(f"\n{'Group':<50} {'Selected Qs':>12} {'Eval Qs':>10} {'Models':>8}")
    print("-" * 90)
    for row in results:
        print(
            f"{row['group']:<50} {row['selected_questions']:>12} "
            f"{row['num_questions']:>10} {row['num_models']:>8}"
        )
    print("-" * 90)
    print(f"{'TOTAL':<50} {'':>12} {total_questions:>10} {'':>8}")
    print(f"\nGroups evaluated: {total_groups}\n")


async def main() -> None:
    if not EXPERIMENT_BASE.exists():
        logger.error("Experiment base not found: {}", EXPERIMENT_BASE)
        sys.exit(1)

    config = load_model_eval_config(CONFIG_PATH)

    orig_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)

    try:
        results: list[dict] = []
        dirs = resolve_experiment_dirs()

        for i, dir_path in enumerate(dirs, 1):
            shared_state_path = dir_path / "shared_state.json"
            if not shared_state_path.exists():
                logger.warning("No shared_state.json, skipping: {}", dir_path.name)
                continue

            logger.info("=" * 60)
            logger.info("[{}/{}] {}", i, len(dirs), dir_path.name)
            logger.info("=" * 60)

            try:
                result = await run_model_eval_agent_from_shared_state(
                    shared_state_path=str(shared_state_path),
                    config=config,
                    registry_path=str(REGISTRY_PATH),
                )

                evaluation_report = collect_evaluation_report(dir_path)
                entry = {
                    "group": dir_path.name,
                    "selected_questions": collect_selected_question_count(dir_path),
                    "num_questions": result.get("num_questions", evaluation_report.get("num_questions", 0)),
                    "num_models": result.get("num_models", evaluation_report.get("num_models", 0)),
                    "output_root": result.get("output_root", evaluation_report.get("output_root", "")),
                }
                results.append(entry)
                logger.info(
                    "  selected_questions={} eval_questions={} models={}",
                    entry["selected_questions"],
                    entry["num_questions"],
                    entry["num_models"],
                )
            except Exception:
                logger.exception("Failed to evaluate {}", dir_path.name)
    finally:
        os.chdir(orig_cwd)

    print_summary(results)

    report_out = EXPERIMENT_BASE / "evaluate_report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config_path": CONFIG_PATH,
                "registry_path": REGISTRY_PATH,
                "target_run_dirs": [str(p) for p in dirs],
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Report saved to {}", report_out)


if __name__ == "__main__":
    asyncio.run(main())
