"""Run a small real qa_agent -> verify_agent smoke experiment.

Usage:
    cd /Users/zhaoziqing/Desktop/benchforge
    python test/run_qa_verify_smoke.py
    python test/run_qa_verify_smoke.py --topic "Artificial Intelligence"
    python test/run_qa_verify_smoke.py --qa-count 5 --mc-count 5 --max-rounds 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from loguru import logger

from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
from benchforge.agents.verify_agent.config_loader import load_verify_agent_config
from benchforge.config.config import load_dotenv
from benchforge.models.loader import ModelLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qa_agent + verify_agent smoke test.")
    parser.add_argument("--task-id", default="qa_verify_smoke", help="Task directory under runs/.")
    parser.add_argument(
        "--run-id",
        default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Run directory name.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Topic to generate from. Repeatable. Defaults to Artificial Intelligence.",
    )
    parser.add_argument("--qa-count", type=int, default=5, help="Target QA question count.")
    parser.add_argument("--mc-count", type=int, default=5, help="Target MC question count.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max rounds per mode.")
    parser.add_argument("--language", default="en", help="Blueprint language.")
    parser.add_argument(
        "--qa-config",
        default=str(project_root / "config" / "qa_agent.yaml"),
        help="Path to qa_agent config.",
    )
    parser.add_argument(
        "--verify-config",
        default=str(project_root / "config" / "verify_agent.yaml"),
        help="Path to verify_agent config.",
    )
    parser.add_argument(
        "--registry-path",
        default=str(project_root / "config" / "model_registry.yaml"),
        help="Path to model registry.",
    )
    return parser.parse_args()


def require_runtime_env() -> None:
    load_dotenv(project_root / ".env")
    if not os.getenv("CUSTOM_API_KEY"):
        raise RuntimeError("CUSTOM_API_KEY is not set in .env or environment.")
    if not os.getenv("CUSTOM_API_BASE_URL"):
        raise RuntimeError("CUSTOM_API_BASE_URL is not set in .env or environment.")


def build_blueprint(args: argparse.Namespace) -> Blueprint:
    topics = args.topics or ["Artificial Intelligence"]
    return Blueprint(
        task_id=args.task_id,
        run_id=args.run_id,
        language=args.language,
        topics=topics,
        modes={
            "qa": ModeCfg(
                count=args.qa_count,
                max_rounds=args.max_rounds,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            ),
            "multiple_choice": ModeCfg(
                count=args.mc_count,
                max_rounds=args.max_rounds,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            ),
        },
    )


def load_verify_model_client(verify_config_path: str, registry_path: str):
    verify_cfg = load_verify_agent_config(verify_config_path)
    registry = load_model_registry(registry_path)
    model_key = verify_cfg.llm_validation.model
    if model_key not in registry:
        raise KeyError(
            f"verify_agent model '{model_key}' not found in registry {registry_path}. "
            f"Available: {list(registry.keys())}"
        )
    return verify_cfg, ModelLoader.load_model(registry[model_key])


def print_generation_summary(report: dict) -> None:
    print("\n=== Generation Report ===")
    print(f"task_id : {report['task_id']}")
    print(f"run_id  : {report['run_id']}")
    for mode, state in report["modes"].items():
        print(
            f"{mode:16s} {state['candidate_count']:>3d}/{state['target_candidate_count']:<3d} "
            f"stopped={state['stopped_reason']}"
        )
    print(f"total_candidates : {report['total_candidates']}")


def print_verify_summary(task_id: str, run_id: str, result, report_path: Path) -> None:
    print("\n=== Verification Result ===")
    print(f"selected_question_ids : {len(result.selected_question_ids)}")
    print(f"failed_by_stage       : {result.failed_by_stage}")
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"citation_passed       : {data.get('citation_passed')}")
        print(f"llm_passed            : {data.get('llm_passed')}")
        print(f"final_selected        : {data.get('final_selected')}")
        print(f"llm_usage             : {data.get('llm_usage')}")
    print(f"run_dir               : runs/{task_id}/{run_id}")
    print(f"validation_report     : {report_path}")


async def main() -> None:
    args = parse_args()
    require_runtime_env()

    run_dir = project_root / "runs" / args.task_id / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(run_dir / "smoke.log"), level="DEBUG", encoding="utf-8")

    blueprint = build_blueprint(args)
    logger.info("Starting qa_agent smoke run: task={} run={}", blueprint.task_id, blueprint.run_id)
    generation_report = await run_generation_agent(
        blueprint=blueprint,
        config_path=args.qa_config,
        registry_path=args.registry_path,
    )
    print_generation_summary(generation_report)

    shared_state_path = run_dir / "shared_state.json"
    if not shared_state_path.exists():
        raise FileNotFoundError(f"shared_state.json not found: {shared_state_path}")

    verify_cfg, verify_model_client = load_verify_model_client(args.verify_config, args.registry_path)
    logger.info("Starting verify_agent on {}", shared_state_path)
    verify_result = await run_verify_agent_from_shared_state(
        shared_state_path=shared_state_path,
        config=verify_cfg,
        model_client=verify_model_client,
    )

    validation_report = run_dir / "validation" / "validation_report.json"
    print_verify_summary(blueprint.task_id, blueprint.run_id, verify_result, validation_report)


if __name__ == "__main__":
    asyncio.run(main())
