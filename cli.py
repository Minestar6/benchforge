"""BenchForge CLI."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchforge.agents.planner_agent.schema import UserIntent
from benchforge.app import run_benchforge
from benchforge.config.config import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BenchForge from user intent.")
    parser.add_argument("goal", nargs="?", help="Natural-language benchmark goal.")
    parser.add_argument("--user-goal", help="Natural-language benchmark goal.")
    parser.add_argument("--task-id", help="Optional stable task id.")
    parser.add_argument("--language", default="zh", help="Benchmark language.")
    parser.add_argument("--seed-topic", action="append", default=[], help="Optional seed topic. Repeatable.")
    parser.add_argument("--qa-target", type=int, default=20, help="Target number of QA questions.")
    parser.add_argument("--mc-target", type=int, default=10, help="Target number of multiple-choice questions.")
    parser.add_argument("--candidate-model", action="append", default=[], help="Candidate model registry key. Repeatable.")
    parser.add_argument("--judge-model", default="deepseek-v4", help="Judge model registry key.")
    parser.add_argument("--planner-model", default="deepseek-v4", help="Planner synthesis model registry key.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum planner rounds.")
    parser.add_argument("--min-selected-per-round", type=int, default=5, help="Minimum selected questions per round.")
    parser.add_argument("--max-total-tokens", type=int, help="Global token budget.")
    parser.add_argument("--config-dir", default="config", help="Base config directory.")
    parser.add_argument("--registry-path", default="config/model_registry.yaml", help="Model registry path.")
    parser.add_argument("--planner-state-dir", help="Planner state output directory.")
    parser.add_argument("--resume-from", help="Resume from a previous planner state directory.")
    parser.add_argument("--resume-latest", action="store_true", help="Auto-resume from the latest run under runs/.")
    return parser


def build_user_intent(args: argparse.Namespace) -> UserIntent:
    user_goal = args.user_goal or args.goal or ""
    if not user_goal and not args.resume_from and not args.resume_latest:
        raise ValueError("user goal is required (unless --resume-from or --resume-latest is used)")
    return UserIntent(
        user_goal=user_goal,
        task_id=args.task_id,
        language=args.language,
        seed_topics=list(args.seed_topic),
        qa_target=args.qa_target,
        multiple_choice_target=args.mc_target,
        candidate_model_names=list(args.candidate_model),
        judge_model_name=args.judge_model,
        planner_model_name=args.planner_model,
        max_rounds=args.max_rounds,
        min_selected_per_round=args.min_selected_per_round,
        max_total_tokens=args.max_total_tokens,
    )


def _find_latest_run() -> Path | None:
    """在 runs/ 下找最新的包含 global_blueprint.json 的 planner 子目录。"""
    runs_dir = Path("runs")
    if not runs_dir.is_dir():
        return None
    blueprints = sorted(
        runs_dir.glob("*/planner/global_blueprint.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return blueprints[0].parent if blueprints else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parent / ".env")

    resume_from: Path | None = None
    if args.resume_latest:
        resume_from = _find_latest_run()
        if resume_from is None:
            parser.error("--resume-latest: no previous runs found under runs/")
    elif args.resume_from:
        resume_from = Path(args.resume_from)

    try:
        intent = build_user_intent(args)
    except ValueError as exc:
        parser.error(str(exc))

    result = asyncio.run(
        run_benchforge(
            intent=intent,
            base_config_dir=Path(args.config_dir),
            registry_path=Path(args.registry_path),
            planner_state_dir=Path(args.planner_state_dir) if args.planner_state_dir else None,
            planner_model_name=args.planner_model,
            resume_from=resume_from,
        )
    )
    print(f"GlobalBlueprint: {result['global_blueprint_path']}")
    print(f"Planner state dir: {result['state_dir']}")
    print(f"Final rounds: {result['planner_state'].current_round}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
