"""Top-level BenchForge app entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from benchforge.agents.planner_agent.blueprint_synthesizer import synthesize_global_blueprint
from benchforge.agents.planner_agent.config_loader import (
    save_global_blueprint,
    load_global_blueprint,
    load_planner_state,
)
from benchforge.agents.planner_agent.planner import run_planner
from benchforge.agents.planner_agent.schema import UserIntent, PlannerState


async def run_benchforge(
    intent: UserIntent,
    base_config_dir: str | Path,
    registry_path: str | Path,
    planner_state_dir: str | Path | None = None,
    planner_model_name: str | None = None,
    planner_model_client=None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run BenchForge from user intent through planner orchestration.

    Args:
        resume_from: 从已有 planner state 目录恢复运行，跳过蓝图合成直接加载已有状态。
    """
    resume_state: PlannerState | None = None

    if resume_from:
        resume_from = Path(resume_from)
        blueprint = load_global_blueprint(resume_from / "global_blueprint.json")
        state_dir = resume_from
        # 找最新的 after_round 快照
        state_files = sorted(state_dir.glob("planner_state_after_round_*.json"))
        if state_files:
            resume_state = load_planner_state(state_files[-1])
            logger.info(f"[App] Resuming from {state_files[-1].name}")
        else:
            logger.info("[App] No round snapshots found, starting from round 0")
        global_blueprint_path = state_dir / "global_blueprint.json"
        print(f"[resume] state_dir={state_dir}")
    else:
        blueprint = await synthesize_global_blueprint(
            intent=intent,
            registry_path=registry_path,
            planner_model_name=planner_model_name,
            model_client=planner_model_client,
        )

        state_dir = Path(planner_state_dir) if planner_state_dir else Path("runs") / blueprint.task_id / "planner"
        state_dir.mkdir(parents=True, exist_ok=True)

        global_blueprint_path = state_dir / "global_blueprint.json"
        save_global_blueprint(blueprint, global_blueprint_path)
        print(f"[start] state_dir={state_dir}")

    final_state = await run_planner(
        global_blueprint=blueprint,
        base_config_dir=base_config_dir,
        registry_path=registry_path,
        state_dir=state_dir,
        resume_state=resume_state,
    )

    return {
        "blueprint": blueprint,
        "planner_state": final_state,
        "state_dir": state_dir,
        "global_blueprint_path": global_blueprint_path,
    }
