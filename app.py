"""Top-level BenchForge app entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchforge.agents.planner_agent.blueprint_synthesizer import synthesize_global_blueprint
from benchforge.agents.planner_agent.config_loader import save_global_blueprint
from benchforge.agents.planner_agent.planner import run_planner
from benchforge.agents.planner_agent.schema import UserIntent


async def run_benchforge(
    intent: UserIntent,
    base_config_dir: str | Path,
    registry_path: str | Path,
    planner_state_dir: str | Path | None = None,
    planner_model_name: str | None = None,
    planner_model_client=None,
) -> dict[str, Any]:
    """Run BenchForge from user intent through planner orchestration."""
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

    final_state = await run_planner(
        global_blueprint=blueprint,
        base_config_dir=base_config_dir,
        registry_path=registry_path,
        state_dir=state_dir,
    )

    return {
        "blueprint": blueprint,
        "planner_state": final_state,
        "state_dir": state_dir,
        "global_blueprint_path": global_blueprint_path,
    }
