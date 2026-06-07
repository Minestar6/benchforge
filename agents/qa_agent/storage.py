"""Storage helpers: save mode outputs, global outputs, generation report."""

import json
from pathlib import Path
from typing import Any

from benchforge.utils.shared_state import build_shared_state, save_shared_state as save_ss

from .state import GlobalState, ModeState


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_mode_outputs(task_id: str, run_id: str, mode: str, mode_state: ModeState) -> None:
    """Write mode-specific files to runs/{task_id}/{run_id}/{mode}/."""
    base = Path("runs") / task_id / run_id / mode

    _write_json(base / "candidate_pool.json", mode_state.candidate_questions)
    _write_json(base / "generation_trace.json", mode_state.trace)
    _write_json(base / "failures.json", mode_state.failures)
    _write_json(base / "mode_state.json", {
        "mode": mode_state.mode,
        "candidate_count": len(mode_state.candidate_questions),
        "difficulty_counts": mode_state.difficulty_counts,
        "topic_counts": mode_state.topic_counts,
        "stopped_reason": mode_state.stopped_reason,
    })


def save_global_outputs(task_id: str, run_id: str, global_state: GlobalState) -> None:
    """Write global files to runs/{task_id}/{run_id}/."""
    base = Path("runs") / task_id / run_id

    _write_json(base / "used_chunks.json", {
        "used_chunk_combinations": [list(c) for c in global_state.used_chunk_combinations],
        "chunk_usage_counts": global_state.chunk_usage_counts,
    })
    _write_json(base / "global_state.json", {
        "global_failures": global_state.global_failures,
        "total_used_combinations": len(global_state.used_chunk_combinations),
    })


def save_shared_state(blueprint: Any) -> dict:
    """Write runs/{task_id}/{run_id}/shared_state.json for downstream agents.

    委托给 benchforge.utils.shared_state 统一实现。
    """
    state = build_shared_state(
        task_id=blueprint.task_id,
        run_id=blueprint.run_id,
        blueprint=blueprint,
    )
    save_ss(state, Path("runs") / blueprint.task_id / blueprint.run_id)
    return state.model_dump()


def save_generation_report(
    task_id: str,
    run_id: str,
    global_state: GlobalState,
    mode_states: dict[str, ModeState],
    mode_cfgs: dict[str, Any],
    config: Any,
) -> dict:
    """Write runs/{task_id}/{run_id}/generation_report.json and return the report dict."""
    from .planner import mode_candidate_target

    modes_summary = {}
    total_candidates = 0

    for mode, ms in mode_states.items():
        target = mode_candidate_target(mode_cfgs[mode], config)
        modes_summary[mode] = {
            "candidate_count": len(ms.candidate_questions),
            "target_candidate_count": target,
            "stopped_reason": ms.stopped_reason,
        }
        total_candidates += len(ms.candidate_questions)

    report = {
        "task_id": task_id,
        "run_id": run_id,
        "modes": modes_summary,
        "total_candidates": total_candidates,
        "global_used_chunk_combinations": len(global_state.used_chunk_combinations),
        "global_failures": global_state.global_failures,
    }
    _write_json(Path("runs") / task_id / run_id / "generation_report.json", report)
    return report
