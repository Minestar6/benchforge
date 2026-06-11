"""Storage helpers for adaptive_qa_agent artifacts."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchforge.schemas import AgentStatus, SharedState

from .state import FeedbackState, ModeState, export_adaptive_metrics


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _mode_dir(task_id: str, run_id: str, mode: str) -> Path:
    return Path("runs") / task_id / run_id / "adaptive" / mode


def _adaptive_dir(task_id: str, run_id: str) -> Path:
    return Path("runs") / task_id / run_id / "adaptive"


def _candidate_to_dict(candidate) -> dict[str, Any]:
    if is_dataclass(candidate):
        return asdict(candidate)
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump()
    if hasattr(candidate, "__dict__"):
        return dict(candidate.__dict__)
    return dict(candidate)


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, SimpleNamespace):
        return {k: _to_plain_data(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(v) for v in value]
    return value


def save_mode_outputs(
    task_id: str,
    run_id: str,
    mode: str,
    mode_state: ModeState,
    feedback_state: FeedbackState,
) -> None:
    base = _mode_dir(task_id, run_id, mode)

    _write_json(base / "candidate_pool.json", [_candidate_to_dict(c) for c in mode_state.candidates])
    _write_jsonl(base / "round_trace.jsonl", mode_state.trace)
    _write_json(base / "failures.json", mode_state.failure_records)
    _write_json(base / "mode_state.json", {
        "mode": mode_state.mode,
        "metrics": mode_state.export_metrics(),
        "stopped_reason": mode_state.stopped_reason,
        "candidate_count": len(mode_state.candidates),
    })
    _write_json(base / "metrics.json", export_adaptive_metrics(mode_state, feedback_state))


def save_global_outputs(
    task_id: str,
    run_id: str,
    mode_states: dict[str, ModeState],
    feedback_states: dict[str, FeedbackState],
) -> None:
    base = _adaptive_dir(task_id, run_id)
    _write_json(base / "global_state.json", {
        "total_candidates": sum(len(state.candidates) for state in mode_states.values()),
        "total_accepted": sum(state.accepted_count for state in mode_states.values()),
        "total_failures": sum(state.failures for state in mode_states.values()),
        "total_used_chunk_combinations": sum(len(state.used_chunk_combinations) for state in mode_states.values()),
        "modes": {
            mode: {
                "state_metrics": state.export_metrics(),
                "feedback_metrics": feedback_states[mode].export_metrics(),
            }
            for mode, state in mode_states.items()
        },
    })


def save_generation_report(
    task_id: str,
    run_id: str,
    mode_states: dict[str, ModeState],
    feedback_states: dict[str, FeedbackState],
) -> dict[str, Any]:
    report = {
        "task_id": task_id,
        "run_id": run_id,
        "modes": {
            mode: {
                "accepted_count": state.accepted_count,
                "candidate_count": len(state.candidates),
                "target_candidates": state.target_candidates,
                "stop_reason": state.stopped_reason,
                "metrics": export_adaptive_metrics(state, feedback_states[mode]),
            }
            for mode, state in mode_states.items()
        },
        "total_candidates": sum(len(state.candidates) for state in mode_states.values()),
        "total_accepted": sum(state.accepted_count for state in mode_states.values()),
    }
    _write_json(_adaptive_dir(task_id, run_id) / "generation_report.json", report)
    return report


def save_shared_state(blueprint: Any) -> SharedState:
    base = Path("runs") / blueprint.task_id / blueprint.run_id
    artifacts = {
        f"{mode}_candidate_pool": str(base / "adaptive" / mode / "candidate_pool.json")
        for mode in blueprint.modes
    }
    artifacts.update({
        "llm_calls": str(base / "llm_calls.jsonl"),
        "generation_report": str(base / "adaptive" / "generation_report.json"),
        "validated_questions": str(base / "validation" / "validated_questions.jsonl"),
    })
    state = SharedState(
        task_id=blueprint.task_id,
        run_id=blueprint.run_id,
        blueprint=_to_plain_data(blueprint),
        blueprint_cache=_to_plain_data(blueprint),
        artifacts=artifacts,
        agent_status={
            "generation": AgentStatus.COMPLETED,
            "verification": AgentStatus.PENDING,
            "evaluation": AgentStatus.PENDING,
        },
    )
    _write_json(base / "shared_state.json", state.model_dump())
    return state
