"""Storage helpers: save mode outputs, global outputs, generation report."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from benchforge.utils.shared_state import build_shared_state, save_shared_state as save_ss

from .state import GlobalState, ModeState, CandidateStatus


def _mode_min_candidate_target(mode_cfg: Any, config: Any) -> int:
    return math.ceil(mode_cfg.count * config.candidate_pool.min_candidate_multiplier)


def _mode_max_candidate_target(mode_cfg: Any, config: Any) -> int:
    return math.ceil(mode_cfg.count * config.candidate_pool.max_candidate_multiplier)


def _mode_min_per_difficulty_targets(mode_cfg: Any, config: Any) -> dict[str, int]:
    total = _mode_min_candidate_target(mode_cfg, config)
    return {
        diff: math.ceil(total * ratio)
        for diff, ratio in (mode_cfg.difficulty_distribution or {}).items()
    }


def _default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)


def _append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")


def append_round_plan(task_id: str, run_id: str, mode: str, round_plan: Any) -> None:
    path = Path("runs") / task_id / run_id / mode / "round_plan.jsonl"
    _append_jsonl(path, dataclasses.asdict(round_plan) if dataclasses.is_dataclass(round_plan) else round_plan)


def append_round_feedback(task_id: str, run_id: str, mode: str, feedback: Any) -> None:
    path = Path("runs") / task_id / run_id / mode / "round_feedback.jsonl"
    _append_jsonl(path, dataclasses.asdict(feedback) if dataclasses.is_dataclass(feedback) else feedback)


def save_mode_metrics(
    task_id: str,
    run_id: str,
    mode: str,
    mode_state: ModeState,
    mode_cfg: Any = None,
    config: Any = None,
) -> None:
    hard_target = (
        mode_cfg.difficulty_distribution.get("hard", 0.6) if mode_cfg is not None else 0.6
    )
    acc_diff = mode_state.get_difficulty_counts(CandidateStatus.ACCEPTED)
    acc_topic = mode_state.get_topic_counts(CandidateStatus.ACCEPTED)
    min_target = _mode_min_candidate_target(mode_cfg, config) if mode_cfg is not None and config is not None else 0
    max_target = _mode_max_candidate_target(mode_cfg, config) if mode_cfg is not None and config is not None else 0
    min_diff_targets = (
        _mode_min_per_difficulty_targets(mode_cfg, config)
        if mode_cfg is not None and config is not None
        else {}
    )
    min_target_reached = all(
        acc_diff.get(diff, 0) >= target for diff, target in min_diff_targets.items()
    ) if min_diff_targets else False
    metrics = {
        "candidate_count": mode_state.accepted_count,
        "accepted_count": mode_state.accepted_count,
        "rejected_count": mode_state.rejected_count,
        "accept_rate": round(mode_state.accept_rate(), 4),
        "topic_distribution": acc_topic,
        "difficulty_distribution": acc_diff,
        "hard_gap": round(mode_state.hard_gap(hard_target), 4),
        "stopped_reason": mode_state.stopped_reason,
        "min_candidate_target": min_target,
        "max_candidate_target": max_target,
        "min_difficulty_targets": min_diff_targets,
        "min_target_reached": min_target_reached,
        "max_target_reached": mode_state.accepted_count >= max_target if max_target > 0 else False,
        "terminal_hard_repair": {
            "executed": mode_state.terminal_hard_repair_executed,
            "rounds": mode_state.terminal_hard_repair_rounds,
            "generated_count": mode_state.terminal_hard_repair_generated_count,
            "accepted_count": mode_state.terminal_hard_repair_accepted_count,
        },
    }
    _write_json(Path("runs") / task_id / run_id / mode / "mode_metrics.json", metrics)


def save_mode_outputs(task_id: str, run_id: str, mode: str, mode_state: ModeState) -> None:
    base = Path("runs") / task_id / run_id / mode

    def _serialize(r):
        return {
            "question_id": r.question_id,
            "question": r.question,
            "answer": r.answer,
            "topic": r.topic,
            "difficulty": r.difficulty,
            "status": r.status.value,
            "source_round": r.source_round,
            "source_strategy": r.source_strategy,
            "chunk_ids": r.chunk_ids,
            "reject_reason": r.reject_reason,
            "parent_question_id": r.parent_question_id,
            "choices": r.choices,

            # Prompt output metadata
            "question_mode": r.question_mode,
            "question_type": r.question_type,
            "required_capability": r.required_capability,
            "estimated_difficulty": r.estimated_difficulty,
            "citations": r.citations,
            "thought_process": r.thought_process,

            # Execution metadata
            "llm_call_id": r.llm_call_id,
            "trace_call_id": r.trace_call_id,
            "chunks": r.chunks,
            "generation_round": r.generation_round,

            # Preserve original model output
            "raw_item": r.raw_item,
       }

    accepted = [r for r in mode_state.candidate_questions if r.status == CandidateStatus.ACCEPTED]
    _write_json(base / "candidate_pool.json", [_serialize(r) for r in accepted])
    _write_json(base / "candidate_pool_all.json", [_serialize(r) for r in mode_state.candidate_questions])
    _write_json(base / "generation_trace.json", mode_state.trace)
    _write_json(base / "failures.json", mode_state.failures)
    _write_json(base / "mode_state.json", {
        "mode": mode_state.mode,
        "candidate_count": mode_state.accepted_count,
        "total_record_count": len(mode_state.candidate_questions),
        "accepted_count": mode_state.accepted_count,
        "rejected_count": mode_state.rejected_count,
        "difficulty_counts": mode_state.get_difficulty_counts(CandidateStatus.ACCEPTED),
        "topic_counts": mode_state.get_topic_counts(CandidateStatus.ACCEPTED),
        "stopped_reason": mode_state.stopped_reason,
        "terminal_hard_repair": {
            "executed": mode_state.terminal_hard_repair_executed,
            "rounds": mode_state.terminal_hard_repair_rounds,
            "generated_count": mode_state.terminal_hard_repair_generated_count,
            "accepted_count": mode_state.terminal_hard_repair_accepted_count,
        },
    })


def save_global_outputs(task_id: str, run_id: str, global_state: GlobalState) -> None:
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
    from .planner import mode_candidate_target, mode_max_candidate_target

    modes_summary = {}
    total_candidates = 0

    for mode, ms in mode_states.items():
        min_target = mode_candidate_target(mode_cfgs[mode], config)
        max_target = mode_max_candidate_target(mode_cfgs[mode], config)
        difficulty_counts = ms.get_difficulty_counts(CandidateStatus.ACCEPTED)
        min_diff_targets = _mode_min_per_difficulty_targets(mode_cfgs[mode], config)
        modes_summary[mode] = {
            "candidate_count": ms.accepted_count,
            "accepted_count": ms.accepted_count,
            "min_candidate_target": min_target,
            "max_candidate_target": max_target,
            "target_candidate_count": min_target,
            "difficulty_counts": difficulty_counts,
            "min_difficulty_targets": min_diff_targets,
            "min_target_reached": all(
                difficulty_counts.get(diff, 0) >= target
                for diff, target in min_diff_targets.items()
            ) if min_diff_targets else False,
            "max_target_reached": ms.accepted_count >= max_target,
            "stopped_reason": ms.stopped_reason,
            "terminal_hard_repair": {
                "executed": ms.terminal_hard_repair_executed,
                "rounds": ms.terminal_hard_repair_rounds,
                "generated_count": ms.terminal_hard_repair_generated_count,
                "accepted_count": ms.terminal_hard_repair_accepted_count,
            },
        }
        total_candidates += ms.accepted_count

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
