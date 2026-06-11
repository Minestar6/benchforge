import math
from pathlib import Path
from types import SimpleNamespace

import yaml
from loguru import logger

from agents.verify_agent.agent import VerifyAgent
from agents.verify_agent.schema import ValidationBlueprintView, ModeCfg

from .state import ModeState, FeedbackState
from .decision import DecisionEngine
from .stop_policy import StopPolicy
from .executor import Executor
from .feedback_mapper import map_from_task_result
from .prompts import PromptBuilder
from .storage import (
    save_mode_outputs,
    save_global_outputs,
    save_generation_report,
    save_shared_state,
)


def _load_adaptive_config(path: str | Path | None = None) -> SimpleNamespace:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return _to_namespace(data)


def _to_namespace(d) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d


async def run_adaptive_generation_agent(
    blueprint,
    config,
    adaptive_config,
    evidence_manager,
    model_client,
    verify_config,
    evolution_tool=None,
):
    prompts = PromptBuilder(blueprint.language)
    verify_agent = VerifyAgent(config=verify_config, model_client=model_client)
    from benchforge.utils.run_context import RunContext
    run_ctx = RunContext(task_id=blueprint.task_id, run_id=blueprint.run_id)
    tracer = run_ctx.create_tracer(agent="adaptive_qa_agent", stage="generation")

    validation_blueprint = ValidationBlueprintView(
        topics=blueprint.topics,
        modes={
            mode: ModeCfg(
                count=mode_cfg.count,
                difficulty_distribution=mode_cfg.difficulty_distribution,
            )
            for mode, mode_cfg in blueprint.modes.items()
        },
    )

    mode_states: dict[str, ModeState] = {}
    feedback_states: dict[str, FeedbackState] = {}

    for mode, mode_cfg in blueprint.modes.items():
        target = math.ceil(mode_cfg.count * config.candidate_pool.target_multiplier)
        logger.info(f"[AdaptiveQAAgent] mode={mode} target={target}")

        state = ModeState(
            mode=mode,
            all_topics=blueprint.topics,
            target_hard_ratio=mode_cfg.difficulty_distribution.get("hard", 0.3),
            target_candidates=target,
            max_rounds=mode_cfg.max_rounds,
        )
        feedback = FeedbackState(window_size=adaptive_config.execution.feedback_window)
        mode_states[mode] = state
        feedback_states[mode] = feedback
        executor = Executor(
            evidence_manager=evidence_manager,
            model_client=model_client,
            evolution_tool=evolution_tool,
            blueprint=blueprint,
            config=config,
            adaptive_config=adaptive_config,
            prompts=prompts,
            tracer=tracer,
        )
        decision_engine = DecisionEngine(adaptive_config)
        stop_policy = StopPolicy(adaptive_config)

        while True:
            stop = stop_policy.check(state)
            if stop.should_stop:
                state.stopped_reason = stop.reason
                logger.info(f"[AdaptiveQAAgent] mode={mode} stopped: {stop.reason}")
                break

            action = decision_engine.decide(state, feedback)
            state.record_action(action.action_type)
            logger.debug(f"[AdaptiveQAAgent] round={state.round_in_mode} action={action.action_type} reason={action.reason}")

            candidates = await executor.run(action, state)

            if candidates:
                result = await verify_agent.run(
                    task_id=blueprint.task_id,
                    run_id=blueprint.run_id,
                    blueprint=validation_blueprint,
                    candidates=candidates,
                )

                validation_dir = Path("runs") / blueprint.task_id / blueprint.run_id / "validation"
                mapped = map_from_task_result(result, candidates, validation_dir)

                round_stats = _aggregate(mapped, is_evolution=(action.action_type == "evolve"))
                feedback.push(round_stats)
                state.update(mapped)
                state.trace.append({
                    "round_in_mode": state.round_in_mode,
                    "action": {
                        "action_type": action.action_type,
                        "reason": action.reason,
                        "topics": list(action.topics),
                        "difficulty": action.difficulty,
                        "evidence_strategy": action.evidence_strategy,
                    },
                    "execution": executor.last_run_details,
                    "round_stats": round_stats,
                    "metrics_snapshot": {
                        **state.export_metrics(),
                        **feedback.export_metrics(),
                    },
                })
                logger.info(f"[AdaptiveQAAgent] round={state.round_in_mode} accepted={state.accepted_count}/{target} stats={round_stats}")
            else:
                state.consecutive_empty += 1
                state.failures += 1
                state.failure_records.append({
                    "round_in_mode": state.round_in_mode,
                    "action_type": action.action_type,
                    "reason": "no_candidates_generated",
                    "execution": executor.last_run_details,
                })
                state.trace.append({
                    "round_in_mode": state.round_in_mode,
                    "action": {
                        "action_type": action.action_type,
                        "reason": action.reason,
                        "topics": list(action.topics),
                        "difficulty": action.difficulty,
                        "evidence_strategy": action.evidence_strategy,
                    },
                    "execution": executor.last_run_details,
                    "round_stats": {"total": 0, "accepted": 0},
                    "metrics_snapshot": {
                        **state.export_metrics(),
                        **feedback.export_metrics(),
                    },
                })

            state.round_in_mode += 1

        save_mode_outputs(blueprint.task_id, blueprint.run_id, mode, state, feedback)

    save_global_outputs(blueprint.task_id, blueprint.run_id, mode_states, feedback_states)
    save_generation_report(blueprint.task_id, blueprint.run_id, mode_states, feedback_states)
    save_shared_state(blueprint)


def _aggregate(mapped: list, is_evolution: bool = False) -> dict:
    stats = {"total": len(mapped), "accepted": 0}
    for item in mapped:
        if item.accepted:
            stats["accepted"] += 1
        elif item.reject_reason:
            stats[item.reject_reason] = stats.get(item.reject_reason, 0) + 1
    if is_evolution:
        stats["evolution_failed"] = stats["total"] - stats["accepted"]
    return stats
