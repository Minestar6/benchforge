import math
from pathlib import Path
from types import SimpleNamespace

import yaml
from loguru import logger

from agents.verify_agent.agent import VerifyAgent
from agents.verify_agent.schema import ValidationBlueprintView, ModeCfg

from .state import ModeState, FeedbackState
from .decision import decide
from .executor import Executor
from .feedback_mapper import map_from_task_result
from .prompts import PromptBuilder


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
        executor = Executor(
            evidence_manager=evidence_manager,
            model_client=model_client,
            evolution_tool=evolution_tool,
            blueprint=blueprint,
            config=config,
            adaptive_config=adaptive_config,
            prompts=prompts,
        )

        while True:
            if state.round_in_mode >= state.max_rounds:
                logger.info(f"[AdaptiveQAAgent] mode={mode} stopped: max_rounds")
                break
            if state.accepted_count >= state.target_candidates:
                logger.info(f"[AdaptiveQAAgent] mode={mode} stopped: target reached")
                break
            if state.consecutive_empty >= adaptive_config.stop.max_empty_rounds:
                logger.info(f"[AdaptiveQAAgent] mode={mode} stopped: max_empty_rounds")
                break
            if state.failures >= adaptive_config.stop.max_failures:
                logger.info(f"[AdaptiveQAAgent] mode={mode} stopped: max_failures")
                break

            action = decide(state, feedback, adaptive_config)
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
                logger.info(f"[AdaptiveQAAgent] round={state.round_in_mode} accepted={state.accepted_count}/{target} stats={round_stats}")
            else:
                state.consecutive_empty += 1
                state.failures += 1

            state.round_in_mode += 1


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
