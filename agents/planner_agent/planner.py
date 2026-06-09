"""PlannerAgent 跨轮主循环（docs/plan-runtime-aligned.md § 6.3）。

职责：
1. 初始化 PlannerState
2. 多轮循环：判断终止条件 → LLM 提案 topics → 规则构建 RoundSpec → 调用 orchestrator → 更新状态
3. 返回最终 PlannerState
"""

import json
import inspect
from pathlib import Path
from loguru import logger

from benchforge.models.loader import ModelLoader
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry

from .schema import GlobalBlueprint, PlannerState, RoundSpec, RoundPlannerHints, RunHistoryEntry
from .config_loader import initialize_planner_state, save_planner_state
from .orchestrator import execute_round


async def run_planner(
    global_blueprint: GlobalBlueprint,
    base_config_dir: str | Path,
    registry_path: str | Path,
    state_dir: str | Path,
) -> PlannerState:
    """执行多轮规划与编排。

    Args:
        global_blueprint: 全局蓝图
        base_config_dir: config/ 目录路径（包含 qa_agent.yaml 等）
        registry_path: model_registry.yaml 路径
        state_dir: planner 状态保存目录

    Returns:
        最终 PlannerState
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 state
    state = initialize_planner_state(global_blueprint)
    save_planner_state(state, state_dir / "planner_state_init.json")

    # 加载 planner 用的 LLM 模型客户端
    registry = load_model_registry(registry_path)
    planner_model_name = global_blueprint.evaluator_defaults.judge_model_name or list(registry.keys())[0]
    planner_model_cfg = registry.get(planner_model_name)
    if not planner_model_cfg:
        raise ValueError(f"Planner model '{planner_model_name}' not found in registry")
    planner_model_client = ModelLoader.load_model(planner_model_cfg)

    logger.info(
        f"[Planner] Start planning: task_id={state.task_id}, "
        f"targets=qa:{global_blueprint.final_targets.qa}/mc:{global_blueprint.final_targets.multiple_choice}"
    )

    latest_gen_fb = None
    latest_val_fb = None
    latest_eval_fb = None

    while True:
        # 1. 终止条件判断
        if _should_stop(state, global_blueprint):
            logger.info(f"[Planner] Stop condition met at round {state.current_round}")
            break

        # 2. 准备下一轮
        state.current_round += 1
        run_id = f"run_{state.current_round:03d}"

        logger.info(f"[Planner] === Round {state.current_round} start ===")

        # 3. LLM 提案 topics
        proposed_topics = await _propose_topics_with_llm(
            state, global_blueprint, planner_model_client, latest_val_fb, latest_eval_fb
        )

        # 4. 规则系统构建 RoundSpec
        round_spec = _build_next_round_spec(
            state, global_blueprint, proposed_topics, run_id, latest_gen_fb, latest_val_fb, latest_eval_fb
        )

        # 落盘 round_spec
        round_spec_path = state_dir / f"round_{state.current_round:03d}_spec.json"
        with open(round_spec_path, "w", encoding="utf-8") as f:
            json.dump(round_spec.model_dump(), f, ensure_ascii=False, indent=2)

        # 5. 执行单轮编排
        gen_fb, val_fb, eval_fb = await execute_round(
            round_spec, global_blueprint, base_config_dir, registry_path
        )
        latest_gen_fb, latest_val_fb, latest_eval_fb = gen_fb, val_fb, eval_fb

        # 6. 更新 PlannerState
        _update_planner_state(state, round_spec, gen_fb, val_fb, eval_fb)

        # 7. 保存状态
        save_planner_state(state, state_dir / f"planner_state_after_round_{state.current_round:03d}.json")

        logger.info(
            f"[Planner] Round {state.current_round} done: "
            f"completed=qa:{state.completed_targets['qa']}/mc:{state.completed_targets['multiple_choice']}, "
            f"tokens={state.resource_usage.total_tokens}"
        )

    logger.info(
        f"[Planner] Planning complete: {state.current_round} rounds, "
        f"final=qa:{state.completed_targets['qa']}/mc:{state.completed_targets['multiple_choice']}"
    )

    return state


def _should_stop(state: PlannerState, blueprint: GlobalBlueprint) -> bool:
    """判断是否满足终止条件。"""
    # max_rounds
    if state.current_round >= blueprint.stop_conditions.max_rounds:
        return True

    # targets_satisfied
    qa_target = blueprint.final_targets.qa
    mc_target = blueprint.final_targets.multiple_choice
    if (
        state.completed_targets["qa"] >= qa_target
        and state.completed_targets["multiple_choice"] >= mc_target
    ):
        return True

    # token_budget
    if blueprint.stop_conditions.max_total_tokens:
        if state.resource_usage.total_tokens >= blueprint.stop_conditions.max_total_tokens:
            return True

    return False


async def _propose_topics_with_llm(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    model_client,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> list[str]:
    """调用 LLM 提案下一轮 topics。

    当前简化实现：直接从 topic_backlog.active 取前 N 个。
    生产版本应调用 model_client.generate() 生成提案。
    """
    # 简化规则：取 backlog 的前 2-3 个 topics
    if not state.topic_backlog.active and state.topic_backlog.deferred:
        state.topic_backlog.active = list(state.topic_backlog.deferred)
        state.topic_backlog.deferred = []

    active_topics = state.topic_backlog.active
    if not active_topics:
        # 无可用 topic，返回空
        logger.warning("[Planner] No active topics in backlog")
        return []

    ranked_topics = list(active_topics)
    if latest_val_fb and latest_val_fb.by_topic:
        ranked_topics = sorted(
            active_topics,
            key=lambda topic: (
                latest_val_fb.by_topic.get(topic, {}).get("selected", 0),
                latest_val_fb.by_topic.get(topic, {}).get("avg_citation_score") or 0.0,
            ),
        )

    # 根据上一轮 feedback 决定 topic 数量
    last_entry = state.run_history[-1] if state.run_history else None
    if last_entry and last_entry.selected_count < blueprint.stop_conditions.min_selected_per_round:
        # 上一轮产出不足，扩大 topic 范围
        num_topics = min(3, len(active_topics))
    else:
        num_topics = min(2, len(active_topics))

    complete = getattr(model_client, "complete", None)
    if complete and inspect.iscoroutinefunction(complete):
        prompt = (
            "You are selecting benchmark topics for the next round.\n"
            f"Candidate topics: {ranked_topics}\n"
            f"Select {num_topics} topics.\n"
            "Return strict JSON only: {\"topics\": [\"topic1\", \"topic2\"]}"
        )
        try:
            response = await complete(
                model=getattr(model_client, "model_name", "planner"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            text = response.get("text", "").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end + 1])
                proposed = parsed.get("topics", [])
                filtered = [topic for topic in proposed if topic in ranked_topics]
                if filtered:
                    return filtered[:num_topics]
        except Exception as exc:
            logger.warning(f"[Planner] LLM topic proposal failed, fallback to deterministic ranking: {exc}")

    return ranked_topics[:num_topics]


def _build_next_round_spec(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    proposed_topics: list[str],
    run_id: str,
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> RoundSpec:
    """规则系统构建 RoundSpec（docs/plan-runtime-aligned.md § 8-9）。"""
    # 计算 target count
    remaining_qa = blueprint.final_targets.qa - state.completed_targets["qa"]
    remaining_mc = blueprint.final_targets.multiple_choice - state.completed_targets["multiple_choice"]

    # 基础目标：每轮至少满足 min_selected_per_round
    base_qa = max(blueprint.stop_conditions.min_selected_per_round, remaining_qa // 3)
    base_mc = max(blueprint.stop_conditions.min_selected_per_round, remaining_mc // 3)

    target_multiplier = 2.0
    if latest_gen_fb and latest_gen_fb.by_mode:
        min_fulfillment = min(mode_fb.fulfillment_rate for mode_fb in latest_gen_fb.by_mode.values())
        if min_fulfillment < 0.6:
            target_multiplier = 2.5
        elif min_fulfillment > 0.9:
            target_multiplier = 1.8
    elif state.run_history:
        last_entry = state.run_history[-1]
        fulfillment_rate = last_entry.selected_count / max(last_entry.qa_target + last_entry.multiple_choice_target, 1)
        if fulfillment_rate < 0.6:
            target_multiplier = 2.5
        elif fulfillment_rate > 0.8:
            target_multiplier = 1.8

    if latest_val_fb and latest_val_fb.quality_signals.duplicate_rate > 0.2:
        target_multiplier = max(1.5, target_multiplier - 0.5)

    qa_target = max(0, min(remaining_qa, int(base_qa * target_multiplier)))
    mc_target = max(0, min(remaining_mc, int(base_mc * target_multiplier)))

    # 难度分布：根据上一轮质量信号调整
    difficulty_distribution = dict(blueprint.default_modes["qa"].difficulty_distribution)
    if latest_val_fb and latest_val_fb.summary.citation_pass_rate < 0.5:
        hard = max(0.0, difficulty_distribution.get("hard", 0.0) - 0.1)
        medium = min(1.0, difficulty_distribution.get("medium", 0.0) + 0.1)
        difficulty_distribution["hard"] = hard
        difficulty_distribution["medium"] = medium

    # eval_profile：根据 token 预算和题目数量决定
    eval_profile = "standard"
    if state.resource_usage.total_tokens > (blueprint.stop_conditions.max_total_tokens or float("inf")) * 0.7:
        eval_profile = "light"
    elif qa_target + mc_target > 50:
        eval_profile = "full"

    # 构造 blueprint dict（兼容 qa_agent.Blueprint）
    blueprint_dict = {
        "task_id": state.task_id,
        "run_id": run_id,
        "language": blueprint.language,
        "topics": proposed_topics,
        "modes": {
            "qa": {
                "count": qa_target,
                "max_rounds": blueprint.default_modes["qa"].max_rounds,
                "difficulty_distribution": difficulty_distribution,
            },
            "multiple_choice": {
                "count": mc_target,
                "max_rounds": blueprint.default_modes["multiple_choice"].max_rounds,
                "difficulty_distribution": dict(blueprint.default_modes["multiple_choice"].difficulty_distribution),
            },
        },
    }

    topics_per_round = len(proposed_topics) if proposed_topics else 1
    verify_agent_patch = {}
    if latest_val_fb and latest_val_fb.summary.citation_pass_rate > 0.7 and latest_val_fb.summary.final_selection_rate < 0.2:
        verify_agent_patch = {
            "llm_validation": {
                "enabled": True,
                "min_overall_score": 0.7,
            }
        }

    return RoundSpec(
        task_id=state.task_id,
        blueprint_id=blueprint.blueprint_id,
        round_id=state.current_round,
        run_id=run_id,
        objective=f"Round {state.current_round}: generate {qa_target} QA + {mc_target} MC on topics {proposed_topics}",
        blueprint=blueprint_dict,
        qa_agent_patch={
            "candidate_pool": {"target_multiplier": target_multiplier},
            "initial_breadth": {
                "enabled": bool(
                    state.current_round == 1
                    or (
                        latest_val_fb is not None
                        and latest_val_fb.summary.final_selected < blueprint.stop_conditions.min_selected_per_round
                    )
                )
            },
            "planner": {"topics_per_round": topics_per_round},
        },
        verify_agent_patch=verify_agent_patch,
        model_eval_agent_patch={
            "models": {"candidate_model_names": list(state.model_pool.active)},
        },
        planner_hints=RoundPlannerHints(eval_profile=eval_profile),
    )


def _update_planner_state(
    state: PlannerState,
    round_spec: RoundSpec,
    gen_fb,
    val_fb,
    eval_fb,
) -> None:
    """更新 PlannerState：累加 completed_targets, resource_usage, run_history。"""
    used_topics = list(round_spec.blueprint.get("topics", []))
    if used_topics:
        remaining_active = [topic for topic in state.topic_backlog.active if topic not in used_topics]
        deferred = [topic for topic in state.topic_backlog.deferred if topic not in used_topics]
        state.topic_backlog.active = remaining_active
        state.topic_backlog.deferred = deferred + used_topics

    # completed_targets
    if val_fb:
        # 从 by_mode 统计 selected 数量
        state.completed_targets["qa"] += val_fb.by_mode.get("qa", {}).get("selected", 0)
        state.completed_targets["multiple_choice"] += val_fb.by_mode.get("multiple_choice", {}).get("selected", 0)

    # resource_usage
    state.resource_usage.total_input_tokens += gen_fb.summary.llm_input_tokens
    state.resource_usage.total_output_tokens += gen_fb.summary.llm_output_tokens
    if val_fb:
        state.resource_usage.total_input_tokens += val_fb.summary.llm_input_tokens
        state.resource_usage.total_output_tokens += val_fb.summary.llm_output_tokens
    if eval_fb:
        state.resource_usage.total_input_tokens += eval_fb.summary.llm_input_tokens
        state.resource_usage.total_output_tokens += eval_fb.summary.llm_output_tokens

    for agent_name, input_tokens, output_tokens in (
        ("qa_agent", gen_fb.summary.llm_input_tokens, gen_fb.summary.llm_output_tokens),
        ("verify_agent", val_fb.summary.llm_input_tokens if val_fb else 0, val_fb.summary.llm_output_tokens if val_fb else 0),
        ("model_eval_agent", eval_fb.summary.llm_input_tokens if eval_fb else 0, eval_fb.summary.llm_output_tokens if eval_fb else 0),
    ):
        if agent_name not in state.resource_usage.by_agent:
            state.resource_usage.by_agent[agent_name] = {"input_tokens": 0, "output_tokens": 0}
        state.resource_usage.by_agent[agent_name]["input_tokens"] += input_tokens
        state.resource_usage.by_agent[agent_name]["output_tokens"] += output_tokens

    state.resource_usage.total_tokens = (
        state.resource_usage.total_input_tokens + state.resource_usage.total_output_tokens
    )

    # run_history
    state.run_history.append(
        RunHistoryEntry(
            round_id=round_spec.round_id,
            run_id=round_spec.run_id,
            topics=round_spec.blueprint["topics"],
            qa_target=round_spec.blueprint["modes"]["qa"]["count"],
            multiple_choice_target=round_spec.blueprint["modes"]["multiple_choice"]["count"],
            selected_count=val_fb.summary.final_selected if val_fb else 0,
            evaluated=eval_fb is not None,
        )
    )
