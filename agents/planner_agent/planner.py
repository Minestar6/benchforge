"""PlannerAgent 跨轮主循环（docs/plan-runtime-aligned.md § 6.3）。

职责：
1. 初始化 PlannerState
2. 多轮循环：判断终止条件 → 诊断 → 控制计划 → topic 工具链 → 构建 RoundSpec → 调用 orchestrator → 更新状态
3. 返回最终 PlannerState
"""

import json
from pathlib import Path
from typing import Any
from loguru import logger

from benchforge.config.config import load_prompt
from benchforge.models.loader import ModelLoader
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry

from .schema import (
    GlobalBlueprint,
    PlannerState,
    RoundSpec,
    RoundPlannerHints,
    RunHistoryEntry,
    QuestionPlan,
    RoundDiagnosis,
    NextRoundControlPlan,
    NextRoundIntegratedPlan,
    NextRoundQuestionPlan,
)
from .config_loader import initialize_planner_state, save_planner_state
from .orchestrator import execute_round
from .topic_search import (
    summarize_topic_feedback,
    expand_topic_candidates,
    rank_topics,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOPIC_ADAPTIVE_RETRIEVER_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "planner_agent" / "topic_adaptive_retriever_prompt.md"


async def run_planner(
    global_blueprint: GlobalBlueprint,
    base_config_dir: str | Path,
    registry_path: str | Path,
    state_dir: str | Path,
) -> PlannerState:
    """执行多轮规划与编排。"""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    state = initialize_planner_state(global_blueprint)
    save_planner_state(state, state_dir / "planner_state_init.json")

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
        if _should_stop(state, global_blueprint):
            logger.info(f"[Planner] Stop condition met at round {state.current_round}")
            break

        state.current_round += 1
        run_id = f"run_{state.current_round:03d}"
        logger.info(f"[Planner] === Round {state.current_round} start ===")

        diagnosis = diagnose_last_round(latest_gen_fb, latest_val_fb, latest_eval_fb)
        question_plan = question_difficulty_evolver(
            state=state,
            blueprint=global_blueprint,
            diagnosis=diagnosis,
            previous_question_plan=state.last_question_plan,
            latest_gen_fb=latest_gen_fb,
            latest_val_fb=latest_val_fb,
        )
        control_plan = control_parameter_tuner(
            state=state,
            blueprint=global_blueprint,
            diagnosis=diagnosis,
            latest_gen_fb=latest_gen_fb,
            latest_val_fb=latest_val_fb,
            latest_eval_fb=latest_eval_fb,
        )

        proposed_topics = await topic_adaptive_retriever(
            state=state,
            blueprint=global_blueprint,
            model_client=planner_model_client,
            topic_budget=question_plan.topic_budget,
            latest_gen_fb=latest_gen_fb,
            latest_val_fb=latest_val_fb,
            latest_eval_fb=latest_eval_fb,
        )

        round_spec = round_spec_builder(
            state=state,
            blueprint=global_blueprint,
            target_topics=proposed_topics,
            run_id=run_id,
            question_plan=question_plan,
            control_plan=control_plan,
        )

        round_spec_path = state_dir / f"round_{state.current_round:03d}_spec.json"
        with open(round_spec_path, "w", encoding="utf-8") as f:
            json.dump(round_spec.model_dump(), f, ensure_ascii=False, indent=2)

        gen_fb, val_fb, eval_fb = await execute_round(
            round_spec, global_blueprint, base_config_dir, registry_path
        )
        latest_gen_fb, latest_val_fb, latest_eval_fb = gen_fb, val_fb, eval_fb

        _update_planner_state(state, round_spec, question_plan, gen_fb, val_fb, eval_fb)
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
    if state.current_round >= blueprint.stop_conditions.max_rounds:
        return True

    qa_target = blueprint.final_targets.qa
    mc_target = blueprint.final_targets.multiple_choice
    if (
        state.completed_targets["qa"] >= qa_target
        and state.completed_targets["multiple_choice"] >= mc_target
    ):
        return True

    if blueprint.stop_conditions.max_total_tokens:
        if state.resource_usage.total_tokens >= blueprint.stop_conditions.max_total_tokens:
            return True

    return False


def _ordered_unique_topics(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        topic = str(item).strip()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        ordered.append(topic)
    return ordered


def _topic_score_list(summary) -> list[dict[str, float | str]]:
    scores = []
    for topic, stats in summary.topic_stats.items():
        value = 0.0
        value += float(stats.get("model_gap") or 0.0) * 4.0
        value += float(stats.get("selected") or 0.0) * 0.3
        value += float(stats.get("avg_citation_score") or 0.0) * 0.2
        value += float(stats.get("avg_llm_overall_score") or 0.0) * 0.2
        value -= float(stats.get("too_easy_ratio") or 0.0) * 1.5
        value -= float(stats.get("all_models_fail_ratio") or 0.0) * 2.0
        value -= float(stats.get("reuse_count") or 0.0) * 0.5
        scores.append({"topic": topic, "score": round(value, 4)})
    return scores


def _extract_topic_selection(text: str, budget: int) -> list[str]:
    stripped = text.strip()
    payload = None
    list_start = stripped.find("[")
    list_end = stripped.rfind("]")
    if list_start >= 0 and list_end > list_start:
        payload = json.loads(stripped[list_start:list_end + 1])
    else:
        object_start = stripped.find("{")
        object_end = stripped.rfind("}")
        if object_start >= 0 and object_end > object_start:
            payload = json.loads(stripped[object_start:object_end + 1])

    if isinstance(payload, dict):
        payload = payload.get("topics") or payload.get("selected_topics") or []
    if not isinstance(payload, list):
        return []
    return _ordered_unique_topics(payload)[: max(1, int(budget or 1))]


def _render_topic_adaptive_prompt(
    user_goal: str,
    topic_score_list: list[dict[str, float | str]],
    related_topics: list[str],
    topic_budget: int,
) -> str:
    prompt_template = load_prompt(_TOPIC_ADAPTIVE_RETRIEVER_PROMPT_PATH)
    replacements = {
        "{user_goal}": json.dumps(user_goal, ensure_ascii=False),
        "{topic_score_list}": json.dumps(topic_score_list, ensure_ascii=False),
        "{related_topics}": json.dumps(related_topics, ensure_ascii=False),
        "{topic_budget}": str(max(1, int(topic_budget or 1))),
    }
    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


async def _select_topics_with_adaptive_prompt(
    blueprint: GlobalBlueprint,
    model_client,
    summary,
    candidate_topics: list[str],
    topic_budget: int,
) -> list[str]:
    complete = getattr(model_client, "complete", None)
    if not complete:
        return []

    prompt = _render_topic_adaptive_prompt(
        user_goal=blueprint.user_goal,
        topic_score_list=_topic_score_list(summary),
        related_topics=candidate_topics,
        topic_budget=topic_budget,
    )
    response = await complete(
        model=getattr(model_client, "model_name", "planner"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    selected = _extract_topic_selection(response.get("text", ""), topic_budget)
    allowed = set(candidate_topics)
    selected = [topic for topic in selected if topic in allowed]
    return selected[: max(1, int(topic_budget or 1))]


async def topic_adaptive_retriever(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    model_client,
    topic_budget: int,
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> list[str]:
    """根据主题反馈和候选扩展检索下一轮目标 topics。"""
    if not state.topic_backlog.active and state.topic_backlog.deferred:
        state.topic_backlog.active = list(state.topic_backlog.deferred)
        state.topic_backlog.deferred = []

    active_topics = state.topic_backlog.active
    if not active_topics and not state.run_history:
        logger.warning("[Planner] No available topics for proposal")
        return []

    summary = summarize_topic_feedback(
        state=state,
        latest_gen_fb=latest_gen_fb,
        latest_val_fb=latest_val_fb,
        latest_eval_fb=latest_eval_fb,
    )
    candidate_topics = await expand_topic_candidates(
        user_goal=blueprint.user_goal,
        state=state,
        model_client=model_client,
        summary=summary,
        budget=topic_budget,
    )

    selected_topics: list[str] = []
    try:
        selected_topics = await _select_topics_with_adaptive_prompt(
            blueprint=blueprint,
            model_client=model_client,
            summary=summary,
            candidate_topics=candidate_topics,
            topic_budget=topic_budget,
        )
    except Exception:
        selected_topics = []

    if not selected_topics:
        selected_topics = rank_topics(
            candidate_topics=candidate_topics,
            budget=topic_budget,
            state=state,
            summary=summary,
        )

    if not selected_topics and active_topics:
        return list(active_topics)[: max(1, topic_budget)]
    return selected_topics


async def _propose_topics_with_llm(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    model_client,
    topic_budget: int | None = None,
    topics_budget: int | None = None,
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> list[str]:
    """兼容旧调用：委托给 topic_adaptive_retriever。"""
    resolved_budget = topic_budget if topic_budget is not None else topics_budget
    return await topic_adaptive_retriever(
        state=state,
        blueprint=blueprint,
        model_client=model_client,
        topic_budget=resolved_budget or 1,
        latest_gen_fb=latest_gen_fb,
        latest_val_fb=latest_val_fb,
        latest_eval_fb=latest_eval_fb,
    )


def diagnose_last_round(
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> RoundDiagnosis:
    if latest_gen_fb and latest_gen_fb.by_mode:
        min_fulfillment = min(mode_fb.fulfillment_rate for mode_fb in latest_gen_fb.by_mode.values())
        if min_fulfillment < 0.5 or latest_gen_fb.summary.global_failures > 0:
            return RoundDiagnosis(
                label="generation_capacity_insufficient",
                confidence=0.8,
                problems=["generation_capacity_insufficient"],
                evidence={
                    "min_fulfillment_rate": min_fulfillment,
                    "global_failures": latest_gen_fb.summary.global_failures,
                },
            )

    if latest_val_fb and latest_eval_fb:
        gap = float(latest_eval_fb.derived_performance_signals.get("overall_model_gap") or 0.0)
        too_easy = float(latest_eval_fb.derived_performance_signals.get("easy_questions_too_easy_ratio") or 0.0)
        all_fail = float(latest_eval_fb.derived_performance_signals.get("all_models_fail_ratio") or 0.0)
        citation_pass = latest_val_fb.summary.citation_pass_rate
        selected_quality = latest_val_fb.quality_signals.avg_llm_overall_score_selected or 0.0
        strong_topics = [
            topic
            for topic, stats in latest_eval_fb.derived_performance_signals.get("by_topic", {}).items()
            if (stats.get("model_gap") or 0.0) >= gap
        ]
        weak_topics = [
            topic
            for topic, stats in latest_eval_fb.derived_performance_signals.get("by_topic", {}).items()
            if (stats.get("model_gap") or 0.0) < gap
        ]
        evidence = {
            "overall_model_gap": gap,
            "easy_questions_too_easy_ratio": too_easy,
            "all_models_fail_ratio": all_fail,
            "citation_pass_rate": citation_pass,
            "avg_llm_overall_score_selected": selected_quality,
            "strong_topics": strong_topics,
            "weak_topics": weak_topics,
        }
        if gap < 0.15 and citation_pass >= 0.75:
            return RoundDiagnosis(
                label="high_quality_low_separation",
                confidence=0.9,
                problems=["low separation"],
                evidence=evidence,
            )
        if gap < 0.15:
            return RoundDiagnosis(
                label="low_quality_low_separation",
                confidence=0.8,
                problems=["low separation", "low quality"],
                evidence=evidence,
            )
        if selected_quality < 0.75 or citation_pass < 0.65:
            return RoundDiagnosis(
                label="high_separation_low_quality",
                confidence=0.75,
                problems=["high separation", "low quality"],
                evidence=evidence,
            )
        return RoundDiagnosis(
            label="high_separation_imbalanced_distribution",
            confidence=0.7,
            problems=["imbalanced separation"],
            evidence=evidence,
        )

    return RoundDiagnosis(label="cold_start", confidence=0.5, problems=["no_feedback"], evidence={})


def _normalize_distribution(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values()) or 1.0
    return {key: value / total for key, value in distribution.items()}


def _rebalance_distribution(
    distribution: dict[str, float],
    increase: str | None = None,
    decrease: str | None = None,
    amount: float = 0.05,
) -> dict[str, float]:
    updated = dict(distribution)
    if increase:
        updated[increase] = min(1.0, updated.get(increase, 0.0) + amount)
    if decrease:
        updated[decrease] = max(0.0, updated.get(decrease, 0.0) - amount)
    return _normalize_distribution(updated)


def _difficulty_selected_rate(latest_val_fb, difficulty: str) -> float | None:
    if not latest_val_fb:
        return None
    stats = (latest_val_fb.by_difficulty or {}).get(difficulty)
    if not stats:
        return None
    total = int(stats.get("selected", 0) or 0) + int(stats.get("reserve", 0) or 0) + int(stats.get("rejected", 0) or 0)
    if total <= 0:
        return None
    return float(int(stats.get("selected", 0) or 0) / total)


def question_difficulty_evolver(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    diagnosis: RoundDiagnosis,
    previous_question_plan: QuestionPlan | None = None,
    latest_gen_fb=None,
    latest_val_fb=None,
) -> NextRoundQuestionPlan:
    """演化下一轮题量、题型、难度和 topic 预算。"""
    remaining_qa = max(0, blueprint.final_targets.qa - state.completed_targets["qa"])
    remaining_mc = max(0, blueprint.final_targets.multiple_choice - state.completed_targets["multiple_choice"])
    base_qa = max(blueprint.stop_conditions.min_selected_per_round, remaining_qa // 3)
    base_mc = max(blueprint.stop_conditions.min_selected_per_round, remaining_mc // 3)

    qa_dist = dict(
        previous_question_plan.qa_difficulty_distribution
        if previous_question_plan
        else blueprint.default_modes["qa"].difficulty_distribution
    )
    mc_dist = dict(
        previous_question_plan.mc_difficulty_distribution
        if previous_question_plan
        else blueprint.default_modes["multiple_choice"].difficulty_distribution
    )
    count_multiplier = 2.0
    min_candidate_multiplier = previous_question_plan.min_candidate_multiplier if previous_question_plan else 1.5
    max_candidate_multiplier = previous_question_plan.max_candidate_multiplier if previous_question_plan else 2.0
    topic_budget = previous_question_plan.topic_budget if previous_question_plan else 2

    label = diagnosis.label
    if label == "high_quality_low_separation":
        qa_dist["hard"] = min(1.0, qa_dist.get("hard", 0.0) + 0.1)
        qa_dist["easy"] = max(0.0, qa_dist.get("easy", 0.0) - 0.1)
        topic_budget = 1
    elif label == "low_quality_low_separation":
        count_multiplier = 2.4
        min_candidate_multiplier = 1.8
        max_candidate_multiplier = 2.4
        topic_budget = 1
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.05)
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.05)
    elif label == "generation_capacity_insufficient":
        count_multiplier = 1.8
        min_candidate_multiplier = 1.3
        max_candidate_multiplier = 1.8
        topic_budget = 1
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.05)
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.05)
    elif label == "high_separation_imbalanced_distribution":
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.05)
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.05)

    if state.current_round == 1 and blueprint.initial_generation_strategy == "balanced_exploration":
        topic_budget = max(topic_budget, 2)
        qa_dist = _normalize_distribution(qa_dist)
        mc_dist = _normalize_distribution(mc_dist)

    qa_mode_fb = (latest_gen_fb.by_mode or {}).get("qa") if latest_gen_fb else None
    if qa_mode_fb and qa_mode_fb.fulfillment_rate < 0.8:
        count_multiplier = min(count_multiplier, 1.8)
        min_candidate_multiplier = max(min_candidate_multiplier, 1.7)
        max_candidate_multiplier = max(max_candidate_multiplier, 2.2)
        topic_budget = min(topic_budget, 1)

    hard_gen_count = 0
    if qa_mode_fb:
        hard_gen_count = int((qa_mode_fb.difficulty_counts or {}).get("hard", 0) or 0)
    hard_selected_rate = _difficulty_selected_rate(latest_val_fb, "hard")
    medium_selected_rate = _difficulty_selected_rate(latest_val_fb, "medium")

    if hard_gen_count >= 3 and hard_selected_rate is not None and hard_selected_rate < 0.2:
        qa_dist = _rebalance_distribution(qa_dist, increase="medium", decrease="hard", amount=0.08)
    elif (
        hard_selected_rate is not None
        and medium_selected_rate is not None
        and hard_selected_rate > medium_selected_rate + 0.2
    ):
        qa_dist = _rebalance_distribution(qa_dist, increase="hard", decrease="easy", amount=0.06)

    if latest_val_fb and latest_val_fb.summary.final_selected < blueprint.stop_conditions.min_selected_per_round:
        min_candidate_multiplier = max(min_candidate_multiplier, 1.8)
        max_candidate_multiplier = max(max_candidate_multiplier, 2.3)
        count_multiplier = min(count_multiplier, 1.9)

    qa_count = max(0, min(remaining_qa, int(base_qa * count_multiplier)))
    mc_count = max(0, min(remaining_mc, int(base_mc * count_multiplier)))

    return NextRoundQuestionPlan(
        qa_count=qa_count,
        mc_count=mc_count,
        qa_difficulty_distribution=_normalize_distribution(qa_dist),
        mc_difficulty_distribution=_normalize_distribution(mc_dist),
        min_candidate_multiplier=min_candidate_multiplier,
        max_candidate_multiplier=max_candidate_multiplier,
        topic_budget=topic_budget,
        min_selected_per_round=(
            previous_question_plan.min_selected_per_round
            if previous_question_plan and previous_question_plan.min_selected_per_round is not None
            else blueprint.stop_conditions.min_selected_per_round
        ),
    )


def control_parameter_tuner(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    diagnosis: RoundDiagnosis,
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> NextRoundControlPlan:
    """调优验证、选择、评估强度等运行控制参数。"""
    min_citation_score = 0.65
    min_chunk_citation_score = 0.85
    min_answer_citation_score = 0.75
    min_overall_score = 0.75
    selection_mode = "light"
    semantic_similarity_threshold = 0.9
    eval_profile = "standard"
    initial_breadth_enabled = state.current_round == 1

    label = diagnosis.label
    if label == "high_quality_low_separation":
        eval_profile = "full"
    elif label == "low_quality_low_separation":
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
    elif label == "high_separation_low_quality":
        min_citation_score = 0.7
        min_chunk_citation_score = 0.88
        min_answer_citation_score = 0.78
        min_overall_score = 0.78
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
    elif label == "high_separation_imbalanced_distribution":
        eval_profile = "full"

    if latest_val_fb:
        if latest_val_fb.quality_signals.duplicate_rate >= 0.25:
            selection_mode = "strict"
            semantic_similarity_threshold = max(semantic_similarity_threshold, 0.92)
        if latest_val_fb.summary.citation_pass_rate < 0.6:
            min_citation_score = max(min_citation_score, 0.68)
            min_chunk_citation_score = max(min_chunk_citation_score, 0.87)
            min_answer_citation_score = max(min_answer_citation_score, 0.77)
        if (latest_val_fb.quality_signals.avg_llm_overall_score_selected or 1.0) < 0.72:
            min_overall_score = max(min_overall_score, 0.78)

    if latest_gen_fb:
        qa_mode_fb = (latest_gen_fb.by_mode or {}).get("qa")
        if qa_mode_fb and qa_mode_fb.fulfillment_rate < 0.7:
            initial_breadth_enabled = True

    if latest_eval_fb:
        gap = float(latest_eval_fb.derived_performance_signals.get("overall_model_gap") or 0.0)
        too_easy = float(latest_eval_fb.derived_performance_signals.get("easy_questions_too_easy_ratio") or 0.0)
        all_fail = float(latest_eval_fb.derived_performance_signals.get("all_models_fail_ratio") or 0.0)
        if gap < 0.1 and too_easy >= 0.4:
            eval_profile = "full"
        if all_fail >= 0.4:
            min_overall_score = min(min_overall_score, 0.75)

    return NextRoundControlPlan(
        citation_enabled=True,
        min_citation_score=min_citation_score,
        min_chunk_citation_score=min_chunk_citation_score,
        min_answer_citation_score=min_answer_citation_score,
        llm_validation_enabled=True,
        min_overall_score=min_overall_score,
        selection_mode=selection_mode,
        semantic_similarity_threshold=semantic_similarity_threshold,
        eval_profile=eval_profile,
        judge_enabled=bool(blueprint.evaluation_requirements.llm_judge_metrics.get("qa")),
        candidate_model_names=list(state.model_pool.active),
        judge_model_name=blueprint.evaluator_defaults.judge_model_name,
        initial_breadth_enabled=initial_breadth_enabled,
    )


def adjust_next_round_plan(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    diagnosis: RoundDiagnosis,
) -> NextRoundIntegratedPlan:
    """兼容旧调用：聚合题目计划和控制参数计划。"""
    question_plan = question_difficulty_evolver(state, blueprint, diagnosis)
    control_plan = control_parameter_tuner(state, blueprint, diagnosis)
    return NextRoundIntegratedPlan(
        **question_plan.model_dump(),
        **control_plan.model_dump(),
    )


def round_spec_builder(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    target_topics: list[str],
    run_id: str,
    question_plan: NextRoundQuestionPlan,
    control_plan: NextRoundControlPlan,
) -> RoundSpec:
    """将规划工具输出落地为 RoundSpec，不做二次推理。"""
    blueprint_dict = {
        "task_id": state.task_id,
        "run_id": run_id,
        "language": blueprint.language,
        "topics": target_topics,
        "modes": {
            "qa": {
                "count": question_plan.qa_count,
                "max_rounds": blueprint.default_modes["qa"].max_rounds,
                "difficulty_distribution": question_plan.qa_difficulty_distribution,
            },
            "multiple_choice": {
                "count": question_plan.mc_count,
                "max_rounds": blueprint.default_modes["multiple_choice"].max_rounds,
                "difficulty_distribution": question_plan.mc_difficulty_distribution,
            },
        },
    }

    verify_agent_patch = {
        "citation_validation": {
            "enabled": control_plan.citation_enabled,
            "min_citation_score": control_plan.min_citation_score,
            "min_chunk_citation_score": control_plan.min_chunk_citation_score,
            "min_answer_citation_score": control_plan.min_answer_citation_score,
        },
        "llm_validation": {
            "enabled": control_plan.llm_validation_enabled,
            "min_overall_score": control_plan.min_overall_score,
        },
        "selection": {
            "mode": control_plan.selection_mode,
            "semantic_similarity_threshold": control_plan.semantic_similarity_threshold,
        },
    }

    return RoundSpec(
        task_id=state.task_id,
        blueprint_id=blueprint.blueprint_id,
        round_id=state.current_round,
        run_id=run_id,
        objective=(
            f"Round {state.current_round}: generate {question_plan.qa_count} QA + "
            f"{question_plan.mc_count} MC on topics {target_topics}"
        ),
        blueprint=blueprint_dict,
        qa_agent_patch={
            "candidate_pool": {
                "min_candidate_multiplier": question_plan.min_candidate_multiplier,
                "max_candidate_multiplier": question_plan.max_candidate_multiplier,
            },
            "initial_breadth": {
                "enabled": control_plan.initial_breadth_enabled
            },
            "planner": {"topics_per_round": question_plan.topic_budget},
        },
        verify_agent_patch=verify_agent_patch,
        model_eval_agent_patch={
            "models": {
                "candidate_model_names": list(control_plan.candidate_model_names),
                "judge_model_name": control_plan.judge_model_name,
            },
            "judge": {"enabled": control_plan.judge_enabled},
        },
        planner_hints=RoundPlannerHints(eval_profile=control_plan.eval_profile),
    )


def _build_next_round_spec(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    proposed_topics: list[str],
    run_id: str,
    control_plan: NextRoundIntegratedPlan,
) -> RoundSpec:
    """兼容旧调用：从聚合计划构建 RoundSpec。"""
    question_plan = NextRoundQuestionPlan(
        qa_count=control_plan.qa_count,
        mc_count=control_plan.mc_count,
        qa_difficulty_distribution=control_plan.qa_difficulty_distribution,
        mc_difficulty_distribution=control_plan.mc_difficulty_distribution,
        min_candidate_multiplier=control_plan.min_candidate_multiplier,
        max_candidate_multiplier=control_plan.max_candidate_multiplier,
        topic_budget=control_plan.topic_budget,
        min_selected_per_round=control_plan.min_selected_per_round,
    )
    runtime_plan = NextRoundControlPlan(
        citation_enabled=control_plan.citation_enabled,
        min_citation_score=control_plan.min_citation_score,
        min_chunk_citation_score=control_plan.min_chunk_citation_score,
        min_answer_citation_score=control_plan.min_answer_citation_score,
        llm_validation_enabled=control_plan.llm_validation_enabled,
        min_overall_score=control_plan.min_overall_score,
        selection_mode=control_plan.selection_mode,
        semantic_similarity_threshold=control_plan.semantic_similarity_threshold,
        eval_profile=control_plan.eval_profile,
        judge_enabled=control_plan.judge_enabled,
        candidate_model_names=control_plan.candidate_model_names,
        judge_model_name=control_plan.judge_model_name,
        initial_breadth_enabled=control_plan.initial_breadth_enabled,
    )
    return round_spec_builder(
        state=state,
        blueprint=blueprint,
        target_topics=proposed_topics,
        run_id=run_id,
        question_plan=question_plan,
        control_plan=runtime_plan,
    )


def _update_planner_state(
    state: PlannerState,
    round_spec: RoundSpec,
    question_plan: QuestionPlan,
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

    if val_fb:
        state.completed_targets["qa"] += val_fb.by_mode.get("qa", {}).get("selected", 0)
        state.completed_targets["multiple_choice"] += val_fb.by_mode.get("multiple_choice", {}).get("selected", 0)

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
    state.last_question_plan = QuestionPlan(**question_plan.model_dump())

    state.run_history.append(
        RunHistoryEntry(
            round_id=round_spec.round_id,
            run_id=round_spec.run_id,
            topics=round_spec.blueprint["topics"],
            qa_target=round_spec.blueprint["modes"]["qa"]["count"],
            multiple_choice_target=round_spec.blueprint["modes"]["multiple_choice"]["count"],
            selected_count=val_fb.summary.final_selected if val_fb else 0,
            evaluated=eval_fb is not None,
            question_plan=QuestionPlan(**question_plan.model_dump()),
        )
    )
