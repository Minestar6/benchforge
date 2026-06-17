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
    NextRoundQuestionPlan,
)
from .config_loader import initialize_planner_state, save_planner_state, load_planner_agent_config
from .orchestrator import execute_round
from .topic_search import (
    summarize_topic_feedback,
    expand_topic_candidates,
    rank_topics,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOPIC_ADAPTIVE_RETRIEVER_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "planner_agent" / "topic_adaptive_retriever_prompt.md"
_DIFFICULTY_COUNT_PLANNER_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "planner_agent" / "difficulty_count_planner_prompt.md"


async def run_planner(
    global_blueprint: GlobalBlueprint,
    base_config_dir: str | Path,
    registry_path: str | Path,
    state_dir: str | Path,
    resume_state: PlannerState | None = None,
) -> PlannerState:
    """执行多轮规划与编排。

    Args:
        resume_state: 从已有快照恢复的 PlannerState，跳过初始化直接从该状态继续。
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    if resume_state is not None:
        state = resume_state
        logger.info(f"[Planner] Resumed from round {state.current_round}, "
                     f"completed=qa:{state.completed_targets['qa']}/mc:{state.completed_targets['multiple_choice']}")
    else:
        state = initialize_planner_state(global_blueprint)
        save_planner_state(state, state_dir / "planner_state_init.json")

    registry = load_model_registry(registry_path)
    planner_model_name = global_blueprint.evaluator_defaults.judge_model_name or list(registry.keys())[0]
    planner_model_cfg = registry.get(planner_model_name)
    if not planner_model_cfg:
        raise ValueError(f"Planner model '{planner_model_name}' not found in registry")
    planner_model_client = ModelLoader.load_model(planner_model_cfg)

    # 加载 planner 配置，决定 evolver 模式
    planner_config = load_planner_agent_config(base_config_dir)
    use_llm_evolver = planner_config.question_difficulty_evolver.enabled

    logger.info(
        f"[Planner] Start planning: task_id={state.task_id}, "
        f"targets=qa:{global_blueprint.final_targets.qa}/mc:{global_blueprint.final_targets.multiple_choice}, "
        f"evolver={'llm' if use_llm_evolver else 'rule'}"
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
        if use_llm_evolver:
            question_plan = await _llm_question_difficulty_evolver(
                state=state,
                blueprint=global_blueprint,
                diagnosis=diagnosis,
                previous_question_plan=state.last_question_plan,
                latest_gen_fb=latest_gen_fb,
                latest_val_fb=latest_val_fb,
                latest_eval_fb=latest_eval_fb,
                model_client=planner_model_client,
            )
        else:
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

        # 校验：有生成目标但无 topic 时跳过本轮，避免空转
        if not proposed_topics and (question_plan.qa_count > 0 or question_plan.mc_count > 0):
            logger.warning(
                f"[Planner] No topics proposed but generation targets exist "
                f"(qa={question_plan.qa_count}, mc={question_plan.mc_count}), skipping round"
            )
            state.current_round -= 1  # 回退轮次计数，避免空转消耗 max_rounds 预算
            continue

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
    """为 LLM prompt 准备 topic 评分列表，权重与 rank_topics 核心因子对齐。"""
    scores = []
    for topic, stats in summary.topic_stats.items():
        value = 0.0
        value += float(stats.get("model_gap") or 0.0) * 4.0
        value += float(stats.get("selected") or 0.0) * 0.6
        value += float(stats.get("candidate_count") or 0.0) * 0.1
        value += float(stats.get("avg_citation_score") or 0.0) * 0.5
        value += float(stats.get("avg_llm_overall_score") or 0.0) * 0.5
        value -= float(stats.get("too_easy_ratio") or 0.0) * 2.0
        value -= float(stats.get("all_models_fail_ratio") or 0.0) * 3.0
        value -= float(stats.get("reuse_count") or 0.0) * 1.25
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
        # 首轮且 seed_topics 为空：尝试从 blueprint 回填
        if blueprint.seed_topics:
            logger.info("[Planner] Backlog empty, reinitializing from blueprint seed_topics")
            state.topic_backlog.active = list(blueprint.seed_topics)
            active_topics = state.topic_backlog.active
        else:
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

    # 将 LLM 扩展的新 topic 写回 backlog，避免 topic 池过早耗尽
    existing_topics = set(state.topic_backlog.active) | set(state.topic_backlog.deferred)
    for topic in candidate_topics:
        if topic not in existing_topics:
            state.topic_backlog.deferred.append(topic)

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


def diagnose_last_round(
    latest_gen_fb=None,
    latest_val_fb=None,
    latest_eval_fb=None,
) -> RoundDiagnosis:
    """分阶段诊断：gen → val-only → val+eval → 兜底，各阶段信号合并而非屏蔽。"""
    # ── Phase 1: 检查生成侧问题（收集证据，不立即返回） ──
    gen_insufficient = False
    gen_evidence: dict[str, Any] = {}
    if latest_gen_fb and latest_gen_fb.by_mode:
        min_fulfillment = min(mode_fb.fulfillment_rate for mode_fb in latest_gen_fb.by_mode.values())
        total_candidates = max(latest_gen_fb.summary.total_candidates, 1)
        failure_rate = latest_gen_fb.summary.global_failures / total_candidates
        if min_fulfillment < 0.5 or failure_rate > 0.3:
            gen_insufficient = True
            gen_evidence = {
                "min_fulfillment_rate": min_fulfillment,
                "global_failures": latest_gen_fb.summary.global_failures,
                "failure_rate": failure_rate,
            }

    # ── Phase 2: 有验证结果但无评估结果 ──
    # 场景：candidates 全部在验证阶段被拒，或评估被跳过
    if latest_val_fb and not latest_eval_fb:
        if latest_val_fb.summary.final_selected == 0:
            problems = ["all_candidates_rejected"]
            evidence = {
                "total_candidates": latest_val_fb.summary.total_candidates,
                "citation_pass_rate": latest_val_fb.summary.citation_pass_rate,
                "llm_pass_rate_after_citation": latest_val_fb.summary.llm_pass_rate_after_citation,
                "duplicate_rate": latest_val_fb.quality_signals.duplicate_rate,
            }
            if gen_insufficient:
                problems.append("generation_capacity_insufficient")
                evidence.update(gen_evidence)
            return RoundDiagnosis(
                label="verification_rejected_all",
                confidence=0.85,
                problems=problems,
                evidence=evidence,
            )
        # val 存在但无 eval（例如 selected > 0 但 eval 因其他原因被跳过）
        # 注：正常流程中 final_selected > 0 时 eval 总会执行，此分支为防御性代码
        if gen_insufficient:
            return RoundDiagnosis(
                label="generation_capacity_insufficient",
                confidence=0.75,
                problems=["generation_capacity_insufficient"],
                evidence={
                    **gen_evidence,
                    "citation_pass_rate": latest_val_fb.summary.citation_pass_rate,
                    "final_selected": latest_val_fb.summary.final_selected,
                },
            )
        # 无 gen 问题、无 eval，但 val 数据存在 → 信息不足
        return RoundDiagnosis(
            label="insufficient_eval_data",
            confidence=0.5,
            problems=["eval_skipped"],
            evidence={"final_selected": latest_val_fb.summary.final_selected},
        )

    # ── Phase 3: 同时有验证和评估结果 ──
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

        # 合并 gen 侧信号
        problems: list[str] = []
        if gen_insufficient:
            problems.append("generation_capacity_insufficient")
            evidence.update(gen_evidence)

        if gap < 0.15 and citation_pass >= 0.75:
            return RoundDiagnosis(
                label="high_quality_low_separation",
                confidence=0.9 if not gen_insufficient else 0.7,
                problems=problems + ["low separation"],
                evidence=evidence,
            )
        if gap < 0.15:
            return RoundDiagnosis(
                label="low_quality_low_separation",
                confidence=0.8 if not gen_insufficient else 0.65,
                problems=problems + ["low separation", "low quality"],
                evidence=evidence,
            )
        if selected_quality < 0.75 or citation_pass < 0.65:
            return RoundDiagnosis(
                label="high_separation_low_quality",
                confidence=0.75 if not gen_insufficient else 0.6,
                problems=problems + ["high separation", "low quality"],
                evidence=evidence,
            )
        return RoundDiagnosis(
            label="high_separation_imbalanced_distribution",
            confidence=0.7 if not gen_insufficient else 0.6,
            problems=problems + ["imbalanced separation"],
            evidence=evidence,
        )

    # ── Phase 4: 仅有 gen 问题，无 val/eval ──
    if gen_insufficient:
        return RoundDiagnosis(
            label="generation_capacity_insufficient",
            confidence=0.8,
            problems=["generation_capacity_insufficient"],
            evidence=gen_evidence,
        )

    # ── Phase 5: 无任何反馈 ──
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


def _difficulty_distribution_of_selected(latest_val_fb, mode: str) -> dict[str, float]:
    """从 ValidatorFeedback 中提取某 mode 的 selected 题目难度分布。"""
    by_diff = latest_val_fb.by_difficulty or {}
    counts = {diff: by_diff.get(diff, {}).get("selected", 0) for diff in ("easy", "medium", "hard")}
    total = sum(counts.values())
    if total <= 0:
        return {"easy": 1/3, "medium": 1/3, "hard": 1/3}
    return {diff: counts[diff] / total for diff in ("easy", "medium", "hard")}


def _extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 响应中提取 JSON 对象。"""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start:end + 1])
    raise ValueError("No JSON object found in LLM response")


async def _llm_question_difficulty_evolver(
    state: PlannerState,
    blueprint: GlobalBlueprint,
    diagnosis: RoundDiagnosis,
    previous_question_plan: QuestionPlan | None,
    latest_gen_fb,
    latest_val_fb,
    latest_eval_fb,
    model_client,
) -> NextRoundQuestionPlan:
    """LLM-based 题目难度演化器。

    将诊断结论和反馈数据注入 prompt，由 LLM 决策下一轮题目计划。
    解析失败或 LLM 不可用时回退到规则引擎。
    """
    # ── 构建 prompt 输入 ──
    remaining = {
        "qa": max(0, blueprint.final_targets.qa - state.completed_targets["qa"]),
        "mc": max(0, blueprint.final_targets.multiple_choice - state.completed_targets["multiple_choice"]),
    }
    if previous_question_plan:
        prev_plan_dict = {
            "qa_count": previous_question_plan.qa_count,
            "mc_count": previous_question_plan.mc_count,
            "qa_difficulty_distribution": previous_question_plan.qa_difficulty_distribution,
            "mc_difficulty_distribution": previous_question_plan.mc_difficulty_distribution,
            "topic_budget": previous_question_plan.topic_budget,
            "min_candidate_multiplier": previous_question_plan.min_candidate_multiplier,
            "max_candidate_multiplier": previous_question_plan.max_candidate_multiplier,
        }
    else:
        prev_plan_dict = None

    gen_stats = None
    if latest_gen_fb:
        by_mode = latest_gen_fb.by_mode or {}
        gen_stats = {
            "qa_candidates": getattr(by_mode.get("qa", None), "candidate_count", 0),
            "mc_candidates": getattr(by_mode.get("multiple_choice", None), "candidate_count", 0),
            "global_failures": latest_gen_fb.summary.global_failures,
        }

    val_stats = None
    if latest_val_fb:
        gen_by_mode = (latest_gen_fb.by_mode or {}) if latest_gen_fb else {}
        val_stats = {
            "qa_generated": getattr(gen_by_mode.get("qa", None), "candidate_count", 0),
            "qa_selected": latest_val_fb.by_mode.get("qa", {}).get("selected", 0),
            "qa_difficulty_distribution": _difficulty_distribution_of_selected(latest_val_fb, "qa"),
            "mc_generated": getattr(gen_by_mode.get("multiple_choice", None), "candidate_count", 0),
            "mc_selected": latest_val_fb.by_mode.get("multiple_choice", {}).get("selected", 0),
            "mc_difficulty_distribution": _difficulty_distribution_of_selected(latest_val_fb, "multiple_choice"),
        }
    eval_stats = None
    if latest_eval_fb:
        by_diff = latest_eval_fb.derived_performance_signals.get("by_difficulty", {}) or {}
        eval_stats = {
            "qa_difficulty_scores": {
                diff: by_diff.get(diff, {}).get("avg_score", 0.0)
                for diff in ("easy", "medium", "hard")
            },
            "mc_difficulty_scores": {
                diff: by_diff.get(diff, {}).get("avg_score", 0.0)
                for diff in ("easy", "medium", "hard")
            },
        }

    prompt_template = load_prompt(_DIFFICULTY_COUNT_PLANNER_PROMPT_PATH)
    replacements = {
        "{current_round}": str(state.current_round),
        "{max_rounds}": str(blueprint.stop_conditions.max_rounds),
        "{remaining_targets}": json.dumps(remaining, ensure_ascii=False),
        "{previous_round_plan}": json.dumps(prev_plan_dict, ensure_ascii=False),
        "{generation_stats}": json.dumps(gen_stats, ensure_ascii=False),
        "{validation_stats}": json.dumps(val_stats, ensure_ascii=False),
        "{evaluation_stats}": json.dumps(eval_stats, ensure_ascii=False),
    }
    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    # ── 调用 LLM ──
    try:
        response = await model_client.complete(
            model=getattr(model_client, "model_name", "planner"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        payload = _extract_json_object(response.get("text", ""))
    except Exception as exc:
        logger.warning(f"[Planner] LLM evolver failed: {exc}, falling back to rule-based")
        return question_difficulty_evolver(
            state=state, blueprint=blueprint, diagnosis=diagnosis,
            previous_question_plan=previous_question_plan,
            latest_gen_fb=latest_gen_fb, latest_val_fb=latest_val_fb,
        )

    # ── 解析并校验输出 ──
    try:
        # 提取必需字段，缺失时回退到规则引擎
        qa_count = int(payload.get("qa_count", 0))
        mc_count = int(payload.get("mc_count", 0))

        qa_dist_raw = payload.get("qa_difficulty_distribution", {})
        mc_dist_raw = payload.get("mc_difficulty_distribution", {})
        qa_dist = {
            "easy": float(qa_dist_raw.get("easy", 0.2)),
            "medium": float(qa_dist_raw.get("medium", 0.5)),
            "hard": float(qa_dist_raw.get("hard", 0.3)),
        }
        mc_dist = {
            "easy": float(mc_dist_raw.get("easy", 0.3)),
            "medium": float(mc_dist_raw.get("medium", 0.5)),
            "hard": float(mc_dist_raw.get("hard", 0.2)),
        }

        qa_count = max(0, min(qa_count, remaining["qa"]))
        mc_count = max(0, min(mc_count, remaining["mc"]))
        qa_dist = _normalize_distribution(qa_dist)
        mc_dist = _normalize_distribution(mc_dist)

        min_mult = max(1.0, float(payload.get("min_candidate_multiplier", 1.5)))
        max_mult = max(min_mult, float(payload.get("max_candidate_multiplier", 2.0)))

        return NextRoundQuestionPlan(
            qa_count=qa_count,
            mc_count=mc_count,
            qa_difficulty_distribution=qa_dist,
            mc_difficulty_distribution=mc_dist,
            topic_budget=max(1, int(payload.get("topic_budget", 2))),
            min_candidate_multiplier=min_mult,
            max_candidate_multiplier=max_mult,
            min_selected_per_round=(
                previous_question_plan.min_selected_per_round
                if previous_question_plan and previous_question_plan.min_selected_per_round is not None
                else blueprint.stop_conditions.min_selected_per_round
            ),
            qa_max_rounds=(
                previous_question_plan.qa_max_rounds
                if previous_question_plan and previous_question_plan.qa_max_rounds is not None
                else blueprint.default_modes["qa"].max_rounds
            ),
            mc_max_rounds=(
                previous_question_plan.mc_max_rounds
                if previous_question_plan and previous_question_plan.mc_max_rounds is not None
                else blueprint.default_modes["multiple_choice"].max_rounds
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(f"[Planner] LLM evolver output invalid: {exc}, falling back to rule-based")
        return question_difficulty_evolver(
            state=state, blueprint=blueprint, diagnosis=diagnosis,
            previous_question_plan=previous_question_plan,
            latest_gen_fb=latest_gen_fb, latest_val_fb=latest_val_fb,
        )


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
    if label == "cold_start":
        # 无反馈数据：保持上一轮参数，仅对首轮设置合理的默认值
        if state.current_round == 1:
            count_multiplier = 1.8
            topic_budget = 2
    elif label == "verification_rejected_all":
        # 全部候选被验证拒绝：扩大候选池、严格筛选、降低 hard 比例避免浪费
        count_multiplier = 2.6
        min_candidate_multiplier = 2.0
        max_candidate_multiplier = 2.6
        topic_budget = 1
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.1)
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.1)
        mc_dist["hard"] = max(0.0, mc_dist.get("hard", 0.0) - 0.1)
        mc_dist["medium"] = min(1.0, mc_dist.get("medium", 0.0) + 0.1)
    elif label == "insufficient_eval_data":
        # 评估数据不足：适度增加候选量以获取更多评估信号
        count_multiplier = 2.0
        min_candidate_multiplier = max(min_candidate_multiplier, 1.8)
        max_candidate_multiplier = max(max_candidate_multiplier, 2.2)
        topic_budget = max(topic_budget, 2)
    elif label == "high_quality_low_separation":
        qa_dist["hard"] = min(1.0, qa_dist.get("hard", 0.0) + 0.1)
        qa_dist["easy"] = max(0.0, qa_dist.get("easy", 0.0) - 0.1)
        mc_dist["hard"] = min(1.0, mc_dist.get("hard", 0.0) + 0.1)
        mc_dist["easy"] = max(0.0, mc_dist.get("easy", 0.0) - 0.1)
        topic_budget = 1
    elif label == "low_quality_low_separation":
        count_multiplier = 2.4
        min_candidate_multiplier = 1.8
        max_candidate_multiplier = 2.4
        topic_budget = 1
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.05)
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.05)
        mc_dist["hard"] = max(0.0, mc_dist.get("hard", 0.0) - 0.05)
        mc_dist["medium"] = min(1.0, mc_dist.get("medium", 0.0) + 0.05)
    elif label == "generation_capacity_insufficient":
        count_multiplier = 1.8
        min_candidate_multiplier = 1.3
        max_candidate_multiplier = 1.8
        topic_budget = 1
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.05)
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.05)
        mc_dist["hard"] = max(0.0, mc_dist.get("hard", 0.0) - 0.05)
        mc_dist["medium"] = min(1.0, mc_dist.get("medium", 0.0) + 0.05)
    elif label == "high_separation_imbalanced_distribution":
        qa_dist["medium"] = min(1.0, qa_dist.get("medium", 0.0) + 0.05)
        qa_dist["hard"] = max(0.0, qa_dist.get("hard", 0.0) - 0.05)
        mc_dist["medium"] = min(1.0, mc_dist.get("medium", 0.0) + 0.05)
        mc_dist["hard"] = max(0.0, mc_dist.get("hard", 0.0) - 0.05)

    if state.current_round == 1 and blueprint.initial_generation_strategy == "balanced_exploration":
        topic_budget = max(topic_budget, 2)
        qa_dist = _normalize_distribution(qa_dist)
        mc_dist = _normalize_distribution(mc_dist)

    qa_mode_fb = (latest_gen_fb.by_mode or {}).get("qa") if latest_gen_fb else None
    mc_mode_fb = (latest_gen_fb.by_mode or {}).get("multiple_choice") if latest_gen_fb else None
    # verification_rejected_all 场景下跳过 fulfillment 检查：
    # 全拒必然导致 fulfillment 低，不应反向削减 count_multiplier
    if label != "verification_rejected_all":
        if (qa_mode_fb and qa_mode_fb.fulfillment_rate < 0.8) or (mc_mode_fb and mc_mode_fb.fulfillment_rate < 0.8):
            count_multiplier = min(count_multiplier, 1.8)
            min_candidate_multiplier = max(min_candidate_multiplier, 1.7)
            max_candidate_multiplier = max(max_candidate_multiplier, 2.2)
            topic_budget = min(topic_budget, 1)

    hard_gen_count = 0
    if qa_mode_fb:
        hard_gen_count += int((qa_mode_fb.difficulty_counts or {}).get("hard", 0) or 0)
    if mc_mode_fb:
        hard_gen_count += int((mc_mode_fb.difficulty_counts or {}).get("hard", 0) or 0)
    hard_selected_rate = _difficulty_selected_rate(latest_val_fb, "hard")
    medium_selected_rate = _difficulty_selected_rate(latest_val_fb, "medium")

    if hard_gen_count >= 3 and hard_selected_rate is not None and hard_selected_rate < 0.2:
        qa_dist = _rebalance_distribution(qa_dist, increase="medium", decrease="hard", amount=0.08)
        mc_dist = _rebalance_distribution(mc_dist, increase="medium", decrease="hard", amount=0.08)
    elif (
        hard_selected_rate is not None
        and medium_selected_rate is not None
        and hard_selected_rate > medium_selected_rate + 0.2
    ):
        qa_dist = _rebalance_distribution(qa_dist, increase="hard", decrease="easy", amount=0.06)
        mc_dist = _rebalance_distribution(mc_dist, increase="hard", decrease="easy", amount=0.06)

    # verification_rejected_all 场景下 final_selected=0 是预期行为，不应再削减
    if label != "verification_rejected_all" and latest_val_fb and latest_val_fb.summary.final_selected < blueprint.stop_conditions.min_selected_per_round:
        min_candidate_multiplier = max(min_candidate_multiplier, 1.8)
        max_candidate_multiplier = max(max_candidate_multiplier, 2.3)
        count_multiplier = min(count_multiplier, 1.9)

    qa_count = max(0, min(remaining_qa, int(base_qa * count_multiplier)))
    mc_count = max(0, min(remaining_mc, int(base_mc * count_multiplier)))

    # 单类达标：如果某一题型已达到目标，下一轮集中资源生成另一题型
    if remaining_qa <= 0 and qa_count > 0:
        qa_count = 0
        logger.info("[Planner] QA target reached, skipping QA generation this round")
    if remaining_mc <= 0 and mc_count > 0:
        mc_count = 0
        logger.info("[Planner] MC target reached, skipping MC generation this round")

    # 动态调整 mode 内 max_rounds：生成能力不足时收紧，质量恢复时放宽
    default_qa_max = blueprint.default_modes["qa"].max_rounds
    default_mc_max = blueprint.default_modes["multiple_choice"].max_rounds
    qa_max_rounds = previous_question_plan.qa_max_rounds if previous_question_plan and previous_question_plan.qa_max_rounds is not None else default_qa_max
    mc_max_rounds = previous_question_plan.mc_max_rounds if previous_question_plan and previous_question_plan.mc_max_rounds is not None else default_mc_max
    if label in ("generation_capacity_insufficient", "verification_rejected_all"):
        qa_max_rounds = max(1, qa_max_rounds - 1)
        mc_max_rounds = max(1, mc_max_rounds - 1)
    elif label in ("high_quality_low_separation", "high_separation_imbalanced_distribution"):
        # 质量良好时逐步恢复到默认值
        qa_max_rounds = min(default_qa_max, qa_max_rounds + 1)
        mc_max_rounds = min(default_mc_max, mc_max_rounds + 1)

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
        qa_max_rounds=qa_max_rounds,
        mc_max_rounds=mc_max_rounds,
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
    min_citation_score = 0.55
    min_chunk_citation_score = 0.60
    min_answer_citation_score = 0.55
    min_overall_score = 0.70
    selection_mode = "light"
    semantic_similarity_threshold = 0.9
    eval_profile = "standard"
    initial_breadth_enabled = state.current_round == 1

    label = diagnosis.label
    if label == "cold_start":
        pass  # 保持默认值，不做调整
    elif label == "verification_rejected_all":
        # 全部被拒：严格筛选 + 提高 citation 阈值 + 全面评估
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
        min_citation_score = max(min_citation_score, 0.72)
        min_chunk_citation_score = max(min_chunk_citation_score, 0.88)
        min_answer_citation_score = max(min_answer_citation_score, 0.80)
        min_overall_score = max(min_overall_score, 0.78)
        eval_profile = "full"
    elif label == "insufficient_eval_data":
        # 评估数据不足：标准评估获取更多信号
        eval_profile = "standard"
    elif label == "high_quality_low_separation":
        eval_profile = "full"
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
    elif label == "low_quality_low_separation":
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
        min_citation_score = max(min_citation_score, 0.68)
        min_chunk_citation_score = max(min_chunk_citation_score, 0.87)
        min_answer_citation_score = max(min_answer_citation_score, 0.77)
    elif label == "high_separation_low_quality":
        min_citation_score = 0.7
        min_chunk_citation_score = 0.88
        min_answer_citation_score = 0.78
        min_overall_score = 0.78
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
    elif label == "generation_capacity_insufficient":
        # 生成能力不足：收紧筛选以防低质量题目通过
        selection_mode = "strict"
        semantic_similarity_threshold = 0.92
        min_citation_score = max(min_citation_score, 0.68)
        min_chunk_citation_score = max(min_chunk_citation_score, 0.87)
        min_answer_citation_score = max(min_answer_citation_score, 0.77)
        min_overall_score = max(min_overall_score, 0.78)
    elif label == "high_separation_imbalanced_distribution":
        eval_profile = "full"
        min_chunk_citation_score = max(min_chunk_citation_score, 0.87)
        min_answer_citation_score = max(min_answer_citation_score, 0.77)

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
        mc_mode_fb = (latest_gen_fb.by_mode or {}).get("multiple_choice")
        if (qa_mode_fb and qa_mode_fb.fulfillment_rate < 0.7) or (mc_mode_fb and mc_mode_fb.fulfillment_rate < 0.7):
            initial_breadth_enabled = True

    if latest_eval_fb:
        gap = float(latest_eval_fb.derived_performance_signals.get("overall_model_gap") or 0.0)
        too_easy = float(latest_eval_fb.derived_performance_signals.get("easy_questions_too_easy_ratio") or 0.0)
        all_fail = float(latest_eval_fb.derived_performance_signals.get("all_models_fail_ratio") or 0.0)
        if gap < 0.1 and too_easy >= 0.4:
            eval_profile = "full"
        if all_fail >= 0.4:
            # 用 min 而非 max：题目难度过高导致所有模型失败时，降低门槛
            # 让高难度题目通过筛选（它们有区分价值，而非低质量）
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
                "max_rounds": (
                    question_plan.qa_max_rounds
                    if question_plan.qa_max_rounds is not None
                    else blueprint.default_modes["qa"].max_rounds
                ),
                "difficulty_distribution": question_plan.qa_difficulty_distribution,
            },
            "multiple_choice": {
                "count": question_plan.mc_count,
                "max_rounds": (
                    question_plan.mc_max_rounds
                    if question_plan.mc_max_rounds is not None
                    else blueprint.default_modes["multiple_choice"].max_rounds
                ),
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

    if gen_fb:
        state.resource_usage.total_input_tokens += gen_fb.summary.llm_input_tokens
        state.resource_usage.total_output_tokens += gen_fb.summary.llm_output_tokens
    if val_fb:
        state.resource_usage.total_input_tokens += val_fb.summary.llm_input_tokens
        state.resource_usage.total_output_tokens += val_fb.summary.llm_output_tokens
    if eval_fb:
        state.resource_usage.total_input_tokens += eval_fb.summary.llm_input_tokens
        state.resource_usage.total_output_tokens += eval_fb.summary.llm_output_tokens

    for agent_name, input_tokens, output_tokens in (
        ("qa_agent", gen_fb.summary.llm_input_tokens if gen_fb else 0, gen_fb.summary.llm_output_tokens if gen_fb else 0),
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
