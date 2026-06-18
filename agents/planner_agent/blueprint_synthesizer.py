"""Synthesize GlobalBlueprint from user intent with an LLM prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchforge.config.config import load_prompt
from benchforge.models.base import BaseModelClient
from benchforge.models.loader import ModelLoader
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry

from .schema import (
    EvaluationRequirements,
    EvaluatorDefaults,
    FinalTargets,
    GlobalBlueprint,
    JudgeMetricDef,
    QuestionModeDefaults,
    StopConditions,
    UserIntent,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "planner_agent" / "blueprint_synthesizer_prompt.md"
_GOAL_ANALYZER_PROMPT_PATH = _PROJECT_ROOT / "prompts" / "planner_agent" / "goal_analyzer_prompt.md"
_DEFAULT_DIFFICULTY = {
    "qa": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "multiple_choice": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
}
_DEFAULT_AUTOMATIC_METRICS = {
    "qa": ["exact_match", "f1"],
    "multiple_choice": ["accuracy"],
}
_DEFAULT_JUDGE_METRICS = {
    "qa": [
        JudgeMetricDef(name="correctness", description="Judge whether the answer is factually correct."),
        JudgeMetricDef(name="faithfulness", description="Judge whether the answer is supported by the provided evidence."),
    ],
    "multiple_choice": [],
}


def _planner_temperature(model_client, default: float = 0.2) -> float:
    value = getattr(model_client, "temperature", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _planner_max_tokens(model_client, cap: int) -> int:
    value = getattr(model_client, "max_tokens", cap)
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = cap
    return max(1, min(resolved, cap))


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Planner synthesis response did not contain a JSON object")
    raw = text[start : end + 1]
    # 尝试修复常见的 JSON 格式错误：最后一个字段后缺少逗号、多逗号等
    import re
    # 移除对象/数组末尾多余的逗号
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # 给未加引号的 key 加引号（使用 \s 分组替代 look-behind）
    raw = re.sub(r"([{,]\s*)(\w[\w\d_]*)\s*(?=:)", lambda m: m.group(1) + '"' + m.group(2) + '"', raw)
    # 修复十六进制转义（\x -> \\x）
    raw = raw.replace("\\x", "\\\\x")
    return json.loads(raw)


def _looks_like_goal_analysis(payload: dict[str, Any]) -> bool:
    return "initial_topics" in payload or "automatic_metrics_by_type" in payload or "llm_eval_enabled" in payload


def _prompt_with_user_goal(prompt_template: str, user_goal: str) -> str:
    return prompt_template.replace("{user_goal}", json.dumps(user_goal, ensure_ascii=False))


def _eval_requirements_from_goal_analysis(payload: dict[str, Any]) -> EvaluationRequirements:
    automatic_metrics = payload.get("automatic_metrics_by_type")
    if not isinstance(automatic_metrics, dict):
        automatic_metrics = {}

    llm_judge_metrics: dict[str, list[dict[str, str]]] = {"qa": [], "multiple_choice": []}
    if bool(payload.get("llm_eval_enabled")):
        raw_metrics = payload.get("qa_llm_eval_metrics")
        if isinstance(raw_metrics, list):
            llm_judge_metrics["qa"] = [item for item in raw_metrics if isinstance(item, dict)]

    return _normalize_eval_requirements(
        {
            "automatic_metrics": automatic_metrics,
            "llm_judge_metrics": llm_judge_metrics,
        }
    )


async def _run_user_goal_analyzer(
    intent: UserIntent,
    selected_planner_model: str,
    model_client: BaseModelClient,
) -> dict[str, Any] | None:
    prompt_template = load_prompt(_GOAL_ANALYZER_PROMPT_PATH)
    prompt = _prompt_with_user_goal(prompt_template, intent.user_goal)
    response = await model_client.complete(
        model=model_client.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=_planner_temperature(model_client),
        max_tokens=_planner_max_tokens(model_client, 800),
    )
    payload = _extract_json_object(response.get("text", ""))
    return payload if _looks_like_goal_analysis(payload) else None


def _normalize_difficulty_distribution(raw: Any, mode: str) -> dict[str, float]:
    defaults = dict(_DEFAULT_DIFFICULTY[mode])
    if not isinstance(raw, dict):
        return defaults

    values = {key: float(raw.get(key, 0.0) or 0.0) for key in ("easy", "medium", "hard")}
    total = sum(values.values())
    if total <= 0:
        return defaults
    return {key: values[key] / total for key in ("easy", "medium", "hard")}


def _normalize_mode_defaults(raw_modes: Any, intent: UserIntent) -> dict[str, QuestionModeDefaults]:
    raw_modes = raw_modes if isinstance(raw_modes, dict) else {}
    return {
        "qa": QuestionModeDefaults(
            max_rounds=max(1, int((raw_modes.get("qa") or {}).get("max_rounds", intent.max_rounds))),
            difficulty_distribution=_normalize_difficulty_distribution(
                (raw_modes.get("qa") or {}).get("difficulty_distribution"),
                "qa",
            ),
        ),
        "multiple_choice": QuestionModeDefaults(
            max_rounds=max(1, int((raw_modes.get("multiple_choice") or {}).get("max_rounds", intent.max_rounds))),
            difficulty_distribution=_normalize_difficulty_distribution(
                (raw_modes.get("multiple_choice") or {}).get("difficulty_distribution"),
                "multiple_choice",
            ),
        ),
    }


def _normalize_eval_requirements(raw: Any) -> EvaluationRequirements:
    raw = raw if isinstance(raw, dict) else {}
    automatic_metrics = raw.get("automatic_metrics", {})
    llm_judge_metrics = raw.get("llm_judge_metrics", {})

    normalized_auto = {}
    for mode in ("qa", "multiple_choice"):
        metrics = automatic_metrics.get(mode)
        if not metrics:
            normalized_auto[mode] = list(_DEFAULT_AUTOMATIC_METRICS[mode])
            continue
        normalized_auto[mode] = [str(metric).strip() for metric in metrics if str(metric).strip()]

    normalized_judge: dict[str, list[JudgeMetricDef]] = {}
    for mode in ("qa", "multiple_choice"):
        if mode not in llm_judge_metrics:
            normalized_judge[mode] = list(_DEFAULT_JUDGE_METRICS[mode])
            continue

        raw_items = llm_judge_metrics.get(mode) or []
        items: list[JudgeMetricDef] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if name and description:
                items.append(JudgeMetricDef(name=name, description=description))
        normalized_judge[mode] = items

    return EvaluationRequirements(
        automatic_metrics=normalized_auto,
        llm_judge_metrics=normalized_judge,
    )


def _resolve_default_models(
    intent: UserIntent,
    registry_path: str | Path,
) -> EvaluatorDefaults:
    registry = load_model_registry(registry_path)
    available = list(registry.keys())
    if not available:
        raise ValueError("Model registry is empty")

    candidate_model_names = intent.candidate_model_names or [available[0]]
    missing_candidates = [name for name in candidate_model_names if name not in registry]
    if missing_candidates:
        raise ValueError(f"Unknown candidate models: {missing_candidates}")

    judge_model_name = intent.judge_model_name or candidate_model_names[0]
    if judge_model_name not in registry:
        raise ValueError(f"Unknown judge model: {judge_model_name}")
    return EvaluatorDefaults(
        candidate_model_names=candidate_model_names,
        judge_model_name=judge_model_name,
    )


async def synthesize_global_blueprint(
    intent: UserIntent,
    registry_path: str | Path,
    planner_model_name: str | None = None,
    planner_model_settings: dict[str, Any] | None = None,
    model_client: BaseModelClient | None = None,
) -> GlobalBlueprint:
    """Generate a validated GlobalBlueprint from user input."""
    registry = load_model_registry(registry_path)
    if model_client is None:
        selected_planner_model = (
            planner_model_name
            or intent.planner_model_name
            or intent.judge_model_name
            or next(iter(registry.keys()), None)
        )
        if not selected_planner_model or selected_planner_model not in registry:
            raise ValueError(f"Planner model '{selected_planner_model}' not found in registry")
        model_client = ModelLoader.load_model(registry[selected_planner_model])
    else:
        selected_planner_model = planner_model_name or intent.planner_model_name or getattr(model_client, "model_name", "planner")

    if planner_model_settings:
        if "temperature" in planner_model_settings:
            setattr(model_client, "temperature", planner_model_settings["temperature"])
        if "max_tokens" in planner_model_settings:
            setattr(model_client, "max_tokens", max(int(planner_model_settings["max_tokens"]), 1))
        if "max_retries" in planner_model_settings:
            setattr(model_client, "max_retries", int(planner_model_settings["max_retries"]))

    goal_analysis = None
    try:
        goal_analysis = await _run_user_goal_analyzer(intent, selected_planner_model, model_client)
    except Exception as exc:
        from loguru import logger
        logger.warning(f"[PlannerBlueprint] goal_analyzer failed with model '{selected_planner_model}': {exc}")
        goal_analysis = None

    if goal_analysis is None:
        prompt_template = load_prompt(_PROMPT_PATH)
        prompt = (
            f"{prompt_template}\n\n"
            f"User intent:\n{json.dumps(intent.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Available model registry keys:\n{json.dumps(list(registry.keys()), ensure_ascii=False)}\n"
        )

        response = await model_client.complete(
            model=model_client.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=_planner_temperature(model_client),
            max_tokens=_planner_max_tokens(model_client, 1200),
        )
        try:
            payload = _extract_json_object(response.get("text", ""))
        except (ValueError, json.JSONDecodeError) as exc2:
            logger.warning(f"[PlannerBlueprint] fallback synthesis also failed: {exc2}")
            raise ValueError(
                "Planner synthesis failed in both primary and fallback paths. "
                "The model did not return valid JSON."
            ) from exc2
    else:
        payload = goal_analysis

    task_id = intent.resolved_task_id()
    topic_key = "initial_topics" if goal_analysis is not None else "seed_topics"
    seed_topics = intent.seed_topics or [str(topic).strip() for topic in payload.get(topic_key, []) if str(topic).strip()]
    if not seed_topics:
        raise ValueError("Synthesized blueprint did not produce any seed topics")

    return GlobalBlueprint(
        task_id=task_id,
        blueprint_id=f"{task_id}-bp",
        user_goal=intent.user_goal,
        language=intent.language,
        seed_topics=seed_topics,
        initial_generation_strategy=str(
            payload.get("initial_generation_strategy") or "balanced_exploration"
        ),
        final_targets=FinalTargets(
            qa=max(0, int(intent.qa_target)),
            multiple_choice=max(0, int(intent.multiple_choice_target)),
        ),
        default_modes=_normalize_mode_defaults(payload.get("default_modes"), intent),
        evaluator_defaults=_resolve_default_models(intent, registry_path),
        evaluation_requirements=(
            _eval_requirements_from_goal_analysis(payload)
            if goal_analysis is not None
            else _normalize_eval_requirements(payload.get("evaluation_requirements"))
        ),
        stop_conditions=StopConditions(
            max_rounds=max(1, int(intent.max_rounds)),
            min_selected_per_round=max(1, int(intent.min_selected_per_round)),
            max_total_tokens=int(intent.max_total_tokens) if intent.max_total_tokens is not None else None,
        ),
    )
