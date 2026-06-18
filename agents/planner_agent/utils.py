"""PlannerAgent 工具函数。"""

from typing import Any

from .schema import GlobalBlueprint


def deep_merge(base: dict[str, Any], *patches: dict[str, Any]) -> dict[str, Any]:
    """递归深度合并多个 dict，后续 patch 覆盖前面的值。

    - None 值跳过（不覆盖），用于表示"不修改此字段"。
    - dict 类型递归合并。
    - list 类型扩展合并（patch list 追加到 base list 后面）。
    - 其他类型直接覆盖。
    """
    result = dict(base)

    for patch in patches:
        for key, value in patch.items():
            if value is None:
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value

    return result


def merge_model_eval_config(base: dict[str, Any], *patches: dict[str, Any]) -> dict[str, Any]:
    """合并 model_eval_agent 配置。

    `metrics` 是评估协议真源，必须整体覆盖，不能按 list 追加。
    其他字段继续沿用通用 deep_merge 行为。
    """
    result = dict(base)

    for patch in patches:
        if not patch:
            continue
        patch_without_metrics = {key: value for key, value in patch.items() if key != "metrics"}
        result = deep_merge(result, patch_without_metrics)
        if "metrics" in patch and patch["metrics"] is not None:
            result["metrics"] = patch["metrics"]

    return result


def build_frozen_metrics_patch(global_blueprint: GlobalBlueprint) -> dict[str, Any]:
    """从 GlobalBlueprint 构建冻结后的 metrics patch。

    GlobalBlueprint.evaluation_requirements 是 planner 后续轮次唯一允许读取的评估指标真源。
    如果蓝图中没有显式 metrics，则返回空 patch，保留基础配置。
    """
    requirements = global_blueprint.evaluation_requirements
    automatic_metrics = requirements.automatic_metrics or {}
    llm_judge_metrics = requirements.llm_judge_metrics or {}

    mode_names = sorted(set(automatic_metrics) | set(llm_judge_metrics))
    if not mode_names:
        return {}

    metrics_patch: dict[str, Any] = {"metrics": {}}
    for mode in mode_names:
        mode_patch: dict[str, Any] = {}

        auto_metrics = automatic_metrics.get(mode)
        if auto_metrics is not None:
            mode_patch["automatic_metrics"] = [
                {"name": metric_name}
                for metric_name in auto_metrics
                if str(metric_name).strip()
            ]

        judge_metrics = llm_judge_metrics.get(mode)
        if judge_metrics is not None:
            mode_patch["llm_judge_metrics"] = [
                {"name": metric.name, "description": metric.description}
                for metric in judge_metrics
                if metric.name and metric.description
            ]

        metrics_patch["metrics"][mode] = mode_patch

    return metrics_patch


def translate_eval_profile(
    eval_profile: str | None,
    global_blueprint: GlobalBlueprint,
) -> dict[str, Any]:
    """将 eval_profile hint 翻译成 model_eval_agent patch。

    职责边界：
    - 只负责评估强度翻译（dataset_evaluation / token defaults）
    - 始终从 GlobalBlueprint 注入冻结后的 metrics 真源
    - 不再在这里改写 judge.enabled，由 RoundSpec.model_eval_agent_patch 控制
    """
    metrics_patch = build_frozen_metrics_patch(global_blueprint)
    if eval_profile is None:
        return metrics_patch

    patch: dict[str, Any] = {}

    if eval_profile == "light":
        patch = {
            "dataset_evaluation": {"enabled": True},
            "models": {"generation_defaults": {"max_tokens": 512}},
        }
    elif eval_profile == "standard":
        patch = {
            "dataset_evaluation": {"enabled": True},
        }
    elif eval_profile == "full":
        patch = {
            "dataset_evaluation": {"enabled": True},
            "models": {
                "generation_defaults": {"max_tokens": 1024},
                "judge_defaults": {"max_tokens": 1200},
            },
        }

    return deep_merge(patch, metrics_patch)
