"""PlannerAgent 工具函数。"""

from typing import Any
from .schema import GlobalBlueprint


def deep_merge(base: dict[str, Any], *patches: dict[str, Any]) -> dict[str, Any]:
    """递归深度合并多个 dict，后续 patch 覆盖前面的值。

    None 值跳过（不覆盖），用于表示"不修改此字段"。

    Args:
        base: 基础配置字典
        *patches: 一个或多个 patch 字典

    Returns:
        合并后的新字典
    """
    result = dict(base)

    for patch in patches:
        for key, value in patch.items():
            if value is None:
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value

    return result


def translate_eval_profile(
    eval_profile: str | None,
    global_blueprint: GlobalBlueprint,
) -> dict[str, Any]:
    """将 eval_profile hint 翻译成 model_eval_agent patch。

    根据 docs/plan-runtime-aligned.md § 9.3：
    - light: dataset_evaluation=true, judge=false, max_tokens=512
    - standard: dataset_evaluation=true, judge=true
    - full: dataset_evaluation=true, judge=true, max_tokens=1024 (gen), 1200 (judge)

    同时注入 GlobalBlueprint.evaluation_requirements.llm_judge_metrics。

    Args:
        eval_profile: "light" | "standard" | "full" | None
        global_blueprint: 全局蓝图（用于读取 llm_judge_metrics）

    Returns:
        model_eval_agent patch dict
    """
    if eval_profile is None:
        return {}

    patch: dict[str, Any] = {}

    if eval_profile == "light":
        patch = {
            "dataset_evaluation": {"enabled": True},
            "models": {"generation_defaults": {"max_tokens": 512}},
            "judge": {"enabled": False},
        }
    elif eval_profile == "standard":
        patch = {
            "dataset_evaluation": {"enabled": True},
            "judge": {"enabled": True},
        }
    elif eval_profile == "full":
        patch = {
            "dataset_evaluation": {"enabled": True},
            "judge": {"enabled": True},
            "models": {
                "generation_defaults": {"max_tokens": 1024},
                "judge_defaults": {"max_tokens": 1200},
            },
        }

    # 注入 llm_judge_metrics（来自全局需求）
    if eval_profile in ("standard", "full"):
        judge_metrics_raw = global_blueprint.evaluation_requirements.llm_judge_metrics
        patch["metrics"] = {}
        for mode, metrics in judge_metrics_raw.items():
            patch["metrics"][mode] = {
                "llm_judge_metrics": [
                    {"name": m.name, "description": m.description}
                    for m in metrics
                ]
            }

    return patch
