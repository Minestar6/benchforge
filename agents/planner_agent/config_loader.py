"""PlannerAgent 配置加载与保存。"""

import json
from pathlib import Path
import yaml

from .schema import GlobalBlueprint, PlannerState


def load_global_blueprint(path: str | Path) -> GlobalBlueprint:
    """从 JSON/YAML 加载 GlobalBlueprint。

    Args:
        path: global_blueprint.json 或 .yaml 文件路径

    Returns:
        GlobalBlueprint 实例
    """
    path = Path(path)

    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

    return GlobalBlueprint(**data)


def save_global_blueprint(blueprint: GlobalBlueprint, path: str | Path) -> None:
    """保存 GlobalBlueprint 为 JSON。

    Args:
        blueprint: GlobalBlueprint 实例
        path: 输出文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(blueprint.model_dump(), f, ensure_ascii=False, indent=2)


def load_planner_state(path: str | Path) -> PlannerState:
    """从 JSON 加载 PlannerState。

    Args:
        path: planner_state.json 文件路径

    Returns:
        PlannerState 实例
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return PlannerState(**data)


def save_planner_state(state: PlannerState, path: str | Path) -> None:
    """保存 PlannerState 为 JSON。

    Args:
        state: PlannerState 实例
        path: 输出文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)


def initialize_planner_state(blueprint: GlobalBlueprint) -> PlannerState:
    """从 GlobalBlueprint 初始化 PlannerState。

    Args:
        blueprint: 全局蓝图

    Returns:
        初始化的 PlannerState
    """
    return PlannerState(
        task_id=blueprint.task_id,
        blueprint_id=blueprint.blueprint_id,
        current_round=0,
        topic_backlog={
            "active": list(blueprint.seed_topics),
            "deferred": [],
        },
        model_pool={
            "active": list(blueprint.evaluator_defaults.candidate_model_names),
            "removed": [],
        },
    )
