"""PlannerAgent 配置加载与保存。"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import yaml

from benchforge.config.config import expand_env_recursive, load_dotenv
from benchforge.utils.paths import get_project_root

from .schema import GlobalBlueprint, PlannerState


_DEFAULT_PROMPTS = {
    "blueprint_synthesizer": "prompts/planner_agent/blueprint_synthesizer_prompt.md",
    "user_goal_analyzer": "prompts/planner_agent/goal_analyzer_prompt.md",
    "topic_adaptive_retriever": "prompts/planner_agent/topic_adaptive_retriever_prompt.md",
    "question_difficulty_evolver": "prompts/planner_agent/difficulty_count_planner_prompt.md",
    "control_parameter_tuner": "prompts/planner_agent/control_parameter_tuner_prompt.md",
}
_DEFAULT_DIFFICULTY = {
    "qa": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    "multiple_choice": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
}


@dataclass
class PlannerModelRef:
    name: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1200
    max_retries: int = 3


@dataclass
class PlannerPromptConfig:
    blueprint_synthesizer: str = _DEFAULT_PROMPTS["blueprint_synthesizer"]
    user_goal_analyzer: str = _DEFAULT_PROMPTS["user_goal_analyzer"]
    topic_adaptive_retriever: str = _DEFAULT_PROMPTS["topic_adaptive_retriever"]
    question_difficulty_evolver: str = _DEFAULT_PROMPTS["question_difficulty_evolver"]
    control_parameter_tuner: str = _DEFAULT_PROMPTS["control_parameter_tuner"]


@dataclass
class QuestionTypeAllocationConfig:
    qa: float = 0.5
    multiple_choice: float = 0.5
    rounding: str = "largest_remainder"


@dataclass
class InitialPlanningConfig:
    question_type_allocation: QuestionTypeAllocationConfig = field(default_factory=QuestionTypeAllocationConfig)
    difficulty_distribution: dict[str, dict[str, float]] = field(default_factory=lambda: dict(_DEFAULT_DIFFICULTY))
    max_rounds_per_mode: int = 3


@dataclass
class DifficultyCountPlannerConfig:
    enabled: bool = False
    fallback: str = "rule_based"


@dataclass
class PlannerAgentConfig:
    model: PlannerModelRef = field(default_factory=PlannerModelRef)
    prompts: PlannerPromptConfig = field(default_factory=PlannerPromptConfig)
    initial_planning: InitialPlanningConfig = field(default_factory=InitialPlanningConfig)
    question_difficulty_evolver: DifficultyCountPlannerConfig = field(default_factory=DifficultyCountPlannerConfig)


def _normalize_difficulty(raw: dict | None) -> dict[str, dict[str, float]]:
    raw = raw if isinstance(raw, dict) else {}
    normalized: dict[str, dict[str, float]] = {}
    for mode, defaults in _DEFAULT_DIFFICULTY.items():
        values = raw.get(mode) if isinstance(raw.get(mode), dict) else defaults
        parts = {key: float((values or {}).get(key, defaults[key]) or 0.0) for key in ("easy", "medium", "hard")}
        total = sum(parts.values()) or 1.0
        normalized[mode] = {key: parts[key] / total for key in ("easy", "medium", "hard")}
    return normalized


def load_planner_agent_config(path_or_dir: str | Path | None) -> PlannerAgentConfig:
    """加载 planner_agent 静态配置；缺失时返回默认值。"""
    if path_or_dir is None:
        path = get_project_root() / "config" / "planner_agent.yaml"
    else:
        path = Path(path_or_dir)
        if path.is_dir() or path.suffix == "":
            path = path / "planner_agent.yaml"

    load_dotenv(get_project_root() / ".env")
    if not path.exists():
        return PlannerAgentConfig()

    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f) or {})

    model_raw = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    prompts_raw = raw.get("prompts") if isinstance(raw.get("prompts"), dict) else {}
    initial_raw = raw.get("initial_planning") if isinstance(raw.get("initial_planning"), dict) else {}
    allocation_raw = initial_raw.get("question_type_allocation") if isinstance(initial_raw.get("question_type_allocation"), dict) else {}
    difficulty_raw = initial_raw.get("difficulty_distribution") if isinstance(initial_raw.get("difficulty_distribution"), dict) else {}
    difficulty_planner_raw = raw.get("question_difficulty_evolver")
    if not isinstance(difficulty_planner_raw, dict):
        difficulty_planner_raw = raw.get("difficulty_count_planner") if isinstance(raw.get("difficulty_count_planner"), dict) else {}

    return PlannerAgentConfig(
        model=PlannerModelRef(
            name=model_raw.get("name") or None,
            temperature=float(model_raw.get("temperature", 0.2)),
            max_tokens=int(model_raw.get("max_tokens", 1200)),
            max_retries=int(model_raw.get("max_retries", 3)),
        ),
        prompts=PlannerPromptConfig(
            blueprint_synthesizer=prompts_raw.get("blueprint_synthesizer", _DEFAULT_PROMPTS["blueprint_synthesizer"]),
            user_goal_analyzer=prompts_raw.get(
                "user_goal_analyzer",
                prompts_raw.get("goal_analyzer", _DEFAULT_PROMPTS["user_goal_analyzer"]),
            ),
            topic_adaptive_retriever=prompts_raw.get("topic_adaptive_retriever", _DEFAULT_PROMPTS["topic_adaptive_retriever"]),
            question_difficulty_evolver=prompts_raw.get(
                "question_difficulty_evolver",
                prompts_raw.get("difficulty_count_planner", _DEFAULT_PROMPTS["question_difficulty_evolver"]),
            ),
            control_parameter_tuner=prompts_raw.get(
                "control_parameter_tuner",
                _DEFAULT_PROMPTS["control_parameter_tuner"],
            ),
        ),
        initial_planning=InitialPlanningConfig(
            question_type_allocation=QuestionTypeAllocationConfig(
                qa=float(allocation_raw.get("qa", 0.5)),
                multiple_choice=float(allocation_raw.get("multiple_choice", 0.5)),
                rounding=str(allocation_raw.get("rounding", "largest_remainder")),
            ),
            difficulty_distribution=_normalize_difficulty(difficulty_raw),
            max_rounds_per_mode=max(1, int(initial_raw.get("max_rounds_per_mode", 3))),
        ),
        question_difficulty_evolver=DifficultyCountPlannerConfig(
            enabled=bool(difficulty_planner_raw.get("enabled", False)),
            fallback=str(difficulty_planner_raw.get("fallback", "rule_based")),
        ),
    )


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
