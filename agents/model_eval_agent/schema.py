"""model_eval_agent 所有核心数据类型。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunConfig:
    shared_state_path: str = ""
    task_id: str = ""
    run_id: str = ""
    input_paths: list[str] = field(default_factory=list)


@dataclass
class DatasetMetricSpec:
    name: str
    threshold: float | None = None


@dataclass
class DatasetEvaluationConfig:
    enabled: bool = True
    metrics: list[DatasetMetricSpec] = field(default_factory=list)


@dataclass
class AutoMetricSpec:
    name: str
    threshold: float | None = None


@dataclass
class JudgeMetricSpec:
    name: str
    description: str
    direction: str = "higher_is_better"  # "higher_is_better" | "lower_is_better"


@dataclass
class QuestionModeMetricPlan:
    automatic_metrics: list[AutoMetricSpec] = field(default_factory=list)
    llm_judge_metrics: list[JudgeMetricSpec] = field(default_factory=list)


@dataclass
class ModelsConfig:
    candidate_model_names: list[str] = field(default_factory=list)
    judge_model_name: str | None = None
    generation_defaults: dict[str, Any] = field(default_factory=dict)
    judge_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    enabled: bool = False
    prompt_system: str = ""
    prompt_user: str = ""


@dataclass
class ModelEvalAgentConfig:
    run: RunConfig
    dataset_evaluation: DatasetEvaluationConfig
    models: ModelsConfig
    metrics: dict[str, QuestionModeMetricPlan]
    judge: JudgeConfig
