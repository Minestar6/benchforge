"""model_eval_agent.yaml → dataclass 配置加载。"""

from pathlib import Path

import yaml

from benchforge.config.config import load_dotenv, expand_env_recursive

from .schema import (
    AutoMetricSpec,
    DatasetEvaluationConfig,
    DatasetMetricSpec,
    JudgeConfig,
    JudgeMetricSpec,
    ModelEvalAgentConfig,
    ModelsConfig,
    QuestionModeMetricPlan,
    RunConfig,
)


def load_model_eval_config(path: str | Path) -> ModelEvalAgentConfig:
    from benchforge.utils.paths import get_project_root
    path = Path(path)
    load_dotenv(get_project_root() / ".env")

    with open(path, encoding="utf-8") as f:
        raw = expand_env_recursive(yaml.safe_load(f))

    run_raw = raw.get("run", {})
    run = RunConfig(
        shared_state_path=run_raw.get("shared_state_path", "").strip(),
        task_id=run_raw.get("task_id", "").strip(),
        run_id=run_raw.get("run_id", "").strip(),
        input_paths=run_raw.get("input_paths", []),
    )

    ds_raw = raw.get("dataset_evaluation", {})
    dataset_evaluation = DatasetEvaluationConfig(
        enabled=ds_raw.get("enabled", True),
        metrics=[
            DatasetMetricSpec(name=m["name"], threshold=m.get("threshold"))
            for m in ds_raw.get("metrics", [])
        ],
    )

    models_raw = raw.get("models", {})
    models = ModelsConfig(
        candidate_model_names=models_raw.get("candidate_model_names", []),
        judge_model_name=models_raw.get("judge_model_name"),
        generation_defaults=models_raw.get("generation_defaults", {}),
        judge_defaults=models_raw.get("judge_defaults", {}),
    )

    metrics: dict[str, QuestionModeMetricPlan] = {}
    for mode, plan_raw in raw.get("metrics", {}).items():
        metrics[mode] = QuestionModeMetricPlan(
            automatic_metrics=[
                AutoMetricSpec(name=m["name"], threshold=m.get("threshold"))
                for m in plan_raw.get("automatic_metrics", [])
            ],
            llm_judge_metrics=[
                JudgeMetricSpec(
                    name=m["name"],
                    description=m.get("description", ""),
                )
                for m in plan_raw.get("llm_judge_metrics", [])
            ],
        )

    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(
        enabled=judge_raw.get("enabled", False),
        prompt_system=judge_raw.get("prompt_system", ""),
        prompt_user=judge_raw.get("prompt_user", ""),
    )

    return ModelEvalAgentConfig(
        run=run,
        dataset_evaluation=dataset_evaluation,
        models=models,
        metrics=metrics,
        judge=judge,
    )
