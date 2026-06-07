"""ModelEvaluationAgent — 模型评测智能体。"""

from .agent import run_model_eval_agent, run_model_eval_agent_from_shared_state
from .schema import ModelEvalAgentConfig, ModelsConfig, RunConfig
from .metrics_registry import AUTO_METRIC_REGISTRY, DATASET_METRIC_REGISTRY

__all__ = [
    "run_model_eval_agent",
    "run_model_eval_agent_from_shared_state",
    "ModelEvalAgentConfig",
    "ModelsConfig",
    "RunConfig",
    "AUTO_METRIC_REGISTRY",
    "DATASET_METRIC_REGISTRY",
]
