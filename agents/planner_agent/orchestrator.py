"""PlannerAgent 单轮编排器（docs/plan-runtime-aligned.md § 6.1）。

职责：
1. 合并 base YAML + patch，落盘 effective config
2. 初始化 EvidenceManager + Generator，调用 run_generation_agent
3. 更新 shared_state，调用 verify_agent / model_eval_agent
4. 返回三份 feedback
"""

import json
import yaml
from pathlib import Path
from loguru import logger

from benchforge.config import QuestionGeneratorConfig
from benchforge.models.loader import ModelLoader
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry
from benchforge.agents.qa_agent.config_loader import load_qa_agent_config
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
from benchforge.agents.qa_agent.evidence_manager import EvidenceManager
from benchforge.agents.qa_agent.generator import Generator
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.verify_agent.config_loader import load_verify_agent_config
from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
from benchforge.agents.model_eval_agent.config_loader import load_model_eval_config
from benchforge.agents.model_eval_agent.agent import run_model_eval_agent_from_shared_state
from benchforge.utils.shared_state import build_shared_state, save_shared_state

from .schema import RoundSpec, GlobalBlueprint, GeneratorFeedback, ValidatorFeedback, EvaluatorFeedback
from .utils import deep_merge, translate_eval_profile
from .feedback import build_generator_feedback, build_validator_feedback, build_evaluator_feedback


def _load_model_client_from_registry(registry: dict, model_name: str):
    model_cfg = registry.get(model_name)
    if not model_cfg:
        raise ValueError(f"Model '{model_name}' not found in registry")
    return ModelLoader.load_model(model_cfg)


def _load_verify_model_client(registry: dict, verify_config):
    if not verify_config.llm_validation.enabled:
        return None
    model_name = verify_config.llm_validation.model
    if not model_name:
        return None
    return _load_model_client_from_registry(registry, model_name)


async def execute_round(
    round_spec: RoundSpec,
    global_blueprint: GlobalBlueprint,
    base_config_dir: str | Path,
    registry_path: str | Path,
) -> tuple[GeneratorFeedback, ValidatorFeedback | None, EvaluatorFeedback | None]:
    """执行单轮编排，返回三份 feedback。

    Args:
        round_spec: 本轮规划
        global_blueprint: 全局蓝图
        base_config_dir: config/ 目录路径（包含 qa_agent.yaml 等）
        registry_path: model_registry.yaml 路径

    Returns:
        (gen_feedback, val_feedback, eval_feedback)
    """
    base_config_dir = Path(base_config_dir)
    run_dir = Path("runs") / round_spec.task_id / round_spec.run_id
    planner_dir = run_dir / "planner"
    planner_dir.mkdir(parents=True, exist_ok=True)

    # 1. 落盘 round_spec.json
    round_spec_path = planner_dir / "round_spec.json"
    with open(round_spec_path, "w", encoding="utf-8") as f:
        json.dump(round_spec.model_dump(), f, ensure_ascii=False, indent=2)

    logger.info(f"[Orchestrator] Round {round_spec.round_id} start, run_id={round_spec.run_id}")

    # 2. 合并 YAML，落盘 effective config
    qa_base_path = base_config_dir / "qa_agent.yaml"
    verify_base_path = base_config_dir / "verify_agent.yaml"
    eval_base_path = base_config_dir / "model_eval_agent.yaml"

    with open(qa_base_path, encoding="utf-8") as f:
        qa_base = yaml.safe_load(f)
    with open(verify_base_path, encoding="utf-8") as f:
        verify_base = yaml.safe_load(f)
    with open(eval_base_path, encoding="utf-8") as f:
        eval_base = yaml.safe_load(f)

    # 翻译 eval_profile
    eval_profile_patch = translate_eval_profile(round_spec.planner_hints.eval_profile, global_blueprint)

    effective_qa = deep_merge(qa_base, round_spec.qa_agent_patch)
    effective_verify = deep_merge(verify_base, round_spec.verify_agent_patch)
    effective_eval = deep_merge(eval_base, round_spec.model_eval_agent_patch, eval_profile_patch)

    # 落盘
    with open(planner_dir / "qa_agent.patch.yaml", "w", encoding="utf-8") as f:
        yaml.dump(round_spec.qa_agent_patch, f, allow_unicode=True)
    with open(planner_dir / "verify_agent.patch.yaml", "w", encoding="utf-8") as f:
        yaml.dump(round_spec.verify_agent_patch, f, allow_unicode=True)
    with open(planner_dir / "model_eval_agent.patch.yaml", "w", encoding="utf-8") as f:
        yaml.dump(round_spec.model_eval_agent_patch, f, allow_unicode=True)
    with open(planner_dir / "translated_model_eval.patch.yaml", "w", encoding="utf-8") as f:
        yaml.dump(eval_profile_patch, f, allow_unicode=True)

    effective_qa_path = planner_dir / "effective_qa_agent.yaml"
    effective_verify_path = planner_dir / "effective_verify_agent.yaml"
    effective_eval_path = planner_dir / "effective_model_eval_agent.yaml"

    with open(effective_qa_path, "w", encoding="utf-8") as f:
        yaml.dump(effective_qa, f, allow_unicode=True)
    with open(effective_verify_path, "w", encoding="utf-8") as f:
        yaml.dump(effective_verify, f, allow_unicode=True)
    with open(effective_eval_path, "w", encoding="utf-8") as f:
        yaml.dump(effective_eval, f, allow_unicode=True)

    # 3. 加载 qa_agent config + 构造 Blueprint
    agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg = load_qa_agent_config(effective_qa_path)

    blueprint = Blueprint(
        task_id=round_spec.blueprint["task_id"],
        run_id=round_spec.blueprint["run_id"],
        language=round_spec.blueprint["language"],
        topics=round_spec.blueprint["topics"],
        modes={
            mode_name: ModeCfg(
                count=mode_cfg["count"],
                max_rounds=mode_cfg["max_rounds"],
                difficulty_distribution=mode_cfg["difficulty_distribution"],
            )
            for mode_name, mode_cfg in round_spec.blueprint["modes"].items()
        },
    )

    # 4. 构建 sys_config（EvidenceManager 需要）
    sys_config = QuestionGeneratorConfig.from_yaml(
        base_config_dir.parent / "question_generator_config.yaml"
        if (base_config_dir.parent / "question_generator_config.yaml").exists()
        else base_config_dir / "question_generator_config.yaml",
        task_id=blueprint.task_id,
        run_id=blueprint.run_id,
    )
    sys_config.retrieval = retrieval_cfg
    sys_config.chunking = chunking_cfg
    sys_config.summarization_chunking = sum_chunking_cfg

    # 5. 加载模型客户端
    registry = load_model_registry(registry_path)
    model_client = _load_model_client_from_registry(registry, model_ref.name)

    # 6. 运行 qa_agent
    logger.info(f"[Orchestrator] Running qa_agent: {len(blueprint.topics)} topics")
    await run_generation_agent(
        blueprint=blueprint,
        config=agent_config,
        evidence_manager=EvidenceManager(sys_config, model_client),
        generator=Generator(),
    )

    # 7. 更新 shared_state（兼容 verify_agent 读 state.blueprint）
    shared_state_path = run_dir / "shared_state.json"
    state = build_shared_state(
        blueprint.task_id,
        blueprint.run_id,
        blueprint,
        run_dir,
        round_id=round_spec.round_id,
        round_spec_ref=str(round_spec_path),
    )

    # 写入 mode_state artifact keys
    for mode in blueprint.modes:
        mode_state_path = run_dir / mode / "mode_state.json"
        if mode_state_path.exists():
            state.add_artifact(f"{mode}_mode_state", str(mode_state_path))

    save_shared_state(state, run_dir)

    gen_feedback = build_generator_feedback(shared_state_path, round_spec)

    if gen_feedback.summary.total_candidates == 0:
        logger.warning("[Orchestrator] No candidates generated, skipping verify/eval")
        return gen_feedback, None, None

    # 8. 运行 verify_agent
    logger.info("[Orchestrator] Running verify_agent")
    verify_config = load_verify_agent_config(effective_verify_path)
    verify_model_client = _load_verify_model_client(registry, verify_config)
    await run_verify_agent_from_shared_state(shared_state_path, verify_config, verify_model_client)

    val_feedback = build_validator_feedback(shared_state_path, round_spec.round_id)
    if not val_feedback or val_feedback.summary.final_selected == 0:
        logger.warning("[Orchestrator] No selected questions, skipping eval")
        return gen_feedback, val_feedback, None

    # 9. 运行 model_eval_agent
    logger.info("[Orchestrator] Running model_eval_agent")
    eval_config = load_model_eval_config(effective_eval_path)
    await run_model_eval_agent_from_shared_state(shared_state_path, eval_config, str(registry_path))

    eval_feedback = build_evaluator_feedback(shared_state_path, round_spec.round_id)

    logger.info(
        f"[Orchestrator] Round {round_spec.round_id} done: "
        f"generated={gen_feedback.summary.total_candidates}, "
        f"selected={val_feedback.summary.final_selected if val_feedback else 0}, "
        f"evaluated={eval_feedback.summary.num_questions if eval_feedback else 0}"
    )

    return gen_feedback, val_feedback, eval_feedback
