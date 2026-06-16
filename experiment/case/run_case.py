"""Case runner: Group D full method + verify_agent driven by YAML configs.

Usage:
  cd /Users/zhaoziqing/Desktop/benchforge
  python experiment/case/run_case.py
"""

# ============================================================
# 运行模式：True = 只跑验证；False = 完整流程（生成 + 验证）
# ============================================================

from __future__ import annotations

VERIFY_ONLY = True
# VERIFY_ONLY = True   # 去掉注释即可只跑 verify

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from loguru import logger

from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
from benchforge.agents.verify_agent.config_loader import load_verify_agent_config
from benchforge.config.config import load_dotenv
from benchforge.models.loader import ModelLoader


BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
BLUEPRINT_PATH = CONFIG_DIR / "case_blueprint.yaml"
QA_CONFIG_PATH = CONFIG_DIR / "qa_agent_case_d_full.yaml"
VERIFY_CONFIG_PATH = CONFIG_DIR / "verify_agent_case.yaml"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_blueprint(raw: dict) -> Blueprint:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_prefix = raw.get("run_id_prefix", "case")
    run_id = f"{run_id_prefix}_{timestamp}"

    modes = {
        mode: ModeCfg(
            count=int(cfg["count"]),
            max_rounds=int(cfg["max_rounds"]),
            difficulty_distribution=dict(cfg["difficulty_distribution"]),
        )
        for mode, cfg in raw["modes"].items()
    }

    return Blueprint(
        task_id=str(raw.get("task_id", "case")),
        run_id=run_id,
        language=str(raw.get("language", "en")),
        topics=list(raw.get("topics", [])),
        modes=modes,
    )


def _save_metadata(run_dir: Path, blueprint: Blueprint, blueprint_raw: dict) -> None:
    meta = dict(blueprint_raw.get("metadata", {}))
    meta.update(
        {
            "task_id": blueprint.task_id,
            "run_id": blueprint.run_id,
            "language": blueprint.language,
            "topics": blueprint.topics,
            "modes": {
                mode: {
                    "count": cfg.count,
                    "max_rounds": cfg.max_rounds,
                    "difficulty_distribution": cfg.difficulty_distribution,
                }
                for mode, cfg in blueprint.modes.items()
            },
            "qa_config_path": str(QA_CONFIG_PATH),
            "verify_config_path": str(VERIFY_CONFIG_PATH),
            "blueprint_config_path": str(BLUEPRINT_PATH),
            "registry_path": str(REGISTRY_PATH),
        }
    )
    with open(run_dir / "case_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_verify_model_client(config_path: Path):
    verify_cfg = load_verify_agent_config(config_path)
    registry = load_model_registry(REGISTRY_PATH)
    model_key = verify_cfg.llm_validation.model
    if model_key not in registry:
        raise KeyError(
            f"verify_agent model '{model_key}' not found in {REGISTRY_PATH}. "
            f"Available: {list(registry.keys())}"
        )
    return verify_cfg, ModelLoader.load_model(registry[model_key])


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("CUSTOM_API_KEY"):
        raise RuntimeError(f"CUSTOM_API_KEY not set. Expected .env at {PROJECT_ROOT / '.env'}")
    if not os.getenv("CUSTOM_API_BASE_URL"):
        raise RuntimeError(f"CUSTOM_API_BASE_URL not set. Expected .env at {PROJECT_ROOT / '.env'}")

    blueprint_raw = _load_yaml(BLUEPRINT_PATH)
    blueprint = _build_blueprint(blueprint_raw)

    if VERIFY_ONLY:
        case_dir = PROJECT_ROOT / "runs" / "case"
        if not case_dir.exists():
            raise FileNotFoundError(f"No runs/case/ directory found")
        dirs = sorted(
            [d for d in case_dir.iterdir() if d.is_dir() and d.name.startswith("d_full_case_")],
            reverse=True,
        )
        if not dirs:
            raise FileNotFoundError(f"No d_full_case_* run dirs found under {case_dir}")
        run_dir = dirs[0]
        shared_state_path = run_dir / "shared_state.json"
        if not shared_state_path.exists():
            raise FileNotFoundError(f"shared_state.json not found: {shared_state_path}")
    else:
        run_dir = PROJECT_ROOT / "runs" / blueprint.task_id / blueprint.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        shared_state_path = run_dir / "shared_state.json"

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    logger.add(
        run_dir / "case.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        encoding="utf-8",
    )

    _save_metadata(run_dir, blueprint, blueprint_raw)

    if VERIFY_ONLY:
        logger.info("Verify-only mode: using {}", shared_state_path)
    else:
        logger.info("Starting case generation: task={} run={}", blueprint.task_id, blueprint.run_id)
        generation_report = await run_generation_agent(
            blueprint=blueprint,
            config_path=QA_CONFIG_PATH,
            registry_path=REGISTRY_PATH,
        )

        print("\n=== Generation Report ===")
        print(f"task_id : {generation_report['task_id']}")
        print(f"run_id  : {generation_report['run_id']}")
        for mode, state in generation_report["modes"].items():
            print(
                f"{mode:16s} {state['candidate_count']:>3d}/{state['target_candidate_count']:<3d} "
                f"stopped={state['stopped_reason']}"
            )
        print(f"total_candidates : {generation_report['total_candidates']}")

    verify_cfg, verify_model_client = _load_verify_model_client(VERIFY_CONFIG_PATH)
    logger.info("Starting verification for {}", shared_state_path)
    verify_result = await run_verify_agent_from_shared_state(
        shared_state_path=shared_state_path,
        config=verify_cfg,
        model_client=verify_model_client,
    )

    validation_report_path = run_dir / "validation" / "validation_report.json"
    validation_report = {}
    if validation_report_path.exists():
        validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))

    print("\n=== Verification Report ===")
    print(f"selected_question_ids : {len(verify_result.selected_question_ids)}")
    print(f"failed_by_stage       : {verify_result.failed_by_stage}")
    if validation_report:
        print(f"citation_passed       : {validation_report.get('citation_passed')}")
        print(f"llm_passed            : {validation_report.get('llm_passed')}")
        print(f"final_selected        : {validation_report.get('final_selected')}")
        print(f"llm_usage             : {validation_report.get('llm_usage')}")

    print(f"\nrun_dir               : {run_dir}")
    print(f"case_metadata         : {run_dir / 'case_metadata.json'}")
    print(f"validation_report     : {validation_report_path}")


if __name__ == "__main__":
    asyncio.run(main())
