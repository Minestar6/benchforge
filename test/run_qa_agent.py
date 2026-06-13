"""Run the qa_agent pipeline end-to-end using qa_agent.yaml.

Blueprint 由调用方程序化构造（未来由规划智能体产生）。
task_id / run_id 也在此指定，不在 YAML 中。
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg

TASK_ID = "task_001"
RUN_ID = "run_001"


def build_blueprint() -> Blueprint:
    """由编排器/规划智能体程序化构造。"""
    return Blueprint(
        task_id=TASK_ID,
        run_id=RUN_ID,
        language="en",
        topics=[
            "Artificial Intelligence",
            "Renewable Energy",
            "Human Evolution",
        ],
        modes={
            "qa": ModeCfg(
                count=20,
                max_rounds=5,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            ),
            "multiple_choice": ModeCfg(
                count=10,
                max_rounds=5,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            ),
        },
    )


async def main():
    blueprint = build_blueprint()

    # 日志落盘
    log_path = Path("runs") / blueprint.task_id / blueprint.run_id / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG", encoding="utf-8")

    # 端到端调用：只需 blueprint + config_path
    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=project_root / "benchforge/config/qa_agent.yaml",
    )

    print(f"\n=== Generation Report ===")
    print(f"task_id : {report['task_id']}")
    print(f"run_id  : {report['run_id']}")
    for mode, s in report["modes"].items():
        print(f"  {mode}: {s['candidate_count']}/{s['target_candidate_count']} candidates  stopped={s['stopped_reason']}")
    print(f"total   : {report['total_candidates']} candidates")
    print(f"combos  : {report['global_used_chunk_combinations']} chunk combinations used")
    print(f"output  : runs/{report['task_id']}/{report['run_id']}/")
    print(f"log     : {log_path}")


if __name__ == "__main__":
    asyncio.run(main())