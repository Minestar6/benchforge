"""Group C：Feedback No Difficulty Evolution — 有反馈但无难度进化。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from benchforge.agents.qa_agent.agent import run_generation_agent
from common import build_blueprint, save_metadata

GROUP_ID = "C"
METHOD = "feedback_no_diff"
CONFIG_PATH = str(Path(__file__).parent / "configs" / "qa_agent_c_feedback_no_difficulty.yaml")


async def run(
    *,
    seed: int,
    topics: list[str],
    task_id: str,
    language: str,
    qa_count: int,
    max_rounds: int,
    difficulty_distribution: dict[str, float],
    mode: str = "qa",
    timestamp: str = "",
):
    blueprint = build_blueprint(
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        topics=topics,
        task_id=task_id,
        language=language,
        mode=mode,
        count=qa_count,
        max_rounds=max_rounds,
        difficulty_distribution=difficulty_distribution,
        timestamp=timestamp,
    )

    output_dir = Path("runs") / blueprint.task_id / blueprint.run_id

    save_metadata(
        output_dir,
        group_id=GROUP_ID,
        method=METHOD,
        seed=seed,
        blueprint=blueprint,
        config_path=CONFIG_PATH,
        extra={
            "enable_feedback": True,
            "enable_hard_generate": False,
            "enable_difficulty_adaptation": False,
            "disable_initial_breadth": False,
        },
    )

    logger.info(f"=== Group C: Feedback No Difficulty Evolution [{mode}] === run_id={blueprint.run_id}")
    await run_generation_agent(
        blueprint=blueprint,
        config_path=CONFIG_PATH,
    )
    logger.info(f"Group C completed. run_id={blueprint.run_id}")


if __name__ == "__main__":
    asyncio.run(run(
        seed=42,
        topics=["Climate Change", "Artificial Intelligence"],
        task_id="qa_ablation",
        language="en",
        qa_count=20, max_rounds=5,
        difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
    ))
