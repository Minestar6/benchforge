"""Group A：Direct Generation — 不经 qa_agent 主循环，直接生成基线。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from common import build_blueprint, save_metadata
from direct_generation import run_direct_generation

GROUP_ID = "A"
METHOD = "direct"
CONFIG_PATH = str(Path(__file__).parent / "configs" / "qa_agent_a_direct.yaml")


async def run(
    *,
    model_client,
    model_name: str,
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
        model_name=model_name,
        extra={
            "enable_feedback": False,
            "enable_hard_generate": False,
            "enable_difficulty_adaptation": False,
            "is_direct_baseline": True,
        },
    )

    logger.info(f"=== Group A: Direct Generation [{mode}] === run_id={blueprint.run_id}")
    await run_direct_generation(
        blueprint=blueprint,
        model_client=model_client,
        model_name=model_name,
        output_dir=output_dir,
        config_path=CONFIG_PATH,
    )
    logger.info(f"Group A completed. run_id={blueprint.run_id}")


if __name__ == "__main__":
    from benchforge.models.fake import FakeModelClient
    asyncio.run(run(
        model_client=FakeModelClient(delay=0.0),
        model_name="fake",
        seed=42,
        topics=["Climate Change", "Artificial Intelligence"],
        task_id="qa_ablation",
        language="en",
        qa_count=20, max_rounds=1,
        difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
    ))
