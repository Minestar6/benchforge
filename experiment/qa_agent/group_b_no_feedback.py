"""Group B：Multi-round No Feedback — 多轮、固定策略/难度、无反馈。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from benchforge.agents.qa_agent.agent import run_generation_agent
from common import build_blueprint, save_metadata

GROUP_ID = "B"
METHOD = "no_feedback"
CONFIG_PATH = str(Path(__file__).parent / "configs" / "qa_agent_b_no_feedback.yaml")


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
            "fixed_strategy": "normal_generate",
            "fixed_difficulty": "medium",
            "disable_initial_breadth": True,
        },
    )

    logger.info(f"=== Group B: Multi-round No Feedback [{mode}] === run_id={blueprint.run_id}")
    await run_generation_agent(
        blueprint=blueprint,
        config_path=CONFIG_PATH,
    )
    logger.info(f"Group B completed. run_id={blueprint.run_id}")


if __name__ == "__main__":
    from benchforge.models.fake import FakeModelClient
    import benchforge.agents.qa_agent.agent as agent_mod
    agent_mod._resolve_model_client_fn = lambda name, path: FakeModelClient(delay=0.0)
    asyncio.run(run(
        model_client=FakeModelClient(delay=0.0),
        model_name="fake",
        seed=42,
        topics=["Climate Change", "Artificial Intelligence"],
        task_id="qa_ablation",
        language="en",
        qa_count=20, max_rounds=5,
        difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
    ))
