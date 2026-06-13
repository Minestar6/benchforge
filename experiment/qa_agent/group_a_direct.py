"""Group A：Direct Generation — max_rounds=1，禁用自适应，作为基线。"""
import asyncio, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from benchforge.models.fake import FakeModelClient
from agents.qa_agent.agent import run_generation_agent
import agents.qa_agent.agent as agent_mod
from fixtures import make_blueprint, CONFIG_PATH


async def run(
    model_client=None,
    model_name: str = "fake",
    topics: list[str] | None = None,
    task_id: str = "exp_task",
    language: str = "en",
    qa_count: int = 5,
    difficulty_distribution: dict[str, float] | None = None,
    **kwargs,
):
    if model_client is None:
        model_client = FakeModelClient(delay=0.0)
    if topics is None:
        topics = ["Climate Change", "Artificial Intelligence"]

    logger.info("=== Group A: Direct Generation ===")
    run_id = f"run_a_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    blueprint = make_blueprint(
        count=qa_count, max_rounds=1, run_id=run_id,
        task_id=task_id, topics=topics, language=language,
        difficulty_distribution=difficulty_distribution,
    )

    agent_mod._resolve_model_client_fn = lambda name, path: model_client

    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=CONFIG_PATH,
    )
    logger.info(f"Group A completed. run_id={run_id}")
    return report


if __name__ == "__main__":
    asyncio.run(run())
