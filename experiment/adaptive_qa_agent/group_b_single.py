"""Group B：Single-round Agent — 有 Verifier，无 Feedback Loop（max_rounds=1）。

支持从 run_all.py 传入真实模型客户端，或独立运行时使用 FakeModelClient。
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from benchforge.models.fake import FakeModelClient
from agents.adaptive_qa_agent.agent import run_adaptive_generation_agent
from agents.verify_agent.schema import ValidationTaskResult
from fixtures import make_blueprint, make_config, make_evidence_manager, make_verify_config, make_adaptive_config


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
        topics = ["topic_a", "topic_b"]

    logger.info("=== Group B: Single-round Agent ===")
    run_id = f"run_b_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    blueprint = make_blueprint(
        count=qa_count, max_rounds=1, run_id=run_id,
        task_id=task_id, topics=topics, language=language,
        difficulty_distribution=difficulty_distribution,
    )
    config = make_config()
    adaptive_config = make_adaptive_config()
    evidence_manager = make_evidence_manager(topics=topics)

    # Group B：Verifier 存在但只跑一轮（部分接受，模拟真实质量过滤）
    async def _partial_accept(**kw):
        candidates = kw["candidates"]
        accepted = [c.question_id for c in candidates[:max(1, len(candidates) // 2)]]
        return ValidationTaskResult(
            task_id=blueprint.task_id, run_id=blueprint.run_id,
            report_path="", selected_question_ids=accepted, failed_by_stage={},
        )

    with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
        MockVerify.return_value.run = AsyncMock(side_effect=_partial_accept)
        await run_adaptive_generation_agent(
            blueprint=blueprint,
            config=config,
            adaptive_config=adaptive_config,
            evidence_manager=evidence_manager,
            model_client=model_client,
            verify_config=make_verify_config(),
        )
    logger.info("Group B completed.")


if __name__ == "__main__":
    # 独立运行时使用 FakeModelClient（测试用途）
    asyncio.run(run())
