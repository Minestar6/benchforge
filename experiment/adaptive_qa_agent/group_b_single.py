"""Group B：Single-round Agent — 有 Verifier，无 Feedback Loop（max_rounds=1）。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from benchforge.models.fake import FakeModelClient
from agents.adaptive_qa_agent.agent import run_adaptive_generation_agent
from agents.verify_agent.schema import ValidationTaskResult
from fixtures import make_blueprint, make_config, make_evidence_manager, make_verify_config, make_adaptive_config


async def run():
    logger.info("=== Group B: Single-round Agent ===")
    blueprint = make_blueprint(count=5, max_rounds=1, run_id="run_b_single")
    config = make_config()
    adaptive_config = make_adaptive_config()
    evidence_manager = make_evidence_manager()

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
            model_client=FakeModelClient(delay=0.0),
            verify_config=make_verify_config(),
        )
    logger.info("Group B completed.")


if __name__ == "__main__":
    asyncio.run(run())
