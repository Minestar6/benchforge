"""Group C：Multi-round Agent (No Difficulty Adaptation)。

关闭 hard_gap / hard_multihop_gap / evolution：
  - hard_gap_threshold=1.0  → hard_gap 永远不超过 target_hard_ratio < 1，规则永不触发
  - too_easy_ratio=1.0      → feedback ratio 最大为 1，evolve 规则永不触发
"""
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
    logger.info("=== Group C: Multi-round Agent (No Difficulty Adaptation) ===")
    blueprint = make_blueprint(count=5, max_rounds=10, run_id="run_c_nodiff")
    config = make_config()
    # 关闭难度自适应：hard_gap_threshold=1.0, too_easy_ratio=1.0
    adaptive_config = make_adaptive_config(hard_gap_threshold=1.0, too_easy_ratio=1.0)
    evidence_manager = make_evidence_manager()

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
    logger.info("Group C completed.")


if __name__ == "__main__":
    asyncio.run(run())
