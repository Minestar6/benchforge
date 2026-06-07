"""运行 verify_agent，对最近题目生成结果进行验证。"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from benchforge.agents.verify_agent.agent import run_verify_agent
from benchforge.agents.verify_agent.config_loader import (
    CitationCfg, LLMValidationCfg, SelectionCfg, VerifyAgentConfig,
)
from benchforge.agents.verify_agent.schema import ModeCfg, ValidationBlueprintView

TASK_ID = "task_001"
RUN_ID = "run_20260602_163344_4441"

blueprint = ValidationBlueprintView(
    topics=["Artificial Intelligence", "Renewable Energy", "Human Evolution"],
    modes={
        "qa": ModeCfg(
            count=20,
            difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
        ),
    },
)

config = VerifyAgentConfig(
    task_id=TASK_ID,
    run_id=RUN_ID,
    input_paths=[f"runs/{TASK_ID}/{RUN_ID}/qa/candidate_pool.json"],
    citation=CitationCfg(
        enabled=True,
        min_citation_score=0.65,
        alpha=0.7,
        beta=0.3,
        citation_match_threshold=0.8,
    ),
    llm_validation=LLMValidationCfg(enabled=False),
    selection=SelectionCfg(
        enabled=True,
        embedding_model=str(project_root / "models" / "all-MiniLM-L6-v2"),
    ),
)


async def main():
    logger.add(f"runs/{TASK_ID}/{RUN_ID}/verify.log", level="DEBUG", encoding="utf-8")
    result = await run_verify_agent(
        input_paths=config.input_paths,
        blueprint=blueprint,
        config=config,
        model_client=None,
    )
    logger.info(f"selected={len(result.selected_question_ids)}")
    logger.info(f"failed_by_stage={result.failed_by_stage}")
    logger.info(f"report={result.report_path}")


if __name__ == "__main__":
    asyncio.run(main())
