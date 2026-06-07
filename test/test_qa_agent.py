"""Test script for Mode-Staged Generation Agent (qa_agent)."""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.config import QuestionGeneratorConfig
from benchforge.models import OpenAIClient
from benchforge.agents.question_generator.modules.evidence_manager import EvidenceManager
from benchforge.agents.question_generator.modules.generator import Generator
from benchforge.agents.qa_agent import run_generation_agent
from benchforge.agents.qa_agent.schema import (
    Blueprint, ModeCfg, AgentConfig,
    CandidatePoolConfig, InitialBreadthConfig, PlannerConfig,
    ChunkMixConfig, ChunkMixDifficulty, ModeAdjustment,
    GenerationYield, ChunkLimitsForMode, ChunkKLimit, RuntimeConfig,
)


def build_default_config() -> AgentConfig:
    return AgentConfig(
        candidate_pool=CandidatePoolConfig(target_multiplier=2.5),
        initial_breadth=InitialBreadthConfig(enabled=True, max_topics_per_round=10, difficulty="medium"),
        planner=PlannerConfig(topics_per_round=3),
        chunk_mix=ChunkMixConfig(
            by_difficulty={
                "easy":   ChunkMixDifficulty(single_ratio=0.8, multi_ratio=0.2),
                "medium": ChunkMixDifficulty(single_ratio=0.5, multi_ratio=0.5),
                "hard":   ChunkMixDifficulty(single_ratio=0.2, multi_ratio=0.8),
            },
            mode_adjustment={
                "qa":              ModeAdjustment(single_delta=0.1),
                "multiple_choice": ModeAdjustment(single_delta=-0.1),
            },
        ),
        generation_yield={
            "qa":              GenerationYield(single_chunk_avg_questions=2.0, multi_chunk_avg_questions=3.0),
            "multiple_choice": GenerationYield(single_chunk_avg_questions=1.5, multi_chunk_avg_questions=2.0),
        },
        chunk_limits={
            "qa": ChunkLimitsForMode(
                single_k=ChunkKLimit(min=1, max=4),
                multi_k=ChunkKLimit(min=0, max=3),
            ),
            "multiple_choice": ChunkLimitsForMode(
                single_k=ChunkKLimit(min=0, max=3),
                multi_k=ChunkKLimit(min=1, max=4),
            ),
        },
        runtime=RuntimeConfig(),
    )


async def run():
    from benchforge.agents.qa_agent.config_loader import load_qa_agent_config

    blueprint, agent_config, model_cfg = load_qa_agent_config("benchforge/config/qa_agent.yaml")

    sys_config = QuestionGeneratorConfig.from_yaml(
        "benchforge/config/question_generator_config.yaml",
        task_id=blueprint.task_id,
        run_id=blueprint.run_id,
    )

    client = OpenAIClient(
        api_key=model_cfg["api_key"],
        model_name=model_cfg["model_name"],
        base_url=model_cfg["base_url"],
    )

    evidence_manager = EvidenceManager(sys_config, client)

    generator = Generator()

    report = await run_generation_agent(
        blueprint=blueprint,
        config=agent_config,
        evidence_manager=evidence_manager,
        generator=generator,
    )

    print(f"\n=== Generation Report ===")
    print(f"task_id : {report['task_id']}")
    print(f"run_id  : {report['run_id']}")
    for mode, summary in report["modes"].items():
        print(
            f"  {mode}: {summary['candidate_count']}/{summary['target_candidate_count']} "
            f"candidates, stopped={summary['stopped_reason']}"
        )
    print(f"total   : {report['total_candidates']} candidates")
    print(f"combos  : {report['global_used_chunk_combinations']} chunk combinations used")
    print(f"Output  : runs/{report['task_id']}/{report['run_id']}/")


if __name__ == "__main__":
    asyncio.run(run())
