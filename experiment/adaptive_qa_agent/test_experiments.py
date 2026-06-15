"""四组实验的 pytest 入口，从项目根目录执行：
  pytest experiment/adaptive_qa_agent/test_experiments.py -v -s
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agents.adaptive_qa_agent.agent import run_adaptive_generation_agent, _load_adaptive_config
from agents.verify_agent.config_loader import VerifyAgentConfig
from agents.verify_agent.schema import ValidationTaskResult
from benchforge.models.fake import FakeModelClient


# ── fixtures ────────────────────────────────────────────────────────────────

def _blueprint(count=5, max_rounds=5, run_id="exp_run"):
    ns = SimpleNamespace()
    ns.task_id = "exp_task"
    ns.run_id = run_id
    ns.language = "en"
    ns.topics = ["topic_a", "topic_b"]
    ns.modes = {
        "qa": SimpleNamespace(
            count=count, max_rounds=max_rounds,
            difficulty_distribution={"easy": 0.2, "medium": 0.3, "hard": 0.5},
        ),
        "mcq": SimpleNamespace(
            count=count, max_rounds=max_rounds,
            difficulty_distribution={"easy": 0.2, "medium": 0.3, "hard": 0.5},
        ),
    }
    return ns


def _config():
    return SimpleNamespace(candidate_pool=SimpleNamespace(target_multiplier=2.0))


def _evidence_manager():
    topics = ["topic_a", "topic_b"]
    def _pool(t):
        chunk = SimpleNamespace(chunk_id=f"{t}_c1", document_id=f"{t}_doc1",
                                text="Evidence.", usage_count=0, qa_score=0.8, mcq_score=0.7)
        return SimpleNamespace(topic=t, single_chunks=[chunk], multi_chunks=[])
    batch = SimpleNamespace(single_chunk_ids=["topic_a_c1"], multi_chunk_ids=[],
                            requested_min_questions=1, requested_target_questions=2)
    em = MagicMock()
    em.evidence_pools = {t: _pool(t) for t in topics}
    em.sample = MagicMock(return_value=batch)
    em.get_evidence_text = MagicMock(return_value="Evidence text.")
    em.get_document_summary = MagicMock(return_value="Summary.")
    em.expand_retrieval = AsyncMock(return_value=None)
    return em


def _adaptive_config(hard_gap_threshold=0.2, too_easy_ratio=0.4):
    cfg = _load_adaptive_config()
    cfg.stop.max_empty_rounds = 3
    cfg.stop.max_failures = 10
    cfg.execution.concurrency = 2
    cfg.generation.max_tokens = 512
    cfg.decision.hard_gap_threshold = hard_gap_threshold
    cfg.decision.too_easy_ratio = too_easy_ratio
    return cfg


def _accept_all_mock(blueprint):
    async def _verify(**kw):
        return ValidationTaskResult(
            task_id=blueprint.task_id, run_id=blueprint.run_id, report_path="",
            selected_question_ids=[c.question_id for c in kw["candidates"]],
            failed_by_stage={},
        )
    m = MagicMock()
    m.run = AsyncMock(side_effect=_verify)
    return m


def _partial_accept_mock(blueprint):
    async def _verify(**kw):
        candidates = kw["candidates"]
        accepted = [c.question_id for c in candidates[:max(1, len(candidates) // 2)]]
        return ValidationTaskResult(
            task_id=blueprint.task_id, run_id=blueprint.run_id, report_path="",
            selected_question_ids=accepted, failed_by_stage={},
        )
    m = MagicMock()
    m.run = AsyncMock(side_effect=_verify)
    return m


# ── Group A: Direct Generation ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_group_a_direct():
    """无 Verifier 质量过滤，max_rounds=1，全部接受。"""
    bp = _blueprint(count=5, max_rounds=1, run_id="run_a_direct")
    with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
        MockVerify.return_value = _accept_all_mock(bp)
        await run_adaptive_generation_agent(
            blueprint=bp, config=_config(), adaptive_config=_adaptive_config(),
            evidence_manager=_evidence_manager(),
            model_client=FakeModelClient(delay=0.0),
            verify_config=VerifyAgentConfig(),
        )


# ── Group B: Single-round Agent ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_group_b_single():
    """有 Verifier，max_rounds=1，部分接受（模拟真实质量过滤）。"""
    bp = _blueprint(count=5, max_rounds=1, run_id="run_b_single")
    with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
        MockVerify.return_value = _partial_accept_mock(bp)
        await run_adaptive_generation_agent(
            blueprint=bp, config=_config(), adaptive_config=_adaptive_config(),
            evidence_manager=_evidence_manager(),
            model_client=FakeModelClient(delay=0.0),
            verify_config=VerifyAgentConfig(),
        )


# ── Group C: Multi-round, No Difficulty Adaptation ───────────────────────────

@pytest.mark.asyncio
async def test_group_c_nodiff():
    """多轮运行，关闭难度自适应（hard_gap_threshold=1.0, too_easy_ratio=1.0）。"""
    bp = _blueprint(count=5, max_rounds=5, run_id="run_c_nodiff")
    cfg = _adaptive_config(hard_gap_threshold=1.0, too_easy_ratio=1.0)
    with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
        MockVerify.return_value = _partial_accept_mock(bp)
        await run_adaptive_generation_agent(
            blueprint=bp, config=_config(), adaptive_config=cfg,
            evidence_manager=_evidence_manager(),
            model_client=FakeModelClient(delay=0.0),
            verify_config=VerifyAgentConfig(),
        )


# ── Group D: Full AdaptiveQAAgent ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_group_d_full():
    """全部机制开启，多轮运行，默认阈值配置。"""
    bp = _blueprint(count=5, max_rounds=5, run_id="run_d_full")
    with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
        MockVerify.return_value = _partial_accept_mock(bp)
        await run_adaptive_generation_agent(
            blueprint=bp, config=_config(), adaptive_config=_adaptive_config(),
            evidence_manager=_evidence_manager(),
            model_client=FakeModelClient(delay=0.0),
            verify_config=VerifyAgentConfig(),
        )
