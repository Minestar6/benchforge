"""共用测试 fixtures：blueprint、config、evidence_manager、verify mock。"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.adaptive_qa_agent.agent import _load_adaptive_config
from agents.verify_agent.config_loader import VerifyAgentConfig
from agents.verify_agent.schema import ValidationTaskResult
from benchforge.models.fake import FakeModelClient


def make_blueprint(
    count: int = 5,
    max_rounds: int = 10,
    run_id: str = "exp_run",
    task_id: str = "exp_task",
    topics: list[str] | None = None,
    language: str = "en",
    difficulty_distribution: dict[str, float] | None = None,
):
    if topics is None:
        topics = ["topic_a", "topic_b"]
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    ns = SimpleNamespace()
    ns.task_id = task_id
    ns.run_id = run_id
    ns.language = language
    ns.topics = topics
    ns.modes = {
        "qa": SimpleNamespace(
            count=count,
            max_rounds=max_rounds,
            difficulty_distribution=difficulty_distribution,
        ),
        "mcq": SimpleNamespace(
            count=count,
            max_rounds=max_rounds,
            difficulty_distribution=difficulty_distribution,
        ),
    }
    return ns


def make_config():
    return SimpleNamespace(candidate_pool=SimpleNamespace(target_multiplier=2.0))


def make_evidence_manager(topics=None):
    if topics is None:
        topics = ("topic_a", "topic_b")
    elif isinstance(topics, list):
        topics = tuple(topics)

    def _pool(topic):
        chunk = SimpleNamespace(
            chunk_id=f"{topic}_c1",
            document_id=f"{topic}_doc1",
            text="Sample evidence text.",
            usage_count=0,
            qa_score=0.8,
            mcq_score=0.7,
        )
        return SimpleNamespace(topic=topic, single_chunks=[chunk], multi_chunks=[])

    pools = {t: _pool(t) for t in topics}
    batch = SimpleNamespace(
        single_chunk_ids=[f"{topics[0]}_c1"],
        multi_chunk_ids=[],
        requested_min_questions=1,
        requested_target_questions=2,
    )
    em = MagicMock()
    em.evidence_pools = pools
    em.sample = MagicMock(return_value=batch)
    em.get_evidence_text = MagicMock(return_value="Sample evidence text.")
    em.get_document_summary = MagicMock(return_value="Document summary.")
    em.expand_retrieval = AsyncMock(return_value=None)
    return em


def make_verify_config():
    return VerifyAgentConfig()


def make_accept_all_verify_mock():
    """返回一个 patch context，verify_agent 接受所有候选题。"""
    def _patch(candidates):
        selected_ids = [c.question_id for c in candidates]
        return ValidationTaskResult(
            task_id="exp_task", run_id="exp_run",
            report_path="",
            selected_question_ids=selected_ids,
            failed_by_stage={},
        )
    mock_instance = MagicMock()
    mock_instance.run = AsyncMock(side_effect=lambda **kw: _patch(kw["candidates"]))
    return mock_instance


def make_reject_all_verify_mock():
    """返回一个 mock，verify_agent 拒绝所有候选题。"""
    mock_instance = MagicMock()
    mock_instance.run = AsyncMock(return_value=ValidationTaskResult(
        task_id="exp_task", run_id="exp_run",
        report_path="", selected_question_ids=[], failed_by_stage={},
    ))
    return mock_instance


def make_adaptive_config(hard_gap_threshold=0.2, too_easy_ratio=0.4, max_rounds_override=None):
    cfg = _load_adaptive_config()
    cfg.stop.max_empty_rounds = 3
    cfg.stop.max_failures = 10
    cfg.execution.concurrency = 2
    cfg.decision.hard_gap_threshold = hard_gap_threshold
    cfg.decision.too_easy_ratio = too_easy_ratio
    return cfg
