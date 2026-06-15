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
    max_rounds: int = 5,
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

    _TOPIC_TEXTS = {
        "Climate Change": (
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "Since the Industrial Revolution, human activities—primarily burning fossil fuels—have been "
            "the main driver of climate change. Rising CO2 levels trap heat in the atmosphere, causing "
            "global warming. Consequences include melting ice caps, rising sea levels, more frequent "
            "extreme weather events, and disruption to ecosystems and agriculture. The Paris Agreement "
            "aims to limit warming to 1.5°C above pre-industrial levels."
        ),
        "Artificial Intelligence": (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural "
            "intelligence displayed by animals including humans. AI research has been defined as the field "
            "of study of intelligent agents, which refers to any system that perceives its environment and "
            "takes actions that maximize its chance of achieving its goals. Modern AI includes machine "
            "learning, deep learning, and large language models. Applications span healthcare diagnostics, "
            "autonomous vehicles, natural language processing, and scientific research."
        ),
    }
    _DEFAULT_TEXT = (
        "This topic covers a broad range of concepts. Key aspects include historical development, "
        "current applications, major challenges, and future directions. Researchers have identified "
        "multiple sub-domains, each requiring specialized knowledge and methodologies."
    )

    def _pool(topic):
        text = _TOPIC_TEXTS.get(topic, _DEFAULT_TEXT)
        chunk = SimpleNamespace(
            chunk_id=f"{topic}_c1",
            document_id=f"{topic}_doc1",
            text=text,
            usage_count=0,
            qa_score=0.8,
            mcq_score=0.7,
        )
        return SimpleNamespace(topic=topic, single_chunks=[chunk], multi_chunks=[])

    pools = {t: _pool(t) for t in topics}

    def _sample(topic, **kwargs):
        return SimpleNamespace(
            single_chunk_ids=[f"{topic}_c1"],
            multi_chunk_ids=[],
            requested_min_questions=1,
            requested_target_questions=2,
        )

    def _get_evidence_text(chunk_id, **kwargs):
        topic = chunk_id.rsplit("_c", 1)[0]
        return _TOPIC_TEXTS.get(topic, _DEFAULT_TEXT)

    em = MagicMock()
    em.evidence_pools = pools
    em.sample = MagicMock(side_effect=_sample)
    em.get_evidence_text = MagicMock(side_effect=_get_evidence_text)
    em.get_document_summary = MagicMock(side_effect=lambda topic, **kw: f"Summary of {topic}.")
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
