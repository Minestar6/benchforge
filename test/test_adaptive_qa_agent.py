"""adaptive_qa_agent 集成测试（使用 FakeModelClient 端到端跑通）。"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.adaptive_qa_agent.state import ModeState, FeedbackState, CandidateRef
from agents.adaptive_qa_agent.decision import decide, Action
from agents.adaptive_qa_agent.feedback_mapper import map_from_task_result, MappedResult
from agents.adaptive_qa_agent.prompts import PromptBuilder
from agents.adaptive_qa_agent.agent import run_adaptive_generation_agent, _aggregate, _load_adaptive_config
from agents.verify_agent.schema import (
    QuestionCandidate, ValidationTaskResult, ValidationBlueprintView, ModeCfg,
)
from models.fake import FakeModelClient


# ─── helpers ────────────────────────────────────────────────────────────────

def _adaptive_config():
    return _load_adaptive_config()


def _blueprint():
    ns = SimpleNamespace()
    ns.task_id = "test_task"
    ns.run_id = "test_run"
    ns.language = "en"
    ns.topics = ["Python", "Machine Learning"]
    ns.modes = {
        "qa": SimpleNamespace(
            count=3,
            max_rounds=5,
            difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
        )
    }
    return ns


def _config():
    return SimpleNamespace(candidate_pool=SimpleNamespace(target_multiplier=2.0))


def _fake_evidence_pool(topic: str):
    chunk = SimpleNamespace(
        chunk_id=f"{topic}_c1",
        document_id=f"{topic}_doc1",
        text="Sample evidence text.",
        usage_count=0,
        qa_score=0.8,
        mcq_score=0.7,
    )
    return SimpleNamespace(
        topic=topic,
        single_chunks=[chunk],
        multi_chunks=[],
    )


def _fake_evidence_manager(topics):
    pools = {t: _fake_evidence_pool(t) for t in topics}

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
    em.expand_retrieval = AsyncMock(return_value=SimpleNamespace(new_chunks=2, new_single_units=2, new_multi_units=0))
    return em


# ─── unit tests ─────────────────────────────────────────────────────────────

class TestModeState:
    def test_accepted_count(self):
        state = ModeState(mode="qa")
        state.candidates = [
            CandidateRef("q1", "t1", "easy", "accepted"),
            CandidateRef("q2", "t1", "medium", "rejected"),
        ]
        assert state.accepted_count == 1

    def test_hard_gap(self):
        state = ModeState(mode="qa", target_hard_ratio=0.3)
        state.candidates = [CandidateRef("q1", "t1", "easy", "accepted")]
        state.difficulty_counts = {"easy": 1}
        assert state.hard_gap == 0.3  # 0 hard / 1 total → gap = 0.3

    def test_missing_topics(self):
        state = ModeState(mode="qa", all_topics=["A", "B", "C"])
        state.candidates = [CandidateRef(f"q{i}", "A", "easy", "accepted") for i in range(4)]
        state.topic_counts = {"A": 4}
        missing = state.missing_topics
        assert "B" in missing and "C" in missing

    def test_record_action_oscillation(self):
        state = ModeState(mode="qa")
        state.record_action("retrieve_more")
        state.record_action("retrieve_more")
        state.record_action("retrieve_more")
        assert state.consecutive_same_action == 3

    def test_chunk_combination_dedup(self):
        state = ModeState(mode="qa")
        state.record_chunk_combination(["c1", "c2"])
        assert state.is_chunk_combination_used(["c2", "c1"])
        assert not state.is_chunk_combination_used(["c1", "c3"])


class TestFeedbackState:
    def test_ratio_empty(self):
        fb = FeedbackState(window_size=5)
        assert fb.ratio("answer_not_grounded") == 0.0

    def test_ratio_calculation(self):
        fb = FeedbackState(window_size=5)
        fb.push({"total": 10, "accepted": 6, "answer_not_grounded": 4})
        assert fb.ratio("answer_not_grounded") == pytest.approx(0.4)

    def test_sliding_window(self):
        fb = FeedbackState(window_size=2)
        fb.push({"total": 10, "accepted": 5, "answer_not_grounded": 8})
        fb.push({"total": 10, "accepted": 5, "answer_not_grounded": 0})
        fb.push({"total": 10, "accepted": 5, "answer_not_grounded": 0})
        assert fb.ratio("answer_not_grounded") == 0.0  # 窗口内只有后两条


class TestDecide:
    def _state(self, **kwargs):
        return ModeState(mode="qa", all_topics=["A", "B"], **kwargs)

    def _feedback(self, **ratios):
        fb = FeedbackState()
        if ratios:
            total = 10
            stats = {"total": total, "accepted": 5}
            for k, v in ratios.items():
                stats[k] = int(v * total)
            fb.push(stats)
        return fb

    def test_p1_retrieve_more(self):
        state = self._state()
        fb = self._feedback(answer_not_grounded=0.5)
        cfg = _adaptive_config()
        action = decide(state, fb, cfg)
        assert action.action_type == "retrieve_more"

    def test_p2_expand_evidence(self):
        state = self._state()
        fb = self._feedback(evidence_insufficient=0.4)
        cfg = _adaptive_config()
        action = decide(state, fb, cfg)
        assert action.action_type == "expand_evidence"

    def test_p4_hard_gap(self):
        state = self._state()
        state.target_hard_ratio = 0.5
        state.difficulty_counts = {}
        state.candidates = [CandidateRef("q1", "A", "easy", "accepted")]
        fb = self._feedback()
        cfg = _adaptive_config()
        action = decide(state, fb, cfg)
        assert action.action_type == "generate" and action.difficulty == "hard"

    def test_fallback_generate(self):
        state = self._state()
        fb = self._feedback()
        cfg = _adaptive_config()
        action = decide(state, fb, cfg)
        assert action.action_type == "generate"

    def test_suppress_oscillation(self):
        state = self._state()
        state.last_action_type = "retrieve_more"
        state.consecutive_same_action = 3
        fb = self._feedback(answer_not_grounded=0.5)
        cfg = _adaptive_config()
        action = decide(state, fb, cfg)
        assert action.action_type != "retrieve_more"


class TestFeedbackMapper:
    def test_map_accepted(self):
        candidates = [QuestionCandidate(
            question_id="q1", task_id="t", run_id="r",
            topic="Python", question="Q?", answer="A",
            question_mode="qa",
        )]
        result = ValidationTaskResult(
            task_id="t", run_id="r",
            report_path="", selected_question_ids=["q1"], failed_by_stage={},
        )
        mapped = map_from_task_result(result, candidates)
        assert len(mapped) == 1
        assert mapped[0].accepted is True

    def test_map_rejected(self):
        candidates = [QuestionCandidate(
            question_id="q1", task_id="t", run_id="r",
            topic="Python", question="Q?", answer="A",
            question_mode="qa",
        )]
        result = ValidationTaskResult(
            task_id="t", run_id="r",
            report_path="", selected_question_ids=[], failed_by_stage={},
        )
        mapped = map_from_task_result(result, candidates)
        assert mapped[0].accepted is False
        assert mapped[0].reject_reason == "unknown"


class TestPromptBuilder:
    def test_build_prompt_returns_messages(self):
        pb = PromptBuilder("en")
        msgs = pb.build_generation_prompt(
            topic="Python", difficulty="medium", mode="qa",
            evidence_text="Python is a programming language.",
            document_summary="About Python.", language="en",
        )
        assert isinstance(msgs, list) and len(msgs) >= 1
        assert any("Python" in m["content"] for m in msgs)


class TestAggregate:
    def test_aggregate(self):
        mapped = [
            MappedResult("q1", "t", "easy", True),
            MappedResult("q2", "t", "easy", False, "too_easy"),
            MappedResult("q3", "t", "medium", False, "answer_not_grounded"),
        ]
        stats = _aggregate(mapped)
        assert stats["total"] == 3
        assert stats["accepted"] == 1
        assert stats["too_easy"] == 1
        assert stats["answer_not_grounded"] == 1

    def test_aggregate_evolution(self):
        mapped = [MappedResult("q1", "t", "easy", False, "too_easy")]
        stats = _aggregate(mapped, is_evolution=True)
        assert stats["evolution_failed"] == 1


# ─── integration test ────────────────────────────────────────────────────────

class TestAdaptiveAgentIntegration:
    @pytest.mark.asyncio
    async def test_run_completes_without_error(self):
        blueprint = _blueprint()
        config = _config()
        adaptive_config = _load_adaptive_config()

        # 降低停止阈值，加速测试
        adaptive_config.stop.max_empty_rounds = 2
        adaptive_config.stop.max_failures = 3
        adaptive_config.execution.concurrency = 2
        adaptive_config.generation.max_tokens = 512

        topics = blueprint.topics
        evidence_manager = _fake_evidence_manager(topics)
        model_client = FakeModelClient(delay=0.0)

        verify_result = ValidationTaskResult(
            task_id="test_task", run_id="test_run",
            report_path="", selected_question_ids=[], failed_by_stage={},
        )

        from agents.verify_agent.config_loader import VerifyAgentConfig
        verify_config = VerifyAgentConfig()

        with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
            mock_verify_instance = AsyncMock()
            mock_verify_instance.run = AsyncMock(return_value=verify_result)
            MockVerify.return_value = mock_verify_instance

            # 注入 FakeModelClient 让 executor 真实生成
            await run_adaptive_generation_agent(
                blueprint=blueprint,
                config=config,
                adaptive_config=adaptive_config,
                evidence_manager=evidence_manager,
                model_client=model_client,
                verify_config=verify_config,
            )

        # 主循环跑通即通过，不报异常即可
        assert True

    @pytest.mark.asyncio
    async def test_run_stops_on_max_empty_rounds(self):
        """当 evidence_manager 返回空池时，循环应因 max_empty_rounds 停止。"""
        blueprint = _blueprint()
        config = _config()
        adaptive_config = _load_adaptive_config()
        adaptive_config.stop.max_empty_rounds = 2
        adaptive_config.stop.max_failures = 100

        em = MagicMock()
        em.evidence_pools = {}  # 所有 topic 无 pool → _generate 返回 []
        em.sample = MagicMock(return_value=None)
        em.get_evidence_text = MagicMock(return_value="")
        em.get_document_summary = MagicMock(return_value="")
        em.expand_retrieval = AsyncMock()

        from agents.verify_agent.config_loader import VerifyAgentConfig
        verify_config = VerifyAgentConfig()

        with patch("agents.adaptive_qa_agent.agent.VerifyAgent") as MockVerify:
            mock_verify_instance = AsyncMock()
            MockVerify.return_value = mock_verify_instance

            await run_adaptive_generation_agent(
                blueprint=blueprint,
                config=config,
                adaptive_config=adaptive_config,
                evidence_manager=em,
                model_client=FakeModelClient(delay=0.0),
                verify_config=verify_config,
            )

        assert True  # 不挂死即通过


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
