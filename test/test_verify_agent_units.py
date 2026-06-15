"""verify_agent 单元测试。"""

import asyncio
import json
import tempfile
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.verify_agent.config_loader import (
    CitationCfg,
    LLMValidationCfg,
    SelectionCfg,
    load_verify_agent_config,
)
from benchforge.agents.verify_agent.citation_validator import (
    run_citation_validation,
    validate_citation,
)
from benchforge.agents.verify_agent.normalizer import (
    _build_chunk_index,
    _normalize_citations,
    load_and_normalize,
)
from benchforge.agents.verify_agent.schema import (
    CitationValidationResult,
    LLMValidationResult,
    ModeCfg,
    QuestionCandidate,
    ValidationBlueprintView,
)
from benchforge.schemas import SharedState, AgentStatus
from benchforge.utils.shared_state import save_shared_state
from benchforge.agents.verify_agent.selector import (
    _compute_bucket_targets,
    _label_to_int_difficulty,
    _normalize_difficulty,
    run_weighted_selection,
)
from benchforge.agents.verify_agent.agent import VerifyAgent
from benchforge.models.fake import FakeModelClient


# ─── helpers ─────────────────────────────────────────────────────────────────

_SENTINEL = object()


def make_candidate(
    question_id="q1",
    question="What is AI?",
    answer="AI is artificial intelligence.",
    citations=_SENTINEL,
    chunks=_SENTINEL,
    chunk_ids=None,
    question_mode="qa",
    estimated_difficulty=5,
    choices=None,
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        task_id="task_test",
        run_id="run_test",
        topic="AI",
        question=question,
        answer=answer,
        question_mode=question_mode,
        citations=citations if citations is not _SENTINEL else ["AI is artificial intelligence."],
        chunks=chunks if chunks is not _SENTINEL else ["AI stands for artificial intelligence, a field of computer science."],
        chunk_ids=chunk_ids or ["doc_abc::chunk_0000"],
        estimated_difficulty=estimated_difficulty,
        choices=choices,
    )


def make_blueprint() -> ValidationBlueprintView:
    return ValidationBlueprintView(
        topics=["AI"],
        modes={
            "qa": ModeCfg(count=10, difficulty_distribution={"easy": 3, "medium": 4, "hard": 3}),
        },
    )


# ─── normalizer tests ─────────────────────────────────────────────────────────

class TestNormalizeCitations:
    def test_list_of_str(self):
        result = _normalize_citations(["citation 1", "citation 2"])
        assert result == ["citation 1", "citation 2"]

    def test_list_of_dict(self):
        result = _normalize_citations([{"text": "citation text"}])
        assert result == ["citation text"]

    def test_empty(self):
        assert _normalize_citations([]) == []

    def test_none(self):
        assert _normalize_citations(None) == []


class TestBuildChunkIndex:
    def test_builds_index_from_json(self, tmp_path):
        chunked = tmp_path / "chunked.json"
        # chunked.json 格式: {topic: [doc_records]}
        data = {
            "AI": [{
                "document_id": "doc_abc",
                "topic": "AI",
                "chunks": [
                    {"chunk_id": "doc_abc::chunk_0000", "chunk_text": "AI is great."},
                    {"chunk_id": "doc_abc::chunk_0001", "chunk_text": "ML is a subset."},
                ]
            }]
        }
        chunked.write_text(json.dumps(data), encoding="utf-8")
        index = _build_chunk_index(chunked)
        assert index["doc_abc::chunk_0000"] == "AI is great."
        assert index["doc_abc::chunk_0001"] == "ML is a subset."

    def test_builds_index_from_jsonl(self, tmp_path):
        """兼容旧 chunked.jsonl 格式。"""
        chunked = tmp_path / "chunked.jsonl"
        data = {
            "document_id": "doc_abc",
            "topic": "AI",
            "chunks": [
                {"chunk_id": "doc_abc::chunk_0000", "chunk_text": "AI is great."},
                {"chunk_id": "doc_abc::chunk_0001", "chunk_text": "ML is a subset."},
            ]
        }
        chunked.write_text(json.dumps(data) + "\n", encoding="utf-8")
        index = _build_chunk_index(chunked)
        assert index["doc_abc::chunk_0000"] == "AI is great."
        assert index["doc_abc::chunk_0001"] == "ML is a subset."

    def test_missing_file(self, tmp_path):
        index = _build_chunk_index(tmp_path / "nonexistent.json")
        assert index == {}


class TestLoadAndNormalize:
    def test_candidate_pool_json(self, tmp_path):
        pool = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence.",
                "question_mode": "qa",
                "citations": ["AI is artificial intelligence."],
                "chunks": ["Some chunk text."],
                "chunk_ids": ["doc_abc::chunk_0000"],
                "topic": "AI",
                "estimated_difficulty": 5,
            }
        ]
        p = tmp_path / "candidate_pool.json"
        p.write_text(json.dumps(pool), encoding="utf-8")

        candidates = load_and_normalize(
            input_paths=[str(p)],
            task_id="task_test",
            run_id="run_test",
        )
        assert len(candidates) == 1
        assert candidates[0].question == "What is AI?"
        assert candidates[0].task_id == "task_test"
        assert len(candidates[0].question_id) > 0

    def test_question_id_generated(self, tmp_path):
        pool = [{"question": "Q1", "answer": "A1", "question_mode": "qa", "topic": "t"}]
        p = tmp_path / "candidate_pool.json"
        p.write_text(json.dumps(pool), encoding="utf-8")
        candidates = load_and_normalize([str(p)], "t", "r")
        assert candidates[0].question_id  # 非空

    def test_chunk_hydrate_from_index(self, tmp_path):
        # chunks 为空 → 从 chunked.json 回查
        pool = [{
            "question": "What?", "answer": "It.", "question_mode": "qa", "topic": "t",
            "chunks": [], "chunk_ids": ["doc_abc::chunk_0000"],
        }]
        pool_file = tmp_path / "candidate_pool.json"
        pool_file.write_text(json.dumps(pool), encoding="utf-8")

        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        chunked = evidence_dir / "chunked.json"
        chunked.write_text(
            json.dumps({
                "t": [{
                    "document_id": "doc_abc",
                    "topic": "t",
                    "chunks": [{"chunk_id": "doc_abc::chunk_0000", "chunk_text": "Hydrated text."}]
                }]
            }),
            encoding="utf-8",
        )

        candidates = load_and_normalize([str(pool_file)], "t", "r")
        assert candidates[0].chunks == ["Hydrated text."]

    def test_accepted_questions_jsonl(self, tmp_path):
        records = [
            {"question": "Q1", "answer": "A1", "question_mode": "qa", "topic": "t"},
            {"question": "Q2", "answer": "A2", "question_mode": "qa", "topic": "t"},
        ]
        p = tmp_path / "accepted_questions.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        candidates = load_and_normalize([str(p)], "t", "r")
        assert len(candidates) == 2


class TestConfigLoader:
    def test_load_strategy_only_yaml(self, tmp_path):
        cfg_file = tmp_path / "verify_agent.yaml"
        cfg_file.write_text(
            """
citation_validation:
  enabled: true
  min_citation_score: 0.7

llm_validation:
  enabled: true
  model: "gpt-4o-mini"
  max_retries: 1

selection:
  enabled: true
  embedding_model: "all-MiniLM-L6-v2"
""".strip(),
            encoding="utf-8",
        )

        cfg = load_verify_agent_config(cfg_file)
        assert cfg.citation.min_citation_score == 0.7
        assert cfg.llm_validation.max_retries == 1
        assert cfg.selection.embedding_model == "all-MiniLM-L6-v2"


@pytest.mark.asyncio
async def test_run_verify_agent_from_shared_state_updates_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "runs" / "task_x" / "run_y"
    run_dir.mkdir(parents=True)
    candidate_path = run_dir / "qa" / "candidate_pool.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("[]", encoding="utf-8")
    chunked_path = run_dir / "evidence" / "chunked.json"
    chunked_path.parent.mkdir(parents=True)
    chunked_path.write_text("{}", encoding="utf-8")

    state = SharedState(
        task_id="task_x",
        run_id="run_y",
        blueprint={"topics": ["AI"], "modes": {"qa": {"count": 1, "difficulty_distribution": {"easy": 1.0}}}},
        artifacts={
            "qa_candidate_pool": str(Path("runs") / "task_x" / "run_y" / "qa" / "candidate_pool.json"),
            "chunked_evidence": str(Path("runs") / "task_x" / "run_y" / "evidence" / "chunked.json"),
        },
        agent_status={"generation": AgentStatus.COMPLETED},
    )
    shared_state_path = save_shared_state(state, run_dir)

    async def fake_run_verify_agent(**kwargs):
        validation_dir = Path("runs") / "task_x" / "run_y" / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "validation_report.json").write_text("{}", encoding="utf-8")
        (validation_dir / "validated_questions.jsonl").write_text("", encoding="utf-8")
        (validation_dir / "weighted_selection.json").write_text("{}", encoding="utf-8")

        class Result:
            pass

        result = Result()
        result.task_id = "task_x"
        result.run_id = "run_y"
        result.report_path = str(validation_dir / "validation_report.json")
        result.selected_question_ids = []
        result.failed_by_stage = {}
        return result

    monkeypatch.setattr(
        "benchforge.agents.verify_agent.agent.run_verify_agent",
        fake_run_verify_agent,
    )

    from benchforge.agents.verify_agent.config_loader import VerifyAgentConfig
    from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state

    await run_verify_agent_from_shared_state(shared_state_path, VerifyAgentConfig(), None)

    updated = SharedState.model_validate_json(shared_state_path.read_text(encoding="utf-8"))
    assert updated.artifact("validation_report") == str(Path("runs") / "task_x" / "run_y" / "validation" / "validation_report.json")
    assert updated.artifact("validated_questions") == str(Path("runs") / "task_x" / "run_y" / "validation" / "validated_questions.jsonl")
    assert updated.artifact("weighted_selection") == str(Path("runs") / "task_x" / "run_y" / "validation" / "weighted_selection.json")


@pytest.mark.asyncio
async def test_verify_agent_skips_llm_validation_when_model_client_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    agent = VerifyAgent(
        config=load_verify_agent_config(project_root / "benchforge/config/verify_agent.yaml"),
        model_client=None,
    )

    result = await agent.run(
        task_id="task_test",
        run_id="run_test",
        blueprint=make_blueprint(),
        candidates=[make_candidate()],
    )
    assert result.task_id == "task_test"


def test_run_weighted_selection_marks_overquota_in_validated_records(monkeypatch):
    blueprint = ValidationBlueprintView(
        topics=["AI"],
        modes={
            "qa": ModeCfg(count=1, difficulty_distribution={"easy": 1}),
        },
    )
    questions = [
        make_candidate(question_id="q1", question="Q1", estimated_difficulty="easy"),
        make_candidate(question_id="q2", question="Q2", estimated_difficulty="easy"),
    ]
    citation_results = {
        q.question_id: CitationValidationResult(
            question_id=q.question_id,
            passed=True,
            answer_citation_score=0.9 if q.question_id == "q1" else 0.8,
            chunk_citation_score=0.9 if q.question_id == "q1" else 0.8,
            citation_score=0.9 if q.question_id == "q1" else 0.8,
            citation_count=1,
            matched_citation_count=1,
            failed_reasons=[],
        )
        for q in questions
    }
    llm_results = {
        q.question_id: LLMValidationResult(
            question_id=q.question_id,
            passed=True,
            overall_score=0.9 if q.question_id == "q1" else 0.8,
            dimensions={},
            failed_reasons=[],
            judge_summary="ok",
        )
        for q in questions
    }

    cfg = SelectionCfg(embedding_model="unused-model")

    monkeypatch.setitem(
        run_weighted_selection.__globals__,
        "_compute_embeddings",
        lambda texts, model_name: __import__("numpy").array([[1.0], [0.0]]),
    )
    result, records = run_weighted_selection(
        questions=questions,
        blueprint=blueprint,
        citation_results=citation_results,
        llm_results=llm_results,
        cfg=cfg,
    )

    status_by_id = {record.question_id: record.final_status for record in records}
    assert len(result.selected_question_ids) == 1
    assert len(result.dropped_as_overquota) == 1
    assert status_by_id[result.dropped_as_overquota[0]] == "overquota"


# ─── citation_validator tests ─────────────────────────────────────────────────

class TestValidateCitation:
    def setup_method(self):
        self.cfg = CitationCfg(
            min_citation_score=0.5,
            alpha=0.7,
            beta=0.3,
            citation_match_threshold=0.6,
        )

    def test_pass_with_good_citations(self):
        c = make_candidate(
            citations=["AI is artificial intelligence."],
            chunks=["AI is artificial intelligence, a broad field of computer science."],
            answer="AI is artificial intelligence.",
        )
        result = validate_citation(c, self.cfg)
        assert result.passed is True
        assert result.citation_score > 0.5

    def test_fail_empty_citations(self):
        c = make_candidate(citations=[])
        result = validate_citation(c, self.cfg)
        assert result.passed is False
        assert "missing_citations" in result.failed_reasons

    def test_fail_low_score(self):
        c = make_candidate(
            citations=["completely unrelated text xyz abc def ghi"],
            chunks=["Python programming language."],
            answer="Python is great.",
        )
        result = validate_citation(c, self.cfg)
        # score 应该很低，fail
        assert result.citation_count == 1

    def test_empty_chunks_gives_zero_chunk_score(self):
        c = make_candidate(
            citations=["some text"],
            chunks=[],
            answer="some text",
        )
        result = validate_citation(c, self.cfg)
        assert result.chunk_citation_score == 0.0

    def test_multiple_choice_uses_correct_option_text_for_answer_score(self):
        c = make_candidate(
            question="Which option is correct?",
            answer="B",
            question_mode="multiple_choice",
            choices=[
                "(A) Wrong option",
                "(B) Correct option supported by evidence",
                "(C) Another wrong option",
                "(D) Final wrong option",
            ],
            citations=["Correct option supported by evidence"],
            chunks=["Correct option supported by evidence appears in the source chunk."],
        )
        result = validate_citation(c, self.cfg)
        assert result.answer_citation_score > 0.8

    def test_run_citation_validation_disabled(self):
        cfg = CitationCfg(enabled=False)
        candidates = [make_candidate("q1"), make_candidate("q2")]
        results = run_citation_validation(candidates, cfg)
        assert all(r.passed for r in results.values())


# ─── selector tests ───────────────────────────────────────────────────────────

class TestNormalizeDifficulty:
    def test_int_easy(self):
        assert _normalize_difficulty(1) == "easy"
        assert _normalize_difficulty(3) == "easy"

    def test_int_medium(self):
        assert _normalize_difficulty(4) == "medium"
        assert _normalize_difficulty(6) == "medium"

    def test_int_hard(self):
        assert _normalize_difficulty(7) == "hard"
        assert _normalize_difficulty(10) == "hard"

    def test_str_label(self):
        assert _normalize_difficulty("easy") == "easy"
        assert _normalize_difficulty("medium") == "medium"
        assert _normalize_difficulty("hard") == "hard"

    def test_str_int(self):
        assert _normalize_difficulty("5") == "medium"

    def test_none(self):
        assert _normalize_difficulty(None) == "unknown"


class TestLabelToIntDifficulty:
    def test_easy(self):
        assert _label_to_int_difficulty("easy") == 2

    def test_medium(self):
        assert _label_to_int_difficulty("medium") == 5

    def test_hard(self):
        assert _label_to_int_difficulty("hard") == 8

    def test_case_insensitive(self):
        assert _label_to_int_difficulty("EASY") == 2
        assert _label_to_int_difficulty("Medium") == 5

    def test_whitespace(self):
        assert _label_to_int_difficulty("  hard  ") == 8

    def test_unknown_fallback(self):
        assert _label_to_int_difficulty("expert") == 5
        assert _label_to_int_difficulty("") == 5


class TestRunWeightedSelection:
    def _make_results(self, candidates):
        from benchforge.agents.verify_agent.schema import CitationValidationResult, LLMValidationResult
        citation_results = {
            c.question_id: CitationValidationResult(
                question_id=c.question_id, passed=True,
                answer_citation_score=0.8, chunk_citation_score=0.9,
                citation_score=0.87, citation_count=1, matched_citation_count=1,
                failed_reasons=[],
            )
            for c in candidates
        }
        llm_results = {
            c.question_id: LLMValidationResult(
                question_id=c.question_id, passed=True,
                overall_score=0.85, dimensions={}, failed_reasons=[], judge_summary="ok",
            )
            for c in candidates
        }
        return citation_results, llm_results

    def test_select_within_quota(self):
        candidates = [make_candidate(f"q{i}", f"Question {i}?", estimated_difficulty=5) for i in range(5)]
        blueprint = ValidationBlueprintView(
            topics=["AI"],
            modes={
                "qa": ModeCfg(count=5, difficulty_distribution={"easy": 1, "medium": 3, "hard": 1}),
            },
        )
        citation_results, llm_results = self._make_results(candidates)

        cfg = SelectionCfg()
        result, records = run_weighted_selection(candidates, blueprint, citation_results, llm_results, cfg)
        assert len(result.selected_question_ids) == 3
        assert len(result.dropped_as_overquota) == 2

    def test_exact_duplicate_removal(self):
        # 两道完全相同的题
        candidates = [
            make_candidate("q1", "What is AI?", estimated_difficulty=5),
            make_candidate("q2", "What is AI?", estimated_difficulty=5),
        ]
        blueprint = make_blueprint()
        citation_results, llm_results = self._make_results(candidates)
        cfg = SelectionCfg()
        result, records = run_weighted_selection(candidates, blueprint, citation_results, llm_results, cfg)
        assert len(result.dropped_as_duplicate) == 1

    def test_reserve_when_no_quota(self):
        candidates = [make_candidate("q1", "Q1?", estimated_difficulty=1)]  # easy bucket
        blueprint = ValidationBlueprintView(
            topics=["AI"],
            modes={
                "qa": ModeCfg(count=3, difficulty_distribution={"easy": 0, "medium": 3, "hard": 0}),
            },
        )
        citation_results, llm_results = self._make_results(candidates)
        cfg = SelectionCfg()
        result, records = run_weighted_selection(candidates, blueprint, citation_results, llm_results, cfg)
        assert "q1" not in result.selected_question_ids

    def test_compute_bucket_targets_from_ratio_blueprint(self):
        blueprint = ValidationBlueprintView(
            topics=["AI"],
            modes={
                "qa": ModeCfg(count=10, difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3}),
            },
        )
        targets = _compute_bucket_targets(blueprint)
        assert targets == {"qa": {"easy": 3, "medium": 4, "hard": 3}}


# ─── llm_validator tests ──────────────────────────────────────────────────────

class TestLLMValidator:
    def _make_fake_client_with_response(self, response_json: dict) -> FakeModelClient:
        """返回一个 fake client，覆写 complete 以返回特定 JSON。"""
        import json as _json

        class _PatchedFakeClient(FakeModelClient):
            async def complete(self, model, messages, **kwargs):
                resp = await super().complete(model, messages, **kwargs)
                resp["text"] = _json.dumps(response_json)
                return resp

        return _PatchedFakeClient(delay=0.0)

    def test_parse_valid_json(self):
        from benchforge.utils.filter import parse_json_response
        data = {
            "passed": True, "overall_score": 0.85,
            "dimensions": {"clarity": 0.9, "answerability": 0.8,
                           "faithfulness": 0.85, "difficulty_alignment": 0.75, "mode_alignment": 0.9},
            "failed_reasons": [], "judge_summary": "Good."
        }
        result = parse_json_response(json.dumps(data))
        assert result["passed"] is True

    def test_parse_json_in_code_block(self):
        from benchforge.utils.filter import parse_json_response
        text = '```json\n{"passed": false, "overall_score": 0.5, "dimensions": {}, "failed_reasons": ["low"], "judge_summary": "bad"}\n```'
        result = parse_json_response(text)
        assert result["passed"] is False

    def test_parse_invalid_raises(self):
        from benchforge.utils.filter import parse_json_response
        with pytest.raises(ValueError):
            parse_json_response("not json at all!!!")

    def test_llm_validation_passed(self):
        good_response = {
            "correctness": 5,
            "answerability": 4,
            "clarity": 4,
            "difficulty_consistency": 4,
            "suggested_difficulty": "medium",
            "reason": "Good question.",
        }
        client = self._make_fake_client_with_response(good_response)
        candidates = [make_candidate()]
        blueprint = make_blueprint()
        cfg = LLMValidationCfg(
            enabled=True,
            min_overall_score=0.75,
            max_concurrency=1,
            max_retries=0,
            prompt_path="",
            prompt_user="",
        )

        results = asyncio.run(
            __import__(
                "benchforge.agents.verify_agent.llm_validator",
                fromlist=["run_llm_validation"]
            ).run_llm_validation(candidates, blueprint, cfg, client, "/tmp/llm_calls.jsonl")
        )
        assert results["q1"].passed is True
        assert results["q1"].overall_score == 0.8125
        assert results["q1"].dimensions == {
            "correctness": 1.0,
            "answerability": 0.75,
            "clarity": 0.75,
            "difficulty_consistency": 0.75,
        }
        assert results["q1"].judge_summary == "Good question."
        assert results["q1"].suggested_difficulty == "medium"

    def test_llm_validation_rejects_below_min_overall_score(self):
        response = {
            "correctness": 3,
            "answerability": 2,
            "clarity": 3,
            "difficulty_consistency": 2,
            "suggested_difficulty": "hard",
            "reason": "Poor quality question.",
        }
        client = self._make_fake_client_with_response(response)
        candidates = [make_candidate()]
        blueprint = make_blueprint()
        cfg = LLMValidationCfg(
            enabled=True,
            min_overall_score=0.75,
            max_concurrency=1,
            max_retries=0,
            prompt_path="",
            prompt_user="",
        )

        from benchforge.agents.verify_agent.llm_validator import run_llm_validation
        results = asyncio.run(
            run_llm_validation(candidates, blueprint, cfg, client, "/tmp/llm_calls.jsonl")
        )
        # raw: 3,2,3,2 → unit: 0.5, 0.25, 0.5, 0.25 → overall = 0.375 < 0.75
        assert results["q1"].passed is False
        assert "overall_score_too_low" in results["q1"].failed_reasons
        assert results["q1"].suggested_difficulty == "hard"

    def test_build_messages_for_multiple_choice_includes_choices_in_question_text(self):
        from benchforge.agents.verify_agent.llm_validator import _build_messages

        candidate = make_candidate(
            question="Which option best defines AI?",
            answer="B",
            question_mode="multiple_choice",
            choices=[
                "(A) A database system",
                "(B) A field focused on intelligent machines",
                "(C) A network cable standard",
                "(D) A graphics format",
            ],
        )
        blueprint = ValidationBlueprintView(
            topics=["AI"],
            modes={
                "multiple_choice": ModeCfg(
                    count=5,
                    difficulty_distribution={"easy": 2, "medium": 2, "hard": 1},
                ),
            },
        )
        messages = _build_messages(
            candidate,
            blueprint,
            "",
            "Question:\n{question}\n\nAnswer:\n{answer}\nDifficulty:\n{difficulty}",
        )

        assert len(messages) == 1
        content = messages[0]["content"]
        assert "Which option best defines AI?" in content
        assert "(B) A field focused on intelligent machines" in content
        assert "Difficulty:\n5" in content

    def test_llm_validation_error_marks_validator_error(self):
        class _ErrorClient(FakeModelClient):
            async def complete(self, model, messages, **kwargs):
                raise RuntimeError("API timeout")

        client = _ErrorClient(delay=0.0)
        candidates = [make_candidate()]
        blueprint = make_blueprint()
        cfg = LLMValidationCfg(enabled=True, max_concurrency=1, max_retries=0)

        from benchforge.agents.verify_agent.llm_validator import run_llm_validation
        results = asyncio.run(
            run_llm_validation(candidates, blueprint, cfg, client, "/tmp/llm_calls.jsonl")
        )
        assert results["q1"].passed is False
        assert "validator_error" in results["q1"].failed_reasons
