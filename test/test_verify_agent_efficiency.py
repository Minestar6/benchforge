import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.verify_agent.agent import (
    _early_exact_dedup_candidates,
    _semantic_near_dedup_candidates,
)
from benchforge.agents.verify_agent.config_loader import CitationCfg, load_verify_agent_config
from benchforge.agents.verify_agent.citation_validator import validate_citation
from benchforge.agents.verify_agent.schema import CitationValidationResult
from benchforge.agents.verify_agent.schema import QuestionCandidate


def _candidate(
    question_id: str,
    question: str,
    citations: list[str] | None = None,
    chunks: list[str] | None = None,
    answer: str = "AI is artificial intelligence.",
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        task_id="task",
        run_id="run",
        topic="AI",
        question=question,
        answer=answer,
        question_mode="qa",
        citations=citations or ["AI is artificial intelligence."],
        chunks=chunks or ["AI is artificial intelligence, a field of computer science."],
        chunk_ids=["doc::chunk_0"],
        estimated_difficulty=5,
    )


def test_citation_validation_requires_both_chunk_and_answer_thresholds():
    cfg = CitationCfg(
        min_chunk_citation_score=0.85,
        min_answer_citation_score=0.75,
        min_citation_score=0.8,
        alpha=0.3,
        beta=0.7,
        citation_match_threshold=0.8,
    )
    candidate = _candidate(
        question_id="q1",
        question="What is AI?",
        citations=["AI is artificial intelligence."],
        chunks=["AI is artificial intelligence, a field of computer science."],
        answer="Machine learning",
    )

    result = validate_citation(candidate, cfg)

    assert result.chunk_citation_score >= cfg.min_chunk_citation_score
    assert result.answer_citation_score < cfg.min_answer_citation_score
    assert result.passed is False
    assert "answer_citation_score_too_low" in result.failed_reasons


def test_early_exact_dedup_candidates_keeps_highest_difficulty_question():
    candidates = [
        _candidate("q1", "What is AI?", answer="a1"),
        _candidate("q2", "What is AI", answer="a2"),
        _candidate("q3", "What is machine learning?"),
    ]
    candidates[0].estimated_difficulty = 3
    candidates[1].estimated_difficulty = 8

    kept, duplicate_of = _early_exact_dedup_candidates(candidates)

    kept_ids = {candidate.question_id for candidate in kept}
    assert kept_ids == {"q2", "q3"}
    assert duplicate_of == {"q1": "q2"}


def test_verify_agent_config_defaults_to_light_selection(tmp_path):
    cfg_file = tmp_path / "verify_agent.yaml"
    cfg_file.write_text(
        """
citation_validation: {}
llm_validation: {}
selection: {}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_verify_agent_config(cfg_file)

    assert cfg.selection.mode == "light"
    assert cfg.selection.semantic_similarity_threshold == 0.9


def test_verify_agent_config_loads_semantic_similarity_threshold(tmp_path):
    cfg_file = tmp_path / "verify_agent.yaml"
    cfg_file.write_text(
        """
citation_validation: {}
llm_validation: {}
selection:
  mode: "light"
  semantic_similarity_threshold: 0.93
""".strip(),
        encoding="utf-8",
    )

    cfg = load_verify_agent_config(cfg_file)

    assert cfg.selection.semantic_similarity_threshold == 0.93


def test_semantic_near_dedup_candidates_keeps_highest_citation_score(monkeypatch):
    candidates = [
        _candidate("q1", "What is artificial intelligence?"),
        _candidate("q2", "How would you define artificial intelligence?"),
        _candidate("q3", "What is machine learning?"),
    ]
    citation_results = {
        "q1": CitationValidationResult(
            question_id="q1",
            passed=True,
            answer_citation_score=0.7,
            chunk_citation_score=0.8,
            citation_score=0.75,
            citation_count=1,
            matched_citation_count=1,
            failed_reasons=[],
        ),
        "q2": CitationValidationResult(
            question_id="q2",
            passed=True,
            answer_citation_score=0.9,
            chunk_citation_score=0.95,
            citation_score=0.92,
            citation_count=1,
            matched_citation_count=1,
            failed_reasons=[],
        ),
        "q3": CitationValidationResult(
            question_id="q3",
            passed=True,
            answer_citation_score=0.85,
            chunk_citation_score=0.88,
            citation_score=0.86,
            citation_count=1,
            matched_citation_count=1,
            failed_reasons=[],
        ),
    }

    monkeypatch.setattr(
        "benchforge.agents.verify_agent.agent._compute_embeddings",
        lambda texts, model_name: __import__("numpy").array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ]),
    )

    kept, duplicate_of = _semantic_near_dedup_candidates(
        candidates,
        citation_results,
        embedding_model="unused-model",
        similarity_threshold=0.9,
    )

    kept_ids = {candidate.question_id for candidate in kept}
    assert kept_ids == {"q2", "q3"}
    assert duplicate_of == {"q1": "q2"}


def test_multiple_choice_dedup_uses_question_plus_choices():
    q1 = _candidate("q1", "Which option is correct?")
    q1.question_mode = "multiple_choice"
    q1.choices = ["A. cat", "B. dog"]
    q1.estimated_difficulty = 3

    q2 = _candidate("q2", "Which option is correct?")
    q2.question_mode = "multiple_choice"
    q2.choices = ["A. fish", "B. bird"]
    q2.estimated_difficulty = 8

    kept, duplicate_of = _early_exact_dedup_candidates([q1, q2])

    kept_ids = {candidate.question_id for candidate in kept}
    assert kept_ids == {"q1", "q2"}
    assert duplicate_of == {}
