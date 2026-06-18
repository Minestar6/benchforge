"""Tests for supplemental effectiveness evaluation helpers."""

from __future__ import annotations

from pathlib import Path

from experiment.case.supplemental_eval import (
    build_combined_subset_report,
    make_eval_tag,
    merge_metric_rows,
    resolve_supplemental_models,
    summarize_llm_judge_rows,
)


def test_resolve_supplemental_models_filters_duplicates_and_baseline() -> None:
    resolved = resolve_supplemental_models(
        requested_models=["deepseek-v4", "glm-4.7", "glm-4.7", "kimi-k2", "fake"],
        registry_models={"deepseek-v4", "glm-4.7", "kimi-k2", "fake"},
        baseline_models=["deepseek-v4", "deepseek-v4-pro", "minimax"],
    )

    assert resolved == ["glm-4.7", "kimi-k2"]


def test_make_eval_tag_uses_models_or_explicit_tag() -> None:
    assert make_eval_tag(["glm-4.7", "kimi-k2"]) == "glm-4_7__kimi-k2"
    assert make_eval_tag(["glm-4.7"], "round 2 + more") == "round_2___more"


def test_merge_metric_rows_combines_model_entries() -> None:
    baseline = [
        {
            "type": "model_auto_metric",
            "question_mode": "qa",
            "metric_name": "semantic_accuracy",
            "threshold": None,
            "question_ids": ["q1", "q2"],
            "models": {"deepseek-v4": {"scores": [1.0, 0.0], "mean": 0.5}},
        }
    ]
    supplemental = [
        {
            "type": "model_auto_metric",
            "question_mode": "qa",
            "metric_name": "semantic_accuracy",
            "threshold": None,
            "question_ids": ["q1", "q2"],
            "models": {"glm-4.7": {"scores": [1.0, 1.0], "mean": 1.0}},
        }
    ]

    merged = merge_metric_rows(baseline, supplemental)

    assert len(merged) == 1
    assert sorted(merged[0]["models"].keys()) == ["deepseek-v4", "glm-4.7"]


def test_summarize_llm_judge_rows_flattens_model_metric_means() -> None:
    summary = summarize_llm_judge_rows(
        [
            {
                "question_mode": "qa",
                "metric_name": "answer_relevance",
                "models": {
                    "deepseek-v4": {"mean": 5.0},
                    "glm-4.7": {"mean": 4.0},
                },
            }
        ]
    )

    assert summary == [
        {
            "question_mode": "qa",
            "metric_name": "answer_relevance",
            "model_name": "deepseek-v4",
            "mean": 5.0,
        },
        {
            "question_mode": "qa",
            "metric_name": "answer_relevance",
            "model_name": "glm-4.7",
            "mean": 4.0,
        },
    ]


def test_build_combined_subset_report_merges_baseline_and_supplemental_models() -> None:
    questions = [
        {
            "question_id": "q1",
            "question_mode": "qa",
            "topic": "AI",
            "estimated_difficulty": "hard",
        },
        {
            "question_id": "q2",
            "question_mode": "multiple_choice",
            "topic": "QC",
            "estimated_difficulty": "medium",
        },
    ]
    dataset_scores = [
        {"metric_name": "citation_score", "score": 0.8, "question_id": "q1", "question_mode": "qa", "topic": "AI"},
        {"metric_name": "citation_score", "score": 0.7, "question_id": "q2", "question_mode": "multiple_choice", "topic": "QC"},
        {"metric_name": "diversity_score", "score": 0.9, "embedding_dispersion": 0.8, "cluster_entropy": 1.0},
    ]
    baseline_automatic = [
        {
            "type": "model_auto_metric",
            "question_mode": "qa",
            "metric_name": "semantic_accuracy",
            "threshold": None,
            "question_ids": ["q1"],
            "models": {
                "deepseek-v4": {"scores": [1.0], "mean": 1.0},
                "minimax": {"scores": [0.0], "mean": 0.0},
            },
        },
        {
            "type": "model_auto_metric",
            "question_mode": "multiple_choice",
            "metric_name": "exact_match",
            "threshold": None,
            "question_ids": ["q2"],
            "models": {
                "deepseek-v4": {"scores": [1.0], "mean": 1.0},
                "minimax": {"scores": [0.0], "mean": 0.0},
            },
        },
    ]
    supplemental_automatic = [
        {
            "type": "model_auto_metric",
            "question_mode": "qa",
            "metric_name": "semantic_accuracy",
            "threshold": None,
            "question_ids": ["q1"],
            "models": {"glm-4.7": {"scores": [0.5], "mean": 0.5}},
        },
        {
            "type": "model_auto_metric",
            "question_mode": "multiple_choice",
            "metric_name": "exact_match",
            "threshold": None,
            "question_ids": ["q2"],
            "models": {"glm-4.7": {"scores": [1.0], "mean": 1.0}},
        },
    ]
    baseline_judge = [
        {
            "type": "llm_judge_metric",
            "question_mode": "qa",
            "metric_name": "answer_relevance",
            "question_ids": ["q1"],
            "models": {"deepseek-v4": {"mean": 5.0}, "minimax": {"mean": 1.0}},
        }
    ]
    supplemental_judge = [
        {
            "type": "llm_judge_metric",
            "question_mode": "qa",
            "metric_name": "answer_relevance",
            "question_ids": ["q1"],
            "models": {"glm-4.7": {"mean": 4.0}},
        }
    ]

    report = build_combined_subset_report(
        subset_name="bfull_eval",
        questions=questions,
        dataset_scores=dataset_scores,
        baseline_automatic_rows=baseline_automatic,
        supplemental_automatic_rows=supplemental_automatic,
        baseline_judge_rows=baseline_judge,
        supplemental_judge_rows=supplemental_judge,
    )

    assert report["benchmark_metrics"]["num_models"] == 3
    assert report["benchmark_metrics"]["model_scores"]["deepseek-v4"] == 1.0
    assert report["benchmark_metrics"]["model_scores"]["glm-4.7"] == 0.75
    assert report["evaluation_result"]["qa"]["model"]["glm-4.7"] == 0.5
    assert report["evaluation_result"]["multiple_choice"]["model"]["glm-4.7"] == 1.0
    assert any(row["model_name"] == "glm-4.7" and row["mean"] == 4.0 for row in report["judge_summary"])
