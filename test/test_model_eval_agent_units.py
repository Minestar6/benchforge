"""model_eval_agent 单元测试。"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchforge.agents.model_eval_agent import agent as agent_module
from benchforge.agents.model_eval_agent.aggregator import build_model_by_dimension
from benchforge.agents.model_eval_agent.auto_metric_executor import run_automatic_metrics
from benchforge.agents.model_eval_agent.dataset_metric_executor import run_dataset_metrics
from benchforge.agents.model_eval_agent.llm_judge_executor import _judge_single
from benchforge.agents.model_eval_agent.schema import (
    AutoMetricSpec,
    DatasetMetricSpec,
    DatasetEvaluationConfig,
    JudgeConfig,
    JudgeMetricSpec,
    ModelEvalAgentConfig,
    ModelsConfig,
    QuestionModeMetricPlan,
    RunConfig,
)
from benchforge.models.base import BaseModelClient


class RecordingJudgeClient(BaseModelClient):
    def __init__(self, response_text='{"scores":{"correctness":0.8},"reason":"ok"}'):
        self.response_text = response_text
        self.calls = []

    async def complete(self, model: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return {
            "text": self.response_text,
            "input_tokens": 10,
            "output_tokens": 5,
            "latency": 0.01,
            "raw": {},
            "llm_call_id": "llm_test_call",
        }

    async def batch_complete(self, model: str, messages_list: list[list[dict[str, str]]], **kwargs):
        return [await self.complete(model, messages, **kwargs) for messages in messages_list]


def test_build_model_by_dimension_filters_scores_to_dimension_subset():
    questions = [
        {"question_id": "q1", "question_mode": "qa", "topic": "AI", "answer": "A1"},
        {"question_id": "q2", "question_mode": "qa", "topic": "ML", "answer": "A2"},
    ]
    automatic_scores = [
        {
            "metric_name": "exact_match",
            "question_mode": "qa",
            "question_ids": ["q1", "q2"],
            "models": {
                "model_a": {
                    "scores": [1.0, 0.0],
                    "passed": [True, False],
                    "mean": 0.5,
                    "pass_rate": 0.5,
                }
            },
        }
    ]
    judge_scores = [
        {
            "metric_name": "correctness",
            "question_mode": "qa",
            "question_ids": ["q1", "q2"],
            "models": {"model_a": {"scores": [0.2, 0.8], "mean": 0.5}},
        },
        {
            "metric_name": "faithfulness",
            "question_mode": "qa",
            "question_ids": ["q1", "q2"],
            "models": {"model_a": {"scores": [0.9, 0.1], "mean": 0.5}},
        },
    ]

    rows = build_model_by_dimension(questions, automatic_scores, judge_scores, "topic")
    row_by_topic = {row["topic"]: row for row in rows}

    assert row_by_topic["AI"]["exact_match"] == 1.0
    assert row_by_topic["AI"]["judge_correctness"] == 0.2
    assert row_by_topic["ML"]["exact_match"] == 0.0
    assert row_by_topic["ML"]["judge_faithfulness"] == 0.1


def test_run_automatic_metrics_aligns_missing_scores_to_shared_question_ids(tmp_path):
    questions = [
        {"question_id": "q1", "question_mode": "qa", "answer": "A1"},
        {"question_id": "q2", "question_mode": "qa", "answer": "A2"},
    ]
    model_responses = [
        {"question_id": "q1", "question_mode": "qa", "model_name": "model_a", "prediction": "A1"},
        {"question_id": "q2", "question_mode": "qa", "model_name": "model_a", "prediction": "wrong"},
        {"question_id": "q1", "question_mode": "qa", "model_name": "model_b", "prediction": "A1"},
        {"question_id": "q2", "question_mode": "qa", "model_name": "model_b", "prediction": "", "error": "timeout"},
    ]
    metric_plans = {
        "qa": QuestionModeMetricPlan(
            automatic_metrics=[AutoMetricSpec(name="exact_match", threshold=1.0)]
        )
    }

    results = run_automatic_metrics(questions, model_responses, metric_plans, tmp_path)
    row = results[0]

    assert row["question_ids"] == ["q1", "q2"]
    assert row["models"]["model_a"]["scores"] == [1.0, 0.0]
    assert row["models"]["model_b"]["scores"] == [1.0, None]


def test_run_dataset_metrics_includes_grouped_citation_rows(tmp_path):
    questions = [
        {
            "question_id": "q1",
            "question_mode": "qa",
            "topic": "AI",
            "estimated_difficulty": "easy",
            "citation_validation": {"citation_score": 0.9},
        },
        {
            "question_id": "q2",
            "question_mode": "qa",
            "topic": "ML",
            "estimated_difficulty": "hard",
            "citation_validation": {"citation_score": 0.7},
        },
    ]
    results = run_dataset_metrics(
        questions,
        [DatasetMetricSpec(name="citation_score", threshold=0.8)],
        tmp_path,
    )

    topic_rows = [
        row for row in results
        if row.get("metric_name") == "citation_score" and row.get("scope") == "topic"
    ]
    assert any(row.get("group_value") == "AI" for row in topic_rows)
    assert any(row.get("group_value") == "ML" for row in topic_rows)


@pytest.mark.asyncio
async def test_judge_single_trace_records_llm_call_id():
    client = RecordingJudgeClient()
    trace = await _judge_single(
        client=client,
        judge_model_name="judge-model",
        question={
            "question_id": "q1",
            "question_mode": "qa",
            "question": "What is AI?",
            "answer": "Artificial intelligence",
            "citations": [],
        },
        model_name="candidate-model",
        prediction="Artificial intelligence",
        metrics=[JudgeMetricSpec(name="correctness", description="is it correct")],
        system_prompt="system",
        user_template="",
        judge_defaults={"temperature": 0.0, "max_tokens": 123},
        llm_trace_path="",
        semaphore=asyncio.Semaphore(1),
    )

    assert trace["llm_call_id"] == "llm_test_call"


@pytest.mark.asyncio
async def test__run_passes_judge_defaults_and_composite_dimension_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    input_path = tmp_path / "accepted_questions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_mode": "qa",
                "question": "What is AI?",
                "answer": "Artificial intelligence",
                "topic": "AI",
                "estimated_difficulty": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recorded_dimension_keys = []
    recorded_judge_defaults = {}

    monkeypatch.setattr(agent_module, "load_model_registry", lambda _: {"candidate": object(), "judge": object()})
    monkeypatch.setattr(
        agent_module,
        "resolve_model_config",
        lambda name, registry: SimpleNamespace(model_name=name),
    )

    judge_client = RecordingJudgeClient()

    def fake_load_model(cfg):
        if cfg.model_name == "judge":
            return judge_client
        return RecordingJudgeClient(response_text="candidate answer")

    monkeypatch.setattr(agent_module.ModelLoader, "load_model", staticmethod(fake_load_model))
    monkeypatch.setattr(agent_module, "run_dataset_metrics", lambda *args, **kwargs: [])

    async def fake_run_candidate_models(*args, **kwargs):
        return [
            {
                "question_id": "q1",
                "question_mode": "qa",
                "model_name": "candidate",
                "prediction": "Artificial intelligence",
                "error": None,
            }
        ]

    monkeypatch.setattr(agent_module, "run_candidate_models", fake_run_candidate_models)
    monkeypatch.setattr(agent_module, "run_automatic_metrics", lambda *args, **kwargs: [])

    async def fake_run_llm_judge(*args, **kwargs):
        recorded_judge_defaults["value"] = kwargs.get("judge_defaults")
        return []

    monkeypatch.setattr(agent_module, "run_llm_judge", fake_run_llm_judge)
    monkeypatch.setattr(agent_module, "build_dataset_quality_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(agent_module, "build_dataset_quality_by_group", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent_module, "build_model_overall_report", lambda *args, **kwargs: [])

    def fake_build_model_by_dimension(questions, automatic_scores, judge_scores, dimension_key):
        recorded_dimension_keys.append(dimension_key)
        return []

    monkeypatch.setattr(agent_module, "build_model_by_dimension", fake_build_model_by_dimension)
    monkeypatch.setattr(agent_module, "write_reports", lambda *args, **kwargs: None)

    config = ModelEvalAgentConfig(
        run=RunConfig(
            task_id="task_x",
            run_id="run_y",
            input_paths=[str(input_path)],
        ),
        dataset_evaluation=DatasetEvaluationConfig(enabled=False, metrics=[]),
        models=ModelsConfig(
            candidate_model_names=["candidate"],
            judge_model_name="judge",
            generation_defaults={},
            judge_defaults={"temperature": 0.0, "max_tokens": 321},
        ),
        metrics={
            "qa": QuestionModeMetricPlan(
                automatic_metrics=[],
                llm_judge_metrics=[JudgeMetricSpec(name="correctness", description="desc")],
            )
        },
        judge=JudgeConfig(enabled=True),
    )

    await agent_module._run(config, registry_path="dummy_registry.yaml")

    assert recorded_judge_defaults["value"] == {"temperature": 0.0, "max_tokens": 321}
    assert recorded_dimension_keys == [
        "question_mode",
        "topic",
        "estimated_difficulty",
        "question_mode_and_difficulty",
        "topic_and_question_mode",
    ]
