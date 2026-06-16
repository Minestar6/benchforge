"""model_eval_agent 单元测试。"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.model_eval_agent import agent as agent_module
from benchforge.agents.model_eval_agent.aggregator import (
    build_model_by_dimension,
    build_model_overall_report,
)
from benchforge.agents.model_eval_agent.auto_metric_executor import run_automatic_metrics
from benchforge.agents.model_eval_agent.dataset_metric_executor import run_dataset_metrics
from benchforge.agents.model_eval_agent.llm_judge_executor import _judge_single, run_llm_judge
from benchforge.agents.model_eval_agent.model_runner import _build_prompt
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
from benchforge.schemas import SharedState, AgentStatus
from benchforge.utils.shared_state import save_shared_state
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


def test_write_questions_jsonl_overwrites_existing_file(tmp_path):
    path = tmp_path / "evaluation_questions.jsonl"
    path.write_text('{"question_id":"old"}\n', encoding="utf-8")

    agent_module._write_jsonl_overwrite(
        path,
        [{"question_id": "q1"}, {"question_id": "q2"}],
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ['{"question_id": "q1"}', '{"question_id": "q2"}']


def test_load_eval_questions_uses_selected_status_not_final_weight(tmp_path):
    validated = tmp_path / "validated_questions.jsonl"
    validated.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "q_selected_low_weight",
                        "final_status": "selected",
                        "final_weight": 0.01,
                        "candidate": {
                            "question_id": "q_selected_low_weight",
                            "question_mode": "qa",
                            "question": "Selected question",
                            "answer": "answer",
                            "topic": "AI",
                        },
                    }
                ),
                json.dumps(
                    {
                        "question_id": "q_reserve_high_weight",
                        "final_status": "reserve",
                        "final_weight": 0.99,
                        "candidate": {
                            "question_id": "q_reserve_high_weight",
                            "question_mode": "qa",
                            "question": "Reserve question",
                            "answer": "answer",
                            "topic": "AI",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    questions = agent_module._load_eval_questions([str(validated)])

    assert [question["question_id"] for question in questions] == ["q_selected_low_weight"]


def test_resolve_input_paths_does_not_fallback_to_quality_weight_artifact(tmp_path):
    run_dir = tmp_path / "runs" / "task_x" / "run_y"
    run_dir.mkdir(parents=True)
    state = SharedState(
        task_id="task_x",
        run_id="run_y",
        blueprint={"topics": [], "modes": {}},
        agent_status={"verification": AgentStatus.COMPLETED},
        artifacts={"accepted_questions_quality": "legacy/accepted_questions_quality.jsonl"},
    )

    resolved = agent_module._resolve_input_paths_from_shared_state(state, run_dir)

    assert resolved == []


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


def test_run_automatic_metrics_supports_semantic_accuracy(tmp_path):
    questions = [
        {"question_id": "q1", "question_mode": "qa", "answer": "A1"},
        {"question_id": "q2", "question_mode": "qa", "answer": "A2"},
    ]
    model_responses = [
        {"question_id": "q1", "question_mode": "qa", "model_name": "model_a", "prediction": "A1"},
        {"question_id": "q2", "question_mode": "qa", "model_name": "model_a", "prediction": "A2"},
    ]
    metric_plans = {
        "qa": QuestionModeMetricPlan(
            automatic_metrics=[AutoMetricSpec(name="semantic_accuracy")]
        )
    }

    import benchforge.agents.model_eval_agent.metrics_registry as metrics_registry

    original_metric = metrics_registry.AUTO_METRIC_REGISTRY["semantic_accuracy"]

    class StubSemanticAccuracy:
        def compute(self, question, prediction, context=None):
            return 1.0 if question["question_id"] == "q1" else 0.0

    metrics_registry.AUTO_METRIC_REGISTRY["semantic_accuracy"] = StubSemanticAccuracy()
    try:
        results = run_automatic_metrics(questions, model_responses, metric_plans, tmp_path)
    finally:
        metrics_registry.AUTO_METRIC_REGISTRY["semantic_accuracy"] = original_metric

    row = results[0]
    assert row["metric_name"] == "semantic_accuracy"
    assert row["models"]["model_a"]["scores"] == [1.0, 0.0]
    assert row["models"]["model_a"]["pass_rate"] is None


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
        tracer=None,
        semaphore=asyncio.Semaphore(1),
    )

    assert trace["llm_call_id"] == "llm_test_call"


def test_build_prompt_multiple_choice_does_not_double_label_choices():
    prompt = _build_prompt(
        {
            "question": "Which option is correct?",
            "question_mode": "multiple_choice",
            "choices": [
                "(A) Wrong",
                "(B) Right",
                "(C) Also wrong",
                "(D) Nope",
            ],
        }
    )
    assert "A. (A) Wrong" not in prompt
    assert "(B) Right" in prompt


@pytest.mark.asyncio
async def test_judge_single_for_multiple_choice_includes_choices():
    client = RecordingJudgeClient()
    trace = await _judge_single(
        client=client,
        judge_model_name="judge-model",
        question={
            "question_id": "q1",
            "question_mode": "multiple_choice",
            "question": "Which option is correct?",
            "answer": "B",
            "choices": ["(A) Wrong", "(B) Right", "(C) Also wrong", "(D) Nope"],
            "citations": [],
        },
        model_name="candidate-model",
        prediction="B",
        metrics=[JudgeMetricSpec(name="correctness", description="is it correct")],
        system_prompt="system",
        user_template="",
        judge_defaults={"temperature": 0.0},
        llm_trace_path="",
        tracer=None,
        semaphore=asyncio.Semaphore(1),
    )
    assert "(B) Right" in trace["prompt"]["user"]


@pytest.mark.asyncio
async def test_judge_single_renders_new_prompt_placeholders():
    client = RecordingJudgeClient(response_text='{"scores":{"correctness":5},"reason":"ok"}')
    trace = await _judge_single(
        client=client,
        judge_model_name="judge-model",
        question={
            "question_id": "q1",
            "question_mode": "qa",
            "question": "What is AI?",
            "answer": "Artificial intelligence",
            "citations": ["AI is the simulation of human intelligence."],
        },
        model_name="candidate-model",
        prediction="Artificial intelligence",
        metrics=[JudgeMetricSpec(name="correctness", description="Whether the answer is correct.")],
        system_prompt="Criteria:\n{metrics}\nFormat:\n{output_format}",
        user_template="<Evidence>\n{citations}\n</Evidence>\n<Question>\n{question}\n</Question>\n<Reference Answer>\n{reference_answer}\n</Reference Answer>\n<Model Answer>\n{model_answer}\n</Model Answer>",
        judge_defaults={"temperature": 0.0},
        llm_trace_path="",
        tracer=None,
        semaphore=asyncio.Semaphore(1),
    )
    assert "correctness" in trace["prompt"]["system"]
    assert '"correctness": <score 1-5>' in trace["prompt"]["system"]
    assert '"scores"' not in trace["prompt"]["system"]
    assert '"reason"' not in trace["prompt"]["system"]
    assert "AI is the simulation of human intelligence." in trace["prompt"]["user"]
    assert "<Reference Answer>" in trace["prompt"]["user"]


def test_parse_judge_output_supports_flat_metric_object():
    from benchforge.agents.model_eval_agent.llm_judge_executor import _parse_judge_output

    parsed, ok, error = _parse_judge_output(
        '{"correctness": 5, "faithfulness": 4}',
        [
            JudgeMetricSpec(name="correctness", description=""),
            JudgeMetricSpec(name="faithfulness", description=""),
        ],
    )

    assert ok is True
    assert error is None
    assert parsed["scores"] == {"correctness": 5.0, "faithfulness": 4.0}


@pytest.mark.asyncio
async def test_run_llm_judge_records_rendered_prompts_file(tmp_path):
    client = RecordingJudgeClient(response_text='{"scores":{"correctness":5},"reason":"ok"}')
    questions = [
        {
            "question_id": "q1",
            "question_mode": "qa",
            "question": "What is AI?",
            "answer": "Artificial intelligence",
            "citations": ["AI is the simulation of human intelligence."],
        }
    ]
    model_responses = [
        {
            "question_id": "q1",
            "question_mode": "qa",
            "model_name": "candidate-model",
            "prediction": "Artificial intelligence",
        }
    ]
    metric_plans = {
        "qa": QuestionModeMetricPlan(
            llm_judge_metrics=[JudgeMetricSpec(name="correctness", description="Whether the answer is correct.")]
        )
    }

    await run_llm_judge(
        questions=questions,
        model_responses=model_responses,
        metric_plans=metric_plans,
        judge_client=client,
        judge_model_name="judge-model",
        judge_config=JudgeConfig(enabled=True, prompt_system="", prompt_user=""),
        judge_defaults={"temperature": 0.0},
        output_dir=tmp_path,
        llm_trace_path="",
        tracer=None,
        max_concurrency=1,
    )

    prompts_file = tmp_path / "traces" / "llm_judge_prompts.jsonl"
    assert prompts_file.exists()
    lines = prompts_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


@pytest.mark.asyncio
async def test_run_llm_judge_skips_multiple_choice_even_if_metrics_configured(tmp_path):
    client = RecordingJudgeClient(response_text='{"scores":{"correctness":5},"reason":"ok"}')
    questions = [
        {
            "question_id": "q1",
            "question_mode": "multiple_choice",
            "question": "Which option is correct?",
            "answer": "B",
            "choices": ["(A) Wrong", "(B) Right"],
            "citations": [],
        }
    ]
    model_responses = [
        {
            "question_id": "q1",
            "question_mode": "multiple_choice",
            "model_name": "candidate-model",
            "prediction": "B",
        }
    ]
    metric_plans = {
        "multiple_choice": QuestionModeMetricPlan(
            llm_judge_metrics=[JudgeMetricSpec(name="correctness", description="Whether the answer is correct.")]
        )
    }

    judge_scores = await run_llm_judge(
        questions=questions,
        model_responses=model_responses,
        metric_plans=metric_plans,
        judge_client=client,
        judge_model_name="judge-model",
        judge_config=JudgeConfig(enabled=True, prompt_system="", prompt_user=""),
        judge_defaults={"temperature": 0.0},
        output_dir=tmp_path,
        llm_trace_path="",
        tracer=None,
        max_concurrency=1,
    )

    assert judge_scores == []
    assert client.calls == []


def test_build_model_reports_follow_available_metrics():
    questions = [
        {"question_id": "q1", "question_mode": "qa", "topic": "AI", "estimated_difficulty": "easy", "answer": "A1"},
    ]
    automatic_scores = [
        {
            "metric_name": "semantic_accuracy",
            "question_mode": "qa",
            "question_ids": ["q1"],
            "models": {"model_a": {"scores": [1.0], "passed": [True], "mean": 1.0, "pass_rate": 1.0}},
        }
    ]
    judge_scores = [
        {
            "metric_name": "correctness",
            "question_mode": "qa",
            "question_ids": ["q1"],
            "models": {"model_a": {"scores": [0.8], "mean": 0.8}},
        },
        {
            "metric_name": "completeness",
            "question_mode": "qa",
            "question_ids": ["q1"],
            "models": {"model_a": {"scores": [0.7], "mean": 0.7}},
        },
    ]

    overall_rows = build_model_overall_report(questions, automatic_scores, judge_scores)
    assert overall_rows[0]["semantic_accuracy"] == 1.0
    assert overall_rows[0]["judge_completeness"] == 0.7

    by_topic = build_model_by_dimension(questions, automatic_scores, judge_scores, "topic")
    assert by_topic[0]["semantic_accuracy"] == 1.0
    assert by_topic[0]["judge_completeness"] == 0.7


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
        lambda *args, **kwargs: SimpleNamespace(model_name=args[0]),
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


@pytest.mark.asyncio
async def test__run_uses_run_root_llm_trace_path(tmp_path, monkeypatch):
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

    recorded = {"candidate_trace": None, "judge_trace": None}

    monkeypatch.setattr(agent_module, "load_model_registry", lambda _: {"candidate": object(), "judge": object()})
    monkeypatch.setattr(
        agent_module,
        "resolve_model_config",
        lambda *args, **kwargs: SimpleNamespace(model_name=args[0]),
    )
    monkeypatch.setattr(
        agent_module.ModelLoader,
        "load_model",
        staticmethod(lambda cfg: RecordingJudgeClient(response_text="candidate answer")),
    )
    monkeypatch.setattr(agent_module, "run_dataset_metrics", lambda *args, **kwargs: [])

    async def fake_run_candidate_models(*args, **kwargs):
        recorded["candidate_trace"] = kwargs.get("llm_trace_path")
        return [
            {
                "question_id": "q1",
                "question_mode": "qa",
                "model_name": "candidate",
                "prediction": "Artificial intelligence",
                "error": None,
            }
        ]

    async def fake_run_llm_judge(*args, **kwargs):
        recorded["judge_trace"] = kwargs.get("llm_trace_path")
        return []

    monkeypatch.setattr(agent_module, "run_candidate_models", fake_run_candidate_models)
    monkeypatch.setattr(agent_module, "run_automatic_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent_module, "run_llm_judge", fake_run_llm_judge)
    monkeypatch.setattr(agent_module, "build_dataset_quality_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(agent_module, "build_dataset_quality_by_group", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent_module, "build_model_overall_report", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent_module, "build_model_by_dimension", lambda *args, **kwargs: [])
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
            judge_defaults={},
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

    expected = str(Path("runs") / "task_x" / "run_y" / "llm_calls.jsonl")
    assert recorded["candidate_trace"] == expected
    assert recorded["judge_trace"] == expected


@pytest.mark.asyncio
async def test__run_raises_when_no_candidate_models_resolve(tmp_path, monkeypatch):
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

    monkeypatch.setattr(agent_module, "load_model_registry", lambda _: {})
    monkeypatch.setattr(agent_module, "run_dataset_metrics", lambda *args, **kwargs: [])

    config = ModelEvalAgentConfig(
        run=RunConfig(
            task_id="task_x",
            run_id="run_y",
            input_paths=[str(input_path)],
        ),
        dataset_evaluation=DatasetEvaluationConfig(enabled=False, metrics=[]),
        models=ModelsConfig(
            candidate_model_names=["missing-model"],
            judge_model_name=None,
            generation_defaults={},
            judge_defaults={},
        ),
        metrics={"qa": QuestionModeMetricPlan(automatic_metrics=[], llm_judge_metrics=[])},
        judge=JudgeConfig(enabled=False),
    )

    with pytest.raises(ValueError, match="No candidate models could be loaded"):
        await agent_module._run(config, registry_path="dummy_registry.yaml")


@pytest.mark.asyncio
async def test_run_model_eval_agent_from_shared_state_updates_llm_calls_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_dir = tmp_path / "runs" / "task_x" / "run_y"
    run_dir.mkdir(parents=True)
    validated_path = run_dir / "validation" / "validated_questions.jsonl"
    validated_path.parent.mkdir(parents=True)
    validated_path.write_text("", encoding="utf-8")

    state = SharedState(
        task_id="task_x",
        run_id="run_y",
        blueprint={},
        artifacts={"validated_questions": str(Path("runs") / "task_x" / "run_y" / "validation" / "validated_questions.jsonl")},
        agent_status={"generation": AgentStatus.COMPLETED, "verification": AgentStatus.COMPLETED},
    )
    shared_state_path = save_shared_state(state, run_dir)

    async def fake_run(config, registry_path=None):
        evaluation_dir = Path("runs") / "task_x" / "run_y" / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        (evaluation_dir / "evaluation_report.json").write_text("{}", encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(agent_module, "_run", fake_run)

    config = ModelEvalAgentConfig(
        run=RunConfig(shared_state_path=str(shared_state_path)),
        dataset_evaluation=DatasetEvaluationConfig(enabled=False, metrics=[]),
        models=ModelsConfig(candidate_model_names=[]),
        metrics={},
        judge=JudgeConfig(enabled=False),
    )

    await agent_module.run_model_eval_agent_from_shared_state(shared_state_path, config)

    updated = SharedState.model_validate_json(shared_state_path.read_text(encoding="utf-8"))
    assert updated.artifact("llm_calls") == str(Path("runs") / "task_x" / "run_y" / "llm_calls.jsonl")
    assert updated.artifact("evaluation_report") == str(Path("runs") / "task_x" / "run_y" / "evaluation" / "evaluation_report.json")
