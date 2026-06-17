"""BenchForge top-level entrypoint tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.planner_agent.schema import (
    EvaluationRequirements,
    EvaluatorDefaults,
    GeneratorFeedback,
    GeneratorFeedbackArtifacts,
    GeneratorFeedbackSummary,
    ModeGeneratorFeedback,
    EvaluatorFeedback,
    EvaluatorFeedbackSummary,
    FinalTargets,
    GlobalBlueprint,
    QuestionModeDefaults,
    StopConditions,
    UserIntent,
    ValidatorFeedback,
    ValidatorFeedbackSummary,
    ValidatorQualitySignals,
)
from benchforge.agents.planner_agent.blueprint_synthesizer import synthesize_global_blueprint
from benchforge.app import run_benchforge
from benchforge.cli import main, build_parser, build_user_intent
from benchforge.config.config import load_prompt
import run_minimal_case


class _BlueprintClient:
    def __init__(self, response_text: str | list[str]):
        self.response_texts = response_text if isinstance(response_text, list) else [response_text]
        self.calls = []
        self.model_name = "planner-model"

    async def complete(self, model: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        index = min(len(self.calls) - 1, len(self.response_texts) - 1)
        return {
            "text": self.response_texts[index],
            "input_tokens": 123,
            "output_tokens": 45,
            "latency": 0.01,
            "raw": {},
        }


@pytest.mark.asyncio
async def test_synthesize_global_blueprint_uses_user_overrides_and_llm_topics():
    client = _BlueprintClient(
        """
        {
          "seed_topics": ["deceptive alignment", "evaluation harnesses"],
          "default_modes": {
            "qa": {"max_rounds": 4, "difficulty_distribution": {"easy": 2, "medium": 3, "hard": 5}},
            "multiple_choice": {"max_rounds": 2, "difficulty_distribution": {"easy": 0.2, "medium": 0.3, "hard": 0.5}}
          },
          "evaluation_requirements": {
            "automatic_metrics": {"qa": ["exact_match"], "multiple_choice": ["accuracy"]},
            "llm_judge_metrics": {
              "qa": [{"name": "correctness", "description": "Judge whether the answer is factually correct."}]
            }
          }
        }
        """
    )
    intent = UserIntent(
        user_goal="生成一套 AI safety 基准题",
        language="zh",
        qa_target=12,
        multiple_choice_target=4,
        candidate_model_names=["deepseek-v3", "glm-4.7"],
        judge_model_name="glm-4.7",
        max_rounds=6,
        min_selected_per_round=4,
    )

    blueprint = await synthesize_global_blueprint(
        intent=intent,
        registry_path=project_root / "benchforge/config/model_registry.yaml",
        model_client=client,
        planner_model_name="fake",
    )

    assert blueprint.user_goal == intent.user_goal
    assert blueprint.language == "zh"
    assert blueprint.final_targets.qa == 12
    assert blueprint.final_targets.multiple_choice == 4
    assert blueprint.seed_topics == ["deceptive alignment", "evaluation harnesses"]
    assert blueprint.evaluator_defaults.candidate_model_names == ["deepseek-v3", "glm-4.7"]
    assert blueprint.evaluator_defaults.judge_model_name == "glm-4.7"
    assert blueprint.stop_conditions.max_rounds == 6
    assert blueprint.stop_conditions.min_selected_per_round == 4
    assert abs(sum(blueprint.default_modes["qa"].difficulty_distribution.values()) - 1.0) < 1e-9
    assert blueprint.evaluation_requirements.llm_judge_metrics["qa"][0].name == "correctness"


@pytest.mark.asyncio
async def test_synthesize_global_blueprint_uses_goal_analyzer_payload():
    client = _BlueprintClient(
        """
        {
          "initial_topics": ["grounded factuality", "citation faithfulness"],
          "initial_generation_strategy": "balanced_exploration",
          "automatic_metrics_by_type": {
            "qa": ["exact_match", "f1"],
            "multiple_choice": ["accuracy"]
          },
          "llm_eval_enabled": true,
          "qa_llm_eval_metrics": [
            {"name": "faithfulness", "description": "Judge whether the answer is supported by the provided evidence."}
          ]
        }
        """
    )
    intent = UserIntent(
        user_goal="生成 grounding benchmark",
        candidate_model_names=["fake"],
        judge_model_name="fake",
    )

    blueprint = await synthesize_global_blueprint(
        intent=intent,
        registry_path=project_root / "benchforge/config/model_registry.yaml",
        model_client=client,
        planner_model_name="fake",
    )

    assert blueprint.seed_topics == ["grounded factuality", "citation faithfulness"]
    assert blueprint.initial_generation_strategy == "balanced_exploration"
    assert blueprint.evaluation_requirements.automatic_metrics["qa"] == ["exact_match", "f1"]
    assert blueprint.evaluation_requirements.llm_judge_metrics["qa"][0].name == "faithfulness"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_run_benchforge_saves_blueprint_and_invokes_planner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    blueprint = GlobalBlueprint(
        task_id="task_demo",
        blueprint_id="bp_demo",
        user_goal="demo",
        language="zh",
        seed_topics=["AI"],
        final_targets=FinalTargets(qa=6, multiple_choice=2),
        default_modes={
            "qa": QuestionModeDefaults(max_rounds=3, difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3}),
            "multiple_choice": QuestionModeDefaults(max_rounds=3, difficulty_distribution={"easy": 0.4, "medium": 0.4, "hard": 0.2}),
        },
        evaluator_defaults=EvaluatorDefaults(candidate_model_names=["fake"], judge_model_name="fake"),
        evaluation_requirements=EvaluationRequirements(),
        stop_conditions=StopConditions(max_rounds=2, min_selected_per_round=2),
    )

    final_state = SimpleNamespace(task_id="task_demo", current_round=2)
    called = {}

    async def fake_synthesize(**kwargs):
        called["intent"] = kwargs["intent"]
        return blueprint

    async def fake_run_planner(global_blueprint, base_config_dir, registry_path, state_dir, resume_state=None):
        called["planner"] = {
            "global_blueprint": global_blueprint,
            "base_config_dir": Path(base_config_dir),
            "registry_path": Path(registry_path),
            "state_dir": Path(state_dir),
            "resume_state": resume_state,
        }
        return final_state

    monkeypatch.setattr("benchforge.app.synthesize_global_blueprint", fake_synthesize)
    monkeypatch.setattr("benchforge.app.run_planner", fake_run_planner)

    result = await run_benchforge(
        intent=UserIntent(user_goal="demo goal"),
        base_config_dir=project_root / "benchforge/config",
        registry_path=project_root / "benchforge/config/model_registry.yaml",
    )

    assert result["blueprint"].task_id == "task_demo"
    assert result["planner_state"] is final_state
    assert called["planner"]["global_blueprint"] is blueprint
    assert called["planner"]["state_dir"] == Path("runs") / "task_demo" / "planner"
    assert (Path("runs") / "task_demo" / "planner" / "global_blueprint.json").exists()


def test_cli_main_builds_user_intent_and_invokes_run_benchforge(monkeypatch):
    called = {}

    async def fake_run_benchforge(**kwargs):
        called["kwargs"] = kwargs
        return {
            "global_blueprint_path": Path("runs/task_demo/planner/global_blueprint.json"),
            "state_dir": Path("runs/task_demo/planner"),
            "planner_state": SimpleNamespace(current_round=2),
        }

    monkeypatch.setattr("benchforge.cli.run_benchforge", fake_run_benchforge)

    rc = main(
        [
            "--user-goal",
            "build safety benchmark",
            "--seed-topic",
            "alignment",
            "--qa-target",
            "8",
            "--mc-target",
            "2",
            "--planner-model",
            "fake",
        ]
    )

    assert rc == 0
    assert called["kwargs"]["intent"].user_goal == "build safety benchmark"
    assert called["kwargs"]["intent"].seed_topics == ["alignment"]
    assert called["kwargs"]["intent"].qa_target == 8
    assert called["kwargs"]["intent"].multiple_choice_target == 2


def test_cli_main_accepts_positional_goal(monkeypatch):
    called = {}

    async def fake_run_benchforge(**kwargs):
        called["kwargs"] = kwargs
        return {
            "global_blueprint_path": Path("runs/task_demo/planner/global_blueprint.json"),
            "state_dir": Path("runs/task_demo/planner"),
            "planner_state": SimpleNamespace(current_round=1),
        }

    monkeypatch.setattr("benchforge.cli.run_benchforge", fake_run_benchforge)

    rc = main(["build a benchmark for grounded QA"])

    assert rc == 0
    assert called["kwargs"]["intent"].user_goal == "build a benchmark for grounded QA"


def test_build_user_intent_uses_deepseek_v4_defaults_for_judge_and_planner():
    parser = build_parser()
    args = parser.parse_args(["build benchmark"])

    intent = build_user_intent(args)

    assert intent.judge_model_name == "deepseek-v4"
    assert intent.planner_model_name == "deepseek-v4"


def test_run_minimal_case_uses_english_goal_and_runnable_defaults(monkeypatch):
    called = {}

    def fake_cli_main(argv=None):
        called["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(run_minimal_case, "cli_main", fake_cli_main)

    rc = run_minimal_case.main()

    assert rc == 0
    assert "--user-goal" in called["argv"]
    goal = called["argv"][called["argv"].index("--user-goal") + 1]
    assert goal == run_minimal_case.DEFAULT_USER_GOAL
    assert "--planner-model" in called["argv"]
    assert called["argv"][called["argv"].index("--planner-model") + 1] == "deepseek-v3.2"


def test_run_minimal_case_forwards_explicit_argv(monkeypatch):
    called = {}

    def fake_cli_main(argv=None):
        called["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(run_minimal_case, "cli_main", fake_cli_main)

    rc = run_minimal_case.main(["--help"])

    assert rc == 0
    assert called["argv"] == ["--help"]


def test_load_prompt_resolves_repo_relative_benchforge_prefix():
    content = load_prompt("benchforge/prompts/verify_agent/quality_system_prompt.md")
    assert content.strip()


@pytest.mark.asyncio
async def test_run_benchforge_from_natural_language_generates_topics_and_executes_pipeline(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    synthesis_client = _BlueprintClient(
        """
        {
          "seed_topics": ["robustness under distribution shift", "grounded generation", "evaluation harness design"],
          "default_modes": {
            "qa": {"max_rounds": 2, "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3}},
            "multiple_choice": {"max_rounds": 2, "difficulty_distribution": {"easy": 0.3, "medium": 0.4, "hard": 0.3}}
          },
          "evaluation_requirements": {
            "automatic_metrics": {"qa": ["exact_match"], "multiple_choice": ["accuracy"]},
            "llm_judge_metrics": {
              "qa": [{"name": "correctness", "description": "Judge whether the answer is factually correct."}]
            }
          }
        }
        """
    )

    class _PlannerLoopClient:
        model_name = "fake"

        async def complete(self, model: str, messages: list[dict[str, str]], **kwargs):
            return {
                "text": '{"topics": ["grounded generation", "evaluation harness design"]}',
                "input_tokens": 10,
                "output_tokens": 5,
                "latency": 0.01,
                "raw": {},
            }

    execute_round_calls = []

    async def fake_execute_round(round_spec, global_blueprint, base_config_dir, registry_path):
        execute_round_calls.append(
            {
                "round_spec": round_spec,
                "global_blueprint": global_blueprint,
                "base_config_dir": Path(base_config_dir),
                "registry_path": Path(registry_path),
            }
        )
        return (
            GeneratorFeedback(
                task_id=global_blueprint.task_id,
                run_id=round_spec.run_id,
                round_id=round_spec.round_id,
                status="success",
                artifacts=GeneratorFeedbackArtifacts(
                    shared_state_path="runs/demo/shared_state.json",
                    generation_report="runs/demo/generation_report.json",
                ),
                summary=GeneratorFeedbackSummary(
                    total_candidates=6,
                    global_used_chunk_combinations=3,
                    global_failures=0,
                    llm_input_tokens=100,
                    llm_output_tokens=50,
                ),
                by_mode={
                    "qa": ModeGeneratorFeedback(
                        candidate_count=4,
                        target_candidate_count=4,
                        fulfillment_rate=1.0,
                    ),
                    "multiple_choice": ModeGeneratorFeedback(
                        candidate_count=2,
                        target_candidate_count=2,
                        fulfillment_rate=1.0,
                    ),
                },
            ),
            ValidatorFeedback(
                task_id=global_blueprint.task_id,
                run_id=round_spec.run_id,
                round_id=round_spec.round_id,
                status="success",
                summary=ValidatorFeedbackSummary(
                    total_candidates=6,
                    citation_passed=6,
                    llm_passed=5,
                    final_selected=3,
                    citation_pass_rate=1.0,
                    llm_pass_rate_after_citation=5 / 6,
                    final_selection_rate=0.5,
                    llm_calls=5,
                    llm_input_tokens=80,
                    llm_output_tokens=20,
                ),
                quality_signals=ValidatorQualitySignals(),
                by_mode={"qa": {"selected": 2}, "multiple_choice": {"selected": 1}},
            ),
            EvaluatorFeedback(
                task_id=global_blueprint.task_id,
                run_id=round_spec.run_id,
                round_id=round_spec.round_id,
                status="success",
                summary=EvaluatorFeedbackSummary(
                    num_questions=3,
                    num_models=1,
                    llm_input_tokens=30,
                    llm_output_tokens=10,
                ),
            ),
        )

    monkeypatch.setattr(
        "benchforge.agents.planner_agent.planner.ModelLoader.load_model",
        lambda cfg: _PlannerLoopClient(),
    )
    monkeypatch.setattr(
        "benchforge.agents.planner_agent.planner.execute_round",
        fake_execute_round,
    )

    result = await run_benchforge(
        intent=UserIntent(
            user_goal="请生成一套面向 RAG 与 grounding 能力评估的中文 benchmark，题目要能区分模型的证据忠实性和鲁棒性。",
            qa_target=4,
            multiple_choice_target=2,
            candidate_model_names=["fake"],
            judge_model_name="fake",
            max_rounds=1,
            min_selected_per_round=2,
        ),
        base_config_dir=project_root / "benchforge/config",
        registry_path=project_root / "benchforge/config/model_registry.yaml",
        planner_model_client=synthesis_client,
        planner_model_name="fake",
    )

    assert result["blueprint"].seed_topics == [
        "robustness under distribution shift",
        "grounded generation",
        "evaluation harness design",
    ]
    assert execute_round_calls
    assert execute_round_calls[0]["round_spec"].blueprint["topics"] == [
        "grounded generation",
        "evaluation harness design",
    ]
    assert result["planner_state"].current_round == 1
