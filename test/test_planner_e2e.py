"""PlannerAgent 端到端测试。"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.agents.planner_agent.schema import (
    GlobalBlueprint,
    FinalTargets,
    QuestionModeDefaults,
    EvaluatorDefaults,
    EvaluationRequirements,
    StopConditions,
    RunHistoryEntry,
    PlannerState,
    QuestionPlan,
    TopicBacklog,
    ModelPool,
)
from benchforge.agents.planner_agent.planner import run_planner
from benchforge.agents.planner_agent.config_loader import load_planner_state, save_planner_state


@pytest.fixture
def test_blueprint():
    """最小 GlobalBlueprint 用于测试。"""
    return GlobalBlueprint(
        task_id="test_task",
        blueprint_id="test_bp",
        user_goal="测试规划智能体",
        language="zh",
        seed_topics=["算法", "数据结构", "设计模式"],
        final_targets=FinalTargets(qa=10, multiple_choice=10),
        default_modes={
            "qa": QuestionModeDefaults(max_rounds=5, difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2}),
            "multiple_choice": QuestionModeDefaults(max_rounds=5, difficulty_distribution={"easy": 0.4, "medium": 0.4, "hard": 0.2}),
        },
        evaluator_defaults=EvaluatorDefaults(candidate_model_names=["deepseek-v3"]),
        evaluation_requirements=EvaluationRequirements(),
        stop_conditions=StopConditions(max_rounds=2, min_selected_per_round=3),
    )


@pytest.fixture
def mock_model_client():
    """Mock 模型客户端，返回合法 JSON 以走通 LLM evolver happy path。"""
    client = MagicMock()
    client.model_name = "test-model"
    client.complete = AsyncMock(return_value={
        "text": (
            '{"qa_count":5,"mc_count":3,'
            '"qa_difficulty_distribution":{"easy":0.2,"medium":0.5,"hard":0.3},'
            '"mc_difficulty_distribution":{"easy":0.3,"medium":0.5,"hard":0.2},'
            '"topic_budget":2,"min_candidate_multiplier":1.5,"max_candidate_multiplier":2.0}'
        )
    })
    return client


@pytest.mark.asyncio
async def test_run_planner_basic(test_blueprint, mock_model_client, tmp_path):
    """测试 run_planner 基础流程：2轮执行 + 状态保存。"""
    base_config_dir = Path("config")
    registry_path = Path("config/model_registry.yaml")
    state_dir = tmp_path / "planner_state"

    # Mock orchestrator.execute_round
    with patch("benchforge.agents.planner_agent.planner.execute_round") as mock_exec:
        # Mock feedback 返回
        from benchforge.agents.planner_agent.schema import (
            GeneratorFeedback,
            GeneratorFeedbackSummary,
            GeneratorFeedbackArtifacts,
            ValidatorFeedback,
            ValidatorFeedbackSummary,
            ValidatorQualitySignals,
            EvaluatorFeedback,
            EvaluatorFeedbackSummary,
        )

        gen_fb = GeneratorFeedback(
            task_id="test_task",
            run_id="run_001",
            round_id=1,
            status="success",
            artifacts=GeneratorFeedbackArtifacts(
                shared_state_path="mock_path",
                generation_report="mock_report",
            ),
            summary=GeneratorFeedbackSummary(
                total_candidates=20,
                global_used_chunk_combinations=10,
                global_failures=0,
                llm_input_tokens=1000,
                llm_output_tokens=500,
            ),
        )

        val_fb = ValidatorFeedback(
            task_id="test_task",
            run_id="run_001",
            round_id=1,
            status="success",
            summary=ValidatorFeedbackSummary(
                total_candidates=20,
                citation_passed=15,
                llm_passed=12,
                final_selected=8,
                citation_pass_rate=0.75,
                llm_pass_rate_after_citation=0.8,
                final_selection_rate=0.4,
                llm_calls=20,
                llm_input_tokens=2000,
                llm_output_tokens=1000,
            ),
            quality_signals=ValidatorQualitySignals(),
            by_mode={"qa": {"selected": 5, "reserve": 1, "rejected": 2}, "multiple_choice": {"selected": 3, "reserve": 0, "rejected": 1}},
        )

        eval_fb = EvaluatorFeedback(
            task_id="test_task",
            run_id="run_001",
            round_id=1,
            status="success",
            summary=EvaluatorFeedbackSummary(
                num_questions=8,
                num_models=1,
                llm_input_tokens=3000,
                llm_output_tokens=1500,
            ),
        )

        mock_exec.return_value = (gen_fb, val_fb, eval_fb)

        # Mock ModelLoader
        with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model") as mock_load:
            mock_load.return_value = mock_model_client

            # Mock load_model_registry
            with patch("benchforge.agents.planner_agent.planner.load_model_registry") as mock_reg:
                mock_reg.return_value = {"deepseek-v3": MagicMock()}

                # 执行 run_planner
                final_state = await run_planner(
                    test_blueprint,
                    base_config_dir,
                    registry_path,
                    state_dir,
                )

    # 验证结果
    assert final_state.current_round == 2  # max_rounds=2
    assert len(final_state.run_history) == 2
    assert final_state.completed_targets["qa"] == 10  # 5 + 5（两轮各5）
    assert final_state.completed_targets["multiple_choice"] == 6  # 3 + 3
    assert final_state.resource_usage.total_tokens > 0

    # 验证状态文件落盘
    assert (state_dir / "planner_state_init.json").exists()
    assert (state_dir / "planner_state_after_round_001.json").exists()
    assert (state_dir / "planner_state_after_round_002.json").exists()
    assert (state_dir / "round_001_spec.json").exists()
    assert (state_dir / "round_002_spec.json").exists()

    # 验证 state 内容
    state_after_r1 = load_planner_state(state_dir / "planner_state_after_round_001.json")
    assert state_after_r1.current_round == 1
    assert state_after_r1.run_history[0].selected_count == 8

    # 验证 round_spec 内容
    with open(state_dir / "round_001_spec.json", encoding="utf-8") as f:
        spec = json.load(f)
        assert spec["round_id"] == 1
        assert spec["task_id"] == "test_task"
        assert "topics" in spec["blueprint"]
        assert spec["planner_hints"]["eval_profile"] in ["light", "standard", "full"]


@pytest.mark.asyncio
async def test_stop_condition_targets_satisfied(test_blueprint, mock_model_client, tmp_path):
    """测试终止条件：targets_satisfied。"""
    # 修改 blueprint，降低 target
    test_blueprint.final_targets = FinalTargets(qa=5, multiple_choice=3)
    test_blueprint.stop_conditions.max_rounds = 5  # 提高 max_rounds，让 targets 优先触发

    state_dir = tmp_path / "planner_state"

    with patch("benchforge.agents.planner_agent.planner.execute_round") as mock_exec:
        from benchforge.agents.planner_agent.schema import (
            GeneratorFeedback,
            GeneratorFeedbackSummary,
            GeneratorFeedbackArtifacts,
            ValidatorFeedback,
            ValidatorFeedbackSummary,
            ValidatorQualitySignals,
        )

        # Mock 第一轮返回足够的 selected
        mock_exec.return_value = (
            GeneratorFeedback(
                task_id="test_task",
                run_id="run_001",
                round_id=1,
                status="success",
                artifacts=GeneratorFeedbackArtifacts(shared_state_path="", generation_report=""),
                summary=GeneratorFeedbackSummary(total_candidates=10, global_used_chunk_combinations=5, global_failures=0),
            ),
            ValidatorFeedback(
                task_id="test_task",
                run_id="run_001",
                round_id=1,
                status="success",
                summary=ValidatorFeedbackSummary(
                    total_candidates=10,
                    citation_passed=10,
                    llm_passed=10,
                    final_selected=8,
                    citation_pass_rate=1.0,
                    llm_pass_rate_after_citation=1.0,
                    final_selection_rate=0.8,
                    llm_calls=10,
                    llm_input_tokens=1000,
                    llm_output_tokens=500,
                ),
                quality_signals=ValidatorQualitySignals(),
                by_mode={"qa": {"selected": 5}, "multiple_choice": {"selected": 3}},
            ),
            None,
        )

        with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model", return_value=mock_model_client):
            with patch("benchforge.agents.planner_agent.planner.load_model_registry", return_value={"deepseek-v3": MagicMock()}):
                final_state = await run_planner(test_blueprint, Path("config"), Path("config/model_registry.yaml"), state_dir)

    # 验证：只运行了1轮就因 targets_satisfied 停止
    assert final_state.current_round == 1
    assert final_state.completed_targets["qa"] >= 5
    assert final_state.completed_targets["multiple_choice"] >= 3


@pytest.mark.asyncio
async def test_run_planner_uses_planner_config_model_not_judge_model(test_blueprint, tmp_path):
    state_dir = tmp_path / "planner_state"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "planner_agent.yaml").write_text(
        "model:\n"
        "  name: planner-model\n"
        "question_difficulty_evolver:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )

    loaded_models = []

    async def _fake_execute_round(*args, **kwargs):
        from benchforge.agents.planner_agent.schema import (
            GeneratorFeedback,
            GeneratorFeedbackSummary,
            GeneratorFeedbackArtifacts,
            ValidatorFeedback,
            ValidatorFeedbackSummary,
            ValidatorQualitySignals,
        )
        return (
            GeneratorFeedback(
                task_id="test_task",
                run_id="run_001",
                round_id=1,
                status="success",
                artifacts=GeneratorFeedbackArtifacts(shared_state_path="", generation_report=""),
                summary=GeneratorFeedbackSummary(total_candidates=10, global_used_chunk_combinations=5, global_failures=0),
            ),
            ValidatorFeedback(
                task_id="test_task",
                run_id="run_001",
                round_id=1,
                status="success",
                summary=ValidatorFeedbackSummary(
                    total_candidates=10,
                    citation_passed=10,
                    llm_passed=10,
                    final_selected=10,
                    citation_pass_rate=1.0,
                    llm_pass_rate_after_citation=1.0,
                    final_selection_rate=1.0,
                    llm_calls=0,
                    llm_input_tokens=0,
                    llm_output_tokens=0,
                ),
                quality_signals=ValidatorQualitySignals(),
                by_mode={"qa": {"selected": 10}, "multiple_choice": {"selected": 10}},
            ),
            None,
        )

    with patch("benchforge.agents.planner_agent.planner.execute_round", new=_fake_execute_round):
        with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model") as mock_load:
            mock_load.side_effect = lambda cfg: loaded_models.append(cfg.model_name) or mock_model_client
            with patch(
                "benchforge.agents.planner_agent.planner.load_model_registry",
                return_value={"planner-model": MagicMock(model_name="planner-model"), "judge-model": MagicMock(model_name="judge-model")},
            ):
                test_blueprint.evaluator_defaults.judge_model_name = "judge-model"
                await run_planner(test_blueprint, config_dir, Path("config/model_registry.yaml"), state_dir)

    assert loaded_models[0] == "planner-model"


@pytest.mark.asyncio
async def test_run_planner_stops_when_topics_cannot_be_proposed(test_blueprint, mock_model_client, tmp_path):
    state_dir = tmp_path / "planner_state"

    with patch("benchforge.agents.planner_agent.planner.execute_round") as mock_exec:
        with patch("benchforge.agents.planner_agent.planner.topic_adaptive_retriever", new=AsyncMock(return_value=[])):
            with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model", return_value=mock_model_client):
                with patch("benchforge.agents.planner_agent.planner.load_model_registry", return_value={"deepseek-v3": MagicMock(model_name="deepseek-v3"), "deepseek-v3.2": MagicMock(model_name="deepseek-v3.2")}):
                    final_state = await run_planner(
                        test_blueprint,
                        Path("config"),
                        Path("config/model_registry.yaml"),
                        state_dir,
                    )

    mock_exec.assert_not_called()
    assert final_state.current_round == 0
    assert final_state.run_history == []
    assert not (state_dir / "round_001_spec.json").exists()


@pytest.mark.asyncio
async def test_run_planner_resume_uses_previous_feedback_for_diagnosis(test_blueprint, mock_model_client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / "planner_state"
    state_dir.mkdir()

    resume_state = PlannerState(
        task_id="test_task",
        blueprint_id="test_bp",
        current_round=1,
        completed_targets={"qa": 4, "multiple_choice": 3},
        topic_backlog=TopicBacklog(active=["算法"], deferred=["数据结构"]),
        model_pool=ModelPool(active=["deepseek-v3"], removed=[]),
        run_history=[
            RunHistoryEntry(
                round_id=1,
                run_id="run_001",
                topics=["算法"],
                qa_target=5,
                multiple_choice_target=3,
                selected_count=8,
                evaluated=True,
                question_plan=QuestionPlan(
                    qa_count=5,
                    mc_count=3,
                    qa_difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
                    mc_difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
                ),
            )
        ],
    )
    save_planner_state(resume_state, state_dir / "planner_state_after_round_001.json")
    with open(state_dir / "round_001_spec.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": "test_task",
                "blueprint_id": "test_bp",
                "round_id": 1,
                "run_id": "run_001",
                "objective": "test",
                "blueprint": {
                    "task_id": "test_task",
                    "run_id": "run_001",
                    "language": "zh",
                    "topics": ["算法"],
                    "modes": {
                        "qa": {"count": 5, "max_rounds": 5, "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3}},
                        "multiple_choice": {"count": 3, "max_rounds": 5, "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}},
                    },
                },
                "qa_agent_patch": {},
                "verify_agent_patch": {},
                "model_eval_agent_patch": {},
                "planner_hints": {"eval_profile": "standard"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    run_dir = tmp_path / "runs" / "test_task" / "run_001"
    (run_dir / "validation").mkdir(parents=True)
    (run_dir / "evaluation" / "dataset_report").mkdir(parents=True)
    with open(run_dir / "shared_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": "test_task",
                "run_id": "run_001",
                "blueprint": {
                    "topics": ["算法"],
                    "modes": {
                        "qa": {"count": 5, "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3}},
                        "multiple_choice": {"count": 3, "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}},
                    },
                },
                "artifacts": {
                    "generation_report": "runs/test_task/run_001/generation_report.json",
                    "validation_report": "runs/test_task/run_001/validation/validation_report.json",
                    "validated_questions": "runs/test_task/run_001/validation/validated_questions.jsonl",
                    "weighted_selection": "runs/test_task/run_001/validation/weighted_selection.json",
                    "evaluation_report": "runs/test_task/run_001/evaluation/evaluation_report.json",
                    "dataset_quality_summary": "runs/test_task/run_001/evaluation/dataset_report/dataset_quality_summary.json",
                },
                "agent_status": {"generation": "completed", "verification": "completed", "evaluation": "completed"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(run_dir / "generation_report.json", "w", encoding="utf-8") as f:
        json.dump({"total_candidates": 10, "global_used_chunk_combinations": 5, "global_failures": 0, "modes": {}}, f)
    with open(run_dir / "validation" / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_candidates": 10,
                "citation_passed": 9,
                "llm_passed": 8,
                "final_selected": 8,
                "failed_by_stage": {},
                "llm_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0},
            },
            f,
        )
    with open(run_dir / "validation" / "validated_questions.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"final_status": "selected", "candidate": {"question_mode": "qa", "estimated_difficulty": "easy", "topic": "算法"}}) + "\n")
    with open(run_dir / "validation" / "weighted_selection.json", "w", encoding="utf-8") as f:
        json.dump({"dropped_as_duplicate": [], "dropped_as_overquota": []}, f)
    with open(run_dir / "evaluation" / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_questions": 8,
                "num_models": 1,
                "discriminative_signals": {
                    "overall_model_gap": 0.05,
                    "discriminative_question_ratio": 0.1,
                    "easy_questions_too_easy_ratio": 0.5,
                    "all_models_fail_ratio": 0.0,
                },
                "by_topic": {"算法": {"question_count": 8, "model_gap": 0.05, "too_easy_ratio": 0.5, "all_models_fail_ratio": 0.0}},
                "by_difficulty": {},
                "by_mode": {},
            },
            f,
        )
    with open(run_dir / "evaluation" / "dataset_report" / "dataset_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump({}, f)

    captured = {}

    def _capture_question_plan(state, blueprint, diagnosis, **kwargs):
        captured["diagnosis_label"] = diagnosis.label
        return QuestionPlan(
            qa_count=1,
            mc_count=0,
            qa_difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
            mc_difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
            topic_budget=1,
        )

    with patch("benchforge.agents.planner_agent.planner.question_difficulty_evolver", side_effect=_capture_question_plan):
        with patch("benchforge.agents.planner_agent.planner.topic_adaptive_retriever", new=AsyncMock(return_value=[])):
            with patch("benchforge.agents.planner_agent.planner.execute_round") as mock_exec:
                with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model", return_value=mock_model_client):
                    with patch("benchforge.agents.planner_agent.planner.load_model_registry", return_value={"deepseek-v3": MagicMock(), "deepseek-v3.2": MagicMock()}):
                        await run_planner(
                            test_blueprint,
                            Path("config"),
                            Path("config/model_registry.yaml"),
                            state_dir,
                            resume_state=resume_state,
                        )

    mock_exec.assert_not_called()
    assert captured["diagnosis_label"] == "high_quality_low_separation"


@pytest.mark.asyncio
async def test_run_planner_resume_missing_artifacts_falls_back_to_cold_start(test_blueprint, mock_model_client, tmp_path):
    state_dir = tmp_path / "planner_state"
    state_dir.mkdir()
    resume_state = PlannerState(
        task_id="test_task",
        blueprint_id="test_bp",
        current_round=1,
        completed_targets={"qa": 4, "multiple_choice": 3},
        topic_backlog=TopicBacklog(active=["算法"], deferred=[]),
        model_pool=ModelPool(active=["deepseek-v3"], removed=[]),
        run_history=[
            RunHistoryEntry(
                round_id=1,
                run_id="run_001",
                topics=["算法"],
                qa_target=5,
                multiple_choice_target=3,
                selected_count=8,
                evaluated=True,
            )
        ],
    )
    captured = {}

    def _capture_question_plan(state, blueprint, diagnosis, **kwargs):
        captured["diagnosis_label"] = diagnosis.label
        return QuestionPlan(
            qa_count=0,
            mc_count=0,
            qa_difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
            mc_difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
            topic_budget=1,
        )

    async def _capture_llm_plan(**kwargs):
        captured["diagnosis_label"] = kwargs["diagnosis"].label
        return QuestionPlan(
            qa_count=0,
            mc_count=0,
            qa_difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
            mc_difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
            topic_budget=1,
        )

    with patch("benchforge.agents.planner_agent.planner._llm_question_difficulty_evolver", side_effect=_capture_llm_plan):
        with patch("benchforge.agents.planner_agent.planner.execute_round") as mock_exec:
            with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model", return_value=mock_model_client):
                with patch("benchforge.agents.planner_agent.planner.load_model_registry", return_value={"deepseek-v3": MagicMock(), "deepseek-v3.2": MagicMock()}):
                    await run_planner(
                        test_blueprint,
                        Path("config"),
                        Path("config/model_registry.yaml"),
                        state_dir,
                        resume_state=resume_state,
                    )

    mock_exec.assert_not_called()
    assert captured["diagnosis_label"] == "cold_start"


@pytest.mark.asyncio
async def test_run_planner_skips_zero_work_round(test_blueprint, mock_model_client, tmp_path):
    state_dir = tmp_path / "planner_state"

    zero_plan = QuestionPlan(
        qa_count=0,
        mc_count=0,
        qa_difficulty_distribution={"easy": 0.2, "medium": 0.5, "hard": 0.3},
        mc_difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2},
        topic_budget=1,
    )

    with patch("benchforge.agents.planner_agent.planner._llm_question_difficulty_evolver", new=AsyncMock(return_value=zero_plan)):
        with patch("benchforge.agents.planner_agent.planner.execute_round") as mock_exec:
            with patch("benchforge.agents.planner_agent.planner.topic_adaptive_retriever", new=AsyncMock(return_value=["算法"])):
                with patch("benchforge.agents.planner_agent.planner.ModelLoader.load_model", return_value=mock_model_client):
                    with patch("benchforge.agents.planner_agent.planner.load_model_registry", return_value={"deepseek-v3": MagicMock(), "deepseek-v3.2": MagicMock()}):
                        final_state = await run_planner(
                            test_blueprint,
                            Path("config"),
                            Path("config/model_registry.yaml"),
                            state_dir,
                        )

    mock_exec.assert_not_called()
    assert final_state.current_round == 0
    assert final_state.run_history == []
