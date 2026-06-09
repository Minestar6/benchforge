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
)
from benchforge.agents.planner_agent.planner import run_planner
from benchforge.agents.planner_agent.config_loader import load_planner_state


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
    """Mock 模型客户端。"""
    client = MagicMock()
    client.generate = AsyncMock(return_value="Mock response")
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
    test_blueprint.stop_conditions.max_rounds = 10  # 提高 max_rounds，让 targets 优先触发

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
