"""experiment/case/run_case_eval.py tests."""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.asyncio
async def test_case_eval_main_uses_latest_case_run_and_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import experiment.case.run_case_eval as case_eval_module

    case_base = tmp_path / "runs" / "case"
    older_dir = case_base / "d_full_case_20260616_120000"
    older_dir.mkdir(parents=True)
    (older_dir / "shared_state.json").write_text("{}", encoding="utf-8")

    run_dir = case_base / "d_full_case_20260616_135135"
    run_dir.mkdir(parents=True)
    (run_dir / "shared_state.json").write_text("{}", encoding="utf-8")

    validation_dir = run_dir / "validation"
    validation_dir.mkdir()
    (validation_dir / "validated_questions.jsonl").write_text(
        json.dumps({"final_status": "selected", "candidate": {"question_id": "q1"}}) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "experiment" / "case" / "configs" / "model_eval_case.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("run:\n  shared_state_path: ''\n", encoding="utf-8")

    monkeypatch.setattr(case_eval_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(case_eval_module, "CASE_BASE", case_base)
    monkeypatch.setattr(case_eval_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(case_eval_module, "REGISTRY_PATH", tmp_path / "config" / "model_registry.yaml")

    called = []

    class FakeConfig:
        def __init__(self):
            self.run = type("Run", (), {"shared_state_path": ""})()

    monkeypatch.setattr(case_eval_module, "load_model_eval_config", lambda _: FakeConfig())

    async def fake_run_model_eval_agent_from_shared_state(shared_state_path, config, registry_path):
        called.append((shared_state_path, registry_path))
        evaluation_dir = Path(shared_state_path).parent / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "task_id": "case",
            "run_id": "d_full_case_20260616_135135",
            "num_questions": 1,
            "num_models": 2,
            "output_root": str(evaluation_dir),
        }
        (evaluation_dir / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(
        case_eval_module,
        "run_model_eval_agent_from_shared_state",
        fake_run_model_eval_agent_from_shared_state,
    )

    await case_eval_module.main()

    assert called == [(str(run_dir / "shared_state.json"), str(tmp_path / "config" / "model_registry.yaml"))]

    report_path = case_base / "evaluate_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_dir"] == str(run_dir)
    assert report["selected_questions"] == 1
    assert report["evaluation"]["num_questions"] == 1
