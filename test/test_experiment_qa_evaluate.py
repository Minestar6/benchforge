"""experiment/qa_agent/evaluate.py tests."""

import json
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.asyncio
async def test_evaluate_main_runs_all_experiments_and_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    import experiment.qa_agent.evaluate as evaluate_module

    experiment_base = tmp_path / "runs" / "qa_agent_exp" / "seed_42"
    older_dir = experiment_base / "A_direct_qa_20260615_100000"
    older_dir.mkdir(parents=True)
    (older_dir / "shared_state.json").write_text("{}", encoding="utf-8")

    run_dir = experiment_base / "A_direct_qa_20260615_200000"
    run_dir.mkdir(parents=True)
    (run_dir / "shared_state.json").write_text("{}", encoding="utf-8")

    validation_dir = run_dir / "validation"
    validation_dir.mkdir()
    (validation_dir / "validated_questions.jsonl").write_text(
        json.dumps({"final_status": "selected", "candidate": {"estimated_difficulty": "hard"}}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluate_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate_module, "EXPERIMENT_BASE", experiment_base)
    monkeypatch.setattr(evaluate_module, "CONFIG_PATH", str(tmp_path / "config" / "model_eval_agent.yaml"))
    monkeypatch.setattr(evaluate_module, "REGISTRY_PATH", str(tmp_path / "config" / "model_registry.yaml"))
    monkeypatch.setattr(evaluate_module, "TARGET_RUN_PREFIXES", ["A_direct_qa_"])

    called = []

    class FakeConfig:
        pass

    monkeypatch.setattr(evaluate_module, "load_model_eval_config", lambda _: FakeConfig())

    async def fake_run_model_eval_agent_from_shared_state(shared_state_path, config, registry_path):
        called.append((shared_state_path, registry_path))
        evaluation_dir = Path(shared_state_path).parent / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "task_id": "task_x",
            "run_id": "run_y",
            "num_questions": 1,
            "num_models": 2,
            "output_root": str(evaluation_dir),
        }
        (evaluation_dir / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(
        evaluate_module,
        "run_model_eval_agent_from_shared_state",
        fake_run_model_eval_agent_from_shared_state,
    )

    await evaluate_module.main()

    assert called == [(str(run_dir / "shared_state.json"), str(tmp_path / "config" / "model_registry.yaml"))]

    report_path = experiment_base / "evaluate_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"][0]["group"] == "A_direct_qa_20260615_200000"
    assert report["results"][0]["num_questions"] == 1
    assert report["results"][0]["selected_questions"] == 1
