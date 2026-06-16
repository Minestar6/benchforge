"""experiment/qa_agent/run_all.py tests."""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.asyncio
async def test_run_all_uses_task_id_without_seed_suffix(monkeypatch):
    import experiment.qa_agent.run_all as run_all_module

    captured_task_ids: list[str] = []

    async def fake_run(**kwargs):
        captured_task_ids.append(kwargs["task_id"])

    monkeypatch.setattr(run_all_module, "CUSTOM_API_KEY", "test-key")
    monkeypatch.setattr(run_all_module, "GROUPS", [("A", fake_run)])
    monkeypatch.setattr(run_all_module, "MODES", ["qa"])
    monkeypatch.setattr(run_all_module, "SEEDS", [42])
    monkeypatch.setattr(run_all_module, "TASK_ID", "ques_generate")

    await run_all_module.main()

    assert captured_task_ids == ["ques_generate"]
