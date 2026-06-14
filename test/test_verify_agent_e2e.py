"""verify_agent 端到端测试：使用 FakeModelClient，验证完整三阶段流程可跑通。"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from benchforge.agents.verify_agent.agent import VerifyAgent, run_verify_agent
from benchforge.agents.verify_agent.config_loader import (
    CitationCfg,
    LLMValidationCfg,
    SelectionCfg,
    VerifyAgentConfig,
)
from benchforge.agents.verify_agent.schema import ModeCfg, ValidationBlueprintView
from benchforge.models.fake import FakeModelClient


GOOD_LLM_RESPONSE = json.dumps({
    "passed": True,
    "overall_score": 0.85,
    "dimensions": {
        "clarity": 0.9,
        "answerability": 0.85,
        "faithfulness": 0.9,
        "difficulty_alignment": 0.8,
        "mode_alignment": 0.9,
    },
    "failed_reasons": [],
    "judge_summary": "Well-formed question with clear answer supported by source.",
})


class ValidatorFakeClient(FakeModelClient):
    """专为 verify_agent 测试定制的 FakeModelClient，返回合法 LLM 评分 JSON。"""

    def __init__(self, response_text: str = GOOD_LLM_RESPONSE, delay: float = 0.0):
        super().__init__(delay=delay)
        self._response_text = response_text

    async def complete(self, model, messages, **kwargs):
        resp = await super().complete(model, messages, **kwargs)
        resp["text"] = self._response_text
        return resp


def _make_candidate_pool(tmp_dir: Path, n: int = 5) -> Path:
    """在 tmp_dir 创建包含 n 道题的 candidate_pool.json。"""
    pool = []
    for i in range(n):
        pool.append({
            "question": f"What is concept {i}?",
            "answer": f"Concept {i} refers to idea number {i} in the domain.",
            "question_mode": "qa",
            "question_type": "factoid",
            "required_capability": "comprehension",
            "estimated_difficulty": (i % 3) * 3 + 3,  # 3, 6, 9, 3, 6
            "topic": "AI concepts",
            "citations": [f"Concept {i} refers to idea number {i} in the domain."],
            "chunks": [
                f"Concept {i} refers to idea number {i} in the domain. "
                f"This concept has been studied extensively."
            ],
            "chunk_ids": [f"doc_test::chunk_{i:04d}"],
        })
    qa_dir = tmp_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    pool_file = qa_dir / "candidate_pool.json"
    pool_file.write_text(json.dumps(pool), encoding="utf-8")
    return pool_file


def _make_config(tmp_dir: Path, pool_file: Path) -> VerifyAgentConfig:
    return VerifyAgentConfig(
        citation=CitationCfg(
            enabled=True,
            min_citation_score=0.3,  # 宽松阈值，确保测试数据通过
            alpha=0.7,
            beta=0.3,
            citation_match_threshold=0.5,
        ),
        llm_validation=LLMValidationCfg(
            enabled=True,
            min_overall_score=0.75,
            max_concurrency=4,
            max_retries=0,
        ),
        selection=SelectionCfg(
            enabled=True,
        ),
    )


def _make_blueprint() -> ValidationBlueprintView:
    return ValidationBlueprintView(
        topics=["AI concepts"],
        modes={
            "qa": ModeCfg(
                count=5,
                difficulty_distribution={"easy": 2, "medium": 2, "hard": 1},
            )
        },
    )


class TestE2EHappyPath:
    def test_run_verify_agent_produces_outputs(self, tmp_path):
        """完整 happy path：全部题通过，产出 validation/ 目录和报告。"""
        # 模拟 runs/task_e2e/run_e2e/ 目录结构
        run_dir = tmp_path / "runs" / "task_e2e" / "run_e2e"
        run_dir.mkdir(parents=True)
        pool_file = _make_candidate_pool(run_dir)

        config = _make_config(tmp_path, pool_file)
        blueprint = _make_blueprint()
        client = ValidatorFakeClient()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = asyncio.run(
                run_verify_agent(
                    input_paths=[str(pool_file)],
                    blueprint=blueprint,
                    config=config,
                    task_id="task_e2e",
                    run_id="run_e2e",
                    model_client=client,
                )
            )
        finally:
            os.chdir(original_cwd)

        # 验证返回值
        assert result.task_id == "task_e2e"
        assert result.run_id == "run_e2e"
        assert isinstance(result.selected_question_ids, list)

        # 验证落盘文件
        validation_dir = tmp_path / "runs" / "task_e2e" / "run_e2e" / "validation"
        assert validation_dir.exists()
        assert (validation_dir / "citation_validation.jsonl").exists()
        assert (validation_dir / "llm_validation.jsonl").exists()
        assert (validation_dir / "validated_questions.jsonl").exists()
        assert (validation_dir / "validation_report.json").exists()

        # 验证报告内容
        report = json.loads((validation_dir / "validation_report.json").read_text())
        assert report["total_candidates"] == 5
        assert report["citation_passed"] >= 0
        assert "final_selected" in report

    def test_citation_validation_jsonl_structure(self, tmp_path):
        """citation_validation.jsonl 每行包含必要字段。"""
        run_dir = tmp_path / "runs" / "task_e2e" / "run_e2e"
        run_dir.mkdir(parents=True)
        pool_file = _make_candidate_pool(run_dir, n=3)

        config = _make_config(tmp_path, pool_file)
        blueprint = _make_blueprint()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            asyncio.run(
                run_verify_agent([str(pool_file)], blueprint, config, task_id="task_e2e", run_id="run_e2e", model_client=ValidatorFakeClient())
            )
        finally:
            os.chdir(original_cwd)

        validation_dir = tmp_path / "runs" / "task_e2e" / "run_e2e" / "validation"
        lines = (validation_dir / "citation_validation.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            rec = json.loads(line)
            assert "question_id" in rec
            assert "passed" in rec
            assert "citation_score" in rec

    def test_without_model_client(self, tmp_path):
        """不传 model_client 时，LLM 阶段跳过，仍能完整运行。"""
        run_dir = tmp_path / "runs" / "task_e2e" / "run_e2e"
        run_dir.mkdir(parents=True)
        pool_file = _make_candidate_pool(run_dir, n=3)

        config = _make_config(tmp_path, pool_file)
        blueprint = _make_blueprint()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = asyncio.run(
                run_verify_agent([str(pool_file)], blueprint, config, task_id="task_e2e", run_id="run_e2e", model_client=None)
            )
        finally:
            os.chdir(original_cwd)

        assert result.task_id == "task_e2e"

    def test_mixed_fail_path(self, tmp_path):
        """混合失败场景：部分题 citations 为空，验证 failed_by_stage 正确统计。"""
        run_dir = tmp_path / "runs" / "task_e2e" / "run_e2e"
        run_dir.mkdir(parents=True)

        # 2 道正常题 + 1 道 citations 为空的题
        pool = [
            {
                "question": "Good question?", "answer": "Good answer.",
                "question_mode": "qa", "topic": "AI",
                "citations": ["Good answer."],
                "chunks": ["Good answer text."],
                "chunk_ids": ["doc_test::chunk_0000"],
                "estimated_difficulty": 5,
            },
            {
                "question": "Another good question?", "answer": "Another good answer.",
                "question_mode": "qa", "topic": "AI",
                "citations": ["Another good answer."],
                "chunks": ["Another good answer text."],
                "chunk_ids": ["doc_test::chunk_0001"],
                "estimated_difficulty": 5,
            },
            {
                "question": "No citations question?", "answer": "No answer.",
                "question_mode": "qa", "topic": "AI",
                "citations": [],  # 空 citations
                "chunks": ["Some chunk."],
                "chunk_ids": ["doc_test::chunk_0002"],
                "estimated_difficulty": 5,
            },
        ]
        qa_dir = run_dir / "qa"
        qa_dir.mkdir(parents=True)
        pool_file = qa_dir / "candidate_pool.json"
        pool_file.write_text(json.dumps(pool), encoding="utf-8")

        config = _make_config(tmp_path, pool_file)
        blueprint = _make_blueprint()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = asyncio.run(
                run_verify_agent([str(pool_file)], blueprint, config, task_id="task_e2e", run_id="run_e2e", model_client=ValidatorFakeClient())
            )
        finally:
            os.chdir(original_cwd)

        # 1 道题应被 citation 拒绝
        assert result.failed_by_stage.get("citation", 0) >= 1

    def test_run_verify_agent_from_shared_state(self, tmp_path):
        """只给 shared_state.json 时，verify_agent 可自动定位 candidate_pool 和 blueprint。"""
        run_dir = tmp_path / "runs" / "task_e2e" / "run_e2e"
        run_dir.mkdir(parents=True)
        _make_candidate_pool(run_dir, n=3)

        shared_state = {
            "task_id": "task_e2e",
            "run_id": "run_e2e",
            "blueprint": {
                "topics": ["AI concepts"],
                "modes": {
                    "qa": {
                        "count": 5,
                        "difficulty_distribution": {"easy": 2, "medium": 2, "hard": 1},
                    }
                },
            },
            "artifacts": {
                "qa_candidate_pool": "runs/task_e2e/run_e2e/qa/candidate_pool.json",
                "chunked_evidence": "runs/task_e2e/run_e2e/evidence/chunked.json",
            },
            "agent_status": {"generation": "completed", "verification": "pending"},
        }
        (run_dir / "shared_state.json").write_text(json.dumps(shared_state), encoding="utf-8")

        config = VerifyAgentConfig(
            citation=CitationCfg(
                enabled=True,
                min_citation_score=0.3,
                alpha=0.7,
                beta=0.3,
                citation_match_threshold=0.5,
            ),
            llm_validation=LLMValidationCfg(
                enabled=True,
                min_overall_score=0.75,
                max_concurrency=4,
                max_retries=0,
            ),
            selection=SelectionCfg(
                enabled=True,
            ),
        )

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state

            result = asyncio.run(
                run_verify_agent_from_shared_state(
                    shared_state_path="runs/task_e2e/run_e2e/shared_state.json",
                    config=config,
                    model_client=ValidatorFakeClient(),
                )
            )
        finally:
            os.chdir(original_cwd)

        assert result.task_id == "task_e2e"
        assert result.run_id == "run_e2e"
        validation_dir = tmp_path / "runs" / "task_e2e" / "run_e2e" / "validation"
        assert (validation_dir / "validation_report.json").exists()
