"""端到端流水线测试：verify_agent → model_eval_agent。

通过 shared_state.json 传递上下文，使用 FakeModelClient 避免真实 API 调用。
验证配置加载、数据传递、产物写入全链路可跑通。
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # benchforge/
# 确保 benchforge 可在任何 CWD 下导入
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 也加入父目录（benchforge 依赖的顶层包）
parent_root = project_root.parent
if str(parent_root) not in sys.path:
    sys.path.insert(0, str(parent_root))

# ═══════════════════════════════════════════════════════════════════
# 测试数据：模拟 qa_agent 产出的 candidate_pool.json
# ═══════════════════════════════════════════════════════════════════

TASK_ID = "e2e_001"
RUN_ID = "run_001"


def build_test_candidates(n: int = 12) -> list[dict]:
    """生成覆盖 3 个 topic、2 种 mode、3 级难度的候选题目。"""
    topics = ["AI Safety", "Quantum Computing", "Climate Science"]
    modes = ["qa", "multiple_choice"]
    difficulties = [
        {"label": "easy", "score": 3},
        {"label": "medium", "score": 6},
        {"label": "hard", "score": 9},
    ]
    candidates = []
    for i in range(n):
        topic = topics[i % len(topics)]
        mode = modes[i % len(modes)]
        diff = difficulties[i % len(difficulties)]
        # 每道题的 citations 包含与 answer 匹配的关键词，确保引用验证通过
        q = {
            "question_id": f"q_{i:04d}",
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "topic": topic,
            "question": f"What is the key insight of {topic} concept #{i}?",
            "answer": f"The key insight is understanding the principle of {topic} model number {i}.",
            "question_mode": mode,
            "question_type": "factual" if mode == "qa" else "conceptual",
            "required_capability": f"understanding {topic}",
            "estimated_difficulty": diff["score"],
            "citations": [
                f"understanding the principle of {topic} model number {i}"
            ],
            "document_id": f"doc_{topic.replace(' ', '_').lower()}",
            "chunk_ids": [f"doc_{topic.replace(' ', '_').lower()}::chunk_{i:04d}"],
            "chunks": [
                (
                    f"The {topic} model number {i} demonstrates that "
                    f"understanding the principle of {topic} model number {i} "
                    f"is essential for further research."
                )
            ],
            "generation_metadata": {
                "llm_call_id": f"llm_call_{i:04d}",
                "generation_round": (i // 4) + 1,
            },
        }
        if mode == "multiple_choice":
            q["choices"] = [
                "(A) understanding the principle",
                "(B) ignoring the principle",
                "(C) a completely unrelated concept",
                "(D) None of the above",
            ]
        candidates.append(q)
    return candidates


def build_test_chunked_evidence(candidates: list[dict]) -> dict[str, list[dict]]:
    """构建 chunked.json 索引（{topic: [doc_records]}），供 verify_agent 的 normalizer 回溯 chunk 文本。"""
    result: dict[str, list[dict]] = {}
    seen_docs: dict[str, dict] = {}
    for c in candidates:
        doc_id = c["document_id"]
        topic = c["topic"]
        if doc_id not in seen_docs:
            seen_docs[doc_id] = {
                "document_id": doc_id,
                "topic": topic,
                "chunks": [],
            }
        for cid in c.get("chunk_ids", []):
            seen_docs[doc_id]["chunks"].append({
                "chunk_id": cid,
                "chunk_text": c.get("chunks", [""])[0] if c.get("chunks") else "",
            })
    for doc in seen_docs.values():
        result.setdefault(doc["topic"], []).append(doc)
    return result


# ═══════════════════════════════════════════════════════════════════
# Fake Model Client（返回合法 JSON，模拟 LLM 行为）
# ═══════════════════════════════════════════════════════════════════

class E2EFakeClient:
    """返回合法评分的假客户端，同时充当候选模型和 judge 模型。"""
    provider = "fake"

    def __init__(self, model_name: str = "fake-model"):
        self.model_name = model_name

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        content = messages[-1].get("content", "") if messages else ""
        # LLM 验证响应（verify_agent）
        if "clarity" in content and "answerability" in content:
            text = json.dumps({
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
                "judge_summary": "Well-formed question.",
            })
        # 模型推理响应（model_eval_agent）
        elif "answer with the option letter" in content.lower():
            text = "A"
        else:
            text = "The key insight is understanding the principle of the model."
        return {
            "text": text,
            "input_tokens": 100,
            "output_tokens": 50,
            "latency": 0.01,
            "raw": {},
            "llm_call_id": "fake_call",
        }

    async def batch_complete(self, model, messages_list, **kwargs):
        return [await self.complete(model, msgs, **kwargs) for msgs in messages_list]


# ═══════════════════════════════════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════════════════════════════════

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


# ═══════════════════════════════════════════════════════════════════
# Stage 1: 准备测试目录和 shared_state.json
# ═══════════════════════════════════════════════════════════════════

def prepare_run_dir() -> tuple[Path, dict]:
    run_dir = Path("runs") / TASK_ID / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    # 写入 candidate_pool
    candidates = build_test_candidates(12)
    qa_dir = run_dir / "qa"
    qa_dir.mkdir(exist_ok=True)
    (qa_dir / "candidate_pool.json").write_text(json.dumps(candidates), encoding="utf-8")

    # 写入 chunked evidence
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    chunked = build_test_chunked_evidence(candidates)
    with open(evidence_dir / "chunked.json", "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)

    # 写入 shared_state.json（模拟 qa_agent 已运行完毕）
    shared_state = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "blueprint": {
            "topics": ["AI Safety", "Quantum Computing", "Climate Science"],
            "modes": {
                "qa": {
                    "count": 6,
                    "difficulty_distribution": {"easy": 2, "medium": 2, "hard": 2},
                },
                "multiple_choice": {
                    "count": 6,
                    "difficulty_distribution": {"easy": 2, "medium": 2, "hard": 2},
                },
            },
        },
        "artifacts": {
            "qa_candidate_pool": str(qa_dir / "candidate_pool.json"),
            "chunked_evidence": str(evidence_dir / "chunked.json"),
        },
        "agent_status": {
            "generation": "completed",
            "verification": "pending",
            "evaluation": "pending",
        },
    }
    (run_dir / "shared_state.json").write_text(
        json.dumps(shared_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"Prepared test data at {run_dir}")
    logger.info(f"  candidates: {len(candidates)}")
    logger.info(f"  topics: {shared_state['blueprint']['topics']}")
    return run_dir, shared_state


# ═══════════════════════════════════════════════════════════════════
# Stage 2: verify_agent
# ═══════════════════════════════════════════════════════════════════

async def stage_verify(run_dir: Path) -> dict:
    from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
    from benchforge.agents.verify_agent.config_loader import (
        CitationCfg, LLMValidationCfg, SelectionCfg, VerifyAgentConfig,
    )

    config = VerifyAgentConfig(
        citation=CitationCfg(
            enabled=True,
            min_citation_score=0.65,
            alpha=0.7,
            beta=0.3,
            citation_match_threshold=0.5,
        ),
        llm_validation=LLMValidationCfg(
            enabled=True,
            model="fake-model",
            temperature=0.0,
            max_tokens=800,
            min_overall_score=0.75,
            max_concurrency=4,
            max_retries=0,
            prompt_system=str(project_root / "prompts/verify_agent/quality_system_prompt.md"),
            prompt_user=str(project_root / "prompts/verify_agent/quality_user_prompt.md"),
        ),
        selection=SelectionCfg(
            enabled=True,
            embedding_model="all-MiniLM-L6-v2",
        ),
    )

    shared_state_path = str(run_dir / "shared_state.json")
    client = E2EFakeClient("verify-judge")

    logger.info("=" * 60)
    logger.info("Stage 2: verify_agent")
    logger.info("=" * 60)

    result = await run_verify_agent_from_shared_state(
        shared_state_path=shared_state_path,
        config=config,
        model_client=client,
    )

    logger.info(f"verify_agent result:")
    logger.info(f"  task_id: {result.task_id}")
    logger.info(f"  run_id: {result.run_id}")
    logger.info(f"  selected: {len(result.selected_question_ids)}")
    logger.info(f"  failed_by_stage: {result.failed_by_stage}")
    logger.info(f"  report: {result.report_path}")

    # 验证产物
    validation_dir = run_dir / "validation"
    assert (validation_dir / "validated_questions.jsonl").exists(), "validated_questions.jsonl not found"
    assert (validation_dir / "validation_report.json").exists(), "validation_report.json not found"

    report = json.loads((validation_dir / "validation_report.json").read_text())
    logger.info(f"  report: total_candidates={report['total_candidates']}, "
                f"citation_passed={report['citation_passed']}, "
                f"llm_passed={report['llm_passed']}, "
                f"final_selected={report['final_selected']}")

    assert report["total_candidates"] == 12
    assert report["citation_passed"] > 0
    assert report["final_selected"] > 0

    # 验证 shared_state 已回写
    state = json.loads((run_dir / "shared_state.json").read_text())
    assert state["agent_status"]["verification"] == "completed"
    assert "validated_questions" in state["artifacts"]

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage 3: model_eval_agent
# ═══════════════════════════════════════════════════════════════════

async def stage_evaluate(run_dir: Path) -> dict:
    from benchforge.agents.model_eval_agent.agent import run_model_eval_agent_from_shared_state
    from benchforge.agents.model_eval_agent.config_loader import (
        DatasetEvaluationConfig, DatasetMetricSpec,
        JudgeConfig, JudgeMetricSpec, ModelEvalAgentConfig,
        ModelsConfig, QuestionModeMetricPlan, RunConfig, AutoMetricSpec,
    )

    config = ModelEvalAgentConfig(
        run=RunConfig(shared_state_path=str(run_dir / "shared_state.json")),
        dataset_evaluation=DatasetEvaluationConfig(
            enabled=True,
            metrics=[DatasetMetricSpec(name="citation_score", threshold=0.85)],
        ),
        models=ModelsConfig(
            candidate_model_names=["candidate-1", "candidate-2"],
            judge_model_name="judge",
            generation_defaults={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
            judge_defaults={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1200},
        ),
        metrics={
            "qa": QuestionModeMetricPlan(
                automatic_metrics=[AutoMetricSpec(name="exact_match", threshold=1.0)],
                llm_judge_metrics=[
                    JudgeMetricSpec(name="correctness",
                                    description="Judge correctness of the answer",
                                    direction="higher_is_better"),
                ],
            ),
            "multiple_choice": QuestionModeMetricPlan(
                automatic_metrics=[AutoMetricSpec(name="accuracy", threshold=1.0)],
                llm_judge_metrics=[],
            ),
        },
        judge=JudgeConfig(
            enabled=True,
            prompt_system=str(project_root / "prompts/model_eval_agent/judge_system_prompt.md"),
            prompt_user=str(project_root / "prompts/model_eval_agent/judge_user_prompt.md"),
        ),
    )

    logger.info("=" * 60)
    logger.info("Stage 3: model_eval_agent")
    logger.info("=" * 60)

    # Monkey-patch: 注册表加载 + 模型加载全部替换为 fake
    import benchforge.agents.model_eval_agent.agent as agent_module

    original_resolve = agent_module.resolve_model_config
    original_load_model = agent_module.ModelLoader.load_model
    original_load_registry = agent_module.load_model_registry

    def fake_resolve(name, registry):
        class FakeCfg:
            model_name = name
            provider = "fake"
        return FakeCfg()

    def fake_load_registry(path):
        return {"candidate-1": None, "candidate-2": None, "judge": None}

    agent_module.resolve_model_config = fake_resolve
    agent_module.ModelLoader.load_model = staticmethod(lambda cfg: E2EFakeClient(cfg.model_name))
    agent_module.load_model_registry = fake_load_registry

    try:
        result = await run_model_eval_agent_from_shared_state(
            shared_state_path=str(run_dir / "shared_state.json"),
            config=config,
            registry_path=str(project_root / "config/model_registry.yaml"),
        )
    finally:
        agent_module.resolve_model_config = original_resolve
        agent_module.ModelLoader.load_model = original_load_model
        agent_module.load_model_registry = original_load_registry

    logger.info(f"model_eval_agent result:")
    logger.info(f"  task_id: {result['task_id']}")
    logger.info(f"  run_id: {result['run_id']}")
    logger.info(f"  num_questions: {result['num_questions']}")
    logger.info(f"  num_models: {result['num_models']}")
    logger.info(f"  output_root: {result['output_root']}")

    eval_dir = run_dir / "evaluation"
    assert (eval_dir / "evaluation_report.json").exists(), "evaluation_report.json not found"

    # 验证 shared_state 已回写
    state = json.loads((run_dir / "shared_state.json").read_text())
    assert state["agent_status"]["evaluation"] == "completed"
    assert "evaluation_report" in state["artifacts"]

    return result


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

async def main():
    import os
    original_cwd = os.getcwd()
    try:
        # 切换到 benchforge 目录
        os.chdir(project_root)
        # 子目录隔离运行产物
        test_work = Path("test/e2e_workdir")
        test_work.mkdir(parents=True, exist_ok=True)
        os.chdir(test_work)

        logger.info("=" * 60)
        logger.info("E2E Pipeline Test: verify_agent → model_eval_agent")
        logger.info(f"Working dir: {os.getcwd()}")
        logger.info("=" * 60)

        run_dir, _ = prepare_run_dir()
        verify_result = await stage_verify(run_dir)
        eval_result = await stage_evaluate(run_dir)

        logger.info("=" * 60)
        logger.info("ALL STAGES PASSED")
        logger.info(f"  verify: {len(verify_result.selected_question_ids)} selected")
        logger.info(f"  eval:   {eval_result['num_questions']} questions × {eval_result['num_models']} models")
        logger.info("=" * 60)

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    asyncio.run(main())
