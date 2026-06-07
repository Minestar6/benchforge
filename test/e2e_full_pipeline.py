"""端到端完整流水线测试：qa_agent → verify_agent → model_eval_agent。

通过 shared_state.json 串联三个阶段，使用 FakeModelClient。
产物保留在 runs/{task_id}/{run_id}/ 下供检查。
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # benchforge/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
parent_root = project_root.parent  # master/
if str(parent_root) not in sys.path:
    sys.path.insert(0, str(parent_root))

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
# 同时写日志文件
# ═══════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════

TASK_ID = "e2e_full"
RUN_ID = "run_001"


# ═══════════════════════════════════════════════════════════════════
# Fake 客户端：模拟 LLM 返回合法 JSON
# ═══════════════════════════════════════════════════════════════════

class E2EFakeClient:
    """模拟 LLM：对不同的 prompt 返回不同格式的合法响应。"""
    provider = "fake"
    model_name = "e2e-fake-model"
    temperature = 0.7
    max_tokens = 2000
    call_count = 0

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        self.call_count += 1
        content = messages[-1].get("content", "") if messages else ""

        # 判断调用类型
        if "<final_summary>" in content:
            # 文档总结
            text = "<final_summary>This document discusses key concepts in the field.</final_summary>"
        elif "clarity" in content and "answerability" in content:
            # 题目验证（verify_agent LLM validation）
            text = json.dumps({
                "passed": True,
                "overall_score": 0.88,
                "dimensions": {
                    "clarity": 0.9,
                    "answerability": 0.85,
                    "faithfulness": 0.9,
                    "difficulty_alignment": 0.85,
                    "mode_alignment": 0.9,
                },
                "failed_reasons": [],
                "judge_summary": "Well-formed question with clear answer.",
            })
        elif "judge" in model.lower() or "scores" in content.lower():
            # 模型评估 judge
            text = json.dumps({
                "scores": {
                    "correctness": 0.85,
                    "completeness": 0.80,
                    "faithfulness": 0.90,
                    "hallucination": 0.10,
                },
                "reason": "Answer aligns well with reference.",
            })
        elif "multiple_choice" in content.lower() or "A)" in content or "choices" in content.lower():
            # MCQ 题目生成
            text = json.dumps([
                {
                    "question": "What is the primary focus of the discussed concept?",
                    "choices": [
                        "(A) Understanding the core principles",
                        "(B) Ignoring fundamental concepts",
                        "(C) A completely unrelated topic",
                        "(D) None of the above",
                    ],
                    "answer": "A",
                    "question_mode": "multiple_choice",
                    "thought_process": "Tests comprehension of the main concept.",
                    "question_type": "conceptual",
                    "required_capability": "understanding core principles",
                    "estimated_difficulty": 5,
                    "citations": ["The document discusses key concepts in the field."],
                },
                {
                    "question": "Which statement best describes the methodology used?",
                    "choices": [
                        "(A) Systematic analysis of data",
                        "(B) Random sampling approach",
                        "(C) Purely theoretical framework",
                        "(D) Ad-hoc experimentation",
                    ],
                    "answer": "A",
                    "question_mode": "multiple_choice",
                    "thought_process": "Requires understanding of research methods.",
                    "question_type": "analytical",
                    "required_capability": "analyzing research methodology",
                    "estimated_difficulty": 6,
                    "citations": ["This document discusses key concepts in the field."],
                },
            ])
        elif "Answer:" in content:
            # 模型推理
            text = "Artificial Intelligence is the study of intelligent agents and machine learning."
        else:
            # QA 题目生成 — citation 必须匹配 Wikipedia chunk 中的实际文本
            text = json.dumps([
                {
                    "question": "What is the main focus of artificial intelligence according to the text?",
                    "answer": "Artificial Intelligence is the study of intelligent agents and machine learning.",
                    "question_mode": "qa",
                    "thought_process": "Tests basic comprehension of the definition of AI.",
                    "question_type": "factual",
                    "required_capability": "understanding core concepts",
                    "estimated_difficulty": 3,
                    "citations": [
                        "Artificial Intelligence (AI) is intelligence demonstrated by machines.",
                        "AI research is defined as the study of intelligent agents.",
                        "Machine learning is a subset of AI",
                    ],
                },
                {
                    "question": "How does deep learning relate to artificial intelligence and what applications does AI have?",
                    "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. AI applications include self-driving cars, medical diagnosis, and language translation.",
                    "question_mode": "qa",
                    "thought_process": "Tests understanding of the hierarchy of AI concepts and real-world applications.",
                    "question_type": "analytical",
                    "required_capability": "analyzing relationships between AI subfields",
                    "estimated_difficulty": 7,
                    "citations": [
                        "Machine learning is a subset of AI that enables systems to learn from data.",
                        "Deep learning uses neural networks with multiple layers.",
                        "Modern AI applications include self-driving cars, medical diagnosis, and language translation.",
                    ],
                },
            ])

        return {
            "text": text,
            "input_tokens": 150,
            "output_tokens": 200,
            "latency": 0.01,
            "raw": {},
            "llm_call_id": f"e2e_fake_{self.call_count:04d}",
        }

    async def batch_complete(self, model, messages_list, **kwargs):
        return [await self.complete(model, msgs, **kwargs) for msgs in messages_list]


# ═══════════════════════════════════════════════════════════════════
# Stage 1: qa_agent — 题目生成
# ═══════════════════════════════════════════════════════════════════

async def stage_generation():
    """运行 qa_agent，生成候选题目池。"""
    from benchforge.agents.qa_agent.config_loader import load_qa_agent_config
    from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
    from benchforge.agents.qa_agent.evidence_manager import EvidenceManager
    from benchforge.agents.qa_agent.generator import Generator
    from benchforge.agents.qa_agent import run_generation_agent
    from benchforge.config import QuestionGeneratorConfig

    logger.info("=" * 60)
    logger.info("Stage 1/3: qa_agent — 题目生成")
    logger.info("=" * 60)

    # 加载 agent 行为配置（不含 blueprint）
    agent_config, model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg = load_qa_agent_config(
        str(project_root / "config/qa_agent.yaml")
    )

    # 程序化构造 Blueprint（小规模，快速验证）
    blueprint = Blueprint(
        task_id=TASK_ID,
        run_id=RUN_ID,
        language="en",
        topics=["Artificial Intelligence"],
        modes={
            "qa": ModeCfg(
                count=4, max_rounds=2,
                difficulty_distribution={"easy": 0.5, "medium": 0.5, "hard": 0.0},
            ),
        },
    )

    # 构造系统配置（从 qa_agent.yaml 的 retrieval/chunking/summarization 组件组装）
    from benchforge.config.config import RetrievalConfig, ChunkingConfig, SummarizationChunkingConfig
    sys_config = QuestionGeneratorConfig(
        retrieval=retrieval_cfg,
        chunking=chunking_cfg,
        summarization_chunking=sum_chunking_cfg,
    )
    sys_config.run.task_id = TASK_ID
    sys_config.run.run_id = RUN_ID
    sys_config.run.output_path = f"./runs/{TASK_ID}/{RUN_ID}"
    sys_config.run.language = "en"

    # Fake 客户端
    client = E2EFakeClient()

    # Monkey-patch Wikipedia 检索 → 返回假数据，无需网络
    import benchforge.agents.qa_agent.evidence_manager as evidence_mod
    from benchforge.schemas import SourceDocument, DocumentStatus

    FAKE_DOC_ID = "doc_e2e_ai_001"
    FAKE_DOC_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    FAKE_DOC_TITLE = "Artificial Intelligence"
    FAKE_DOC_CONTENT = (
        "Artificial Intelligence (AI) is intelligence demonstrated by machines. "
        "AI research is defined as the study of intelligent agents. "
        "Machine learning is a subset of AI that enables systems to learn from data. "
        "Deep learning uses neural networks with multiple layers. "
        "Modern AI applications include self-driving cars, medical diagnosis, and language translation. "
        "Natural language processing allows machines to understand human language. "
        "Computer vision enables machines to interpret visual information. "
        "Reinforcement learning trains agents through reward signals. "
        "AI ethics addresses the moral implications of artificial intelligence. "
        "The field was founded in 1956 at the Dartmouth workshop. "
        + "AI research continues to evolve rapidly. " * 80  # 填充到足够 chunk_size
    )

    class FakeSearchResult:
        url = FAKE_DOC_URL
        title = FAKE_DOC_TITLE

    def fake_search_wikipedia(query, language="en", max_pages=5, **kwargs):
        return [FakeSearchResult()]

    def fake_fetch_wikipedia_page(result, run_id, language="en", **kwargs):
        return SourceDocument(
            document_id=FAKE_DOC_ID,
            run_id=run_id,
            topic="Artificial Intelligence",
            language=language,
            title=FAKE_DOC_TITLE,
            url=FAKE_DOC_URL,
            summary="Artificial Intelligence is the study of intelligent agents and machine learning.",
            content=FAKE_DOC_CONTENT,
            status=DocumentStatus.FETCHED,
        )

    # 直接替换 evidence_manager 模块中的函数引用
    evidence_mod.search_wikipedia = fake_search_wikipedia
    evidence_mod.fetch_wikipedia_page = fake_fetch_wikipedia_page

    # 运行生成
    report = await run_generation_agent(
        blueprint=blueprint,
        config=agent_config,
        evidence_manager=EvidenceManager(sys_config, client),
        generator=Generator(),
    )

    logger.info(f"qa_agent done: {report['total_candidates']} total candidates")
    for mode, s in report["modes"].items():
        logger.info(f"  {mode}: {s['candidate_count']}/{s['target_candidate_count']} stopped={s['stopped_reason']}")

    # 检查 shared_state.json 已生成
    ss_path = Path("runs") / TASK_ID / RUN_ID / "shared_state.json"
    assert ss_path.exists(), f"shared_state.json not found at {ss_path}"
    state = json.loads(ss_path.read_text(encoding="utf-8"))
    logger.info(f"shared_state: agent_status={state['agent_status']}")
    logger.info(f"shared_state: artifacts={list(state['artifacts'].keys())}")

    return report


# ═══════════════════════════════════════════════════════════════════
# Stage 2: verify_agent — 题目验证
# ═══════════════════════════════════════════════════════════════════

async def stage_verification():
    """运行 verify_agent，从 shared_state 读取候选池并验证。"""
    from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
    from benchforge.agents.verify_agent.config_loader import (
        CitationCfg, LLMValidationCfg, SelectionCfg, VerifyAgentConfig,
    )

    logger.info("=" * 60)
    logger.info("Stage 2/3: verify_agent — 题目验证")
    logger.info("=" * 60)

    config = VerifyAgentConfig(
        citation=CitationCfg(
            enabled=True, min_citation_score=0.65,
            alpha=0.7, beta=0.3, citation_match_threshold=0.5,
        ),
        llm_validation=LLMValidationCfg(
            enabled=True, model="e2e-fake-model",
            temperature=0.0, max_tokens=800,
            min_overall_score=0.75, max_concurrency=4, max_retries=0,
            prompt_system=str(project_root / "prompts/verify_agent/quality_system_prompt.md"),
            prompt_user=str(project_root / "prompts/verify_agent/quality_user_prompt.md"),
        ),
        selection=SelectionCfg(enabled=True, embedding_model="all-MiniLM-L6-v2"),
    )

    shared_state_path = str(Path("runs") / TASK_ID / RUN_ID / "shared_state.json")
    client = E2EFakeClient()

    result = await run_verify_agent_from_shared_state(
        shared_state_path=shared_state_path,
        config=config,
        model_client=client,
    )

    logger.info(f"verify_agent done: selected={len(result.selected_question_ids)}")
    logger.info(f"  failed_by_stage: {result.failed_by_stage}")

    if len(result.selected_question_ids) == 0:
        logger.warning("No questions passed verification — skipping model_eval stage")
        return result

    validation_dir = Path("runs") / TASK_ID / RUN_ID / "validation"
    report = json.loads((validation_dir / "validation_report.json").read_text(encoding="utf-8"))
    logger.info(f"  report: total={report['total_candidates']} "
                f"citation_pass={report['citation_passed']} "
                f"llm_pass={report['llm_passed']} "
                f"final={report['final_selected']}")

    # 验证 shared_state 已回写
    state = json.loads((Path("runs") / TASK_ID / RUN_ID / "shared_state.json").read_text(encoding="utf-8"))
    assert state["agent_status"]["verification"] == "completed"
    assert "validated_questions" in state["artifacts"]

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage 3: model_eval_agent — 模型评估
# ═══════════════════════════════════════════════════════════════════

async def stage_evaluation():
    """运行 model_eval_agent，从 shared_state 读取验证题目并评估。"""
    import benchforge.agents.model_eval_agent.agent as agent_module
    from benchforge.agents.model_eval_agent.config_loader import (
        DatasetEvaluationConfig, DatasetMetricSpec,
        JudgeConfig, JudgeMetricSpec, ModelEvalAgentConfig,
        ModelsConfig, QuestionModeMetricPlan, RunConfig, AutoMetricSpec,
    )

    logger.info("=" * 60)
    logger.info("Stage 3/3: model_eval_agent — 模型评估")
    logger.info("=" * 60)

    config = ModelEvalAgentConfig(
        run=RunConfig(shared_state_path=str(Path("runs") / TASK_ID / RUN_ID / "shared_state.json")),
        dataset_evaluation=DatasetEvaluationConfig(
            enabled=True,
            metrics=[DatasetMetricSpec(name="citation_score", threshold=0.85)],
        ),
        models=ModelsConfig(
            candidate_model_names=["candidate-qwen", "candidate-llama"],
            judge_model_name="e2e-judge",
            generation_defaults={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
            judge_defaults={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1200},
        ),
        metrics={
            "qa": QuestionModeMetricPlan(
                automatic_metrics=[
                    AutoMetricSpec(name="exact_match", threshold=1.0),
                    AutoMetricSpec(name="f1", threshold=0.7),
                    AutoMetricSpec(name="bertscore", threshold=0.85),
                ],
                llm_judge_metrics=[
                    JudgeMetricSpec(name="correctness",
                                    description="Whether the answer correctly addresses the question.",
                                    direction="higher_is_better"),
                    JudgeMetricSpec(name="completeness",
                                    description="Whether the answer covers all key information.",
                                    direction="higher_is_better"),
                    JudgeMetricSpec(name="faithfulness",
                                    description="Whether the answer is supported by evidence.",
                                    direction="higher_is_better"),
                    JudgeMetricSpec(name="hallucination",
                                    description="Whether the answer contains unsupported claims.",
                                    direction="lower_is_better"),
                ],
            ),
        },
        judge=JudgeConfig(
            enabled=True,
            prompt_system=str(project_root / "prompts/model_eval_agent/judge_system_prompt.md"),
            prompt_user=str(project_root / "prompts/model_eval_agent/judge_user_prompt.md"),
        ),
    )

    # Monkey-patch: 用 fake client 替代真实模型加载
    orig_resolve = agent_module.resolve_model_config
    orig_load_model = agent_module.ModelLoader.load_model
    orig_load_registry = agent_module.load_model_registry

    def fake_registry(path):
        return {"candidate-qwen": None, "candidate-llama": None, "e2e-judge": None}

    def fake_resolve(name, registry):
        class FC:
            model_name = name
            provider = "fake"
        return FC()

    agent_module.resolve_model_config = fake_resolve
    agent_module.ModelLoader.load_model = staticmethod(lambda cfg: E2EFakeClient())
    agent_module.load_model_registry = fake_registry

    try:
        result = await agent_module.run_model_eval_agent_from_shared_state(
            shared_state_path=str(Path("runs") / TASK_ID / RUN_ID / "shared_state.json"),
            config=config,
            registry_path=str(project_root / "config/model_registry.yaml"),
        )
    finally:
        agent_module.resolve_model_config = orig_resolve
        agent_module.ModelLoader.load_model = orig_load_model
        agent_module.load_model_registry = orig_load_registry

    logger.info(f"model_eval_agent done: {result['num_questions']} questions x {result['num_models']} models")
    logger.info(f"  output_root: {result['output_root']}")

    # 验证产物
    eval_dir = Path("runs") / TASK_ID / RUN_ID / "evaluation"
    assert (eval_dir / "evaluation_report.json").exists()
    assert (eval_dir / "model_report" / "model_responses.jsonl").exists()

    # 验证 shared_state 回写
    state = json.loads((Path("runs") / TASK_ID / RUN_ID / "shared_state.json").read_text(encoding="utf-8"))
    assert state["agent_status"]["evaluation"] == "completed"

    return result


# ═══════════════════════════════════════════════════════════════════
# 产物汇总
# ═══════════════════════════════════════════════════════════════════

def print_artifacts():
    """打印所有产物文件清单。"""
    run_dir = Path("runs") / TASK_ID / RUN_ID
    logger.info("=" * 60)
    logger.info("产物文件清单")
    logger.info("=" * 60)

    state = json.loads((run_dir / "shared_state.json").read_text(encoding="utf-8"))
    logger.info(f"shared_state: generation={state['agent_status'].get('generation')}, "
                f"verification={state['agent_status'].get('verification')}, "
                f"evaluation={state['agent_status'].get('evaluation')}")

    for f in sorted(run_dir.rglob("*")):
        if f.is_file() and f.name != "pipeline.log":
            size = f.stat().st_size
            if f.suffix in (".jsonl",):
                lines = len(f.read_text(encoding="utf-8").strip().splitlines())
                logger.info(f"  {f.relative_to(run_dir)} ({lines} lines, {size}B)")
            else:
                logger.info(f"  {f.relative_to(run_dir)} ({size}B)")


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

async def main():
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        # 清理旧产物（忽略被锁的日志文件）
        import shutil
        old_runs = Path("runs") / TASK_ID
        if old_runs.exists():
            shutil.rmtree(old_runs, ignore_errors=True)

        # 日志落盘（在清理之后，避免文件被锁）
        logger.add(
            "runs/e2e_full/pipeline.log",
            level="DEBUG",
            encoding="utf-8",
            rotation="10 MB",
        )

        logger.info("Pipeline start: {} / {}", TASK_ID, RUN_ID)
        logger.info("Working dir: {}", os.getcwd())

        # Stage 1
        gen_report = await stage_generation()

        # Stage 2
        verify_result = await stage_verification()

        # Stage 3（仅在有选中题目时运行）
        if len(verify_result.selected_question_ids) > 0:
            eval_result = await stage_evaluation()
        else:
            eval_result = {"num_questions": 0, "num_models": 0}

        # 汇总
        print_artifacts()

        logger.info("=" * 60)
        logger.info("ALL 3 STAGES PASSED")
        logger.info(f"  qa_agent:     {gen_report['total_candidates']} candidates generated")
        logger.info(f"  verify_agent: {len(verify_result.selected_question_ids)} questions selected")
        logger.info(f"  model_eval:   {eval_result['num_questions']} questions x {eval_result['num_models']} models")
        logger.info(f"  artifacts:    runs/{TASK_ID}/{RUN_ID}/")
        logger.info("=" * 60)

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    asyncio.run(main())
