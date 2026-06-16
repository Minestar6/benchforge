"""端到端完整流水线测试：qa_agent → verify_agent → model_eval_agent。"""
import asyncio, json, sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
parent_root = project_root.parent
if str(parent_root) not in sys.path:
    sys.path.insert(0, str(parent_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

TASK_ID = "e2e_full"
RUN_ID = "run_001"


class E2EFakeClient:
    provider = "fake"
    model_name = "e2e-fake-model"
    temperature = 0.7
    max_tokens = 2000
    call_count = 0

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        self.call_count += 1
        content = messages[-1].get("content", "") if messages else ""
        if "<final_summary>" in content:
            text = "<final_summary>This document discusses key concepts in the field.</final_summary>"
        elif "clarity" in content and "answerability" in content:
            text = json.dumps({"passed": True, "overall_score": 0.88,
                "dimensions": {"clarity": 0.9, "answerability": 0.85, "faithfulness": 0.9,
                               "difficulty_alignment": 0.85, "mode_alignment": 0.9},
                "failed_reasons": [], "judge_summary": "Well-formed question with clear answer."})
        elif "judge" in model.lower() or "scores" in content.lower():
            text = json.dumps({"scores": {"correctness": 0.85, "completeness": 0.80,
                "faithfulness": 0.90, "hallucination": 0.10}, "reason": "Answer aligns well with reference."})
        elif "multiple_choice" in content.lower() or "A)" in content or "choices" in content.lower():
            text = json.dumps([
                {"question": "What is the primary focus?", "choices": ["(A) Core principles", "(B) Ignoring concepts",
                 "(C) Unrelated topic", "(D) None"], "answer": "A", "question_mode": "multiple_choice",
                 "thought_process": "Tests comprehension.", "question_type": "conceptual",
                 "required_capability": "understanding", "estimated_difficulty": 5,
                 "citations": ["The document discusses key concepts."]},
                {"question": "Which statement best describes the methodology?", "choices": [
                 "(A) Systematic analysis", "(B) Random sampling", "(C) Theoretical framework",
                 "(D) Ad-hoc"], "answer": "A", "question_mode": "multiple_choice",
                 "thought_process": "Requires understanding of methods.", "question_type": "analytical",
                 "required_capability": "analyzing methods", "estimated_difficulty": 6,
                 "citations": ["This document discusses key concepts."]}])
        elif "Answer:" in content:
            text = "Artificial Intelligence is the study of intelligent agents and machine learning."
        else:
            text = json.dumps([
                {"question": "What is the main focus of AI according to the text?",
                 "answer": "Artificial Intelligence is the study of intelligent agents and machine learning.",
                 "question_mode": "qa", "thought_process": "Tests basic comprehension.",
                 "question_type": "factual", "required_capability": "understanding core concepts",
                 "estimated_difficulty": 3,
                 "citations": ["Artificial Intelligence (AI) is intelligence demonstrated by machines.",
                               "AI research is defined as the study of intelligent agents.",
                               "Machine learning is a subset of AI"]},
                {"question": "How does deep learning relate to AI?",
                 "answer": "Deep learning is a subset of machine learning using neural networks.",
                 "question_mode": "qa", "thought_process": "Tests understanding of AI hierarchy.",
                 "question_type": "analytical", "required_capability": "analyzing relationships",
                 "estimated_difficulty": 7,
                 "citations": ["Machine learning is a subset of AI.",
                               "Deep learning uses neural networks with multiple layers.",
                               "Modern AI applications include self-driving cars."]}])
        return {"text": text, "input_tokens": 150, "output_tokens": 200, "latency": 0.01,
                "raw": {}, "llm_call_id": f"e2e_fake_{self.call_count:04d}"}

    async def batch_complete(self, model, messages_list, **kwargs):
        return [await self.complete(model, msgs, **kwargs) for msgs in messages_list]


async def stage_generation():
    from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg
    from benchforge.agents.qa_agent import run_generation_agent
    from benchforge.schemas import SourceDocument, DocumentStatus

    logger.info("=" * 60)
    logger.info("Stage 1/3: qa_agent — 题目生成")
    logger.info("=" * 60)

    blueprint = Blueprint(task_id=TASK_ID, run_id=RUN_ID, language="en",
        topics=["Artificial Intelligence"],
        modes={"qa": ModeCfg(count=4, max_rounds=2,
            difficulty_distribution={"easy": 0.5, "medium": 0.5, "hard": 0.0})})

    client = E2EFakeClient()

    # Monkey-patch Wikipedia 检索 + 模型解析
    import benchforge.agents.qa_agent.evidence_manager as evidence_mod
    import benchforge.agents.qa_agent.agent as agent_mod
    agent_mod._resolve_model_client_fn = lambda name, path: client

    FAKE_DOC_ID = "doc_e2e_ai_001"
    FAKE_DOC_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    FAKE_DOC_TITLE = "Artificial Intelligence"
    FAKE_DOC_CONTENT = (
        "Artificial Intelligence (AI) is intelligence demonstrated by machines. "
        "AI research is defined as the study of intelligent agents. "
        "Machine learning is a subset of AI that enables systems to learn from data. "
        "Deep learning uses neural networks with multiple layers. "
        "Modern AI applications include self-driving cars, medical diagnosis, and language translation. "
        + "AI research continues to evolve rapidly. " * 80)

    class FakeSearchResult:
        url = FAKE_DOC_URL
        title = FAKE_DOC_TITLE

    evidence_mod.search_wikipedia = lambda query, **kwargs: [FakeSearchResult()]
    evidence_mod.fetch_wikipedia_page = lambda result, run_id, **kwargs: SourceDocument(
        document_id=FAKE_DOC_ID, run_id=run_id, topic="Artificial Intelligence",
        language="en", title=FAKE_DOC_TITLE, url=FAKE_DOC_URL,
        summary="Artificial Intelligence is the study of intelligent agents and machine learning.",
        content=FAKE_DOC_CONTENT, status=DocumentStatus.FETCHED)

    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=str(project_root / "config/qa_agent.yaml"))

    logger.info(f"qa_agent done: {report['total_candidates']} total candidates")
    ss_path = Path("runs") / TASK_ID / RUN_ID / "shared_state.json"
    assert ss_path.exists()
    return report


async def stage_verification():
    from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
    from benchforge.agents.verify_agent.config_loader import (
        CitationCfg, LLMValidationCfg, SelectionCfg, VerifyAgentConfig)

    logger.info("=" * 60)
    logger.info("Stage 2/3: verify_agent — 题目验证")
    logger.info("=" * 60)

    config = VerifyAgentConfig(
        citation=CitationCfg(enabled=True, min_citation_score=0.65, alpha=0.7, beta=0.3, citation_match_threshold=0.5),
        llm_validation=LLMValidationCfg(enabled=True, model="e2e-fake-model", temperature=0.0, max_tokens=800,
            min_overall_score=0.75, max_concurrency=4, max_retries=0,
            prompt_path=str(project_root / "prompts/verify_agent/quality_prompt.md"),
            prompt_user=""),
        selection=SelectionCfg(enabled=True, embedding_model="all-MiniLM-L6-v2"))

    shared_state_path = str(Path("runs") / TASK_ID / RUN_ID / "shared_state.json")
    result = await run_verify_agent_from_shared_state(shared_state_path, config, E2EFakeClient())
    logger.info(f"verify_agent done: selected={len(result.selected_question_ids)}")
    return result


async def stage_evaluation():
    import benchforge.agents.model_eval_agent.agent as agent_module
    from benchforge.agents.model_eval_agent.config_loader import (
        DatasetEvaluationConfig, DatasetMetricSpec, JudgeConfig, JudgeMetricSpec,
        ModelEvalAgentConfig, ModelsConfig, QuestionModeMetricPlan, RunConfig, AutoMetricSpec)

    logger.info("=" * 60)
    logger.info("Stage 3/3: model_eval_agent — 模型评估")
    logger.info("=" * 60)

    config = ModelEvalAgentConfig(
        run=RunConfig(shared_state_path=str(Path("runs") / TASK_ID / RUN_ID / "shared_state.json")),
        dataset_evaluation=DatasetEvaluationConfig(enabled=True,
            metrics=[DatasetMetricSpec(name="citation_score", threshold=0.85)]),
        models=ModelsConfig(candidate_model_names=["candidate-qwen", "candidate-llama"],
            judge_model_name="e2e-judge",
            generation_defaults={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
            judge_defaults={"temperature": 0.0, "top_p": 1.0, "max_tokens": 1200}),
        metrics={"qa": QuestionModeMetricPlan(
            automatic_metrics=[AutoMetricSpec(name="exact_match", threshold=1.0),
                AutoMetricSpec(name="f1", threshold=0.7), AutoMetricSpec(name="bertscore", threshold=0.85)],
            llm_judge_metrics=[JudgeMetricSpec(name="correctness", description="Answer correctness."),
                JudgeMetricSpec(name="completeness", description="Answer completeness."),
                JudgeMetricSpec(name="faithfulness", description="Evidence support."),
                JudgeMetricSpec(name="hallucination", description="Unsupported claims.")])},
        judge=JudgeConfig(enabled=True,
            prompt_system=str(project_root / "prompts/model_eval_agent/judge_system_prompt.md"),
            prompt_user=str(project_root / "prompts/model_eval_agent/judge_user_prompt.md")))

    orig_resolve = agent_module.resolve_model_config
    orig_load_model = agent_module.ModelLoader.load_model
    orig_load_registry = agent_module.load_model_registry
    agent_module.resolve_model_config = lambda name, reg: type("FC", (), {"model_name": name, "provider": "fake"})()
    agent_module.ModelLoader.load_model = staticmethod(lambda cfg: E2EFakeClient())
    agent_module.load_model_registry = lambda path: {"candidate-qwen": None, "candidate-llama": None, "e2e-judge": None}
    try:
        result = await agent_module.run_model_eval_agent_from_shared_state(
            shared_state_path=str(Path("runs") / TASK_ID / RUN_ID / "shared_state.json"),
            config=config, registry_path=str(project_root / "config/model_registry.yaml"))
    finally:
        agent_module.resolve_model_config = orig_resolve
        agent_module.ModelLoader.load_model = orig_load_model
        agent_module.load_model_registry = orig_load_registry
    logger.info(f"model_eval_agent done: {result['num_questions']} questions x {result['num_models']} models")
    return result


def print_artifacts():
    run_dir = Path("runs") / TASK_ID / RUN_ID
    logger.info("=" * 60)
    logger.info("产物文件清单")
    logger.info("=" * 60)
    state = json.loads((run_dir / "shared_state.json").read_text(encoding="utf-8"))
    logger.info(f"shared_state: gen={state['agent_status'].get('generation')}, "
                f"verify={state['agent_status'].get('verification')}, "
                f"eval={state['agent_status'].get('evaluation')}")
    for f in sorted(run_dir.rglob("*")):
        if f.is_file() and f.name != "pipeline.log":
            logger.info(f"  {f.relative_to(run_dir)} ({f.stat().st_size}B)")


async def main():
    import os, shutil
    original_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        old_runs = Path("runs") / TASK_ID
        if old_runs.exists():
            shutil.rmtree(old_runs, ignore_errors=True)
        logger.add("runs/e2e_full/pipeline.log", level="DEBUG", encoding="utf-8", rotation="10 MB")
        logger.info(f"Pipeline start: {TASK_ID} / {RUN_ID}")
        gen_report = await stage_generation()
        verify_result = await stage_verification()
        eval_result = await stage_evaluation() if len(verify_result.selected_question_ids) > 0 else {"num_questions": 0, "num_models": 0}
        print_artifacts()
        logger.info("ALL 3 STAGES PASSED")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    asyncio.run(main())