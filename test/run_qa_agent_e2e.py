# -*- coding: utf-8 -*-
"""qa_agent 端到端手动实验脚本。

演示新 API：run_generation_agent(blueprint, config_path)

用法:
    cd benchforge/
    python test/run_qa_agent_e2e.py          # fake 模型（免网络，始终可运行）
    python test/run_qa_agent_e2e.py real     # 真实模型（需 .env 配置）
"""

import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ─── 路径设置 ────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from loguru import logger

# ─── Blueprint 构造 ──────────────────────────────────────────────────────────
from benchforge.agents.qa_agent.schema import Blueprint, ModeCfg


def build_blueprint(
    task_id: str = "exp_e2e",
    run_id: str | None = None,
    topics: list[str] | None = None,
    qa_count: int = 20,
    mcq_count: int = 10,
    max_rounds: int = 5,
    language: str = "en",
) -> Blueprint:
    """构造 Blueprint——告诉 agent 生成什么。"""
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if topics is None:
        topics = ["Python (programming language)", "Machine learning"]

    return Blueprint(
        task_id=task_id,
        run_id=run_id,
        language=language,
        topics=topics,
        modes={
            "qa": ModeCfg(
                count=qa_count,
                max_rounds=max_rounds,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            ),
            "multiple_choice": ModeCfg(
                count=mcq_count,
                max_rounds=max_rounds,
                difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3},
            ),
        },
    )


# ─── Fake 模型客户端 ─────────────────────────────────────────────────────────
class FakeModelClient:
    """返回可通过 LightweightFilter 的合法 QA/MCQ JSON。"""

    provider = "fake"
    model_name = "fake-e2e"
    temperature = 0.7
    max_tokens = 2000
    call_count = 0

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        self.call_count += 1
        content = messages[-1].get("content", "") if messages else ""

        if "<final_summary>" in content:
            text = "<final_summary>An overview of the topic with key concepts.</final_summary>"
        elif "multiple_choice" in content.lower():
            text = json.dumps([
                {
                    "question": "Which of the following best describes the topic?",
                    "choices": [
                        "(A) An approach based on core principles",
                        "(B) A completely unrelated method",
                        "(C) No relevant information available",
                        "(D) Only historical context matters",
                    ],
                    "answer": "A",
                    "question_mode": "multiple_choice",
                    "thought_process": "Tests understanding of the core concept.",
                    "question_type": "conceptual",
                    "required_capability": "understanding core principles",
                    "estimated_difficulty": 3,
                    "citations": ["The text describes an approach based on core principles."],
                },
                {
                    "question": "What is a key characteristic of the topic according to the evidence?",
                    "choices": [
                        "(A) It emphasizes fundamental methodology",
                        "(B) It ignores practical applications",
                        "(C) It has no identifiable characteristics",
                        "(D) Only theoretical aspects are relevant",
                    ],
                    "answer": "A",
                    "question_mode": "multiple_choice",
                    "thought_process": "Requires extracting key characteristics from evidence.",
                    "question_type": "analytical",
                    "required_capability": "extracting key characteristics",
                    "estimated_difficulty": 5,
                    "citations": ["The topic emphasizes fundamental methodology."],
                },
            ])
        else:
            text = json.dumps([
                {
                    "question": "What is the primary focus of this topic according to the evidence?",
                    "answer": "It focuses on the application of core principles.",
                    "question_mode": "qa",
                    "thought_process": "Tests basic comprehension of the topic.",
                    "question_type": "factual",
                    "required_capability": "understanding core principles",
                    "estimated_difficulty": 2,
                    "citations": ["The text discusses the application of core principles."],
                },
                {
                    "question": "How does the methodology relate to practical outcomes?",
                    "answer": "The methodology enables practical applications by establishing foundational principles.",
                    "question_mode": "qa",
                    "thought_process": "Tests understanding of methodology-outcome relationships.",
                    "question_type": "analytical",
                    "required_capability": "analyzing methodology",
                    "estimated_difficulty": 6,
                    "citations": [
                        "The methodology enables practical applications.",
                        "Foundational principles are established first.",
                    ],
                },
            ])
        return {
            "text": text,
            "input_tokens": 100,
            "output_tokens": 150,
            "latency": 0.01,
            "raw": {},
            "llm_call_id": f"fake_{self.call_count:04d}",
        }


# ─── 主逻辑 ──────────────────────────────────────────────────────────────────

async def run_fake():
    """使用 fake 模型运行——不需要 API key，不需要网络。"""
    from benchforge.agents.qa_agent import run_generation_agent
    import benchforge.agents.qa_agent.agent as agent_mod
    import benchforge.agents.qa_agent.evidence_manager as evidence_mod
    from benchforge.schemas import SourceDocument, DocumentStatus

    logger.info("=" * 60)
    logger.info("qa_agent E2E — Fake Model")
    logger.info("=" * 60)

    blueprint = build_blueprint(qa_count=4, mcq_count=2, max_rounds=2)

    # ── 注入 fake 组件 ──
    client = FakeModelClient()

    # 模型解析 hook
    agent_mod._resolve_model_client_fn = lambda name, path: client

    # Wikipedia 检索 mock
    FAKE_DOC_CONTENT = (
        "Python is a high-level programming language. "
        "It emphasizes code readability with significant whitespace. "
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data without explicit programming. "
        + "Additional context for chunking. " * 80
    )

    class _FakeSearchResult:
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        title = "Python (programming language)"

    evidence_mod.search_wikipedia = lambda query, **kwargs: [_FakeSearchResult()]
    evidence_mod.fetch_wikipedia_page = lambda result, run_id, **kwargs: SourceDocument(
        document_id="doc_fake_001",
        run_id=run_id,
        topic="Python",
        language="en",
        title="Python (programming language)",
        url=_FakeSearchResult.url,
        summary="Python is a high-level, general-purpose programming language.",
        content=FAKE_DOC_CONTENT,
        status=DocumentStatus.FETCHED,
    )

    # ── 调用 ──
    config_path = project_root / "config" / "qa_agent.yaml"
    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=str(config_path),
    )

    _print_report(report)
    return report


async def run_real():
    """使用真实模型运行——需要 .env 中配置 CUSTOM_API_KEY 和 CUSTOM_API_BASE_URL。"""
    from benchforge.agents.qa_agent import run_generation_agent

    # 检查 API key
    env_path = project_root / ".env"
    has_key = False
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("CUSTOM_API_KEY="):
                    val = line.strip().split("=", 1)[1]
                    if val:
                        has_key = True
                        break
    if not has_key:
        has_key = bool(os.getenv("CUSTOM_API_KEY"))

    if not has_key:
        logger.error("CUSTOM_API_KEY not set in .env or environment")
        logger.error(f"  Create {project_root / '.env'} with:")
        logger.error("  CUSTOM_API_KEY=your-key")
        logger.error("  CUSTOM_API_BASE_URL=https://your-endpoint/v1")
        return None

    logger.info("=" * 60)
    logger.info("qa_agent E2E — Real Model (deepseek-v3)")
    logger.info("=" * 60)

    blueprint = build_blueprint(qa_count=10, mcq_count=10)

    config_path = project_root / "config" / "qa_agent.yaml"
    report = await run_generation_agent(
        blueprint=blueprint,
        config_path=str(config_path),
    )

    _print_report(report)
    return report


def _print_report(report: dict):
    run_dir = Path("runs") / report["task_id"] / report["run_id"]

    print(f"\n{'=' * 60}")
    print(f"Generation Report")
    print(f"{'=' * 60}")
    print(f"  task_id : {report['task_id']}")
    print(f"  run_id  : {report['run_id']}")

    for mode, s in report["modes"].items():
        print(
            f"  {mode:20s}: {s['candidate_count']:>3d}/{s['target_candidate_count']:<3d}"
            f"  stopped={s['stopped_reason']}"
        )
    print(f"  {'total':20s}: {report['total_candidates']:>3d} candidates")
    print(f"  {'chunk combos':20s}: {report['global_used_chunk_combinations']}")
    print(f"\n  Output: {run_dir}")

    # 产物清单
    if run_dir.exists():
        print(f"\n  Artifacts:")
        for f in sorted(run_dir.rglob("*")):
            if f.is_file():
                print(f"    {f.relative_to(run_dir)}")


async def main():
    # use_real = "real" in sys.argv[1:]
    use_real = True
    if use_real:
        report = await run_real()
    else:
        report = await run_fake()

    if report:
        print(f"\nDone. task_id={report['task_id']}, run_id={report['run_id']}")


if __name__ == "__main__":
    asyncio.run(main())
