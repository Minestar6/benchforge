"""仅运行检索步骤：预设 50 个主题，每主题 saliency top-4，保存全部文档。

输出目录：runs/retrieval_collect/<run_id>/
  topics.json       — 50 个主题
  documents.jsonl   — 全部抓取文档（每行一个 SourceDocument）
  summary.json      — 统计摘要
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from benchforge.agents.qa_agent.config_loader import load_qa_agent_config
from benchforge.utils import fetch_wikipedia_page, search_wikipedia

# ── 50 个预设主题 ──────────────────────────────────────────────────────────────

TOPICS = [
    # Natural Science
    "Quantum Mechanics", "Black Hole", "CRISPR", "Dark Matter", "Photosynthesis",
    "Plate Tectonics", "Superconductivity", "Mitosis", "Periodic Table", "General Relativity",
    # Technology
    "Artificial Intelligence", "Blockchain", "Nuclear Fusion", "Semiconductor", "Internet of Things",
    "Large Language Model", "Reinforcement Learning", "Computer Vision", "Quantum Computing", "Robotics",
    # History & Civilization
    "Byzantine Empire", "Silk Road", "French Revolution", "Roman Republic", "Renaissance",
    "World War II", "Industrial Revolution", "Mongol Empire", "Ancient Egypt", "Cold War",
    # Medicine & Biology
    "Human Genome Project", "Vaccine", "Alzheimer's Disease", "Antibiotic Resistance",
    "Stem Cell", "Cancer Immunotherapy", "COVID-19", "Human Microbiome", "Epigenetics", "Malaria",
    # Society & Economy
    "Globalization", "Climate Change", "Cryptocurrency",
    "Social Media", "Renewable Energy",
    # Arts & Culture
    "Jazz", "Impressionism", "Shakespeare", "Buddhism", "Olympic Games",
]


# ── 单主题检索 ────────────────────────────────────────────────────────────────

async def retrieve_topic(
    topic: str,
    retrieval_cfg,
    run_id: str,
    language: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async with semaphore:
        loop = asyncio.get_event_loop()

        # search_wikipedia 是同步 IO，放进线程池避免阻塞事件循环
        results = await loop.run_in_executor(
            None,
            lambda: search_wikipedia(
                query=topic,
                language=language,
                max_pages=retrieval_cfg.max_pages,
                request_timeout=retrieval_cfg.request_timeout,
                saliency_rerank=retrieval_cfg.saliency_rerank,
                saliency_top_k=retrieval_cfg.saliency_top_k,
                saliency_start_date=retrieval_cfg.saliency_start_date,
                saliency_end_date=retrieval_cfg.saliency_end_date,
            ),
        )

        docs = []
        for result in results:
            doc = await loop.run_in_executor(
                None,
                lambda r=result: fetch_wikipedia_page(
                    result=r,
                    run_id=run_id,
                    language=language,
                    request_timeout=retrieval_cfg.request_timeout,
                    min_paragraph_tokens=retrieval_cfg.min_paragraph_tokens,
                ),
            )
            if doc.status.value != "failed":
                docs.append(doc)
                logger.info(f"[{topic}] fetched: {doc.title} ({len(doc.content)} chars)")
            else:
                logger.warning(f"[{topic}] fetch failed: {result.url}")

        return docs


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def main() -> None:
    config_path = project_root / "benchforge/config/qa_agent.yaml"
    _, _, retrieval_cfg, _, _ = load_qa_agent_config(config_path)

    # 强制 top-k=4
    retrieval_cfg.saliency_top_k = 4
    retrieval_cfg.saliency_rerank = True

    language = "en"
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path("runs/retrieval_collect") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.add(str(out_dir / "run.log"), level="DEBUG", encoding="utf-8")
    logger.info(f"Output: {out_dir}")

    topics = TOPICS
    logger.info(f"Using {len(topics)} predefined topics")

    topics_path = out_dir / "topics.json"
    topics_path.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved topics → {topics_path}")

    # Step 2：并发检索，每主题 top-4
    # 并发度 2：Wikipedia 限流较严，控制请求速率
    semaphore = asyncio.Semaphore(2)
    tasks = [
        retrieve_topic(
            topic=t,
            retrieval_cfg=retrieval_cfg,
            run_id=run_id,
            language=language,
            semaphore=semaphore,
        )
        for t in topics
    ]

    logger.info(f"Retrieving documents for {len(topics)} topics (concurrency=5)...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Step 3：去重并保存
    seen_urls: set[str] = set()
    all_docs = []
    failed_topics = []

    for topic, result in zip(topics, results):
        if isinstance(result, Exception):
            logger.error(f"Topic '{topic}' failed: {result}")
            failed_topics.append(topic)
            continue
        for doc in result:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                all_docs.append(doc)

    docs_path = out_dir / "documents.jsonl"
    with open(docs_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(doc.model_dump_json() + "\n")
    logger.info(f"Saved {len(all_docs)} documents → {docs_path}")

    # Step 4：统计摘要
    summary = {
        "run_id": run_id,
        "topics_count": len(topics),
        "documents_count": len(all_docs),
        "failed_topics": failed_topics,
        "saliency_top_k": retrieval_cfg.saliency_top_k,
        "language": language,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Retrieval Collect ===")
    print(f"topics      : {len(topics)}")
    print(f"documents   : {len(all_docs)}")
    print(f"failed      : {len(failed_topics)}")
    print(f"output      : {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
