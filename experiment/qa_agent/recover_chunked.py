"""恢复 A_direct 运行的 chunked.json。

问题：direct_generation.py 使用 EvidenceManager 检索并分块了文档，但从未将 chunk
索引持久化到 evidence/chunked.json。verify_agent 的 citation validation 依赖该文件
来水合 candidate.chunks 字段。缺失导致 chunk_score 恒为 0，所有 candidate 在 citation
阶段被淘汰。

恢复策略：
  - 对相同 topics 重新执行 Wikipedia 搜索 + 文档分块
  - document_id 基于 URL 的 SHA256 hash，相同 URL → 相同 document_id
  - saliency rerank 基于历史数据（2022-2025），搜索结果应当稳定
  - Wikipedia 摘要来自页面元数据，无需 LLM 调用

用法：
  python experiment/qa_agent/recover_chunked.py

风险：
  - Wikipedia 页面内容可能已更新，chunk 文本可能不完全一致
  - 搜索 API 返回结果可能有变化，导致 document_id 不匹配
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# 确保项目根在 sys.path 中（与 verify_all.py 一致）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # benchforge/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))  # project/

from loguru import logger

from agents.qa_agent.config_loader import load_qa_agent_config
from agents.qa_agent.evidence_manager import EvidenceManager
from utils.artifact_store import ArtifactStore


# ── 配置 ──────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "configs" / "qa_agent_a_direct.yaml"
RUNS_BASE = PROJECT_ROOT / "runs" / "qa_agent_exp" / "seed_42"


def find_runs_needing_recovery(base: Path) -> list[Path]:
    """找到所有缺少 evidence/chunked.json 的 A_direct 运行目录。"""
    candidates = sorted(base.glob("A_direct_*"))
    needing: list[Path] = []
    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        chunked = run_dir / "evidence" / "chunked.json"
        if chunked.exists():
            logger.info(f"SKIP (already exists): {run_dir.name}")
            continue
        # 确认有 candidate_pool.json（说明生成阶段已完成）
        has_candidates = any(run_dir.glob("*/candidate_pool.json"))
        if has_candidates:
            needing.append(run_dir)
        else:
            logger.info(f"SKIP (no candidates): {run_dir.name}")
    return needing


async def recover_single_run(run_dir: Path) -> bool:
    """对单个 run 重建 chunked.json。"""
    run_name = run_dir.name
    logger.info(f"--- Recovering: {run_name} ---")

    # 1. 读取 metadata
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"metadata.json not found in {run_dir}")
        return False

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    topics = metadata.get("topics", [])
    language = metadata.get("language", "en")
    run_id = metadata.get("run_id", run_name)
    task_id = metadata.get("task_id", "")

    logger.info(f"  topics={topics}  language={language}")

    # 2. 加载配置
    agent_config, _model_ref, retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg = \
        load_qa_agent_config(CONFIG_PATH)

    # 3. 构造 EvidenceManager（与 direct_generation.py 完全一致）
    class _EvidenceConfig:
        __slots__ = ("retrieval", "chunking", "summarization_chunking", "multi_chunk")

        def __init__(self, retrieval, chunking, summarization_chunking, multi_chunk):
            self.retrieval = retrieval
            self.chunking = chunking
            self.summarization_chunking = summarization_chunking
            self.multi_chunk = multi_chunk

        def get_resolved_output_path(self) -> Path:
            return run_dir

    evidence_config = _EvidenceConfig(retrieval_cfg, chunking_cfg, sum_chunking_cfg, multi_chunk_cfg)
    # model_client=None：Wikipedia 摘要来自页面元数据，无需 LLM
    evidence_mgr = EvidenceManager(evidence_config, model_client=None)

    # 4. 构造 minimal plan 对象（prepare_evidence 只需 .language 和 .run_id）
    plan = SimpleNamespace(language=language, run_id=run_id)

    # 5. 并行检索所有 topic
    async def _prepare_topic(topic: str):
        return topic, await evidence_mgr.prepare_evidence(topic, plan)

    logger.info(f"  Retrieving evidence for {len(topics)} topics...")
    topic_evidence = await asyncio.gather(*[_prepare_topic(t) for t in topics])

    # 6. 构建 chunked_rows_all（与 agent.py 完全一致的格式）
    chunked_rows_all: list[dict] = []
    for topic, (chunks, evidence_pool) in topic_evidence:
        evidence_mgr.evidence_pools[topic] = evidence_pool

        chunks_by_doc: dict[str, list] = {}
        for chunk in chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_chunks_sorted = sorted(doc_chunks, key=lambda c: c.chunk_index)
            source_doc = evidence_mgr.documents.get(doc_id)
            chunked_rows_all.append({
                "document_id": doc_id,
                "topic": topic,
                "document_title": source_doc.title if source_doc else "",
                "document_url": source_doc.url if source_doc else "",
                "document_text": source_doc.content if source_doc else "",
                "document_summary": evidence_mgr.document_summaries.get(doc_id, ""),
                "source": "initial_retrieval",
                "chunks": [
                    {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "chunk_text": c.text}
                    for c in doc_chunks_sorted
                ],
            })

        logger.info(f"  [{topic}] {len(chunks_by_doc)} docs, {len(chunks)} chunks")

    # 7. 保存 chunked.json
    if not chunked_rows_all:
        logger.warning(f"  No evidence retrieved for {run_name}!")
        return False

    chunked_by_topic: dict[str, list] = {}
    for row in chunked_rows_all:
        t = row.get("topic", "unknown")
        chunked_by_topic.setdefault(t, []).append(row)

    evidence_store = ArtifactStore(str(run_dir / "evidence"))
    evidence_store.save_json("chunked.json", chunked_by_topic)

    total_docs = len(chunked_rows_all)
    total_chunks = sum(len(row["chunks"]) for row in chunked_rows_all)
    logger.info(f"  Saved: {total_docs} docs, {total_chunks} chunks → {run_dir / 'evidence' / 'chunked.json'}")

    # 8. 验证：检查 candidate 的 document_id 能在 chunked.json 中找到
    matched = 0
    missing = 0
    missing_ids: set[str] = set()
    for mode_dir_name in ["qa", "multiple_choice"]:
        cp = run_dir / mode_dir_name / "candidate_pool.json"
        if cp.exists():
            with open(cp, encoding="utf-8") as f:
                candidates = json.load(f)
            all_doc_ids = {row["document_id"] for row in chunked_rows_all}
            for c in candidates:
                doc_id = c.get("document_id", "")
                if doc_id in all_doc_ids:
                    matched += 1
                else:
                    missing += 1
                    missing_ids.add(doc_id)

    if missing > 0:
        logger.warning(
            f"  Document ID mismatch: {matched} matched, {missing} missing "
            f"(missing doc_ids: {list(missing_ids)[:5]}...)"
        )
    else:
        logger.info(f"  All {matched} candidate document_ids matched!")

    return True


async def main():
    runs = find_runs_needing_recovery(RUNS_BASE)
    if not runs:
        logger.info("No runs need recovery.")
        return

    logger.info(f"Found {len(runs)} run(s) needing recovery:")
    for r in runs:
        logger.info(f"  - {r.name}")

    success = 0
    for run_dir in runs:
        try:
            ok = await recover_single_run(run_dir)
            if ok:
                success += 1
        except Exception:
            logger.exception(f"Failed to recover {run_dir.name}")

    logger.info(f"Done: {success}/{len(runs)} succeeded")


if __name__ == "__main__":
    asyncio.run(main())
