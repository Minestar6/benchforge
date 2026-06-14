"""EvidenceManager 模块：检索、分块、证据池、采样。"""

import json
from pathlib import Path
import re
from typing import Any
from dataclasses import dataclass

from loguru import logger

from benchforge.utils import (
    search_wikipedia,
    fetch_wikipedia_page,
    chunk_document,
    get_document_source,
)
from benchforge.utils.multi_chunk import (
    MultiChunkBuilder,
    build_evidence_pool_from_chunks,
)
from benchforge.utils.sampling import (
    BroadExplorationSampling,
    GapDrivenSampling,
)
from benchforge.utils.planning import format_evidence_texts


@dataclass
class ExpandResult:
    """扩展检索结果。"""
    new_chunks: int
    new_single_units: int
    new_multi_units: int


class EvidenceManager:
    """证据管理器。

    职责：
    - 检索主题相关文档
    - 生成文档摘要
    - 切分 chunk
    - 构建证据池
    - 采样证据（支持单/多 chunk）
    - 扩展检索
    """

    def __init__(self, config: Any, model_client: Any | None = None):
        """初始化证据管理器。

        Args:
            config: 配置对象
            model_client: 模型客户端（用于生成文档摘要）
        """
        self.config = config
        self.model_client = model_client
        self.multi_chunk_builder = MultiChunkBuilder()
        self.document_summaries: dict[str, str] = {}
        self.documents: dict[str, Any] = {}  # document_id -> SourceDocument
        self.retrieved_urls: set[str] = set()
        self.evidence_pools: dict[str, Any] = {}

    def _llm_trace_path(self) -> str | None:
        if hasattr(self.config, "get_resolved_output_path"):
            return str(self.config.get_resolved_output_path() / "llm_calls.jsonl")
        return None

    def _multi_chunk_cfg(self) -> Any:
        return getattr(self.config, "multi_chunk", None)

    def _resolve_multi_target_count(
        self,
        single_units: list[Any],
        *,
        expansion: bool = False,
    ) -> int:
        mcfg = self._multi_chunk_cfg()

        h_min = getattr(mcfg, "h_min", 2)
        h_max = getattr(mcfg, "h_max", 5)
        num_multihops_factor = getattr(mcfg, "num_multihops_factor", 1)

        max_units = getattr(
            mcfg,
            "max_units_per_expansion" if expansion else "max_units_per_topic",
            40 if expansion else 80,
        )

        return self.multi_chunk_builder.calculate_yourbench_target_count(
            single_chunks=single_units,
            h_min=h_min,
            h_max=h_max,
            num_multihops_factor=num_multihops_factor,
            max_units=max_units,
        )

    def _build_multi_units(
        self,
        single_units: list[Any],
        doc_summaries: dict[str, str],
        *,
        expansion: bool = False,
    ) -> list[Any]:
        mcfg = self._multi_chunk_cfg()

        h_min = getattr(mcfg, "h_min", 2)
        h_max = getattr(mcfg, "h_max", 5)
        combinations_per_doc_factor = getattr(mcfg, "combinations_per_doc_factor", 1)

        target_count = self._resolve_multi_target_count(
            single_units,
            expansion=expansion,
        )

        return self.multi_chunk_builder.build_multi_chunk_units_smart(
            single_units,
            doc_summaries,
            target_count=target_count,
            h_min=h_min,
            h_max=h_max,
            combinations_per_doc_factor=combinations_per_doc_factor,
        )

    async def prepare_evidence(
        self,
        topic: str,
        plan: Any,
    ) -> tuple[list[Any], Any]:
        """准备证据池。

        步骤：
        1. 检索主题相关文档
        2. 生成文档摘要
        3. 切分 chunk
        4. 构建证据池

        Args:
            topic: 主题名称
            plan: 生成计划

        Returns:
            (chunk 列表, 证据池)
        """
        # 新主题，重置组合历史和 URL 记录
        self.retrieved_urls.clear()

        # 检索文档（可选 saliency 重排：按 Wikimedia 访问量筛选最显著页面）
        _ret = self.config.retrieval
        search_results = search_wikipedia(
            query=topic,
            language=plan.language,
            max_pages=_ret.max_pages,
            request_timeout=_ret.request_timeout,
            saliency_rerank=_ret.saliency_rerank,
            saliency_top_k=_ret.saliency_top_k,
            saliency_start_date=_ret.saliency_start_date,
            saliency_end_date=_ret.saliency_end_date,
        )

        if not search_results:
            return [], None

        # 抓取并处理文档
        all_chunks: list[Any] = []

        for result in search_results:
            # 去重检查：跳过已抓取的 URL
            if result.url in self.retrieved_urls:
                logger.debug(f"Skipping duplicate URL (initial): {result.url}")
                continue

            # 记录已抓取的 URL
            self.retrieved_urls.add(result.url)

            document = fetch_wikipedia_page(
                result=result,
                run_id=plan.run_id,
                language=plan.language,
                request_timeout=_ret.request_timeout,
                min_paragraph_tokens=_ret.min_paragraph_tokens,
            )

            if document.status.value == "failed":
                continue

            self.documents[document.document_id] = document

            # Wikipedia 且导言段非空：直接用，否则 LLM 生成
            if get_document_source(document.url) == "wikipedia" and document.summary:
                self.document_summaries[document.document_id] = document.summary
            else:
                self.document_summaries[document.document_id] = (
                    await self._generate_document_summary(document)
                )

            # 题目生成分块（小 chunk）
            chunks = chunk_document(
                document=document,
                chunk_size=self.config.chunking.chunk_size,
                overlap=self.config.chunking.overlap,
            )

            all_chunks.extend(chunks)

        # 构建证据池
        evidence_pool = self._build_evidence_pool(topic, all_chunks)

        return all_chunks, evidence_pool

    async def _generate_document_summary(
        self,
        document: Any,
    ) -> str:
        """生成文档摘要（使用独立的大 chunk 参数，降低成本）。

        参考 YourBench 实现：
        - 使用独立的总结分段参数（summarization_chunking）
        - 大块减少 LLM 调用次数

        步骤：
        1. 用总结分段参数对文档进行分段
        2. 对每个大 chunk 生成摘要
        3. 如果有多个 chunk，合并摘要

        Args:
            document: 源文档

        Returns:
            文档摘要
        """
        if not document.content:
            return document.summary or ""

        # 使用总结分段参数进行分段（大 chunk，减少调用次数）
        from benchforge.utils import chunk_document
        summarization_chunks = chunk_document(
            document=document,
            chunk_size=self.config.summarization_chunking.chunk_size,
            overlap=self.config.summarization_chunking.overlap,
        )

        if not summarization_chunks:
            return document.summary or ""

        # 如果没有 model_client，使用简化实现
        if not self.model_client:
            summaries = []
            for chunk in summarization_chunks[:2]:
                summaries.append(chunk.text[:300])
            return " | ".join(summaries)

        # Stage 1: 生成 chunk summaries（大 chunk，调用次数少）
        chunk_summaries = []

        for chunk in summarization_chunks:
            try:
                prompt = self._get_summarization_prompt(chunk.text)

                response = await self.model_client.complete(
                    model=getattr(self.model_client, 'model_name', 'gpt-4o'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,  # 增加输出长度，因为输入是更大的 chunk
                    llm_trace_path=self._llm_trace_path(),
                )

                # 提取摘要
                summary_match = re.search(
                    r'<final_summary>(.*?)</final_summary>',
                    response["text"],
                    re.DOTALL | re.IGNORECASE
                )
                if summary_match:
                    summary = summary_match.group(1).strip()
                else:
                    summary = response["text"].strip()[:500]

                chunk_summaries.append(summary)

            except Exception as e:
                logger.warning(f"Failed to summarize chunk {chunk.chunk_id}: {e}")
                chunk_summaries.append("")

        # Stage 2: 如果有多个 chunk，合并摘要
        if len(chunk_summaries) > 1:
            try:
                bullet_list = "\n".join(f"- {s}" for s in chunk_summaries if s)
                combine_prompt = self._get_combine_summaries_prompt(bullet_list)

                response = await self.model_client.complete(
                    model=getattr(self.model_client, 'model_name', 'gpt-4o'),
                    messages=[{"role": "user", "content": combine_prompt}],
                    temperature=0.3,
                    max_tokens=800,  # 合并摘要需要更多 tokens
                    llm_trace_path=self._llm_trace_path(),
                )

                final_match = re.search(
                    r'<final_summary>(.*?)</final_summary>',
                    response["text"],
                    re.DOTALL | re.IGNORECASE
                )
                if final_match:
                    return final_match.group(1).strip()
                else:
                    return response["text"].strip()
            except Exception as e:
                logger.warning(f"Failed to combine summaries: {e}")

        # 如果只有一个 chunk 或合并失败，返回第一个摘要
        return chunk_summaries[0] if chunk_summaries else document.summary or ""

    def _get_summarization_prompt(self, text: str) -> str:
        """获取 summarization prompt（从文件加载）。

        Args:
            text: 文本内容

        Returns:
            prompt
        """
        prompt_path = (
            Path(__file__).parent.parent.parent.parent
            / "prompts" / "question_generator" / "summarization_user_prompt.md"
        )

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            return template.replace("{document}", text)
        else:
            return f"""<document>
{text}
</document>

Provide a concise summary in <final_summary> tags."""

    def _get_combine_summaries_prompt(self, bullet_list: str) -> str:
        """获取合并摘要 prompt（从文件加载）。

        Args:
            bullet_list: chunk 摘要列表

        Returns:
            prompt
        """
        prompt_path = (
            Path(__file__).parent.parent.parent.parent
            / "prompts" / "question_generator" / "combine_summaries_user_prompt.md"
        )

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            return template.replace("{chunk_summaries}", bullet_list)
        else:
            return f"""<chunk_summaries>
{bullet_list}
</chunk_summaries>

Provide a concise overview in <final_summary> tags."""

    def _build_evidence_pool(
        self,
        topic: str,
        chunks: list[Any],
    ) -> Any:
        """构建证据池。

        Args:
            topic: 主题名称
            chunks: chunk 列表

        Returns:
            证据池
        """
        from benchforge.schemas import EvidencePool

        # 按文档分组 chunks
        chunks_by_doc: dict[str, list[Any]] = {}
        for chunk in chunks:
            if chunk.document_id not in chunks_by_doc:
                chunks_by_doc[chunk.document_id] = []
            chunks_by_doc[chunk.document_id].append(chunk)

        # 构建单证据单元
        single_units = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_summary = self.document_summaries.get(doc_id, "")
            doc_units = build_evidence_pool_from_chunks(
                doc_chunks,
                topic,
                document_summary=doc_summary,
            )
            single_units.extend(doc_units)

        # 构建多证据单元
        doc_summaries = {
            doc_id: self.document_summaries.get(doc_id, "")
            for doc_id in chunks_by_doc.keys()
        }

        multi_units = self._build_multi_units(
            single_units,
            doc_summaries,
            expansion=False,
        )

        # 创建证据池
        pool = EvidencePool(
            topic=topic,
            single_chunks=single_units,
            multi_chunks=multi_units,
        )

        return pool

    def sample(
        self,
        evidence_pool: Any,
        topic: str,
        target_mode: str,
        target_difficulty: str,
        prefer_multi_chunk: bool = False,
        round_num: int = 1,
        remaining: int = 1,
    ) -> Any:
        """采样证据，确保 chunk 组合唯一。

        使用采样策略：
        - 第 1 轮：BroadExplorationSampling（广度探索）
        - 其他轮：GapDrivenSampling（缺口驱动）

        Args:
            evidence_pool: 证据池
            topic: 主题名称
            target_mode: 目标模式
            target_difficulty: 目标难度
            prefer_multi_chunk: 是否偏好多 chunk
            round_num: 轮次（用于选择采样策略）
            remaining: 剩余缺口数量

        Returns:
            生成批次
        """
        # 选择采样策略
        if round_num == 1:
            sampler = BroadExplorationSampling()
        else:
            sampler = GapDrivenSampling()

        # 计算请求数量（冗余策略）
        num_evidence = self._calculate_num_evidence(remaining)

        # ???????? sample_chunks + GlobalState ?????
        batch = sampler.sample(
            pool=evidence_pool,
            target_mode=target_mode,
            target_difficulty=target_difficulty,
            num_evidence=num_evidence,
            prefer_multi_chunk=prefer_multi_chunk,
        )
        # 设置请求数量
        min_questions, target_questions = self._calculate_batch_request_counts(remaining)
        batch.requested_min_questions = min_questions
        batch.requested_target_questions = target_questions

        # 更新使用计数
        for unit in evidence_pool.single_chunks:
            if unit.chunk_id in batch.single_chunk_ids:
                unit.usage_count += 1

        for unit in evidence_pool.multi_chunks:
            if unit.unit_id in batch.multi_chunk_ids:
                unit.usage_count += 1

        return batch

    def _calculate_num_evidence(self, remaining_count: int) -> int:
        """计算需要采样的证据数量。

        Args:
            remaining_count: 剩余目标数量

        Returns:
            证据数量
        """
        # 最多 5 个证据单元
        return min(5, remaining_count + 2)

    def _calculate_batch_request_counts(
        self,
        remaining_count: int,
    ) -> tuple[int, int]:
        """计算批次请求的题目数量。

        使用冗余策略而非精确计算。

        Args:
            remaining_count: 剩余目标数量

        Returns:
            (最小请求数, 目标请求数)
        """
        if remaining_count <= 2:
            return remaining_count + 1, remaining_count + 2
        elif remaining_count <= 5:
            return remaining_count + 1, remaining_count + 3
        else:
            return remaining_count + 2, remaining_count + 4

    async def expand_retrieval(
        self,
        topic: str,
        queries: list[str],
        language: str,
        run_id: str,
        evidence_pool: Any,
    ) -> ExpandResult:
        """扩展检索（带去重）。

        Args:
            topic: 主题名称
            queries: 查询列表
            language: 语言
            run_id: 运行 ID
            evidence_pool: 现有证据池（用于扩展）

        Returns:
            扩展结果
        """
        from benchforge.schemas import TopicState

        all_chunks: list[Any] = []
        skipped_urls: list[str] = []

        for query in queries:
            results = search_wikipedia(
                query=query,
                language=language,
                max_pages=2,
            )

            for result in results:
                # 去重检查：跳过已抓取的 URL
                if result.url in self.retrieved_urls:
                    skipped_urls.append(result.url)
                    logger.debug(f"Skipping duplicate URL: {result.url}")
                    continue

                # 记录已抓取的 URL
                self.retrieved_urls.add(result.url)

                document = fetch_wikipedia_page(
                    result=result,
                    run_id=run_id,
                    language=language,
                    request_timeout=self.config.retrieval.request_timeout,
                    min_paragraph_tokens=self.config.retrieval.min_paragraph_tokens,
                )

                if document.status.value == "failed":
                    continue

                self.documents[document.document_id] = document

                # Wikipedia 且导言段非空：直接用，否则 LLM 生成
                if get_document_source(document.url) == "wikipedia" and document.summary:
                    self.document_summaries[document.document_id] = document.summary
                else:
                    self.document_summaries[document.document_id] = (
                        await self._generate_document_summary(document)
                    )

                # 题目生成分块（小 chunk）
                chunks = chunk_document(
                    document=document,
                    chunk_size=self.config.chunking.chunk_size,
                    overlap=self.config.chunking.overlap,
                )

                all_chunks.extend(chunks)

        if skipped_urls:
            logger.info(f"Skipped {len(skipped_urls)} duplicate URLs during expansion")

        # 构建新单元
        single_units = build_evidence_pool_from_chunks(
            all_chunks,
            topic,
            document_summary="",
        )

        # 构建多 chunk 单元
        doc_summaries = {
            chunk.document_id: self.document_summaries.get(chunk.document_id, "")
            for chunk in all_chunks
        }

        multi_units = self._build_multi_units(
            single_units,
            doc_summaries,
            expansion=True,
        )

        # 扩展到现有证据池
        if evidence_pool:
            evidence_pool.single_chunks.extend(single_units)
            evidence_pool.multi_chunks.extend(multi_units)

        self._append_expanded_evidence_to_disk(
            topic=topic,
            chunks=all_chunks,
            single_units=single_units,
            multi_units=multi_units,
        )

        return ExpandResult(
            new_chunks=len(all_chunks),
            new_single_units=len(single_units),
            new_multi_units=len(multi_units),
        )

    def _append_expanded_evidence_to_disk(
        self,
        topic: str,
        chunks: list[Any],
        single_units: list[Any],
        multi_units: list[Any],
    ) -> None:
        """将 EXPAND_EVIDENCE 新增证据持久化到 runs/{task_id}/{run_id}/evidence。"""
        if not hasattr(self.config, "get_resolved_output_path"):
            return

        evidence_dir = Path(self.config.get_resolved_output_path()) / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # 1. append chunked.jsonl
        chunks_by_doc: dict[str, list[Any]] = {}
        for chunk in chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        chunked_rows: list[dict[str, Any]] = []

        for doc_id, doc_chunks in chunks_by_doc.items():
            doc_chunks_sorted = sorted(
                doc_chunks,
                key=lambda c: getattr(c, "chunk_index", 0),
            )
            source_doc = self.documents.get(doc_id)

            chunked_rows.append(
                {
                    "document_id": doc_id,
                    "topic": topic,
                    "document_title": source_doc.title if source_doc else "",
                    "document_url": source_doc.url if source_doc else "",
                    "document_text": source_doc.content if source_doc else "",
                    "document_summary": self.document_summaries.get(doc_id, ""),
                    "chunks": [
                        {
                            "chunk_id": c.chunk_id,
                            "chunk_text": c.text,
                        }
                        for c in doc_chunks_sorted
                    ],
                    "source": "expand_retrieval",
                }
            )

        if chunked_rows:
            with open(evidence_dir / "chunked.jsonl", "a", encoding="utf-8") as f:
                for row in chunked_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 2. merge single_units.json
        single_path = evidence_dir / "single_units.json"
        if single_path.exists():
            with open(single_path, "r", encoding="utf-8") as f:
                single_data = json.load(f)
        else:
            single_data = {}

        single_data.setdefault(topic, [])
        single_data[topic].extend(
            [
                {
                    "chunk_id": u.chunk_id,
                    "document_id": u.document_id,
                    "text": getattr(u, "text", ""),
                    "qa_score": u.qa_score,
                    "mcq_score": u.mcq_score,
                    "hard_score": u.hard_score,
                    "source": "expand_retrieval",
                }
                for u in single_units
            ]
        )

        with open(single_path, "w", encoding="utf-8") as f:
            json.dump(single_data, f, ensure_ascii=False, indent=2)

        # 3. merge multi_units.json
        multi_path = evidence_dir / "multi_units.json"
        if multi_path.exists():
            with open(multi_path, "r", encoding="utf-8") as f:
                multi_data = json.load(f)
        else:
            multi_data = {}

        multi_data.setdefault(topic, [])
        multi_data[topic].extend(
            [
                {
                    "unit_id": u.unit_id,
                    "document_id": u.document_id,
                    "chunk_ids": list(u.chunk_ids),
                    "texts": list(getattr(u, "texts", [])),
                    "qa_score": u.qa_score,
                    "mcq_score": u.mcq_score,
                    "hard_score": u.hard_score,
                    "source": "expand_retrieval",
                }
                for u in multi_units
            ]
        )

        with open(multi_path, "w", encoding="utf-8") as f:
            json.dump(multi_data, f, ensure_ascii=False, indent=2)

    def get_evidence_text(
        self,
        batch: Any,
        evidence_pool: Any,
    ) -> str:
        """获取格式化后的证据文本。

        Args:
            batch: 生成批次
            evidence_pool: 证据池

        Returns:
            格式化后的证据文本
        """
        # 获取证据单元
        single_units = [
            u for u in evidence_pool.single_chunks
            if u.chunk_id in batch.single_chunk_ids
        ]
        multi_units = [
            u for u in evidence_pool.multi_chunks
            if u.unit_id in batch.multi_chunk_ids
        ]

        return format_evidence_texts(single_units, multi_units)

    def get_document_summary(
        self,
        batch: Any,
        evidence_pool: Any,
    ) -> str:
        """获取文档摘要。

        Args:
            batch: 生成批次
            evidence_pool: 证据池

        Returns:
            文档摘要
        """
        single_units = [
            u for u in evidence_pool.single_chunks
            if u.chunk_id in batch.single_chunk_ids
        ]

        if single_units and single_units[0].document_id in self.document_summaries:
            return self.document_summaries[single_units[0].document_id]

        return ""
