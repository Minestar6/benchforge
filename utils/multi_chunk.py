"""多证据单元构建模块。"""

import hashlib
from typing import Any

import numpy as np
import tiktoken

from benchforge.schemas import (
    SourceChunk,
    SingleChunkUnit,
    MultiChunkUnit,
    TopicStatus,
)
from benchforge.utils.signals import SignalCalculator


class MultiChunkBuilder:
    """多证据单元构建器。"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.signal_calculator = SignalCalculator(encoding_name)

    def build_multi_chunk_units(
        self,
        single_chunks: list[SingleChunkUnit],
        max_chunk_distance: int = 3,
        max_total_tokens: int = 1500,
        min_chunk_count: int = 2,
        max_chunk_count: int = 3,
    ) -> list[MultiChunkUnit]:
        """相邻滑动窗口构建（保留作备用）。"""
        if not single_chunks:
            return []

        chunks_by_doc: dict[str, list[SingleChunkUnit]] = {}
        for chunk in single_chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        multi_chunks: list[MultiChunkUnit] = []
        for doc_id, doc_chunks in chunks_by_doc.items():
            sorted_chunks = sorted(doc_chunks, key=lambda x: self._extract_chunk_index(x.chunk_id))

            for i in range(len(sorted_chunks)):
                for window_size in range(min_chunk_count, max_chunk_count + 1):
                    end = i + window_size
                    if end > len(sorted_chunks):
                        continue
                    window = sorted_chunks[i:end]
                    if self._check_distance_constraint(window, max_chunk_distance):
                        mc = self._build_multi_chunk_from_window(window)
                        if mc and self._check_token_constraint(mc, max_total_tokens):
                            multi_chunks.append(mc)

        return multi_chunks

    def build_multi_chunk_units_smart(
        self,
        single_chunks: list[SingleChunkUnit],
        document_summaries: dict[str, str],
        target_count: int = 10,
        h_min: int = 2,
        h_max: int = 4,
        combinations_per_doc_factor: int = 2,
    ) -> list[MultiChunkUnit]:
        """YourBench 风格：基于文档哈希的确定性随机采样，支持非相邻 chunk 组合。

        Args:
            single_chunks: 单证据单元列表
            document_summaries: 文档摘要 {document_id: summary}
            target_count: 最终保留的 multi-chunk 单元数量上限
            h_min: 每个组合最少 chunk 数（含）
            h_max: 每个组合最多 chunk 数（含）
            combinations_per_doc_factor: 每篇文档生成的候选组合数 = n_chunks // factor
        """
        if not single_chunks:
            return []

        chunks_by_doc: dict[str, list[SingleChunkUnit]] = {}
        for chunk in single_chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        candidates: list[tuple[MultiChunkUnit, float]] = []

        for doc_id, doc_chunks in chunks_by_doc.items():
            n = len(doc_chunks)
            if n < h_min:
                continue

            sorted_chunks = sorted(doc_chunks, key=lambda x: self._extract_chunk_index(x.chunk_id))

            # 基于 doc_id 生成确定性种子（对齐 YourBench）
            seed = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % (2 ** 31)
            rng = np.random.default_rng(seed)

            n_combinations = max(1, int(n * combinations_per_doc_factor))
            seen: set[tuple[int, ...]] = set()

            for _ in range(n_combinations * 10):  # 超采样后去重
                if len(seen) >= n_combinations:
                    break
                h = int(rng.integers(h_min, min(h_max, n) + 1))
                indices = tuple(sorted(rng.choice(n, size=h, replace=False).tolist()))
                if indices in seen:
                    continue
                seen.add(indices)

                window = [sorted_chunks[i] for i in indices]
                mc = self._build_multi_chunk_from_window(window)
                if mc is None:
                    continue

                summary = document_summaries.get(doc_id, "")
                score = self._calculate_multi_chunk_score(mc, summary, prioritize_hard_score=True)
                candidates.append((mc, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [mc for mc, _ in candidates[:target_count]]

    def _build_multi_chunk_from_window(
        self,
        chunks: list[SingleChunkUnit],
    ) -> MultiChunkUnit | None:
        if not chunks:
            return None

        combined_text = " ".join(chunk.text for chunk in chunks)
        all_tags: set[str] = set()
        for chunk in chunks:
            all_tags.update(chunk.tags)

        doc_id = chunks[0].document_id
        topic = chunks[0].topic
        scores = self.signal_calculator.calculate_all_scores(combined_text)

        return MultiChunkUnit(
            document_id=doc_id,
            topic=topic,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            texts=[chunk.text for chunk in chunks],
            tags=list(all_tags),
            mcq_score=scores["mcq_score"],
            qa_score=scores["qa_score"],
            hard_score=scores["hard_score"],
        )

    def _calculate_multi_chunk_score(
        self,
        multi_chunk: MultiChunkUnit,
        document_summary: str,
        prioritize_hard_score: bool,
    ) -> float:
        combined_text = " ".join(multi_chunk.texts)
        base_score = multi_chunk.hard_score if prioritize_hard_score else (
            multi_chunk.mcq_score + multi_chunk.qa_score
        ) / 2
        alignment = self.signal_calculator.calculate_summary_alignment_signal(combined_text, document_summary)
        length_signal = self.signal_calculator.calculate_length_signal(combined_text)
        return 0.6 * base_score + 0.25 * alignment + 0.15 * length_signal

    def _check_distance_constraint(self, chunks: list[SingleChunkUnit], max_distance: int) -> bool:
        indices = [self._extract_chunk_index(c.chunk_id) for c in chunks]
        return max(indices) - min(indices) <= max_distance

    def _check_token_constraint(self, multi_chunk: MultiChunkUnit, max_tokens: int) -> bool:
        combined_text = " ".join(multi_chunk.texts)
        tokens = self.encoding.encode(combined_text, disallowed_special=())
        return len(tokens) <= max_tokens

    @staticmethod
    def _extract_chunk_index(chunk_id: str) -> int:
        try:
            if "::chunk_" in chunk_id:
                return int(chunk_id.split("chunk_")[-1])
            import re
            match = re.search(r'(\d+)$', chunk_id)
            return int(match.group(1)) if match else 0
        except (ValueError, AttributeError):
            return 0

    def calculate_yourbench_target_count(
        self,
        single_chunks: list[SingleChunkUnit],
        h_min: int = 2,
        h_max: int = 5,
        num_multihops_factor: int = 1,
        max_units: int | None = None,
    ) -> int:
        """按 YourBench 风格计算 multi-chunk 目标数量。

        设计原则：
        - 不枚举所有组合，按每篇文档的 chunk 数生成受控数量
        - 最终使用 max_units 控制全局上限
        """
        if not single_chunks:
            return 0

        chunks_by_doc: dict[str, list[SingleChunkUnit]] = {}
        for chunk in single_chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

        total = 0
        factor = max(1, int(num_multihops_factor))

        for _doc_id, doc_chunks in chunks_by_doc.items():
            n = len(doc_chunks)
            if n < h_min:
                continue

            effective_h_max = min(h_max, n)

            # YourBench 风格基础数量：按文档 chunk 数缩放
            doc_target = max(1, n // factor)

            # 防止每组过大时生成过多无效组合
            if doc_target * effective_h_max > n:
                doc_target = max(1, n // effective_h_max)

            total += doc_target

        if max_units is not None:
            total = min(total, int(max_units))

        return max(0, total)


def build_evidence_pool_from_chunks(
    chunks: list[SourceChunk],
    topic: str,
    document_summary: str = "",
) -> list[SingleChunkUnit]:
    """从源 chunks 构建单证据单元。"""
    calculator = SignalCalculator()

    units = []
    for chunk in chunks:
        scores = calculator.calculate_all_scores(chunk.text, document_summary)
        tags = calculator.calculate_tags(chunk.text)

        unit = SingleChunkUnit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            topic=topic,
            text=chunk.text,
            tags=tags,
            mcq_score=scores["mcq_score"],
            qa_score=scores["qa_score"],
            hard_score=scores["hard_score"],
        )
        units.append(unit)

    return units


