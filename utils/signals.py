"""信号计算模块：评估文本块质量的多维信号。"""

import re
from typing import Any

import tiktoken


class SignalCalculator:
    """文本质量信号计算器。"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def calculate_all_scores(self, text: str, document_summary: str = "") -> dict[str, float]:
        """综合质量评分，返回 mcq_score / qa_score / hard_score。"""
        if not text.strip():
            return {"mcq_score": 0.0, "qa_score": 0.0, "hard_score": 0.0}
        tokens = self._token_count(text)
        # 信息密度基础分
        if 100 <= tokens <= 800:
            density = 0.8
        elif tokens < 100:
            density = tokens / 100 * 0.8
        else:
            density = max(0.2, 0.8 - (tokens - 800) / 2000)
        # 与摘要的对齐分
        alignment = self.calculate_summary_alignment_signal(text, document_summary) if document_summary else 0.5
        # 文本长度分
        length_score = self.calculate_length_signal(text)
        # 事实密度（实体数量）
        entities = len(re.findall(r'[A-Z][a-z]+', text))
        fact_density = min(1.0, max(0.1, entities / max(1, tokens / 100)))
        # MCQ 分数：侧重信息密度和事实
        mcq_score = round(0.35 * density + 0.25 * fact_density + 0.25 * alignment + 0.15 * length_score, 3)
        # QA 分数：侧重对齐和完整性
        qa_score = round(0.30 * density + 0.20 * fact_density + 0.35 * alignment + 0.15 * length_score, 3)
        # Hard 分数：对齐程度低 + 多实体 = 更适合 hard 题
        hard_score = round(0.25 * density + 0.30 * fact_density + 0.15 * alignment + 0.30 * (1 - length_score), 3)
        return {"mcq_score": mcq_score, "qa_score": qa_score, "hard_score": hard_score}

    def calculate_summary_alignment_signal(
        self, text: str, document_summary: str
    ) -> float:
        """计算文本与文档摘要的对齐程度（0-1）。"""
        if not text.strip() or not document_summary.strip():
            return 0.5
        text_words = set(text.lower().split())
        summary_words = set(document_summary.lower().split())
        if not summary_words:
            return 0.5
        overlap = len(text_words & summary_words)
        return round(min(1.0, overlap / max(1, len(summary_words))), 3)

    def calculate_length_signal(self, text: str) -> float:
        """计算文本长度适宜度（0-1）。理想范围 100-600 tokens。"""
        if not text.strip():
            return 0.0
        tokens = self._token_count(text)
        if 100 <= tokens <= 600:
            return 1.0
        elif tokens < 100:
            return round(tokens / 100, 3)
        else:
            return round(max(0.2, 1.0 - (tokens - 600) / 2000), 3)

    def calculate_tags(self, text: str) -> list[str]:
        """从文本中提取标签（关键词、领域标识）。"""
        tags: list[str] = []
        if not text.strip():
            return tags
        tokens = self._token_count(text)
        if tokens < 100:
            tags.append("short")
        elif tokens <= 500:
            tags.append("medium")
        else:
            tags.append("long")
        # 检查是否包含数字/数据
        if re.search(r'\d+', text):
            tags.append("numeric")
        # 检查是否包含引用
        if re.search(r'(?:cited|reference|according to|et al\.)', text, re.IGNORECASE):
            tags.append("scholarly")
        return tags
