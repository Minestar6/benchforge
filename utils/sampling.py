"""采样策略模块（基于设计文档 12）。"""

import random
from typing import Any

import numpy as np

from benchforge.schemas import (
    EvidencePool,
    SingleChunkUnit,
    MultiChunkUnit,
    GenerationBatch,
    QuestionMode,
    Difficulty,
)
from benchforge.utils.signals import SignalCalculator


class SamplingStrategy:
    """采样策略基类。"""

    def __init__(
        self,
        seed: int | None = None,
    ):
        """初始化采样策略。

        Args:
            seed: 随机种子
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample(
        self,
        pool: EvidencePool,
        target_mode: str,
        target_difficulty: str,
        num_evidence: int,
        prefer_multi_chunk: bool = False,
    ) -> GenerationBatch:
        """采样证据单元。

        Args:
            pool: 证据池
            target_mode: 目标模式
            target_difficulty: 目标难度
            num_evidence: 需要的证据单元数量
            prefer_multi_chunk: 是否优先多 chunk

        Returns:
            生成批次
        """
        raise NotImplementedError


class BroadExplorationSampling(SamplingStrategy):
    """广度探索采样。

    用途：
    1. 主题第一轮
    2. 主题内部尚未出现明显主缺口时

    特点：
    1. 进行广覆盖加权采样，不是纯随机
    2. 优先低使用、高基础分数的证据单元
    3. 目标是快速摸清主题内部的证据结构
    """

    def __init__(self, seed: int | None = None):
        """初始化广度探索采样。

        Args:
            seed: 随机种子
        """
        super().__init__(seed)

    def sample(
        self,
        pool: EvidencePool,
        target_mode: str,
        target_difficulty: str,
        num_evidence: int = 5,
        prefer_multi_chunk: bool = False,
    ) -> GenerationBatch:
        """执行广度探索采样。

        Args:
            pool: 证据池
            target_mode: 目标模式
            target_difficulty: 目标难度
            num_evidence: 需要的证据单元数量
            prefer_multi_chunk: 是否优先多 chunk

        Returns:
            生成批次
        """
        # 准备候选单元
        candidates: list[tuple[str, float, str]] = []

        # 单 chunk 候选
        for chunk in pool.single_chunks:
            score = self._calculate_exploration_score(
                chunk,
                target_mode,
                target_difficulty,
            )
            candidates.append((chunk.chunk_id, score, "single"))

        # 多 chunk 候选
        for unit in pool.multi_chunks:
            score = self._calculate_exploration_score(
                unit,
                target_mode,
                target_difficulty,
            )
            candidates.append((unit.unit_id, score, "multi"))

        # 加权采样
        selected_ids: list[str] = []
        selected_single: list[str] = []
        selected_multi: list[str] = []

        if not candidates:
            return GenerationBatch(
                topic=pool.topic,
                target_mode=target_mode,
                target_difficulty=target_difficulty,
                remaining_count=num_evidence,
                single_chunk_ids=[],
                multi_chunk_ids=[],
                prompt_template_id=self._get_template_id(target_mode),
                requested_min_questions=num_evidence,
                requested_target_questions=num_evidence + 2,
            )

        # 获取分数用于加权采样
        ids = [c[0] for c in candidates]
        scores = np.array([c[1] for c in candidates])
        types_map = {c[0]: c[2] for c in candidates}

        # 归一化分数为概率
        if scores.sum() > 0:
            probs = scores / scores.sum()
        else:
            probs = np.ones(len(scores)) / len(scores)

        # 采样（避免重复）
        sampled_indices = np.random.choice(
            len(ids),
            size=min(num_evidence, len(ids)),
            replace=False,
            p=probs,
        )

        # 根据类型分类
        for idx in sampled_indices:
            unit_id = ids[idx]
            unit_type = types_map[unit_id]
            selected_ids.append(unit_id)

            if unit_type == "single":
                selected_single.append(unit_id)
            else:
                selected_multi.append(unit_id)

        # 调整多 chunk 比例
        if prefer_multi_chunk and pool.multi_chunks:
            # 增加多 chunk 比例
            target_multi_ratio = 0.6
            current_multi_ratio = len(selected_multi) / len(selected_ids) if selected_ids else 0

            if current_multi_ratio < target_multi_ratio:
                # 替换一些单 chunk 为多 chunk
                multi_candidates = [
                    (u.unit_id, u.mcq_score if target_mode == "multiple_choice" else u.qa_score)
                    for u in pool.multi_chunks
                    if u.unit_id not in selected_multi
                ]
                if multi_candidates:
                    multi_candidates.sort(key=lambda x: x[1], reverse=True)

                    num_to_replace = int(
                        (target_multi_ratio - current_multi_ratio) * len(selected_ids)
                    )
                    num_to_replace = min(num_to_replace, len(multi_candidates))

                    # 替换
                    for i in range(num_to_replace):
                        if selected_single:
                            selected_single.pop()
                        selected_multi.append(multi_candidates[i][0])

        return GenerationBatch(
            topic=pool.topic,
            target_mode=target_mode,
            target_difficulty=target_difficulty,
            remaining_count=num_evidence,
            single_chunk_ids=selected_single,
            multi_chunk_ids=selected_multi,
            prompt_template_id=self._get_template_id(target_mode),
            requested_min_questions=num_evidence,
            requested_target_questions=num_evidence + 2,
        )

    def _calculate_exploration_score(
        self,
        unit: SingleChunkUnit | MultiChunkUnit,
        target_mode: str,
        target_difficulty: str,
    ) -> float:
        """计算探索分数。

        优先低使用、高基础分数的单元。

        Args:
            unit: 证据单元
            target_mode: 目标模式
            target_difficulty: 目标难度

        Returns:
            探索分数
        """
        # 基础分数
        if target_mode == "multiple_choice":
            base_score = unit.mcq_score
        else:
            base_score = unit.qa_score

        # 难度权重
        if target_difficulty == "hard":
            base_score = base_score * 0.7 + unit.hard_score * 0.3

        # 使用惩罚（越使用越低）
        usage_penalty = 1.0 / (1.0 + unit.usage_count)

        return base_score * usage_penalty

    @staticmethod
    def _get_template_id(target_mode: str) -> str:
        """获取模板 ID。

        Args:
            target_mode: 目标模式

        Returns:
            模板 ID
        """
        if target_mode == "multiple_choice":
            return "mcq_generation_v1"
        else:
            return "qa_generation_v1"


class GapDrivenSampling(SamplingStrategy):
    """缺口驱动采样。

    用途：
    1. 主题内部已经出现明确主缺口时

    特点：
    1. 缺 multiple_choice 时提高 mcq_score 权重
    2. 缺 qa 时提高 qa_score 权重
    3. 缺 hard 时提高 hard_score 权重
    4. hard 不足时优先提高 multi_chunks 占比
    """

    def __init__(self, seed: int | None = None):
        """初始化缺口驱动采样。

        Args:
            seed: 随机种子
        """
        super().__init__(seed)

    def sample(
        self,
        pool: EvidencePool,
        target_mode: str,
        target_difficulty: str,
        num_evidence: int = 5,
        prefer_multi_chunk: bool = False,
    ) -> GenerationBatch:
        """执行缺口驱动采样。

        Args:
            pool: 证据池
            target_mode: 目标模式
            target_difficulty: 目标难度
            num_evidence: 需要的证据单元数量
            prefer_multi_chunk: 是否优先多 chunk

        Returns:
            生成批次
        """
        # 准备候选单元
        candidates: list[tuple[str, float, str]] = []

        # 单 chunk 候选
        for chunk in pool.single_chunks:
            score = self._calculate_gap_score(
                chunk,
                target_mode,
                target_difficulty,
            )
            candidates.append((chunk.chunk_id, score, "single"))

        # 多 chunk 候选
        for unit in pool.multi_chunks:
            score = self._calculate_gap_score(
                unit,
                target_mode,
                target_difficulty,
            )
            candidates.append((unit.unit_id, score, "multi"))

        # 如果是 hard 题目且有多 chunk，增加多 chunk 权重
        if target_difficulty == "hard" and pool.multi_chunks:
            for unit in pool.multi_chunks:
                # 额外加分
                for i, (uid, score, utype) in enumerate(candidates):
                    if uid == unit.unit_id:
                        candidates[i] = (uid, score * 1.5, "multi")
                        break

        if not candidates:
            return GenerationBatch(
                topic=pool.topic,
                target_mode=target_mode,
                target_difficulty=target_difficulty,
                remaining_count=num_evidence,
                single_chunk_ids=[],
                multi_chunk_ids=[],
                prompt_template_id=self._get_template_id(target_mode),
                requested_min_questions=num_evidence,
                requested_target_questions=num_evidence + 2,
            )

        # 获取分数用于加权采样
        ids = [c[0] for c in candidates]
        scores = np.array([c[1] for c in candidates])
        types_map = {c[0]: c[2] for c in candidates}

        # 归一化分数为概率
        if scores.sum() > 0:
            probs = scores / scores.sum()
        else:
            probs = np.ones(len(scores)) / len(scores)

        # 采样
        sampled_indices = np.random.choice(
            len(ids),
            size=min(num_evidence, len(ids)),
            replace=False,
            p=probs,
        )

        # 根据类型分类
        selected_single: list[str] = []
        selected_multi: list[str] = []

        for idx in sampled_indices:
            unit_id = ids[idx]
            unit_type = types_map[unit_id]

            if unit_type == "single":
                selected_single.append(unit_id)
            else:
                selected_multi.append(unit_id)

        return GenerationBatch(
            topic=pool.topic,
            target_mode=target_mode,
            target_difficulty=target_difficulty,
            remaining_count=num_evidence,
            single_chunk_ids=selected_single,
            multi_chunk_ids=selected_multi,
            prompt_template_id=self._get_template_id(target_mode),
            requested_min_questions=num_evidence,
            requested_target_questions=num_evidence + 2,
        )

    def _calculate_gap_score(
        self,
        unit: SingleChunkUnit | MultiChunkUnit,
        target_mode: str,
        target_difficulty: str,
    ) -> float:
        """计算缺口分数。

        Args:
            unit: 证据单元
            target_mode: 目标模式
            target_difficulty: 目标难度

        Returns:
            缺口分数
        """
        # 基础分数（根据目标模式）
        if target_mode == "multiple_choice":
            base_score = unit.mcq_score * 1.3  # 提高 MCQ 权重
        else:
            base_score = unit.qa_score * 1.3  # 提高 QA 权重

        # 难度权重
        if target_difficulty == "hard":
            hard_weight = 1.5
            base_score = base_score * 0.5 + unit.hard_score * hard_weight
        elif target_difficulty == "easy":
            base_score = base_score * 0.8  # easy 不需要太复杂的证据
        # medium 保持原样

        # 多 chunk 额外加分（hard 题时）
        if isinstance(unit, MultiChunkUnit) and target_difficulty == "hard":
            base_score *= 1.4

        # 使用惩罚
        usage_penalty = 1.0 / (1.0 + unit.usage_count)

        return base_score * usage_penalty

    @staticmethod
    def _get_template_id(target_mode: str) -> str:
        """获取模板 ID。

        Args:
            target_mode: 目标模式

        Returns:
            模板 ID
        """
        if target_mode == "multiple_choice":
            return "mcq_generation_v1"
        else:
            return "qa_generation_v1"


def get_sampling_strategy(
    strategy_type: str,
    seed: int | None = None,
) -> SamplingStrategy:
    """获取采样策略。

    Args:
        strategy_type: 策略类型 ("broad_exploration" 或 "gap_driven")
        seed: 随机种子

    Returns:
        采样策略实例
    """
    if strategy_type == "broad_exploration":
        return BroadExplorationSampling(seed)
    elif strategy_type == "gap_driven":
        return GapDrivenSampling(seed)
    else:
        raise ValueError(f"未知的采样策略: {strategy_type}")