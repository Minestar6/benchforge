"""计划编译和状态更新辅助工具。"""

from typing import Any

from benchforge.schemas import (
    GenerationPlan,
    TopicState,
    TopicStatus,
    QuestionMode,
    Difficulty,
    QuestionModeTarget,
    EvidenceStats,
    EvidenceTypeStats,
    NextStepAction,
    NextStepPlan,
)


def compile_generation_plan(plan: GenerationPlan) -> dict[str, TopicState]:
    """编译生成计划，初始化主题状态。

    Args:
        plan: 生成计划

    Returns:
        主题状态字典
    """
    topic_states: dict[str, TopicState] = {}

    # 1. 展开 mode × difficulty 目标数
    global_targets = _expand_mode_targets(plan.mode_targets)

    # 2. 按主题均分
    topic_targets = _distribute_across_topics(
        global_targets,
        plan.topics,
    )

    # 3. 初始化 TopicState
    for topic in plan.topics:
        topic_states[topic] = TopicState(
            topic=topic,
            status=TopicStatus.PENDING,
            current_round=0,
            target_counts=topic_targets.get(topic, {}),
            completed_counts={},
            remaining_counts=topic_targets.get(topic, {}),
        )

    return topic_states


def _expand_mode_targets(
    mode_targets: dict[str, QuestionModeTarget],
) -> dict[str, int]:
    """展开模式目标为具体难度数量。

    使用最大余数法分配余数。

    Args:
        mode_targets: 模式目标

    Returns:
        展开后的目标 {mode:difficulty: count}
    """
    expanded = {}

    for mode, target in mode_targets.items():
        count = target.count
        distribution = target.difficulty_distribution

        # 计算浮点目标
        float_targets = {
            f"{mode}:{diff}": count * ratio
            for diff, ratio in distribution.items()
        }

        # 向下取整
        floor_targets = {
            key: int(value)
            for key, value in float_targets.items()
        }

        # 计算余数
        remainder_keys = [
            (key, float_targets[key] - floor_targets[key])
            for key in floor_targets.keys()
        ]
        remainder_keys.sort(key=lambda x: x[1], reverse=True)

        # 计算已分配总数
        allocated = sum(floor_targets.values())
        remainder = count - allocated

        # 分配余数
        for i in range(remainder):
            if i < len(remainder_keys):
                key = remainder_keys[i][0]
                floor_targets[key] += 1

        expanded.update(floor_targets)

    return expanded


def _distribute_across_topics(
    global_targets: dict[str, int],
    topics: list[str],
) -> dict[str, dict[str, int]]:
    """将全局目标均分到各主题。

    Args:
        global_targets: 全局目标
        topics: 主题列表

    Returns:
        各主题目标 {topic: {mode:difficulty: count}}
    """
    topic_targets = {}

    if not topics:
        return topic_targets

    num_topics = len(topics)

    for key, total in global_targets.items():
        base = total // num_topics
        remainder = total % num_topics

        for i, topic in enumerate(topics):
            if topic not in topic_targets:
                topic_targets[topic] = {}

            count = base
            if i < remainder:
                count += 1

            topic_targets[topic][key] = count

    return topic_targets


def update_topic_state(
    state: TopicState,
    batch: dict[str, Any],
) -> TopicState:
    """更新主题状态。

    Args:
        state: 当前主题状态
        batch: 本轮生成批次信息

    Returns:
        更新后的主题状态
    """

    # 更新已完成计数
    for key, count in batch.get("completed_counts", {}).items():
        if key not in state.completed_counts:
            state.completed_counts[key] = 0
        state.completed_counts[key] += count

    # 更新剩余计数
    for key in state.target_counts.keys():
        completed = state.completed_counts.get(key, 0)
        target = state.target_counts.get(key, 0)
        state.remaining_counts[key] = max(0, target - completed)

    return state


def identify_main_gap(state: TopicState) -> tuple[str, int] | None:
    """识别当前主题的主缺口。

    优先级：
    1. hard 优先
    2. 剩余数量大的优先

    Args:
        state: 主题状态

    Returns:
        (缺口键, 剩余数量) 或 None
    """
    gaps = []

    for key, remaining in state.remaining_counts.items():
        if remaining > 0:
            gaps.append((key, remaining))

    if not gaps:
        return None

    # hard 优先排序
    gaps.sort(
        key=lambda x: (
            0 if "hard" in x[0] else (1 if "medium" in x[0] else 2),
            -x[1],  # 剩余数量降序
        )
    )

    return gaps[0]


def update_evidence_stats(
    stats: EvidenceStats,
    batch: dict[str, Any],
) -> EvidenceStats:
    """更新证据统计。

    独立更新单 chunk 和多 chunk 的统计。

    Args:
        stats: 当前统计
        batch: 本轮批次信息

    Returns:
        更新后的统计
    """
    alpha = 0.3

    # 单 chunk 统计
    used_single = batch.get("used_single_chunks", 0)
    if used_single > 0:
        candidate_rate = batch.get("candidate_count", 0) / used_single
        valid_rate = batch.get("valid_count", 0) / used_single

        stats.single_chunk_stats.avg_candidate_count = (
            alpha * candidate_rate +
            (1 - alpha) * stats.single_chunk_stats.avg_candidate_count
        )
        stats.single_chunk_stats.avg_valid_count = (
            alpha * valid_rate +
            (1 - alpha) * stats.single_chunk_stats.avg_valid_count
        )

        # 更新模式分布
        for mode, count in batch.get("single_mode_counts", {}).items():
            total = batch.get("valid_count", 0)
            if total > 0:
                ratio = count / total
                dist = stats.single_chunk_stats.mode_distribution
                dist[mode] = alpha * ratio + (1 - alpha) * dist.get(mode, 0)

        # 更新难度分布
        for diff, count in batch.get("single_difficulty_counts", {}).items():
            total = batch.get("valid_count", 0)
            if total > 0:
                ratio = count / total
                dist = stats.single_chunk_stats.difficulty_distribution
                dist[diff] = alpha * ratio + (1 - alpha) * dist.get(diff, 0)

    # 多 chunk 统计
    used_multi = batch.get("used_multi_chunks", 0)
    if used_multi > 0:
        # 多 chunk 的候选率假设与单 chunk 相同（因为来自同一批生成）
        candidate_rate = batch.get("candidate_count", 0) / max(used_single + used_multi, 1)
        valid_rate = batch.get("valid_count", 0) / max(used_single + used_multi, 1)

        stats.multi_chunk_stats.avg_candidate_count = (
            alpha * candidate_rate +
            (1 - alpha) * stats.multi_chunk_stats.avg_candidate_count
        )
        stats.multi_chunk_stats.avg_valid_count = (
            alpha * valid_rate +
            (1 - alpha) * stats.multi_chunk_stats.avg_valid_count
        )

        # 更新模式分布
        for mode, count in batch.get("multi_mode_counts", {}).items():
            total = batch.get("valid_count", 0)
            if total > 0:
                ratio = count / total
                dist = stats.multi_chunk_stats.mode_distribution
                dist[mode] = alpha * ratio + (1 - alpha) * dist.get(mode, 0)

        # 更新难度分布
        for diff, count in batch.get("multi_difficulty_counts", {}).items():
            total = batch.get("valid_count", 0)
            if total > 0:
                ratio = count / total
                dist = stats.multi_chunk_stats.difficulty_distribution
                dist[diff] = alpha * ratio + (1 - alpha) * dist.get(diff, 0)

    return stats


def check_topic_completion(state: TopicState) -> bool:
    """检查主题是否完成。

    Args:
        state: 主题状态

    Returns:
        是否完成
    """
    for key, target in state.target_counts.items():
        completed = state.completed_counts.get(key, 0)
        if completed < target:
            return False
    return True


def calculate_batch_request_counts(remaining_count: int) -> tuple[int, int]:
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


def get_difficulty_instructions(difficulty: str) -> str:
    """获取难度附加说明。

    Args:
        difficulty: 难度级别

    Returns:
        附加说明文本
    """
    instructions = {
        "easy": """
**Easy 题目要求**:
- 强调单证据片段的直接可回答性
- 避免需要复杂推理或跨片段整合
- 答案应当直接在证据中可找到
- 适合测试基本事实理解
""",
        "medium": """
**Medium 题目要求**:
- 强调同一证据单元中的多事实整合
- 需要一定的理解和推理
- 可能需要解释概念或机制
- 适合测试综合理解能力
""",
        "hard": """
**Hard 题目要求**:
- 强调比较、因果、机制、约束等复杂推理
- 需要非显式推断和深度理解
- 可能需要跨证据片段整合
- 适合测试高级分析和推理能力
""",
    }
    return instructions.get(difficulty, "")


def format_evidence_texts(
    single_chunks: list[Any],
    multi_chunks: list[Any],
) -> str:
    """格式化证据文本。

    Args:
        single_chunks: 单证据单元列表
        multi_chunks: 多证据单元列表

    Returns:
        格式化后的文本
    """
    parts = []

    for i, chunk in enumerate(single_chunks, 1):
        parts.append(f"### 单证据片段 {i}")
        parts.append(f"ID: {chunk.chunk_id}")
        parts.append(chunk.text)
        parts.append("")

    for i, unit in enumerate(multi_chunks, 1):
        parts.append(f"### 多证据组合 {i}")
        parts.append(f"ID: {unit.unit_id}")
        parts.append("包含片段:")
        for j, text in enumerate(unit.texts, 1):
            parts.append(f"{j}. {text}")
        parts.append("")

    return "\n".join(parts)


def get_template_path(template_id: str) -> str:
    """获取模板路径。

    Args:
        template_id: 模板 ID

    Returns:
        模板路径
    """
    template_map = {
        "mcq_generation_v1": "prompts/mcq_generation.md",
        "qa_generation_v1": "prompts/qa_generation.md",
        "next_step_planning_v1": "prompts/next_step_planning.md",
    }
    return template_map.get(template_id, f"prompts/{template_id}.md")


def build_allowed_actions(state: TopicState, max_rounds_per_topic: int) -> list[NextStepAction]:
    """构建允许的动作列表。

    基于当前状态返回可执行的下一步动作。

    Args:
        state: 当前主题状态
        max_rounds_per_topic: 每主题最大轮数

    Returns:
        允许的动作列表
    """
    actions = [NextStepAction.CONTINUE_GENERATION]

    # 检查是否有 hard 难度缺口
    has_hard_gap = any(
        key.endswith(":hard") and value > 0
        for key, value in state.remaining_counts.items()
    )

    if has_hard_gap:
        actions.append(NextStepAction.INCREASE_MULTI_CHUNK_RATIO)
        actions.append(NextStepAction.ENABLE_HARDENING)
        actions.append(NextStepAction.EXPAND_RETRIEVAL)

    # 检查是否达到最大轮数
    if state.current_round + 1 >= max_rounds_per_topic:
        actions.append(NextStepAction.DEFER_TOPIC)

    return actions


def build_next_step_plan(
    topic: str,
    target_gap: str,
    allowed_actions: list[str],
    prefer_multi_chunk: bool,
) -> NextStepPlan:
    """构建下一步计划。

    基于允许的动作和目标缺口决定下一步。

    Args:
        topic: 主题名称
        target_gap: 目标缺口键（如 "qa:hard"）
        allowed_actions: 允许的动作字符串列表
        prefer_multi_chunk: 是否偏好多 chunk

    Returns:
        下一步计划
    """
    actions_str = [str(a) if isinstance(a, NextStepAction) else a for a in allowed_actions]

    # 优先级：expand_retrieval > continue_generation
    if "expand_retrieval" in actions_str and target_gap.endswith(":hard") and not prefer_multi_chunk:
        return NextStepPlan(
            topic=topic,
            action=NextStepAction.EXPAND_RETRIEVAL,
            target_gap=target_gap,
            retrieval_expansion_queries=[topic],
            reason=f"hard gap '{target_gap}' persists, expanding retrieval",
        )

    if "increase_multi_chunk_ratio" in actions_str and target_gap.endswith(":hard"):
        return NextStepPlan(
            topic=topic,
            action=NextStepAction.INCREASE_MULTI_CHUNK_RATIO,
            target_gap=target_gap,
            prefer_multi_chunk=True,
            reason=f"hard gap '{target_gap}' needs multi-chunk evidence",
        )

    if "enable_hardening" in actions_str:
        return NextStepPlan(
            topic=topic,
            action=NextStepAction.ENABLE_HARDENING,
            target_gap=target_gap,
            hardening_enabled=True,
            additional_instructions="Focus on generating harder questions that require multi-step reasoning.",
            reason=f"hard gap '{target_gap}' needs hardening",
        )

    return NextStepPlan(
        topic=topic,
        action=NextStepAction.CONTINUE_GENERATION,
        target_gap=target_gap,
        reason="continue with current evidence",
    )


def identify_global_gap(topic_states: dict[str, TopicState]) -> tuple[str | None, list[str]]:
    """识别全局最大的模式难度缺口。

    Args:
        topic_states: 所有主题状态

    Returns:
        (缺口键, 主题列表) 或 (None, [])
    """
    gap_totals: dict[str, int] = {}
    gap_topics: dict[str, list[str]] = {}

    for topic, state in topic_states.items():
        if state.status != TopicStatus.DEFERRED:
            continue

        for key, remaining in state.remaining_counts.items():
            if remaining <= 0:
                continue
            gap_totals[key] = gap_totals.get(key, 0) + remaining
            gap_topics.setdefault(key, []).append(topic)

    if not gap_totals:
        return None, []

    # 按缺口数量降序，然后按模式难度排序（hard > medium > easy）
    def sort_key(item: tuple[str, int]) -> tuple[int, str]:
        key, count = item
        difficulty_order = {"hard": 0, "medium": 1, "easy": 2}
        diff = key.split(":")[-1] if ":" in key else "medium"
        return (-count, difficulty_order.get(diff, 1), key)

    best_key = sorted(gap_totals.items(), key=sort_key)[0][0]
    return best_key, gap_topics.get(best_key, [])