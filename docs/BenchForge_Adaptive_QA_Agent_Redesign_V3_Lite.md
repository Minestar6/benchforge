# BenchForge Adaptive QA Agent 技术方案 V3-Lite（修订版）

## 0. 设计原则

- **真正独立**：不 import qa_agent 内部模块（planner/sampling/generator/executor），只依赖公共接口
- 决策逻辑用带优先级的函数链，不搞规则引擎
- 依赖的公共接口：`EvidenceManager.sample()` / `EvidenceManager.expand_retrieval()` / `BaseModelClient.complete()` / `VerifyAgent.run()`
- 增量配置与全局配置分离：adaptive agent 只定义决策阈值和 prompt 模板

---

## 1. 目录结构

```text
agents/adaptive_qa_agent/
  __init__.py
  agent.py              # 主循环
  state.py              # ModeState + FeedbackState
  decision.py           # 决策逻辑（带优先级的函数链）
  executor.py           # 并发执行（调用公共接口）
  feedback_mapper.py    # VerifyAgent 输出 → 决策信号
  evolution.py          # grounded question evolution
  prompts.py            # prompt 组装（自有模板）
  config.yaml           # 增量配置（决策阈值 + 执行参数）
```

依赖的公共接口（不复制、不改造、不 import 内部实现）：

```text
agents.qa_agent.evidence_manager.EvidenceManager   — .sample() / .expand_retrieval() / .get_evidence_text() / .get_document_summary()
agents.verify_agent.agent.VerifyAgent              — .run(task_id, run_id, blueprint, candidates) -> ValidationTaskResult
agents.verify_agent.schema.*                       — FinalStatus, QuestionCandidate, ValidatedQuestionRecord, ValidationTaskResult 等
models.base.BaseModelClient                        — .complete(model, messages, **kwargs) / .model_name (属性)
```

不依赖（与 qa_agent 解耦）：

```text
agents.qa_agent.planner        — 不引用 compute_dynamic_chunk_k / mode_candidate_target
agents.qa_agent.sampling       — 不引用 sample_chunks / record_global_chunk_usage
agents.qa_agent.generator      — 不引用 Generator 类
agents.qa_agent.executor       — 不引用 Executor 类
```

---

## 2. 配置分层

adaptive agent 的 `config.yaml` 只定义增量配置：

```yaml
# agents/adaptive_qa_agent/config.yaml

decision:
  answer_not_grounded_ratio: 0.4
  evidence_insufficient_ratio: 0.3
  hard_gap_threshold: 0.2
  not_multihop_ratio: 0.5
  too_easy_ratio: 0.4
  topic_gap_ratio: 0.3            # 缺失 topic 占总 topic 比例超过此值时触发
  evolution_fail_cap: 0.7
  max_consecutive_same_action: 3  # 同一非 generate action 连续触发上限

execution:
  concurrency: 5
  feedback_window: 5

generation:
  model: null                 # null 则使用 model_client.model_name（注入时已确定的模型）
  temperature: 0.7
  max_tokens: 4096

stop:
  max_empty_rounds: 3
  max_failures: 10            # 生成失败（解析失败/空结果/采样耗尽）累计次数上限，不含验证被拒
```

主循环接收的参数：

```python
async def run_adaptive_generation_agent(
    blueprint,             # 含 modes, topics, language, task_id, run_id
    config,                # 全局配置（含 candidate_pool.target_multiplier）
    adaptive_config,       # 增量配置（上述 YAML，加载后为 dataclass/namespace 对象）
    evidence_manager,      # EvidenceManager 实例（已初始化好 evidence_pools）
    model_client,          # BaseModelClient 实例（含 .model_name 属性）
    verify_config,         # VerifyAgentConfig（验证配置）
)
```

**配置加载约定**：YAML 加载后统一转为嵌套 `SimpleNamespace`（或 dataclass），保证所有层级用 attr 访问（`cfg.field`），不使用 dict 的 `cfg["field"]` 或 `cfg.get()`。

---

## 3. 状态 (state.py)

```python
from collections import deque
from dataclasses import dataclass, field


@dataclass
class CandidateRef:
    question_id: str
    topic: str
    difficulty: str
    status: str  # accepted | rejected
    reject_reason: str | None = None


@dataclass
class ModeState:
    mode: str
    all_topics: list[str] = field(default_factory=list)
    target_hard_ratio: float = 0.3
    target_candidates: int = 100
    max_rounds: int = 20
    round_in_mode: int = 0
    candidates: list[CandidateRef] = field(default_factory=list)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    consecutive_empty: int = 0
    failures: int = 0
    # 采样去重：记录已使用的 chunk 组合，避免重复采样
    used_chunk_combinations: set[frozenset[str]] = field(default_factory=set)
    # 防震荡：记录上一轮 action_type
    last_action_type: str = ""
    consecutive_same_action: int = 0

    @property
    def accepted_count(self) -> int:
        return sum(1 for c in self.candidates if c.status == "accepted")

    @property
    def hard_gap(self) -> float:
        total = max(1, self.accepted_count)
        hard = self.difficulty_counts.get("hard", 0)
        return max(0.0, self.target_hard_ratio - hard / total)

    @property
    def missing_topics(self) -> list[str]:
        if not self.all_topics:
            return []
        avg = self.accepted_count / max(1, len(self.all_topics))
        threshold = max(1.0, avg * 0.5)  # 低于平均值50%才算缺失，避免几乎所有topic都"缺失"
        return [t for t in self.all_topics if self.topic_counts.get(t, 0) < threshold]

    @property
    def active_topics(self) -> list[str]:
        return self.missing_topics or self.all_topics

    def record_action(self, action_type: str):
        """记录本轮 action，用于防震荡。首次出现计为 1。"""
        if action_type == self.last_action_type:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
            self.last_action_type = action_type

    def update(self, mapped_results: list):
        """每轮结束后更新（仅当有验证结果时调用）"""
        self.consecutive_empty = 0
        for item in mapped_results:
            ref = CandidateRef(
                question_id=item.question_id,
                topic=item.topic,
                difficulty=item.difficulty,
                status="accepted" if item.accepted else "rejected",
                reject_reason=item.reject_reason,
            )
            self.candidates.append(ref)
            if item.accepted:
                self.difficulty_counts[ref.difficulty] = self.difficulty_counts.get(ref.difficulty, 0) + 1
                self.topic_counts[ref.topic] = self.topic_counts.get(ref.topic, 0) + 1

    def record_chunk_combination(self, chunk_ids: list[str]):
        """记录使用过的 chunk 组合"""
        self.used_chunk_combinations.add(frozenset(chunk_ids))

    def is_chunk_combination_used(self, chunk_ids: list[str]) -> bool:
        return frozenset(chunk_ids) in self.used_chunk_combinations


class FeedbackState:
    """滑动窗口反馈统计。不使用 dataclass 以避免 deque 初始化问题。"""

    def __init__(self, window_size: int = 5):
        self._window: deque[dict] = deque(maxlen=window_size)
        self.total_generated: int = 0
        self.total_accepted: int = 0

    def push(self, round_stats: dict):
        self._window.append(round_stats)
        self.total_generated += round_stats["total"]
        self.total_accepted += round_stats["accepted"]

    def ratio(self, key: str) -> float:
        if not self._window:
            return 0.0
        total = sum(r["total"] for r in self._window)
        bad = sum(r.get(key, 0) for r in self._window)
        return bad / max(1, total)
```

分母统一为 `accepted_count`（不含 rejected），避免 rejection 放大"不足"假象。

---

## 4. 决策 (decision.py)

**配置约定**：`adaptive_config` 各子节点统一使用 `dataclass` 或 `SimpleNamespace` 访问（即 `cfg.field` 形式），YAML 加载时统一转为对象。不混用 dict 和 attr 访问。

```python
from dataclasses import dataclass, field


@dataclass
class Action:
    action_type: str
    topics: list[str] = field(default_factory=list)
    difficulty: str | None = None
    evidence_strategy: str | None = None
    reason: str = ""


def decide(state, feedback, adaptive_config) -> Action:
    cfg = adaptive_config.decision

    # 防震荡：如果同一非 generate action 连续触发超过阈值，强制降级
    max_repeat = cfg.max_consecutive_same_action

    def _should_suppress(action_type: str) -> bool:
        if action_type == "generate":
            return False
        return (state.last_action_type == action_type
                and state.consecutive_same_action >= max_repeat)

    # P1: grounding 失败多 → 检索新文档
    if (feedback.ratio("answer_not_grounded") > cfg.answer_not_grounded_ratio
            and not _should_suppress("retrieve_more")):
        return Action("retrieve_more", topics=state.missing_topics,
                      evidence_strategy="new_document", reason="grounding failure high")

    # P2: evidence 不足 → 扩展证据
    if (feedback.ratio("evidence_insufficient") > cfg.evidence_insufficient_ratio
            and not _should_suppress("expand_evidence")):
        return Action("expand_evidence", topics=state.missing_topics,
                      evidence_strategy="neighbor", reason="evidence insufficient")

    # P3: hard 缺口大 + multihop 不足
    if state.hard_gap > cfg.hard_gap_threshold and feedback.ratio("not_multihop") > cfg.not_multihop_ratio:
        return Action("generate", difficulty="hard",
                      evidence_strategy="multi_group", reason="hard+multihop gap")

    # P4: hard 缺口大
    if state.hard_gap > cfg.hard_gap_threshold:
        return Action("generate", difficulty="hard",
                      evidence_strategy="high_hard_score", reason="hard gap")

    # P5: topic 覆盖不足
    topic_gap_ratio = len(state.missing_topics) / max(1, len(state.all_topics))
    if topic_gap_ratio > cfg.topic_gap_ratio:
        return Action("generate", topics=state.missing_topics, reason="topic gap")

    # P6: too_easy 多 + evolution 未失控
    if (feedback.ratio("too_easy") > cfg.too_easy_ratio
            and feedback.ratio("evolution_failed") < cfg.evolution_fail_cap
            and not _should_suppress("evolve")):
        return Action("evolve", reason="too many easy, evolve up")

    # Fallback
    return Action("generate", reason="default")
```

---

## 5. 执行 (executor.py) — 自有生成，不依赖 qa_agent

**核心区别**：不 import Generator 类，不 import planner/sampling。直接调用 `EvidenceManager.sample()` 获取 batch，调用 `model_client.complete()` 生成题目。

```python
import asyncio
import json
from agents.verify_agent.schema import QuestionCandidate


class Executor:
    def __init__(self, evidence_manager, model_client, evolution_tool,
                 blueprint, config, adaptive_config, prompts):
        self.evidence_manager = evidence_manager
        self.model_client = model_client
        self.evolution_tool = evolution_tool
        self.blueprint = blueprint
        self.config = config
        self.adaptive_config = adaptive_config
        self.prompts = prompts
        self.sem = asyncio.Semaphore(adaptive_config.execution.concurrency)

    async def run(self, action, state) -> list[QuestionCandidate]:
        topics = action.topics or state.active_topics

        if action.action_type == "retrieve_more":
            await self._concurrent(topics, self._retrieve, action, state)
            return await self._concurrent(topics, self._generate, action, state)

        if action.action_type == "expand_evidence":
            await self._concurrent(topics, self._expand, action, state)
            return await self._concurrent(topics, self._generate, action, state)

        if action.action_type == "evolve":
            return await self._evolve(state)

        return await self._concurrent(topics, self._generate, action, state)

    async def _concurrent(self, items, fn, action, state) -> list[QuestionCandidate]:
        async def bounded(item):
            async with self.sem:
                return await fn(item, action, state)
        results = await asyncio.gather(*[bounded(i) for i in items])
        return [q for batch in results for q in batch]

    async def _generate(self, topic, action, state) -> list[QuestionCandidate]:
        """单 topic 生成：调用 EvidenceManager.sample() + model_client.complete()"""
        difficulty = action.difficulty or "medium"
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        if not evidence_pool:
            return []

        batch = self.evidence_manager.sample(
            evidence_pool=evidence_pool,
            topic=topic,
            target_mode=state.mode,
            target_difficulty=difficulty,
            prefer_multi_chunk=(action.evidence_strategy == "multi_group"),
            round_num=state.round_in_mode + 1,
            remaining=max(1, state.target_candidates - state.accepted_count),
        )
        if not batch or not (batch.single_chunk_ids or batch.multi_chunk_ids):
            return []

        # 采样去重：检查 chunk 组合是否已使用
        chunk_ids = list(batch.single_chunk_ids or []) + list(batch.multi_chunk_ids or [])
        if state.is_chunk_combination_used(chunk_ids):
            return []
        state.record_chunk_combination(chunk_ids)

        evidence_text = self.evidence_manager.get_evidence_text(batch, evidence_pool)
        document_summary = self.evidence_manager.get_document_summary(batch, evidence_pool)

        messages = self.prompts.build_generation_prompt(
            topic=topic,
            difficulty=difficulty,
            mode=state.mode,
            evidence_text=evidence_text,
            document_summary=document_summary,
            language=self.blueprint.language,
            constraints=action.evidence_strategy,
        )

        gen_cfg = self.adaptive_config.generation
        model = gen_cfg.model or self.model_client.model_name
        response = await self.model_client.complete(
            model=model,
            messages=messages,
            temperature=gen_cfg.temperature,
            max_tokens=gen_cfg.max_tokens,
        )

        return self._parse_response(response["text"], topic, state, batch)

    def _parse_response(self, text, topic, state, batch) -> list[QuestionCandidate]:
        try:
            items = json.loads(text) if text.strip().startswith("[") else [json.loads(text)]
        except json.JSONDecodeError:
            return []

        candidates = []
        chunk_ids = list(batch.single_chunk_ids or []) + list(batch.multi_chunk_ids or [])
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                continue
            candidates.append(QuestionCandidate(
                question_id=f"{self.blueprint.run_id}_{state.mode}_{topic}_{state.round_in_mode}_{i}",
                task_id=self.blueprint.task_id,
                run_id=self.blueprint.run_id,
                topic=topic,
                question=item["question"],
                answer=item["answer"],
                question_mode=state.mode,
                estimated_difficulty=item.get("difficulty", "medium"),
                chunk_ids=chunk_ids,
            ))
        return candidates

    async def _retrieve(self, topic, action, state) -> list[QuestionCandidate]:
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        await self.evidence_manager.expand_retrieval(
            topic=topic,
            queries=[topic],
            language=self.blueprint.language,
            run_id=self.blueprint.run_id,
            evidence_pool=evidence_pool,
        )
        return []

    async def _expand(self, topic, action, state) -> list[QuestionCandidate]:
        evidence_pool = self.evidence_manager.evidence_pools.get(topic)
        await self.evidence_manager.expand_retrieval(
            topic=topic,
            queries=[f"{topic} details", f"{topic} related"],
            language=self.blueprint.language,
            run_id=self.blueprint.run_id,
            evidence_pool=evidence_pool,
        )
        return []

    async def _evolve(self, state) -> list[QuestionCandidate]:
        if not self.evolution_tool:
            return []
        easy_pool = [c for c in state.candidates
                     if c.status == "accepted" and c.difficulty in ("easy", "medium")]
        if not easy_pool:
            return []
        return await self.evolution_tool.evolve(easy_pool[:5], state.mode)
```

**关键设计选择**：
- 采样：直接调用 `EvidenceManager.sample()` 公共 API，它内部已实现去重和策略选择
- 去重：executor 额外维护 `state.used_chunk_combinations`，避免对同一 chunk 组合重复生成
- 生成：直接调用 `model_client.complete()`，使用自有 prompt 模板
- 不引入 qa_agent 的 Generator、planner、sampling 任何内部模块

---

## 6. 归因映射 (feedback_mapper.py)

把 VerifyAgent 的验证结果转为决策信号。

**设计说明**：
- `ValidationTaskResult` 返回值含 `selected_question_ids` + `failed_by_stage` 用于快速判断通过/失败
- 逐题归因需要读取 `validated_questions.jsonl`，但仅在需要详细 reject_reason 时读取
- 采用"优先返回值 + 按需文件读取"策略，避免每轮全量扫描文件

**关键修正**：只有 `FinalStatus.selected` 计为 accepted，`reserve` 不计（因为下游评估只使用 selected）。

```python
from dataclasses import dataclass
from pathlib import Path
from agents.verify_agent.schema import (
    FinalStatus, ValidatedQuestionRecord, ValidationTaskResult, QuestionCandidate,
)


@dataclass
class MappedResult:
    question_id: str
    topic: str
    difficulty: str
    accepted: bool
    reject_reason: str | None = None


def map_from_task_result(
    result: ValidationTaskResult,
    candidates: list[QuestionCandidate],
    validation_dir: Path | None = None,
) -> list[MappedResult]:
    """优先从 ValidationTaskResult 返回值映射，按需读文件获取详细归因。

    Args:
        result: VerifyAgent.run() 的返回值
        candidates: 本轮生成的候选题
        validation_dir: 验证落盘目录（可选，提供时读取详细归因）
    """
    selected_ids = set(result.selected_question_ids)

    # 快速路径：仅从返回值判断 accepted/rejected
    results = []
    candidate_map = {c.question_id: c for c in candidates}

    # 如果有落盘文件，读取详细归因
    detailed_reasons: dict[str, str] = {}
    if validation_dir and (validation_dir / "validated_questions.jsonl").exists():
        records = _load_round_records(validation_dir, set(candidate_map.keys()))
        for rec in records:
            if rec.question_id not in selected_ids:
                detailed_reasons[rec.question_id] = _classify_rejection(rec)

    for qid, cand in candidate_map.items():
        accepted = qid in selected_ids
        reject_reason = None if accepted else detailed_reasons.get(qid, "unknown")
        results.append(MappedResult(
            question_id=qid,
            topic=cand.topic,
            difficulty=str(cand.estimated_difficulty or "medium"),
            accepted=accepted,
            reject_reason=reject_reason,
        ))
    return results


def _load_round_records(
    validation_dir: Path, question_ids: set[str]
) -> list[ValidatedQuestionRecord]:
    """从落盘 jsonl 读取本轮相关记录（仅过滤本轮 ID，避免全量扫描）"""
    jsonl_path = validation_dir / "validated_questions.jsonl"
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = ValidatedQuestionRecord.model_validate_json(line)
            if rec.question_id in question_ids:
                records.append(rec)
    return records


def _classify_rejection(rec: ValidatedQuestionRecord) -> str:
    status = rec.final_status

    if status == FinalStatus.rejected_citation.value:
        if rec.citation_validation and rec.citation_validation.answer_citation_score < 0.3:
            return "answer_not_grounded"
        return "evidence_insufficient"

    if status == FinalStatus.rejected_llm.value:
        if rec.llm_validation:
            dims = rec.llm_validation.dimensions
            reasons = rec.llm_validation.failed_reasons
            if dims.get("difficulty", 1.0) < 0.4:
                return "too_easy"
            if "not_multihop" in reasons or "single_source" in reasons:
                return "not_multihop"
        return "format_error"

    if status == FinalStatus.duplicate.value:
        return "duplicate"

    if status == FinalStatus.overquota.value:
        return "overquota"

    if status == FinalStatus.reserve.value:
        return "reserve"

    return "format_error"
```

---

## 7. 主循环 (agent.py)

```python
import math
from pathlib import Path
from agents.verify_agent.agent import VerifyAgent
from agents.verify_agent.schema import QuestionCandidate, ValidationBlueprintView, ModeCfg
from .state import ModeState, FeedbackState
from .decision import decide
from .executor import Executor
from .feedback_mapper import map_from_task_result
from .prompts import PromptBuilder


async def run_adaptive_generation_agent(
    blueprint, config, adaptive_config,
    evidence_manager, model_client, verify_config,
    evolution_tool=None,
):
    """独立 adaptive agent 入口。

    不引用 qa_agent 的 planner/sampling/generator/executor。
    """
    prompts = PromptBuilder(blueprint.language)
    verify_agent = VerifyAgent(config=verify_config, model_client=model_client)

    validation_blueprint = ValidationBlueprintView(
        topics=blueprint.topics,
        modes={
            mode: ModeCfg(
                count=mode_cfg.count,
                difficulty_distribution=mode_cfg.difficulty_distribution,
            )
            for mode, mode_cfg in blueprint.modes.items()
        },
    )

    for mode, mode_cfg in blueprint.modes.items():
        target = math.ceil(mode_cfg.count * config.candidate_pool.target_multiplier)

        state = ModeState(
            mode=mode,
            all_topics=blueprint.topics,
            target_hard_ratio=mode_cfg.difficulty_distribution.get("hard", 0.3),
            target_candidates=target,
            max_rounds=mode_cfg.max_rounds,
        )
        feedback = FeedbackState(window_size=adaptive_config.execution.feedback_window)
        executor = Executor(
            evidence_manager=evidence_manager,
            model_client=model_client,
            evolution_tool=evolution_tool,
            blueprint=blueprint,
            config=config,
            adaptive_config=adaptive_config,
            prompts=prompts,
        )

        while True:
            if state.round_in_mode >= state.max_rounds:
                break
            if state.accepted_count >= state.target_candidates:
                break
            if state.consecutive_empty >= adaptive_config.stop.max_empty_rounds:
                break
            if state.failures >= adaptive_config.stop.max_failures:
                break

            action = decide(state, feedback, adaptive_config)
            state.record_action(action.action_type)

            candidates = await executor.run(action, state)

            if candidates:
                # 调用 VerifyAgent.run() 公共接口，直接使用返回值
                result = await verify_agent.run(
                    task_id=blueprint.task_id,
                    run_id=blueprint.run_id,
                    blueprint=validation_blueprint,
                    candidates=candidates,
                )

                # 从返回值映射，按需读取落盘文件获取详细归因
                validation_dir = Path("runs") / blueprint.task_id / blueprint.run_id / "validation"
                mapped = map_from_task_result(result, candidates, validation_dir)

                round_stats = _aggregate(mapped, is_evolution=(action.action_type == "evolve"))
                feedback.push(round_stats)
                state.update(mapped)
            else:
                state.consecutive_empty += 1
                state.failures += 1  # 生成为空（解析失败/采样耗尽）计入失败

            state.round_in_mode += 1


def _aggregate(mapped: list, is_evolution: bool = False) -> dict:
    stats = {"total": len(mapped), "accepted": 0}
    for item in mapped:
        if item.accepted:
            stats["accepted"] += 1
        elif item.reject_reason:
            stats[item.reject_reason] = stats.get(item.reject_reason, 0) + 1
    if is_evolution:
        stats["evolution_failed"] = stats["total"] - stats["accepted"]
    return stats
```

---

## 8. 与现有系统的对接点

| 对接点 | 公共接口 | adaptive agent 使用方式 |
|--------|---------|----------------------|
| 采样 | `EvidenceManager.sample(evidence_pool, topic, mode, difficulty, ...)` | executor._generate 直接调用 |
| 证据扩展 | `EvidenceManager.expand_retrieval(topic, queries, language, run_id, pool)` | executor._retrieve / _expand 调用 |
| 证据文本 | `EvidenceManager.get_evidence_text(batch, pool)` | executor._generate 获取 prompt 素材 |
| 文档摘要 | `EvidenceManager.get_document_summary(batch, pool)` | executor._generate 获取 prompt 素材 |
| 生成 | `BaseModelClient.complete(model, messages, **kwargs)` | executor._generate 直接调用 |
| 验证 | `VerifyAgent.run(task_id, run_id, blueprint, candidates) -> ValidationTaskResult` | 主循环调用，直接使用返回值中的 selected_question_ids |
| 验证详情 | `validated_questions.jsonl` 落盘文件 | feedback_mapper 按需读取，仅获取被拒原因归因 |
| 停止目标 | `ceil(mode_cfg.count * config.candidate_pool.target_multiplier)` | 主循环初始化时自行计算 |

**不对接（解耦）：**
- `qa_agent.planner.compute_dynamic_chunk_k` — 不需要，由 EvidenceManager.sample() 内部处理采样量
- `qa_agent.sampling.sample_chunks` — 不需要，由 EvidenceManager.sample() 替代
- `qa_agent.generator.Generator` — 不需要，自行调用 model_client + 自有 prompt

---

## 9. 切换入口

```yaml
qa_generation_agent_type: adaptive  # or legacy
```

```python
if config.qa_generation_agent_type == "legacy":
    return await run_generation_agent(...)
if config.qa_generation_agent_type == "adaptive":
    return await run_adaptive_generation_agent(...)
```

---

## 10. 何时升级

- 规则超过 12 条 → 引入规则表（参考 V3 的 DecisionEngine 方案）
- 需要运行时动态调参 → 引入 YAML 热加载
- EvidenceManager.sample() 不能满足特殊采样策略 → 引入自有采样逻辑
- 归因映射无法覆盖新的 verify_agent 输出 → 升级为共享 schema

---

## 11. 实施阶段

### Phase 1: 骨架 + 配置 (1天)

```text
创建 agents/adaptive_qa_agent/
实现 state.py（ModeState + CandidateRef + FeedbackState）
实现 decision.py（Action + decide()）
实现 agent.py 主循环框架
编写 config.yaml
```

验收：空数据下循环正常启停，decide() 返回合理 action。

### Phase 2: 生成接入（自有实现） (2天)

```text
实现 executor.py：
  _generate 调用 EvidenceManager.sample() + model_client.complete()
  _retrieve / _expand 调用 EvidenceManager.expand_retrieval()
  多 topic 并发（semaphore）
实现 prompts.py：自有 prompt 模板，注入 hard/multihop constraints
```

验收：单 mode 下能生成题目，输出格式为 QuestionCandidate 列表。

### Phase 3: 反馈驱动 + 归因映射 (2天)

```text
实现 feedback_mapper.py：
  map_from_task_result() 优先从返回值获取 accepted/rejected
  _load_round_records() 按需读取 jsonl 获取详细归因
  只有 FinalStatus.selected 计为 accepted
主循环接入 VerifyAgent.run()，直接使用 ValidationTaskResult 返回值
FeedbackState 滑动窗口驱动 decide() 切换动作
```

验收：高 grounding 失败时自动切 retrieve_more；hard 不足时切 hard generation。

### Phase 4: Evolution + 端到端 (1天)

```text
实现 evolution.py：
  定义 EvolutionTool 接口协议：
    async def evolve(candidates: list[CandidateRef], mode: str) -> list[QuestionCandidate]
  从已验证通过的题目中恢复原文（通过 validated_questions.jsonl 中的 candidate 字段）
  调用 model_client 生成进化版本
进化产出统一走 VerifyAgent.run() 验证
evolution_failed 计入 feedback
配置切换入口
端到端集成测试
```

验收：全流程跑通，evolution 失败率高时自动停止。

---

## 12. 接口验证清单（实施前 Spike）

实施前需确认以下公共接口的**实际签名**与本文档假设一致：

| 接口 | 文档假设 | 需确认 |
|------|---------|--------|
| `EvidenceManager.sample()` | 支持 `prefer_multi_chunk`, `target_mode`, `target_difficulty`, `round_num`, `remaining` 参数 | ✅ 已确认签名匹配 |
| `EvidenceManager.sample()` 返回值 | 含 `.single_chunk_ids` 和 `.multi_chunk_ids` 属性 | 需确认 GenerationBatch 的字段 |
| `EvidenceManager.expand_retrieval()` | 参数为 `topic, queries, language, run_id, evidence_pool` | ✅ 已确认签名匹配 |
| `EvidenceManager.evidence_pools` | 是 `dict[str, Any]`，key 为 topic | 需确认 |
| `VerifyAgent.run()` | 返回 `ValidationTaskResult` 含 `selected_question_ids: list[str]` | ✅ 已确认 |
| `ValidationTaskResult` | 不含逐题 ValidatedQuestionRecord（需从文件读取） | ✅ 已确认，report_path + selected_ids + failed_by_stage |
| `BaseModelClient.complete()` | 返回 `dict` 含 `"text"` key | 需确认返回值格式 |
| `BaseModelClient.model_name` | 属性，返回当前客户端绑定的模型标识 | 需确认（用于 generation.model 为 null 时的回退） |

**Evolution 接口协议**（需实现）：

```python
from typing import Protocol

class EvolutionTool(Protocol):
    async def evolve(
        self, candidates: list[CandidateRef], mode: str
    ) -> list[QuestionCandidate]:
        """将简单题进化为更难的版本。

        实现需自行从 validated_questions.jsonl 恢复题目原文，
        或通过 question_id 查询已有记录获取完整 QuestionCandidate。
        """
        ...
```
