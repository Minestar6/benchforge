# BenchForge 独立题目生成 Agent 技术方案 V3

## 0. 本版修订说明

相对 V2 的关键修订：

1. DecisionEngine 从硬编码 if-else 改为**规则表 + 显式优先级 + 打分**架构
2. Executor 支持 **ActionBatch** 批量并发执行，对齐当前 qa_agent 的一轮多 topic 并行能力
3. 状态管理引入**双轨反馈统计**（lifetime + recent_window）和内存上限控制
4. GroundedQuestionEvolution 输出**强制重入完整验证链路**
5. 保留 V2 的独立部署、命名规避、兼容策略等正确设计不变

---

## 1. 新 Agent 定位

命名：

```text
AdaptiveQuestionGenerationAgent
```

目录：

```text
agents/adaptive_qa_agent/
```

职责：

```text
根据目标配额、验证反馈、topic 覆盖、难度分布、chunk 使用状态，动态选择下一轮题目生成动作。
```

核心结构：

```text
Feedback-driven
Rule-based Decision (规则表 + 优先级)
Batch Executor (并发)
Validation Feedback Loop (双轨统计)
```

---

## 2. 目录结构

```text
agents/
  adaptive_qa_agent/
    __init__.py
    agent.py
    state.py
    actions.py
    decision_engine.py
    rules.py
    executor.py
    feedback.py
    stop_policy.py

    prompts/
      base_prompt.py
      strategy_constraints.py
      prompt_builder.py

    tools/
      evidence_group_builder.py
      retrieval_expansion_tool.py
      generation_tool.py
      grounded_question_evolution_tool.py

    config/
      decision_engine.yaml
      stop_policy.yaml
```

说明：

```text
- 不再有 local_utils/，只复制确需改造的文件，其余直接 import benchforge.utils
- config/ 存放所有阈值和规则配置，不散在代码中
```

---

## 3. 与当前实现的兼容策略

保持不动：

```text
agents/qa_agent/
utils/
```

切换入口：

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

## 4. 主循环

```python
async def run_adaptive_generation_agent(...):
    for mode in modes:
        mode_state = ModeState(mode=mode)
        feedback_state = FeedbackState()

        while True:
            stop, reason = stop_policy.should_stop(mode_state, config)
            if stop:
                break

            action_batch = decision_engine.decide(
                state=mode_state,
                config=config,
                feedback=feedback_state,
            )

            results = await executor.execute_batch(action_batch, mode_state)

            validation_report = await validate_candidates(results.candidates)

            feedback_state.update(validation_report)
            mode_state.update(results, validation_report)
            decision_engine.tick_cooldowns()
```

---

## 5. StopPolicy

文件：`stop_policy.py`，阈值来自 `config/stop_policy.yaml`。

条件：

```text
max_rounds_reached
candidate_pool_sufficient
consecutive_empty_rounds_reached
failure_limit_reached
target_distribution_satisfied
no_improvement_rounds_reached (基于 recent_window)
```

```python
class StopPolicy:
    def should_stop(self, state, config) -> tuple[bool, str | None]:
        if state.round_in_mode >= config.stop.max_rounds:
            return True, "max_rounds_reached"
        if state.accepted_count >= config.stop.target_candidates:
            return True, "candidate_pool_sufficient"
        if state.consecutive_empty_rounds >= config.stop.max_empty_rounds:
            return True, "consecutive_empty_rounds_reached"
        if state.failures_count >= config.stop.max_failures:
            return True, "failure_limit_reached"
        return False, None
```

原则：

```text
不让 LLM 决定是否停止
不让 tool 内部绕过 stop policy
所有终止条件统一收敛到 StopPolicy
```

---

## 6. DecisionEngine — 规则表 + 优先级 + 打分

### 6.1 设计原则

```text
不再使用线性 if-else
每条规则独立定义，可配置优先级和阈值
支持多规则同时命中时的冲突消解
所有阈值集中在 decision_engine.yaml
```

### 6.2 RuleSpec 定义

文件：`rules.py`

```python
@dataclass
class RuleSpec:
    rule_id: str
    priority: int                          # 越大越优先
    exclusive_group: str | None            # 同组互斥，只取最高优先级
    cooldown_rounds: int = 0              # 触发后 N 轮内不再触发
    predicate: Callable[[ModeState, FeedbackState, Config], bool]
    score_fn: Callable[[ModeState, FeedbackState, Config], float] | None = None
    action_factory: Callable[[ModeState, FeedbackState, Config], ActionPlanItem]
```

### 6.3 规则配置示例 (decision_engine.yaml)

```yaml
rules:
  - rule_id: retrieve_more_on_grounding_failure
    priority: 90
    exclusive_group: evidence_acquisition
    cooldown_rounds: 2
    thresholds:
      answer_not_grounded_ratio: 0.4

  - rule_id: expand_evidence_on_insufficiency
    priority: 85
    exclusive_group: evidence_acquisition
    cooldown_rounds: 1
    thresholds:
      evidence_insufficient_ratio: 0.3

  - rule_id: multihop_generation
    priority: 80
    exclusive_group: hard_generation
    thresholds:
      hard_gap_ratio: 0.3
      not_multihop_ratio: 0.5

  - rule_id: hard_generation
    priority: 75
    exclusive_group: hard_generation
    thresholds:
      hard_gap_ratio: 0.3

  - rule_id: topic_focused_generation
    priority: 70
    exclusive_group: null
    thresholds:
      topic_gap_count: 2

  - rule_id: grounded_evolution
    priority: 60
    exclusive_group: evolution
    cooldown_rounds: 3
    thresholds:
      too_easy_ratio: 0.4
      grounded_medium_pool_min: 5

  # 组合规则：hard_gap + topic_gap 同时满足
  - rule_id: hard_topic_combined
    priority: 95
    exclusive_group: hard_generation
    thresholds:
      hard_gap_ratio: 0.2
      topic_gap_count: 1

  - rule_id: default_generate
    priority: 0
    exclusive_group: null
    # predicate 始终返回 True，作为 fallback 兜底
    # 仅当所有其它规则都不命中时才生效
```

### 6.4 规则注册与加载机制

YAML 定义阈值和优先级，Python 注册 predicate 和 action_factory：

```python
from typing import Callable

# 装饰器注册
_PREDICATE_REGISTRY: dict[str, Callable] = {}
_ACTION_FACTORY_REGISTRY: dict[str, Callable] = {}

def register_rule(rule_id: str):
    def decorator(cls):
        _PREDICATE_REGISTRY[rule_id] = cls.predicate
        _ACTION_FACTORY_REGISTRY[rule_id] = cls.action_factory
        return cls
    return decorator

@register_rule("retrieve_more_on_grounding_failure")
class RetrieveMoreOnGroundingFailure:
    @staticmethod
    def predicate(state, feedback, config) -> bool:
        return feedback.recent.answer_not_grounded_ratio > config.thresholds["answer_not_grounded_ratio"]

    @staticmethod
    def action_factory(state, feedback, config) -> ActionPlanItem:
        return ActionPlanItem(
            action_type="retrieve_more_documents",
            work_units=[WorkUnit(topic=t) for t in state.active_topics],
            evidence_strategy="new_document_retrieval",
            reason="grounding failure ratio high in recent window",
        )

@register_rule("default_generate")
class DefaultGenerate:
    @staticmethod
    def predicate(state, feedback, config) -> bool:
        return True  # fallback，始终命中

    @staticmethod
    def action_factory(state, feedback, config) -> ActionPlanItem:
        return ActionPlanItem(
            action_type="generate",
            work_units=[WorkUnit(topic=t) for t in state.active_topics],
        )

# 加载流程
def load_rules(yaml_path: str) -> list[RuleSpec]:
    raw = yaml.safe_load(open(yaml_path))
    rules = []
    for entry in raw["rules"]:
        rules.append(RuleSpec(
            rule_id=entry["rule_id"],
            priority=entry["priority"],
            exclusive_group=entry.get("exclusive_group"),
            cooldown_rounds=entry.get("cooldown_rounds", 0),
            predicate=_PREDICATE_REGISTRY[entry["rule_id"]],
            action_factory=_ACTION_FACTORY_REGISTRY[entry["rule_id"]],
        ))
    return rules
```

### 6.5 决策执行流程

```python
class DecisionEngine:
    def __init__(self, rules: list[RuleSpec], config):
        self.rules = rules
        self.config = config
        self.cooldown_tracker: dict[str, int] = {}

    def decide(self, state, config, feedback) -> ActionBatch:
        # 1. 收集所有命中规则（排除 cooldown 中的）
        fired = [
            r for r in self.rules
            if r.predicate(state, feedback, config)
            and self._not_in_cooldown(r)
        ]

        # 2. 按 exclusive_group 去冲突（同组保留最高 priority）
        resolved = self._resolve_exclusive_groups(fired)

        # 3. 同 priority 时按 score_fn 打分排序
        resolved.sort(key=lambda r: (r.priority, self._score(r, state, feedback)), reverse=True)

        # 4. 取 top-1 规则生成 ActionBatch
        if not resolved:
            return self._default_action(state)

        winner = resolved[0]
        self._apply_cooldown(winner)
        plan_item = winner.action_factory(state, feedback, config)
        return ActionBatch(items=[plan_item])

    def tick_cooldowns(self):
        """每轮结束后调用，递减所有 cooldown 计数器"""
        expired = []
        for rule_id, remaining in self.cooldown_tracker.items():
            if remaining <= 1:
                expired.append(rule_id)
            else:
                self.cooldown_tracker[rule_id] = remaining - 1
        for rule_id in expired:
            del self.cooldown_tracker[rule_id]
```

### 6.6 组合规则说明

```text
hard_gap + topic_gap 同时满足时，触发 hard_topic_combined（priority=95）
它的 action_factory 生成针对缺失 topic 的 hard 生成动作
优先级高于单独的 hard_generation(75) 和 topic_focused_generation(70)
```

---

## 7. ActionBatch 与并发执行

### 7.1 ActionPlanItem 与 ActionBatch

文件：`actions.py`

```python
@dataclass
class ActionPlanItem:
    action_type: str
    work_units: list[WorkUnit]
    difficulty: str | None = None
    evidence_strategy: str | None = None
    prompt_strategy: str | None = None
    reason: str | None = None
    concurrency_limit: int = 5            # 该动作内部 work_unit 的并发上限

@dataclass
class WorkUnit:
    topic: str | None = None
    evidence_group_id: str | None = None
    chunk_ids: list[str] | None = None

@dataclass
class ActionBatch:
    items: list[ActionPlanItem]
    merge_policy: str = "append"  # append | replace | dedupe
```

### 7.2 Executor 并发模型

文件：`executor.py`

```python
class AdaptiveActionExecutor:
    async def execute_batch(self, batch: ActionBatch, state) -> BatchResult:
        all_results = []
        for item in batch.items:
            results = await self._execute_item_concurrent(item, state)
            all_results.extend(results)
        return BatchResult(candidates=all_results)

    async def _execute_item_concurrent(self, item, state) -> list:
        semaphore = asyncio.Semaphore(item.concurrency_limit)

        async def run_unit(unit: WorkUnit):
            async with semaphore:
                return await self._dispatch(item.action_type, unit, state)

        tasks = [run_unit(u) for u in item.work_units]
        return await asyncio.gather(*tasks)
```

### 7.3 并发规则

```text
同类生成动作：按 topic / evidence_group 并发，受 semaphore 限制
检索扩展动作：按 topic 并发
会修改共享池的动作（如 evolution）：串行执行或受锁保护
```

语义：**一轮有一个主动作类型，但该动作内部多 work_unit 并行执行**，与当前 qa_agent 一轮多 topic 并行一致。

---

## 8. FeedbackState — 双轨统计 + 内存控制

### 8.1 双轨设计

文件：`feedback.py`

```python
@dataclass
class FeedbackWindow:
    """最近 N 轮的统计，用于驱动决策"""
    window_size: int = 5
    rounds: deque[RoundFeedback] = field(default_factory=lambda: deque(maxlen=5))

    @property
    def answer_not_grounded_ratio(self) -> float:
        if not self.rounds:
            return 0.0
        total = sum(r.total_count for r in self.rounds)
        bad = sum(r.answer_not_grounded_count for r in self.rounds)
        return bad / max(1, total)

    @property
    def evidence_insufficient_ratio(self) -> float:
        if not self.rounds:
            return 0.0
        total = sum(r.total_count for r in self.rounds)
        bad = sum(r.evidence_insufficient_count for r in self.rounds)
        return bad / max(1, total)

    @property
    def too_easy_ratio(self) -> float:
        if not self.rounds:
            return 0.0
        total = sum(r.total_count for r in self.rounds)
        bad = sum(r.too_easy_count for r in self.rounds)
        return bad / max(1, total)

    @property
    def not_multihop_ratio(self) -> float:
        if not self.rounds:
            return 0.0
        total = sum(r.total_count for r in self.rounds)
        bad = sum(r.not_multihop_count for r in self.rounds)
        return bad / max(1, total)

    @property
    def evolution_failed_ratio(self) -> float:
        attempted = sum(r.evolution_attempted for r in self.rounds)
        failed = sum(r.evolution_failed for r in self.rounds)
        return failed / max(1, attempted)

    def push(self, round_fb: RoundFeedback):
        self.rounds.append(round_fb)  # deque(maxlen=N) 自动淘汰最旧

@dataclass
class RoundFeedback:
    round_id: int
    total_count: int
    accepted_count: int
    rejected_count: int
    too_easy_count: int
    not_multihop_count: int
    answer_not_grounded_count: int
    evidence_insufficient_count: int
    duplicate_count: int
    format_error_count: int
    evolution_attempted: int = 0
    evolution_failed: int = 0

@dataclass
class LifetimeStats:
    """全局累计统计，仅用于监控和最终报告"""
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    reject_reason_counts: dict[str, int] = field(default_factory=dict)
    evolution_attempted: int = 0
    evolution_failed: int = 0

    def accumulate(self, round_fb: RoundFeedback):
        self.total_generated += round_fb.total_count
        self.total_accepted += round_fb.accepted_count
        self.total_rejected += round_fb.rejected_count
        self.evolution_attempted += round_fb.evolution_attempted
        self.evolution_failed += round_fb.evolution_failed

@dataclass
class FeedbackState:
    lifetime: LifetimeStats      # 全局累计，仅用于监控和报告
    recent: FeedbackWindow       # 最近 N 轮，用于驱动 DecisionEngine
    
    def update(self, validation_report):
        round_fb = self._aggregate_round(validation_report)
        self.recent.push(round_fb)
        self.lifetime.accumulate(round_fb)
```

### 8.2 DecisionEngine 使用 recent 而非 lifetime

```python
# DecisionEngine 中的 predicate 示例
def predicate(state, feedback, config):
    return feedback.recent.answer_not_grounded_ratio > config.thresholds.answer_not_grounded_ratio
```

全局 lifetime 只做兜底监控和最终报告，不直接驱动下一轮动作选择。

### 8.3 内存上限控制

```python
@dataclass
class ModeState:
    mode: str
    round_in_mode: int = 1
    # 内存中只保留轻量索引
    candidate_index: list[CandidateRef] = field(default_factory=list)
    # 完整题目落盘到 artifact_store
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    consecutive_empty_rounds: int = 0
    failures_count: int = 0
    stopped_reason: str | None = None

    @property
    def accepted_count(self) -> int:
        return sum(1 for c in self.candidate_index if c.status == "accepted")

@dataclass
class CandidateRef:
    question_id: str
    topic: str
    difficulty: str
    status: str           # accepted | rejected | pending
    reject_reason: str | None = None
```

```text
完整题目内容通过 ArtifactStore 落盘
内存中仅保留 CandidateRef 索引
避免长时间运行后内存无限增长
```

---

## 9. GroundedQuestionEvolution — 强制重入验证

### 9.1 定位

```text
兜底工具，不是 hard 题生成主路径
```

### 9.2 触发条件

```python
# 允许触发
grounded == True
answerable == True
evidence_sufficient == True
difficulty in ["easy", "medium"]
reject_reason == "too_easy"

# 禁止触发
answer_not_grounded / evidence_insufficient / answer_ambiguous / duplicate / format_error
```

### 9.3 输出与重入验证

```python
@dataclass
class EvolvedCandidate:
    evolved_question: str
    answer: str
    difficulty: str
    difficulty_reason: str
    required_evidence_ids: list[str]
    reasoning_path: list[str]
    # lineage 追踪
    source_question_id: str
    evolution_round: int
    evolution_strategy: str
```

**关键规则：evolved_question 不是最终产物，只是新的 candidate。**

重入流程：

```text
GroundedQuestionEvolutionTool 输出 EvolvedCandidate
  ↓
重新进入完整验证链路：
  1. citation / evidence consistency 检查
  2. grounded / answerable / difficulty / multihop 验证
  3. dedup（含与原题去重）
  4. final selection
  ↓
验证通过 → 进入 accepted 池
验证失败 → 计入 evolution_failed_count → 反馈给 DecisionEngine
```

### 9.4 反向反馈

```python
# FeedbackState 增加
evolution_attempted_count: int = 0
evolution_failed_count: int = 0

# DecisionEngine 规则中增加保护
def grounded_evolution_predicate(state, feedback, config):
    if feedback.recent.evolution_failed_ratio > 0.7:
        return False  # 连续失败率过高时停止 evolution
    return feedback.recent.too_easy_ratio > config.thresholds.too_easy_ratio
```

---

## 10. EvidenceGroupBuilder

文件：`tools/evidence_group_builder.py`

来源：复制并改造 `utils/multi_chunk.py`（仅此文件需要改造，其余 utils 直接引用）。

策略：

```text
same_document_neighbor
same_entity
same_topic
high_hard_score
new_document_retrieval
multi_chunk_unit
```

输出：

```python
@dataclass
class EvidenceGroup:
    chunk_ids: list[str]
    evidence_units: list[dict]
    strategy: str
    expected_reasoning_type: str
```

---

## 11. Prompt 设计

```text
prompts/
  base_prompt.py           # 统一格式、字段、基础 grounding 要求
  strategy_constraints.py  # 按策略注入 hard/multihop/topic/evolution 约束
  prompt_builder.py        # 组装 base + constraints
```

Hard Constraints 输出格式：

```json
{
  "question": "...",
  "answer": "...",
  "difficulty": "hard",
  "difficulty_reason": "...",
  "reasoning_type": "...",
  "required_evidence_ids": ["chunk_1", "chunk_3"],
  "reasoning_path": ["..."]
}
```

---

## 12. 验证接口契约

生成 Agent 与验证 Agent 之间的共享 schema：

```python
@dataclass
class ValidationReport:
    """验证智能体的标准化输出"""
    items: list[ValidationItem]

@dataclass
class ValidationItem:
    question_id: str
    grounded: bool
    answerable: bool
    evidence_sufficient: bool
    difficulty_assessed: str
    is_multihop: bool
    is_duplicate: bool
    accepted: bool
    reject_reason: str | None = None
    # reject_reason 枚举值：
    #   answer_not_grounded | evidence_insufficient | too_easy
    #   not_multihop | duplicate | format_error | answer_ambiguous
```

---

## 13. 实施阶段

### Phase 1: 骨架搭建

```text
创建 agents/adaptive_qa_agent/
实现 state / actions / stop_policy
实现 DecisionEngine 框架 + 规则加载
编写 decision_engine.yaml 初始规则集
```

验收：能跑通空循环，输出 stop reason，规则加载无报错。

### Phase 2: 生成与并发

```text
实现 ActionBatch / Executor 并发模型
generation_tool 包装当前 generator
prompt_builder 注入 hard/multihop constraints
```

验收：单 mode 下能并发生成多 topic 题目，生成数量和当前实现对齐。

### Phase 3: EvidenceGroupBuilder

```text
从 utils/multi_chunk.py 改造
支持 hard/multihop evidence group 构建
```

验收：给定 seed chunks 能输出合格 evidence group。

### Phase 4: Feedback Loop

```text
实现 FeedbackState 双轨统计
实现 FeedbackWindow 滑动窗口
接入验证智能体，遵守 ValidationReport 契约
DecisionEngine 根据 recent window 驱动动作
```

验收：多轮循环中 DecisionEngine 能根据反馈切换动作类型。

### Phase 5: Grounded Evolution + 重入验证

```text
实现 grounded_question_evolution_tool
输出强制重入完整验证链路
增加 evolution_failed_count 反向反馈
```

验收：evolution 后的题目经过完整验证，失败时 DecisionEngine 能自动降低 evolution 频率。

---

## 14. 最终架构总结

```text
AdaptiveQuestionGenerationAgent
+ DecisionEngine（规则表 + 优先级 + 打分 + cooldown）
+ StopPolicy（统一终止收敛）
+ ActionBatch + Executor（一轮多 work_unit 并发）
+ EvidenceGroupBuilder
+ Prompt Strategy Injection
+ FeedbackState（双轨：recent_window 驱动 + lifetime 监控）
+ GroundedEvolution（强制重入验证 + 失败反馈）
+ ValidationReport 接口契约
```

不应当：

```text
直接改旧 qa_agent
继续使用 planner 命名
让 LLM 每轮自主决策
让 DifficultyEvolution 成为主路径
在代码中硬编码阈值
把完整题目常驻内存
Evolution 后跳过验证
```
