# BenchForge 轻量化 Adaptive QA Agent 技术方案

## 0. 文档目标

这份文档不是继续扩展完整重构版架构，而是定义一版：

```text
可实现
可稳定运行
改动面可控
便于和现有 qa_agent 做 A/B 对比
```

的轻量化 Adaptive QA Agent。

目标是：

```text
在 agents/adp_qa_agent/ 下单独实现一个新智能体版本，
但尽量复用现有 qa_agent 的执行能力、并发能力、采样能力和落盘能力。
```

不直接改写 `agents/qa_agent/` 主路径。

---

## 1. 设计原则

### 1.1 核心原则

1. **独立目录实现**
   - 新版本放在 `agents/adp_qa_agent/`
   - 不污染现有 `qa_agent` 主路径
   - 便于 A/B 对比与回退

2. **轻量决策，重度复用**
   - 新智能体只新增：
     - 决策层
     - 反馈聚合层
     - 少量策略工具
   - 不重写现有：
     - generator
     - evidence preparation
     - chunk sampling
     - topic 并发执行
     - artifact 基础落盘

3. **先做最小可运行版本**
   - 第一版只支持少量动作类型
   - 不引入复杂规则引擎框架
   - 不引入完整新 validation contract
   - 不引入多 action batch 调度

4. **验证智能体仍然是真实质量判断来源**
   - `adp_qa_agent` 不重写 grounded/answerable/difficulty/multihop 判定
   - 继续消费现有 `verify_agent` 输出

5. **Evolution 只作为支路**
   - 不是主路径
   - 必须重新进入完整验证流程

---

## 2. 与现有复杂方案的取舍

### 2.1 保留的部分

从较完整的 Adaptive 方案中保留这些核心能力：

- 基于反馈的下一轮动作选择
- 根据 topic gap / difficulty gap / grounding 问题调整生成策略
- 必要时执行 grounded question evolution
- 使用近期反馈而非只看全局累计

### 2.2 删掉或推迟的部分

第一版不做：

- 完整 `RuleSpec + exclusive_group + score_fn + cooldown` 规则系统
- 多 action 的 `ActionBatch` 调度器
- 独立的 `StopPolicy` 配置体系
- 全新的验证接口契约
- 完整的新 state / artifact 体系
- 大规模复制现有 `qa_agent` 基础模块

这些都属于第二阶段再考虑的增强项。

---

## 3. 新智能体定位

命名：

```text
AdaptiveQuestionGenerationAgent
```

目录：

```text
agents/
  adp_qa_agent/
    __init__.py
    agent.py
    state.py
    decision.py
    feedback.py
    config_loader.py
    tools/
      generation_tool.py
      evidence_tool.py
      evolution_tool.py
```

职责：

```text
在现有 qa_agent 的基础执行能力之上，
根据验证反馈动态调整 topic、difficulty 和 evidence strategy，
提高最终题目质量与分布覆盖。
```

---

## 4. 与现有 qa_agent 的关系

### 4.1 复用关系

`adp_qa_agent` 不应复制整套 `qa_agent`，而应尽量复用：

- `qa_agent.generator.Generator`
- `qa_agent.evidence_manager.EvidenceManager`
- `qa_agent.sampling.*`
- `qa_agent.executor.execute_mode_round_plan`
- `qa_agent.executor.mode_should_stop`
- `qa_agent.planner.build_mode_round_plan` 的部分逻辑
- `qa_agent.storage` 中已有的 artifact 落盘方式

### 4.2 新旧差异

现有 `qa_agent` 的主要逻辑是：

```text
固定的 mode loop
固定的 round plan 生成
按 topic 并发生成
候选池达到目标后停止
```

`adp_qa_agent` 的增量在于：

```text
每轮先看反馈
根据反馈决定当前轮的策略动作
再调用已有生成执行能力完成本轮
```

所以本质上：

```text
qa_agent = 执行框架
adp_qa_agent = 轻量决策层 + 复用 qa_agent 执行框架
```

---

## 5. 第一版支持的动作

第一版只支持 6 类动作：

```python
GENERATE = "generate"
TOPIC_FOCUS_GENERATE = "topic_focus_generate"
HARD_GENERATE = "hard_generate"
MULTIHOP_GENERATE = "multihop_generate"
EXPAND_EVIDENCE = "expand_evidence"
EVOLVE_GROUNDED_QUESTION = "evolve_grounded_question"
```

说明：

- `generate`
  - 默认动作
  - 按当前 topic 分布正常生成

- `topic_focus_generate`
  - 针对 topic coverage 不足的主题补题

- `hard_generate`
  - 当 hard gap 明显时，提高 hard 比例

- `multihop_generate`
  - 当 hard 题不足且题目不具备多跳性质时，使用 multi-evidence 策略

- `expand_evidence`
  - 当 grounding / evidence support 问题突出时，优先扩大证据来源

- `evolve_grounded_question`
  - 对已 grounding 但过于简单的题做演化
  - 必须重新进入验证链路

---

## 6. 第一版不支持的能力

第一版明确不做：

- 多 action 同轮混合调度
- 复杂优先级冲突求解器
- 通用 rule plugin 框架
- 工具之间的自由编排图
- tool 内部自主停止
- LLM 自主决定下一轮动作

这些能力会显著扩大实现复杂度，不适合作为第一版。

---

## 7. 主循环

主循环保持接近现有 `qa_agent`，只在每轮前后插入新逻辑。

```python
async def run_adaptive_generation_agent(...):
    for mode in blueprint.modes:
        mode_state = ModeState(mode=mode)
        feedback_state = FeedbackState()

        while True:
            stop, reason = should_stop(mode_state, blueprint, config)
            if stop:
                mode_state.stopped_reason = reason
                break

            action = decide_next_action(
                mode_state=mode_state,
                feedback_state=feedback_state,
                blueprint=blueprint,
                config=config,
            )

            round_plan = build_adaptive_round_plan(
                action=action,
                mode_state=mode_state,
                blueprint=blueprint,
                config=config,
            )

            generation_result = await run_generation_round(
                round_plan=round_plan,
                mode_state=mode_state,
                ...
            )

            validation_feedback = await load_or_request_validation_feedback(
                generation_result=generation_result,
                ...
            )

            feedback_state.update(validation_feedback)
            update_mode_state(mode_state, generation_result, validation_feedback)
```

原则：

- 不改现有多 topic 并发核心
- 决策逻辑只发生在每轮开始前
- 验证反馈只发生在每轮结束后

---

## 8. 决策层

文件：

```text
agents/adp_qa_agent/decision.py
```

### 8.1 为什么不用完整规则引擎

完整规则引擎虽然通用，但第一版成本过高。

第一版采用：

```text
有限动作集合
显式优先级顺序
集中阈值配置
少量组合规则
```

也就是：

```text
不是散落的 if-else
但也不是完整通用 rule engine
```

### 8.2 第一版决策顺序

建议顺序如下：

```python
def decide_next_action(mode_state, feedback_state, blueprint, config) -> Action:
    recent = feedback_state.recent

    if recent.answer_not_grounded_rate > config.thresholds.answer_not_grounded_rate:
        return Action(type="expand_evidence", reason="grounding failure high")

    if recent.evidence_insufficient_rate > config.thresholds.evidence_insufficient_rate:
        return Action(type="expand_evidence", reason="evidence insufficient")

    if (
        mode_state.hard_gap_ratio > config.thresholds.hard_gap_ratio
        and recent.not_multihop_rate > config.thresholds.not_multihop_rate
    ):
        return Action(type="multihop_generate", reason="need harder multihop questions")

    if mode_state.hard_gap_ratio > config.thresholds.hard_gap_ratio:
        return Action(type="hard_generate", reason="hard gap high")

    if mode_state.topic_gap_count > 0:
        return Action(type="topic_focus_generate", reason="topic coverage insufficient")

    if (
        recent.too_easy_rate > config.thresholds.too_easy_rate
        and mode_state.grounded_easy_medium_count >= config.thresholds.evolution_pool_min
    ):
        return Action(type="evolve_grounded_question", reason="too many easy grounded questions")

    return Action(type="generate", reason="default generation")
```

### 8.3 为什么这一版仍可接受

因为它具备：

- 集中阈值
- 可解释优先级
- 有限动作空间
- 低实现成本

相比完整 V3，它更适合第一版落地。

---

## 9. 决策配置

文件：

```text
agents/adp_qa_agent/config/decision.yaml
```

示例：

```yaml
thresholds:
  answer_not_grounded_rate: 0.35
  evidence_insufficient_rate: 0.30
  hard_gap_ratio: 0.25
  not_multihop_rate: 0.40
  too_easy_rate: 0.35
  evolution_pool_min: 5

strategy:
  topic_focus_topics_per_round: 3
  hard_generation_topics_per_round: 2
  multihop_topics_per_round: 2
```

要求：

- 所有阈值集中定义
- 不能散在代码里
- 第一版允许固定优先级写在代码中

---

## 10. 状态设计

文件：

```text
agents/adp_qa_agent/state.py
```

### 10.1 ModeState

```python
@dataclass
class ModeState:
    mode: str
    round_in_mode: int = 1
    candidate_index: list[CandidateRef] = field(default_factory=list)
    difficulty_counts: dict[str, int] = field(default_factory=dict)
    topic_counts: dict[str, int] = field(default_factory=dict)
    consecutive_empty_rounds: int = 0
    failures_count: int = 0
    stopped_reason: str | None = None

    @property
    def accepted_count(self) -> int:
        return sum(1 for item in self.candidate_index if item.status == "accepted")
```

### 10.2 CandidateRef

```python
@dataclass
class CandidateRef:
    question_id: str
    topic: str
    difficulty: str
    status: str
    reject_reason: str | None = None
    source_question_id: str | None = None
    is_evolved: bool = False
```

### 10.3 状态原则

- 完整题目内容不常驻内存
- 内存中只保留轻量索引
- 完整记录继续落盘

这能避免长期运行后内存膨胀。

---

## 11. 反馈设计

文件：

```text
agents/adp_qa_agent/feedback.py
```

### 11.1 第一版反馈只保留最有价值的信号

```python
@dataclass
class RoundFeedback:
    round_id: int
    total_count: int
    accepted_count: int
    too_easy_count: int
    not_multihop_count: int
    answer_not_grounded_count: int
    evidence_insufficient_count: int
    duplicate_count: int
    evolution_attempted: int = 0
    evolution_failed: int = 0
    by_topic: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, int] = field(default_factory=dict)

@dataclass
class RecentFeedbackWindow:
    rounds: deque[RoundFeedback] = field(default_factory=lambda: deque(maxlen=5))
```

### 11.2 Recent 指标

必须至少提供：

- `accepted_rate`
- `too_easy_rate`
- `not_multihop_rate`
- `answer_not_grounded_rate`
- `evidence_insufficient_rate`
- `duplicate_rate`

### 11.3 为什么只看 recent

决策主要基于最近几轮，而不是全局累计。

全局累计只用于最终统计和报告，不驱动下一轮动作。

---

## 12. 验证反馈来源

第一版不定义新验证协议，直接适配现有 `verify_agent` 输出。

适配层职责：

```text
从 validated_questions / validation_report / weighted_selection 中提炼出 RoundFeedback
```

最小适配字段：

- topic
- difficulty
- final_status
- reject_reason
- grounded / evidence support 类失败原因
- 是否 duplicate

如果现有验证输出缺字段，则由适配层补推导逻辑，但不改写验证主逻辑。

---

## 13. Stop 条件

第一版不单独引入全新 `StopPolicy` 模块，直接复用现有 `qa_agent` 的 stop 语义，并只补一个轻量条件。

保留：

- `max_rounds_reached`
- `candidate_pool_sufficient`
- `consecutive_empty_rounds_reached`
- `failure_limit_reached`

可新增：

- `no_recent_improvement`

但第一版即使不加这个条件也可以接受。

原则：

- 不让 LLM 决定停止
- 不让 tool 自主中断轮次

---

## 14. 各工具职责

### 14.1 generation_tool.py

职责：

```text
对现有 execute_mode_round_plan 做薄包装
```

不重写生成主体。

输入：

- 当前 Action
- 当前 round plan
- 当前 mode_state

输出：

- 当前轮生成结果

### 14.2 evidence_tool.py

职责：

```text
在 expand_evidence / multihop_generate / hard_generate 时，
构建更适合的 evidence group 或 topic evidence set
```

复用：

- 现有 `EvidenceManager`
- 现有 `multi_chunk` 相关逻辑

### 14.3 evolution_tool.py

职责：

```text
对已 grounding 且 too_easy 的题做演化
```

但输出的不是 accepted question，而是：

```text
新的 candidate
```

然后必须重新进入验证。

---

## 15. RoundPlan 构造

第一版不需要重写整个 round planner，只需要根据 Action 对已有 round plan 参数做覆盖。

例如：

### 默认 generate

- 使用现有 `build_mode_round_plan`

### topic_focus_generate

- 强制 topics = 缺失 coverage 的 topic 子集

### hard_generate

- 强制 difficulty = hard

### multihop_generate

- difficulty = hard
- evidence strategy = multi-evidence / cross-chunk

### expand_evidence

- 当前轮先准备更多 evidence，再进入下一次生成

### evolve_grounded_question

- 不走普通 topic 生成
- 直接从 eligible grounded candidates 中取输入

---

## 16. Grounded Question Evolution

### 16.1 定位

它是兜底支路，不是主路径。

### 16.2 触发条件

只允许对满足以下条件的题触发：

- 已通过 grounding
- 已通过 answerability
- evidence sufficient
- 当前被判定 `too_easy`
- 原题 difficulty 为 easy 或 medium

禁止对这些题触发：

- grounding 失败
- evidence 不足
- duplicate
- 格式错误
- 含歧义

### 16.3 输出结构

```python
@dataclass
class EvolvedCandidate:
    source_question_id: str
    evolved_question: str
    answer: str
    difficulty: str
    required_evidence_ids: list[str]
    reasoning_path: list[str]
```

### 16.4 必须重新验证

流程：

```text
evolution_tool 输出 EvolvedCandidate
  ↓
重新进入 verify_agent
  ↓
验证通过才进入 accepted 池
验证失败计入 evolution_failed
```

这是第一版必须保证的安全约束。

---

## 17. 对比实验指标

因为它是独立 `adp_qa_agent/`，必须能和 legacy `qa_agent` 做对比。

建议对比：

- 最终 accepted 题数量
- hard 题占比
- multihop 题占比
- grounding 通过率
- duplicate rate
- topic coverage
- token 成本
- 时间成本

如果第一版不能提升全部指标，也至少要回答：

```text
adaptive 策略是否提高了 hard / multihop / grounding 相关质量
```

---

## 18. 实施阶段

### Phase 1

```text
创建 agents/adp_qa_agent/
实现最小 state / feedback / decision / config_loader
```

### Phase 2

```text
包装现有 qa_agent 执行能力
实现 generate / topic_focus_generate / hard_generate
```

### Phase 3

```text
接入 verify_agent 反馈适配
实现 multihop_generate / expand_evidence
```

### Phase 4

```text
实现 evolution_tool
实现 evolution -> revalidate 链路
```

### Phase 5

```text
跑 legacy vs adaptive 对比实验
```

---

## 19. 第一版成功标准

第一版成功不等于“架构最优”，而是满足：

1. 能独立运行
2. 不破坏现有 `qa_agent`
3. 可以稳定输出题目
4. 能消费现有 `verify_agent` 反馈
5. 能根据反馈改变下一轮策略
6. Evolution 不会跳过验证
7. 能和 legacy `qa_agent` 做对比

---

## 20. 最终结论

这版方案本质上是：

```text
独立 adp_qa_agent
+ 轻量决策层
+ 最小反馈闭环
+ 少量策略工具
+ 复用现有 qa_agent 执行能力
```

不是：

```text
完整重写一个新的生成系统
```

这更适合作为当前 BenchForge 的第一版 adaptive 生成智能体实现方案。
